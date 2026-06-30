#!/usr/bin/env python3
"""Opt-in eval: agent + VLM-critique fixes fighter facing on a seeded bug.

Runs the full loop over a minimal HTML seed with an intentional facing bug
(both fighters stay facing +1; no flip toward opponent). Preflight: local VLM
must say NO on the seed screenshot (see scripts/smoke_vlm_facing_sanity.py).
PASS when post-run VLM Q4 is YES on a crossover screenshot; state probes are
secondary. Main agent /vlm-critique is unchanged — still useful for simpler checks.

Needs: local MLX model, Chromium (visible), ~3-8 min cold.

Usage:
    MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 \\
      .venv/bin/python eval/eval_vlm_facing_fix.py
    .venv/bin/python eval/eval_vlm_facing_fix.py --max-iters 3 --headless
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Before mlx/tokenizers import — avoids leaked-semaphore warning at exit.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend as backend_mod  # noqa: E402
from agent import AgentEvent, GameAgent  # noqa: E402
from tools import LiveBrowser  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO / "eval" / "fixtures" / "seed_fighters_facing_bug.html"
_FIXTURE_ASSETS = _FIXTURE.parent / "seed_fighters_facing_bug_assets"
_OUT_STEM = _FIXTURE.stem
_PLAYTESTS = _REPO / "memory" / "visual_playtests.jsonl"

GOAL = (
    "Two-player 1v1 fighter versus duel. Use ONLY the existing blue_idle and "
    "red_idle PNGs in seed_fighters_facing_bug_assets — no new art, no combat, no "
    "HUD. ArrowLeft and ArrowRight move player 1 horizontally. Both fighters "
    "must always face each other (flip sprites toward the opponent when relative "
    "position changes)."
)

_CROSSOVER_PROBE_EXPR = (
    "(()=>{const s=window.state;if(!s||!s.p1||!s.p2)return true;"
    "const u=window.game&&window.game.update;if(typeof u!=='function')return true;"
    "const bak={x1:s.p1.x,f1:s.p1.facing,f2:s.p2.facing};"
    "s.p1.x=s.p2.x+80;u();"
    "const sorted=[s.p1,s.p2].sort((a,b)=>(a.x||0)-(b.x||0));"
    "const ok=sorted[0].facing>0&&sorted[1].facing<0;"
    "s.p1.x=bak.x1;s.p1.facing=bak.f1;s.p2.facing=bak.f2;u();return ok;})()"
)

_CROSSOVER_MOVE_JS = (
    "(()=>{const s=window.state;if(!s||!s.p1||!s.p2)return false;"
    "const u=window.game&&window.game.update;if(typeof u!=='function')return false;"
    "s.p1.x=s.p2.x+80;u();return true;})()"
)

_RAF_SETTLE_JS = (
    "(async()=>{for(let i=0;i<4;i++)await new Promise(r=>requestAnimationFrame(r));})()"
)

from eval.vlm_facing_sanity import (  # noqa: E402
    ask_facing_vlm,
    run_vlm_facing_sanity,
)

_FACING_TRACE_KEYS = (
    "facing",
    "face each other",
    "face toward",
    "flip",
    "orientation",
    "mirror",
    "scale(-1",
)


def _print_event(ev: AgentEvent) -> None:
    """Mirror coder.py progress lines so unattended eval runs stay visible."""
    if ev.kind == "phase":
        print(f"\n── {ev.text} ──", flush=True)
    elif ev.kind == "activity":
        label = (ev.data or {}).get("label") or ev.text
        role = (ev.data or {}).get("role")
        suffix = f" [{role}]" if role else ""
        print(f"  … {label}{suffix}", flush=True)
    elif ev.kind == "code":
        d = ev.data or {}
        print(
            f"  wrote {ev.text} ({d.get('size', 0)} bytes; {d.get('materialize', 'n/a')})",
            flush=True,
        )
    elif ev.kind == "test":
        ok = (ev.data or {}).get("ok", False)
        pp = (ev.data or {}).get("probes_passed")
        pt = (ev.data or {}).get("probes_total")
        tag = "TEST OK" if ok else "TEST FAILED"
        probe_bits = f" probes={pp}/{pt}" if pt is not None else ""
        print(f"  {tag}{probe_bits}", flush=True)
        if not ok and ev.text:
            print(f"  --- report ---\n{ev.text[:1200]}\n  --- /report ---", flush=True)
    elif ev.kind == "done":
        print(f"\nDONE — {ev.text}", flush=True)
    elif ev.kind == "error":
        print(f"\n! ERROR: {ev.text}", flush=True)
    elif ev.kind == "info" and ev.text:
        if "step-mode auto-armed" in ev.text or "await_user" in ev.text:
            print(f"  i {ev.text}", flush=True)
        elif ev.text.startswith("visual critic") or "VLM-CRITIQUE" in ev.text:
            print(f"  vlm: {ev.text[:200]}", flush=True)
    elif ev.kind == "await_user":
        print(f"\n⏸  step-mode pause ignored (unattended eval): {ev.text}", flush=True)


def _stage_fixture_assets(out_dir: Path) -> Path:
    """Copy fixture sprites beside the runtime HTML (<stem>_assets convention)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_assets = out_dir / f"{_OUT_STEM}_assets"
    dst_assets.mkdir(parents=True, exist_ok=True)
    for name in ("blue_idle.png", "red_idle.png"):
        shutil.copy2(_FIXTURE_ASSETS / name, dst_assets / name)
    return out_dir / f"{_OUT_STEM}.html"


def _facing_recipe_probe_exprs() -> dict[str, str]:
    """Load auto_actors_face_* probe exprs from canvas-two-actors-facing."""
    out: dict[str, str] = {}
    if not _PLAYTESTS.is_file():
        return out
    for line in _PLAYTESTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") != "canvas-two-actors-facing":
            continue
        for ap in (rec.get("recipe") or {}).get("auto_probes") or []:
            name = ap.get("name") or ""
            expr = ap.get("expr") or ""
            if name and expr:
                out[name] = expr
        break
    return out


def _probe_ok(report: dict | None, name: str) -> bool | None:
    if not report:
        return None
    for p in report.get("probes") or []:
        if p.get("name") == name:
            return bool(p.get("ok"))
    return None


def _probes_total(report: dict | None) -> int:
    if not report:
        return 0
    probes = report.get("probes") or []
    if probes:
        return len(probes)
    total = report.get("probes_total")
    if isinstance(total, int):
        return total
    passed = report.get("probes_passed")
    if isinstance(passed, int) and passed == 0 and report.get("ok") is True:
        return 0
    return 0


def _trace_records(trace_path: Path | None) -> list[dict]:
    if not trace_path or not trace_path.is_file():
        return []
    out: list[dict] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _trace_has_facing_critique(records: list[dict]) -> bool:
    for rec in records:
        kind = rec.get("kind") or ""
        if kind not in (
            "visual_playtest",
            "visual_playtest_parsed",
            "structured_critic_via_local_vlm",
            "vision_judge",
            "critic_note",
        ):
            continue
        blob = json.dumps(rec, ensure_ascii=False).lower()
        if any(k in blob for k in _FACING_TRACE_KEYS):
            return True
    return False


def _trace_has_patch_after_iter1(records: list[dict]) -> bool:
    for rec in records:
        if rec.get("kind") != "patch_outcome":
            continue
        if int(rec.get("iteration") or 0) >= 2 and rec.get("applied"):
            return True
    return False


async def _post_run_facing_check(
    html_path: Path,
    *,
    headless: bool,
) -> dict | None:
    """Run facing probes + crossover sim on the shipped HTML."""
    if not html_path.is_file():
        return None
    recipe_probes = _facing_recipe_probe_exprs()
    probes: list[dict] = []
    for name in ("auto_actors_face_each_other", "auto_actors_face_each_other_strict"):
        expr = recipe_probes.get(name)
        if expr:
            probes.append({"name": name, "expr": expr})
    probes.append({"name": "facing_crossover_strict", "expr": _CROSSOVER_PROBE_EXPR})

    browser = LiveBrowser(run_seconds=2.0, headless=headless)
    try:
        await browser.start()
        report = await browser.load_and_test(html_path, probes=probes)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}
    finally:
        try:
            await browser.close()
        except Exception:
            pass
    return report


async def _post_run_vlm_facing_q4(
    html_path: Path,
    *,
    headless: bool,
) -> dict | None:
    """After crossover, ask VLM Q4 whether sprites visually face each other."""
    if not html_path.is_file():
        return None
    from eval.vlm_facing_sanity import _resolve_vlm_path

    model_path = _resolve_vlm_path()
    if not model_path:
        return {"ok": False, "error": "VISION_JUDGE disabled or no local VLM"}

    browser = LiveBrowser(run_seconds=1.0, headless=headless)
    try:
        await browser.start()
        file_url = html_path.resolve().as_uri()
        await browser._page.goto(file_url, wait_until="load", timeout=15_000)
        settle = await browser._wait_for_session_assets_ready()
        if settle.get("need") and not settle.get("ready"):
            return {
                "ok": False,
                "error": "asset_decode_settle not ready",
                "settle": settle,
            }
        crossed = await browser._safe_eval(_CROSSOVER_MOVE_JS)
        if crossed is not True:
            return {"ok": False, "error": "crossover move failed"}
        await browser._page.evaluate(_RAF_SETTLE_JS)
        png = await browser._page.screenshot(type="png", full_page=False)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    vlm = await ask_facing_vlm(
        png,
        model_path=model_path,
        context=(
            "You are reviewing one screenshot of a 2-player fighting game AFTER "
            "player 1 has moved to the right side of player 2."
        ),
    )
    return {
        "ok": vlm.get("ok"),
        "q4_answer": vlm.get("q_answer"),
        "parse_rate": vlm.get("parse_rate"),
        "raw_preview": vlm.get("raw_preview"),
        "error": vlm.get("error"),
    }


def _mlx_teardown() -> None:
    """Release MLX Metal context before hard exit (avoids segfault on teardown)."""
    try:
        from backend import MLXBackend

        MLXBackend._drop_after_crash()
    except Exception:
        pass


async def _run_once(
    backend_inst,
    *,
    max_iters: int,
    headless: bool,
    out_dir: Path,
) -> tuple[dict | None, dict | None, dict | None, str | None, Path | None]:
    out_path = _stage_fixture_assets(out_dir)
    browser = LiveBrowser(run_seconds=3.0, headless=headless)
    try:
        await browser.start()
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"Chromium start failed: {e}", None

    agent = GameAgent(
        backend=backend_inst,
        out_path=out_path,
        browser=browser,
        max_iters=max_iters,
        prompt_version="v1",
        playbook_top_k=6,
        memory_root="memory",
        seed_file=_FIXTURE,
        use_vlm_critique=True,
    )
    # Unattended eval: never auto-arm step-mode on test failure (default True
    # in GameAgent — TUI /wait on only). Visible browser ≠ interactive wait.
    agent.set_step_mode(False)
    agent.set_auto_step_on_failure(False)

    def on_token(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()

    agent.set_token_callback(on_token)

    print(f"  html: {out_path}", flush=True)
    print("  wait/step-mode: OFF (unattended eval)", flush=True)
    print("  streaming model tokens to stdout…", flush=True)

    err = None
    try:
        async for ev in agent.run(GOAL, patch_only=True):
            _print_event(ev)
            # Belt-and-suspenders: never block on stdin if step-mode slips on.
            if ev.kind == "await_user":
                agent.signal_step_continue()
            if ev.kind == "error":
                err = ev.text
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    agent_report = getattr(agent, "_last_test_report", None)
    trace_path = getattr(agent, "trace_path", None)

    best = out_path.with_name(out_path.stem + ".best.html")
    check_html = best if best.is_file() else out_path
    post_report = await _post_run_facing_check(check_html, headless=headless)
    vlm_report = await _post_run_vlm_facing_q4(check_html, headless=headless)
    return agent_report, post_report, vlm_report, err, trace_path


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mlx", help="mlx | mlx-server | ollama | auto")
    ap.add_argument("--model", default=None, help="override model id/path (else MLX_MODEL env)")
    ap.add_argument("--max-iters", type=int, default=3, help="build iterations (default 3)")
    ap.add_argument("--headless", action="store_true", help="Chromium without visible window")
    ap.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip VLM preflight (not recommended; facing Q4 may false-pass)",
    )
    args = ap.parse_args()

    if not _FIXTURE.is_file():
        print(f"seed fixture missing: {_FIXTURE}", file=sys.stderr)
        return 2
    if not (_FIXTURE_ASSETS / "blue_idle.png").is_file() or not (_FIXTURE_ASSETS / "red_idle.png").is_file():
        print(f"assets missing under {_FIXTURE_ASSETS}", file=sys.stderr)
        return 2

    try:
        info = backend_mod.detect_backend(args.backend)
    except Exception as e:  # noqa: BLE001
        print(f"backend resolution failed: {e}", file=sys.stderr)
        return 2
    if args.model:
        info = backend_mod.BackendInfo(
            name=info.name,
            model=args.model,
            source=f"--model {args.model!r}",
            endpoint=info.endpoint,
        )
    backend_inst = backend_mod.make_backend(info)
    print(
        f"backend: {info.name} · model: {info.model} · max_iters={args.max_iters} "
        f"· vlm-critique=ON · headless={args.headless} · patch-only"
    )
    print(f"goal: {GOAL}\n")

    if not args.skip_sanity:
        print("VLM facing sanity (seed bug must get Q=NO)…", flush=True)
        sanity = await run_vlm_facing_sanity(_FIXTURE, headless=True)
        print(
            f"  sanity: q_answer={sanity.get('q_answer')!r}  "
            f"model={sanity.get('model_path')}",
            flush=True,
        )
        if sanity.get("raw_preview"):
            print(f"  raw: {sanity.get('raw_preview')!r}", flush=True)
        if not sanity.get("sanity_ok"):
            err_low = (sanity.get("error") or "").lower()
            if not sanity.get("q_answer") and (
                "playwright" in err_low or "executable doesn't exist" in err_low
            ):
                print("\n[ABORT] Chromium/Playwright not available for sanity screenshot.", file=sys.stderr)
                return 2
            print(
                "\n[SKIP] VLM facing sanity FAILED — this model cannot detect the seed bug. "
                "Run scripts/smoke_vlm_facing_sanity.py after switching MLX_MODEL. "
                "Use --skip-sanity to force eval anyway.",
                file=sys.stderr,
            )
            if sanity.get("error"):
                print(f"  error: {sanity['error']}", file=sys.stderr)
            return 3
        print("  sanity PASS\n", flush=True)
    else:
        print("WARN: --skip-sanity — post-run VLM Q4 may false-pass\n", flush=True)

    print("Progress: phase / activity / TEST OK|FAILED lines stream below.\n", flush=True)

    t0 = time.time()
    agent_report, post_report, vlm_report, err, trace_path = await _run_once(
        backend_inst,
        max_iters=args.max_iters,
        headless=args.headless,
        out_dir=Path("games") / "eval_vlm_facing",
    )
    dt = time.time() - t0

    agent_total = _probes_total(agent_report)
    face_ok = _probe_ok(post_report, "auto_actors_face_each_other")
    strict_ok = _probe_ok(post_report, "auto_actors_face_each_other_strict")
    cross_ok = _probe_ok(post_report, "facing_crossover_strict")
    post_total = _probes_total(post_report)

    vlm_q4_ok = vlm_report.get("ok") if vlm_report else None
    primary = vlm_q4_ok is True
    records = _trace_records(trace_path)
    secondary = (
        post_total > 0
        and face_ok is True
        and strict_ok is True
        and cross_ok is True
    )
    tertiary = _trace_has_facing_critique(records) or _trace_has_patch_after_iter1(records)

    status = "PASS" if primary else "FAIL"
    print(f"\n[{status}] facing eval ({dt:.0f}s)")
    print(f"  PRIMARY (VLM Q4 after crossover): {vlm_q4_ok!r}")
    if vlm_report:
        print(f"    q4_answer={vlm_report.get('q4_answer')!r}  parse_rate={vlm_report.get('parse_rate')}")
        if vlm_report.get("raw_preview"):
            print(f"    raw: {vlm_report.get('raw_preview')!r}")
        if vlm_report.get("error"):
            print(f"    error: {vlm_report.get('error')}")
    print(f"  secondary (state probes): face={face_ok!r} strict={strict_ok!r} cross={cross_ok!r}")
    print(f"  agent probes_total={agent_total}  post-check probes_total={post_total}")
    print(f"  tertiary (trace VLM/patch signal): {tertiary}")
    if agent_total == 0:
        print("  WARN: agent run had 0 probes — patch-only auto-probe injection may be broken")
    if err:
        print(f"  error: {err}")
    if agent_report is not None:
        print(f"  agent final ok={agent_report.get('ok')!r}  probes_passed={agent_report.get('probes_passed')}")
    if post_report is not None and post_report.get("error"):
        print(f"  post-check error: {post_report.get('error')}")
    if trace_path:
        print(f"  trace: {trace_path}")
        print(f"  triage: .venv/bin/python scripts/enrich_trace.py {trace_path.stem} --timeline")

    return 0 if primary else 1


if __name__ == "__main__":
    _rc = 2
    try:
        _rc = asyncio.run(main())
    finally:
        _mlx_teardown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_rc)
