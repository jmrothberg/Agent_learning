#!/usr/bin/env python3
"""Headless Chromium smoke for asset-decode settle + undrawn detection.

Loads eval/fixtures/dojo_fighters_asset_smoke.html (real dojo sprites from
goodgame/) through LiveBrowser — no GameAgent, no coder LLM.

Run from repo root (Playwright Chromium required):

    .venv/bin/python scripts/_smoke_asset_decode_settle.py
    .venv/bin/python scripts/_smoke_asset_decode_settle.py --full
    .venv/bin/python scripts/_smoke_asset_decode_settle.py --vlm

Exits 0 on success, 1 on assertion failure, 2 on missing deps / VLM model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REPO = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO / "eval" / "fixtures" / "dojo_fighters_asset_smoke.html"

_DEFAULT_SMOKE_VLM_QUERY = os.environ.get(
    "SMOKE_VLM_MODEL",
    os.environ.get("VISION_JUDGE_MODEL", "qwen3.6-27b-mxfp8"),
)

_FACE_PROBE_EXPR = (
    "(()=>{const s=window.state||window.gameState;if(!s)return true;"
    "const p1=s.p1||s.player1||s.fighter1||(Array.isArray(s.players)&&s.players[0])"
    "||(Array.isArray(s.fighters)&&s.fighters[0]);"
    "const p2=s.p2||s.player2||s.fighter2||(Array.isArray(s.players)&&s.players[1])"
    "||(Array.isArray(s.fighters)&&s.fighters[1]);"
    "if(!p1||!p2)return true;"
    "if(typeof p1.facing!=='number'||typeof p2.facing!=='number')return true;"
    "if(p1.facing===0&&p2.facing===0)return false;"
    "return Math.sign(p1.facing)!==Math.sign(p2.facing);})()"
)

_VLM_CHECKLIST = (
    "Q1: Are two distinct fighter characters visible on screen?\n"
    "Q2: Is one character on the left and one on the right?\n"
    "Q3: Do both characters appear to face toward each other?\n"
)

_VLM_PROMPT = (
    "You are reviewing one screenshot of a 2D fighting game on a dojo stage.\n\n"
    "Answer each numbered question by re-emitting Qn: YES, NO, or UNCLEAR "
    "(one line per question, in order). Do not add other prose.\n\n"
    + _VLM_CHECKLIST
)


def _load_facing_recipe_checklist() -> list[str]:
    path = _REPO / "memory" / "visual_playtests.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("id") == "canvas-two-actors-facing":
            return list((rec.get("recipe") or {}).get("checklist") or [])[:3]
    return [q.split(":", 1)[-1].strip() for q in _VLM_CHECKLIST.strip().splitlines()]


def _build_vlm_prompt() -> str:
    items = _load_facing_recipe_checklist()[:2]
    numbered = "\n".join(f"Q{i + 1}: {q}" for i, q in enumerate(items))
    return (
        "You are reviewing one screenshot of a 2D fighting game on a dojo stage.\n\n"
        "Answer each numbered question by re-emitting Qn: YES, NO, or UNCLEAR "
        "(one line per question, in order). Do not add other prose.\n\n"
        f"{numbered}\n"
    )


def _parse_vlm_checklist(raw: str, *, n_questions: int = 3) -> dict[int, str]:
    answers: dict[int, str] = {}
    pat = re.compile(
        r"^\s*Q?\s*(\d+)\s*[:.\-)]\s*"
        r"(?:\d+\s*[.)]\s*)?"
        r"(yes|no|unclear|y|n|u)\b",
        re.I | re.M,
    )
    for m in pat.finditer(raw or ""):
        idx = int(m.group(1))
        word = m.group(2).lower()
        if word in ("y", "yes"):
            answers[idx] = "yes"
        elif word in ("n", "no"):
            answers[idx] = "no"
        else:
            answers[idx] = "unclear"
    return answers


def _vlm_checklist_passes(raw: str) -> tuple[bool, str]:
    """Pass when Q1+Q2 are YES (two characters visible, left vs right).

    Q3 (facing) is judged by auto_actors_face_each_other on window.state —
    idle sprite art often looks forward/symmetric so the VLM cannot reliably
  read facing from a still frame.
    """
    answers = _parse_vlm_checklist(raw)
    q1 = answers.get(1, "").lower()
    q2 = answers.get(2, "").lower()
    if q1 == "yes" and q2 == "yes":
        return True, raw
    # Fallback: bare YES when model ignores checklist format.
    low = (raw or "").strip().lower()
    if low == "yes" or low.startswith("yes"):
        return True, raw
    if answers:
        detail = ", ".join(f"Q{k}={v}" for k, v in sorted(answers.items()))
        return False, f"{raw}\n(parsed: {detail})"
    return False, raw or "(empty)"


async def _run_vlm_check(page, model_path: str) -> tuple[bool, str]:
    from vision_judge import run_local_vlm_prompt

    await page.evaluate(
        "(async()=>{for(let i=0;i<3;i++)"
        "await new Promise(r=>requestAnimationFrame(r));})()"
    )
    png = await page.screenshot(type="png", full_page=False)
    raw = await run_local_vlm_prompt(
        prompt=_build_vlm_prompt(),
        images=[png],
        model_path=model_path,
        max_tokens=256,
    )
    if not raw:
        return False, "VLM returned empty response"
    return _vlm_checklist_passes(raw)


async def amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Load all dojo PNGs")
    parser.add_argument("--vlm", action="store_true", help="Qwen3.6-27B-mxfp8 screenshot check")
    parser.add_argument(
        "--no-settle", action="store_true",
        help="Skip asset_decode_settle (regression guard — expect undrawn fail)",
    )
    args = parser.parse_args(argv)

    if args.vlm:
        from vision_judge import is_enabled
        if not is_enabled():
            print("SKIP: VISION_JUDGE=0")
            return 0

    if not _FIXTURE.is_file():
        print(f"FAIL: fixture missing: {_FIXTURE}")
        return 2

    try:
        from tools import LiveBrowser
    except Exception as e:
        print(f"FAIL: could not import tools.LiveBrowser — {e}")
        return 2

    html_path = _FIXTURE
    tmp_dir: Path | None = None
    if args.full:
        tmp_dir = Path(tempfile.mkdtemp(prefix="dojo_smoke_full_"))
        html_src = _FIXTURE.read_text(encoding="utf-8")
        html_src = html_src.replace(
            "const entries = (window.__SMOKE_FULL_ASSETS ? FULL_ASSETS : MIN_ASSETS);",
            "const entries = FULL_ASSETS;",
        )
        html_path = tmp_dir / "dojo_full.html"
        html_path.write_text(html_src, encoding="utf-8")

    vlm_path: str | None = None
    if args.vlm:
        try:
            from vision_judge import _resolve_local_mlx_vlm
        except Exception as e:
            print(f"FAIL: vision_judge unavailable — {e}")
            return 2
        vlm_path = _resolve_local_mlx_vlm(_DEFAULT_SMOKE_VLM_QUERY)
        if not vlm_path:
            print(
                f"FAIL: local VLM not found for query {_DEFAULT_SMOKE_VLM_QUERY!r} "
                f"(expected ~/MLX_Models/Qwen3.6-27B-mxfp8)"
            )
            return 2
        print(f"[vlm] resolved {_DEFAULT_SMOKE_VLM_QUERY!r} -> {vlm_path}")

    run_secs = 0.35 if args.no_settle else 4.0
    lb = LiveBrowser(headless=True, run_seconds=run_secs)
    failures: list[str] = []
    try:
        await lb.start()
        print(f"[1] loading {html_path}")
        report = await lb.load_and_test(
            html_path,
            probes=[{"name": "auto_actors_face_each_other", "expr": _FACE_PROBE_EXPR}],
            asset_decode_settle=not args.no_settle,
        )

        settle = report.get("asset_decode_settle") or {}
        dac = report.get("drawn_asset_check") or {}
        undrawn = dac.get("undrawn") or []
        soft = report.get("soft_warnings") or []
        blocking_undrawn = any(
            "ASSETS_LOADED_BUT_UNDRAWN" in w for w in soft
        )

        print()
        print("== asset_decode_settle ==")
        print(f"  {settle}")
        print("== drawn_asset_check ==")
        print(f"  undrawn: {undrawn}")
        print(f"  ok: {report.get('ok')}")
        print(f"  probes: {report.get('probes')}")

        if args.no_settle:
            if not settle.get("skipped"):
                failures.append("--no-settle: expected asset_decode_settle.skipped")
            if settle.get("ready") is True:
                failures.append("--no-settle: settle should not report ready when skipped")
            if undrawn:
                print("  (undrawn present without settle — timing regression reproduced)")
        else:
            if settle.get("ready") is not True:
                failures.append(f"asset_decode_settle.ready is not true: {settle!r}")
            if blocking_undrawn:
                failures.append(
                    "ASSETS_LOADED_BUT_UNDRAWN still blocking in soft_warnings"
                )
            for stem in ("blue_idle", "red_idle", "dojo_bg"):
                if stem in undrawn:
                    failures.append(f"expected {stem} drawn but it is undrawn")
            probes = report.get("probes") or []
            face = next((p for p in probes if p.get("name") == "auto_actors_face_each_other"), None)
            if not face or face.get("ok") is not True:
                failures.append(f"auto_actors_face_each_other failed: {face!r}")
            if report.get("ok") is False and blocking_undrawn:
                failures.append("report ok=False due to undrawn gate")

        if args.vlm and vlm_path and not failures:
            ok_vlm, raw = await _run_vlm_check(lb._page, vlm_path)
            print()
            print("== VLM ==")
            print(raw[:500])
            if not ok_vlm:
                failures.append("VLM did not confirm two fighters visible and facing")

        if failures:
            print()
            print("== FAIL ==")
            for f in failures:
                print(f"  - {f}")
            return 1
        print()
        print("== PASS — asset decode settle smoke ==")
        return 0
    finally:
        await lb.close()
        if tmp_dir:
            try:
                for p in tmp_dir.iterdir():
                    p.unlink()
                tmp_dir.rmdir()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
