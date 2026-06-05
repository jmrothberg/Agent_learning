#!/usr/bin/env python3
"""Layer-1 eval: one Phase-A planning turn per curated prompt against the local
model (MLX qwen by default), then assert each prompt's CRITICAL FEATURE — with
NO Phase-B build loop, NO Chromium, NO diffuser.

Why this is cheap: GameAgent.run() yields an AgentEvent(kind="plan") with the
full plan reply BEFORE it generates assets/sounds. We break on that event, so
the diffuser is never invoked; browser=None means Chromium never launches. The
MLX backend holds weights at class level, so N sequential turns reuse the model.

This exercises the real plan-time memory injection (playbook, opening-book
outlines/playtests/audits, visual auto-probes, the art/3D/multi-frame intent
detectors) feeding the model, then checks the parsed <plan>/<criteria>/<probes>/
<assets> against the `expect` block co-located in each prompt.

Usage:
    .venv/bin/python eval/eval_prompts_plan.py                 # all prompts
    .venv/bin/python eval/eval_prompts_plan.py --limit 3       # first 3
    .venv/bin/python eval/eval_prompts_plan.py --names street-fighter,chess
    .venv/bin/python eval/eval_prompts_plan.py --only 1
    MLX_MODEL=/path/to/Qwen3.6-27B-mxfp8 .venv/bin/python eval/eval_prompts_plan.py
    .venv/bin/python eval/eval_prompts_plan.py --backend ollama
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend as backend_mod  # noqa: E402
from agent import GameAgent  # noqa: E402
from assets import parse_assets_block_with_meta  # noqa: E402
from memory import GameMemory, Playbook  # noqa: E402
from prompt_library import load_prompt_library  # noqa: E402

try:
    from sounds import parse_sounds_block
except Exception:  # sounds optional
    parse_sounds_block = None


def _check_expect(plan_reply: str, expect: dict) -> dict:
    """Return {pass, checks{...}, counts{...}} for one prompt's critical feature."""
    criteria = (GameAgent._extract_criteria(plan_reply) or "")
    probes = GameAgent._extract_probes(plan_reply) or []
    assets, _dropped = parse_assets_block_with_meta(plan_reply)
    asset_names = [str(a.get("name", "")).lower() for a in assets]
    crit_l = criteria.lower()

    want_assets = [s.lower() for s in expect.get("asset_names_any", [])]
    want_crit = [s.lower() for s in expect.get("criteria_kw_any", [])]
    min_probes = int(expect.get("min_probes", 0))

    assets_ok = (not want_assets) or any(
        sub in nm for nm in asset_names for sub in want_assets
    )
    crit_ok = (not want_crit) or any(kw in crit_l for kw in want_crit)
    probes_ok = len(probes) >= min_probes

    return {
        "pass": assets_ok and crit_ok and probes_ok,
        "checks": {"assets": assets_ok, "criteria": crit_ok, "probes": probes_ok},
        "counts": {"assets": len(assets), "probes": len(probes),
                   "criteria_chars": len(criteria)},
        "asset_names": asset_names,
    }


def _memory_engagement(mem, pb, goal: str) -> dict:
    """What each memory subsystem returns for this goal (kept in the record so a
    failure can be traced to a memory gap)."""
    rec, _ = mem.find_visual_playtest_for(goal=goal, plan_text="", asset_names=[])
    oh = mem.retrieve_implementation_outline(goal)
    sk = mem.retrieve_skeleton(goal)
    return {
        "visual_recipe": rec.id if rec else None,
        "outline": oh.item.id if oh else None,
        "skeleton": sk.name if sk else None,
        "playbook_ids": [h.bullet.id for h in pb.retrieve(goal, stage="plan", k=8)],
        "playtest_ids": [h.item.id for h in mem.retrieve_playtests(goal)],
        "asset_audit_ids": [h.item.id for h in mem.retrieve_asset_audits(goal)],
        "anim_audit_ids": [h.item.id for h in mem.retrieve_animation_audits(goal)],
    }


async def _plan_once(backend_inst, goal: str, out_dir: Path, name: str):
    """Run Phase A only into a PERSISTENT out_dir; return (plan_reply, error,
    trace_path). Breaks on the plan event so no assets/sounds are generated and
    (browser=None) no Chromium launches. The agent writes its full plan-time
    trace (.jsonl + .conversation.md) under out_dir/traces/ — kept, not discarded."""
    if True:
        out_dir.mkdir(parents=True, exist_ok=True)
        agent = GameAgent(
            backend=backend_inst,
            out_path=out_dir / f"{name}.html",
            browser=None,            # no Chromium
            max_iters=1,
            prompt_version="v1",
            playbook_top_k=6,        # engage playbook injection
            memory_root="memory",
        )
        plan_reply = None
        err = None
        try:
            # plan_only=True runs through criteria/probes analysis and writes a
            # complete trace, then returns before asset gen — so we consume to
            # completion (no early break) and the kept trace is fully informative.
            async for ev in agent.run(goal, plan_only=True):
                if ev.kind == "plan":
                    plan_reply = ev.text
                elif ev.kind == "error":
                    err = ev.text
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        return plan_reply, err, getattr(agent, "trace_path", None)


def _coverage(prompts) -> int:
    """Layer-0: print which memory subsystem engages for each prompt. No model.

    Answers 'see how ALL the memory systems work' in one human-readable table.
    """
    mem = GameMemory(root="memory")
    pb = Playbook(root="memory")
    GENERIC = {"canvas-controllable-player", "generic-canvas-game-baseline"}
    hdr = f"{'#':>2} {'game':<16} {'visual':<26} {'outline':<26} {'skeleton':<22} {'pb':>3} {'pt':>3} {'aa':>3} {'an':>3}"
    print(hdr)
    print("-" * len(hdr))
    weak = []
    for g in prompts:
        goal = g["prompt"]
        rec, _ = mem.find_visual_playtest_for(goal=goal, plan_text="", asset_names=[])
        vid = rec.id if rec else "(none)"
        oh = mem.retrieve_implementation_outline(goal)
        oid = oh.item.id if oh else "(none)"
        sk = mem.retrieve_skeleton(goal)
        skn = sk.name if sk else "(none)"
        npb = len(pb.retrieve(goal, stage="plan", k=8))
        npt = len(mem.retrieve_playtests(goal))
        naa = len(mem.retrieve_asset_audits(goal))
        nan = len(mem.retrieve_animation_audits(goal))
        if vid in GENERIC or npb == 0:
            weak.append(g["name"])
        print(f"{g['n']:>2} {g['name']:<16} {vid:<26} {oid:<26} {skn:<22} {npb:>3} {npt:>3} {naa:>3} {nan:>3}")
    print("\nlegend: pb=playbook bullets · pt=playtests · aa=asset-audits · an=animation-audits")
    print(f"{len(prompts)} prompts · {len(prompts)-len(weak)} with genre-specific visual recipe + playbook engaged")
    if weak:
        print(f"weak/generic engagement: {weak}")
    return 0


def _select(prompts, args):
    if args.only is not None:
        return [g for g in prompts if g["n"] == args.only]
    if args.names:
        wanted = {n.strip().lower() for n in args.names.split(",")}
        return [g for g in prompts if g["name"].lower() in wanted]
    if args.limit is not None:
        return prompts[: args.limit]
    return prompts


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mlx", help="mlx | ollama | auto")
    ap.add_argument("--model", default=None, help="override model id/path")
    ap.add_argument("--only", type=int, default=None, help="single prompt number")
    ap.add_argument("--names", default=None, help="comma list of prompt names")
    ap.add_argument("--limit", type=int, default=None, help="first K prompts")
    ap.add_argument("--coverage", action="store_true",
                    help="Layer-0 memory engagement matrix; no model, fast")
    args = ap.parse_args()

    prompts = _select(load_prompt_library(), args)
    if not prompts:
        print("no prompts selected", file=sys.stderr)
        return 2

    if args.coverage:
        return _coverage(prompts)

    try:
        info = backend_mod.detect_backend(args.backend)
    except Exception as e:  # noqa: BLE001
        print(f"backend resolution failed: {e}", file=sys.stderr)
        return 2
    if args.model:
        info = backend_mod.BackendInfo(
            name=info.name, model=args.model,
            source=f"--model {args.model!r}", endpoint=info.endpoint,
        )
    backend_inst = backend_mod.make_backend(info)

    # Persistent trace root (KEPT — under games/, so on disk for analysis).
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path("games") / "eval-traces" / f"eval_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "eval_summary.jsonl"
    report_path = root / "eval_report.md"

    mem, pb = GameMemory(root="memory"), Playbook(root="memory")
    print(f"backend: {info.name} · model: {info.model}")
    print(f"traces : {root}")
    print(f"prompts: {len(prompts)} · planning-only (no browser, no diffuser)\n")

    rows = []  # each: dict with g, res|None, dt, memcov, trace_path
    for g in prompts:
        t0 = time.time()
        plan_reply, err, trace_path = await _plan_once(
            backend_inst, g["prompt"], root, g["name"])
        dt = round(time.time() - t0, 1)
        memcov = _memory_engagement(mem, pb, g["prompt"])
        tp = str(trace_path) if trace_path else None
        if err or not plan_reply:
            print(f"{g['n']:>2} {g['name']:<16} ERROR  {err or 'no plan reply'}")
            res = None
            rec = {"n": g["n"], "name": g["name"], "error": err or "no plan reply",
                   "gen_seconds": dt, "memory": memcov, "trace_path": tp}
        else:
            res = _check_expect(plan_reply, g.get("expect", {}))
            (root / f"{g['name']}.plan.md").write_text(plan_reply, encoding="utf-8")
            c = res["checks"]
            print(f"{g['n']:>2} {g['name']:<16} {'PASS' if res['pass'] else 'FAIL'}  "
                  f"assets={'Y' if c['assets'] else 'n'} "
                  f"criteria={'Y' if c['criteria'] else 'n'} "
                  f"probes={'Y' if c['probes'] else 'n'} "
                  f"({res['counts']['assets']}a/{res['counts']['probes']}p, {dt:.0f}s)")
            rec = {"n": g["n"], "name": g["name"], "pass": res["pass"],
                   "checks": res["checks"], "counts": res["counts"],
                   "asset_names": res["asset_names"], "expect": g.get("expect", {}),
                   "criteria_excerpt": (GameAgent._extract_criteria(plan_reply) or "")[:300],
                   "gen_seconds": dt, "memory": memcov, "trace_path": tp}
        rows.append({"g": g, "res": res, "dt": dt, "mem": memcov, "trace": tp})
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    done = [r for r in rows if r["res"] is not None]
    passed = [r for r in done if r["res"]["pass"]]
    failed = [r for r in done if not r["res"]["pass"]]
    errored = [r for r in rows if r["res"] is None]

    # Aggregate report — the improvement substrate.
    lines = [f"# Prompt-library plan eval — {stamp}", "",
             f"- model: `{info.model}`",
             f"- {len(passed)}/{len(done)} critical features present"
             f"{f', {len(errored)} errored' if errored else ''}", ""]
    if failed:
        lines.append("## Critical-feature MISSES (improvement signals)")
        for r in failed:
            miss = [k for k, v in r["res"]["checks"].items() if not v]
            lines.append(
                f"- **{r['g']['name']}** — failed {miss} · "
                f"expect={r['g'].get('expect', {})} · "
                f"got assets={r['res']['asset_names'][:8]} · "
                f"trace=`{r['trace']}`")
        lines.append("")
    if errored:
        lines.append("## Errored")
        lines += [f"- {r['g']['name']}: {r}" for r in errored] + [""]
    lines.append("## Full matrix (verdict · assets · probes · memory engaged)")
    lines.append("| # | game | verdict | a | p | visual_recipe | outline | skeleton |")
    lines.append("|--:|------|---------|--:|--:|---------------|---------|----------|")
    for r in rows:
        res = r["res"]
        v = "ERR" if res is None else ("PASS" if res["pass"] else "FAIL")
        a = "" if res is None else res["counts"]["assets"]
        p = "" if res is None else res["counts"]["probes"]
        m = r["mem"]
        lines.append(f"| {r['g']['n']} | {r['g']['name']} | {v} | {a} | {p} | "
                     f"{m['visual_recipe']} | {m['outline']} | {m['skeleton']} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nsummary: {len(passed)}/{len(done)} critical features present"
          f"{f' · {len(errored)} errored' if errored else ''}")
    print(f"kept   : {summary_path}")
    print(f"report : {report_path}")
    return 0 if len(passed) == len(done) and not errored else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
