#!/usr/bin/env python3
"""Layer-2b eval: seed-EDIT robustness against the local model (opt-in).

Runs real build/iterate turns over a seed HTML fixture (no Chromium: browser
is None, so we measure MATERIALIZATION not gameplay) and asserts the single
thing seed edits kept failing at on slow local models: the model actually
emitted code that CHANGED the file.

PASS for a scenario = the agent materialized a new HTML body whose bytes differ
from the seed (a <patch> applied or an <html_file> rewrite landed). FAIL = the
turn produced no usable code (deliberation / loop / format stall) or left the
seed byte-identical.

This is the regression guard for the Fieldrunners-class failure
(trace 20260626_102307 iters 4-5: feedback produced no saved code) and the
seed-edit hardening generally. It is OPT-IN (needs a real model) — never part
of the pure-pytest suite.

Usage:
    MLX_MODEL=~/MLX_Models/GLM-5.2-MLX-4bit .venv/bin/python eval/eval_seed_edits.py
    .venv/bin/python eval/eval_seed_edits.py --backend ollama
    .venv/bin/python eval/eval_seed_edits.py --only 1
    .venv/bin/python eval/eval_seed_edits.py --max-iters 2
    .venv/bin/python eval/eval_seed_edits.py --patch-only --only 1
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend as backend_mod  # noqa: E402
from agent import GameAgent  # noqa: E402

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "seed_tower_defense.html"

# Edit goals tailored to the TD fixture. Each is a localized, verifiable edit
# a weak model should be able to land in one or two turns. Genre-free shapes:
# resize, recolor, reposition, add-structure, behavior-tweak.
SCENARIOS: list[dict] = [
    {"name": "bigger_towers", "goal": "make the towers render bigger — draw them as 28x28 squares instead of 20x20"},
    {"name": "recolor_creeps", "goal": "change the creep color from red to bright yellow"},
    {"name": "faster_creeps", "goal": "make the creeps move faster by increasing their speed"},
    {"name": "second_path", "goal": "add a second straight enemy path 64 pixels below the first one and draw it"},
    {"name": "tower_range_ring", "goal": "draw a faint range circle around every placed tower"},
]


def _select(scenarios, args):
    if args.only is not None:
        return [scenarios[args.only - 1]] if 1 <= args.only <= len(scenarios) else []
    if args.names:
        wanted = {n.strip().lower() for n in args.names.split(",")}
        return [s for s in scenarios if s["name"].lower() in wanted]
    return scenarios


async def _edit_once(backend_inst, goal: str, out_dir: Path, name: str, max_iters: int, *, patch_only: bool = False):
    """Run a real edit turn over the seed fixture. Returns
    (materialized: bool, changed: bool, err: str|None, trace_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_text = _FIXTURE.read_text(encoding="utf-8")
    agent = GameAgent(
        backend=backend_inst,
        out_path=out_dir / f"{name}.html",
        browser=None,                 # measure materialization, not gameplay
        max_iters=max_iters,
        prompt_version="v1",
        playbook_top_k=6,
        memory_root="memory",
        seed_file=_FIXTURE,
    )
    err = None
    try:
        async for ev in agent.run(goal, patch_only=patch_only):
            if ev.kind == "error":
                err = ev.text
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    cur = agent._current_file or ""
    materialized = bool(cur)
    changed = materialized and cur.strip() != seed_text.strip()
    return materialized, changed, err, getattr(agent, "trace_path", None)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mlx", help="mlx | ollama | auto")
    ap.add_argument("--model", default=None, help="override model id/path")
    ap.add_argument("--only", type=int, default=None, help="single scenario number (1-based)")
    ap.add_argument("--names", default=None, help="comma list of scenario names")
    ap.add_argument("--max-iters", type=int, default=2, help="iterations per scenario")
    ap.add_argument(
        "--patch-only",
        action="store_true",
        help="skip Phase A planning + phase_a asset generation (canvas seed edits)",
    )
    args = ap.parse_args()

    if not _FIXTURE.exists():
        print(f"seed fixture missing: {_FIXTURE}", file=sys.stderr)
        return 2

    scenarios = _select(SCENARIOS, args)
    if not scenarios:
        print("no scenarios selected", file=sys.stderr)
        return 2

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
    print(f"backend: {info.name} · model: {info.model} · max_iters={args.max_iters}"
          + (" · patch-only" if args.patch_only else ""))

    out_root = Path("games") / "eval_seed_edits"
    passed = 0
    for i, sc in enumerate(scenarios, 1):
        t0 = time.time()
        materialized, changed, err, trace = await _edit_once(
            backend_inst, sc["goal"], out_root / sc["name"], sc["name"],
            args.max_iters, patch_only=args.patch_only,
        )
        ok = changed
        passed += int(ok)
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        detail = "changed" if changed else ("materialized-but-identical" if materialized else "no-code")
        print(f"[{status}] {i:>2} {sc['name']:<18} {detail:<26} {dt:6.0f}s"
              + (f"  err={err}" if err else ""))
        if trace:
            print(f"        trace: {trace}")

    print(f"\n{passed}/{len(scenarios)} scenarios materialized a changed file")
    return 0 if passed == len(scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
