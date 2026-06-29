#!/usr/bin/env python3
"""Run run_07 Batch A (GLM, no VLM) then Batch B (Qwen, VLM) — one overnight session.

No wake-up between batches. Default: no pause between games (--wait-for-monitor 0).
Optional handoff: pass --wait-for-monitor 1800 + watcher with --auto-release.

  cd /Users/jonathanrothberg/Agent_learning
  bash eval/tune_run07_chain.sh

Or with log tee:
  bash eval/tune_run07_chain.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOOP = REPO / "eval" / "tune_serial_loop.py"
PYTHON = REPO / ".venv" / "bin" / "python"

BIG_OUT = REPO / "games" / "tune_serial10" / "run_07_big"
VLM_OUT = REPO / "games" / "tune_serial10" / "run_07_vlm"
CHAIN_META = REPO / "games" / "tune_serial10" / "run_07" / "chain_status.json"

DEFAULT_GLM = os.path.expanduser("~/MLX_Models/GLM-5.2-MLX-4bit")
DEFAULT_QWEN = os.path.expanduser("~/MLX_Models/Qwen3.6-27B-mxfp8")


def _completed(out_dir: Path) -> int:
    ck = out_dir / "tune_checkpoint.json"
    if not ck.is_file():
        return 0
    try:
        d = json.loads(ck.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    labels = d.get("completed_labels") or []
    return int(d.get("completed_count") or len(labels))


def _run_batch(
    *,
    name: str,
    goals_file: Path,
    out_dir: Path,
    model: str,
    no_vlm: bool,
    max_iters: int,
    wait_for_monitor: float,
    resume: bool,
    best_of_n: int,
    dry_run: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(LOOP),
        "--goals-file",
        str(goals_file),
        "--out-dir",
        str(out_dir),
        "--model",
        model,
        "--max-iters",
        str(max_iters),
        "--best-of-n",
        str(best_of_n),
        "--wait-for-monitor",
        str(wait_for_monitor),
    ]
    if no_vlm:
        cmd.append("--no-vlm-critique")
    if resume:
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")

    print(
        f"\n{'=' * 60}\n"
        f"run_07 chain — batch {name}\n"
        f"  goals: {goals_file.relative_to(REPO)}\n"
        f"  out:   {out_dir.relative_to(REPO)}\n"
        f"  model: {model}\n"
        f"  vlm:   {'OFF' if no_vlm else 'ON'}\n"
        f"  max_iters: {max_iters}\n"
        f"{'=' * 60}\n",
        flush=True,
    )
    if dry_run:
        print(" ".join(cmd))
        return 0

    env = os.environ.copy()
    env["LLM_BACKEND"] = "mlx"
    env["MLX_MODEL"] = model
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(cmd, cwd=str(REPO), env=env)


def _write_chain_status(active: str, *, big_done: int, vlm_done: int) -> None:
    CHAIN_META.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active_batch": active,
        "big": {"out_dir": str(BIG_OUT.relative_to(REPO)), "completed": big_done, "jobs_total": 6},
        "vlm": {"out_dir": str(VLM_OUT.relative_to(REPO)), "completed": vlm_done, "jobs_total": 5},
    }
    CHAIN_META.write_text(json.dumps(body, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glm-model", default=DEFAULT_GLM)
    ap.add_argument("--qwen-model", default=DEFAULT_QWEN)
    ap.add_argument(
        "--wait-for-monitor",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Inter-game handoff wait (both batches). 0=no pause (default).",
    )
    ap.add_argument("--max-iters-big", type=int, default=6)
    ap.add_argument("--max-iters-vlm", type=int, default=2)
    ap.add_argument("--best-of-n", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force-big",
        action="store_true",
        help="Re-run Batch A even if checkpoint already 6/6",
    )
    args = ap.parse_args()
    resume = not args.no_resume

    if not PYTHON.is_file():
        print("ERROR: .venv missing — run ./scripts/setup.sh", file=sys.stderr)
        return 1

    big_done = _completed(BIG_OUT)
    vlm_done = _completed(VLM_OUT)

    run_big = big_done < 6 or args.force_big
    rc = 0

    if run_big:
        _write_chain_status("big", big_done=big_done, vlm_done=vlm_done)
        rc = _run_batch(
            name="A (GLM, no VLM)",
            goals_file=REPO / "eval" / "tune_run07_big.txt",
            out_dir=BIG_OUT,
            model=args.glm_model,
            no_vlm=True,
            max_iters=args.max_iters_big,
            wait_for_monitor=args.wait_for_monitor,
            resume=resume,
            best_of_n=args.best_of_n,
            dry_run=args.dry_run,
        )
        big_done = _completed(BIG_OUT)
        if rc != 0 and big_done < 6:
            print(
                f"warning: Batch A exited {rc} with {big_done}/6 complete — "
                f"continuing to Batch B anyway",
                file=sys.stderr,
            )

    _write_chain_status("vlm", big_done=big_done, vlm_done=vlm_done)
    rc_b = _run_batch(
        name="B (Qwen, VLM ON)",
        goals_file=REPO / "eval" / "tune_run07_vlm.txt",
        out_dir=VLM_OUT,
        model=args.qwen_model,
        no_vlm=False,
        max_iters=args.max_iters_vlm,
        wait_for_monitor=args.wait_for_monitor,
        resume=resume,
        best_of_n=args.best_of_n,
        dry_run=args.dry_run,
    )
    vlm_done = _completed(VLM_OUT)
    _write_chain_status("done", big_done=big_done, vlm_done=vlm_done)

    print(
        f"\nrun_07 chain finished — Batch A {big_done}/6 · Batch B {vlm_done}/5",
        flush=True,
    )
    return 0 if big_done >= 6 and vlm_done >= 5 and rc_b == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
