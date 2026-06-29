#!/usr/bin/env python3
"""Release tune_serial_loop after inter-game triage/fixes (no Terminal Enter).

Run from Cursor after you patch code/memory/prompts for the game in pending:
  .venv/bin/python eval/tune_inter_game_ready.py \\
      --out-dir games/tune_serial10/run_07_big \\
      --note "playbook: added chaser spawn bullet"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval.inter_game_sync import load_pending, write_ready  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="Batch out dir (same as tune_serial_loop)")
    ap.add_argument("--note", default="", help="What you fixed (logged in ready file)")
    ap.add_argument(
        "--skip-if-no-pending",
        action="store_true",
        help="Exit 0 when no pending (idempotent for monitor auto-release)",
    )
    args = ap.parse_args()
    out_dir = (REPO / args.out_dir).resolve()
    pending = load_pending(out_dir)
    if not pending:
        if args.skip_if_no_pending:
            return 0
        print(f"no {out_dir}/inter_game_pending.json — nothing to release", file=sys.stderr)
        return 1
    path = write_ready(out_dir, note=args.note, released_by="agent")
    print(f"released next game → {path}")
    if args.note:
        print(f"  note: {args.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
