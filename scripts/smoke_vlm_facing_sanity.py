#!/usr/bin/env python3
"""Preflight: can the local VLM detect fighters NOT facing each other?

Loads eval/fixtures/seed_fighters_facing_bug.html (intentional facing bug),
screenshots it, asks the facing question. **PASS (exit 0)** only when the VLM
answers NO — i.e. it sees the bug. If it answers YES, the model is not a
trustworthy facing judge for eval or post-run Q4 gates.

Run before eval/eval_vlm_facing_fix.py when testing a new VLM:

    MLX_MODEL=~/MLX_Models/<your-vlm> \\
      .venv/bin/python scripts/smoke_vlm_facing_sanity.py

Exits 0 = sanity pass, 1 = VLM failed to detect bug, 2 = missing deps/fixture.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from eval.vlm_facing_sanity import FACING_Q, run_vlm_facing_sanity  # noqa: E402

_FIXTURE = _REPO / "eval" / "fixtures" / "seed_fighters_facing_bug.html"


async def _main() -> int:
    if not _FIXTURE.is_file():
        print(f"fixture missing: {_FIXTURE}", file=sys.stderr)
        return 2
    print(f"facing sanity: {FACING_Q}")
    print(f"fixture: {_FIXTURE}")
    rep = await run_vlm_facing_sanity(_FIXTURE, headless=True)
    print(f"model: {rep.get('model_path')}")
    print(f"q_answer: {rep.get('q_answer')!r}  (want 'no' on seed bug)")
    if rep.get("raw_preview"):
        print(f"raw: {rep.get('raw_preview')!r}")
    if rep.get("error"):
        print(f"error: {rep['error']}", file=sys.stderr)
    err = (rep.get("error") or "").lower()
    if not rep.get("q_answer") and ("playwright" in err or "executable doesn't exist" in err):
        print("hint: run `playwright install` (or env -u PLAYWRIGHT_BROWSERS_PATH)", file=sys.stderr)
        return 2
    if rep.get("sanity_ok"):
        print("SANITY PASS — VLM can detect facing wrong on seed bug")
        return 0
    print(
        "SANITY FAIL — VLM did not reject the seed bug; do not trust facing Q4 / eval PRIMARY",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
