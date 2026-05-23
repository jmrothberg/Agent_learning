"""Phase 0.5 — structured patch_outcome trace events.

When `apply_patches` runs (via `_materialize`), the agent emits a
`patch_outcome` JSONL record so postmortems can reason about recurring
patch failure patterns across sessions. The 2026-05-22 chess trace had
patch failures that silently produced broken outputs — without structured
per-block detail in the trace, postmortems can't see WHICH searches keep
failing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )


def test_patch_outcome_trace_records_per_block_success_and_failure(tmp_path):
    a = _agent(tmp_path)
    a._current_file = (
        "<html><body><script>\n"
        "  const G = { score: 0 };\n"
        "  const TILE = 32;\n"
        "</script></body></html>\n"
    )
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "  const TILE = 32;\n"
        "=======\n"
        "  const TILE = 48;\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
        "<patch>\n"
        "<<<<<<< SEARCH\n"
        "  let kbCursor = {r:0,c:0};\n"
        "=======\n"
        "  let kbCursor = {r:7,c:4};\n"
        ">>>>>>> REPLACE\n"
        "</patch>\n"
    )

    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    html, _msg = asyncio.run(a._materialize(reply, dry_run=False))
    assert html is not None  # one patch applied

    patch_outcomes = [t for t in traces if t.get("kind") == "patch_outcome"]
    assert len(patch_outcomes) == 1
    rec = patch_outcomes[0]
    assert rec["total"] == 2
    assert rec["applied"] == 1
    assert rec["failed"] == 1
    blocks = rec["blocks"]
    assert len(blocks) == 2
    # First patch (TILE) applied; second (kbCursor — not in baseline) failed.
    assert blocks[0]["applied"] is True
    assert blocks[1]["applied"] is False
    assert "let kbCursor" in blocks[1]["search_head"]
    # No nearest anchor expected for `let kbCursor` because it doesn't
    # exist in the file at all (find_anchor walks SEARCH lines that DO
    # appear; here none do). But the failure entry must carry a reason.
    assert blocks[1]["reason"]


def test_patch_outcome_trace_dry_run_skipped(tmp_path):
    # Dry-run materializations (best-of-N candidate scoring) MUST NOT
    # emit patch_outcome traces — the chosen candidate emits its own.
    a = _agent(tmp_path)
    a._current_file = "<html><body><script>const X = 1;</script></body></html>\n"
    reply = (
        "<patch>\n"
        "<<<<<<< SEARCH\nconst X = 1;\n=======\nconst X = 2;\n>>>>>>> REPLACE\n"
        "</patch>\n"
    )
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    asyncio.run(a._materialize(reply, dry_run=True))

    assert not any(t.get("kind") == "patch_outcome" for t in traces)
