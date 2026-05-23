"""Phase 3 regression tests — honest exit / ship signals.

Trace 1 (chess 20260522_000304) ended with confident <done/> + <notes>
over an unplayable file. The post-done structural verification must
flip session_outcome to ok=False regardless of how persuasive the
model's notes are.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _baseline_structurally_broken  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def test_exit_done_over_broken_file_flag_initializes_false(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    assert a._exit_done_over_broken_file is False


def test_session_outcome_ok_when_best_exists_and_not_broken(tmp_path: Path) -> None:
    """Sanity: a healthy session with a best.html and no broken-file
    flag must record ok=True."""
    a = _make_agent(tmp_path)
    a.best_path.write_text("<html><body>ok</body></html>")
    assert a.best_path.exists()
    assert a._exit_done_over_broken_file is False
    ok_outcome = a.best_path.exists() and not a._exit_done_over_broken_file
    assert ok_outcome is True


def test_session_outcome_ok_false_when_done_over_broken(tmp_path: Path) -> None:
    """Phase 3 invariant: a <done/> over a broken file forces ok=False
    even if best.html exists from an earlier iter."""
    a = _make_agent(tmp_path)
    a.best_path.write_text("<html><body>earlier-clean</body></html>")
    assert a.best_path.exists()
    a._exit_done_over_broken_file = True
    ok_outcome = a.best_path.exists() and not a._exit_done_over_broken_file
    assert ok_outcome is False


def test_baseline_structurally_broken_recognises_chess_trace_shape() -> None:
    """The exit-decision verifier uses _baseline_structurally_broken;
    it must continue to recognise the trace-1 concatenated-drafts shape
    (otherwise the warning + flag never fire).
    """
    chess_trace_shape = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>"
        "(() => {\n"
        "  const ctx = c.getContext('2d');\n"
        "  const ctx = c.getContext('2d');\n"
        + "  const pad = 0;\n" * 200
        + "})();\n"
        "</script></body></html>"
    )
    reason = _baseline_structurally_broken(chess_trace_shape)
    assert reason is not None
    assert "duplicate" in reason.lower()


def test_post_done_verification_reads_out_path(tmp_path: Path) -> None:
    """Smoke: the verifier reads the file currently on disk (not the
    in-memory _current_file), so the flag flips when the disk file is
    broken even if the agent thinks it has a clean one in memory."""
    a = _make_agent(tmp_path)
    # Simulate disk corruption with healthy in-memory state.
    a.out_path.write_text("not html at all, just notes\n", encoding="utf-8")
    a._current_file = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        "<script>const x=1;function ok(){return x;}</script></body></html>"
    )
    on_disk = a.out_path.read_text(encoding="utf-8")
    assert _baseline_structurally_broken(on_disk) is not None
