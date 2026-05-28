"""Tests for `GameAgent.revert_to_iter` — the `/revert` slash command's
agent-side implementation.

Background: audit of 10 recent traces (2026-05-25) showed harness gates
catch only ~5-10% of "model takes liberty beyond user scope" failures
across the full feedback shape. Rather than build more gates, the agent
got a user-controlled escape hatch: one keystroke rolls the on-disk
game file back to a previous iter's snapshot. These tests pin the
behavior so a future edit can't regress it.

Pattern follows tests/test_compaction.py:_make_agent — instantiates a
real GameAgent with stub browser and tmp_path-rooted paths; no Ollama,
no Chromium, no memory subsystem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html><body>initial</body></html>")
    fake_browser = MagicMock()
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _seed_snapshot(agent: GameAgent, iter_n: int, content: str) -> Path:
    agent.snapshots_dir.mkdir(parents=True, exist_ok=True)
    p = agent.snapshots_dir / f"iter_{iter_n:02d}.html"
    p.write_text(content, encoding="utf-8")
    return p


def _seed_iter_summary(agent: GameAgent, iter_n: int, ok: bool) -> None:
    agent.trace_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": "2026-05-25T00:00:00Z",
        "kind": "iter_summary",
        "iteration": iter_n,
        "ok": ok,
        "probes_passed": 5 if ok else 0,
        "probes_total": 5,
        "soft_warnings_count": 0,
        "page_errors_count": 0,
        "console_errors_count": 0,
        "frozen_canvas": False,
        "entity_missing_count": 0,
        "fail_reason": "ok" if ok else "broken",
    }
    with agent.trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_revert_to_last_clean_iter(tmp_path):
    """No arg → most-recent clean iter snapshot. Iters 1,3 clean; iter 5
    broken. revert_to_iter(None) lands on iter 3.
    """
    a = _make_agent(tmp_path)
    iter1 = "<html><body>iter1 clean</body></html>"
    iter2 = "<html><body>iter2 broken</body></html>"
    iter3 = "<html><body>iter3 clean</body></html>"
    iter5 = "<html><body>iter5 broken</body></html>"
    _seed_snapshot(a, 1, iter1)
    _seed_snapshot(a, 2, iter2)
    _seed_snapshot(a, 3, iter3)
    _seed_snapshot(a, 5, iter5)
    _seed_iter_summary(a, 1, ok=True)
    _seed_iter_summary(a, 2, ok=False)
    _seed_iter_summary(a, 3, ok=True)
    _seed_iter_summary(a, 5, ok=False)
    # Simulate the agent currently holding the broken iter-5 file.
    a.out_path.write_text(iter5, encoding="utf-8")
    a._current_file = iter5

    result = a.revert_to_iter(None)
    assert result["ok"] is True, result
    assert result["to_iter"] == 3
    assert result["source"] == "snapshot"
    assert a.out_path.read_text(encoding="utf-8") == iter3
    assert a._current_file == iter3


def test_revert_to_specific_iter(tmp_path):
    """Explicit iter number rolls back to that snapshot regardless of ok status."""
    a = _make_agent(tmp_path)
    _seed_snapshot(a, 1, "<html>one</html>")
    _seed_snapshot(a, 2, "<html>two</html>")
    _seed_snapshot(a, 3, "<html>three</html>")
    a.out_path.write_text("<html>broken</html>", encoding="utf-8")

    result = a.revert_to_iter(1)
    assert result["ok"] is True
    assert result["to_iter"] == 1
    assert result["source"] == "snapshot"
    assert a.out_path.read_text(encoding="utf-8") == "<html>one</html>"


def test_revert_falls_back_to_best_when_no_clean_iter(tmp_path):
    """No clean iter snapshot, but best.html exists → fall back to best."""
    a = _make_agent(tmp_path)
    _seed_snapshot(a, 1, "<html>broken1</html>")
    _seed_snapshot(a, 2, "<html>broken2</html>")
    _seed_iter_summary(a, 1, ok=False)
    _seed_iter_summary(a, 2, ok=False)
    best_content = "<html>best preserved</html>"
    a.best_path.parent.mkdir(parents=True, exist_ok=True)
    a.best_path.write_text(best_content, encoding="utf-8")
    a.out_path.write_text("<html>broken2</html>", encoding="utf-8")

    result = a.revert_to_iter(None)
    assert result["ok"] is True
    assert result["source"] == "best"
    assert result["to_iter"] is None
    assert a.out_path.read_text(encoding="utf-8") == best_content


def test_revert_errors_when_nothing_to_revert_to(tmp_path):
    """No snapshots, no best.html → clean error, no file change."""
    a = _make_agent(tmp_path)
    original = "<html>original on disk</html>"
    a.out_path.write_text(original, encoding="utf-8")

    result = a.revert_to_iter(None)
    assert result["ok"] is False
    assert "nothing to revert" in (result.get("error") or "").lower()
    # File on disk unchanged
    assert a.out_path.read_text(encoding="utf-8") == original


def test_revert_errors_when_requested_iter_missing(tmp_path):
    """`/revert 7` when iter 7 snapshot doesn't exist → clean error
    listing the iters that do exist.
    """
    a = _make_agent(tmp_path)
    _seed_snapshot(a, 1, "<html>one</html>")
    _seed_snapshot(a, 3, "<html>three</html>")
    original = "<html>current</html>"
    a.out_path.write_text(original, encoding="utf-8")

    result = a.revert_to_iter(7)
    assert result["ok"] is False
    err = (result.get("error") or "")
    assert "iter 7" in err
    # Error names what IS available so the user can pick another.
    assert "1" in err and "3" in err
    # File unchanged.
    assert a.out_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Side effects (state reset + trace event)
# ---------------------------------------------------------------------------


def test_revert_resets_stale_per_iter_state_flags(tmp_path):
    """After revert, recovery flags from the round-1/2 Wolfenstein fix
    bundle must be cleared — they referred to the iter that just got
    rewound away, so they're stale and would mislead the next turn.
    """
    a = _make_agent(tmp_path)
    _seed_snapshot(a, 1, "<html>one</html>")
    _seed_iter_summary(a, 1, ok=True)
    a.out_path.write_text("<html>broken</html>", encoding="utf-8")
    # Simulate the agent having queued a recovery for the broken iter
    a._context_pressure_pending = True
    a._context_pressure_streak = 2
    a._dead_first_build_pending = True
    a._format_stuck_streak = 3
    a._last_no_usable_code_fingerprint = "deadbeef"
    a._last_failed_patch_anchors = {"some-anchor"}
    a._pending_coaching = ["VISUAL CRITIC: facing wrong way"]
    a._previous_report_ok = False
    a._last_stream_silent = True

    result = a.revert_to_iter(None)
    assert result["ok"] is True
    # All stale recovery / coaching flags reset.
    assert a._context_pressure_pending is False
    assert a._context_pressure_streak == 0
    assert a._dead_first_build_pending is False
    assert a._format_stuck_streak == 0
    assert a._last_no_usable_code_fingerprint is None
    assert a._last_failed_patch_anchors == set()
    assert a._pending_coaching == []
    assert a._previous_report_ok is None
    assert a._last_stream_silent is False


def test_revert_emits_trace_event_with_required_fields(tmp_path):
    """`user_revert` trace event must carry from/to/source/path/bytes
    so postmortem analysis can correlate it with the failure it undid.
    """
    a = _make_agent(tmp_path)
    _seed_snapshot(a, 2, "<html>iter2 clean content</html>")
    _seed_iter_summary(a, 2, ok=True)
    a._last_tested_iter = 5
    a.out_path.write_text("<html>broken</html>", encoding="utf-8")

    result = a.revert_to_iter(None)
    assert result["ok"] is True

    # Read trace JSONL for the user_revert event
    events = []
    with a.trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "user_revert":
                events.append(obj)
    assert len(events) == 1, f"expected exactly one user_revert event, got {len(events)}"
    ev = events[0]
    assert ev["from_iter"] == 5
    assert ev["to_iter"] == 2
    assert ev["source"] == "snapshot"
    assert ev["source_path"].endswith("iter_02.html")
    assert ev["file_bytes"] == len("<html>iter2 clean content</html>")
