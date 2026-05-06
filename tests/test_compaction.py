"""Tests for the pi-mono-style structured compaction in agent.py.

We don't run a real model — we instantiate GameAgent with a stub browser
and exercise _build_structured_summary / _prune_messages directly with
synthetic state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _PRUNE_KEEP_RECENT_TURNS, _STRUCTURED_PRUNE_THRESHOLD  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    """Build a minimal GameAgent instance for state-only tests.

    We avoid touching Ollama, the browser, or the memory subsystem by
    pointing all paths at tmp_path.
    """
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    fake_browser = MagicMock()
    agent = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    return agent


def test_structured_summary_minimal(tmp_path):
    """Empty session — summary must still produce all required sections."""
    a = _make_agent(tmp_path)
    a._goal = "Build asteroids"
    summary = a._build_structured_summary()
    assert "## Goal" in summary
    assert "Build asteroids" in summary
    assert "## Progress" in summary
    assert "not yet built" in summary
    assert "## Critical context" in summary
    assert "source of truth" in summary


def test_structured_summary_includes_criteria(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "Snake"
    a._criteria = "Basic: snake moves\nEdge: wraps screen edges"
    summary = a._build_structured_summary()
    assert "## Acceptance criteria" in summary
    assert "snake moves" in summary
    assert "wraps screen edges" in summary


def test_structured_summary_progress_states(tmp_path):
    """Snapshot N > 0 with previous_report_ok True/False/None."""
    a = _make_agent(tmp_path)
    a._goal = "Pong"
    a._snapshot_n = 3

    a._previous_report_ok = True
    s = a._build_structured_summary()
    assert "iteration 3: PASSED" in s

    a._previous_report_ok = False
    s = a._build_structured_summary()
    assert "iteration 3: FAILING" in s

    a._previous_report_ok = None
    s = a._build_structured_summary()
    assert "iteration 3: status unknown" in s


def test_structured_summary_stuck_streak_visible(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "x"
    a._snapshot_n = 4
    a._previous_report_ok = False
    a._stuck_streak = 3
    s = a._build_structured_summary()
    assert "stuck-streak: 3" in s


def test_structured_summary_includes_diagnose_and_report(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "x"
    a._snapshot_n = 2
    a._previous_report_ok = False
    a._last_diagnose = "keyup handler clears the wrong slot"
    a._last_report_summary = "TEST FAILED — 1 console error: TypeError at line 42"
    s = a._build_structured_summary()
    assert "## Key decisions" in s
    assert "keyup handler" in s
    assert "## Last test report" in s
    assert "TypeError at line 42" in s


def test_prune_messages_no_op_when_short(tmp_path):
    """Below the keep-recent threshold, prune is a no-op."""
    a = _make_agent(tmp_path)
    a._messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    before = list(a._messages)
    a._prune_messages()
    assert a._messages == before


def test_prune_messages_default_elision_path(tmp_path):
    """Between KEEP+1 and STRUCTURED threshold: HTML in older turns is
    replaced with size markers; message count stays the same."""
    a = _make_agent(tmp_path)
    msgs = [{"role": "system", "content": "sys"}]
    # Build enough messages to exceed KEEP_RECENT but stay under structured.
    big_html = "<html_file>" + ("a" * 5000) + "</html_file>"
    for i in range(_PRUNE_KEEP_RECENT_TURNS + 3):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"turn {i} {big_html}"})
    a._messages = msgs
    n_before = len(a._messages)
    a._prune_messages()
    assert len(a._messages) == n_before  # elision keeps shape
    # Older turns no longer carry the inline HTML.
    older = a._messages[1:1 + (n_before - 1 - _PRUNE_KEEP_RECENT_TURNS)]
    for m in older:
        assert "[omitted:" in m["content"]
    # Most-recent K turns untouched.
    recent = a._messages[-_PRUNE_KEEP_RECENT_TURNS:]
    for m in recent:
        assert "[omitted:" not in m["content"]


def test_prune_messages_structured_path(tmp_path):
    """Above the structured threshold: messages 1..cutoff get replaced by
    ONE state-anchor user message; system + last K turns survive."""
    a = _make_agent(tmp_path)
    a._goal = "Make asteroids"
    a._snapshot_n = 5
    a._previous_report_ok = False
    a._stuck_streak = 2
    a._last_report_summary = "TEST FAILED: 2 errors"

    msgs = [{"role": "system", "content": "sys-original"}]
    n_extra = _STRUCTURED_PRUNE_THRESHOLD + 3
    for i in range(n_extra):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"turn-{i}"})
    a._messages = msgs
    a._prune_messages()

    # System message preserved at index 0.
    assert a._messages[0]["content"] == "sys-original"
    # Index 1 is the state anchor.
    anchor = a._messages[1]
    assert anchor["role"] == "user"
    assert "STATE ANCHOR" in anchor["content"]
    assert "Make asteroids" in anchor["content"]
    assert "stuck-streak: 2" in anchor["content"]
    assert "TEST FAILED: 2 errors" in anchor["content"]
    # Last K turns are the original last K.
    expected_recent = msgs[-_PRUNE_KEEP_RECENT_TURNS:]
    actual_recent = a._messages[-_PRUNE_KEEP_RECENT_TURNS:]
    assert actual_recent == expected_recent
    # Total count: system + anchor + last K = 2 + K.
    assert len(a._messages) == 2 + _PRUNE_KEEP_RECENT_TURNS


def test_structured_summary_critical_context_always_present(tmp_path):
    """Whatever the state, the 'source of truth' rule must appear so the
    model never loses the patch-against-current-file invariant."""
    a = _make_agent(tmp_path)
    s = a._build_structured_summary()
    assert "source of truth" in s
    assert "patch against" in s.lower()
