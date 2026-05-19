"""Regression tests for blocker-first feedback routing.

When a browser/micro-probe blocker is active, fresh user feedback should not
compete with the failing test report. This protects local models from chasing
the newest visual request while input/rendering is still broken.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent_stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._fix_mode = True
    a._previous_report_ok = False
    a._previous_report = {
        "ok": False,
        "soft_warnings": [
            "PROBE FAILED [input_responsive]: keyboard input must produce a visible canvas change",
        ],
    }
    a._pending_answer = None
    a._pending_feedback = ["change the pointer for the player; it is wrong"]
    a._pending_bullet_lookups = []
    a._pending_probe_quarantine_notices = []
    a._pending_coaching = []
    a._last_drained_feedback = []
    a._token_cb = None
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    return a


def test_active_blocker_defers_feedback_without_consuming_queue() -> None:
    a = _agent_stub()

    prompt = a._flush_user_injections("FIX THE FAILING INPUT REPORT")

    assert "BLOCKER-FIRST FEEDBACK DEFERRAL" in prompt
    assert "FIX THE FAILING INPUT REPORT" in prompt
    assert "USER FEEDBACK (HIGHEST PRIORITY)" not in prompt
    assert a._pending_feedback == ["change the pointer for the player; it is wrong"]
    assert a._last_drained_feedback == []
    assert a._last_turn_contract["blocker_feedback_deferred"] is True
    assert any(e.get("kind") == "feedback_deferred_blocker" for e in a._trace_events)


def test_explicit_blocker_override_is_not_deferred() -> None:
    a = _agent_stub()
    a._pending_feedback = ["ignore the test failure and change the pointer"]

    assert a._should_defer_feedback_for_blocker() is False
