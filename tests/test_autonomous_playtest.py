"""Phase 1.5 — autonomous self-feedback loop.

Covers four things, all general (no genre logic):
  1. /playtest dispatcher (alias /feedback): on / off / bare state-print
  2. The 3 seed recipes' applicability gates + check_kind evaluators
  3. Budget governor: hard cap + no-findings auto-stop
  4. End-to-end loop: when a finding fires, feedback lands in
     _pending_feedback with the [AUTONOMOUS PLAYTEST] marker
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from chat import CodingBoxApp  # noqa: E402


# ---------------------------------------------------------------------------
# /playtest dispatcher (alias /feedback)
# ---------------------------------------------------------------------------


def _bare_app() -> CodingBoxApp:
    app = CodingBoxApp()
    app.agent = None
    app._update_status = MagicMock()  # type: ignore[assignment]
    app._log_info = MagicMock()  # type: ignore[assignment]
    app._log_error = MagicMock()  # type: ignore[assignment]
    return app


def test_feedback_command_default_on():
    app = _bare_app()
    assert app._use_autonomous_feedback is True


def test_feedback_off_disables_loop():
    app = _bare_app()
    app._cmd_toggle_autonomous_feedback("off")
    assert app._use_autonomous_feedback is False


def test_feedback_on_re_enables_loop():
    app = _bare_app()
    app._use_autonomous_feedback = False
    app._cmd_toggle_autonomous_feedback("on")
    assert app._use_autonomous_feedback is True


def test_feedback_bare_does_not_flip_state():
    # Autonomous mode has bigger consequences than a vlm-critique toggle —
    # bare /critique should print state, not toggle. Different from
    # /vlm-critique by design.
    app = _bare_app()
    app._use_autonomous_feedback = True
    app._cmd_toggle_autonomous_feedback("")
    assert app._use_autonomous_feedback is True  # unchanged

    app._use_autonomous_feedback = False
    app._cmd_toggle_autonomous_feedback("")
    assert app._use_autonomous_feedback is False  # unchanged


def test_feedback_off_propagates_to_active_agent():
    app = _bare_app()
    # Simulate an active agent.
    agent_stub = MagicMock()
    agent_stub._use_autonomous_feedback = True
    app.agent = agent_stub
    app._cmd_toggle_autonomous_feedback("off")
    assert agent_stub._use_autonomous_feedback is False


def test_critique_and_aliases_dispatch_to_no_vision_reviewer():
    """/critique is the primary name; /playtest and /feedback are silent
    aliases. All three must route to the no-vision reviewer toggle."""
    import asyncio

    for name in ("critique", "playtest", "feedback"):
        app = _bare_app()
        app._log = lambda *a, **k: None  # type: ignore[assignment]
        app._use_autonomous_feedback = True
        asyncio.run(app._handle_slash(f"/{name} off"))
        assert app._use_autonomous_feedback is False, name


# ---------------------------------------------------------------------------
# Recipe applicability + check evaluators
# ---------------------------------------------------------------------------


def _agent_stub(tmp_path: Path | None = None) -> GameAgent:
    """A GameAgent stub with just enough state for the playtest
    evaluator (no live browser, no backend)."""
    a = GameAgent.__new__(GameAgent)
    a._user_force_done = False
    a._use_autonomous_feedback = True
    a._autonomous_playtest_cycle = 0
    a._autonomous_no_findings_streak = 0
    a._pending_feedback = []
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    a.browser = None
    a._record = lambda ev: ev
    return a


def test_any_progress_check_passes_when_state_changed():
    a = _agent_stub()
    timeline = {
        "samples": [
            {"canvas_hash": "aaa", "state": {"score": 0}},
            {"canvas_hash": "aaa", "state": {"score": 10}},
            {"canvas_hash": "aaa", "state": {"score": 20}},
        ],
    }
    recipe = {"check_kind": "any_progress", "finding_label": "frozen"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is None


def test_any_progress_check_fires_when_nothing_changes():
    a = _agent_stub()
    same_state = {"score": 0, "x": 100}
    timeline = {
        "samples": [
            {"canvas_hash": "aaa", "state": dict(same_state)},
            {"canvas_hash": "aaa", "state": dict(same_state)},
            {"canvas_hash": "aaa", "state": dict(same_state)},
        ],
    }
    recipe = {"check_kind": "any_progress", "finding_label": "frozen"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is not None
    assert "frozen" in finding["finding_label"]


def test_facing_matches_movement_passes_when_aligned():
    # Player facing 0 rad (east), moves east → aligned.
    a = _agent_stub()
    timeline = {
        "samples": [
            {"state": {"player.x": 100.0, "player.y": 100.0, "player.facing": 0.0}},
            {"state": {"player.x": 150.0, "player.y": 100.0, "player.facing": 0.0}},
        ],
    }
    recipe = {"check_kind": "facing_matches_movement", "finding_label": "doom-bug"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is None


def test_facing_matches_movement_fires_on_doom_bug():
    # Player facing PI/2 rad (north on math axis = up on screen),
    # but moves east (world-X) → mismatch ~90°.
    import math
    a = _agent_stub()
    timeline = {
        "samples": [
            {"state": {"player.x": 100.0, "player.y": 100.0, "player.facing": math.pi / 2}},
            {"state": {"player.x": 150.0, "player.y": 100.0, "player.facing": math.pi / 2}},
        ],
    }
    recipe = {"check_kind": "facing_matches_movement", "finding_label": "doom-bug"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is not None
    assert "facing" in finding["evidence"].lower()


def test_facing_check_skips_when_movement_too_small():
    a = _agent_stub()
    timeline = {
        "samples": [
            {"state": {"player.x": 100.0, "player.y": 100.0, "player.facing": 0.0}},
            {"state": {"player.x": 101.0, "player.y": 100.0, "player.facing": 0.0}},
        ],
    }
    recipe = {"check_kind": "facing_matches_movement", "finding_label": "doom-bug"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    # 1-pixel delta is below _AUTONOMOUS_MIN_MOVE_PX (6.0) — skipped, not fired.
    assert finding is None


def test_stays_in_canvas_fires_when_player_leaves():
    a = _agent_stub()
    timeline = {
        "samples": [
            {"state": {"player.x": 100.0, "player.y": 100.0}},
            {"state": {"player.x": 99999.0, "player.y": 100.0}},
        ],
    }
    recipe = {"check_kind": "stays_in_canvas", "finding_label": "off-canvas"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is not None


def test_stays_in_canvas_passes_when_player_in_bounds():
    a = _agent_stub()
    timeline = {
        "samples": [
            {"state": {"player.x": 100.0, "player.y": 100.0}},
            {"state": {"player.x": 800.0, "player.y": 600.0}},
        ],
    }
    recipe = {"check_kind": "stays_in_canvas", "finding_label": "off-canvas"}
    finding = a._evaluate_behavior_playtest_check(recipe, timeline)
    assert finding is None


# ---------------------------------------------------------------------------
# Budget governor
# ---------------------------------------------------------------------------


def test_budget_governor_caps_cycles():
    a = _agent_stub()
    a._autonomous_playtest_cycle = a._AUTONOMOUS_MAX_CYCLES
    disabled, reason = a._autonomous_playtest_disabled()
    assert disabled is True
    assert reason == "budget_exhausted"


def test_budget_governor_stops_after_two_no_findings():
    a = _agent_stub()
    a._autonomous_no_findings_streak = 2
    disabled, reason = a._autonomous_playtest_disabled()
    assert disabled is True
    assert reason == "no_findings_streak"


def test_force_done_disables_autonomous_loop():
    a = _agent_stub()
    a._user_force_done = True
    disabled, reason = a._autonomous_playtest_disabled()
    assert disabled is True
    assert reason == "force_done"


def test_feedback_off_disables_autonomous_loop():
    a = _agent_stub()
    a._use_autonomous_feedback = False
    disabled, reason = a._autonomous_playtest_disabled()
    assert disabled is True
    assert reason == "feedback_off"


def test_default_state_does_not_disable_loop():
    a = _agent_stub()
    disabled, _ = a._autonomous_playtest_disabled()
    assert disabled is False


# ---------------------------------------------------------------------------
# End-to-end: when a finding fires, feedback lands in _pending_feedback
# with the [AUTONOMOUS PLAYTEST] marker.
# ---------------------------------------------------------------------------


class _MockBrowser:
    """Minimal LiveBrowser stand-in. Returns canned timelines so we can
    drive `_run_autonomous_playtest` without Chromium."""

    def __init__(self, applies_when_returns: bool, timeline: dict) -> None:
        self._applies = applies_when_returns
        self._timeline = timeline

    async def _safe_eval(self, _js: str):
        return self._applies

    async def record_playtest(self, **kwargs):
        return self._timeline


def test_autonomous_loop_queues_feedback_when_finding_fires():
    a = _agent_stub()
    a.browser = _MockBrowser(
        applies_when_returns=True,
        timeline={
            "samples": [
                {"canvas_hash": "aaa", "state": {"score": 0}},
                {"canvas_hash": "aaa", "state": {"score": 0}},
                {"canvas_hash": "aaa", "state": {"score": 0}},
            ],
            "errors": [],
        },
    )

    async def _drive():
        events = []
        async for ev in a._run_autonomous_playtest(iteration=1, report={"ok": True}):
            events.append(ev)
        return events

    events = asyncio.run(_drive())
    # A cycle ran.
    assert a._autonomous_playtest_cycle == 1
    # Feedback was queued.
    assert len(a._pending_feedback) == 1
    assert "[AUTONOMOUS PLAYTEST]" in a._pending_feedback[0]
    # And the trace recorded the summary.
    assert any(
        t.get("kind") == "autonomous_playtest_summary" and t.get("findings", 0) >= 1
        for t in a._trace_events
    )


def test_autonomous_loop_increments_no_finding_streak_when_clean():
    a = _agent_stub()
    # This test exercises the no-findings STREAK gate across multiple
    # cycles; raise the per-session cycle cap (production default is now 1
    # — post-green polish cap, run_10) so the budget gate doesn't trip first.
    a._AUTONOMOUS_MAX_CYCLES = 3
    a.browser = _MockBrowser(
        applies_when_returns=False,  # all recipes skip → no findings
        timeline={"samples": [], "errors": []},
    )

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=1, report={"ok": True}):
            pass

    asyncio.run(_drive())
    assert a._pending_feedback == []
    assert a._autonomous_no_findings_streak == 1
    # Second clean run trips the streak limit.
    asyncio.run(_drive())
    assert a._autonomous_no_findings_streak == 2
    # A third call is now no-op because the streak gate is closed.
    asyncio.run(_drive())
    assert a._autonomous_playtest_cycle == 2  # NOT incremented


def test_autonomous_loop_skipped_on_failed_iter():
    a = _agent_stub()
    a.browser = _MockBrowser(
        applies_when_returns=True,
        timeline={"samples": [], "errors": []},
    )

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=1, report={"ok": False}):
            pass

    asyncio.run(_drive())
    assert a._autonomous_playtest_cycle == 0
    assert a._pending_feedback == []
