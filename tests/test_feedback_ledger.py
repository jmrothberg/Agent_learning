"""Tests for the feedback-ledger round (plan feedback_ledger_and_fps_gate_fix).

Trace-backed fixes from run build-a-first-person-3d-shoote_20260611_163325:

  1. Feedback ledger — session-scoped queued → applying → applied record
     rendered in the status panel (the old "Queued (N)" section was empty
     within milliseconds, so the user never saw their prompts acknowledged).
  2. Extension-path acknowledgment — feedback typed after <done/> bypasses
     agent._pending_feedback entirely; it now gets a ledger entry too.
  3. status_snapshot observability — pending_feedback / ledger_tail fields,
     included in the dedupe signature so queue changes write a row.
  4. Trace conciseness — static goal/files fields carried forward instead
     of repeated on every snapshot row.
  5. FPS yaw false positive — z-axis leaves are position leaves, and the
     control-recovery recheck accepts ANY position-leaf change (W moves
     relative to facing in first-person games).

Pure-function / source-pinned tests — no model, no Chromium.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402
from agent import AgentEvent, GameAgent  # noqa: E402
from chat import CodingBoxApp  # noqa: E402
from tools import _is_position_leaf  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=4,
        memory_root=str(tmp_path / "memory"),
    )


def _app_with_ledger() -> CodingBoxApp:
    app = CodingBoxApp()
    app.agent = None
    return app


# ---------------------------------------------------------------------------
# 1. Feedback ledger — transitions
# ---------------------------------------------------------------------------

def test_reconcile_flips_drained_to_applying():
    app = _app_with_ledger()
    app.agent = MagicMock()
    app.agent._pending_feedback = []  # the agent drained the item
    app._iteration_label = "3/6"
    app._feedback_ledger = [
        {"text": "fix the gun", "status": "queued", "iter": None, "ok": None},
    ]
    app._reconcile_feedback_ledger()
    assert app._feedback_ledger[0]["status"] == "applying"
    assert app._feedback_ledger[0]["iter"] == "3/6"


def test_reconcile_keeps_pending_items_queued():
    app = _app_with_ledger()
    app.agent = MagicMock()
    app.agent._pending_feedback = ["still waiting"]
    app._feedback_ledger = [
        {"text": "still waiting", "status": "queued", "iter": None, "ok": None},
    ]
    app._reconcile_feedback_ledger()
    assert app._feedback_ledger[0]["status"] == "queued"


def test_reconcile_noop_without_agent_or_ledger():
    app = _app_with_ledger()
    app.agent = None
    app._feedback_ledger = [
        {"text": "x", "status": "queued", "iter": None, "ok": None},
    ]
    app._reconcile_feedback_ledger()  # must not raise
    assert app._feedback_ledger[0]["status"] == "queued"
    app.agent = MagicMock()
    app._feedback_ledger = []
    app._reconcile_feedback_ledger()  # must not raise on empty ledger


def test_test_event_flips_applying_to_applied():
    app = _app_with_ledger()
    app.agent = MagicMock()
    app.agent._pending_feedback = []
    app._iteration_label = "4/6"
    app._iter_decision_verbose = False
    app._feedback_ledger = [
        {"text": "move pistol up", "status": "applying", "iter": "4/6", "ok": None},
        {"text": "already done", "status": "applied", "iter": "2/6", "ok": True},
    ]
    # Stub UI-touching methods so _handle_event runs headless.
    app._flush_stream = MagicMock()
    app._update_status = MagicMock()
    app._log = MagicMock()
    app._maybe_trigger_profile_review = MagicMock()
    ev = AgentEvent("test", "report text", {
        "ok": True, "errors": [], "soft_warnings": [],
        "probes": [{"name": "p1", "ok": True}],
    })
    app._handle_event(ev)
    assert app._feedback_ledger[0]["status"] == "applied"
    assert app._feedback_ledger[0]["ok"] is True
    assert app._feedback_ledger[0]["iter"] == "4/6"
    # Prior applied entries are untouched.
    assert app._feedback_ledger[1]["iter"] == "2/6"


def test_test_event_records_failing_checks():
    app = _app_with_ledger()
    app.agent = MagicMock()
    app.agent._pending_feedback = []
    app._iteration_label = "5/6"
    app._iter_decision_verbose = False
    app._feedback_ledger = [
        {"text": "fb", "status": "applying", "iter": None, "ok": None},
    ]
    app._flush_stream = MagicMock()
    app._update_status = MagicMock()
    app._log = MagicMock()
    app._maybe_trigger_profile_review = MagicMock()
    ev = AgentEvent("test", "report", {
        "ok": False, "errors": [], "soft_warnings": ["W"],
        "probes": [{"name": "p1", "ok": False}],
    })
    app._handle_event(ev)
    assert app._feedback_ledger[0]["status"] == "applied"
    assert app._feedback_ledger[0]["ok"] is False


def test_reset_status_state_clears_ledger():
    app = _app_with_ledger()
    app._feedback_ledger = [
        {"text": "x", "status": "applied", "iter": "1/6", "ok": True},
    ]
    app._reset_status_state()
    assert app._feedback_ledger == []


def test_intake_site_appends_queued_entry():
    # Source-pin: on_input_submitted appends a queued ledger entry right
    # after agent.add_user_feedback(text).
    src = inspect.getsource(CodingBoxApp.on_input_submitted)
    idx = src.index("add_user_feedback(text)")
    after = src[idx:idx + 600]
    assert "_feedback_ledger.append" in after
    assert '"queued"' in after


# ---------------------------------------------------------------------------
# 1b. Feedback ledger — status panel rendering
# ---------------------------------------------------------------------------

def test_render_includes_ledger_section():
    app = _app_with_ledger()
    app._feedback_ledger = [
        {"text": "make the gun bigger", "status": "queued", "iter": None, "ok": None},
    ]
    out = app._render_iteration_block()
    assert "Feedback (1):" in out
    assert "queued" in out
    assert "make the gun bigger" in out


def test_render_ledger_status_badges():
    app = _app_with_ledger()
    app._feedback_ledger = [
        {"text": "a" * 80, "status": "applied", "iter": "7/6", "ok": True},
        {"text": "b", "status": "applied", "iter": "8/6", "ok": False},
        {"text": "c", "status": "applying", "iter": None, "ok": None},
    ]
    out = app._render_iteration_block()
    assert "✓ applied (iter 7/6)" in out
    assert "△ applied (iter 8/6, checks failing)" in out
    assert "→ applying" in out
    # Long texts are previewed, not dumped wholesale.
    assert "a" * 80 not in out
    assert "a" * 60 + "…" in out


def test_render_ledger_caps_at_last_four():
    app = _app_with_ledger()
    app._feedback_ledger = [
        {"text": f"item {i}", "status": "applied", "iter": str(i), "ok": True}
        for i in range(6)
    ]
    out = app._render_iteration_block()
    assert "Feedback (6):" in out
    assert "item 0" not in out and "item 1" not in out
    assert "item 2" in out and "item 5" in out


def test_render_without_ledger_has_no_section():
    app = _app_with_ledger()
    app._feedback_ledger = []
    out = app._render_iteration_block()
    assert "Feedback (" not in out


# ---------------------------------------------------------------------------
# 2. Extension-path acknowledgment
# ---------------------------------------------------------------------------

def test_extend_session_appends_applying_entry():
    # Source-pin: _extend_session records the feedback as already
    # "applying" (it bypasses agent._pending_feedback) and acknowledges
    # receipt in the log like the queued path does.
    src = inspect.getsource(CodingBoxApp._extend_session)
    assert "_feedback_ledger.append" in src
    assert '"applying"' in src
    assert "received" in src


# ---------------------------------------------------------------------------
# 3 + 4. status_snapshot observability + conciseness (agent.trace_status)
# ---------------------------------------------------------------------------

def _snapshot_rows(agent: GameAgent) -> list[dict]:
    rows = []
    for line in agent.trace_path.read_text().splitlines():
        obj = json.loads(line)
        if obj.get("kind") == "status_snapshot":
            rows.append(obj)
    return rows


def test_update_status_payload_has_queue_fields():
    # Source-pin: the TUI snapshot payload carries pending_feedback +
    # ledger_tail so traces can answer "was my feedback queued?".
    src = inspect.getsource(CodingBoxApp._update_status)
    assert '"pending_feedback"' in src
    assert '"ledger_tail"' in src


def test_trace_status_writes_row_on_pending_feedback_change(tmp_path):
    a = _make_agent(tmp_path)
    base = {"activity": "idle", "phase": "build", "goal": "g", "files": {"game": "x"}}
    a.trace_status({**base, "pending_feedback": 0, "ledger_tail": None})
    # Same activity/phase — only the queue count changed. Must still write.
    a.trace_status({**base, "pending_feedback": 1, "ledger_tail": "queued"})
    rows = _snapshot_rows(a)
    assert len(rows) == 2
    assert rows[1]["pending_feedback"] == 1
    assert rows[1]["ledger_tail"] == "queued"


def test_trace_status_dedupes_unchanged_sig(tmp_path):
    a = _make_agent(tmp_path)
    snap = {"activity": "idle", "phase": "build", "pending_feedback": 0}
    a.trace_status(dict(snap))
    a.trace_status(dict(snap))  # identical signature — skipped
    assert len(_snapshot_rows(a)) == 1


def test_trace_status_carries_forward_goal_and_files(tmp_path):
    a = _make_agent(tmp_path)
    base = {"goal": "snake", "files": {"game": "g.html"}}
    a.trace_status({"activity": "idle", **base})
    a.trace_status({"activity": "iter 1 reply", **base})
    a.trace_status({"activity": "idle2", "goal": "snake 2", "files": {"game": "g.html"}})
    rows = _snapshot_rows(a)
    assert len(rows) == 3
    # First row carries the static fields.
    assert rows[0]["goal"] == "snake" and rows[0]["files"] == {"game": "g.html"}
    # Second row: unchanged statics are elided (carry-forward semantics).
    assert "goal" not in rows[1] and "files" not in rows[1]
    # Third row: goal changed — statics reappear.
    assert rows[2]["goal"] == "snake 2"


# ---------------------------------------------------------------------------
# 5. FPS yaw false positive (z-axis position leaves + any-leaf recovery)
# ---------------------------------------------------------------------------

def test_position_leaf_names_include_z_axis():
    assert _is_position_leaf("playerPos.z")
    assert _is_position_leaf("player.worldz")
    assert _is_position_leaf("cam.tz")
    assert _is_position_leaf("p.posz")
    # Rotation is NOT position — yaw alone must not count as movement.
    assert not _is_position_leaf("player.yaw")
    assert not _is_position_leaf("camera.angle")


def test_recovery_recheck_accepts_any_position_leaf():
    # Source-pin: _recovery_leaves_move_again falls back to ANY
    # position-leaf change (FPS: W moves along z after the view rotated,
    # not the originally-recorded x).
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    start = src.index("async def _recovery_leaves_move_again")
    end = src.index("await _induce_combat_before_recovery()")
    block = src[start:end]
    assert "_is_position_leaf" in block
    # Original-leaf match is still checked first (cheapest, most specific).
    assert "leaf in moved for leaf in recovery_leaves" in block


def test_recovery_yaw_rotation_scenario_passes():
    # Simulates the FPS trace: recovery recorded playerPos.x, but after
    # the sweep rotated the view, W changes playerPos.z. The recheck
    # decision (original leaves OR any position leaf) must read "moved".
    recovery_leaves = ["playerPos.x"]
    moved = {"playerPos.z"}
    decision = any(leaf in moved for leaf in recovery_leaves) or any(
        _is_position_leaf(leaf) for leaf in moved
    )
    assert decision is True


def test_recovery_true_stun_lock_still_trips():
    # Permanent stun-lock: NOTHING position-like moves — only a flag.
    recovery_leaves = ["player.x"]
    moved = {"player.stunned"}
    decision = any(leaf in moved for leaf in recovery_leaves) or any(
        _is_position_leaf(leaf) for leaf in moved
    )
    assert decision is False
