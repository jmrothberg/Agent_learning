"""Tests for the /allroles status-panel visibility fix.

When a user runs `/allroles` ON with a single backend loaded (the
common Claude Opus 4.7 / single-cloud-model case), the architect and
critic roles run on the coder backend via the fallthrough in
`agent.GameAgent.get_backend(role)`. Before this fix, the status
panel still hardcoded the slot-1 header as "Activity (coder):" so
the architect/critic streams were invisible — users couldn't see
the architect drive Phase A planning or the critic fire after a
clean iter, and naturally asked "is the architect actually running?"

The fix:
  1. Slot-1 header swaps to the active role when no dedicated slot
     2/3 is staged for that role (chat.py _render_activity_line).
  2. A "Roles: /allroles ON" indicator appears in the status panel
     header when both architect-split and vlm-critique are enabled
     (chat.py _render_mode_row).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


# --- Change 1: dynamic slot-1 header when multiplexed --------------------


def test_slot1_header_shows_architect_when_active_and_no_slot_staged() -> None:
    """The motivating bug: user runs /allroles with one backend, Phase A
    streams the architect role on the coder slot, and the header used
    to lie ('Activity (coder)'). After the fix the header reflects
    the active role."""
    app = CodingBoxApp()
    # Pretend a stream is in flight on the coder slot, role=architect.
    app._activity_role = "architect"
    app._activity_label = "planning reply"
    app._is_streaming = True
    app._stream_tokens = 16
    import time as _t
    app._stream_started_at = _t.monotonic() - 1.0
    app._last_token_at = _t.monotonic() - 0.1
    # No slot 2/3 staged — single-backend /allroles case.
    assert not app._session_backend2 and not app._session_backend3

    line = app._render_activity_line()
    assert "Activity (architect):" in line
    # And the bare "Activity (coder):" should NOT appear, because slot 1
    # is currently representing the architect role.
    assert "Activity (coder):" not in line


def test_slot1_header_shows_critic_when_critic_streams_on_coder_backend() -> None:
    app = CodingBoxApp()
    app._activity_role = "critic"
    app._activity_label = "Auditing screenshot..."
    app._is_streaming = True
    app._stream_tokens = 200
    import time as _t
    app._stream_started_at = _t.monotonic() - 2.0
    app._last_token_at = _t.monotonic() - 0.5

    line = app._render_activity_line()
    assert "Activity (critic):" in line
    assert "Activity (coder):" not in line


def test_slot1_header_stays_coder_when_idle() -> None:
    """When nothing is streaming, _activity_role resets to 'coder'
    (chat.py:5959). The header must remain 'Activity (coder)' so the
    panel is stable between turns."""
    app = CodingBoxApp()
    app._activity_role = "coder"
    app._activity_label = ""
    assert not app._is_streaming

    line = app._render_activity_line()
    assert "Activity (coder):" in line
    # No spurious architect/critic header during idle.
    assert "Activity (architect):" not in line
    assert "Activity (critic):" not in line


def test_slot1_header_stays_coder_when_dedicated_critic_slot_staged() -> None:
    """If the user staged a real slot-2 critic backend, the critic does
    NOT multiplex onto slot 1 — the slot-2 row handles its display.
    Slot 1 must keep the 'coder' header."""
    app = CodingBoxApp()
    app._session_backend2 = object()
    app._session_role2 = "critic"
    app._session_model2 = "external-critic"
    app._activity_role = "critic"  # currently streaming on slot 2
    app._activity_label = "Auditing screenshot..."
    app._model2_is_streaming = True
    app._model2_stream_tokens = 100
    import time as _t
    app._model2_stream_started_at = _t.monotonic() - 1.0
    app._model2_last_token_at = _t.monotonic() - 0.2

    line = app._render_activity_line()
    # Slot 1 stays "coder" because the critic routes to slot 2.
    assert "Activity (coder):" in line
    # The critic row appears (slot-2 row was already rendering correctly).
    assert "Activity (critic):" in line


def test_slot1_header_stays_coder_when_coder_streams() -> None:
    """During build/fix turns the active role IS 'coder' and the
    header should remain 'Activity (coder)' — no swap needed."""
    app = CodingBoxApp()
    app._activity_role = "coder"
    app._activity_label = "streaming coder…"
    app._is_streaming = True
    app._stream_tokens = 500
    import time as _t
    app._stream_started_at = _t.monotonic() - 5.0
    app._last_token_at = _t.monotonic() - 0.1

    line = app._render_activity_line()
    assert "Activity (coder):" in line
    assert "Activity (architect):" not in line


# --- Change 2: /allroles ON indicator -------------------------------------


def test_mode_line_shows_allroles_on_single_backend() -> None:
    """When both architect-split and vlm-critique are on AND no slot 2/3
    is staged, the Mode area should show the /allroles indicator with
    the multiplex explanation."""
    app = CodingBoxApp()
    app._use_architect_split = True
    app._use_vlm_critique = True

    line = app._render_mode_row()
    assert "/allroles ON" in line
    # The single-backend explanation should mention the slot-1 header
    # behavior so the user knows what to look for.
    assert "one loaded LLM" in line or "multiplexed" in line


def test_mode_line_shows_allroles_with_staged_slots() -> None:
    """When /allroles is on AND the user staged a dedicated critic slot,
    the indicator phrasing should reflect the mixed setup (not pretend
    everything multiplexes onto slot 1)."""
    app = CodingBoxApp()
    app._use_architect_split = True
    app._use_vlm_critique = True
    app._session_backend2 = object()
    app._session_role2 = "critic"

    line = app._render_mode_row()
    assert "/allroles ON" in line
    assert "staged" in line or "multiplex" in line.lower()


def test_mode_line_omits_allroles_when_off() -> None:
    app = CodingBoxApp()
    app._use_architect_split = False
    app._use_vlm_critique = False

    line = app._render_mode_row()
    assert "/allroles" not in line
    assert "architect-split" not in line
    assert "vlm-critique" not in line


def test_mode_line_partial_features_shown_separately() -> None:
    """If only one of the two features is on (e.g. user manually
    toggled vlm-critique), surface that fact rather than implying
    /allroles is active."""
    app = CodingBoxApp()
    app._use_architect_split = False
    app._use_vlm_critique = True

    line = app._render_mode_row()
    assert "/allroles ON" not in line
    assert "vlm-critique ON" in line


def test_mode_line_prefers_agent_state_when_session_running() -> None:
    """When a session is running, the live agent's feature flags are
    the source of truth (the chat-level state may have changed but
    the running session might have its own state). Verify the mode
    line reads from the agent when available."""
    app = CodingBoxApp()
    app._use_architect_split = False
    app._use_vlm_critique = False

    class _Stub:
        _use_architect_split = True
        _use_vlm_critique = True
        _is_vlm = None
    app.agent = _Stub()

    line = app._render_mode_row()
    assert "/allroles ON" in line
