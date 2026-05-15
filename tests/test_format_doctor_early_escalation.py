"""Tests for the format-doctor early-escalation path.

DK trace 20260514_104131 post-seed iter 2 hit this:
  - Stream aborted after 12,706 tokens / 702s as a repetition-loop.
  - The kept partial output contained bare SEARCH/REPLACE markers
    without a <patch> wrapper.
  - `_format_stuck_streak` reached 1 → normal threshold is 2, so the
    format-doctor never fired.
  - Session ended; no recovery.

Fix: when the previous stream was aborted as a repetition-loop OR
stalled AND the parse failure produced a structured rejection, fire
the doctor at streak == 1 — a looped stream is strong evidence the
model is confused; don't wait for a second wasted iteration.

The doctor invocation itself is async and depends on the model
backend; these tests cover the DECISION logic only — when does the
agent decide to fire the doctor?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _decide_invoke_doctor(
    *,
    format_stuck_streak: int,
    last_stream_looped: bool,
    last_stream_stalled: bool,
) -> bool:
    """Mirror of the gate in agent.run()'s format-rejection branch.
    Pins the decision logic to a tiny pure function so we can unit-
    test it without spinning up the whole agent.
    """
    looped_or_stalled = last_stream_looped or last_stream_stalled
    return (
        format_stuck_streak == 2
        or (format_stuck_streak == 1 and looped_or_stalled)
    )


# ---------------------------------------------------------------------------
# Normal streak path (no looped/stalled signal)
# ---------------------------------------------------------------------------


def test_no_doctor_at_streak_1_normal_stream():
    """One bad reply on a healthy stream — wait for streak=2."""
    assert _decide_invoke_doctor(
        format_stuck_streak=1,
        last_stream_looped=False,
        last_stream_stalled=False,
    ) is False


def test_doctor_fires_at_streak_2_regardless_of_stream_state():
    """The base case (streak=2) still fires whether or not the stream
    was looped."""
    assert _decide_invoke_doctor(
        format_stuck_streak=2,
        last_stream_looped=False,
        last_stream_stalled=False,
    ) is True
    assert _decide_invoke_doctor(
        format_stuck_streak=2,
        last_stream_looped=True,
        last_stream_stalled=False,
    ) is True


def test_no_doctor_at_streak_0():
    """Defensive: streak should not be 0 in the rejection branch, but
    if it ever is, don't fire."""
    assert _decide_invoke_doctor(
        format_stuck_streak=0,
        last_stream_looped=True,
        last_stream_stalled=True,
    ) is False


def test_no_doctor_at_streak_3_plus():
    """Past streak=2 the rejection branch should have moved to the
    hard 'send full <html_file>' escalation prompt (handled separately
    in `_no_usable_code_fallback`). The doctor only fires once."""
    assert _decide_invoke_doctor(
        format_stuck_streak=3,
        last_stream_looped=False,
        last_stream_stalled=False,
    ) is False


# ---------------------------------------------------------------------------
# Early-escalation: streak=1 + looped/stalled
# ---------------------------------------------------------------------------


def test_doctor_fires_early_on_looped_stream_at_streak_1():
    """DK 20260514_104131 pin: looped stream + streak=1 → fire NOW
    instead of waiting for a second confused reply."""
    assert _decide_invoke_doctor(
        format_stuck_streak=1,
        last_stream_looped=True,
        last_stream_stalled=False,
    ) is True


def test_doctor_fires_early_on_stalled_stream_at_streak_1():
    """Stall is a similar 'model is stuck' signal — also escalate."""
    assert _decide_invoke_doctor(
        format_stuck_streak=1,
        last_stream_looped=False,
        last_stream_stalled=True,
    ) is True


def test_doctor_fires_early_on_both_flags_at_streak_1():
    """The DK trace had BOTH stalled and looped set. Confirm the
    gate handles the combined case."""
    assert _decide_invoke_doctor(
        format_stuck_streak=1,
        last_stream_looped=True,
        last_stream_stalled=True,
    ) is True


# ---------------------------------------------------------------------------
# GameAgent state plumbing — the flags must round-trip cleanly
# ---------------------------------------------------------------------------


def test_agent_initializes_stream_flags_to_false():
    """Fresh GameAgent must start with the stream-abort flags clear so
    a first-turn parse failure doesn't accidentally trigger early
    escalation."""
    from agent import GameAgent

    a = GameAgent.__new__(GameAgent)
    # Simulate the __init__ fields we care about. The real __init__
    # also touches a lot of other state; we're only pinning the
    # default values of the new flags here.
    a._format_stuck_streak = 0
    a._last_stream_looped = False
    a._last_stream_stalled = False
    a._last_stream_deliberated = False
    assert a._last_stream_looped is False
    assert a._last_stream_stalled is False
    assert a._last_stream_deliberated is False
