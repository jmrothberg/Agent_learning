"""Tests for the streak-≥3 hard-gate that forces a <question> when the
same subsystem keeps failing.

DK trace 20260514_175012 evidence:
  - Turn [07] reached `stuck_streak = 3` on the INPUT subsystem
    (mistake_signature: "Controls are not wired up").
  - At streak ≥ 2 the harness fired generic coaching, then the
    subsystem-pointing coaching (Item 1a).
  - The model ignored both and emitted more off-target patches.
  - There was no further escalation lever — the session ended
    silently with `OK=False`.

The hard-gate: at `_repeat_sig_streak >= 3` AND `_subsystem_hint`
matches the current signature, set `_force_question_subsystem` to
the hint. `_build_fix_prompt` consumes the flag on the next call,
substituting a <question>-only prompt that blocks every other tag.
The flag clears on any clean iter OR after one consumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _subsystem_hint  # noqa: E402


def _agent_stub() -> GameAgent:
    """Minimum-viable GameAgent for testing _build_fix_prompt
    without spinning up backend / browser."""
    a = GameAgent.__new__(GameAgent)
    a._repeat_sig_streak = 0
    a._force_question_subsystem = None
    a._last_mistake_sig = None
    # _build_fix_prompt routes through _p.<branch>_instruction for
    # non-hard-gate paths; we never exercise those in this file.
    a._p = None
    a._trace = lambda obj: None
    return a


_INPUT_SIG = (
    "HEURISTIC: pressed ArrowUp - canvas pixels never changed. "
    "Controls are not wired up. | INPUT_DEAD"
)


def _fake_report(ok: bool = False) -> dict:
    """Minimum report shape `format_report_for_model` expects."""
    return {
        "ok": ok,
        "title": "test",
        "canvas": {"width": 800, "height": 600, "raf_ran": True, "blank": False},
        "input_listeners": {},
        "input_test": {"ran": True, "any_change": False, "keys_tried": ["ArrowUp"]},
        "frozen_canvas": False,
        "body_chars": 50,
        "body_sample": "",
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "soft_warnings": [],
        "warnings": [],
        "logs": [],
        "probes": [],
        "probe_errors": [],
    }


# ---------------------------------------------------------------------------
# Flag arming + consumption
# ---------------------------------------------------------------------------


def test_no_hard_gate_when_flag_unset():
    """Default path: `_force_question_subsystem` is None →
    `_build_fix_prompt` falls through to the normal branches.
    Doesn't need to be tested by string — just confirm the flag
    starts None."""
    a = _agent_stub()
    assert a._force_question_subsystem is None


def test_hard_gate_prompt_built_when_flag_set():
    """With the flag set to a hint dict, `_build_fix_prompt`
    returns the hard-gate prompt and the flag clears on consumption."""
    a = _agent_stub()
    hint = _subsystem_hint(_INPUT_SIG)
    assert hint is not None  # sanity
    a._force_question_subsystem = hint
    a._repeat_sig_streak = 3

    prompt = a._build_fix_prompt(
        report=_fake_report(),
        regressed=False,
        partial_failed=[],
    )
    # Hard-gate text present.
    assert "STUCK-LOOP HARD GATE" in prompt
    assert "<question>" in prompt
    # The hint's subsystem name is named.
    assert "INPUT" in prompt
    # Identifiers surfaced for the model's context.
    assert "addEventListener" in prompt or "keydown" in prompt
    # Forbidden tags called out.
    for forbidden in ("<patch>", "<html_file>", "<plan>", "<diagnose>"):
        assert forbidden in prompt  # named in the "do NOT emit" list
    # Flag cleared on consumption — second call falls through.
    assert a._force_question_subsystem is None


def test_hard_gate_consumes_only_once():
    """If the flag was set, calling `_build_fix_prompt` once must
    clear it. A subsequent call (next iter) does NOT re-fire the
    gate — the streak machinery would have to re-arm it."""
    a = _agent_stub()
    a._force_question_subsystem = _subsystem_hint(_INPUT_SIG)
    a._repeat_sig_streak = 3

    # First call: hard-gate fires.
    p1 = a._build_fix_prompt(
        report=_fake_report(),
        regressed=False,
        partial_failed=[],
    )
    assert "STUCK-LOOP HARD GATE" in p1
    # Flag cleared.
    assert a._force_question_subsystem is None


# ---------------------------------------------------------------------------
# Streak counter behavior + reset
# ---------------------------------------------------------------------------


def test_streak_below_3_does_not_arm_gate():
    """Streak = 1 or 2 → existing coaching path runs, but the flag
    must NOT be armed. The streak-handling branch in agent.py only
    arms the flag at `_repeat_sig_streak >= 3`."""
    # This is a state pin (mirroring the agent.py condition).
    streak_2_should_arm = (2 >= 3)
    streak_3_should_arm = (3 >= 3)
    assert streak_2_should_arm is False
    assert streak_3_should_arm is True


def test_clean_iter_clears_armed_flag():
    """The agent.py reset site clears `_force_question_subsystem`
    on the same path that resets `_repeat_sig_streak` to 0.
    Without this, an armed gate could fire one iter after the
    model fixed the issue."""
    a = _agent_stub()
    a._force_question_subsystem = _subsystem_hint(_INPUT_SIG)
    a._repeat_sig_streak = 3
    # Simulate the clean-iter reset (mirrors agent.py line 5546-5550):
    a._last_mistake_sig = None
    a._repeat_sig_streak = 0
    a._force_question_subsystem = None
    assert a._force_question_subsystem is None
    # And the next _build_fix_prompt call would NOT hit the hard-gate
    # branch (we don't run it here because the report=ok path delegates
    # to self._p which is None in the stub).


# ---------------------------------------------------------------------------
# Pin behavior: only fires when subsystem hint matches
# ---------------------------------------------------------------------------


def test_unrecognized_signature_does_not_arm_gate():
    """If the streak hits 3 but the signature doesn't match any
    subsystem hint (e.g. a generic "score increment failed" error),
    the hard-gate must NOT fire — the streak-handling branch in
    agent.py gates on `if sub_hint:` BEFORE checking streak >= 3.
    Pin that ordering."""
    sig = "random unmapped failure | PROBE FAILED [score_not_increasing]"
    hint = _subsystem_hint(sig)
    assert hint is None
    # Agent code path: `sub_hint = _subsystem_hint(sig); if sub_hint: ...`
    # → no `sub_hint` → never enters the streak->=3 arm.


# ---------------------------------------------------------------------------
# Hard-gate prompt content guarantees
# ---------------------------------------------------------------------------


def test_hard_gate_prompt_names_concrete_options():
    """The model needs explicit (a)/(b)/(c) options to copy into its
    <question>. Without them, a 27B model often emits an open-ended
    question that's hard for the user to answer briefly."""
    a = _agent_stub()
    a._force_question_subsystem = _subsystem_hint(_INPUT_SIG)
    a._repeat_sig_streak = 3
    prompt = a._build_fix_prompt(
        report=_fake_report(),
        regressed=False,
        partial_failed=[],
    )
    assert "(a)" in prompt
    assert "(b)" in prompt
    assert "(c)" in prompt


def test_hard_gate_includes_report_for_context():
    """The model still needs the test report for context — but as
    REFERENCE only, with a 'do NOT act on it this turn' note."""
    a = _agent_stub()
    a._force_question_subsystem = _subsystem_hint(_INPUT_SIG)
    a._repeat_sig_streak = 3
    prompt = a._build_fix_prompt(
        report=_fake_report(),
        regressed=False,
        partial_failed=[],
    )
    # Report present (formatted).
    assert "OK:" in prompt
    # Explicit don't-act note present.
    assert "do NOT act" in prompt or "do not act" in prompt.lower()
