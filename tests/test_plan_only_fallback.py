"""Tests for the plan-only / no-usable-code fallback selection.

Background: when the model emits a <plan> block but no code on iter 1
of a fresh session (no baseline file on disk), the harness previously
waited for a SECOND consecutive plan-only iter before escalating to
the strong "MUST produce code" directive. With no baseline, the second
strike is wasted by definition — there's nothing the model can
legitimately be diagnosing yet, and Phase A already supplied the plan.
One-strike escalation saves a full iter (~60-120s on local models).

The asteroid_20260510_173200 and donkey-kong_20260513_153626 traces
both showed reasoner-class models hitting exactly this pattern. The
fix is generic — it gates on "no baseline" rather than the model
name — so it benefits any model that defaults to "think more" when
uncertain about phase boundaries.
"""

from agent import GameAgent


def _fallback(plan_only: bool, has_existing_file: bool, streak: int):
    """Call the static helper directly — it has no instance state."""
    return GameAgent._no_usable_code_fallback(
        plan_only=plan_only,
        has_existing_file=has_existing_file,
        consecutive_plan_only=streak,
    )


def test_first_strike_no_baseline_escalates_immediately():
    """Iter 1 plan-only with no baseline file: the model must emit
    code THIS turn. Strong-language fallback expected on streak=1."""
    text, reset = _fallback(plan_only=True, has_existing_file=False, streak=1)
    low = text.lower()
    # Strong escalation markers — pick a couple of phrases unique to
    # the no-baseline branch so the test rejects a soft fallback.
    assert "build phase" in low or "required this turn" in low, (
        f"expected escalation directive, got: {text!r}"
    )
    assert "html_file" in low
    assert "do not re-emit" in low
    # Streak does NOT reset on a first-strike escalation — a second
    # strike would still need the LOOP DETECTED hard-break.
    assert reset is False


def test_first_strike_does_not_forbid_assets_reemission():
    """Refinement from DK trace 20260513_153626 conversation.md:
    the iter-1 user turn often stacks USER FEEDBACK + MEDIA-CHANGE
    DIRECTIVE + the plan-only fallback. The MEDIA-CHANGE DIRECTIVE
    explicitly invites <assets> re-emission for art-change feedback.

    The first-strike coach must NOT forbid <assets> globally — that
    would contradict the user's request when they've asked for a
    sprite refresh. Plan/criteria/probes are session-fixed; assets
    are not."""
    text, _ = _fallback(plan_only=True, has_existing_file=False, streak=1)
    low = text.lower()
    # The prohibition list must NOT mention <assets>. The earlier
    # draft of the coach had "Do NOT re-emit <plan>, <criteria>,
    # <probes>, or <assets>"; the conversation.md from restart-2 of
    # the DK trace caught this conflicting with the MEDIA-CHANGE
    # DIRECTIVE in the same user turn.
    assert "or <assets>" not in low
    # And the coach should explicitly permit <assets> as an optional
    # companion to the required <html_file>, so the model isn't
    # confused by the apparent contradiction with the directive.
    assert "may emit" in low and "<assets>" in low


def test_first_strike_with_baseline_uses_soft_fallback():
    """When there's an existing file, the model may have a legitimate
    reason to re-emit <plan> (e.g. user asked for a redesign). Don't
    escalate on the first strike here."""
    text, reset = _fallback(plan_only=True, has_existing_file=True, streak=1)
    low = text.lower()
    # Soft fallback wording — no LOOP DETECTED, no BUILD PHASE.
    assert "loop detected" not in low
    assert "build phase" not in low
    # Still tells the model to stop re-emitting plan.
    assert "plan" in low and "html_file" in low
    assert reset is False


def test_second_strike_triggers_loop_break_and_resets():
    """The existing 2-strike hard-break path is unchanged."""
    text, reset = _fallback(plan_only=True, has_existing_file=True, streak=2)
    low = text.lower()
    assert "loop detected" in low
    assert "must produce code" in low
    # Counter resets so escalation doesn't stack on iter 3, 4, ...
    assert reset is True


def test_second_strike_no_baseline_also_triggers_loop_break():
    """Streak >= 2 always trips the hard-break, regardless of baseline.
    Defensive: the new one-strike rule means we'd usually never get
    here on a no-baseline session, but if we do, escalate maximally."""
    text, reset = _fallback(plan_only=True, has_existing_file=False, streak=2)
    low = text.lower()
    assert "loop detected" in low
    assert reset is True


def test_not_plan_only_returns_generic_reminder():
    """No <plan>, no code at all — generic "send patches or html_file"
    nudge. No streak reset, no escalation."""
    text, reset = _fallback(plan_only=False, has_existing_file=True, streak=0)
    low = text.lower()
    assert "patch" in low and "html_file" in low
    assert "loop detected" not in low
    assert "build phase" not in low
    assert reset is False


def test_not_plan_only_no_baseline_uses_first_build_lock():
    """No baseline + no code tags should get a strict first-build lock.

    Regression for DOOM trace 20260517_155638 where prose mentions of
    `<html_file>` kept repeating without any real tagged output.
    """
    text, reset = _fallback(plan_only=False, has_existing_file=False, streak=0)
    low = text.lower()
    assert "first build required" in low
    assert "code only" in low
    assert "first non-whitespace token" in low
    assert "<html_file>" in text
    assert reset is False


def test_restart_temperature_bias_pattern_is_stable():
    """Restart attempts should diversify decode paths deterministically."""
    assert GameAgent._restart_temperature_bias(0) == 0.0
    assert GameAgent._restart_temperature_bias(1) == -0.20
    assert GameAgent._restart_temperature_bias(2) == 0.10
    assert GameAgent._restart_temperature_bias(3) == -0.30
    assert GameAgent._restart_temperature_bias(4) == 0.20
