"""Tests for the behavior-bug detector that suppresses the MEDIA-CHANGE
DIRECTIVE on user feedback.

DK trace 20260514_104131 background:
  - User typed: "mario does not climb the ladder, even when below it
    and i push the key up, dont change anything else."
  - The harness's `_feedback_is_art_change` returned True because
    "mario" and "ladder" were both registered asset names.
  - That triggered the MEDIA-CHANGE DIRECTIVE which told the model:
    "The feedback above is about ART/SOUND, not code."
  - The model dutifully emitted <assets> re-renders instead of fixing
    the ladder-climb bug. The .jsonl shows 7 consecutive
    `media_change_directive_injected ... art_change: true` events.

`_feedback_is_behavior_bug(text)` catches the misrouting by detecting
negation + behavior-verb patterns and explicit complaint nouns.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import _feedback_is_behavior_bug  # noqa: E402


# ---------------------------------------------------------------------------
# Positive cases — should fire (suppress MEDIA-CHANGE DIRECTIVE)
# ---------------------------------------------------------------------------


def test_dk_trace_feedback_fires():
    """The literal text from DK 20260514_104131 turn [03]. Must fire
    or the directive misrouting persists."""
    text = (
        "we've tried three times, the game works great, just mario does "
        "not climb the ladder, even when below it and i push the key "
        "up, dont change anything else."
    )
    assert _feedback_is_behavior_bug(text) is True


def test_negated_climb_fires():
    assert _feedback_is_behavior_bug("mario does not climb the ladder") is True
    assert _feedback_is_behavior_bug("mario doesn't climb up") is True
    assert _feedback_is_behavior_bug("mario can't climb") is True


def test_negated_movement_fires():
    assert _feedback_is_behavior_bug("the player can't move") is True
    assert _feedback_is_behavior_bug("ship won't move") is True
    assert _feedback_is_behavior_bug("nothing moves when I press up") is True


def test_negated_response_fires():
    assert _feedback_is_behavior_bug(
        "controls aren't responding to keyboard"
    ) is True
    assert _feedback_is_behavior_bug(
        "the game doesn't respond to clicks"
    ) is True


def test_explicit_complaint_words_fire():
    assert _feedback_is_behavior_bug("the game is broken") is True
    assert _feedback_is_behavior_bug("there's a bug in the spawn logic") is True
    assert _feedback_is_behavior_bug("the page is frozen") is True
    assert _feedback_is_behavior_bug("everything keeps crashing") is True
    assert _feedback_is_behavior_bug("mario is stuck on the platform") is True
    assert _feedback_is_behavior_bug("the animation glitches") is True


def test_nothing_happens_fires():
    """'nothing happens' patterns — a common code-bug phrasing where
    user names no specific verb but is still describing missing
    behavior. Caught via the 'nothing' negation token + a behavior
    verb within window."""
    assert _feedback_is_behavior_bug(
        "nothing happens when i press space"
    ) is True


def test_barrels_dont_roll_fires():
    """Genre-agnostic check: a different behavior verb (roll) — the
    list isn't keyed to mario / DK / platformer vocabulary."""
    assert _feedback_is_behavior_bug("barrels don't roll down") is True


def test_fails_to_pattern_fires():
    """'fails to X' is another common bug-reporting phrasing."""
    assert _feedback_is_behavior_bug(
        "the player fails to register key presses"
    ) is True


# ---------------------------------------------------------------------------
# Negative cases — must NOT fire (let directive route as before)
# ---------------------------------------------------------------------------


def test_visual_complaint_does_not_fire():
    """'doesn't look right' is a pure visual complaint — should route
    to art-change, NOT be treated as a behavior bug."""
    assert _feedback_is_behavior_bug("the dragon doesn't look right") is False
    assert _feedback_is_behavior_bug("the princess sprite looks ugly") is False


def test_explicit_art_change_does_not_fire():
    assert _feedback_is_behavior_bug(
        "change the sprite to be more colorful"
    ) is False
    assert _feedback_is_behavior_bug(
        "redraw mario with a bigger hat"
    ) is False
    assert _feedback_is_behavior_bug(
        "make the explosion sound louder"
    ) is False


def test_positive_progress_does_not_fire():
    assert _feedback_is_behavior_bug("the game works great") is False
    assert _feedback_is_behavior_bug(
        "we've tried three times and it looks good"
    ) is False


def test_empty_inputs_do_not_fire():
    assert _feedback_is_behavior_bug("") is False
    assert _feedback_is_behavior_bug(None or "") is False


# ---------------------------------------------------------------------------
# Integration: the misroute is now blocked end-to-end
# ---------------------------------------------------------------------------


def test_dk_feedback_misrouting_suppressed():
    """Pin the integration: with the literal DK feedback text AND the
    registered asset names from the session, the existing
    `_feedback_is_art_change` STILL fires (asset names match), but the
    behavior-bug detector also fires — so the wrapper in
    `_flush_user_injections` will suppress the MEDIA-CHANGE
    DIRECTIVE."""
    from agent import _feedback_is_art_change

    text = (
        "mario does not climb the ladder, even when below it and i "
        "push the key up, dont change anything else."
    )
    asset_names = [
        "mario", "ladder", "barrel", "dk", "princess", "girder",
    ]
    # The art-change detector still mis-fires on its own (asset names
    # in text). That's pinned so a future refactor of that classifier
    # doesn't silently change behavior under us.
    assert _feedback_is_art_change(text, asset_names) is True
    # But the behavior-bug detector catches the misroute — the wrapper
    # uses it to suppress the directive.
    assert _feedback_is_behavior_bug(text) is True
