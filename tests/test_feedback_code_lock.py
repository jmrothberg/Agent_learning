"""Tests for the broadened `_CODE_LOCK_PATTERNS` in agent.py.

DK trace 20260514_175012 evidence: at turn [03] the user typed
*"you mixed up the mario and maria, just switch the references to
them, this is trivial no other changes there"*. The previous patterns
all required the literal word "code" or a specific media noun
(asset/sprite/image/art/png/graphic/picture/icon), so this phrasing
slipped through, `rewrite_exemption_armed` fired (confirmed in the
.jsonl at 22:24:01), and the model emitted a full <html_file>
rewrite instead of the minimal patch the user asked for.

The fix adds 7 new patterns covering common minimal-scope phrasings
that don't mention "code" or a media noun directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import _feedback_locks_code  # noqa: E402


# ---------------------------------------------------------------------------
# Trace-pin: the literal DK 20260514_175012 user text must now lock code
# ---------------------------------------------------------------------------


def test_dk_trace_20260514_175012_user_text_locks_code():
    """The literal phrasing from the trace must trigger code-lock so
    the rewrite exemption does NOT arm and the model emits a minimal
    patch instead of a full file rewrite."""
    text = (
        "you mixed up the mario and maria, just switch the references "
        "to them, this is trivial no other changes there"
    )
    assert _feedback_locks_code(text) is True


# ---------------------------------------------------------------------------
# Each new pattern: positive case
# ---------------------------------------------------------------------------


def test_no_other_changes_locks():
    assert _feedback_locks_code("fix the typo, no other changes") is True
    assert _feedback_locks_code("no other change please") is True
    assert _feedback_locks_code("no other changes needed") is True


def test_trivial_change_locks():
    assert _feedback_locks_code("this is trivial — just a swap") is True
    assert _feedback_locks_code("trivial fix to the constant") is True
    assert _feedback_locks_code(
        "Should be a trivial change to the title text"
    ) is True
    assert _feedback_locks_code("trivial swap of two names") is True


def test_just_minimal_verb_locks():
    assert _feedback_locks_code(
        "just swap the references to them"
    ) is True
    assert _feedback_locks_code("just switch X and Y") is True
    assert _feedback_locks_code("just rename foo to bar") is True
    assert _feedback_locks_code("just flip the boolean") is True


def test_only_minimal_verb_locks():
    assert _feedback_locks_code("only swap the two names") is True
    assert _feedback_locks_code("only rename the variable") is True


def test_leave_rest_locks():
    assert _feedback_locks_code(
        "fix the score color, leave the rest alone"
    ) is True
    assert _feedback_locks_code(
        "swap mario/maria, leave the rest"
    ) is True


def test_nothing_else_locks():
    assert _feedback_locks_code("rename foo to bar, nothing else") is True
    assert _feedback_locks_code(
        "nothing else needs to change"
    ) is True


# ---------------------------------------------------------------------------
# Backwards-compat: existing patterns must still work
# ---------------------------------------------------------------------------


def test_no_changes_to_code_still_works():
    assert _feedback_locks_code("no changes to the code") is True


def test_dont_touch_code_still_works():
    assert _feedback_locks_code("don't touch the code") is True


def test_only_the_asset_still_works():
    assert _feedback_locks_code("only the sprite needs swap") is True


def test_without_changing_code_still_works():
    assert _feedback_locks_code(
        "regenerate the asset without changing the code"
    ) is True


# ---------------------------------------------------------------------------
# Negative cases: must NOT false-positive on unrelated text
# ---------------------------------------------------------------------------


def test_unrelated_text_does_not_lock():
    """Generic feedback that's NOT a scope-lock signal must not trigger."""
    assert _feedback_locks_code("the player should jump higher") is False
    assert _feedback_locks_code("add a score multiplier") is False
    assert _feedback_locks_code(
        "the barrels need to roll faster"
    ) is False
    assert _feedback_locks_code("") is False


def test_describing_a_bug_does_not_lock():
    """A user describing what's broken (no minimal-scope intent
    expressed) must not lock."""
    assert _feedback_locks_code(
        "mario does not climb when I press up"
    ) is False
    assert _feedback_locks_code(
        "controls are completely broken"
    ) is False


def test_just_alone_does_not_lock():
    """The word 'just' on its own (without a minimal-scope verb) must
    not lock. Otherwise feedback like 'just make it look better' would
    spuriously suppress rewrites."""
    assert _feedback_locks_code("just make it look better") is False
    assert _feedback_locks_code("just like the original") is False


def test_only_alone_does_not_lock():
    """Same defense for 'only'."""
    assert _feedback_locks_code(
        "only when ammo is depleted"
    ) is False


# ---------------------------------------------------------------------------
# Combined scope-lock + behavior-bug feedback (the trace pattern)
# ---------------------------------------------------------------------------


def test_scope_lock_with_behavior_bug_phrasing_still_locks():
    """User reports a bug AND scopes the fix — both signals fire.
    The code-lock detector still locks; the behavior-bug detector
    suppresses MEDIA-CHANGE separately (see test_feedback_behavior_bug)."""
    text = (
        "the score does not update, just swap the references — no "
        "other changes"
    )
    assert _feedback_locks_code(text) is True
