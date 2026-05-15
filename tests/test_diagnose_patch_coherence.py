"""Tests for the diagnose-vs-patch subsystem-coherence check.

DK trace 20260514_175012 turn [08] evidence:
  - `mistake_signature` had been "Controls are not wired up" for 3
    iterations (INPUT subsystem per `_subsystem_hint`).
  - The model's `<diagnose>` named "barrel drop threshold" + "player
    procedural fallback coordinate bug" — NO mention of any input
    identifier (addEventListener, keydown, keys, KEYMAP, ...).
  - The model's `<patch>` SEARCH blocks targeted barrel math and
    rendering coords — NO input-handler code.

The coherence check queues a coaching message for the next user
turn when both the diagnose AND the patches ignore the implicated
subsystem. Light touch — does not reject the patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from patches import Patch  # noqa: E402


# ---------------------------------------------------------------------------
# `_diagnose_mentions_subsystem` — does the diagnose body name the area?
# ---------------------------------------------------------------------------


_INPUT_IDENTS = ("addEventListener", "keydown", "keyup",
                 "KeyboardEvent", "code", "KEYMAP", "keys")


def test_diagnose_mentions_input_subsystem():
    diag = (
        "The keydown handler is not registering arrow keys "
        "because the listener attaches to document instead of "
        "window."
    )
    assert GameAgent._diagnose_mentions_subsystem(diag, _INPUT_IDENTS) is True


def test_diagnose_addressing_different_subsystem():
    """DK turn [08] diagnose shape — ignores input entirely."""
    diag = (
        "The barrel drop threshold (22) and ladder transfer "
        "threshold (20) are too generous. Additionally, the "
        "player's procedural fallback drawing has a coordinate "
        "bug in the flipped-facing branch."
    )
    assert GameAgent._diagnose_mentions_subsystem(diag, _INPUT_IDENTS) is False


def test_diagnose_case_insensitive():
    """Match must be case-insensitive — model might capitalize
    differently than the hint table."""
    diag = "The KEYDOWN handler is broken."
    assert GameAgent._diagnose_mentions_subsystem(diag, _INPUT_IDENTS) is True


def test_empty_diagnose_returns_false():
    assert GameAgent._diagnose_mentions_subsystem("", _INPUT_IDENTS) is False
    assert GameAgent._diagnose_mentions_subsystem(None, _INPUT_IDENTS) is False


def test_empty_identifiers_returns_false():
    """Defensive — never report a mention when the identifier set
    is empty (would otherwise vacuously match anything)."""
    assert GameAgent._diagnose_mentions_subsystem(
        "any text", (),
    ) is False


# ---------------------------------------------------------------------------
# `_patches_touch_subsystem_idents` — do the patches target the area?
# ---------------------------------------------------------------------------


def test_patch_touching_input_handler_search():
    """A patch whose SEARCH references the keydown listener counts
    as touching the subsystem."""
    patches = [Patch(
        search="window.addEventListener('keydown', e => {",
        replace="window.addEventListener('keydown', function (e) {",
    )]
    assert GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    ) is True


def test_patch_adding_input_wiring_in_replace():
    """A patch that ADDS input wiring (in REPLACE) without the SEARCH
    side mentioning it also counts. This catches the case where the
    model wires up missing input handlers to a previously-bare
    function."""
    patches = [Patch(
        search="function init() {",
        replace=(
            "function init() {\n"
            "  window.addEventListener('keydown', e => {\n"
            "    keys[e.code] = true;\n"
            "  });"
        ),
    )]
    assert GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    ) is True


def test_patch_ignoring_input_subsystem():
    """DK turn [08] shape — patches target barrels and rendering,
    not input."""
    patches = [
        Patch(
            search="if (b.x >= g.x1 - 22) drop = true;",
            replace="if (b.x >= g.x1 - 5) drop = true;",
        ),
        Patch(
            search="ctx.fillRect(4, 0, w, h);",
            replace="ctx.fillRect(0, 0, w, h);",
        ),
    ]
    assert GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    ) is False


def test_no_patches_returns_false():
    assert GameAgent._patches_touch_subsystem_idents([], _INPUT_IDENTS) is False
    assert GameAgent._patches_touch_subsystem_idents(None, _INPUT_IDENTS) is False


def test_empty_identifiers_returns_false():
    patches = [Patch(search="anything", replace="anything")]
    assert GameAgent._patches_touch_subsystem_idents(patches, ()) is False


# ---------------------------------------------------------------------------
# Combined: would the coherence note fire on the DK trace turn [08]?
# ---------------------------------------------------------------------------


def test_dk_turn_8_shape_triggers_coherence_mismatch():
    """End-to-end pin: the DK turn [08] diagnose + patch shape
    should trigger the coherence-mismatch coaching."""
    diag = (
        "The barrel drop threshold ... and ladder transfer "
        "threshold ... are too generous. Additionally, the "
        "player's procedural fallback drawing has a coordinate "
        "bug in the flipped-facing branch."
    )
    patches = [
        Patch(
            search="if (b.x >= g.x1 - 22) drop = true;",
            replace="if (b.x >= g.x1 - 5) drop = true;",
        ),
    ]
    # Neither diag nor patches touch input — coherence mismatch.
    diag_touches = GameAgent._diagnose_mentions_subsystem(
        diag, _INPUT_IDENTS,
    )
    patch_touches = GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    )
    assert diag_touches is False
    assert patch_touches is False
    # → The agent should queue the coherence-note coaching.


def test_diagnose_mentions_but_patches_miss_does_not_trigger():
    """If the diagnose acknowledges the subsystem, give the model
    benefit of the doubt — maybe it's about to rewrite in a
    follow-up turn. Don't fire the coherence note."""
    diag = (
        "The keydown handler attaches to document instead of "
        "window — that's why arrow keys don't register."
    )
    patches = [Patch(  # patch targets something else this turn
        search="state.score = 0;",
        replace="state.score = 0; state.lives = 3;",
    )]
    diag_touches = GameAgent._diagnose_mentions_subsystem(
        diag, _INPUT_IDENTS,
    )
    patch_touches = GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    )
    assert diag_touches is True
    assert patch_touches is False
    # The wrapper in agent.py checks `not diag_mentions AND not patch_touches`,
    # so diagnose match alone suppresses the coherence note.


def test_patches_touch_but_diagnose_misses_does_not_trigger():
    """Symmetric: if the patches address the subsystem, no coaching
    even if the diagnose forgot to name it."""
    diag = "Fixing a small naming bug."  # generic, no input mention
    patches = [Patch(
        search="window.addEventListener('keydown', e => {",
        replace="window.addEventListener('keydown', function (e) {",
    )]
    diag_touches = GameAgent._diagnose_mentions_subsystem(
        diag, _INPUT_IDENTS,
    )
    patch_touches = GameAgent._patches_touch_subsystem_idents(
        patches, _INPUT_IDENTS,
    )
    assert diag_touches is False
    assert patch_touches is True
