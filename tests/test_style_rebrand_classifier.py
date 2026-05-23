"""Phase 0.11 — descriptive-verb art-change + style-rebrand classifier.

Catches the failure shape from 2026-05-22 chess trace (third session):
the user said *"All of the images ... need to be animated as if the
chess pieces are fantasy monsters, not look like regular chess pieces"*
and *"game works great, i just want ALL new graphics so the pieces ...
look like monsters"*. Both phrasings use descriptive verbs ("need to be",
"want", "look like") + stem-only asset references ("pawn") that the
pre-fix classifier missed → MEDIA-CHANGE DIRECTIVE never fired → the
model emitted 14 more wrong-style sprites with the SAME prompt pattern.

Tests are genre-free; failure phrases describe rendering modality
(graphics / images / pieces) not subject matter.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    _feedback_is_art_change,
    _feedback_requests_style_rebrand,
)


_ASSETS = [
    "white_pawn_idle", "white_pawn_walk", "white_pawn_smash",
    "black_pawn_idle", "black_pawn_walk",
    "white_rook_idle", "black_rook_idle",
    "white_king_idle", "black_king_idle",
]


# ---------------------------------------------------------------------------
# Descriptive-verb expansion of art_change
# ---------------------------------------------------------------------------


def test_descriptive_verb_look_paired_with_art_noun_fires():
    assert _feedback_is_art_change("the images should look more polished", _ASSETS) is True


def test_descriptive_verb_want_paired_with_art_noun_fires():
    assert _feedback_is_art_change(
        "i just want all new graphics that look like monsters", _ASSETS
    ) is True


def test_descriptive_verb_need_paired_with_art_noun_fires():
    assert _feedback_is_art_change(
        "all the sprites need to be more colorful", _ASSETS
    ) is True


def test_fuzzy_stem_match_fires_art_change_without_full_asset_name():
    # User says "all the pawns" — full names are like white_pawn_idle.
    # Pre-Phase 0.11, _name_in_text required substring match against the
    # canonical full name, which would miss "pawn" → "white_pawn_idle".
    # Phase 0.11 integrates _resolve_fuzzy_asset_stems.
    assert _feedback_is_art_change("the pawns look terrible", _ASSETS) is True


def test_negative_control_behavior_ask_without_art_noun_stays_false():
    # "the player should jump higher" has 'should' (now in _MEDIA_VERBS)
    # but no art noun. Must NOT trigger art_change.
    assert _feedback_is_art_change(
        "the player should jump higher when arrow up is held", _ASSETS
    ) is False


def test_negative_control_feature_request_no_art_noun_stays_false():
    assert _feedback_is_art_change("add a high score screen", _ASSETS) is False


# ---------------------------------------------------------------------------
# Style rebrand classifier
# ---------------------------------------------------------------------------


def test_style_rebrand_fires_on_real_session_feedback_one():
    fb = (
        "All of the images for the chess pieces, resting, walking, "
        "smashing need to be animated as if the chess pieces are "
        "fantasy monsters, not look like regular chess pieces, so "
        "fantasy monster versions of pawns, rooks, bishops, queen, "
        "kink, knights"
    )
    assert _feedback_requests_style_rebrand(fb) is True
    assert _feedback_is_art_change(fb, _ASSETS) is True


def test_style_rebrand_fires_on_real_session_feedback_two():
    fb = (
        "game works great, i just want ALL new graphics so the pieces "
        "and animations look like monsters"
    )
    assert _feedback_requests_style_rebrand(fb) is True
    assert _feedback_is_art_change(fb, _ASSETS) is True


def test_style_rebrand_phrases():
    positives = [
        "make them look like medieval warriors",
        "should be themed as cyberpunk",
        "in the style of 8-bit pixel art",
        "i want a different style for the sprites",
        "all new graphics please",
        "redesign them all to be more elegant",
        "the pieces look like generic chess, not what i wanted",
    ]
    for fb in positives:
        assert _feedback_requests_style_rebrand(fb), (
            f"expected style_rebrand for: {fb!r}"
        )


def test_style_rebrand_negative_controls():
    negatives = [
        "fix the broken move generation",
        "the AI is too slow",
        "add a high score screen",
        "use the existing pawn sprite as a starting point for a walk frame",
        "the player position is wrong",
    ]
    for fb in negatives:
        assert not _feedback_requests_style_rebrand(fb), (
            f"unexpected style_rebrand match for: {fb!r}"
        )
