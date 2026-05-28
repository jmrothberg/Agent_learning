"""Pin the three classifier bugs surfaced by the 2026-05-23 Doom trace
(write-a-game-of-doom-a-first-p_20260523_204625) plus the new
/rawfeedback kill switch.

Background: user repeatedly typed feedback about movement direction
and pistol orientation. The harness misrouted every turn:

  Iter 5 user: "do not make a NEW asset, i think the pistal maybe
                facing the wrong way ... inverted facing the wrong way"
  → Orientation classifier should fire. Instead the literal blocker
    `\\bnew\\s+asset\\b` matched (ignoring the preceding "do not"),
    orientation_change was False, MEDIA-CHANGE DIRECTIVE was injected,
    agent emitted <assets> regen — EXACTLY what the user forbade.

  Iter 7 user: "...you see backwards not the way the arrow is in the
                maze view, and the down key moves you forward.
                why are they directions and views getting reversed..."
  → Clear movement/control bug. _feedback_is_behavior_bug missed it
    because the bug-detector only matched negation ("X doesn't Y"),
    not "X does the OPPOSITE of Y". The fuzzy stem matcher also
    mis-mapped 'view' (from "maze view") to [pistol_view, shotgun_view]
    because 'view' wasn't in _NON_DISTINCTIVE_ASSET_STEMS — surfacing
    the pistol sprite in the MEDIA-CHANGE block when the user was
    talking about camera perspective.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent import (
    GameAgent,
    _feedback_is_art_change,
    _feedback_is_behavior_bug,
    _feedback_is_orientation_change,
    _resolve_fuzzy_asset_stems,
    _phrase_is_negated,
)


DOOM_ASSETS = [
    "imp_idle", "imp_walk1", "imp_walk2", "imp_walk3",
    "demon_idle", "demon_walk1", "demon_walk2", "demon_walk3",
    "cacodemon_idle", "cacodemon_walk1", "cacodemon_walk2",
    "health_pickup", "armor_pickup", "ammo_pickup", "shotgun_pickup",
    "pistol_view", "shotgun_view",
    "stone_wall", "stone_floor", "stone_ceiling",
]
DOOM_SOUNDS = [
    "pistol_shot", "shotgun_shot", "enemy_death", "pickup",
    "player_hit", "music",
]


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ----------------------------------------------------------------------
# Fix 1: 'view' is non-distinctive — "maze view" must NOT map to
# pistol_view / shotgun_view.
# ----------------------------------------------------------------------


def test_maze_view_does_not_stem_match_pistol_view() -> None:
    text = (
        "so strange sometimes you see backwards not the way the arrow"
        " is in the maze view, and the down key moves you forward."
    )
    stem_map = _resolve_fuzzy_asset_stems(text, DOOM_ASSETS)
    # 'view' is a common English noun and is now in the
    # non-distinctive stem set — should NOT map "maze view" to the
    # pistol_view / shotgun_view sprites.
    assert "view" not in stem_map, (
        f"'view' stem-matched as asset reference: {stem_map}"
    )


def test_distinctive_pistol_stem_still_matches() -> None:
    """The fix demotes 'view' but pistol/shotgun are still distinctive."""
    text = "the pistol sprite looks wrong"
    stem_map = _resolve_fuzzy_asset_stems(text, DOOM_ASSETS)
    # 'pistol' is distinctive and should match pistol_view + pistol_shot
    # (anything ending in _<pistol>_? — actually walks tokens
    # right-to-left, so pistol_view → 'view' is non-distinct → 'pistol'
    # is picked).
    assert "pistol" in stem_map
    assert "pistol_view" in stem_map["pistol"]


# ----------------------------------------------------------------------
# Fix 2: negation-aware orientation regen blockers.
# ----------------------------------------------------------------------


DOOM_ITER5_FEEDBACK = (
    "do not make a NEW asset, i think the pistal maybe facing the"
    " wrong way, it should be placed like in the asset, so you see"
    " the barrel, i think it is inverted facing the wrong way"
)


def test_negation_helper_basics() -> None:
    text = "do not make a new asset for cpu_warrior"
    new_idx = text.find("new asset")
    assert _phrase_is_negated(text, new_idx) is True

    text2 = "make a new asset for cpu_warrior, no need to flip"
    new_idx2 = text2.find("new asset")
    assert _phrase_is_negated(text2, new_idx2) is False


def test_orientation_fires_when_new_asset_is_negated() -> None:
    """The user is FORBIDDING regen and asking for orientation fix.
    Previously the literal `\\bnew\\s+asset\\b` blocker matched and
    suppressed orientation_change. Negation-aware check fixes this."""
    assert _feedback_is_orientation_change(DOOM_ITER5_FEEDBACK) is True


def test_orientation_still_suppressed_on_unambiguous_regen_request() -> None:
    """Regression guard: the previous test_orientation_regen_blocker_
    suppresses cases must still suppress (no negation present)."""
    assert _feedback_is_orientation_change(
        "make a new asset for cpu_warrior facing the other way"
    ) is False
    assert _feedback_is_orientation_change(
        "regenerate the player kick so it mirrors correctly"
    ) is False
    assert _feedback_is_orientation_change(
        "redraw the punch sprite, the current one is bad"
    ) is False


# ----------------------------------------------------------------------
# Fix 3: inverted-behavior patterns added to _feedback_is_behavior_bug.
# ----------------------------------------------------------------------


DOOM_ITER7_FEEDBACK = (
    "so strange sometimes you see backwards not the way the arrow is"
    " in the maze view, and the down key moves you forward. why are"
    " they directions and views getting reversed at times."
)


def test_inverted_direction_classifies_as_behavior_bug() -> None:
    """User reports 'directions ... getting reversed' — a clear code
    bug that the old negation-only classifier missed entirely."""
    assert _feedback_is_behavior_bug(DOOM_ITER7_FEEDBACK) is True


def test_down_key_moves_forward_classifies_as_behavior_bug() -> None:
    """Standalone input-mismatch sentence must fire on its own."""
    assert _feedback_is_behavior_bug("the down key moves you forward") is True
    assert _feedback_is_behavior_bug("up arrow goes backwards") is True


def test_movement_reversed_classifies_as_behavior_bug() -> None:
    assert _feedback_is_behavior_bug(
        "the movement controls are reversed"
    ) is True
    assert _feedback_is_behavior_bug(
        "controls are inverted, the wrong key fires"
    ) is True


def test_visual_only_inversion_does_not_classify_as_behavior_bug() -> None:
    """Regression guard: 'the dragon looks weird' must stay on the
    art-change path. Pure visual descriptions should not fire the new
    inverted-behavior patterns (they require input/control vocabulary)."""
    assert _feedback_is_behavior_bug(
        "the dragon sprite looks odd"
    ) is False
    # 'inverted' alone (no control noun) should NOT fire behavior_bug.
    # It's an orientation request about a sprite.
    assert _feedback_is_behavior_bug(
        "the pistol sprite is inverted"
    ) is False


def test_doom_iter7_routing_to_media_change_suppressed(tmp_path: Path) -> None:
    """End-to-end: with the inverted-behavior classifier wired up,
    the doom iter-7 feedback must NOT inject the MEDIA-CHANGE
    DIRECTIVE — even though 'view' would have stem-matched (had it
    not been demoted to non-distinctive) and 'directions reversed'
    is now a behavior bug."""
    a = _make_agent(tmp_path)
    a._session_assets = {n: tmp_path / f"{n}.png" for n in DOOM_ASSETS}
    a._session_sounds = {n: tmp_path / f"{n}.ogg" for n in DOOM_SOUNDS}
    a._pending_feedback.append(DOOM_ITER7_FEEDBACK)
    rendered = a._flush_user_injections(base_message="<base>")
    assert "MEDIA-CHANGE DIRECTIVE" not in rendered, (
        "User reported a movement bug; harness wrapped it as ART/SOUND."
    )


def test_doom_iter5_routes_to_orientation_not_media_change(tmp_path: Path) -> None:
    """End-to-end: 'do not make a NEW asset ... facing the wrong way
    ... inverted' must inject ORIENTATION-CHANGE (canvas mirror recipe)
    and NOT MEDIA-CHANGE (asset regen)."""
    a = _make_agent(tmp_path)
    a._session_assets = {n: tmp_path / f"{n}.png" for n in DOOM_ASSETS}
    a._session_sounds = {n: tmp_path / f"{n}.ogg" for n in DOOM_SOUNDS}
    a._pending_feedback.append(DOOM_ITER5_FEEDBACK)
    rendered = a._flush_user_injections(base_message="<base>")
    assert "ORIENTATION-CHANGE DIRECTIVE" in rendered
    assert "MEDIA-CHANGE DIRECTIVE" not in rendered
    assert a._last_turn_contract["orientation_change"] is True


# ----------------------------------------------------------------------
# Fix 4: /rawfeedback kill switch — when off, no directive wrapping
# regardless of what the classifiers think.
# ----------------------------------------------------------------------


def test_raw_feedback_mode_suppresses_all_directives(tmp_path: Path) -> None:
    """With _use_feedback_directives=False, even genuine art-change
    feedback must NOT inject MEDIA-CHANGE / SCOPED-CHANGE / etc."""
    a = _make_agent(tmp_path)
    a._session_assets = {"barrel": tmp_path / "barrel.png"}
    a._session_sounds = {"music": tmp_path / "music.ogg"}
    a._use_feedback_directives = False  # /rawfeedback on
    a._pending_feedback.append(
        "redraw the barrel sprite as a metal canister, no code changes"
    )
    rendered = a._flush_user_injections(base_message="<base>")
    # Basic wrapper still present.
    assert "USER FEEDBACK (HIGHEST PRIORITY)" in rendered
    assert "[USER NOTE]" in rendered
    # Directive blocks all suppressed.
    assert "MEDIA-CHANGE DIRECTIVE" not in rendered
    assert "SCOPED-CHANGE DIRECTIVE" not in rendered
    assert "ORIENTATION-CHANGE DIRECTIVE" not in rendered
    assert "FEEDBACK SCOPE ARBITRATION" not in rendered
    assert "Stems the user referenced" not in rendered
    # Turn contract reflects raw mode.
    assert a._last_turn_contract.get("raw_feedback_mode") is True
    assert a._last_turn_contract["art_change"] is False
    # Pending queue drained.
    assert list(a._pending_feedback) == []


def test_raw_feedback_mode_clears_stale_scoped_constraints(tmp_path: Path) -> None:
    """Switching to raw mode mid-session must not leave scoped
    constraints from a previous classifier-driven turn lingering."""
    a = _make_agent(tmp_path)
    a._session_assets = {"barrel": tmp_path / "barrel.png"}
    # Pretend a prior turn had set scoped constraints.
    a._scoped_constraints = {"mode": "media_only", "max_patch_count": 1}
    a._use_feedback_directives = False
    a._pending_feedback.append("just listen and fix the direction bug")
    a._flush_user_injections(base_message="<base>")
    assert a._scoped_constraints is None


def test_directives_still_fire_when_raw_mode_is_off(tmp_path: Path) -> None:
    """Default mode still routes art-change feedback through MEDIA-CHANGE.
    Regression guard so the raw switch doesn't accidentally globally
    disable the classifiers for users who don't flip it."""
    a = _make_agent(tmp_path)
    a._session_assets = {"barrel": tmp_path / "barrel.png"}
    # _use_feedback_directives defaults to True.
    a._pending_feedback.append(
        "redraw the barrel sprite as a metal canister"
    )
    rendered = a._flush_user_injections(base_message="<base>")
    assert "MEDIA-CHANGE DIRECTIVE" in rendered
