"""Tests for art-change feedback handling.

Motivating trace: games/traces/centipede-game-with-super-nice_20260512_180020.
User asked to change one sprite "no code changes"; model rewrote code
into procedural drawing → regression. The fix:

  1. _feedback_locks_code  detects "no code changes" / "only the asset"
     phrasing and suppresses the one-shot rewrite exemption.
  2. _feedback_is_art_change detects an art-change intent so the
     MEDIA-CHANGE DIRECTIVE can be injected into the next user turn.
  3. _flush_user_injections wires both into the rendered prompt and
     gates _allow_one_rewrite accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    GameAgent,
    _feedback_is_art_change,
    _feedback_requests_existing_media,
    _feedback_mentions_scoped_behavior_change,
    _feedback_is_sound_change,
    _feedback_locks_code,
)


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


# ---------------------------------------------------------------------------
# _feedback_locks_code
# ---------------------------------------------------------------------------

def test_locks_code_explicit_no_code_changes() -> None:
    assert _feedback_locks_code("no changes to the code")
    assert _feedback_locks_code("no code changes please")
    assert _feedback_locks_code("don't change the code")
    assert _feedback_locks_code("do not touch the code")
    assert _feedback_locks_code("without changing the code")


def test_locks_code_only_or_just_phrasing() -> None:
    assert _feedback_locks_code("only the asset")
    assert _feedback_locks_code("just that one asset")
    assert _feedback_locks_code("only the sprite please")
    assert _feedback_locks_code("just the png")


def test_locks_code_centipede_trace_actual_phrasing() -> None:
    # Verbatim from games/traces/centipede-game-with-super-nice_20260512_180020.
    feedback = (
        "make the additional segments more round with moving legs, so it "
        "looks more connected, only change the centipiede_tail no other "
        "asset or code, just that one asset no changes to the code."
    )
    assert _feedback_locks_code(feedback)


def test_locks_code_normal_feedback_does_not_match() -> None:
    assert not _feedback_locks_code("fix the mouse look")
    assert not _feedback_locks_code("add powerups and a boss fight")
    assert not _feedback_locks_code("the ship moves too slow")
    assert not _feedback_locks_code("redraw the player to be bigger")


# ---------------------------------------------------------------------------
# _feedback_is_art_change
# ---------------------------------------------------------------------------

def test_art_change_matches_asset_name() -> None:
    assert _feedback_is_art_change(
        "make the centipede_tail rounder", ["centipede_tail", "player"]
    )


def test_art_change_matches_canonicalized_asset_name() -> None:
    # User typed "centipede tail" (space); asset name is "centipede_tail".
    assert _feedback_is_art_change(
        "make the centipede tail rounder", ["centipede_tail"]
    )
    # Hyphen variant.
    assert _feedback_is_art_change(
        "redraw the player-ship", ["player_ship"]
    )


def test_art_change_matches_noun_plus_verb() -> None:
    assert _feedback_is_art_change("change the sprite", [])
    assert _feedback_is_art_change("redraw the art", [])
    assert _feedback_is_art_change("update the graphics", [])


def test_existing_media_request_suppresses_regen_intent() -> None:
    assert _feedback_requests_existing_media(
        "some animations are missing, dont redo them, use them, the original ones"
    )
    assert _feedback_requests_existing_media(
        "use existing p2_idle frames, do not regenerate"
    )
    assert not _feedback_requests_existing_media(
        "redraw the p2 block animation with new art"
    )


def test_art_change_negative_cases() -> None:
    assert not _feedback_is_art_change("the ship moves too slow", [])
    assert not _feedback_is_art_change("add a boss fight", [])
    # Asset noun without a verb — not enough signal.
    assert not _feedback_is_art_change("the sprite", [])


# ---------------------------------------------------------------------------
# _flush_user_injections integration
# ---------------------------------------------------------------------------

def test_flush_injects_asset_change_directive_when_code_locked(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {
        "centipede_tail": tmp_path / "centipede_tail.png",
        "player_ship": tmp_path / "player_ship.png",
    }
    a._pending_feedback.append(
        "only change the centipede_tail, no changes to the code"
    )
    rendered = a._flush_user_injections(base_message="<base>")

    assert "USER FEEDBACK (HIGHEST PRIORITY)" in rendered
    assert "MEDIA-CHANGE DIRECTIVE" in rendered
    assert "centipede_tail" in rendered
    assert "player_ship" in rendered
    assert "<assets>" in rendered


def test_flush_suppresses_rewrite_exemption_when_code_locked(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"tail": tmp_path / "tail.png"}
    a._pending_feedback.append(
        "only the tail asset, no code changes"
    )
    assert a._allow_one_rewrite is False
    a._flush_user_injections(base_message="<base>")
    assert a._allow_one_rewrite is False, (
        "code-lock phrasing must suppress the rewrite exemption"
    )


def test_flush_arms_rewrite_exemption_on_normal_feedback(
    tmp_path: Path,
) -> None:
    # Preserves the prior contract: plain feedback still arms the
    # one-shot rewrite license.
    a = _make_agent(tmp_path)
    a._pending_feedback.append("fix the mouse look and add powerups")
    assert a._allow_one_rewrite is False
    a._flush_user_injections(base_message="<base>")
    assert a._allow_one_rewrite is True


# ---------------------------------------------------------------------------
# _feedback_is_sound_change
# ---------------------------------------------------------------------------

def test_sound_change_matches_sound_name() -> None:
    assert _feedback_is_sound_change(
        "make the laser_shoot louder", ["laser_shoot", "explosion"]
    )


def test_sound_change_matches_noun_plus_verb() -> None:
    assert _feedback_is_sound_change("remake the laser sound", [])
    assert _feedback_is_sound_change("regenerate the music", [])
    assert _feedback_is_sound_change("change the sfx", [])


def test_sound_change_negative_cases() -> None:
    assert not _feedback_is_sound_change("make the ship turn faster", [])
    assert not _feedback_is_sound_change("redraw the sprite", ["explosion"])


def test_flush_injects_sound_directive_when_sound_change(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_sounds = {"laser": tmp_path / "laser.ogg"}
    a._pending_feedback.append(
        "remake the laser sound to be deeper, only the audio"
    )
    rendered = a._flush_user_injections(base_message="<base>")

    assert "MEDIA-CHANGE DIRECTIVE" in rendered
    assert "<sounds>" in rendered
    assert "laser" in rendered


def test_flush_injects_combined_directive_when_assets_and_sounds(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"player": tmp_path / "player.png"}
    a._session_sounds = {"laser": tmp_path / "laser.ogg"}
    a._pending_feedback.append(
        "redraw the player and remake the laser, no code changes"
    )
    rendered = a._flush_user_injections(base_message="<base>")

    assert "MEDIA-CHANGE DIRECTIVE" in rendered
    assert "<assets>" in rendered and "<sounds>" in rendered
    assert "player" in rendered and "laser" in rendered
    assert a._allow_one_rewrite is False  # "no code changes" locked it


def test_flush_no_directive_when_no_assets_in_session(
    tmp_path: Path,
) -> None:
    # Code-lock alone (without any session assets to target) still
    # suppresses the rewrite exemption, but we skip the directive
    # because we have nothing to point the model at.
    a = _make_agent(tmp_path)
    a._pending_feedback.append("no code changes, just tune the physics")
    a._flush_user_injections(base_message="<base>")
    # No assets → no directive.
    # (We don't assert _allow_one_rewrite here; code_locked still
    # suppresses it, and that's the desired behavior.)
    assert a._allow_one_rewrite is False


def test_flush_persists_scoped_constraints_for_media_only_turn(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"tail": tmp_path / "tail.png"}
    a._pending_feedback.append("redraw only the tail sprite, no code changes")
    a._flush_user_injections(base_message="<base>")

    assert a._scoped_constraints is not None
    assert a._scoped_constraints["mode"] == "media_only"
    assert a._scoped_constraints["media_name_lock"] is True
    assert a._scoped_constraints["allowed_asset_names"] == ["tail"]


def test_flush_routes_behavior_worded_animation_feedback_to_patch_mode(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    text = (
        "only change the animation so the kick turns around to face the CPU, "
        "no code changes elsewhere"
    )
    assert _feedback_mentions_scoped_behavior_change(text) is True
    a._pending_feedback.append(text)
    a._flush_user_injections(base_message="<base>")

    assert a._scoped_constraints is not None
    assert a._scoped_constraints["mode"] == "single_patch"
    assert a._scoped_constraints["max_patch_count"] == 1
    assert a._scoped_constraints["require_scope_probe"] is True


def test_flush_sets_preserve_baseline_guard_on_clean_scoped_tweak(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"player": tmp_path / "player.png"}
    a._previous_report_ok = True
    a._previous_report = {
        "errors": [],
        "soft_warnings": [],
        "page_errors": [],
        "console_errors": [],
        "probes": [{"ok": True}],
    }
    a._pending_feedback.append("only make movement faster; no other changes")
    a._flush_user_injections(base_message="<base>")

    assert a._scoped_constraints is not None
    assert a._scoped_constraints["mode"] == "single_patch"
    assert a._scoped_constraints["preserve_baseline"] is True
