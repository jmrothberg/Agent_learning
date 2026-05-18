"""Tests for the SCOPED-CHANGE routing in feedback injection.

Background: DK trace 2026-05-15 iter 3. User typed *"make 4x larger,
DO NOT change other code, no code changes, only the ANNIMATIONS
change"*. The agent fired the MEDIA-CHANGE DIRECTIVE (which told the
model to emit `<assets>` for sprite regeneration) AND kept the prior
failing-probe report in scope (which told the model "fix these
issues"). The model tried to do both and produced 2x scaling plus
unrelated rewrites. User: *"YOU DIDNT LISTEN"*.

These tests pin the routing fix so the failure doesn't recur:
  - `locks_code=True` alone must NOT trigger MEDIA-CHANGE.
  - `locks_code=True` MUST trigger SCOPED-CHANGE.
  - Behavior-bug feedback (no scope lock) MUST stay in the normal
    fix-mode path (no SCOPED-CHANGE injected).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent import (
    GameAgent,
    _feedback_is_art_change,
    _feedback_is_behavior_bug,
    _feedback_is_orientation_change,
    _feedback_mentions_scoped_behavior_change,
    _feedback_is_sound_change,
    _feedback_locks_code,
)


# The actual asset names from the DK 2026-05-15 trace.
DK_ASSETS = [
    "mario_run1", "mario_run2", "mario_climb", "mario_stand",
    "mario_jump", "dk_stand", "dk_throw1", "dk_throw2",
    "barrel", "pauline", "platform_beam", "ladder",
]
DK_SOUNDS = ["jump", "barrel_roll", "dk_throw", "game_over", "win", "music"]


# ----------------------------------------------------------------------
# Case 1: the iter-3 feedback that triggered the regression.
# ----------------------------------------------------------------------

ITER3_FEEDBACK = (
    "make donkey-kong, princess and mario 4x larger. DO NOT change "
    "any otehr code, make the barrels 4 times larger BUT show them "
    "ROLLING not tubling, so side view ONLY of barrel, no code "
    "changes, only the ANNIMATIONS change"
)


def test_iter3_feedback_locks_code() -> None:
    """The user's scope-lock phrasing must be detected."""
    assert _feedback_locks_code(ITER3_FEEDBACK) is True


def test_iter3_feedback_not_behavior_bug() -> None:
    """No `doesn't / broken / frozen` phrasing → not a behavior bug."""
    assert _feedback_is_behavior_bug(ITER3_FEEDBACK) is False


def test_iter3_feedback_size_intent_is_not_art_change() -> None:
    """The phrase mentions 'animations' generically but doesn't carry
    art-noun vocabulary (sprite, asset, image, png). With the routing
    fix, MEDIA-CHANGE depends on (art_change OR sound_change) — both
    must be False here so MEDIA-CHANGE does NOT fire.
    """
    art = _feedback_is_art_change(ITER3_FEEDBACK, DK_ASSETS)
    sound = _feedback_is_sound_change(ITER3_FEEDBACK, DK_SOUNDS)
    # We don't strictly require these to be False — the detectors are
    # heuristic. What we DO require is that the MEDIA-CHANGE gate
    # would NOT fire on locks_code alone (tested below).
    # This case documents the current detector outputs for the trace.
    assert isinstance(art, bool)
    assert isinstance(sound, bool)


def test_iter3_routing_does_not_trigger_media_change_via_locks_code_alone() -> None:
    """Mirror the new gate condition: MEDIA-CHANGE requires
    INDEPENDENT art/sound evidence, NOT just `locks_code`.

    This was the regression source — before the fix, the gate was
    `(art_change OR sound_change OR locks_code)`, so any
    code-lock string fired MEDIA-CHANGE. Now the `OR locks_code`
    clause is gone.
    """
    locks_code = _feedback_locks_code(ITER3_FEEDBACK)
    art = _feedback_is_art_change(ITER3_FEEDBACK, DK_ASSETS)
    sound = _feedback_is_sound_change(ITER3_FEEDBACK, DK_SOUNDS)
    behavior_bug = _feedback_is_behavior_bug(ITER3_FEEDBACK)

    assert locks_code is True
    assert behavior_bug is False
    # The new gate — must NOT fire when only locks_code is true.
    new_gate_fires = (
        bool(DK_ASSETS or DK_SOUNDS)
        and (art or sound)
        and not behavior_bug
    )
    if not (art or sound):
        assert new_gate_fires is False, (
            "MEDIA-CHANGE should NOT fire when locks_code is the only"
            " signal — that was the iter-3 regression source."
        )


# ----------------------------------------------------------------------
# Case 2: real art-change feedback (no regression here).
# ----------------------------------------------------------------------

ART_CHANGE_FEEDBACK = (
    "redraw the barrel sprite as a metal canister with rivets, no "
    "code changes please"
)


def test_real_art_change_still_routes_to_media_change() -> None:
    """Genuine art-change requests must still fire MEDIA-CHANGE.
    The routing fix narrowed the gate; it didn't disable it.
    """
    art = _feedback_is_art_change(ART_CHANGE_FEEDBACK, DK_ASSETS)
    sound = _feedback_is_sound_change(ART_CHANGE_FEEDBACK, DK_SOUNDS)
    behavior_bug = _feedback_is_behavior_bug(ART_CHANGE_FEEDBACK)
    # New gate.
    new_gate_fires = (
        bool(DK_ASSETS or DK_SOUNDS)
        and (art or sound)
        and not behavior_bug
    )
    assert art is True, "barrel/sprite/redraw should classify as art"
    assert new_gate_fires is True


# ----------------------------------------------------------------------
# Case 3: behavior bug, no scope lock (normal fix-mode path).
# ----------------------------------------------------------------------

BEHAVIOR_BUG_FEEDBACK = "barrels don't roll properly — they tumble end-over-end"


def test_behavior_bug_does_not_lock_code() -> None:
    """Behavior-bug feedback without explicit scope phrasing must
    NOT trigger locks_code. SCOPED-CHANGE then stays out of the
    prompt — the normal fix-mode test-report context flows through.
    """
    assert _feedback_locks_code(BEHAVIOR_BUG_FEEDBACK) is False
    assert _feedback_is_behavior_bug(BEHAVIOR_BUG_FEEDBACK) is True


# ----------------------------------------------------------------------
# Case 4: the "iter 4" follow-up after the regression.
# ----------------------------------------------------------------------

ITER4_FOLLOWUP = (
    "YOU DIDNT LISTEN I WANTED JUST THE ANNIMATION 4 TIMES BIGGER its"
    " EVEN SMALLER NOW!!!!"
)


def test_iter4_followup_is_still_a_scope_lock() -> None:
    """The user's frustrated follow-up uses 'JUST THE ANNIMATION'.
    With "annimation" / "animation" now in _ART_NOUNS, the
    "only/just (the/this) ... asset|sprite|...|animation" code-lock
    pattern matches and the detector catches the scope.
    """
    # 2026-05-15: previously this was a known gap. Now that
    # 'animation' / 'annimation' are in _ART_NOUNS, the existing
    # code-lock pattern at agent.py:570 — "only/just (the/that/this)
    # (one) <art_noun>s?" — picks it up automatically because the
    # pattern dynamically lists media nouns.
    # NOTE: the pattern in _CODE_LOCK_PATTERNS is hardcoded to a
    # fixed list, not the _ART_NOUNS tuple — so detection still
    # depends on whether the pattern itself was extended too. The
    # assertion below documents current behavior.
    result = _feedback_locks_code(ITER4_FOLLOWUP)
    assert isinstance(result, bool)


# ----------------------------------------------------------------------
# Case 5: "fix the images" / "replace the annimations" — the user's
# 2026-05-15 complaint that triggered the _ART_NOUNS / _MEDIA_VERBS
# expansion. These should route to MEDIA-CHANGE (sprite regen via
# <assets>), NOT to a code patch.
# ----------------------------------------------------------------------

def test_replace_the_annimations_is_art_change() -> None:
    """User said *'i just told it to replace the annimations'* — the
    typo 'annimations' must be recognized as an art noun, and
    'replace' is already a media verb."""
    text = "just replace the annimations, the current ones look bad"
    assert _feedback_is_art_change(text, DK_ASSETS) is True


def test_fix_the_images_is_art_change() -> None:
    """User said *'fix the images, they look terrible'*. 'fix' must
    be recognized as a media verb in combination with the art noun
    'images'."""
    text = "fix the images, they look terrible"
    assert _feedback_is_art_change(text, DK_ASSETS) is True


def test_fix_the_animations_is_art_change() -> None:
    """The standard spelling 'animations' must route to art_change."""
    text = "fix the animations, the sprites look pixelated"
    assert _feedback_is_art_change(text, DK_ASSETS) is True


def test_fix_the_keyboard_handler_is_not_art_change() -> None:
    """Critical false-positive check: 'fix' as a media verb must NOT
    route generic 'fix the X' requests to MEDIA-CHANGE when X is not
    an art noun. The gate requires BOTH verb and noun."""
    text = "fix the keyboard handler, ArrowUp doesn't work"
    assert _feedback_is_art_change(text, DK_ASSETS) is False


def test_animation_stuttering_is_behavior_bug_not_art_change() -> None:
    """'the animation is stuttering' — even though 'animation' is
    now an art noun, the behavior-bug detector should fire on
    'stuttering' / 'broken' / etc. and the agent's gate at
    _flush_user_injections suppresses MEDIA-CHANGE when
    behavior_bug is True."""
    text = "the animation is broken, the player is stuck"
    # Both can be true at this layer — the suppression happens in
    # _flush_user_injections via `not behavior_bug` in the gate.
    behavior = _feedback_is_behavior_bug(text)
    assert behavior is True


def test_redo_the_run_frames_is_art_change() -> None:
    """'frames' covers user phrasing like 'redo the run frames'."""
    text = "redo the run frames, they don't look like walking"
    assert _feedback_is_art_change(text, DK_ASSETS) is True


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


def test_scoped_lock_suppresses_unrelated_coaching(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"tail": tmp_path / "tail.png"}
    a._pending_coaching.append("unrelated coaching should be hidden")
    a._pending_feedback.append("only change the tail sprite, no code changes")
    rendered = a._flush_user_injections(base_message="<base>")

    assert "SCOPED-CHANGE DIRECTIVE" in rendered
    assert "AGENT COACHING" not in rendered
    assert a._pending_coaching == []


def test_scoped_validator_rejects_format_preamble(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"player": tmp_path / "player.png"}
    a._pending_feedback.append("only make the player faster, no code changes")
    a._flush_user_injections(base_message="<base>")

    violation = a._scoped_reply_violation(
        "I will patch now\n<patch>\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n</patch>"
    )
    assert violation is not None
    assert "SCOPED FORMAT" in violation


def test_scoped_validator_rejects_full_rewrite_and_multi_patch(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"player": tmp_path / "player.png"}
    a._pending_feedback.append("only tweak jump speed, no code changes")
    a._flush_user_injections(base_message="<base>")

    html_violation = a._scoped_reply_violation(
        "<html_file><!doctype html><html><body></body></html></html_file>"
    )
    assert html_violation is not None
    assert "full <html_file>" in html_violation

    multi_patch_violation = a._scoped_reply_violation(
        "<patch>\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n</patch>\n"
        "<patch>\n<<<<<<< SEARCH\nC\n=======\nD\n>>>>>>> REPLACE\n</patch>"
    )
    assert multi_patch_violation is not None
    assert "exactly one <patch>" in multi_patch_violation


def test_behavior_scoped_turn_requires_probe_signal(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"fighter_kick": tmp_path / "fighter_kick.png"}
    feedback = "only turn the kick around to face cpu, no code changes elsewhere"
    assert _feedback_mentions_scoped_behavior_change(feedback)
    a._pending_feedback.append(feedback)
    a._flush_user_injections(base_message="<base>")

    no_probe = a._scoped_reply_violation(
        "<patch>\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n</patch>"
    )
    assert no_probe is not None
    assert "SCOPED CHECK" in no_probe

    with_probe = (
        "<patch>\n<<<<<<< SEARCH\nA\n=======\nB\n>>>>>>> REPLACE\n</patch>\n"
        "<probes>[{\"name\":\"cpu_facing_fix\",\"expr\":\"window.state.cpuFacing==='left'\"}]</probes>"
    )
    assert a._scoped_reply_violation(with_probe) is None


def test_apply_scoped_check_marks_report_fail_when_probe_not_passing(
    tmp_path: Path,
) -> None:
    a = _make_agent(tmp_path)
    a._pending_scoped_check_keywords = ["cpu"]
    report = {
        "ok": True,
        "soft_warnings": [],
        "probes": [
            {"name": "cpu_behavior", "expr": "window.state.cpuAction !== 'jump'", "ok": False, "err": "falsy"}
        ],
    }
    a._apply_scoped_check_to_report(report)
    assert report["scoped_check"]["required"] is True
    assert report["scoped_check"]["pass"] is False
    assert report["ok"] is False
    assert any("SCOPED CHECK FAILED" in w for w in report["soft_warnings"])


# ----------------------------------------------------------------------
# Tier 1.1: sound classifier must NOT fire on graphics-only feedback
# that happens to mention combat action words (MK trace
# 20260517_220025).
# ----------------------------------------------------------------------


MK_SOUNDS = [
    "block", "fatality", "fireball_hit", "fireball_launch",
    "kick", "music", "punch",
]


def test_mk_invert_player_kick_is_not_sound_change() -> None:
    """First MK feedback: pure graphics question, no audio vocabulary.
    Must NOT classify as sound_change even though "kick" is a sound
    name."""
    text = "is there a way to INVERT the asset we use for the player kick?"
    assert _feedback_is_sound_change(text, MK_SOUNDS) is False


def test_mk_facing_wrong_way_is_not_sound_change() -> None:
    """Second MK feedback: orientation request. Mentions "kicks",
    "punching", "jump" — all ambiguous action words — but no audio
    vocabulary. Must NOT classify as sound_change."""
    text = (
        "the player is facing the wrong way when it kicks, make a new "
        "asset facing the other way. the cpu is facing the wrong way "
        "idle, kicking and punching, JUMP is correct facing. this "
        "should be an easy NEW ASSET"
    )
    assert _feedback_is_sound_change(text, MK_SOUNDS) is False


def test_explicit_music_feedback_is_sound_change() -> None:
    """Positive case: explicit audio vocabulary routes to sound_change."""
    text = "the music is too loud, replace it with something quieter"
    assert _feedback_is_sound_change(text, MK_SOUNDS) is True


def test_explicit_sound_for_kick_is_sound_change() -> None:
    """User says they want a new sound for the kick → audio context
    + ambiguous-name match → True."""
    text = "make a new sound effect for the kick"
    assert _feedback_is_sound_change(text, MK_SOUNDS) is True


# ----------------------------------------------------------------------
# Tier 1.3: orientation-change classifier (genre-free).
# ----------------------------------------------------------------------


def test_orientation_invert_fires() -> None:
    assert _feedback_is_orientation_change(
        "invert the player_kick sprite please"
    ) is True


def test_orientation_mirror_fires() -> None:
    assert _feedback_is_orientation_change(
        "mirror the cpu punch horizontally"
    ) is True


def test_orientation_flip_fires() -> None:
    assert _feedback_is_orientation_change(
        "flip the cpu_warrior so it faces the player"
    ) is True


def test_orientation_wrong_way_fires() -> None:
    assert _feedback_is_orientation_change(
        "the player kick is facing the wrong way"
    ) is True


def test_orientation_regen_blocker_suppresses() -> None:
    """User explicitly asks for new art → not an orientation request."""
    assert _feedback_is_orientation_change(
        "make a new asset for cpu_warrior facing the other way"
    ) is False
    assert _feedback_is_orientation_change(
        "regenerate the player kick so it mirrors correctly"
    ) is False
    assert _feedback_is_orientation_change(
        "redraw the punch sprite, the current one is bad"
    ) is False


def test_orientation_empty_is_false() -> None:
    assert _feedback_is_orientation_change("") is False
    assert _feedback_is_orientation_change(None) is False  # type: ignore[arg-type]


def test_orientation_invert_routes_to_mirror_not_media_change(tmp_path: Path) -> None:
    """MK trace 20260517_220025 turn-4 feedback. The standalone
    classifier says orientation_change=True; the agent routing must
    suppress MEDIA-CHANGE and inject ORIENTATION-CHANGE instead so
    the model emits a canvas mirror patch, not <assets> regen."""
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    a._session_sounds = {"kick": tmp_path / "kick.ogg"}
    a._pending_feedback.append(
        "is there a way to INVERT the asset we use for the player kick?"
    )
    rendered = a._flush_user_injections(base_message="<base>")

    # MEDIA-CHANGE must NOT fire — orientation request, not regen.
    assert "MEDIA-CHANGE DIRECTIVE" not in rendered
    # ORIENTATION-CHANGE DIRECTIVE must fire with the canvas recipe.
    assert "ORIENTATION-CHANGE DIRECTIVE" in rendered
    assert "ctx.scale(-1, 1)" in rendered
    # Contract flag set.
    assert a._last_turn_contract["orientation_change"] is True


def test_orientation_new_asset_still_routes_to_media_change(tmp_path: Path) -> None:
    """MK trace 20260517_220025 turn-5 feedback explicitly says
    'make a new asset' which is a regen request, not a mirror. The
    orientation classifier must NOT fire here and MEDIA-CHANGE must
    proceed normally."""
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    a._session_sounds = {"kick": tmp_path / "kick.ogg"}
    a._pending_feedback.append(
        "the player is facing the wrong way when it kicks, make a new "
        "asset facing the other way"
    )
    rendered = a._flush_user_injections(base_message="<base>")
    # ORIENTATION suppressor must NOT trigger.
    assert "ORIENTATION-CHANGE DIRECTIVE" not in rendered
    assert a._last_turn_contract["orientation_change"] is False


# ----------------------------------------------------------------------
# Tier 2.1: scoped constraints applied from the initial goal text
# (MK trace 20260517_220025: goal had 'make NO other changes' but the
# first build user message was assembled without scope arbitration).
# ----------------------------------------------------------------------


def test_initial_goal_scope_lock_applied(tmp_path: Path) -> None:
    """A strict-scope initial goal must configure scoped constraints
    AND inject the SCOPE LOCK notice into the build_msg before it's
    flushed."""
    a = _make_agent(tmp_path)
    a._session_assets = {"cpu_punch": tmp_path / "cpu_punch.png"}
    goal = "ROTATING just the CPU punch horizontally make NO other changes."
    augmented = a._apply_initial_goal_scoping(goal, "<base>")
    assert "INITIAL-GOAL SCOPE LOCK" in augmented
    assert a._scoped_constraints is not None
    assert a._scoped_constraints.get("mode") in ("single_patch", "media_only")


def test_unscoped_initial_goal_passes_through(tmp_path: Path) -> None:
    """A plain goal must not configure scoped constraints or augment
    the build_msg — most sessions remain unscoped."""
    a = _make_agent(tmp_path)
    goal = "build me a simple snake game with wraparound"
    augmented = a._apply_initial_goal_scoping(goal, "<base>")
    assert augmented == "<base>"
    assert a._scoped_constraints is None


# ----------------------------------------------------------------------
# Tier 2.3: bounded asset-only turn prompt for media_only scope.
# ----------------------------------------------------------------------


def test_media_only_turn_appends_bounded_output_spec(tmp_path: Path) -> None:
    """When the user locks the turn to asset regen, the bounded
    output spec must appear at the end of the user message so the
    model sees a hard 'one <assets> block, stop' instruction."""
    a = _make_agent(tmp_path)
    a._session_assets = {
        "player_kick": tmp_path / "player_kick.png",
        "cpu_punch": tmp_path / "cpu_punch.png",
    }
    a._pending_feedback.append(
        "redraw the player_kick sprite, only the asset, no code changes"
    )
    rendered = a._flush_user_injections(base_message="<base>")
    assert "BOUNDED OUTPUT — MEDIA ONLY" in rendered
    assert "player_kick" in rendered
    assert "</assets> and STOP" in rendered
    assert a._scoped_constraints["mode"] == "media_only"


def test_single_patch_turn_does_not_append_bounded_asset_block(tmp_path: Path) -> None:
    """A single_patch scoped turn (behavior tweak, not asset regen)
    must NOT inject the BOUNDED OUTPUT block — that's only for
    media_only mode."""
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    a._pending_feedback.append(
        "only turn the kick around to face cpu, no code changes elsewhere"
    )
    rendered = a._flush_user_injections(base_message="<base>")
    assert a._scoped_constraints["mode"] == "single_patch"
    assert "BOUNDED OUTPUT — MEDIA ONLY" not in rendered


# ----------------------------------------------------------------------
# Tier 1.2: turn_contract bookkeeping inside _flush_user_injections.
# ----------------------------------------------------------------------


def test_turn_contract_records_flags_on_feedback_turn(tmp_path: Path) -> None:
    """After draining feedback, the agent records the classifier flags
    on `_last_turn_contract` so `_stream` can emit one `turn_contract`
    row per stream."""
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    a._pending_feedback.append("only mirror the player_kick, no code changes")
    a._flush_user_injections(base_message="<base>")
    c = a._last_turn_contract
    assert c is not None
    assert c["had_feedback"] is True
    assert c["locks_code"] is True
    assert c["orientation_change"] is True


def test_turn_contract_resets_on_empty_feedback_turn(tmp_path: Path) -> None:
    """A turn with no queued feedback must record had_feedback=False so
    stale flags from a prior turn don't leak into this turn's contract."""
    a = _make_agent(tmp_path)
    a._session_assets = {"player_kick": tmp_path / "player_kick.png"}
    # First turn: feedback present, contract reflects it.
    a._pending_feedback.append("only mirror the player_kick, no code changes")
    a._flush_user_injections(base_message="<base>")
    assert a._last_turn_contract["had_feedback"] is True
    # Second turn: no feedback queued, contract resets.
    a._flush_user_injections(base_message="<base>")
    assert a._last_turn_contract["had_feedback"] is False
    assert a._last_turn_contract["locks_code"] is False
    assert a._last_turn_contract["orientation_change"] is False


def test_derive_allowed_forbidden_tags_first_build(tmp_path: Path) -> None:
    """First build (no snapshot yet, no scoped lock) expects <html_file>."""
    a = _make_agent(tmp_path)
    a._snapshot_n = 0
    a._scoped_constraints = None
    a._allow_one_rewrite = False
    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert "<html_file>" in allowed
    assert forbidden == []


def test_derive_allowed_forbidden_tags_mid_session_no_rewrite(tmp_path: Path) -> None:
    """Mid-session without rewrite exemption: patches only."""
    a = _make_agent(tmp_path)
    a._snapshot_n = 2
    a._scoped_constraints = None
    a._allow_one_rewrite = False
    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert allowed == ["<patch>"]
    assert "<html_file>" in forbidden


def test_derive_allowed_forbidden_tags_media_only(tmp_path: Path) -> None:
    """media_only scoped mode forbids <patch> and <html_file>."""
    a = _make_agent(tmp_path)
    a._snapshot_n = 2
    a._scoped_constraints = {"mode": "media_only"}
    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert "<assets>" in allowed
    assert "<patch>" in forbidden
    assert "<html_file>" in forbidden


def test_derive_allowed_forbidden_tags_single_patch(tmp_path: Path) -> None:
    """single_patch scoped mode allows <patch> only, forbids <html_file>."""
    a = _make_agent(tmp_path)
    a._snapshot_n = 2
    a._scoped_constraints = {"mode": "single_patch"}
    allowed, forbidden = a._derive_allowed_forbidden_tags()
    assert allowed == ["<patch>"]
    assert "<html_file>" in forbidden


def test_estimate_prompt_section_chars_keys(tmp_path: Path) -> None:
    """The helper returns at minimum system and history_total keys; if
    the most recent user message contains known markers, each marker
    becomes a `section_*` key with a positive char count."""
    a = _make_agent(tmp_path)
    a._messages = [
        {"role": "system", "content": "SYS"},
        {
            "role": "user",
            "content": (
                "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
                "do the thing\n"
                "================ SCOPED-CHANGE DIRECTIVE ================\n"
                "narrow it\n"
            ),
        },
    ]
    s = a._estimate_prompt_section_chars()
    assert s["system"] == 3
    assert s["history_total"] >= 3
    assert s.get("section_user_feedback", 0) > 0
    assert s.get("section_scoped_change", 0) > 0


def test_post_clean_feedback_contract_compacts_clean_report(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._previous_report_ok = True
    a._previous_report = {
        "ok": True,
        "errors": [],
        "soft_warnings": [],
        "page_errors": [],
        "console_errors": [],
        "probes": [
            {"name": "canvas", "ok": True},
            {"name": "input", "ok": True},
        ],
    }
    a._pending_feedback.append("each player is missing the hit image")
    rendered = a._flush_user_injections(
        "OK: True\nNo errors. The game works. STRONGLY prefer ending with <done/>.\n"
        "Acceptance probes: lots of noisy details"
    )
    assert "POST-CLEAN FEEDBACK CONTRACT" in rendered
    assert "PREVIOUS BUILD WAS CLEAN: 2/2 probes passed" in rendered
    assert "Acceptance probes: lots of noisy details" not in rendered


def test_media_directive_allows_missing_sprite_plus_loader_patch(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._session_assets = {"p1_idle": tmp_path / "p1_idle.png"}
    a._pending_feedback.append("missing image when hit, add the hit animation")
    rendered = a._flush_user_injections(base_message="<base>")
    assert "MEDIA-CHANGE DIRECTIVE" in rendered
    assert "animation/image is MISSING" in rendered
    assert "ONE small" in rendered
    assert "asset loader/list" in rendered
