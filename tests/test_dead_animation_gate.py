"""Tests for the dead-animation gate + context-specific animation critic.

Background — trace `a-super-graphical-adventure-ga_20260530` (3D adventure):
the player + NPC walk frames were generated via from_image at strength 0.40
chained frame-from-previous, so every frame came back ~99% identical to idle
(deltas 0.004-0.012). The characters slid across the field as static images —
"fully animated walking" never happened. The near-idle detector measured it
but only WARNED; the visual critic's movement recipe had no animation question.

Fixes verified here:
  - `_dead_anim_frames` (near-identical from_image frames) is now ADVISORY:
    `_apply_dead_animation_check_to_report` surfaces it in `warnings` (the
    non-gating channel) and does NOT flip report["ok"]=False. Changed
    2026-06-01 after trace build-a-single-screen-2d-fight_20260531_214215: a
    cosmetic img2img sprite warning held a behaviorally-correct build hostage
    across BOTH a local model and Opus 4.8 (probes 8/8, patches applied, input
    PASS — yet ok stayed False forever because the prescribed img2img fix is
    the path the user's own A/B finding marks as broken). Behavioral probes
    gate shipping; cosmetics inform.
  - `_animation_expected` is signal-driven (declared/dead frames or game
    controls), not a genre table.
  - `_augment_recipe_for_animation` appends a context-specific "is it actually
    walking/kicking?" question to a CLONE of the recipe (cached recipe intact).
  - tools.py: asset_usage accepts WebGL texture wiring (three.js never calls
    drawImage); before_mid_after consults the real static_action signal, not a
    requestAnimationFrame-in-source rubber-stamp.

Pure-logic paths run live; browser-dependent recipe execution is source-pinned.
"""

from __future__ import annotations

import inspect

from unittest.mock import MagicMock

import agent
import tools
from agent import GameAgent
from memory import VisualPlaytestRecipe


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ---- dead-animation advisory (NOT a hard gate) ------------------------------

def test_dead_anim_frames_are_advisory_not_blocking(tmp_path):
    """Dead frames surface in `warnings` and must NOT flip ok=False.

    Regression guard for the unwinnable loop in trace
    build-a-single-screen-2d-fight_20260531_214215: a behaviorally-correct
    build (probes passing) must still ship even with a cosmetic dead-sprite
    warning, because the prescribed img2img remedy is the path the user's A/B
    finding documents as broken.
    """
    a = _make_agent(tmp_path)
    a._dead_anim_frames = {"hero_walk1": 0.007, "hero_walk2": 0.006}
    report = {"ok": True, "soft_warnings": [], "warnings": []}
    a._apply_dead_animation_check_to_report(report)
    # ok is UNTOUCHED — cosmetics do not gate shipping.
    assert report["ok"] is True
    # not routed to the gating channel.
    assert report["soft_warnings"] == []
    # but the model still sees it, in the non-gating `warnings` channel.
    joined = "\n".join(report["warnings"])
    assert "DEAD ANIMATION" in joined
    assert "hero_walk1" in joined and "hero_walk2" in joined
    # advisory framing so the model knows it is not a blocker.
    assert "does not block shipping" in joined
    assert report.get("dead_anim_frames") == {"hero_walk1": 0.007, "hero_walk2": 0.006}


def test_dead_anim_does_not_flip_a_clean_report(tmp_path):
    """Even when the only finding is dead frames, a probe-clean report stays ok."""
    a = _make_agent(tmp_path)
    a._dead_anim_frames = {"player_block": 0.029}
    report = {"ok": True, "soft_warnings": [], "warnings": []}
    a._apply_dead_animation_check_to_report(report)
    assert report["ok"] is True


def test_no_dead_frames_is_noop(tmp_path):
    a = _make_agent(tmp_path)
    a._dead_anim_frames = {}
    report = {"ok": True, "soft_warnings": [], "warnings": []}
    a._apply_dead_animation_check_to_report(report)
    assert report["ok"] is True
    assert report["soft_warnings"] == []
    assert report["warnings"] == []


# ---- animation-expected signal ---------------------------------------------

def test_animation_expected_from_declared_frames(tmp_path):
    a = _make_agent(tmp_path)
    assert a._animation_expected() is False  # nothing declared yet, benign goal
    a._declared_anim_frames = True
    assert a._animation_expected() is True


def test_animation_expected_from_dead_frames(tmp_path):
    a = _make_agent(tmp_path)
    a._dead_anim_frames = {"hero_walk1": 0.005}
    assert a._animation_expected() is True


# ---- context-specific recipe augmentation ----------------------------------

def _recipe(checklist):
    return VisualPlaytestRecipe(
        id="canvas-controllable-player",
        kind="visual_playtest",
        content="x",
        recipe={"checklist": list(checklist), "fix_hint": "base hint"},
    )


def test_augment_appends_animation_question_without_mutating_original(tmp_path):
    a = _make_agent(tmp_path)
    a._declared_anim_frames = True
    base = _recipe(["Is there a player visible?"])
    aug = a._augment_recipe_for_animation(base)
    # clone, not the same object
    assert aug is not base
    # original untouched (cached recipe must stay clean)
    assert base.recipe["checklist"] == ["Is there a player visible?"]
    assert base.recipe["fix_hint"] == "base hint"
    # clone gained the animation question + an animation fix hint
    assert len(aug.recipe["checklist"]) == 2
    assert "SPECIFIC motion" in aug.recipe["checklist"][-1]
    # The fix hint must NOT prescribe regenerating frames (changed 2026-06-01):
    # img2img can't change a pose and fresh txt2img breaks character
    # consistency, so the hint is informational/cosmetic only. Guard against
    # any regen suggestion creeping back in.
    hint = aug.recipe["fix_hint"]
    assert "cosmetic" in hint.lower() or "does not block" in hint.lower()
    assert "from_image" not in hint
    assert "strength" not in hint.lower()
    # The hint may MENTION regeneration only to forbid it ("do NOT ...
    # regenerate"); it must never PRESCRIBE it. Check the actionable verbs.
    assert "re-emit" not in hint.lower()
    assert "do not try to regenerate" in hint.lower()


def test_augment_skips_when_no_animation_expected(tmp_path):
    a = _make_agent(tmp_path)  # no declared/dead frames, benign goal
    base = _recipe(["Is there a player visible?"])
    assert a._augment_recipe_for_animation(base) is base


def test_augment_skips_recipe_that_already_asks_about_animation(tmp_path):
    a = _make_agent(tmp_path)
    a._declared_anim_frames = True
    # fighter recipe already has the "same character mid-move" question
    fighter = _recipe([
        "Are TWO characters visible?",
        "Does each character keep the same character mid-move, not a different sprite?",
    ])
    assert a._augment_recipe_for_animation(fighter) is fighter


# ---- browser-dependent recipe executor (source-pinned) ----------------------

def test_asset_usage_accepts_webgl_textures():
    src = inspect.getsource(tools.LiveBrowser._run_opening_book_recipes)
    # three.js games never call drawImage — texture wiring must count as "used".
    assert "uses_texture" in src
    assert "CanvasTexture" in src and "TextureLoader" in src


def test_before_mid_after_uses_static_action_not_raf_stamp():
    src = inspect.getsource(tools.LiveBrowser._run_opening_book_recipes)
    # the before_mid_after / event_window branch now keys off the real
    # static_action signal, and no longer rubber-stamps on RAF-in-source.
    assert "static_action" in src
    assert "requestAnimationFrame" not in src
