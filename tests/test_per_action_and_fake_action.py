"""Per-action snapshots + fake-action ("lines acting like a kick") detection.

Two user-driven needs (2026-06-03):
  1. A fighting game has MANY named actions (J kick, K kick, L fireball, jump,
     duck). The harness pressed every key and captured a frame per key, but
     COLLAPSED them to one — so the trace had a single action image and the
     critic only judged one action. Now every per-action frame is saved
     (iter_NN_action_<KeyCode>.png) so each action's graphics are debuggable.
  2. The model fakes a "kick" by scribbling a limb with ctx.lineTo/fillRect over
     the idle sprite instead of swapping to the kick SPRITE. A real action draws
     a NEW drawImage source; a fake one only code-draws. ACTION_DRAWN_NOT_SPRITED
     flags the fake (gating soft_warning) so the model must use the sprite.

Source-pinned (load_and_test needs a live browser); mirrors
test_procedural_regression_detector.py.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402


# ---- stroke/line shim (so code-drawn limbs are visible, not just fillRect) --

def test_stroke_shim_present():
    src = inspect.getsource(tools_module)
    assert "window.__strokeEvents" in src
    # patches the line/path drawing methods
    for m in ("stroke", "lineTo"):
        assert f'"{m}"' in src
    assert "__strokeEvents.n++" in src


# ---- per-action frames: all captured, all saved -----------------------------

def test_input_smoke_returns_all_action_frames():
    src = inspect.getsource(tools_module._input_smoke_test.fget) \
        if isinstance(tools_module.LiveBrowser.__dict__.get("_input_smoke_test"), property) \
        else inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    # returns the full per-key frame map, not just the single peak frame
    assert "action_frames_png_bytes" in src
    assert "action_candidates.items()" in src


def test_load_and_test_saves_each_action_frame():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    # pops the per-key bytes and writes one file per action key
    assert 'input_test.pop("action_frames_png_bytes"' in src
    assert "_{keycode}.png" in src or "{stem}_{keycode}" in src
    # surfaces the saved paths in the report for trace debugging
    assert 'report["action_frames"]' in src


# ---- fake-action detection ---------------------------------------------------

def test_input_smoke_records_fake_action_signal():
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    # snapshots draw state before/after each hold: new sprite src + code-draw delta
    assert "_DRAW_STATE_JS" in src
    assert "new_sprite_src" in src
    assert "code_draw_delta" in src
    assert '"fake_actions"' in src


def test_harness_emits_action_drawn_not_sprited():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "ACTION_DRAWN_NOT_SPRITED" in src
    start = src.index("ACTION_DRAWN_NOT_SPRITED")
    excerpt = src[start:start + 900]
    # actionable: tells the model to use sprite() for the action, not code lines
    assert "sprite()" in excerpt
    assert "lineTo" in excerpt or "fillRect" in excerpt


def test_fake_action_gated_on_sprites_present():
    """Must NOT fire on a legitimately-procedural game (no sprite assets) —
    only when referenced_assets exist and a key code-drew without a new sprite."""
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    start = src.index("ACTION_DRAWN_NOT_SPRITED")
    block_start = src.rfind("if isinstance(_fake_actions", 0, start)
    assert block_start != -1
    guard = src[block_start:start]
    assert "referenced_assets" in guard
    # the per-key condition requires NO new sprite src AND positive code-draw
    cond = src[block_start:start + 400]
    assert "new_sprite_src" in cond and "code_draw_delta" in cond


def test_action_drawn_not_sprited_is_genre_free():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    start = src.index("ACTION_DRAWN_NOT_SPRITED")
    excerpt = src[start:start + 900].lower()
    for term in ("street fighter", "kombat", "mario", "pac-man", "doom"):
        assert term not in excerpt


# ---- code-drawn EFFECT bolted on top of a real sprite action ----------------

def test_per_key_signal_separates_stroke_delta():
    """The per-key signal must record stroke_delta (stroke/arc/lineTo) apart
    from fillRect, so 'code effect over a sprite' is distinguishable from a
    background fillRect."""
    src = inspect.getsource(tools_module.LiveBrowser._input_smoke_test)
    assert "stroke_delta" in src


def test_harness_emits_code_drawn_over_sprite():
    """The line+ball 'kick effect' over a real kick sprite (two_kickers test3)
    must be flagged: sprite WAS drawn (new_sprite_src) but stroke/arc code-draw
    also spiked. Distinct from ACTION_DRAWN_NOT_SPRITED (no sprite at all)."""
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "CODE_DRAWN_OVER_SPRITE" in src
    start = src.index("CODE_DRAWN_OVER_SPRITE")
    # condition: requires a new sprite src AND a stroke delta over threshold
    block_start = src.rfind("over_sprite = [", 0, start)
    assert block_start != -1
    cond = src[block_start:start]
    assert 'info.get("new_sprite_src")' in cond
    assert 'stroke_delta' in cond
    # actionable + genre-free
    excerpt = src[start:start + 700].lower()
    assert "sprite" in excerpt
    for term in ("street fighter", "kombat", "mario", "doom"):
        assert term not in excerpt


def test_two_detectors_are_mutually_exclusive_by_construction():
    """ACTION_DRAWN_NOT_SPRITED requires NOT new_sprite_src; CODE_DRAWN_OVER_SPRITE
    requires new_sprite_src. A single key can't trigger both."""
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    fake_i = src.index("ACTION_DRAWN_NOT_SPRITED")
    over_i = src.index("CODE_DRAWN_OVER_SPRITE")
    fake_cond = src[src.rfind("faked = [", 0, fake_i):fake_i]
    over_cond = src[src.rfind("over_sprite = [", 0, over_i):over_i]
    assert "not info.get(\"new_sprite_src\")" in fake_cond
    assert 'info.get("new_sprite_src")' in over_cond and 'not info.get("new_sprite_src")' not in over_cond
