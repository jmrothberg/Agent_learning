"""Tests for the action-frame capture + derived-sprite sanity warning.

Background — the failing session (make-the-most-graphic-animated, 2026-05-28):
the agent spent 5 iterations failing to make a visible punch/kick while every
iteration reported "probes 7/7 / TEST OK". Two harness blind spots:

  1. The visual critic only ever saw the startup frame and the post-smoke-test
     RESTING frame, so a brief ~0.5s attack was never on screen — the critic
     returned "UNCLEAR — no active attack visible" forever.
  2. A `from_image`-derived "punch" sprite that came out identical to idle was
     never flagged; the model (which can't see its art) debugged logic instead.

Fix:
  - `_canvas_hash_distance` (pure) — magnitude of canvas change between two
    `_CANVAS_HASH_JS` strings.
  - `_input_smoke_test` captures one screenshot at the moment a held key
    produces peak change (the "action frame"); discards it unless the key was
    input-attributable. `load_and_test` writes it to `report["screenshot_action"]`.
  - `run_visual_critic` accepts a 3rd `action_png` image and routes
    action/animation questions to it.
  - `generate_assets` records `parent_delta` for `from_image` specs; the agent
    warns the model when a derived frame is near-identical to its parent.

These are CI-safe: the pure helper runs directly; the rest are source-pin
assertions (mirroring tests/test_input_smoke_state_global.py) since the browser
and the diffusion pipeline can't run in CI.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import assets  # noqa: E402
import tools  # noqa: E402


# ---- pure helper: _canvas_hash_distance ------------------------------------

def test_canvas_hash_distance_fraction():
    assert tools._canvas_hash_distance("a,a,a,a", "a,b,a,a") == 0.25


def test_canvas_hash_distance_equal_is_zero():
    assert tools._canvas_hash_distance("a,b,c,d", "a,b,c,d") == 0.0


def test_canvas_hash_distance_all_different():
    assert tools._canvas_hash_distance("a,a", "b,b") == 1.0


def test_canvas_hash_distance_none_on_bad_input():
    assert tools._canvas_hash_distance(None, "a,b") is None
    assert tools._canvas_hash_distance("a,b", None) is None
    assert tools._canvas_hash_distance("", "") is None
    # mismatched cell counts → not comparable
    assert tools._canvas_hash_distance("a,b,c", "a,b") is None


# ---- _input_smoke_test captures + gates the action frame -------------------

def _smoke_src() -> str:
    return inspect.getsource(tools.LiveBrowser._input_smoke_test)


def test_smoke_test_captures_action_frame_while_held():
    src = _smoke_src()
    # An ambient floor is computed so baseline drift never wins.
    assert "ambient_floor" in src
    assert "_canvas_hash_distance(ambient_a, ambient_b)" in src
    # A per-key candidate screenshot is taken inside the hold.
    assert "action_candidates[k]" in src
    assert "self._page.screenshot(" in src
    # The capture must happen BEFORE the key is released (still held).
    cap = src.find("action_candidates[k] = (held_dist, _png)")
    up = src.find("keyboard.up(k)", cap)
    assert cap > -1 and up > cap, "action frame must be captured while key held"


def test_smoke_test_selects_responsive_transient_action_frame():
    src = _smoke_src()
    # Winner must be input-attributable (responsive) AND transient (reverts
    # after release) — so a screen-wiping restart key never wins.
    assert "responsive_evidence" in src
    assert "_ACTION_TRANSIENT_MAX_RATIO" in src
    assert "per_key_release_dist" in src


def test_smoke_test_returns_action_fields():
    src = _smoke_src()
    assert '"action_frame_png_bytes": action_frame_png' in src
    assert '"action_key": best_action_key' in src
    # Existing keys must remain intact (other tests depend on them).
    assert '"summary": summary' in src
    assert '"responsive_evidence": responsive_evidence' in src


# ---- load_and_test surfaces the action frame -------------------------------

def test_load_and_test_writes_action_screenshot():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "screenshot_action_path" in src
    assert 'report["screenshot_action"]' in src
    # Raw bytes are popped out of input_test before the report is built.
    assert 'input_test.pop("action_frame_png_bytes"' in src


# ---- visual critic consumes the 3rd image ----------------------------------

def test_run_visual_critic_accepts_action_png():
    sig = inspect.signature(agent.GameAgent.run_visual_critic)
    assert "action_png" in sig.parameters


def test_run_visual_critic_assembles_three_images_and_prompt():
    src = inspect.getsource(agent.GameAgent.run_visual_critic)
    # action frame appended to the image list
    assert "images.append(action_png)" in src
    # prompt mentions Image 3 / the action frame
    assert "Image 3" in src
    # stops inviting UNCLEAR on action visibility
    assert "unclear" in src.lower()


def test_build_visual_playtest_prompt_routes_to_action_frame():
    src = inspect.getsource(agent.GameAgent._build_visual_playtest_prompt)
    assert "action_png" in src
    assert "Image 3" in src


def test_spawn_visual_critic_threads_action_bytes():
    sig = inspect.signature(agent.GameAgent._spawn_visual_critic)
    assert "action_bytes" in sig.parameters


# ---- derived-frame sanity in assets.py -------------------------------------

def test_assets_defines_derived_frame_threshold():
    assert isinstance(assets._DERIVED_FRAME_MIN_DELTA, float)
    assert 0.0 < assets._DERIVED_FRAME_MIN_DELTA < 1.0


def test_generate_assets_records_parent_delta():
    src = inspect.getsource(assets.generate_assets)
    assert "_derived_frame_delta(" in src
    assert 'stat["parent_delta"]' in src
    assert "from_image" in src


def test_agent_warns_on_near_identical_derived_frame():
    # The warning lives in the asset-generation helper coroutine.
    src = inspect.getsource(agent.GameAgent._maybe_generate_assets_and_sounds)
    assert "_DERIVED_FRAME_MIN_DELTA" in src
    assert "ASSET SANITY WARNING" in src
    assert "self._pending_feedback.append" in src
