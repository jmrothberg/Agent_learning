"""Tests for the action-key parsing + animation-liveness (static-pose) gate.

Background — trace `short-and-done-first-the-promp_20260529` (SOTA Opus 4.7
fighting game): every iteration reported TEST OK / probes 8/9 / visual critic
Q1-Q7 all YES, yet the attacks were single static held poses, never animated.
Root causes proven from the trace:

  1. `_input_smoke_test` only pressed movement keys (Arrows/Space/WASD), never
     the punch/kick keys (KeyF/KeyG/...), so no action frame was ever captured
     (every visual_critic_start had image_count:2) and the critic rubber-stamped.
  2. "Animated" was never verified — a static pose held for the whole move
     passed everything.

Fix:
  - `_parse_action_keys` extracts the event.code tokens the model declared in
    <criteria>; `_input_smoke_test` presses those too.
  - During the per-key hold, 3 canvas hashes are sampled; if the responsive
    action key produces ~no in-hold motion while the canvas animates elsewhere,
    `static_action` is set and `load_and_test` appends a STATIC-ACTION
    soft_warning that flips report["ok"]=False (blocks <done/>).
  - When no action frame exists but actions are expected, the visual-critic
    prompt instructs NO/UNCLEAR instead of a YES rubber stamp.

CI-safe: pure-helper tests run live; the browser-dependent paths are pinned
with inspect.getsource (mirrors tests/test_action_frame_capture.py).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import tools  # noqa: E402


# ---- _parse_action_keys (pure) ---------------------------------------------

def test_parse_action_keys_extracts_codes_in_order():
    keys = tools._parse_action_keys("Press KeyF to punch, KeyG kick, ArrowUp jump")
    assert keys == ["KeyF", "KeyG", "ArrowUp"]


def test_parse_action_keys_dedupes_preserving_first_seen():
    assert tools._parse_action_keys("KeyF then KeyF and Space and Space") == ["KeyF", "Space"]


def test_parse_action_keys_ignores_prose_letters():
    # Loose "press F" must NOT match — only literal event.code tokens.
    assert tools._parse_action_keys("press f to punch, hit the g key") == []


def test_parse_action_keys_handles_digits_numpad_arrows():
    keys = tools._parse_action_keys("Numpad1 punch, Digit2 special, ArrowLeft, Enter")
    assert keys == ["Numpad1", "Digit2", "ArrowLeft", "Enter"]


def test_parse_action_keys_empty():
    assert tools._parse_action_keys("", None) == []


def test_static_pose_threshold_sane():
    assert isinstance(tools._STATIC_POSE_MAX_INHOLD, float)
    assert 0.0 < tools._STATIC_POSE_MAX_INHOLD < 0.05


# ---- _input_smoke_test wiring (source-pin) ---------------------------------

def _smoke_src() -> str:
    return inspect.getsource(tools.LiveBrowser._input_smoke_test)


def test_smoke_test_takes_criteria_and_parses_action_keys():
    sig = inspect.signature(tools.LiveBrowser._input_smoke_test)
    assert "criteria" in sig.parameters
    src = _smoke_src()
    assert "_parse_action_keys(criteria" in src
    assert "default_keys" in src
    assert "[:16]" in src  # capped


def test_smoke_test_samples_three_hold_frames_and_computes_motion():
    src = _smoke_src()
    assert "hold_hashes" in src
    assert "per_key_hold_motion" in src
    # in-hold motion = max pairwise distance of the sampled frames
    assert "_canvas_hash_distance(hold_hashes[0], hold_hashes[1])" in src


def test_smoke_test_flags_static_action_with_guards():
    src = _smoke_src()
    assert "_STATIC_POSE_MAX_INHOLD" in src
    # must require the game is animating elsewhere (not a paused/static game)
    assert "ambient_canvas_changed" in src
    # only the validated responsive action key
    assert "best_action_key" in src
    assert '"static_action": static_action' in src


# ---- load_and_test gating (source-pin) -------------------------------------

def test_load_and_test_appends_static_action_soft_warning():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "STATIC-ACTION:" in src
    assert 'report["soft_warnings"].append' in src
    # the append must precede the final ok recompute so ok flips to False
    i_append = src.find("STATIC-ACTION:")
    i_ok = src.rfind('report["ok"] = len(report["errors"]) == 0')
    assert i_append != -1 and i_ok != -1 and i_append < i_ok


# ---- critic anti-rubber-stamp (source-pin) ---------------------------------

def test_critic_prompt_refuses_yes_without_action_frame():
    src = inspect.getsource(agent.GameAgent._build_visual_playtest_prompt)
    # "actions expected" is now the broader `_animation_expected()` signal
    # (covers walk/kick/etc., not just attack keywords).
    assert "_animation_expected" in src
    assert "no active-input" in src.lower()
    assert "do NOT answer YES" in src


# ---- trace observability (source-pin) --------------------------------------

def test_iter_summary_logs_action_and_static_fields():
    src = inspect.getsource(agent.GameAgent.run)
    assert '"action_frame_captured"' in src
    assert '"static_action": _static_action' in src


def test_playbook_injected_trace_present():
    src = inspect.getsource(agent.GameAgent._retrieve_playbook_block)
    assert '"kind": "playbook_injected"' in src


# ---- stuck-player detection ("doesn't move") -------------------------------

def test_position_leaf_helper():
    assert tools._is_position_leaf("player.x")
    assert tools._is_position_leaf("pacman.tileY")
    assert tools._is_position_leaf("s.gridX")
    assert tools._is_position_leaf("hero.row")
    assert not tools._is_position_leaf("player.dir")
    assert not tools._is_position_leaf("player.nextDir")
    assert not tools._is_position_leaf("score")
    assert not tools._is_position_leaf("")


def test_movement_keys_are_arrows_and_wasd():
    for k in ("ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
              "KeyW", "KeyA", "KeyS", "KeyD"):
        assert k in tools._MOVEMENT_KEYS
    # action keys must NOT count as movement
    assert "KeyF" not in tools._MOVEMENT_KEYS
    assert "Space" not in tools._MOVEMENT_KEYS


def test_smoke_test_distinguishes_move_from_registered():
    src = _smoke_src()
    assert "movement_position_changed" in src
    assert "movement_registered_without_move" in src
    assert "has_position_state" in src
    assert '"input_registered_without_move"' in src
    # only movement keys, only when the game has position state
    assert "_MOVEMENT_KEYS" in src
    assert "_is_position_leaf" in src


def test_load_and_test_gates_on_stuck_player():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "PLAYER-STUCK" in src
    assert 'input_test.get("input_registered_without_move")' in src
    assert 'report["soft_warnings"].append' in src
