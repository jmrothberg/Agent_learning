"""Tests for the `expects_game_controls` keyword detector and the
synthetic `input_responsive` probe added to tools.py.

DK trace 20260514_104131 background:
  - Phase A criteria mentioned `ArrowRight moves the player`,
    `climb ladders with ArrowUp/ArrowDown`, etc.
  - The shipped game's keyboard input was silently broken; the
    harness's input_test reported FAIL — canvas pixels never changed.
  - The page had a clickable restart button, so the existing input-
    dead branch took the "treating as DOM-driven" path, leaving
    ok=True and emitting "STRONGLY prefer <done/>".
  - The model shipped with <done/>. User confirmed game was broken.

This file tests the fix at two layers:
  1. `expects_game_controls(*texts)` — tokenizing keyword detector
     that fires on Arrow / WASD / Space / move / climb / jump etc.
  2. The synthesis itself is end-to-end against `LiveBrowser`, which
     needs Chromium. The integration is covered indirectly via the
     keyword detector here; a future trace replay will verify the
     full chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import expects_game_controls  # noqa: E402


# ---------------------------------------------------------------------------
# Positive cases — the keyword set fires
# ---------------------------------------------------------------------------


def test_arrow_keys_fire():
    assert expects_game_controls(
        "Basic: Player moves left and right with ArrowLeft/ArrowRight."
    ) is True


def test_wasd_fires():
    assert expects_game_controls("Move with WASD; space to jump") is True


def test_climb_fires():
    """The DK trace's specific verb."""
    assert expects_game_controls(
        "Mario can climb ladders with ArrowUp/ArrowDown"
    ) is True


def test_jump_and_fire_fire():
    assert expects_game_controls("Press space to jump") is True
    assert expects_game_controls("Hold mouse to fire bullets") is True


def test_full_dk_criteria_fires():
    """The literal <criteria> body from DK 20260514_104131. Must fire
    or the fix is useless."""
    criteria = (
        "Basic: Player character is visible on the lowest platform at "
        "game start; pressing ArrowRight moves the player right along "
        "the platform; barrels spawn from Donkey Kong and roll across "
        "the screen.\n"
        "Edge: Barrels reverse direction at platform edges and roll "
        "down ladders to the next level; player can climb ladders with "
        "ArrowUp/ArrowDown and is blocked from climbing when no ladder "
        "is present.\n"
        "Stress: After 30+ seconds of gameplay with multiple barrels "
        "on screen, frame rate remains steady (~60fps); restarting the "
        "game (Space/Enter on game over) fully resets all barrels."
    )
    assert expects_game_controls(criteria) is True


def test_multiple_args_any_match_fires():
    """`expects_game_controls` accepts multiple texts (e.g. goal +
    criteria + plan); ANY match returns True."""
    assert expects_game_controls(
        "Boring calculator",
        "Press ArrowRight to ...",
    ) is True


# ---------------------------------------------------------------------------
# Negative cases — must not false-positive
# ---------------------------------------------------------------------------


def test_substring_inside_word_does_not_fire():
    """`key` inside `monkey`, `arrow` inside `arrowroot`, etc. —
    tokenized matching prevents substring false positives."""
    assert expects_game_controls("A monkey eats arrowroot") is False
    # `press` IS in the keyword set (covers "press space to ..."), so
    # "Press the buttons" returns True. That's intentional, not a
    # false positive — if a goal says "press", the input modality is
    # explicit even when buttons are involved.
    assert expects_game_controls("Press the buttons") is True


def test_pure_dom_app_does_not_fire():
    """A genuinely DOM-driven page: calculator, todo list, color picker
    — none of these mention game controls."""
    assert expects_game_controls(
        "Build a calculator with buttons for digits and operators. "
        "Show the result in a large display."
    ) is False
    assert expects_game_controls(
        "Todo list with checkboxes and a text input"
    ) is False


def test_empty_inputs_do_not_fire():
    assert expects_game_controls("") is False
    assert expects_game_controls(None or "") is False
    # Multiple empty inputs.
    assert expects_game_controls("", "", "") is False


def test_case_insensitive():
    assert expects_game_controls("CLIMB LADDERS WITH ARROW UP") is True
    assert expects_game_controls("WaSd") is True


# ---------------------------------------------------------------------------
# Pin the DK probes nudge logic to the keyword detector
# ---------------------------------------------------------------------------


def test_dk_criteria_combined_with_classifier_triggers_full_chain():
    """End-to-end: when the actual DK criteria are passed AND the
    actual probes are passed, BOTH (a) the nudge should classify the
    probes as 0% dynamic AND (b) the keyword detector should fire on
    the criteria. The combination of those two signals is the fix."""
    from agent import GameAgent

    dk_criteria = (
        "Basic: Player character is visible on the lowest platform at "
        "game start; pressing ArrowRight moves the player right along "
        "the platform; barrels spawn from Donkey Kong and roll across "
        "the screen."
    )
    dk_probes = [
        {"name": "canvas_exists",
         "expr": "!!document.querySelector('canvas')"},
        {"name": "player_state",
         "expr": "window.state && typeof window.state.player === "
                 "'object' && typeof window.state.player.x === "
                 "'number' && typeof window.state.player.y === "
                 "'number'"},
        {"name": "barrels_active",
         "expr": "window.state && Array.isArray(window.state.barrels) "
                 "&& window.state.barrels.length > 0"},
        {"name": "score_visible",
         "expr": "document.getElementById('score') && "
                 "document.getElementById('score').textContent.length "
                 "> 0"},
        {"name": "game_not_blank",
         "expr": "(function(){var c=document.querySelector('canvas');"
                 "if(!c||!c.width||!c.height)return false;try{return "
                 "c.toDataURL().length>200;}catch(e){return true;}})()"},
    ]
    # Probe classifier: all structural.
    pq = GameAgent._classify_probes_dynamic(dk_probes)
    assert pq["ratio"] == 0.0
    # Keyword detector: controls expected.
    assert expects_game_controls(dk_criteria) is True
