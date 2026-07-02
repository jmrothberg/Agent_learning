"""Open-domain routing: synthetic goals must hit genre-specific memory."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory, _RECIPE_TO_OUTLINE  # noqa: E402
from agent import GameAgent  # noqa: E402

_REPO = Path(__file__).parent.parent
_GENERIC = {"canvas-controllable-player", "generic-canvas-game-baseline"}
_SPECIALIZED_3D = {
    "canvas_3d_basic.html",
    "canvas_voxel_minecraft_basic.html",
}

# (goal snippet, expected visual recipe id)
_SYNTHETIC = [
    ("side scrolling brawler punch waves of enemies from both sides", "canvas-side-scroll-beat-em-up"),
    ("two fighters facing each other on a dojo stage with health bars 1v1", "canvas-two-actors-facing"),
    ("wireframe vector tank first person glowing lines on black", "canvas-vector-wireframe"),
    ("asteroids style ship thrust rotate irregular polygon rocks wrap screen", "canvas-top-down-action"),
    ("first person doom raycaster three.js shoot monsters", "canvas-3d-first-person"),
    ("chess board hotseat click legal moves check checkmate", "canvas-board-game"),
    ("pac-man maze dots ghosts corridors", "canvas-grid-navigation"),
    ("tetris falling tetromino lines clear rotate drop", "canvas-puzzle-grid"),
    ("bejeweled swap adjacent gems match three cascade", "canvas-match3"),
    ("frogger lanes traffic river hop crossing", "canvas-lane-crossing"),
    ("breakout paddle bricks ball bounce", "canvas-paddle-ball"),
    ("super mario side scroll platform jump gravity", "canvas-side-scroll-platformer"),
    ("donkey kong ladders girders climb barrels vertical", "canvas-vertical-platformer"),
    ("outrun mode7 perspective road race car", "canvas-racing-perspective"),
    ("zelda overworld tile map npc dialogue hearts", "canvas-overworld-rpg"),
    ("monkey island point and click inventory hotspot dialog", "canvas-point-and-click"),
    ("qbert isometric pyramid hop cubes recolor", "canvas-pyramid-hopper"),
    ("dragons lair qte timed reaction cutscene prompt window", "canvas-cutscene-qte"),
    ("place towers along a path to stop creeps waves", "canvas-tower-defense"),
    ("rhythm game notes scroll hit line lanes timing combo", "canvas-rhythm"),
    ("idle clicker cookie generators passive income", "canvas-idle-clicker"),
    ("bullet hell danmaku boss spiral pattern dodge", "canvas-bullet-hell"),
    ("type falling words before they hit the bottom keyboard", "canvas-word-typing"),
    ("pinball flippers bumpers ball drain plunger", "canvas-pinball"),
    ("stealth guards vision cone patrol sneak unseen", "canvas-stealth"),
    ("roguelike procedural dungeon turn based fog of war", "canvas-roguelike"),
    ("stack blocks crane drop tower balance topple", "canvas-stacking-physics"),
    ("particle fireworks explosion sparks gravity fade", "canvas-vfx-fluid"),
    ("simcity zones residential commercial roads economy", "canvas-city-builder"),
    ("elite space trading market cargo fuel jump systems", "canvas-space-trading"),
    ("solitaire klondike tableau foundation drag cards", "canvas-card-tabletop"),
    ("slingshot angry birds gravity arc knock blocks", "canvas-physics-projectile"),
    ("torch lit dungeon fog of darkness radial glow only visible near player", "canvas-lit-dungeon"),
    ("mobile touch virtual d-pad joystick phone controls", "canvas-mobile-touch"),
    ("single character animation showcase one fighter no opponent", "canvas-single-fighter"),
    ("snake grid grow food wraparound walls", "canvas-grid-navigation"),
    ("pong two paddles bounce ball score", "canvas-paddle-ball"),
    ("missile command crosshair cities intercept incoming", "canvas-top-down-action"),
    ("1942 vertical scrolling shoot em up plane formation", "canvas-top-down-action"),
    ("space invaders grid aliens march shoot bunker", "canvas-top-down-action"),
]


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    dst = tmp_path_factory.mktemp("memcopy") / "mem"
    shutil.copytree(_REPO / "memory", dst)
    return GameMemory(root=str(dst))


@pytest.mark.parametrize("goal,visual_id", _SYNTHETIC)
def test_synthetic_visual_routing(mem, goal, visual_id):
    rec, _ = mem.find_visual_playtest_for(goal=goal, plan_text="", asset_names=[])
    assert rec is not None, f"no recipe for: {goal!r}"
    assert rec.id not in _GENERIC, f"generic fallthrough for: {goal!r}"
    assert rec.id == visual_id, f"goal={goal!r} expected {visual_id}, got {rec.id}"


@pytest.mark.parametrize("goal,visual_id", _SYNTHETIC)
def test_synthetic_outline_matches_visual_map(mem, goal, visual_id):
    rec, _ = mem.find_visual_playtest_for(goal=goal, plan_text="", asset_names=[])
    want_outline = _RECIPE_TO_OUTLINE.get(rec.id)
    assert want_outline, f"no outline map for visual {rec.id}"
    hit = mem.retrieve_implementation_outline(goal)
    assert hit is not None
    assert hit.item.id == want_outline, (
        f"goal={goal!r} visual={rec.id} expected outline {want_outline}, "
        f"got {hit.item.id}"
    )


def test_wireframe_not_threejs_skeleton(mem):
    goal = "battlezone wireframe vector tank radar first person"
    sk = mem.retrieve_skeleton(goal)
    assert sk.name == "canvas_vector_wireframe_basic.html"


def test_2d_shooter_not_3d_skeleton(mem):
    goal = "galaga formation shooter space aliens wave"
    sk = mem.retrieve_skeleton(goal)
    assert sk.name not in _SPECIALIZED_3D


# ---------------------------------------------------------------------------
# GENERALITY — "works for ANY game", not just the genres above.
#
# The tests above pin genre-SPECIFIC routing. These prove the complementary
# guarantee: a goal that matches NO known genre still inherits a genre-free
# engine contract (state-on-window / input / dt / restart) plus the working
# game-loop + input snippets, so the model never plans a build from scratch.
# All model-free / browser-free (TEST.md: pure-function only).
# ---------------------------------------------------------------------------

# Deliberately novel goals that name no known genre or title.
_NOVEL_GOALS = [
    "herd glowing fireflies into jars before dawn",
    "tune three radio dials until the static becomes music",
    "guide a paper boat down a rain gutter collecting bottle caps",
    "balance spinning plates on poles as the wind rises",
    "sort falling autumn leaves into matching colored piles",
]

# The engine skeleton every open-domain build pins (agent.py ~15084).
_ENGINE_COMPONENTS = ["fixed-timestep-game-loop", "input-manager-buffered"]


@pytest.fixture
def agent(tmp_path):
    """GameAgent wired to a COPY of the real curated memory — no model, no
    browser, no writes back into the repo memory/."""
    dst = tmp_path / "mem"
    shutil.copytree(_REPO / "memory", dst)
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=4,
        memory_root=str(dst),
    )


def test_universal_outline_exposes_state_input_restart(agent):
    """The genre-free baseline outline carries the contract every game needs:
    state exposed on window, input, and restart."""
    hit = agent._memory._outline_item_by_id("outline-controllable-canvas-game")
    assert hit is not None, "universal fallback outline missing from memory"
    blob = (hit.item.content + " " + json.dumps(hit.item.recipe or {})).lower()
    for token in ("state", "window", "input", "restart"):
        assert token in blob, f"universal outline missing '{token}' contract"


def test_novel_goal_still_gets_a_baseline_outline(agent, monkeypatch):
    """When retrieval matches NO outline (genuinely novel goal), the opening
    book must STILL inject the universal controllable-canvas-game outline so
    the model never plans from a blank state/input/restart contract."""
    # Force the "nothing matched" branch deterministically.
    monkeypatch.setattr(
        agent._memory, "retrieve_implementation_outline",
        lambda *a, **k: None,
    )
    block, hits = agent._retrieve_opening_book_block("zzzqqq wholly novel thing")
    assert block.strip(), "no opening-book block produced for novel goal"
    outline_ids = [h["id"] for h in hits if h.get("kind") == "outline"]
    assert "outline-controllable-canvas-game" in outline_ids
    assert "outline-controllable-canvas-game" in block


@pytest.mark.parametrize("goal", _NOVEL_GOALS)
def test_open_domain_pins_engine_components(agent, goal):
    """Open-domain builds pin the engine snippets so a weak model copies a
    working loop+input instead of inventing a broken one — regardless of the
    goal's words (Jaccard alone would miss these novel goals)."""
    block = agent._retrieve_components_block(
        goal, stage="plan", k=4, ensure_ids=_ENGINE_COMPONENTS,
    )
    assert block.strip(), f"no components block for novel goal: {goal!r}"
    for cid in _ENGINE_COMPONENTS:
        assert cid in block, f"engine component {cid} not pinned for {goal!r}"


def test_components_by_ids_returns_runnable_engine_snippets(agent):
    """The pinned ids resolve to real snippets WITH code bodies, not names."""
    hits = agent._memory.components_by_ids(_ENGINE_COMPONENTS)
    got = {h.item.id for h in hits}
    assert set(_ENGINE_COMPONENTS) <= got, got
    for h in hits:
        code = str((h.item.recipe or {}).get("code") or "").strip()
        assert code, f"{h.item.id} has no code body"


@pytest.mark.parametrize("goal,_visual_id", _SYNTHETIC)
def test_every_genre_gets_an_opening_contract(agent, goal, _visual_id):
    """Sweep EVERY known genre: each must yield a non-empty opening book with
    an outline. Generality is not limited to a hand-picked subset of games."""
    block, hits = agent._retrieve_opening_book_block(goal)
    assert block.strip(), f"empty opening book for genre goal: {goal!r}"
    assert any(h.get("kind") == "outline" for h in hits), f"no outline for {goal!r}"
