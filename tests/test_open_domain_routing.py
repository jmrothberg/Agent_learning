"""Open-domain routing: synthetic goals must hit genre-specific memory."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory, _RECIPE_TO_OUTLINE  # noqa: E402

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
    ("qbert isometric pyramid hop cubes recolor", "canvas-isometric-tile"),
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
