"""Regression: vector wireframe games (Battlezone / Star Wars) must route to
2D canvas wireframe memory, not three.js FPS or generic fallbacks."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory  # noqa: E402
from prompt_library import effective_prompt, load_prompt_library  # noqa: E402
from prompts_v1 import first_build_instruction, plan_instruction  # noqa: E402

_REPO = Path(__file__).parent.parent
_SHIPPED = _REPO / "memory" / "prompt_library.jsonl"
_GENERIC_VISUAL = {"canvas-controllable-player", "generic-canvas-game-baseline"}


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    dst = tmp_path_factory.mktemp("memcopy") / "mem"
    shutil.copytree(_REPO / "memory", dst)
    return GameMemory(root=str(dst))


@pytest.fixture(scope="module")
def wireframe_prompts():
    lib = {p["name"]: p for p in load_prompt_library(_SHIPPED)}
    for name in ("battlezone", "star-wars", "asteroids"):
        assert name in lib, f"missing prompt library entry: {name}"
    return lib


def test_battlezone_and_star_wars_expect_blocks(wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        exp = wireframe_prompts[name]["expect"]
        assert exp["visual_recipe"] == "canvas-vector-wireframe"
        assert exp["outline"] == "outline-vector-wireframe"


def test_wireframe_prompts_route_to_wireframe_recipe(mem, wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        prompt = wireframe_prompts[name]["prompt"]
        rec, _ = mem.find_visual_playtest_for(goal=prompt, plan_text="", asset_names=[])
        assert rec is not None, f"{name}: no visual recipe"
        assert rec.id == "canvas-vector-wireframe", (
            f"{name}: expected canvas-vector-wireframe, got {rec.id}"
        )


def test_wireframe_prompts_avoid_generic_visual(mem, wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        prompt = wireframe_prompts[name]["prompt"]
        rec, _ = mem.find_visual_playtest_for(goal=prompt, plan_text="", asset_names=[])
        assert rec.id not in _GENERIC_VISUAL, f"{name} fell to generic visual recipe"


def test_wireframe_prompts_use_wireframe_outline(mem, wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        prompt = wireframe_prompts[name]["prompt"]
        hit = mem.retrieve_implementation_outline(prompt)
        assert hit is not None, f"{name}: no outline"
        assert hit.item.id == "outline-vector-wireframe", (
            f"{name}: expected outline-vector-wireframe, got {hit.item.id}"
        )


def test_wireframe_prompts_avoid_threejs_skeleton(mem, wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        prompt = wireframe_prompts[name]["prompt"]
        sk = mem.retrieve_skeleton(prompt)
        assert sk.name == "canvas_vector_wireframe_basic.html", (
            f"{name}: expected wireframe skeleton, got {sk.name}"
        )


def test_wireframe_plan_instruction_suppresses_art_and_3d(wireframe_prompts):
    for name in ("battlezone", "star-wars"):
        pi = plan_instruction(goal=wireframe_prompts[name]["prompt"])
        assert "ART INTENT DETECTED" not in pi, f"{name}: false art nudge"
        assert "3D INTENT DETECTED" not in pi, f"{name}: false 3d nudge"
        assert "WIREFRAME VECTOR INTENT DETECTED" in pi, f"{name}: missing wireframe nudge"


def test_asteroids_still_routes_top_down(mem, wireframe_prompts):
    ast = wireframe_prompts["asteroids"]["prompt"]
    rec, _ = mem.find_visual_playtest_for(goal=ast, plan_text="", asset_names=[])
    assert rec.id == "canvas-top-down-action"
    hit = mem.retrieve_implementation_outline(ast)
    assert hit.item.id == "outline-top-down-action"


def test_640png_appendix_keeps_wireframe_class(mem, wireframe_prompts):
    """BATTLE10: TUI /640png Battlezone must not retrieve Tetris/puzzle-grid."""
    for name in ("battlezone", "star-wars"):
        goal = effective_prompt(wireframe_prompts[name], jmr_png_mode=True)
        assert "drawImage columns" not in goal
        rec, diag = mem.find_visual_playtest_for(
            goal=goal, plan_text="", asset_names=[],
        )
        assert rec is not None, f"{name}: no visual recipe"
        assert rec.id == "canvas-vector-wireframe", (
            f"{name}: expected canvas-vector-wireframe, got {rec.id}; "
            f"top={diag.get('top_candidates')}"
        )
        hit = mem.retrieve_implementation_outline(goal)
        assert hit is not None, f"{name}: no outline"
        assert hit.item.id == "outline-vector-wireframe", (
            f"{name}: expected outline-vector-wireframe, got {hit.item.id}"
        )
        sk = mem.retrieve_skeleton(goal)
        assert sk.name == "canvas_vector_wireframe_basic.html", (
            f"{name}: expected wireframe skeleton, got {sk.name}"
        )


def test_wireframe_jmr_png_plan_does_not_demand_assets(wireframe_prompts):
    """Phase A must not require <assets> while the wireframe nudge forbids them."""
    goal = effective_prompt(wireframe_prompts["battlezone"], jmr_png_mode=True)
    pi = plan_instruction(goal=goal, jmr_png_mode=True, simulator_mode=True)
    assert "WIREFRAME VECTOR INTENT DETECTED" in pi
    assert "No <assets>/<sounds>/<videos>" in pi
    assert "jmr:spr" not in pi
    assert "STEM-N.png" not in pi
    assert "STEM-N" not in pi


def test_wireframe_jmr_png_first_build_does_not_demand_sheets(wireframe_prompts):
    goal = effective_prompt(wireframe_prompts["battlezone"], jmr_png_mode=True)
    msg = first_build_instruction(
        "<html><canvas></canvas></html>",
        None,
        jmr_png_mode=True,
        simulator_mode=True,
        goal=goal,
    )
    assert "jmr:spr" not in msg
    assert "pixel map" in msg.lower() or "data:image" in msg or "vector" in msg.lower()
