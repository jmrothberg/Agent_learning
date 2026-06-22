"""Regression: side-scroll beat-em-ups must not route to 1v1 fighter memory."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory  # noqa: E402
from prompt_library import load_prompt_library  # noqa: E402
from prompts_v1 import plan_instruction  # noqa: E402

_REPO = Path(__file__).parent.parent
_SHIPPED = _REPO / "memory" / "prompt_library.jsonl"


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    dst = tmp_path_factory.mktemp("memcopy") / "mem"
    shutil.copytree(_REPO / "memory", dst)
    return GameMemory(root=str(dst))


@pytest.fixture(scope="module")
def lib():
    return {p["name"]: p for p in load_prompt_library(_SHIPPED)}


def test_kung_fu_master_expect_blocks(lib):
    exp = lib["kung-fu-master"]["expect"]
    assert exp["visual_recipe"] == "canvas-side-scroll-beat-em-up"
    assert exp["outline"] == "outline-side-scroll-beat-em-up"


def test_kung_fu_routes_beat_em_up(mem, lib):
    prompt = lib["kung-fu-master"]["prompt"]
    rec, _ = mem.find_visual_playtest_for(goal=prompt, plan_text="", asset_names=[])
    assert rec is not None
    assert rec.id == "canvas-side-scroll-beat-em-up"
    hit = mem.retrieve_implementation_outline(prompt)
    assert hit is not None
    assert hit.item.id == "outline-side-scroll-beat-em-up"


def test_street_fighter_still_routes_two_actors(mem, lib):
    prompt = lib["street-fighter"]["prompt"]
    rec, _ = mem.find_visual_playtest_for(goal=prompt, plan_text="", asset_names=[])
    assert rec.id == "canvas-two-actors-facing"
    hit = mem.retrieve_implementation_outline(prompt)
    assert hit.item.id == "outline-two-actors-facing"


def test_kung_fu_plan_has_beat_em_up_nudge(lib):
    pi = plan_instruction(goal=lib["kung-fu-master"]["prompt"])
    assert "BEAT-EM-UP INTENT DETECTED" in pi


def test_synthetic_brawler_not_facing(mem):
    goal = "side scrolling brawler walk right and punch waves of enemies"
    rec, _ = mem.find_visual_playtest_for(goal=goal, plan_text="", asset_names=[])
    assert rec.id == "canvas-side-scroll-beat-em-up"
