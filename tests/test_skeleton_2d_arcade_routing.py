"""Regression: plain 2D arcade goals must NOT inherit a 3D / board / dungeon
scaffold (memory._SKELETON_GENERIC_TOKENS distinctiveness gate, 2026-05-31).

Mirrors tests/test_prompt_library_coverage.py — it drives the REAL curated
/games prompts (memory/prompt_library.jsonl) through retrieve_skeleton, so it
exercises exactly the text production sees. Before the gate, bundled specialized
scaffolds were exempt from any floor and a single incidental shared token routed
a 2D arcade goal to a wrong specialized scaffold (asteroids/galaga/centipede/
qbert -> canvas_3d on "space"/"vector"/"game"/"projection"; pong/missile-command
-> canvas_lit_dungeon on "light"; breakout -> canvas_crawler on "slide"; snake
-> canvas_rpg on "grid"; tetris -> canvas_cards on "grid"/"puzzle"; frogger/
monkey-island -> canvas_board_turn on "move"/"player"/"select").

A bundled scaffold may now only win if the goal shares a NON-generic token with
its sidecar, so these fall back to the safe generic v2 canvas. The GENUINE
specialized picks (sokoban/pacman -> grid, 1942 -> scrolling) and the modality
picks (chess/doom/minecraft) share a distinctive token and survive — see
test_skeleton_retrieval.py.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory  # noqa: E402
from prompt_library import load_prompt_library  # noqa: E402

_REPO = Path(__file__).parent.parent

# Specialized scaffolds a flat 2D arcade goal must never inherit.
_SPECIALIZED_3D_BOARD_DUNGEON = {
    "canvas_3d_basic.html",
    "canvas_voxel_minecraft_basic.html",
    "canvas_board_turn_basic.html",
    "canvas_lit_dungeon_basic.html",
    "canvas_crawler_basic.html",
    "canvas_cards_basic.html",
    "canvas_rpg_basic.html",
}

# Curated prompts that are flat 2D arcade games (NOT 3D/board/turn-based).
# pac-man -> grid and 1942 -> scrolling are genuine specialized picks and are
# intentionally excluded here (covered by test_skeleton_retrieval.py).
_ARCADE_2D_NAMES = {
    "centipede", "asteroids", "galaga", "frogger", "tetris", "breakout",
    "snake", "dig-dug", "qbert", "missile-command", "joust", "pong",
    "monkey-island",
}


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    # Copy shipped memory/ into a tmp root whose parent isn't '.', so GameMemory
    # treats base==live (matches test_prompt_library_coverage.py).
    dst = tmp_path_factory.mktemp("memcopy") / "mem"
    shutil.copytree(_REPO / "memory", dst)
    return GameMemory(root=str(dst))


@pytest.fixture(scope="module")
def arcade_prompts():
    lib = {p["name"]: p for p in load_prompt_library(_REPO / "memory" / "prompt_library.jsonl")}
    present = [(n, lib[n]["prompt"]) for n in sorted(_ARCADE_2D_NAMES) if n in lib]
    assert present, "no arcade prompts found in the library"
    return present


def test_known_arcade_prompts_present(arcade_prompts):
    # Guard against silent skips if the library is renamed/trimmed.
    assert len(arcade_prompts) >= 10, [n for n, _ in arcade_prompts]


def test_2d_arcade_avoids_3d_board_dungeon_skeleton(mem, arcade_prompts):
    misses = []
    for name, prompt in arcade_prompts:
        hit = mem.retrieve_skeleton(prompt)
        if hit.name in _SPECIALIZED_3D_BOARD_DUNGEON:
            misses.append(f"{name} -> {hit.name}")
    assert not misses, "2D arcade goals routed to specialized scaffolds: " + ", ".join(misses)


def test_genuine_specialized_picks_still_win(mem):
    """The distinctiveness gate must not over-correct: goals that share a
    distinctive token with a specialized sidecar still get that scaffold."""
    assert mem.retrieve_skeleton(
        "sokoban push boxes onto targets on a grid"
    ).name == "canvas_grid_basic.html"
    assert mem.retrieve_skeleton(
        "pac man with ghosts"
    ).name == "canvas_grid_basic.html"
