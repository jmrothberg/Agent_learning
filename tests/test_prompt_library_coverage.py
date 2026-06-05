"""Layer-0 memory coverage matrix for the curated /games prompt library.

Model-free and deterministic: for every prompt in memory/prompt_library.jsonl,
call each memory subsystem's retrieval function directly and assert it engages
with a genre-appropriate result (the `expect` block co-located in each prompt).
This makes permanent + regression-guarded the cross-check that verified the 26
prompts route correctly (2026-05-31).

Hermetic: memory is constructed with an EMPTY live-store root so per-machine
games/game-memory/ cruft cannot perturb routing.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory, Playbook  # noqa: E402
from prompt_library import load_prompt_library  # noqa: E402

_REPO = Path(__file__).parent.parent
_SHIPPED = _REPO / "memory" / "prompt_library.jsonl"
_GENERIC_VISUAL = {"canvas-controllable-player", "generic-canvas-game-baseline"}


@pytest.fixture(scope="module")
def games():
    rows = load_prompt_library(_SHIPPED)
    assert rows, "prompt library failed to load"
    return rows


@pytest.fixture(scope="module")
def mem(tmp_path_factory):
    # Copy shipped memory/ into a tmp root whose parent isn't '.', so GameMemory
    # treats base==live (no per-machine games/game-memory live-store cruft).
    dst = tmp_path_factory.mktemp("memcopy") / "mem"
    shutil.copytree(_REPO / "memory", dst)
    return GameMemory(root=str(dst))


@pytest.fixture(scope="module")
def playbook(tmp_path_factory):
    live = tmp_path_factory.mktemp("empty_live_pb")
    return Playbook(base_root=str(_REPO / "memory"), live_root=str(live))


def test_every_prompt_has_expect_block(games):
    for g in games:
        assert "expect" in g, f"prompt #{g['n']} {g['name']} missing expect block"
        for key in ("visual_recipe", "outline"):
            assert g["expect"].get(key), f"{g['name']} expect missing {key}"


def test_visual_recipe_routes_to_expected_genre(games, mem):
    misses = []
    for g in games:
        rec, _ = mem.find_visual_playtest_for(goal=g["prompt"], plan_text="", asset_names=[])
        got = rec.id if rec else "(none)"
        if got != g["expect"]["visual_recipe"]:
            misses.append(f"{g['name']}: expected {g['expect']['visual_recipe']}, got {got}")
    assert not misses, "visual-recipe routing changed:\n  " + "\n  ".join(misses)


def test_no_prompt_falls_to_generic_visual_recipe(games, mem):
    generic = []
    for g in games:
        rec, _ = mem.find_visual_playtest_for(goal=g["prompt"], plan_text="", asset_names=[])
        if rec is None or rec.id in _GENERIC_VISUAL:
            generic.append(g["name"])
    assert not generic, f"these prompts fall to the generic visual recipe: {generic}"


def test_implementation_outline_routes_to_expected(games, mem):
    misses = []
    for g in games:
        hit = mem.retrieve_implementation_outline(g["prompt"])
        got = hit.item.id if hit else "(none)"
        if got != g["expect"]["outline"]:
            misses.append(f"{g['name']}: expected {g['expect']['outline']}, got {got}")
    assert not misses, "outline routing changed:\n  " + "\n  ".join(misses)


def test_skeleton_retrieval_engages_for_every_prompt(games, mem):
    for g in games:
        sk = mem.retrieve_skeleton(g["prompt"])
        assert sk is not None and sk.name, f"{g['name']}: no skeleton returned"
        assert sk.html and "<" in sk.html, f"{g['name']}: skeleton has no html"


def test_playbook_engages_at_plan_stage(games, playbook):
    for g in games:
        hits = playbook.retrieve(g["prompt"], stage="plan", k=8)
        assert hits, f"{g['name']}: playbook returned no bullets at plan stage"


def test_opening_book_recipes_are_callable(games, mem):
    # playtests / asset-audits / animation-audits must at least retrieve cleanly.
    for g in games:
        assert isinstance(mem.retrieve_playtests(g["prompt"]), list)
        assert isinstance(mem.retrieve_asset_audits(g["prompt"]), list)
        assert isinstance(mem.retrieve_animation_audits(g["prompt"]), list)


def test_mistakes_mechanism_works_with_synthetic_signature(mem):
    # mistakes are keyed by an error signature (only produced by a failed build),
    # so verify the retrieval mechanism with a synthetic signature.
    hits = mem.retrieve_mistakes("ReferenceError: foo is not defined")
    assert isinstance(hits, list)
