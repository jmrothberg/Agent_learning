"""Coverage test for the visual playtest recipe library.

The user's "top 25 game archetypes" list (2026-05-24) is the canonical
proof that the mechanism-keyed library covers the breadth of canvas
games the agent might be asked to build. Every game in the list MUST
resolve to a real mechanism recipe (not the generic fallback) when
matched against its short description.

When this test starts failing, that's the signal a new recipe is
needed in `memory/visual_playtests.jsonl` (or an existing recipe's
keywords need expanding) — NOT the signal to relax the test.

The list intentionally uses ONLY the short-description text the user
would naturally type as a goal; no game name is required. If a goal
match relies entirely on a strong-hook game name and not on
mechanism vocabulary, the recipe is too narrow.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory as memory_mod  # noqa: E402


REPO_MEMORY = Path(__file__).parent.parent / "memory"


# (game name, short-description goal text). The goal text comes from
# the user's table; it's intentionally close to what a user would
# actually type as a /new goal.
ARCHETYPES_25 = [
    ("Pac-Man",             "maze chase pellets ghosts"),
    # Donkey Kong now routes to canvas-vertical-platformer (added
    # 2026-05-24 after the DK trace showed canvas-side-scroll-platformer's
    # checklist missed the vertical-specific failure modes: player not
    # at bottom, ladders not snapped to floors, barrels not cascading).
    ("Donkey Kong",         "vertical platformer ladders barrels jumping rescue"),
    ("Galaga",              "space shooter enemy waves player ship"),
    ("Tetris",              "falling block puzzle rotation line clearing"),
    ("Arkanoid",            "breakout paddle ball bricks power-ups"),
    ("Gauntlet",            "top-down dungeon rooms keys monsters"),
    ("Double Dragon",       "side-scrolling beat em up punch kick enemies"),
    ("Contra",              "run gun platformer jumping shooting bosses"),
    ("Street Fighter",      "two fighters health bars special moves blocking"),
    ("Out Run",             "racing road perspective speed steering"),
    ("Punch-Out!!",         "boxing dodge punch enemy tells stamina"),
    ("Bubble Bobble",       "single-screen bubble action trap enemies"),
    ("Q*bert",              "isometric tile puzzle hop change colors"),
    ("Robotron",            "twin-stick arena shooter swarms"),
    ("Ghosts n Goblins",    "side-scrolling action platformer throwing weapons armor"),
    ("Battle City",         "top-down tank battle destructible walls enemy tanks"),
    ("Frogger",             "frog crossing lanes traffic logs timing"),
    ("Rogue",               "roguelike procedural dungeon turn-based loot"),
    ("SimCity",             "city builder zones roads budget growth"),
    ("Prince of Persia",    "cinematic platformer sword combat traps precise jumps"),
    ("Maniac Mansion",      "point-and-click adventure rooms inventory characters"),
    ("Ultima IV",           "open-world RPG towns overworld NPC dialogue quests"),
    ("Elite",               "space trading combat ship pirates upgrades"),
    ("Castle Wolfenstein",  "stealth maze guards rooms keys disguise"),
    ("Dungeon Master",      "first-person grid-based 3D dungeon party monsters items"),
]


def test_every_archetype_resolves_to_a_recipe() -> None:
    """All 25 archetypes must match a real mechanism recipe.

    If this fails for a game: either add a new recipe to
    memory/visual_playtests.jsonl, or expand an existing recipe's
    applies_keywords to include the mechanism vocabulary from the
    archetype's short description.
    """
    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    misses: list[tuple[str, str]] = []
    for game, goal in ARCHETYPES_25:
        r, _ = mem.find_visual_playtest_for(goal=goal)
        if r is None or r.id == "generic-canvas-game-baseline":
            misses.append((game, goal))
    assert not misses, (
        "Some archetypes hit the generic fallback — add or broaden "
        f"a mechanism recipe for: {misses}"
    )


def test_archetypes_have_high_quality_matches() -> None:
    """Soft check: most matches should score >= 2 (clear mechanism
    overlap) or use strong_hook (game name). A score of 1 means the
    recipe matched on a single token and is probably a coincidence.
    """
    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    weak: list[tuple[str, str, float]] = []
    for game, goal in ARCHETYPES_25:
        r, diag = mem.find_visual_playtest_for(goal=goal)
        if r is None:
            continue  # covered by the test above
        top = diag["top_candidates"][0]
        score = top["score"]
        via = top["via"]
        # Strong-hook (1001+) is decisive even at "score 1001"; overlap
        # matches need >= 2 to be confident.
        if via == "overlap" and score < 2.0:
            weak.append((game, r.id, score))
    assert not weak, (
        "Some archetypes matched a recipe on only 1 keyword overlap — "
        "the match is probably coincidental. Add more mechanism "
        f"vocabulary to the recipe: {weak}"
    )


def test_known_assignments() -> None:
    """Pin the exact expected recipe ID for the archetypes whose
    mechanism is unambiguous. If a refactor of the library accidentally
    re-routes one of these, this test catches it immediately.
    """
    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    expected = {
        # short_goal_text → recipe id
        "maze chase pellets ghosts": "canvas-grid-navigation",
        "falling block puzzle rotation line clearing": "canvas-puzzle-grid",
        "breakout paddle ball bricks power-ups": "canvas-paddle-ball",
        "racing road perspective speed steering": "canvas-racing-perspective",
        "two fighters health bars special moves blocking": "canvas-two-actors-facing",
        "frog crossing lanes traffic logs timing": "canvas-lane-crossing",
        "city builder zones roads budget growth": "canvas-city-builder",
        "point-and-click adventure rooms inventory characters": "canvas-point-and-click",
        "open-world RPG towns overworld NPC dialogue quests": "canvas-overworld-rpg",
        "space trading combat ship pirates upgrades": "canvas-space-trading",
        "isometric tile puzzle hop change colors": "canvas-isometric-tile",
        "first-person grid-based 3D dungeon party monsters items": "canvas-3d-first-person",
        # Vertical platformers (donkey-kong 2026-05-24 trace). Distinct
        # from canvas-side-scroll-platformer: ladders + cascading
        # hazards + bottom-to-top progression are the signal.
        "vertical platformer ladders barrels jumping rescue": "canvas-vertical-platformer",
        "burger maker climbing ladders dodging cooks": "canvas-vertical-platformer",
        # Horizontal platformers stay with side-scroll-platformer.
        "side scrolling platformer hero jumping on platforms": "canvas-side-scroll-platformer",
    }
    for goal, expected_id in expected.items():
        r, diag = mem.find_visual_playtest_for(goal=goal)
        assert r is not None, f"{goal!r} hit fallback"
        assert r.id == expected_id, (
            f"{goal!r} matched {r.id!r}, expected {expected_id!r}; "
            f"top: {diag['top_candidates']}"
        )


def test_newly_covered_mechanism_families_route() -> None:
    """The mechanism families that previously hit the generic fallback (or
    only a broad neighbour) must now resolve to their dedicated recipe, so
    the VLM critic gets a precise checklist instead of the open-ended
    fallback. match-3 intentionally stays on canvas-puzzle-grid (already a
    strong fit; a competing recipe would risk strong-hook collisions)."""
    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    expected = {
        "tower defense place turrets to stop waves of creeps following a path": "canvas-tower-defense",
        "rhythm music game hit notes in time as they reach the target line": "canvas-rhythm",
        "idle incremental clicker game click to earn buy upgrades auto income": "canvas-idle-clicker",
        "bullet hell dodge dense enemy bullet patterns shmup boss": "canvas-bullet-hell",
        "typing game type the falling words before they reach the bottom": "canvas-word-typing",
        "pinball flippers launch ball bumpers ramps physics gravity score": "canvas-pinball",
        "stealth game sneak past guards with vision cones avoid detection": "canvas-stealth",
        "roguelike procedural dungeon turn-based permadeath loot levels": "canvas-roguelike",
        "stacking physics tower stack balance dont topple": "canvas-stacking-physics",
        "match three puzzle swap adjacent gems to clear lines and cascade combos": "canvas-puzzle-grid",
    }
    for goal, expected_id in expected.items():
        r, diag = mem.find_visual_playtest_for(goal=goal)
        assert r is not None, f"{goal!r} hit fallback"
        assert r.id == expected_id, (
            f"{goal!r} matched {r.id!r}, expected {expected_id!r}; "
            f"top: {diag['top_candidates']}"
        )


def test_library_has_at_least_one_recipe_per_mechanism_family() -> None:
    """Pin that the library covers the major mechanism families
    surfaced by the 25-archetype audit, so future trimming doesn't
    accidentally remove one."""
    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    all_recipes = mem.load_visual_playtests()
    ids = {r.id for r in all_recipes}
    required = {
        "canvas-controllable-player",
        "canvas-grid-navigation",
        "canvas-two-actors-facing",
        "canvas-side-scroll-platformer",
        "canvas-vertical-platformer",  # NEW 2026-05-24 (donkey-kong trace)
        "canvas-3d-first-person",
        "canvas-top-down-action",
        "canvas-board-game",
        "canvas-puzzle-grid",
        "canvas-racing-perspective",
        "canvas-vfx-fluid",
        "canvas-paddle-ball",
        "canvas-lane-crossing",
        "canvas-point-and-click",
        "canvas-isometric-tile",
        "canvas-overworld-rpg",
        "canvas-city-builder",
        "canvas-space-trading",
        # 2026-06-13 deepening: dedicated recipes for families that
        # previously hit the generic fallback or only a broad neighbour.
        "canvas-tower-defense",
        "canvas-rhythm",
        "canvas-idle-clicker",
        "canvas-bullet-hell",
        "canvas-word-typing",
        "canvas-pinball",
        "canvas-stealth",
        "canvas-roguelike",
        "canvas-stacking-physics",
        "generic-canvas-game-baseline",
    }
    missing = required - ids
    assert not missing, (
        f"Recipe library is missing mechanism families: {missing}"
    )
