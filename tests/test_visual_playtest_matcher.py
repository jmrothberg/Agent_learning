"""Visual-playtest retrieval — matches recipes via goal + plan text +
asset names when the goal doesn't name a game.

The user's concern (2026-05-24): real goals don't always name the
game. "Build a game where you collect dots and avoid ghosts in
corridors" describes Pac-Man without saying "Pac-Man". The matcher
must lean on:

  1. Mechanic keywords in the goal (`corridors`, `collect`, `avoid`)
  2. The model's <plan> text (post-Phase A — describes mechanics)
  3. Asset names emitted by the model (`imp`, `pawn`, `ghost`)
  4. Strong-hook game names when present (`doom`, `chess`)

Tests are pure-function — they pass a synthetic recipe list to the
module-level `find_best_visual_playtest` and assert the right
recipe wins. No `GameMemory` instance, no disk I/O.

The seed library will land in a follow-up commit; this test file
pins the MATCHER behavior so the library can be evolved without
breaking retrieval.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory as memory_mod  # noqa: E402
from memory import (
    VisualPlaytestRecipe,
    find_best_visual_playtest,
    _visual_match_tokens,
)


def _recipe(rid: str, *, applies_keywords: list[str], strong_hooks: list[str] | None = None, min_matches: int = 2) -> VisualPlaytestRecipe:
    rec_dict = {
        "applies_keywords": applies_keywords,
        "applies_min_matches": min_matches,
        "checklist": ["Is X visible?"],
        "format": "yes_no_per_line",
    }
    if strong_hooks:
        rec_dict["strong_hooks"] = strong_hooks
    return VisualPlaytestRecipe(
        id=rid,
        kind="visual_playtest",
        content=f"Test recipe {rid}",
        tags=applies_keywords,
        source_tier="root",
        verified=True,
        helpful=0,
        harmful=0,
        recipe=rec_dict,
        trace_ids=[],
        pass_count=0,
        false_positive_count=0,
        last_verified_at="",
    )


def _grid_recipe() -> VisualPlaytestRecipe:
    return _recipe(
        "canvas-grid-navigation",
        applies_keywords=[
            "maze", "corridor", "corridors", "grid", "tile", "tiles",
            "dot", "dots", "pellet", "pellets", "navigate", "wall",
            "walls", "labyrinth", "dungeon", "ghost", "ghosts",
            "pacman", "sokoban",
        ],
        strong_hooks=["pacman", "pac-man", "sokoban"],
    )


def _fps_recipe() -> VisualPlaytestRecipe:
    return _recipe(
        "canvas-3d-first-person",
        applies_keywords=[
            "first-person", "firstperson", "fps", "shooter", "weapon",
            "crosshair", "perspective", "raycaster", "billboard",
            "doom", "wolfenstein", "quake",
        ],
        strong_hooks=["doom", "wolfenstein", "quake"],
    )


def _voxel_recipe() -> VisualPlaytestRecipe:
    return _recipe(
        "canvas-voxel-sandbox",
        applies_keywords=[
            "voxel", "voxels", "minecraft", "sandbox", "block", "blocks",
            "place", "break", "hotbar", "chunk", "terrain", "cube",
        ],
        strong_hooks=["minecraft"],
    )


def test_voxel_sandbox_goal_prefers_voxel_over_fps() -> None:
    """Paraphrase voxel goal must not inherit FPS-only auto_probes."""
    recipes = [_fps_recipe(), _voxel_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal=(
            "first-person voxel sandbox: explore cube blocks, break and "
            "place with a hotbar, chunked terrain, pointer-lock mouse look"
        ),
    )
    assert r is not None
    assert r.id == "canvas-voxel-sandbox"


def test_doom_goal_stays_fps_not_voxel() -> None:
    recipes = [_fps_recipe(), _voxel_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal="doom first-person shooter raycaster maze monsters crosshair weapon",
    )
    assert r is not None
    assert r.id == "canvas-3d-first-person"


def _board_recipe() -> VisualPlaytestRecipe:
    return _recipe(
        "canvas-board-game",
        applies_keywords=[
            "board", "grid", "piece", "pieces", "chess", "checkers",
            "tile", "8x8", "rook", "knight", "pawn", "king",
            "queen", "bishop", "reversi", "othello",
        ],
        strong_hooks=["chess", "checkers", "reversi", "othello"],
    )


def _fighter_recipe() -> VisualPlaytestRecipe:
    return _recipe(
        "canvas-two-actors-facing",
        applies_keywords=[
            "fighter", "fighting", "punch", "kick", "block",
            "fireball", "1v1", "player1", "player2", "mortal-kombat",
            "kombat", "street-fighter", "duel", "versus",
        ],
        strong_hooks=["mortal-kombat", "street-fighter"],
    )


# ----------------------------------------------------------------------
# Tokenizer behavior.
# ----------------------------------------------------------------------


def test_tokenizer_drops_stopwords_and_short_tokens() -> None:
    toks = _visual_match_tokens("Make a game about the dots in a maze")
    # Stopwords absent.
    for stop in ("the", "a", "make", "game", "about"):
        assert stop not in toks
    # Mechanics present.
    assert "dots" in toks
    assert "maze" in toks


def test_tokenizer_emits_2_and_3_grams() -> None:
    toks = _visual_match_tokens("first person shooter game")
    # Joined 2-gram.
    assert "firstperson" in toks
    # Hyphenated 2-gram.
    assert "first-person" in toks


def test_tokenizer_handles_empty_and_none() -> None:
    assert _visual_match_tokens("") == set()
    assert _visual_match_tokens(None) == set()  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Goal-only matching: user names the game.
# ----------------------------------------------------------------------


def test_strong_hook_wins_on_single_token() -> None:
    """A goal that just names the game ('doom') must resolve via
    strong-hook bypass even though it has only ONE relevant token."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, diag = find_best_visual_playtest(recipes, goal="doom clone")
    assert r is not None
    assert r.id == "canvas-3d-first-person"
    assert diag["top_candidates"][0]["via"] == "strong_hook"


def test_strong_hook_for_chess() -> None:
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, _ = find_best_visual_playtest(recipes, goal="build me a chess game")
    assert r is not None
    assert r.id == "canvas-board-game"


# ----------------------------------------------------------------------
# Goal-only matching: user describes the mechanic WITHOUT naming the game.
# This is the case the user flagged.
# ----------------------------------------------------------------------


def test_pacman_described_by_mechanic_only() -> None:
    """User goal: collect dots while avoiding ghosts in corridors.
    Doesn't say 'pacman' — must still resolve via mechanic overlap."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, diag = find_best_visual_playtest(
        recipes,
        goal="game where you collect dots while avoiding ghosts in corridors",
    )
    assert r is not None, f"expected grid-navigation, got None. diag={diag}"
    assert r.id == "canvas-grid-navigation"


def test_fighter_described_by_mechanic_only() -> None:
    """User goal: two-player versus, punch and kick. No 'kombat' /
    'street fighter' mentioned."""
    recipes = [_grid_recipe(), _fps_recipe(), _fighter_recipe(), _board_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal="two player versus game with punch and kick and fireball moves",
    )
    assert r is not None
    assert r.id == "canvas-two-actors-facing"


def test_fps_described_by_mechanic_only() -> None:
    """User goal: 3D first-person shooter feel without naming Doom."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal="first person shooter with a weapon and crosshair in a 3D perspective",
    )
    assert r is not None
    assert r.id == "canvas-3d-first-person"


# ----------------------------------------------------------------------
# Multi-signal matching: vague goal + rich plan text + asset names.
# ----------------------------------------------------------------------


def test_vague_goal_resolves_via_plan_text() -> None:
    """Goal is one ambiguous word; the model's plan text describes
    the mechanic clearly. Matcher must pick up plan text too."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal="arcade clone",
        plan_text=(
            "Mechanics: navigate a maze of corridors collecting dots; "
            "ghosts patrol the maze and chase the player. "
            "Controls: arrow keys move the player through walls "
            "and corridors."
        ),
    )
    assert r is not None
    assert r.id == "canvas-grid-navigation"


def test_vague_goal_resolves_via_asset_names() -> None:
    """Goal is generic; the model's declared asset names ('pawn',
    'rook', 'king') strongly signal a board game even without any
    board keyword in the goal."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, _ = find_best_visual_playtest(
        recipes,
        goal="two-player strategy game",
        asset_names=[
            "white_pawn", "black_pawn", "white_rook", "black_rook",
            "white_king", "black_king", "white_queen", "black_queen",
        ],
    )
    assert r is not None
    assert r.id == "canvas-board-game"


def test_combined_signals_outweigh_partial_match() -> None:
    """Goal has 1 keyword that matches recipe A weakly; plan text +
    assets have 4 keywords that match recipe B strongly. Recipe B
    must win."""
    recipes = [_grid_recipe(), _fighter_recipe(), _board_recipe()]
    # "grid" goal would naively match grid-navigation, but the plan
    # is clearly chess.
    r, _ = find_best_visual_playtest(
        recipes,
        goal="grid-based strategy game",
        plan_text=(
            "Two-player turn-based board with pieces; alternating "
            "moves; legal-move validation; king-in-check detection."
        ),
        asset_names=["pawn", "rook", "king", "queen", "knight", "bishop"],
    )
    assert r is not None
    assert r.id == "canvas-board-game"


# ----------------------------------------------------------------------
# Negative cases: no match.
# ----------------------------------------------------------------------


def test_no_match_returns_none() -> None:
    """A goal totally unrelated to any seeded recipe must return
    None (the caller falls back to the generic baseline)."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, diag = find_best_visual_playtest(
        recipes,
        goal="a meditation timer with breathing prompts",
    )
    assert r is None
    assert diag["top_candidates"] == []


def test_below_min_matches_returns_none() -> None:
    """One weak token-overlap isn't enough — min_matches default 2."""
    r, _ = find_best_visual_playtest(
        [_grid_recipe()],  # has 'tile' in its keywords
        goal="tile-painting app",  # only 'tile' overlaps
    )
    assert r is None


def test_empty_inputs_return_none() -> None:
    r, diag = find_best_visual_playtest([], goal="anything")
    assert r is None
    r, diag = find_best_visual_playtest([_grid_recipe()], goal="")
    assert r is None


# ----------------------------------------------------------------------
# Diagnostics surface enough info to debug retrieval misses.
# ----------------------------------------------------------------------


def test_diag_lists_top_candidates_with_scores_and_via() -> None:
    """When the matcher picks a winner, diag.top_candidates must
    carry id + score + via so the trace can record WHY this recipe
    won. Aids post-session analysis."""
    recipes = [_grid_recipe(), _fps_recipe(), _board_recipe()]
    r, diag = find_best_visual_playtest(
        recipes,
        goal="maze of corridors with dots",
    )
    assert r is not None
    top = diag["top_candidates"][0]
    assert "id" in top
    assert "score" in top
    assert "via" in top


def test_diag_match_tokens_sample_present() -> None:
    """diag.match_tokens_sample lets postmortem trace mining see
    what tokens the matcher actually had to work with."""
    _, diag = find_best_visual_playtest(
        [_grid_recipe()],
        goal="navigate a maze of corridors collecting dots",
    )
    assert "match_tokens_sample" in diag
    assert isinstance(diag["match_tokens_sample"], list)
    assert len(diag["match_tokens_sample"]) > 0


def test_sprite_scale_appendix_does_not_strong_hook_fixed_shooter() -> None:
    """run_17: SPRITE SCALE text said 'invaders' and strong-hooked
    canvas-fixed-shooter onto Roguelike/Pinball. Appendix must be stripped."""
    from pathlib import Path
    from memory import GameMemory, find_best_visual_playtest

    root = Path(__file__).resolve().parents[1] / "memory"
    mem = GameMemory(root=str(root))
    recipes = mem.load_visual_playtests()
    scale = (
        " SPRITE SCALE (required): Main enemies, invaders, gems, blocks, "
        "and props must be clearly visible."
    )
    rogue = (
        "Build a Roguelike Dungeon game. Procedurally generate rooms "
        "connected by corridors; fog-of-war reveals tiles near the player."
    )
    winner, diag = find_best_visual_playtest(recipes, goal=rogue + scale)
    assert winner is not None
    assert winner.id != "canvas-fixed-shooter", diag.get("top_candidates")
    assert winner.id in ("canvas-roguelike", "canvas-grid-navigation")


# ----------------------------------------------------------------------
# Module wiring: the constants + class exist and are loadable.
# ----------------------------------------------------------------------


def test_visual_playtest_recipe_is_dataclass_with_expected_shape() -> None:
    r = VisualPlaytestRecipe(
        id="x", kind="visual_playtest", content="", tags=[],
        source_tier="root", verified=True, helpful=0, harmful=0,
        recipe={}, trace_ids=[], pass_count=0,
        false_positive_count=0, last_verified_at="",
    )
    assert r.id == "x"
    assert r.kind == "visual_playtest"


def test_visual_playtests_filename_constant_exists() -> None:
    assert hasattr(memory_mod, "VISUAL_PLAYTESTS_FILENAME")
    assert memory_mod.VISUAL_PLAYTESTS_FILENAME == "visual_playtests.jsonl"


def test_game_memory_load_method_present() -> None:
    """GameMemory must expose load_visual_playtests + find method."""
    assert hasattr(memory_mod.GameMemory, "load_visual_playtests")
    assert hasattr(memory_mod.GameMemory, "find_visual_playtest_for")
