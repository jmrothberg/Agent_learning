"""Grid-maze-chase modality: open-domain retrieval + probe wiring.

Pac-Man is one example theme — Ms Pac-Man and reskin goals (different
pursuer nouns) must hit the same memory + pass the same generic probes.
No title-locked wiring (no requirement for 'ghost' in HTML or state).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory as memory_mod  # noqa: E402
from memory import GameMemory, Playbook  # noqa: E402


_REPO = Path(__file__).parent.parent / "memory"


def _load_grid_recipe() -> memory_mod.VisualPlaytestRecipe:
    path = _REPO / memory_mod.VISUAL_PLAYTESTS_FILENAME
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("id") == "canvas-grid-navigation":
            return memory_mod.VisualPlaytestRecipe.from_record(rec, source_tier="root")
    raise AssertionError("canvas-grid-navigation missing from visual_playtests.jsonl")


def _plan_playbook_ids(goal: str) -> list[str]:
    pb = Playbook(base_root=str(_REPO))
    hits = pb.retrieve(goal, stage="plan", k=8)
    return [h.bullet.id for h in hits]


def test_ms_pacman_goal_retrieves_maze_bullets() -> None:
    ids = _plan_playbook_ids("Ms Pac-Man maze chase pellets four pursuers")
    assert "pacman-maze-copy-dont-generate" in ids or "grid-chase-vulnerability-timer" in ids


def test_reskin_pursuer_goal_retrieves_maze_bullets() -> None:
    goal = (
        "maze game, four pursuers chase you through corridors, "
        "power pickup makes them flee — naked people instead of ghosts"
    )
    ids = _plan_playbook_ids(goal)
    assert "pacman-maze-copy-dont-generate" in ids or "grid-chase-vulnerability-timer" in ids


def test_grid_recipe_keywords_include_chaser_modality() -> None:
    recipe = _load_grid_recipe()
    kw = recipe.recipe.get("applies_keywords") or []
    for token in ("chaser", "pursuer", "vulnerable", "ms"):
        assert token in kw, f"missing modality keyword {token!r}"


def test_grid_recipe_auto_probes_are_genre_free() -> None:
    recipe = _load_grid_recipe()
    auto = recipe.recipe.get("auto_probes") or []
    names = {ap["name"] for ap in auto}
    assert "auto_player_not_in_wall" in names
    assert "auto_maze_has_walls" in names
    assert "auto_chasers_array_present" in names
    assert "auto_chaser_moves_autonomously" in names
    assert "auto_vulnerability_mechanism_exposed" in names
    assert "auto_collectibles_counter" in names
    joined = " ".join(ap.get("expr") or "" for ap in auto).lower()
    # Probes may fall back to ghosts[] for legacy games, but must not require it.
    assert "state.ghosts" not in joined or "||" in joined
    assert "frightened" not in joined or "fright|scared|flee|vulner" in joined


def test_grid_recipe_chaser_probe_uses_generic_arrays() -> None:
    recipe = _load_grid_recipe()
    auto = recipe.recipe.get("auto_probes") or []
    move = next(ap for ap in auto if ap["name"] == "auto_chaser_moves_autonomously")
    expr = move["expr"]
    assert "chasers" in expr
    assert "enemies" in expr or "pursuers" in expr


def test_outline_grid_navigation_probes_are_behavioral() -> None:
    mem = GameMemory(root="memory")
    hit = mem.retrieve_implementation_outline(
        "navigate corridors collect pellets while pursuers chase"
    )
    assert hit is not None
    assert hit.item.id == "outline-grid-navigation"
    probes = (hit.item.recipe or {}).get("probes") or []
    text = " ".join(probes).lower()
    assert "chaser" in text or "chasers" in text
    assert "vulnerable" in text or "flee" in text
    assert "ghost" not in text


def test_playbook_grid_chase_bullet_exists() -> None:
    pb = Playbook(base_root=str(_REPO))
    assert any(b.id == "grid-chase-vulnerability-timer" for b in pb.load_all())
