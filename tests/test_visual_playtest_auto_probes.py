"""Deterministic auto-probe layer for visual-playtest recipes.

Pairs with the VLM checklist: when a recipe matches the session,
its `auto_probes` get injected into `self._probes` and run every
iter. The VLM checklist is a perception layer (may be wrong); the
auto-probe is a state-shape assertion (deterministic).

Motivating trace: mortal-kombat 2026-05-24 iter 12 flipped the
sprite-facing condition wholesale (`facing === -1` → `facing === 1`),
making both fighters face the same direction. The VLM might or might
not have caught it from a screenshot; the auto-probe asserts
`Math.sign(p1.facing) !== Math.sign(p2.facing)` directly against
state — fails the iter the regression lands.

Tests are structural (recipe library carries auto_probes for the
right mechanisms; JS strings are syntactically balanced + conservative
when fields absent) plus an end-to-end injection test (recipe
matches → auto_probes appear in `self._probes`).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import memory as memory_mod  # noqa: E402

# The visual-playtest library is a hand-edited data file at the repo's
# `memory/visual_playtests.jsonl` (not seeded from Python). Tests load
# from the canonical committed file so they exercise the real library
# the agent uses in production.
_REPO_MEMORY = Path(__file__).parent.parent / "memory"


def _load_seeded_recipes() -> list[memory_mod.VisualPlaytestRecipe]:
    """Read the canonical committed visual_playtests.jsonl directly."""
    import json
    path = _REPO_MEMORY / memory_mod.VISUAL_PLAYTESTS_FILENAME
    out: list[memory_mod.VisualPlaytestRecipe] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(
            memory_mod.VisualPlaytestRecipe.from_record(rec, source_tier="root")
        )
    return out


def _make_agent(tmp_path: Path, *, goal: str = "") -> GameAgent:
    """Construct an agent pointed at the REPO `memory/` so it sees the
    canonical visual_playtests.jsonl (and all other shipped recipes)."""
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(_REPO_MEMORY),
    )
    a._goal = goal
    return a


# ----------------------------------------------------------------------
# Library wiring: the right recipes carry auto_probes; the JS is balanced.
# ----------------------------------------------------------------------


def test_seeded_recipes_with_auto_probes_are_balanced() -> None:
    """For every recipe that declares auto_probes, the JS expressions
    must be syntactically balanced (parens / braces) so they don't
    crash the browser eval."""
    recipes = _load_seeded_recipes()
    found_any = False
    for r in recipes:
        auto = r.recipe.get("auto_probes") or []
        for ap in auto:
            found_any = True
            expr = ap.get("expr") or ""
            assert expr, f"recipe {r.id} probe {ap.get('name')} has empty expr"
            assert expr.count("(") == expr.count(")"), (
                f"recipe {r.id} probe {ap.get('name')}: parens unbalanced"
            )
            assert expr.count("{") == expr.count("}"), (
                f"recipe {r.id} probe {ap.get('name')}: braces unbalanced"
            )
            # Must be an IIFE so it works as a standalone probe expression.
            assert expr.startswith("(()=>{"), (
                f"recipe {r.id} probe {ap.get('name')}: not an IIFE"
            )
            assert expr.endswith("})()"), (
                f"recipe {r.id} probe {ap.get('name')}: not an IIFE"
            )
    assert found_any, "no recipes seeded with auto_probes — wiring broken"


def test_two_actors_facing_recipe_has_facing_probe() -> None:
    """The mortal-kombat regression recipe MUST have an auto-probe
    that asserts opposite-sign facing."""
    recipes = _load_seeded_recipes()
    fighter = [r for r in recipes if r.id == "canvas-two-actors-facing"][0]
    auto = fighter.recipe.get("auto_probes") or []
    assert auto, "canvas-two-actors-facing must have auto_probes"
    names = [ap["name"] for ap in auto]
    assert "auto_actors_face_each_other" in names
    # The expr must reference Math.sign and facing.
    expr = next(ap["expr"] for ap in auto if ap["name"] == "auto_actors_face_each_other")
    assert "Math.sign" in expr
    assert "facing" in expr


def test_topdown_facing_probe_optional_for_fixed_shooters() -> None:
    """Centipede/Galaga blasters need not expose player.facing — the
    auto-probe must pass when no facing/angle/direction field exists."""
    recipes = _load_seeded_recipes()
    topdown = [r for r in recipes if r.id == "canvas-top-down-action"][0]
    expr = next(
        ap["expr"] for ap in (topdown.recipe.get("auto_probes") or [])
        if ap["name"] == "auto_topdown_player_can_face_directions"
    )
    assert "hasFacing" in expr
    assert "hasAngle" in expr
    assert "hasDir" in expr
    # Simulated fixed-shooter player (no facing fields) must pass.
    js = (
        "const window={state:{player:{x:100,y:400,firing:0}}};"
        + expr
    )
    # Can't eval JS in pytest easily — structural check only.
    assert "if(!hasFacing&&!hasAngle&&!hasDir)returntrue;" in expr.replace(" ", "")
    """canvas-grid-navigation must have a probe asserting the player
    isn't standing inside a wall cell."""
    recipes = _load_seeded_recipes()
    grid = [r for r in recipes if r.id == "canvas-grid-navigation"][0]
    auto = grid.recipe.get("auto_probes") or []
    names = [ap["name"] for ap in auto]
    assert "auto_player_not_in_wall" in names


def test_controllable_player_recipe_has_bounds_probe() -> None:
    recipes = _load_seeded_recipes()
    ctrl = [r for r in recipes if r.id == "canvas-controllable-player"][0]
    auto = ctrl.recipe.get("auto_probes") or []
    names = [ap["name"] for ap in auto]
    assert "auto_player_within_canvas_bounds" in names


def test_fps_recipe_has_map_and_enemy_probes() -> None:
    recipes = _load_seeded_recipes()
    fps = [r for r in recipes if r.id == "canvas-3d-first-person"][0]
    names = {ap["name"] for ap in (fps.recipe.get("auto_probes") or [])}
    assert "auto_fp_map_has_structure" in names
    assert "auto_fp_enemies_present" in names


def test_voxel_recipe_has_hotbar_probe() -> None:
    recipes = _load_seeded_recipes()
    voxel = [r for r in recipes if r.id == "canvas-voxel-sandbox"]
    assert voxel, "canvas-voxel-sandbox missing from visual_playtests.jsonl"
    names = {ap["name"] for ap in (voxel[0].recipe.get("auto_probes") or [])}
    assert "auto_voxel_hotbar_has_multiple_types" in names


def test_auto_probes_are_conservative_default_true() -> None:
    """Every auto-probe expression must contain at least one
    `return true` so it doesn't fail games that simply don't expose
    the relevant state shape. Conservative-by-default rule."""
    recipes = _load_seeded_recipes()
    for r in recipes:
        for ap in (r.recipe.get("auto_probes") or []):
            expr = ap.get("expr") or ""
            assert "return true" in expr, (
                f"{r.id}/{ap.get('name')}: must include `return true` "
                "as the fallback path — probes that fail on missing "
                "state shapes break games that legitimately don't expose "
                "those fields."
            )


def test_no_auto_probes_on_unsuitable_recipes() -> None:
    """Recipes whose mechanisms don't have a clean state-shape
    assertion (vfx-fluid, racing-perspective, puzzle-grid, etc.)
    should NOT carry auto_probes — better to leave the VLM as the
    sole signal than add a probe that fires unpredictably across
    legitimate state shapes."""
    recipes = _load_seeded_recipes()
    # vfx-fluid: no canonical state shape ("particles" array varies
    # too much across implementations).
    vfx = [r for r in recipes if r.id == "canvas-vfx-fluid"][0]
    assert not vfx.recipe.get("auto_probes")
    # Generic baseline: by design has no auto-probes.
    gen = [r for r in recipes if r.id == "generic-canvas-game-baseline"][0]
    assert not gen.recipe.get("auto_probes")


# ----------------------------------------------------------------------
# Injection behavior: probe set grows with auto-probes; idempotent;
# preserves model-authored probes.
# ----------------------------------------------------------------------


def test_facing_eval_goal_injects_both_auto_probes(tmp_path: Path) -> None:
    """Eval GOAL (patch-only fighter seed) must match canvas-two-actors-facing."""
    goal = (
        "Two-player 1v1 fighter versus duel. Use ONLY the existing blue_idle and "
        "red_idle PNGs — no new art, no combat, no HUD. Both fighters must always "
        "face each other."
    )
    a = _make_agent(tmp_path, goal=goal)
    a._probes = []
    a._session_assets = {
        "blue_idle": tmp_path / "blue_idle.png",
        "red_idle": tmp_path / "red_idle.png",
    }
    a._maybe_inject_visual_playtest_auto_probes()
    names = [p["name"] for p in a._probes]
    assert "auto_actors_face_each_other" in names
    assert "auto_actors_face_each_other_strict" in names


def test_inject_adds_recipe_auto_probes_to_session(tmp_path: Path) -> None:
    """When a recipe matches, its auto-probes should land in self._probes."""
    a = _make_agent(tmp_path, goal="mortal kombat style two player fighter")
    a._probes = [{"name": "canvas_present", "expr": "!!document.querySelector('canvas')"}]
    a._maybe_inject_visual_playtest_auto_probes()
    names = [p["name"] for p in a._probes]
    assert "canvas_present" in names  # model probe preserved
    assert "auto_actors_face_each_other" in names  # auto-probe added


def test_inject_is_idempotent(tmp_path: Path) -> None:
    """Repeated calls must not duplicate the same probe."""
    a = _make_agent(tmp_path, goal="mortal kombat fighter")
    a._probes = []
    a._maybe_inject_visual_playtest_auto_probes()
    n1 = len(a._probes)
    a._maybe_inject_visual_playtest_auto_probes()
    a._maybe_inject_visual_playtest_auto_probes()
    n2 = len(a._probes)
    assert n1 == n2, f"auto-probes duplicated on repeated injection ({n1} → {n2})"


def test_inject_noop_when_no_recipe_matches(tmp_path: Path) -> None:
    """A goal that doesn't match any mechanism recipe leaves probes alone."""
    a = _make_agent(tmp_path, goal="meditation timer with breathing prompts")
    a._probes = [{"name": "canvas_present", "expr": "!!document.querySelector('canvas')"}]
    a._maybe_inject_visual_playtest_auto_probes()
    assert len(a._probes) == 1
    assert a._probes[0]["name"] == "canvas_present"


def test_inject_when_probes_empty(tmp_path: Path) -> None:
    """Model may have emitted no <probes> block; auto-probes still inject."""
    a = _make_agent(tmp_path, goal="mortal kombat fighter game")
    a._probes = []
    a._maybe_inject_visual_playtest_auto_probes()
    names = [p["name"] for p in a._probes]
    assert "auto_actors_face_each_other" in names


def test_inject_when_probes_none(tmp_path: Path) -> None:
    """Defensive: self._probes can be None on a fresh agent before
    planning has run."""
    a = _make_agent(tmp_path, goal="pacman style maze game")
    a._probes = None  # type: ignore[assignment]
    a._maybe_inject_visual_playtest_auto_probes()
    assert a._probes is not None
    names = [p["name"] for p in a._probes]
    assert "auto_player_not_in_wall" in names


# ----------------------------------------------------------------------
# The motivating regression: would the new probe catch the mortal-kombat
# iter-12 flip? Simulate with a fake state-shape evaluation.
# ----------------------------------------------------------------------


def _eval_facing_probe(p1_facing: float, p2_facing: float) -> bool:
    """Crude Python simulation of the probe's JS logic against a
    synthetic state shape — same conservative defaults."""
    if p1_facing == 0 and p2_facing == 0:
        return False
    import math
    return math.copysign(1, p1_facing) != math.copysign(1, p2_facing)


def test_facing_probe_passes_when_actors_face_each_other() -> None:
    """Healthy state — P1 facing right (+1), P2 facing left (-1)."""
    assert _eval_facing_probe(1, -1) is True
    assert _eval_facing_probe(-1, 1) is True


def test_facing_probe_fails_on_wholesale_flip_regression() -> None:
    """Mortal-kombat iter-12 regression: both fighters' facing
    got flipped the same way — both now face right (or both left).
    Probe must FAIL this state."""
    assert _eval_facing_probe(1, 1) is False
    assert _eval_facing_probe(-1, -1) is False


def test_facing_probe_fails_when_both_facing_zero() -> None:
    """Edge: both .facing default to 0 — not a meaningful facing.
    Probe fails so the model gets a signal something's wrong."""
    assert _eval_facing_probe(0, 0) is False
