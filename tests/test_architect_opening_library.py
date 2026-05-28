"""Tests for the 2026-05-25 architect-side opening library expansion.

Context: the user asked for "massive memory improvements" specifically for
the architect role, framed as a chess-opening library — many precomputed
game-shape outlines retrieved selectively, never bloating the prompt.

`retrieve_implementation_outline` returns at most ONE outline per session
(k=1), so adding many doesn't risk prompt bloat — it just covers more
game shapes. The library grew from 3 → 19 outlines mirroring the
mechanism shapes in `memory/visual_playtests.jsonl`.

These tests pin:
 1. All 19 outlines exist + parse + have minimum required fields.
 2. Per-shape retrieval lands on the right outline for a representative
    goal text (the chess-opening-correctness criterion).
 3. Each outline's content stays under a per-entry budget so the
    architect's char_budget=2600 isn't blown on one outline alone.
 4. The library doesn't crowd retrieval on goals where no outline should
    fire strongly (a simple "todo list" returns either nothing or a
    weak score).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTLINES_PATH = PROJECT_ROOT / "memory" / "implementation_outlines.jsonl"

# Mapping of representative goal text → outline_id that MUST win.
# These are the chess-opening correctness assertions: a session whose goal
# text matches a known game shape MUST get the matching outline.
SHAPE_TO_OUTLINE = [
    ("navigate a maze full of dots with ghosts in corridors", "outline-grid-navigation"),
    ("side scrolling platformer with jumps and platforms", "outline-side-scroll-platformer"),
    ("first-person 3D shooter raycaster with walls and enemies", "outline-3d-first-person"),
    ("top-down action game with arrow key movement", "outline-top-down-action"),
    ("puzzle game where blocks drop and clear rows", "outline-puzzle-grid"),
    ("racing game with road perspective and steering", "outline-racing-perspective"),
    ("paddle and ball game with bricks", "outline-paddle-ball"),
    ("lane crossing game with traffic obstacles", "outline-lane-crossing"),
    ("point and click adventure with scenes and inventory", "outline-point-and-click"),
    ("isometric tile city builder", "outline-isometric-tile"),
    ("overworld RPG with NPCs and dialogue", "outline-overworld-rpg"),
    ("vertical platformer with ladders and barrels donkey kong style",
     "outline-vertical-platformer"),
    ("two character fighting game with punch kick fireball",
     "outline-two-actors-facing"),
    ("particle simulation with fire and smoke effects", "outline-vfx-fluid"),
    ("city building tycoon with resources and buildings", "outline-city-builder"),
    ("space trading exploration with systems and markets", "outline-space-trading"),
    ("chess game with two players and pieces", "outline-turn-based-board"),
    ("checkers two player game", "outline-turn-based-board"),
    ("tic tac toe", "outline-turn-based-board"),
    ("simple canvas arcade game with player input", "outline-controllable-canvas-game"),
    ("asset-backed animated game with sprite frames", "outline-asset-backed-animation"),
]


def _load_outlines() -> list[dict]:
    out = []
    with OUTLINES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_all_outlines_parse_and_have_required_fields():
    outlines = _load_outlines()
    assert len(outlines) >= 19, f"expected ≥19 outlines, got {len(outlines)}"
    for o in outlines:
        assert o.get("id"), o
        assert o.get("kind") == "implementation_outline"
        assert o.get("content"), f"{o['id']} has empty content"
        assert isinstance(o.get("tags"), list) and o["tags"], (
            f"{o['id']} has empty tags"
        )
        assert o.get("source_tier") == "root"


def test_per_outline_content_fits_architect_budget():
    """Each outline's content stays under 900 chars so the architect's
    char_budget=2600 has room for playtests + asset_audits +
    animation_audits in the same block. If this fires, an outline ran
    too long — trim it; don't bump the budget."""
    BUDGET_PER_OUTLINE = 900
    for o in _load_outlines():
        n = len(o["content"])
        assert n <= BUDGET_PER_OUTLINE, (
            f"{o['id']}: {n} chars exceeds per-outline budget "
            f"{BUDGET_PER_OUTLINE} — trim the outline rather than bumping "
            f"the budget."
        )


def test_each_shape_retrieves_correct_outline():
    """The opening-library correctness assertion: each representative
    goal MUST retrieve the expected outline. If this fails, either the
    new outline's tags are too narrow OR the existing outlines' tags
    are too broad and crowding out the right answer."""
    from memory import GameMemory

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = GameMemory(root="memory")
        misses = []
        for goal, expected in SHAPE_TO_OUTLINE:
            hit = mem.retrieve_implementation_outline(goal, modality=None)
            actual = hit.item.id if hit else None
            if actual != expected:
                misses.append((goal, expected, actual))
    finally:
        os.chdir(old_cwd)
    assert not misses, (
        f"{len(misses)} shape→outline mismatches:\n"
        + "\n".join(f"  {g!r} expected {e}, got {a}" for g, e, a in misses)
    )


def test_simple_unrelated_goal_does_not_match_strongly():
    """A goal like 'todo list app' shouldn't strongly match any game
    outline — most outlines should score 0 or below the retrieval
    threshold. The library being large must not crowd unrelated goals."""
    from memory import GameMemory

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = GameMemory(root="memory")
        hit = mem.retrieve_implementation_outline("todo list app", None)
    finally:
        os.chdir(old_cwd)
    # Either nothing matches OR the match is weak. We don't assert no
    # match (low-score matches are fine); we assert the score is below
    # a "strong match" threshold so the unrelated goal isn't getting
    # spurious architect guidance.
    if hit is not None:
        assert hit.score < 0.10, (
            f"todo list app got a too-strong outline match: "
            f"{hit.item.id} score={hit.score:.3f}"
        )


def test_doom_trace_goal_routes_to_3d_first_person():
    """Regression for the 2026-05-25 doom trace (maket-the-most-graphic-
    version_20260525_182007). The actual goal from that session repeats
    'animation'/'animated'/'animations' 4 times — without Fix A's tag
    rebalancing, `outline-asset-backed-animation` won (score 0.0732)
    over `outline-3d-first-person` (score 0.0448), so all the 3D
    architect guidance never reached Phase A. Fix A: tighten the
    asset-backed outline's generic tokens + boost the 3D outline with
    doom-vocabulary (monster, weapon, shooter, walls, floor, ceiling).
    """
    import os

    from memory import GameMemory

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = GameMemory(root="memory")
        goal_doom = (
            "maket the most graphic version of DOOM, with amazing animations, "
            "and all the game play of the original game of doom, fantastic "
            "animated monsters, large and super animated so they have smoth "
            "animation, detailed wall and floor patterns at the highes "
            "resolution, true first person shooter view of weapons, "
            "incredicble graphics when a monster is injured or killed, foxus "
            "on fantastic high resultion graphics"
        )
        hit = mem.retrieve_implementation_outline(goal_doom, None)
    finally:
        os.chdir(old_cwd)
    assert hit is not None
    assert hit.item.id == "outline-3d-first-person", (
        f"doom-goal regression: expected outline-3d-first-person, "
        f"got {hit.item.id} (score {hit.score:.4f}). Either the "
        f"asset-backed-animation outline regrew generic tokens, or the "
        f"3D outline lost its doom-vocabulary boost."
    )


def test_3d_outline_wins_when_3d_keywords_dominate():
    """Guard adjacent to the doom-trace regression: a goal naming the 3D
    voxel shape with animation vocabulary still picks the 3D outline.
    (Shorter / more ambiguous 3D-FPS-shape goals — e.g. just 'first
    person shooter' — are NOT pinned here; those are too short to
    score reliably against any outline. Real-world FPS goals tend to
    be long like the doom trace and route correctly.)"""
    import os

    from memory import GameMemory

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = GameMemory(root="memory")
        hit = mem.retrieve_implementation_outline(
            "3D voxel game with animated character animations everywhere",
            None,
        )
    finally:
        os.chdir(old_cwd)
    assert hit is not None
    assert hit.item.id == "outline-3d-first-person", (
        f"got {hit.item.id}, expected outline-3d-first-person"
    )


def test_outline_content_mentions_state_or_probes_or_recommendation():
    """Each outline should give the architect actionable guidance —
    state shape, probe templates, or a recommendation. A vague outline
    that doesn't name concrete fields is dead weight. We accept ANY of
    several actionable markers."""
    actionable_markers = (
        "state.", "state[", "window.state", "window.gameState",
        "probe", "Probe",
        "recommend", "PREFER", "MUST", "should",
        "ctx.", "drawImage", "addEventListener",
    )
    for o in _load_outlines():
        c = o["content"]
        assert any(m in c for m in actionable_markers), (
            f"{o['id']} content has no actionable markers — "
            f"add a state.X reference, a probe, or a concrete recommendation."
        )
