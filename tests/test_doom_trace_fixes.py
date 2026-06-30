"""Regression tests for Fix A + Fix B against the 2026-05-25 doom trace
(`maket-the-most-graphic-version_20260525_182007`).

That trace burned 6+ user feedback turns asking the model to flip the
arrow-key direction (a one-line `Math.sin/Math.cos` sign flip). Audit
revealed two structural gaps the previous memory work didn't close:

  Fix A: the architect picked `outline-asset-backed-animation` (generic,
         won on animation-heavy goal) over `outline-3d-first-person`
         (specific). Doom-vocabulary boost + tightening of generic
         tags made the 3D outline win.

  Fix B: scope-discipline playbook bullets (scope-locked-by-user-language,
         patch-budget-when-scope-locked, vlm-critic-can-mislead-on-
         orientation) never fired because playbook retrieval keyed
         only on `self._goal` — never on per-turn user feedback.
         _retrieve_playbook_block now concatenates the most recent
         drained feedback to the query.

These tests pin both fixes so a future edit can't regress them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Verbatim goal text from the trace's session_start event.
DOOM_TRACE_GOAL = (
    "maket the most graphic version of DOOM, with amazing animations, "
    "and all the game play of the original game of doom, fantastic "
    "animated monsters, large and super animated so they have smoth "
    "animation, detailed wall and floor patterns at the highes "
    "resolution, true first person shooter view of weapons, "
    "incredicble graphics when a monster is injured or killed, foxus "
    "on fantastic high resultion graphics"
)

# Verbatim user feedback from the trace (4-times-repeated "fix the arrows").
DOOM_TRACE_SCOPED_FEEDBACK = (
    "JUST change the direction the player moves with the arrow keys "
    "they are REVERSED ive asked you 4 times for the same simpl trivial FIX"
)


# ---------------------------------------------------------------------------
# Fix A: outline routing
# ---------------------------------------------------------------------------


def test_fix_a_doom_trace_goal_routes_to_3d_first_person_outline():
    """The literal trace goal must pick outline-3d-first-person, not the
    generic asset-backed-animation outline."""
    from memory import GameMemory

    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        mem = GameMemory(root="memory")
        hit = mem.retrieve_implementation_outline(DOOM_TRACE_GOAL, None)
    finally:
        os.chdir(old_cwd)
    assert hit is not None
    assert hit.item.id == "outline-3d-first-person", (
        f"Fix A regression: doom trace goal routed to {hit.item.id} "
        f"(score {hit.score:.4f}) instead of outline-3d-first-person. "
        f"Either the asset-backed outline regained generic tokens, OR "
        f"the 3D outline lost its doom-vocabulary boost. See "
        f"`memory/implementation_outlines.jsonl` tags."
    )


def test_fix_a_3d_outline_carries_doom_vocabulary_tags():
    """The 3D outline must keep the doom-vocabulary tags that were
    added in Fix A. Without them, the doom-trace routing test above
    would still pass on small variations of goal text but would fail
    on the real trace text."""
    import json

    REQUIRED_TAGS = {
        "monster", "monsters", "weapon", "weapons", "shooter",
        "walls", "floor", "ceiling",
    }
    with (PROJECT_ROOT / "memory" / "implementation_outlines.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["id"] == "outline-3d-first-person":
                missing = REQUIRED_TAGS - set(d.get("tags", []))
                assert not missing, (
                    f"outline-3d-first-person missing required Fix-A "
                    f"vocabulary tags: {missing}"
                )
                return
    raise AssertionError("outline-3d-first-person not found in JSONL")


# ---------------------------------------------------------------------------
# Fix B: playbook retrieval uses user feedback
# ---------------------------------------------------------------------------


def _make_agent_with_goal(tmp_path: Path, goal: str) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html><body>x</body></html>")
    fake_browser = MagicMock()
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    a._goal = goal
    return a


def test_fix_b_feedback_changes_retrieval_to_responsive_bullets(tmp_path):
    """The actual trace failure: the user typed scope-lock feedback
    ('JUST change ... they are REVERSED ... ive asked you 4 times')
    and the playbook surfaced nothing responsive to the complaint.
    With Fix B, retrieval must (a) change when feedback is added,
    (b) surface a bullet whose content directly addresses the
    feedback (either a scope-discipline bullet OR the bullet that
    contains the exact reversed-arrows fix). The trace event must
    record feedback_in_query=True so postmortem readers can tell."""
    # Retrieve WITHOUT feedback (baseline).
    a0 = _make_agent_with_goal(tmp_path, DOOM_TRACE_GOAL)
    a0._last_drained_feedback = []
    a0._pending_feedback = []
    events0: list[dict] = []
    orig0 = a0._trace
    a0._trace = lambda obj: events0.append(obj) or orig0(obj)
    a0._retrieve_playbook_block(a0._goal, code="", stage="code")
    no_fb_evs = [e for e in events0 if e.get("kind") == "playbook_retrieved"]
    no_fb_ids = set(no_fb_evs[-1]["ids"]) if no_fb_evs else set()
    assert no_fb_evs and no_fb_evs[-1].get("feedback_in_query") is False

    # Retrieve WITH the scope-locked feedback.
    a = _make_agent_with_goal(tmp_path, DOOM_TRACE_GOAL)
    a._last_drained_feedback = [DOOM_TRACE_SCOPED_FEEDBACK]
    events: list[dict] = []
    orig = a._trace
    a._trace = lambda obj: events.append(obj) or orig(obj)
    a._retrieve_playbook_block(a._goal, code="", stage="code")
    fb_evs = [e for e in events if e.get("kind") == "playbook_retrieved"]
    assert fb_evs, "no playbook_retrieved event emitted"
    fb_ids = set(fb_evs[-1]["ids"])

    # (a) Trace event records that feedback was in the query.
    assert fb_evs[-1].get("feedback_in_query") is True, (
        "playbook_retrieved trace event missing feedback_in_query=True"
    )

    # (b) Retrieval result CHANGED when feedback was added (Fix B is
    # actually doing something, not a no-op).
    assert fb_ids != no_fb_ids, (
        f"Fix B no-op: retrieval result identical with and without "
        f"feedback. no_feedback={sorted(no_fb_ids)}, "
        f"with_feedback={sorted(fb_ids)}"
    )

    # (c) At least one surfaced bullet is RESPONSIVE to the user's
    # complaint — either a scope-discipline bullet (telling the model
    # to make a minimal scoped change) OR a bullet that contains the
    # specific arrow-key-direction fix shape (fps-camera-and-movement-
    # vectors carries 'flip the sign of Math.sin/Math.cos in the
    # move vector, NOT the input handler' — exactly the user's bug).
    responsive_bullets = {
        # Scope-discipline family
        "scope-locked-by-user-language",
        "patch-budget-when-scope-locked",
        "add-feature-do-not-touch-working-code",
        "vlm-critic-can-mislead-on-orientation",
        "scope-flip-minimal-change",
        "previous-user-fix-is-locked",
        # Bullet whose content carries the exact reversed-arrows fix
        "fps-camera-and-movement-vectors",
        "fps-minimap-radar-yaw-arrow",
    }
    overlap = responsive_bullets & fb_ids
    assert overlap, (
        f"Fix B regression: no bullets responsive to scope-locked "
        f"reversed-arrows feedback surfaced. Retrieved: {sorted(fb_ids)}"
    )


def test_fix_b_no_feedback_means_no_extra_query_tokens(tmp_path):
    """When no user feedback is pending, retrieval falls back to
    goal-only (the prior behavior). feedback_in_query flag must be
    False on the trace event so future audits can tell which sessions
    used Fix B."""
    a = _make_agent_with_goal(tmp_path, DOOM_TRACE_GOAL)
    a._last_drained_feedback = []
    a._pending_feedback = []

    events: list[dict] = []
    orig_trace = a._trace
    a._trace = lambda obj: events.append(obj) or orig_trace(obj)

    a._retrieve_playbook_block(a._goal, code="", stage="code")

    retrieval_events = [e for e in events if e.get("kind") == "playbook_retrieved"]
    if retrieval_events:
        assert retrieval_events[-1].get("feedback_in_query") is False, (
            "feedback_in_query should be False when no feedback queued"
        )


def test_fix_b_works_across_all_game_shapes(tmp_path):
    """The user's framing: 'i want any game better not just doom!!!!'
    Fix B (playbook retrieval includes per-turn feedback) is a general
    mechanism — verify it surfaces scope-responsive bullets across SIX
    different game shapes, not just FPS. Each case: a goal of the
    shape + a scoped feedback phrase typical for that shape. The
    retrieval must surface AT LEAST ONE bullet from either the
    scope-discipline family OR a shape-specific bullet that directly
    addresses the feedback's failure mode.
    """
    SHAPE_CASES = [
        # (label, goal, scoped feedback)
        ("FIGHTER",  "two character fighter with punch kick fireball",
                     "JUST flip the punch direction only the punch nothing else"),
        ("PADDLE",   "paddle ball breakout with bricks",
                     "only change the ball speed nothing else"),
        ("CHESS",    "chess game with pieces and turns",
                     "do not change any code just fix the castling move"),
        ("MAZE",     "pacman maze with corridors and ghosts",
                     "no other changes just make the maze bigger"),
        ("PLATFORMER", "side scrolling platformer with jumps",
                     "ONLY add a new powerup pickup nothing else changes"),
        ("FPS",      "first person shooter with monsters and weapons",
                     "JUST change the direction the player moves with the arrow keys they are REVERSED ive asked you 4 times for the same simpl trivial FIX"),
    ]
    SCOPE_OR_DIRECT_RESPONSIVE = {
        # Scope-discipline family (added Tier-1 round)
        "scope-locked-by-user-language",
        "patch-budget-when-scope-locked",
        "add-feature-do-not-touch-working-code",
        "vlm-critic-can-mislead-on-orientation",
        "scope-flip-minimal-change",
        "previous-user-fix-is-locked",
        # Shape-specific bullets that ARE directly responsive
        "fps-camera-and-movement-vectors",   # exact arrow-flip fix
        "breakout-ball-launch",              # paddle/ball physics
        "ball-paddle-angle-bias",            # paddle/ball physics
        "chess-board-orientation",           # board games
    }

    misses = []
    for label, goal, feedback in SHAPE_CASES:
        a = _make_agent_with_goal(tmp_path, goal)
        a._last_drained_feedback = [feedback]
        events: list[dict] = []
        orig = a._trace
        a._trace = lambda obj: events.append(obj) or orig(obj)
        a._retrieve_playbook_block(a._goal, code="", stage="code")
        retrieval_events = [e for e in events if e.get("kind") == "playbook_retrieved"]
        if not retrieval_events:
            misses.append((label, "no playbook_retrieved event"))
            continue
        ids = set(retrieval_events[-1]["ids"])
        if not (SCOPE_OR_DIRECT_RESPONSIVE & ids):
            misses.append((label, f"no responsive bullet — got {sorted(ids)}"))
        if retrieval_events[-1].get("feedback_in_query") is not True:
            misses.append((label, "feedback_in_query not True"))
    assert not misses, (
        "Fix B regression on at least one game shape:\n"
        + "\n".join(f"  {l}: {reason}" for l, reason in misses)
    )


def test_every_visual_playtest_recipe_has_fix_hint(tmp_path):
    """User direction: 'any game better, not just doom'. Every mechanism
    recipe in the library must carry a fix_hint so the critic's coaching
    always surfaces the minimal-fix shape — not just for FPS/3D. This
    pins universal coverage so a future recipe addition can't silently
    skip the fix_hint field."""
    import json
    n = 0
    missing = []
    with (PROJECT_ROOT / "memory" / "visual_playtests.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            n += 1
            fh = d.get("recipe", {}).get("fix_hint")
            if not fh or not fh.strip():
                missing.append(d["id"])
    assert not missing, (
        f"{len(missing)} of {n} recipes missing fix_hint: {missing}"
    )


def test_most_visual_playtest_recipes_have_auto_probes(tmp_path):
    """Most recipes should carry auto_probes for deterministic critic
    backstop. A few recipes are intentionally excluded — their game state
    shapes vary too much for a meaningful objective probe, so the VLM
    checklist is the verification:
      - canvas-vfx-fluid, generic-canvas-game-baseline (original authors)
      - canvas-lit-dungeon, canvas-mobile-touch (added 2026-06-02): a lighting
        composite / a touch-control overlay have no canonical numeric state to
        assert; a `return true` no-op probe would be noise. The checklist
        ('is most of the screen dark with a lit radius', 'are on-screen touch
        controls visible') is the real check.
    Pin coverage at >= 17 so any future recipe addition can't skip the
    auto_probe field without a deliberate exclusion."""
    import json
    n = 0; with_probes = 0
    EXCLUDED = {
        "canvas-vfx-fluid", "generic-canvas-game-baseline",
        "canvas-lit-dungeon", "canvas-mobile-touch",
    }
    with (PROJECT_ROOT / "memory" / "visual_playtests.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            n += 1
            ap = d.get("recipe", {}).get("auto_probes") or []
            if ap:
                with_probes += 1
            elif d["id"] not in EXCLUDED:
                raise AssertionError(
                    f"{d['id']} has no auto_probes but isn't on the "
                    f"project-author exclusion list {EXCLUDED}. Add "
                    f"auto_probes OR justify the exclusion."
                )
    assert with_probes >= 17, (
        f"only {with_probes}/{n} recipes have auto_probes; pin floor "
        f"is 17 (19 minus 2 author-excluded)"
    )


def test_fix_b_pending_feedback_used_when_drained_is_empty(tmp_path):
    """If `_retrieve_playbook_block` is called BEFORE feedback drain
    (e.g. from a planning code path), `_pending_feedback` is the
    available source. Fix B must use it as a fallback."""
    a = _make_agent_with_goal(tmp_path, DOOM_TRACE_GOAL)
    a._last_drained_feedback = []
    a._pending_feedback = [DOOM_TRACE_SCOPED_FEEDBACK]

    events: list[dict] = []
    orig_trace = a._trace
    a._trace = lambda obj: events.append(obj) or orig_trace(obj)

    a._retrieve_playbook_block(a._goal, code="", stage="code")

    retrieval_events = [e for e in events if e.get("kind") == "playbook_retrieved"]
    assert retrieval_events
    assert retrieval_events[-1].get("feedback_in_query") is True
