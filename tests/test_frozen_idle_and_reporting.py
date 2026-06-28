"""Frozen-canvas-at-idle downgrade + higher-signal iter_summary + single-character
visual recipe (2026-05-31).

From the `here-s-a-tight-test-prompt 20260530` trace: a single-character animation
test idled on a static sprite, was hard-flagged FROZEN-CANVAS (flipping ok=False)
even though probes passed 7/7 and input changed the canvas — and that false blocker
starved a follow-up two-player feature request. Also the trace recorded only a
soft_warnings COUNT (not the texts), and the critic used the two-fighter recipe for
a one-character goal.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402
import tools  # noqa: E402
from memory import find_best_visual_playtest, VisualPlaytestRecipe  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent


# ---- Agent fix A1: frozen-at-idle is non-blocking when input responds -------

def test_frozen_canvas_idle_by_design_does_not_hard_block():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    # the frozen branch classifies input-responsiveness and only the
    # NON-responsive case appends the ok-flipping FROZEN-CANVAS soft_warning
    assert "frozen_canvas_input_responsive" in src
    assert "FROZEN-AT-IDLE" in src
    # the blocking soft_warning is gated behind the input-not-responsive branch
    fi = src.find("if frozen is True:")
    assert fi != -1
    branch = src[fi:fi + 1600]
    assert "input_responsive" in branch
    assert 'report["soft_warnings"].append' in branch  # still blocks a TRUE freeze


# ---- Reporting: iter_summary carries WHY it blocked + deferred feedback ------

def test_iter_summary_records_soft_warnings_and_deferred_feedback():
    src = inspect.getsource(agent.GameAgent.run)
    # the iter_summary payload now includes the actual warning texts, the
    # frozen false-positive classifier, and any queued (deferrable) feedback
    assert '"soft_warnings": [str(w)' in src
    assert '"frozen_canvas_input_responsive"' in src
    assert '"pending_feedback"' in src and '"pending_feedback_count"' in src


# ---- Memory B2: single-character goal picks the single-fighter recipe -------

def _load_recipes():
    recs = []
    with (PROJECT_ROOT / "memory" / "visual_playtests.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs.append(VisualPlaytestRecipe(
                id=d["id"], kind=d["kind"], content=d["content"],
                tags=d.get("tags", []), recipe=d["recipe"]))
    return recs


def test_single_character_goal_selects_single_fighter_recipe():
    recs = _load_recipes()
    r, _ = find_best_visual_playtest(
        recs,
        goal="A single-character animation test, one martial artist fighter with "
             "punch kick jump duck run fireball, each animated",
        plan_text="", asset_names=[])
    assert r is not None and r.id == "canvas-single-fighter"


def test_two_fighter_goal_still_selects_two_actors_recipe():
    recs = _load_recipes()
    r, _ = find_best_visual_playtest(
        recs,
        goal="street fighter, two fighters versus, punch kick fireball, health bars",
        plan_text="", asset_names=[])
    assert r is not None and r.id == "canvas-two-actors-facing"


def test_single_fighter_recipe_checks_one_character_and_distinct_pose():
    recs = {r.id: r for r in _load_recipes()}
    cl = " ".join(recs["canvas-single-fighter"].recipe["checklist"]).lower()
    assert "one character" in cl
    assert "non-blank" in cl or "hud" in cl
    assert recs["canvas-single-fighter"].recipe.get("auto_probes")  # has a probe
