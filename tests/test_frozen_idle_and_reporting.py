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
    branch = src[fi:fi + 2800]
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


# ---- Run_05 quality fixes: turn-based board idle + pose-only undrawn -------

def test_turn_based_board_frozen_is_non_blocking():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "_turn_based_board_idle" in src
    assert "turn-based board is static" in src


def test_undrawn_pose_frames_helper():
    drawn = "gold_gumdrop_idle.png purple_jelly_idle.png"
    undrawn = ["gold_gumdrop_hop_up", "gold_gumdrop_hop_land"]
    assert tools._undrawn_are_animation_poses_only(undrawn, drawn) is True
    assert tools._undrawn_are_animation_poses_only(
        ["gold_gumdrop_idle"], drawn
    ) is False


def test_probe_pointer_board_click_patch():
    expr = (
        "(async()=>{const c=document.querySelector('canvas');"
        "const r=c.getBoundingClientRect();"
        "const sx=r.left+r.width*(3/8),sy=r.top+r.height*(7/8);"
        "c.dispatchEvent(new MouseEvent('mousedown',{clientX:sx,clientY:sy}));"
        "return true;})()"
    )
    patched = tools._patch_probe_pointer_board_clicks(expr)
    assert "__harnessPointerClick" in patched
    assert "__harnessOccupiedBoardClick" in patched
    assert "window.__harnessPointerClick(c,sx,sy)" in patched


def test_webgl_blank_view_is_advisory_not_gating():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "_is_webgl_or_three_game" in src
    assert "ADVISORY (non-blocking): 3D/canvas view appears blank" in src


def test_holochess_iter1_not_rejected_for_else_branches():
    """Run_05 holochess: 33 KB complete game must not fail materialize on
    normal chess `} else {` branches or repeated session asset paths."""
    import json
    import re
    from agent import _baseline_structurally_broken
    trace = PROJECT_ROOT / (
        "games/tune_serial10/run_05/traces/"
        "11_build_a_holochess_game__8x8_boar__run_20260629_114835_314603.jsonl"
    )
    if not trace.exists():
        return
    reply = None
    for line in trace.open():
        d = json.loads(line)
        if d.get("kind") == "assistant_reply" and d.get("iteration") == 1:
            reply = d["reply"]
            break
    assert reply
    m = re.search(r"<html_file>\s*(.*?)\s*</html_file>", reply, re.DOTALL | re.I)
    html = m.group(1).strip()
    assert len(html) > 20000
    mp = tools.run_micro_probes(html)
    assert mp["ok"] is True, mp.get("errors")
    assert _baseline_structurally_broken(html) is None
    assert tools._is_benign_script_repeat_line("} else {")
    assert tools._is_benign_script_repeat_identifier(
        "_build_a_holochess_game__8x8_boar_assets"
    )
