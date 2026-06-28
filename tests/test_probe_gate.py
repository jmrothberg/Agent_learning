"""Tests for the harness changes that make probes ENFORCING (A1)
and require a clean-streak before <done/> ships (A2).

All pure-function — no Chromium, no model. Each test < 5 ms.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402
from tools import (  # noqa: E402
    _classify_probe_eval_error,
    _format_probe_failure_warning,
    _normalize_probe_expr,
)


# ---------------------------------------------------------------------------
# A1 — probe failures must surface as soft_warnings (gating ok=True via
# the existing formula). We exercise the exact code shape the harness
# uses: build a stub report + probes list, run the helper, assert.
# ---------------------------------------------------------------------------


def _stub_report() -> dict:
    """Mimic what _build_report returns BEFORE the post-step that
    promotes probe failures into soft_warnings."""
    return {
        "ok": True,
        "errors": [],
        "warnings": [],
        "soft_warnings": [],
        "logs": [],
        "title": "test",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "body_chars": 100,
        "body_sample": "...",
        "screenshot": None,
        "screenshot_before": None,
        "frozen_canvas": False,
        "input_test": {"ran": True, "any_change": True, "first_responsive_key": "ArrowRight"},
        "probes": [],
    }


def _apply_probe_gate(report: dict) -> None:
    """Inline copy of the post-step that lives in
    LiveBrowser.load_and_test (tools.py around line 1057). Test-only
    so we don't need to spin up Chromium to verify A1.
    """
    for p in (report.get("probes") or []):
        if p.get("ok"):
            continue
        report["soft_warnings"].append(_format_probe_failure_warning(p))
    report["ok"] = (
        len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
    )


def test_a1_no_probes_keeps_ok_true():
    r = _stub_report()
    _apply_probe_gate(r)
    assert r["ok"] is True
    assert r["soft_warnings"] == []


def test_a1_all_probes_pass_keeps_ok_true():
    r = _stub_report()
    r["probes"] = [
        {"name": "canvas_present", "expr": "...", "ok": True, "err": ""},
        {"name": "player_alive",   "expr": "...", "ok": True, "err": ""},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is True


def test_a1_one_probe_fails_flips_ok_to_false():
    """The exact failure mode from the doom-fps trace."""
    r = _stub_report()
    r["probes"] = [
        {"name": "canvas_present",  "expr": "!!document.querySelector('canvas')", "ok": True, "err": ""},
        {"name": "player_alive",    "expr": "window.game.player.health > 0",      "ok": False, "err": ""},
        {"name": "monster_exists",  "expr": "window.game.monsters.length > 0",    "ok": False, "err": ""},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is False
    # Both failed probes get a clear soft_warning so the model can act.
    matching = [w for w in r["soft_warnings"] if "PROBE FAILED" in w]
    assert len(matching) == 2
    assert any("player_alive" in w for w in matching)
    assert any("monster_exists" in w for w in matching)


def test_a1_failed_probe_with_runtime_error_surfaces_message():
    """If a probe threw at evaluate-time, its `err` should appear in
    the soft_warning so the model knows it's a syntax/typo issue."""
    r = _stub_report()
    r["probes"] = [
        {"name": "broken", "expr": "window.gam.player", "ok": False,
         "err": "TypeError: Cannot read properties of undefined"},
    ]
    _apply_probe_gate(r)
    assert r["ok"] is False
    found = [w for w in r["soft_warnings"] if "broken" in w]
    assert found
    assert "TypeError" in found[0]


def test_probe_eval_error_message_blames_probe_not_game():
    r = _stub_report()
    r["probes"] = [
        {
            "name": "move_right",
            "expr": "(()=>{const x0=state.player.x;return new Promise(r=>r(true));});",
            "ok": False,
            "err": "Page.evaluate: SyntaxError: missing ) after argument list",
            "kind": "eval_error",
            "error_class": "syntax_error",
        },
    ]
    _apply_probe_gate(r)
    assert r["ok"] is False
    warning = r["soft_warnings"][0]
    assert "PROBE BROKEN [move_right]" in warning
    assert "This is the probe, not the game" in warning
    assert "fix the game so it evaluates truthy" not in warning


def test_probe_expr_normalization_strips_semicolon_and_invokes_arrow_iife():
    expr = "(()=>{ return Promise.resolve(true); });"
    assert _normalize_probe_expr(expr).endswith("()")
    assert not _normalize_probe_expr(expr).endswith(";")


def test_probe_expr_normalization_leaves_invoked_iife_alone():
    expr = "(()=>true)()"
    assert _normalize_probe_expr(expr) == expr


def test_probe_eval_error_classifier():
    assert _classify_probe_eval_error("SyntaxError: missing ) after argument list") == "syntax_error"
    assert _classify_probe_eval_error("ReferenceError: state is not defined") == "reference_error"
    assert _classify_probe_eval_error("evaluated falsy") is None


# ---------------------------------------------------------------------------
# A2 — clean-streak counter increments on ok=True, resets on ok=False.
# Default streak threshold is 2; <done/> shouldn't ship at streak=1.
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def test_a2_streak_initial_state(tmp_path):
    a = _make_agent(tmp_path)
    assert a._consecutive_clean_iters == 0
    assert a._min_clean_streak_to_ship == 2


def test_a2_streak_increments_on_clean(tmp_path):
    """Simulate the per-iter counter update from agent.run()."""
    a = _make_agent(tmp_path)

    def step(ok: bool):
        if ok:
            a._consecutive_clean_iters += 1
        else:
            a._consecutive_clean_iters = 0

    step(True);  assert a._consecutive_clean_iters == 1
    step(True);  assert a._consecutive_clean_iters == 2
    step(True);  assert a._consecutive_clean_iters == 3
    step(False); assert a._consecutive_clean_iters == 0
    step(True);  assert a._consecutive_clean_iters == 1


def test_a2_streak_threshold_reflects_done_eligibility(tmp_path):
    """The conditional in agent.run() requires
    consecutive_clean_iters >= min_clean_streak_to_ship before honoring
    a model-issued <done/>."""
    a = _make_agent(tmp_path)
    a._consecutive_clean_iters = 1
    streak_ok = a._consecutive_clean_iters >= a._min_clean_streak_to_ship
    assert streak_ok is False  # one clean iter ≠ ship

    a._consecutive_clean_iters = 2
    streak_ok = a._consecutive_clean_iters >= a._min_clean_streak_to_ship
    assert streak_ok is True


# ---------------------------------------------------------------------------
# B1 — 3D intent detector
# ---------------------------------------------------------------------------


def test_b1_3d_keyword_first_person():
    """Two-word phrases like 'first person' detected via word-join."""
    kws = prompts_v1._detect_3d_intent("make a first person shooter")
    assert "firstperson" in kws or "first-person" in kws


def test_b1_3d_genre_title_does_not_fire():
    # Genre/title phrases like "doom like" are NOT 3D modality triggers —
    # no hardcoded genre/title names in Python. Title-level 3D routing is
    # data-driven (visual_playtests.jsonl strong_hooks).
    assert prompts_v1._detect_3d_intent("doom like maze game with monsters") == []
    # A real rendering-modality phrase still fires.
    assert prompts_v1._detect_3d_intent("first person raycaster game") != []


def test_b1_3d_keyword_three_d():
    kws = prompts_v1._detect_3d_intent("3D voxel adventure")
    assert "3d" in kws
    assert "voxel" in kws


def test_b1_3d_no_match_for_2d():
    """Plain 2D goals should not trigger the 3D nudge."""
    assert prompts_v1._detect_3d_intent("make a snake game") == []
    assert prompts_v1._detect_3d_intent("calculator app with light theme") == []
    # Importantly — bare 'doom' alone (no 'like' / 'fps' / 'first-person')
    # doesn't fire. We don't hardcode genre names.
    assert prompts_v1._detect_3d_intent("doom") == []


def test_b1_3d_nudge_appears_in_plan_instruction():
    out = prompts_v1.plan_instruction(goal="first person shooter with monsters")
    assert "3D INTENT DETECTED" in out
    assert "three.js" in out
    assert "file://" in out


def test_b1_no_3d_nudge_for_2d_goal():
    out = prompts_v1.plan_instruction(goal="snake game with pixel art")
    assert "3D INTENT DETECTED" not in out


def test_b1_both_nudges_can_fire_together():
    """A '3D first person shooter with sprite art' goal should trigger
    BOTH the ART INTENT and the 3D INTENT callouts — they're additive."""
    out = prompts_v1.plan_instruction(goal="first person shooter with sprite art")
    assert "ART INTENT DETECTED" in out
    assert "3D INTENT DETECTED" in out


def test_b1_hard_rules_require_threejs_file_runtime():
    joined = "\n".join(prompts_v1.HARD_RULES)
    assert "three.js/WebGL" in joined
    assert "file://" in joined


# ---------------------------------------------------------------------------
# B2 — Mixed sprite/procedural guideline appears in system prompt
# ---------------------------------------------------------------------------


def test_b2_sprite_procedural_mix_in_system_prompt():
    sp = prompts_v1.SYSTEM_PROMPT
    # The new guideline says: sprites for static, procedural for
    # destructible/state-rich. Look for the key concept.
    assert "DESTRUCTIBLE" in sp or "destructible" in sp
    assert "procedural" in sp.lower()


# ---------------------------------------------------------------------------
# B3 — Sprite-orientation guidance in the asset paths block
# ---------------------------------------------------------------------------


def test_b3_orientation_block_removed(tmp_path):
    # Sprite orientation/facing guidance is no longer injected by the asset
    # pipeline — it lives in the playbook (directional-art-faces-right). The
    # pipeline must stay free of art-direction policy.
    import assets
    asset_dir = tmp_path / "x_assets"
    asset_dir.mkdir()
    ship_png = asset_dir / "ship.png"
    ship_png.write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder; filter only checks exists()
    block = assets.render_asset_paths_block(
        {"ship": ship_png},
        tmp_path / "x.html",
    )
    assert "ORIENTATION" not in block
    assert "facing right" not in block


# ---------------------------------------------------------------------------
# Tier 4 — coverage-gap synthetic probes & Phase B probe re-parse
#
# Background: the asteroid_20260510_164857 trace showed that when the
# planning-turn coverage check flagged "Edge"/"Stress" criteria with no
# matching probes, the model ignored the soft-warning form across 4
# iters. The new behavior synthesizes a failing probe per gap and lets
# Phase B re-parse <probes> when the model emits a new block. These
# tests cover the slugifier, the gap detection, and the re-parse
# eligibility logic without needing Chromium or a model.
# ---------------------------------------------------------------------------


def test_slugify_criterion_strips_category_label():
    from tools import _slugify_criterion
    # Slug is capped at 32 chars; the leading category label is stripped.
    assert _slugify_criterion("Edge: pressing space restarts the game") \
        == "pressing_space_restarts_the_game"
    assert _slugify_criterion("Basic: ship visible at startup") == "ship_visible_at_startup"
    # Stress label stripped; lowercase + underscore separators.
    assert _slugify_criterion("Stress: frame rate steady after 20 kills") \
        == "frame_rate_steady_after_20_kills"
    # Cap respected on overlong inputs.
    long_in = "Edge: this is a very long criterion line that runs on past the 32-char cap"
    out = _slugify_criterion(long_in)
    assert len(out) <= 32
    assert out.startswith("this_is_a_very_long")


def test_slugify_criterion_no_label_keeps_text():
    from tools import _slugify_criterion
    # No "Foo:" prefix → entire string is the body.
    assert _slugify_criterion("Player can fire bullets") == "player_can_fire_bullets"


def test_slugify_criterion_empty_falls_back():
    from tools import _slugify_criterion
    # Pure punctuation / empty → safe default.
    assert _slugify_criterion("") == "criterion"
    assert _slugify_criterion("    !@#$    ") == "criterion"


def test_coverage_gap_detection_finds_uncovered_criterion():
    """Sanity: when criteria mention a behavior no probe references,
    _criteria_coverage_gaps returns that criterion line."""
    from tools import _criteria_coverage_gaps
    criteria = (
        "Basic: ship visible at startup.\n"
        "Edge: after game over, pressing space restarts the game."
    )
    probes = [
        {"name": "ship_visible", "expr": "!!window.state.ship"},
        # No probe for the Edge criterion (no mention of "restart" or "game over").
    ]
    gaps = _criteria_coverage_gaps(criteria, probes)
    assert len(gaps) == 1
    assert "Edge" in gaps[0]


def test_coverage_gap_clears_when_probe_added():
    """After the model adds a probe referencing the criterion, the gap
    detector finds no more uncovered lines. This is the path the
    Phase B re-parse exercises — re-emit <probes>, run the gap check
    again, update self._planning_coverage_gaps.

    Note: the gap detector uses lowercased word overlap, so the new
    probe's name/expr must literally contain words from the criterion
    line (e.g. "restarts" or "pressing"). The model is expected to
    use criterion-derived names (the synthetic probe's err message
    tells it to)."""
    from tools import _criteria_coverage_gaps
    criteria = "Edge: pressing space restarts the game after game over."
    probes_before = [{"name": "ship_visible", "expr": "!!window.state.ship"}]
    # New probe whose name reuses words from the criterion ("restarts",
    # "space") → 2 overlapping words → gap closes.
    probes_after = [
        {"name": "ship_visible", "expr": "!!window.state.ship"},
        {"name": "restarts_on_space",
         "expr": "window.state && state.lives === 3 /* after restart */"},
    ]
    assert _criteria_coverage_gaps(criteria, probes_before)  # gap present
    assert _criteria_coverage_gaps(criteria, probes_after) == []  # closed


def _phase_b_reparse_gate(agent, reply: str) -> bool:
    """Inline copy of the Phase B re-parse gate from agent.py so we
    can test the three conditions without running the full async
    iter loop."""
    reply_low = reply.lower()
    has_code = ("<patch>" in reply_low) or ("<html_file>" in reply_low)
    return bool(agent._planning_coverage_gaps) and ("<probes>" in reply_low) and has_code


def test_agent_phase_b_reparse_gate_requires_existing_gap(tmp_path):
    """No gaps → gate closed even when reply has probes AND code."""
    a = _make_agent(tmp_path)
    a._planning_coverage_gaps = []  # no gaps to close
    reply = "<probes>[]</probes>\n<patch>foo</patch>"
    assert _phase_b_reparse_gate(a, reply) is False


def test_agent_phase_b_reparse_gate_opens_when_gaps_and_code_present(tmp_path):
    """Open: unresolved gaps + <probes> + <patch>. The 'doing real
    work' condition is what we just added after the asteroid_173200
    trace: the model has to emit usable code alongside the new
    probes, not just echo the plan shape."""
    a = _make_agent(tmp_path)
    a._planning_coverage_gaps = ["Edge: pressing space restarts the game."]
    reply = "before <probes>[...]</probes>\n<patch>diff</patch> after"
    assert _phase_b_reparse_gate(a, reply) is True

    # html_file form also opens the gate.
    reply2 = "<html_file><!doctype html>...</html_file>\n<probes>[...]</probes>"
    assert _phase_b_reparse_gate(a, reply2) is True


def test_agent_phase_b_reparse_gate_closed_when_no_code(tmp_path):
    """The asteroid_173200 regression: gaps exist, reply has probes,
    but ZERO code. Gate must STAY CLOSED — credit is reserved for
    replies that move the loop forward."""
    a = _make_agent(tmp_path)
    a._planning_coverage_gaps = ["Edge: pressing space restarts the game."]
    # Model emitted plan + probes + assets + sounds and no code.
    reply = (
        "<plan>...</plan>\n"
        "<criteria>Edge: ...</criteria>\n"
        "<probes>[{\"name\":\"x\",\"expr\":\"true\"}]</probes>\n"
        "<assets>[]</assets>\n"
        "<sounds>[]</sounds>"
    )
    assert _phase_b_reparse_gate(a, reply) is False


# ---------------------------------------------------------------------------
# Harness invariant (genre-free): a failing SYNTHETIC coverage_gap probe
# must gate report["ok"] == False — same as a failing model-authored probe.
#
# Bug (battlezone trace 20260622, GLM-5.2): load_and_test synthesized the
# coverage_gap probe and listed it as FAIL in report["probes"], but never
# added the matching PROBE FAILED soft_warning. The final ok-recompute
# (`ok = no errors and no soft_warnings`) therefore left ok=True — the
# report showed "6/7, FAIL coverage_gap__x" yet still shipped GREEN.
#
# These tests exercise the REAL extracted gate `_apply_coverage_gap_gate`
# (the same function load_and_test calls), so a regression that drops the
# soft_warning append turns them RED. No Chromium, no model, no genre.
# ---------------------------------------------------------------------------


def test_synthetic_coverage_gap_probe_fails_report_ok():
    from tools import _apply_coverage_gap_gate
    report = _stub_report()
    # One passing model probe that does NOT cover the second criterion.
    probes = [{"name": "thing_present", "expr": "!!window.state.thing"}]
    report["probes"] = list(probes)
    probe_results = list(probes)
    criteria = (
        "Basic: the thing is present at startup.\n"
        "Behavior: pressing the action key fires a projectile."
    )
    probe_results = _apply_coverage_gap_gate(report, criteria, probes, probe_results)
    # Same ok-recompute formula load_and_test applies after the gate.
    report["ok"] = (
        len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
    )
    assert report["ok"] is False
    assert any(
        w.startswith("PROBE FAILED [coverage_gap__")
        for w in report["soft_warnings"]
    )
    # The synthetic FAIL is also visible in the probe list.
    assert any(
        p["name"].startswith("coverage_gap__") and not p["ok"]
        for p in probe_results
    )


def test_synthetic_coverage_gap_advisory_when_all_model_probes_pass():
    """Round 1 street-fighter trace: model probes passed but Edge criterion
    had no matching probe — synthetic coverage_gap burned iter 1."""
    from tools import _apply_coverage_gap_gate
    report = _stub_report()
    probes = [{"name": "punch_works", "expr": "state.punching === true"}]
    probe_results = [{"name": "punch_works", "expr": "...", "ok": True}]
    report["probes"] = list(probe_results)
    criteria = (
        "Basic: punch animation fires.\n"
        "Edge: hit-stagger recovers after ~300ms."
    )
    probe_results = _apply_coverage_gap_gate(report, criteria, probes, probe_results)
    report["ok"] = (
        len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
    )
    assert report["ok"] is True
    assert not any(
        p["name"].startswith("coverage_gap__") for p in probe_results
    )
    assert any("ADVISORY" in w for w in report.get("warnings", []))


def test_clean_report_stays_ok_when_no_coverage_gap():
    """Guard: when every criterion is covered, the gate adds nothing —
    no synthetic probe, no soft_warning — and ok stays True. The
    coverage_gap fix must NOT over-gate clean games."""
    from tools import _apply_coverage_gap_gate
    report = _stub_report()
    probes = [{
        "name": "fires_projectile",
        "expr": "state.bullets.length > 0 /* action key fires a projectile */",
    }]
    report["probes"] = list(probes)
    probe_results = list(probes)
    criteria = "Behavior: pressing the action key fires a projectile."
    before_warnings = len(report["soft_warnings"])
    before_probes = len(probe_results)
    probe_results = _apply_coverage_gap_gate(report, criteria, probes, probe_results)
    report["ok"] = (
        len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
    )
    assert report["ok"] is True
    assert len(report["soft_warnings"]) == before_warnings
    assert len(probe_results) == before_probes
    assert all(
        not p["name"].startswith("coverage_gap__") for p in probe_results
    )
