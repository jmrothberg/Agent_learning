"""Tests for the probe-quality classifier added to agent.py.

DK trace 20260513_185815 rationale: the Phase-A probes were all
structural-presence checks (e.g. `!!window.state`, `typeof
state.player === 'object'`). A game that renders a static HUD passes
them all — so the harness signal said "ok" while gameplay was
missing. The classifier flags this pattern at plan time so the model
can add dynamic-behavior probes before Phase B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


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


# ---------------------------------------------------------------------------
# Per-expression classification
# ---------------------------------------------------------------------------


def test_structural_existence_check_is_not_dynamic():
    assert GameAgent._is_dynamic_probe("!!window.state") is False
    assert GameAgent._is_dynamic_probe("typeof state.player === 'object'") is False
    assert GameAgent._is_dynamic_probe("window.hasOwnProperty('y')") is False
    assert GameAgent._is_dynamic_probe("Array.isArray(state.barrels)") is False


def test_truthy_check_alone_is_not_dynamic():
    assert GameAgent._is_dynamic_probe("!!state && !!state.player") is False


def test_zero_threshold_is_not_dynamic():
    """Comparisons against 0 are presence-of-non-empty, NOT
    behavior-over-time. `state.score > 0` can be true at init if the
    game starts with score=1; `state.player.x > 0` is true for any
    game whose player has positive starting coords. DK-trace pin:
    `state.barrels.length > 0` was being classified as dynamic and
    suppressing the nudge — that's the false-negative this fixes."""
    assert GameAgent._is_dynamic_probe("state.score > 0") is False
    assert GameAgent._is_dynamic_probe("state.player.x > 0") is False
    assert GameAgent._is_dynamic_probe("0 < state.barrels.length") is False


def test_meaningful_threshold_against_state_is_dynamic():
    """A non-zero threshold against a non-structural property is
    a genuine "game advanced past this point" check."""
    assert GameAgent._is_dynamic_probe("state.frame >= 30") is True
    assert GameAgent._is_dynamic_probe("state.score >= 10") is True
    assert GameAgent._is_dynamic_probe("state.player.x > 100") is True


def test_length_comparisons_are_never_dynamic():
    """`.length`, `.size`, `.width`, `.height` are structural-presence
    properties; comparing them to ANY number is a "the thing has at
    least N items / N pixels" check, true at init."""
    assert GameAgent._is_dynamic_probe(
        "state.barrels.length > 0"
    ) is False
    assert GameAgent._is_dynamic_probe(
        "state.barrels.length >= 1"
    ) is False
    assert GameAgent._is_dynamic_probe(
        "c.toDataURL().length > 200"
    ) is False
    assert GameAgent._is_dynamic_probe(
        "canvas.width > 100"
    ) is False
    assert GameAgent._is_dynamic_probe(
        "el.textContent.length > 0"
    ) is False


def test_promise_and_await_is_dynamic():
    expr = "await new Promise(r => setTimeout(r, 500))"
    assert GameAgent._is_dynamic_probe(expr) is True
    assert GameAgent._is_dynamic_probe("new Promise(r => r()).then(()=>1)") is True


def test_raf_and_timers_are_dynamic():
    assert GameAgent._is_dynamic_probe(
        "requestAnimationFrame(()=>{})"
    ) is True
    assert GameAgent._is_dynamic_probe("setTimeout(()=>{}, 100)") is True
    assert GameAgent._is_dynamic_probe("setInterval(()=>{}, 16)") is True
    assert GameAgent._is_dynamic_probe(
        "performance.now() - t0 > 100"
    ) is True


def test_canvas_pixel_read_is_dynamic():
    expr = "ctx.getImageData(0, 0, 100, 100).data.some(v => v !== 0)"
    assert GameAgent._is_dynamic_probe(expr) is True


def test_input_dispatch_then_state_delta_is_dynamic():
    """The canonical input→delta probe the prompts now hand the model:
    record a coord, dispatch a KeyboardEvent, await a frame, assert the
    coord changed. Its `await`+`setTimeout` make it dynamic — this is
    the probe shape the probe-quality re-prompt asks for, so the
    classifier MUST recognise it (else the retry would loop or the
    nudge would mis-fire on a genuinely dynamic plan)."""
    expr = (
        "(async()=>{if(!window.state||!state.player)return false;"
        "const x0=state.player.x;window.dispatchEvent(new KeyboardEvent("
        "'keydown',{code:'ArrowRight',bubbles:true}));await new Promise("
        "r=>setTimeout(r,250));window.dispatchEvent(new KeyboardEvent("
        "'keyup',{code:'ArrowRight',bubbles:true}));return "
        "state.player.x!==x0;})()"
    )
    assert GameAgent._is_dynamic_probe(expr) is True
    # And in a bulk set alongside structural probes, the ratio is > 0.
    probes = [
        {"name": "canvas_present", "expr": "!!document.querySelector('canvas')"},
        {"name": "state_exists", "expr": "!!window.state"},
        {"name": "input_moves_player", "expr": expr},
    ]
    result = GameAgent._classify_probes_dynamic(probes)
    assert result["dynamic"] == ["input_moves_player"]
    assert result["ratio"] > 0.0


def test_delta_marker_identifiers_alone_are_not_dynamic():
    """The previous classifier flagged `lastFireTime > 0` and `prev`
    as dynamic on identifier-tokens alone. That's a false positive —
    `lastFireTime > 0` is just "the gun has been fired at init"; an
    `prev` reference without an explicit time-delta (await/setTimeout)
    isn't doing anything. Dynamic checks require an explicit time
    signal."""
    assert GameAgent._is_dynamic_probe("lastFireTime > 0") is False
    assert GameAgent._is_dynamic_probe("prev !== state.x") is False
    # Paired with a timer though, it's genuinely dynamic.
    assert GameAgent._is_dynamic_probe(
        "(async()=>{const p=state.x; await new Promise(r=>setTimeout(r,500)); return p !== state.x;})()"
    ) is True


def test_empty_expr_is_not_dynamic():
    assert GameAgent._is_dynamic_probe("") is False
    assert GameAgent._is_dynamic_probe(None or "") is False


# ---------------------------------------------------------------------------
# Tier 2.4: probe-bait lint (MK trace 20260517_220025).
# ---------------------------------------------------------------------------


def test_probe_bait_flag_caught_when_patch_sets_same_property():
    """MK trace iter 2: the probe checks state.cpuPunchFlipped===true
    and the same iter's patch assigns `cpuPunchFlipped: true` inside
    reset(). The bait lint must flag this as probe_bait_flag."""
    probes = [{
        "name": "cpu_punch_flipped",
        "expr": (
            "(()=>{try{const s=window.state;if(!s)return false;"
            "return s.cpuPunchFlipped===true;}catch(e){return false;}})()"
        ),
    }]
    replace_text = (
        "    state = {\n"
        "      player: createFighter(200, 1),\n"
        "      cpu: createFighter(600, -1),\n"
        "      round: 1,\n"
        "      cpuPunchFlipped: true\n"
        "    };\n"
    )
    findings = GameAgent._probes_baited_by_patches(probes, [replace_text])
    assert len(findings) == 1
    assert findings[0]["kind"] == "probe_bait_flag"
    assert "cpuPunchFlipped" in findings[0]["message"]


def test_probe_bait_flag_not_flagged_when_assignment_is_dynamic():
    """If the patch assigns the property to a non-literal expression
    (`x = computeFlip()`), the bait lint must NOT flag — that's a
    legitimate value check."""
    probes = [{
        "name": "cpu_flipped",
        "expr": "state.cpuFlipped === true",
    }]
    # Dynamic assignment, not literal true/false.
    replace_text = "state.cpuFlipped = computeFlipState(cpu);"
    findings = GameAgent._probes_baited_by_patches(probes, [replace_text])
    assert findings == []


def test_probe_bait_flag_not_flagged_when_property_unrelated():
    """If the patch assigns a different property than the probe
    checks, no bait flag."""
    probes = [{
        "name": "frames_advancing",
        "expr": "window.state.frame > 30",
    }]
    replace_text = "state.cpuPunchFlipped: true,"
    findings = GameAgent._probes_baited_by_patches(probes, [replace_text])
    assert findings == []


def test_probe_bait_flag_handles_window_assignment_form():
    """window.X = true (instead of state.X) also counts as bait."""
    probes = [{
        "name": "flag_check",
        "expr": "window.someFlag === true",
    }]
    replace_text = "window.someFlag = true;"
    findings = GameAgent._probes_baited_by_patches(probes, [replace_text])
    assert len(findings) == 1
    assert findings[0]["kind"] == "probe_bait_flag"


def test_probe_bait_flag_empty_inputs():
    assert GameAgent._probes_baited_by_patches([], ["x = true"]) == []
    assert GameAgent._probes_baited_by_patches(
        [{"name": "p", "expr": "state.x === true"}], [],
    ) == []


# ---------------------------------------------------------------------------
# Probe eval-error recovery (Claude MK trace 20260518_150416).
# ---------------------------------------------------------------------------


def _eval_error_report() -> dict:
    # Non-syntax transient eval error — this kind (TypeError from a
    # property read against a not-yet-initialised state field) is
    # exactly the case the 2-strike policy was designed for. It often
    # clears on the next iter once the game is fully loaded. The
    # one-strike fast-path for syntax errors (added 2026-05-23) is
    # tested by `test_syntax_error_probe_quarantined_immediately` below.
    return {
        "ok": False,
        "errors": [],
        "warnings": [],
        "soft_warnings": [
            "PROBE BROKEN [move_right]: bad",
        ],
        "page_errors": [],
        "console_errors": [],
        "canvas": {"blank": False},
        "input_test": {"ran": True, "any_change": True},
        "probes": [
            {
                "name": "move_right",
                "expr": "state.player.x > 0",
                "ok": False,
                "err": "TypeError: Cannot read properties of undefined (reading 'x')",
                "kind": "eval_error",
                "error_class": "type_error",
            },
            {"name": "frame_alive", "expr": "state.frame > 1", "ok": True, "err": ""},
        ],
        "probe_errors": ["move_right: TypeError: Cannot read properties of undefined (reading 'x')"],
        "probe_eval_errors": [
            {
                "name": "move_right",
                "expr_preview": "state.player.x > 0",
                "error_class": "type_error",
                "err": "TypeError: Cannot read properties of undefined (reading 'x')",
            },
        ],
    }


def _syntax_error_report() -> dict:
    # Unparseable JS expression — the probe will NEVER evaluate
    # differently on re-run. Quarantine immediately to avoid masking
    # other harness signals across iters (the 2026-05-23 chess trace
    # had iters 2 + 3 chasing this same syntax error).
    return {
        "ok": False,
        "errors": [],
        "warnings": [],
        "soft_warnings": [
            "PROBE BROKEN [stress_20_moves_layout_holds]: bad",
        ],
        "page_errors": [],
        "console_errors": [],
        "canvas": {"blank": False},
        "input_test": {"ran": True, "any_change": True},
        "probes": [
            {
                "name": "stress_20_moves_layout_holds",
                "expr": "new Promise(r=>{window.game.reset();let i=0;",  # unterminated
                "ok": False,
                "err": "SyntaxError: Unexpected token ')'",
                "kind": "eval_error",
                "error_class": "syntax_error",
            },
            {"name": "frame_alive", "expr": "state.frame > 1", "ok": True, "err": ""},
        ],
        "probe_errors": ["stress_20_moves_layout_holds: SyntaxError: Unexpected token ')'"],
        "probe_eval_errors": [
            {
                "name": "stress_20_moves_layout_holds",
                "expr_preview": "new Promise(r=>{window.game.reset();let i=0;",
                "error_class": "syntax_error",
                "err": "SyntaxError: Unexpected token ')'",
            },
        ],
    }


def test_probe_eval_error_first_iter_remains_blocking(tmp_path: Path):
    """One eval error can be transient; first occurrence stays blocking."""
    a = _make_agent(tmp_path)
    a._probes = list(_eval_error_report()["probes"])
    report = _eval_error_report()
    a._handle_probe_eval_errors(report, iteration=1)
    assert report["ok"] is False
    assert any(p["name"] == "move_right" for p in report["probes"])
    assert a._probe_eval_error_streak["move_right"] == 1


def test_probe_quarantine_after_second_eval_error(tmp_path: Path):
    """Second consecutive eval error quarantines the probe and removes
    it from current + future probe sets."""
    a = _make_agent(tmp_path)
    a._probes = list(_eval_error_report()["probes"])
    first = _eval_error_report()
    a._handle_probe_eval_errors(first, iteration=1)
    second = _eval_error_report()
    a._handle_probe_eval_errors(second, iteration=2)
    assert second["ok"] is True
    assert all(p["name"] != "move_right" for p in second["probes"])
    assert all(p["name"] != "move_right" for p in a._probes)
    assert a._pending_probe_quarantine_notices
    assert "move_right" in a._pending_probe_quarantine_notices[0]


def test_probe_eval_error_shape_can_soften_before_quarantine(tmp_path: Path):
    """C1: if the same shape already errored once, a healthy page can
    soften an eval-error-only failure."""
    a = _make_agent(tmp_path)
    report = _eval_error_report()
    shape = a._probe_shape_key(report["probes"][0])
    a._probe_eval_error_shape_streak[shape] = 2
    a._handle_probe_eval_errors(report, iteration=3)
    assert report["ok"] is True
    assert report["probes"][0]["ok"] is True
    assert "downgraded" in report["probes"][0]


def test_syntax_error_probe_quarantined_immediately(tmp_path: Path):
    """One-strike fast-path: a probe with a JavaScript SyntaxError
    cannot evaluate differently on a re-run (the expression itself is
    unparseable). Quarantine on the FIRST occurrence rather than the
    second so the broken probe doesn't mask other harness signals.

    The 2026-05-23 chess trace had iters 2 and 3 both reporting
    `PROBE BROKEN [stress_20_moves_layout_holds]: SyntaxError:
    Unexpected token ')'`; the real bug (assets loaded but never
    drawn) was hidden behind the syntax error for two iters. With
    the fast-path, iter 1 already quarantines the broken probe
    and iter 2 sees the real signal.
    """
    a = _make_agent(tmp_path)
    a._probes = list(_syntax_error_report()["probes"])
    report = _syntax_error_report()
    a._handle_probe_eval_errors(report, iteration=1)
    # Quarantined on iter 1 — no second iter required.
    assert all(
        p["name"] != "stress_20_moves_layout_holds"
        for p in report["probes"]
    )
    assert all(
        p["name"] != "stress_20_moves_layout_holds"
        for p in a._probes
    )
    # Notice carries the fast-path-specific wording so the model
    # knows it needs to re-emit the probe with valid JS.
    assert a._pending_probe_quarantine_notices
    notice = a._pending_probe_quarantine_notices[0]
    assert "QUARANTINED IMMEDIATELY" in notice
    assert "syntax error" in notice.lower()


def test_non_syntax_first_eval_error_stays_two_strike(tmp_path: Path):
    """The one-strike fast-path is SCOPED to syntax errors. A runtime
    TypeError (state.player.x against an undefined state — common
    transient on iter 1 before the game finishes initialising) still
    gets the tolerant two-strike treatment so the harness doesn't
    over-quarantine probes that would clear on their own."""
    a = _make_agent(tmp_path)
    a._probes = list(_eval_error_report()["probes"])
    report = _eval_error_report()
    a._handle_probe_eval_errors(report, iteration=1)
    # NOT quarantined on iter 1 — the probe stays in the active list.
    assert any(p["name"] == "move_right" for p in report["probes"])
    assert any(p["name"] == "move_right" for p in a._probes)
    # Streak counter ticks but no notice yet.
    assert a._probe_eval_error_streak.get("move_right") == 1
    assert not a._pending_probe_quarantine_notices


def test_signature_focus_identifiers_filters_browser_internals():
    sig = (
        "Page.evaluate at UtilityScript.evaluate and "
        "UtilityScript.anonymous while state.player.x failed"
    )
    ids = GameAgent._signature_focus_identifiers(sig)
    assert "state.player.x" in ids
    assert "Page.evaluate" not in ids
    assert "UtilityScript.evaluate" not in ids


def test_repeated_feedback_detector(tmp_path: Path):
    a = _make_agent(tmp_path)
    a._recent_feedback_texts = [
        "make the characters face each other, punches face opponent",
    ]
    found = a._detect_repeated_feedback(
        "the characters still need to face each other and punch toward opponent"
    )
    assert found is not None
    assert found["overlap"] >= 0.5


def test_repeated_feedback_detector_ignores_unrelated(tmp_path: Path):
    a = _make_agent(tmp_path)
    a._recent_feedback_texts = ["make the characters face each other"]
    assert a._detect_repeated_feedback("add a fireball command") is None


# ---------------------------------------------------------------------------
# Bulk classification
# ---------------------------------------------------------------------------


def test_classify_all_structural_returns_zero_ratio():
    probes = [
        {"name": "p1", "expr": "!!window.state"},
        {"name": "p2", "expr": "typeof state.player === 'object'"},
        {"name": "p3", "expr": "Array.isArray(state.barrels)"},
    ]
    result = GameAgent._classify_probes_dynamic(probes)
    assert result["ratio"] == 0.0
    assert result["dynamic"] == []
    assert sorted(result["structural"]) == ["p1", "p2", "p3"]


def test_classify_mixed_returns_partial_ratio():
    probes = [
        {"name": "exists", "expr": "!!state.player"},
        {"name": "frames",
         "expr": "performance.now() > 1000 && state.frame > 30"},
    ]
    result = GameAgent._classify_probes_dynamic(probes)
    assert result["ratio"] == 0.5
    assert result["dynamic"] == ["frames"]
    assert result["structural"] == ["exists"]


def test_classify_empty_returns_zero_ratio():
    result = GameAgent._classify_probes_dynamic([])
    assert result["ratio"] == 0.0
    assert result["dynamic"] == []
    assert result["structural"] == []


# ---------------------------------------------------------------------------
# The DK-trace probes — pinned regression
# ---------------------------------------------------------------------------


def test_dk_trace_probes_classify_all_structural():
    """The actual <probes> from the failing DK session — should all
    classify as structural so the nudge fires."""
    dk_probes = [
        {"name": "state_exists", "expr": "!!window.state && typeof window.state === 'object'"},
        {"name": "player_exists", "expr": "!!state && !!state.player"},
        {"name": "barrels_array", "expr": "Array.isArray(state.barrels)"},
        {"name": "princess_exists", "expr": "!!state && !!state.princess"},
        {"name": "hud_visible", "expr": "document.getElementById('hud') !== null"},
    ]
    result = GameAgent._classify_probes_dynamic(dk_probes)
    assert result["ratio"] == 0.0, (
        "DK trace probes should classify as all-structural — that's "
        "the failure pattern the nudge exists to catch."
    )


def test_dk_trace_20260514_probes_classify_all_structural():
    """The literal probe set from
    games/traces/donkey-kong-game-animated-donk_20260514_104131
    Turn 02. Under the OLD classifier this came out at 60% dynamic
    (length > 0 / textContent.length > 0 / toDataURL().length > 200
    all matched the over-broad numeric-comparison rule) so the nudge
    stayed silent on a game that shipped with broken keyboard input.
    Pin the new verdict: all five are structural."""
    probes = [
        {"name": "canvas_exists",
         "expr": "!!document.querySelector('canvas')"},
        {"name": "player_state",
         "expr": "window.state && typeof window.state.player === "
                 "'object' && typeof window.state.player.x === "
                 "'number' && typeof window.state.player.y === "
                 "'number'"},
        {"name": "barrels_active",
         "expr": "window.state && Array.isArray(window.state.barrels) "
                 "&& window.state.barrels.length > 0"},
        {"name": "score_visible",
         "expr": "document.getElementById('score') && "
                 "document.getElementById('score').textContent.length "
                 "> 0"},
        {"name": "game_not_blank",
         "expr": "(function(){var c=document.querySelector('canvas');"
                 "if(!c||!c.width||!c.height)return false;try{return "
                 "c.toDataURL().length>200;}catch(e){return true;}})()"},
    ]
    result = GameAgent._classify_probes_dynamic(probes)
    assert result["ratio"] == 0.0, (
        f"DK 20260514 probes should classify as all-structural "
        f"but got ratio={result['ratio']}: dynamic={result['dynamic']}"
    )


# ---------------------------------------------------------------------------
# Partial-quarantine gate (serial10 chess game 5) + strong-model regression
# baseline (serial10 games 2 Asteroids / 7 Mario shipped clean).
# ---------------------------------------------------------------------------


def _clean_recipe_report() -> dict:
    """A clean, recipe-matched game (mirrors serial10 games 2/7): every probe
    passes, no eval/syntax errors, healthy page — the strong-model case the
    partial-quarantine gate must NEVER flip to ok=False."""
    return {
        "ok": True,
        "errors": [],
        "soft_warnings": [],
        "page_errors": [],
        "console_errors": [],
        "canvas": {"blank": False},
        "input_test": {"ran": True, "any_change": True},
        "probes": [
            {"name": "ship_moves", "expr": "state.ship.x !== 400", "ok": True, "err": ""},
            {"name": "frame_alive", "expr": "state.frame > 1", "ok": True, "err": ""},
        ],
    }


def test_clean_recipe_matched_game_stays_ok_no_partial_gate(tmp_path: Path):
    """Strong-model regression baseline: a recipe-matched game where every
    probe PASSES (serial10 Asteroids/Mario) must stay ok=True. The
    partial-quarantine gate only fires on a SYNTAX quarantine, so a clean
    build can never be over-gated by it."""
    a = _make_agent(tmp_path)
    a._active_visual_playtest_recipe_id = "canvas-top-down-action"  # recipe matched
    report = _clean_recipe_report()
    a._probes = list(report["probes"])
    a._handle_probe_eval_errors(report, iteration=2)
    assert report["ok"] is True
    assert a._partial_quarantine_gate_used == 0
    assert not report.get("soft_warnings")


def test_partial_quarantine_gate_blocks_clean_ship_on_recipe_match(tmp_path: Path):
    """serial10 chess game 5: a behavioral probe syntax-quarantined while
    OTHER probes survive used to ship clean (5/6 + <confirm_done/>). On a
    recipe-matched game the partial-quarantine gate now holds ok=False so the
    dead behavioral gate cannot be masked."""
    a = _make_agent(tmp_path)
    a._active_visual_playtest_recipe_id = "canvas-board-game"
    a._probes = list(_syntax_error_report()["probes"])
    report = _syntax_error_report()
    a._handle_probe_eval_errors(report, iteration=1)
    assert report["ok"] is False
    assert a._partial_quarantine_gate_used == 1
    assert any(
        "malformed" in str(sw).lower()
        for sw in (report.get("soft_warnings") or [])
    )


def test_partial_quarantine_gate_skipped_without_recipe_match(tmp_path: Path):
    """Conservative scoping: the gate fires ONLY on recipe-matched games. A
    novel/unmatched game with a syntax-quarantined probe is NOT gated by it
    (it ships on the harness's own checks), so the gate can never over-gate
    an out-of-genre build from a strong model."""
    a = _make_agent(tmp_path)
    # _active_visual_playtest_recipe_id stays None (no recipe matched)
    a._probes = list(_syntax_error_report()["probes"])
    report = _syntax_error_report()
    a._handle_probe_eval_errors(report, iteration=1)
    assert a._partial_quarantine_gate_used == 0


def test_partial_quarantine_gate_is_bounded_by_cap(tmp_path: Path):
    """The gate is bounded: after _PARTIAL_QUARANTINE_GATE_CAP iters it stops
    blocking so a model that cannot author a parseable replacement does not
    loop forever (mirrors the all-probes-quarantined gate's anti-stuck
    contract)."""
    a = _make_agent(tmp_path)
    a._active_visual_playtest_recipe_id = "canvas-board-game"
    for it in range(1, a._PARTIAL_QUARANTINE_GATE_CAP + 3):
        a._probes = list(_syntax_error_report()["probes"])
        report = _syntax_error_report()
        a._handle_probe_eval_errors(report, iteration=it)
    assert a._partial_quarantine_gate_used == a._PARTIAL_QUARANTINE_GATE_CAP
