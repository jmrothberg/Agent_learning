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
                "expr": "(()=>{const x0=state.player.x;return true;});",
                "ok": False,
                "err": "SyntaxError: missing ) after argument list",
                "kind": "eval_error",
                "error_class": "syntax_error",
            },
            {"name": "frame_alive", "expr": "state.frame > 1", "ok": True, "err": ""},
        ],
        "probe_errors": ["move_right: SyntaxError: missing ) after argument list"],
        "probe_eval_errors": [
            {
                "name": "move_right",
                "expr_preview": "(()=>{const x0=state.player.x;return true;});",
                "error_class": "syntax_error",
                "err": "SyntaxError: missing ) after argument list",
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
