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

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


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
