"""Tests for the harness gating-trap fixes from trace 20260612_171752.

Three trace-backed fixes, covered with pure-function / source-pinned tests
(no model, no Chromium):

  1. Probe ordering — read-only probes run before side-effecting ones
     (restart_resets dispatched KeyR → reset() zeroed state.frame right
     before raf_firing read `frame > 0`; the model burned 3 iters on the
     contradiction). Falsy read-only probes retry once; failures that ran
     after a side-effecting probe carry an ordering hint.
  2. ACTION_DRAWN_NOT_SPRITED persistence downgrade — first occurrence
     gates, a repeat on a behaviorally-green build (all probes pass, zero
     errors) demotes to the non-gating warnings channel, mirroring the
     ASSETS_LOADED_BUT_UNDRAWN gate.
  3. best.html cosmetic-only save — a build whose ok=False comes solely
     from cosmetic sprite-family soft_warnings (probes green, no errors)
     still saves best.html (the trace ended best_exists=False on a
     7/7-probes playable game).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402
from agent import GameAgent, _report_green_except_cosmetic_sprites  # noqa: E402
from tools import _probe_has_side_effects  # noqa: E402


# ---------------------------------------------------------------------------
# 1a. Side-effect classifier (pure function)
# ---------------------------------------------------------------------------

def test_dispatching_probe_is_side_effecting():
    expr = (
        "(async()=>{window.dispatchEvent(new KeyboardEvent('keydown',"
        "{code:'KeyR'}));await new Promise(r=>setTimeout(r,300));"
        "return window.gameState.scene===0;})()"
    )
    assert _probe_has_side_effects(expr) is True


def test_reset_and_click_calls_are_side_effecting():
    assert _probe_has_side_effects("window.game.reset() === undefined") is True
    assert _probe_has_side_effects(
        "document.querySelector('button').click() || true"
    ) is True


def test_readonly_probes_are_not_side_effecting():
    for expr in (
        "window.gameState && window.gameState.frame > 0",
        "!!document.querySelector('canvas')",
        "typeof window.gameState.score === 'number'",
        "(()=>{const c=document.querySelector('canvas');"
        "return c.toDataURL().length>500;})()",
    ):
        assert _probe_has_side_effects(expr) is False, expr


def test_await_probe_runs_in_second_group():
    # Pure timing probes still get delayed to the side-effect group —
    # harmless (only ordering changes, never the result).
    assert _probe_has_side_effects(
        "(async()=>{await new Promise(r=>setTimeout(r,100));return true;})()"
    ) is True


# ---------------------------------------------------------------------------
# 1b. Probe loop wiring (source-pinned)
# ---------------------------------------------------------------------------

def test_probe_loop_orders_readonly_first():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "_probe_has_side_effects" in src
    # Read-only partition concatenated before the effectful one.
    i = src.index("ordered = (")
    block = src[i:i + 400]
    assert "if not _probe_has_side_effects" in block
    # Report restores the original probe order.
    assert "results_by_idx[i] for i in sorted(results_by_idx)" in src


def test_probe_loop_retries_falsy_readonly_once():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    i = src.index("if not ok and not err and not is_effectful:")
    block = src[i:i + 400]
    # Retry only after a short delay, and only re-evaluates the same expr.
    assert "asyncio.sleep" in block
    assert "await self._run_probe(pexpr)" in block


def test_probe_failure_carries_ordering_hint():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    i = src.index("ran after side-effecting probe(s)")
    # Hint only on falsy results (not eval errors) after an effectful probe.
    guard = src[src.rfind("if not entry", 0, i):i]
    assert "effectful_run_so_far" in guard
    assert "not err_kind" in guard


def test_consecutive_side_effecting_probes_reset_between():
    """P3 (run_04 holochess iter 1): a side-effecting probe that runs after a
    prior side-effecting probe resets the game first (when reset() exists) so
    a leftover mid-animation state can't block the next probe's clicks."""
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    i = src.index("if is_effectful and effectful_run_so_far:")
    block = src[i:i + 500]
    # Calls game.reset()/restart() and waits a beat before running the probe.
    assert "g.reset||g.restart" in block
    assert "asyncio.sleep" in block
    # The reset happens BEFORE the probe runs this iteration.
    run_i = src.index("ok, err, err_kind = await self._run_probe(pexpr)", i)
    assert run_i > i


# ---------------------------------------------------------------------------
# 2. ACTION_DRAWN_NOT_SPRITED persistence downgrade (source-pinned,
#    mirroring test_undrawn_gates_first_occurrence_then_demotes)
# ---------------------------------------------------------------------------

def test_action_not_sprited_gates_first_then_demotes():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "_action_not_sprited_seen_before" in src
    start = src.index("ACTION_DRAWN_NOT_SPRITED")
    window = src[start:start + 3000]
    # Demotion branch requires green probes AND no errors AND a repeat.
    assert "_probes_green_fa" in window and "_no_errors_fa" in window
    assert "ADVISORY (non-blocking)" in window
    # First occurrence still gates via soft_warnings.
    assert 'report["soft_warnings"].append(_faked_text)' in window


def test_action_not_sprited_streak_resets_when_finding_clears():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "self._action_not_sprited_seen_before = False" in src


def test_action_not_sprited_flag_initialized():
    src = inspect.getsource(tools_module.LiveBrowser.__init__)
    assert "self._action_not_sprited_seen_before = False" in src


# ---------------------------------------------------------------------------
# 3. Cosmetic-only best.html save
# ---------------------------------------------------------------------------

def _cosmetic_green_report() -> dict:
    return {
        "ok": False,
        "errors": [],
        "page_errors": [],
        "console_errors": [],
        "soft_warnings": [
            "ACTION_DRAWN_NOT_SPRITED [Space]: pressing this key changed "
            "the canvas by CODE-DRAWING but did NOT draw a different sprite",
        ],
        "warnings": [],
        "probes": [
            {"name": "p1", "ok": True},
            {"name": "p2", "ok": True},
        ],
    }


def test_cosmetic_only_report_qualifies():
    assert _report_green_except_cosmetic_sprites(_cosmetic_green_report()) is True


def test_all_cosmetic_family_prefixes_qualify():
    for prefix in (
        "ASSETS_LOADED_BUT_UNDRAWN [a, b]: 2/9 asset PNG(s) loaded",
        "ACTION_DRAWN_NOT_SPRITED [Space]: pressing this key",
        "CODE_DRAWN_OVER_SPRITE [KeyJ]: the action DID draw its sprite",
    ):
        r = _cosmetic_green_report()
        r["soft_warnings"] = [prefix]
        assert _report_green_except_cosmetic_sprites(r) is True, prefix


def test_ok_true_report_does_not_double_save():
    r = _cosmetic_green_report()
    r["ok"] = True
    assert _report_green_except_cosmetic_sprites(r) is False


def test_failing_probe_disqualifies():
    r = _cosmetic_green_report()
    r["probes"][1]["ok"] = False
    assert _report_green_except_cosmetic_sprites(r) is False


def test_errors_disqualify():
    for field in ("errors", "page_errors", "console_errors"):
        r = _cosmetic_green_report()
        r[field] = ["TypeError: boom"]
        assert _report_green_except_cosmetic_sprites(r) is False, field


def test_non_cosmetic_soft_warning_disqualifies():
    r = _cosmetic_green_report()
    r["soft_warnings"].append(
        "PROBE FAILED [raf_firing]: `window.gameState.frame > 0` — falsy"
    )
    assert _report_green_except_cosmetic_sprites(r) is False


def test_no_probes_or_no_softs_disqualify():
    r = _cosmetic_green_report()
    r["probes"] = []
    assert _report_green_except_cosmetic_sprites(r) is False
    r = _cosmetic_green_report()
    r["soft_warnings"] = []
    assert _report_green_except_cosmetic_sprites(r) is False


def test_agent_save_site_wires_cosmetic_fallback():
    src = inspect.getsource(GameAgent)
    assert "elif _report_green_except_cosmetic_sprites(report):" in src
    # Degraded save is traced so the run jsonl shows WHY best.html exists
    # despite ok=False.
    i = src.index("elif _report_green_except_cosmetic_sprites(report):")
    block = src[i:i + 1200]
    assert "_save_best(new_html)" in block
    assert "best_saved_cosmetic_only" in block
