"""Regression tests for the gameplay-state-global mismatch.

The bug: tools.py's _GAMESTATE_SNAPSHOT_JS was sampling window.gameState
while every system-prompt example, every won-skeleton on disk, and
every agent-generated game uses window.state. This made the input
smoke test silently degenerate to canvas-hash only, which fails on
any auto-animating game. Sessions shipped with broken controls.

Fix (2026-05-16): walk a small candidate list (`state`, `gameState`,
`game`, `GAME`, `world`) and take the first object. `state` is the
documented convention; the others are back-compat.

These tests assert:

  1. The JS source string lists `state` BEFORE `gameState`.
  2. The input_test return shape includes a `summary` and
     `responsive_evidence`.
  3. The format_report_for_model line consumes `summary`.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools  # noqa: E402


def _smoke_test_src() -> str:
    """The full source of the _input_smoke_test method, for grepping.

    2026-06-10: _GAMESTATE_SNAPSHOT_JS was hoisted to module level so the
    state-timeline sampler (capability-round item 4) shares the exact same
    flattening. The smoke test still evaluates that constant, so include
    it here — the candidate-ordering assertions below now grep the
    module-level JS plus the method body."""
    return tools._GAMESTATE_SNAPSHOT_JS + inspect.getsource(
        tools.LiveBrowser._input_smoke_test
    )


def test_snapshot_walks_window_state_first():
    """The candidate list must put `state` first so the documented
    convention wins. Catches the original bug if it ever regresses."""
    src = _smoke_test_src()
    # `state` must appear in the candidate list BEFORE `gameState`
    # for the first-match-wins loop to pick the right global.
    idx_state = src.find("'state'")
    idx_gs = src.find("'gameState'")
    assert idx_state > -1, "candidate list must include 'state'"
    assert idx_gs > -1, "candidate list must keep 'gameState' for back-compat"
    assert idx_state < idx_gs, (
        f"'state' must come before 'gameState' in the candidates list; "
        f"got state@{idx_state}, gameState@{idx_gs}"
    )


def test_snapshot_includes_back_compat_globals():
    """`game` / `GAME` / `world` are kept as fallbacks for unusual
    naming choices the model might make. Loose match so we don't
    over-constrain the exact ordering."""
    src = _smoke_test_src()
    for name in ("'state'", "'gameState'", "'game'", "'world'"):
        assert name in src, f"snapshot candidate list missing {name}"


def test_input_test_return_has_summary_field():
    """The new `summary` field is the model-facing one-liner that
    names exactly which keys moved which fields. The format_report
    function reads this directly — no fallback to a synthesized
    message anymore."""
    src = _smoke_test_src()
    # The return dict must include `summary` and `responsive_evidence`.
    assert '"summary": summary' in src
    assert '"responsive_evidence": responsive_evidence' in src


def test_format_report_uses_summary():
    """format_report_for_model reads input_test['summary']. If someone
    refactors the test return shape and drops `summary`, the report
    silently goes back to vague messages — guard against that."""
    fmt_src = inspect.getsource(tools.format_report_for_model)
    assert 'it.get("summary")' in fmt_src or "it.get('summary')" in fmt_src


def test_input_evidence_rejects_per_entity_animation_noise():
    """Per-entity array deltas are weak proof of direct key response.

    A held key should not get credit merely because an autonomous object
    or visual effect changed while the key was down.
    """
    assert tools._input_evidence_is_plausible("objects.3.x") is False
    assert tools._input_evidence_is_plausible("effects.0.age") is False


def test_input_evidence_accepts_direct_state_and_collection_length():
    assert tools._input_evidence_is_plausible("player.x") is True
    assert tools._input_evidence_is_plausible("camera.zoom") is True
    assert tools._input_evidence_is_plausible("things.length") is True


def test_prompt_and_smoke_test_agree_on_window_state():
    """End-to-end sanity: the prompt teaches `window.state` and the
    smoke test now samples `window.state`. If either side drifts, the
    gameplay verification path breaks again."""
    import prompts_v1
    prompt_src = inspect.getsource(prompts_v1)
    assert "window.state" in prompt_src
    smoke = _smoke_test_src()
    assert "'state'" in smoke
