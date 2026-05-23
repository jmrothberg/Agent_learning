"""Phase 1.5.1 + 1.5.2 — catch "state has entity but canvas doesn't
render it" and surface autonomous-playtest skip reasons.

The 2026-05-22 Pac-Man trace shipped a game with no Pac-Man visible
because `gameState.pacman.x !== undefined` passed (state has the
entity) but the draw() loop never referenced the sprite. Existing
probes check state, not visible rendering.

These fixes catch that class of bug generally — no genre logic. The
JS check is exercised here as a Python-level smoke (the actual
canvas-pixel sampling requires a real Chromium and is exercised by
the existing tools-side tests that already run Chromium).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from tools import _ENTITY_RENDERED_JS  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 1.5.1 — JS expression contract
# ---------------------------------------------------------------------------


def test_entity_rendered_js_handles_no_state():
    """The expression must return null (not raise) when window.state /
    window.gameState are absent — most pure-DOM apps have neither."""
    # JS contract check: the expression structurally accepts the absence
    # of both globals. Verify by inspecting the JS source string.
    assert "window.state || window.gameState" in _ENTITY_RENDERED_JS
    assert "if (!s || typeof s !== 'object') return null" in _ENTITY_RENDERED_JS


def test_entity_rendered_js_handles_no_canvas():
    """Pure-DOM games (todo lists, calculators) have no canvas. The
    check must return null cleanly so the harness skips it."""
    assert "if (!c || !c.width || !c.height) return null" in _ENTITY_RENDERED_JS


def test_entity_rendered_js_finds_candidates_by_shape_not_name():
    """No genre-specific names (player, pacman, mario, etc.). The
    candidate-finding logic must use STRUCTURAL detection — any
    top-level field whose value is an object with numeric .x/.y."""
    # The candidate filter — verify no hardcoded entity names.
    forbidden = [
        "pacman", "ghost", "mario", "luigi", "ship", "player",
        "hero", "fighter", "asteroid", "alien",
    ]
    for term in forbidden:
        assert term not in _ENTITY_RENDERED_JS.lower(), (
            f"_ENTITY_RENDERED_JS must NOT mention genre name {term!r}"
        )


def test_entity_rendered_js_tries_multiple_position_interpretations():
    """Tile-vs-pixel ambiguity: `pacman.x = 14` could be column 14 or
    pixel 14. The expression tries pixel coordinates AND several common
    arcade tile sizes (28-wide = Pac-Man, 32, 20, 16, 8) and picks the
    interpretation where the position is most likely rendered."""
    assert "kind: 'pixel'" in _ENTITY_RENDERED_JS
    assert "tile${n}" in _ENTITY_RENDERED_JS
    # Verify several tile candidates exist.
    for n in (28, 32, 20, 16, 8):
        assert str(n) in _ENTITY_RENDERED_JS


def test_entity_rendered_js_flag_threshold_is_strict():
    """The check flags entities where >80% of the surrounding patch is
    background. Looser thresholds (e.g. 50%) would over-flag entities
    that are partially transparent or near edges; stricter (e.g. 95%)
    would miss the Pac-Man case where the entity has zero rendering at
    all. 80% is the chosen middle ground."""
    assert "0.80" in _ENTITY_RENDERED_JS


# ---------------------------------------------------------------------------
# Phase 1.5.2 — autonomous-playtest skip tracing
# ---------------------------------------------------------------------------


def _agent_stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._user_force_done = False
    a._use_autonomous_feedback = True
    a._autonomous_playtest_cycle = 0
    a._autonomous_no_findings_streak = 0
    a._pending_feedback = []
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    a.browser = None
    a._record = lambda ev: ev
    return a


def test_autonomous_skip_traces_when_feedback_off():
    a = _agent_stub()
    a._use_autonomous_feedback = False

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=1, report={"ok": True}):
            pass

    asyncio.run(_drive())

    skips = [t for t in a._trace_events if t.get("kind") == "autonomous_playtest_skipped"]
    assert len(skips) == 1
    assert skips[0]["reason"] == "disabled:feedback_off"


def test_autonomous_skip_traces_when_iter_failed():
    a = _agent_stub()

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=2, report={"ok": False}):
            pass

    asyncio.run(_drive())

    skips = [t for t in a._trace_events if t.get("kind") == "autonomous_playtest_skipped"]
    assert len(skips) == 1
    assert skips[0]["reason"] == "iter_failed"


def test_autonomous_skip_traces_when_force_done():
    a = _agent_stub()
    a._user_force_done = True

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=1, report={"ok": True}):
            pass

    asyncio.run(_drive())

    skips = [t for t in a._trace_events if t.get("kind") == "autonomous_playtest_skipped"]
    assert len(skips) == 1
    assert skips[0]["reason"] == "disabled:force_done"


def test_autonomous_skip_traces_when_no_browser():
    a = _agent_stub()
    a.browser = None  # explicit

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=1, report={"ok": True}):
            pass

    asyncio.run(_drive())

    skips = [t for t in a._trace_events if t.get("kind") == "autonomous_playtest_skipped"]
    assert len(skips) == 1
    assert skips[0]["reason"] == "no_browser"


def test_autonomous_skip_includes_iteration_for_correlation():
    a = _agent_stub()
    a._use_autonomous_feedback = False

    async def _drive():
        async for _ in a._run_autonomous_playtest(iteration=7, report={"ok": True}):
            pass

    asyncio.run(_drive())

    skips = [t for t in a._trace_events if t.get("kind") == "autonomous_playtest_skipped"]
    assert skips[0]["iteration"] == 7
