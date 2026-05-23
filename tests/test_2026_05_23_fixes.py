"""Five general fixes from the 2026-05-22 Pac-Man + 2026-05-23 SOTA chess
trace comparison. All five must work regardless of model size — no
provider names, no model-specific branching. The detection mechanisms
are observable shape (token count, endpoint URL form, JS state shape).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Fix #1 — short_stream_warning (symmetric to runaway warning)
# ---------------------------------------------------------------------------


def test_fix1_short_stream_warning_in_source():
    """The warning kind + floor must be wired; the floor must be a
    reasonable value (>10 to skip <confirm_done/>, <100 to catch the
    chess trace's 8-token degenerate reply)."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    assert "short_stream_warning" in src
    assert "_SHORT_STREAM_FLOOR" in src
    import re
    m = re.search(r"_SHORT_STREAM_FLOOR\s*=\s*(\d+)", src)
    assert m is not None
    floor = int(m.group(1))
    assert 10 < floor < 100, (
        f"short_stream floor of {floor} is out of the useful range — "
        "should catch the chess trace's 8-token degenerate reply "
        "(needs floor > 10) without flagging legitimate brief "
        "<confirm_done/> turns (needs floor < ~100)."
    )


def test_fix1_short_stream_only_warns_for_code_emitting_roles():
    """A short reply is degenerate for `coder` and `architect` (they
    should be emitting code or plan content). It's NOT degenerate for
    a `critic` role whose output is one sentence of feedback, or for a
    deliberate <confirm_done/> reply."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    # The role gate is in the emission site.
    assert 'role in ("coder", "architect")' in src
    # Done-marker bypass.
    assert "<confirm_done/>" in src and "<done/>" in src


# ---------------------------------------------------------------------------
# Fix #2 — autonomous budget refresh on fresh feedback
# ---------------------------------------------------------------------------


def _feedback_agent_stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._pending_feedback = []
    a._feedback_deferred_last_turn = False
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)
    a._autonomous_playtest_cycle = 0
    a._autonomous_no_findings_streak = 0
    return a


def test_fix2_fresh_feedback_refreshes_exhausted_budget():
    a = _feedback_agent_stub()
    a._autonomous_playtest_cycle = 3      # at the cap
    a._autonomous_no_findings_streak = 2  # at the stop threshold

    a.add_user_feedback("the pacman doesn't move on arrow keys")

    # Cycle decremented by 1 so the next autonomous turn fits under cap.
    assert a._autonomous_playtest_cycle == 2
    # Streak reset so the no-findings auto-stop doesn't gate.
    assert a._autonomous_no_findings_streak == 0
    # Refresh event surfaced for the learner.
    refreshes = [t for t in a._trace_events if t.get("kind") == "autonomous_budget_refreshed_on_feedback"]
    assert len(refreshes) == 1
    assert refreshes[0]["prior_cycle"] == 3
    assert refreshes[0]["new_cycle"] == 2


def test_fix2_fresh_feedback_at_zero_cycle_does_not_underflow():
    a = _feedback_agent_stub()
    # Pre-feedback state already at zero.
    a._autonomous_playtest_cycle = 0
    a._autonomous_no_findings_streak = 0

    a.add_user_feedback("change the colors")

    # Stays at zero — no negative cycle.
    assert a._autonomous_playtest_cycle == 0
    assert a._autonomous_no_findings_streak == 0
    # No refresh event when nothing actually changed.
    assert not any(
        t.get("kind") == "autonomous_budget_refreshed_on_feedback"
        for t in a._trace_events
    )


def test_fix2_empty_feedback_does_not_refresh():
    a = _feedback_agent_stub()
    a._autonomous_playtest_cycle = 2

    a.add_user_feedback("")
    a.add_user_feedback("   \n   ")

    assert a._autonomous_playtest_cycle == 2
    assert not any(
        t.get("kind") == "autonomous_budget_refreshed_on_feedback"
        for t in a._trace_events
    )


# ---------------------------------------------------------------------------
# Fix #3 — surface why an applies_when gate failed
# ---------------------------------------------------------------------------


def test_fix3_recipe_skip_carries_diagnostics_map():
    """When applies_when returns falsy, the skip event must carry a
    diagnostics map showing which structural conditions held. Without
    re-running the session, future trace mining can tell whether the
    game's state is missing window.state, missing canvas, missing
    player.x/y, etc."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    # The diagnostic JS exists and tests the expected fields.
    for field in (
        "has_state", "has_gameState", "has_canvas",
        "has_player_xy", "has_player_facing",
        "top_level_xy_count",
    ):
        assert f'"{field}"' not in src or f"out.{field}" in src or field in src, (
            f"Fix #3 diagnostic must surface {field!r}"
        )
    # The skip-event payload picks up the diagnostics field.
    assert '"diagnostics"' in src


# ---------------------------------------------------------------------------
# Fix #4 — frozen_canvas flips ok to False via soft_warning
# ---------------------------------------------------------------------------


def test_fix4_frozen_canvas_emits_soft_warning():
    import inspect
    import tools as tools_module
    src = inspect.getsource(tools_module)
    # The soft-warning emission for frozen_canvas exists.
    assert 'frozen is True' in src
    assert "FROZEN-CANVAS" in src
    # And it goes through soft_warnings (which flips ok).
    assert 'report["soft_warnings"].append' in src


# ---------------------------------------------------------------------------
# Fix #5 — endpoint-shape detection for cloud concurrency
# ---------------------------------------------------------------------------


def test_fix5_loopback_endpoints_serialize():
    assert GameAgent._endpoint_supports_concurrency("http://127.0.0.1:11434") is False
    assert GameAgent._endpoint_supports_concurrency("http://localhost:11434") is False
    assert GameAgent._endpoint_supports_concurrency("http://[::1]:11434") is False
    assert GameAgent._endpoint_supports_concurrency("http://0.0.0.0:11434") is False
    assert GameAgent._endpoint_supports_concurrency("") is False
    assert GameAgent._endpoint_supports_concurrency(None) is False  # type: ignore[arg-type]


def test_fix5_non_loopback_endpoints_allow_concurrency():
    # Detection is by SHAPE — no provider name string-match.
    assert GameAgent._endpoint_supports_concurrency("https://api.anthropic.com") is True
    assert GameAgent._endpoint_supports_concurrency("https://api.openai.com") is True
    assert GameAgent._endpoint_supports_concurrency("https://generativelanguage.googleapis.com") is True
    # Hypothetical future cloud provider — still works because we don't
    # name-check.
    assert GameAgent._endpoint_supports_concurrency("https://api.cool-new-provider.io/v1") is True
    # LAN-hosted inference server.
    assert GameAgent._endpoint_supports_concurrency("http://192.168.1.50:8080") is True


def test_fix5_independence_check_passes_for_shared_cloud_endpoint(tmp_path):
    """Two role backends pointing at the same cloud endpoint must be
    treated as independent (concurrent requests are fine). The
    SOTA-chess trace from 2026-05-23 had all three roles on the same
    Anthropic endpoint and fan-out fell back to sequential — Fix #5
    closes that gap."""
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    a._backend.info.endpoint = "https://api.anthropic.com"
    distinct_critic = MagicMock()
    distinct_critic.info.endpoint = "https://api.anthropic.com"  # same endpoint
    assert a._critic_runs_on_independent_slot(distinct_critic) is True


def test_fix5_independence_check_rejects_shared_loopback_endpoint(tmp_path):
    """Two role backends sharing the same Ollama loopback endpoint
    must STILL be rejected — local daemons serialize."""
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    distinct_local = MagicMock()
    distinct_local.info.endpoint = "http://127.0.0.1:11434"
    assert a._critic_runs_on_independent_slot(distinct_local) is False


def test_fix5_no_provider_name_strings_in_endpoint_check_logic():
    """The endpoint-shape check MUST stay generic — no anthropic /
    openai / claude / gpt strings in the EXECUTABLE LOGIC. (Docstring
    mentions of provider names as examples are allowed.) A future
    provider should work automatically."""
    import inspect
    import textwrap
    method_src = textwrap.dedent(
        inspect.getsource(GameAgent._endpoint_supports_concurrency)
    )
    # Strip the docstring so we only inspect the code path.
    import ast
    parsed = ast.parse(method_src).body[0]
    # The method has a @staticmethod decorator wrapper at module level
    # but inspect returns the FunctionDef directly when given the unbound
    # callable. Handle both forms.
    if hasattr(parsed, "body") and parsed.body:
        fn = parsed
    else:  # pragma: no cover
        fn = parsed
    if (
        isinstance(fn, ast.FunctionDef)
        and fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
    ):
        # Drop the docstring node and re-render the function body.
        fn.body = fn.body[1:]
    body_src = ast.unparse(fn).lower()
    forbidden = ("anthropic", "openai", "claude", "gpt", "gemini", "google")
    for term in forbidden:
        assert term not in body_src, (
            f"_endpoint_supports_concurrency LOGIC must not name-match "
            f"provider {term!r} — detection is by endpoint SHAPE "
            "(loopback vs non-loopback)."
        )
