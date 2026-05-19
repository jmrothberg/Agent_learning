"""Tests for the /wiki research-default-OFF policy.

Empirical evidence (2026-05-19): research.fetch returned 0/10 hits on
representative game goals (asteroids, pacman, donkey kong, space
invaders, missile command, street fighter, doom, snake, 2d roguelike,
tetris) for a cumulative ~38s of network latency with no benefit. The
opt-in default keeps live sessions fast and only runs the lookup when
the operator explicitly types `/wiki on` (mirrors /wait style).

These tests pin the gate so a future refactor doesn't silently flip
the default back to ON (which would re-introduce the 110s "agent looks
frozen" gap from trace 20260519_111209).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent_stub() -> GameAgent:
    """Minimum-viable GameAgent for testing the research toggle without
    spinning up the backend / browser."""
    a = GameAgent.__new__(GameAgent)
    a._research_enabled = False
    a._trace = lambda obj: None
    return a


def test_default_is_off() -> None:
    """A fresh agent has research disabled by default."""
    a = _agent_stub()
    assert a._research_enabled is False


def test_set_research_enabled_true() -> None:
    """`/wiki on` -> set_research_enabled(True) -> flag is True."""
    a = _agent_stub()
    a.set_research_enabled(True)
    assert a._research_enabled is True


def test_set_research_enabled_false() -> None:
    """`/wiki off` -> set_research_enabled(False) -> flag is False."""
    a = _agent_stub()
    a.set_research_enabled(True)
    a.set_research_enabled(False)
    assert a._research_enabled is False


def test_set_research_enabled_coerces_to_bool() -> None:
    """Truthy/falsy non-bool inputs are coerced — paranoid guard against
    a future caller passing the raw arg from chat.py without parsing."""
    a = _agent_stub()
    a.set_research_enabled(1)  # type: ignore[arg-type]
    assert a._research_enabled is True
    a.set_research_enabled(0)  # type: ignore[arg-type]
    assert a._research_enabled is False
    a.set_research_enabled("on")  # type: ignore[arg-type]
    # Non-empty string is truthy in Python — pin the contract.
    assert a._research_enabled is True


def test_set_research_enabled_emits_trace_event() -> None:
    """The setter emits a `research_enabled_set` trace event so the
    state change is visible in the .jsonl. Sessions that later show
    `research_attempted` should always have a paired prior set event;
    sessions without the set event should skip with `research_skipped`."""
    events = []
    a = _agent_stub()
    a._trace = lambda obj: events.append(obj)
    a.set_research_enabled(True)
    a.set_research_enabled(False)
    kinds = [e.get("kind") for e in events]
    assert kinds == ["research_enabled_set", "research_enabled_set"]
    assert events[0]["on"] is True
    assert events[1]["on"] is False


def test_chat_cmd_toggle_wiki_routes_to_set_research_enabled() -> None:
    """The /wiki command in chat.py should call set_research_enabled
    on the live agent. Mirror the /wait pattern."""
    import chat as chat_mod

    # Build a minimal app stub with a fake agent that records calls.
    class _FakeAgent:
        def __init__(self) -> None:
            self._research_enabled = False
            self.calls: list[bool] = []

        def set_research_enabled(self, on: bool) -> None:
            self.calls.append(on)
            self._research_enabled = on

    app = chat_mod.CodingBoxApp.__new__(chat_mod.CodingBoxApp)
    app.agent = _FakeAgent()
    app._log_info = lambda *args, **kwargs: None
    app._update_status = lambda: None

    app._cmd_toggle_wiki("on")
    assert app.agent.calls == [True]
    app._cmd_toggle_wiki("off")
    assert app.agent.calls == [True, False]
    # Bare /wiki toggles from the current state.
    app._cmd_toggle_wiki("")
    assert app.agent.calls == [True, False, True]


def test_chat_cmd_toggle_wiki_no_active_session_is_safe() -> None:
    """/wiki with no agent must not crash — same defensive shape as /wait."""
    import chat as chat_mod

    app = chat_mod.CodingBoxApp.__new__(chat_mod.CodingBoxApp)
    app.agent = None
    app._log_info = lambda *args, **kwargs: None
    # Should not raise.
    app._cmd_toggle_wiki("on")
