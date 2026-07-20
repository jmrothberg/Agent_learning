"""Tests for `/ask` read-only Q&A (TUI slash command)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import AgentEvent, GameAgent  # noqa: E402
from chat import CodingBoxApp  # noqa: E402
from prompts_v1 import ASK_ADVICE_SYSTEM, ask_advice_instruction, ask_instruction  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    agent._goal = "build a point-and-click adventure"
    agent._criteria = "- help overlay lists puzzle steps"
    agent._messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "plan turn"},
        {"role": "assistant", "content": "<plan>done</plan>"},
    ]
    agent._snapshot_n = 2
    # Length must clear run_ask_turn's has_game threshold (>=200 chars)
    # so game-mode asks stay grounded in best.html, not advice mode.
    agent.best_path.write_text(
        "<html><body><canvas id='c'></canvas>"
        "<script>function digAtSkullBeach(){ /* puzzle helper */ }"
        "function helpOverlay(){ return 'dig at skull beach'; }"
        "// padding so excerpt/game detection treats this as a real build"
        "</script></body></html>",
        encoding="utf-8",
    )
    return agent


async def _collect_ask(agent: GameAgent, question: str) -> list[AgentEvent]:
    out: list[AgentEvent] = []
    async for ev in agent.run_ask_turn(question):
        out.append(ev)
    return out


def test_ask_instruction_includes_question_and_context() -> None:
    body = ask_instruction(
        question="how does digging work?",
        goal="adventure game",
        criteria="- dig at beach",
        report_text="ok=True",
        html_excerpt="<script>dig()</script>",
        asset_names=["beach_bg", "shovel"],
    )
    assert "how does digging work?" in body
    assert "adventure game" in body
    assert "beach_bg" in body
    assert "READ-ONLY ASK TURN" in body
    assert "do NOT emit <patch>" in body


def test_ask_advice_instruction_is_freeform() -> None:
    body = ask_advice_instruction(
        question="which 80s game benefits from rich art?",
        goal="",
    )
    assert "ADVISORY ASK TURN" in body
    assert "which 80s game benefits from rich art?" in body
    assert "freeform advice" in body
    assert "do NOT emit <patch>" in body
    assert "game-design" in ASK_ADVICE_SYSTEM.lower() or "advisor" in ASK_ADVICE_SYSTEM.lower()


def test_sanitize_ask_reply_strips_code_tags() -> None:
    raw = (
        "Here is the logic.\n"
        "<patch>SEARCH\nx\nREPLACE\ny\n</patch>\n"
        "<done/>"
    )
    clean, stripped = GameAgent._sanitize_ask_reply(raw)
    assert stripped is True
    assert "<patch>" not in clean
    assert "<done" not in clean
    assert "Here is the logic." in clean


def test_run_ask_turn_preserves_messages_and_snapshot(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    before_messages = list(agent._messages)
    before_snapshot = agent._snapshot_n
    before_best = agent.best_path.read_text(encoding="utf-8")

    async def fake_stream(on_token, **kwargs):
        return "Digging calls digAtSkullBeach() when the shovel is selected."

    agent._stream = fake_stream  # type: ignore[method-assign]
    agent.set_token_callback(lambda _p: None)

    events = asyncio.run(_collect_ask(agent, "how do you dig at skull beach?"))

    assert agent._messages == before_messages
    assert agent._snapshot_n == before_snapshot
    assert agent._pending_feedback == []
    assert agent.best_path.read_text(encoding="utf-8") == before_best
    assert any(ev.kind == "info" and ev.data.get("ask") for ev in events)
    assert any("digAtSkullBeach" in (ev.text or "") for ev in events)


def test_run_ask_turn_strips_patch_tags_without_applying(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    before_best = agent.best_path.read_text(encoding="utf-8")

    async def fake_stream(on_token, **kwargs):
        return (
            "It works like this.\n"
            "<patch>SEARCH\ndigAtSkullBeach\nREPLACE\nbroken\n</patch>"
        )

    agent._stream = fake_stream  # type: ignore[method-assign]
    agent.set_token_callback(lambda _p: None)

    events = asyncio.run(_collect_ask(agent, "how does dig work?"))

    assert agent.best_path.read_text(encoding="utf-8") == before_best
    info = next(ev for ev in events if ev.kind == "info")
    assert "<patch>" not in (info.text or "")
    assert info.data.get("tags_stripped") is True


def test_run_ask_turn_advice_mode_without_best_html(tmp_path: Path) -> None:
    """No best.html → freeform advice (pre-/new), not an error."""
    agent = _make_agent(tmp_path)
    agent.best_path.unlink()
    agent._messages = []
    traced: list[dict] = []
    agent._trace = lambda row: traced.append(row)  # type: ignore[method-assign]

    async def fake_stream(on_token, **kwargs):
        return "A side-scroller with parallax art layers."

    agent._stream = fake_stream  # type: ignore[method-assign]
    agent.set_token_callback(lambda _p: None)

    events = asyncio.run(
        _collect_ask(agent, "which genre benefits most from rich art?")
    )

    assert not any(ev.kind == "error" for ev in events)
    assert any(ev.kind == "info" and ev.data.get("ask") for ev in events)
    assert agent._pending_feedback == []
    ctx = next(r for r in traced if r.get("kind") == "user_ask_context")
    assert ctx["ask_mode"] == "advice"
    assert ctx["message_count"] == 2


def test_run_ask_turn_traces_full_reply(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    traced: list[dict] = []
    agent._trace = lambda row: traced.append(row)  # type: ignore[method-assign]

    async def fake_stream(on_token, **kwargs):
        return "Full answer about digAtSkullBeach()."

    agent._stream = fake_stream  # type: ignore[method-assign]
    agent.set_token_callback(lambda _p: None)

    asyncio.run(_collect_ask(agent, "how do you dig?"))

    user_ask = next(r for r in traced if r.get("kind") == "user_ask")
    assert user_ask["reply"] == "Full answer about digAtSkullBeach()."
    assert "digAtSkullBeach" in user_ask["reply_preview"]
    assert user_ask.get("phase") == "ask"


def test_run_ask_turn_uses_lean_context(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    seen_messages: list[list] = []
    traced: list[dict] = []
    agent._trace = lambda row: traced.append(row)  # type: ignore[method-assign]

    async def fake_stream(on_token, **kwargs):
        seen_messages.append(list(agent._messages))
        return "ok"

    agent._stream = fake_stream  # type: ignore[method-assign]
    agent.set_token_callback(lambda _p: None)

    asyncio.run(_collect_ask(agent, "why?"))

    assert len(seen_messages) == 1
    msgs = seen_messages[0]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1].get("phase") == "ask"
    ctx = next(r for r in traced if r.get("kind") == "user_ask_context")
    assert ctx["history_chars_dropped"] > 0
    assert ctx["message_count"] == 2
    assert ctx.get("ask_mode") == "game"


def test_cmd_ask_does_not_queue_feedback() -> None:
    app = CodingBoxApp()
    app._log = MagicMock()  # type: ignore[assignment]
    app._log_info = MagicMock()  # type: ignore[assignment]
    app._log_error = MagicMock()  # type: ignore[assignment]
    app._handle_event = MagicMock()  # type: ignore[assignment]
    app._is_streaming = False
    app._model2_is_streaming = False
    app._model3_is_streaming = False

    agent = MagicMock()
    agent._pending_feedback = []

    async def fake_ask(question: str):
        yield AgentEvent("info", "answer text", {"ask": True})

    agent.run_ask_turn = fake_ask
    app.agent = agent

    asyncio.run(app._run_ask_worker("how does digging work?"))

    agent.add_user_feedback.assert_not_called()
    assert agent._pending_feedback == []
    assert app._ask_in_flight is False


def test_cmd_ask_works_without_session_agent() -> None:
    """Pre-game /ask must not require self.agent (advice before /new)."""
    app = CodingBoxApp()
    app._log_info = MagicMock()  # type: ignore[assignment]
    app._log_error = MagicMock()  # type: ignore[assignment]
    app._is_streaming = False
    app._model2_is_streaming = False
    app._model3_is_streaming = False
    app.agent = None
    ask_agent = MagicMock()
    app._ensure_ask_agent = MagicMock(return_value=ask_agent)  # type: ignore[method-assign]

    def _fake_run_worker(coro, exclusive=False):
        # Production schedules the coroutine; close it so pytest doesn't warn.
        if hasattr(coro, "close"):
            coro.close()

    app.run_worker = MagicMock(side_effect=_fake_run_worker)  # type: ignore[method-assign]

    asyncio.run(app._cmd_ask("which genre benefits from rich art?"))

    app._ensure_ask_agent.assert_called()
    app._log_error.assert_not_called()
    app.run_worker.assert_called_once()
    # Must not claim an active session is required.
    for call in app._log_error.call_args_list:
        assert "active session" not in str(call)


def test_ensure_ask_agent_prefers_session_agent() -> None:
    app = CodingBoxApp()
    session = MagicMock()
    app.agent = session
    app._ask_only_agent = MagicMock()
    assert app._ensure_ask_agent() is session
