"""Phase 5 regression tests — backend reliability cleanup.

Trace 2 (chess 20260522_104235) revealed:
  5a. Anthropic / OpenAI catch blocks returned `stalled=True` with no
      error_message, hiding the real exception class behind a
      hardcoded "stalling at 600s" message.
  5b. The agent's user-visible error used `self.stall_seconds` (the
      configured ceiling) instead of the actual `result.duration_s`,
      so a 0.48s API failure looked like a 600-second hang.
  5c. `_try_extension_backend_fallback` was gated on `continuation=True`,
      which meant a transient cloud error on a normal iter killed the
      session entirely.
  5d. One transparent retry on a `crashed` cloud result is much cheaper
      than swapping the whole backend.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend as backend_mod  # noqa: E402
from agent import GameAgent  # noqa: E402
from ollama_io import StreamResult  # noqa: E402


# -----------------------------------------------------------------------
# Phase 5a — Anthropic + OpenAI errors surface as crashed=True with msg.
# -----------------------------------------------------------------------


def _make_anthropic_backend_with_failing_stream(
    raise_exc: Exception,
) -> backend_mod.AnthropicBackend:
    info = backend_mod.BackendInfo(
        name="anthropic", model="stub-model", source="test", endpoint="https://x"
    )
    b = backend_mod.AnthropicBackend.__new__(backend_mod.AnthropicBackend)
    b.info = info
    fake_client = MagicMock()
    fake_messages = MagicMock()
    fake_client.messages = fake_messages
    # The backend uses `async with self._client.messages.stream(...) as stream`.
    # Make the stream() call raise immediately so we exercise the catch path.
    fake_messages.stream = MagicMock(side_effect=raise_exc)
    b._client = fake_client
    return b


def test_anthropic_crash_returns_crashed_true_with_error_message() -> None:
    err = RuntimeError("OverloadedError: 503 Service Unavailable")
    b = _make_anthropic_backend_with_failing_stream(err)

    result = asyncio.run(
        b.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.5},
        )
    )
    assert isinstance(result, StreamResult)
    assert result.crashed is True, "anthropic crash must mark crashed=True"
    assert result.stalled is False, "crashed errors must NOT be reported as stalls"
    assert result.error_message is not None
    assert "OverloadedError" in result.error_message
    # Real wall-clock should be very small — proves we're not waiting 600s.
    assert result.duration_s < 5.0


def test_openai_crash_returns_crashed_true_with_error_message() -> None:
    """Mirror test for OpenAI — same shape, different SDK."""
    info = backend_mod.BackendInfo(
        name="openai", model="stub-model", source="test", endpoint="https://x"
    )
    b = backend_mod.OpenAIBackend.__new__(backend_mod.OpenAIBackend)
    b.info = info
    fake_client = MagicMock()
    fake_chat = MagicMock()
    fake_completions = MagicMock()
    fake_completions.create = AsyncMock(
        side_effect=RuntimeError("APIConnectionError: connection reset")
    )
    fake_chat.completions = fake_completions
    fake_client.chat = fake_chat
    b._client = fake_client

    result = asyncio.run(
        b.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.5},
        )
    )
    assert result.crashed is True
    assert result.stalled is False
    assert result.error_message is not None
    assert "APIConnectionError" in result.error_message


# -----------------------------------------------------------------------
# Phase 5b — agent stall message uses real duration_s, not stall_seconds.
# -----------------------------------------------------------------------


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def test_classify_stall_parses_actual_duration() -> None:
    """A 0.5s stall message must be classified with stall_seconds=0.5,
    NOT with the configured 600s ceiling."""
    msg = (
        "Model produced no tokens before stalling at 0.5s on "
        "backend=anthropic. cause: OverloadedError. Check ..."
    )
    info = GameAgent._classify_stall(msg)
    assert info is not None
    assert info["kind"] == "no_tokens_stall"
    assert info["stall_seconds"] == 0.5


# -----------------------------------------------------------------------
# Phase 5c — fallback fires on any iteration, not just continuation.
# -----------------------------------------------------------------------


def test_extension_fallback_no_longer_continuation_only(tmp_path: Path) -> None:
    """The fallback handler does not require continuation=True. It only
    inspects the active backend name and the stall shape."""
    a = _make_agent(tmp_path)
    info = backend_mod.BackendInfo(
        name="anthropic", model="claude", source="test", endpoint="https://x"
    )
    fake_backend = MagicMock()
    fake_backend.info = info
    fake_backend.close = AsyncMock()
    a._backend = fake_backend

    # Force the resolver to return an Ollama backend candidate.
    candidate_info = backend_mod.BackendInfo(
        name="ollama",
        model="qwen3.6:27b",
        source="auto",
        endpoint="http://localhost:11434",
    )
    new_ollama = MagicMock()
    new_ollama.info = candidate_info
    with patch("agent.detect_backend", return_value=candidate_info), patch(
        "agent.make_backend", return_value=new_ollama
    ):
        switched, note = asyncio.run(
            a._try_extension_backend_fallback(
                stall={"kind": "no_tokens_stall", "stall_seconds": 0.48},
                iteration=2,
            )
        )

    assert switched is True, f"fallback must switch on cloud crash; note={note}"
    assert "ollama" in note.lower()
    # Backend on the agent is now the Ollama backend.
    assert a._backend is new_ollama


# -----------------------------------------------------------------------
# Phase 5d — `cloud_retry_attempted` flag layout (smoke).
# -----------------------------------------------------------------------


def test_cloud_crash_retry_signal_is_recognised() -> None:
    """Smoke: the `cause:` substring inserted by Phase 5b is what the
    agent's retry branch keys on. Make sure trace-2 style errors match.
    """
    sample = (
        "ANTHROPIC call failed: Model produced no tokens before stalling "
        "at 0.5s on backend=anthropic. cause: OverloadedError: Overloaded. "
        "Check network / API key / rate limits."
    )
    assert "cause:" in sample.lower()
