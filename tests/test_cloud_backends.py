"""Tests for the OpenAI + Anthropic cloud backends.

Pure-function coverage only — no network calls. Verifies:

  1. /list inventory is GATED on the API-key env var. Without the key,
     entries are hidden; with the key, the curated model list is
     returned. Critical because cloud entries dangling in /list when
     the key isn't set would just produce confusing /load failures.

  2. detect_backend honors LLM_BACKEND=openai / =anthropic and raises a
     clear error when the key is missing.

  3. make_backend dispatches openai/anthropic names to the right class
     (the constructor will refuse to build without a key, so we only
     assert the class identity).

  4. AnthropicBackend's request-time split of system vs messages keeps
     the system content out of the user-role message array. This is
     where the Anthropic API differs from OpenAI/Ollama and where a
     bug would silently corrupt the prompt.

  5. P0a (MK trace 20260528) — Trailing assistant tag-opener prefill is
     FOLDED into the preceding user message as a format hint before the
     API call. Opus 4.7+ models hard-reject `{"role":"assistant",
     "content":"<diagnose>"}` with a 400, regardless of whitespace
     trimming. The previous "rstrip()" sanitizer is replaced by a fold.

  6. P0a safety net — _anthropic_prepare_messages itself folds short
     tag-opener trailing assistant turns even if the caller forgot.

  7. P0b — _ANTHROPIC_NON_RETRYABLE_400_PHRASES lists the exact strings
     the agent retry classifier matches against; verifies the agent
     won't burn a same-payload retry on a known shape error.

  8. P0c — _try_extension_backend_fallback accepts mlx OR ollama as the
     local fallback target (previously: ollama-only, which rejected MLX
     in the MK trace despite MLX being the loaded local backend).

The key text itself is never written into the test fixtures — tests use
"sk-test" sentinels so a CI log dump can't leak anything sensitive.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import backend as backend_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Inventory gating
# ---------------------------------------------------------------------------

def test_openai_inventory_empty_without_key() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        models, default = backend_mod.list_openai_inventory()
        assert models == []
        assert default is None


def test_openai_inventory_populated_with_key() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        models, default = backend_mod.list_openai_inventory()
        assert len(models) >= 1
        assert default == backend_mod._OPENAI_DEFAULT_MODEL
        assert default in models


def test_anthropic_inventory_empty_without_key() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        models, default = backend_mod.list_anthropic_inventory()
        assert models == []
        assert default is None


def test_anthropic_inventory_populated_with_key() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        models, default = backend_mod.list_anthropic_inventory()
        assert len(models) >= 2
        assert default == backend_mod._ANTHROPIC_DEFAULT_MODEL
        assert default in models
        assert "claude-fable-5" in models
        assert "claude-opus-4-8" in models


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------

def test_detect_backend_openai_without_key_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            backend_mod.detect_backend("openai")


def test_detect_backend_openai_with_key() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        info = backend_mod.detect_backend("openai")
        assert info.name == "openai"
        assert info.model == backend_mod._OPENAI_DEFAULT_MODEL
        assert "OPENAI_API_KEY" in info.source
        assert info.endpoint.startswith("https://api.openai.com")


def test_detect_backend_anthropic_without_key_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            backend_mod.detect_backend("anthropic")


def test_detect_backend_anthropic_with_key() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        info = backend_mod.detect_backend("anthropic")
        assert info.name == "anthropic"
        assert info.model == backend_mod._ANTHROPIC_DEFAULT_MODEL
        assert "ANTHROPIC_API_KEY" in info.source
        assert info.endpoint.startswith("https://api.anthropic.com")


def test_detect_backend_claude_alias() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        info = backend_mod.detect_backend("claude")
        assert info.name == "anthropic"


# ---------------------------------------------------------------------------
# make_backend dispatch
# ---------------------------------------------------------------------------

def test_make_backend_dispatches_openai() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        info = backend_mod.BackendInfo(
            name="openai",
            model=backend_mod._OPENAI_DEFAULT_MODEL,
            source="test",
            endpoint=backend_mod.openai_endpoint_url(),
        )
        b = backend_mod.make_backend(info)
        assert isinstance(b, backend_mod.OpenAIBackend)
        assert b.info.model == backend_mod._OPENAI_DEFAULT_MODEL


def test_make_backend_dispatches_anthropic() -> None:
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        info = backend_mod.BackendInfo(
            name="anthropic",
            model=backend_mod._ANTHROPIC_DEFAULT_MODEL,
            source="test",
            endpoint=backend_mod.anthropic_endpoint_url(),
        )
        b = backend_mod.make_backend(info)
        assert isinstance(b, backend_mod.AnthropicBackend)
        assert b.info.model == backend_mod._ANTHROPIC_DEFAULT_MODEL


def test_make_backend_openai_without_key_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        info = backend_mod.BackendInfo(
            name="openai", model="gpt-5", source="test",
            endpoint=backend_mod.openai_endpoint_url(),
        )
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            backend_mod.make_backend(info)


# ---------------------------------------------------------------------------
# Anthropic system/message split
# ---------------------------------------------------------------------------

def test_anthropic_splits_system_message_from_history() -> None:
    """Anthropic's API requires the system prompt as a top-level kwarg.
    The backend must lift it out of the messages array and concatenate
    consecutive system messages."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
        info = backend_mod.BackendInfo(
            name="anthropic",
            model=backend_mod._ANTHROPIC_DEFAULT_MODEL,
            source="test",
            endpoint=backend_mod.anthropic_endpoint_url(),
        )
        b = backend_mod.make_backend(info)

        # Simulate the split logic directly — extracting the same
        # comprehension the backend uses on the input messages.
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "system", "content": "Tools available: foo, bar."},
            {"role": "user", "content": "Build the game."},
            {"role": "assistant", "content": "<plan>...</plan>"},
            {"role": "user", "content": "Continue."},
        ]
        # Mirror backend's split via the shared helper.
        system_text, msgs = backend_mod._anthropic_prepare_messages(messages)

        # The combined system text must include both fragments.
        assert "You are an agent." in system_text
        assert "Tools available: foo, bar." in system_text
        # The messages array passed to Anthropic must contain ONLY
        # user/assistant turns.
        assert all(m["role"] in ("user", "assistant") for m in msgs)
        assert len(msgs) == 3

        # Sanity — the backend wired AsyncAnthropic correctly.
        assert b._client is not None


# ---------------------------------------------------------------------------
# P0a — Tag-opener fold (MK trace 20260528)
# ---------------------------------------------------------------------------

def test_anthropic_prefill_folded_into_user_message() -> None:
    """P0a safety net: a trailing assistant `<diagnose>` opener must be
    folded into the preceding user message as a format hint, because
    Opus 4.7+ rejects ALL assistant-final messages with a 400.
    """
    _, msgs = backend_mod._anthropic_prepare_messages([
        {"role": "user", "content": "Fix the game."},
        {"role": "assistant", "content": "<diagnose>\n"},
    ])
    # No trailing assistant — fold happened.
    assert msgs[-1]["role"] == "user"
    # The user message now carries the format hint with the tag opener.
    assert "<diagnose>" in msgs[-1]["content"]
    assert "Fix the game." in msgs[-1]["content"]
    assert "FORMAT" in msgs[-1]["content"].upper()


def test_anthropic_prefill_fold_handles_html_file_opener() -> None:
    """The first-build rescue prefill is multi-line (`<html_file>\\n
    <!DOCTYPE html>\\n`). Only the first line (the tag) should land in
    the format hint."""
    _, msgs = backend_mod._anthropic_prepare_messages([
        {"role": "user", "content": "Rebuild the game."},
        {"role": "assistant", "content": "<html_file>\n<!DOCTYPE html>\n"},
    ])
    assert msgs[-1]["role"] == "user"
    assert "<html_file>" in msgs[-1]["content"]
    # The DOCTYPE noise should NOT leak into the hint.
    assert "<!DOCTYPE" not in msgs[-1]["content"]


def test_anthropic_does_not_fold_long_assistant_reply() -> None:
    """A real model reply (>200 chars or not starting with `<`) must be
    preserved verbatim — only short tag openers are folded."""
    long_reply = "x" * 400
    _, msgs = backend_mod._anthropic_prepare_messages([
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": long_reply},
    ])
    # Trailing assistant still present (real reply).
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == long_reply


def test_anthropic_stream_chat_folds_prefill_into_user() -> None:
    """End-to-end: stream_chat passes a payload with NO trailing
    assistant message; the fold places the tag opener as a hint inside
    the last user turn."""
    info = backend_mod.BackendInfo(
        name="anthropic",
        model=backend_mod._ANTHROPIC_DEFAULT_MODEL,
        source="test",
        endpoint=backend_mod.anthropic_endpoint_url(),
    )
    b = backend_mod.AnthropicBackend.__new__(backend_mod.AnthropicBackend)
    b.info = info

    captured: dict = {}

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        @property
        def text_stream(self):
            return self._empty()

        async def _empty(self):
            return
            yield  # pragma: no cover — makes this an async generator

        async def get_final_message(self):
            msg = MagicMock()
            msg.usage = None
            msg.stop_reason = "end_turn"
            return msg

    fake_messages = MagicMock()

    def _capture_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    fake_messages.stream = MagicMock(side_effect=_capture_stream)
    b._client = MagicMock()
    b._client.messages = fake_messages

    asyncio.run(
        b.stream_chat(
            messages=[
                {"role": "user", "content": "Continue."},
                {"role": "assistant", "content": "<diagnose>\n"},
            ],
        )
    )

    sent = captured["messages"]
    # MK-trace bug: trailing assistant prefill was sent and rejected
    # with 400. Fold replaces that with a user message carrying the
    # opener hint.
    assert sent[-1]["role"] == "user"
    assert "<diagnose>" in sent[-1]["content"]
    assert "Continue." in sent[-1]["content"]


# ---------------------------------------------------------------------------
# P0b — Non-retryable Anthropic 400 phrases
# ---------------------------------------------------------------------------

def test_anthropic_non_retryable_400_phrases_present() -> None:
    """The agent retry classifier must match the exact substrings
    Anthropic returns for assistant-prefill payload-shape 400s."""
    import agent as agent_mod
    phrases = agent_mod._ANTHROPIC_NON_RETRYABLE_400_PHRASES
    assert isinstance(phrases, tuple)
    assert "does not support assistant message prefill" in phrases
    assert "must end with a user message" in phrases
    # All phrases must be lower-cased so the err_str.lower() match works.
    for p in phrases:
        assert p == p.lower(), f"phrase must be lowercase: {p!r}"


# ---------------------------------------------------------------------------
# P0c — Extension fallback accepts MLX or Ollama
# ---------------------------------------------------------------------------

def test_extension_fallback_accepts_mlx() -> None:
    """When Anthropic stalls and detect_backend resolves MLX, the
    fallback must accept it (MK trace previously rejected MLX with
    `(not ollama)`)."""
    import agent as agent_mod
    src = __import__("inspect").getsource(
        agent_mod.GameAgent._try_extension_backend_fallback
    )
    # The new loop accepts mlx OR ollama, not ollama-only.
    assert 'resolved.name in ("mlx", "ollama")' in src
    # Old failure mode string is gone (so trace mining can grep for the
    # new "no local MLX or Ollama" phrasing instead).
    assert "no local Ollama backend could" not in src


def test_extension_fallback_skips_when_not_cloud() -> None:
    """The fallback should still only trigger from cloud backends, not
    from a local backend stall."""
    import agent as agent_mod
    src = __import__("inspect").getsource(
        agent_mod.GameAgent._try_extension_backend_fallback
    )
    # MLX-source guard preserved (no silent switch on MLX stalls).
    assert "MLX stall — staying on MLX" in src
    # Cloud-source allowlist now covers anthropic AND openai (both
    # benefit from local fallback on transient outage).
    assert 'backend_name not in ("anthropic", "openai")' in src
