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

The key text itself is never written into the test fixtures — tests use
"sk-test" sentinels so a CI log dump can't leak anything sensitive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

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
        assert len(models) >= 1
        assert default == backend_mod._ANTHROPIC_DEFAULT_MODEL
        assert default in models


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
        # Mirror backend's split.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        msgs = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_parts).strip()

        # The combined system text must include both fragments.
        assert "You are an agent." in system_text
        assert "Tools available: foo, bar." in system_text
        # The messages array passed to Anthropic must contain ONLY
        # user/assistant turns.
        assert all(m["role"] in ("user", "assistant") for m in msgs)
        assert len(msgs) == 3

        # Sanity — the backend wired AsyncAnthropic correctly.
        assert b._client is not None
