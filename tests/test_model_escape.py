"""Escape hatch when Ollama cannot load a model blob (/model after session end)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import (  # noqa: E402
    CodingBoxApp,
    _KNOWN_BROKEN_TAGS,
    _is_ollama_load_failure,
    mark_broken_ollama_tag,
)


def _app_with_agent(*, session_done: bool) -> CodingBoxApp:
    app = CodingBoxApp.__new__(CodingBoxApp)
    app.agent = MagicMock()
    app.agent._backend = MagicMock()
    app._session_done = session_done
    app._session_backend = None
    app._session_backend_info = None
    app._session_model = "bad-model:latest"
    app._next_backend = None
    app._next_model = None
    app._last_listing = [("ollama", "good-model:27b")]
    app._is_streaming = False
    app._out_path = None
    app._goal = "build a game"
    app._run_profile = "local_auto"
    app._log_info = lambda *a, **k: None
    app._log_error = lambda *a, **k: None
    app._update_status = lambda *a, **k: None
    app._update_mode_bar = lambda *a, **k: None
    return app


def test_ollama_load_failure_detection() -> None:
    assert _is_ollama_load_failure(
        "unable to load model: /usr/share/ollama/.ollama/models/blobs/sha256-abc "
        "(status code: 500)"
    )
    assert not _is_ollama_load_failure("connection refused")


def test_mark_broken_tag_dedupes() -> None:
    _KNOWN_BROKEN_TAGS.discard("test-broken:latest")
    mark_broken_ollama_tag("test-broken:latest")
    mark_broken_ollama_tag("test-broken:latest")
    assert "test-broken:latest" in _KNOWN_BROKEN_TAGS
    _KNOWN_BROKEN_TAGS.discard("test-broken:latest")


def test_model_swap_updates_agent_when_session_done() -> None:
    app = _app_with_agent(session_done=True)
    fake_backend = MagicMock()
    fake_info = MagicMock()
    fake_info.name = "ollama"
    fake_info.model = "good-model:27b"
    with patch("chat.backend_mod.make_backend", return_value=fake_backend), patch(
        "chat.backend_mod.ollama_endpoint_url", return_value="http://127.0.0.1:11434"
    ), patch(
        "chat.backend_mod.BackendInfo", return_value=fake_info
    ):
        ok = app._apply_model_to_active_session(
            "ollama", "good-model:27b", source="test",
        )
    assert ok is True
    assert app.agent._backend is fake_backend
    assert app._session_model == "good-model:27b"


def test_cmd_set_model_swaps_when_session_done() -> None:
    app = _app_with_agent(session_done=True)
    with patch.object(app, "_apply_model_to_active_session", return_value=True) as swap:
        app._cmd_set_model("1")
    swap.assert_called_once_with("ollama", "good-model:27b", source="/model hot-swap")
    assert app._next_model == "good-model:27b"
