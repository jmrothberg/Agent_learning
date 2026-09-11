"""TUI /server on and /model N server — oMLX without LLM_BACKEND=mlx-server."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


def _bare_app() -> CodingBoxApp:
    app = CodingBoxApp()
    app.agent = None
    app._update_status = MagicMock()  # type: ignore[assignment]
    app._update_mode_bar = MagicMock()  # type: ignore[assignment]
    app._log_info = MagicMock()  # type: ignore[assignment]
    app._log_error = MagicMock()  # type: ignore[assignment]
    return app


def test_split_mlx_server_token():
    assert CodingBoxApp._split_mlx_server_token("12 server") == ("12", True)
    assert CodingBoxApp._split_mlx_server_token("Qwen3.8 server") == ("Qwen3.8", True)
    assert CodingBoxApp._split_mlx_server_token("12 local") == ("12", False)
    assert CodingBoxApp._split_mlx_server_token("12 in-process") == ("12", False)
    assert CodingBoxApp._split_mlx_server_token("12") == ("12", None)
    assert CodingBoxApp._split_mlx_server_token("server") == ("server", None)


def test_server_off_and_on(monkeypatch):
    app = _bare_app()
    assert app._mlx_via_server is False
    app._cmd_set_mlx_server("off")
    assert app._mlx_via_server is False
    monkeypatch.setattr(
        "backend.ensure_omlx_server", lambda: "http://127.0.0.1:8000",
    )
    app._cmd_set_mlx_server("on")
    assert app._mlx_via_server is True
    app._cmd_set_mlx_server("")
    logged = " ".join(str(c.args[0]) for c in app._log_info.call_args_list)
    assert "ON" in logged or "on" in logged.lower()


def test_model_n_server_stages_mlx_via_omlx(monkeypatch):
    app = _bare_app()
    app._last_listing = [
        ("mlx", "/Users/me/MLX_Models/Qwen3.8-27B-mxfp8"),
    ]
    monkeypatch.setattr(
        "backend.ensure_omlx_server", lambda: "http://127.0.0.1:8000",
    )
    monkeypatch.setattr(
        "backend.omlx_api_model_id",
        lambda p: Path(p).name if p else p,
    )
    app._cmd_set_model("1 server")
    assert app._mlx_via_server is True
    assert app._next_backend == "mlx"
    assert app._next_model == "Qwen3.8-27B-mxfp8"


def test_model_server_ignored_on_ollama_pick(monkeypatch):
    app = _bare_app()
    app._last_listing = [("ollama", "qwen3.6:27b")]
    monkeypatch.setattr(
        "backend.ensure_omlx_server",
        lambda: (_ for _ in ()).throw(RuntimeError("should not start oMLX")),
    )
    app._cmd_set_model("1 server")
    assert app._mlx_via_server is False
    assert app._next_backend == "ollama"
    assert app._next_model == "qwen3.6:27b"
    logged = " ".join(str(c.args[0]) for c in app._log_info.call_args_list)
    assert "ignored" in logged.lower()


def test_bare_model_server_is_server_on(monkeypatch):
    app = _bare_app()
    monkeypatch.setattr(
        "backend.ensure_omlx_server", lambda: "http://127.0.0.1:8000",
    )
    app._cmd_set_model("server")
    assert app._mlx_via_server is True


def test_help_server_topic_is_simple():
    import tui_help

    assert tui_help.normalize_help_topic("server") == "server"
    assert tui_help.normalize_help_topic("omlx") == "server"
    text = "\n".join(tui_help.help_topic_lines("server") or []).lower()
    assert "/server on" in text
    assert "/model" in text and "server" in text
    assert "critic" in text
    critic = "\n".join(tui_help.help_topic_lines("critic") or []).lower()
    assert "/server on" in critic
