"""Bare /seed and /ref open a native file picker; /seed clear still clears."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from chat import CodingBoxApp, _pick_file_native


def _app() -> CodingBoxApp:
    app = CodingBoxApp()
    logs: list[str] = []
    app._log_info = lambda msg: logs.append(str(msg))  # type: ignore[method-assign]
    app._log_error = lambda msg: logs.append("ERR:" + str(msg))  # type: ignore[method-assign]
    app._logs = logs  # type: ignore[attr-defined]
    return app


def test_bare_seed_opens_picker_and_stages(tmp_path: Path) -> None:
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    app = _app()
    with patch("chat._pick_file_native", return_value=html) as pick:
        app._cmd_set_seed("")
    pick.assert_called_once()
    assert app._next_seed == html.resolve()


def test_seed_clear_drops_staged(tmp_path: Path) -> None:
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    app = _app()
    app._next_seed = html
    app._cmd_set_seed("clear")
    assert app._next_seed is None


def test_bare_seed_cancel_leaves_unstaged() -> None:
    app = _app()
    with patch("chat._pick_file_native", return_value=None):
        app._cmd_set_seed("")
    assert app._next_seed is None


def test_bare_ref_opens_picker(tmp_path: Path) -> None:
    png = tmp_path / "mood.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    app = _app()
    with patch("chat._pick_file_native", return_value=png) as pick:
        app._cmd_attach_ref_image("")
    pick.assert_called_once()
    assert app._staged_ref_image_bytes is not None
    assert app._staged_ref_image_name == "mood.png"


def test_pick_file_native_signature() -> None:
    # Importable helper used by /seed and /ref.
    assert callable(_pick_file_native)
