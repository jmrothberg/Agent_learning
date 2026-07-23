"""Tests for /assets staging (bring your own PNGs into a new session)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chat import CodingBoxApp


def _app() -> CodingBoxApp:
    app = CodingBoxApp()
    app._log_info = lambda msg: None  # type: ignore[method-assign]
    app._log_error = lambda msg: None  # type: ignore[method-assign]
    return app


def test_assets_help_explains_seed_ref_and_assets() -> None:
    import tui_help

    lines = tui_help.help_topic_lines("assets") or []
    text = "\n".join(lines).lower()
    assert "/assets" in text
    assert "/seed" in text
    assert "/ref" in text
    assert "vlm" in text
    assert "no vlm" in text or "text coder" in text
    assert "space invaders" in text
    assert "invader_a" in text
    assert "sprite('player')" in "\n".join(lines)
    assert "quick pick" in text
    assert tui_help.normalize_help_topic("asset") == "assets"
    assert tui_help.normalize_help_topic("bring-art") == "assets"


def test_ref_help_points_to_assets_not_copy() -> None:
    import tui_help

    text = "\n".join(tui_help.help_topic_lines("ref") or []).lower()
    assert "not" in text and ("sprite" in text or "copy" in text)
    assert "/assets" in text


def test_cmd_stage_assets_file_and_folder(tmp_path: Path) -> None:
    app = _app()
    png = tmp_path / "hero.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    folder = tmp_path / "pack"
    folder.mkdir()
    (folder / "alien.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (folder / "notes.txt").write_text("ignore")

    app._cmd_stage_assets(str(png))
    assert len(app._staged_asset_paths) == 1
    assert app._staged_asset_paths[0].stem == "hero"

    app._cmd_stage_assets(str(folder))
    stems = {p.stem for p in app._staged_asset_paths}
    assert stems == {"hero", "alien"}

    app._cmd_stage_assets("")
    assert app._staged_asset_paths == []


def test_apply_staged_assets_copies_into_session_dir(tmp_path: Path) -> None:
    app = _app()
    src = tmp_path / "ship.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    app._staged_asset_paths = [src.resolve()]
    agent = SimpleNamespace(_session_assets={})
    dest_dir = tmp_path / "game_assets"
    app._apply_staged_assets_to_session(agent, dest_dir)
    assert (dest_dir / "ship.png").is_file()
    assert "ship" in agent._session_assets
    # Sticky: staging list remains for another /new
    assert len(app._staged_asset_paths) == 1
