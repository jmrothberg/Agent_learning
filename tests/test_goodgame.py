"""Tests for goodgame.promote_session_game."""

from pathlib import Path

from goodgame import promote_session_game


def test_promote_copies_html_and_asset_dirs(tmp_path: Path) -> None:
    games = tmp_path / "games"
    games.mkdir()
    stem = "snake_20260525_120000"
    out = games / f"{stem}.html"
    best = games / f"{stem}.best.html"
    out.write_text("<html>live</html>", encoding="utf-8")
    best.write_text("<html>best</html>", encoding="utf-8")

    assets = games / f"{stem}_assets"
    assets.mkdir()
    (assets / "ship.png").write_bytes(b"png")
    sounds = games / f"{stem}_sounds"
    sounds.mkdir()
    (sounds / "beep.ogg").write_bytes(b"ogg")

    dest = tmp_path / "goodgame"
    copied = promote_session_game(
        out_path=out,
        best_path=best,
        assets_dir=assets,
        sounds_dir=sounds,
        dest_root=dest,
    )

    assert (dest / f"{stem}.html").read_text(encoding="utf-8") == "<html>best</html>"
    assert (dest / f"{stem}_assets" / "ship.png").read_bytes() == b"png"
    assert (dest / f"{stem}_sounds" / "beep.ogg").read_bytes() == b"ogg"
    assert copied["html"] == dest / f"{stem}.html"


def test_promote_falls_back_to_live_html(tmp_path: Path) -> None:
    games = tmp_path / "games"
    games.mkdir()
    stem = "minimal_20260525_120001"
    out = games / f"{stem}.html"
    out.write_text("<html>only</html>", encoding="utf-8")

    dest = tmp_path / "goodgame"
    promote_session_game(out_path=out, dest_root=dest)

    assert (dest / f"{stem}.html").read_text(encoding="utf-8") == "<html>only</html>"
