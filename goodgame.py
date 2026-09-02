"""Promote a finished session into goodgame/ (tracked; your git workflow picks it up)."""

from __future__ import annotations

import shutil
from pathlib import Path

from tools import html_owns_its_folder

GOODGAME_DIR = Path("goodgame")

_MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".ogg", ".mp3", ".wav", ".m4a",
    ".mp4", ".webm",
})


def promote_session_game(
    *,
    out_path: Path,
    best_path: Path | None = None,
    assets_dir: Path | None = None,
    sounds_dir: Path | None = None,
    dest_root: Path | None = None,
) -> dict[str, Path | None]:
    """Copy the shipped HTML plus media into goodgame/.

    Per-game folders (`NAME/NAME.html`) copy as `goodgame/NAME/` with
    HTML + sprites/sounds at top level. Legacy sessions still copy
    `{stem}.html` plus `{stem}_assets/` / `{stem}_sounds/`.
    """
    out_path = Path(out_path)
    stem = out_path.stem
    root = Path(dest_root) if dest_root is not None else GOODGAME_DIR
    root.mkdir(parents=True, exist_ok=True)

    src_html = Path(best_path) if best_path and Path(best_path).is_file() else out_path
    if not src_html.is_file():
        raise FileNotFoundError(f"no game HTML to promote: {src_html}")

    if html_owns_its_folder(out_path):
        dest_dir = root / stem
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True)
        dest_html = dest_dir / out_path.name
        shutil.copy2(src_html, dest_html)
        copied: dict[str, Path | None] = {
            "html": dest_html, "assets": dest_dir, "sounds": dest_dir,
        }
        game_dir = out_path.parent
        for f in game_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() == ".html":
                continue
            if f.suffix.lower() not in _MEDIA_SUFFIXES:
                continue
            shutil.copy2(f, dest_dir / f.name)
        return copied

    dest_html = root / f"{stem}.html"
    shutil.copy2(src_html, dest_html)

    copied = {"html": dest_html, "assets": None, "sounds": None}

    for key, src_dir in (("assets", assets_dir), ("sounds", sounds_dir)):
        if src_dir is None:
            continue
        src = Path(src_dir)
        if not src.is_dir():
            continue
        dest = root / f"{stem}_{key}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        copied[key] = dest

    return copied
