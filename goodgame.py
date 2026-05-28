"""Promote a finished session into goodgame/ (tracked; your git workflow picks it up)."""

from __future__ import annotations

import shutil
from pathlib import Path

GOODGAME_DIR = Path("goodgame")


def promote_session_game(
    *,
    out_path: Path,
    best_path: Path | None = None,
    assets_dir: Path | None = None,
    sounds_dir: Path | None = None,
    dest_root: Path | None = None,
) -> dict[str, Path | None]:
    """Copy best.html (or live .html) plus *_assets/ and *_sounds/ into goodgame/.

    Uses out_path.stem so relative ASSET_DIR / SND_DIR paths in the HTML stay valid.
    Returns dict with keys: html, assets, sounds (values None if not copied).
    """
    out_path = Path(out_path)
    stem = out_path.stem
    root = Path(dest_root) if dest_root is not None else GOODGAME_DIR
    root.mkdir(parents=True, exist_ok=True)

    src_html = Path(best_path) if best_path and Path(best_path).is_file() else out_path
    if not src_html.is_file():
        raise FileNotFoundError(f"no game HTML to promote: {src_html}")

    dest_html = root / f"{stem}.html"
    shutil.copy2(src_html, dest_html)

    copied: dict[str, Path | None] = {"html": dest_html, "assets": None, "sounds": None}

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
