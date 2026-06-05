#!/usr/bin/env python3
"""Preview a folder of game assets from the terminal.

Drag a folder onto the terminal (or pass a path) after this command:

    .venv/bin/python scripts/play_folder.py

- Mostly audio  → plays each clip in sorted order (afplay / ffplay / open).
- Mostly images → opens them all in Preview (arrow keys to step through).

Mixed folders pick the larger group unless you pass --sounds or --images.
"""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".tif", ".tiff"}
)
SOUND_EXTS = frozenset(
    {".ogg", ".mp3", ".wav", ".m4a", ".aiff", ".aif", ".caf", ".flac", ".aac"}
)


def parse_path_arg(raw_args: list[str]) -> Path:
    """Accept a path pasted or drag-dropped from the terminal."""
    if not raw_args:
        raise SystemExit(
            "Usage: play_folder.py <folder>\n"
            "Tip: type the command, then drag a folder onto the terminal."
        )
    raw = " ".join(raw_args).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    # Terminal drag-drop sometimes escapes spaces as '\ '
    raw = raw.replace("\\ ", " ")
    path = Path(raw).expanduser()
    if not path.exists():
        # Drag-drop can leave shell-style quoting; try shlex as a fallback.
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = []
        if parts:
            path = Path(parts[0]).expanduser()
    if not path.exists():
        raise SystemExit(f"Not found: {raw!r}")
    if not path.is_dir():
        raise SystemExit(f"Not a folder: {path}")
    return path.resolve()


def collect_files(folder: Path, exts: frozenset[str], *, recursive: bool) -> list[Path]:
    globber = folder.rglob if recursive else folder.glob
    hits = [p for p in globber("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(hits, key=lambda p: p.name.lower())


def pick_mode(
    images: list[Path],
    sounds: list[Path],
    force: str | None,
) -> str:
    if force in {"images", "sounds"}:
        return force
    if images and not sounds:
        return "images"
    if sounds and not images:
        return "sounds"
    if not images and not sounds:
        raise SystemExit(
            f"No supported images ({', '.join(sorted(IMAGE_EXTS))}) or "
            f"sounds ({', '.join(sorted(SOUND_EXTS))}) found."
        )
    if len(sounds) > len(images):
        return "sounds"
    if len(images) > len(sounds):
        return "images"
    print(
        f"Mixed folder ({len(sounds)} sounds, {len(images)} images); "
        "showing images. Use --sounds to play audio instead."
    )
    return "images"


def _run(cmd: list[str]) -> int:
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        return 127


def play_sound(path: Path) -> None:
    """Play one clip; try native players before falling back to open."""
    print(f"  ▶ {path.name}")
    if platform.system() == "Darwin":
        for cmd in (
            ["afplay", str(path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            ["open", str(path)],
        ):
            if _run(cmd) == 0:
                return
        print(f"    ✗ could not play {path.name}", file=sys.stderr)
        return
    for cmd in (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ["aplay", str(path)],
        ["paplay", str(path)],
        ["xdg-open", str(path)],
    ):
        if _run(cmd) == 0:
            return
    print(f"    ✗ could not play {path.name}", file=sys.stderr)


def show_images(paths: list[Path]) -> None:
    if not paths:
        return
    print(f"Opening {len(paths)} image(s) in Preview (use ←/→ to step through).")
    if platform.system() == "Darwin":
        rc = _run(["open", "-a", "Preview", *[str(p) for p in paths]])
        if rc == 0:
            return
    # Linux / fallback: one window per file is crude but works everywhere.
    for path in paths:
        print(f"  🖼 {path.name}")
        if _run(["xdg-open", str(path)]) != 0:
            print(f"    ✗ could not open {path.name}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play sounds or show images from a folder (drag-drop friendly)."
    )
    parser.add_argument(
        "folder",
        nargs="*",
        help="Folder path — drag onto the terminal after typing this command",
    )
    parser.add_argument(
        "--sounds",
        action="store_const",
        const="sounds",
        dest="mode",
        help="Force audio playback even in a mixed folder",
    )
    parser.add_argument(
        "--images",
        action="store_const",
        const="images",
        dest="mode",
        help="Force image preview even in a mixed folder",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Only look at files directly inside the folder",
    )
    args = parser.parse_args()

    folder = parse_path_arg(args.folder)
    recursive = not args.no_recurse
    images = collect_files(folder, IMAGE_EXTS, recursive=recursive)
    sounds = collect_files(folder, SOUND_EXTS, recursive=recursive)
    mode = pick_mode(images, sounds, args.mode)

    print(f"{folder}")
    if mode == "sounds":
        print(f"Playing {len(sounds)} sound(s)…")
        for path in sounds:
            play_sound(path)
        return 0

    show_images(images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
