#!/usr/bin/env python3
"""Interactive sprite drawer — same Z-Image-Turbo pipeline as the agent.

Walks you through prompts, then writes PNGs under games/_draw/.

Usage:
    .venv/bin/python scripts/draw_game_art.py

Requires GPU setup: ./scripts/install_diffuser.sh
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import assets  # noqa: E402


def _read(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return raw or default


def _yes_no(prompt: str, *, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _read(f"{prompt} ({hint})", "y" if default else "n").lower()
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default


def _slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "-", text.strip().lower()).strip("-")
    return (s[:40] or "sprites")


def _ensure_transparent(prompt: str) -> str:
    if "transparent" in prompt.lower():
        return prompt
    return f"{prompt}, transparent background"


def _with_style(prompt: str, style: str) -> str:
    prompt = prompt.strip()
    if style and style.lower() not in prompt.lower():
        prompt = f"{prompt}, {style}"
    return _ensure_transparent(prompt)


def _parse_size(raw: str, default: int = 512) -> tuple[int, int]:
    if not raw:
        return (default, default)
    try:
        return assets._parse_size(raw)
    except Exception:
        print(f"  (could not parse {raw!r}, using {default}px)")
        return (default, default)


def _collect_single_specs(style: str, size: tuple[int, int]) -> list[dict]:
    print()
    print("Enter each sprite. Leave the name blank when you are done.")
    specs: list[dict] = []
    while True:
        print()
        name = _read("Name")
        if not name:
            break
        desc = _read("Description")
        if not desc:
            print("  (skipped — need a description)")
            continue
        specs.append(
            {
                "name": name,
                "prompt": _with_style(desc, style),
                "size": size,
            }
        )
    return specs


def _collect_animation_specs(style: str, size: tuple[int, int]) -> list[dict]:
    print()
    print("Animation mode — one character, multiple poses.")
    print("The first frame is the base look; later frames reuse it for consistency.")
    print()

    char_name = _read("Character name", "hero")
    char_desc = _read("Character look (colors, outfit, facing, art style)")
    if not char_desc:
        raise SystemExit("Need a character description.")

    idle_name = _read("Base frame name", f"{char_name}_idle")
    idle_pose = _read("Base pose", "standing idle, feet together")
    idle_prompt = _with_style(
        f"{char_desc}, {idle_pose}" if idle_pose else char_desc,
        style,
    )

    specs: list[dict] = [
        {"name": idle_name, "prompt": idle_prompt, "size": size},
    ]

    print()
    print("Add pose frames (kick, walk, block, …). Leave name blank to finish.")
    while True:
        print()
        frame_name = _read("Frame name")
        if not frame_name:
            break
        pose = _read("Pose / action")
        if not pose:
            print("  (skipped — need a pose description)")
            continue
        specs.append(
            {
                "name": frame_name,
                "prompt": _ensure_transparent(pose),
                "size": size,
                "from_image": idle_name,
            }
        )
    if len(specs) < 2:
        print("Only one frame — treating this as a single-image draw.")
    return specs


def _summarize_specs(specs: list[dict]) -> None:
    print()
    print(f"Sprites to generate: {len(specs)}")
    for spec in specs:
        chain = ""
        if spec.get("from_image"):
            chain = f"  (from {spec['from_image']})"
        w, h = spec["size"]
        print(f"  • {spec['name']}  {w}×{h}{chain}")
        print(f"    {spec['prompt'][:120]}{'…' if len(spec['prompt']) > 120 else ''}")


def _open_folder(folder: Path) -> None:
    if platform.system() != "Darwin":
        subprocess.run(["xdg-open", str(folder)], check=False)
        return
    subprocess.run(["open", str(folder)], check=False)


def main() -> int:
    print("Game art drawer (Z-Image-Turbo)")
    print("Same diffuser the agent uses — PNGs land in games/_draw/.")
    print()

    project = _read("Project name (for the output folder)", "sprites")
    style = _read(
        "Shared style note (optional)",
        "pixel-art game sprite, clean edges",
    )
    size = _parse_size(_read("Sprite size in pixels", "512"))

    print()
    print("Mode:")
    print("  1) Single images — unrelated sprites")
    print("  2) Animation — one character, multiple poses")
    mode = _read("Choice", "1")

    if mode in {"2", "a", "anim", "animation", "animated"}:
        specs = _collect_animation_specs(style, size)
    else:
        specs = _collect_single_specs(style, size)

    if not specs:
        print("Nothing to generate.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "games" / "_draw" / f"{_slug(project)}_{stamp}_assets"
    cache_dir = REPO / "games" / "_asset_cache"

    _summarize_specs(specs)
    print()
    print(f"Output: {out_dir}")
    if not _yes_no("Generate now?", default=True):
        print("Cancelled.")
        return 0

    print()
    print("Loading diffuser (first run may take 30–60s) …")
    gen = assets.try_load_image_generator()
    if gen is None:
        print("✗ Could not load Z-Image-Turbo.")
        print("  Run ./scripts/install_diffuser.sh first.")
        return 1

    t0 = time.time()
    paths = assets.generate_assets(
        specs,
        session_dir=out_dir,
        cache_dir=cache_dir,
        image_generator=gen,
    )
    elapsed = time.time() - t0

    print()
    if not paths:
        print(f"✗ No sprites generated ({elapsed:.1f}s).")
        err = getattr(gen, "_last_error", None)
        if err:
            print(f"  {err}")
        return 2

    print(f"Done in {elapsed:.1f}s — {len(paths)} PNG(s):")
    for name in sorted(paths):
        path = paths[name]
        kb = path.stat().st_size // 1024 if path.exists() else 0
        print(f"  {name}: {path}  ({kb} KB)")

    print()
    if _yes_no("Open output folder?", default=True):
        _open_folder(out_dir.resolve())

    print()
    print("Preview them:")
    print(f"  .venv/bin/python scripts/play_folder.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
