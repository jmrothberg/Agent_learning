#!/usr/bin/env python3
"""Quick independent sprite-gen test: generate five Doom-themed pixel-art
sprites via the self-contained Z-Image-Turbo pipeline. Verifies the
pipeline runs end-to-end on this machine (cold-start + 5x warm
inference) before trusting it inside the agent loop.

Output: games/_smoke/doom_sprites/{imp,marine,shotgun,medkit,floor_tile}.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets


SPECS = [
    {
        "name": "imp",
        "prompt": (
            "pixel art sprite of a doom imp demon, brown skin, spiked "
            "shoulders, fireball in hand, snarling, 1990s shareware FPS "
            "style, front-facing, transparent background"
        ),
        "size": (256, 256),
    },
    {
        "name": "marine",
        "prompt": (
            "pixel art sprite of a doom space marine, green armor, "
            "holding a shotgun, helmet visor, 1990s shareware FPS "
            "style, front-facing, transparent background"
        ),
        "size": (256, 256),
    },
    {
        "name": "shotgun",
        "prompt": (
            "pixel art sprite of a pump-action shotgun, side view, "
            "dark wood and steel, doom 1993 weapon pickup style, "
            "transparent background"
        ),
        "size": (256, 256),
    },
    {
        "name": "medkit",
        "prompt": (
            "pixel art sprite of a small white medkit with a red cross, "
            "doom 1993 health pickup style, isometric tilt, "
            "transparent background"
        ),
        "size": (128, 128),
    },
    {
        "name": "floor_tile",
        "prompt": (
            "pixel art seamless tileable floor texture, dark grimy "
            "metal grating with rivets, doom 1993 hellish base "
            "aesthetic, top-down view"
        ),
        "size": (256, 256),
    },
]


def main() -> int:
    out_dir = Path(__file__).parent.parent / "games" / "_smoke" / "doom_sprites"
    cache_dir = Path(__file__).parent.parent / "games" / "_asset_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading ZImageTurboGenerator …")
    gen = assets.try_load_image_generator()
    if gen is None:
        print("    FAIL: try_load_image_generator() returned None.")
        print("    Likely missing: torch / diffusers / model files.")
        print("    Run scripts/install_diffuser.sh first.")
        return 1
    print(f"    OK: {type(gen).__name__} (model_path={gen.model_path})")

    print()
    print(f"[2] Generating {len(SPECS)} sprites — first call cold-loads")
    print("    pipeline into MPS VRAM (~30-60s), then ~3-14s per image …")
    t0 = time.time()
    paths = assets.generate_assets(
        SPECS,
        session_dir=out_dir,
        cache_dir=cache_dir,
        image_generator=gen,
    )
    elapsed = time.time() - t0
    print()

    expected = {s["name"] for s in SPECS}
    got = set(paths.keys())
    missing = expected - got
    if missing:
        print(f"    FAIL: missing {missing} after {elapsed:.1f}s")
        return 2

    bad = []
    for name in expected:
        path = paths[name]
        size = path.stat().st_size if path.exists() else 0
        marker = "OK " if size >= 5000 else "TINY"
        print(f"    [{marker}] {name}: {size:>8,} bytes  {path}")
        if size < 5000:
            bad.append(name)
    print()
    print(f"total elapsed: {elapsed:.1f}s ({elapsed / len(SPECS):.1f}s per sprite avg)")

    stats = getattr(gen, "last_stats", None)
    if stats:
        print(f"per-asset stats: {stats}")

    if bad:
        print(f"FAIL: {bad} are suspiciously small (<5KB)")
        return 3

    print()
    print("PASS. Open the sprites:")
    print(f"  open {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
