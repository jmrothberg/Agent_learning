#!/usr/bin/env python3
"""One-shot smoke test: generate a single 'doom' sprite via the
self-contained Z-Image-Turbo loader. Used to verify the install
works end-to-end. Saves the PNG into games/_smoke/doom.png and
prints the absolute path on success.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets


def main() -> int:
    out_dir = Path(__file__).parent.parent / "games" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Constructing ZImageTurboGenerator (no model load yet) …")
    gen = assets.try_load_image_generator()
    if gen is None:
        print("    ✗ try_load_image_generator() returned None.")
        print("    Likely missing: torch / diffusers / CUDA / model files.")
        print("    Run scripts/install_diffuser.sh first.")
        return 1
    print(f"    ✓ {type(gen).__name__} ready (model_path={gen.model_path})")

    print()
    print("[2] First .generate() call — pipeline loads into GPU VRAM.")
    print("    Expect 30-60s on first run (downloads weights from HF if")
    print("    /home/jonathan/Models_Diffusers/Z-Image-Turbo/ is missing),")
    print("    then ~3s of inference …")
    t0 = time.time()
    spec = [{
        "name": "doom",
        "prompt": (
            "doom video game cover art, demon with red skin and horns "
            "screaming, dark gritty pixel-art aesthetic, 1990s shareware "
            "FPS feel, dramatic shadows, transparent background"
        ),
        "size": (256, 256),
    }]
    paths = assets.generate_assets(
        spec,
        session_dir=out_dir,
        cache_dir=Path(__file__).parent.parent / "games" / "_asset_cache",
        image_generator=gen,
    )
    elapsed = time.time() - t0
    print()
    if not paths:
        print(f"    ✗ generate_assets returned empty after {elapsed:.1f}s.")
        return 2
    for name, path in paths.items():
        size = path.stat().st_size if path.exists() else 0
        print(f"    ✓ {name}: {path}")
        print(f"      size: {size:,} bytes")
    print(f"    total elapsed: {elapsed:.1f}s")
    print()
    print("Open the PNG to inspect:")
    print(f"  xdg-open {paths['doom']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
