#!/usr/bin/env python3
"""Pre-generate the stock SFX pack used as a fallback library for
first-iter audio.

Generating a sound takes 2-5s on Stable Audio Open. A typical game
wants 4-8 sounds. That's 10-40s of Phase A on iter 1, every session,
and the model often skips <sounds> entirely to dodge the cost. The
result: first-iter games are silent.

This script runs once at setup time. It synthesizes a small bank of
universal SFX (jump, pickup, hit, win, lose, click, laser, explosion)
into `games/memory/asset_library/sounds/` and indexes them so any
future session can hit them via the cross-session asset library. The
prompt-side guideline (prompts_v1.py's <sounds> FormatSpec) tells the
model these names exist; the asset library serves them for free.

Idempotent: re-running is a no-op for entries that already exist. No
internet access required at runtime (Stable Audio runs locally).

Usage:
    .venv/bin/python scripts/build_stock_sounds.py
    .venv/bin/python scripts/build_stock_sounds.py --force   # regen
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Universal game SFX. Names match common asset-naming conventions so
# models can guess them. Prompts are kept short, descriptive, and
# stylistically neutral so the cached output works across genres.
STOCK_SOUNDS: list[dict] = [
    {
        "name": "jump",
        "prompt": "short retro arcade jump blip, 8-bit, bright, no music",
        "duration": 0.35,
    },
    {
        "name": "pickup",
        "prompt": "short coin pickup chime, bright bell, 8-bit, no music",
        "duration": 0.40,
    },
    {
        "name": "hit",
        "prompt": "short impact thud, dull crunch, no music",
        "duration": 0.30,
    },
    {
        "name": "win",
        "prompt": "short upward victory fanfare, three bright tones, no music",
        "duration": 1.20,
    },
    {
        "name": "lose",
        "prompt": "short descending sad trombone, three falling tones, no music",
        "duration": 1.20,
    },
    {
        "name": "click",
        "prompt": "short ui click tick, sharp short button press, no music",
        "duration": 0.15,
    },
    {
        "name": "laser",
        "prompt": "short laser zap, descending pew, sci-fi arcade, no music",
        "duration": 0.40,
    },
    {
        "name": "explosion",
        "prompt": "short explosion crunch, low rumble with crackle, no music",
        "duration": 0.80,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a stock sound is already present in the library.",
    )
    args = parser.parse_args()

    try:
        from asset_library import AssetLibrary, _tokenize
        from sounds import (
            try_load_audio_generator,
            _safe_generate,
            _safe_filename,
        )
    except ImportError as exc:
        print(f"[stock-sounds] import failed: {exc}", file=sys.stderr)
        return 1

    library = AssetLibrary()
    library._ensure_dirs()
    print(f"[stock-sounds] library root: {library.root}")

    gen = try_load_audio_generator("stable-audio-open-1.0")
    if gen is None:
        print(
            "[stock-sounds] no audio generator available (torch/diffusers/"
            "Stable Audio model not installed). Stock pack will be empty; "
            "the agent runs fine without it.",
            file=sys.stderr,
        )
        return 0

    n_added = 0
    n_skipped = 0
    n_failed = 0
    with tempfile.TemporaryDirectory(prefix="stock_sounds_") as tmp:
        tmp_path = Path(tmp)
        for spec in STOCK_SOUNDS:
            name = _safe_filename(spec["name"])
            prompt = spec["prompt"]
            duration = float(spec["duration"])
            # Skip if a library entry already covers this prompt+duration.
            if not args.force:
                hit = library.retrieve(
                    prompt=prompt, modality="sound", size_or_duration=duration,
                )
                if hit is not None:
                    print(
                        f"[stock-sounds] {name:10s}  skip  "
                        f"(library hit: {hit.entry.path})"
                    )
                    n_skipped += 1
                    continue
            print(f"[stock-sounds] {name:10s}  gen   prompt={prompt!r}")
            ogg_path = _safe_generate(gen, prompt, duration)
            if ogg_path is None:
                err = getattr(gen, "_last_error", "unknown")
                print(
                    f"[stock-sounds] {name:10s}  FAIL  {err}",
                    file=sys.stderr,
                )
                n_failed += 1
                continue
            # Move into a stable temp name so admit() can copy it cleanly.
            target = tmp_path / f"{name}.ogg"
            try:
                import shutil
                shutil.copy2(ogg_path, target)
            except Exception as exc:
                print(
                    f"[stock-sounds] {name:10s}  FAIL  copy: {exc}",
                    file=sys.stderr,
                )
                n_failed += 1
                continue
            entry = library.admit(
                prompt=prompt,
                modality="sound",
                size_or_duration=duration,
                source_path=target,
            )
            if entry is None:
                print(
                    f"[stock-sounds] {name:10s}  FAIL  admission rejected",
                    file=sys.stderr,
                )
                n_failed += 1
                continue
            print(f"[stock-sounds] {name:10s}  ok    -> {entry.path}")
            n_added += 1

    print(
        f"\n[stock-sounds] done: added={n_added} skipped={n_skipped} "
        f"failed={n_failed}"
    )
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
