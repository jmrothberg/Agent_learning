#!/usr/bin/env python3
"""One-shot smoke test: generate three short OGGs via the
self-contained Stable Audio Open loader. Used to verify the install
works end-to-end. Saves the audio into games/_smoke/ and prints the
absolute paths on success.

Run after `pip install -r requirements-diffuser.txt` (which now
includes soundfile alongside the existing diffusers/torch deps).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sounds


def main() -> int:
    out_dir = Path(__file__).parent.parent / "games" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Constructing StableAudioGenerator (no model load yet) …")
    gen = sounds.try_load_audio_generator()
    if gen is None:
        print("    ✗ try_load_audio_generator() returned None.")
        print("    Likely missing: torch / diffusers / soundfile / GPU.")
        print("    Run scripts/install_diffuser.sh first.")
        return 1
    print(f"    ✓ {type(gen).__name__} ready (model_path={gen.model_path})")

    print()
    print("[2] First .generate() call — pipeline loads into GPU/MPS.")
    print("    Expect 30-60s on first run (downloads weights from HF if")
    print("    no local stable-audio-open-small/ exists), then a few")
    print("    seconds of inference per clip …")
    specs = [
        {
            "name": "laser",
            "prompt": "short retro arcade laser shot, 8-bit synth blip, sharp attack",
            "duration": 0.4,
            "loop": False,
        },
        {
            "name": "explosion",
            "prompt": "short pixelated explosion, 8-bit boom with crunchy decay",
            "duration": 0.8,
            "loop": False,
        },
        {
            "name": "music",
            "prompt": "loopable 8-bit chiptune background music, 90 bpm, upbeat arcade feel",
            "duration": 6.0,
            "loop": True,
        },
    ]
    t0 = time.time()
    paths = sounds.generate_sounds(
        specs,
        session_dir=out_dir / "audio",
        cache_dir=Path(__file__).parent.parent / "games" / "_sound_cache",
        audio_generator=gen,
    )
    elapsed = time.time() - t0
    print()
    if not paths:
        print(f"    ✗ generate_sounds returned empty after {elapsed:.1f}s.")
        per_sound = getattr(gen, "last_stats", []) or []
        for s in per_sound:
            err = (s or {}).get("error")
            if err:
                print(f"      - {s.get('name','?')}: {err}")
        return 2
    for name, path in paths.items():
        size = path.stat().st_size if path.exists() else 0
        print(f"    ✓ {name}: {path}")
        print(f"      size: {size:,} bytes")
    print(f"    total elapsed: {elapsed:.1f}s")
    print()
    print("Play one to verify (macOS):  afplay  <path>")
    print("Play one to verify (Linux):  aplay   <path>  (or paplay)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
