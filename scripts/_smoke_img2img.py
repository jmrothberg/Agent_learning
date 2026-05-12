#!/usr/bin/env python3
"""End-to-end smoke test for the SD-Turbo img2img animation chain.

Generates frame 1 from text (Z-Image-Turbo), then frame 2 from frame 1
as init image (SD-Turbo img2img at strength=0.45). Writes both PNGs
into games/_smoke/img2img/ and prints a coarse similarity score so
you can verify the frames are coherent (a walk-cycle, not two random
sprites) without launching the full agent.

Coherence read:
    similarity > 0.55  → strongly correlated frames (good)
    0.30–0.55          → recognizably related but loose (still OK for
                          gameplay; tighten strength if needed)
    < 0.30             → effectively independent (something is wrong;
                          check SD-Turbo install or the prompts)

The score is a normalized mean-squared-error of luminance between the
two output frames at the same resolution. It's a quick proxy, not a
proper perceptual metric — the eye remains the final judge.

Usage:
    .venv/bin/python scripts/_smoke_img2img.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets


FRAME1_PROMPT = (
    "8-bit pixel-art green space invader alien, front view, "
    "antennae up, legs together, transparent background"
)
FRAME2_PROMPT = (
    "8-bit pixel-art green space invader alien, front view, "
    "antennae up, legs apart in mid-step, transparent background"
)


def _luminance_similarity(p1: Path, p2: Path) -> float:
    """Compute 1 - normalized MSE of luminance between two images
    resized to a common 64x64 grid. Returns 0..1 where 1 = identical
    and 0 = maximum dissimilarity (e.g. inverted images).
    """
    from PIL import Image
    a = Image.open(p1).convert("L").resize((64, 64), Image.LANCZOS)
    b = Image.open(p2).convert("L").resize((64, 64), Image.LANCZOS)
    pa = list(a.getdata())
    pb = list(b.getdata())
    if not pa or not pb:
        return 0.0
    sq_diff = sum((x - y) * (x - y) for x, y in zip(pa, pb)) / len(pa)
    # 255**2 = 65025 is the worst case per-pixel; normalize to [0, 1].
    return max(0.0, 1.0 - sq_diff / 65025.0)


def main() -> int:
    out_dir = Path(__file__).parent.parent / "games" / "_smoke" / "img2img"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Constructing ZImageTurboGenerator …")
    txt2img = assets.try_load_image_generator()
    if txt2img is None:
        print("    ✗ try_load_image_generator() returned None.")
        print("    Likely missing: torch / diffusers / GPU / model files.")
        print("    Run scripts/install_diffuser.sh first.")
        return 1
    print(f"    ✓ {type(txt2img).__name__} ready")

    print("[2] Constructing Img2ImgGenerator (SD-Turbo) …")
    i2i = assets.try_load_img2img_generator()
    if i2i is None:
        print("    ✗ try_load_img2img_generator() returned None.")
        print("    Run scripts/install_diffuser.sh (without --skip-img2img).")
        return 1
    print(f"    ✓ {type(i2i).__name__} ready (model_path={i2i.model_path})")

    print("[3] Generating frame 1 (txt2img) …")
    t0 = time.time()
    f1_src = txt2img.generate(FRAME1_PROMPT)
    if f1_src is None:
        print(f"    ✗ txt2img failed: {txt2img._last_error}")
        return 2
    print(f"    ✓ frame 1 in {time.time() - t0:.1f}s → {f1_src}")
    f1_dst = out_dir / "frame1.png"
    Path(f1_src).rename(f1_dst)

    print("[4] Generating frame 2 (img2img from frame 1, strength=0.45) …")
    t0 = time.time()
    f2_src = i2i.generate(FRAME2_PROMPT, str(f1_dst), strength=0.45)
    if f2_src is None:
        print(f"    ✗ img2img failed: {i2i._last_error}")
        return 3
    print(f"    ✓ frame 2 in {time.time() - t0:.1f}s → {f2_src}")
    f2_dst = out_dir / "frame2.png"
    Path(f2_src).rename(f2_dst)

    print("[5] Comparing frames …")
    sim = _luminance_similarity(f1_dst, f2_dst)
    print(f"    luminance similarity = {sim:.3f}")
    if sim > 0.55:
        verdict = "STRONG — frames are visibly the same character"
    elif sim > 0.30:
        verdict = "MODERATE — frames are recognizably related"
    else:
        verdict = "WEAK — frames look independent (check strength / install)"
    print(f"    verdict: {verdict}")

    print()
    print(f"Outputs:\n  {f1_dst}\n  {f2_dst}")
    print("Open both side-by-side to verify visually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
