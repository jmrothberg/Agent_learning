#!/usr/bin/env python3
"""Diagnostic: Z-Image-Turbo on MPS produces NaN at fp16. Try fp32 to
confirm the fix path. One image only — keeps it cheap.

Saves to games/_smoke/dtype_check_fp32.png. Pass = produced PNG > 5KB
with non-uniform pixel values.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    import numpy as np
    import torch
    from diffusers import ZImagePipeline
    from PIL import Image

    if not torch.backends.mps.is_available():
        print("MPS unavailable; skipping.")
        return 1

    model_path = "/Users/jonathanrothberg/Diffusion_Models/Z-Image-Turbo"
    out = Path(__file__).parent.parent / "games" / "_smoke" / "dtype_check_fp32.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path} at fp32 on MPS …")
    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    pipe.to("mps")
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Running inference (1 image, 9 steps) …")
    t0 = time.time()
    g = torch.Generator("mps").manual_seed(42)
    result = pipe(
        prompt="pixel art sprite of a doom imp demon, brown skin, snarling, transparent background",
        height=768,
        width=768,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=g,
    )
    elapsed = time.time() - t0
    print(f"  inference {elapsed:.1f}s")

    img = result.images[0]
    arr = np.array(img)
    print(f"  image shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  pixel min/max/mean: {arr.min()}/{arr.max()}/{arr.mean():.1f}")
    nonzero_unique = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
    print(f"  unique pixel values: {nonzero_unique}")

    img.save(out, format="PNG")
    size = out.stat().st_size
    print(f"  saved {out} ({size:,} bytes)")

    if size < 5000 or nonzero_unique < 100:
        print("FAIL: output looks degenerate (NaN or uniform).")
        return 2
    print("PASS: fp32 on MPS produces a real image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
