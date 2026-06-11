#!/usr/bin/env python3
"""Local video generation wrapper — Wan2.2-TI2V-5B, cross-platform.

ONE entry point for humans, the agent (<videos> tag via videos.py), and
scripts. Backend is picked by platform:

  - macOS (Apple Silicon): `mlxgen` CLI in the dedicated `.venv-video`
    virtualenv (installed separately from the agent's main .venv so
    mlx-gen's own mlx pin can never break the LLM backend).
    Default model: AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit (~17 GB).
  - Linux (NVIDIA CUDA): in-process diffusers WanPipeline /
    WanImageToVideoPipeline using the SAME .venv this script runs in
    (torch + diffusers come from ./scripts/install_diffuser.sh).
    Default model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (HF cache, first use).

Text-to-video:
    .venv/bin/python scripts/generate_video.py \
        --prompt "a knight runs across a collapsing drawbridge, cinematic" \
        --out games/mygame_videos/intro.mp4

Image-to-video (animate a Z-Image-Turbo still — keeps cutscene art
consistent with in-game sprites):
    .venv/bin/python scripts/generate_video.py \
        --prompt "the dragon slowly wakes and rears its head, embers drift" \
        --image games/mygame_assets/key_dragon_wake.png \
        --out games/mygame_videos/dragon_reveal.mp4

Defaults are tuned for game cutscenes: 832x480, 49 frames written at
12 fps (~4 s of slow cinematic motion, ~2-5 min to generate).
Width/height must be multiples of 32; frames must be 4k+1 (Wan VAE).
Note: the mlx backend snaps the output to the input image's aspect for
image-to-video (e.g. square key art -> square clip) — use CSS
`object-fit:cover` in the game so either shape fills the canvas.

Env overrides:
    VIDEO_MODEL          model alias/repo/path (per-backend default above)
    VIDEO_VENV           venv holding mlxgen (default <repo>/.venv-video; macOS only)
    DIFFUSER_CUDA_DEVICE pin the CUDA index on Linux (same var assets.py uses)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLX_DEFAULT_MODEL = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
DIFFUSERS_DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _mlxgen_bin() -> Path:
    venv = Path(os.environ.get("VIDEO_VENV", REPO / ".venv-video"))
    exe = venv / "bin" / "mlxgen"
    if not exe.exists():
        sys.exit(
            f"mlxgen not found at {exe}\n"
            "Install it first:  python3 -m venv .venv-video && "
            ".venv-video/bin/pip install -U mlx-gen\n"
            "(./scripts/setup.sh does this automatically on macOS)"
        )
    return exe


def _run_mlx(args: argparse.Namespace, out: Path) -> int:
    """macOS backend: shell out to mlx-gen in .venv-video."""
    model = os.environ.get("VIDEO_MODEL", MLX_DEFAULT_MODEL)
    cmd = [
        str(_mlxgen_bin()), "generate",
        "--model", model,
        "--prompt", args.prompt,
        "--negative-prompt", args.negative,
        "--width", str(args.width),
        "--height", str(args.height),
        "--frames", str(args.frames),
        "--fps", str(args.fps),
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--seed", str(args.seed),
        "--output", str(out),
    ]
    if args.image:
        cmd += ["--image", str(Path(args.image).resolve()), "--task", "i2v"]
    print(f"[generate_video] backend=mlx-gen model={model}")
    return subprocess.run(cmd).returncode


def _run_diffusers(args: argparse.Namespace, out: Path) -> int:
    """Linux/CUDA backend: in-process diffusers WanPipeline.

    Imported lazily so the module stays importable (and --help fast) on
    machines without torch. Weights download to the HF cache on first
    use (~25 GB for the 5B Diffusers layout).
    """
    try:
        import torch
        from diffusers import AutoencoderKLWan
        from diffusers.utils import export_to_video
    except Exception as e:
        sys.exit(
            f"torch/diffusers import failed ({type(e).__name__}: {e}).\n"
            "Run ./scripts/install_diffuser.sh first."
        )
    if not torch.cuda.is_available():
        sys.exit("no CUDA device — Wan2.2 on CPU is impractical; aborting.")
    # Respect the same GPU pin the sprite/sound diffusers use.
    pin = (os.environ.get("DIFFUSER_CUDA_DEVICE") or "").strip()
    device = f"cuda:{pin}" if pin.isdigit() else "cuda"

    model = os.environ.get("VIDEO_MODEL", DIFFUSERS_DEFAULT_MODEL)
    print(f"[generate_video] backend=diffusers model={model} device={device}")
    # Wan's VAE wants float32; the transformer runs bf16 (per HF model card).
    vae = AutoencoderKLWan.from_pretrained(
        model, subfolder="vae", torch_dtype=torch.float32,
    )
    if args.image:
        from diffusers import WanImageToVideoPipeline
        from diffusers.utils import load_image
        pipe = WanImageToVideoPipeline.from_pretrained(
            model, vae=vae, torch_dtype=torch.bfloat16,
        )
    else:
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            model, vae=vae, torch_dtype=torch.bfloat16,
        )
    # CPU offload keeps the 5B model inside consumer-GPU VRAM (e.g. 4090);
    # costs some speed but never OOMs. The pin still routes compute.
    try:
        pipe.enable_model_cpu_offload(device=device)
    except Exception:
        pipe.to(device)

    kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    if args.image:
        kwargs["image"] = load_image(str(Path(args.image).resolve()))
    frames = pipe(**kwargs).frames[0]
    export_to_video(frames, str(out), fps=args.fps)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt", required=True, help="motion/scene description")
    ap.add_argument("--image", help="optional first-frame PNG (image-to-video)")
    ap.add_argument("--out", default="video.mp4", help="output .mp4 path")
    ap.add_argument("--width", type=int, default=832, help="multiple of 32")
    ap.add_argument("--height", type=int, default=480, help="multiple of 32")
    ap.add_argument("--frames", type=int, default=49, help="must be 4k+1 (e.g. 17/49/121)")
    ap.add_argument("--fps", type=int, default=12, help="container fps (12 = slow cinematic)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative", default="", help="negative prompt")
    args = ap.parse_args()

    if args.width % 32 or args.height % 32:
        sys.exit("width/height must be multiples of 32")
    if (args.frames - 1) % 4:
        sys.exit("frames must be 4k+1 (17, 49, 121, ...)")
    if args.image and not Path(args.image).exists():
        sys.exit(f"input image not found: {args.image}")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate_video] {args.width}x{args.height} x{args.frames}f @ {args.fps}fps -> {out}")
    t0 = time.time()
    if sys.platform == "darwin":
        rc = _run_mlx(args, out)
    else:
        rc = _run_diffusers(args, out)
    dt = time.time() - t0
    if rc == 0 and out.exists():
        mb = out.stat().st_size / 1e6
        print(f"[generate_video] OK in {dt:.0f}s — {out} ({mb:.1f} MB)")
    else:
        print(f"[generate_video] FAILED (rc={rc}) after {dt:.0f}s")
        rc = rc or 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
