#!/usr/bin/env bash
# Install the full diffusion GPU stack into Agent_learning/.venv:
#   • Sprite generation (<assets>) — macOS: FLUX2-klein via mflux; else
#     Z-Image-Turbo (torch + diffusers git + transformers…)
#   • Stable Audio Open (<sounds>)      — layers requirements-diffuser.txt
#     (soundfile, torchsde, mflux on Darwin, pinned transformers/accelerate)
#
# Cross-platform: Linux (CUDA) vs macOS (MPS) vs other (CPU — slow).
# `./scripts/setup.sh` calls this once; you can also run this script alone
# after creating `.venv`.
#
# Model weights: run `./scripts/setup.sh` — auto-detects FLUX2-klein (macOS)
# and Z-Image-Turbo, writes DIFFUSION_MODELS_DIR to .env. Stable Audio still
# downloads on first <sounds> unless cached.

set -euo pipefail

# B3: img2img is an opt-out (--skip-img2img) — SD-Turbo weights are
# small (~2 GB) and most game goals benefit from animation chaining.
SKIP_IMG2IMG=0
for arg in "$@"; do
  case "$arg" in
    --skip-img2img) SKIP_IMG2IMG=1 ;;
  esac
done

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "ERROR: no .venv/ in $(pwd) — run setup.sh / pip install first."
  exit 1
fi

PIP=".venv/bin/pip"
PY=".venv/bin/python"

# ---------------------------------------------------------------------------
# Platform detection. Z-Image-Turbo support is recent — only diffusers
# git HEAD knows about ZImagePipeline (no tagged release as of
# 2026-05-06). The torch flavor differs by platform:
#
#   Linux + NVIDIA   →  cu130 nightly (Blackwell-class GPUs need cu13)
#   macOS            →  generic nightly (Apple Silicon MPS support
#                       built into the wheels from PyPI)
#   anything else    →  CPU torch from PyPI (warned: too slow for
#                       Z-Image-Turbo in practice)
#
# Override CUDA version with $TORCH_CUDA (e.g. 121, 124, 130) if your
# GPU needs an older one.
# ---------------------------------------------------------------------------

OS="$(uname -s)"

# Auto-pick PyTorch wheel CUDA tag from the installed NVIDIA driver unless
# TORCH_CUDA is already set. Nightly cu130 needs a CUDA 13.x driver; driver
# 12.x (e.g. 570 + CUDA 12.8) needs stable cu126 wheels.
if [ -z "${TORCH_CUDA:-}" ] && [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  _cuda_drv_major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.[0-9]*/\1/p' | head -1)"
  case "${_cuda_drv_major:-}" in
    12) TORCH_CUDA=126 ;;
    13) TORCH_CUDA=130 ;;
  esac
fi
TORCH_CUDA="${TORCH_CUDA:-130}"

case "$OS" in
  Darwin)
    PLATFORM_LABEL="macOS (MPS / Apple Silicon)"
    TORCH_INDEX_FLAGS="--pre --index-url https://download.pytorch.org/whl/nightly/cpu"
    DEVICE_NOTE="Apple Silicon Macs use MPS via the same PyTorch wheel — Z-Image-Turbo on MPS is EXPERIMENTAL."
    ;;
  Linux)
    PLATFORM_LABEL="Linux + NVIDIA (CUDA $TORCH_CUDA)"
    if [ "$TORCH_CUDA" = "130" ]; then
      TORCH_INDEX_FLAGS="--pre --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA}"
    else
      # Stable wheels for cu126/cu124/cu121 — matches CUDA 12.x drivers.
      TORCH_INDEX_FLAGS="--index-url https://download.pytorch.org/whl/cu${TORCH_CUDA}"
    fi
    DEVICE_NOTE="If your GPU needs a different CUDA version, re-run with TORCH_CUDA=121 (or 124, 126, etc)."
    ;;
  *)
    PLATFORM_LABEL="$OS (no GPU support)"
    TORCH_INDEX_FLAGS="--pre"
    DEVICE_NOTE="WARNING: no recognized GPU platform; CPU inference is impractically slow for Z-Image-Turbo."
    ;;
esac

echo "Platform: $PLATFORM_LABEL"
echo "          $DEVICE_NOTE"
echo

echo "[1/4] Installing torch + torchvision + torchaudio …"
# shellcheck disable=SC2086
$PIP install --force-reinstall $TORCH_INDEX_FLAGS torch torchvision torchaudio

echo
echo "[2/4] Installing diffusers from git HEAD (needed for ZImagePipeline + Stable Audio) …"
$PIP install --upgrade git+https://github.com/huggingface/diffusers

echo
echo "[3/4] Installing transformers / accelerate / safetensors / pillow …"
$PIP install --upgrade transformers accelerate safetensors pillow

echo
echo "[4/4] Layering requirements-diffuser.txt (soundfile, torchsde, mflux on macOS) …"
$PIP install -r requirements-diffuser.txt

echo
echo "Verifying GPU + diffusers + sprites + audio Python deps …"
$PY - <<'PY'
import importlib.util as iu
ok_torch = iu.find_spec("torch") is not None
ok_diffusers = iu.find_spec("diffusers") is not None
print(f"  torch installed:        {ok_torch}")
print(f"  diffusers installed:    {ok_diffusers}")
if ok_torch:
    import torch
    print(f"  torch version:          {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    print(f"  CUDA available:         {cuda_ok}")
    print(f"  MPS available:          {mps_ok}")
    if cuda_ok:
        print(f"  CUDA device:            {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU memory free/total:  {free/1e9:.1f}GB / {total/1e9:.1f}GB")
    elif not (cuda_ok or mps_ok):
        print("  No GPU detected — Z-Image-Turbo will refuse to load.")
if ok_diffusers:
    try:
        from diffusers import ZImagePipeline  # noqa: F401
        print(f"  ZImagePipeline class:   importable ✓")
    except Exception as e:
        print(f"  ZImagePipeline class:   IMPORT FAILED — {e!r}")
    try:
        from diffusers import StableAudioPipeline  # noqa: F401
        print(f"  StableAudioPipeline:    importable ✓")
    except Exception as e:
        print(f"  StableAudioPipeline:    IMPORT FAILED — {e!r}")
    # B3: SD-Turbo img2img for animation-frame chaining.
    try:
        from diffusers import AutoPipelineForImage2Image  # noqa: F401
        print(f"  AutoPipelineForImg2Img: importable ✓")
    except Exception as e:
        print(f"  AutoPipelineForImg2Img: IMPORT FAILED — {e!r}")
for mod in ("soundfile", "torchsde"):
    ok = iu.find_spec(mod) is not None
    print(f"  {mod + ':':24} {'✓' if ok else 'MISSING'}")
import sys
if sys.platform == "darwin":
    mflux_cli = __import__("shutil").which("mflux-generate-flux2")
    print(f"  mflux-generate-flux2:   {'✓ ' + mflux_cli if mflux_cli else 'MISSING (pip install mflux)'}")
PY

if [ "$SKIP_IMG2IMG" = "0" ]; then
  echo
  echo "[5/5] Pre-fetching SD-Turbo (~2 GB) for img2img animation chaining …"
  echo "      (pass --skip-img2img to skip; the agent still works without it,"
  echo "       but animation frames will be uncorrelated)"
  $PY - <<'PY'
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download(
        repo_id="stabilityai/sd-turbo",
        allow_patterns=[
            "*.json", "*.txt",
            "scheduler/*", "tokenizer/*", "text_encoder/*",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    )
    print(f"  ✓ SD-Turbo cached at {p}")
except Exception as e:
    print(f"  SD-Turbo download SKIPPED — {type(e).__name__}: {e!s}")
    print(f"  (the agent will retry on first img2img request)")
PY
fi

echo
echo "Now running assets.try_load_image_generator() to confirm the loader is wired:"
$PY - <<'PY'
import assets
gen = assets.try_load_image_generator()
print(f"  → {gen!r}")
if gen is None:
    print("  Still None — no local weights and/or no GPU/mflux on this host.")
else:
    kind = type(gen).__name__
    if kind == "Flux2KleinMfluxGenerator":
        print("  ✓ FLUX2-klein (mflux) selected — sprites generate via mflux-generate-flux2")
        print("    (~10-15s first image incl. load, then faster; ~13 GB peak MLX RAM)")
    else:
        print("  ✓ ready. Z-Image-Turbo loads on the first <assets> request")
        print("    (~30-60s once per session, then ~2-4s per sprite on CUDA;")
        print("     longer on MPS).")
PY
