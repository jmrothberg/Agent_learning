#!/usr/bin/env bash
# Install the full diffusion GPU stack into Agent_learning/.venv:
#   • Z-Image-Turbo sprites (<assets>)  — torch + diffusers git + transformers…
#   • Stable Audio Open (<sounds>)      — layers requirements-diffuser.txt
#     (soundfile, torchsde, pinned transformers/accelerate/safetensors)
#
# Cross-platform: Linux (CUDA) vs macOS (MPS) vs other (CPU — slow).
# `./scripts/setup.sh` calls this once; you can also run this script alone
# after creating `.venv`.
#
# Model weights are NOT downloaded here — they populate ~/.cache/huggingface/hub/
# on first <assets>/<sounds> or scripts/_smoke_*.py (login rarely needed; README if 403).

set -euo pipefail

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
TORCH_CUDA="${TORCH_CUDA:-130}"

case "$OS" in
  Darwin)
    PLATFORM_LABEL="macOS (MPS / Apple Silicon)"
    TORCH_INDEX_FLAGS="--pre --index-url https://download.pytorch.org/whl/nightly/cpu"
    DEVICE_NOTE="Apple Silicon Macs use MPS via the same PyTorch wheel — Z-Image-Turbo on MPS is EXPERIMENTAL."
    ;;
  Linux)
    PLATFORM_LABEL="Linux + NVIDIA (CUDA $TORCH_CUDA)"
    TORCH_INDEX_FLAGS="--pre --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA}"
    DEVICE_NOTE="If your GPU needs a different CUDA version, re-run with TORCH_CUDA=121 (or 124, etc)."
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
$PIP install $TORCH_INDEX_FLAGS torch torchvision torchaudio

echo
echo "[2/4] Installing diffusers from git HEAD (needed for ZImagePipeline + Stable Audio) …"
$PIP install --upgrade git+https://github.com/huggingface/diffusers

echo
echo "[3/4] Installing transformers / accelerate / safetensors / pillow …"
$PIP install --upgrade transformers accelerate safetensors pillow

echo
echo "[4/4] Layering requirements-diffuser.txt (soundfile, torchsde, version pins) …"
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
for mod in ("soundfile", "torchsde"):
    ok = iu.find_spec(mod) is not None
    print(f"  {mod + ':':24} {'✓' if ok else 'MISSING'}")
PY

echo
echo "Now running assets.try_load_image_generator() to confirm the loader is wired:"
$PY - <<'PY'
import assets
gen = assets.try_load_image_generator()
print(f"  → {gen!r}")
if gen is None:
    print("  Still None — most likely no GPU on this host.")
else:
    print("  ✓ ready. Z-Image-Turbo will load on the first <assets> request")
    print("    (~30-60s once per session, then ~2-4s per sprite on CUDA;")
    print("     longer on MPS).")
PY
