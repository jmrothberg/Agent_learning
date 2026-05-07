#!/usr/bin/env bash
# Install the optional Z-Image-Turbo sprite-generation GPU stack into
# Agent_learning/.venv. Re-uses pip's wheel cache so if you already
# have these versions in another venv on this machine, it's fast.
#
# After this completes, sessions whose Phase A includes an <assets>
# block will generate real PNG sprites in-process (no server) and
# inject the paths into the first-build prompt.
#
# Skip this if you don't have a CUDA GPU, or just don't want sprite
# generation — the agent runs fine without it (procedural ctx.fillRect
# drawing, like every prior version).

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "ERROR: no .venv/ in $(pwd) — run setup.sh / pip install first."
  exit 1
fi

PIP=".venv/bin/pip"
PY=".venv/bin/python"

# ---------------------------------------------------------------------------
# Z-Image-Turbo support is recent — only the diffusers git HEAD knows
# about ZImagePipeline (no tagged release as of 2026-05-06). torch
# nightly with CUDA 13 is what current GPUs (Blackwell, etc.) need.
# These versions match the user's working Colossal_Cave setup.
# ---------------------------------------------------------------------------

echo "[1/3] Installing torch nightly (cu130) — re-uses wheel cache if present …"
$PIP install --pre \
  --index-url https://download.pytorch.org/whl/nightly/cu130 \
  torch torchvision torchaudio

echo
echo "[2/3] Installing diffusers from git HEAD (needed for ZImagePipeline) …"
$PIP install --upgrade git+https://github.com/huggingface/diffusers

echo
echo "[3/3] Installing transformers / accelerate / safetensors / pillow …"
$PIP install --upgrade transformers accelerate safetensors pillow

echo
echo "Verifying CUDA + diffusers + ZImagePipeline are wired up …"
$PY - <<'PY'
import importlib.util as iu
ok_torch = iu.find_spec("torch") is not None
ok_diffusers = iu.find_spec("diffusers") is not None
print(f"  torch installed:        {ok_torch}")
print(f"  diffusers installed:    {ok_diffusers}")
if ok_torch:
    import torch
    print(f"  torch version:          {torch.__version__}")
    print(f"  CUDA available:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:            {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU memory free/total:  {free/1e9:.1f}GB / {total/1e9:.1f}GB")
if ok_diffusers:
    try:
        from diffusers import ZImagePipeline
        print(f"  ZImagePipeline class:   importable ✓")
    except Exception as e:
        print(f"  ZImagePipeline class:   IMPORT FAILED — {e!r}")
PY

echo
echo "Now running assets.try_load_image_generator() to confirm the loader is wired:"
$PY - <<'PY'
import assets
gen = assets.try_load_image_generator()
print(f"  → {gen!r}")
if gen is None:
    print("  Still None — see warnings above. Most likely no CUDA GPU on this host.")
else:
    print("  ✓ ready. Z-Image-Turbo will load on the first <assets> request")
    print("    (~30-60s once per session, then ~2-4s per sprite).")
PY
