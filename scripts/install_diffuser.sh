#!/usr/bin/env bash
# Install the optional Z-Image-Turbo sprite-generation GPU stack into
# Agent_learning/.venv. ~5GB download + ~5GB model on first <assets> use.
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

echo "Installing Z-Image-Turbo dependencies into .venv …"
.venv/bin/pip install -r requirements-diffuser.txt

echo
echo "Verifying CUDA + diffusers are wired up …"
.venv/bin/python - <<'PY'
import importlib.util as iu
ok_torch = iu.find_spec("torch") is not None
ok_diffusers = iu.find_spec("diffusers") is not None
print(f"  torch installed:     {ok_torch}")
print(f"  diffusers installed: {ok_diffusers}")
if ok_torch:
    import torch
    print(f"  torch version:       {torch.__version__}")
    print(f"  CUDA available:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:         {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU memory:          {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")
PY

echo
echo "Now running assets.try_load_image_generator() — this should print"
echo "a ZImageTurboGenerator (NOT None) to confirm the loader is wired:"
.venv/bin/python - <<'PY'
import assets
gen = assets.try_load_image_generator()
print(f"  → {gen!r}")
if gen is None:
    print("  (still None — check the warnings above. Most likely no CUDA GPU.)")
else:
    print("  ✓ ready. Z-Image-Turbo will load on the first <assets> request.")
PY
