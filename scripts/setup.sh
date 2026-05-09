#!/usr/bin/env bash
#
# One-shot setup for Agent_learning on a fresh machine.
#
#   ./scripts/setup.sh                   # sensible defaults (auto-detect GPU)
#   ./scripts/setup.sh --no-gpu          # core only, no torch/diffusers (no
#                                        #   sprites, no audio gen)
#   ./scripts/setup.sh --recreate-venv   # nuke .venv and start over (use this
#                                        #   if the venv was created in a
#                                        #   different directory and the
#                                        #   interpreter shim is now stale)
#   ./scripts/setup.sh --skip-playwright # skip `playwright install chromium`
#   ./scripts/setup.sh --skip-tests      # skip the pytest verification step
#   ./scripts/setup.sh -h | --help
#
# What this script does, in order:
#   1. Verify Python 3.10+ is on PATH (as `python3`).
#   2. Create or reuse `.venv/` in the repo root.
#   3. `pip install -r requirements.txt` (core: ollama client, playwright,
#      textual, pytest, etc).
#   4. `playwright install chromium` (the browser the harness drives).
#   5. Optional GPU stack — `./scripts/install_diffuser.sh` (torch +
#      diffusers + transformers + accelerate + safetensors + soundfile).
#      Enables BOTH sprite generation (Z-Image-Turbo) and sound generation
#      (Stable Audio Open). On Apple Silicon uses MPS; on Linux+NVIDIA
#      uses CUDA via the appropriate torch nightly.
#   6. Run the pytest suite end-to-end as a sanity check (170 tests, < 1 s).
#   7. Print a next-steps banner with the gated-model + Ollama hints.
#
# Idempotent: re-running on a healthy install is a no-op (~5 seconds).
# Cross-platform: tested on macOS (Apple Silicon + MPS) and Ubuntu Linux
# (with NVIDIA + CUDA, or CPU-only if --no-gpu).

set -euo pipefail

# --- locate repo root ------------------------------------------------------

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# --- arg parsing -----------------------------------------------------------

WITH_GPU="auto"
SKIP_PLAYWRIGHT=0
SKIP_TESTS=0
RECREATE_VENV=0

usage() {
    sed -n '2,18p' "$0"
    exit "${1:-0}"
}

for arg in "$@"; do
    case "$arg" in
        --no-gpu)             WITH_GPU="off" ;;
        --gpu|--with-gpu)     WITH_GPU="on"  ;;
        --skip-playwright)    SKIP_PLAYWRIGHT=1 ;;
        --skip-tests)         SKIP_TESTS=1 ;;
        --recreate-venv)      RECREATE_VENV=1 ;;
        -h|--help)            usage 0 ;;
        *)                    echo "unknown flag: $arg" >&2; usage 1 ;;
    esac
done

# --- helpers ---------------------------------------------------------------

# Pretty step header. `step "1/7" "Description"` prints a numbered banner
# so the user can follow which phase is in flight.
step() {
    printf '\n\033[1;36m[%s] %s\033[0m\n' "$1" "$2"
}

ok() {
    printf '   \033[1;32m✓\033[0m %s\n' "$1"
}

warn() {
    printf '   \033[1;33m!\033[0m %s\n' "$1"
}

die() {
    printf '\n\033[1;31mERROR:\033[0m %s\n' "$1" >&2
    exit 1
}

# --- platform detect -------------------------------------------------------

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos"  ;;
    Linux)  PLATFORM="linux"  ;;
    *)      PLATFORM="other"  ;;
esac

# `auto` resolves to `on` on Mac+Linux, `off` elsewhere. The user can still
# explicitly opt in or out.
if [ "$WITH_GPU" = "auto" ]; then
    case "$PLATFORM" in
        macos|linux) WITH_GPU="on" ;;
        *)           WITH_GPU="off" ;;
    esac
fi

echo "Agent_learning setup"
echo "  repo:       $ROOT"
echo "  platform:   $PLATFORM ($OS)"
echo "  GPU stack:  $WITH_GPU"
echo "  playwright: $([ $SKIP_PLAYWRIGHT -eq 0 ] && echo on || echo skipped)"
echo "  tests:      $([ $SKIP_TESTS -eq 0 ] && echo on || echo skipped)"

# --- 1. python check -------------------------------------------------------

step "1/7" "Python 3.10+ check"

if ! command -v python3 >/dev/null 2>&1; then
    if [ "$PLATFORM" = "linux" ]; then
        die "python3 not found. On Ubuntu: sudo apt install python3 python3-venv python3-pip"
    else
        die "python3 not found. Install Python 3.10+ first (https://www.python.org/downloads/)."
    fi
fi

PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="${PY_VER%.*}"
PY_MINOR="${PY_VER#*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "python3 is $PY_VER, need >= 3.10. Install a newer Python."
fi
ok "python3 $PY_VER at $(command -v python3)"

# --- 2. venv ---------------------------------------------------------------

step "2/7" "Virtual environment (.venv/)"

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

# Detect a broken venv (interpreter shim points at a path that doesn't
# exist anymore — common after the repo is moved or renamed). If broken,
# rebuild silently rather than confuse the user with a stale shim.
venv_is_broken=0
if [ -d ".venv" ] && [ ! -x "$VENV_PY" ]; then
    venv_is_broken=1
    warn ".venv/ exists but $VENV_PY is missing — venv is broken (probably moved repo). Rebuilding."
fi

if [ $RECREATE_VENV -eq 1 ] || [ $venv_is_broken -eq 1 ]; then
    if [ -d ".venv" ]; then
        rm -rf .venv
        ok "removed old .venv/"
    fi
fi

if [ ! -d ".venv" ]; then
    if ! python3 -m venv .venv 2>/tmp/venv_err.$$; then
        cat /tmp/venv_err.$$ >&2 || true
        rm -f /tmp/venv_err.$$
        if [ "$PLATFORM" = "linux" ]; then
            die "python3 -m venv failed. On Ubuntu, install: sudo apt install python3-venv"
        else
            die "python3 -m venv failed (see error above)."
        fi
    fi
    rm -f /tmp/venv_err.$$
    ok "created .venv/"
else
    ok ".venv/ already present"
fi

# Sanity-check the rebuilt venv before we trust it.
if ! "$VENV_PY" --version >/dev/null 2>&1; then
    die ".venv/bin/python is not runnable. Try: ./scripts/setup.sh --recreate-venv"
fi

# Upgrade pip itself once, quietly. Old pip mishandles some wheels; cheap
# to keep current.
"$VENV_PIP" install --upgrade pip wheel >/dev/null
ok "pip upgraded"

# --- 3. core deps ----------------------------------------------------------

step "3/7" "Core Python deps (requirements.txt)"

"$VENV_PIP" install -r requirements.txt
ok "core deps installed"

# --- 4. playwright ---------------------------------------------------------

step "4/7" "Playwright Chromium"

if [ $SKIP_PLAYWRIGHT -eq 1 ]; then
    warn "skipped (--skip-playwright). chat.py / coder.py will fail until you run: $VENV_PY -m playwright install chromium"
else
    # `playwright install chromium` is itself idempotent — it skips download
    # if the browser is already cached at ~/.cache/ms-playwright on Linux
    # or ~/Library/Caches/ms-playwright on macOS.
    "$VENV_PY" -m playwright install chromium
    ok "Chromium ready"
fi

# --- 5. optional GPU stack -------------------------------------------------

step "5/7" "GPU stack (torch + diffusers + soundfile)"

if [ "$WITH_GPU" = "off" ]; then
    warn "skipped (--no-gpu). Sprite + sound generation will be unavailable; agent runs fine without them."
else
    # install_diffuser.sh:
    #   - picks the right torch wheel index for the platform
    #   - installs diffusers from git HEAD (Z-Image-Turbo support)
    #   - installs transformers + accelerate + safetensors + pillow
    # We then layer requirements-diffuser.txt on top, which adds soundfile
    # (the OGG encoder used by sounds.py) and re-asserts the upper-bound
    # versions for transformers / accelerate / safetensors.
    bash ./scripts/install_diffuser.sh
    "$VENV_PIP" install -r requirements-diffuser.txt
    ok "GPU stack installed"
fi

# --- 6. tests --------------------------------------------------------------

step "6/7" "Test suite (pytest)"

if [ $SKIP_TESTS -eq 1 ]; then
    warn "skipped (--skip-tests)"
else
    "$VENV_PY" -m pytest tests/ -q
    ok "all tests passed"
fi

# --- 7. next steps ---------------------------------------------------------

step "7/7" "Next steps"

cat <<EOF

   Activate the venv (or use the absolute path to its python):
     source .venv/bin/activate
     # or just: .venv/bin/python chat.py

   Make sure Ollama is running with at least one model:
     ollama serve &        # in a separate terminal
     ollama list           # confirm a usable tag is available
     # optional: warm a model at the default 32K context
     ollama run --ctx-size 32768 qwen3.6:35b

   Or use MLX on Apple Silicon (often faster than Ollama):
     mlx_lm.server --model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit --port 8080

   Run it:
     .venv/bin/python chat.py                        # TUI (recommended)
     .venv/bin/python coder.py "build snake"          # one-shot CLI

EOF

if [ "$WITH_GPU" = "on" ]; then
    cat <<'EOF'
   GENERATED ASSETS — sprite art (Z-Image-Turbo, ~5 GB):
     Auto-downloads from HuggingFace on first <assets> request. No license
     gating. Smoke-test with:
       .venv/bin/python scripts/_smoke_doom.py

   GENERATED SOUNDS — Stable Audio Open 1.0 (~5 GB):
     Gated model. Two one-time steps before the first <sounds> request:
       1. Accept the license at:
            https://huggingface.co/stabilityai/stable-audio-open-1.0
       2. Authenticate the CLI so diffusers can download:
            .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
            # (or: export HF_TOKEN=hf_xxxxxxxxxxxx)
     Smoke-test with:
       .venv/bin/python scripts/_smoke_audio.py

EOF
fi

# --- macOS / Apple Silicon: MLX wired-memory hint --------------------------
# `mlx_lm.server`'s default Metal allocator cap (`iogpu.wired_limit_mb`) is
# typically ~75% of physical RAM. On big models with long context the
# weights + KV cache can exceed it; the generate thread then dies with
# `[metal::malloc] Resource limit (...) exceeded.` and the HTTP layer
# silently keeps responding to /v1/models with no tokens flowing. Show a
# RAM-aware suggested override here so the user can sysctl it before
# starting mlx_lm.server. We don't run sudo for them — the value persists
# only until reboot, so the user is already going to redo it after every
# reboot, and surfacing the command is just as good.
if [ "$PLATFORM" = "macos" ] && command -v sysctl >/dev/null 2>&1; then
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    if [ "$RAM_BYTES" -gt 0 ]; then
        # MB = bytes / 1024 / 1024.   Recommended cap = RAM_MB - 16384
        # (leave 16 GB for the OS + apps). Use bash arithmetic so no
        # awk/python dependency creeps in.
        RAM_MB=$(( RAM_BYTES / 1024 / 1024 ))
        REC_LIMIT_MB=$(( RAM_MB - 16384 ))
        if [ "$REC_LIMIT_MB" -lt 16384 ]; then
            REC_LIMIT_MB=$REC_LIMIT_MB
        fi
        # Display total in human-friendly GB, rounded to 1 decimal.
        RAM_GB=$(( (RAM_MB + 512) / 1024 ))
        cat <<EOF
   MLX on Apple Silicon — raise the Metal wired-memory cap before
   starting mlx_lm.server (per-boot; needs sudo):
     sudo sysctl iogpu.wired_limit_mb=${REC_LIMIT_MB}    # for your ${RAM_GB} GB Mac
   See README §MLX memory limit on Apple Silicon for why and how to persist.

EOF
    fi
fi

echo "   Setup complete."
