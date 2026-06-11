#!/usr/bin/env bash
#
# One-shot setup for Agent_learning on a fresh machine.
#
#   ./scripts/setup.sh                   # DEFAULT: core + Chromium + mlx-lm (arm64 Mac)
#                                        #         + FULL GPU stack (torch/diffusers/sprites/audio)
#                                        #         + video cutscenes (Wan2.2 — mlx-gen on Mac)
#   ./scripts/setup.sh --no-gpu          # rare: skip torch/diffusers (no Z-Image / Stable Audio)
#   ./scripts/setup.sh --no-video        # skip the video-generation stack (<videos> disabled)
#   ./scripts/setup.sh --recreate-venv   # nuke .venv and start over (use this
#                                        #   if the venv was created in a
#                                        #   different directory and the
#                                        #   interpreter shim is now stale)
#   ./scripts/setup.sh --skip-playwright # skip `playwright install chromium`
#   ./scripts/setup.sh --skip-tests      # skip the pytest verification step
#   ./scripts/setup.sh --non-interactive # no prompts; auto-detect weights only
#   ./scripts/setup.sh --no-mlx-tools    # Apple Silicon: skip mlx-lm pip install
#   ./scripts/setup.sh -h | --help
#
# What this script does, in order:
#   1. Verify Python 3.10+ is on PATH (as `python3`).
#   2. Create or reuse `.venv/` in the repo root.
#   3. `pip install -r requirements.txt` (core: ollama client, playwright,
#      textual, pytest, etc).
#   4. On macOS arm64 (unless --no-mlx-tools): `pip install -r requirements-mlx.txt`
#      so `mlx_lm.server` is available for the MLX backend.
#   5. `env -u PLAYWRIGHT_BROWSERS_PATH playwright install chromium` — browser
#      cache goes to ~/Library/Caches/ms-playwright (Mac) or ~/.cache/ms-playwright.
#   6. GPU stack — `./scripts/install_diffuser.sh` installs torch + diffusers git +
#      transformers AND layers `requirements-diffuser.txt` (soundfile, torchsde)
#      in one script — Z-Image sprites + Stable Audio Open pip deps together.
#      Omit with `--no-gpu`. Then prompts for Z-Image-Turbo weights (or downloads
#      from HuggingFace when you press Enter with no local copy) and writes
#      DIFFUSION_MODELS_DIR into `.env` for chat.py / coder.py.
#   7. Video cutscenes (<videos> tag, Wan2.2-TI2V-5B) — on macOS arm64 creates
#      the dedicated `.venv-video/` and pip-installs mlx-gen (weights ~17 GB
#      lazy-download to the HF cache on first clip). On Ubuntu/Linux the
#      diffusers stack from step 6 already covers Wan (~25 GB lazy-download on
#      first <videos>). Omit with `--no-video` (auto-off under `--no-gpu`).
#   8. Run the pytest suite end-to-end as a sanity check (~190 tests, < 20 s).
#   9. Print MLX / Ollama next-steps + HF download recovery (only if 403/401).
#
# Idempotent: re-running on a healthy install is a no-op (~5 seconds).
# Cross-platform: tested on macOS (Apple Silicon + MPS) and Ubuntu Linux
# (with NVIDIA + CUDA; Apple Silicon + MPS; use --no-gpu only when neither applies).

set -euo pipefail

# --- locate repo root ------------------------------------------------------

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# --- arg parsing -----------------------------------------------------------

WITH_GPU="auto"
WITH_VIDEO="auto"
SKIP_PLAYWRIGHT=0
SKIP_TESTS=0
RECREATE_VENV=0
MLX_TOOLS_SKIP=0
NONINTERACTIVE=0

usage() {
    sed -n '2,17p' "$0"
    exit "${1:-0}"
}

for arg in "$@"; do
    case "$arg" in
        --no-gpu)             WITH_GPU="off" ;;
        --gpu|--with-gpu)     WITH_GPU="on"  ;;
        --no-video)           WITH_VIDEO="off" ;;
        --skip-playwright)    SKIP_PLAYWRIGHT=1 ;;
        --skip-tests)         SKIP_TESTS=1 ;;
        --recreate-venv)      RECREATE_VENV=1 ;;
        --no-mlx-tools)       MLX_TOOLS_SKIP=1 ;;
        --non-interactive)    NONINTERACTIVE=1 ;;
        -h|--help)            usage 0 ;;
        *)                    echo "unknown flag: $arg" >&2; usage 1 ;;
    esac
done

# Pipelines / CI have no TTY — skip weight prompts unless forced interactive.
if [ ! -t 0 ]; then
    NONINTERACTIVE=1
fi

# --- helpers ---------------------------------------------------------------

# Pretty step header. `step "1/9" "Description"` prints a numbered banner
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

# Return 0 when the venv's Playwright Chromium binary is on disk.
# IDE sandboxes sometimes set PLAYWRIGHT_BROWSERS_PATH to a temp dir, so
# callers must `env -u PLAYWRIGHT_BROWSERS_PATH` when invoking this.
playwright_chromium_ready() {
    env -u PLAYWRIGHT_BROWSERS_PATH "$VENV_PY" - <<'PY' >/dev/null 2>&1
import os, sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    exe = p.chromium.executable_path
    if not os.path.isfile(exe):
        sys.exit(1)
PY
}

# Upsert one KEY=VALUE line in repo-root .env (gitignored; chat.py loads it).
write_env_var() {
    local key="$1"
    local val="$2"
    "$VENV_PY" - "$key" "$val" "$ROOT/.env" <<'PY'
import sys
from pathlib import Path
key, val, path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
lines = path.read_text().splitlines() if path.exists() else []
out = [ln for ln in lines if not ln.startswith(key + "=")]
out.append(f"{key}={val}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(out) + "\n")
PY
}

# Print candidate DIFFUSION_MODELS_DIR roots that already contain Z-Image-Turbo.
find_zimage_roots() {
    local bases=()
    if [ "$PLATFORM" = "linux" ] && [ -d "/data/Diffusion_Models" ]; then
        bases+=("/data/Diffusion_Models")
    fi
    if [ "$PLATFORM" = "macos" ]; then
        bases+=(
            "$HOME/Diffusion_Models"
            "$HOME/.Diffusion_Models"
            "$HOME/Models_Diffusers"
            "$HOME/.Models_Diffusers"
        )
    else
        bases+=(
            "$HOME/.Models_Diffusers"
            "$HOME/Models_Diffusers"
            "$HOME/.Diffusion_Models"
            "$HOME/Diffusion_Models"
        )
    fi
    local base
    for base in "${bases[@]}"; do
        if [ -f "$base/Z-Image-Turbo/model_index.json" ]; then
            printf '%s\n' "$base"
        fi
    done
}

# Resolve a user-entered path to the parent dir for DIFFUSION_MODELS_DIR.
# Accepts either .../Z-Image-Turbo or its parent (.../Diffusion_Models).
resolve_diffusion_models_dir() {
    local raw="${1:-}"
    raw="${raw/#\~/$HOME}"
    raw="${raw%/}"
    if [ -z "$raw" ]; then
        return 1
    fi
    if [ -f "$raw/model_index.json" ]; then
        dirname "$raw"
        return 0
    fi
    if [ -f "$raw/Z-Image-Turbo/model_index.json" ]; then
        printf '%s\n' "$raw"
        return 0
    fi
    return 1
}

download_zimage_turbo() {
    local parent="$1"
    mkdir -p "$parent"
    warn "Downloading Z-Image-Turbo from HuggingFace (~32 GB one-time) …"
    DIFFUSION_DOWNLOAD_ROOT="$parent" "$VENV_PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
root = os.path.expanduser(os.environ["DIFFUSION_DOWNLOAD_ROOT"])
path = snapshot_download(
    "Tongyi-MAI/Z-Image-Turbo",
    local_dir=os.path.join(root, "Z-Image-Turbo"),
)
print(path)
PY
}

configure_diffusion_weights() {
    if [ "$WITH_GPU" != "on" ]; then
        return 0
    fi

    local detected_root=""
    detected_root="$(find_zimage_roots | head -1 || true)"

    # Respect an existing .env unless we are interactive and user picks anew.
    if [ -f "$ROOT/.env" ] && grep -q '^DIFFUSION_MODELS_DIR=' "$ROOT/.env" 2>/dev/null; then
        local existing
        existing="$(grep '^DIFFUSION_MODELS_DIR=' "$ROOT/.env" | tail -1 | cut -d= -f2-)"
        if [ -n "$existing" ] && [ -f "${existing/#\~/$HOME}/Z-Image-Turbo/model_index.json" ]; then
            ok "DIFFUSION_MODELS_DIR already in .env → ${existing}"
            return 0
        fi
    fi

    if [ "$NONINTERACTIVE" -eq 1 ]; then
        if [ -n "$detected_root" ]; then
            write_env_var "DIFFUSION_MODELS_DIR" "$detected_root"
            ok "DIFFUSION_MODELS_DIR=$detected_root (auto-detected → .env)"
        else
            warn "no local Z-Image-Turbo found (re-run interactively to download)"
        fi
        return 0
    fi

    printf '\n'
    echo "   Z-Image-Turbo sprite weights"
    echo "   Stable Audio Open uses the same root (or HuggingFace on first <sounds>)."
    if [ -n "$detected_root" ]; then
        echo "   Local copy found: $detected_root/Z-Image-Turbo"
        printf '   Path to Z-Image-Turbo directory [Enter = use local copy]: '
    else
        echo "   No local copy found on this machine."
        printf '   Path to Z-Image-Turbo directory [Enter = download from HuggingFace]: '
    fi
    local zpath=""
    read -r zpath || zpath=""
    zpath="${zpath/#\~/$HOME}"

    local models_dir=""
    if [ -z "$zpath" ]; then
        if [ -n "$detected_root" ]; then
            models_dir="$detected_root"
        else
            local dl_root="$HOME/Models_Diffusers"
            if [ "$PLATFORM" = "macos" ]; then
                dl_root="$HOME/Diffusion_Models"
            fi
            download_zimage_turbo "$dl_root"
            models_dir="$dl_root"
        fi
    else
        if ! models_dir="$(resolve_diffusion_models_dir "$zpath")"; then
            warn "not a Z-Image-Turbo tree (need model_index.json) — skipping .env"
            warn "chat.py will download from HuggingFace on first <assets> instead."
            return 0
        fi
    fi

    write_env_var "DIFFUSION_MODELS_DIR" "$models_dir"
    ok "DIFFUSION_MODELS_DIR=$models_dir → .env (chat.py will use local weights)"
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

# Video cutscene stack (<videos> tag, Wan2.2). Follows the GPU switch:
# auto-on when the GPU stack is on, forced off under --no-gpu (the Linux
# backend rides on torch/diffusers from step 6; the Mac backend is
# pointless without the rest of the media stack anyway).
if [ "$WITH_VIDEO" = "auto" ]; then
    WITH_VIDEO="$WITH_GPU"
fi
if [ "$WITH_GPU" = "off" ]; then
    WITH_VIDEO="off"
fi

MLX_TOOLS=0
if [ "$PLATFORM" = "macos" ] && [ "$(uname -m)" = "arm64" ] && [ "$MLX_TOOLS_SKIP" -eq 0 ]; then
    MLX_TOOLS=1
fi

echo "Agent_learning setup"
echo "  repo:       $ROOT"
echo "  platform:   $PLATFORM ($OS)"
echo "  GPU stack:  $WITH_GPU"
echo "  video gen:  $WITH_VIDEO"
echo "  mlx-lm:     $([ $MLX_TOOLS -eq 1 ] && echo on || echo off)"
echo "  playwright: $([ $SKIP_PLAYWRIGHT -eq 0 ] && echo on || echo skipped)"
echo "  tests:      $([ $SKIP_TESTS -eq 0 ] && echo on || echo skipped)"

# --- 1. python check -------------------------------------------------------

step "1/9" "Python 3.10+ check"

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

step "2/9" "Virtual environment (.venv/)"

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

step "3/9" "Core Python deps (requirements.txt)"

"$VENV_PIP" install -r requirements.txt
ok "core deps installed"

# --- 4. mlx-lm (Apple Silicon arm64) ---------------------------------------

step "4/9" "MLX server package (requirements-mlx.txt)"

if [ $MLX_TOOLS -eq 1 ]; then
    "$VENV_PIP" install -r requirements-mlx.txt
    ok "mlx-lm installed (mlx_lm.server CLI in this venv)"
fi

# --- 5. playwright ---------------------------------------------------------

step "5/9" "Playwright Chromium"

if [ $SKIP_PLAYWRIGHT -eq 1 ]; then
    warn "skipped (--skip-playwright). Run when ready: env -u PLAYWRIGHT_BROWSERS_PATH $VENV_PY -m playwright install chromium"
else
    # IDE sandboxes sometimes set PLAYWRIGHT_BROWSERS_PATH to a temp dir;
    # unset so browsers land in the normal OS cache (~/.cache/ms-playwright).
    if playwright_chromium_ready; then
        ok "Chromium already cached"
    else
        warn "Chromium missing — downloading (first run or after playwright upgrade) …"
        env -u PLAYWRIGHT_BROWSERS_PATH "$VENV_PY" -m playwright install chromium
    fi
    if ! playwright_chromium_ready; then
        die "Chromium install failed. Re-run: env -u PLAYWRIGHT_BROWSERS_PATH $VENV_PY -m playwright install chromium"
    fi
    ok "Chromium ready"
    if [ "$PLATFORM" = "linux" ]; then
        warn "If launch fails with missing .so libs: sudo env -u PLAYWRIGHT_BROWSERS_PATH $VENV_PY -m playwright install-deps chromium"
    fi
fi

# --- 6. optional GPU stack -------------------------------------------------

step "6/9" "GPU stack — torch + diffusers + sprites + audio (skip with --no-gpu)"

if [ "$WITH_GPU" = "off" ]; then
    warn "skipped (--no-gpu). No Z-Image / Stable Audio — sprite + sound generation unavailable."
else
    # Match PyTorch wheel CUDA tag to the installed NVIDIA driver unless the
    # caller already set TORCH_CUDA. Nightly cu130 needs CUDA 13.x driver;
    # driver 12.x (e.g. 570 + CUDA 12.8) needs stable cu126 wheels.
    if [ -z "${TORCH_CUDA:-}" ] && [ "$PLATFORM" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
        _cuda_drv_major="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.[0-9]*/\1/p' | head -1)"
        case "${_cuda_drv_major:-}" in
            12) export TORCH_CUDA=126 ;;
            13) export TORCH_CUDA=130 ;;
        esac
        if [ -n "${TORCH_CUDA:-}" ]; then
            ok "auto TORCH_CUDA=${TORCH_CUDA} from nvidia-smi (override: TORCH_CUDA=124 ./scripts/setup.sh)"
        fi
    fi
    # One script: torch + diffusers + transformers + requirements-diffuser.txt
    # (soundfile, torchsde) — sprites AND Stable Audio Open deps together.
    bash ./scripts/install_diffuser.sh
    ok "GPU stack installed (sprites + sound pip deps)"
    configure_diffusion_weights
fi

# --- 7. video cutscenes (Wan2.2) -------------------------------------------

step "7/9" "Video cutscenes — Wan2.2-TI2V-5B (skip with --no-video)"

if [ "$WITH_VIDEO" = "off" ]; then
    warn "skipped (--no-video / --no-gpu). The <videos> tag will be a silent no-op."
elif [ "$PLATFORM" = "macos" ] && [ "$(uname -m)" = "arm64" ]; then
    # mlx-gen lives in its OWN venv: it pins its own mlx version, which
    # must never fight the agent's mlx-lm pin in the main .venv.
    VIDEO_VENV="$ROOT/.venv-video"
    if [ ! -x "$VIDEO_VENV/bin/pip" ]; then
        python3 -m venv "$VIDEO_VENV"
        ok "created dedicated video venv at .venv-video/"
    fi
    "$VIDEO_VENV/bin/pip" install -q -U pip mlx-gen
    if [ -x "$VIDEO_VENV/bin/mlxgen" ]; then
        ok "mlx-gen installed — model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit (~17 GB) lazy-downloads on first clip"
    else
        warn "mlx-gen install did not produce .venv-video/bin/mlxgen — <videos> will be a no-op until fixed"
    fi
elif [ "$PLATFORM" = "linux" ]; then
    # Nothing extra to install: the diffusers stack from step 6 covers
    # Wan2.2 (WanPipeline). Weights lazy-download on first use.
    ok "Linux uses the step-6 diffusers stack — Wan-AI/Wan2.2-TI2V-5B-Diffusers (~25 GB) lazy-downloads to the HF cache on first <videos>"
else
    warn "no supported video backend on this platform — <videos> will be a silent no-op."
fi

# --- 8. tests --------------------------------------------------------------

step "8/9" "Test suite (pytest)"

if [ $SKIP_TESTS -eq 1 ]; then
    warn "skipped (--skip-tests)"
else
    "$VENV_PY" -m pytest tests/ -q
    ok "all tests passed"
fi

# --- 9. next steps ---------------------------------------------------------

step "9/9" "Next steps"

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
     mlx_lm.server --model /Users/jonathanrothberg_1/MLX_Models/Qwen3.6-27B-mxfp8 --port 8080

   Want to run DeepSeek-V4 Flash / Pro? mlx-lm 0.31.3 ships a broken stub.
   Patch your existing mlx_lm install (auto-detects which Python owns it):
     ./scripts/install_mlx_v4_fix.sh
   See README §"DeepSeek-V4 on MLX" for details + rollback.

   Run it:
     .venv/bin/python chat.py                        # TUI (recommended)
     .venv/bin/python coder.py "build snake"          # one-shot CLI

EOF

if [ "$WITH_GPU" = "on" ]; then
    cat <<'EOF'
   SPRITE WEIGHTS — Z-Image-Turbo:
     setup.sh writes DIFFUSION_MODELS_DIR to .env when a local tree exists
     (Linux: /data/Diffusion_Models, Mac: ~/Diffusion_Models, etc.).
     Re-run ./scripts/setup.sh interactively to pick a path or download (~32 GB).
       .venv/bin/python scripts/_smoke_doom.py

   GENERATED SOUNDS — Stable Audio Open 1.0 (~9 GB cache typical):
     Downloads on first <sounds> or smoke unless cached under DIFFUSION_MODELS_DIR.
       .venv/bin/python scripts/_smoke_audio.py
     Only if download fails with 403/401:
       • https://huggingface.co/stabilityai/stable-audio-open-1.0 — agree if prompted
       • .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
         (or export HF_TOKEN=hf_…)

EOF
fi

if [ "$WITH_VIDEO" = "on" ]; then
    cat <<'EOF'
   VIDEO CUTSCENES — Wan2.2-TI2V-5B (<videos> tag, ~3 min per 4s clip):
     Weights lazy-download to the HF cache on the first clip.
     Try it standalone (T2V; add --image <png> for image-to-video):
       .venv/bin/python scripts/generate_video.py \
           --prompt "a knight runs across a collapsing drawbridge, cinematic" \
           --out /tmp/test_clip.mp4

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
