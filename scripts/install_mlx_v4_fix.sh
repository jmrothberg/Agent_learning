#!/usr/bin/env bash
#
# Install the in-flight mlx-lm + transformers branches that add working
# DeepSeek-V4 Flash/Pro support, into whichever Python owns this
# machine's `mlx_lm.server` command.
#
# Why this exists: mlx-lm 0.31.3 (the version on PyPI as of May 2026)
# ships an *incomplete* `models/deepseek_v4.py` stub. Loading a public
# DeepSeek-V4 conversion against the stub fails with
#   ValueError: Received N parameters not in model: lm_head.weight, ...
# The full implementation (HyperConnection / Sinkhorn-Knopp / FP8 block
# dequant / HyperHead / sliced wo_a / hash-routed MoE / sqrtsoftplus
# scoring) lives in two open PRs that haven't been released yet:
#   - ml-explore/mlx-lm  PR #1192  "Add DeepSeek-v4 (Flash/Pro)"
#   - huggingface/transformers PR #45643 (DeepSeek V4 tokenizer fixes)
# This script installs both PR heads.
#
# Cross-machine note: `mlx_lm.server` may live in different Pythons on
# different machines (python.org installer Python 3.11 on this machine,
# Homebrew Python on others, conda elsewhere). We resolve via the
# `mlx_lm.server` script's shebang, so the install always lands in the
# right interpreter without hardcoded paths.
#
# Usage:
#   ./scripts/install_mlx_v4_fix.sh                  # install / re-install
#   ./scripts/install_mlx_v4_fix.sh --rollback       # back to upstream stable
#   ./scripts/install_mlx_v4_fix.sh -h | --help
#
# When upstream PRs merge and a new mlx-lm release lands on PyPI:
#   ./scripts/install_mlx_v4_fix.sh --rollback
# That `pip install --upgrade --force-reinstall mlx-lm transformers`
# pulls the new release and overwrites the git-installed version, so
# you stay current as new local models ship.

set -euo pipefail

# --- arg parsing ------------------------------------------------------------

MODE="install"
for arg in "$@"; do
    case "$arg" in
        --rollback) MODE="rollback" ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg" >&2
            sed -n '2,32p' "$0" >&2
            exit 1
            ;;
    esac
done

# --- helpers ----------------------------------------------------------------

die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
info() { printf '\033[1;36m·\033[0m %s\n' "$*"; }

# --- 1. resolve the Python that owns mlx_lm.server -------------------------

MLX_SERVER_PATH="$(command -v mlx_lm.server 2>/dev/null || true)"
if [ -z "$MLX_SERVER_PATH" ]; then
    die "mlx_lm.server not found on PATH. Install mlx-lm first:
        pip install --user mlx-lm
    Then re-run this script. (mlx-lm only works on Apple Silicon.)"
fi
info "found mlx_lm.server at: $MLX_SERVER_PATH"

SHEBANG="$(head -1 "$MLX_SERVER_PATH" 2>/dev/null || true)"
case "$SHEBANG" in
    "#!"*)
        # Strip "#!" + any leading spaces; take the first whitespace-
        # separated token in case the shebang has extra args.
        MLX_PY="${SHEBANG#\#!}"
        MLX_PY="${MLX_PY%%[[:space:]]*}"
        ;;
    *)
        die "could not parse shebang in $MLX_SERVER_PATH (got: $SHEBANG)"
        ;;
esac
[ -x "$MLX_PY" ] || die "Python from shebang not executable: $MLX_PY"
info "owning Python: $MLX_PY"
"$MLX_PY" --version

# Some macOS Python distributions force --user installs (PEP 668 marker).
# We pass --user explicitly anyway; this works for both system Python and
# user-installed Pythons. If pip reports "externally-managed-environment"
# we add --break-system-packages as a fallback.
PIP_USER_FLAG="--user"
PIP_BREAK_FLAG=""
if "$MLX_PY" -m pip install --user --dry-run pip 2>&1 | grep -q "externally-managed"; then
    warn "this Python is marked externally-managed; will add --break-system-packages"
    PIP_BREAK_FLAG="--break-system-packages"
fi

# --- 2a. rollback path ------------------------------------------------------

if [ "$MODE" = "rollback" ]; then
    info "rolling back to upstream stable mlx-lm + transformers from PyPI"
    "$MLX_PY" -m pip install $PIP_USER_FLAG $PIP_BREAK_FLAG \
        --force-reinstall --upgrade --no-cache-dir mlx-lm transformers
    ok   "rolled back. The git-installed PR heads have been replaced by PyPI."
    "$MLX_PY" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
    "$MLX_PY" -c "import transformers; print('transformers', transformers.__version__)"
    exit 0
fi

# --- 2b. install path -------------------------------------------------------

info "installing PR #1192 (mlx-lm) + PR #45643 (transformers) into:"
info "  $MLX_PY"

"$MLX_PY" -m pip install $PIP_USER_FLAG $PIP_BREAK_FLAG \
    --force-reinstall --no-cache-dir \
    "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1192/head" \
    "git+https://github.com/huggingface/transformers.git@refs/pull/45643/head"

# --- 3. verify -------------------------------------------------------------

info "verifying the V4 implementation landed (PR file should be 50KB+; the broken 0.31.3 stub was ~16KB)"
"$MLX_PY" - <<'PY'
import os
from mlx_lm.models import deepseek_v4
size = os.path.getsize(deepseek_v4.__file__)
print(f"deepseek_v4.py: {size:,} bytes at {deepseek_v4.__file__}")
# Imports that ONLY exist in the PR #1192 implementation:
need = ("HyperConnection", "HyperHead", "hc_expand")
src = open(deepseek_v4.__file__).read()
missing = [n for n in need if n not in src]
if missing:
    raise SystemExit(f"PR #1192 didn't fully land — missing symbols: {missing}")
if size < 30_000:
    raise SystemExit(f"deepseek_v4.py is suspiciously small ({size} bytes); PR #1192 not installed")
print("OK: deepseek_v4.py contains PR #1192 features")
PY

ok "DeepSeek-V4 Flash/Pro should now load. Start the server normally:"
echo "     mlx_lm.server --model /path/to/DeepSeek-V4-Flash-... --port 8080"
echo "  Or in chat.py:  /list  →  /launch <N>"
echo
info "to switch back to upstream stable later:"
echo "     ./scripts/install_mlx_v4_fix.sh --rollback"
