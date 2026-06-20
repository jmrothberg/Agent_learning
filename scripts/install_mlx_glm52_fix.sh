#!/usr/bin/env bash
#
# Install mlx-lm support for GLM-5.2 (glm_moe_dsa) DSA cross-layer indexer
# sharing (IndexShare). PyPI mlx-lm 0.31.3 maps glm_moe_dsa to a bare
# deepseek_v32 subclass and builds an indexer on every layer; GLM-5.2 only
# has indexer weights on "full" layers and shares top-k on "shared" layers.
# Loading without the fix fails with:
#   ValueError: Missing 285 parameters: model.layers.11.self_attn.indexer...
#
# Upstream fix: ml-explore/mlx-lm PR #1410
#   https://github.com/ml-explore/mlx-lm/pull/1410
#
# This script checks whether #1410 has merged:
#   - merged + fix on PyPI  → pip install mlx-lm from PyPI
#   - merged, not on PyPI yet → pip install mlx-lm @ main
#   - still open            → pip install @ refs/pull/1410/head
#
# Python resolution (in order):
#   1. This repo's .venv/bin/python (if present)
#   2. Shebang of `mlx_lm.server` on PATH (same as install_mlx_v4_fix.sh)
#
# Usage:
#   ./scripts/install_mlx_glm52_fix.sh                  # install / re-install
#   ./scripts/install_mlx_glm52_fix.sh --status           # merge + install check only
#   ./scripts/install_mlx_glm52_fix.sh --rollback       # back to upstream PyPI
#   ./scripts/install_mlx_glm52_fix.sh -h | --help
#
# After any mlx-lm reinstall, re-copy minimax_m3.py from your MiniMax model
# dir if you use MiniMax-M3 (see README "MLX upgrades — MiniMax-M3").

set -euo pipefail

PR_NUM=1410
REPO="ml-explore/mlx-lm"
PR_URL="https://github.com/${REPO}/pull/${PR_NUM}"

# --- arg parsing ------------------------------------------------------------

MODE="install"
for arg in "$@"; do
    case "$arg" in
        --rollback) MODE="rollback" ;;
        --status)   MODE="status" ;;
        -h|--help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg" >&2
            sed -n '2,28p' "$0" >&2
            exit 1
            ;;
    esac
done

# --- helpers ----------------------------------------------------------------

die()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
info() { printf '\033[1;36m·\033[0m %s\n' "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_VENV_PY="${SCRIPT_DIR}/../.venv/bin/python"

resolve_mlx_python() {
    if [ -x "$REPO_VENV_PY" ]; then
        if "$REPO_VENV_PY" -c "import mlx_lm" >/dev/null 2>&1; then
            echo "$REPO_VENV_PY"
            return 0
        fi
        warn "repo .venv exists but mlx_lm not importable; trying mlx_lm.server"
    fi

    local mlx_server_path
    mlx_server_path="$(command -v mlx_lm.server 2>/dev/null || true)"
    if [ -z "$mlx_server_path" ]; then
        die "no mlx_lm Python found. Install mlx-lm first:
        python3 -m venv .venv && .venv/bin/pip install mlx-lm
    or: pip install --user mlx-lm"
    fi
    info "found mlx_lm.server at: $mlx_server_path"

    local shebang mlx_py
    shebang="$(head -1 "$mlx_server_path" 2>/dev/null || true)"
    case "$shebang" in
        "#!"*)
            mlx_py="${shebang#\#!}"
            mlx_py="${mlx_py%%[[:space:]]*}"
            ;;
        *)
            die "could not parse shebang in $mlx_server_path (got: $shebang)"
            ;;
    esac
    [ -x "$mlx_py" ] || die "Python from shebang not executable: $mlx_py"
    echo "$mlx_py"
}

# Returns "true" or "false" on stdout.
pr_merged() {
    if command -v gh >/dev/null 2>&1; then
        local merged
        merged="$(gh pr view "$PR_NUM" -R "$REPO" --json merged -q .merged 2>/dev/null || true)"
        if [ -n "$merged" ]; then
            echo "$merged"
            return 0
        fi
        warn "gh pr view failed; falling back to GitHub API"
    fi
    local json
    json="$(curl -fsSL "https://api.github.com/repos/${REPO}/pulls/${PR_NUM}" 2>/dev/null)" \
        || die "could not query GitHub for PR #${PR_NUM} (install gh or check network)"
    python3 - <<'PY' "$json"
import json, sys
d = json.loads(sys.argv[1])
print("true" if d.get("merged_at") else "false")
PY
}

verify_glm52_fix() {
    # Exit 0 when PR #1410 features are present; non-zero otherwise.
    "$MLX_PY" - <<'PY'
import os
import sys
try:
    from mlx_lm.models import glm_moe_dsa
except ImportError as e:
    print(f"import failed: {e}")
    sys.exit(1)

path = glm_moe_dsa.__file__
size = os.path.getsize(path)
src = open(path, encoding="utf-8").read()
need = (
    "GlmMoeDsaAttention",
    "indexer_types",
    "skip_topk",
    "prev_topk_indices",
    "make_cache",
)
missing = [sym for sym in need if sym not in src]
if missing:
    print(f"glm_moe_dsa.py missing PR #1410 symbols: {missing}")
    print(f"  file: {path} ({size} bytes)")
    sys.exit(1)
if size < 8_000:
    print(f"glm_moe_dsa.py too small ({size} bytes); still the bare dsv32 stub")
    sys.exit(1)
print(f"OK: glm_moe_dsa.py has IndexShare support ({size:,} bytes)")
print(f"    {path}")
PY
}

pip_install() {
    "$MLX_PY" -m pip install $PIP_USER_FLAG $PIP_BREAK_FLAG "$@"
}

# --- 1. resolve Python ------------------------------------------------------

MLX_PY="$(resolve_mlx_python)"
info "target Python: $MLX_PY"
"$MLX_PY" --version

PIP_USER_FLAG="--user"
PIP_BREAK_FLAG=""
if "$MLX_PY" -m pip install --user --dry-run pip 2>&1 | grep -q "externally-managed"; then
    warn "this Python is marked externally-managed; will add --break-system-packages"
    PIP_BREAK_FLAG="--break-system-packages"
fi

# --- 2. merge status --------------------------------------------------------

MERGED="$(pr_merged)"
if [ "$MERGED" = "true" ]; then
    info "PR #${PR_NUM} is merged into ${REPO}"
else
    info "PR #${PR_NUM} is still open (draft): ${PR_URL}"
fi

# --- 3. status-only mode ----------------------------------------------------

if [ "$MODE" = "status" ]; then
    if verify_glm52_fix; then
        ok "installed mlx-lm already has GLM-5.2 IndexShare support"
    else
        warn "installed mlx-lm does NOT have GLM-5.2 IndexShare support"
        if [ "$MERGED" = "true" ]; then
            info "PR merged — run without --status to install from PyPI (or main)"
        else
            info "PR open — run without --status to install PR head"
        fi
        exit 1
    fi
    "$MLX_PY" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
    exit 0
fi

# --- 4a. rollback -------------------------------------------------------------

if [ "$MODE" = "rollback" ]; then
    info "rolling back to upstream stable mlx-lm from PyPI"
    pip_install --force-reinstall --upgrade --no-cache-dir mlx-lm
    ok "rolled back to PyPI mlx-lm"
    "$MLX_PY" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
    if verify_glm52_fix; then
        ok "PyPI release already includes GLM-5.2 IndexShare (no PR branch needed)"
    else
        warn "PyPI release is still the bare glm_moe_dsa stub — GLM-5.2 will not load"
    fi
    exit 0
fi

# --- 4b. install ------------------------------------------------------------

if verify_glm52_fix >/dev/null 2>&1; then
    ok "GLM-5.2 IndexShare support already installed — nothing to do"
    "$MLX_PY" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"
    exit 0
fi

if [ "$MERGED" = "true" ]; then
    info "PR merged — trying PyPI mlx-lm first"
    pip_install --force-reinstall --upgrade --no-cache-dir mlx-lm
    if verify_glm52_fix >/dev/null 2>&1; then
        ok "installed GLM-5.2 support from PyPI"
    else
        warn "PyPI mlx-lm does not include the fix yet — installing ${REPO} @ main"
        pip_install --force-reinstall --no-cache-dir \
            "git+https://github.com/${REPO}.git@main"
    fi
else
    info "installing PR #${PR_NUM} head into: $MLX_PY"
    pip_install --force-reinstall --no-cache-dir \
        "git+https://github.com/${REPO}.git@refs/pull/${PR_NUM}/head"
fi

# --- 5. verify ----------------------------------------------------------------

info "verifying glm_moe_dsa IndexShare implementation"
verify_glm52_fix || die "install finished but GLM-5.2 fix verification failed"
"$MLX_PY" -c "import mlx_lm; print('mlx_lm', mlx_lm.__version__)"

ok "GLM-5.2 (glm_moe_dsa) should now load."
echo
info "example:"
echo "     MLX_MODEL=/path/to/GLM-5.2-MLX-4bit .venv/bin/python chat.py"
echo
info "if you use MiniMax-M3, re-copy minimax_m3.py into mlx_lm/models/ after this"
info "to switch back to upstream PyPI later:"
echo "     ./scripts/install_mlx_glm52_fix.sh --rollback"
