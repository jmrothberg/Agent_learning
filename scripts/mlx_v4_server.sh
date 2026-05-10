#!/usr/bin/env bash
#
# Start mlx_lm.server on a DeepSeek-V4 model with the correct flags.
#
#   ./scripts/mlx_v4_server.sh                     # auto-pick first V4 model in ~/MLX_Models
#   ./scripts/mlx_v4_server.sh <model_path>        # explicit path
#
# The only thing this script does that bare `mlx_lm.server` doesn't:
# pass `--prefill-step-size 512`. Without that flag, V4 (Flash or Pro)
# blows up the moment any single prefill chunk exceeds ~1.3K tokens
# because its Indexer attention path materializes a single Metal buffer
# of size O(L^2 * k) — see PR ml-explore/mlx-lm#1192 review thread for
# the cubic-growth analysis. With chunk = 512 you stay well under the
# cliff regardless of prompt length.
#
# Default port 8080 matches mlx_lm.server's default and what backend.py
# probes when no MLX_HOST env is set. Override via the PORT env var.

set -euo pipefail

PORT="${PORT:-8080}"
MLX_DIR="${MLX_MODELS_DIR:-$HOME/MLX_Models}"

resolve_model() {
    if [ "$#" -ge 1 ] && [ -n "$1" ]; then
        echo "$1"
        return
    fi
    # Auto-pick: first directory under ~/MLX_Models whose config.json
    # says model_type=deepseek_v4. Cross-machine — we don't hardcode
    # quant names like "mxfp8" or "8bit".
    for d in "$MLX_DIR"/*/; do
        cfg="$d/config.json"
        [ -f "$cfg" ] || continue
        if grep -q '"model_type"[[:space:]]*:[[:space:]]*"deepseek_v4"' "$cfg" 2>/dev/null; then
            echo "${d%/}"
            return
        fi
    done
    echo "ERROR: no DeepSeek-V4 model found under $MLX_DIR" >&2
    echo "       (looking for any subdirectory whose config.json says model_type=deepseek_v4)" >&2
    echo "       Pass an explicit path: $0 /path/to/your/V4-model" >&2
    exit 1
}

MODEL="$(resolve_model "$@")"

if [ ! -d "$MODEL" ]; then
    echo "ERROR: model directory does not exist: $MODEL" >&2
    exit 1
fi
if [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: not a model directory (no config.json): $MODEL" >&2
    exit 1
fi

echo "→ starting mlx_lm.server on $MODEL"
echo "  port=$PORT  prefill-step-size=512"
echo
exec mlx_lm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --prefill-step-size 512
