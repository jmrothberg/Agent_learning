#!/usr/bin/env bash
#
# Start mlx_lm.server on a DeepSeek-V4 model with the correct flags.
#
#   ./scripts/mlx_v4_server.sh                     # auto-pick first V4 model in ~/MLX_Models
#   ./scripts/mlx_v4_server.sh <model_path>        # explicit path
#
# The only thing this script does that bare `mlx_lm.server` doesn't:
# pass `--prefill-step-size 1024`. Without that flag, V4 (Flash or Pro)
# blows up the moment any single prefill chunk exceeds ~1.3K tokens
# because its Indexer attention path materializes a single Metal buffer
# of size O(L^2 * k) — see PR ml-explore/mlx-lm#1192 review thread for
# the cubic-growth analysis. The PR reviewer's measured table:
#
#   L=1024  k=256   34 GB    works
#   L=1280  k=320   67 GB    works (tight)
#   L=1500  k=375  108 GB    crashes (Metal per-buffer cap ~86 GB)
#   L=2048  k=512  275 GB    crashes
#
# Choosing 1024: the safe-anywhere default per the PR reviewer; at the
# limit (34 GB) you keep ~5% more prefill throughput vs 512. The Metal
# per-buffer cap is hardware (~86 GB on M-series), NOT total RAM, so
# a 512 GB Mac doesn't get a higher safe ceiling than a 64 GB Mac.
# Override with CHUNK env var if you want to experiment:
#   CHUNK=512  ./scripts/mlx_v4_server.sh   # paranoid
#   CHUNK=1280 ./scripts/mlx_v4_server.sh   # last documented-safe step
#
# Default port 8080 matches mlx_lm.server's default and what backend.py
# probes when no MLX_HOST env is set. Override via the PORT env var.

set -euo pipefail

PORT="${PORT:-8080}"
CHUNK="${CHUNK:-1024}"
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
echo "  port=$PORT  prefill-step-size=$CHUNK  (override with CHUNK env)"
echo
exec mlx_lm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --prefill-step-size "$CHUNK"
