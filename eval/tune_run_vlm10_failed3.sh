#!/usr/bin/env bash
# Re-run the 3 run_vlm10 failures (PoP, Monkey Island, Dragon's Lair) after
# harness/memory fixes. Fresh Python process picks up all code changes.
# Log: games/tune_serial10/run_vlm10_failed3/overnight.log
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_vlm10_failed3"
GOALS="$REPO_ROOT/eval/tune_run_vlm10_failed3.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/Qwen3.6-27B-mxfp8}"
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
cp "$GOALS" "$OUT/goals.txt"
LOG="$OUT/overnight.log"

echo "=== run_vlm10_failed3 start $(date -u +%Y-%m-%dT%H:%M:%SZ) jobs=3 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo ">>> Cursor watcher: .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_vlm10_failed3 --jobs-total 3 --interval 30 --sync-loop" | tee -a "$LOG"

ARGS=(--goals-file "$OUT/goals.txt" --out-dir "$OUT" --model "$MODEL" --backend mlx --max-iters 3 --retries 2 --retry-delay 30 --stall-seconds 1200 --best-of-n 1)
if [[ "$WAIT" != "0" && "$WAIT" != "0.0" ]]; then
  ARGS+=(--wait-for-monitor "$WAIT")
fi

exec caffeinate -dims env -u PLAYWRIGHT_BROWSERS_PATH \
  LLM_BACKEND=mlx \
  MLX_MODEL="$MODEL" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_serial_loop.py" \
  "${ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG"
