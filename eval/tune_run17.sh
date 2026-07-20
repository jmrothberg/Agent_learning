#!/usr/bin/env bash
# run_17 — 14 goals: run_15+16 fails (11) + 3 fresh graphics games.
# GLM-5.2-MLX-4bit, no VLM. Match run_16 throughput: max-iters 3, retries 0.
# Log: games/tune_serial10/run_17/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX). Cursor watcher triages finished
# traces while games keep going.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_17"
GOALS="$REPO_ROOT/eval/tune_run17_rerun_fails.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_17 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=3 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL jobs=14 vlm=OFF" | tee -a "$LOG"
echo ">>> CURSOR WATCHER:" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_17 --jobs-total 14 --interval 30 --sync-loop" | tee -a "$LOG"
echo "" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --no-vlm-critique --resume --max-iters 3 --retries 0)
if [[ "$WAIT" != "0" && "$WAIT" != "0.0" ]]; then
  ARGS+=(--wait-for-monitor "$WAIT")
fi

exec caffeinate -dims env -u PLAYWRIGHT_BROWSERS_PATH \
  LLM_BACKEND=mlx \
  MLX_MODEL="$MODEL" \
  PYTHONUNBUFFERED=1 \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_serial_loop.py" \
  "${ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG"
