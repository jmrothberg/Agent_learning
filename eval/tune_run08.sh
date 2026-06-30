#!/usr/bin/env bash
# run_08 — 10 games overnight with Cursor watcher handoff between games.
# Log: games/tune_serial10/run_08/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor runs the watcher
# in parallel — see eval/OPERATIONS.md § run_08.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_08"
GOALS="$REPO_ROOT/eval/tune_run08_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
# Block after each game until Cursor watcher triages + releases (no Enter, no stdin).
WAIT="${TUNE_WAIT_FOR_MONITOR:-1800}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_08 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"

exec caffeinate -dims env -u PLAYWRIGHT_BROWSERS_PATH \
  LLM_BACKEND=mlx \
  MLX_MODEL="$MODEL" \
  PYTHONUNBUFFERED=1 \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_serial_loop.py" \
  --goals-file "$GOALS" \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --no-vlm-critique \
  --resume \
  --retries 2 \
  --retry-delay 30 \
  --wait-for-monitor "$WAIT" \
  "$@" 2>&1 | tee -a "$LOG"
