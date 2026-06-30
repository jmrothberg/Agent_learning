#!/usr/bin/env bash
# run_08 — 10 games overnight, flat-out batch + parallel Cursor watcher.
# Log: games/tune_serial10/run_08/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor watcher runs in
# parallel — triages finished traces and patches harness/memory while games
# keep going. See eval/OPERATIONS.md § run_08.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_08"
GOALS="$REPO_ROOT/eval/tune_run08_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
# 0 = no pause between games — batch runs flat-out all night.
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_08 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless — Chromium window opens for each game)" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --no-vlm-critique --resume --retries 2 --retry-delay 30)
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
