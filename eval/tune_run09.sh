#!/usr/bin/env bash
# run_10 validation — 10 games, flat-out batch + parallel Cursor watcher learn loop.
# Validates run_09 harness/memory fixes under --max-iters 4.
# Log: games/tune_serial10/run_10/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor watcher triages
# finished traces and patches harness/memory/prompts while games keep going.
# --max-iters 4 --retries 0: fix source for next game, not more HTML patches.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_10"
GOALS="$REPO_ROOT/eval/tune_run09_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
# 0 = no pause between games — batch runs flat-out all night.
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_10 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=4 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless — Chromium window opens for each game)" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --no-vlm-critique --resume --max-iters 4 --retries 0)
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
