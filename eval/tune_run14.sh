#!/usr/bin/env bash
# run_14 — 10 ALL-NEW games, Qwen3.6-27B-mxfp8, VLM critique ON.
# Flat-out batch + Cursor watcher learn loop (fix harness/memory as each
# game finishes — do not wait until the end).
# Log: games/tune_serial10/run_14/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor watcher triages
# finished traces and patches harness/memory/prompts while games keep going.
# --max-iters 4 --retries 0: fix source for next game, not more HTML patches.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_14"
GOALS="$REPO_ROOT/eval/tune_run14_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/Qwen3.6-27B-mxfp8}"
# 0 = no pause between games — batch runs flat-out all night.
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_14 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=4 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "vlm_critique=ON" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless — Chromium window opens for each game)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> REQUIRED IN CURSOR (parallel watcher — triage traces, patch harness/memory):" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_14 --jobs-total 10 --interval 30 --sync-loop" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# VLM critique ON = omit --no-vlm-critique (tune_serial_loop default).
ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --resume --max-iters 4 --retries 0)
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
