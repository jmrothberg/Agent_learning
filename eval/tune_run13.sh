#!/usr/bin/env bash
# run_13 — 10 ALL-NEW games, GLM-5.2-MLX-4bit, --no-vlm-critique.
# Flat-out batch + Cursor watcher learn loop (fix harness/memory as each
# game finishes — do not wait until the end).
# Log: games/tune_serial10/run_13/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor watcher triages
# finished traces and patches harness/memory/prompts while games keep going.
# --max-iters 4 --retries 0: fix source for next game, not more HTML patches.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_13"
GOALS="$REPO_ROOT/eval/tune_run13_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
# 0 = no pause between games — batch runs flat-out all night.
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_13 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=4 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "vlm_critique=OFF" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless — Chromium window opens for each game)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> REQUIRED IN CURSOR (parallel watcher — triage traces, patch harness/memory):" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_13 --jobs-total 10 --interval 30 --sync-loop" | tee -a "$LOG"
echo "" | tee -a "$LOG"

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
