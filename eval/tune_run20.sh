#!/usr/bin/env bash
# run_20 — Qwen3.8-Flash-Next campaign: 8 /640png + 4 full-HTML, VLM off.
# Goals: eval/tune_campaign_qwen38_goals.txt (12 games). --max-iters 3 --retries 0.
# Log: games/tune_serial10/run_20/overnight.log
#
# DO NOT run this inside Cursor's integrated terminal (wrong Playwright arch).
# Agent launch: bash eval/launch_overnight_batch.sh eval/tune_run20.sh
# Then Cursor Shell (block_until_ms=0, visible): tune_overnight_monitor.py … run_20
# See eval/OPERATIONS.md § HARD RULES + § Campaign bench.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_20"
GOALS="$REPO_ROOT/eval/tune_campaign_qwen38_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/Qwen3.8-Flash-Next-MLX-8bit-MTP}"
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

echo "=== run_20 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=3 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL jobs=12 vlm=OFF campaign=qwen38_html_640png" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> CURSOR WATCHER:" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_20 --jobs-total 12 --interval 30 --sync-loop" | tee -a "$LOG"
echo "" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --backend mlx --no-vlm-critique --resume --max-iters 3 --retries 0)
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
