#!/usr/bin/env bash
# run_16 — 10 fresh graphics-heavy games, GLM-5.2-MLX-4bit, no VLM critique.
# Starts only after run_15 has exited; safe to launch while run_15 finalizes.
# Log: games/tune_serial10/run_16/overnight.log
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_16"
GOALS="$REPO_ROOT/eval/tune_run16_goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

# Evidence (run_14+15): first-clean clusters at 1–2; cap 2 loses ~4 rescues,
# cap 4 mostly burns fail-side churn for one Doom rescue → default max-iters=3.
echo "=== run_16 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=3 retries=0 ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "vlm_critique=OFF graphics_heavy=10" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> CURSOR WATCHER:" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_16 --jobs-total 10 --interval 30 --sync-loop" | tee -a "$LOG"
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
