#!/usr/bin/env bash
# run_07 — both batches back-to-back (A: GLM no VLM, B: Qwen + VLM). One Terminal paste.
# Log: games/tune_serial10/run_07/chain.log
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_07"
mkdir -p "$OUT"
LOG="$OUT/chain.log"
# 0 = no pause between games (batch runs flat-out overnight).
WAIT="${TUNE_WAIT_FOR_MONITOR:-1800}"

echo "=== run_07 chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s ===" | tee -a "$LOG"

exec caffeinate -dims env PYTHONUNBUFFERED=1 \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_run07_chain.py" \
  --wait-for-monitor "$WAIT" \
  "$@" 2>&1 | tee -a "$LOG"
