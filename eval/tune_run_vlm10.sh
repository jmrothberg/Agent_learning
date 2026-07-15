#!/usr/bin/env bash
# run_vlm10 — 10 graphical games, Qwen 27B + VLM critique ON, flat-out batch + Cursor watcher.
# Log: games/tune_serial10/run_vlm10/overnight.log
#
# Terminal.app ONLY (visible Chromium + MLX cold load). Cursor watcher triages
# finished traces and patches harness/memory/prompts while games keep going.
# --max-iters 3 --retries 2: per-game crash retry + alignment/VLM validation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT="$REPO_ROOT/games/tune_serial10/run_vlm10"
GOALS="$OUT/goals.txt"
MODEL="${MLX_MODEL:-$HOME/MLX_Models/Qwen3.6-27B-mxfp8}"
WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
mkdir -p "$OUT"
LOG="$OUT/overnight.log"

# Assemble 10 single-line goals (library prompts like dragons-lair embed newlines).
"$REPO_ROOT/.venv/bin/python" - <<'PY' "$GOALS"
import sys
from pathlib import Path
from prompt_library import load_prompt_library

out = Path(sys.argv[1])
repo = out.parents[3]
r12 = [
    l.strip()
    for l in (repo / "eval/tune_run12_goals.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")
]
r08 = [
    l.strip()
    for l in (repo / "eval/tune_run08_goals.txt").read_text().splitlines()
    if l.strip() and not l.startswith("#")
][5:8]
by_name = {p["name"]: " ".join(p["prompt"].split()) for p in load_prompt_library()}
extra = [by_name[n] for n in ("fighter-showcase", "1942", "dragons-lair")]
goals = r12 + r08 + extra
assert len(goals) == 10, goals
out.write_text("\n".join(goals) + "\n", encoding="utf-8")
print(f"assembled {len(goals)} goals -> {out}")
PY

if [[ ! -f "$GOALS" ]]; then
  echo "ERROR: goals file missing: $GOALS" >&2
  echo "Run goal assembly first (see eval/tune_run_vlm10.sh header in plan)." >&2
  exit 1
fi

echo "=== run_vlm10 start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=3 retries=2 vlm=ON ===" | tee -a "$LOG"
echo "goals=$GOALS model=$MODEL" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless — Chromium window opens for each game)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> REQUIRED IN CURSOR (parallel watcher — triage traces, patch harness/memory):" | tee -a "$LOG"
echo ">>> .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_vlm10 --jobs-total 10 --interval 30 --sync-loop" | tee -a "$LOG"
echo "" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL" --backend mlx --resume --max-iters 3 --retries 2 --retry-delay 30 --stall-seconds 1200 --best-of-n 1)
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
