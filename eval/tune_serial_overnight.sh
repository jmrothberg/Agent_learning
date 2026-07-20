#!/usr/bin/env bash
# Overnight watchdog — restarts tune_serial_loop until all goals are delivered.
# Full ops guide (launch, triage, artifact paths): eval/OPERATIONS.md
# Survives parent crashes, SIGSEGV atexit, and Cursor terminal aborts.
#
# Launch in Terminal.app ONLY (not Cursor). Log: $TUNE_OUT_DIR/overnight.log
# (written via tee inside this script — do NOT also redirect nohup stdout to that file).
#
# Round 2 / GLM-5.2-MLX-4bit (recommended over mxfp4 for stability):
#   cd /Users/jonathanrothberg/Agent_learning
#   bash eval/tune_run08.sh
#   tail -f games/tune_serial10/run_08/overnight.log
# Cursor watcher (required — see eval/OPERATIONS.md § run_08):
#   .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_08 --jobs-total 10 --interval 30 --sync-loop
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${TUNE_OUT_DIR:-games/tune_serial10/run_08}"
GOALS="${TUNE_GOALS_FILE:-eval/tune_run08_goals.txt}"
MLX_MODEL="${MLX_MODEL:-$HOME/MLX_Models/GLM-5.2-MLX-4bit}"
JOBS_TOTAL="$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "$GOALS" || true)"
LOG="$OUT_DIR/overnight.log"
PIDFILE="$OUT_DIR/overnight.pid"
mkdir -p "$OUT_DIR"

_preflight() {
  local err=0
  if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "ERROR: missing .venv — run ./scripts/setup.sh first" >&2
    err=1
  fi
  if [[ ! -f "$GOALS" ]]; then
    echo "ERROR: goals file not found: $GOALS" >&2
    err=1
  fi
  if [[ "$JOBS_TOTAL" -lt 1 ]]; then
    echo "ERROR: no goals in $GOALS" >&2
    err=1
  fi
  if [[ ! -e "$MLX_MODEL" ]]; then
    echo "ERROR: MLX_MODEL path not found: $MLX_MODEL" >&2
    echo "  Round 2 expects: \$HOME/MLX_Models/GLM-5.2-MLX-4bit" >&2
    err=1
  fi
  if [[ "$err" -ne 0 ]]; then
    exit 1
  fi
}

_preflight
echo "$$" >"$PIDFILE"

echo "=== tune_serial_overnight start $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ ===" | tee -a "$LOG"
echo "out_dir=$OUT_DIR goals=$GOALS model=$MLX_MODEL jobs=$JOBS_TOTAL" | tee -a "$LOG"

_run_once() {
  WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
    LOOP_ARGS=(
    --goals-file "$GOALS"
    --out-dir "$OUT_DIR"
    --model "$MLX_MODEL"
    --no-vlm-critique
    --resume
    # Match run_15/16 throughput (run_17 regression: default max-iters=6 + retries=2).
    --max-iters "${TUNE_MAX_ITERS:-3}"
    --retries "${TUNE_RETRIES:-0}"
    --retry-delay 30
  )
  if [[ "$WAIT" != "0" && "$WAIT" != "0.0" ]]; then
    LOOP_ARGS+=(--wait-for-monitor "$WAIT")
  fi
  env -u PLAYWRIGHT_BROWSERS_PATH \
    LLM_BACKEND=mlx \
    MLX_MODEL="$MLX_MODEL" \
    PYTHONUNBUFFERED=1 \
    "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_serial_loop.py" \
      "${LOOP_ARGS[@]}" \
      2>&1 | tee -a "$LOG"
}

_completed_count() {
  "$REPO_ROOT/.venv/bin/python" - <<PY
import json
from pathlib import Path
p = Path("$OUT_DIR") / "tune_checkpoint.json"
if not p.is_file():
    print(0)
else:
    d = json.loads(p.read_text())
    print(int(d.get("completed_count") or len(d.get("completed_labels") or [])))
PY
}

# Seed checkpoint for any delivered .best.html not yet recorded.
"$REPO_ROOT/.venv/bin/python" - <<PY "$OUT_DIR" "$JOBS_TOTAL"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
jobs_total = int(sys.argv[2])
ck_path = out / "tune_checkpoint.json"
ck = {"completed_labels": [], "results": []}
if ck_path.is_file():
    try:
        ck = json.loads(ck_path.read_text())
    except Exception:
        pass
labels = list(ck.get("completed_labels") or [])
for best in sorted(out.glob("*.best.html")):
    label = best.name.replace(".best.html", "")
    if label not in labels:
        labels.append(label)
if labels != list(ck.get("completed_labels") or []):
    ck["completed_labels"] = labels
    ck["completed_count"] = len(labels)
    ck["jobs_total"] = jobs_total
    ck_path.write_text(json.dumps(ck, indent=2))
    print(f"seeded checkpoint: {len(labels)} delivered — {labels}")
PY

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "--- watchdog attempt $ATTEMPT $(date -u +%Y-%m-%dT%H:%M:%SZ) ---" | tee -a "$LOG"
  _run_once || true
  DONE="$(_completed_count)"
  echo "completed $DONE / $JOBS_TOTAL" | tee -a "$LOG"
  if [[ "$DONE" -ge "$JOBS_TOTAL" ]]; then
    echo "=== ALL $JOBS_TOTAL GAMES DELIVERED $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
    exit 0
  fi
  echo "sleep 60s then restart…" | tee -a "$LOG"
  sleep 60
done
