#!/usr/bin/env bash
# Overnight watchdog — restarts tune_serial_loop until all goals are delivered.
# Survives parent crashes, SIGSEGV atexit, and Cursor terminal aborts.
#
# Usage:
#   cd /Users/jonathanrothberg/Agent_learning
#   MLX_MODEL=~/MLX_Models/Qwen3.6-27B-mxfp8 nohup eval/tune_serial_overnight.sh &
#   tail -f games/tune_serial10/run_03/overnight.log
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${TUNE_OUT_DIR:-games/tune_serial10/run_03}"
GOALS="${TUNE_GOALS_FILE:-eval/tune_serial10_goals.txt}"
MLX_MODEL="${MLX_MODEL:-$HOME/MLX_Models/Qwen3.6-27B-mxfp8}"
JOBS_TOTAL="$(grep -cve '^[[:space:]]*$' -e '^[[:space:]]*#' "$GOALS" || true)"
LOG="$OUT_DIR/overnight.log"
mkdir -p "$OUT_DIR"

echo "=== tune_serial_overnight start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "out_dir=$OUT_DIR goals=$GOALS model=$MLX_MODEL jobs=$JOBS_TOTAL" | tee -a "$LOG"

_run_once() {
  env -u PLAYWRIGHT_BROWSERS_PATH \
    LLM_BACKEND=mlx \
    MLX_MODEL="$MLX_MODEL" \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python eval/tune_serial_loop.py \
      --goals-file "$GOALS" \
      --out-dir "$OUT_DIR" \
      --resume \
      --retries 2 \
      --retry-delay 30 \
      2>&1 | tee -a "$LOG"
}

_completed_count() {
  .venv/bin/python - <<PY
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
.venv/bin/python - <<'PY' "$OUT_DIR"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
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
    ck["jobs_total"] = 10
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
  if [ "$DONE" -ge "$JOBS_TOTAL" ]; then
    echo "=== ALL $JOBS_TOTAL GAMES DELIVERED $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
    exit 0
  fi
  echo "sleep 60s then restart…" | tee -a "$LOG"
  sleep 60
done
