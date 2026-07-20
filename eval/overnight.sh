#!/usr/bin/env bash
# eval/overnight.sh — ONE overnight entry point.
#
# Pick prompt_library numbers, an MLX model, and VLM yes/no. That is all.
#
#   bash eval/overnight.sh --prompts 54,28,21 --model GLM-5.2-MLX-4bit --vlm no
#   bash eval/overnight.sh 54,28,21 GLM-5.2-MLX-4bit no          # same (positional)
#   bash eval/overnight.sh --list                                 # show # → name
#
# From Cursor the agent runs this with full OS perms: it opens Terminal.app for
# the batch and prints the exact Cursor Shell watcher command (block_until_ms=0).
# Never ask the human to paste. Never run the batch inside Cursor's integrated
# terminal. See eval/OPERATIONS.md § HARD RULES.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash eval/overnight.sh --prompts N,N,... --model NAME_OR_PATH --vlm yes|no
  bash eval/overnight.sh N,N,... NAME_OR_PATH yes|no
  bash eval/overnight.sh --list

Options:
  --prompts   Comma-separated prompt_library numbers and/or names (required)
  --model     MLX folder name under ~/MLX_Models/ or an absolute path (required)
  --vlm       yes|no  (VLM critique; only useful if a VLM is available)
  --run-id    Optional out dir name (default: next run_N under games/tune_serial10/)
  --max-iters Default 3
  --retries   Default 0
  --list      Print prompt_library numbers and exit
  --dry-run   Resolve goals + print watcher cmd; do not open Terminal or start batch
  -h|--help   This help

Agent launch (Cursor, full OS perms) then Cursor Shell watcher:
  bash eval/overnight.sh --prompts 54,28 --model GLM-5.2-MLX-4bit --vlm no
EOF
}

PROMPTS=""
MODEL=""
VLM=""
RUN_ID=""
MAX_ITERS="${TUNE_MAX_ITERS:-3}"
RETRIES="${TUNE_RETRIES:-0}"
LIST_ONLY=0
DRY_RUN=0

# Positional shorthand: overnight.sh 54,28 GLM-5.2-MLX-4bit no
if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  PROMPTS="$1"
  shift
  if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then MODEL="$1"; shift; fi
  if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then VLM="$1"; shift; fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --vlm) VLM="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --max-iters) MAX_ITERS="${2:-}"; shift 2 ;;
    --retries) RETRIES="${2:-}"; shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$LIST_ONLY" -eq 1 ]]; then
  "$REPO_ROOT/.venv/bin/python" - <<'PY'
from prompt_library import load_prompt_library
for p in load_prompt_library():
    print(f"{p['n']:3d}  {p.get('name','')}  —  {p.get('title','')}")
PY
  exit 0
fi

if [[ -z "$PROMPTS" || -z "$MODEL" || -z "$VLM" ]]; then
  echo "error: need --prompts, --model, and --vlm (yes|no)" >&2
  usage >&2
  exit 2
fi

case "$(echo "$VLM" | tr '[:upper:]' '[:lower:]')" in
  y|yes|on|1|true) VLM=yes ;;
  n|no|off|0|false) VLM=no ;;
  *) echo "error: --vlm must be yes or no (got: $VLM)" >&2; exit 2 ;;
esac

if [[ "$MODEL" == /* ]] || [[ -d "$MODEL" ]]; then
  MODEL_PATH="$MODEL"
elif [[ -d "$HOME/MLX_Models/$MODEL" ]]; then
  MODEL_PATH="$HOME/MLX_Models/$MODEL"
else
  echo "error: model not found: $MODEL (tried $HOME/MLX_Models/$MODEL)" >&2
  exit 2
fi

if [[ -z "$RUN_ID" ]]; then
  max=0
  shopt -s nullglob
  for d in "$REPO_ROOT"/games/tune_serial10/run_*; do
    [[ -d "$d" ]] || continue
    n="${d##*/run_}"
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > max )); then max=$n; fi
  done
  shopt -u nullglob
  RUN_ID="run_$((max + 1))"
fi
OUT="$REPO_ROOT/games/tune_serial10/$RUN_ID"
GOALS="$OUT/goals.txt"
LOG="$OUT/overnight.log"
mkdir -p "$OUT"

export OVERNIGHT_PROMPTS="$PROMPTS"
export OVERNIGHT_GOALS="$GOALS"
export OVERNIGHT_RUN_ID="$RUN_ID"
export OVERNIGHT_MODEL_PATH="$MODEL_PATH"
export OVERNIGHT_VLM="$VLM"

JOBS="$("$REPO_ROOT/.venv/bin/python" - <<'PY'
import os
import sys
from pathlib import Path
from prompt_library import load_prompt_library, get_prompt

sel = [s.strip() for s in os.environ["OVERNIGHT_PROMPTS"].split(",") if s.strip()]
if not sel:
    print("error: empty --prompts", file=sys.stderr)
    sys.exit(2)
lib = load_prompt_library()
by_name = {str(p.get("name", "")).lower(): p for p in lib}
goals = []
meta = []
for tok in sel:
    rec = None
    if tok.isdigit():
        rec = get_prompt(int(tok))
    if rec is None:
        rec = by_name.get(tok.lower())
    if rec is None:
        print(f"error: unknown prompt: {tok!r}", file=sys.stderr)
        sys.exit(2)
    goals.append(" ".join(str(rec["prompt"]).split()))
    meta.append(f"# {rec['n']} {rec.get('name','')} — {rec.get('title','')}")
out = Path(os.environ["OVERNIGHT_GOALS"])
header = [
    f"# overnight {os.environ['OVERNIGHT_RUN_ID']} — prompts={os.environ['OVERNIGHT_PROMPTS']}",
    f"# model={os.environ['OVERNIGHT_MODEL_PATH']}",
    f"# vlm={os.environ['OVERNIGHT_VLM']}",
] + meta + [""]
out.write_text("\n".join(header + goals) + "\n", encoding="utf-8")
for line in meta:
    print(line, file=sys.stderr)
print(len(goals))
PY
)"

WAIT="${TUNE_WAIT_FOR_MONITOR:-0}"
WATCH_CMD=".venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/${RUN_ID} --jobs-total ${JOBS} --interval 30 --sync-loop"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== dry-run ${RUN_ID} (${JOBS} games, vlm=${VLM}) ==="
  echo "out: games/tune_serial10/${RUN_ID}"
  echo "goals: ${GOALS}"
  echo "model: ${MODEL_PATH}"
  echo "prompts: ${PROMPTS}"
  echo ">>> CURSOR WATCHER: ${WATCH_CMD}"
  exit 0
fi

# From Cursor / non-Terminal: open real Terminal.app and exit (agent then starts watcher).
if [[ "${OVERNIGHT_CHILD:-}" != "1" && "${TERM_PROGRAM:-}" != "Apple_Terminal" ]]; then
  CHILD_CMD="cd ${REPO_ROOT} && OVERNIGHT_CHILD=1 bash ${REPO_ROOT}/eval/overnight.sh --prompts ${PROMPTS} --model ${MODEL_PATH} --vlm ${VLM} --run-id ${RUN_ID} --max-iters ${MAX_ITERS} --retries ${RETRIES}"
  osascript -e 'tell application "Terminal" to activate' \
    -e "tell application \"Terminal\" to do script \"${CHILD_CMD}\""
  echo "=== overnight ${RUN_ID} → Terminal.app (${JOBS} games, vlm=${VLM}) ==="
  echo "out: games/tune_serial10/${RUN_ID}"
  echo "model: ${MODEL_PATH}"
  echo "prompts: ${PROMPTS}"
  echo ""
  echo ">>> CURSOR WATCHER (Shell tool, block_until_ms=0 — NOT nohup):"
  echo ">>> ${WATCH_CMD}"
  exit 0
fi

echo "=== overnight ${RUN_ID} start $(date -u +%Y-%m-%dT%H:%M:%SZ) wait_for_monitor=${WAIT}s max_iters=${MAX_ITERS} retries=${RETRIES} ===" | tee -a "$LOG"
echo "prompts=${PROMPTS} jobs=${JOBS} model=${MODEL_PATH} vlm=${VLM}" | tee -a "$LOG"
echo "goals=${GOALS}" | tee -a "$LOG"
echo "visible_chromium=yes (no --headless)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ">>> CURSOR WATCHER:" | tee -a "$LOG"
echo ">>> ${WATCH_CMD}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

ARGS=(--goals-file "$GOALS" --out-dir "$OUT" --model "$MODEL_PATH" --backend mlx --resume --max-iters "$MAX_ITERS" --retries "$RETRIES")
if [[ "$VLM" == "no" ]]; then
  ARGS+=(--no-vlm-critique)
fi
if [[ "$WAIT" != "0" && "$WAIT" != "0.0" ]]; then
  ARGS+=(--wait-for-monitor "$WAIT")
fi

exec caffeinate -dims env -u PLAYWRIGHT_BROWSERS_PATH \
  LLM_BACKEND=mlx \
  MLX_MODEL="$MODEL_PATH" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/eval/tune_serial_loop.py" \
  "${ARGS[@]}" \
  2>&1 | tee -a "$LOG"
