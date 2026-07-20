#!/usr/bin/env bash
# eval/overnight.sh — ONE overnight entry point.
#
# Interactive (no command line — just run / double-click Overnight.command):
#   bash eval/overnight.sh
#   bash eval/overnight.sh --interactive
#
# CLI still works for agents:
#   bash eval/overnight.sh --prompts 54,28,21 --model GLM-5.2-MLX-4bit --vlm no
#   bash eval/overnight.sh 54,28,21 GLM-5.2-MLX-4bit no
#   bash eval/overnight.sh --list
#
# From Cursor with flags: opens Terminal.app for the batch and prints the watcher
# line. Interactive mode asks questions inside Terminal, then starts.
# See eval/OPERATIONS.md § HARD RULES.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash eval/overnight.sh                    # interactive Q&A in Terminal
  open Overnight.command                    # same (double-click in Finder)
  bash eval/overnight.sh --prompts N,N,... --model NAME --vlm yes|no
  bash eval/overnight.sh N,N,... NAME yes|no
  bash eval/overnight.sh --list

Options:
  --prompts     Comma-separated prompt_library numbers and/or names
  --model       MLX folder name under ~/MLX_Models/ or absolute path
  --vlm         yes|no
  --run-id      Optional out dir (default: next run_N)
  --max-iters   Default 3
  --retries     Default 0
  --interactive Force the question UI
  --list        Print prompt library and exit
  --dry-run     Resolve goals + print watcher; do not start batch
  -h|--help     This help
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
INTERACTIVE=0

# No args → interactive
if [[ $# -eq 0 ]]; then
  INTERACTIVE=1
fi

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
    --interactive|-i) INTERACTIVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

print_library() {
  "$REPO_ROOT/.venv/bin/python" - <<'PY'
from prompt_library import load_prompt_library
for p in load_prompt_library():
    print(f"{p['n']:3d}  {p.get('name','')}  —  {p.get('title','')}")
PY
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
  print_library
  exit 0
fi

# If interactive requested but we're not in Terminal.app yet, open Terminal and ask there.
if [[ "$INTERACTIVE" -eq 1 && "${OVERNIGHT_CHILD:-}" != "1" && "${TERM_PROGRAM:-}" != "Apple_Terminal" ]]; then
  CHILD_CMD="cd ${REPO_ROOT} && OVERNIGHT_CHILD=1 bash ${REPO_ROOT}/eval/overnight.sh --interactive"
  osascript -e 'tell application "Terminal" to activate' \
    -e "tell application \"Terminal\" to do script \"${CHILD_CMD}\""
  echo "Opened Terminal.app — answer the questions there to start the overnight batch."
  echo "After it starts, run the watcher in Cursor (the Terminal window will print the exact command)."
  exit 0
fi

interactive_ask() {
  echo ""
  echo "════════════════════════════════════════════════════════"
  echo "  Overnight batch — answer the questions, then it starts"
  echo "════════════════════════════════════════════════════════"
  echo ""
  echo "Canned prompts (enter numbers separated by commas):"
  echo ""
  print_library
  echo ""
  while true; do
    read -r -p "Prompt numbers (e.g. 54,28,21): " PROMPTS
    PROMPTS="$(echo "$PROMPTS" | tr -d '[:space:]')"
    if [[ -n "$PROMPTS" ]]; then break; fi
    echo "  Need at least one number."
  done

  echo ""
  read -r -p "Max iterations per game [${MAX_ITERS}]: " ans
  if [[ -n "${ans:-}" ]]; then MAX_ITERS="$ans"; fi
  if ! [[ "$MAX_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: max-iters must be a positive integer" >&2
    exit 2
  fi

  echo ""
  echo "VLM critique checks orientation/art with a vision model (slower; needs a VLM)."
  while true; do
    read -r -p "Enable VLM critique? [y/N]: " ans
    ans="$(echo "${ans:-n}" | tr '[:upper:]' '[:lower:]')"
    case "$ans" in
      y|yes) VLM=yes; break ;;
      n|no|"") VLM=no; break ;;
      *) echo "  Enter y or n." ;;
    esac
  done

  echo ""
  echo "MLX models in ~/MLX_Models:"
  local models=()
  local i=1 default_i=1
  shopt -s nullglob
  local d
  for d in "$HOME"/MLX_Models/*/; do
    [[ -d "$d" ]] || continue
    local name
    name="$(basename "$d")"
    models+=("$name")
    echo "  $i) $name"
    if [[ "$name" == "GLM-5.2-MLX-4bit" ]]; then default_i=$i; fi
    i=$((i + 1))
  done
  shopt -u nullglob
  if [[ ${#models[@]} -eq 0 ]]; then
    echo "error: no models found in $HOME/MLX_Models" >&2
    exit 2
  fi
  echo ""
  while true; do
    read -r -p "Model number [${default_i}]: " ans
    ans="${ans:-$default_i}"
    if [[ "$ans" =~ ^[0-9]+$ ]] && (( ans >= 1 && ans <= ${#models[@]} )); then
      MODEL="${models[$((ans - 1))]}"
      break
    fi
    echo "  Enter a number 1–${#models[@]}."
  done

  echo ""
  echo "── Summary ──"
  echo "  prompts:    $PROMPTS"
  echo "  max-iters:  $MAX_ITERS"
  echo "  vlm:        $VLM"
  echo "  model:      $MODEL"
  echo ""
  read -r -p "Start overnight batch now? [Y/n]: " ans
  ans="$(echo "${ans:-y}" | tr '[:upper:]' '[:lower:]')"
  case "$ans" in
    y|yes|"") ;;
    *) echo "Cancelled."; exit 0 ;;
  esac
}

if [[ "$INTERACTIVE" -eq 1 ]]; then
  interactive_ask
elif [[ -z "$PROMPTS" || -z "$MODEL" || -z "$VLM" ]]; then
  echo "Missing --prompts / --model / --vlm. Starting interactive mode…"
  echo ""
  INTERACTIVE=1
  if [[ "${OVERNIGHT_CHILD:-}" != "1" && "${TERM_PROGRAM:-}" != "Apple_Terminal" ]]; then
    CHILD_CMD="cd ${REPO_ROOT} && OVERNIGHT_CHILD=1 bash ${REPO_ROOT}/eval/overnight.sh --interactive"
    osascript -e 'tell application "Terminal" to activate' \
      -e "tell application \"Terminal\" to do script \"${CHILD_CMD}\""
    echo "Opened Terminal.app — answer the questions there."
    exit 0
  fi
  interactive_ask
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
  echo "max-iters: ${MAX_ITERS}"
  echo ">>> CURSOR WATCHER: ${WATCH_CMD}"
  exit 0
fi

# CLI path from Cursor (non-interactive, not already Terminal): open Terminal and exit.
if [[ "$INTERACTIVE" -eq 0 && "${OVERNIGHT_CHILD:-}" != "1" && "${TERM_PROGRAM:-}" != "Apple_Terminal" ]]; then
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
echo ">>> CURSOR WATCHER (start this in Cursor so harness fixes run in parallel):" | tee -a "$LOG"
echo ">>> ${WATCH_CMD}" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo ""
echo "Batch starting. In Cursor, run the watcher:"
echo "  ${WATCH_CMD}"
echo ""

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
