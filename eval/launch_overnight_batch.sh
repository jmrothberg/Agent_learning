#!/usr/bin/env bash
# Open overnight batch in REAL Terminal.app (never Cursor's integrated terminal).
#
# Usage (Cursor agent runs this — NEVER ask the human to paste):
#   bash eval/launch_overnight_batch.sh eval/tune_run18.sh
#
# Then IMMEDIATELY start the watcher in a Cursor Shell with block_until_ms=0
# so it appears in the IDE terminals panel. NEVER nohup / disown the watcher.
#
# See eval/OPERATIONS.md § HARD RULES.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_REL="${1:?usage: bash eval/launch_overnight_batch.sh eval/tune_runXX.sh}"
case "$SCRIPT_REL" in
  /*) SCRIPT_ABS="$SCRIPT_REL" ;;
  *)  SCRIPT_ABS="$REPO_ROOT/$SCRIPT_REL" ;;
esac
if [[ ! -f "$SCRIPT_ABS" ]]; then
  echo "error: batch script not found: $SCRIPT_ABS" >&2
  exit 1
fi

# Keep the do-script string simple (no shell %q) — paths have no spaces in this repo.
CMD="cd ${REPO_ROOT} && bash ${SCRIPT_ABS}"
osascript -e "tell application \"Terminal\" to activate" \
  -e "tell application \"Terminal\" to do script \"${CMD}\""

echo "opened Terminal.app → ${CMD}"
base="$(basename "$SCRIPT_ABS" .sh)"   # tune_run18
run_num="${base#tune_run}"             # 18
echo "NEXT — Cursor Shell ONLY (visible panel, block_until_ms=0; NOT nohup):"
echo "  .venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_${run_num} --jobs-total N --interval 30 --sync-loop"
