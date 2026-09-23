#!/bin/zsh
# Start LoRA training jobs in the background. They survive closing Terminal/Cursor.
#
#   ./start.sh            monitor + trainer            (default)
#   ./start.sh all        monitor + downloader + trainer (HTML-game LoRA only)
#   ./start.sh monitor    progress page only            http://127.0.0.1:$LORA_PORT/
#   ./start.sh train      trainer (run_slices.py) only
#   ./start.sh ingest     GitHub game downloader only   (HTML-game LoRA only)
#
# Pick a LoRA project with env vars (defaults = HTML-game LoRA on Qwen3.8-27B):
#   LORA_ROOT=~/MLX_Models/html_game_sft     data + adapters + snapshots + logs
#   LORA_BASE=~/MLX_Models/Qwen3.8-27B-mxfp8 base model (never modified)
#   LORA_PY=~/Agents/.venv/bin/python        python that has mlx-vlm
#   LORA_PORT=8766                           progress page port
#   LORA_TRAIN_ARGS="..."                    extra train_lora.py flags
#   LORA_GAME_FILTER=1                       0 = train every row, not just games
#   LORA_SLICE_MIN=30                        minutes per slice / snapshot
set -e
HERE="${0:A:h}"
export LORA_ROOT="${LORA_ROOT:-$HOME/MLX_Models/html_game_sft}"
export LORA_BASE="${LORA_BASE:-$HOME/MLX_Models/Qwen3.8-27B-mxfp8}"
export LORA_PY="${LORA_PY:-$HOME/Agents/.venv/bin/python}"
export LORA_PORT="${LORA_PORT:-8766}"
what="${1:-default}"

mkdir -p "$LORA_ROOT/logs" "$LORA_ROOT/jsonl"
cd "$LORA_ROOT"
echo "LORA_ROOT=$LORA_ROOT"
echo "LORA_BASE=$LORA_BASE"

start_monitor() {
  nohup "$LORA_PY" "$HERE/serve_progress.py" >> logs/dashboard.log 2>&1 &
  disown
  echo "monitor  http://127.0.0.1:$LORA_PORT/   (log: $LORA_ROOT/logs/dashboard.log)"
}
start_ingest() {
  SKIP_SPLIT=1 nohup "$LORA_PY" "$HERE/ingest.py" >> logs/ingest.log 2>&1 &
  disown
  echo "ingest   (log: $LORA_ROOT/logs/ingest.log)"
}
start_train() {
  nohup "$LORA_PY" "$HERE/run_slices.py" >> logs/supervisor.log 2>&1 &
  disown
  echo "trainer  (log: $LORA_ROOT/logs/train.log)   tail -f $LORA_ROOT/logs/train.log"
}

case "$what" in
  default) start_monitor; start_train ;;
  all)     start_monitor; start_ingest; start_train ;;
  monitor) start_monitor ;;
  train)   start_train ;;
  ingest)  start_ingest ;;
  *) echo "usage: $0 [default|all|monitor|train|ingest]"; exit 1 ;;
esac
