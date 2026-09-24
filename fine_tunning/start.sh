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
#
# SMALL MODEL (full fine-tune of a ~1B base on HTML/JS, code in small/, README §9):
#   ./start.sh small              monitor + train_small.py (starts or resumes)
#   ./start.sh small-train        train_small.py only
#   ./start.sh small-monitor      progress page only
#   ./start.sh small-data         build shards from scratch: own games + github-code-clean
#   ./start.sh small-retok DIR    re-tokenize DIR/shards for SMALL_BASE (new model, same data)
#   ./start.sh small-quality      CPU quality passes (score, then edu + browser), niced
#   SMALL_ROOT=~/MLX_Models/html_js_small     shards + quality.sqlite + checkpoints + logs
#   SMALL_BASE=~/MLX_Models/MiniCPM5-1B-Base  base model (never modified)
#   SMALL_PORT=8767                           progress page port
#   SMALL_TRAIN_ARGS="..."                    extra train_small.py flags
#   BROWSER_PY=~/Agent_learning/.venv/bin/python   python with playwright (browser pass)
set -e
HERE="${0:A:h}"
export LORA_ROOT="${LORA_ROOT:-$HOME/MLX_Models/html_game_sft}"
export LORA_BASE="${LORA_BASE:-$HOME/MLX_Models/Qwen3.8-27B-mxfp8}"
export LORA_PY="${LORA_PY:-$HOME/Agents/.venv/bin/python}"
export LORA_PORT="${LORA_PORT:-8766}"
what="${1:-default}"

# SMALL MODEL: own env and folder; the LoRA folders below are not touched.
if [[ "$what" == small* ]]; then
  export SMALL_ROOT="${SMALL_ROOT:-$HOME/MLX_Models/html_js_small}"
  export SMALL_BASE="${SMALL_BASE:-$HOME/MLX_Models/MiniCPM5-1B-Base}"
  export SMALL_PORT="${SMALL_PORT:-8767}"
  BROWSER_PY="${BROWSER_PY:-$HOME/Agent_learning/.venv/bin/python}"
  mkdir -p "$SMALL_ROOT/logs"
  cd "$SMALL_ROOT"
  echo "SMALL_ROOT=$SMALL_ROOT"
  echo "SMALL_BASE=$SMALL_BASE"
  [[ -f "$SMALL_BASE/config.json" ]] || { echo "no model at SMALL_BASE (see README §9 to download)"; exit 1; }
  small_monitor() {
    LORA_ROOT="$SMALL_ROOT" LORA_PORT="$SMALL_PORT" nohup "$LORA_PY" "$HERE/serve_progress.py" >> logs/dashboard.log 2>&1 &
    disown
    echo "monitor  http://127.0.0.1:$SMALL_PORT/   (log: $SMALL_ROOT/logs/dashboard.log)"
  }
  small_train() {
    # One GPU: refuse a second trainer, whichever SMALL_ROOT it uses.
    if pgrep -f "train_small.py" > /dev/null; then
      echo "train_small.py already running (pid $(pgrep -f train_small.py | head -1))"; return
    fi
    ls shards/*.jsonl > /dev/null 2>&1 || { echo "no shards in $SMALL_ROOT/shards (run small-data or small-retok)"; exit 1; }
    nohup "$LORA_PY" "$HERE/small/train_small.py" ${=SMALL_TRAIN_ARGS} >> logs/train.out 2>&1 &
    disown
    echo "trainer  (log: $SMALL_ROOT/logs/train.log)   tail -f $SMALL_ROOT/logs/train.log"
  }
  case "$what" in
    small)         small_monitor; small_train ;;
    small-train)   small_train ;;
    small-monitor) small_monitor ;;
    small-data)
      nohup zsh -c "'$LORA_PY' '$HERE/small/data.py' --source own && '$LORA_PY' '$HERE/small/data.py' --source gcc --workers 8 --max-chunks 50" >> logs/data.out 2>&1 &
      disown
      echo "data     (log: $SMALL_ROOT/logs/data.out)" ;;
    small-retok)
      [[ -n "$2" ]] || { echo "usage: $0 small-retok <other SMALL_ROOT with shards/>"; exit 1; }
      nohup "$LORA_PY" "$HERE/small/data.py" --source retok --from "$2" --workers 12 >> logs/data.out 2>&1 &
      disown
      echo "retok    $2/shards -> $SMALL_ROOT/shards   (log: $SMALL_ROOT/logs/data.out)" ;;
    small-quality)
      nohup zsh -c "nice -n 19 '$LORA_PY' '$HERE/small/quality_worker.py' --workers 16 && \
        { PLAYWRIGHT_BROWSERS_PATH=\$HOME/Library/Caches/ms-playwright nice -n 19 '$BROWSER_PY' '$HERE/small/quality_worker.py' --browser --workers 6 >> logs/browser.out 2>&1 & \
          nice -n 19 '$LORA_PY' '$HERE/small/quality_worker.py' --edu --workers 12; wait; }" >> logs/quality.out 2>&1 &
      disown
      echo "quality  (logs: $SMALL_ROOT/logs/quality.out, browser.out)" ;;
    *) echo "usage: $0 [small|small-train|small-monitor|small-data|small-retok DIR|small-quality]"; exit 1 ;;
  esac
  exit 0
fi

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
