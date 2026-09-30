#!/bin/zsh
# Start training jobs in the background. Each one is reparented to launchd
# (PPID 1) in its own session, so quitting Cursor or closing Terminal does not
# stop it. nohup is not enough: a Cursor shell kills its process group.
#
#   ./start.sh            monitor + trainer            (default)
#   ./start.sh all        monitor + downloader + trainer (HTML-game LoRA only)
#   ./start.sh monitor    progress page only            http://127.0.0.1:$LORA_PORT/
#   ./start.sh train      trainer (run_slices.py) only
#   ./start.sh ingest     GitHub game downloader only   (HTML-game LoRA only)
#   ./start.sh server     Qwen server on :8000, restarted if it exits
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
#   ./start.sh small              page + trainer + review (no Stack download)
#   ./start.sh small continue     same, plus the Stack download
#   ./start.sh small-train        train_small.py only
#   ./start.sh small-monitor      progress page only
#   ./start.sh small-data         build shards from scratch: own games + github-code-clean
#   ./start.sh small-retok DIR    re-tokenize DIR/shards for SMALL_BASE (new model, same data)
#   ./start.sh small-quality      CPU quality passes (score, then edu + browser), niced
#   ./start.sh small-stack        The Stack v2 download (resumes logs/stack_state.json)
#   ./start.sh small-watch        every 30 min: keep last_good, resume it if loss blows up
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

# Leave the Cursor/Terminal process group. The parent waits until the grandchild
# has called setsid, then exits, so the job's parent is launchd.
detach() {
  local log="$1"
  shift
  "$LORA_PY" -c '
import os, sys
log, cmd = sys.argv[1], sys.argv[2:]
r, w = os.pipe()
if os.fork() > 0:
    os.close(w)
    os.read(r, 1)
    raise SystemExit(0)
os.close(r)
os.setsid()
if os.fork() > 0:
    os.close(w)
    os._exit(0)
os.write(w, b"1")
os.close(w)
fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.close(fd)
dn = os.open(os.devnull, os.O_RDONLY)
os.dup2(dn, 0)
os.close(dn)
os.execvp(cmd[0], cmd)
' "$log" "$@"
}

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
  # First run on a Mac: make the venv and fetch the default 1B base. Shards must already be copied.
  if [[ "$what" == "small" ]]; then
    ls "$SMALL_ROOT"/shards/*.jsonl >/dev/null 2>&1 || ls "$SMALL_ROOT"/sets/*/tokens/*.jsonl >/dev/null 2>&1 || { echo "no token shards in $SMALL_ROOT — copy sets/ or shards/ first"; exit 1; }
    if [[ ! -x "$LORA_PY" ]]; then
      echo "creating ${LORA_PY:h:h}"
      python3.12 -m venv "${LORA_PY:h:h}"
      "$LORA_PY" -m pip install mlx mlx-lm tokenizers numpy pyarrow huggingface_hub torch transformers playwright
      "$LORA_PY" -m playwright install chromium
    fi
    [[ -x "$BROWSER_PY" ]] || BROWSER_PY="$LORA_PY"
    if [[ ! -f "$SMALL_BASE/config.json" && "$SMALL_BASE" == "$HOME/MLX_Models/MiniCPM5-1B-Base" ]]; then
      echo "downloading MiniCPM5-1B-Base"
      mkdir -p "${SMALL_BASE:h}"
      "${LORA_PY:h}/hf" download openbmb/MiniCPM5-1B-Base --local-dir "$SMALL_BASE"
    fi
  fi
  [[ -f "$SMALL_BASE/config.json" ]] || { echo "no model at SMALL_BASE (see README §9 to download)"; exit 1; }
  # Python only. A zsh wrapper's command line names every pass, including ones that already exited.
  quality_py() {
    local flag="$1"
    ps -ax -o command= | awk -v flag="$flag" '
      /quality_worker\.py/ && /Python/ && !/zsh/ && !/awk/ {
        if (flag == "") found = 1
        else {
          n = split($0, a, " ")
          for (i = 1; i <= n; i++) if (a[i] == flag) found = 1
        }
      }
      END { exit found ? 0 : 1 }
    '
  }
  # macOS ps lstart is "Thu Sep 24 22:44:19 2026" (day may be space-padded).
  # Returns 0 when any file is newer than the process, or the start time cannot be read.
  files_newer_than_pid() {
    local pid="$1" proc src f
    shift
    proc=$(ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//')
    proc=$(date -j -f "%a %b %e %T %Y" "$proc" +%s 2>/dev/null || true)
    [[ -n "$proc" ]] || return 0
    for f in "$@"; do
      src=$(stat -f %m "$f")
      if (( src > proc )); then
        return 0
      fi
    done
    return 1
  }
  # Wait until pid is gone. The trainer is never in this list.
  wait_gone() {
    local pid="$1" i=0
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && (( i < 25 )); do
      sleep 0.2
      i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  }
  small_monitor() {
    local pid
    pid=$(lsof -nP -iTCP:"$SMALL_PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
    if [[ -n "$pid" ]]; then
      if files_newer_than_pid "$pid" "$HERE/serve_progress.py" "$HERE/progress.html"; then
        echo "monitor code is newer than pid $pid; reloading the page only"
        wait_gone "$pid"
      else
        echo "monitor already up  http://127.0.0.1:$SMALL_PORT/"
        return
      fi
    fi
    LORA_ROOT="$SMALL_ROOT" LORA_PORT="$SMALL_PORT" SMALL_BASE="$SMALL_BASE" detach "$SMALL_ROOT/logs/dashboard.log" "$LORA_PY" "$HERE/serve_progress.py"
    echo "monitor  http://127.0.0.1:$SMALL_PORT/   (log: $SMALL_ROOT/logs/dashboard.log)"
  }
  small_train() {
    # Page Stop stays stopped until Resume. Do not undo it from this script.
    if [[ -f "$SMALL_ROOT/logs/hold.json" ]] && grep -q '"state": "held"' "$SMALL_ROOT/logs/hold.json"; then
      echo "training is stopped from the page (logs/hold.json). Press Resume there. This does not start the trainer."
      return
    fi
    # One GPU: refuse a second trainer, whichever SMALL_ROOT it uses.
    if pgrep -f "train_small.py" > /dev/null; then
      echo "train_small.py already running (pid $(pgrep -f train_small.py | head -1))"; return
    fi
    ls shards/*.jsonl > /dev/null 2>&1 || ls sets/*/tokens/*.jsonl > /dev/null 2>&1 || { echo "no token shards in $SMALL_ROOT (sets/*/tokens or shards/)"; exit 1; }
    detach "$SMALL_ROOT/logs/train.out" "$LORA_PY" "$HERE/small/train_small.py" ${=SMALL_TRAIN_ARGS}
    echo "trainer  (log: $SMALL_ROOT/logs/train.log)   tail -f $SMALL_ROOT/logs/train.log"
  }
  small_watch() {
    # Shell only. Keeps last_good and restarts from it. Does not call an agent.
    # A newer script is reloaded. The trainer is left for the new watcher (it honors page Stop).
    local pid
    pid=$(pgrep -f "small/watch_train.sh" | head -1 || true)
    if [[ -n "$pid" ]]; then
      if files_newer_than_pid "$pid" "$HERE/small/watch_train.sh"; then
        echo "watch script is newer than pid $pid; reloading"
        wait_gone "$pid"
      else
        echo "watch already running (pid $pid)"
        return
      fi
    fi
    detach "$SMALL_ROOT/logs/watch.log" /bin/zsh "$HERE/small/watch_train.sh"
    echo "watch    every 30 min (log: $SMALL_ROOT/logs/watch.log)"
  }
  small_quality() {
    # Each pass on its own. File check stays up and scores shards the download adds.
    # Running this again starts only the passes that are not already up.
    if ! quality_py "--follow"; then
      detach "$SMALL_ROOT/logs/quality.out" nice -n 19 "$LORA_PY" -u "$HERE/small/quality_worker.py" --follow --workers 16
      echo "file check   (log: $SMALL_ROOT/logs/quality.out)"
    else
      echo "file check already running"
    fi
    if ! quality_py "--edu"; then
      # A clean exit means nothing left to score: stop. A crash starts the pass again.
      # The command line includes --edu so the page shows it running until that exit.
      detach "$SMALL_ROOT/logs/edu.out" zsh -c "while true; do nice -n 19 '$LORA_PY' -u '$HERE/small/quality_worker.py' --edu --follow --workers 12; status=\$?; if [[ \$status -eq 0 ]]; then exit 0; fi; sleep 15; done"
      echo "code quality (log: $SMALL_ROOT/logs/edu.out)"
    else
      echo "code quality already running"
    fi
    if ! quality_py "--follow-browser"; then
      detach "$SMALL_ROOT/logs/browser.out" env PLAYWRIGHT_BROWSERS_PATH="$HOME/Library/Caches/ms-playwright" \
        nice -n 19 "$BROWSER_PY" -u "$HERE/small/quality_worker.py" --follow-browser --workers 6
      echo "browser      (log: $SMALL_ROOT/logs/browser.out)"
    else
      echo "browser already running"
    fi
  }
  case "$what" in
    small)
      # Page, trainer, and quality. `continue` is the only extra: resume the Stack download.
      small_monitor
      small_train
      small_watch
      small_quality
      if [[ "${2:-}" == "continue" ]]; then
        if pgrep -f "data.py --source stack" > /dev/null; then
          echo "stack download already running (pid $(pgrep -f 'data.py --source stack' | head -1))"
        else
          detach "$SMALL_ROOT/logs/stack.out" nice -n 5 "$LORA_PY" -u "$HERE/small/data.py" --source stack --workers 32 --max-tokens 8000000000
          echo "stack    (log: $SMALL_ROOT/logs/stack.out)   resumes logs/stack_state.json"
        fi
      elif [[ -n "${2:-}" ]]; then
        echo "usage: $0 small [continue]"; exit 1
      fi
      ;;
    small-train)   small_train ;;
    small-monitor) small_monitor ;;
    small-data)
      detach "$SMALL_ROOT/logs/data.out" zsh -c "'$LORA_PY' '$HERE/small/data.py' --source own && '$LORA_PY' '$HERE/small/data.py' --source gcc --workers 8 --max-chunks 50"
      echo "data     (log: $SMALL_ROOT/logs/data.out)" ;;
    small-retok)
      [[ -n "$2" ]] || { echo "usage: $0 small-retok <other SMALL_ROOT with shards/>"; exit 1; }
      detach "$SMALL_ROOT/logs/data.out" "$LORA_PY" "$HERE/small/data.py" --source retok --from "$2" --workers 12
      echo "retok    $2/shards -> $SMALL_ROOT/shards   (log: $SMALL_ROOT/logs/data.out)" ;;
    small-quality) small_quality ;;
    small-watch)   small_watch ;;
    small-synth)
      "$LORA_PY" -u "$HERE/small/data.py" --source synth
      ;;
    small-export)
      if pgrep -f "data.py --source export" > /dev/null; then
        echo "export already running (pid $(pgrep -f 'data.py --source export' | head -1))"; return
      fi
      detach "$SMALL_ROOT/logs/export.out" "$LORA_PY" -u "$HERE/small/data.py" --source export
      echo "export   train text from tokens (log: $SMALL_ROOT/logs/export.out)"
      ;;
    small-stack)
      # Resumes logs/stack_state.json. Same workers and cap as the Sep 24 run.
      if pgrep -f "data.py --source stack" > /dev/null; then
        echo "stack download already running (pid $(pgrep -f 'data.py --source stack' | head -1))"; return
      fi
      detach "$SMALL_ROOT/logs/stack.out" nice -n 5 "$LORA_PY" -u "$HERE/small/data.py" --source stack --workers 32 --max-tokens 8000000000
      echo "stack    (log: $SMALL_ROOT/logs/stack.out)   resumes logs/stack_state.json" ;;
    *) echo "usage: $0 [small|small-train|small-monitor|small-data|small-retok DIR|small-quality|small-stack|small-synth|small-export|small-watch]"; exit 1 ;;
  esac
  exit 0
fi

mkdir -p "$LORA_ROOT/logs" "$LORA_ROOT/jsonl"
cd "$LORA_ROOT"
echo "LORA_ROOT=$LORA_ROOT"
echo "LORA_BASE=$LORA_BASE"

start_monitor() {
  detach "$LORA_ROOT/logs/dashboard.log" "$LORA_PY" "$HERE/serve_progress.py"
  echo "monitor  http://127.0.0.1:$LORA_PORT/   (log: $LORA_ROOT/logs/dashboard.log)"
}
start_ingest() {
  SKIP_SPLIT=1 detach "$LORA_ROOT/logs/ingest.log" "$LORA_PY" "$HERE/ingest.py"
  echo "ingest   (log: $LORA_ROOT/logs/ingest.log)"
}
start_train() {
  detach "$LORA_ROOT/logs/supervisor.log" "$LORA_PY" "$HERE/run_slices.py"
  echo "trainer  (log: $LORA_ROOT/logs/train.log)   tail -f $LORA_ROOT/logs/train.log"
}
# If port 8000 is open, only watch. If that process exits, start it again.
start_mlx_server() {
  if pgrep -f "MLX_SERVER_KEEPALIVE" >/dev/null; then
    echo "server watchdog already running (log: $LORA_ROOT/logs/mlx_server.log)"
    return
  fi
  detach "$LORA_ROOT/logs/mlx_server.log" env \
    MLX_VLM_PRELOAD_MODEL="$LORA_BASE" \
    MLX_VLM_MAX_NUM_SEQS="${MLX_VLM_MAX_NUM_SEQS:-64}" \
    MLX_VLM_LOG_PROGRESS_INTERVAL=0 \
    MLX_VLM_ENABLE_THINKING=0 \
    LORA_PY="$LORA_PY" \
    zsh -c '
    # MLX_SERVER_KEEPALIVE
    while true; do
      if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        sleep 20
        continue
      fi
      echo "mlx server starting $(date)"
      "$LORA_PY" -c "import logging; logging.basicConfig(level=logging.WARNING); import uvicorn; uvicorn.run(\"mlx_vlm.server:app\", host=\"127.0.0.1\", port=8000, workers=1, access_log=False, log_level=\"warning\")"
      echo "mlx server exited $(date), restarting in 5s"
      sleep 5
    done
  '
  echo "server watchdog  http://127.0.0.1:8000/  (log: $LORA_ROOT/logs/mlx_server.log)"
}

case "$what" in
  default) start_monitor; start_train ;;
  all)     start_monitor; start_ingest; start_train ;;
  monitor) start_monitor ;;
  train)   start_train ;;
  ingest)  start_ingest ;;
  server)  start_mlx_server ;;
  *) echo "usage: $0 [default|all|monitor|train|ingest|server]"; exit 1 ;;
esac
