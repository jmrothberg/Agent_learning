#!/bin/zsh
# Manage train_small.py. One status line per check. No agent in the loop.
#
# Every SMALL_WATCH_SEC (default 30 min):
#   healthy loss  -> copy checkpoints/latest to checkpoints/last_good
#   loss above 3  -> resume last_good at a lower peak (1e-5, then 3e-6, then 1e-6)
#   trainer dead, last loss still healthy -> resume checkpoints/latest at the same peak
# Never starts from the base model when a checkpoint is already on disk.
# Never deletes a checkpoint. Old last_good copies move to checkpoints/kept/.
#
# Env: SMALL_ROOT, SMALL_BASE (required), SMALL_WATCH_SEC, SMALL_WATCH_BAD_LOSS.
set -u
ROOT="${SMALL_ROOT:?'set SMALL_ROOT'}"
BASE="${SMALL_BASE:?'set SMALL_BASE'}"
INTERVAL="${SMALL_WATCH_SEC:-1800}"
BAD="${SMALL_WATCH_BAD_LOSS:-3}"
START="${0:A:h:h}/start.sh"
CKPT="$ROOT/checkpoints"
LOG="$ROOT/logs/train.log"
STATE="$ROOT/logs/watch_state"
mkdir -p "$ROOT/logs" "$CKPT"

say() { print -r -- "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

trainer_pid() {
  pgrep -f '/small/train_small.py' | head -n 1 || true
}

# Float compare. Returns 0 when loss is above the bad line.
loss_is_bad() {
  awk -v l="$1" -v b="$BAD" 'BEGIN { exit !(l+0 > b+0) }'
}

last_iter() {
  [[ -f "$LOG" ]] || return 1
  grep 'Train loss' "$LOG" | tail -n 1
}

peak_lr() {
  [[ -f "$LOG" ]] || return 1
  grep '^train ' "$LOG" | tail -n 1 | sed -n 's/.* lr=\([^ ]*\).*/\1/p'
}

counter_update() {
  sed -n 's/.*"update": *\([0-9][0-9]*\).*/\1/p' "$1/counters.json" | head -n 1
}

# A save writes model, then optimizer, then counters, each via rename.
# Wait until counters is the newest file and has been still for 2 minutes.
checkpoint_ready() {
  local d="$1" now newest=0 m f
  [[ -f "$d/counters.json" && -f "$d/model.safetensors" && -f "$d/optimizer.safetensors" ]] || return 1
  now=$(date +%s)
  for f in counters.json model.safetensors optimizer.safetensors; do
    m=$(stat -f %m "$d/$f")
    (( m > newest )) && newest=$m
  done
  (( now - newest >= 120 )) || return 1
  m=$(stat -f %m "$d/counters.json")
  (( m >= $(stat -f %m "$d/model.safetensors") && m >= $(stat -f %m "$d/optimizer.safetensors") ))
}

free_kb() { df -k "$CKPT" | awk 'NR==2 { print $4 }'; }

# Keep the newest healthy save. Previous copies move aside; nothing is removed.
save_last_good() {
  local upd prev avail stamp
  checkpoint_ready "$CKPT/latest" || return 0
  upd=$(counter_update "$CKPT/latest")
  [[ -n "$upd" ]] || return 0
  if [[ -f "$CKPT/last_good/counters.json" ]]; then
    prev=$(counter_update "$CKPT/last_good")
    [[ -n "$prev" && "$upd" -gt "$prev" ]] || return 0
  fi
  avail=$(free_kb)
  if (( avail < 500000000 )); then
    say "disk low avail_kb=$avail skip last_good copy"
    return 0
  fi
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  if [[ -d "$CKPT/last_good_prev" ]]; then
    mkdir -p "$CKPT/kept"
    mv "$CKPT/last_good_prev" "$CKPT/kept/last_good_$stamp"
  fi
  if [[ -d "$CKPT/last_good" ]]; then
    mv "$CKPT/last_good" "$CKPT/last_good_prev"
  fi
  cp -cR "$CKPT/latest" "$CKPT/last_good" || cp -R "$CKPT/latest" "$CKPT/last_good"
  say "saved last_good update=$upd"
}

stop_trainer() {
  local pid i
  pid=$(trainer_pid)
  [[ -n "$pid" ]] || return 0
  say "stop trainer pid=$pid"
  kill "$pid" || true
  i=0
  while kill -0 "$pid" 2>/dev/null && (( i < 45 )); do
    sleep 1
    i=$((i + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" || true
    sleep 1
  fi
}

# Park the dead latest, put last_good back, hide the dead loss curve.
restore_last_good() {
  local stamp
  [[ -d "$CKPT/last_good" ]] || { say "no last_good; not starting from the base"; return 1; }
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  if [[ -d "$CKPT/latest" ]]; then
    mv "$CKPT/latest" "$CKPT/diverged_$stamp"
  fi
  cp -cR "$CKPT/last_good" "$CKPT/latest" || cp -R "$CKPT/last_good" "$CKPT/latest"
  if [[ -f "$LOG" ]]; then
    mv "$LOG" "$ROOT/logs/train_diverged_$stamp.log"
  fi
  say "restored last_good update=$(counter_update "$CKPT/latest")"
}

blows() {
  [[ -f "$STATE" ]] || { print 0; return; }
  sed -n 's/^blows=//p' "$STATE" | head -n 1
}

set_blows() { print -r -- "blows=$1" > "$STATE"; }

# Step the peak down. Empty output means stop restarting.
next_lr() {
  case "$1" in
    5e-5|5e-05|0.00005) print -r -- 1e-5 ;;
    1e-5|1e-05|0.00001) print -r -- 3e-6 ;;
    3e-6|3e-06|0.000003) print -r -- 1e-6 ;;
    1e-6|1e-06|0.000001) print -r -- "" ;;
    *) print -r -- 3e-6 ;;
  esac
}

restart_at() {
  local lr="$1"
  say "restart lr=$lr"
  SMALL_ROOT="$ROOT" SMALL_BASE="$BASE" SMALL_TRAIN_ARGS="--lr $lr" "$START" small-train
}

check() {
  local pid line loss step lr n
  pid=$(trainer_pid)
  line=$(last_iter || true)
  loss=""
  step=""
  if [[ -n "$line" ]]; then
    loss=$(print -r -- "$line" | sed -n 's/.*Train loss \([0-9.][0-9.]*\).*/\1/p')
    step=$(print -r -- "$line" | sed -n 's/^Iter \([0-9][0-9]*\).*/\1/p')
  fi
  lr=$(peak_lr || true)
  [[ -n "$lr" ]] || lr="1e-5"

  if [[ -n "$pid" && -n "$loss" ]] && loss_is_bad "$loss"; then
    n=$(blows)
    n=$((n + 1))
    set_blows "$n"
    say "blow-up loss=$loss step=$step peak=$lr count=$n"
    stop_trainer
    if (( n > 2 )); then
      say "stopped restarting after $n blow-ups"
      return 0
    fi
    lr=$(next_lr "$lr")
    [[ -n "$lr" ]] || { say "no lower lr; left stopped"; return 0; }
    restore_last_good || return 0
    restart_at "$lr"
    return 0
  fi

  if [[ -z "$pid" ]]; then
    if [[ -n "$loss" ]] && loss_is_bad "$loss"; then
      say "trainer down after bad loss=$loss; waiting for a last_good restore on the next blow-up check"
      n=$(blows)
      n=$((n + 1))
      set_blows "$n"
      if (( n > 2 )); then
        say "stopped restarting after $n blow-ups"
        return 0
      fi
      lr=$(next_lr "$lr")
      [[ -n "$lr" ]] || { say "no lower lr; left stopped"; return 0; }
      restore_last_good || return 0
      restart_at "$lr"
      return 0
    fi
    if [[ ! -d "$CKPT/latest" && ! -d "$CKPT/last_good" ]]; then
      say "trainer down and no checkpoint; not starting from the base"
      return 0
    fi
    say "trainer down loss=${loss:-none} step=${step:-none}; resume peak=$lr"
    restart_at "$lr"
    return 0
  fi

  save_last_good
  say "ok loss=${loss:-waiting} step=${step:-0} peak=$lr pid=$pid"
}

say "watch start interval=${INTERVAL}s bad_loss>$BAD root=$ROOT"
while true; do
  check || say "check failed status=$?"
  sleep "$INTERVAL"
done
