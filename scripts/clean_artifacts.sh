#!/usr/bin/env bash
# Wipe stale per-session artifacts from games/ so the working tree stays
# focused on what's actually useful. Run any time the games/ directory
# starts feeling cluttered with old runs, or when you want a fresh
# baseline for the agent's "memory" subsystem.
#
# DELETES (deliberately destructive — these are all generated artifacts
# that the agent re-creates on its next session, no manual work lost):
#   - games/<slug>_<ts>.html  + .best.html siblings
#   - games/traces/*          (.log + .jsonl + .conversation.md)
#   - games/snapshots/<slug>_<ts>/   (per-iteration HTML + screenshots)
#   - games/game-memory/skeletons/won_*   (auto-promoted past wins — risky;
#                                     can lock in the wrong game from a
#                                     mislabeled session)
#   - games/goals/*            (per-session outcome cache)
#   - games/game-memory/mistakes.jsonl    (small but stale)
#   - games/tune/*/                  (historical tune-battery run dirs;
#                                     KEEPS games/tune/battery.jsonl which
#                                     is the canonical test-definition
#                                     file, not generated output)
#   - games/tune_serial10/run_*/     (serial tune batches — HTML + traces)
#
# KEEPS (load-bearing — agent depends on these):
#   - memory/playbook.jsonl              (hand-curated seed bullets)
#   - memory/skeletons/canvas_basic.html (bundled default skeleton)
#   - goodgame/                          (curated wins — NEVER deleted here)
#   - games/_asset_cache/                      (Z-Image-Turbo PNG cache;
#                                               cache hits = free)
#   - games/_smoke/doom.png                    (one-off proof-of-concept;
#                                               tiny + harmless. Remove
#                                               by hand if you want it
#                                               gone.)
#
# Pass --yes (or -y) to skip the confirmation prompt.

set -euo pipefail
cd "$(dirname "$0")/.."

YES=""
for a in "$@"; do
    case "$a" in
        --yes|-y) YES=1 ;;
    esac
done

count_lines() {
    if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi
}

# `find` over `ls glob` so an empty match doesn't abort under set -e.
n_html=$(find games -maxdepth 1 -name "*.html" -type f 2>/dev/null | wc -l)
n_traces=$(find games/traces -maxdepth 1 -type f 2>/dev/null | wc -l)
n_snapshots=$(find games/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
n_won=$(find games/game-memory/skeletons -maxdepth 1 -name "won_*" 2>/dev/null | wc -l)
n_goals=$(find games/goals -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
n_mistakes=$(count_lines games/game-memory/mistakes.jsonl)
n_tune_runs=$(find games/tune -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
n_tune_serial=$(find games/tune_serial10 -maxdepth 1 -mindepth 1 -type d -name "run_*" 2>/dev/null | wc -l)

echo "About to delete:"
echo "  $n_html  per-session game HTML files"
echo "  $n_traces  trace files (.log + .jsonl + .conversation.md)"
echo "  $n_snapshots  snapshot directories"
echo "  $n_won  auto-promoted skeletons (won_*)"
echo "  $n_goals  per-session goal records"
echo "  $n_mistakes  lines of mistakes.jsonl"
echo "  $n_tune_runs  tune-battery run directories"
echo "  $n_tune_serial  tune_serial10 run directories (run_01, run_02, …)"
echo
echo "Keeping:"
echo "  memory/playbook.jsonl"
echo "  memory/skeletons/canvas_basic.html"
echo "  goodgame/                      (curated wins — never touched)"
echo "  games/tune/battery.jsonl       (canonical test goals)"
echo "  games/_asset_cache/  (cache speedup; hits are free)"
echo "  games/_smoke/        (small smoke-test artifacts)"
echo "  nohup.out            (delete manually if desired)"

if [ -z "$YES" ]; then
    printf "Proceed? [y/N] "
    read -r ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "aborted."; exit 1 ;;
    esac
fi

# Order matters: wipe per-session asset/sound dirs FIRST so that if the
# script is interrupted mid-run we don't leave orphaned _assets/_sounds
# dirs without their matching .html. HTML and traces go last.

# Per-session asset dirs (games/<slug>_<ts>_assets/). The shared cache
# at games/_asset_cache/ stays — that's what makes re-runs free.
find games -maxdepth 1 -mindepth 1 -type d -name "*_assets" \
    -not -name "_asset_cache" -exec rm -rf {} +

# Per-session sound dirs (games/<slug>_<ts>_sounds/). Same logic — the
# shared cache at games/_sound_cache/ stays so re-runs hit it.
find games -maxdepth 1 -mindepth 1 -type d -name "*_sounds" \
    -not -name "_sound_cache" -exec rm -rf {} +

# Sweep up any leftover empty dirs (failed asset generations leave
# behind 0-byte session dirs even when no HTML was produced).
find games -maxdepth 1 -mindepth 1 -type d \
    \( -name "*_assets" -o -name "*_sounds" \) -empty -delete 2>/dev/null || true

# Per-session HTML files (don't touch directory structure).
rm -f games/*.html

# Traces: directory contents, but keep the directory itself.
if [ -d games/traces ]; then
    find games/traces -maxdepth 1 -type f -delete
fi

# Snapshots: drop ALL per-session subdirs; keep the parent.
if [ -d games/snapshots ]; then
    find games/snapshots -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} +
fi

# Skeletons: drop only won_* (KEEP canvas_basic.html and any other
# hand-placed file).
rm -f games/game-memory/skeletons/won_*

# Goals: drop every per-session subdir.
if [ -d games/goals ]; then
    find games/goals -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} +
fi

# Mistakes: drop the file entirely (recreated on next session).
rm -f games/game-memory/mistakes.jsonl

# Tune-battery runs: drop subdirs, keep battery.jsonl test definitions.
if [ -d games/tune ]; then
    find games/tune -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} +
fi

# Serial tune batches: drop run_XX dirs (HTML, traces, overnight.log, checkpoints).
if [ -d games/tune_serial10 ]; then
    find games/tune_serial10 -maxdepth 1 -mindepth 1 -type d -name "run_*" -exec rm -rf {} +
fi

echo
echo "✓ cleaned. games/ is now:"
du -sh games/* 2>/dev/null | sort -k2
