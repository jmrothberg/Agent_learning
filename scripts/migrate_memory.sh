#!/usr/bin/env bash
# scripts/migrate_memory.sh — Seamless division of memory into three tiers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_MEM="$REPO_ROOT/memory"
OLD_LEGACY_MEM="$REPO_ROOT/games/memory"
LIVE_MEM="$REPO_ROOT/games/game-memory"
SHORT_TERM_MEM="$REPO_ROOT/games/goals"

echo "=== Tiered Memory Migration Tool ==="

# 1. Handle legacy games/memory if it exists but root memory does not
if [ -d "$OLD_LEGACY_MEM" ] && [ ! -d "$OLD_MEM" ]; then
    echo "Found legacy games/memory directory. Moving it to root memory for unified transition..."
    mv "$OLD_LEGACY_MEM" "$OLD_MEM"
fi

# Ensure target directories exist
echo "Creating new tiered memory directories..."
mkdir -p "$LIVE_MEM/skeletons" "$SHORT_TERM_MEM"

# 2. Migrate Goals (Short-Term Outcome History)
if [ -d "$OLD_MEM/goals" ]; then
    n_goals=$(find "$OLD_MEM/goals" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    if [ "$n_goals" -gt 0 ]; then
        echo "Migrating $n_goals goal records to short-term storage (games/goals/)..."
        # Copy everything inside memory/goals/ to games/goals/
        cp -a "$OLD_MEM/goals"/. "$SHORT_TERM_MEM/"
        echo "Removing moved goals from pristine storage..."
        rm -rf "$OLD_MEM/goals"
    else
        rm -rf "$OLD_MEM/goals"
    fi
fi

# 3. Migrate Won Skeletons (Game-Specific Skeletons)
if [ -d "$OLD_MEM/skeletons" ]; then
    n_won=$(find "$OLD_MEM/skeletons" -maxdepth 1 -name "won_*" 2>/dev/null | wc -l)
    if [ "$n_won" -gt 0 ]; then
        echo "Migrating $n_won won skeletons to live learned storage (games/game-memory/skeletons/)..."
        # Move all won_* files (html and json sidecars)
        find "$OLD_MEM/skeletons" -maxdepth 1 -name "won_*" -exec cp -a {} "$LIVE_MEM/skeletons/" \;
        echo "Removing moved won skeletons from pristine storage..."
        find "$OLD_MEM/skeletons" -maxdepth 1 -name "won_*" -delete
    fi
fi

# 4. Migrate Playbook
if [ -f "$OLD_MEM/playbook.jsonl" ]; then
    echo "Migrating live playbook to live learned storage (games/game-memory/playbook.jsonl)..."
    cp -p "$OLD_MEM/playbook.jsonl" "$LIVE_MEM/playbook.jsonl"
else
    echo "No playbook found in pristine root. Initializing a clean live playbook..."
    # If no base playbook exists, python boot will auto-seed both on next run.
fi

# 5. Migrate Mistakes
if [ -f "$OLD_MEM/mistakes.jsonl" ]; then
    echo "Migrating mistakes to live learned storage (games/game-memory/mistakes.jsonl)..."
    mv "$OLD_MEM/mistakes.jsonl" "$LIVE_MEM/mistakes.jsonl"
fi

# 6. Reset Pristine Reference Playbook to Verified Seed Bullets
echo "Resetting pristine reference playbook to verified seed bullets..."
mkdir -p "$OLD_MEM"
python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); import memory; memory.Playbook('$OLD_MEM')._save_all_to_path(memory.SEED_BULLETS, memory.Path('$OLD_MEM/playbook.jsonl'))"

# 7. Ensure pristine skeletons contains all generic scaffolds
echo "Ensuring pristine skeletons are populated..."
python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT'); import memory; memory.GameMemory('$OLD_MEM').ensure()"

echo "=== Migration Complete ==="
echo "Pristine Memory:     $OLD_MEM"
echo "Learned Memory:      $LIVE_MEM"
echo "Short-Term Memory:   $SHORT_TERM_MEM"
echo "=========================="
