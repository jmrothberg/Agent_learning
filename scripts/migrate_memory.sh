#!/usr/bin/env bash
# scripts/migrate_memory.sh — Safely migrate games/memory/ to memory/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_MEM="$REPO_ROOT/games/memory"
NEW_MEM="$REPO_ROOT/memory"

echo "=== Memory Migration Tool ==="

if [ ! -d "$OLD_MEM" ]; then
    echo "No legacy memory directory found at $OLD_MEM."
    echo "Ensuring new memory directory structure exists..."
    mkdir -p "$NEW_MEM/skeletons" "$NEW_MEM/goals"
    echo "Done."
    exit 0
fi

echo "Found legacy memory at: $OLD_MEM"
echo "Creating new memory directory at: $NEW_MEM"
mkdir -p "$NEW_MEM"

# Copy all contents recursively
echo "Copying memory contents..."
cp -a "$OLD_MEM"/. "$NEW_MEM/"

# Verification check
if [ -f "$NEW_MEM/playbook.jsonl" ]; then
    echo "Verification SUCCESS: playbook.jsonl copied successfully."
    echo "Removing legacy memory directory..."
    rm -rf "$OLD_MEM"
    echo "Migration completed successfully!"
else
    echo "ERROR: Verification failed. playbook.jsonl not found in $NEW_MEM."
    echo "Aborting cleanup. Your files in $OLD_MEM have been left untouched."
    exit 1
fi
