#!/usr/bin/env bash
# Pull newest code from GitHub (Agent_learning_overlay). Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git pull --ff-only origin main
echo "Up to date: $(git log -1 --oneline)"
