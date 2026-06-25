#!/bin/bash
# Double-click this in Finder — starts Z-Image and opens Asset Studio in your browser.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x ".venv/bin/python" ]]; then
  osascript -e 'display alert "Asset Studio needs setup" message "Run ./scripts/setup.sh in the Agent_learning folder first."' || true
  exit 1
fi
exec .venv/bin/python scripts/asset_studio.py
