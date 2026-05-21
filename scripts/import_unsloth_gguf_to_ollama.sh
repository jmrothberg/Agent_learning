#!/usr/bin/env bash
# Import a text-only Unsloth/HF GGUF blob into native Ollama format.
#
# Online/registry tools involved:
#   1. Hugging Face Hub  ->  ollama run hf.co/unsloth/...  (pulls GGUF + mmproj)
#   2. gguf-to-ollama    ->  https://github.com/jonathanhecl/gguf-to-ollama/releases
#   3. ollama create     ->  writes Ollama manifest (library tag you can ollama run)
#
# HF pulls that include mmproj often fail with "unknown architecture: qwen35"
# on the llama.cpp path. Fix: re-import the *text* GGUF only with RENDERER/PARSER
# qwen3.5 (same as official qwen3.6:27b).
#
# VISION (image input): Unsloth ships text + mmproj as separate GGUFs. Ollama 0.24
# cannot run that split stack on the native engine; dual-FROM still fails.
# This script builds a TEXT-ONLY tag — do not use it for screenshots.
# For image input in Ollama, use an official fused build (vision-capable), e.g.:
#   ollama run qwen3.6:27b
# Optional broken experiment (text + mmproj, not loadable today):
#   scripts/ollama_models/qwen3.6-27b-unsloth-q8-with-mmproj.Modelfile
#
# Usage:
#   1. ollama run hf.co/unsloth/Qwen3.6-27B-GGUF:Q8_0   # HF pull (text GGUF + mmproj)
#   2. ./scripts/import_unsloth_gguf_to_ollama.sh        # registers text-only tag (keep this)
#   ollama run qwen3.6-27b-unsloth-q8   # Q8 text/chat only — NOT a VLM, no screenshots
#   ollama run qwen3.6:27b              # vision-capable (official Q4), use for images
#
# Best Ubuntu Q8 + image input (official fused GGUF, vision-capable):
#   ollama pull qwen3.6:27b-q8_0               # or register from -partial blob:
#   ollama create qwen3.6:27b-q8_0 -f scripts/ollama_models/qwen3.6-27b-q8_0-official.Modelfile
#   OLLAMA_MODEL=qwen3.6:27b-q8_0 .venv/bin/python chat.py

set -euo pipefail

# Keep this tag for working Q8 text on Ubuntu. Name is intentionally not "*vlm*".
MODEL_TAG="${MODEL_TAG:-qwen3.6-27b-unsloth-q8}"
HF_TAG="${HF_TAG:-hf.co/unsloth/Qwen3.6-27B-GGUF:Q8_0}"
OLLAMA_MODELS="${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE_MODELFILE="$ROOT/scripts/ollama_models/qwen3.6-27b-unsloth-q8.Modelfile"
BUILD_MODELFILE="${TMPDIR:-/tmp}/qwen3.6-27b-unsloth-q8.build.Modelfile"

_resolve_text_gguf_blob() {
  # Prefer explicit path, then HF manifest model layer, then known Q8 digest.
  if [[ -n "${GGUF_BLOB:-}" && -f "$GGUF_BLOB" ]]; then
    echo "$GGUF_BLOB"
    return 0
  fi
  local known="${OLLAMA_MODELS}/blobs/sha256-f93f517f38e696d35a1a7df2c0e3155a64f4c4dcd662107a146ae263f7fb14ce"
  if [[ -f "$known" ]]; then
    echo "$known"
    return 0
  fi
  local manifest="${OLLAMA_MODELS}/manifests/hf.co/unsloth/Qwen3.6-27B-GGUF/Q8_0"
  if [[ -f "$manifest" ]]; then
    python3 - "$manifest" "$OLLAMA_MODELS" <<'PY'
import json, sys
manifest, root = sys.argv[1], sys.argv[2]
data = json.load(open(manifest))
layers = data.get("layers") or []
# Largest model layer = main text GGUF (mmproj is ~0.9GB).
best = max(
    (l for l in layers if l.get("mediaType") == "application/vnd.ollama.image.model"),
    key=lambda l: int(l.get("size") or 0),
    default=None,
)
if not best:
    sys.exit(1)
digest = best["digest"].removeprefix("sha256:")
print(f"{root}/blobs/sha256-{digest}")
PY
    return 0
  fi
  return 1
}

if ! GGUF_BLOB="$(_resolve_text_gguf_blob)"; then
  echo "Text GGUF blob not found. Pulling ${HF_TAG} (downloads text + mmproj)..." >&2
  ollama pull "$HF_TAG"
  GGUF_BLOB="$(_resolve_text_gguf_blob)" || {
    echo "Still no text GGUF blob after pull." >&2
    exit 1
  }
fi

if [[ ! -f "$TEMPLATE_MODELFILE" ]]; then
  echo "Missing template: $TEMPLATE_MODELFILE" >&2
  exit 1
fi

# Build Modelfile with the resolved blob path (template may have a stale hardcoded path).
{
  head -n 1 "$TEMPLATE_MODELFILE"
  echo "FROM ${GGUF_BLOB}"
  tail -n +3 "$TEMPLATE_MODELFILE"
} > "$BUILD_MODELFILE"

echo "Creating TEXT-ONLY Ollama model ${MODEL_TAG} (no mmproj, no image input)..."
echo "  weights: ${GGUF_BLOB}"
ollama create "$MODEL_TAG" -f "$BUILD_MODELFILE"
echo "Done. Text/chat: ollama run ${MODEL_TAG}"
echo "Images/screenshots: use qwen3.6:27b or qwen3.6:27b-q8_0 (vision-capable), not this tag."
