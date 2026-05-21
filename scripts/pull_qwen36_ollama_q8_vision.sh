#!/usr/bin/env bash
# Ubuntu + Ollama: ONE model — Q8_0 GGUF with vision (image input).
#
# SAFE tag (verified linux/amd64 GGUF, NOT MLX):
#   qwen3.6:27b-q8_0
#
# macOS / MLX ONLY on Ubuntu (412: this model requires macOS) — do NOT use:
#   qwen3.6:27b-mxfp8
#   qwen3.6:27b-nvfp4
#   qwen3.6:27b-mlx-bf16
#   qwen3.6:27b-coding-mxfp8
#   qwen3.6:27b-coding-nvfp4
#
# Already installed, vision-capable but Q4 not Q8:
#   qwen3.6:27b  (alias of qwen3.6:27b-q4_K_M)

set -euo pipefail
TAG="qwen3.6:27b-q8_0"
LOG="${LOG:-/tmp/ollama-pull-q8-vlm.log}"

echo "Pulling ${TAG} only (GGUF Q8_0, vision, Linux)..."
echo "Log: ${LOG}"
ollama pull "$TAG" 2>&1 | tee "$LOG"
echo ""
ollama show "$TAG" | grep -iE "quantization|Capabilities|vision" || true
echo ""
echo "Run: OLLAMA_MODEL=${TAG} .venv/bin/python chat.py"
