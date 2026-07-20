#!/bin/bash
# Double-click this in Finder to open Terminal and start the overnight Q&A.
# Asks: prompt numbers → max iterations → VLM yes/no → MLX model → confirm → run.
cd "$(dirname "$0")"
exec bash eval/overnight.sh --interactive
