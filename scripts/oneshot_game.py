#!/usr/bin/env python3
"""One-shot the model: bypass the agent loop entirely.

Sends ONE prompt to the local 27B (or whichever MLX model is loaded),
streams the response, strips any tags/code-fences, and saves to a
single HTML file. No planning phase, no playbook, no iteration loop,
no assets pipeline, no probes. Just: goal → HTML.

Use when the agent's multi-turn dynamics are doing more harm than
good (the 27B hits repetition loops on complex multi-stage prompts;
a single complete generation often produces a better playable game
than 8 iters of plan/fix/critique).

Usage:
    .venv/bin/python scripts/oneshot_game.py \\
        "DOOM-style first-person shooter with three.js" \\
        --out games/my_game.html
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import backend as backend_mod  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert HTML5 game engineer. Output a complete, working "
    "single-file HTML5 game when asked. The reply MUST be ONE complete "
    "HTML file: <!DOCTYPE html> through </html>. No prose before or "
    "after. No <plan>, <criteria>, <probes>, <patch>, or any other "
    "XML-style tags around the code. Just the HTML.\n\n"
    "HARD RULES:\n"
    "- Single file: HTML + CSS + JS inline. CDN <script src> is fine.\n"
    "- requestAnimationFrame for the loop, NOT setInterval.\n"
    "- e.code for keyboard ('KeyW', 'ArrowUp', 'Space'), not e.key.\n"
    "- Wrap the frame body in try/catch logging to console.error.\n"
    "- Visible score + clear lose condition + a way to restart.\n"
    "- DPR scaling so HiDPI displays aren't blurry.\n"
    "- Place entities at runtime by scanning empty cells, never \n"
    "  hand-verify coordinates in your reply.\n"
    "- Ship a COMPLETE, PLAYABLE game in one reply. Do not abbreviate\n"
    "  with '// rest of code' or any elision marker. Do not split the\n"
    "  output into multiple files."
)


def strip_wrappers(text: str) -> str:
    """Remove markdown fences, leading prose, and trailing prose.
    Return the bare HTML between the first '<!DOCTYPE' (or '<html') and
    the last '</html>'. Fallback to the raw text if neither found.
    """
    if not text:
        return text
    # Drop ```html ... ``` fences if present.
    fence = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Cut to the first DOCTYPE or <html>.
    m = re.search(r"<!doctype\s+html|<html\b", text, re.IGNORECASE)
    if m:
        text = text[m.start():]
    # Cut after the last </html>.
    m2 = re.search(r"</html\s*>", text, re.IGNORECASE)
    if m2:
        text = text[:m2.end()]
    return text.strip()


async def main_async(goal: str, out_path: Path, model_override: str | None,
                     stall_s: float, overall_s: float) -> int:
    import os
    # detect_backend reads MLX_MODEL from env. Honor --model by
    # setting it on the way in if the caller passed one.
    if model_override:
        os.environ["MLX_MODEL"] = model_override
    info = backend_mod.detect_backend("mlx")
    print(f"[oneshot] backend={info.name} model={info.model}", flush=True)
    back = backend_mod.MLXBackend(info)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    started = time.monotonic()
    chars_written = [0]
    last_chunk_at = [time.monotonic()]

    def on_token(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()
        chars_written[0] += len(piece)
        last_chunk_at[0] = time.monotonic()

    print("[oneshot] streaming...\n", flush=True)
    result = await back.stream_chat(
        messages,
        on_token=on_token,
        stall_seconds=stall_s,
        overall_seconds=overall_s,
        max_retries=0,
    )
    elapsed = time.monotonic() - started

    raw_text = result.text or ""
    html = strip_wrappers(raw_text)
    if not html or "<html" not in html.lower():
        print(f"\n[oneshot] FAILED — no HTML in response "
              f"(got {len(raw_text)} chars)",
              file=sys.stderr)
        # Still save the raw text for debug.
        debug_path = out_path.with_suffix(".raw.txt")
        debug_path.write_text(raw_text, encoding="utf-8")
        print(f"[oneshot] raw text saved to {debug_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n\n[oneshot] WROTE {out_path} ({len(html):,} chars, "
          f"{elapsed:.0f}s, {result.completion_tokens or 0} tokens, "
          f"stalled={result.stalled} looped={result.looped})",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", help="Game description, plain English.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Path to save the HTML file.")
    ap.add_argument("--model", default=None,
                    help="MLX model path/id override. Default: env or auto.")
    ap.add_argument("--stall-seconds", type=float, default=120.0,
                    help="Abort if the stream stalls for this many "
                         "seconds with no tokens (default 120).")
    ap.add_argument("--overall-seconds", type=float, default=900.0,
                    help="Total time budget for the stream (default 900s).")
    args = ap.parse_args()

    try:
        rc = asyncio.run(main_async(
            args.goal, args.out, args.model,
            args.stall_seconds, args.overall_seconds,
        ))
    except KeyboardInterrupt:
        print("\n[oneshot] interrupted by user", file=sys.stderr)
        rc = 130
    return rc


if __name__ == "__main__":
    sys.exit(main())
