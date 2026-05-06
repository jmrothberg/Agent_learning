"""Integration test: confirm the research-grounded planning prompt
produces a Missile Command plan that matches the actual game mechanics
(cities, crosshair, counter-missiles), not Space Invaders.

Compares against the broken run captured at:
  games/traces/game-of-misile-command-good-gr_20260505_133453.conversation.md

That session's plan said: "missile launcher that fires projectiles to
destroy incoming enemies; ArrowUp/ArrowDown to aim, Space to fire" —
Space Invaders. This test verifies the new prompt grounds the model
in the actual game.

Runs ONE Ollama planning call (no iteration, no browser). Wall-clock
~30-60s on a loaded gpt-oss model.

Usage:
    .venv/bin/python tests/test_research_planning.py
        # uses gpt-oss:latest (override with OLLAMA_MODEL env)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402

import prompts_v1  # noqa: E402
import research  # noqa: E402


GOAL = (
    "game of misile command, good graphics good sound. like the original "
    "arcade game for misile command."
)

MISSILE_KEYWORDS = (
    # Real Missile Command mechanics — at least 3 of these should appear
    # in a grounded plan. The previous (broken) plan hit ZERO.
    "city", "cities",
    "crosshair", "cursor", "mouse", "trackball", "click",
    "counter-missile", "counter missile", "anti-ballistic", "intercept", "explosion",
    "battery", "batteries", "silo",
    "icbm", "ballistic", "incoming", "rain",
    "defend", "protect",
    "fireball",
)

SPACE_INVADERS_KEYWORDS = (
    # Failure mode — these were in the broken plan and shouldn't appear.
    "arrowleft", "arrowright", "left/right",
    "fires upward", "fires up", "shoot upward", "ship moves",
)


async def main() -> int:
    model = os.environ.get("OLLAMA_MODEL", "gpt-oss:latest")
    print(f"[1/3] Fetching Wikipedia reference for: {GOAL[:80]!r}")
    ref = research.fetch(GOAL)
    if not ref:
        print("FAIL: no reference fetched.")
        return 1
    title_line = ref.splitlines()[1] if "\n" in ref else "?"
    print(f"      ✓ {title_line}  ({len(ref)} chars)")
    if "Missile Command" not in ref:
        print("FAIL: reference doesn't mention 'Missile Command'.")
        return 1

    print(f"[2/3] Building grounded planning prompt")
    plan_msg = prompts_v1.plan_instruction(reference_block=ref)
    system_msg = prompts_v1.SYSTEM_PROMPT.replace("{goal}", GOAL)

    print(f"[3/3] Streaming plan from {model} (this loads the model first run)…")
    client = ollama.AsyncClient()
    pieces: list[str] = []
    try:
        stream = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": plan_msg},
            ],
            stream=True,
            options={"num_ctx": 8192},
        )
        async for chunk in stream:
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                pieces.append(piece)
                # Print streaming so the user sees progress.
                sys.stdout.write(piece)
                sys.stdout.flush()
    except Exception as e:
        print(f"\nFAIL: Ollama stream failed: {e!r}")
        return 1
    print()

    plan = "".join(pieces).lower()

    print()
    print("=" * 60)
    hits = [k for k in MISSILE_KEYWORDS if k in plan]
    misses = [k for k in SPACE_INVADERS_KEYWORDS if k in plan]
    print(f"Missile-Command keywords matched ({len(hits)}/{len(MISSILE_KEYWORDS)}): {hits}")
    print(f"Space-Invaders keywords (should be 0): {misses}")

    if len(hits) >= 3 and not misses:
        print("✓ PASS — plan describes Missile Command, not Space Invaders.")
        return 0
    if len(hits) >= 3 and misses:
        print(f"~ MIXED — plan mentions correct mechanics but also {misses}.")
        return 2
    print("✗ FAIL — plan does not look like Missile Command.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
