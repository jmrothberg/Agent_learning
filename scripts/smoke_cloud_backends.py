"""Smoke-test the cloud backends end-to-end.

Reads OPENAI_API_KEY and ANTHROPIC_API_KEY from the shell env (never
from disk), sends a 4-word prompt to each newest model, and prints the
response + token counts. Use this after first export to confirm the
key is good and the model id is current.

Run:
    .venv/bin/python scripts/smoke_cloud_backends.py
    .venv/bin/python scripts/smoke_cloud_backends.py openai
    .venv/bin/python scripts/smoke_cloud_backends.py anthropic
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load project .env (gitignored, chmod 600) so this entry point sees
# the same OPENAI_API_KEY / ANTHROPIC_API_KEY that chat.py + coder.py do.
try:
    from dotenv import load_dotenv
    # override=True so .env wins over empty/stale shell vars; matches chat.py.
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    if (ROOT / ".env").exists():
        print(
            "note: .env present but python-dotenv not installed; "
            "run `.venv/bin/pip install python-dotenv` or export keys "
            "manually in your shell.",
            file=sys.stderr,
        )

import backend as backend_mod  # noqa: E402


SYS = "You are a code assistant. Respond in at most 4 words."
USER = "Say hello in Spanish."


async def _smoke(name: str) -> int:
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("  SKIP openai — OPENAI_API_KEY not set")
            return 0
        info = backend_mod.detect_backend("openai")
    elif name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  SKIP anthropic — ANTHROPIC_API_KEY not set")
            return 0
        info = backend_mod.detect_backend("anthropic")
    else:
        print(f"  unknown backend: {name}")
        return 2

    print(f"\n[{name}] model={info.model}")
    print(f"  source: {info.source}")
    bk = backend_mod.make_backend(info)

    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
    ]
    print(f"  > {USER}")
    sys.stdout.write("  < ")
    sys.stdout.flush()

    def on_tok(s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    res = await bk.stream_chat(
        messages,
        on_token=on_tok,
        options={"max_tokens": 50, "temperature": 0.0},
        stall_seconds=30.0,
        overall_seconds=60.0,
    )
    print()
    print(
        f"  tokens: {res.tokens}  duration: {res.duration_s:.1f}s  "
        f"prompt={res.prompt_tokens}  completion={res.completion_tokens}  "
        f"stalled={res.stalled}"
    )
    await bk.close()
    if res.stalled or not res.text.strip():
        print("  FAIL — stalled or empty reply")
        return 1
    print("  PASS")
    return 0


async def main() -> int:
    targets = sys.argv[1:] or ["openai", "anthropic"]
    print("=" * 60)
    print(" CLOUD BACKEND SMOKE")
    print("=" * 60)
    print(f"  OPENAI_API_KEY    set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"  ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")

    rc = 0
    for t in targets:
        rc |= await _smoke(t)
    print()
    print("=" * 60)
    print(" DONE" if rc == 0 else " FAILED")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
