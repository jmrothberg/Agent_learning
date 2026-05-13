"""Live test — narrow, surgical: given the new MEDIA-CHANGE DIRECTIVE
in the user-turn, does the real model emit <assets> for the named
asset (mid-session re-render), and does the harness then actually
replace the PNG?

This bypasses Phase A so it doesn't matter whether the small model
emits <assets> spontaneously for the planning turn — that's a
separate concern. What we want to validate here is:

  given a session that already has assets and the user types
  "redraw the X asset, no code changes", the model picks the
  <assets> path.

We do this by:
  1. Generating ONE real PNG via Z-Image-Turbo (cold-load if first run).
  2. Building the same system prompt the agent uses.
  3. Building a user-turn message that exactly mirrors what
     `_flush_user_injections` produces when a session has one asset and
     the user typed art-change feedback.
  4. Calling the REAL backend (Ollama / MLX) once.
  5. Parsing the reply for an <assets> block targeting the asset.
  6. Running the real diffuser pipeline on the parsed spec.
  7. Confirming the PNG bytes actually changed.

Run:
    .venv/bin/python scripts/live_test_asset_change.py
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backend as backend_mod  # noqa: E402
import assets as assets_mod  # noqa: E402
import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402
from tools import LiveBrowser  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


GOAL = "tiny bouncing player_ship demo"
ASSET_NAME = "player_ship"
ORIGINAL_PROMPT = (
    "pixel-art retro spaceship facing right, silver hull, "
    "white outline, transparent background"
)
FEEDBACK = (
    "redraw the player_ship asset as a small pink heart with white "
    "sparkles on a transparent background. Only the asset, no code "
    "changes."
)


def _sha8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


async def _stream(backend, messages, label: str) -> str:
    print(f"\n  [model] streaming {label} …")
    buf: list[str] = []
    last_kind = ""

    def on_tok(s: str) -> None:
        nonlocal last_kind
        if last_kind != "tok":
            sys.stdout.write("\n    ")
            last_kind = "tok"
        buf.append(s)
        sys.stdout.write(s)
        sys.stdout.flush()

    res = await backend.stream_chat(
        messages,
        on_token=on_tok,
        stall_seconds=120.0,
        overall_seconds=300.0,
    )
    print()
    return res.text


def _extract_assets_block(reply: str) -> str | None:
    m = re.search(r"<assets>(.*?)</assets>", reply, flags=re.S | re.I)
    return m.group(0) if m else None


async def main() -> int:
    print("=" * 72)
    print(" LIVE TEST — change one asset via the new MEDIA-CHANGE DIRECTIVE")
    print("=" * 72)

    info = backend_mod.detect_backend("auto")
    print(f"  backend:  {info.name} model={info.model}  ({info.source})")
    bk = backend_mod.make_backend(info)

    out = ROOT / "games" / "live_test_asset_change.html"
    out.parent.mkdir(exist_ok=True)
    assets_dir = out.parent / "live_test_asset_change_assets"
    assets_dir.mkdir(exist_ok=True)

    # 1. Generate the original PNG via the real diffuser.
    print("\n[1] Loading Z-Image-Turbo and generating original "
          f"{ASSET_NAME}.png …")
    gen = assets_mod.try_load_image_generator()
    if gen is None:
        print("    FAIL — diffuser not reachable. See ./scripts/install_diffuser.sh")
        return 2
    t0 = time.time()
    produced = await asyncio.to_thread(
        assets_mod.generate_assets,
        [{"name": ASSET_NAME, "prompt": ORIGINAL_PROMPT, "size": (128, 128)}],
        assets_dir,
        image_generator=gen,
    )
    if ASSET_NAME not in produced:
        print("    FAIL — generate_assets did not produce the asset.")
        return 2
    png_path = Path(produced[ASSET_NAME])
    pre_bytes = png_path.read_bytes()
    print(f"    OK  -> {png_path}  ({len(pre_bytes)}B  sha8={_sha8(pre_bytes)}  "
          f"in {time.time()-t0:.1f}s)")

    # 2. Construct a GameAgent so we can use its prompt assembly +
    #    feedback wrapper + asset pipeline. No need to call run().
    out.write_text(
        "<!DOCTYPE html><html><body><canvas id='c'></canvas>"
        f"<script>const img=new Image();img.src='./live_test_asset_change_assets/{ASSET_NAME}.png';"
        "</script></body></html>"
    )
    agent = GameAgent(
        backend=bk,
        out_path=out,
        browser=LiveBrowser(viewport=(800, 600), headless=True),  # never started
        max_iters=2,
        prompt_version="v1",
    )
    # Seed session state to simulate "user is mid-session with one asset".
    agent._session_assets[ASSET_NAME] = png_path
    agent._messages = []  # be explicit

    # 3. Build the system prompt the way agent.run() would.
    sys_prompt = prompts_v1.build_system_prompt(GOAL)
    sys_msg = {"role": "system", "content": sys_prompt}

    # 4. Queue feedback and flush it into a user-turn message — this is
    #    the EXACT prompt shape the model would see in a real session
    #    right after the user typed the feedback.
    agent.add_user_feedback(FEEDBACK)
    user_turn = agent._flush_user_injections(base_message=(
        "Continue the session. The current file on disk is small and "
        "working; address the user feedback above with minimal change."
    ))

    # Sanity-check the rendered prompt.
    assert "MEDIA-CHANGE DIRECTIVE" in user_turn, (
        "directive missing from user turn — code change did not land?"
    )
    assert "<assets>" in user_turn and ASSET_NAME in user_turn, (
        "directive missing asset name or <assets> tag"
    )
    print("\n[2] User-turn prompt assembled with MEDIA-CHANGE DIRECTIVE.")
    print(f"    sys prompt:  {len(sys_prompt)} chars")
    print(f"    user turn:   {len(user_turn)} chars")

    # 5. Call the real model.
    print("\n[3] Calling real backend …")
    reply = await _stream(
        bk,
        [sys_msg, {"role": "user", "content": user_turn}],
        label="user-turn",
    )

    # 6. Parse for <assets>.
    block = _extract_assets_block(reply)
    print()
    print("[4] Parsing model reply …")
    if not block:
        print("    FAIL — model did NOT emit <assets>. The directive was")
        print("           in the prompt but the model chose another path.")
        return 1
    parsed = assets_mod.parse_assets_block(reply)
    print(f"    OK  — found <assets> block, {len(parsed)} spec(s).")
    for s in parsed:
        print(f"      - name={s.get('name')!r}  prompt={s.get('prompt','')[:80]!r}…")

    targets = [s for s in parsed if s.get("name") == ASSET_NAME]
    if not targets:
        print(f"    FAIL — <assets> block does not target {ASSET_NAME!r}.")
        return 1

    # 7. Run the real asset pipeline against the parsed spec.
    print(f"\n[5] Re-rendering {ASSET_NAME} from the model's prompt …")
    t1 = time.time()
    produced2 = await asyncio.to_thread(
        assets_mod.generate_assets,
        targets,
        assets_dir,
        image_generator=gen,
    )
    if ASSET_NAME not in produced2:
        print("    FAIL — generate_assets did not produce the re-rendered asset.")
        return 2
    post_path = Path(produced2[ASSET_NAME])
    post_bytes = post_path.read_bytes()
    print(f"    OK  -> {post_path}  ({len(post_bytes)}B  sha8={_sha8(post_bytes)}  "
          f"in {time.time()-t1:.1f}s)")

    # 8. Verify.
    print()
    print("=" * 72)
    print(" RESULTS")
    print("=" * 72)
    print(f"  asset name:        {ASSET_NAME}")
    print(f"  original prompt:   {ORIGINAL_PROMPT[:70]}")
    print(f"  model's new prompt:{(targets[0].get('prompt','') or '')[:70]}")
    print(f"  PNG pre:           sha8={_sha8(pre_bytes)}  ({len(pre_bytes)} B)")
    print(f"  PNG post:          sha8={_sha8(post_bytes)}  ({len(post_bytes)} B)")
    png_changed = pre_bytes != post_bytes
    print()
    print(f"  PNG re-rendered?   {png_changed}")
    print(f"  <patch> in reply?  {'<patch>' in reply.lower()}")
    print(f"  <html_file>?       {'<html_file>' in reply.lower()}")
    print()
    if png_changed:
        print("  PASS — model picked the <assets> path; asset was re-rendered.")
        return 0
    print("  FAIL — model emitted <assets> but the new PNG matches the old.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
