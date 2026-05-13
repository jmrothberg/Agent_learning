"""Demo: art-change feedback now routes to <assets> re-render instead
of procedural-drawing rewrites.

Motivating trace:
    games/traces/centipede-game-with-super-nice_20260512_180020.*

The user typed "only change the centipede_tail no other asset or
code, just that one asset no changes to the code" and the model
replied "I can't generate new image assets in this environment - I
can only modify the HTML file" — then rewrote a drawSprite() call
into procedural ctx.* code (regression, auto-reverted). The harness
*can* re-render assets mid-session; the model just didn't know.

This script runs the relevant agent code paths end-to-end with stubs
in place of Z-Image-Turbo and Chromium so it's fast (≈1 s) and has
no GPU/network dependency.

Run:
    .venv/bin/python scripts/demo_asset_change_feedback.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import GameAgent  # noqa: E402


class _StubImageGenerator:
    """Drop-in for ZImageTurboGenerator. Each call writes a unique
    grey PNG so we can SEE the file change between calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_stats: list[dict] = []

    def generate(self, prompt: str) -> str:
        from PIL import Image
        self.calls.append(prompt)
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        shade = 60 + 50 * len(self.calls)
        Image.new("RGB", (768, 768), (shade, shade, shade)).save(f.name)
        return f.name


def _hr(title: str = "") -> None:
    bar = "─" * 72
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def _build_agent(tmp: Path) -> GameAgent:
    out = tmp / "game.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp / "memory"),
    )
    a._asset_generator = _StubImageGenerator()
    seed_dir = tmp / "game_assets"
    seed_dir.mkdir(exist_ok=True)
    for name in ("player_ship", "centipede_head", "centipede_tail", "mushroom"):
        p = seed_dir / f"{name}.png"
        p.write_bytes(b"\x89PNG seed")
        a._session_assets[name] = p
    return a


async def _drain(agen) -> list:
    return [ev async for ev in agen]


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  PASS  {msg}")
        else:
            failures.append(msg)
            print(f"  FAIL  {msg}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        agent = _build_agent(tmp)

        _hr("STEP 1 — simulate user typing feedback while watching the game")
        feedback = (
            "make the additional segments more round with moving legs, so "
            "it looks more connected, only change the centipede_tail no "
            "other asset or code, just that one asset no changes to the "
            "code."
        )
        print(f"  USER FEEDBACK > {feedback}")
        agent._pending_feedback.append(feedback)

        _hr("STEP 2 — _flush_user_injections renders the next user turn")
        before_flag = agent._allow_one_rewrite
        rendered = agent._flush_user_injections(
            base_message="<base iteration prompt>"
        )
        print(rendered)

        _hr("STEP 3 — assertions on the rendered prompt + agent state")
        check(
            "USER FEEDBACK (HIGHEST PRIORITY)" in rendered,
            "USER FEEDBACK block is present (existing behavior preserved)",
        )
        check(
            "MEDIA-CHANGE DIRECTIVE" in rendered,
            "new MEDIA-CHANGE DIRECTIVE was injected",
        )
        check(
            "centipede_tail" in rendered and "player_ship" in rendered,
            "existing asset names are listed for the model to target",
        )
        check(
            "<assets>" in rendered,
            "directive names the <assets> tag explicitly",
        )
        check(
            before_flag is False and agent._allow_one_rewrite is False,
            "_allow_one_rewrite stayed CLOSED (user locked the code)",
        )

        _hr("STEP 4 — model replies with <assets> targeting centipede_tail")
        reply = (
            "<assets>[{\"name\": \"centipede_tail\", \"prompt\": \"round "
            "green pixel-art centipede tail with two animated legs, "
            "transparent background\"}]</assets>"
        )
        print(reply)
        before = agent._session_assets["centipede_tail"]
        before_size = Path(before).stat().st_size
        print(f"  BEFORE  centipede_tail -> {before}")
        print(f"          file size {before_size} bytes")

        asyncio.run(_drain(agent._maybe_generate_assets_and_sounds(
            reply, trigger="mid_session",
        )))

        after = agent._session_assets["centipede_tail"]
        after_size = Path(after).stat().st_size
        print(f"  AFTER   centipede_tail -> {after}")
        print(f"          file size {after_size} bytes")

        _hr("STEP 5 — assertions on the regenerated asset")
        check(
            Path(after).exists() and after_size > 0,
            "regenerated centipede_tail PNG exists on disk and is non-empty",
        )
        check(
            str(after) != str(before),
            "session pointer moved (asset was actually re-rendered)",
        )
        for name in ("player_ship", "centipede_head", "mushroom"):
            check(
                agent._session_assets[name].name == f"{name}.png",
                f"{name} untouched (mid-session merge did not clobber it)",
            )
        check(
            any(
                "Mid-session asset/sound additions" in fb
                and "centipede_tail" in fb
                for fb in agent._pending_feedback
            ),
            "next user turn will surface the new PNG path to the model",
        )

        _hr("STEP 6 — control: normal feedback should still arm the rewrite gate")
        agent2 = _build_agent(tmp)
        check(
            agent2._allow_one_rewrite is False,
            "(control) starts with _allow_one_rewrite = False",
        )
        agent2._pending_feedback.append("fix the mouse-look and add powerups")
        agent2._flush_user_injections("next-turn message")
        check(
            agent2._allow_one_rewrite is True,
            "(control) plain multi-issue feedback still arms the rewrite "
            "exemption — gate only closes when user locks code",
        )

    _hr()
    if failures:
        print(f"DEMO FAILED — {len(failures)} assertion(s):")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("DEMO PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
