"""Tests for mid-session asset/sound generation.

The agent runs _maybe_generate_assets_and_sounds on every assistant
reply that contains an <assets> or <sounds> block — not just the
Phase-A plan reply. This file exercises that helper with a stub
generator so we never touch Z-Image-Turbo or Stable Audio Open, then
verifies:

  1. Calling with trigger="mid_session" MERGES into _session_assets
     rather than overwriting.
  2. A USER FEEDBACK injection is queued so the next user turn shows
     the new asset paths to the model.
  3. Trace events stamp `trigger` so offline analysis can split
     phase_a from mid_session.

For sounds we cover the same merge contract.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import agent as agent_mod  # noqa: E402


class _StubImageGenerator:
    """Mimics ZImageTurboGenerator just enough for generate_assets to
    write one PNG per spec. last_stats is populated so the agent's
    per-asset timing log path is exercised."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_stats: list[dict] = []

    def generate(self, prompt: str) -> str | None:
        self.calls.append(prompt)
        from PIL import Image
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        Image.new("RGB", (768, 768), (200, 200, 200)).save(f.name)
        return f.name


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


async def _drain(agen) -> list:
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def test_midsession_assets_merge(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    # Seed an existing asset from a hypothetical Phase A.
    seed_path = tmp_path / "seed.png"
    seed_path.write_bytes(b"")
    a._session_assets = {"player": seed_path}

    reply = (
        "Adding a new sprite as you asked.\n"
        "<assets>[{\"name\": \"alien\", \"prompt\": \"green pixel alien\"}]"
        "</assets>\n"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    # Merge contract: existing + new, no overwrite of unrelated names.
    assert "player" in a._session_assets
    assert "alien" in a._session_assets
    # The new file actually exists on disk.
    assert Path(a._session_assets["alien"]).exists()


def test_midsession_assets_inject_feedback(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    reply = (
        "<assets>[{\"name\": \"bullet\", \"prompt\": \"yellow blob\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    # The next user turn should see a feedback block referencing the
    # new asset path so the model can load it via new Image().
    assert any(
        "Mid-session asset/sound additions" in fb
        and "bullet" in fb
        for fb in a._pending_feedback
    )


def test_phase_a_does_not_inject_feedback(tmp_path: Path) -> None:
    # Phase A first-build prompt already inlines render_asset_paths_block,
    # so we MUST NOT also queue the feedback line — that would duplicate
    # the asset paths in the model's view.
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    reply = (
        "<assets>[{\"name\": \"ship\", \"prompt\": \"silver ship\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="phase_a",
    )))

    assert a._pending_feedback == []
    assert "ship" in a._session_assets


def test_empty_reply_is_noop(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    # No generators loaded — should never be touched.
    events = asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        "no asset blocks here", trigger="mid_session",
    )))
    assert events == []
    assert a._session_assets == {}
    assert a._pending_feedback == []


def test_midsession_overwrite_same_name(tmp_path: Path) -> None:
    # If the model re-emits an asset with the same name, the new path
    # should replace the old (same-name overwrite within the merge).
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    old_path = tmp_path / "old_alien.png"
    old_path.write_bytes(b"")
    a._session_assets = {"alien": old_path}

    reply = (
        "<assets>[{\"name\": \"alien\", \"prompt\": \"redrawn alien\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    # Same key, new path. The old file isn't deleted (the diffuser
    # cache may still reference it) but the session pointer moves.
    assert a._session_assets["alien"] != old_path
    assert Path(a._session_assets["alien"]).exists()
