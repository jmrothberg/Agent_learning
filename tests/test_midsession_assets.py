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
        "Mid-session asset/sound/video additions" in fb
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


def test_midsession_same_name_skips_loader_block(tmp_path: Path) -> None:
    """MK trace 20260517_220025 fix: when an asset is regenerated with
    a name already referenced by the on-disk HTML, the heavy
    `ASSETS_LOADER` block must NOT be injected — the regenerated PNG
    has already replaced the file, and the existing drawImage() call
    picks it up. Emit a compact MEDIA REGEN COMPLETE confirmation
    instead so the model doesn't think it needs to add a loader."""
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    # Simulate the current on-disk HTML referencing `player_kick` via
    # the canonical ASSETS map pattern.
    a._current_file = (
        "<html><body><canvas></canvas><script>"
        "const ASSETS = {};"
        "ASSETS.player_kick = new Image();"
        "ASSETS['player_kick'].src = './g_assets/player_kick.png';"
        "</script></body></html>"
    )
    old = tmp_path / "old_player_kick.png"
    old.write_bytes(b"old bytes")
    a._session_assets = {"player_kick": old}

    reply = (
        "<assets>[{\"name\": \"player_kick\", \"prompt\": \"mirrored kick\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    assert a._pending_feedback, "expected a queued feedback line"
    fb = a._pending_feedback[0]
    # Compact confirmation — name appears, but NOT the heavy loader.
    assert "MEDIA REGEN COMPLETE" in fb
    assert "player_kick" in fb
    assert "changed" in fb
    assert "ASSETS_LOADER" not in fb
    assert "ULTRA IMPORTANT" not in fb
    assert "async function loadAssets" not in fb


def test_midsession_same_name_trace_records_old_new_hash(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    a._current_file = (
        "<html><body><script>"
        "const ASSETS = {}; ASSETS.player_kick = new Image();"
        "</script></body></html>"
    )
    old = tmp_path / "old_player_kick.png"
    old.write_bytes(b"old bytes")
    a._session_assets = {"player_kick": old}
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = (
        "<assets>[{\"name\": \"player_kick\", \"prompt\": \"mirrored kick\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    event = next(
        t for t in traces
        if t.get("kind") == "midsession_asset_injection_queued"
    )
    rep = event["asset_replacements"]["player_kick"]
    assert rep["old_hash"] is not None
    assert rep["new_hash"] is not None
    assert rep["old_hash"] != rep["new_hash"]
    assert rep["changed"] is True


def test_midsession_new_name_still_emits_loader_block(tmp_path: Path) -> None:
    """Negative control: a brand-new asset name (not in HTML) must
    still get the full loader block so the model knows to wire it in."""
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    # Current HTML has no asset references at all.
    a._current_file = "<html><body><canvas></canvas></body></html>"

    reply = (
        "<assets>[{\"name\": \"explosion\", \"prompt\": \"orange burst\"}]"
        "</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    assert a._pending_feedback, "expected a queued feedback line"
    fb = a._pending_feedback[0]
    # New name → full loader block injected.
    assert "Mid-session asset/sound/video additions" in fb
    assert "explosion" in fb


def test_scoped_media_lock_drops_new_asset_names(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._asset_generator = _StubImageGenerator()
    existing = tmp_path / "player.png"
    existing.write_bytes(b"")
    a._session_assets = {"player": existing}
    a._scoped_constraints = {
        "mode": "media_only",
        "media_name_lock": True,
        "allowed_asset_names": ["player"],
        "allowed_sound_names": [],
    }
    reply = (
        "<assets>["
        "{\"name\": \"player\", \"prompt\": \"updated player\"},"
        "{\"name\": \"enemy_new\", \"prompt\": \"new enemy\"}"
        "]</assets>"
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    assert "enemy_new" not in a._session_assets
    assert any(
        "SCOPED MEDIA NAME LOCK" in fb and "player" in fb
        for fb in a._pending_feedback
    )
