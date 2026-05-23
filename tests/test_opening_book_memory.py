from __future__ import annotations

import asyncio
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import GameAgent  # noqa: E402
from memory import (  # noqa: E402
    ASSET_AUDITS_FILENAME,
    PLAYTESTS_FILENAME,
    GameMemory,
    OpeningBookItem,
    render_opening_book_block,
)
from tools import LiveBrowser  # noqa: E402


def test_opening_book_retrieval_caps_and_prefers_root(tmp_path: Path) -> None:
    mem = GameMemory(root=tmp_path / "memory")
    mem.ensure()

    live = OpeningBookItem(
        id="controllable-movement-delta",
        kind="playtest",
        content="noisy duplicate should not shadow root",
        tags=["input", "movement", "controls"],
        source_tier="live",
        verified=True,
        pass_count=1,
        recipe={"type": "input_delta"},
    )
    assert mem.append_live_opening_book_item(PLAYTESTS_FILENAME, live)

    hits = mem.retrieve_playtests("player moves with arrow keys", k=3)

    assert 0 < len(hits) <= 3
    assert hits[0].item.source_tier == "root"
    assert hits[0].item.id == "controllable-movement-delta"


def test_live_memory_requires_verified_positive_score(tmp_path: Path) -> None:
    mem = GameMemory(root=tmp_path / "memory")
    mem.ensure()
    bad = OpeningBookItem(
        id="live-unverified-asset-check",
        kind="asset_audit",
        content="asset sprite drawImage loader check",
        tags=["asset", "sprite", "drawImage"],
        source_tier="live",
        verified=False,
        pass_count=0,
        recipe={"type": "asset_usage"},
    )
    assert mem.append_live_opening_book_item(ASSET_AUDITS_FILENAME, bad)

    hits = mem.retrieve_asset_audits("use generated sprite assets", k=5)

    assert all(h.item.id != "live-unverified-asset-check" for h in hits)


def test_opening_book_block_is_compact_no_trace_dump(tmp_path: Path) -> None:
    mem = GameMemory(root=tmp_path / "memory")
    mem.ensure()
    block = render_opening_book_block(
        mem.retrieve_implementation_outline("top down player movement", "canvas"),
        mem.retrieve_playtests("top down player movement", "canvas"),
        mem.retrieve_asset_audits("top down player movement", "canvas"),
        mem.retrieve_animation_audits("top down player movement", "canvas"),
    )

    assert "<opening_book>" in block
    assert "Traceback" not in block
    assert "Conversation dump" not in block
    assert len(block) < 3200


def test_agent_retrieves_opening_book_block(tmp_path: Path) -> None:
    agent = GameAgent(
        model="stub",
        out_path=tmp_path / "game.html",
        browser=None,
        memory_root=tmp_path / "memory",
    )

    block, hits = agent._retrieve_opening_book_block(
        "build a player movement game with sprite assets",
        stage="plan",
    )

    assert "<opening_book>" in block
    assert hits
    assert len([h for h in hits if h["kind"] == "playtest"]) <= 3


def test_opening_book_asset_recipe_execution(tmp_path: Path) -> None:
    html = tmp_path / "game.html"
    html.write_text(
        "<script>const ASSETS={}; const ASSET_LIST=[['hero','./x_assets/hero.png']];</script>",
        encoding="utf-8",
    )
    browser = LiveBrowser(headless=True)

    checks = asyncio.run(browser._run_opening_book_recipes(
        html,
        [{"id": "asset-usage", "kind": "asset_audit", "recipe": {"type": "asset_usage"}}],
        input_test={"ran": False},
        canvas_info=None,
        frozen=None,
    ))

    assert checks[0]["ok"] is False
    assert checks[0]["hard"] is True
