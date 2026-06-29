from __future__ import annotations

import asyncio
import inspect
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
from tools import LiveBrowser, pointclick_opening_book_applicable  # noqa: E402


def test_visual_playtest_rows_carry_outline_id() -> None:
    """Phase 4 (4A): every recipe→outline link is now on the data ROW
    (recipe.outline_id), consistent with the _RECIPE_TO_OUTLINE fallback."""
    import memory as memory_module
    mem = GameMemory(root="memory")
    rows = mem.load_visual_playtests()
    for item in rows:
        rec = item.recipe if isinstance(item.recipe, dict) else {}
        expected = memory_module._RECIPE_TO_OUTLINE.get(item.id)
        if expected is None:
            continue
        assert rec.get("outline_id") == expected, item.id


def test_outline_routing_uses_row_outline_id_not_just_dict(monkeypatch) -> None:
    """Proves the row's outline_id is the runtime source: with the
    _RECIPE_TO_OUTLINE fallback emptied, a tower-defense goal STILL routes to
    outline-tower-defense via the data row."""
    import memory as memory_module
    monkeypatch.setattr(memory_module, "_RECIPE_TO_OUTLINE", {})
    mem = GameMemory(root="memory")
    hit = mem.retrieve_implementation_outline(
        "open-field fieldrunners tower defense: place turrets to stop waves"
    )
    assert hit is not None
    assert hit.item.id == "outline-tower-defense"


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


def test_plan_turn_injects_opening_book_before_plan_contract() -> None:
    src = GameAgent.run_loop_inspect_source()
    assert "plan_opening_book_injected" in src
    assert "Use the opening-book recipes above when choosing your" in src
    assert "adapt them to the user's goal" in src or "them to the user's goal when it specifies" in src
    # B2: plan-turn opening-book budget was raised 1200 -> 3000 so the model
    # sees the full state contract at the moment it writes its <probes>.
    assert "char_budget=3000" in src


def test_pointclick_puzzle_chain_recipe_is_executable() -> None:
    src = inspect.getsource(LiveBrowser._run_opening_book_recipes)
    assert "pointclick_puzzle_chain" in src
    assert "selectedItem" in src
    assert "state.inventory[]" in src
    assert "pointclick_opening_book_applicable" in src


def test_pointclick_opening_book_not_applicable_for_voxel_goal() -> None:
    assert pointclick_opening_book_applicable(
        "first-person voxel sandbox minecraft blocks break place",
        visual_recipe_id="canvas-3d-first-person",
    ) is False


def test_pointclick_opening_book_applicable_for_p_and_c_goal() -> None:
    assert pointclick_opening_book_applicable(
        "monkey island point and click adventure inventory hotspots",
        visual_recipe_id="canvas-point-and-click",
    ) is True


def test_pointclick_skipped_on_non_p_and_c_goal(tmp_path: Path) -> None:
    html = tmp_path / "game.html"
    html.write_text("<script>window.state={player:{x:0}};</script>", encoding="utf-8")
    browser = LiveBrowser(headless=True)

    checks = asyncio.run(browser._run_opening_book_recipes(
        html,
        [{
            "id": "pointclick-puzzle-chain",
            "kind": "playtest",
            "recipe": {"type": "pointclick_puzzle_chain"},
        }],
        input_test={"ran": False},
        canvas_info=None,
        frozen=None,
        goal="first-person voxel sandbox with three.js blocks",
        visual_recipe_id="canvas-3d-first-person",
    ))

    assert len(checks) == 1
    assert checks[0]["ok"] is True
    assert checks[0]["hard"] is False
    assert checks[0].get("skipped") is True


async def _pointclick_hard_fail_when_applicable(tmp_path: Path) -> None:
    html = tmp_path / "game.html"
    html.write_text("<script>window.state={};</script>", encoding="utf-8")
    browser = LiveBrowser(headless=True)

    async def _fake_eval(_js: str):
        return {"ok": False, "missing": ["state.inventory[]", "scene hotspots[]"]}

    browser._safe_eval = _fake_eval  # type: ignore[method-assign]
    checks = await browser._run_opening_book_recipes(
        html,
        [{
            "id": "pointclick-puzzle-chain",
            "kind": "playtest",
            "recipe": {"type": "pointclick_puzzle_chain"},
        }],
        input_test={"ran": False},
        canvas_info=None,
        frozen=None,
        goal="point and click adventure with inventory and hotspots",
        visual_recipe_id="canvas-point-and-click",
    )

    assert len(checks) == 1
    assert checks[0]["ok"] is False
    assert checks[0]["hard"] is True
    assert "inventory" in checks[0]["err"]


def test_pointclick_hard_fails_when_p_and_c_state_missing(tmp_path: Path) -> None:
    asyncio.run(_pointclick_hard_fail_when_applicable(tmp_path))


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
