"""Plan-stage VLM_CHECKLIST injection from visual_playtests.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from memory import render_opening_book_block, render_vlm_checklist_section  # noqa: E402

REPO_MEMORY = Path(__file__).parent.parent / "memory"


def test_render_vlm_checklist_section_caps_items() -> None:
    import memory as memory_mod

    mem = memory_mod.GameMemory(root=str(REPO_MEMORY))
    recipe, _ = mem.find_visual_playtest_for(
        goal="two fighters health bars special moves blocking",
    )
    assert recipe is not None
    assert recipe.id == "canvas-two-actors-facing"
    section = render_vlm_checklist_section(recipe, max_items=5)
    assert section.startswith("VLM_CHECKLIST [canvas-two-actors-facing]")
    assert "/vlm-critique is on" in section
    assert "Q1:" in section
    assert "Q4:" in section
    assert "Q5:" not in section


def test_fighter_goal_injects_vlm_checklist_at_plan_stage(tmp_path: Path) -> None:
    agent = GameAgent(
        model="stub",
        out_path=tmp_path / "game.html",
        browser=None,
        memory_root=str(REPO_MEMORY),
    )
    agent._goal = "mortal kombat two fighters facing each other with health bars"
    block, _hits = agent._retrieve_opening_book_block(
        agent._goal,
        stage="plan",
    )
    assert "VLM_CHECKLIST [canvas-two-actors-facing]" in block
    assert "two distinct characters" in block.lower()


def test_code_stage_omits_vlm_checklist(tmp_path: Path) -> None:
    agent = GameAgent(
        model="stub",
        out_path=tmp_path / "game.html",
        browser=None,
        memory_root=str(REPO_MEMORY),
    )
    block, _ = agent._retrieve_opening_book_block(
        "two fighters health bars special moves",
        stage="code",
    )
    assert "VLM_CHECKLIST" not in block


def test_playbook_bullets_use_vlm_critique_wording() -> None:
    ids = {
        "ladder-snap-to-platform-y",
        "vlm-critic-can-mislead-on-orientation",
        "animation-frames-consistent-character",
        "attack-sprite-wrong-direction-flip-in-code",
        "prefer-code-fix-over-asset-regen",
        "draw-fighters-large",
    }
    by_id: dict[str, dict] = {}
    for line in (REPO_MEMORY / "playbook.jsonl").open():
        o = json.loads(line)
        if o["id"] in ids:
            by_id[o["id"]] = o
    assert len(by_id) == len(ids)
    for bid, o in by_id.items():
        assert "/vlm-critique" in o["content"], bid
        assert "visual critic" not in o["content"].lower(), bid


def test_opening_book_block_accepts_vlm_checklist_section() -> None:
    section = (
        "VLM_CHECKLIST [canvas-two-actors-facing] (checked when /vlm-critique is on):\n"
        "- Q1: Are TWO distinct character figures visible on the canvas?"
    )
    block = render_opening_book_block(
        None, [], [], [],
        vlm_checklist=section,
    )
    assert "VLM_CHECKLIST" in block
    assert "<opening_book>" in block
