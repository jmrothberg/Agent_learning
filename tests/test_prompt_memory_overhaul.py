"""Prompt + memory quality overhaul regression tests (2026-07-01)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import memory as memory_module  # noqa: E402
from agent import GameAgent  # noqa: E402
from memory import render_outline_traps_only  # noqa: E402
from prompts_v1 import (  # noqa: E402
    _detect_canvas_entity_intent,
    _detect_open_field_td_intent,
    _detect_pinball_intent,
    plan_instruction,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUTLINES_PATH = PROJECT_ROOT / "memory" / "implementation_outlines.jsonl"
PLAYBOOK_PATH = PROJECT_ROOT / "memory" / "playbook.jsonl"


def _load_outlines() -> list[dict]:
    out = []
    for line in OUTLINES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _playbook_ids() -> set[str]:
    ids: set[str] = set()
    for line in PLAYBOOK_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def test_pinball_outline_has_plunger_and_bumper_traps() -> None:
    outlines = {o["id"]: o for o in _load_outlines()}
    traps = " ".join(outlines["outline-pinball"]["recipe"]["traps"]).lower()
    assert "plunger" in traps
    assert "cooldown" in traps or "ping-pong" in traps


def test_new_playbook_bullets_exist() -> None:
    ids = _playbook_ids()
    for pid in (
        "circle-collision-pushout-cooldown",
        "launch-into-playfield",
        "flipper-segment-boost",
    ):
        assert pid in ids


def test_outline_trap_promotion_playbook_refs_valid() -> None:
    ids = _playbook_ids()
    for line in (PROJECT_ROOT / "memory" / "visual_playtests.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        refs = (o.get("recipe") or {}).get("playbook_refs") or {}
        for lst in refs.values():
            for rid in lst:
                assert rid in ids, f"{o['id']} refs missing playbook {rid}"


def test_render_outline_traps_only_respects_budget() -> None:
    recipe = _load_outlines()[0]["recipe"]
    block = render_outline_traps_only(recipe, char_budget=400)
    assert "OUTLINE TRAPS" in block
    assert len(block) <= 420


def test_canvas_entity_nudge_fires_for_pinball_without_art_keyword() -> None:
    out = plan_instruction(goal="Build a pinball table with flippers and bumpers")
    assert "CANVAS ENTITY ART" in out


def test_canvas_entity_nudge_skips_wireframe() -> None:
    out = plan_instruction(
        goal="vector line art wireframe battlezone tank combat on black background"
    )
    assert "CANVAS ENTITY ART" not in out


def test_pinball_table_nudge_fires() -> None:
    out = plan_instruction(goal="Build a pinball game with plunger and drain")
    assert "PINBALL / TABLE PHYSICS" in out


def test_detect_canvas_entity_intent_requires_two_entities() -> None:
    assert len(_detect_canvas_entity_intent("build a pinball table")) < 2
    assert len(_detect_canvas_entity_intent("pinball with flippers and bumpers")) >= 2


def test_detect_pinball_intent() -> None:
    assert _detect_pinball_intent("make a pinball game") != []
    assert _detect_pinball_intent("flipper bumper table") != []


def test_detect_open_field_td_intent() -> None:
    assert _detect_open_field_td_intent("Build a Fieldrunners game") != []
    assert _detect_open_field_td_intent(
        "open-field tower defense with BFS pathfinding"
    ) != []
    assert _detect_open_field_td_intent("simple pong with paddle") == []


def test_open_field_td_nudge_fires() -> None:
    out = plan_instruction(
        goal="Build a Fieldrunners open-field tower defense with Tesla and Flame towers"
    )
    assert "BEAM TOWERS (Tesla / Flame)" in out
    assert "NOT rotatable" in out
    assert "PROCEDURAL attack" in out


def test_plan_nudges_include_new_ids() -> None:
    for nid in ("canvas-entity-art", "pinball-table", "open-field-td"):
        assert memory_module.load_plan_nudge(nid).strip(), nid


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )


def test_outline_traps_injected_on_memory_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(tmp_path)
    agent._goal = "pinball table with flippers"
    block = agent._retrieve_outline_traps_block(
        agent._goal,
        "CONTROL-NOT-RECOVERED: ball stuck",
        failure_class="memory_gap",
    )
    assert "OUTLINE TRAPS" in block
    assert "plunger" in block.lower() or "flipper" in block.lower()


def test_outline_traps_skipped_when_unrelated_failure(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    agent._goal = "calculator app"
    block = agent._retrieve_outline_traps_block(
        agent._goal,
        "Title mismatch only",
        failure_class="harness_bug",
    )
    assert block == ""


def test_pinball_skeleton_mapped() -> None:
    from memory import _RECIPE_TO_SKELETON
    assert _RECIPE_TO_SKELETON.get("canvas-pinball") == "canvas_pinball_basic"
    skel = PROJECT_ROOT / "memory" / "skeletons" / "canvas_pinball_basic.html"
    assert skel.exists()
    assert "resolveCircle" in skel.read_text()


def test_prompt_library_pinball_expects_assets_hint() -> None:
    for line in (PROJECT_ROOT / "memory" / "prompt_library.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("name") == "pinball":
            names = o.get("expect", {}).get("asset_names_any") or []
            assert "ball" in names
            break
    else:
        pytest.fail("pinball eval prompt not found")
