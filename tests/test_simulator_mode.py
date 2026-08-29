"""Simulator mode (`/640`, `/media off`) — prompt + pipeline gating."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402
from tools import _canvas_default_size_warning  # noqa: E402


def test_build_system_prompt_simulator_drops_media_and_injects_jmr_block():
    sp = prompts_v1.build_system_prompt(
        "space invaders with pixel sprites",
        model_class="large",
        simulator_mode=True,
    )
    tags_start = sp.index("<output-tags>")
    tags_end = sp.index("</output-tags>")
    tags_block = sp[tags_start:tags_end]
    assert "  <assets>" not in tags_block
    assert "  <sounds>" not in tags_block
    assert "  <videos>" not in tags_block
    assert "<simulator-target>" in sp
    assert "640" in sp
    assert "Object.keys" in sp
    assert "per-tick maze" in sp or "BFS" in sp
    # Lean: /640 must not ship the full large schema (~23KB).
    assert len(sp) < 12000


def test_plan_instruction_simulator_has_no_expected_assets_block():
    sim = prompts_v1.plan_instruction(
        goal="pixel sprite shooter with doom 3d",
        simulator_mode=True,
    )
    assert "EXPECTED — emit an <assets>" not in sim
    assert "<sounds>" not in sim or "Do NOT" in sim or "No <assets>" in sim
    assert "three.js" not in sim.lower()
    assert "cdn.jsdelivr" not in sim
    assert "<plan>" in sim
    assert "640" in sim


def test_plan_instruction_simulator_suppresses_art_nudge():
    normal = prompts_v1.plan_instruction(goal="pixel sprite shooter")
    sim = prompts_v1.plan_instruction(
        goal="pixel sprite shooter", simulator_mode=True,
    )
    assert "MUST emit an <assets>" in normal or "REQUIRED" in normal
    assert "MUST emit an <assets>" not in sim
    assert "canvas-entity" not in sim.lower() or "EMIT <assets>" not in sim


def test_game_agent_media_pipeline_toggle():
    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    assert a.media_pipeline_enabled()
    a.set_simulator_mode(True)
    assert not a.media_pipeline_enabled()
    a.set_simulator_mode(False)
    assert a.media_pipeline_enabled()


def test_canvas_default_size_hint_640_in_simulator_mode():
    msg = _canvas_default_size_warning(
        {"width": 300, "height": 150}, "<html><canvas></canvas></html>",
        simulator_mode=True,
    )
    assert msg is not None
    assert "640" in msg
    assert "800" not in msg


def test_maybe_generate_assets_skipped_in_simulator_mode():
    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    a.set_simulator_mode(True)
    reply = '<assets>[{"name":"hero","prompt":"knight"}]</assets>'

    async def _run():
        events = []
        async for ev in a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"):
            events.append(ev)
        return events

    assert asyncio.run(_run()) == []
