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
    assert "pixel map" in sp.lower() or "Pixel maps" in sp
    assert "NOT colored circles" in sp or "placeholders" in sp.lower()
    # Lean: /640 must not ship the full large schema (~23KB).
    assert len(sp) < 13000


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
    assert "pixel" in sim.lower() or "sprite" in sim.lower()
    assert "placeholder" in sim.lower() or "geometric" in sim.lower()


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


def test_simulator_falls_back_skeleton_with_asset_refs():
    from memory import SkeletonHit

    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    a.set_simulator_mode(True)
    skel = SkeletonHit(
        name="won_fieldrunners.html",
        html="drawImage(ASSETS.tower_gun); const p='./x_assets/creep.png';",
        score=1.0,
        source_goal="Build a Fieldrunners game",
    )
    fallback, reason = a._local_should_fallback_skeleton(skel)
    assert fallback is True
    assert "simulator" in reason


def test_simulator_falls_back_oversized_skeleton():
    from memory import SkeletonHit

    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    a.set_simulator_mode(True)
    skel = SkeletonHit(
        name="won_huge.html",
        html="x" * 30_000,
        score=1.0,
        source_goal="big game",
    )
    fallback, reason = a._local_should_fallback_skeleton(skel)
    assert fallback is True
    assert "too large" in reason


def test_simulator_lean_budget_prefers_components():
    """Mirror Fieldrunners 20260829: opening ate budget, BFS components dropped."""
    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    a.set_simulator_mode(True)
    a.set_lean_prompt(True)
    opening = "O" * 4064
    components = "C" * 2296  # bfs-grid-sized
    playbook = "P" * 900
    ob, cb, pb = a._apply_lean_memory_budget(opening, components, playbook)
    assert cb == components
    assert len(ob) + len(cb) + len(pb) <= a._LEAN_MEMORY_COMBINED_BUDGET
    assert len(cb) >= len(ob)  # components kept preferentially


def test_simulator_placeholder_art_helper_still_detects_boxes():
    """Detector kept for diagnostics; harness no longer fails the run on it."""
    from tools import simulator_placeholder_art_soft_warning

    html = """
    <canvas></canvas><script>
    function drawTower(t){ ctx.fillStyle='#48f'; ctx.fillRect(t.x,t.y,16,16); }
    function drawEnemy(e){ ctx.fillStyle='#f84'; ctx.fillRect(e.x,e.y,12,12); }
    function draw(){ drawTower(t); drawEnemy(e); ctx.fillRect(0,0,10,10);
      ctx.fillRect(1,1,1,1); ctx.fillRect(2,2,1,1); ctx.fillRect(3,3,1,1); }
    </script>
    """
    msg = simulator_placeholder_art_soft_warning(html, goal="open-field tower defense")
    assert msg is not None and "PLACEHOLDER-GEOMETRY-ART" in msg


def test_simulator_placeholder_art_helper_allows_pixel_maps():
    from tools import simulator_placeholder_art_soft_warning

    html = """
    <script>
    const TOWER=[[0,1,1,0,1,0],[1,1,1,1,1,1],[0,1,1,0,1,0],[1,0,1,0,1,0],[0,1,0,1,0,1],[1,1,0,0,1,1]];
    const CREEP=[[0,1,0,1,0,1],[1,1,1,1,1,1],[0,1,1,1,1,0],[1,0,1,0,1,0],[0,1,0,1,0,1],[1,1,1,0,1,1]];
    const SHOT=[[0,1,0,1,0,0],[1,1,1,1,1,0],[0,1,1,1,0,0],[1,0,1,0,1,0],[0,1,0,1,0,0],[0,0,1,0,0,0]];
    function drawTower(t){ blit(TOWER,t.x,t.y,2); }
    function drawEnemy(e){ blit(CREEP,e.x,e.y,2); }
    function blit(m,x,y,s){ for(let r=0;r<m.length;r++)for(let c=0;c<m[r].length;c++)
      if(m[r][c]) ctx.fillRect(x+c*s,y+r*s,s,s); }
    </script>
    """
    assert simulator_placeholder_art_soft_warning(html, goal="tower defense") is None


def test_simulator_system_prompt_uses_jmr_hard_rules_not_cdn():
    sp = prompts_v1.build_system_prompt(
        "fieldrunners tower defense", model_class="large", simulator_mode=True,
    )
    assert "HARD_RULES" not in sp  # content check below
    assert "640" in sp
    assert "pixel map" in sp.lower() or "Pixel maps" in sp or "pixel maps" in sp
    assert "FIRST <html_file>" in sp or "FIRST build" in sp.lower() or "turn 1" in sp.lower()
    # Standard CDN / assetsReady rules must not fight /640.
    assert "Phaser, three.js" not in sp
    assert "_assetsReady" not in sp


def test_standard_prompt_rejects_box_entities_as_final_art():
    sp = prompts_v1.build_system_prompt(
        "space invaders", model_class="large", simulator_mode=False,
    )
    assert "solid fillRect" in sp or "colored boxes" in sp.lower() or "NEVER leave solid" in sp


def test_simulator_ensures_pixel_map_playbook():
    a = GameAgent(model="stub", out_path=Path("games/test_sim.html"))
    a.set_simulator_mode(True)
    ids = a._first_build_playbook_ensure_ids("open-field tower defense maze")
    assert ids is not None
    assert "classic-arcade-pixel-maps" in ids


def test_simulator_first_build_asks_for_pixel_art_upfront():
    msg = prompts_v1.first_build_instruction(
        "<html><canvas></canvas></html>",
        None,
        simulator_mode=True,
    )
    assert "pixel map" in msg.lower() or "data:image" in msg
    assert "fix art later" in msg.lower() or "together on turn 1" in msg


def test_games_list_header_is_not_a_rich_closing_tag():
    """`[/640 pixel-map goals]` crashes RichLog (MarkupError on restart+/games)."""
    src = Path(__file__).resolve().parents[1].joinpath("chat.py").read_text(
        encoding="utf-8"
    )
    assert "[/640 pixel-map goals]" not in src
    assert "(/640 pixel-map goals)" in src
    assert "(/640png sheet goals)" in src
    # Error log body must be escaped so a MarkupError cannot crash again.
    assert 'self._log(f"[red]![/red] {_esc(text)}")' in src


def test_jmr_png_system_prompt_keeps_assets_drops_sounds():
    sp = prompts_v1.build_system_prompt(
        "space invaders",
        model_class="large",
        jmr_png_mode=True,
    )
    tags_start = sp.index("<output-tags>")
    tags_end = sp.index("</output-tags>")
    tags_block = sp[tags_start:tags_end]
    assert "  <assets>" in tags_block
    assert "  <sounds>" not in tags_block
    assert "  <videos>" not in tags_block
    assert "jmr:spr" in sp
    assert "STEM-N.png" in sp or "STEM-0.png" in sp
    assert "Object.keys" in sp
    assert "quoted literal" in sp.lower() or "dest x,y" in sp.lower()
    assert "640" in sp
    assert "Phaser, three.js" not in sp


def test_jmr_png_plan_instruction_expects_assets():
    plan = prompts_v1.plan_instruction(
        goal="pixel sprite shooter", jmr_png_mode=True,
    )
    assert "<assets>" in plan
    assert "jmr:spr" in plan or "STEM" in plan
    assert "No <sounds>" in plan or "No <sounds>/<videos>" in plan


def test_jmr_png_enables_sprite_pipeline():
    a = GameAgent(model="stub", out_path=Path("games/test_jmr.png.html"))
    a.set_jmr_png_mode(True)
    assert a.media_pipeline_enabled()
    assert a._simulator_mode
    a.set_simulator_mode(True)
    a.set_jmr_png_mode(False)
    assert not a.media_pipeline_enabled()


def test_jmr_png_playbook_pin():
    a = GameAgent(model="stub", out_path=Path("games/test_jmr.png.html"))
    a.set_jmr_png_mode(True)
    ids = a._first_build_playbook_ensure_ids("open-field tower defense maze")
    assert ids is not None
    assert "jmr-png-sheets" in ids
    assert "classic-arcade-pixel-maps" not in ids


def test_jmr_png_first_build_asks_for_jmr_spr():
    msg = prompts_v1.first_build_instruction(
        "<html><canvas></canvas></html>",
        None,
        jmr_png_mode=True,
        has_generated_assets=True,
    )
    assert "jmr:spr" in msg
    assert "sprite()" not in msg or "Do NOT use sprite()" in msg
    assert "640" in msg


def test_jmr_png_maybe_generate_not_skipped():
    a = GameAgent(model="stub", out_path=Path("games/test_jmr.png.html"))
    a.set_jmr_png_mode(True)
    reply = '<assets>[{"name":"hero","prompt":"knight"}]</assets>'

    async def _run():
        events = []
        async for ev in a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"):
            events.append(ev)
        return events

    # Pipeline is enabled; stub has no diffuser so we get info events, not [].
    events = asyncio.run(_run())
    assert events != []
