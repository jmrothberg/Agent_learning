"""Regression tests for 3D navigation convention fixes (2026-06-30).

Pins per-modality movement bases in skeletons, playbook cross-modality
suppression, minimap bullet retrieval, FPS-vs-grid playtest routing, and
the _identifier_occurrence_slice agent crash fix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent
from agent_prompts import PromptBuildingMixin
from memory import GameMemory, Playbook, find_best_visual_playtest
from modality import detect_fps_navigation_modality
from prompts_v1 import input_moves_player_probe_expr
from tools import _threejs_manual_navigation_basis_risk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKELETONS = PROJECT_ROOT / "memory" / "skeletons"


def test_identifier_occurrence_slice_callable_on_instance() -> None:
    """PromptBuildingMixin._identifier_occurrence_slice must accept self."""
    html = "line one\nconst yaw = state.player.yaw;\nline three"
    out = PromptBuildingMixin._identifier_occurrence_slice(html, ["yaw"])
    assert "yaw" in out
    assert "2:" in out or "  2:" in out


def test_canvas_3d_basic_uses_camera_relative_movement() -> None:
    src = (SKELETONS / "canvas_3d_basic.html").read_text()
    assert "applyQuaternion(camera.quaternion)" in src
    assert "window.state" in src
    assert "camera.position.x -=" not in src
    assert "requestPointerLock" in src


def test_canvas_vector_wireframe_forward_uses_plus_cos_z() -> None:
    src = (SKELETONS / "canvas_vector_wireframe_basic.html").read_text()
    assert "Math.cos(state.player.yaw)" in src
    assert "strokeSeg" in src
    assert "KeyA" in src and "KeyD" in src


def test_canvas_mode7_uses_same_angle_in_update_and_draw() -> None:
    src = (SKELETONS / "canvas_mode7_basic.html").read_text()
    assert "p.angle" in src
    assert "Math.cos(p.angle)" in src
    assert "Math.sin(p.angle)" in src


def test_voxel_skeleton_exposes_window_state() -> None:
    src = (SKELETONS / "canvas_voxel_minecraft_basic.html").read_text()
    assert "window.state" in src
    assert "applyQuaternion" in src


def test_input_moves_player_probe_threejs_for_fps_goal() -> None:
    expr = input_moves_player_probe_expr(goal="first person doom shooter 3d")
    assert "ArrowUp" in expr
    assert "player.x" in expr or "p.x" in expr


def test_input_moves_player_probe_wireframe_for_battlezone() -> None:
    expr = input_moves_player_probe_expr(
        goal="battlezone wireframe vector tank first person",
    )
    assert "ArrowLeft" in expr
    assert "yaw" in expr


def test_plan_time_wireframe_probe_from_goal_without_wireframe_word() -> None:
    expr = input_moves_player_probe_expr(goal="battlezone tank vector combat")
    assert "ArrowLeft" in expr
    assert "yaw" in expr


def test_threejs_forward_basis_aligns_with_camera() -> None:
    import math

    yaw = math.pi / 2
    cam_x, cam_z = -math.sin(yaw), -math.cos(yaw)
    old_mx, old_mz = math.sin(yaw), -math.cos(yaw)
    new_mx, new_mz = -math.sin(yaw), -math.cos(yaw)
    assert old_mx * cam_x + old_mz * cam_z < 0
    assert abs(new_mx * cam_x + new_mz * cam_z - 1.0) < 1e-9


def test_modality_detect_fps_navigation_wireframe_goal() -> None:
    assert detect_fps_navigation_modality(goal="battlezone tank vector") == "wireframe"


def test_modality_detect_fps_navigation_threejs_code() -> None:
    assert detect_fps_navigation_modality(
        goal="maze shooter",
        code="new THREE.PerspectiveCamera();",
    ) == "threejs"


def test_modality_detect_fps_navigation_mode7() -> None:
    assert detect_fps_navigation_modality(goal="mode7 racer") == "mode7"


def test_pointer_lock_component_threejs_convention() -> None:
    import json

    line = next(
        ln for ln in (PROJECT_ROOT / "memory" / "components.jsonl").read_text().splitlines()
        if '"id": "pointer-lock-fps-look"' in ln
    )
    obj = json.loads(line)
    code = obj["recipe"]["code"]
    assert "fx = -Math.sin(yaw)" in code
    assert "yaw -= e.movementX" in code
    assert "fz = Math.cos(p.yaw)" not in code


@pytest.fixture
def playbook() -> Playbook:
    old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        return Playbook(root="memory")
    finally:
        os.chdir(old_cwd)


def test_playbook_battlezone_surfaces_wireframe_not_fps_camera(playbook: Playbook) -> None:
    goal = "battlezone wireframe vector tank combat radar"
    hits = playbook.retrieve(goal, k=6, stage="code")
    ids = [h.bullet.id for h in hits]
    assert "wireframe-fps-movement-vectors" in ids
    assert ids.index("wireframe-fps-movement-vectors") < (
        ids.index("fps-camera-and-movement-vectors")
        if "fps-camera-and-movement-vectors" in ids else 999
    )


def test_playbook_doom_surfaces_fps_camera(playbook: Playbook) -> None:
    goal = "first person doom shooter three.js weapon crosshair"
    hits = playbook.retrieve(
        goal, k=6, stage="code",
        modality_tokens=["3d", "first-person", "fps", "doom"],
    )
    ids = [h.bullet.id for h in hits]
    assert "fps-camera-and-movement-vectors" in ids


def test_playbook_minimap_feedback_retrieves_minimap_bullet(playbook: Playbook) -> None:
    goal = "first person maze shooter"
    feedback = (
        "minimap arrow mirrored wrong way when I turn right "
        "the map line goes counter clockwise"
    )
    query = f"{goal} {feedback} {feedback}"
    hits = playbook.retrieve(query, k=6, stage="code")
    ids = [h.bullet.id for h in hits]
    assert "fps-minimap-radar-yaw-arrow" in ids


def test_playbook_suppression_wireframe_blocks_fps_camera() -> None:
    suppressed = GameAgent._playbook_suppressed_bullet_ids(
        goal="battlezone wireframe vector tank",
        active_skeleton="canvas_vector_wireframe_basic.html",
        code="function strokeSeg() {}",
    )
    assert "fps-camera-and-movement-vectors" in suppressed
    assert "fps-minimap-radar-yaw-arrow" in suppressed
    assert "wireframe-fps-movement-vectors" not in suppressed


def test_playbook_suppression_plan_time_wireframe_goal() -> None:
    suppressed = GameAgent._playbook_suppressed_bullet_ids(
        goal="battlezone tank vector combat",
        active_skeleton=None,
        code="",
    )
    assert "fps-camera-and-movement-vectors" in suppressed


def test_playbook_suppression_threejs_blocks_wireframe_bullet() -> None:
    suppressed = GameAgent._playbook_suppressed_bullet_ids(
        goal="first person doom shooter",
        active_skeleton="canvas_3d_basic.html",
        code="new THREE.PerspectiveCamera()",
    )
    assert "wireframe-fps-movement-vectors" in suppressed
    assert "wireframe-minimap-radar-yaw-arrow" in suppressed


def test_threejs_navigation_basis_risk_detector() -> None:
    bad = (
        "const fx = Math.sin(yaw), fz = -Math.cos(yaw);\n"
        "camera.rotation.y = state.player.yaw;\n"
        "new THREE.WebGLRenderer();"
    )
    good = (
        "move.applyQuaternion(camera.quaternion);\n"
        "camera.rotation.y = state.player.yaw;\n"
        "new THREE.WebGLRenderer();"
    )
    assert _threejs_manual_navigation_basis_risk(bad) is True
    assert _threejs_manual_navigation_basis_risk(good) is False


def test_playbook_suppression_mode7_blocks_all_fps_nav_bullets() -> None:
    suppressed = GameAgent._playbook_suppressed_bullet_ids(
        goal="mode7 racer",
        active_skeleton="canvas_mode7_basic.html",
        code="",
    )
    assert "fps-camera-and-movement-vectors" in suppressed
    assert "wireframe-fps-movement-vectors" in suppressed
    assert "fps-minimap-radar-yaw-arrow" in suppressed
    assert "wireframe-minimap-radar-yaw-arrow" in suppressed


def test_playbook_retrieves_3d_navigation_invariants(playbook: Playbook) -> None:
    query = (
        "movement camera opposite facing walk backwards gun wrong axis "
        "three.js first person"
    )
    hits = playbook.retrieve(query, k=8, stage="code")
    ids = [h.bullet.id for h in hits]
    assert "3d-navigation-modality-invariants" in ids


def test_fps_playtest_routing_prefers_3d_when_code_has_yaw() -> None:
    mem = GameMemory(root=str(PROJECT_ROOT / "memory"))
    recipes = [r for r in mem.load_visual_playtests() if r.id in (
        "canvas-grid-navigation", "canvas-3d-first-person", "canvas-controllable-player",
    )]
    code = (
        "const state = { player: { x:0, z:0, yaw: 0 } };\n"
        "new THREE.PerspectiveCamera();\n"
        "document.requestPointerLock();\n"
    )
    winner, _diag = find_best_visual_playtest(
        recipes,
        goal="maze too dark add map in top right corner",
        plan_text="",
        asset_names=[],
        code=code,
    )
    assert winner is not None
    assert winner.id == "canvas-3d-first-person"


def _make_agent(tmp_path: Path, goal: str) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html><body>x</body></html>")
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    a._goal = goal
    return a


def test_continuation_feedback_in_playbook_query(tmp_path: Path) -> None:
    a = _make_agent(tmp_path, "first person doom maze")
    a._last_drained_feedback = []
    a._pending_feedback = []
    a._continuation_feedback = (
        "minimap arrow mirrored when turning right fix the map only"
    )
    events: list[dict] = []
    orig = a._trace
    a._trace = lambda obj: events.append(obj) or orig(obj)
    a._retrieve_playbook_block(a._goal, code="", stage="code")
    fb_evs = [e for e in events if e.get("kind") == "playbook_retrieved"]
    assert fb_evs
    assert fb_evs[-1].get("feedback_in_query") is True
