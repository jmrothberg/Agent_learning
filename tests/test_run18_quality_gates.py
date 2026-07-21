"""run_18 quality gates: screenshot sky/ink, obstacle depth, opaque scenery, pins."""
from __future__ import annotations

import inspect
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

import tools
from agent import GameAgent


def _png_bytes(color: tuple[int, int, int], size: int = 64) -> bytes:
    im = Image.new("RGB", (size, size), color)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _png_sparse_dark_green() -> bytes:
    """~2% near-black green ink on black — Battlezone-like."""
    im = Image.new("RGB", (128, 128), (0, 0, 0))
    px = im.load()
    for i in range(0, 128, 16):
        for j in range(3):
            px[i, 40 + j] = (10, 51, 10)  # ~#0a330a
            px[20 + j, i] = (10, 26, 10)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _png_sparse_bright_cyan() -> bytes:
    """Sparse but bright lines — Asteroids-like (thick enough to survive resize)."""
    im = Image.new("RGB", (128, 128), (0, 0, 0))
    px = im.load()
    for i in range(0, 128, 12):
        for t in range(3):
            px[min(127, i + t), 60] = (0, 255, 255)
            px[60, min(127, i + t)] = (0, 255, 180)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_viewport_sky_dominant():
    a = tools.analyze_viewport_png(_png_bytes((135, 206, 235)))
    assert a is not None
    assert a["dominant_frac"] >= 0.95


def test_analyze_viewport_multi_hue_not_sky():
    im = Image.new("RGB", (64, 64), (20, 20, 20))
    px = im.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = ((x * 3) % 255, (y * 5) % 255, (x + y) % 255)
    buf = BytesIO()
    im.save(buf, format="PNG")
    a = tools.analyze_viewport_png(buf.getvalue())
    assert a is not None
    assert a["dominant_frac"] < 0.95


def test_webgl_empty_screenshot_soft_warning_sky():
    html = "<script src='three.min.js'></script><canvas></canvas>"
    a = tools.analyze_viewport_png(_png_bytes((135, 206, 235)))
    w = tools.webgl_empty_screenshot_soft_warning(html, a)
    assert w and "EMPTY-3D-VIEW" in w


def test_webgl_empty_screenshot_skips_rich_view():
    html = "<script>THREE.WebGLRenderer</script>"
    im = Image.new("RGB", (64, 64))
    px = im.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = (80 + (x % 40), 40 + (y % 30), 30)
    buf = BytesIO()
    im.save(buf, format="PNG")
    a = tools.analyze_viewport_png(buf.getvalue())
    assert tools.webgl_empty_screenshot_soft_warning(html, a) is None


def test_dim_vector_ink_catches_dark_sparse():
    html = "<canvas id=c></canvas><script>ctx.stroke()</script>"
    a = tools.analyze_viewport_png(_png_sparse_dark_green())
    w = tools.dim_vector_ink_soft_warning(html, a)
    assert w and "DIM-VECTOR-SCENE" in w


def test_dim_vector_ink_skips_bright_sparse():
    html = "<canvas></canvas><script>stroke</script>"
    a = tools.analyze_viewport_png(_png_sparse_bright_cyan())
    assert tools.dim_vector_ink_soft_warning(html, a) is None


def test_dim_vector_skips_webgl():
    html = "THREE.WebGLRenderer"
    a = tools.analyze_viewport_png(_png_sparse_dark_green())
    assert tools.dim_vector_ink_soft_warning(html, a) is None


def test_obstacle_depth_stall_when_z_frozen_and_distance_advances():
    a = {"zs": [7000.0, 7200.0, 7400.0], "distance": 100.0}
    b = {"zs": [7000.0, 7200.0, 7400.0], "distance": 400.0}
    w = tools.obstacle_depth_stall_soft_warning(a, b)
    assert w and "OBSTACLE-DEPTH-STALL" in w


def test_obstacle_depth_ok_when_z_approaches():
    a = {"zs": [7000.0, 7200.0], "distance": 100.0}
    b = {"zs": [5000.0, 5200.0], "distance": 400.0}
    assert tools.obstacle_depth_stall_soft_warning(a, b) is None


def test_opaque_scenery_soft_warning(tmp_path: Path):
    # Mid-alpha figure with opaque gray edges (wall leftovers).
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    px = im.load()
    for x in range(64):
        for y in range(64):
            if x < 4 or x >= 60 or y < 4 or y >= 60:
                px[x, y] = (120, 120, 115, 255)  # wall gray edge
            elif 16 <= x < 48 and 16 <= y < 48:
                px[x, y] = (40, 180, 40, 255)  # figure
    path = tmp_path / "monster_idle.png"
    im.save(path)
    w = tools.opaque_scenery_soft_warning_for_png("monster_idle", path)
    assert w and "OPAQUE-SPRITE-SCENERY" in w


def test_opaque_scenery_skips_clean_sprite(tmp_path: Path):
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    px = im.load()
    for x in range(20, 44):
        for y in range(20, 44):
            px[x, y] = (40, 180, 40, 255)
    path = tmp_path / "monster_idle.png"
    im.save(path)
    assert tools.opaque_scenery_soft_warning_for_png("monster_idle", path) is None


def test_opaque_scenery_skips_keyart_even_when_boss_in_name(tmp_path: Path):
    """Role skip: keyart/title plates must not arm OPAQUE-SPRITE via token 'boss'.

    Trace build-a-doom-game-first-person_20260721_132716: keyart_boss matched
    _CHARACTER_SPRITE_NAME_RE and hard soft_warning'd a probe-green build.
    Same wall-edge pixels must still warn for monster_idle.
    """
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    px = im.load()
    for x in range(64):
        for y in range(64):
            if x < 4 or x >= 60 or y < 4 or y >= 60:
                px[x, y] = (120, 120, 115, 255)
            elif 16 <= x < 48 and 16 <= y < 48:
                px[x, y] = (40, 180, 40, 255)
    path = tmp_path / "keyart_boss.png"
    im.save(path)
    assert tools.opaque_scenery_soft_warning_for_png("keyart_boss", path) is None
    assert tools.opaque_scenery_soft_warning_for_png("title_boss", path) is None
    # Real character stem still warns on the same pixels.
    assert tools.opaque_scenery_soft_warning_for_png("monster_idle", path)


def test_webgl_undrawn_gate_skipped_in_source():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "not _is_webgl_or_three_game(_src_html)" in src
    assert "EMPTY-3D-VIEW" in src or "webgl_empty_screenshot_soft_warning" in src


def test_webgl_blank_readpixels_stays_advisory_only():
    """readPixels blank path remains advisory; screenshot gate is the blocker."""
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "ADVISORY (non-blocking): 3D/canvas view appears blank" in src
    assert "webgl_empty_screenshot_soft_warning" in src


def _agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(Path(__file__).resolve().parents[1] / "memory"),
    )


def test_ensure_ids_wireframe_and_trench(tmp_path: Path):
    ag = _agent(tmp_path)
    ag._session_assets = [{"name": "x"}]  # would pin drawImage if 2D
    ag._active_visual_playtest_recipe_id = "canvas-vector-wireframe"
    ids = ag._first_build_playbook_ensure_ids(
        "Build a wireframe vector trench run with towers"
    )
    assert ids
    assert "vector-stroke-contrast" in ids
    assert "trench-depth-vector-spawn" in ids
    assert "draw-generated-sprites-not-boxes" not in ids


def test_ensure_ids_fps_omits_drawimage(tmp_path: Path):
    ag = _agent(tmp_path)
    ag._session_assets = [{"name": "wall"}]
    ag._active_visual_playtest_recipe_id = "canvas-3d-first-person"
    ids = ag._first_build_playbook_ensure_ids(
        "Build a first-person maze with three.js pointer-lock"
    )
    assert ids
    assert "fps-camera-and-movement-vectors" in ids
    assert "draw-generated-sprites-not-boxes" not in ids


def test_ensure_ids_fixed_shooter_keeps_drawimage(tmp_path: Path):
    ag = _agent(tmp_path)
    ag._session_assets = [{"name": "ship"}]
    ag._active_visual_playtest_recipe_id = "canvas-fixed-shooter"
    ids = ag._first_build_playbook_ensure_ids("Build a fixed shooter")
    assert "draw-generated-sprites-not-boxes" in ids


def test_frozen_merge_keeps_class_pins(tmp_path: Path):
    ag = _agent(tmp_path)
    ag._current_goal = "Build a 2D wireframe vector tank game with radar"
    ag._active_visual_playtest_recipe_id = "canvas-vector-wireframe"
    report = {
        "frozen_canvas": True,
        "soft_warnings": ["FROZEN-CANVAS: idle"],
        "probes": [],
    }
    ids = ag._playbook_ensure_ids_for_report(report)
    assert "raf-must-start" in ids
    assert "vector-stroke-contrast" in ids or "wireframe-fps-movement-vectors" in ids


def test_report_pins_new_soft_warnings(tmp_path: Path):
    ag = _agent(tmp_path)
    ids = ag._playbook_ensure_ids_for_report(
        {
            "soft_warnings": [
                "EMPTY-3D-VIEW: sky",
                "OBSTACLE-DEPTH-STALL: z",
                "OPAQUE-SPRITE-SCENERY [monster]: wall",
            ],
            "probes": [],
        }
    )
    assert "voxel-mesh-simple-or-groups" in ids
    assert "trench-depth-vector-spawn" in ids
    assert "character-sprite-isolation" in ids


def test_playbook_new_ids_exist():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "memory" / "playbook.jsonl"
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.add(json.loads(line)["id"])
    assert "vector-stroke-contrast" in ids
    assert "voxel-mesh-simple-or-groups" in ids
    assert "character-sprite-isolation" in ids


def test_prompt_library_run18_wording():
    from prompt_library import get_prompt, load_prompt_library

    by_name = {p["name"]: p for p in load_prompt_library()}
    fw = by_name["particle-fireworks"]["prompt"].lower()
    assert "auto" in fw and ("intentional" in fw or "timer" in fw)
    rp = by_name["rampage"]["prompt"].lower()
    assert "isolated" in rp or "figure only" in rp or "no building" in rp
    bz = by_name["battlezone"]["prompt"].lower()
    assert "bright" in bz or "#0f0" in bz
    assert get_prompt(28) is not None
