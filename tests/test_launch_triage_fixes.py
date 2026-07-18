"""Launch/playfield triage fixes (pinball trace 20260701_211752)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from prompts_v1 import fix_instruction  # noqa: E402
from tools import (  # noqa: E402
    _criteria_declares_keyboard_player_movement,
    _recovery_is_physics_ball_only,
    _rotation_mechanic_action_keys,
)

PROJECT_ROOT = Path(__file__).parent.parent


def test_class_launch_playfield_beats_undrawn_memory_gap() -> None:
    cls, reason = GameAgent._classify_failure(
        ok=False,
        materialized=True,
        stall_reason=None,
        coaching_suppressed=False,
        asset_reprompt_cleared=False,
        art_intent=True,
        undrawn_present=True,
        launch_playfield_probe_failed=True,
    )
    assert cls == "memory_gap"
    assert "launch/playfield" in reason
    assert "undrawn" not in reason


def test_recovery_skip_cnr_for_ball_only() -> None:
    crit = "KeyZ left flipper; Space plunger; R restart."
    assert not _criteria_declares_keyboard_player_movement(crit)
    assert _recovery_is_physics_ball_only(["ball.x"], crit)


def test_recovery_keeps_cnr_for_player_movement() -> None:
    crit = "Basic: KeyW moves the player up; ArrowRight walks right."
    assert _criteria_declares_keyboard_player_movement(crit)
    assert not _recovery_is_physics_ball_only(["player.x"], crit)


def test_rotation_mechanic_excludes_flipper_keys() -> None:
    crit = "Controls: KeyZ left flipper, Slash right flipper, Space plunger."
    keys = _rotation_mechanic_action_keys(crit)
    assert "KeyZ" in keys


def test_fix_instruction_leads_with_probe_failure() -> None:
    out = fix_instruction(
        "ISSUES: cosmetic stuff",
        "<html></html>",
        probe_failure_block="- auto_body_enters_playfield: falsy",
    )
    assert out.index("PROBE FAILURE") < out.index("ISSUES")


def test_canvas_pinball_auto_probes_async_launch() -> None:
    for line in (PROJECT_ROOT / "memory" / "visual_playtests.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("id") != "canvas-pinball":
            continue
        names = {p["name"] for p in o["recipe"]["auto_probes"]}
        assert "auto_launch_increases_speed" in names
        assert "auto_body_enters_playfield" in names
        launch = next(p for p in o["recipe"]["auto_probes"] if p["name"] == "auto_launch_increases_speed")
        assert "launched===true" not in launch["expr"]
        assert "async" in launch["expr"]
        # run_16: prior probes mutate ball — reseat before playfield entry check
        body = next(p for p in o["recipe"]["auto_probes"] if p["name"] == "auto_body_enters_playfield")
        assert "KeyR" in body["expr"] or "reset" in body["expr"]
        return
    pytest.fail("canvas-pinball not found")


def test_playbook_ensure_ids_on_auto_probe_failure(tmp_path: Path) -> None:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    agent._active_visual_playtest_recipe_id = "canvas-pinball"
    report = {
        "probes": [
            {"name": "auto_body_enters_playfield", "ok": False},
            {"name": "canvas_present", "ok": True},
        ],
    }
    ids = agent._playbook_ensure_ids_for_report(report)
    assert "launch-into-playfield" in ids
    assert "circle-collision-pushout-cooldown" in ids


def test_playbook_ensure_ids_on_undrawn_warning(tmp_path: Path) -> None:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    report = {
        "probes": [],
        "soft_warnings": [
            "ASSETS_LOADED_BUT_UNDRAWN [hero_walk1, hero_walk2]",
        ],
    }
    ids = agent._playbook_ensure_ids_for_report(report)
    assert "draw-generated-sprites-not-boxes" in ids
    assert "animation-frames-consistent-character" in ids


def test_playbook_ensure_ids_on_control_not_recovered(tmp_path: Path) -> None:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    report = {
        "probes": [],
        "control_not_recovered": {"key": "ArrowRight", "moved_at_start": True},
    }
    ids = agent._playbook_ensure_ids_for_report(report)
    assert "stun-timer-before-early-return" in ids


def test_playbook_ensure_ids_on_camera_moves_probe(tmp_path: Path) -> None:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    report = {
        "probes": [{"name": "camera_moves", "ok": False}],
    }
    ids = agent._playbook_ensure_ids_for_report(report)
    assert "parallax-coordinate-camera" in ids


def test_playbook_ensure_ids_on_hotspot_alignment(tmp_path: Path) -> None:
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    agent = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(PROJECT_ROOT / "memory"),
    )
    report = {
        "probes": [],
        "soft_warnings": [
            "OPENING BOOK CHECK FAILED [pointclick-puzzle-chain]: HOTSPOT_ALIGNMENT_MISS",
        ],
    }
    ids = agent._playbook_ensure_ids_for_report(report)
    assert "pointclick-hotspot-from-source-art" in ids
