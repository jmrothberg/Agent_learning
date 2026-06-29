"""Run_06 general improvements — drawImage contract, memory, harness softening."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
import tools  # noqa: E402
from memory import Playbook  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent


def test_assets_format_no_mix_freely():
    blob = " ".join(prompts_v1.ASSETS_FORMAT.guidelines)
    assert "MIX sprites and procedural drawing freely" not in blob
    assert "DRAW CONTRACT" in blob
    assert "sprite(key)" in blob or "drawImage" in blob


def test_generated_sprite_draw_contract_helper():
    c = prompts_v1.generated_sprite_draw_contract()
    assert "GENERATED-SPRITE DRAW CONTRACT" in c
    assert "drawImage" in c
    assert "MISSING" in c
    assert "window.state" in c


def test_first_build_seed_framing_when_assets_exist():
    body = prompts_v1.first_build_instruction(
        "<html>seed</html>",
        has_generated_assets=True,
    )
    assert "REPLACE any procedural entity draw bodies" in body
    assert "window.state = state" in body


def test_playbook_draw_generated_sprites_retrieves_for_art_goal():
    pb = Playbook()
    goal = (
        "Build a platformer with colorful sprites and drawImage for every "
        "generated character enemy pickup"
    )
    hits = pb.retrieve(goal, stage="code", k=8)
    ids = {h.bullet.id for h in hits}
    assert "draw-generated-sprites-not-boxes" in ids


def test_skeletons_show_drawimage_pattern():
    for name in (
        "canvas_board_turn_basic.html",
        "canvas_platformer_basic.html",
        "canvas_basic_v2.html",
    ):
        text = (PROJECT_ROOT / "memory/skeletons" / name).read_text()
        assert "drawImage" in text, name
        assert "sprite(" in text, name
        assert "MISSING" in text, name


def test_undrawn_state_gated_helper():
    assert tools._undrawn_likely_state_gated(["boss_idle", "boss_attack"])
    assert not tools._undrawn_likely_state_gated(["hero_idle", "enemy_walk1"])


def test_entity_render_js_uses_bounding_box_sample():
    src = tools._ENTITY_RENDERED_JS
    assert "halfW" in src
    assert "halfH" in src
    assert "getImageData(bx, by, bw, bh)" in src


def test_entity_not_rendered_advisory_when_probes_green():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "ENTITY-NOT-RENDERED" in src
    assert "ADVISORY (non-blocking) — " in src
    assert "_ent_probes_green" in src


def test_undrawn_first_occurrence_advisory_when_state_gated():
    src = inspect.getsource(tools.LiveBrowser.load_and_test)
    assert "_undrawn_likely_state_gated" in src
    assert "_undrawn_state_gated" in src


def test_outline_turn_based_board_mentions_drawimage():
    for line in (PROJECT_ROOT / "memory/implementation_outlines.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("id") == "outline-turn-based-board":
            assert "drawimage" in o.get("content", "").lower()
            break
    else:
        raise AssertionError("outline-turn-based-board not found")
