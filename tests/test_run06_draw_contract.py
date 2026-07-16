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
    # run_13: undrawn-art self-check before emit
    assert "SELF-CHECK" in c
    assert "GENERATED ASSETS" in c


def test_probes_format_requires_self_contained_helpers():
    """run_13 Elite Trader: bare simulateClick() in probes failed forever."""
    blob = " ".join(prompts_v1.PROBES_FORMAT.guidelines)
    assert "SELF-CONTAINED" in blob
    assert "simulateClick" in blob


def test_playbook_self_contained_probe_bullet():
    pb = Playbook()
    ids = {b.id for b in pb.load_all()}
    assert "probe-self-contained-no-bare-helpers" in ids


def test_first_build_seed_framing_when_assets_exist():
    body = prompts_v1.first_build_instruction(
        "<html>seed</html>",
        has_generated_assets=True,
    )
    assert "REPLACE any procedural entity draw bodies" in body
    assert "window.state = state" in body


def test_playbook_draw_generated_sprites_has_shooter_tags_for_centipede():
    """First-build pins this bullet via ensure_ids when assets exist; tags
    must include centipede/galaga so plan-stage retrieval can find it."""
    pb = Playbook()
    bullet = next(b for b in pb.load_all() if b.id == "draw-generated-sprites-not-boxes")
    for tag in ("centipede", "galaga", "fixed-shooter", "drawimage"):
        assert tag in bullet.tags


def test_playbook_draw_generated_sprites_retrieves_for_art_goal():
    pb = Playbook()
    goal = (
        "Build a platformer with colorful sprites and drawImage for every "
        "generated character enemy pickup"
    )
    hits = pb.retrieve(goal, stage="plan", k=8)
    ids = {h.bullet.id for h in hits}
    assert "draw-generated-sprites-not-boxes" in ids


def test_playbook_run06_kung_fu_bullets_retrieve():
    pb = Playbook()
    goal = (
        "Side-scrolling beat-em-up with crouch punch kick video cutscene "
        "floor intro boss entrance enemies spawn waves"
    )
    hits = pb.retrieve(goal, stage="code", k=10)
    ids = {h.bullet.id for h in hits}
    assert "harness-movement-not-action-gated" in ids
    assert "video-intro-must-enter-play" in ids


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
    assert "_sprite_paths_wrapper" in src


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
