"""Trace-backed QTE quality + loop-hardening checks.

These tests cover the data/prompt/loop changes motivated by
build-a-single-screen-quick-ti_20260612_225857: generated media existed but
was not wired, QTE probes were brittle, continuation feedback was reported as
applied after rejected patches, and visual quality regressed without a clean
best.html anchor.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chat as chat_module  # noqa: E402
import prompts_v1  # noqa: E402
import tools as tools_module  # noqa: E402
from assets import render_asset_paths_block  # noqa: E402
from agent import GameAgent  # noqa: E402
from memory import GameMemory  # noqa: E402

ROOT = Path(__file__).parent.parent


def _jsonl_record(path: str, key: str, value: str) -> dict:
    for line in (ROOT / path).read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get(key) == value:
            return obj
    raise AssertionError(f"missing {key}={value} in {path}")


def test_dragons_lair_prompt_requires_generated_media_wiring():
    rec = _jsonl_record("memory/prompt_library.jsonl", "name", "dragons-lair")
    prompt = rec["prompt"]
    for term in (
        "full-screen generated background PNG",
        "drawImage(bg_scene",
        "0-index state.scene",
        "muted full-screen overlays",
        "onended/onerror",
    ):
        assert term in prompt
    assert "Do NOT treat 'painted background' as canvas painting" in prompt


def test_dragons_lair_director_prompt_is_local_model_safe():
    rec = _jsonl_record("memory/prompt_library.jsonl", "name", "dragons-lair-director")
    prompt = rec["prompt"]
    assert "PHASE A REQUIREMENT" in prompt
    assert "<assets>" in prompt and "<sounds>" in prompt and "<videos>" in prompt
    assert rec["expect"]["visual_recipe"] == "canvas-cutscene-qte"
    assert rec["expect"]["outline"] == "outline-cutscene-qte"
    # Concise rewrite (2026-06-13): the field-by-field ROOMS spec and the
    # exhaustive asset-name list were moved out to the outline/skeleton, but
    # the hard requirements (media up front, ROOMS-driven, harness-visible
    # input, skippable video overlays, RAF-immediate) stay in the goal.
    for term in (
        "ROOMS array",
        "requestAnimationFrame(frame) immediately",
        "onended/onerror always continues",
        "pipeline runs up front",
        "0-index state.scene",
    ):
        assert term in prompt
    # Media asset families are referenced by prefix, not an exhaustive list.
    assert "bg_*" in prompt and "hero_*" in prompt and "key_*" in prompt
    # Videos stay up front but as a small local-model-safe set.
    assert "cutscene videos for intro, a funny fail, and victory" in prompt
    assert "18 video" not in prompt
    # The bloated field-by-field spec is gone (now in the skeleton/outline).
    # NOTE (2026-06-14): no raw len() ceiling here — an arbitrary char cap
    # artificially blocked the deliberate CHARACTER CONSISTENCY block. Bloat is
    # guarded SEMANTICALLY instead: the field-by-field hazardPath spec must be
    # absent and asset families must be referenced by prefix (checked above).
    assert "hazardPath:{from:{x,y}" not in prompt


def test_timed_media_components_retrieve_without_dragon_words():
    mem = GameMemory()
    mem.ensure()
    goals = [
        "boss telegraphs an axe swing; player must press dodge during a timed cue window",
        "rhythm game with notes entering a hit window and skippable intro video",
        "cinematic trap rooms with scripted hazards, generated media, and cutscene overlays",
    ]
    all_ids: set[str] = set()
    for goal in goals:
        hits = mem.retrieve_components(goal, modality=["canvas"], k=5)
        ids = {h.item.id for h in hits}
        all_ids |= ids
        assert "timed-window-qte-manager" in ids
        assert ids & {"nonblocking-media-loader", "skippable-video-overlay", "room-script-runner"}
    assert "normalized-room-animation" in all_ids


def test_timed_media_components_have_safe_code_snippets():
    mem = GameMemory()
    mem.ensure()
    records = {i.id: i for i in mem.load_components()}
    required = {
        "timed-window-qte-manager",
        "nonblocking-media-loader",
        "skippable-video-overlay",
        "normalized-room-animation",
        "room-script-runner",
    }
    assert required <= set(records)
    loader = records["nonblocking-media-loader"].recipe["code"]
    assert "requestAnimationFrame(frame)" in loader
    assert "loadAssets();" in loader
    assert "loadAssets().then" not in loader
    video = records["skippable-video-overlay"].recipe["code"]
    assert "onended" in video and "onerror" in video
    assert "finishVideo" in video
    timed = records["timed-window-qte-manager"].recipe["code"]
    assert "inputFlash" in timed and "lastInputCode" in timed and "openNow" in timed
    norm = records["normalized-room-animation"].recipe["code"]
    assert "heroBoxes" in norm and "hazardPath" in norm


def test_qte_playbook_bullets_are_code_stage_retrievable():
    for bullet_id in (
        "qte-timed-input-window",
        "cutscene-frame-cycling",
        "scripted-scene-state-machine",
    ):
        rec = _jsonl_record("memory/playbook.jsonl", "id", bullet_id)
        assert rec["harmful"] == 0
        assert rec["helpful"] >= 1
    qte = _jsonl_record("memory/playbook.jsonl", "id", "qte-timed-input-window")
    assert "openNow" in qte["content"]
    assert "inputFlash" in qte["content"]
    scene = _jsonl_record("memory/playbook.jsonl", "id", "scripted-scene-state-machine")
    assert "ZERO-INDEXED" in scene["content"]
    assert "scene>=1" in scene["content"]
    load = _jsonl_record("memory/playbook.jsonl", "id", "sprite-gen-wait-for-load")
    assert "Start requestAnimationFrame(frame) immediately" in load["content"]
    assert "Promise.all" in load["content"]


def test_qte_outline_and_visual_recipe_reject_procedural_backdrops():
    outline = _jsonl_record(
        "memory/implementation_outlines.jsonl", "id", "outline-cutscene-qte"
    )
    assert outline["verified"] is True
    assert "Draw generated bg_* PNGs first with drawImage" in outline["content"]
    assert "scene>=1" in " ".join(outline["recipe"]["traps"])
    visual = _jsonl_record(
        "memory/visual_playtests.jsonl", "id", "canvas-cutscene-qte"
    )
    assert visual["verified"] is True
    checks = " ".join(visual["recipe"]["checklist"])
    assert "procedural rectangles" in checks
    assert "pink/MISSING" in checks
    assert "auto_cutscene_restart_zero_index" in json.dumps(visual["recipe"])


def test_qte_skeleton_uses_harness_visible_zero_index_contract():
    skel = (ROOT / "memory/skeletons/canvas_cutscene_qte_basic.html").read_text()
    for term in (
        "state.scene is ZERO-indexed",
        "qteOpenNow",
        "state.inputFlash = 150",
        "window.game = { reset }",
        "MISSING GENERATED BACKGROUND",
        "scoreEl.textContent",
    ):
        assert term in skel


def test_qte_plan_instruction_includes_mechanism_nudge():
    assert prompts_v1._detect_qte_intent("quick time reaction scene")
    out = prompts_v1.plan_instruction(
        goal="Build a quick-time reaction cutscene game with videos"
    )
    assert "TIMED-REACTION / QTE INTENT DETECTED" in out
    assert "inputFlash" in out
    assert "scene>=1" in out
    assert "onended/onerror" in out


def test_generated_assets_block_starts_raf_immediately(tmp_path):
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    asset_dir = tmp_path / "game_assets"
    asset_dir.mkdir()
    sprite = asset_dir / "knight_idle.png"
    sprite.write_bytes(b"fake")
    block = render_asset_paths_block({"knight_idle": sprite}, html)
    assert "Start RAF IMMEDIATELY" in block
    assert "requestAnimationFrame(frame);" in block
    assert "loadAssets();" in block
    assert "loadAssets().then" not in block


def test_small_system_prompt_keeps_explicit_media_tags():
    sys_prompt = prompts_v1.build_system_prompt(
        "Build a quick-time cutscene game with generated assets, sounds, and videos",
        model_class="small",
    )
    assert "<assets>" in sys_prompt
    assert "<sounds>" in sys_prompt
    assert "<videos>" in sys_prompt


def test_probe_lint_flags_fragile_scene_ge_one():
    findings = GameAgent._lint_probes([
        {"name": "hud_visible", "expr": "window.state && state.scene >= 1"},
    ])
    assert any(f["kind"] == "fragile_initial_scene_index" for f in findings)


def test_chat_feedback_ledger_is_write_honest():
    src = inspect.getsource(chat_module.CodingBoxApp._handle_event)
    assert '"wrote"' in src
    assert 'startswith("no usable code:")' in src
    assert '"no_usable"' in src
    status_src = Path(chat_module.__file__).read_text()
    assert "wrote code" in status_src
    assert "no usable code" in status_src
    paths_src = inspect.getsource(chat_module.CodingBoxApp.action_show_log_paths)
    assert "self._best_path.exists()" in paths_src
    assert "none saved" in paths_src


def test_agent_loop_hardening_hooks_are_wired():
    src = inspect.getsource(GameAgent)
    for term in (
        "PATCH SURGERY MODE",
        "USER FEEDBACK OVERRIDES STALE BLOCKER",
        "visual_regression_snapshot_revert",
        "<html_file> rejected: full rewrite on a failing",
        "FACTUAL LAST REPORT",
    ):
        assert term in src


def test_procedural_regression_reports_likely_source_sites():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "likely_source_sites" in src
    assert "Likely source site(s)" in src
    assert "function drawbg" in src.lower()

