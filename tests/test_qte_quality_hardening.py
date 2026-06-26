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
    assert rec["expect"]["visual_recipe"] == "canvas-cutscene-qte"
    assert rec["expect"]["outline"] == "outline-cutscene-qte"
    # Visual-first; 8-scene arcade list is a default template, goal can override count/style.
    assert "VISUAL RULE" in prompt
    assert "USER GOAL WINS" in prompt
    assert "DEFAULT 8-SCENE TEMPLATE" in prompt
    assert "10 scenes" in prompt  # explicit flexibility example for the model
    assert "cartoon" in prompt.lower()
    assert "bg_drawbridge" in prompt and "bg_dragons_lair" in prompt
    for term in (
        "drawImage(bg",
        "0-index state.scene",
        "onended/onerror",
        "requestAnimationFrame(frame) immediately",
        "pipeline runs up front",
        "<assets>",
        "<sounds>",
        "<videos>",
        "SCENES array",
        "bg_*",
        "knight_*",
        "key_*",
    ):
        assert term in prompt, f"missing {term!r}"
    assert ("cutscene videos" in prompt or "I2V videos" in prompt) and "intro" in prompt and "victory" in prompt
    assert "18 video" not in prompt
    assert "hazardPath:{from:{x,y}" not in prompt


def test_dragons_lair_outline_lists_eight_canonical_scenes():
    outline = _jsonl_record(
        "memory/implementation_outlines.jsonl", "id", "outline-cutscene-qte"
    )
    scenes = outline["recipe"]["dragons_lair_scenes"]
    assert len(scenes) == 8  # reference template when goal is silent
    assert scenes[0]["bg"] == "bg_drawbridge"
    assert scenes[-1]["bg"] == "bg_dragons_lair"
    assert sum(len(s["steps"]) for s in scenes) == 14  # 2 single-beat + 6 dual-beat scenes
    note = outline["recipe"].get("dragons_lair_scenes_note", "")
    assert "user goal" in note.lower() or "follow the user goal" in note.lower()


def test_dragons_lair_eight_scenes_playbook_exists():
    rec = _jsonl_record("memory/playbook.jsonl", "id", "dragons-lair-eight-scenes")
    assert "default template" in rec["content"].lower()
    assert "cartoon" in rec["content"].lower()
    assert "flexible" in rec["tags"] or "override" in rec["tags"]
    assert rec["harmful"] == 0


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
    flex = (visual.get("content") or "").lower()
    assert "procedural rectangles" in checks
    assert "pink/MISSING" in checks
    assert "auto_cutscene_restart_zero_index" in json.dumps(visual["recipe"])
    assert "default 8-scene template" in flex or "user's scene count" in flex


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
    assert "MEMORY vs YOUR GOAL" in out


def test_user_goal_wins_playbook_retrieves_on_generic_goals():
    from memory import Playbook
    pb = Playbook(base_root=str(ROOT / "memory"))
    goal = "build a game with 10 cartoon levels and custom characters"
    hits = pb.retrieve(goal, stage="plan", k=12)
    ids = {h.bullet.id for h in hits}
    assert "user-goal-wins-over-templates" in ids


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


# ----------------------------------------------------------------------
# Timing verification (QTE timing via memory, 2026-06-14).
# An input pressed OUTSIDE its window must not score as a hit — the
# "react at the right moment" guarantee, encoded as data not code.
# ----------------------------------------------------------------------


def test_qte_recipe_has_window_gating_probe():
    """canvas-cutscene-qte must carry the deterministic out-of-window
    rejection probe, reading the windowOpen/expectKey/result contract."""
    visual = _jsonl_record(
        "memory/visual_playtests.jsonl", "id", "canvas-cutscene-qte"
    )
    probes = visual["recipe"].get("auto_probes") or []
    gate = [p for p in probes if p.get("name") == "auto_qte_input_gated_by_window"]
    assert gate, "missing auto_qte_input_gated_by_window probe"
    expr = gate[0]["expr"]
    for field in ("expectKey", "windowOpen", "result", "sceneEnteredAt"):
        assert field in expr, f"probe expr missing {field}"
    # It must dispatch the expected key and only fail on a closed-window hit.
    assert "KeyboardEvent" in expr
    assert "succ" in expr


def test_qte_gating_probe_is_injected_for_dragons_lair_goal():
    """The matcher resolves a QTE goal to canvas-cutscene-qte and the
    gating probe is present so the harness will run it."""
    mem = GameMemory(root=str(ROOT / "memory"))
    recipe, _diag = mem.find_visual_playtest_for(
        goal="dragons lair laserdisc quick-time reaction cutscene duck jump sword"
    )
    assert recipe is not None
    assert recipe.id == "canvas-cutscene-qte"
    names = [p.get("name") for p in (recipe.recipe.get("auto_probes") or [])]
    assert "auto_qte_input_gated_by_window" in names


def test_qte_playbook_states_verification_contract():
    qte = _jsonl_record("memory/playbook.jsonl", "id", "qte-timed-input-window")
    assert "windowOpen" in qte["content"]
    assert "result" in qte["content"]
    # Generalized beyond Dragon's Lair to all timed-input mechanics.
    for tag in ("rhythm", "parry", "timing", "reaction-window"):
        assert tag in qte["tags"], f"missing tag {tag}"


def test_qte_outline_probes_out_of_window_rejection():
    outline = _jsonl_record(
        "memory/implementation_outlines.jsonl", "id", "outline-cutscene-qte"
    )
    probes = " ".join(outline["recipe"].get("probes") or [])
    assert "OUT of window" in probes
    assert "no success" in probes

