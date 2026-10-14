"""Lean system-prompt routing for LOCAL models (2026-06-13).

A local VLM like qwen3.6:27b classifies as `mid`, which used to render the
~20KB system prompt and bury the model. Lean mode renders the compact `small`
schema for local backends; maintainer docs are no longer injected. SOTA/large/cloud
keep the full prompt.

Pure-function tests: no model or Chromium calls.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path, *, backend_name: str, model: str) -> GameAgent:
    be = MagicMock()
    be.info.name = backend_name
    be.info.model = model
    out = tmp_path / "g.html"
    out.write_text("<html></html>")
    return GameAgent(model=model, out_path=out, browser=MagicMock(), max_iters=2, backend=be)


def test_local_mid_vlm_routes_to_lean_small(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    assert a._model_class == "mid"
    assert a._lean_prompt_active() is True
    assert a._system_prompt_class() == "small"


def test_local_ollama_mid_routes_to_lean(tmp_path):
    a = _agent(tmp_path, backend_name="ollama", model="qwen3.6:27b")
    assert a._lean_prompt_active() is True
    assert a._system_prompt_class() == "small"


def test_cloud_large_keeps_full_prompt(tmp_path):
    a = _agent(tmp_path, backend_name="anthropic", model="claude-opus-4-8")
    assert a._model_class == "large"
    assert a._lean_prompt_active() is False
    assert a._system_prompt_class() == "large"


def test_explicit_off_overrides_local_auto(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    a.set_lean_prompt(False)
    assert a._lean_prompt_active() is False
    assert a._system_prompt_class() == "mid"


def test_explicit_on_forces_lean_even_on_cloud(tmp_path):
    a = _agent(tmp_path, backend_name="anthropic", model="claude-opus-4-8")
    a.set_lean_prompt(True)
    # large tier never downshifts the schema, but the flag is honored as
    # "active"; only `mid` maps to small. Large stays large by design.
    assert a._lean_prompt_active() is True
    assert a._system_prompt_class() == "large"


def test_lean_media_goal_keeps_media_tags(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    goal = (
        "Make a Dragon's-Lair QTE game with generated cel-animation sprites, "
        "sounds, music, and cutscene videos"
    )
    sp = prompts_v1.build_system_prompt(goal, model_class=a._system_prompt_class())
    assert "<assets>" in sp and "<sounds>" in sp and "<videos>" in sp


def test_qwen38_lean_prompt_starts_raf_before_asset_decode(tmp_path):
    """qwen3.8-27b is a local mid model, so it renders the small schema.
    That schema used to say only `await img.decode()`, which blocked the
    frame loop. The prompt the model actually sees must start RAF first,
    draw PNGs with sprite()/drawImage, unlock audio on a gesture, and
    keep a video from stalling on onerror."""
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.8-27b")
    assert a._model_class == "mid"
    assert a._system_prompt_class() == "small"
    goal = (
        "first-person maze with three.js, generated sprites, laser sounds, "
        "and an intro cutscene video"
    )
    sp = prompts_v1.build_system_prompt(goal, model_class=a._system_prompt_class())
    assert "requestAnimationFrame before loadAssets()" in sp
    assert "do not await it before the loop" in sp
    assert "TextureLoader" in sp
    assert "AudioContext.resume" in sp
    assert "onended AND onerror" in sp
    assert "exact MP4" in sp
    plan = prompts_v1.plan_instruction(goal=goal, model_class="small")
    assert "babylon" not in plan.lower()
    assert "three@0.160" in plan
    assert "TextureLoader" in plan
    assert "camera.rotation" in plan
    assert "AudioContext.resume" in plan
    assert "await img.decode()" not in plan or "inside that" in plan


def test_lean_memory_budget_drops_lowest_priority(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    opening = "O" * 1700
    components = "C" * 2200
    playbook = "P" * 1500
    ob, cb, pb = a._apply_lean_memory_budget(opening, components, playbook)
    # opening + components fit under 4500; playbook is dropped.
    assert ob == opening
    assert cb == components
    assert pb == ""


def test_lean_memory_budget_keeps_higher_priority_when_components_huge(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    opening = "O" * 1700
    components = "C" * 3200  # 1700+3200 > 4500 -> components dropped
    playbook = "P" * 1000
    ob, cb, pb = a._apply_lean_memory_budget(opening, components, playbook)
    assert ob == opening
    assert cb == ""           # higher-priority dropped -> lower also dropped
    assert pb == ""           # strict priority: no lower block survives a drop


def test_lean_memory_budget_noop_for_cloud(tmp_path):
    a = _agent(tmp_path, backend_name="anthropic", model="claude-opus-4-8")
    opening = "O" * 4000
    components = "C" * 4000
    playbook = "P" * 4000
    ob, cb, pb = a._apply_lean_memory_budget(opening, components, playbook)
    assert (ob, cb, pb) == (opening, components, playbook)


def test_lean_memory_budget_protect_playbook_survives_opening_fill(tmp_path):
    """Mirrors run_10: opening ~4064 chars leaves no room for playbook unless
    protect_playbook keeps the pinned sprite-wiring bullet."""
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    budget = GameAgent._LEAN_MEMORY_COMBINED_BUDGET
    opening = "O" * (budget - 100)
    components = "C" * 1800
    playbook = "P" * 900
    ob, cb, pb = a._apply_lean_memory_budget(
        opening, components, playbook, protect_playbook=True,
    )
    assert ob == opening
    assert pb == playbook
    assert cb == ""


def test_qwen36_27b_is_detected_as_vlm():
    """The user runs qwen3.6:27b specifically for its vision — so /allroles'
    per-iter visual critic must actually engage. Confirm the modality
    classifier recognizes the family (both `-` and `:` quant spellings)."""
    from backend import classify_model_modality
    assert classify_model_modality("qwen3.6-27b") == "vlm"
    assert classify_model_modality("qwen3.6:27b") == "vlm"
    assert classify_model_modality("mlx-community/Qwen3.6-27B-mxfp8") == "vlm"
    assert classify_model_modality("mlx-community/Qwen3.8-27B-mxfp8") == "vlm"
    # A plain coder model is text-only -> critic correctly skipped.
    assert classify_model_modality("qwen2.5-coder-32b-instruct") != "vlm"


def test_allroles_help_no_longer_claims_separate_architect_pass():
    import tui_help
    arch = "\n".join(tui_help.help_topic_lines("architect") or [])
    assert "does NOT add a second planning generation" in arch
    allroles = "\n".join(tui_help.help_topic_lines("allroles") or []).lower()
    assert "/critic" in allroles
    assert "vlm-critique" in allroles or "vision" in allroles
    assert "architect" in allroles

