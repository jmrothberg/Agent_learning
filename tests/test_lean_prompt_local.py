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


def test_lean_system_prompt_is_small_sized(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    sp = prompts_v1.build_system_prompt("make a snake game", model_class=a._system_prompt_class())
    assert len(sp) <= 6600, f"lean system prompt too big: {len(sp)}"


def test_lean_media_goal_keeps_media_tags(tmp_path):
    a = _agent(tmp_path, backend_name="mlx", model="qwen3.6-27b")
    goal = (
        "Make a Dragon's-Lair QTE game with generated cel-animation sprites, "
        "sounds, music, and cutscene videos"
    )
    sp = prompts_v1.build_system_prompt(goal, model_class=a._system_prompt_class())
    assert "<assets>" in sp and "<sounds>" in sp and "<videos>" in sp


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


def test_qwen36_27b_is_detected_as_vlm():
    """The user runs qwen3.6:27b specifically for its vision — so /allroles'
    per-iter visual critic must actually engage. Confirm the modality
    classifier recognizes the family (both `-` and `:` quant spellings)."""
    from backend import classify_model_modality
    assert classify_model_modality("qwen3.6-27b") == "vlm"
    assert classify_model_modality("qwen3.6:27b") == "vlm"
    assert classify_model_modality("mlx-community/Qwen3.6-27B-mxfp8") == "vlm"
    # A plain coder model is text-only -> critic correctly skipped.
    assert classify_model_modality("qwen2.5-coder-32b-instruct") != "vlm"


def test_allroles_help_no_longer_claims_separate_architect_pass():
    import tui_help
    arch = "\n".join(tui_help.help_topic_lines("architect") or [])
    assert "does NOT add a second planning generation" in arch
    allroles = "\n".join(tui_help.help_topic_lines("allroles") or [])
    assert "per-iter visual critic" in allroles
    assert "VLM-capable" in allroles

