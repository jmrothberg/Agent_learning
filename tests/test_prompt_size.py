"""Prompt-size guards for the small-model trim path.

Stop-Losing-To-OneShot Track C: when the agent runs against a coder-class
mid-size local LLM (qwen2.5-coder-32B, deepseek-coder-33B), it builds the
system prompt with `model_class="small"`, which drops <assets>, <sounds>,
<lookup_bullet>, <anti-patterns>, <reasoning-license>, <user-presence>
and collapses <workflow> + <iteration-policy>. The target is ≤ 6 KB so
the model spends its capacity on the game, not the schema.

These tests stay pure-function (no model, no Chromium) and run in well
under a second.
"""
from __future__ import annotations

from prompts_v1 import build_system_prompt


def test_small_model_prompt_under_six_kilobytes():
    p = build_system_prompt("snake game", model_class="small")
    assert len(p) <= 6_000, f"small-model prompt {len(p)} chars exceeds 6 KB target"


def test_small_model_prompt_drops_optional_tags():
    p = build_system_prompt("snake game", model_class="small")
    for missing in ("<assets>", "<sounds>", "<lookup_bullet>"):
        assert missing not in p, f"{missing} should be dropped in small-model prompt"
    assert "<reasoning-license>" not in p
    assert "<user-presence>" not in p
    assert "<anti-patterns>" not in p


def test_small_model_prompt_keeps_core_tags():
    p = build_system_prompt("snake game", model_class="small")
    for required in ("<plan>", "<criteria>", "<probes>", "<html_file>",
                     "<patch>", "<diagnose>", "<done/>"):
        assert required in p, f"{required} missing from small-model prompt"


def test_large_model_prompt_unchanged():
    p = build_system_prompt("snake game", model_class="large")
    for required in ("<assets>", "<sounds>", "<lookup_bullet>",
                     "<reasoning-license>", "<user-presence>",
                     "<anti-patterns>"):
        assert required in p, f"{required} missing from large-model prompt"


def test_mid_model_prompt_keeps_assets_drops_anti_patterns():
    p = build_system_prompt("snake game", model_class="mid")
    assert "<assets>" in p
    assert "<sounds>" in p
    assert "<anti-patterns>" not in p
