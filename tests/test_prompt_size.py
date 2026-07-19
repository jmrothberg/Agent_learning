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

import json

import pytest

from agent_prompts import PromptBuildingMixin
from prompts_v1 import build_system_prompt


def test_small_model_prompt_under_six_kilobytes():
    """Target is ≤ ~6 KB so the small-model prompt stays lean. 2026-05-21
    bumped the cap from 6_000 to 6_300 to admit the new window-state
    hard-rule (Expose state on window: window.gameState = state; ...).
    Evidence: that rule prevents the single most common probe failure
    across May 20-21 traces (pac/dk/sf/doom/FPS all hit it). The +200
    chars is justified by the failure mode it eliminates.
    2026-06-12: bumped 6_300 to 6_600 to admit the minimal
    TODOS_FORMAT_SMALL spec — todo-driven CURRENT TASK turns require the
    small class to know the <todos> tag, and one-objective-per-turn is
    the biggest reliability lever for this class.
    2026-07-01: bumped 6_600 to 6_900 for window._assetsReady hard-rule
    (harness asset-settle contract; run_08 M4/P1)."""
    p = build_system_prompt("snake game", model_class="small")
    assert len(p) <= 6_900, f"small-model prompt {len(p)} chars exceeds 6.9 KB target"


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


# ---------------------------------------------------------------------------
# Fix A — <html_file> markdown-fence warning. Generic across-model
# protection against the failure shape from classic-doom-style
# 20260512_153449 where DeepSeek-V4 emitted a stray closing ``` inside
# the <html_file> body.
# ---------------------------------------------------------------------------


def test_large_model_prompt_includes_markdown_fence_warning():
    """The warning must reach reasoning-class models (where the failure
    actually showed up). The HTML_FORMAT guideline is in the always-on
    set, so it should be in every prompt size."""
    p = build_system_prompt("doom-style first-person shooter", model_class="large")
    assert "RAW HTML" in p or "raw HTML" in p, (
        "markdown-fence warning missing from large-model system prompt"
    )
    # Specific phrasing that prohibits the failure pattern:
    assert "markdown code fences" in p or "```html" in p


def test_small_model_prompt_includes_markdown_fence_warning():
    """Small models can also exhibit this if trained on markdown corpora.
    The warning must survive the small-model trim path."""
    p = build_system_prompt("snake game", model_class="small")
    assert "RAW HTML" in p or "raw HTML" in p, (
        "markdown-fence warning missing from small-model trimmed prompt"
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "All checks passed. Emit <done/> or make one clean follow-up.",
        (
            "REPLY FORMAT FOR THIS TURN\nFix the failing probe.\n\n"
            "CURRENT FILE ON DISK:\n```html\n<html></html>\n```\n"
        ),
        "POST-CLEAN REGRESSION: revert one minimal change.\n\nLast known-good follows.",
        (
            "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
            "move only the score\n"
            "================ SCOPED-CHANGE DIRECTIVE ================\n"
            "do not touch gameplay\n"
        ),
        (
            "================ TRUNCATION DETECTED ================\n"
            "Emit one complete replacement file."
        ),
    ],
    ids=["clean", "failed", "regressed", "scoped", "truncation"],
)
def test_prompt_section_collection_reconstructs_branch_bytes(prompt: str) -> None:
    sections = PromptBuildingMixin._collect_prompt_sections(prompt)
    reconstructed = "".join(section["text"] for section in sections)
    assert reconstructed.encode("utf-8") == prompt.encode("utf-8")


def test_prompt_section_collection_preserves_unmatched_prefix_and_known_blocks() -> None:
    prompt = (
        "arbitrary unmatched prefix\n\n"
        "================ USER FEEDBACK (HIGHEST PRIORITY) ================\n"
        "make it blue\n\n"
        "================ SCOPED-CHANGE DIRECTIVE ================\n"
        "only recolor it\n"
        "<opening_book>\noutline\n</opening_book>\n"
        "<components>\ncomponent\n</components>\n"
        "<playbook>\nlesson\n</playbook>\n"
        "CURRENT FILE ON DISK:\n<html></html>"
    )
    sections = PromptBuildingMixin._collect_prompt_sections(prompt)
    assert sections[0] == {
        "id": "core",
        "source": "other",
        "text": "arbitrary unmatched prefix\n\n",
    }
    assert [section["id"] for section in sections[1:]] == [
        "user_feedback",
        "scoped_change",
        "opening_book",
        "components",
        "playbook",
        "current_file",
    ]
    assert "".join(section["text"] for section in sections) == prompt


def test_prompt_section_manifest_is_bounded_text_free_and_exactly_accounted() -> None:
    marker = "================ USER FEEDBACK (HIGHEST PRIORITY) ================"
    prompt = "".join(f"{marker}\nfeedback {i}\n" for i in range(20))
    sections = PromptBuildingMixin._collect_prompt_sections(prompt)
    manifest, other_chars = PromptBuildingMixin._bounded_prompt_section_manifest(
        sections,
    )
    serialized = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")

    assert len(manifest) <= 12
    assert len(serialized) <= 1024
    assert all(set(row) == {"id", "source", "chars"} for row in manifest)
    assert all("text" not in row for row in manifest)
    assert sum(int(row["chars"]) for row in manifest) + other_chars == len(prompt)
    assert other_chars > 0
