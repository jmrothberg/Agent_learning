"""Tests for the <todos> deepagents-style artifact added to agent.py +
prompts_v1.py.

The model emits a checklist of remaining work, the harness persists it
to disk, and `_build_structured_summary` replays it across compaction
so long sessions don't lose track of "what's left to ship."
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_extract_todos_returns_none_when_absent():
    assert GameAgent._extract_todos("no tags here") is None
    assert GameAgent._extract_todos("") is None
    assert GameAgent._extract_todos("<plan>some plan</plan>") is None


def test_extract_todos_pulls_a_simple_checklist():
    reply = (
        "<plan>here's the plan</plan>\n"
        "<todos>\n"
        "[ ] wire space-bar to fire\n"
        "[x] reset lastFireTime in reset()\n"
        "[ ] add HUD score counter\n"
        "</todos>\n"
    )
    todos = GameAgent._extract_todos(reply)
    assert todos is not None
    assert "wire space-bar to fire" in todos
    assert "[x]" in todos
    assert "[ ]" in todos


def test_extract_todos_strips_whitespace():
    reply = (
        "<todos>\n"
        "    \n"
        "[ ] item one\n"
        "    \n"
        "</todos>\n"
    )
    todos = GameAgent._extract_todos(reply)
    assert todos is not None
    assert todos.startswith("[ ] item one") or "item one" in todos


def test_extract_todos_handles_dash_prefix_markdown():
    """Models often write `- [ ] item` in Markdown style; we don't
    normalize the marker but should still capture it."""
    reply = (
        "<todos>\n"
        "- [ ] item one\n"
        "- [x] item two\n"
        "</todos>\n"
    )
    todos = GameAgent._extract_todos(reply)
    assert todos is not None
    assert "item one" in todos
    assert "item two" in todos


def test_extract_todos_after_thinking_block():
    """Reasoning-model prefix should be stripped before extraction."""
    reply = (
        "<think>thinking thinking thinking</think>\n"
        "<todos>[ ] item one</todos>\n"
    )
    todos = GameAgent._extract_todos(reply)
    assert todos is not None
    assert "item one" in todos


# ---------------------------------------------------------------------------
# Format spec
# ---------------------------------------------------------------------------


def test_todos_format_registered_in_all_formats():
    import prompts_v1

    names = [f.name for f in prompts_v1.ALL_FORMATS]
    assert "<todos>" in names


def test_todos_appears_in_default_system_prompt():
    import prompts_v1

    sp = prompts_v1.build_system_prompt("test goal")
    assert "<todos>" in sp


def test_todos_dropped_from_small_class_for_size_budget():
    """The small-class prompt has a hard 6 KB cap (test_prompt_size.py).
    Adding <todos> with its guideline pushed it over the limit, so the
    tag is intentionally in _SMALL_DROP. Pin that so a refactor doesn't
    silently re-add it and break the size budget."""
    import prompts_v1

    sp_small = prompts_v1.build_system_prompt("test goal", model_class="small")
    assert "<todos>" not in sp_small
    # Still present in the default (large) prompt.
    sp_large = prompts_v1.build_system_prompt("test goal", model_class="large")
    assert "<todos>" in sp_large
