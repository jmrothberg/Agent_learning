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


# ---------------------------------------------------------------------------
# Todo-driven execution (2026-06-12): parse → select → CURRENT TASK contract
# ---------------------------------------------------------------------------


def test_parse_todo_items_handles_marker_dialects():
    text = (
        "[ ] plain open\n"
        "- [x] dash done\n"
        "* [X] star done upper\n"
        "header line without checkbox\n"
        "  - [ ] indented open\n"
    )
    items = GameAgent._parse_todo_items(text)
    assert items == [
        (False, "plain open"),
        (True, "dash done"),
        (True, "star done upper"),
        (False, "indented open"),
    ]


def test_parse_todo_items_empty_and_no_markers():
    assert GameAgent._parse_todo_items("") == []
    assert GameAgent._parse_todo_items("just prose\nno boxes") == []


def _todo_stub() -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._todos_items = []
    a._todo_nag_counts = {}
    a._current_todo = None
    a._trace = lambda obj: None
    return a


def test_select_next_todo_skips_done_and_respects_nag_cap():
    a = _todo_stub()
    a._todos_items = [(True, "done item"), (False, "open one"), (False, "open two")]
    assert a._select_next_todo() == "open one"
    # Two nags on "open one" → fall through to the next open item.
    a._todo_nag_counts[GameAgent._norm_todo("open one")] = 2
    assert a._select_next_todo() == "open two"
    a._todo_nag_counts[GameAgent._norm_todo("open two")] = 2
    assert a._select_next_todo() is None


def test_capture_todos_updates_structured_items(tmp_path):
    a = _todo_stub()
    a.trace_path = tmp_path / "t.jsonl"
    a._session_id = "sess"
    a._capture_todos("<todos>\n[ ] wire input\n[x] scaffold\n</todos>")
    assert a._todos_items == [(False, "wire input"), (True, "scaffold")]


class _PromptStub:
    @staticmethod
    def post_clean_instruction(report_text):
        return "POST_CLEAN_BASE"


def _fix_prompt_stub() -> GameAgent:
    """Minimum-viable agent for the ok-branch of _build_fix_prompt."""
    a = _todo_stub()
    a._force_question_subsystem = None
    a._scoped_change_active = False
    a._format_report_for_model = lambda r: "REPORT"
    a._p = _PromptStub()
    a._iters_remaining = 3
    a._pending_feedback = []
    a._pending_answer = None
    a._internal_feedback_texts = set()
    a._user_force_done = False
    a._current_file = "<html>game</html>"
    # Polish branch attrs — capped out so tests exercise the plain
    # post-clean path when no todo contract fires.
    a._polish_turns_used = 99
    return a


def test_clean_iter_with_open_todos_injects_current_task_contract():
    a = _fix_prompt_stub()
    a._todos_items = [(True, "scaffold"), (False, "wire space-bar to fire")]
    traced = []
    a._trace = lambda obj: traced.append(obj)
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" in prompt
    assert "wire space-bar to fire" in prompt
    # Truth-source file inlined so <patch> SEARCH can anchor.
    assert "<html>game</html>" in prompt
    assert a._current_todo == "wire space-bar to fire"
    kinds = [t.get("kind") for t in traced]
    assert "todo_contract_injected" in kinds


def test_clean_iter_with_pending_feedback_skips_todo_contract():
    a = _fix_prompt_stub()
    a._todos_items = [(False, "wire space-bar to fire")]
    a._pending_feedback = ["make the ship blue"]
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" not in prompt
    assert a._current_todo is None


def test_contract_fires_over_internal_only_feedback():
    """2026-06-12 (trace 20260612_132314): at every clean iter an
    [AUTONOMOUS PLAYTEST] finding (agent-generated) was pending, so the
    CURRENT TASK contract never fired all session. Internal-only pending
    feedback must NOT block the contract — the finding still flows into
    the same turn via _flush_user_injections."""
    a = _fix_prompt_stub()
    a._todos_items = [(False, "wire space-bar to fire")]
    finding = "[AUTONOMOUS PLAYTEST] timer never advances in intro phase"
    a._pending_feedback = [finding]
    a._internal_feedback_texts = {finding}
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" in prompt
    assert a._current_todo == "wire space-bar to fire"


def test_contract_blocked_by_genuine_user_feedback_mixed_with_internal():
    """A REAL user ask in the same batch still wins the turn."""
    a = _fix_prompt_stub()
    a._todos_items = [(False, "wire space-bar to fire")]
    finding = "[AUTONOMOUS PLAYTEST] timer never advances"
    a._pending_feedback = [finding, "make the ship blue"]
    a._internal_feedback_texts = {finding}
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" not in prompt
    assert a._current_todo is None


def test_contract_blocked_by_pending_answer():
    """A queued user ANSWER is genuine input — contract must wait."""
    a = _fix_prompt_stub()
    a._todos_items = [(False, "wire space-bar to fire")]
    a._pending_answer = "yes, rewrite the input system"
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" not in prompt


def test_clean_iter_with_all_done_todos_skips_contract():
    a = _fix_prompt_stub()
    a._todos_items = [(True, "scaffold"), (True, "wire input")]
    # Polish branch needs these attrs to decide; cap it out so the
    # plain post-clean path is taken.
    a._polish_turns_used = 99
    prompt = a._build_fix_prompt(
        report={"ok": True}, regressed=False, partial_failed=[],
    )
    assert "CURRENT TASK" not in prompt


def test_todo_drift_fires_when_task_left_unchecked():
    a = _todo_stub()
    traced = []
    a._trace = lambda obj: traced.append(obj)
    a._current_todo = "wire space-bar to fire"
    a._todos_items = [(False, "wire space-bar to fire")]
    a._todo_drift_check()
    assert any(t.get("kind") == "todo_drift" for t in traced)
    assert a._current_todo is None


def test_todo_drift_silent_when_task_checked_off():
    a = _todo_stub()
    traced = []
    a._trace = lambda obj: traced.append(obj)
    a._current_todo = "wire space-bar to fire"
    a._todos_items = [(True, "wire space-bar to fire")]
    a._todo_drift_check()
    assert not any(t.get("kind") == "todo_drift" for t in traced)
    assert a._current_todo is None


def test_todos_minimal_spec_in_small_class():
    """2026-06-12: <todos> is RE-ENABLED for the small class via the
    minimal TODOS_FORMAT_SMALL spec (todo-driven CURRENT TASK turns help
    small models the most). Pin presence AND that the small prompt uses
    the terse guideline, not the full one (size budget lives in
    test_prompt_size.py)."""
    import prompts_v1

    sp_small = prompts_v1.build_system_prompt("test goal", model_class="small")
    assert "<todos>" in sp_small
    # The full TODOS_FORMAT guideline mentions compaction replay; the
    # minimal one must not (it's the size-budget marker).
    assert "replays it across compaction" not in sp_small
    # Still present in the default (large) prompt with the full guideline.
    sp_large = prompts_v1.build_system_prompt("test goal", model_class="large")
    assert "<todos>" in sp_large
    assert "replays it across compaction" in sp_large
