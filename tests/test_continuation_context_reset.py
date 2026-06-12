"""Tests for fresh-context continuations (2026-06-12).

When a continuation request arrives on a CLEAN game, the agent replaces
the accumulated message history with [system prompt, state anchor]
before the continuation message is appended — the frontier-agent
"fresh subagent context" pattern. Evidence: trace 20260612_004616
carried ~61K prompt tokens into its continuation turns despite 7
structured compactions; 27B-class local models degrade well before
that. Gated so mid-debugging history is never discarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _stub(n_history: int = 8) -> GameAgent:
    a = GameAgent.__new__(GameAgent)
    a._messages = [{"role": "system", "content": "SYS"}]
    for i in range(n_history):
        role = "user" if i % 2 == 0 else "assistant"
        a._messages.append({"role": role, "content": f"turn {i} " * 50})
    a._traced: list[dict] = []
    a._trace = lambda obj: a._traced.append(obj)
    a._build_structured_summary = lambda: "SUMMARY-BODY"
    return a


def test_reset_applies_on_clean_prior_session():
    a = _stub()
    before = len(a._messages)
    assert before > 3
    applied = a._maybe_reset_continuation_context(True)
    assert applied is True
    # [system, anchor] only — continuation msg is appended by run() after.
    assert len(a._messages) == 2
    assert a._messages[0]["content"] == "SYS"
    assert "fresh-context" in a._messages[1]["content"]
    assert "SUMMARY-BODY" in a._messages[1]["content"]
    kinds = [t.get("kind") for t in a._traced]
    assert "continuation_context_reset" in kinds
    ev = next(t for t in a._traced
              if t.get("kind") == "continuation_context_reset")
    assert ev["before_messages"] == before
    assert ev["after_messages"] == 2


def test_no_reset_when_prior_session_not_clean():
    """Mid-debugging history must never be discarded."""
    a = _stub()
    before = list(a._messages)
    assert a._maybe_reset_continuation_context(False) is False
    assert a._messages == before
    assert not a._traced


def test_no_reset_on_tiny_history():
    a = _stub(n_history=2)  # 3 messages total — not worth replacing
    before = list(a._messages)
    assert a._maybe_reset_continuation_context(True) is False
    assert a._messages == before


def test_no_reset_without_system_prompt_at_index_zero():
    a = _stub()
    a._messages[0] = {"role": "user", "content": "not a system prompt"}
    before = list(a._messages)
    assert a._maybe_reset_continuation_context(True) is False
    assert a._messages == before


def test_anchor_build_failure_falls_back_to_append_behavior():
    a = _stub()
    before = list(a._messages)

    def _boom():
        raise RuntimeError("anchor exploded")

    a._build_structured_summary = _boom
    assert a._maybe_reset_continuation_context(True) is False
    assert a._messages == before
    kinds = [t.get("kind") for t in a._traced]
    assert "continuation_context_reset_failed" in kinds
