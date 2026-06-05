"""Token-aware compaction (_prune_messages).

Trace `short-and-done-first-the-promp_20260529` hit 3 lossy compactions while
using a fraction of the context window, because `_prune_messages` compacted on
MESSAGE COUNT (>14) and ignored `num_ctx`. On a 200k-ctx local model that
throws away the playbook and earlier user-feedback items needlessly (the CPU
request vanished between iters). Fix: only do the lossy structured-summary
compaction when the last coder prompt used >= _COMPACT_PRESSURE of num_ctx, or
as a hard message-count safety cap when token stats are missing.

These exercise `_prune_messages` directly with a lightweight stub `self`
(it touches only a handful of attributes), so no model/browser is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402


class _Stub:
    """Minimal stand-in exposing exactly what _prune_messages reads."""

    def __init__(self, n_messages, pressure):
        # index 0 = system; rest alternate user/assistant
        self._messages = [{"role": "system", "content": "SYS"}]
        for i in range(1, n_messages):
            role = "assistant" if i % 2 == 0 else "user"
            self._messages.append({"role": role, "content": f"msg{i} body"})
        self._last_prompt_pressure = pressure
        self._last_prompt_tokens = int(pressure * 100000)
        self.num_ctx = 100000
        self.traced = []

    def _build_structured_summary(self):
        return "GOAL: x\nPROGRESS: y"

    def _summarize_content(self, c):
        return c  # identity — exercise the elision path without mutation

    def _trace(self, obj):
        self.traced.append(obj)


def _has_anchor(stub):
    return any("STATE ANCHOR" in (m.get("content") or "") for m in stub._messages)


def test_compaction_constants_sane():
    assert 0.5 < agent._COMPACT_PRESSURE < 0.95
    assert agent._COMPACT_MESSAGE_CAP > agent._STRUCTURED_PRUNE_THRESHOLD


def test_low_pressure_keeps_full_history():
    # 30 messages but only 30% of the window used → NO lossy compaction.
    stub = _Stub(n_messages=30, pressure=0.30)
    agent.GameAgent._prune_messages(stub)
    assert not _has_anchor(stub), "must NOT compact under low token pressure"
    assert len(stub._messages) == 30, "history length preserved (elision only)"
    assert not any(t.get("kind") == "structured_compaction" for t in stub.traced)


def test_high_pressure_triggers_structured_compaction():
    stub = _Stub(n_messages=30, pressure=0.80)
    agent.GameAgent._prune_messages(stub)
    assert _has_anchor(stub), "must compact when window is under pressure"
    assert len(stub._messages) < 30
    evt = [t for t in stub.traced if t.get("kind") == "structured_compaction"]
    assert evt and evt[0]["reason"] == "token_pressure"


def test_count_cap_fallback_when_no_token_stats():
    # pressure unknown (0.0) but message count blows past the safety cap.
    stub = _Stub(n_messages=agent._COMPACT_MESSAGE_CAP + 5, pressure=0.0)
    agent.GameAgent._prune_messages(stub)
    assert _has_anchor(stub)
    evt = [t for t in stub.traced if t.get("kind") == "structured_compaction"]
    assert evt and evt[0]["reason"] == "count_cap"


def test_tiny_history_is_noop():
    stub = _Stub(n_messages=3, pressure=0.99)
    agent.GameAgent._prune_messages(stub)
    assert not _has_anchor(stub)
    assert len(stub._messages) == 3
