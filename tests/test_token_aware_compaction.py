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


# ---------------------------------------------------------------------------
# KV prefix-cache friendliness (Sept 2026): on oMLX / in-process MLX the
# per-turn elision is deferred so history stays append-only (cache hit);
# it runs as a batch once the projected prompt nears the ceiling.
# ---------------------------------------------------------------------------

class _Info:
    def __init__(self, name):
        self.name = name


class _Backend:
    def __init__(self, name):
        self.info = _Info(name)


def _report_stub(n_messages, backend_name, pressure=0.0):
    """Stub whose old assistant turns carry a big HTML blob so the default
    elision path has something to rewrite."""
    stub = _Stub(n_messages=n_messages, pressure=pressure)
    big = "<html_file>" + ("x" * 5000) + "</html_file>"
    for m in stub._messages[1:]:
        if m["role"] == "assistant":
            m["content"] = "reply " + big
    stub._backend = _Backend(backend_name)
    # Real _summarize_content so elision is observable.
    stub._summarize_content = lambda c: agent.GameAgent._summarize_content(stub, c)
    stub._SUMMARIZE_HTML_RE = agent.GameAgent._SUMMARIZE_HTML_RE
    stub._SUMMARIZE_FENCE_RE = agent.GameAgent._SUMMARIZE_FENCE_RE
    stub._SUMMARIZE_PROBES_RE = agent.GameAgent._SUMMARIZE_PROBES_RE
    return stub


def test_lazy_elision_defers_on_prefix_cache_backend(monkeypatch):
    monkeypatch.delenv("AGENT_PREFIX_CACHE_FRIENDLY", raising=False)
    stub = _report_stub(n_messages=12, backend_name="mlx-server")
    agent.GameAgent._prune_messages(stub)
    # Nothing rewritten → byte-identical prefix for the server's KV cache.
    assert all("HARNESS-OMITTED" not in m["content"] for m in stub._messages)
    kinds = [t.get("kind") for t in stub.traced]
    assert kinds.count("prune_deferred_prefix_cache") == 1
    # Second turn in the same streak does not re-trace (no debug spam).
    agent.GameAgent._prune_messages(stub)
    assert [t.get("kind") for t in stub.traced].count("prune_deferred_prefix_cache") == 1


def test_lazy_elision_runs_as_batch_near_ceiling(monkeypatch):
    monkeypatch.delenv("AGENT_PREFIX_CACHE_FRIENDLY", raising=False)
    stub = _report_stub(n_messages=12, backend_name="mlx-server")
    # Last real prompt already at the lazy threshold → elide now (batch).
    stub._last_prompt_tokens = int(agent._COMPACT_TOKEN_CEILING * 0.8) + 1
    stub._last_prompt_pressure = 0.3  # below structured-compaction gate
    agent.GameAgent._prune_messages(stub)
    older = stub._messages[1: len(stub._messages) - agent._PRUNE_KEEP_RECENT_TURNS]
    assert any("HARNESS-OMITTED-PRIOR-HTML" in m["content"] for m in older if m["role"] == "assistant")
    assert not _has_anchor(stub)


def test_eager_elision_kept_for_ollama_and_env_off(monkeypatch):
    monkeypatch.delenv("AGENT_PREFIX_CACHE_FRIENDLY", raising=False)
    stub = _report_stub(n_messages=12, backend_name="ollama")
    agent.GameAgent._prune_messages(stub)
    assert any("HARNESS-OMITTED-PRIOR-HTML" in m["content"] for m in stub._messages)
    # Env kill-switch restores eager elision on oMLX too.
    monkeypatch.setenv("AGENT_PREFIX_CACHE_FRIENDLY", "0")
    stub = _report_stub(n_messages=12, backend_name="mlx-server")
    agent.GameAgent._prune_messages(stub)
    assert any("HARNESS-OMITTED-PRIOR-HTML" in m["content"] for m in stub._messages)
    # Env opt-in enables deferral for Ollama.
    monkeypatch.setenv("AGENT_PREFIX_CACHE_FRIENDLY", "1")
    stub = _report_stub(n_messages=12, backend_name="ollama")
    agent.GameAgent._prune_messages(stub)
    assert all("HARNESS-OMITTED" not in m["content"] for m in stub._messages)
