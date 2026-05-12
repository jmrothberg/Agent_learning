"""Tests for the pi-mono-style structured compaction in agent.py.

We don't run a real model — we instantiate GameAgent with a stub browser
and exercise _build_structured_summary / _prune_messages directly with
synthetic state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _PRUNE_KEEP_RECENT_TURNS, _STRUCTURED_PRUNE_THRESHOLD  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    """Build a minimal GameAgent instance for state-only tests.

    We avoid touching Ollama, the browser, or the memory subsystem by
    pointing all paths at tmp_path.
    """
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    fake_browser = MagicMock()
    agent = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    return agent


def test_structured_summary_minimal(tmp_path):
    """Empty session — summary must still produce all required sections."""
    a = _make_agent(tmp_path)
    a._goal = "Build asteroids"
    summary = a._build_structured_summary()
    assert "## Goal" in summary
    assert "Build asteroids" in summary
    assert "## Progress" in summary
    assert "not yet built" in summary
    assert "## Critical context" in summary
    assert "source of truth" in summary


def test_structured_summary_includes_criteria(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "Snake"
    a._criteria = "Basic: snake moves\nEdge: wraps screen edges"
    summary = a._build_structured_summary()
    assert "## Acceptance criteria" in summary
    assert "snake moves" in summary
    assert "wraps screen edges" in summary


def test_structured_summary_progress_states(tmp_path):
    """Snapshot N > 0 with previous_report_ok True/False/None."""
    a = _make_agent(tmp_path)
    a._goal = "Pong"
    a._snapshot_n = 3

    a._previous_report_ok = True
    s = a._build_structured_summary()
    assert "iteration 3: PASSED" in s

    a._previous_report_ok = False
    s = a._build_structured_summary()
    assert "iteration 3: FAILING" in s

    a._previous_report_ok = None
    s = a._build_structured_summary()
    assert "iteration 3: status unknown" in s


def test_structured_summary_stuck_streak_visible(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "x"
    a._snapshot_n = 4
    a._previous_report_ok = False
    a._stuck_streak = 3
    s = a._build_structured_summary()
    assert "stuck-streak: 3" in s


def test_structured_summary_includes_diagnose_and_report(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "x"
    a._snapshot_n = 2
    a._previous_report_ok = False
    a._last_diagnose = "keyup handler clears the wrong slot"
    a._last_report_summary = "TEST FAILED — 1 console error: TypeError at line 42"
    s = a._build_structured_summary()
    assert "## Key decisions" in s
    assert "keyup handler" in s
    assert "## Last test report" in s
    assert "TypeError at line 42" in s


def test_prune_messages_no_op_when_short(tmp_path):
    """Below the keep-recent threshold, prune is a no-op."""
    a = _make_agent(tmp_path)
    a._messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    before = list(a._messages)
    a._prune_messages()
    assert a._messages == before


def test_prune_messages_default_elision_path(tmp_path):
    """Between KEEP+1 and STRUCTURED threshold: HTML in older turns is
    replaced with size markers; message count stays the same."""
    a = _make_agent(tmp_path)
    msgs = [{"role": "system", "content": "sys"}]
    # Build enough messages to exceed KEEP_RECENT but stay under structured.
    big_html = "<html_file>" + ("a" * 5000) + "</html_file>"
    for i in range(_PRUNE_KEEP_RECENT_TURNS + 3):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"turn {i} {big_html}"})
    a._messages = msgs
    n_before = len(a._messages)
    a._prune_messages()
    assert len(a._messages) == n_before  # elision keeps shape
    # Older turns no longer carry the inline HTML.
    older = a._messages[1:1 + (n_before - 1 - _PRUNE_KEEP_RECENT_TURNS)]
    for m in older:
        assert "[omitted:" in m["content"]
    # Most-recent K turns untouched.
    recent = a._messages[-_PRUNE_KEEP_RECENT_TURNS:]
    for m in recent:
        assert "[omitted:" not in m["content"]


def test_prune_messages_structured_path(tmp_path):
    """Above the structured threshold: messages 1..cutoff get replaced by
    ONE state-anchor user message; system + last K turns survive."""
    a = _make_agent(tmp_path)
    a._goal = "Make asteroids"
    a._snapshot_n = 5
    a._previous_report_ok = False
    a._stuck_streak = 2
    a._last_report_summary = "TEST FAILED: 2 errors"

    msgs = [{"role": "system", "content": "sys-original"}]
    n_extra = _STRUCTURED_PRUNE_THRESHOLD + 3
    for i in range(n_extra):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"turn-{i}"})
    a._messages = msgs
    a._prune_messages()

    # System message preserved at index 0.
    assert a._messages[0]["content"] == "sys-original"
    # Index 1 is the state anchor.
    anchor = a._messages[1]
    assert anchor["role"] == "user"
    assert "STATE ANCHOR" in anchor["content"]
    assert "Make asteroids" in anchor["content"]
    assert "stuck-streak: 2" in anchor["content"]
    assert "TEST FAILED: 2 errors" in anchor["content"]
    # Last K turns are the original last K.
    expected_recent = msgs[-_PRUNE_KEEP_RECENT_TURNS:]
    actual_recent = a._messages[-_PRUNE_KEEP_RECENT_TURNS:]
    assert actual_recent == expected_recent
    # Total count: system + anchor + last K = 2 + K.
    assert len(a._messages) == 2 + _PRUNE_KEEP_RECENT_TURNS


def test_structured_summary_critical_context_always_present(tmp_path):
    """Whatever the state, the 'source of truth' rule must appear so the
    model never loses the patch-against-current-file invariant."""
    a = _make_agent(tmp_path)
    s = a._build_structured_summary()
    assert "source of truth" in s
    assert "patch against" in s.lower()


# ---------------------------------------------------------------------------
# Fix #3 — one-shot <html_file> rewrite exemption after feedback drain
#
# Trace: classic-doom-style 20260512_111015 — user typed multi-issue
# feedback ("gun + mouse + powerups + demons"), model wanted holistic
# rewrite, agent rejected it on multiple consecutive extension turns
# and the session ran out of iters.
# ---------------------------------------------------------------------------


def _real_baseline_html() -> str:
    """A 3-KB syntactically-valid game file that is NOT degenerate
    (passes _is_degenerate_baseline) so the rewrite-rejection branch
    is actually exercised in the test rather than carved out by the
    skeleton-detector."""
    body = "; ".join(f"const v{i} = {i}" for i in range(120))
    return (
        "<!DOCTYPE html><html><head></head><body>"
        "<canvas id='c' width='800' height='600'></canvas>"
        f"<script>{body};\nfunction draw() {{ /* real */ }}\n"
        "requestAnimationFrame(draw);</script>"
        "</body></html>"
    )


def test_rewrite_exemption_armed_by_feedback_drain(tmp_path):
    """Feeding pending_feedback through _flush_user_injections must set
    the one-shot exemption flag."""
    a = _make_agent(tmp_path)
    assert a._allow_one_rewrite is False
    a._pending_feedback.append("fix the mouse look")
    a._flush_user_injections("next-turn message")
    assert a._allow_one_rewrite is True


def test_rewrite_exemption_not_armed_without_feedback(tmp_path):
    """Calls to _flush_user_injections without feedback must NOT
    spuriously arm the exemption — only fresh user feedback should."""
    a = _make_agent(tmp_path)
    a._flush_user_injections("regular turn message")
    assert a._allow_one_rewrite is False


def test_rewrite_exemption_consumed_after_one_attempt(tmp_path):
    """First materialization after the exemption is armed must clear
    the flag — even if the model later tries another rewrite in the
    same extension, the second attempt must be rejected."""
    import asyncio
    a = _make_agent(tmp_path)
    a._current_file = _real_baseline_html()
    a._snapshot_n = 1
    a._allow_one_rewrite = True
    # Build a minimal `<html_file>` reply.
    new_html = (
        "<!DOCTYPE html><html><body><canvas id='c' width='800' height='600'>"
        "</canvas><script>" + ("// real-code line\n" * 200) +
        "</script></body></html>"
    )
    reply = f"<html_file>\n{new_html}\n</html_file>"
    materialized, msg = asyncio.run(a._materialize(reply, dry_run=False))
    assert materialized is not None, f"first rewrite must succeed (msg={msg})"
    assert a._allow_one_rewrite is False, "flag must be consumed"
    # Second rewrite attempt in the same extension must now be rejected.
    second_html = new_html.replace("// real-code", "// second")
    reply2 = f"<html_file>\n{second_html}\n</html_file>"
    res2, msg2 = asyncio.run(a._materialize(reply2, dry_run=False))
    assert res2 is None
    assert "baseline file already exists" in msg2


def test_rewrite_rejected_when_no_feedback_drain(tmp_path):
    """Sanity: without feedback, a baseline-rewrite is still rejected
    (the prior behavior is preserved on normal iter-by-iter flow)."""
    import asyncio
    a = _make_agent(tmp_path)
    a._current_file = _real_baseline_html()
    a._snapshot_n = 1
    new_html = (
        "<!DOCTYPE html><html><body><canvas id='c' width='800' height='600'>"
        "</canvas><script>" + ("// other\n" * 200) +
        "</script></body></html>"
    )
    res, msg = asyncio.run(a._materialize(
        f"<html_file>\n{new_html}\n</html_file>", dry_run=False,
    ))
    assert res is None
    assert "baseline file already exists" in msg


# ---------------------------------------------------------------------------
# Fix C — truncation-aware fix-turn routing
#
# When the on-disk file has open <html>/<body>/<script> with no matching
# close, _build_fix_prompt must route through truncation_recovery_instruction
# and NOT inline the broken file as truth source. The classic-doom
# 20260512_153449 trace showed an iter-2 prompt of 34,137 BPE tokens —
# of which ~7K was the broken iter-1 HTML — stall MLX at 60s with zero
# output. Skipping the inline drops prompt size dramatically.
# ---------------------------------------------------------------------------


def _truncated_file() -> str:
    """A 3 KB+ file that opens <html>/<body>/<script> but never closes
    them — mirrors the classic-doom 20260512_153449 failure shape."""
    body = "\n".join([f"  const v{i} = {i};" for i in range(160)])
    return (
        "<!DOCTYPE html>\n"
        "<html lang='en'><head><title>FPS</title></head>\n"
        "<body>\n<canvas id='c' width='800' height='600'></canvas>\n"
        f"<script>(()=>{{\n{body}\n"
        # truncated: no closing })(), </script>, </body>, </html>.
    )


def test_build_fix_prompt_routes_truncation_to_recovery(tmp_path):
    """When the current file has open <script> with no close, the fix
    prompt must use truncation_recovery wording and skip the file body."""
    a = _make_agent(tmp_path)
    a._current_file = _truncated_file()
    fake_report = {
        "ok": False,
        "errors": ["<script> opened but never closed"],
        "console_errors": [],
        "page_errors": [],
        "probe_errors": [],
        "soft_warnings": [],
        "warnings": [],
        "logs": [],
        "title": "X",
        "canvas": None,
        "input_listeners": {},
        "input_test": None,
        "frozen_canvas": None,
        "body_chars": 0,
        "body_sample": "",
        "probes": [],
    }
    prompt = a._build_fix_prompt(
        report=fake_report, regressed=False, partial_failed=[],
    )
    # The truncation-recovery wording is present.
    assert "TRUNCATION DETECTED" in prompt
    # The fixture has <html>, <body>, <script> all unclosed. _truncation_reason
    # reports the outermost open tag first, which is the highest-leverage
    # recovery signal — see test_truncation_reports_outermost_first.
    assert "unclosed <html>" in prompt
    # The broken file body is NOT inlined as truth source — the
    # const-v0/v1/v159 sea would otherwise be visible.
    assert "const v0" not in prompt
    assert "const v159" not in prompt
    # The fresh-rewrite directive is present.
    assert "fresh, complete <html_file>" in prompt or "fresh complete <html_file>" in prompt


def test_build_fix_prompt_normal_path_still_inlines_file(tmp_path):
    """Regression guard: a complete (non-truncated) file must still go
    through the normal fix_instruction path with the file inlined."""
    a = _make_agent(tmp_path)
    a._current_file = _real_baseline_html()
    a._criteria = "Basic: works"
    fake_report = {
        "ok": False,
        "errors": ["console.log('something went wrong')"],
        "console_errors": ["console.log('something went wrong')"],
        "page_errors": [],
        "probe_errors": [],
        "soft_warnings": [],
        "warnings": [],
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": {"ran": False, "any_change": None, "keys_tried": []},
        "frozen_canvas": False,
        "body_chars": 100,
        "body_sample": "...",
        "probes": [],
    }
    prompt = a._build_fix_prompt(
        report=fake_report, regressed=False, partial_failed=[],
    )
    assert "TRUNCATION DETECTED" not in prompt
    # The normal fix-instruction inlines the current file somewhere.
    # _real_baseline_html includes a distinctive `function draw() { /* real */ }`
    # marker; assert it survives into the prompt.
    assert "/* real */" in prompt
