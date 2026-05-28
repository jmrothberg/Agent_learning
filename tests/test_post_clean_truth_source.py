"""Post-clean fix-prompt truth-source inject.

Evidence: fighing-game trace 20260519_153115 iter 3→4. Iter 3 was clean,
the user gave feedback ("fire ball needs to travel further"), and the
agent built iter 4's user message via the post_clean branch of
`_build_fix_prompt`. The current `post_clean_instruction` does NOT inline
the file on disk, so the model patched against a remembered/imagined
shape of `drawFighter`. Its SEARCH did not match disk; 1/2 patches
applied; the iter failed with `partial patch apply`. The conversation.md
literally says: "I need to check the actual current file ... I will
assume the draw logic already handles flipping".

Fix: when `report["ok"]` AND there is queued user feedback (or a pending
question answer), append a `CURRENT FILE ON DISK` block to the post_clean
instruction. Same pattern continuation_instruction / fix_instruction
already use. Below a generous file-size cap so we do not blow context.

These tests are pure-state — no model, no Chromium, no disk model load.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    fake_browser = MagicMock()
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _clean_report() -> dict:
    """Minimal `report["ok"]=True` payload for the post_clean branch."""
    return {
        "ok": True,
        "errors": [],
        "console_errors": [],
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


# ---------- the key change ------------------------------------------------


def test_post_clean_with_pending_feedback_inlines_current_file(tmp_path):
    """Iter N-1 was clean and the user has queued feedback for iter N.
    The fix prompt MUST include `CURRENT FILE ON DISK` so the next
    <patch>'s SEARCH anchors against disk truth, not against memory."""
    a = _make_agent(tmp_path)
    # Distinctive marker so we can assert the file body landed in the
    # prompt body and not just the directive sentence.
    a._current_file = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>"
        "function drawFighter(f) { /* SENTINEL_FUNC */ }"
        "</script></body></html>"
    )
    a._pending_feedback.append("fire ball needs to travel further")

    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )

    # Truth-source block present.
    assert "CURRENT FILE ON DISK" in prompt
    # File body landed in the prompt (real bytes, not just the directive).
    assert "SENTINEL_FUNC" in prompt
    # Post-clean text still present (we APPEND, not replace).
    assert "<done/>" in prompt or "STRONGLY prefer" in prompt


def test_post_clean_without_feedback_stays_lean(tmp_path):
    """No queued feedback → no truth-source inject. The post_clean text
    is meant to encourage <done/>; we do not want to bloat it with the
    full file when the model is just being told 'you may ship'."""
    a = _make_agent(tmp_path)
    a._current_file = (
        "<!DOCTYPE html><html><body><canvas id='c'></canvas><script>"
        "function drawFighter(f) { /* SENTINEL_FUNC */ }"
        "</script></body></html>"
    )
    # No pending_feedback, no pending_answer.

    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )

    # No truth-source inject.
    assert "CURRENT FILE ON DISK" not in prompt
    assert "SENTINEL_FUNC" not in prompt


def test_post_clean_huge_file_skips_inject(tmp_path):
    """Files larger than the inject cap fall back to the lean post_clean
    instruction. Truth-source inject is intended for the common case
    (single-file games are ≤ ~30-50 KB); huge files should not balloon
    the prompt by another 100 KB."""
    a = _make_agent(tmp_path)
    # Size > 60_000 (the cap chosen in agent.py); content distinctive.
    a._current_file = (
        "<!DOCTYPE html><html><body><script>// HUGE_FILE_SENTINEL\n"
        + ("const v = 1;\n" * 7000)
        + "</script></body></html>"
    )
    assert len(a._current_file) > 60_000
    a._pending_feedback.append("change something")

    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )

    assert "CURRENT FILE ON DISK" not in prompt
    assert "HUGE_FILE_SENTINEL" not in prompt


def test_post_clean_pending_answer_also_triggers_inject(tmp_path):
    """A pending answer to a model `<question>` is also a high-likelihood
    next-turn-emits-<patch> signal, so the truth source should inject."""
    a = _make_agent(tmp_path)
    a._current_file = (
        "<!DOCTYPE html><html><body><script>"
        "/* SENTINEL_PENDING_ANSWER */"
        "</script></body></html>"
    )
    a._pending_answer = "yes, use 32x32"

    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )

    assert "CURRENT FILE ON DISK" in prompt
    assert "SENTINEL_PENDING_ANSWER" in prompt


def test_post_clean_failed_branch_unchanged(tmp_path):
    """Regression guard: this change must only affect the post_clean
    branch. A failed report still goes through fix_instruction / the
    existing focused-slice + full-file inject path."""
    a = _make_agent(tmp_path)
    a._current_file = (
        "<!DOCTYPE html><html><body><script>"
        "function drawFighter(f) { /* FAILED_BRANCH_SENTINEL */ }"
        "</script></body></html>"
    )
    a._pending_feedback.append("anything")

    failed_report = _clean_report()
    failed_report["ok"] = False
    failed_report["errors"] = ["something blew up"]

    prompt = a._build_fix_prompt(
        report=failed_report, regressed=False, partial_failed=[],
    )

    # The existing fix_instruction also inlines the file, but with its
    # own header — confirm we did not double-inject our new block.
    # (Our new block uses the literal phrase "if you emit a <patch>".)
    assert "if you emit a <patch>" not in prompt
    # The failed branch still gets the file body via fix_instruction.
    assert "FAILED_BRANCH_SENTINEL" in prompt
