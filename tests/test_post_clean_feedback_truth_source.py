"""Tests for inlining CURRENT FILE ON DISK on post-clean feedback turns
and on turns that consume a `<question>` answer.

Motivating trace:
  games/traces/make-a-first-person-shooter-ga_20260523_152317.log

Bug: when iterations passed clean and the user typed mid-session
feedback ("remove the computer-drawn circles", "shift the muzzle
flash up 75px"), the agent assembled POST-CLEAN FEEDBACK CONTRACT +
USER FEEDBACK but did NOT inline the file. After compaction, the
original <html_file> was gone from history, so the model literally
said "I genuinely do not have the file contents in my context this
turn" and emitted <question> instead of patching. When the user
answered, the file was still not inlined, so the model gave up.

This is the post-clean analogue of test_continuation_truth_source.py.
The fix path lives in agent.py:_flush_user_injections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path: Path, file_text: str = "") -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text(file_text or "<html><body></body></html>")
    a = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    a._current_file = file_text
    return a


_POST_CLEAN_BASE = (
    "ok\n\nNo errors. The game works. STRONGLY prefer ending with "
    "<done/>.\nOnly send code if you have ONE small concrete improvement."
)


def test_post_clean_feedback_inlines_current_file(tmp_path: Path) -> None:
    file_text = (
        '<!doctype html><script>'
        '/* MUZZLE_FLASH_ANCHOR */'
        'function drawGun(){ ctx.arc(0,0,5,0,Math.PI*2); }'
        '</script>'
    )
    a = _make_agent(tmp_path, file_text)
    a.add_user_feedback("remove the computer drawn circles")

    out = a._flush_user_injections(_POST_CLEAN_BASE)
    assert "USER FEEDBACK" in out
    assert "remove the computer drawn circles" in out
    # The file body must be inlined as the truth source so the model
    # can write a SEARCH/REPLACE against the real text.
    assert "CURRENT FILE ON DISK" in out
    assert "SOURCE OF TRUTH" in out
    assert "/* MUZZLE_FLASH_ANCHOR */" in out
    assert "function drawGun(){ ctx.arc(0,0,5,0,Math.PI*2); }" in out


def test_answer_to_question_inlines_current_file(tmp_path: Path) -> None:
    """When the model emits <question> on a clean turn and the user
    answers, the next turn must inline the file — otherwise the model
    is still patching blind."""
    file_text = '<!doctype html><script>const X="UNIQUE_TOKEN_42";</script>'
    a = _make_agent(tmp_path, file_text)
    a.add_user_answer("you have all the code! remove the circles")

    # base_message can be empty for an answer-only turn.
    out = a._flush_user_injections("")
    assert "USER ANSWER" in out
    assert "you have all the code" in out
    assert "CURRENT FILE ON DISK" in out
    assert "UNIQUE_TOKEN_42" in out


def test_post_clean_no_feedback_does_not_inline_file(tmp_path: Path) -> None:
    """Cost guard: don't pay the 7K-token inline cost on a turn where
    the model is just being told 'clean — prefer <done/>'."""
    file_text = '<!doctype html><script>const Y="SHOULD_NOT_APPEAR";</script>'
    a = _make_agent(tmp_path, file_text)

    out = a._flush_user_injections(_POST_CLEAN_BASE)
    assert "CURRENT FILE ON DISK" not in out
    assert "SHOULD_NOT_APPEAR" not in out


def test_post_clean_feedback_empty_current_file_no_inline(tmp_path: Path) -> None:
    """Defensive: don't render an empty CURRENT FILE block when the
    agent's on-disk view hasn't been populated yet."""
    a = _make_agent(tmp_path, "")
    a._current_file = ""
    a.add_user_feedback("move the gun 10 px right")

    out = a._flush_user_injections(_POST_CLEAN_BASE)
    assert "USER FEEDBACK" in out
    # No file → no inline block.
    assert "CURRENT FILE ON DISK" not in out
