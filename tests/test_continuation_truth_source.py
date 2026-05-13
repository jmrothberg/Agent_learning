"""Tests for the CURRENT FILE ON DISK truth source on continuation /
restart turns.

Motivating trace:
  games/traces/a-game-of-donkey-kong-with-a-w_20260512_201139.log

Bug: the agent's continuation prompt said "the file is unchanged on
disk" but didn't include the file text. Model wrote a patch targeting
`state.princessTimer` when the file actually has `state.princess.timer`
— patch failed, iter wasted, regression risk. Worst on restarts (msg
history starts empty) and on weaker local models (more prone to
hallucinating variable names without the truth source in view).

Fix: `prompts_v1.continuation_instruction(current_file)` inlines the
full file under the standard "CURRENT FILE ON DISK ... SOURCE OF
TRUTH" header that fix_instruction uses, and agent.py's continuation
branch routes through it.

These tests cover three independent code paths:
  1. The prompt helper itself.
  2. The agent's continuation branch (new feedback after <done/>).
  3. The restart case (continuation=True on a fresh-process agent
     whose message history is empty).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import prompts_v1  # noqa: E402
from agent import GameAgent  # noqa: E402


def test_helper_includes_current_file() -> None:
    file_text = "<!doctype html><script>const X=42;</script>"
    msg = prompts_v1.continuation_instruction(file_text)
    assert "CURRENT FILE ON DISK" in msg
    assert "SOURCE OF TRUTH" in msg
    assert "character-for-character" in msg
    assert file_text in msg


def test_helper_handles_long_file_without_truncation() -> None:
    # 30 KB game — typical size after a real session. Must NOT be
    # silently truncated; the patch matcher needs the exact bytes.
    body = "\n".join(f"const v{i} = {i};" for i in range(2000))
    file_text = f"<!doctype html><script>{body}</script>"
    msg = prompts_v1.continuation_instruction(file_text)
    assert body in msg
    assert len(file_text) > 20_000  # sanity
    # The truth-source header must come BEFORE the file content so the
    # model reads the framing first.
    assert msg.index("CURRENT FILE ON DISK") < msg.index(body)


def _make_agent(tmp_path: Path, file_text: str) -> GameAgent:
    out = tmp_path / "g.html"
    out.write_text(file_text)
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


async def _consume_until(agen, stop_on: str) -> list:
    """Drive the async iterator until we've seen the named event-text
    fragment (so we know the continuation branch fired) without making
    a real model call."""
    out = []
    async for ev in agen:
        out.append(ev)
        if ev.text and stop_on in ev.text:
            # One more event to let the message append complete.
            try:
                out.append(await agen.__anext__())
            except (StopAsyncIteration, Exception):
                pass
            break
    return out


def test_continuation_branch_inlines_truth_source_into_messages(
    tmp_path: Path,
) -> None:
    """Driving run(..., continuation=True) on an agent with an existing
    _current_file must append a user message containing both the
    CURRENT FILE ON DISK header AND the file's exact text."""
    file_text = (
        '<!doctype html><script>'
        'state.princess = { timer: 0, frame: 0 };'
        '</script>'
    )
    a = _make_agent(tmp_path, file_text)
    a._current_file = file_text

    # Drive just enough of the loop for the continuation branch to
    # land its message. The branch happens BEFORE any model call, so
    # we can stop the iterator immediately after the "continuing on
    # existing file" info event fires.
    gen = a.run("change the princess timer to 0.35s", continuation=True)
    asyncio.run(_consume_until(gen, "continuing on existing file"))

    # The user message appended by the continuation branch must
    # contain the file truth source.
    user_msgs = [m for m in a._messages if m.get("role") == "user"]
    assert user_msgs, "no user message appended on continuation"
    last = user_msgs[-1]["content"]
    assert "CURRENT FILE ON DISK" in last
    assert "state.princess = { timer: 0, frame: 0 }" in last
    # The user's actual feedback must also be in there (via the
    # standard USER FEEDBACK banner injected by _flush_user_injections).
    assert "change the princess timer to 0.35s" in last


def test_restart_case_loads_file_then_inlines_truth_source(
    tmp_path: Path,
) -> None:
    """Restart scenario: a fresh process starts a continuation against
    an on-disk file the agent hasn't loaded yet. The continuation
    branch must read the file from disk first, then inline it as the
    truth source so the model sees actual content (not "the file is
    unchanged" with no body)."""
    file_text = (
        '<!doctype html><script>const TIMER = "this_is_unique_text_12345";</script>'
    )
    out = tmp_path / "restart.html"
    out.write_text(file_text)
    # Fresh agent: _current_file starts empty.
    a = GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    assert a._current_file == ""

    gen = a.run("tighten the timer", continuation=True)
    asyncio.run(_consume_until(gen, "continuing on existing file"))

    # The branch must have populated _current_file from disk AND
    # inlined it.
    assert a._current_file == file_text
    user_msgs = [m for m in a._messages if m.get("role") == "user"]
    assert user_msgs
    last = user_msgs[-1]["content"]
    assert "this_is_unique_text_12345" in last, (
        "restart-continuation prompt is missing the file's actual "
        "content; weaker local models will hallucinate patch anchors"
    )
    assert "CURRENT FILE ON DISK" in last
