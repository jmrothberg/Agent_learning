"""Tests for /unqueue — drop queued feedback before the next user-turn."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def test_unqueue_bare_pops_only_last_typed_feedback(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("first")
    a.add_user_feedback("second")
    r = a.unqueue_pending_input()
    assert r["ok"] is True
    assert r["which"] == "last_feedback"
    assert a._pending_feedback == ["first"]
    assert r["removed"][0]["preview"].startswith("second")


def test_unqueue_by_index(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("one")
    a.add_user_feedback("two")
    a.add_user_feedback("three")
    r = a.unqueue_pending_input("2")
    assert r["ok"] is True
    assert a._pending_feedback == ["one", "three"]


def test_unqueue_all_clears_feedback_and_answer(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("keep me until all")
    a.add_user_answer("yes")
    r = a.unqueue_pending_input("all")
    assert r["ok"] is True
    assert a._pending_feedback == []
    assert a._pending_answer is None
    assert len(r["removed"]) == 2


def test_unqueue_answer_only(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("still here")
    a.add_user_answer("nope")
    r = a.unqueue_pending_input("answer")
    assert r["ok"] is True
    assert a._pending_feedback == ["still here"]
    assert a._pending_answer is None


def test_unqueue_bare_does_not_touch_answer(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_answer("only answer")
    r = a.unqueue_pending_input()
    assert r["ok"] is False
    assert a._pending_answer == "only answer"


def test_unqueue_bare_keeps_older_feedback(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("keep this")
    a.add_user_feedback("oops accidental")
    a.unqueue_pending_input()
    assert a._pending_feedback == ["keep this"]


def test_unqueue_empty_returns_error(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    r = a.unqueue_pending_input()
    assert r["ok"] is False


def test_unqueue_bad_index_returns_error(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a.add_user_feedback("only")
    r = a.unqueue_pending_input("9")
    assert r["ok"] is False
