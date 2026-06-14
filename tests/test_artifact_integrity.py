"""Trace/log/conversation artifact identity tests.

Seeded runs intentionally reuse the canonical game basename so the live
HTML, best.html, assets, and sounds stay in one place. Trace artifacts must
not reuse that bare basename, or a later seed edit can overwrite .log /
.conversation.md while the .jsonl is appended from a different goal.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from agent import GameAgent


def _make_agent(tmp_path: Path, name: str = "game.html") -> GameAgent:
    out = tmp_path / name
    out.write_text("<html></html>", encoding="utf-8")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def test_trace_artifacts_use_per_run_id_not_game_basename(tmp_path: Path) -> None:
    a = _make_agent(tmp_path, "mortal.html")

    assert a._session_id == "mortal"
    assert a._artifact_id.startswith("mortal__run_")
    assert a.trace_path.name == f"{a._artifact_id}.jsonl"
    assert a.conversation_path.name == f"{a._artifact_id}.conversation.md"
    assert a.snapshots_dir.name == a._artifact_id

    # Game artifacts remain canonical and reusable.
    assert a.best_path.name == "mortal.best.html"


def test_same_game_basename_gets_distinct_artifact_paths(tmp_path: Path) -> None:
    a1 = _make_agent(tmp_path, "mortal.html")
    a2 = _make_agent(tmp_path, "mortal.html")

    assert a1._session_id == a2._session_id == "mortal"
    assert a1._artifact_id != a2._artifact_id
    assert a1.trace_path != a2.trace_path
    assert a1.conversation_path != a2.conversation_path
    assert a1.snapshots_dir != a2.snapshots_dir
    # But they still share the same live game/best paths.
    assert a1.out_path == a2.out_path
    assert a1.best_path == a2.best_path


def test_conversation_dump_is_retired_no_op(tmp_path: Path) -> None:
    # One-complete-trace (2026-06-14): the .conversation.md sibling is retired.
    # Its content (system prompt, every user turn, every assistant reply) now
    # lives in the .jsonl via system_prompt_built.system_prompt,
    # stream_start.turn_input, and assistant_reply.reply. _dump_conversation()
    # is a no-op and must NOT write the redundant file.
    a = _make_agent(tmp_path, "mortal.html")
    a._goal = "mortal kombat goal"
    a._messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    a._dump_conversation()

    assert not a.conversation_path.exists()

