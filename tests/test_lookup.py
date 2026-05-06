"""Tests for <lookup_bullet> tag extraction + injection in agent.py.

Pi-mono "skills" pattern: when the playbook block is rendered in hybrid
mode, only the top-N bullets ship with their bodies; the rest are
indexed by ID. The model can emit <lookup_bullet>id</lookup_bullet> in
its reply to ask for the body of any indexed entry — the agent resolves
the lookup against the playbook and injects the body into the next
user-turn message.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _MAX_BULLET_LOOKUPS_PER_TURN  # noqa: E402
from memory import Bullet  # noqa: E402


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    # Replace the seeded playbook with our small fixture set.
    a._playbook._save_all([
        Bullet(id="thrust-rule", content="Compute velocity from facing angle: vx = cos(a)*s, vy = sin(a)*s.", tags=["ship", "thrust"]),
        Bullet(id="dpr-rule", content="Use device-pixel-ratio scaling so retina displays aren't blurry.", tags=["canvas", "dpr"]),
        Bullet(id="raf-rule", content="Drive animation with requestAnimationFrame, never setInterval.", tags=["raf", "loop"]),
    ])
    return a


def test_no_lookup_tags_does_nothing(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._extract_and_queue_lookups("a normal reply with no lookup tags")
    assert a._pending_bullet_lookups == []


def test_single_lookup_resolves_to_body(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    reply = "<diagnose>...</diagnose>\n<lookup_bullet>thrust-rule</lookup_bullet>"
    a._extract_and_queue_lookups(reply)
    assert len(a._pending_bullet_lookups) == 1
    block = a._pending_bullet_lookups[0]
    assert "PLAYBOOK LOOKUP RESULTS" in block
    assert "[thrust-rule]" in block
    assert "Compute velocity from facing angle" in block
    assert "tags=[ship,thrust]" in block


def test_multiple_lookups_in_one_reply(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    reply = (
        "<lookup_bullet>thrust-rule</lookup_bullet>\n"
        "Some prose between.\n"
        "<lookup_bullet>raf-rule</lookup_bullet>"
    )
    a._extract_and_queue_lookups(reply)
    block = a._pending_bullet_lookups[0]
    assert "[thrust-rule]" in block
    assert "[raf-rule]" in block
    assert "requestAnimationFrame" in block


def test_unknown_id_surfaces_not_found(tmp_path: Path) -> None:
    """Typos and stale IDs must produce a 'NOT FOUND' marker so the
    model knows its lookup missed."""
    a = _make_agent(tmp_path)
    a._extract_and_queue_lookups("<lookup_bullet>does-not-exist</lookup_bullet>")
    block = a._pending_bullet_lookups[0]
    assert "[does-not-exist]" in block
    assert "NOT FOUND" in block


def test_duplicate_lookups_collapse(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    reply = (
        "<lookup_bullet>thrust-rule</lookup_bullet>\n"
        "<lookup_bullet>thrust-rule</lookup_bullet>\n"
        "<lookup_bullet>thrust-rule</lookup_bullet>"
    )
    a._extract_and_queue_lookups(reply)
    block = a._pending_bullet_lookups[0]
    assert block.count("[thrust-rule]") == 1


def test_lookup_cap_respected(tmp_path: Path) -> None:
    """Floods of <lookup_bullet> tags get capped per turn."""
    a = _make_agent(tmp_path)
    # Add lots of bullets so all IDs resolve.
    a._playbook._save_all([
        Bullet(id=f"b{i}", content=f"body {i}", tags=["x"])
        for i in range(_MAX_BULLET_LOOKUPS_PER_TURN + 5)
    ])
    tags = "\n".join(
        f"<lookup_bullet>b{i}</lookup_bullet>"
        for i in range(_MAX_BULLET_LOOKUPS_PER_TURN + 5)
    )
    a._extract_and_queue_lookups(tags)
    block = a._pending_bullet_lookups[0]
    # Only the first _MAX_BULLET_LOOKUPS_PER_TURN distinct IDs should resolve.
    resolved_ids = sum(1 for i in range(_MAX_BULLET_LOOKUPS_PER_TURN + 5)
                       if f"[b{i}]" in block)
    assert resolved_ids == _MAX_BULLET_LOOKUPS_PER_TURN


def test_flush_drains_pending_lookups(tmp_path: Path) -> None:
    """_flush_user_injections prepends queued lookups, then clears."""
    a = _make_agent(tmp_path)
    a._extract_and_queue_lookups("<lookup_bullet>dpr-rule</lookup_bullet>")
    assert len(a._pending_bullet_lookups) == 1
    out = a._flush_user_injections("base message after lookups")
    assert "PLAYBOOK LOOKUP RESULTS" in out
    assert "Use device-pixel-ratio scaling" in out
    # Queue is drained after flush.
    assert a._pending_bullet_lookups == []
    # Base message still appears at the end.
    assert "base message after lookups" in out
    assert out.index("LOOKUP RESULTS") < out.index("base message after lookups")


def test_empty_reply_no_op(tmp_path: Path) -> None:
    a = _make_agent(tmp_path)
    a._extract_and_queue_lookups("")
    a._extract_and_queue_lookups(None)  # type: ignore[arg-type]
    assert a._pending_bullet_lookups == []
