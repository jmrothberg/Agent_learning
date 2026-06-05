"""Planning-turn retry: when a planning reply parses with no usable
<criteria>/<probes> (a degenerate repetition cut-off truncated the structured
output), run() re-streams ONCE and recovers. Trace-evidenced 2026-05-31
(street-fighter / bomberman plan eval). Deterministic — _stream is scripted,
no model/browser."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402

_COMPLETE = (
    "<plan>2D fighter</plan>\n"
    "<criteria>Both fighters visible facing each other with health bars; "
    "J punches, K kicks.</criteria>\n"
    '<probes>[{"name":"a","expr":"1"},{"name":"b","expr":"1"},'
    '{"name":"c","expr":"1"}]</probes>'
)
# Truncated by a repetition tail — no </plan>, no <criteria>, no <probes>.
_INCOMPLETE = "<plan>2D fighter, cooldown reset cooldown reset cooldown reset"


def _agent(tmp_path):
    a = GameAgent(model="stub", out_path=tmp_path / "g.html",
                  browser=MagicMock(), max_iters=1,
                  memory_root=str(tmp_path / "mem"))

    async def _no_vlm(role):
        return False
    a._detect_vlm = _no_vlm  # avoid backend capability probe
    return a


def _drive(agent, replies):
    seq = iter(replies)

    async def fake_stream(on_token, **kw):
        return next(seq)
    agent._stream = fake_stream
    events = []

    async def go():
        async for ev in agent.run("a 2D fighting game with punch and kick",
                                  plan_only=True):
            events.append(ev)
    asyncio.run(go())
    return events


def test_incomplete_plan_triggers_one_retry_and_recovers(tmp_path):
    a = _agent(tmp_path)
    events = _drive(a, [_INCOMPLETE, _COMPLETE])
    # Recovered criteria + probes from the retry.
    assert a._criteria and "health" in a._criteria.lower()
    assert a._probes and len(a._probes) >= 3
    # Two plan events: the incomplete one, then the retry.
    assert [e.kind for e in events].count("plan") == 2


def test_complete_plan_does_not_retry(tmp_path):
    a = _agent(tmp_path)
    # If a second _stream were called, next() would raise StopIteration.
    events = _drive(a, [_COMPLETE])
    assert a._criteria and a._probes
    assert [e.kind for e in events].count("plan") == 1


def test_retry_is_bounded_to_once(tmp_path):
    a = _agent(tmp_path)
    # Both attempts incomplete -> retry fires once, then proceeds (no loop).
    events = _drive(a, [_INCOMPLETE, _INCOMPLETE])
    assert [e.kind for e in events].count("plan") == 1  # retry reply had nothing parseable -> not re-yielded
    assert a._plan_retry_done is True


# ---------------------------------------------------------------------------
# Probe-quality retry: a plan that parses fine but whose probes are ALL
# structural-only gets ONE corrective re-prompt for an input→delta probe.
# 2026-05-31 prompt-library sweep: the passive nudge alone left 24/26
# prompts at probe_quality_ratio 0.0, so this escalates to a re-stream.
# ---------------------------------------------------------------------------

# Parses cleanly (criteria + 3 probes) but every probe is structural-only.
_STRUCTURAL_ONLY = (
    "<plan>top-down mover</plan>\n"
    "<criteria>Player visible; ArrowRight moves the player right.</criteria>\n"
    '<probes>[{"name":"canvas","expr":"!!document.querySelector(\'canvas\')"},'
    '{"name":"state","expr":"!!window.state"},'
    '{"name":"player","expr":"!!state && !!state.player"}]</probes>'
)
# Retry reply: keeps structural probes AND adds an input→delta dynamic probe.
_WITH_DYNAMIC = (
    "<plan>top-down mover</plan>\n"
    "<criteria>Player visible; ArrowRight moves the player right.</criteria>\n"
    '<probes>[{"name":"canvas","expr":"!!document.querySelector(\'canvas\')"},'
    '{"name":"input_moves_player","expr":"(async()=>{const x0=state.player.x;'
    "window.dispatchEvent(new KeyboardEvent('keydown',{code:'ArrowRight'}));"
    'await new Promise(r=>setTimeout(r,250));return state.player.x!==x0;})()"}]'
    "</probes>"
)


def test_all_structural_probes_trigger_probe_quality_retry(tmp_path):
    a = _agent(tmp_path)
    events = _drive(a, [_STRUCTURAL_ONLY, _WITH_DYNAMIC])
    # Retry was adopted: probes now include the dynamic input→delta probe.
    assert a._probe_quality_retry_done is True
    names = [p.get("name") for p in a._probes]
    assert "input_moves_player" in names
    assert GameAgent._classify_probes_dynamic(a._probes)["ratio"] > 0.0
    # Two plan events: the structural-only one, then the recovered retry.
    assert [e.kind for e in events].count("plan") == 2


def test_plan_with_dynamic_probe_does_not_retry(tmp_path):
    a = _agent(tmp_path)
    # A single reply that already has a dynamic probe -> no second _stream.
    events = _drive(a, [_WITH_DYNAMIC])
    assert a._probe_quality_retry_done is False
    assert [e.kind for e in events].count("plan") == 1


def test_probe_quality_retry_keeps_original_when_no_improvement(tmp_path):
    a = _agent(tmp_path)
    # Retry STILL all-structural -> not adopted; original probes kept,
    # flag set so it fires only once.
    events = _drive(a, [_STRUCTURAL_ONLY, _STRUCTURAL_ONLY])
    assert a._probe_quality_retry_done is True
    assert GameAgent._classify_probes_dynamic(a._probes)["ratio"] == 0.0
    # Original structural probes retained (3 of them).
    assert len(a._probes) == 3
    # Only the first plan was yielded; the non-improving retry was not.
    assert [e.kind for e in events].count("plan") == 1
