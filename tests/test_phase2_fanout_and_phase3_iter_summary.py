"""Phase 2A (parallel best-of-N) + Phase 3 (iter_summary + surprise).

All checks general — no genre logic, GPU assignment unchanged, single-slot
falls back to existing sequential behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _stub_agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )


# ---------------------------------------------------------------------------
# Phase 2A — slot detection / availability
# ---------------------------------------------------------------------------


def test_single_slot_returns_only_slot1(tmp_path):
    """No secondary backends configured → only slot1 available →
    fan-out is bypassed → sequential best_of_n is used (existing
    behavior preserved)."""
    a = _stub_agent(tmp_path)
    a._backend2 = None
    a._backend3 = None
    slots = a._available_sampler_slots()
    assert [label for _, label in slots] == ["slot1"]


def test_independent_slots_all_listed(tmp_path):
    """When slot2/slot3 are distinct backends at distinct endpoints,
    all three appear and can be fanned out across."""
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11435"
    bk3 = MagicMock()
    bk3.info.endpoint = "http://127.0.0.1:11436"
    a._backend2 = bk2
    a._backend3 = bk3
    a._critic_task = None
    slots = a._available_sampler_slots()
    assert [label for _, label in slots] == ["slot1", "slot2", "slot3"]


def test_shared_endpoint_collapsed(tmp_path):
    """Two backend objects pointing at the same Ollama endpoint must
    NOT both be used — concurrent requests would just queue at the
    daemon. The duplicate is dropped."""
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11434"  # same!
    bk3 = MagicMock()
    bk3.info.endpoint = "http://127.0.0.1:11436"
    a._backend2 = bk2
    a._backend3 = bk3
    a._critic_task = None
    slots = a._available_sampler_slots()
    assert [label for _, label in slots] == ["slot1", "slot3"]


def test_critic_busy_slot_excluded(tmp_path):
    """When a concurrent critic task is running on slot 2, the
    fan-out skips slot 2 — running another LLM stream there would
    queue at the same daemon and steal time from the critic."""
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11435"
    bk3 = MagicMock()
    bk3.info.endpoint = "http://127.0.0.1:11436"
    a._backend2 = bk2
    a._backend3 = bk3
    # Make critic point at slot 2 AND mark the task as in-flight.
    a._model2_role = "critic"
    a._model3_role = "architect"
    busy_task = MagicMock()
    busy_task.done.return_value = False
    a._critic_task = busy_task
    slots = a._available_sampler_slots()
    assert [label for _, label in slots] == ["slot1", "slot3"]


def test_done_critic_task_does_not_block_slot(tmp_path):
    """A FINISHED critic task no longer holds its slot."""
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11435"
    a._backend2 = bk2
    a._model2_role = "critic"
    done_task = MagicMock()
    done_task.done.return_value = True
    a._critic_task = done_task
    slots = a._available_sampler_slots()
    assert [label for _, label in slots] == ["slot1", "slot2"]


# ---------------------------------------------------------------------------
# Phase 2A — fan-out scoring + non-slot1-winner surprise
# ---------------------------------------------------------------------------


def test_fanout_picks_highest_score_winner(tmp_path):
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    a._backend.stream_chat = MagicMock()
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11435"
    a._backend2 = bk2
    a._critic_task = None
    a._messages = [{"role": "user", "content": "test"}]
    a.num_ctx = 4096
    a.stall_seconds = 60.0
    a.overall_seconds = 300.0
    a._keep_alive_for_backend = MagicMock(return_value=None)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    from backend import StreamResult

    async def fake_stream_chat_slot1(*args, **kwargs):
        return StreamResult(text="slot1-output", tokens=100, duration_s=2.0, stalled=False)

    async def fake_stream_chat_slot2(*args, **kwargs):
        return StreamResult(text="slot2-output", tokens=120, duration_s=2.2, stalled=False)

    a._backend.stream_chat = fake_stream_chat_slot1
    bk2.stream_chat = fake_stream_chat_slot2

    async def scorer(text):
        # Score slot2's output higher than slot1's.
        if "slot2" in text:
            return 75.0, {"report_ok": False, "kind": "candidate"}
        return 30.0, {"report_ok": False, "kind": "candidate"}

    slots = a._available_sampler_slots()

    async def _drive():
        return await a._fan_out_best_of_n_across_slots(
            slots=slots, n=2, scorer=scorer,
        )

    winner, all_cands = asyncio.run(_drive())
    assert winner is not None
    assert "slot2" in winner.text
    assert winner.score == 75.0
    # Surprise event for non-slot1 winner.
    assert any(
        t.get("kind") == "surprise"
        and t.get("category") == "non_slot1_bon_winner"
        for t in traces
    )
    # best_of_n_attempt summary present with both candidates.
    bon_events = [t for t in traces if t.get("kind") == "best_of_n_attempt"]
    assert len(bon_events) == 1
    assert bon_events[0]["winner_slot"] == "slot2"
    assert len(bon_events[0]["candidate_summary"]) == 2


def test_fanout_slot1_winner_no_surprise(tmp_path):
    """Slot 1 winning is the expected case — no surprise event."""
    a = _stub_agent(tmp_path)
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    bk2 = MagicMock()
    bk2.info.endpoint = "http://127.0.0.1:11435"
    a._backend2 = bk2
    a._critic_task = None
    a._messages = [{"role": "user", "content": "test"}]
    a.num_ctx = 4096
    a.stall_seconds = 60.0
    a.overall_seconds = 300.0
    a._keep_alive_for_backend = MagicMock(return_value=None)
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    from backend import StreamResult

    async def fake_slot1(*args, **kwargs):
        return StreamResult(text="slot1-output", tokens=100, duration_s=2.0, stalled=False)

    async def fake_slot2(*args, **kwargs):
        return StreamResult(text="slot2-output", tokens=120, duration_s=2.2, stalled=False)

    a._backend.stream_chat = fake_slot1
    bk2.stream_chat = fake_slot2

    async def scorer(text):
        if "slot1" in text:
            return 90.0, {"report_ok": True, "kind": "candidate"}
        return 30.0, {"report_ok": False, "kind": "candidate"}

    slots = a._available_sampler_slots()

    async def _drive():
        return await a._fan_out_best_of_n_across_slots(slots=slots, n=2, scorer=scorer)

    winner, _ = asyncio.run(_drive())
    assert winner is not None
    assert "slot1" in winner.text
    # No non-slot1 surprise.
    assert not any(
        t.get("kind") == "surprise"
        and t.get("category") == "non_slot1_bon_winner"
        for t in traces
    )


# ---------------------------------------------------------------------------
# Phase 3 — iter_summary content + surprise rules
# ---------------------------------------------------------------------------


def test_phase_3_emits_iter_summary_event_signature():
    """Verify the source has the iter_summary emission with the
    expected payload fields. Direct integration through agent.run
    requires a full Chromium + backend stack, so this is a structural
    check on the source."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    # The emission site exists.
    assert '"kind": "iter_summary"' in src
    # Required payload fields.
    for key in (
        "probes_passed", "probes_total", "soft_warnings_count",
        "page_errors_count", "console_errors_count", "fail_reason",
        "entity_missing_count",
    ):
        assert f'"{key}"' in src, f"iter_summary payload missing field {key!r}"


def test_phase_3_surprise_rules_in_source():
    """The two surprise categories — state_vs_render_gap and
    regression_after_clean_iter — must be wired."""
    import inspect
    import agent as agent_module
    src = inspect.getsource(agent_module)
    assert '"category": "state_vs_render_gap"' in src
    assert '"category": "regression_after_clean_iter"' in src
    assert '"category": "non_slot1_bon_winner"' in src


def test_learner_reflector_prompt_documents_new_kinds():
    """The reflector must know what the new trace kinds mean, or its
    JSON output will treat them as noise."""
    import inspect
    import learner as learner_module
    src = inspect.getsource(learner_module)
    for kind in (
        "iter_summary", "surprise", "best_of_n_attempt",
        "patch_outcome", "slow_prefill",
        "autonomous_playtest_skipped",
    ):
        assert kind in src, (
            f"REFLECTOR_SYSTEM must document trace kind {kind!r}"
        )
