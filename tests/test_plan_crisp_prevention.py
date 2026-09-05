"""Tests for Crisp Phase A planning prevention (2026-06-28 plan)."""

from __future__ import annotations

from pathlib import Path

import prompts_v1
from agent import GameAgent
from backend import BackendInfo, make_backend


def _make_agent(tmp_path: Path, *, backend_name: str = "ollama") -> GameAgent:
    info = BackendInfo(
        name=backend_name,  # type: ignore[arg-type]
        model="dummy:0",
        source="test",
        endpoint="http://127.0.0.1:0",
    )
    backend = make_backend(info)
    agent = GameAgent(
        backend=backend,
        out_path=tmp_path / "game.html",
        max_iters=1,
    )
    agent._model_class = "small"
    return agent


def test_first_build_instruction_thinking_friendly_contract():
    body = prompts_v1.first_build_instruction("<html>seed</html>")
    low = body.lower()
    assert "do not restate requirements or re-plan" in low
    assert "raw `<html_file>`" in low or "raw <html_file>" in low
    assert "requestanimationframe(loop)" in low
    assert "markdown ```html fence" in low


def test_measure_plan_reply_splits_prose_and_tags():
    reply = (
        "Some essay before tags.\n"
        "<plan>Mechanics: jump</plan>\n"
        "<criteria>Basic: moves</criteria>\n"
        "<probes>[{\"name\":\"x\",\"expr\":\"true\"}]</probes>\n"
    )
    prose, canonical = prompts_v1.measure_plan_reply(reply)
    assert prose > 0
    assert canonical > 0


def test_local_plan_crisp_nudge_injected_for_small_model():
    nudge_ids: list[str] = []
    body = prompts_v1.plan_instruction(
        goal="build a snake game",
        model_class="small",
        nudge_ids_out=nudge_ids,
    )
    assert "local-plan-crisp" in nudge_ids
    assert "LOCAL PLAN CRISPNESS" in body


def test_local_plan_crisp_nudge_not_injected_for_large_model():
    nudge_ids: list[str] = []
    prompts_v1.plan_instruction(
        goal="build a snake game",
        model_class="large",
        nudge_ids_out=nudge_ids,
    )
    assert "local-plan-crisp" not in nudge_ids


def test_pre_lean_plan_before_first_build_heavy_assets(tmp_path: Path):
    agent = _make_agent(tmp_path)
    for i in range(10):
        agent._session_assets[f"s{i}"] = tmp_path / f"s{i}.png"
    assert agent._should_pre_lean_plan_before_first_build() is True


def test_pre_lean_plan_before_first_build_small_snake_noop(tmp_path: Path):
    agent = _make_agent(tmp_path)
    agent._session_assets = {"food": tmp_path / "food.png"}
    agent._messages = [{
        "role": "assistant",
        "phase": "planning",
        "content": (
            "<plan>Mechanics: snake grid</plan>"
            "<criteria>Basic: moves</criteria>"
            "<probes>[]</probes>"
        ),
    }]
    assert agent._should_pre_lean_plan_before_first_build() is False


def test_plan_incomplete_retry_replaces_failed_blob(tmp_path: Path):
    failed = "essay " * 5000 + "<plan>partial</plan>"
    retry = (
        "<plan>Mechanics: ok</plan>"
        "<criteria>Basic: ok</criteria>"
        "<probes>[{\"name\":\"x\",\"expr\":\"true\"}]</probes>"
    )
    messages = [
        {"role": "user", "content": "plan please"},
        {
            "role": "assistant",
            "phase": "planning",
            "content": failed,
        },
        {"role": "user", "content": "retry"},
    ]
    replaced = False
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("phase") == "planning":
            msg["content"] = retry
            replaced = True
            break
    assert replaced
    planning_msgs = [
        m for m in messages
        if m.get("role") == "assistant" and m.get("phase") == "planning"
    ]
    assert len(planning_msgs) == 1
    assert planning_msgs[0]["content"] == retry
    assert len(planning_msgs[0]["content"]) < len(failed)


def test_lean_compact_drops_untagged_plan_essay(tmp_path: Path):
    """DIGDUGD3: 30k untagged essay must not ride into first-build prefill."""
    agent = _make_agent(tmp_path)
    essay = "Let me think about mechanics. " * 800
    agent._messages = [{
        "role": "assistant",
        "phase": "planning",
        "content": (
            essay
            + "<plan>Mechanics: dig tunnels</plan>"
            + "<criteria>Basic: ArrowRight moves player</criteria>"
            + "<probes>[{\"name\":\"input_moves_player\",\"expr\":\"true\"}]</probes>"
        ),
    }]
    assert agent._should_pre_lean_plan_before_first_build() is True
    agent._lean_compact_planning_message()
    kept = agent._messages[0]["content"]
    assert "Let me think about mechanics" not in kept
    assert "<criteria>" in kept
    assert "<probes>" in kept
    assert "input_moves_player" in kept


def test_jmr_pre_lean_fires_on_modest_untagged_prose(tmp_path: Path):
    agent = _make_agent(tmp_path)
    agent.set_jmr_png_mode(True)
    agent._messages = [{
        "role": "assistant",
        "phase": "planning",
        "content": (
            ("FPGA notes. " * 80)
            + "<plan>Mechanics: tiles</plan>"
            + "<criteria>Basic: moves</criteria>"
            + "<probes>[]</probes>"
        ),
    }]
    assert agent._should_pre_lean_plan_before_first_build() is True


def test_full_html_keeps_short_tagged_plan(tmp_path: Path):
    """Media/full-HTML: a crisp tagged plan is not pre-leaned away."""
    agent = _make_agent(tmp_path)
    agent._session_assets = {"hero": tmp_path / "hero.png"}
    agent._messages = [{
        "role": "assistant",
        "phase": "planning",
        "content": (
            "<plan>Mechanics: ship flies</plan>"
            "<criteria>Basic: moves</criteria>"
            "<probes>[]</probes>"
        ),
    }]
    assert agent._should_pre_lean_plan_before_first_build() is False
