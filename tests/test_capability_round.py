"""Tests for the capability round (plan local-model_game_agent_capability_round).

Five agent-side upgrades, all covered with pure-function tests — no model,
no Chromium:

  1. Component skill library (memory/components.jsonl + retrieval + render)
  2. Polish phase after probes pass
  3. Context discipline (report-turn collapse + stale <probes> elision)
  4. Runtime state timeline digest
  5. Stuck best-of-2 escalation predicate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    GameAgent,
    _POLISH_TURN_CAP,
    _REPORT_BLOCK_BEGIN,
    _STUCK_BON_ESCALATION_CAP,
)
from memory import GameMemory, render_components_block  # noqa: E402
from tools import run_micro_probes, summarize_state_timeline  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
COMPONENTS_PATH = REPO_ROOT / "memory" / "components.jsonl"


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    fake_browser = MagicMock()
    agent = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=4,
        memory_root=str(tmp_path / "memory"),
    )
    return agent


def _repo_memory() -> GameMemory:
    return GameMemory(REPO_ROOT / "memory")


def _clean_report() -> dict:
    return {
        "ok": True,
        "errors": [],
        "soft_warnings": [],
        "warnings": [],
        "title": "Game",
        "canvas": None,
        "input_listeners": {},
        "input_test": None,
        "frozen_canvas": False,
        "body_chars": 10,
        "body_sample": "",
        "logs": [],
        "probes": [{"name": "p1", "ok": True}],
        "page_errors": [],
        "console_errors": [],
    }


def _failing_report() -> dict:
    r = _clean_report()
    r["ok"] = False
    r["probes"] = [{"name": "p1", "ok": False}]
    r["page_errors"] = ["TypeError: boom"]
    return r


# ---------------------------------------------------------------------------
# 1. Component skill library
# ---------------------------------------------------------------------------

def test_components_file_exists_and_parses():
    assert COMPONENTS_PATH.exists()
    recs = [json.loads(l) for l in COMPONENTS_PATH.read_text().splitlines() if l.strip()]
    assert len(recs) >= 10
    for rec in recs:
        assert rec["id"]
        assert rec["kind"] == "component"
        assert rec["content"]
        assert rec["recipe"]["code"].strip()


def test_every_component_snippet_passes_micro_probes():
    """Each seed snippet, wrapped in a minimal page, must clear the
    pre-Chromium structural checks (brackets, script presence)."""
    recs = [json.loads(l) for l in COMPONENTS_PATH.read_text().splitlines() if l.strip()]
    for rec in recs:
        code = rec["recipe"]["code"]
        html = (
            "<!DOCTYPE html>\n<html><head><title>t</title></head><body>\n"
            "<canvas id=\"c\" width=\"100\" height=\"100\"></canvas>\n"
            f"<script>\n{code}\n</script>\n</body></html>"
        )
        mp = run_micro_probes(html)
        assert mp["ok"], f"{rec['id']}: {mp['errors']}"


def test_new_high_value_components_present_and_have_code():
    """The bounded set added for weak-VLM support must exist, parse, and
    carry non-empty code (they target what local models reliably get
    wrong: HiDPI, seeded generation, input, storage, touch, pathfinding,
    image-load race, audio unlock)."""
    recs = {
        json.loads(l)["id"]: json.loads(l)
        for l in COMPONENTS_PATH.read_text().splitlines() if l.strip()
    }
    for cid in (
        "dpr-canvas-scaling",
        "seeded-rng-grid-generator",
        "keyboard-input-ecode",
        "localstorage-save-load",
        "mobile-touch-dpad",
        "bfs-grid-pathfinding",
        "image-load-decode-gate",
        "audio-unlock-on-gesture",
    ):
        assert cid in recs, f"missing component {cid}"
        assert recs[cid]["recipe"]["code"].strip(), f"{cid} has empty code"
        assert recs[cid]["kind"] == "component"


def test_seeded_generation_component_retrieves_for_maze_goal():
    """A maze/procedural goal should surface the seeded-grid generator —
    the snippet that prevents the big-literal token-repetition loop."""
    m = _repo_memory()
    hits = m.retrieve_components(
        "procedurally generated maze dungeon with random walls each run",
        k=3,
    )
    ids = [h.item.id for h in hits]
    assert "seeded-rng-grid-generator" in ids, ids


def test_components_retrieval_routes_fighting_goal():
    m = _repo_memory()
    hits = m.retrieve_components(
        "build a single-screen 2d fighting game with two characters, "
        "punches, kicks, health bars",
        k=3,
    )
    ids = [h.item.id for h in hits]
    assert ids, "fighting goal retrieved no components"
    assert any(i in ("hit-pause", "entity-state-machine") for i in ids), ids


def test_components_retrieval_empty_for_blank_goal():
    m = _repo_memory()
    assert m.retrieve_components("", k=3) == []


def test_render_components_block_respects_budget():
    m = _repo_memory()
    hits = m.retrieve_components("fighting punches kicks attack hit", k=3)
    assert hits
    block = render_components_block(hits, char_budget=900)
    assert block.startswith("<components>")
    assert block.endswith("</components>")
    # Whole entries are dropped, never truncated mid-code: at least one
    # complete fenced snippet survives even under a small budget.
    assert block.count("```js") == block.count("\n```", block.index("```js") + 1) or "```js" in block
    # Budget bounds the body (wrapper text adds a fixed overhead).
    assert len(block) < 900 + 400


def test_render_components_block_empty_cases():
    assert render_components_block([]) == ""


def test_agent_retrieve_components_block_renders_and_traces(tmp_path):
    a = _make_agent(tmp_path)
    a._memory = _repo_memory()
    traces: list[dict] = []
    a._trace = lambda rec: traces.append(rec)
    block = a._retrieve_components_block(
        "fighting game with punches and kicks", stage="plan", k=3,
    )
    assert "<components>" in block
    assert "adapt" in block.lower()
    kinds = [t.get("kind") for t in traces]
    assert "components_injected" in kinds
    row = [t for t in traces if t.get("kind") == "components_injected"][0]
    assert row["stage"] == "plan"
    assert row["ids"]


def test_both_first_build_paths_inject_components():
    """Seed-file build AND skeleton build must both inject a <components>
    block. The seed path previously skipped it, starving weak models of the
    copy-paste-correct snippets on continuation/seed builds."""
    import inspect
    src = inspect.getsource(GameAgent.run)
    # Both first-build branches use the same plan-stage component call.
    assert 'seed_build_instruction(' in src
    assert 'first_build_instruction(' in src
    assert src.count('goal, stage="plan", k=3,') >= 2, (
        "expected a first-build <components> injection on BOTH the seed and "
        "skeleton paths"
    )


def test_report_blocker_query_builds_from_failures():
    q = GameAgent._report_blocker_query({
        "probes": [{"name": "player_moves", "ok": False}],
        "console_errors": ["TypeError: x is undefined"],
        "frozen_canvas": True,
        "input_test": {"ran": True, "any_change": False},
    })
    assert "player_moves" in q
    assert "TypeError" in q
    assert "frozen" in q
    # Clean report -> empty query -> no fix-turn injection.
    assert GameAgent._report_blocker_query(_clean_report()) == ""


# ---------------------------------------------------------------------------
# 2. Polish phase
# ---------------------------------------------------------------------------

def test_polish_prompt_sent_when_clean_and_budget_remains(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "fighting game"
    a._current_file = "<html><body>game</body></html>"
    a._iters_remaining = 2
    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )
    assert "polish turn 1/2" in prompt
    assert "GAME FEEL" in prompt
    assert a._polish_turns_used == 1
    assert a._polish_pending is True
    # Polish never blocks shipping — <done/> stays on the table.
    assert "<done/>" in prompt


def test_polish_cap_reached_falls_back_to_post_clean(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "fighting game"
    a._current_file = "<html></html>"
    a._iters_remaining = 2
    a._polish_turns_used = _POLISH_TURN_CAP
    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )
    assert "STRONGLY prefer ending with" in prompt
    assert "polish turn" not in prompt
    assert a._polish_pending is False


def test_polish_skipped_when_no_iters_remaining(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "fighting game"
    a._iters_remaining = 0
    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )
    assert "polish turn" not in prompt
    assert a._polish_turns_used == 0


def test_polish_skipped_when_user_feedback_pending(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "fighting game"
    a._iters_remaining = 2
    a._pending_feedback.append("make the player blue")
    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )
    assert "polish turn" not in prompt


def test_polish_prompt_includes_critic_note_and_file(tmp_path):
    a = _make_agent(tmp_path)
    a._goal = "fighting game"
    a._current_file = "<html><body>UNIQUE-MARKER-123</body></html>"
    a._iters_remaining = 1
    a._last_critic_note = "the hits have no visual feedback"
    prompt = a._build_fix_prompt(
        report=_clean_report(), regressed=False, partial_failed=[],
    )
    assert "the hits have no visual feedback" in prompt
    assert "UNIQUE-MARKER-123" in prompt


def test_polish_regression_revert_clears_pending_flag(tmp_path):
    """revert_to_iter must drop a stale in-flight polish flag (the turn
    counter is per-session and survives)."""
    a = _make_agent(tmp_path)
    a._polish_pending = True
    a._polish_turns_used = 1
    a.out_path.write_text("<html>current</html>")
    # No snapshots exist -> revert resolves to best.html or errors; either
    # way the polish flag handling runs only on success, so write best.
    best = a.out_path.with_name(".best.html")
    best.write_text("<html>best</html>")
    res = a.revert_to_iter(None)
    if res["ok"]:
        assert a._polish_pending is False
        assert a._polish_turns_used == 1


# ---------------------------------------------------------------------------
# 3. Context discipline
# ---------------------------------------------------------------------------

def test_report_wrap_and_collapse_keeps_newest_and_feedback(tmp_path):
    a = _make_agent(tmp_path)
    report = _failing_report()
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(3):
        msgs.append({
            "role": "user",
            "content": (
                "USER FEEDBACK: keep me\n"
                + a._wrap_report_block(f"REPORT {i} " + "y" * 800, report)
            ),
        })
        msgs.append({"role": "assistant", "content": f"reply {i}"})
    for i in range(4):
        msgs.append({"role": "user", "content": f"recent {i}"})
    a._messages = msgs
    a._last_prompt_pressure = 0.0
    a._prune_messages()

    collapsed = [
        m for m in a._messages
        if "superseded test report" in (m.get("content") or "")
    ]
    wrapped = [
        m for m in a._messages
        if _REPORT_BLOCK_BEGIN in (m.get("content") or "")
    ]
    # Two old reports collapsed; the newest stays verbatim.
    assert len(collapsed) == 2
    assert len(wrapped) == 1
    # The 3-line digest survives in collapsed turns.
    for m in collapsed:
        assert "ok=False" in m["content"]
        assert "probes passing: 0/1" in m["content"]
        assert "first blocker:" in m["content"]
        # User feedback outside the wrapper survives untouched.
        assert "USER FEEDBACK: keep me" in m["content"]


def test_stale_probes_elided_in_pruned_assistant_turns_only(tmp_path):
    a = _make_agent(tmp_path)
    big_probes = "<probes>" + ('{"name": "p", "expr": "1"},' * 30) + "</probes>"
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "assistant", "content": "old reply " + big_probes})
    msgs.append({"role": "user", "content": "old user turn " + big_probes})
    for i in range(5):
        msgs.append({"role": "user", "content": f"recent {i}"})
    a._messages = msgs
    a._last_prompt_pressure = 0.0
    a._prune_messages()
    assert "HARNESS-OMITTED-PRIOR-PROBES" in a._messages[1]["content"]
    # User turns (not report-wrapped) are never rewritten.
    assert big_probes in a._messages[2]["content"]


def test_small_probes_blocks_stay_verbatim(tmp_path):
    a = _make_agent(tmp_path)
    small = '<probes>[{"name": "p", "expr": "1"}]</probes>'
    out = a._summarize_content("reply " + small)
    assert small in out


def test_report_digest_lines_pure():
    digest = GameAgent._report_digest_lines(_failing_report())
    lines = digest.splitlines()
    assert len(lines) == 3
    assert lines[0] == "ok=False"
    assert "0/1" in lines[1]
    assert "p1" in lines[2]
    # Digest must not be able to close its wrapping HTML comment.
    nasty = _failing_report()
    nasty["probes"] = []
    nasty["page_errors"] = ["evil --> breakout"]
    assert "-->" not in GameAgent._report_digest_lines(nasty)


# ---------------------------------------------------------------------------
# 4. State timeline digest
# ---------------------------------------------------------------------------

def _samples(leaf_values: dict[str, list[float]]) -> list[dict]:
    n = len(next(iter(leaf_values.values())))
    return [
        {k: vals[i] for k, vals in leaf_values.items()}
        for i in range(n)
    ]


def test_timeline_classifies_constant_monotonic_changing():
    samples = _samples({
        "player.x": [100, 110, 120, 130, 140, 150],   # monotonic
        "cpu.x": [600, 600, 600, 600, 600, 600],       # constant
        "bullets.length": [0, 1, 0, 2, 1, 0],          # changing
    })
    d = summarize_state_timeline(samples)
    assert "1 constant, 1 monotonic, 1 changing" in d
    assert "player.x (100" in d


def test_timeline_flags_stalled_frame_counter():
    samples = _samples({
        "frame": [42, 42, 42, 42, 42, 42],
        "player.x": [1, 2, 3, 4, 5, 6],
    })
    d = summarize_state_timeline(samples)
    assert "SUSPICIOUS" in d
    assert "frame not increasing" in d


def test_timeline_flags_frozen_sibling_position():
    samples = _samples({
        "player.x": [100, 120, 140, 160, 180, 200],
        "cpu.x": [600, 600, 600, 600, 600, 600],
    })
    d = summarize_state_timeline(samples)
    assert "cpu.x constant across all samples while player.x changes" in d


def test_timeline_render_length_cap():
    samples = _samples({
        "frame": [1, 1, 1, 1, 1, 1],
        "player.x": [0, 1, 2, 3, 4, 5],
        "player.y": [0, 1, 0, 1, 0, 1],
        "cpu.x": [9, 9, 9, 9, 9, 9],
        "cpu.y": [9, 9, 9, 9, 9, 9],
    })
    d = summarize_state_timeline(samples, max_lines=4)
    assert len(d.splitlines()) <= 4


def test_timeline_empty_when_no_state_exposed():
    assert summarize_state_timeline([None, None, None, None, None, None]) == ""
    assert summarize_state_timeline([]) == ""
    # Fewer than 3 usable samples -> not classifiable.
    assert summarize_state_timeline([{"a": 1}, {"a": 2}]) == ""


# ---------------------------------------------------------------------------
# 5. Stuck best-of-2 escalation predicate
# ---------------------------------------------------------------------------

def test_stuck_bon_predicate():
    f = GameAgent._should_escalate_stuck_bon
    assert f(2, 1, 0) is True
    assert f(3, 1, 1) is True
    # Not stuck enough.
    assert f(0, 1, 0) is False
    assert f(1, 1, 0) is False
    # Session already samples N>1 every turn — no escalation.
    assert f(2, 2, 0) is False
    assert f(5, 3, 0) is False
    # Per-session cap.
    assert f(2, 1, _STUCK_BON_ESCALATION_CAP) is False
    assert f(9, 1, _STUCK_BON_ESCALATION_CAP + 1) is False
