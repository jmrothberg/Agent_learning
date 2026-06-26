"""Phase 2 task-ledger seeding tests.

The <todos> ledger (parse / inject / CURRENT TASK contract / drift check)
already existed; Phase 2 adds SEEDING it from the goal's clauses when the
model has not emitted its own <todos> yet, so a multi-part edit / listed ask
gets a per-turn checklist from turn 1. Pure-function: no model, no browser.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    a._trace = lambda obj: None
    return a


# ---------------------------------------------------------------------------
# _parse_task_steps
# ---------------------------------------------------------------------------


def test_parse_steps_comma_listed():
    steps = GameAgent._parse_task_steps(
        "flip the ship sprite, make the score text bigger, add a restart button"
    )
    assert len(steps) == 3
    assert steps[0].startswith("flip the ship")
    assert "restart button" in steps[-1]


def test_parse_steps_then_and_semicolons():
    steps = GameAgent._parse_task_steps(
        "spawn the enemies; then draw the towers; then wire the sell button"
    )
    assert len(steps) == 3


def test_parse_steps_numbered_lines():
    steps = GameAgent._parse_task_steps(
        "1. implement BFS pathfinding\n2. place the turrets\n3. draw the sprites"
    )
    assert len(steps) == 3
    assert steps[0] == "implement BFS pathfinding"


def test_parse_steps_terse_single_clause_returns_empty_or_one():
    # A short single-clause goal must not be force-split into noise.
    assert GameAgent._parse_task_steps("build a snake game") in ([], ["build a snake game"])


def test_parse_steps_does_not_split_bare_and():
    # "cat and mouse" is a compound noun, not two steps.
    steps = GameAgent._parse_task_steps("build a cat and mouse chase game")
    assert len(steps) == 1


def test_parse_steps_drops_tiny_fragments_and_caps():
    steps = GameAgent._parse_task_steps(
        "do a, b, c, add the real working pause menu, fix the broken sell logic"
    )
    # 'a' / 'b' / 'c' fragments (<3 words) dropped; real clauses kept.
    assert all(len(s.split()) >= 3 for s in steps)


# ---------------------------------------------------------------------------
# _seed_task_ledger_from_goal
# ---------------------------------------------------------------------------


def test_seed_ledger_from_multipart_goal(tmp_path):
    a = _agent(tmp_path)
    a._goal = "flip the ship sprite, make the score text bigger, add a restart button"
    a._seed_task_ledger_from_goal()
    assert len(a._todos_items) == 3
    assert all(done is False for done, _ in a._todos_items)
    assert "- [ ]" in a._todos_text


def test_seed_ledger_noop_for_single_clause(tmp_path):
    a = _agent(tmp_path)
    a._goal = "build a snake game with a wraparound board"
    a._seed_task_ledger_from_goal()
    assert a._todos_items == []
    assert a._todos_text == ""


def test_seed_ledger_does_not_overwrite_model_todos(tmp_path):
    a = _agent(tmp_path)
    a._goal = "flip the ship, make the score bigger, add a restart button"
    # Model already emitted a real todos list — seeding must be a no-op.
    a._todos_items = [(False, "model owned item")]
    a._todos_text = "- [ ] model owned item"
    a._seed_task_ledger_from_goal()
    assert a._todos_items == [(False, "model owned item")]


def test_seed_ledger_sets_harness_owned_flag(tmp_path):
    a = _agent(tmp_path)
    a._goal = "flip the ship, make the score bigger, add a restart button"
    a._seed_task_ledger_from_goal()
    assert a._todos_seeded_by_harness is True


# ---------------------------------------------------------------------------
# Outline-order source (b): complex fresh build whose goal is one clause
# ---------------------------------------------------------------------------


def test_outline_order_seeds_when_goal_is_single_clause(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    a._goal = "build an open-field fieldrunners tower defense"  # one clause

    class _Item:
        recipe = {"order": ["compute BFS path", "place turrets on grid",
                            "aim and fire at creeps", "draw all sprites"]}

    class _Hit:
        item = _Item()
        score = 0.9

    monkeypatch.setattr(a._memory, "retrieve_implementation_outline", lambda g: _Hit())
    a._seed_task_ledger_from_goal()
    assert len(a._todos_items) == 4
    assert a._todos_items[0][1] == "compute BFS path"


def test_outline_order_ignored_below_floor(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    a._goal = "build a snake game"

    class _Hit:
        class item:  # noqa: N801
            recipe = {"order": ["a", "b", "c"]}
        score = 0.1  # below _OPEN_DOMAIN_OUTLINE_FLOOR

    monkeypatch.setattr(a._memory, "retrieve_implementation_outline", lambda g: _Hit())
    a._seed_task_ledger_from_goal()
    assert a._todos_items == []


def test_outline_order_not_used_for_seed_edits(tmp_path, monkeypatch):
    a = _agent(tmp_path)
    a._goal = "make it better"
    a.seed_file = tmp_path / "seed.html"  # seed edit -> outline source disabled

    called = {"n": 0}

    def _boom(g):
        called["n"] += 1
        raise AssertionError("outline retrieval must not run for seed edits")

    monkeypatch.setattr(a._memory, "retrieve_implementation_outline", _boom)
    a._seed_task_ledger_from_goal()
    assert called["n"] == 0
    assert a._todos_items == []


# ---------------------------------------------------------------------------
# Harness done-marking after materialize (seeded ledgers only)
# ---------------------------------------------------------------------------


def test_mark_ledger_progress_marks_present_steps(tmp_path):
    a = _agent(tmp_path)
    a._goal = "add a pause menu, wire the sell button, draw a range circle"
    a._seed_task_ledger_from_goal()
    assert len(a._todos_items) == 3
    # The materialized file mentions every token of the first two steps
    # (pause/menu and wire/sell/button); the range-circle step is absent.
    a._current_file = "function pauseMenu(){} function wireSellButton(){ refund(); }"
    a._mark_ledger_progress()
    done = {t for d, t in a._todos_items if d}
    assert any("pause menu" in t for t in done)
    assert any("sell button" in t for t in done)
    # The range-circle step is NOT in the file -> stays open.
    assert any("range circle" in t and not d for d, t in a._todos_items)


def test_mark_ledger_progress_noop_when_model_owned(tmp_path):
    a = _agent(tmp_path)
    a._todos_items = [(False, "wire the sell button")]
    a._todos_text = "- [ ] wire the sell button"
    a._todos_seeded_by_harness = False  # model owns it
    a._current_file = "function sellButton(){}"
    a._mark_ledger_progress()
    assert a._todos_items == [(False, "wire the sell button")]


def test_capture_todos_hands_ownership_to_model(tmp_path):
    a = _agent(tmp_path)
    a._goal = "flip the ship, make the score bigger, add a restart button"
    a._seed_task_ledger_from_goal()
    assert a._todos_seeded_by_harness is True
    a._capture_todos("<todos>\n- [ ] model step one\n- [x] model step two\n</todos>")
    assert a._todos_seeded_by_harness is False
    assert len(a._todos_items) == 2


# ---------------------------------------------------------------------------
# State-anchor survival
# ---------------------------------------------------------------------------


def test_task_progress_in_state_anchor(tmp_path):
    a = _agent(tmp_path)
    a._goal = "add a pause menu, wire the sell button, draw a range circle"
    a._seed_task_ledger_from_goal()
    anchor = a._build_structured_summary()
    assert "## Task progress" in anchor
    assert "0/3 steps done" in anchor


# ---------------------------------------------------------------------------
# Phase 4B — gated one-objective-first-build nudge
# ---------------------------------------------------------------------------


def _arm_outline_ledger(a):
    """Put the agent in the state _seed_task_ledger_from_goal leaves after an
    outline_order seed (>=3 steps), pretending the backend is local."""
    a._is_local_backend = lambda: True
    a._todos_items = [(False, "build the path"), (False, "place towers"),
                      (False, "spawn waves")]
    a._todos_seeded_by_harness = True
    a._ledger_source = "outline_order"


def test_one_objective_nudge_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ONE_STEP_FIRST_BUILD", raising=False)
    a = _agent(tmp_path)
    _arm_outline_ledger(a)
    assert a._one_objective_first_build_nudge() == ""


def test_one_objective_nudge_fires_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ONE_STEP_FIRST_BUILD", "1")
    a = _agent(tmp_path)
    _arm_outline_ledger(a)
    out = a._one_objective_first_build_nudge()
    assert "ONE-OBJECTIVE FIRST BUILD" in out
    assert "build the path" in out  # step 1 named


def test_one_objective_nudge_skips_goal_clause_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ONE_STEP_FIRST_BUILD", "1")
    a = _agent(tmp_path)
    _arm_outline_ledger(a)
    a._ledger_source = "goal_clauses"  # not a strong outline match
    assert a._one_objective_first_build_nudge() == ""


def test_one_objective_nudge_skips_cloud_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ONE_STEP_FIRST_BUILD", "1")
    a = _agent(tmp_path)
    _arm_outline_ledger(a)
    a._is_local_backend = lambda: False  # cloud/large backend
    assert a._one_objective_first_build_nudge() == ""
