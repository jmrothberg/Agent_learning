"""Phase 1 seed-edit scenario bank (stub, no model).

Locks the seed-edit classifiers that arm the single-patch scope lock and the
orientation/size fast-paths (agent._goal_is_small_scope_edit and friends), so
a refactor of those heuristics cannot silently change how a weak local model
is steered on a seed first build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402

_BANK = Path(__file__).parent.parent / "eval" / "seed_edit_scenarios.jsonl"
_FIXTURE = Path(__file__).parent.parent / "eval" / "fixtures" / "seed_tower_defense.html"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _actual(key: str, goal: str) -> bool:
    if key == "small_scope_edit":
        return agent._goal_is_small_scope_edit(goal)
    if key == "orientation_change":
        return agent._feedback_is_orientation_change(goal)
    if key == "size_change":
        return agent._feedback_requests_size_change(goal)
    raise KeyError(key)


def test_seed_fixture_exists_and_loads():
    """The opt-in GLM eval (eval/eval_seed_edits.py) patches this fixture."""
    assert _FIXTURE.exists()
    html = _FIXTURE.read_text(encoding="utf-8")
    assert "window.state" in html and "startWave" in html


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["name"])
def test_seed_edit_classification(row):
    goal = row["goal"]
    for key, expected in row["expect"].items():
        got = _actual(key, goal)
        assert got == expected, (
            f"{row['name']}: {key} expected {expected} got {got}\n  goal={goal!r}"
        )
