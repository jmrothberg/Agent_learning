"""failure_class routing bank — where triage should send edits first."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_BANK = Path(__file__).parent.parent / "eval" / "failure_class_routing.jsonl"
_REPO = Path(__file__).parent.parent


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for line in _BANK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("row", _load_rows(), ids=lambda r: r["failure_class"])
def test_failure_class_edit_targets_exist(row: dict) -> None:
    for rel in row.get("edit_first") or []:
        p = _REPO / rel
        assert p.exists(), f"{row['failure_class']}: missing edit target {rel}"


def test_bank_covers_known_failure_classes() -> None:
    classes = {r["failure_class"] for r in _load_rows()}
    assert classes >= {"harness_bug", "memory_gap", "local_llm_limit", "none"}
