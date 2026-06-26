"""Phase 1 modality/gate scenario bank (stub, no model).

Data-driven companion to test_modality_disambiguation.py /
test_action_gate_non_combat_keys.py: each row in
eval/modality_scenarios.jsonl pins the beat-em-up suppression and the
non-combat-key exclusion so new cases are added as data, not code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts_v1 import _detect_beat_em_up_intent  # noqa: E402
from tools import _non_combat_action_keys  # noqa: E402

_BANK = Path(__file__).parent.parent / "eval" / "modality_scenarios.jsonl"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["name"])
def test_modality_scenario(row):
    expect = row["expect"]
    if "beat_em_up" in expect:
        hits = _detect_beat_em_up_intent(row["goal"])
        got = bool(hits)
        assert got == expect["beat_em_up"], (
            f"{row['name']}: beat_em_up expected {expect['beat_em_up']} "
            f"got {got} (hits={hits})"
        )
    if "non_combat_keys" in expect:
        crit = row.get("criteria", "")
        got_keys = _non_combat_action_keys(crit)
        assert got_keys == set(expect["non_combat_keys"]), (
            f"{row['name']}: non_combat_keys expected "
            f"{sorted(expect['non_combat_keys'])} got {sorted(got_keys)}"
        )
