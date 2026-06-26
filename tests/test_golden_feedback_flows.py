"""Golden feedback-flow regression bank from trace 20260626_102307
(Fieldrunners tower-defense, GLM-5.2, ~15 tok/s).

This trace is the POSITIVE reference: iters 2-3 did exactly what the user
asked (art regen -> 3/3 patches, then a 1/1 behavior-bug sell fix), and
iters 4-5 are the FAILURE inputs we are hardening recovery around.

These are pure-function stubs over the harness feedback classifiers
(`agent._feedback_is_*`). They lock the *classification* that drives the
router so a future refactor of the predicates cannot silently break the
working feedback->asset->patch flow (golden iters 2-3) or change how the
failure inputs (iters 4-5) are categorized.

No model, no browser. Rows live in eval/golden_feedback_flows.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402

_BANK = Path(__file__).parent.parent / "eval" / "golden_feedback_flows.jsonl"


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for line in _BANK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# Map each expectation key to the predicate that produces it. Predicates
# that take the asset roster get it; the rest take text only.
def _actual(key: str, text: str, asset_names: list[str]) -> bool:
    if key == "art_change":
        return agent._feedback_is_art_change(text, asset_names)
    if key == "behavior_bug":
        return agent._feedback_is_behavior_bug(text)
    if key == "orientation_change":
        return agent._feedback_is_orientation_change(text)
    if key == "requests_existing_media":
        return agent._feedback_requests_existing_media(text)
    if key == "ui_feature":
        return agent._feedback_is_ui_feature(text)
    raise KeyError(key)


def test_bank_loads_and_is_nonempty():
    rows = _load_rows()
    assert rows, "golden_feedback_flows.jsonl is empty"
    # Both golden-success rows (iters 2-3) must be present — they are the
    # must-not-break reference.
    roles = {r["role"] for r in rows}
    assert "golden_success" in roles


@pytest.mark.parametrize("row", _load_rows(), ids=lambda r: r["name"])
def test_golden_feedback_classification(row):
    text = row["feedback"]
    asset_names = row.get("asset_names", [])
    for key, expected in row["expect"].items():
        actual = _actual(key, text, asset_names)
        assert actual == expected, (
            f"{row['name']}: {key} expected {expected} got {actual}\n"
            f"  feedback={text!r}"
        )


def test_golden_iter2_routes_art_not_behavior():
    """Iter 2 (user-approved): unique-sprite request must read as an
    art change and NOT a behavior bug, or the router would skip asset
    generation."""
    rows = {r["name"]: r for r in _load_rows()}
    r = rows["iter2_unique_sprites"]
    assert agent._feedback_is_art_change(r["feedback"], r["asset_names"]) is True
    assert agent._feedback_is_behavior_bug(r["feedback"]) is False


def test_golden_iter3_routes_behavior_not_art():
    """Iter 3 (user-approved): the sell-button bug must read as a
    behavior bug (patch-only), not an art change."""
    rows = {r["name"]: r for r in _load_rows()}
    r = rows["iter3_sell_refund"]
    assert agent._feedback_is_behavior_bug(r["feedback"]) is True
    assert agent._feedback_is_art_change(r["feedback"], r["asset_names"]) is False


def test_golden_iter5_retry_is_not_itself_an_art_request():
    """A bare retry nudge is correctly NOT an art request on its own.
    This is exactly why the router dropped the standing iter-4 art ask;
    Phase 0D-6 must retain the standing request despite this False."""
    rows = {r["name"]: r for r in _load_rows()}
    r = rows["iter5_retry_keeps_art"]
    assert agent._feedback_is_art_change(r["feedback"], r["asset_names"]) is False
    assert agent._feedback_is_behavior_bug(r["feedback"]) is False


# ---------------------------------------------------------------------------
# Phase 4 (4A): feedback-classification EXTENSION layer (data-editable vocab).
# ---------------------------------------------------------------------------


def test_feedback_patterns_extension_is_consumed():
    """The classifier UNIONS code vocab with memory/feedback_patterns.jsonl.
    'reskin the tileset' uses ONLY data-file tokens (reskin=media_verb,
    tileset=art_noun) — neither is in agent._MEDIA_VERBS / _ART_NOUNS — so a
    True classification proves the data file is read at runtime."""
    assert "reskin" not in agent._MEDIA_VERBS
    assert "tileset" not in agent._ART_NOUNS
    assert agent._feedback_is_art_change("reskin the tileset", []) is True


def test_feedback_patterns_row_change_changes_classification(monkeypatch):
    """Editing a row changes classification: with the loader stubbed empty,
    the data-only phrase no longer classifies as an art change."""
    import memory as memory_module
    monkeypatch.setattr(memory_module, "_FEEDBACK_PATTERNS_CACHE", {}, raising=False)
    monkeypatch.setattr(memory_module, "load_feedback_patterns", lambda c: ())
    assert agent._feedback_is_art_change("reskin the tileset", []) is False
