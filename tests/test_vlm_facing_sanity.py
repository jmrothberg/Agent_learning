"""Tests for eval/vlm_facing_sanity.py (no MLX / Chromium)."""
from __future__ import annotations

from eval.vlm_facing_sanity import facing_sanity_passes, parse_facing_answer


def test_facing_sanity_passes_only_on_no() -> None:
    assert facing_sanity_passes("no") is True
    assert facing_sanity_passes("yes") is False
    assert facing_sanity_passes("unclear") is False
    assert facing_sanity_passes(None) is False


def test_parse_facing_answer_yes_no() -> None:
    p = parse_facing_answer("Q1: YES")
    assert p["q_answer"] == "yes"
    assert p["parse_rate"] == 1.0
    p2 = parse_facing_answer("Q1: 1: NO")
    assert p2["q_answer"] == "no"
