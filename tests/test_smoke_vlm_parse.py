"""Unit tests for VLM checklist parsing in asset-decode smoke script."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._smoke_asset_decode_settle import (  # noqa: E402
    _parse_vlm_checklist,
    _vlm_checklist_passes,
)


def test_vlm_checklist_passes_on_q1_q2_yes() -> None:
    raw = "Q1: YES\nQ2: YES\nQ3: UNCLEAR"
    ok, _ = _vlm_checklist_passes(raw)
    assert ok


def test_vlm_checklist_fails_on_q1_no() -> None:
    raw = "Q1: NO\nQ2: YES\nQ3: YES"
    ok, _ = _vlm_checklist_passes(raw)
    assert not ok


def test_vlm_bare_yes_fallback() -> None:
    ok, _ = _vlm_checklist_passes("YES")
    assert ok


def test_parse_vlm_checklist_tolerant_format() -> None:
    ans = _parse_vlm_checklist("Q1. yes\n2) NO\nQ3 - unclear")
    assert ans[1] == "yes"
    assert ans[2] == "no"
    assert ans[3] == "unclear"
