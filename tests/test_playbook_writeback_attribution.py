"""Playbook write-back attribution gate (trace 20260613_213711).

The online playbook learner used to bump `harmful` on every injected bullet
whenever the stuck-streak hit 3 — regardless of WHY the iter failed. In the
Opus 4.8 Dragon's-Lair run the blocker was a model-authored `<probes>`
failure (`state_room0` racing the input smoke test), which has nothing to do
with the QTE playbook bullets, yet those hand-curated seeds were penalized to
harmful=2.

`GameAgent._failure_blames_code(report)` gates the penalty: only genuine code
defects (page errors, FROZEN-CANVAS, ENTITY-NOT-RENDERED, controls-not-wired)
count — not model-probe / coverage-gap authoring artifacts. These tests pin
that behavior as a pure function (no model / Chromium).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_model_probe_failure_alone_does_not_blame_code():
    report = {
        "page_errors": [],
        "soft_warnings": ["PROBE FAILED [state_room0]: `state.room===0` — false."],
        "probes": [{"name": "state_room0", "ok": False}],
    }
    assert GameAgent._failure_blames_code(report) is False


def test_coverage_gap_synthetic_probe_does_not_blame_code():
    report = {
        "soft_warnings": ["PROBE FAILED [coverage_gap__restart]: synthetic."],
        "probes": [{"name": "coverage_gap__restart", "ok": False}],
    }
    assert GameAgent._failure_blames_code(report) is False


def test_page_error_blames_code():
    report = {"page_errors": ["TypeError: x is undefined"], "soft_warnings": []}
    assert GameAgent._failure_blames_code(report) is True


def test_frozen_canvas_blames_code():
    report = {"soft_warnings": ["FROZEN-CANVAS: 32x32 hash unchanged ..."]}
    assert GameAgent._failure_blames_code(report) is True


def test_controls_not_wired_input_responsive_blames_code():
    # The harness-synthesized behavioral probe is a real defect even though
    # it carries the PROBE FAILED prefix.
    report = {
        "soft_warnings": ["PROBE FAILED [input_responsive]: controls not wired."],
        "probes": [{"name": "input_responsive", "ok": False}],
    }
    assert GameAgent._failure_blames_code(report) is True


def test_entity_not_rendered_blames_code():
    report = {"soft_warnings": ["ENTITY-NOT-RENDERED [player]: in state but not drawn."]}
    assert GameAgent._failure_blames_code(report) is True


def test_clean_report_does_not_blame_code():
    report = {"page_errors": [], "soft_warnings": [], "probes": []}
    assert GameAgent._failure_blames_code(report) is False


def test_dropped_assets_pending_does_not_blame_code():
    report = {
        "soft_warnings": [
            "ASSETS_DROPPED_PENDING [hazard_dragon_jaw]: harness per-turn cap",
        ],
    }
    assert GameAgent._failure_blames_code(report) is False
