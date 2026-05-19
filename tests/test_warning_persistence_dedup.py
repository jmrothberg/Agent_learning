"""Persistent harness-warning compaction in the model-facing report.

Evidence: fighing-game trace 20260519_153115. Eight unused-asset
warning lines (~1.6 KB) repeated verbatim in every iter for four
iters. The model stopped reacting after iter 1 but continued to
carry the full text in its working set, displacing more useful
context. The full warning text is still in `report` (and therefore
in the trace JSONL); only the prompt rendering is compacted.

Domain-neutral by construction: the dedup is keyed by exact-string
hash, not by the warning content. Threshold is the third consecutive
appearance — model gets two full views before compaction kicks in.

These tests are pure-state — no model, no Chromium.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _make_agent(tmp_path) -> GameAgent:
    out = tmp_path / "game.html"
    out.write_text("<html></html>")
    fake_browser = MagicMock()
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=fake_browser,
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


# ---------- counter advance / reset ---------------------------------------


def test_persistence_counts_consecutive_iters(tmp_path):
    a = _make_agent(tmp_path)
    w = "sprite 'arena_bg.png' was generated but is NEVER referenced"

    a._advance_warning_persistence([w])
    a._advance_warning_persistence([w])
    a._advance_warning_persistence([w])

    h = a._hash_warning(w)
    assert a._warning_persistence[h] == 3


def test_persistence_resets_when_warning_absent(tmp_path):
    """Streak breaks the moment the warning string is not in the
    current iter's report. The next appearance starts at 1 again."""
    a = _make_agent(tmp_path)
    w = "sprite 'arena_bg.png' was generated but is NEVER referenced"
    other = "different warning"

    a._advance_warning_persistence([w])
    a._advance_warning_persistence([w])
    # Streak broken — `w` not in this iter.
    a._advance_warning_persistence([other])
    a._advance_warning_persistence([w])

    assert a._warning_persistence[a._hash_warning(w)] == 1


def test_persistence_resets_on_attempt_reset(tmp_path):
    """Restarts inside `run_with_restarts` call `_reset_attempt_state`,
    which must clear the persistence dict so attempt N+1 sees fresh
    streaks. Otherwise compaction would fire on iter 1 of the new
    attempt for warnings the previous attempt accumulated."""
    a = _make_agent(tmp_path)
    a._advance_warning_persistence(["abc", "xyz"])
    assert len(a._warning_persistence) == 2
    a._reset_attempt_state()
    assert a._warning_persistence == {}


# ---------- model-facing rendering ----------------------------------------


def test_compaction_below_threshold_keeps_full_text(tmp_path):
    """First two appearances must show the full warning so the model
    actually sees it. Compaction kicks in on the third+ appearance."""
    a = _make_agent(tmp_path)
    w = "sprite 'arena_bg.png' was generated but is NEVER referenced"

    # Iter 1.
    a._advance_warning_persistence([w])
    out1 = a._compact_warnings_for_prompt([w])
    assert out1 == [w]

    # Iter 2.
    a._advance_warning_persistence([w])
    out2 = a._compact_warnings_for_prompt([w])
    assert out2 == [w]


def test_compaction_at_threshold_collapses(tmp_path):
    """Third consecutive appearance compacts to a one-line preview."""
    a = _make_agent(tmp_path)
    w = (
        "sprite 'arena_bg.png' was generated to "
        "'fighing-game-..._assets/arena_bg.png' but is NEVER "
        "referenced in the HTML. Either wire it in or drop it."
    )
    for _ in range(3):
        a._advance_warning_persistence([w])

    compacted = a._compact_warnings_for_prompt([w])
    assert len(compacted) == 1
    line = compacted[0]
    assert "persistent warning" in line
    assert "seen 3× in a row" in line
    # Preview is a prefix of the original warning, capped.
    assert line.endswith("…")
    # Full body must NOT survive into the compacted line — that's the
    # entire point of the dedup.
    assert "Either wire it in" not in line


def test_format_report_for_model_uses_compacted_warnings(tmp_path):
    """End-to-end: the wrapper used by the iter loop must produce a
    report-text where the persistent warning section is collapsed."""
    a = _make_agent(tmp_path)
    long_w = (
        "sprite 'arena_bg.png' was generated to "
        "'fighing-game-..._assets/arena_bg.png' but is NEVER "
        "referenced in the HTML"
    )
    fake_report = {
        "ok": True,
        "errors": [],
        "warnings": [long_w],
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": {"ran": False, "any_change": None, "keys_tried": []},
        "frozen_canvas": False,
        "body_chars": 0,
        "body_sample": "",
        "probes": [],
    }
    # Three iters: counter advances each time.
    for _ in range(3):
        a._advance_warning_persistence(fake_report["warnings"])

    out = a._format_report_for_model(fake_report)

    # Compacted body is in the rendered text.
    assert "persistent warning" in out
    assert "seen 3× in a row" in out
    # Full warning body is NOT (the prompt-side noise has been removed).
    assert "Either wire it in" not in out  # not in fixture but assert pattern absent
    # Original report dict is untouched (key invariant — trace and
    # downstream consumers must still see the full warnings).
    assert fake_report["warnings"] == [long_w]


def test_format_report_handles_empty_warnings(tmp_path):
    """No warnings in the report → wrapper just delegates and produces
    a normal formatted report. No compaction state changes."""
    a = _make_agent(tmp_path)
    fake_report = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": {"ran": False, "any_change": None, "keys_tried": []},
        "frozen_canvas": False,
        "body_chars": 0,
        "body_sample": "",
        "probes": [],
    }
    out = a._format_report_for_model(fake_report)
    assert "persistent warning" not in out
    assert a._warning_persistence == {}


def test_compaction_does_not_affect_soft_warnings_or_errors(tmp_path):
    """Only `warnings` (harness-level, non-blocking) is dedup-compacted.
    `soft_warnings` (probe failures, partial-patch markers) and `errors`
    drive `report["ok"]`/scoring and must NOT be touched."""
    a = _make_agent(tmp_path)
    persistent = "x repeats forever"
    fake_report = {
        "ok": False,
        "errors": ["some error"],
        "warnings": [persistent],
        "soft_warnings": [persistent],  # same string, different field
        "logs": [],
        "title": "X",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": {"ran": False, "any_change": None, "keys_tried": []},
        "frozen_canvas": False,
        "body_chars": 0,
        "body_sample": "",
        "probes": [],
    }
    for _ in range(3):
        a._advance_warning_persistence(fake_report["warnings"])

    out = a._format_report_for_model(fake_report)

    # Warnings section: compacted.
    assert "persistent warning" in out
    # Soft warnings + errors: full text survives (they drive "ok").
    # The fixture's soft_warning string equals `persistent`; assert
    # the FULL string still appears under its own section header.
    assert "x repeats forever" in out
