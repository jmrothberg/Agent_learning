"""Auto-revert must not undo honored user feedback on probe-count alone.

DOOM3DFI 20260902_134411 iter 3: model baked transparent floor/ceil PNG
margins opaque (correct). A brittle ``floor_not_solid`` variance probe
failed → auto-revert restored the gappy iter-2 file. These tests pin the
decision helper so that failure mode stays fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_skip_probe_only_drop_when_user_feedback_honored() -> None:
    should, tag = GameAgent._auto_revert_should_fire(
        prev_ok=True,
        current_ok=False,
        new_page_errors=False,
        fewer_probes=True,
        new_coverage_gaps=False,
        user_feedback_honored=True,
    )
    assert should is False
    assert tag == "skip_probe_only_user_feedback"


def test_still_revert_probe_drop_without_user_feedback() -> None:
    should, tag = GameAgent._auto_revert_should_fire(
        prev_ok=True,
        current_ok=False,
        new_page_errors=False,
        fewer_probes=True,
        new_coverage_gaps=False,
        user_feedback_honored=False,
    )
    assert should is True
    assert tag == "fewer_probes"


def test_still_revert_on_new_page_errors_even_with_user_feedback() -> None:
    should, tag = GameAgent._auto_revert_should_fire(
        prev_ok=True,
        current_ok=False,
        new_page_errors=True,
        fewer_probes=True,
        new_coverage_gaps=False,
        user_feedback_honored=True,
    )
    assert should is True
    assert tag == "hard_regression"


def test_still_revert_on_new_coverage_gaps_with_user_feedback() -> None:
    should, tag = GameAgent._auto_revert_should_fire(
        prev_ok=True,
        current_ok=False,
        new_page_errors=False,
        fewer_probes=False,
        new_coverage_gaps=True,
        user_feedback_honored=True,
    )
    assert should is True
    assert tag == "hard_regression"


def test_no_revert_when_prev_not_ok_or_current_ok() -> None:
    should, _ = GameAgent._auto_revert_should_fire(
        prev_ok=False,
        current_ok=False,
        new_page_errors=False,
        fewer_probes=True,
        new_coverage_gaps=False,
        user_feedback_honored=False,
    )
    assert should is False
    should2, _ = GameAgent._auto_revert_should_fire(
        prev_ok=True,
        current_ok=True,
        new_page_errors=False,
        fewer_probes=True,
        new_coverage_gaps=False,
        user_feedback_honored=False,
    )
    assert should2 is False


def test_agent_source_wires_auto_revert_skip() -> None:
    """Guard: loop must call the helper and trace auto_revert_skipped."""
    src = (Path(__file__).resolve().parents[1] / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "_auto_revert_should_fire" in src
    assert "auto_revert_skipped" in src
    assert "skip_probe_only_user_feedback" in src
