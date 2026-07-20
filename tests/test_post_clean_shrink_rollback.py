"""Post-clean shrink rollback — reject truncated materialize that would
destroy a green build; let well-formed shrinks through to browser judgment.

Castle courtyard 20260720_193459: truncated mega-patch applied, file shrank
25723→10276 with unclosed <html>/<body>. Guard must reject before write.
A well-formed shrink (user asked to simplify) must NOT be rejected — it is
written and coached via post_clean_shrink_detected instead, so the guard
cannot thrash a legitimate simplification request.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_helpers import (  # noqa: E402
    _truncation_reason,
    should_reject_post_clean_shrink,
)
from agent import GameAgent  # noqa: E402


def test_should_reject_post_clean_shrink_castle_ratio() -> None:
    # Castle iter 3: 25723 → 10276 (~40% kept), truncated, after a clean iter.
    assert should_reject_post_clean_shrink(
        previous_ok=True, before_bytes=25723, after_bytes=10276, truncated=True,
    )


def test_well_formed_shrink_is_not_rejected() -> None:
    # Intentional simplification: big shrink but structurally complete —
    # must be written (browser probes / auto-revert judge it), never
    # rejected, or a legitimate "remove the second level" request thrashes.
    assert not should_reject_post_clean_shrink(
        previous_ok=True, before_bytes=25723, after_bytes=10276, truncated=False,
    )


def test_should_reject_under_threshold_keeps() -> None:
    assert not should_reject_post_clean_shrink(
        previous_ok=True, before_bytes=10000, after_bytes=8500, truncated=True,
    )


def test_should_reject_requires_previous_ok() -> None:
    assert not should_reject_post_clean_shrink(
        previous_ok=False, before_bytes=25723, after_bytes=10276, truncated=True,
    )
    assert not should_reject_post_clean_shrink(
        previous_ok=None, before_bytes=25723, after_bytes=10276, truncated=True,
    )


def test_should_reject_zero_before_bytes() -> None:
    assert not should_reject_post_clean_shrink(
        previous_ok=True, before_bytes=0, after_bytes=0, truncated=True,
    )


def test_agent_rejects_truncated_shrink_without_writing(tmp_path: Path) -> None:
    """Materialize-path reject: working file on disk stays intact."""
    out = tmp_path / "game.html"
    baseline = "<!DOCTYPE html><html><body><canvas></canvas><script>\n" + (
        "x=1;\n" * 400
    ) + "</script></body></html>"
    out.write_text(baseline, encoding="utf-8")
    a = GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )
    a._current_file = baseline
    a._previous_report_ok = True
    traces: list[dict] = []
    a._trace = traces.append  # type: ignore[method-assign]

    # Truncated stub — same shape the castle iter 3 wrote to disk.
    shrunk = "<!DOCTYPE html><html><body><script>x"
    trunc = _truncation_reason(shrunk)
    assert trunc is not None

    # Simulate the loop body after _materialize returns shrunk HTML.
    new_html: str | None = shrunk
    before = len(a._current_file or "")
    if should_reject_post_clean_shrink(
        previous_ok=a._previous_report_ok,
        before_bytes=before,
        after_bytes=len(new_html),
        truncated=trunc is not None,
    ):
        a._post_clean_shrink_detected = True
        a._trace({
            "kind": "post_clean_shrink_rollback",
            "iteration": 3,
            "before_bytes": before,
            "after_bytes": len(new_html),
            "reason": trunc,
            "action": "rollback",
        })
        new_html = None

    assert new_html is None
    assert a._post_clean_shrink_detected is True
    assert out.read_text(encoding="utf-8") == baseline
    assert a._current_file == baseline
    kinds = [t.get("kind") for t in traces]
    assert "post_clean_shrink_rollback" in kinds
    assert "post_clean_shrink_detected" not in kinds
    ev = next(t for t in traces if t["kind"] == "post_clean_shrink_rollback")
    assert ev["action"] == "rollback"
    assert ev["reason"] == "unclosed <html>"
    assert ev["before_bytes"] == before
    assert ev["after_bytes"] == len(shrunk)


def test_agent_writes_well_formed_shrink(tmp_path: Path) -> None:
    """A complete smaller file passes the guard (detect-only path)."""
    baseline = "<!DOCTYPE html><html><body><canvas></canvas><script>\n" + (
        "x=1;\n" * 400
    ) + "</script></body></html>"
    shrunk = (
        "<!DOCTYPE html><html><body><canvas></canvas>"
        "<script>x=1;</script></body></html>"
    )
    assert _truncation_reason(shrunk) is None
    assert not should_reject_post_clean_shrink(
        previous_ok=True,
        before_bytes=len(baseline),
        after_bytes=len(shrunk),
        truncated=_truncation_reason(shrunk) is not None,
    )
