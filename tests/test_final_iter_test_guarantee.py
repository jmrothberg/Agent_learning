"""Tests for the final-iter test guarantee added to agent.GameAgent.

DK trace 20260513_185815 rationale: the agent shipped a correct full
<html_file> on Turn 14 that was never tested because the loop hit its
max_iters budget while in the rejection branch. The final-iter test
guarantee ensures `_last_materialized_iter > _last_tested_iter` at any
exit point triggers one closing browser test, so the outcome reflects
the latest shipped code, not a stale snapshot.

The verifier is async + Chromium, so this test uses a fake browser
double to exercise the logic without launching a real browser.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class _FakeBrowser:
    """Captures load_and_test calls; returns canned reports."""

    def __init__(self, ok: bool):
        self._ok = ok
        self.calls: list[dict] = []

    async def load_and_test(self, path, **kwargs):  # noqa: D401
        self.calls.append({"path": str(path), "kwargs": kwargs})
        # Minimal report shape — the helper only reads .get("ok") and
        # passes the dict to format_report_for_model.
        return {
            "ok": self._ok,
            "page_errors": [],
            "errors": [],
            "soft_warnings": [],
            "warnings": [],
            "title": "fake",
            "canvas": None,
            "input_listeners": {},
            "input_test": None,
            "frozen_canvas": None,
            "body_chars": 0,
            "body_sample": "",
            "logs": [],
            "probes": [],
            "screenshot": None,
            "screenshot_before": None,
        }


def _make_agent_for_test(tmpdir: Path):
    """Build a minimum-viable GameAgent without spinning up the model
    backend. Skips __init__ via __new__ + manual field population."""
    from agent import GameAgent

    a = GameAgent.__new__(GameAgent)
    a.out_path = tmpdir / "out.html"
    a.best_path = tmpdir / "out.best.html"
    a.trace_path = tmpdir / "traces" / "out.jsonl"
    a.trace_path.parent.mkdir(parents=True, exist_ok=True)
    a._session_id = "test_session"
    a._current_file = ""
    a._last_materialized_iter = 0
    a._last_tested_iter = 0
    a._last_report_summary = ""
    a._last_test_report = None
    a._probes = []
    a._criteria = ""
    a._recorded_events: list = []  # consumed by _record but irrelevant here

    def _record(ev):
        a._recorded_events.append(ev)
        return ev

    def _trace(obj):
        pass

    def _save_best(html):
        try:
            a.best_path.write_text(html, encoding="utf-8")
            return a.best_path
        except Exception:
            return None

    a._record = _record
    a._trace = _trace
    a._save_best = _save_best
    return a


async def _drain(agen):
    return [ev async for ev in agen]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_skip_when_nothing_materialized():
    """No materialize happened → no test should run."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=True)
        # _last_materialized_iter = 0 → guard skips.
        events = asyncio.run(_drain(a._final_iter_test_if_needed()))
        assert events == []
        assert a.browser.calls == []


def test_skip_when_already_tested():
    """Materialized and tested at the same iter → guard skips."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=True)
        a._current_file = "<!DOCTYPE html><html><body>hi</body></html>"
        a._last_materialized_iter = 3
        a._last_tested_iter = 3
        events = asyncio.run(_drain(a._final_iter_test_if_needed()))
        assert events == []
        assert a.browser.calls == []


def test_runs_test_when_materialized_beyond_tested():
    """The DK-trace scenario: latest code was materialized on iter 14
    but the verifier only saw iter 13 (or earlier). Guard MUST fire."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=True)
        a._current_file = "<!DOCTYPE html><html><body>final</body></html>"
        a._last_materialized_iter = 14
        a._last_tested_iter = 13
        events = asyncio.run(_drain(a._final_iter_test_if_needed()))
        # One browser test call ran.
        assert len(a.browser.calls) == 1
        # _last_tested_iter caught up.
        assert a._last_tested_iter == 14
        # A test event was emitted for the UI.
        kinds = [getattr(ev, "kind", None) for ev in events]
        assert "test" in kinds


def test_promotes_to_best_on_final_pass():
    """When the final test passes and best.html doesn't yet exist
    (session was failing up to the final emit), promote so
    `_record_session_outcome(ok=best.exists())` reports ok=True."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=True)
        a._current_file = "<!DOCTYPE html><html><body>winner</body></html>"
        a._last_materialized_iter = 5
        a._last_tested_iter = 4
        assert not a.best_path.exists()
        asyncio.run(_drain(a._final_iter_test_if_needed()))
        # best.html now reflects the final passing code.
        assert a.best_path.exists()
        assert "winner" in a.best_path.read_text(encoding="utf-8")


def test_no_best_promotion_on_final_fail():
    """If the final test fails, don't lie about success by promoting
    a broken file to best."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=False)
        a._current_file = "<!DOCTYPE html><html><body>broken</body></html>"
        a._last_materialized_iter = 5
        a._last_tested_iter = 4
        asyncio.run(_drain(a._final_iter_test_if_needed()))
        assert not a.best_path.exists()


def test_no_op_when_current_file_empty():
    """Materialize-iter advanced but current_file is empty (defensive
    guard) — don't write an empty file to disk."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a = _make_agent_for_test(tmp)
        a.browser = _FakeBrowser(ok=True)
        a._current_file = ""
        a._last_materialized_iter = 5
        a._last_tested_iter = 4
        events = asyncio.run(_drain(a._final_iter_test_if_needed()))
        assert events == []
        assert a.browser.calls == []
