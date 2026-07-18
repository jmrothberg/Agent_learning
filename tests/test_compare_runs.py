"""compare_runs offline metrics — infra vs never_clean separation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.compare_runs import _trace_metrics  # noqa: E402


def test_infra_crash_not_counted_as_never_clean(tmp_path: Path):
    run = tmp_path / "run_x"
    traces = run / "traces"
    traces.mkdir(parents=True)
    # Backend crash: retrieval + stream_done.crashed, zero iter_summary.
    (traces / "crash__run_1.jsonl").write_text(
        json.dumps({"kind": "playbook_retrieved", "ids": ["b1"]}) + "\n"
        + json.dumps({"kind": "stream_done", "crashed": True, "tokens": 0}) + "\n"
        + json.dumps({
            "kind": "session_outcome", "ok": False,
            "code_materialized": False, "backend_crashed": True,
        }) + "\n",
        encoding="utf-8",
    )
    # Clean pass for denominator sanity.
    (traces / "ok__run_1.jsonl").write_text(
        json.dumps({"kind": "iter_summary", "iteration": 1, "ok": True,
                     "retrieved_ids": ["b2"], "tok_per_s": 10.0}) + "\n"
        + json.dumps({
            "kind": "session_outcome", "ok": True,
            "code_materialized": True, "backend_crashed": False,
        }) + "\n",
        encoding="utf-8",
    )
    metrics = _trace_metrics(run)
    assert metrics["infra_failed"] == 1
    assert metrics["never_clean"] == 0
    assert metrics["avg_first_clean"] == 1.0
