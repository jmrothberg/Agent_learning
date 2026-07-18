"""Offline playbook credit + coder writeback-on regression."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import credit_bullets as cb  # noqa: E402
from memory import Bullet, Playbook  # noqa: E402


def test_credit_from_trace_helpful_when_first_clean():
    records = [
        {"kind": "playbook_retrieved", "ids": ["a", "b"]},
        {"kind": "iter_summary", "iteration": 1, "ok": False},
        {"kind": "iter_summary", "iteration": 2, "ok": True},
    ]
    helpful, harmful = cb.credit_from_trace(records)
    assert helpful == ["a", "b"]
    assert harmful == []


def test_credit_from_trace_harmful_when_never_clean():
    records = [
        {"kind": "playbook_retrieved", "ids": ["x"]},
        {"kind": "iter_summary", "iteration": 1, "ok": False},
        {"kind": "iter_summary", "iteration": 2, "ok": False},
    ]
    helpful, harmful = cb.credit_from_trace(records)
    assert helpful == []
    assert harmful == ["x"]


def test_credit_skips_backend_crash_session():
    """Infra crash with retrieval must not poison playbook harmful counters."""
    records = [
        {"kind": "playbook_retrieved", "ids": ["x"]},
        {"kind": "stream_done", "crashed": True, "tokens": 0},
        {"kind": "session_outcome", "ok": False,
         "code_materialized": False, "backend_crashed": True},
    ]
    helpful, harmful = cb.credit_from_trace(records)
    assert helpful == [] and harmful == []


def test_credit_batch_dedupes_via_ledger(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    pb = Playbook(base_root=str(mem))
    pb.add(Bullet(id="b1", content="rule one", tags=["t"]))

    batch = tmp_path / "batch"
    traces = batch / "traces"
    traces.mkdir(parents=True)
    tpath = traces / "game__run_1.jsonl"
    tpath.write_text(
        json.dumps({"kind": "playbook_retrieved", "ids": ["b1"]}) + "\n"
        + json.dumps({"kind": "iter_summary", "iteration": 1, "ok": True}) + "\n",
        encoding="utf-8",
    )

    r1 = cb.credit_batch(batch, memory_root=mem, dry_run=False)
    assert "game__run_1" in r1["credited_traces"]
    assert r1["helpful_deltas"].get("b1") == 1

    bullets = pb.load_all()
    assert bullets[0].helpful == 1

    r2 = cb.credit_batch(batch, memory_root=mem, dry_run=False)
    assert r2["credited_traces"] == []
    assert "game__run_1" in r2["skipped_traces"]
    # No double credit
    assert pb.load_all()[0].helpful == 1


def test_coder_playbook_writeback_defaults_on():
    """Tune path (coder.py) must keep playbook writeback ON by default."""
    src = Path("coder.py").read_text(encoding="utf-8")
    assert "playbook_writeback: bool = True" in src
    assert "--no-playbook-writeback" in src
