"""Unit tests for system_tests.py (no qwen / Chromium)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import system_tests as st


def test_load_system_battery(tmp_path: Path) -> None:
    bat = tmp_path / "battery.jsonl"
    bat.write_text(
        '{"slug": "a", "goal": "g", "suite": "smoke", "pass_criteria": {"input_responsive": true}}\n',
        encoding="utf-8",
    )
    tests = st.load_system_battery(bat)
    assert len(tests) == 1
    assert tests[0].slug == "a"
    assert tests[0].pass_criteria["input_responsive"] is True


def test_filter_tests_suite() -> None:
    tests = [
        st.SystemBatteryTest("smoke-move", "g1", suite="smoke"),
        st.SystemBatteryTest("pacman-hard", "g2", suite="pacman"),
    ]
    smoke = st.filter_tests(tests, suite="smoke", slugs=None)
    assert [t.slug for t in smoke] == ["smoke-move"]
    full = st.filter_tests(tests, suite="full", slugs=None)
    assert len(full) == 2


def test_validate_three_model_endpoints_collapsed(monkeypatch) -> None:
    monkeypatch.setattr(
        st.backend_mod, "ollama_endpoint_url",
        lambda slot=1: "http://127.0.0.1:11434",
    )
    monkeypatch.setattr(st.backend_mod, "_endpoint_ready", lambda _ep: True)
    ok, errors, eps = st.validate_three_model_endpoints()
    assert not ok
    assert any("collapsed" in e for e in errors)


def test_validate_three_model_endpoints_distinct(monkeypatch) -> None:
    def fake_url(slot: int = 1) -> str:
        return {1: "http://127.0.0.1:11434", 2: "http://127.0.0.1:11435", 3: "http://127.0.0.1:11436"}[slot]

    monkeypatch.setattr(st.backend_mod, "ollama_endpoint_url", fake_url)
    monkeypatch.setattr(st.backend_mod, "_endpoint_ready", lambda _ep: True)
    ok, errors, eps = st.validate_three_model_endpoints()
    assert ok
    assert not errors
    assert len(set(eps)) == 3


def test_parse_trace_events(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        "\n".join([
            json.dumps({"kind": "session_start"}),
            json.dumps({"kind": "stream_start"}),
            json.dumps({"kind": "opening_book_retrieved", "count": 2}),
            json.dumps({"kind": "stream_done", "duration_s": 12.5}),
            json.dumps({"event": "test", "ok": False}),
        ]),
        encoding="utf-8",
    )
    analysis = st.parse_trace(trace)
    assert analysis.session_start
    assert analysis.opening_book_retrieved == 1
    assert analysis.stream_done_count == 1
    assert analysis.test_events >= 1


def test_evaluate_pass_criteria_input_responsive() -> None:
    report = {
        "input_test": {"ran": True, "any_change": True},
        "page_errors": [],
        "errors": [],
    }
    trace = st.TraceAnalysis(opening_book_retrieved=1)
    ok, results = st.evaluate_pass_criteria(
        {"input_responsive": True, "opening_book_trace": True},
        last_report=report,
        trace=trace,
        test_dir=Path("/nonexistent"),
        three_model=False,
    )
    assert ok
    assert results["input_responsive"]


def test_evaluate_pass_criteria_sidecar_requires_activity() -> None:
    trace = st.TraceAnalysis()
    ok, results = st.evaluate_pass_criteria(
        {"sidecar_trace_when_three_model": True},
        last_report={},
        trace=trace,
        test_dir=Path("/nonexistent"),
        three_model=True,
    )
    assert not ok
    assert not results["sidecar_trace_when_three_model"]


def test_score_observability() -> None:
    trace = st.TraceAnalysis(
        session_start=True,
        kinds={"session_start", "stream_start", "test"},
        test_events=2,
        line_count=50,
    )
    test_dir = Path(__file__).parent
    score, notes = st.score_observability(trace, test_dir, None)
    assert score >= 50


def test_write_system_summary(tmp_path: Path) -> None:
    gpu = st.GpuHygieneReport(placement_mode="auto-pinned")
    results = [
        st.SystemTestResult(
            slug="smoke-move", goal="g", model="qwen3.6:27b",
            suite="smoke", max_iters=2, passed=True, harness_ok=True,
            observability_score=80,
        ),
    ]
    st.write_system_summary(
        tmp_path,
        model="qwen3.6:27b",
        suite="smoke",
        three_model=True,
        started_at="2026-05-21T12:00:00",
        duration_s=120.0,
        gpu_report=gpu,
        results=results,
        infrastructure_ok=True,
    )
    text = (tmp_path / "SYSTEM_SUMMARY.md").read_text(encoding="utf-8")
    assert "GPU hygiene" in text
    assert "smoke-move" in text
    assert "Cleanup (optional, manual)" in text


def test_run_gpu_hygiene_writes_snapshots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(st, "_capture_nvidia_smi", lambda: "GPU0: test")
    monkeypatch.setattr(st, "_snapshot_ollama_ps", lambda _bases: {"http://127.0.0.1:11434": []})
    monkeypatch.setattr(st.backend_mod, "_ollama_running_models", lambda _b: [])
    report = st.run_gpu_hygiene(tmp_path, three_model=False, do_unload=False)
    assert (tmp_path / "gpu_nvidia_smi_before.txt").exists()
    assert (tmp_path / "ollama_ps_before.json").exists()
    assert report.ok or report.errors  # ok when no errors


def test_default_battery_file_exists() -> None:
    assert st.DEFAULT_BATTERY.exists()


def test_classify_infrastructure_fail_result() -> None:
    r = st.SystemTestResult(
        slug="x", goal="g", model="m", suite="smoke", max_iters=1,
        infrastructure_ok=False,
    )
    assert r.short_status() == "INFRA FAIL"


def test_confirm_slow_tests_pacman_suite_declined(monkeypatch) -> None:
    monkeypatch.setattr(st, "prompt_yes_no", lambda _p, **kw: False)
    tests = [st.SystemBatteryTest("pacman-hard", "goal", suite="pacman")]
    out, notes = st.confirm_slow_tests_before_run(tests, suite="pacman", assume_yes=False)
    assert out == []
    assert notes


def test_confirm_slow_tests_pacman_suite_accepted(monkeypatch) -> None:
    monkeypatch.setattr(st, "prompt_yes_no", lambda _p, **kw: True)
    tests = [st.SystemBatteryTest("pacman-hard", "goal", suite="pacman")]
    out, notes = st.confirm_slow_tests_before_run(tests, suite="pacman", assume_yes=False)
    assert len(out) == 1
    assert not notes


def test_confirm_slow_tests_smoke_skips_prompt() -> None:
    tests = [st.SystemBatteryTest("smoke-move", "goal", suite="smoke")]
    out, notes = st.confirm_slow_tests_before_run(tests, suite="smoke", assume_yes=False)
    assert len(out) == 1
    assert not notes


def test_confirm_slow_test_run_assume_yes() -> None:
    assert st.confirm_slow_test_run("pacman-hard", assume_yes=True)


def test_confirm_slow_test_run_declined(monkeypatch) -> None:
    monkeypatch.setattr(st, "prompt_yes_no", lambda _p, **kw: False)
    assert not st.confirm_slow_test_run("pacman-hard", assume_yes=False)


def test_partition_slow_tests() -> None:
    tests = [
        st.SystemBatteryTest("smoke-move", "g"),
        st.SystemBatteryTest("pacman-hard", "g"),
    ]
    fast, slow = st.partition_slow_tests(tests)
    assert [t.slug for t in fast] == ["smoke-move"]
    assert [t.slug for t in slow] == ["pacman-hard"]
