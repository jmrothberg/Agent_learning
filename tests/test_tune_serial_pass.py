"""Honest PASS scoring for tune_serial_loop (run_06 false 6/6 PASS)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval.tune_serial_loop as loop  # noqa: E402
import eval.tune_overnight_monitor as monitor  # noqa: E402

REPO = Path(__file__).parent.parent
RUN06 = REPO / "games" / "tune_serial10" / "run_06"


def test_stream_instance_method_regression():
    import inspect
    from agent import GameAgent

    assert not isinstance(inspect.getattr_static(GameAgent, "_stream"), classmethod)


def test_wait_for_monitor_handoff_uses_out_root():
    src = (REPO / "eval" / "tune_serial_loop.py").read_text(encoding="utf-8")
    assert "out_dir.glob" not in src
    assert "out_root.glob" in src


def test_serial_loop_rereads_goals_file_each_game():
    """Mid-batch edits to --goals-file must apply to the next game without
    restarting the parent (run_15: bloated ASSET MUSTS stayed in memory)."""
    src = (REPO / "eval" / "tune_serial_loop.py").read_text(encoding="utf-8")
    assert "Re-read --goals-file each game" in src
    assert "refreshed = _load_goals(args)" in src


def test_serial_loop_crash_bonus_retry_on_sigkill():
    """exit<0 with no HTML gets one free retry even when --retries 0
    (run_15 Dragon/Prince/Doom jetsam left no code). Soft fails still
    respect the retries budget."""
    src = (REPO / "eval" / "tune_serial_loop.py").read_text(encoding="utf-8")
    assert "crash_bonus_retry" in src
    assert "crash_bonus_used" in src
    assert "exit<0 with no delivered HTML" in src


def test_counts_as_pass_requires_best_or_iter_ok(tmp_path: Path):
    out = tmp_path / "01_game.html"
    out.write_text("x" * 600, encoding="utf-8")
    assert loop._counts_as_pass(out) is False

    best = tmp_path / "01_game.best.html"
    best.write_text("x" * 600, encoding="utf-8")
    assert loop._counts_as_pass(out) is True


def test_classify_outcome_from_trace_iter_ok(tmp_path: Path):
    out = tmp_path / "01_game.html"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace = trace_dir / "01_game__run_20260101_120000_000000.jsonl"
    trace.write_text(
        json.dumps({"kind": "iter_summary", "ok": True}) + "\n",
        encoding="utf-8",
    )
    # glob searches games/**/traces/{stem}__run_*.jsonl
    games = tmp_path
    stem = out.stem
    # Patch by placing under games/ layout the helper expects
    real_trace = tmp_path / "games" / "sub" / "traces" / f"{stem}__run_test.jsonl"
    real_trace.parent.mkdir(parents=True)
    real_trace.write_text(
        json.dumps({"kind": "iter_summary", "ok": True}) + "\n",
        encoding="utf-8",
    )
    # _trace_last_iter_ok glob is REPO_ROOT / "games" — use monkeypatch
    import eval.tune_serial_loop as mod

    orig = mod.REPO_ROOT

    class _Root:
        pass

    fake = _Root()
    fake.__truediv__ = lambda self, other: tmp_path / other if other == "games" else tmp_path / other
    # simpler: write trace where glob from REPO_ROOT/games finds it
    mod.REPO_ROOT = tmp_path
    try:
        tdir = tmp_path / "games" / "batch" / "traces"
        tdir.mkdir(parents=True)
        (tdir / f"{stem}__run_x.jsonl").write_text(
            json.dumps({"kind": "iter_summary", "ok": True}) + "\n",
            encoding="utf-8",
        )
        assert loop._classify_outcome(out) == "fresh_pass"
    finally:
        mod.REPO_ROOT = orig


def test_classify_fresh_fail_when_no_ship_artifact(tmp_path: Path):
    out = tmp_path / "03_game.html"
    import eval.tune_serial_loop as mod

    orig = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        tdir = tmp_path / "games" / "batch" / "traces"
        tdir.mkdir(parents=True)
        stem = out.stem
        lines = [
            {"kind": "event", "event": "error", "text_preview": "get_backend() missing role"},
            {"kind": "session_outcome", "ok": False, "best_path_exists": False},
        ]
        (tdir / f"{stem}__run_x.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n",
            encoding="utf-8",
        )
        assert loop._classify_outcome(out) == "fresh_fail"
        assert loop._counts_as_pass(out) is False
    finally:
        mod.REPO_ROOT = orig


def test_run06_monitor_relabels_old_checkpoint():
    """run_06 claimed 6/6 PASS; traces show only 01-02 fresh_pass."""
    if not RUN06.is_dir():
        return
    payload = monitor.snapshot(RUN06, jobs_total=6)
    outcomes = {g["label"]: g["outcome"] for g in payload["games"]}
    assert outcomes.get("01_build_a_donkey_kong_game__single") == "fresh_pass"
    assert outcomes.get("02_build_a_kung_fu_master_game__sid") == "fresh_pass"
    for label in (
        "03_build_a_fieldrunners_game__open",
        "04_build_a_joust_game__flap_and_jou",
        "05_build_a_checkers_game__8x8_board",
        "06_build_a_holochess_game__8x8_boar",
    ):
        assert outcomes.get(label) == "fresh_fail", label
    assert payload["fresh_pass_count"] == 2
    assert payload["fresh_fail_count"] == 4


def test_effective_outcome_never_pass_on_session_ok_false():
    raw = {"exit_code": 0}
    sig = {"iter_summaries": 0, "session_ok": False, "last_iter_ok": None}
    assert monitor._effective_outcome(raw, sig) == "fresh_fail"


def test_run_vlm10_goal_assembly_produces_ten_goals():
    """Mirror eval/tune_run_vlm10.sh assembly — batch list must stay stable."""
    from prompt_library import load_prompt_library

    repo = REPO
    r12 = [
        line.strip()
        for line in (repo / "eval/tune_run12_goals.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    r08 = [
        line.strip()
        for line in (repo / "eval/tune_run08_goals.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ][5:8]
    by_name = {
        p["name"]: " ".join(p["prompt"].split())
        for p in load_prompt_library()
    }
    extra = [by_name[n] for n in ("fighter-showcase", "1942", "dragons-lair")]
    goals = r12 + r08 + extra
    assert len(goals) == 10
    assert goals[0].startswith("Build a Prince of Persia")
    assert goals[6].startswith("Build a Monkey Island")
    assert goals[9].startswith("Build a Dragon's Lair")
