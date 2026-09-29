"""Honest PASS scoring for tune_serial_loop (run_06 false 6/6 PASS)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import eval.tune_serial_loop as loop  # noqa: E402
import eval.tune_overnight_monitor as monitor  # noqa: E402

REPO = Path(__file__).parent.parent


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


def test_coder_stdout_ignores_closed_pipe(monkeypatch):
    """A detached batch reader must not abort before HTML materialization."""
    import coder
    from agent import AgentEvent

    class ClosedPipe:
        writes = 0

        def write(self, _text: str) -> None:
            self.writes += 1
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self) -> None:
            raise AssertionError("write should fail before flush")

    pipe = ClosedPipe()
    monkeypatch.setattr(coder, "_stdout_broken", False)
    monkeypatch.setattr(coder.sys, "stdout", pipe)

    coder._print_event(AgentEvent("info", "reply is ready to materialize"))
    assert coder._stdout_broken is True

    # Once the pipe is known closed, later events do not attempt another write.
    coder._print_event(AgentEvent("code", "game.html"))
    assert pipe.writes == 1

    token_pipe = ClosedPipe()
    monkeypatch.setattr(coder, "_stdout_broken", False)
    monkeypatch.setattr(coder.sys, "stdout", token_pipe)
    coder._stdout_write("<html_file>")
    assert coder._stdout_broken is True
    assert token_pipe.writes == 1


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


def test_effective_outcome_never_pass_on_session_ok_false():
    raw = {"exit_code": 0}
    sig = {"iter_summaries": 0, "session_ok": False, "last_iter_ok": None}
    assert monitor._effective_outcome(raw, sig) == "fresh_fail"


def test_child_env_for_goal_arms_jmr_png_only_when_target_says_so():
    """Campaign mix: /640png goals get AGENT_JMR_PNG; full-HTML clears it."""
    base = {"LLM_BACKEND": "mlx", "AGENT_JMR_PNG": "1", "AGENT_SIMULATOR": "1"}
    png = loop._child_env_for_goal(base, "Build Dig Dug. TARGET=/640png JMR sheets.")
    assert png.get("AGENT_JMR_PNG") == "1"
    assert png.get("AGENT_SIMULATOR") == "1"
    full = loop._child_env_for_goal(base, "Build a Doom game with three.js CDN.")
    assert "AGENT_JMR_PNG" not in full
    assert "AGENT_SIMULATOR" not in full
    sim = loop._child_env_for_goal({}, "Build Asteroids. TARGET=/640 only.")
    assert sim.get("AGENT_SIMULATOR") == "1"
    assert "AGENT_JMR_PNG" not in sim

