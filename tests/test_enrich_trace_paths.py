"""Path resolution for scripts/enrich_trace.py (tune batch + TUI traces)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_enrich_trace():
    path = Path(__file__).parent.parent / "scripts" / "enrich_trace.py"
    spec = importlib.util.spec_from_file_location("_enrich_trace_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_snapshots_dir_tune_batch_layout(tmp_path: Path, monkeypatch) -> None:
    et = _load_enrich_trace()
    monkeypatch.setattr(et, "REPO_ROOT", tmp_path)
    trace = tmp_path / "games" / "tune_serial10" / "run_06" / "traces" / "01_foo__run_2026.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    snap = et._snapshots_dir_for_trace(trace)
    assert snap == tmp_path / "games" / "tune_serial10" / "run_06" / "snapshots" / "01_foo__run_2026"


def test_snapshots_dir_tui_layout(tmp_path: Path, monkeypatch) -> None:
    et = _load_enrich_trace()
    monkeypatch.setattr(et, "REPO_ROOT", tmp_path)
    trace = tmp_path / "games" / "traces" / "snake__run_2026.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    snap = et._snapshots_dir_for_trace(trace)
    assert snap == tmp_path / "games" / "snapshots" / "snake__run_2026"


def test_find_trace_matches_tune_and_tui(tmp_path: Path, monkeypatch) -> None:
    et = _load_enrich_trace()
    monkeypatch.setattr(et, "REPO_ROOT", tmp_path)
    tui = tmp_path / "games" / "traces" / "donkey__run_old.jsonl"
    tune = tmp_path / "games" / "tune_serial10" / "run_06" / "traces" / "donkey__run_new.jsonl"
    tui.parent.mkdir(parents=True)
    tune.parent.mkdir(parents=True)
    tui.write_text("{}")
    tune.write_text("{}")
    matches = et._find_trace_matches("donkey")
    assert len(matches) == 2
    assert tui in matches and tune in matches


def test_resolve_full_path_relative_to_repo(tmp_path: Path, monkeypatch) -> None:
    et = _load_enrich_trace()
    monkeypatch.setattr(et, "REPO_ROOT", tmp_path)
    trace = tmp_path / "games" / "traces" / "x.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}")
    got_trace, snap = et._resolve_paths("games/traces/x.jsonl")
    assert got_trace.resolve() == trace.resolve()
    assert snap.name == "x"
