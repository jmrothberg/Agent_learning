"""Phase 3 media auto-probe injection tests.

`_maybe_inject_media_probes()` adds deterministic media-wiring probes for the
assets / videos a session generated. Conservative: only clean-signal probes,
each idempotent, none injected when no media exists. Pure-function: no model,
no browser eval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def _agent(tmp_path: Path) -> GameAgent:
    a = GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )
    a._trace = lambda obj: None
    a._probes = []
    return a


def _names(a) -> set[str]:
    return {(p.get("name") or "") for p in (a._probes or [])}


def test_no_media_no_probes(tmp_path):
    a = _agent(tmp_path)
    a._maybe_inject_media_probes()
    assert _names(a) == set()


def test_assets_inject_miss_probe(tmp_path):
    a = _agent(tmp_path)
    a._session_assets = {"hero": tmp_path / "hero.png"}
    a._maybe_inject_media_probes()
    assert "auto_no_missing_asset_placeholders" in _names(a)


def test_videos_inject_video_probe(tmp_path):
    a = _agent(tmp_path)
    a._session_videos = {"intro": tmp_path / "intro.mp4"}
    a._maybe_inject_media_probes()
    assert "auto_video_present" in _names(a)
    expr = next(p["expr"] for p in a._probes if p["name"] == "auto_video_present")
    assert "querySelector('video')" in expr


def test_sounds_only_injects_no_false_failing_probe(tmp_path):
    # Deliberate Phase 3 decision: the injected sound loader uses script-scoped
    # `const SOUNDS` + `new Audio()` with NO window-observable signal, so a
    # gating `auto_sounds_wired` probe would false-fail games that DO wire sound
    # (violating the anti-false-positive rule). Sounds therefore inject nothing
    # here; "silent game is a regression" stays in advisory channels.
    a = _agent(tmp_path)
    a._session_sounds = {"shoot": tmp_path / "shoot.ogg"}
    a._maybe_inject_media_probes()
    assert _names(a) == set()


def test_idempotent_no_duplicates(tmp_path):
    a = _agent(tmp_path)
    a._session_assets = {"hero": tmp_path / "hero.png"}
    a._session_videos = {"intro": tmp_path / "intro.mp4"}
    a._maybe_inject_media_probes()
    a._maybe_inject_media_probes()
    names = [p.get("name") for p in a._probes]
    assert len(names) == len(set(names)), f"duplicate probes injected: {names}"
    assert "auto_no_missing_asset_placeholders" in names
    assert "auto_video_present" in names
