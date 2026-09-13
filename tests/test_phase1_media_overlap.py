"""Phase 1 media overlap: media_kinds filter + pending path blocks."""
from __future__ import annotations
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent if False else "/Users/jonathanrothberg/Agent_learning"))
import agent_assets
from sounds import render_sound_paths_block
from videos import render_video_paths_block
from agent import GameAgent

def test_maybe_generate_accepts_media_kinds():
    src = inspect.getsource(agent_assets.AssetGenerationMixin._maybe_generate_assets_and_sounds)
    assert "media_kinds" in src
    assert '"assets" not in _kinds' in src or "'assets' not in _kinds" in src

def test_pending_sound_block_includes_missing_files(tmp_path):
    p = tmp_path / "missing.ogg"
    block = render_sound_paths_block({"shoot": p}, tmp_path / "g.html", pending=True)
    assert "pending" in block.lower()
    assert "shoot" in block
    assert render_sound_paths_block({"shoot": p}, tmp_path / "g.html", pending=False) == ""

def test_pending_video_block_includes_missing_files(tmp_path):
    p = tmp_path / "intro.mp4"
    block = render_video_paths_block({"intro": p}, tmp_path / "g.html", pending=True)
    assert "pending" in block.lower()
    assert "intro" in block

def test_phase_a_overlap_wiring_in_agent_source():
    src = Path("/Users/jonathanrothberg/Agent_learning/agent.py").read_text()
    assert 'media_kinds={"assets"}' in src
    assert 'media_kinds={"sounds", "videos"}' in src
    assert "media_overlapped_seconds" in src
    assert "_pending_media_task" in src
