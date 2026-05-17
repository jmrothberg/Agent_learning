"""Tests for sound-reference alignment and local trace hardening helpers."""

from __future__ import annotations

from pathlib import Path

from agent import GameAgent
from backend import BackendInfo, make_backend
from memory import SkeletonHit
from patches import FormatRejection


def _make_agent(tmp_path: Path, *, backend_name: str) -> GameAgent:
    info = BackendInfo(
        name=backend_name,  # type: ignore[arg-type]
        model="dummy:0",
        source="test",
        endpoint="http://127.0.0.1:0",
    )
    backend = make_backend(info)
    return GameAgent(
        backend=backend,
        out_path=tmp_path / "game.html",
        max_iters=1,
    )


def test_scan_extracts_sound_refs_bracket_and_dot():
    html = """
    play(SOUNDS['laser_shot']);
    SOUNDS.music_loop.play();
    """
    refs = GameAgent._scan_html_for_sound_refs(html)
    assert refs == {"laser_shot", "music_loop"}


def test_scan_extracts_sound_refs_path_and_list_names():
    html = """
    const src = './session_sounds/explosion_small.ogg';
    const soundNames = ['laser_shot', 'enemy_laser', 'game_over'];
    """
    refs = GameAgent._scan_html_for_sound_refs(html)
    assert "explosion_small" in refs
    assert "laser_shot" in refs
    assert "enemy_laser" in refs
    assert "game_over" in refs


def test_sound_alignment_detects_gap_and_coaches(tmp_path: Path):
    agent = _make_agent(tmp_path, backend_name="ollama")
    sounds_dir = tmp_path / "sounds"
    sounds_dir.mkdir()
    for n in ("laser_shot", "enemy_laser"):
        p = sounds_dir / f"{n}.ogg"
        p.write_bytes(b"OggS")
        agent._session_sounds[n] = p

    html = """
    play(SOUNDS['laser_shot']);
    play(SOUNDS['enemy_laser']);
    play(SOUNDS['music_loop']);
    """
    missing = agent._check_sound_alignment(html)
    assert missing == {"music_loop"}
    assert agent._pending_coaching, "expected coaching for missing sounds"
    txt = agent._pending_coaching[-1]
    assert "music_loop" in txt
    assert "ERR_FILE_NOT_FOUND" in txt
    assert "<sounds>" in txt


def test_sound_alignment_no_gap_is_silent(tmp_path: Path):
    agent = _make_agent(tmp_path, backend_name="ollama")
    sounds_dir = tmp_path / "sounds"
    sounds_dir.mkdir()
    for n in ("laser_shot", "enemy_laser"):
        p = sounds_dir / f"{n}.ogg"
        p.write_bytes(b"OggS")
        agent._session_sounds[n] = p
    html = "play(SOUNDS['laser_shot']); play(SOUNDS['enemy_laser']);"
    assert agent._check_sound_alignment(html) == set()
    assert not agent._pending_coaching


def test_local_first_build_nudge_only_for_local_backend(tmp_path: Path):
    local_agent = _make_agent(tmp_path, backend_name="ollama")
    cloud_agent = _make_agent(tmp_path, backend_name="anthropic")
    for i in range(10):
        local_agent._session_assets[f"a{i}"] = tmp_path / f"a{i}.png"
        cloud_agent._session_assets[f"a{i}"] = tmp_path / f"a{i}.png"
    assert local_agent._local_first_build_nudge()
    assert cloud_agent._local_first_build_nudge() == ""


def test_local_skeleton_guard_fallback_on_low_overlap(tmp_path: Path):
    agent = _make_agent(tmp_path, backend_name="ollama")
    agent._session_assets = {"player_ship": tmp_path / "player_ship.png"}
    agent._session_sounds = {"laser_shot": tmp_path / "laser_shot.ogg"}
    skel = SkeletonHit(
        name="won_old.html",
        html=(
            "const p = ASSETS.player; "
            "const m = SOUNDS.music; "
            "const src = './old_sounds/explosion.ogg';"
        ),
        score=0.9,
        source_goal="old goal",
    )
    fallback, reason = agent._local_should_fallback_skeleton(skel)
    assert fallback is True
    assert "low media-name overlap" in reason


def test_local_skeleton_guard_keeps_when_overlap_is_good(tmp_path: Path):
    agent = _make_agent(tmp_path, backend_name="ollama")
    agent._session_assets = {"player_ship": tmp_path / "player_ship.png"}
    agent._session_sounds = {"laser_shot": tmp_path / "laser_shot.ogg"}
    skel = SkeletonHit(
        name="won_old.html",
        html="drawSprite(ASSETS.player_ship); play(SOUNDS['laser_shot']);",
        score=0.9,
        source_goal="old goal",
    )
    fallback, reason = agent._local_should_fallback_skeleton(skel)
    assert fallback is False
    assert reason == ""


def test_loop_truncation_fallback_local_disallows_question():
    rejection = FormatRejection(
        kind="unclosed_html_file",
        hint="missing </html_file>",
        detail="(detail body)",
    )
    text, _ = GameAgent._no_usable_code_fallback(
        plan_only=False,
        has_existing_file=False,
        consecutive_plan_only=0,
        rejection=rejection,
        format_stuck_streak=1,
        prior_stream_looped=True,
        prior_loop_kind="adjacent_line_spam",
        prior_loop_line="x = 1;",
        is_local_backend=True,
    )
    low = text.lower()
    assert "<question>" not in low
    assert "do not ask the user a question" in low
    assert "smaller complete <html_file>" in low
