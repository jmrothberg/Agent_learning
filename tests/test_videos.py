"""Tests for videos.py (video cutscene pipeline).

Pure-function coverage only — no mlx-gen / torch / diffusers required
to run these. The actual generation path is the subprocess wrapper
scripts/generate_video.py, exercised manually / by the live check.
Mirrors tests/test_sounds.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import videos  # noqa: E402


# ---------------------------------------------------------------------------
# parse_videos_block — tolerant JSON parsing, same shape as sounds/assets
# ---------------------------------------------------------------------------


def test_parse_videos_block_basic():
    reply = (
        "<videos>\n"
        "[\n"
        "  {\"name\": \"intro\", \"prompt\": \"castle at dusk, slow push-in\","
        " \"image\": \"key_intro\", \"seconds\": 4},\n"
        "  {\"name\": \"victory\", \"prompt\": \"confetti falls, camera orbits\"}\n"
        "]\n"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 2
    assert out[0] == {
        "name": "intro", "prompt": "castle at dusk, slow push-in",
        "seconds": 4.0, "image": "key_intro",
    }
    # image defaults to None; seconds defaults to 4.0
    assert out[1]["image"] is None
    assert out[1]["seconds"] == videos._DEFAULT_SECONDS


def test_parse_videos_block_no_tag_returns_empty():
    assert videos.parse_videos_block("no tags here") == []
    assert videos.parse_videos_block("") == []
    assert videos.parse_videos_block(None) == []


def test_parse_videos_block_drops_specs_with_empty_prompt():
    """Empty PROMPT drops the entry; empty NAME auto-fills video_<i>."""
    reply = (
        "<videos>"
        "[{\"name\":\"\", \"prompt\":\"x\"},"
        " {\"name\":\"a\", \"prompt\":\"\"},"
        " {\"name\":\"keep\", \"prompt\":\"valid\"}]"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    names = {s["name"] for s in out}
    assert "keep" in names
    assert "a" not in names
    assert any(n.startswith("video_") for n in names)


def test_parse_videos_block_clamps_seconds():
    reply = (
        "<videos>"
        "[{\"name\":\"long\", \"prompt\":\"p1\", \"seconds\": 60},"
        " {\"name\":\"short\", \"prompt\":\"p2\", \"seconds\": 0.5},"
        " {\"name\":\"bad\", \"prompt\":\"p3\", \"seconds\": \"huh\"}]"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    by_name = {s["name"]: s for s in out}
    assert by_name["long"]["seconds"] == videos._MAX_SECONDS
    assert by_name["short"]["seconds"] == videos._MIN_SECONDS
    assert by_name["bad"]["seconds"] == videos._DEFAULT_SECONDS


def test_parse_videos_block_duration_alias():
    """Models copy the <sounds> key name — "duration" works as an alias
    for "seconds" (observed in the 20260611 snake-cutscene live run)."""
    reply = (
        "<videos>"
        "[{\"name\":\"a\", \"prompt\":\"p\", \"duration\": 3.0},"
        " {\"name\":\"b\", \"prompt\":\"q\", \"seconds\": 5, \"duration\": 2}]"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    by_name = {s["name"]: s for s in out}
    assert by_name["a"]["seconds"] == 3.0
    assert by_name["b"]["seconds"] == 5.0   # explicit "seconds" wins


def test_parse_videos_block_caps_per_turn():
    items = ",".join(
        f"{{\"name\":\"v{i}\", \"prompt\":\"clip number {i}\"}}"
        for i in range(10)
    )
    out = videos.parse_videos_block(f"<videos>[{items}]</videos>")
    assert len(out) == videos._MAX_VIDEOS_PER_TURN


def test_parse_videos_block_dedupes_same_prompt():
    reply = (
        "<videos>"
        "[{\"name\":\"a\", \"prompt\":\"the SAME clip\"},"
        " {\"name\":\"b\", \"prompt\":\"the  same   clip\"},"
        " {\"name\":\"c\", \"prompt\":\"a different clip\"}]"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 2
    assert {s["name"] for s in out} == {"a", "c"}


def test_parse_videos_block_same_prompt_different_image_kept():
    """Same prompt seeded from different key art is NOT a dupe."""
    reply = (
        "<videos>"
        "[{\"name\":\"a\", \"prompt\":\"camera pans\", \"image\":\"key_a\"},"
        " {\"name\":\"b\", \"prompt\":\"camera pans\", \"image\":\"key_b\"}]"
        "</videos>"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 2


def test_parse_videos_block_json_fence_tolerated():
    reply = (
        "<videos>\n```json\n"
        "[{\"name\":\"intro\", \"prompt\":\"fenced\"}]\n"
        "```\n</videos>"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 1
    assert out[0]["name"] == "intro"


def test_parse_videos_block_truncated_stream_recovers():
    """Stream cut off mid-list: complete entries are recovered."""
    reply = (
        "<videos>\n"
        "[{\"name\":\"intro\", \"prompt\":\"complete entry\", \"seconds\": 4},\n"
        " {\"name\":\"death\", \"prompt\":\"another complete one\"},\n"
        " {\"name\":\"trunc\", \"prompt\":\"this one was cut o"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 2
    assert {s["name"] for s in out} == {"intro", "death"}


def test_parse_videos_block_ignores_thinking_prose(tmp_path):
    """A <videos> mention inside <think> CoT must not corrupt the parse."""
    reply = (
        "<think>I could emit <videos> with junk here [not json</think>\n"
        "<videos>[{\"name\":\"real\", \"prompt\":\"the real one\"}]</videos>"
    )
    out = videos.parse_videos_block(reply)
    assert len(out) == 1
    assert out[0]["name"] == "real"


# ---------------------------------------------------------------------------
# generate_videos — cache + injected fake generator (no subprocess)
# ---------------------------------------------------------------------------


class _FakeGen:
    """Stands in for VideoGenerator: writes a tiny file at out_path."""

    def __init__(self, fail_names: set[str] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_names = fail_names or set()
        self._last_error: str | None = None
        self.last_stats: list[dict] = []

    def generate(self, prompt, out_path, *, seconds, image_path=None):
        self.calls.append({
            "prompt": prompt, "out": Path(out_path),
            "seconds": seconds, "image_path": image_path,
        })
        if any(f in str(out_path) for f in self.fail_names):
            self._last_error = "fake failure"
            return None
        Path(out_path).write_bytes(b"\x00fakemp4")
        return str(out_path)


def test_generate_videos_empty_specs_no_op(tmp_path):
    assert videos.generate_videos([], tmp_path / "s") == {}


def test_generate_videos_writes_session_and_cache(tmp_path):
    gen = _FakeGen()
    specs = [{"name": "intro", "prompt": "p", "seconds": 4.0, "image": None}]
    out = videos.generate_videos(
        specs, tmp_path / "sess", cache_dir=tmp_path / "cache",
        video_generator=gen,
    )
    assert set(out) == {"intro"}
    assert out["intro"].exists()
    assert out["intro"].parent == (tmp_path / "sess")
    # one cache entry written
    assert len(list((tmp_path / "cache").glob("intro__*.mp4"))) == 1
    assert len(gen.calls) == 1


def test_generate_videos_cache_hit_skips_generation(tmp_path):
    specs = [{"name": "intro", "prompt": "p", "seconds": 4.0, "image": None}]
    g1 = _FakeGen()
    videos.generate_videos(
        specs, tmp_path / "s1", cache_dir=tmp_path / "cache",
        video_generator=g1,
    )
    g2 = _FakeGen()
    out2 = videos.generate_videos(
        specs, tmp_path / "s2", cache_dir=tmp_path / "cache",
        video_generator=g2,
    )
    assert set(out2) == {"intro"}
    assert g2.calls == []          # cache hit — no generation
    assert g2.last_stats[0]["cache_hit"] is True


def test_generate_videos_image_field_resolves_session_asset(tmp_path):
    art = tmp_path / "key_intro.png"
    art.write_bytes(b"\x89PNG fake")
    gen = _FakeGen()
    specs = [{
        "name": "intro", "prompt": "p", "seconds": 4.0, "image": "key_intro",
    }]
    out = videos.generate_videos(
        specs, tmp_path / "sess", cache_dir=tmp_path / "cache",
        video_generator=gen, asset_paths={"key_intro": art},
    )
    assert set(out) == {"intro"}
    assert gen.calls[0]["image_path"] == art


def test_generate_videos_unknown_image_degrades_to_t2v(tmp_path):
    gen = _FakeGen()
    specs = [{
        "name": "intro", "prompt": "p", "seconds": 4.0, "image": "no_such",
    }]
    out = videos.generate_videos(
        specs, tmp_path / "sess", cache_dir=tmp_path / "cache",
        video_generator=gen, asset_paths={},
    )
    assert set(out) == {"intro"}
    assert gen.calls[0]["image_path"] is None      # degraded, not failed


def test_generate_videos_partial_failure_keeps_batch(tmp_path):
    gen = _FakeGen(fail_names={"bad"})
    specs = [
        {"name": "bad", "prompt": "p1", "seconds": 4.0, "image": None},
        {"name": "good", "prompt": "p2", "seconds": 4.0, "image": None},
    ]
    out = videos.generate_videos(
        specs, tmp_path / "sess", cache_dir=tmp_path / "cache",
        video_generator=gen,
    )
    assert set(out) == {"good"}
    errs = [s for s in gen.last_stats if s.get("error")]
    assert len(errs) == 1 and errs[0]["name"] == "bad"


# ---------------------------------------------------------------------------
# render_video_paths_block — injection block for the build prompt
# ---------------------------------------------------------------------------


def test_render_video_paths_block_empty_inputs():
    assert videos.render_video_paths_block({}, "/tmp/x.html") == ""
    # paths that don't exist on disk are filtered → empty block
    assert videos.render_video_paths_block(
        {"ghost": Path("/nonexistent/ghost.mp4")}, "/tmp/x.html",
    ) == ""


def test_render_video_paths_block_relative_paths_and_loader(tmp_path):
    html = tmp_path / "game.html"
    vdir = tmp_path / "game_videos"
    vdir.mkdir()
    intro = vdir / "intro.mp4"
    intro.write_bytes(b"x")
    block = videos.render_video_paths_block({"intro": intro}, html)
    assert "./game_videos/intro.mp4" in block
    # the loader pattern essentials
    assert "playCut" in block
    assert "muted" in block
    assert "onDone" in block          # skip-and-fallback continuation
    assert "keydown" in block         # any-key skip
    assert "NEVER stall" in block


def test_render_video_paths_block_filters_missing(tmp_path):
    html = tmp_path / "game.html"
    vdir = tmp_path / "v"
    vdir.mkdir()
    real = vdir / "real.mp4"
    real.write_bytes(b"x")
    block = videos.render_video_paths_block(
        {"real": real, "ghost": vdir / "ghost.mp4"}, html,
    )
    assert "real.mp4" in block
    assert "ghost" not in block


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_safe_filename():
    assert videos._safe_filename("intro scene!") == "intro_scene"
    assert videos._safe_filename("") == "video"
    assert len(videos._safe_filename("x" * 100)) <= 48


def test_cache_key_normalizes_prompt():
    a = videos._cache_key("m", "A  Big   Clip", 4.0, "")
    b = videos._cache_key("m", "a big clip", 4.0, "")
    assert a == b
    # image signature changes the key
    c = videos._cache_key("m", "a big clip", 4.0, "123:456")
    assert c != a
