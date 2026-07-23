"""Tests for sounds.py (audio asset pipeline).

Pure-function coverage only — no torch / diffusers / soundfile required
to run these. The model-loading + generation paths are covered by the
end-to-end smoke test scripts/_smoke_audio.py, which IS gated on the
GPU stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sounds  # noqa: E402


# ---------------------------------------------------------------------------
# parse_sounds_block — tolerant JSON parsing, the same way assets.py does
# ---------------------------------------------------------------------------


def test_parse_sounds_block_basic():
    reply = (
        "<sounds>\n"
        "[\n"
        "  {\"name\": \"laser\", \"prompt\": \"8-bit laser\", \"duration\": 0.4},\n"
        "  {\"name\": \"music\", \"prompt\": \"chiptune\", \"duration\": 12, \"loop\": true}\n"
        "]\n"
        "</sounds>"
    )
    out = sounds.parse_sounds_block(reply)
    assert len(out) == 2
    assert out[0] == {"name": "laser", "prompt": "8-bit laser", "duration": 0.4, "loop": False}
    assert out[1]["loop"] is True
    assert out[1]["duration"] == 12.0


def test_parse_sounds_block_drops_specs_with_empty_prompt():
    """Mirrors assets.py: an empty PROMPT drops the entry (the model
    can't generate audio without a description), but an empty NAME
    auto-fills as `sound_<i>` so the entry isn't lost."""
    reply = (
        "<sounds>"
        "[{\"name\":\"\", \"prompt\":\"x\"},"
        " {\"name\":\"a\", \"prompt\":\"\"},"
        " {\"name\":\"keep\", \"prompt\":\"valid\"}]"
        "</sounds>"
    )
    out = sounds.parse_sounds_block(reply)
    # Two survive: the empty-name one (renamed to sound_1) and the
    # explicitly-named one. The empty-prompt entry is dropped.
    names = {s["name"] for s in out}
    assert "keep" in names
    assert "a" not in names
    assert any(n.startswith("sound_") for n in names)


def test_parse_sounds_block_clamps_duration():
    reply = (
        "<sounds>"
        "[{\"name\":\"too_long\",\"prompt\":\"x\",\"duration\":999},"
        " {\"name\":\"too_short\",\"prompt\":\"x\",\"duration\":0.001},"
        " {\"name\":\"nan_friendly\",\"prompt\":\"x\",\"duration\":\"not a number\"}]"
        "</sounds>"
    )
    out = sounds.parse_sounds_block(reply)
    durs = {s["name"]: s["duration"] for s in out}
    assert durs["too_long"] == sounds._MAX_DURATION_S
    assert durs["too_short"] == sounds._MIN_DURATION_S
    # Unparsable duration falls back to the default rather than crashing.
    assert durs["nan_friendly"] == sounds._DEFAULT_DURATION_S


def test_parse_sounds_block_caps_count():
    """Cap kicks in even when entries are all distinct (i.e. dedupe
    didn't collapse them). Use unique prompts so the cap is what stops
    growth, not the dedupe set."""
    items = ",".join(
        f'{{"name":"s{i}","prompt":"distinct prompt {i}"}}' for i in range(20)
    )
    reply = f"<sounds>[{items}]</sounds>"
    out = sounds.parse_sounds_block(reply)
    assert len(out) == sounds._MAX_SOUNDS_PER_TURN


def test_parse_sounds_block_tolerates_truncated_stream():
    # Stream ended before </sounds> AND before the closing `]`.
    # Truncation-repair drops the incomplete trailing entry.
    reply = (
        "<sounds>\n"
        "[\n"
        "  {\"name\": \"laser\", \"prompt\": \"8-bit laser\"},\n"
        "  {\"name\": \"music\", \"prompt\": \"loopable chiptu"
    )
    out = sounds.parse_sounds_block(reply)
    assert len(out) == 1
    assert out[0]["name"] == "laser"


def test_parse_sounds_block_tolerates_json_fence():
    reply = (
        "<sounds>\n```json\n"
        "[{\"name\":\"laser\",\"prompt\":\"x\"}]\n"
        "```\n</sounds>"
    )
    out = sounds.parse_sounds_block(reply)
    assert len(out) == 1


def test_parse_sounds_block_empty_when_tag_absent():
    assert sounds.parse_sounds_block("no tag here") == []
    assert sounds.parse_sounds_block("") == []


# ---------------------------------------------------------------------------
# render_sound_paths_block — model-facing loader instructions
# ---------------------------------------------------------------------------


def test_render_sound_paths_block_includes_loop_flag(tmp_path):
    html_path = tmp_path / "game.html"
    snd_dir = tmp_path / "g_sounds"
    snd_dir.mkdir()
    paths = {
        "laser": snd_dir / "laser.ogg",
        "music": snd_dir / "music.ogg",
    }
    for p in paths.values():
        p.write_bytes(b"OggS\x00\x00")           # placeholder; filter only checks exists()
    block = sounds.render_sound_paths_block(
        paths, html_path, looping_names={"music"},
    )
    assert "laser" in block and "music" in block
    # The looping music entry gets `true` in the loader entries; SFX get `false`.
    assert "['music', './g_sounds/music.ogg', true]" in block
    assert "['laser', './g_sounds/laser.ogg', false]" in block
    # Available-sounds list marks looping entries explicitly.
    assert "(looping background)" in block


def test_render_sound_paths_block_returns_empty_when_no_paths(tmp_path):
    assert sounds.render_sound_paths_block({}, tmp_path / "game.html") == ""


# ---------------------------------------------------------------------------
# safe-filename and cache-key — the small helpers prompts depend on
# ---------------------------------------------------------------------------


def test_safe_filename_strips_unsafe_characters():
    assert sounds._safe_filename("hello world!") == "hello_world"
    assert sounds._safe_filename("../etc/passwd") == "etc_passwd"
    # Long names get capped.
    assert len(sounds._safe_filename("x" * 200)) <= 48
    # Empty / all-unsafe input falls back to a stable default.
    assert sounds._safe_filename("???") == "sound"


def test_cache_key_is_normalization_stable():
    # Whitespace and case differences must NOT bust the cache.
    a = sounds._cache_key("m", "  Loud  Boom  ", 0.5)
    b = sounds._cache_key("m", "loud boom", 0.5)
    assert a == b
    # Different duration → different key.
    c = sounds._cache_key("m", "loud boom", 1.0)
    assert a != c


# ---------------------------------------------------------------------------
# generate_sounds cache filename layout — human-readable
# ---------------------------------------------------------------------------


class _StubAudioGenerator:
    """Writes a tiny placeholder .ogg per call. Matches the surface
    used by generate_sounds (just a `.generate(prompt, duration_s)`
    method returning a path). last_stats is populated for parity with
    the real generator."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.last_stats: list[dict] = []

    def generate(self, prompt: str, duration_s: float = 1.0) -> str | None:
        import tempfile
        self.calls.append(prompt)
        f = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        f.write(b"OggS\x00\x00")  # placeholder; downstream only cares about path/exists
        f.close()
        return f.name


def test_cache_filenames_are_human_readable(tmp_path):
    """Cache files should land at `<name>__<hash6>.ogg` so a user can
    scan `_sound_cache/` and know what's in there."""
    cache = tmp_path / "cache"
    sounds.generate_sounds(
        [{"name": "shoot", "prompt": "8-bit laser pew", "duration": 0.3, "loop": False}],
        tmp_path / "s",
        cache_dir=cache,
        audio_generator=_StubAudioGenerator(),
    )
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 1
    fname = files[0]
    assert fname.startswith("shoot__"), fname
    assert fname.endswith(".ogg"), fname
    stem = fname[len("shoot__"):-len(".ogg")]
    assert len(stem) == 6 and all(c in "0123456789abcdef" for c in stem)


def test_cache_same_name_different_prompts_coexist_sounds(tmp_path):
    """Same name + different prompts → two distinct cache files. Without
    this, a later session would silently reuse the wrong audio."""
    cache = tmp_path / "cache"
    gen = _StubAudioGenerator()
    sounds.generate_sounds(
        [{"name": "explosion", "prompt": "small boom", "duration": 0.5, "loop": False}],
        tmp_path / "a", cache_dir=cache, audio_generator=gen,
    )
    sounds.generate_sounds(
        [{"name": "explosion", "prompt": "huge boom", "duration": 0.5, "loop": False}],
        tmp_path / "b", cache_dir=cache, audio_generator=gen,
    )
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 2, files
    assert all(f.startswith("explosion__") for f in files)
    assert files[0] != files[1]


def test_resolve_stable_audio_uses_hf_cache(tmp_path: Path, monkeypatch) -> None:
    """Second Mac: Stable Audio often lives only under ~/.cache/huggingface/hub."""
    import os

    fake_home = tmp_path / "home"
    snap = (
        fake_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--stabilityai--stable-audio-open-1.0"
        / "snapshots"
        / "deadbeef"
    )
    snap.mkdir(parents=True)
    (snap / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(os.path, "expanduser", lambda *_: str(fake_home))
    monkeypatch.delenv("AUDIO_MODELS_DIR", raising=False)
    monkeypatch.delenv("DIFFUSION_MODELS_DIR", raising=False)
    monkeypatch.setattr(sounds, "_MODEL_SEARCH_DIRS", [])
    assert sounds._resolve_stable_audio_path() == str(snap)

