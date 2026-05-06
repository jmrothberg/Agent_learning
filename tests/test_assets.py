"""Tests for the asset-generation pipeline (assets.py).

We don't actually run Z-Image-Turbo here — that needs CUDA + diffusers
+ a real model checkpoint. Instead we inject a `StubGenerator` that
writes 1×1 PNGs, exercising the full parse → cache → save → render
path without the GPU dependency.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402
from assets import (  # noqa: E402
    _cache_key,
    _parse_size,
    _safe_filename,
    generate_assets,
    parse_assets_block,
    render_asset_paths_block,
)


# ---------------------------------------------------------------------------
# Stub generator — writes a tiny PNG without needing torch/diffusers/CUDA
# ---------------------------------------------------------------------------


class StubGenerator:
    """Test double for ImageGenerator.

    Each call writes a 768×768 PNG (matching Z-Image-Turbo's native
    output size) so the resize path in generate_assets exercises a
    real downscale.
    """

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_for: set[str] = fail_for or set()

    def generate(self, prompt: str) -> str | None:
        self.calls.append(prompt)
        if prompt in self.fail_for:
            return None
        from PIL import Image
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        # Distinct color per prompt so we can verify caching collapses
        # repeated requests but distinct prompts produce distinct files.
        seed = sum(ord(c) for c in prompt) % 256
        Image.new("RGB", (768, 768), (seed, 128, 255 - seed)).save(f.name)
        return f.name


# ---------------------------------------------------------------------------
# parse_assets_block
# ---------------------------------------------------------------------------


def test_parse_basic():
    reply = '''
<assets>
[
  {"name": "ship", "prompt": "pixel ship facing right"},
  {"name": "rock", "prompt": "grey rock", "size": "64x64"}
]
</assets>
'''
    out = parse_assets_block(reply)
    assert len(out) == 2
    assert out[0]["name"] == "ship"
    assert out[0]["prompt"] == "pixel ship facing right"
    assert out[0]["size"] == (128, 128)  # default
    assert out[1]["size"] == (64, 64)


def test_parse_missing_tag_returns_empty():
    assert parse_assets_block("no tag here") == []
    assert parse_assets_block("") == []
    assert parse_assets_block(None) == []  # type: ignore


def test_parse_strips_json_fence():
    reply = '''<assets>
```json
[{"name": "x", "prompt": "y"}]
```
</assets>'''
    assert len(parse_assets_block(reply)) == 1


def test_parse_drops_specs_missing_prompt():
    """`prompt` is the only truly required field — without it we have
    nothing to send the diffuser. `name` is auto-filled when missing
    or blank (asset_1, asset_2, …) so the model doesn't lose work
    over a slightly malformed spec."""
    reply = '''<assets>
[
  {"name": "ok", "prompt": "valid"},
  {"prompt": "missing name"},
  {"name": "missing-prompt"},
  {"name": "", "prompt": "blank name"}
]
</assets>'''
    out = parse_assets_block(reply)
    # Three keep paths: explicit name, auto-named (no name), auto-named (blank name).
    # One drop path: no prompt.
    assert len(out) == 3
    names = [s["name"] for s in out]
    assert "ok" in names
    assert "missing-prompt" not in names
    # Auto-named entries follow the asset_<i> pattern.
    assert any(n.startswith("asset_") for n in names)


def test_parse_caps_at_max_per_turn():
    reply = '<assets>' + str([
        {"name": f"a{i}", "prompt": f"p{i}"} for i in range(20)
    ]).replace("'", '"') + '</assets>'
    out = parse_assets_block(reply)
    assert len(out) == assets._MAX_ASSETS_PER_TURN


def test_parse_malformed_json_returns_empty():
    reply = '<assets>not json{[</assets>'
    assert parse_assets_block(reply) == []


def test_parse_size_int_default():
    reply = '''<assets>[{"name":"x","prompt":"y","size": 96}]</assets>'''
    out = parse_assets_block(reply)
    assert out[0]["size"] == (96, 96)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_size_variants():
    assert _parse_size(64) == (64, 64)
    assert _parse_size("128") == (128, 128)
    assert _parse_size("64x96") == (64, 96)
    assert _parse_size("32X32") == (32, 32)


def test_parse_size_clamps():
    """Big or zero values get bounded so we can't accidentally request a
    20000×20000 sprite."""
    assert _parse_size(99999)[0] <= 1024
    assert _parse_size(0)[0] >= 1


def test_safe_filename():
    assert _safe_filename("ship") == "ship"
    assert _safe_filename("ship/sprite.png") == "ship_sprite_png"
    assert _safe_filename("../../etc/passwd") == "etc_passwd"
    assert _safe_filename("") == "asset"
    assert _safe_filename("a" * 200).startswith("aaaa")
    assert len(_safe_filename("a" * 200)) <= 48


def test_cache_key_stable_across_whitespace():
    """Trivial whitespace + casing differences shouldn't bust the cache."""
    k1 = _cache_key("Z", "Hello World", (64, 64))
    k2 = _cache_key("Z", "  hello   world ", (64, 64))
    assert k1 == k2


def test_cache_key_size_matters():
    """But size differences DO bust the cache — different sizes are
    different artifacts."""
    k1 = _cache_key("Z", "x", (64, 64))
    k2 = _cache_key("Z", "x", (128, 128))
    assert k1 != k2


def test_cache_key_model_matters():
    k1 = _cache_key("Z-Image-Turbo", "x", (64, 64))
    k2 = _cache_key("Flux", "x", (64, 64))
    assert k1 != k2


# ---------------------------------------------------------------------------
# generate_assets
# ---------------------------------------------------------------------------


def test_generate_no_specs_is_noop(tmp_path: Path):
    out = generate_assets([], tmp_path / "session", image_generator=StubGenerator())
    assert out == {}


def test_generate_no_generator_returns_empty(tmp_path: Path, monkeypatch):
    """When ImageGenerator can't be loaded, return {} without crashing.

    We force `try_load_image_generator` to return None; the function
    must not try to call `.generate()` on a None object.
    """
    monkeypatch.setattr(assets, "try_load_image_generator", lambda *a, **k: None)
    specs = [{"name": "ship", "prompt": "pixel ship", "size": (64, 64)}]
    out = generate_assets(specs, tmp_path / "session")
    assert out == {}


def test_generate_writes_pngs(tmp_path: Path):
    specs = [
        {"name": "ship", "prompt": "pixel ship", "size": (64, 64)},
        {"name": "rock", "prompt": "grey rock", "size": (32, 32)},
    ]
    gen = StubGenerator()
    out = generate_assets(specs, tmp_path / "session", image_generator=gen)
    assert set(out.keys()) == {"ship", "rock"}
    for path in out.values():
        assert path.exists()
        assert path.suffix == ".png"
        # Verify resize actually happened.
        from PIL import Image
        with Image.open(path) as img:
            assert img.size in {(64, 64), (32, 32)}
    assert len(gen.calls) == 2


def test_generate_caches_by_content(tmp_path: Path):
    """Second call for the same (prompt, size) must NOT re-invoke the
    generator — cache hit."""
    specs = [{"name": "ship", "prompt": "pixel ship", "size": (64, 64)}]
    gen = StubGenerator()
    cache = tmp_path / "cache"

    out1 = generate_assets(
        specs, tmp_path / "s1", cache_dir=cache, image_generator=gen,
    )
    out2 = generate_assets(
        specs, tmp_path / "s2", cache_dir=cache, image_generator=gen,
    )
    assert len(gen.calls) == 1, "second call should hit cache"
    assert out1["ship"].exists() and out2["ship"].exists()
    # Different session dirs but same source pixels (link or copy from cache).
    assert out1["ship"] != out2["ship"]
    assert out1["ship"].read_bytes() == out2["ship"].read_bytes()


def test_generate_individual_failure_doesnt_kill_batch(tmp_path: Path):
    """If one asset fails to generate, the others still come back."""
    specs = [
        {"name": "ship",  "prompt": "pixel ship", "size": (64, 64)},
        {"name": "broken","prompt": "WILL FAIL",  "size": (64, 64)},
        {"name": "rock",  "prompt": "grey rock",  "size": (64, 64)},
    ]
    gen = StubGenerator(fail_for={"WILL FAIL"})
    out = generate_assets(specs, tmp_path / "session", image_generator=gen)
    assert "ship" in out
    assert "rock" in out
    assert "broken" not in out


def test_generate_sanitizes_dangerous_names(tmp_path: Path):
    """Names with path-traversal characters must NOT escape session_dir."""
    specs = [{"name": "../../escape", "prompt": "x", "size": (32, 32)}]
    out = generate_assets(specs, tmp_path / "session", image_generator=StubGenerator())
    # Name was sanitized to "_._._escape" or similar; result MUST live
    # inside session_dir.
    for path in out.values():
        assert path.is_file()
        assert (tmp_path / "session").resolve() in path.parents


# ---------------------------------------------------------------------------
# render_asset_paths_block
# ---------------------------------------------------------------------------


def test_render_block_uses_relative_paths(tmp_path: Path):
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    asset_dir = tmp_path / "game_assets"
    asset_dir.mkdir()
    ship = asset_dir / "ship.png"
    ship.write_bytes(b"\x89PNG fake")
    block = render_asset_paths_block({"ship": ship}, html)
    assert "GENERATED ASSETS" in block
    assert "./game_assets/ship.png" in block
    assert "image-load-race" in block  # links to playbook bullet


def test_render_block_empty_input_returns_empty():
    assert render_asset_paths_block({}, "/tmp/anywhere.html") == ""


def test_render_block_skips_rel_when_outside_html_dir(tmp_path: Path):
    """If the asset path is outside the html dir (cache fallback), the
    block still renders — using the absolute path."""
    html = tmp_path / "subdir" / "game.html"
    html.parent.mkdir()
    html.write_text("")
    elsewhere = tmp_path / "elsewhere.png"
    elsewhere.write_bytes(b"x")
    block = render_asset_paths_block({"x": elsewhere}, html)
    assert "x:" in block
    assert "GENERATED ASSETS" in block
