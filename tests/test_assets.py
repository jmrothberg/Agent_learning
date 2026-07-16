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
        if any(f in prompt for f in self.fail_for):
            return None
        from PIL import Image, ImageDraw
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        # White corners so chroma-key keeps the center blob; solid fills were
        # keyed to empty PNGs and made every from_image frame look identical
        # (which falsely tripped the run_13 pose-retry path in unit tests).
        seed = sum(ord(c) for c in prompt) % 256
        img = Image.new("RGB", (768, 768), (255, 255, 255))
        ImageDraw.Draw(img).rectangle(
            [192, 192, 576, 576], fill=(seed, 128, 255 - seed)
        )
        img.save(f.name)
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
    assert out[0]["size"] == (512, 512)  # default 512 for a plain prompt (no hi-res cue)
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
    """A request well over the cap is truncated to _MAX_ASSETS_PER_TURN.
    Uses 2x the cap so the test stays correct if the cap moves again."""
    cap = assets._MAX_ASSETS_PER_TURN
    reply = '<assets>' + str([
        {"name": f"a{i}", "prompt": f"p{i}"} for i in range(cap * 2)
    ]).replace("'", '"') + '</assets>'
    out = parse_assets_block(reply)
    assert len(out) == cap


def test_parse_with_meta_surfaces_dropped_names():
    """The agent uses parse_assets_block_with_meta() to know which
    asset names were dropped so it can coach the model. The DK trace
    failure pattern: model asked for 14 sprites, harness silently
    kept only the first 8, model's code referenced all 14, browser
    404'd on the 6 dropped ones — and the model spent multiple iters
    patching drawImage instead of asking for the missing assets."""
    from assets import parse_assets_block_with_meta
    cap = assets._MAX_ASSETS_PER_TURN
    n = cap + 5
    reply = '<assets>' + str([
        {"name": f"sprite_{i}", "prompt": f"p{i}"} for i in range(n)
    ]).replace("'", '"') + '</assets>'
    specs, dropped, dropped_specs = parse_assets_block_with_meta(reply)
    assert len(specs) == cap
    assert dropped == [f"sprite_{i}" for i in range(cap, n)]
    assert len(dropped_specs) == 5
    assert dropped_specs[0]["name"] == f"sprite_{cap}"
    kept_names = {s["name"] for s in specs}
    assert f"sprite_0" in kept_names
    assert f"sprite_{cap - 1}" in kept_names
    assert f"sprite_{cap}" not in kept_names


def test_autogen_pending_dropped_assets_method_exists():
    from agent_assets import AssetGenerationMixin
    assert hasattr(AssetGenerationMixin, "_maybe_autogen_pending_dropped_assets")


def test_parse_with_meta_no_overflow_returns_empty_dropped():
    """Happy path — request fits under cap, no dropped names."""
    from assets import parse_assets_block_with_meta
    reply = '<assets>[{"name":"a","prompt":"p"},{"name":"b","prompt":"q"}]</assets>'
    specs, dropped, dropped_specs = parse_assets_block_with_meta(reply)
    assert len(specs) == 2
    assert dropped == []
    assert dropped_specs == []


def test_prefer_video_seed_assets_rescues_key_victory():
    """run_14 Dragon's Lair: FIFO cap dropped key_victory (video i2v seed).
    Rescue it by evicting a non-seed hazard so cutscenes keep locked look."""
    from assets import prefer_video_seed_assets
    kept = (
        [{"name": f"bg_{i}", "prompt": f"stage {i}"} for i in range(3)]
        + [{"name": f"hazard_{i}", "prompt": f"threat {i}"} for i in range(5)]
        + [{"name": "key_intro", "prompt": "intro still"}]
        + [{"name": "key_fail", "prompt": "fail still"}]
    )
    dropped_specs = [{"name": "key_victory", "prompt": "victory still"}]
    dropped_names = ["key_victory"]
    videos = [
        {"name": "intro", "image": "key_intro"},
        {"name": "fail", "image": "key_fail"},
        {"name": "victory", "image": "key_victory"},
    ]
    new_kept, new_dropped, new_dropped_specs = prefer_video_seed_assets(
        kept, dropped_names, dropped_specs, videos,
    )
    kept_names = {s["name"] for s in new_kept}
    assert "key_victory" in kept_names
    assert "key_intro" in kept_names and "key_fail" in kept_names
    assert all(n.startswith("bg_") for n in kept_names if n.startswith("bg_"))
    assert "key_victory" not in new_dropped
    assert any(n.startswith("hazard_") for n in new_dropped)
    assert len(new_kept) == len(kept)  # cap size unchanged


def test_parse_malformed_json_returns_empty():
    reply = '<assets>not json{[</assets>'
    assert parse_assets_block(reply) == []


def test_parse_recovers_from_truncated_stream():
    """Real failure mode from May 7 FPS run: model emitted <assets>
    + a JSON list, but the stream ended mid-prompt before the closing
    `]` and `</assets>`. We should recover all complete entries and
    drop the incomplete trailing one."""
    reply = '''<plan>doom shooter</plan>
<criteria>...</criteria>
<probes>[]</probes>
<assets>
[
  {"name": "demon",   "prompt": "pixel-art red demon"},
  {"name": "imp",     "prompt": "pixel-art brown imp"},
  {"name": "shotgun", "prompt": "pixel-art shotgun first person"},
  {"name": "wall",    "prompt": "pixel-art stone wall texture"},
  {"name": "muzzle_flash", "prompt": "pixel-art yellow muzzle flas'''
    out = parse_assets_block(reply)
    assert len(out) == 4   # demon, imp, shotgun, wall — muzzle_flash was incomplete
    names = [s["name"] for s in out]
    assert "demon" in names
    assert "wall" in names
    assert "muzzle_flash" not in names


def test_parse_recovers_with_no_closing_bracket():
    """Variant: stream truncated INSIDE a complete object, before the
    list bracket closes. The last `}` is well-formed, so we recover."""
    reply = '<assets>[{"name":"a","prompt":"p1"},{"name":"b","prompt":"p2"}'
    out = parse_assets_block(reply)
    assert len(out) == 2
    assert [s["name"] for s in out] == ["a", "b"]


def test_parse_truncation_recovery_falls_through_on_no_objects():
    """If the truncated body has no complete `{...}` we can find,
    return [] instead of crashing."""
    reply = '<assets>\n[\n  {"incomplete'
    out = parse_assets_block(reply)
    assert out == []


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


def test_cache_filenames_are_human_readable(tmp_path: Path):
    """Cache files should land at `<name>__<hash6>.png` so a user can
    `ls _asset_cache/` and recognize what's in there.

    Regression guard: previously the cache used `<sha256[:32]>.png`,
    which made the cache dir illegible.
    """
    specs = [{"name": "player_ship", "prompt": "cyan ship", "size": (64, 64)}]
    gen = StubGenerator()
    cache = tmp_path / "cache"
    generate_assets(specs, tmp_path / "s", cache_dir=cache, image_generator=gen)
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 1
    fname = files[0]
    assert fname.startswith("player_ship__"), fname
    assert fname.endswith(".png"), fname
    # exactly 6 hex chars between the `__` separator and `.png`
    stem = fname[len("player_ship__"):-len(".png")]
    assert len(stem) == 6 and all(c in "0123456789abcdef" for c in stem)


def test_cache_same_name_different_prompts_coexist(tmp_path: Path):
    """Two specs with the same `name` but different prompts must map to
    distinct cache files (different content hash). Otherwise a later
    session would silently reuse the wrong sprite."""
    cache = tmp_path / "cache"
    gen = StubGenerator()
    generate_assets(
        [{"name": "ship", "prompt": "silver ship", "size": (64, 64)}],
        tmp_path / "a", cache_dir=cache, image_generator=gen,
    )
    generate_assets(
        [{"name": "ship", "prompt": "red ship", "size": (64, 64)}],
        tmp_path / "b", cache_dir=cache, image_generator=gen,
    )
    files = sorted(p.name for p in cache.iterdir())
    # Both start with `ship__`, but with different hash6 suffixes.
    assert len(files) == 2, files
    assert all(f.startswith("ship__") for f in files)
    assert files[0] != files[1]


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
    # The block should show the actual loading pattern (await img.decode()
    # + drawImage) so the model has working code to copy, not just a
    # reference to a playbook bullet it'd have to look up.
    assert "img.decode()" in block
    assert "drawImage" in block
    # One failed decode must not prevent RAF/input from starting.
    assert "requestAnimationFrame(frame);" in block
    assert "loadAssets();" in block
    assert "loadAssets().then" not in block
    # And it should be insistent — qwen3.6-class models default to
    # procedural drawing without explicit "MUST" framing.
    assert "MUST USE THEM" in block or "REGRESSION" in block
    assert "const PATHS = {" in block
    assert "ASSETS[name] = img;" in block
    assert block.index("ASSETS[name] = img;") < block.index("await img.decode();")
    assert "function drawEntity(" in block
    assert "MANDATORY ITER-1 WIRING" in block


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


# ---------------------------------------------------------------------------
# B2 — img2img schema (from_image, strength) and topological ordering
# ---------------------------------------------------------------------------


def test_parse_preserves_from_image_and_strength():
    reply = '''
<assets>
[
  {"name": "alien1", "prompt": "8-bit alien legs together"},
  {"name": "alien2", "prompt": "8-bit alien legs apart",
   "from_image": "alien1", "strength": 0.4}
]
</assets>
'''
    out = parse_assets_block(reply)
    assert len(out) == 2
    assert "from_image" not in out[0]
    assert out[1]["from_image"] == "alien1"
    assert abs(out[1]["strength"] - 0.4) < 1e-9


def test_parse_strength_clamps_and_defaults():
    """Out-of-range strength gets clamped; missing strength defaults to 0.45."""
    reply = '''
<assets>
[
  {"name": "a", "prompt": "p1"},
  {"name": "b", "prompt": "p2", "from_image": "a"},
  {"name": "c", "prompt": "p3", "from_image": "a", "strength": 99.0},
  {"name": "d", "prompt": "p4", "from_image": "a", "strength": -1.0}
]
</assets>
'''
    out = parse_assets_block(reply)
    assert out[1]["strength"] == 0.45
    assert out[2]["strength"] == 1.0
    assert out[3]["strength"] == 0.05


def test_topo_sort_places_parent_before_child():
    """When the child is declared first, the topological sort must
    reorder so the parent is generated before the child reads it."""
    specs = [
        {"name": "child", "prompt": "p2", "size": (128, 128), "from_image": "parent", "strength": 0.4},
        {"name": "parent", "prompt": "p1", "size": (128, 128)},
    ]
    sorted_specs = assets._topo_sort_specs(specs)
    names = [s["name"] for s in sorted_specs]
    assert names.index("parent") < names.index("child")


def test_topo_sort_handles_chain_of_three():
    specs = [
        {"name": "f3", "prompt": "p3", "size": (128, 128), "from_image": "f2", "strength": 0.4},
        {"name": "f1", "prompt": "p1", "size": (128, 128)},
        {"name": "f2", "prompt": "p2", "size": (128, 128), "from_image": "f1", "strength": 0.4},
    ]
    names = [s["name"] for s in assets._topo_sort_specs(specs)]
    assert names == ["f1", "f2", "f3"]


def test_topo_sort_cycle_falls_back_to_input_order():
    """A cycle (a→b→a) is malformed input; don't loop forever, just
    return the original list and let the per-spec code mark the missing
    parents as errors."""
    specs = [
        {"name": "a", "prompt": "p", "size": (128, 128), "from_image": "b", "strength": 0.4},
        {"name": "b", "prompt": "p", "size": (128, 128), "from_image": "a", "strength": 0.4},
    ]
    sorted_specs = assets._topo_sort_specs(specs)
    assert len(sorted_specs) == 2  # didn't drop anything


class Img2ImgStubGenerator:
    """Test double for the SD-Turbo wrapper. Writes a 512×512 PNG whose
    color is derived from BOTH the prompt and the init image so we can
    verify the init was honored, not silently ignored.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self._last_error: str | None = None

    def generate(self, prompt: str, init_image_path: str,
                 *, strength: float = 0.45, num_inference_steps: int = 2) -> str | None:
        self.calls.append((prompt, init_image_path, strength))
        from PIL import Image
        init = Image.open(init_image_path).convert("RGB")
        # Mix init avg color with prompt hash so the test can prove
        # init_image actually contributed.
        seed = sum(ord(c) for c in prompt) % 256
        avg = init.resize((1, 1)).getpixel((0, 0))
        out = ((avg[0] + seed) % 256, avg[1], avg[2])
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        Image.new("RGB", (512, 512), out).save(f.name)
        return f.name


def test_generate_assets_pose_frame_uses_txt2img_merged(tmp_path):
    """A from_image pose frame is rendered as TXT2IMG with the parent's
    character prompt merged with the pose clause — NOT img2img. img2img at
    guidance_scale=0 can't change a pose (proven 2026-05-30, animation_ab/);
    txt2img + shared description + fixed seed gives the same character in the
    new pose."""
    specs = [
        {"name": "idle", "prompt": "alien standing", "size": (64, 64)},
        {"name": "punch", "prompt": "alien arm extended punching", "size": (64, 64),
         "from_image": "idle", "strength": 0.55},
    ]
    txt2img = StubGenerator()
    i2i = Img2ImgStubGenerator()
    out = generate_assets(
        specs, tmp_path / "session",
        cache_dir=tmp_path / "cache",
        image_generator=txt2img,
        img2img_generator=i2i,
    )
    assert set(out.keys()) == {"idle", "punch"}
    assert len(i2i.calls) == 0, "pose frames must NOT use img2img"
    assert len(txt2img.calls) == 2, "both frames render via txt2img"
    # the punch frame's prompt merges pose clause FIRST then parent character
    punch_prompt = txt2img.calls[1]
    assert "alien standing" in punch_prompt and "arm extended punching" in punch_prompt
    # Pose-first order (run_13 1942): do not let idle orientation dominate.
    assert punch_prompt.index("arm extended punching") < punch_prompt.index("alien standing")


def test_strip_idle_orientation_locks():
    from assets import _strip_idle_orientation_locks
    p = "top-down fighter plane facing straight up, navy blue body"
    out = _strip_idle_orientation_locks(p)
    assert "facing straight up" not in out.lower()
    assert "navy blue body" in out
    assert "fighter plane" in out


def test_generate_assets_pose_retry_when_near_identical(tmp_path):
    """run_13: near-identical derived frames get one amplified txt2img retry."""

    class NearIdenticalThenDistinct(StubGenerator):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        def generate(self, prompt: str) -> str | None:
            self.calls.append(prompt)
            self._n += 1
            from PIL import Image, ImageDraw
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            # White corners survive chroma-key; center blob differs only on retry.
            img = Image.new("RGB", (128, 128), (255, 255, 255))
            color = (200, 40, 40) if self._n <= 2 else (40, 40, 200)
            ImageDraw.Draw(img).rectangle([32, 32, 96, 96], fill=color)
            img.save(f.name)
            return f.name

    specs = [
        {"name": "idle", "prompt": "fighter idle stance", "size": (64, 64)},
        {
            "name": "punch",
            "prompt": "fighter arm extended punch",
            "size": (64, 64),
            "from_image": "idle",
            "strength": 0.55,
        },
    ]
    gen = NearIdenticalThenDistinct()
    out = generate_assets(
        specs, tmp_path / "session",
        cache_dir=tmp_path / "cache",
        image_generator=gen,
        img2img_generator=None,
    )
    assert set(out.keys()) == {"idle", "punch"}
    # idle + first pose + amplified retry
    assert len(gen.calls) == 3
    assert any("EXTREME visible pose asymmetry" in c for c in gen.calls)
    stats = getattr(gen, "last_stats", [])
    punch_stat = next(s for s in stats if s.get("name") == "punch")
    assert punch_stat.get("pose_retry") is True
    assert punch_stat.get("parent_delta", 0) >= assets._DERIVED_FRAME_MIN_DELTA


def test_generate_assets_falls_back_to_txt2img_when_img2img_missing(tmp_path):
    """If img2img wrapper isn't available (None), the chained child
    still generates via txt2img — no chain, but no asset is lost."""
    specs = [
        {"name": "walk1", "prompt": "alien legs together", "size": (64, 64)},
        {"name": "walk2", "prompt": "alien legs apart", "size": (64, 64),
         "from_image": "walk1", "strength": 0.45},
    ]
    txt2img = StubGenerator()
    out = generate_assets(
        specs, tmp_path / "session",
        cache_dir=tmp_path / "cache",
        image_generator=txt2img,
        img2img_generator=None,
    )
    assert set(out.keys()) == {"walk1", "walk2"}
    # Both frames went through txt2img — chain unavailable.
    assert len(txt2img.calls) == 2


def test_render_block_emits_robust_sprite_resolver(tmp_path):
    """The injected loader must provide a sprite(key) resolver that tolerates
    key-naming drift (the #1 cause of 'generated art but game shows boxes':
    code builds 'left_idle' but asset is 'left_fighter_idle') and a LOUD
    MISSING marker on a true miss — never a clean fillRect block that hides
    the bug. Added 2026-06-03 after a two-kickers run rendered solid blocks
    because spriteKey != generated asset name."""
    from assets import render_asset_paths_block
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    ad = tmp_path / "game_assets"
    ad.mkdir()
    p = ad / "left_fighter_idle.png"
    p.write_bytes(b"\x89PNG fake")
    block = render_asset_paths_block({"left_fighter_idle": p}, html)
    assert "function sprite(key)" in block          # robust accessor
    assert "MISSING" in block                        # loud marker, not a tidy block
    assert "naturalWidth" in block                   # guards against undecoded
    # normalized/token matching so 'left_idle' resolves to 'left_fighter_idle'
    assert "norm" in block and "includes" in block


def test_render_block_sprite_resolver_is_self_healing(tmp_path):
    """The injected sprite() must cache the resolved IMAGE OBJECT, not a
    readiness verdict, and check naturalWidth AT DRAW TIME — otherwise a
    matched-but-still-decoding PNG gets cached as a permanent miss and the
    game shows boxes until the page is reloaded (the holochess 2026-06-24
    'reload a few times and the art appears' race). Regression guard for the
    sprite-loader negative-cache fix."""
    from assets import render_asset_paths_block
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    ad = tmp_path / "game_assets"
    ad.mkdir()
    p = ad / "hero_idle.png"
    p.write_bytes(b"\x89PNG fake")
    block = render_asset_paths_block({"hero_idle": p}, html)
    # MUST NOT emit the negative-cache form that stores null for an image
    # that simply has not decoded yet (naturalWidth still 0 on first frame).
    assert "ok ? img : null" not in block
    # Cache hits immediately; cache null only after _assetsReady (Chrome race).
    assert "_assetsReady" in block
    assert "_spriteCache[key] = img || null" not in block
    assert "if (img)" in block and "else if (_assetsReady)" in block
    # Readiness is verified where the sprite is DRAWN, not where it is cached.
    assert "img.complete && img.naturalWidth > 0" in block
    # __assetMisses records a true no-match key, not a still-decoding image.
    assert "(window.__assetMisses = window.__assetMisses || {})[key] = 1" in block


# ---------------------------------------------------------------------------
# sprite() resolver mirror — same algorithm as injected JS in render block
# ---------------------------------------------------------------------------


def _norm_asset_key(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _entity_tokens(want: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+\d+|\d+|[a-z]+", want)


def _primary_tokens(want: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+|\d+", want)


def _prefix_overlap(want: str, candidate: str) -> int:
    n = 0
    for i, ch in enumerate(want):
        if i >= len(candidate) or candidate[i] != ch:
            break
        n += 1
    return n


def _mirror_resolve_sprite_key(key: str, names: list[str]) -> str | None:
    """Python mirror of the injected sprite() fuzzy resolver (assets.py)."""
    if key in names:
        return key
    want = _norm_asset_key(key)
    hit: str | None = None
    for n in names:
        if _norm_asset_key(n) == want:
            return n
    for n in names:
        nn = _norm_asset_key(n)
        if nn in want or want in nn:
            hit = n
            break
    if hit:
        return hit
    toks = _primary_tokens(want)
    best = 0
    ties: list[str] = []
    for n in names:
        nn = _norm_asset_key(n)
        sc = sum(len(t) for t in toks if len(t) >= 3 and t in nn)
        if sc > best:
            best = sc
            ties = [n]
        elif sc == best and sc > 0:
            ties.append(n)
    if len(ties) == 1:
        return ties[0]
    if len(ties) > 1:
        etoks = _entity_tokens(want)
        tb = -1
        tb_hit = ties[0]
        for n in ties:
            nn = _norm_asset_key(n)
            sc2 = sum(len(t) for t in etoks if len(t) >= 2 and t in nn)
            if sc2 > tb:
                tb = sc2
                tb_hit = n
            elif sc2 == tb:
                pref = _prefix_overlap(want, nn)
                old_pref = _prefix_overlap(want, _norm_asset_key(tb_hit))
                if pref > old_pref:
                    tb_hit = n
        return tb_hit
    return None


def test_sprite_resolver_key_drift_includes():
    """Shorter code key contained in longer asset name (includes tier)."""
    names = ["left_fighter_idle"]
    assert _mirror_resolve_sprite_key("left_fighter", names) == "left_fighter_idle"
    assert _mirror_resolve_sprite_key("hero", ["hero_idle"]) == "hero_idle"


def test_sprite_resolver_parallel_prefix_tiebreak():
    names = ["f1_walk", "f2_walk", "f1_idle", "f2_idle"]
    assert _mirror_resolve_sprite_key("f2_walk", names) == "f2_walk"
    assert _mirror_resolve_sprite_key("f1_walk", names) == "f1_walk"
    blue_red = ["blue_special", "red_special", "blue_idle", "red_idle"]
    assert _mirror_resolve_sprite_key("blue_special", blue_red) == "blue_special"
    assert _mirror_resolve_sprite_key("red_special", blue_red) == "red_special"
    p12 = ["p1_special", "p2_special"]
    assert _mirror_resolve_sprite_key("p2_special", p12) == "p2_special"


def test_sprite_resolver_single_entity_unchanged():
    names = ["hero_idle", "hero_walk"]
    assert _mirror_resolve_sprite_key("hero_idle", names) == "hero_idle"
    assert _mirror_resolve_sprite_key("hero_walk", names) == "hero_walk"


def test_render_block_flushes_cache_on_assets_ready(tmp_path: Path):
    html = tmp_path / "game.html"
    html.write_text("<html></html>")
    ad = tmp_path / "game_assets"
    ad.mkdir()
    p = ad / "ship.png"
    p.write_bytes(b"\x89PNG fake")
    block = render_asset_paths_block({"ship": p}, html)
    flush_idx = block.index("for (const k in _spriteCache) delete _spriteCache[k];")
    ready_idx = block.index("_assetsReady = true;")
    assert flush_idx < ready_idx
