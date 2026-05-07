"""Tests for Tier 1 (probe-signal honesty) + Tier 2 (log enrichment).

All pure-function — no Chromium, no model, no GPU. Each test < 5 ms.

Tier 1 covered:
  - 1.1 Chromium CORS launch flags wired in tools.py
  - 1.2 non_blank example uses a CORS-safe pattern; runner downgrades
        SecurityError-class probe failures to passes
  - 1.3 Z-Image-Turbo chroma-key alpha pass

Tier 2 covered:
  - 2.1 probes_parsed trace includes full text
  - 2.2 per-asset stats stash on the generator
  - 2.3 code_snapshot trace includes html_sha256
  - 2.4 console / page error split + format_report_for_model rendering
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402
import tools   # noqa: E402
import prompts_v1  # noqa: E402


# ---------------------------------------------------------------------------
# 1.1 — Chromium launch flags present in BOTH async + sync paths
# ---------------------------------------------------------------------------


def test_11_async_launch_args_include_cors_flags():
    """Read tools.py and verify the LiveBrowser launch_args list
    includes both flags. Avoids actually starting Chromium."""
    src = Path("tools.py").read_text()
    # Grep around the async launch site.
    assert "--allow-file-access-from-files" in src
    assert "--disable-web-security" in src
    # Both flags should appear in the LiveBrowser.start path AND the
    # sync test_html_file path. That's two occurrences each at minimum.
    assert src.count("--allow-file-access-from-files") >= 2
    assert src.count("--disable-web-security") >= 2


# ---------------------------------------------------------------------------
# 1.2 — non_blank example uses a CORS-safe pattern; runner downgrades
# tainted-canvas probe errors.
# ---------------------------------------------------------------------------


def test_12_non_blank_example_does_not_use_getImageData():
    """Old example called getImageData; the new one falls back via
    try/catch so even on a tainted canvas it returns truthy."""
    src = Path("prompts_v1.py").read_text()
    # The example now uses toDataURL with a try/catch fallback to
    # plain dimension check.
    assert 'name":"non_blank"' in src
    # Find the non_blank example line.
    line = next(
        l for l in src.splitlines() if '"name":"non_blank"' in l
    )
    assert "toDataURL" in line
    # Crucially: NO getImageData (the old broken pattern).
    assert "getImageData" not in line


def test_12_taint_signal_phrases_recognized():
    """The probe-runner downgrades probes whose err contains tainted /
    cross-origin / SecurityError. Verify these phrases match the
    detection in tools.py."""
    src = Path("tools.py").read_text()
    # The detection section should reference all three phrases.
    for phrase in ("tainted", "cross-origin", "securityerror"):
        assert phrase in src.lower()


# ---------------------------------------------------------------------------
# 1.3 — chroma-key pass
# ---------------------------------------------------------------------------


def _solid_bg_image(bg, fg_box, size=(64, 64)):
    """Helper: image with a solid background and a colored rectangle."""
    from PIL import Image
    img = Image.new("RGB", size, bg)
    px = img.load()
    x0, y0, x1, y1 = fg_box
    for y in range(y0, y1):
        for x in range(x0, x1):
            px[x, y] = (200, 0, 0)
    return img


def test_13_chromakey_white_bg_becomes_alpha():
    img = _solid_bg_image((255, 255, 255), (20, 20, 44, 44))
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert keyed.mode == "RGBA"
    assert stats["bg_color"] == (255, 255, 255)
    # Most pixels (background) are now transparent.
    assert stats["alpha_pixel_ratio"] > 0.5
    # Corner is transparent.
    assert keyed.getpixel((0, 0))[3] == 0
    # Center (the red box) stays opaque.
    assert keyed.getpixel((32, 32))[3] == 255


def test_13_chromakey_black_bg_becomes_alpha():
    """Some Z-Image-Turbo prompts produce black bg; the detector should
    pick that up too."""
    img = _solid_bg_image((0, 0, 0), (20, 20, 44, 44))
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert stats["bg_color"] == (0, 0, 0)
    assert keyed.getpixel((0, 0))[3] == 0
    assert keyed.getpixel((32, 32))[3] == 255


def test_13_chromakey_no_dominant_bg_skips_masking():
    """When the corners disagree (no clear bg), don't mask — we'd risk
    eating real pixels."""
    from PIL import Image
    img = Image.new("RGB", (64, 64))
    # Set every corner a different color.
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((63, 0), (0, 255, 0))
    img.putpixel((0, 63), (0, 0, 255))
    img.putpixel((63, 63), (255, 255, 0))
    img.putpixel((32, 0), (255, 0, 255))
    img.putpixel((0, 32), (0, 255, 255))
    img.putpixel((32, 63), (128, 128, 128))
    img.putpixel((63, 32), (200, 200, 50))
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert keyed.mode == "RGBA"
    assert stats["bg_color"] is None
    assert stats["alpha_pixel_ratio"] == 0.0


# ---------------------------------------------------------------------------
# 2.1 — probes_parsed trace includes the full text
# ---------------------------------------------------------------------------


def test_21_probes_full_text_logged_in_agent():
    """Read agent.py source to confirm the probes_parsed trace dict
    has a 'full' field with name+expr per probe (not just count)."""
    src = Path("agent.py").read_text()
    # The probes_parsed trace should now build a 'full' list.
    assert "probes_parsed" in src
    assert '"full":' in src
    # The full entries should record both name and expr.
    snippet = src[src.find("probes_parsed"):src.find("probes_parsed") + 800]
    assert '"name"' in snippet and '"expr"' in snippet


# ---------------------------------------------------------------------------
# 2.2 — per-asset stats stashed on the generator instance
# ---------------------------------------------------------------------------


class _TimedStubGenerator:
    """Stub that counts calls + writes 1×1 PNGs via PIL."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_stats = None  # generate_assets writes here

    def generate(self, prompt: str) -> str | None:
        self.calls += 1
        from PIL import Image
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        # Solid white bg so the chroma-key has work to do.
        img = Image.new("RGB", (768, 768), (255, 255, 255))
        # A tiny red square so the alpha ratio < 1.
        for y in range(380, 388):
            for x in range(380, 388):
                img.putpixel((x, y), (200, 0, 0))
        img.save(f.name)
        return f.name


def test_22_per_asset_stats_attached_to_generator(tmp_path):
    gen = _TimedStubGenerator()
    specs = [
        {"name": "ship", "prompt": "pixel ship", "size": (64, 64)},
        {"name": "rock", "prompt": "grey rock", "size": (32, 32)},
    ]
    out = assets.generate_assets(specs, tmp_path / "session", image_generator=gen)
    assert set(out.keys()) == {"ship", "rock"}
    stats = gen.last_stats
    assert stats is not None
    assert len(stats) == 2
    for s in stats:
        # Mandatory fields per the spec in agent.py
        assert "name" in s
        assert "prompt" in s
        assert "target_size" in s
        assert "cache_hit" in s
        assert "gen_seconds" in s
        # bg_color/alpha_pixel_ratio populated by the chroma-key pass
        assert "bg_color" in s
        assert "alpha_pixel_ratio" in s
    # First call: cache miss; second call (same specs) should be a hit.
    out2 = assets.generate_assets(specs, tmp_path / "session2", image_generator=gen)
    assert all(s["cache_hit"] for s in gen.last_stats), gen.last_stats


def test_22_per_asset_stats_capture_chromakey(tmp_path):
    """The white-bg StubGenerator → 1.3's chroma-key should fire and
    record bg_color + alpha_pixel_ratio."""
    gen = _TimedStubGenerator()
    specs = [{"name": "x", "prompt": "p", "size": (64, 64)}]
    assets.generate_assets(specs, tmp_path / "session", image_generator=gen)
    s = gen.last_stats[0]
    assert s["bg_color"] == [255, 255, 255]   # white detected
    assert s["alpha_pixel_ratio"] > 0.5       # most pixels masked


# ---------------------------------------------------------------------------
# 2.3 — code_snapshot trace event includes html_sha256
# ---------------------------------------------------------------------------


def test_23_code_snapshot_trace_includes_sha256():
    src = Path("agent.py").read_text()
    # The new trace block emits "code_snapshot" with html_sha256.
    assert '"kind": "code_snapshot"' in src
    assert "html_sha256" in src


# ---------------------------------------------------------------------------
# 2.4 — error-source split + formatter rendering
# ---------------------------------------------------------------------------


def _stub_report() -> dict:
    return {
        "ok": False,
        "errors": [],
        "console_errors": [],
        "page_errors": [],
        "probe_errors": [],
        "warnings": [],
        "soft_warnings": [],
        "logs": [],
        "title": "t",
        "canvas": {"width": 800, "height": 600, "blank": False, "raf_ran": True},
        "input_listeners": {"total": 1, "document": 0, "window": 1, "body": 0, "other": 0},
        "input_test": None,
        "frozen_canvas": False,
        "body_chars": 100,
        "body_sample": "...",
    }


def test_24_format_renders_page_errors_separately():
    r = _stub_report()
    r["page_errors"] = ["UNCAUGHT TypeError: foo is not a function"]
    r["console_errors"] = ["console.error: bar"]
    r["errors"] = r["page_errors"] + r["console_errors"]
    text = tools.format_report_for_model(r)
    # Page errors get the loud header, console errors get a milder one.
    assert "PAGE ERRORS" in text
    assert "CONSOLE ERRORS" in text
    # The combined "ERRORS (must fix):" section should NOT appear when
    # split feeds are populated.
    assert "ERRORS (must fix)" not in text


def test_24_format_falls_back_to_union_when_split_missing():
    """Sync test_html_file path doesn't populate the split lists. The
    formatter must fall back to report['errors'] in that case."""
    r = _stub_report()
    r["errors"] = ["UNCAUGHT TypeError: foo is not a function"]
    # console_errors / page_errors stay empty (sync path).
    text = tools.format_report_for_model(r)
    assert "ERRORS (must fix)" in text
    assert "TypeError" in text
    # No PAGE ERRORS / CONSOLE ERRORS headers when split is empty.
    assert "PAGE ERRORS" not in text
    assert "CONSOLE ERRORS" not in text


def test_24_taint_downgrade_logic_in_source():
    """Verify the runner has the downgrade branch for tainted-canvas
    probe failures (so a bogus harness-side error doesn't gate ship)."""
    src = Path("tools.py").read_text()
    assert "downgraded" in src
    # The downgrade triggers on the taint phrases.
    assert "tainted" in src
