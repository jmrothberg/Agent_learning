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


def test_13_chromakey_near_white_5_of_8_when_figure_touches_edge():
    """run_15 Rampage: figure eats left edge samples so only 5/8 agree on
    white — still chroma-key (opaque idle boxes otherwise)."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), (252, 253, 253))
    # Figure touches left corners + left mid — breaks strict 6/8 consensus.
    for y in range(64):
        for x in range(8):
            img.putpixel((x, y), (180, 100, 70))
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert stats["bg_color"] is not None
    assert stats["alpha_pixel_ratio"] > 0.4
    # Right edge was backdrop → transparent.
    assert keyed.getpixel((63, 32))[3] == 0
    # Figure stays opaque.
    assert keyed.getpixel((2, 32))[3] == 255


def test_13_chromakey_border_majority_when_figure_fills_most_edges():
    """run_15 Rampage punch: only ~3/8 edge samples white, but border strip
    is still majority near-white — chroma via border-majority fallback."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), (254, 254, 254))
    # Non-matching edge colors so 8-point consensus fails (like punch1).
    img.putpixel((0, 0), (255, 253, 253))
    img.putpixel((63, 0), (161, 153, 139))
    img.putpixel((0, 63), (4, 4, 3))
    img.putpixel((63, 63), (36, 34, 28))
    img.putpixel((32, 0), (254, 255, 254))
    img.putpixel((32, 63), (253, 255, 253))
    img.putpixel((0, 32), (165, 178, 155))
    img.putpixel((63, 32), (112, 101, 88))
    # Interior figure + most of the canvas fill with opaque body, leaving
    # a white halo near the top/right that dominates the border strip.
    for y in range(8, 64):
        for x in range(0, 56):
            img.putpixel((x, y), (40, 120, 60))
    # Keep a white border band on top + right (backdrop).
    for x in range(64):
        for y in range(0, 6):
            img.putpixel((x, y), (254, 254, 254))
    for y in range(64):
        for x in range(58, 64):
            img.putpixel((x, y), (254, 254, 254))
    keyed, stats = assets._chroma_key_to_rgba(img)
    assert stats["bg_color"] is not None
    assert stats.get("checkerboard_bg") is None
    assert stats["alpha_pixel_ratio"] > 0.15
    assert keyed.getpixel((60, 2))[3] == 0
    assert keyed.getpixel((20, 40))[3] == 255


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


def test_24_strict_runtime_marker_detects_threejs_candidate():
    assert tools._is_threejs_candidate_html(
        "<script src='https://cdn.jsdelivr.net/npm/three@0.160/build/three.min.js'></script>"
    )
    assert not tools._is_threejs_candidate_html("<canvas id='c'></canvas>")


def test_24_strict_failure_classifier_cors():
    kind, summary, hint = tools._classify_strict_file_failure(
        page_errors=["UNCAUGHT: SecurityError: blocked by CORS policy"],
        console_errors=[],
        canvas_info={"raf_ran": True},
    )
    assert kind == "cors_blocked"
    assert "SecurityError" in summary or "security" in summary.lower()
    assert "file://-safe" in hint


def test_24_format_renders_strict_runtime_summary():
    r = _stub_report()
    r["strict_file_runtime"] = {
        "checked": True,
        "status": "fail",
        "failure_type": "cors_blocked",
        "summary": "SecurityError: blocked by CORS policy",
        "hints": ["Use file://-safe texture loading."],
    }
    text = tools.format_report_for_model(r)
    assert "Strict file:// runtime: FAIL [cors_blocked]" in text
    assert "Strict fix hint" in text


def test_24_strict_runtime_check_infra_error_is_nonfatal(monkeypatch):
    class _BrokenPlaywright:
        def __enter__(self):
            raise RuntimeError("playwright unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(tools, "sync_playwright", lambda: _BrokenPlaywright())
    out = tools._run_strict_file_runtime_check(Path("/tmp/nope.html"))
    assert out["status"] == "infra_error"
    assert out["failure_type"] == "infra_error"


def test_24_format_renders_scoped_check_summary():
    r = _stub_report()
    r["scoped_check"] = {
        "required": True,
        "keywords": ["cpu", "facing"],
        "pass": False,
    }
    text = tools.format_report_for_model(r)
    assert "Scoped check: FAIL (cpu, facing)" in text


# ---------------------------------------------------------------------------
# A1 — crash-site source slice (tools.extract_crash_source_slices)
# ---------------------------------------------------------------------------


def test_a1_extracts_source_line_with_arrow(tmp_path):
    """The space-invaders failure mode: the report had a `file://...:482:7`
    stack frame but no line content. With A1, the report carries the
    actual offending source line so a smaller LLM can patch the right
    spot without reverse-line-counting a 700-line file.
    """
    f = tmp_path / "game.html"
    f.write_text(
        "line1\n"
        "line2\n"
        "line3\n"
        "line4\n"
        "    b.y += ALIEN_BULLET_SPEED * dt;  // line5\n"
        "line6\n"
        "line7\n"
        "line8\n"
    )
    err = (
        "game crash: TypeError: Cannot read properties of undefined (reading 'y')\n"
        f"    at update (file://{f}:5:7)\n"
        "    at frame"
    )
    slices = tools.extract_crash_source_slices([err], file_filter=f)
    assert len(slices) == 1
    assert slices[0]["line"] == 5
    assert ">" in slices[0]["snippet"]
    assert "b.y += ALIEN_BULLET_SPEED" in slices[0]["snippet"]
    assert "line4" in slices[0]["snippet"]  # context window includes neighbors


def test_a1_dedup_same_path_line(tmp_path):
    """A 5-deep stack trace that hits the same line repeatedly must not
    produce 5 copies of the same snippet."""
    f = tmp_path / "game.html"
    f.write_text("a\nb\nc\nd\n")
    err = (
        f"E\n    at f (file://{f}:2:1)\n"
        f"    at g (file://{f}:2:1)\n"
        f"    at h (file://{f}:2:1)"
    )
    slices = tools.extract_crash_source_slices([err], file_filter=f)
    assert len(slices) == 1


def test_a1_respects_file_filter(tmp_path):
    """Frames into CDN scripts or framework files (not the game file)
    must be ignored when file_filter is provided — keeps the report focused
    on the user's actual source."""
    game = tmp_path / "game.html"
    framework = tmp_path / "framework.js"
    game.write_text("x\ny\nz\n")
    framework.write_text("a\nb\nc\n")
    err = (
        f"E\n    at f (file://{framework}:2:1)\n"
        f"    at g (file://{game}:1:1)"
    )
    slices = tools.extract_crash_source_slices([err], file_filter=game)
    assert len(slices) == 1
    assert slices[0]["line"] == 1


def test_a1_renders_in_format_report():
    """End-to-end: format_report_for_model emits the SOURCE NEAR ERROR
    section when slices are attached."""
    r = _stub_report()
    r["page_errors"] = ["UNCAUGHT TypeError at file:///x.html:5:1"]
    r["errors"] = list(r["page_errors"])
    r["crash_source_slices"] = [{
        "path": "/tmp/x.html",
        "line": 5,
        "col": 1,
        "snippet": "    4: b\n  > 5: c.y\n    6: d",
    }]
    text = tools.format_report_for_model(r)
    assert "SOURCE NEAR ERROR" in text
    assert "x.html:5" in text
    assert "c.y" in text


def test_a1_no_slice_when_no_stack_frame():
    """A console.error with no file URL must not produce a slice."""
    slices = tools.extract_crash_source_slices(
        ["plain log message with no stack frame here"]
    )
    assert slices == []


def test_format_suppresses_cascading_issues_on_syntax_page_error():
    """Local-LLM noise: syntax page error → drop soft_warning flood."""
    r = _stub_report()
    r["page_errors"] = ["UNCAUGHT: Unexpected token ')'"]
    r["errors"] = list(r["page_errors"])
    r["soft_warnings"] = [
        "HEURISTIC: <canvas> exists but requestAnimationFrame never fired",
        "HEURISTIC: pressed ArrowUp — canvas pixels never changed",
        "PROBE FAILED [player_visible]: state.player undefined",
    ]
    r["probes"] = [
        {"name": "player_visible", "ok": False, "expr": "window.state && state.player"},
        {"name": "canvas_present", "ok": True, "expr": "!!document.querySelector('canvas')"},
    ]
    text = tools.format_report_for_model(r)
    assert "PAGE ERRORS" in text
    assert "Unexpected token" in text
    assert "cascading soft-warning" in text
    assert "requestAnimationFrame never fired" not in text
    assert "details omitted until syntax is fixed" in text
    assert "window.state && state.player" not in text  # full expr suppressed
