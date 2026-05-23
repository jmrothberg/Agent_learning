"""Drawn-asset detector — catches "model loaded the PNG into ASSETS[name]
but never called drawImage with it" failures.

The 2026-05-23 chess trace had the model write `new Image() + img.decode()`
on 36 piece sprites and then draw chess pieces via `ctx.fillText` Unicode
glyphs. Loader-presence check passed; the user saw no monsters.

The fix is structural in tools.py:
  1. drawImage shim records every call's source URL into
     window.__drawImageEvents (next to the existing __audioEvents shim).
  2. After the input smoke test, the harness diffs the recorded sources
     against asset PNGs referenced in the HTML.
  3. >=1/3 of referenced assets undrawn + canvas non-blank + RAF firing
     → ASSETS_LOADED_BUT_UNDRAWN soft warning, ok flips False.

Tests are structural (inspect the JS shim + Python harness code) plus
a generic check that the warning shape itself is genre-free.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402


def test_drawimage_shim_present_in_init_js():
    """The shim must monkey-patch CanvasRenderingContext2D.prototype.drawImage
    and push records into window.__drawImageEvents."""
    src = inspect.getsource(tools_module)
    # The shim's three structural anchors.
    assert "window.__drawImageEvents = []" in src
    assert "CanvasRenderingContext2D.prototype.drawImage" in src
    # Records carry the same {t, src} flat shape as __audioEvents.
    assert "window.__drawImageEvents.push" in src


def test_drawimage_shim_captures_source_url_robustly():
    """The shim must capture image.src (full URL) when available, with
    sensible fallbacks for non-image draw sources (canvas, ImageBitmap,
    generic HTML elements). Failure to capture src would produce no
    matchable entries, defeating the detector."""
    src = inspect.getsource(tools_module)
    # Every fallback path must exist.
    for pattern in (
        "image.src",
        "image.currentSrc",
        "HTMLCanvasElement",
        "ImageBitmap",
    ):
        assert pattern in src, (
            f"drawImage shim must handle source kind: {pattern!r}"
        )


def test_drawimage_shim_buffer_is_capped():
    """A long-running game may fire hundreds of drawImage calls per
    frame. The shim must cap the recorded buffer so window.__drawImageEvents
    doesn't grow unboundedly. The harness only needs the SET of sources,
    not the count per source."""
    src = inspect.getsource(tools_module)
    # Some cap exists.
    assert "__drawImageEvents.length <" in src


def test_harness_emits_assets_loaded_but_undrawn_warning():
    """Verify the Python harness side reads __drawImageEvents, builds a
    referenced-asset map from the HTML, and emits the expected warning
    when the intersection is small."""
    src = inspect.getsource(tools_module)
    # Read site.
    assert 'window.__drawImageEvents || []' in src
    # Reads the HTML to find referenced PNG paths.
    assert '_assets/' in src
    # Warning string.
    assert "ASSETS_LOADED_BUT_UNDRAWN" in src
    # Concrete actionable hint (drawImage replacement).
    assert "ctx.drawImage" in src


def test_assets_loaded_but_undrawn_threshold_requires_significant_gap():
    """The check must avoid false-firing when most assets ARE drawn
    and a small minority happens to be unused — that's normal for
    games where some assets are conditional (game-over screen,
    second-level boss, etc.). Threshold: at least 1/3 of referenced
    assets must be undrawn."""
    src = inspect.getsource(tools_module)
    # Threshold formula must be present.
    assert "len(referenced_assets) // 3" in src


def test_drawn_asset_check_requires_raf_and_non_blank():
    """Don't false-fire on pages that haven't started rendering yet.
    The check is gated on RAF having fired AND canvas not being blank;
    if those held and assets are still undrawn, the gap is real."""
    src = inspect.getsource(tools_module)
    assert 'canvas_info.get("raf_ran")' in src
    assert 'canvas_info.get("blank") is False' in src


def test_drawn_asset_check_is_genre_free():
    """The detector must not match by genre / game name. It works on
    SHAPE: a path matching `*_assets/*.png` referenced by the HTML,
    not the entity it represents."""
    # Look at the warning emission text block only.
    src = inspect.getsource(tools_module)
    start = src.index("ASSETS_LOADED_BUT_UNDRAWN")
    end = start + 2000  # generous window past the warning
    excerpt = src[start:end].lower()
    forbidden = ("pacman", "pac-man", "mario", "chess piece", "ghost", "ship")
    for term in forbidden:
        assert term not in excerpt, (
            f"ASSETS_LOADED_BUT_UNDRAWN emission text mentions {term!r} — "
            "the detector must stay genre-free."
        )
