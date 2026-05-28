"""Procedural-regression detector — catches "model declared sprites
but draws colored rectangles instead" failures.

Companion to ASSETS_LOADED_BUT_UNDRAWN (drawImage shim). That detector
says WHICH assets weren't drawn; this one says WHAT was drawn in their
place. Combined: model + harness gets a clear "you're regressing
entities to fillRect placeholders" signal.

Motivating trace: mortal-kombat 2026-05-24 iter 5 — VLM critic
described P2 as "a massive solid blue rectangle rather than a
character sprite". The drawImage detector would have flagged it
eventually (sprite was loaded, never drawn) but the cause signal
(big fillRect being drawn instead) was invisible to the harness
before this commit.

Structural tests (inspect tools.py source) + a genre-freeness check.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402


def test_fillrect_shim_present_in_init_js():
    """fillRect must be monkey-patched the same way drawImage is, with
    records pushed into window.__fillRectEvents."""
    src = inspect.getsource(tools_module)
    assert "window.__fillRectEvents = []" in src
    assert "CanvasRenderingContext2D.prototype.fillRect" in src
    assert "window.__fillRectEvents.push" in src


def test_fillrect_shim_filters_to_big_rects_only():
    """UI elements (HUD bars, borders, score backgrounds) are typically
    thin in ONE dimension (e.g., 200×16 health bar). The shim must
    filter to ≥32×32 in BOTH dimensions so it only catches
    entity-sized placeholders."""
    src = inspect.getsource(tools_module)
    # Threshold must be present in the JS shim.
    assert "aw >= 32" in src
    assert "ah >= 32" in src


def test_fillrect_shim_records_size():
    """Width and height must be captured so the post-iter report can
    cite median area in the soft warning."""
    src = inspect.getsource(tools_module)
    # Records carry w and h fields.
    assert '"w": aw' in src or "w: aw" in src
    assert '"h": ah' in src or "h: ah" in src


def test_fillrect_shim_buffer_is_capped():
    """4000-entry cap, same as drawImage shim, to prevent unbounded
    growth on long-running games."""
    src = inspect.getsource(tools_module)
    assert "__fillRectEvents.length <" in src


def test_harness_reads_fillrect_events():
    """Python-side post-iter check must read window.__fillRectEvents."""
    src = inspect.getsource(tools_module)
    assert 'window.__fillRectEvents || []' in src


def test_harness_emits_procedural_regression_warning():
    """Warning string and key signals must be present."""
    src = inspect.getsource(tools_module)
    assert "PROCEDURAL_REGRESSION_SUSPECTED" in src
    # Concrete actionable hint pointing to ctx.drawImage.
    proc_start = src.index("PROCEDURAL_REGRESSION_SUSPECTED")
    proc_excerpt = src[proc_start:proc_start + 1500]
    assert "ctx.drawImage" in proc_excerpt


def test_procedural_regression_requires_3_plus_sprites():
    """The detector must not false-fire on legitimate procedural games
    (snake, tetris, pong) that have NO sprite assets declared. Gate:
    referenced_assets >= 3."""
    src = inspect.getsource(tools_module)
    # Threshold present.
    assert "len(referenced_assets) >= 3" in src


def test_procedural_regression_requires_ratio_gate():
    """Tile-based backgrounds (e.g., a maze drawn as 30×30 fillRect
    tiles) will produce many big rectangles per frame BUT also draw
    sprites for entities. The detector must use a ratio (big_rect
    must outnumber drawImage by 5:1) so legit tile-background games
    don't false-fire."""
    src = inspect.getsource(tools_module)
    # Ratio gate: big_rect_count > 5 * draw_image_count means draw_image_count < big_rect_count // 5.
    assert "big_rect_count // 5" in src


def test_procedural_regression_requires_raf_and_non_blank():
    """Same guard as the drawImage detector — don't fire on pages
    that haven't started rendering."""
    src = inspect.getsource(tools_module)
    # The block reuses canvas_info checks; verify by counting occurrences
    # of those gate strings around PROCEDURAL_REGRESSION_SUSPECTED.
    proc_start = src.index("PROCEDURAL_REGRESSION_SUSPECTED")
    # Walk backwards from the warning emission to find the if-block.
    block_start = src.rfind("if (", 0, proc_start)
    block_excerpt = src[block_start:proc_start]
    assert 'canvas_info.get("raf_ran")' in block_excerpt
    assert 'canvas_info.get("blank") is False' in block_excerpt


def test_procedural_regression_warning_is_genre_free():
    """Warning text must not mention any specific game / character /
    genre. Structural signal only."""
    src = inspect.getsource(tools_module)
    start = src.index("PROCEDURAL_REGRESSION_SUSPECTED")
    end = start + 2000
    excerpt = src[start:end].lower()
    forbidden = (
        "pacman", "pac-man", "mario", "chess piece", "ghost",
        "doom", "fps", "fighter", "kombat", "street fighter",
        "platformer", "shooter",
    )
    for term in forbidden:
        assert term not in excerpt, (
            f"PROCEDURAL_REGRESSION_SUSPECTED text mentions {term!r} — "
            "the detector must stay genre-free."
        )


def test_procedural_regression_report_field_documented():
    """The detector adds a structured `procedural_regression` field to
    the report carrying the counts, so jsonl trace analysis can mine
    the pattern across sessions without re-parsing the warning text."""
    src = inspect.getsource(tools_module)
    assert 'report["procedural_regression"]' in src
    # Field must carry the counts.
    for key in (
        '"referenced_assets":',
        '"big_rect_count":',
        '"draw_image_count":',
        '"median_rect_area_px":',
    ):
        assert key in src, f"report['procedural_regression'] missing key: {key!r}"
