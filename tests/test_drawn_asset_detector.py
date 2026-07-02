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


def test_asset_decode_settle_wait_before_undrawn_audit():
    """Chromium may read __drawImageEvents before async loadAssets() finishes.
    The harness must poll _assetsReady / ASSETS decode, tick extra RAF frames,
    and expose asset_decode_settle on the report before the undrawn diff."""
    src = inspect.getsource(tools_module)
    assert "_wait_for_session_assets_ready" in src
    assert 'report["asset_decode_settle"]' in src
    assert "_ASSET_DECODE_SETTLE_JS" in src
    assert "window._assetsReady===true" in src


def test_drawn_blob_uses_path_boundary_fname_match():
    """Stem/filename matching must not false-positive on substrings
    (e.g. idle.png inside blue_idle.png)."""
    assert tools_module._drawn_blob_contains_asset_fname(
        "file:///tmp/foo/blue_idle.png?x=1", "blue_idle.png"
    )
    assert not tools_module._drawn_blob_contains_asset_fname(
        "file:///tmp/foo/blue_idle.png", "idle.png"
    )


def test_decode_timing_demotion_when_settle_and_probes_green():
    """When decode settled and the canvas is advancing with green probes,
    ASSETS_LOADED_BUT_UNDRAWN demotes to advisory (Chromium timing FP)."""
    src = inspect.getsource(tools_module)
    assert "_decode_timing_demote" in src
    assert "Chromium timing false positive" in src


# ---------------------------------------------------------------------------
# Undrawn-FP fixes (run_10 Q*bert trace 20260702_204842)
# ---------------------------------------------------------------------------

def test_directional_hop_and_bounce_stems_are_pose_frames():
    """hero_hop_ul / enemy_bounce style stems must classify as animation
    poses, and their idle counterpart must resolve through the directional
    tail (hero_hop_ul -> hero_idle)."""
    assert tools_module._stem_looks_like_animation_pose("hero_hop_ul")
    assert tools_module._stem_looks_like_animation_pose("enemy_bounce")
    assert tools_module._stem_looks_like_animation_pose("enemy_bounce_2")
    blob = "file:///tmp/x_assets/hero_idle.png\nfile:///tmp/x_assets/enemy_idle.png"
    assert tools_module._idle_counterpart_drawn("hero_hop_ul", blob)
    assert tools_module._idle_counterpart_drawn("enemy_bounce_2", blob)
    # Base sprite NOT drawn -> no counterpart credit.
    assert not tools_module._idle_counterpart_drawn("ghost_hop_dr", blob)


def test_pose_only_undrawn_set_with_directional_hops():
    """A Q*bert-shaped undrawn set (all 4 diagonal hop frames + enemy
    bounce) counts as poses-only when the idle bases drew."""
    blob = "file:///g_assets/hero_idle.png\nfile:///g_assets/enemy_idle.png"
    undrawn = ["hero_hop_ul", "hero_hop_ur", "hero_hop_dl", "hero_hop_dr", "enemy_bounce"]
    assert tools_module._undrawn_are_animation_poses_only(undrawn, blob)


def test_sprite_draw_proven_demotes_undrawn_gate():
    """When the input smoke test recorded a new sprite src on a keypress
    (fake_actions[key].new_sprite_src), the undrawn finding must demote to
    advisory instead of gating — the game provably draws sprites."""
    src = inspect.getsource(tools_module)
    assert "_sprite_draw_proven" in src
    assert '"sprite_draw_proven"' in src
    # The demote condition includes the proven-draw path.
    assert "_sprite_draw_proven and _no_errors and _game_advancing" in src


def test_asset_settle_runs_before_observation_window():
    """The decode settle must run BEFORE the frozen-canvas observation
    window / input smoke test, not only right before the undrawn audit —
    otherwise early drawImage events are missed on slow decodes."""
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    settle_idx = src.index("_wait_for_session_assets_ready")
    window_idx = src.index("Sleep half the budget")
    assert settle_idx < window_idx, (
        "asset decode settle must be called before the observation window"
    )


def test_iter_summary_persists_drawn_asset_check():
    """iter_summary trace events must carry drawn_asset_check and
    asset_decode_settle so undrawn-FP triage works from the .jsonl alone."""
    import agent as agent_module
    src = inspect.getsource(agent_module)
    anchor = src.index('"kind": "iter_summary"')
    block = src[anchor:anchor + 6000]
    assert '"drawn_asset_check"' in block
    assert '"asset_decode_settle"' in block


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
