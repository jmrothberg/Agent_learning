"""Status-panel: "Memory in use" block.

The user asked (2026-05-24) that the status panel show how and when
memory items are being used so they can see they're in use. Without
this row the memory layers (skeleton selection, visual playtest
recipe matching, auto-probe injection, opening-book retrieval) are
invisible to the user — they fire silently in the trace.

`_render_memory_block` in chat.py shows three groups when populated:
  - Skeleton (selected at session start)
  - Visual playtest recipe (mechanism-keyed, with auto-probe names)
  - Opening-book hits this turn (outline / playtest / asset_audit /
    animation_audit IDs)

The (older) playbook bullet row stays in its own
`_render_playbook_block` — separate visual section.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


def _app_stub() -> CodingBoxApp:
    """Construct a CodingBoxApp without going through Textual init —
    matching the existing test_auto_staff pattern."""
    app = CodingBoxApp.__new__(CodingBoxApp)
    return app


def _agent_stub(**fields) -> MagicMock:
    """Minimal agent stand-in carrying the fields the block reads."""
    a = MagicMock()
    defaults = {
        "_active_skeleton": None,
        "_active_visual_playtest_recipe_id": None,
        "_active_visual_playtest_auto_probes": [],
        "_active_opening_book_recipes": [],
    }
    defaults.update(fields)
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


# ----------------------------------------------------------------------
# Empty case — render block is silent when no memory is active.
# ----------------------------------------------------------------------


def test_empty_agent_renders_nothing() -> None:
    """When nothing's been retrieved yet the block is empty (so it
    doesn't pollute the panel with placeholder text on iter 0)."""
    app = _app_stub()
    app.agent = _agent_stub()
    out = app._render_memory_block()
    assert out == ""


def test_no_agent_renders_nothing() -> None:
    """Defensive: before a session starts, agent is None."""
    app = _app_stub()
    app.agent = None
    out = app._render_memory_block()
    assert out == ""


# ----------------------------------------------------------------------
# Skeleton row.
# ----------------------------------------------------------------------


def test_skeleton_row_shows_active_name() -> None:
    app = _app_stub()
    app.agent = _agent_stub(_active_skeleton="canvas_3d_basic.html")
    out = app._render_memory_block()
    assert "Memory in use" in out
    assert "skeleton:" in out
    assert "canvas_3d_basic.html" in out


# ----------------------------------------------------------------------
# Visual playtest recipe row.
# ----------------------------------------------------------------------


def test_visual_playtest_recipe_shows_id() -> None:
    app = _app_stub()
    app.agent = _agent_stub(
        _active_visual_playtest_recipe_id="canvas-two-actors-facing",
    )
    out = app._render_memory_block()
    assert "vlm-critique checklist:" in out
    assert "canvas-two-actors-facing" in out


def test_visual_playtest_recipe_shows_auto_probes_when_present() -> None:
    app = _app_stub()
    app.agent = _agent_stub(
        _active_visual_playtest_recipe_id="canvas-two-actors-facing",
        _active_visual_playtest_auto_probes=["auto_actors_face_each_other"],
    )
    out = app._render_memory_block()
    assert "1 auto-probe(s)" in out
    assert "auto_actors_face_each_other" in out


def test_visual_playtest_recipe_no_auto_probes_omits_label() -> None:
    """Recipes without auto_probes (e.g. canvas-vfx-fluid) still show
    the recipe id but skip the auto-probe addendum."""
    app = _app_stub()
    app.agent = _agent_stub(
        _active_visual_playtest_recipe_id="canvas-vfx-fluid",
        _active_visual_playtest_auto_probes=[],
    )
    out = app._render_memory_block()
    assert "canvas-vfx-fluid" in out
    assert "auto-probe" not in out


# ----------------------------------------------------------------------
# Opening-book hits row.
# ----------------------------------------------------------------------


def test_opening_book_hits_grouped_by_kind() -> None:
    """Hits should group cleanly by kind so the user sees outline /
    playtest / asset_audit / animation_audit on separate lines instead
    of one long blob."""
    app = _app_stub()
    app.agent = _agent_stub(
        _active_opening_book_recipes=[
            {"kind": "outline", "id": "outline-controllable-canvas-game", "score": 0.5, "recipe": {}},
            {"kind": "playtest", "id": "controllable-movement-delta", "score": 0.3, "recipe": {}},
            {"kind": "playtest", "id": "held-key-stays-in-bounds", "score": 0.2, "recipe": {}},
            {"kind": "asset_audit", "id": "generated-assets-loaded-and-drawn", "score": 0.4, "recipe": {}},
            {"kind": "animation_audit", "id": "movement-has-midframe", "score": 0.3, "recipe": {}},
        ],
    )
    out = app._render_memory_block()
    assert "opening book (this turn):" in out
    assert "outline: outline-controllable-canvas-game" in out
    # Two playtests on the same line.
    assert "playtest: controllable-movement-delta, held-key-stays-in-bounds" in out
    assert "asset_audit: generated-assets-loaded-and-drawn" in out
    assert "animation_audit: movement-has-midframe" in out


def test_opening_book_no_hits_omits_section() -> None:
    """When the opening book pulls nothing this turn, the section
    doesn't appear (rather than rendering an empty header)."""
    app = _app_stub()
    app.agent = _agent_stub(
        _active_skeleton="canvas_basic_v2.html",
        _active_opening_book_recipes=[],
    )
    out = app._render_memory_block()
    assert "opening book" not in out


# ----------------------------------------------------------------------
# Full integration — all three rows together.
# ----------------------------------------------------------------------


def test_full_memory_block_layout() -> None:
    """End-to-end: skeleton + visual playtest + opening-book all render
    as one Memory in use block in the expected order."""
    app = _app_stub()
    app.agent = _agent_stub(
        _active_skeleton="canvas_grid_basic.html",
        _active_visual_playtest_recipe_id="canvas-grid-navigation",
        _active_visual_playtest_auto_probes=["auto_player_not_in_wall"],
        _active_opening_book_recipes=[
            {"kind": "outline", "id": "outline-controllable-canvas-game", "score": 0.5, "recipe": {}},
            {"kind": "playtest", "id": "controllable-movement-delta", "score": 0.3, "recipe": {}},
        ],
    )
    out = app._render_memory_block()
    # Single header at the top.
    assert out.count("Memory in use") == 1
    # Order: skeleton → visual playtest → opening book.
    sk = out.index("skeleton:")
    vp = out.index("vlm-critique checklist:")
    ob = out.index("opening book")
    assert sk < vp < ob
