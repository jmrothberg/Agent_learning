"""Tests for GameMemory.retrieve_skeleton() — modality detector + fallback.

Added 2026-05-21 after May 20-21 trace evidence showed 4/4 newest sessions
(chess, pac-man, doom, FPS) fell through to canvas_basic.html at score 0.0:
- chess goal "Game of chess human vs computer" -> wrong scaffold (generic)
- pac-man goal "pac man with ghosts" -> missed canvas_grid_basic.html
- doom / FPS -> missed canvas_3d_basic.html

The modality detector + threshold relaxation + v1->v2 fallback fix these
exactly. These tests are the regression guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import GameMemory  # noqa: E402


def _gm(tmp_path: Path) -> GameMemory:
    """Fresh memory rooted in tmp_path so live/base distinction is clean."""
    gm = GameMemory(root=str(tmp_path / "memory"))
    gm.ensure()
    return gm


# ---------------------------------------------------------------------------
# Modality detector — strong-hook single-token matches
# ---------------------------------------------------------------------------


def test_chess_picks_board_turn(tmp_path: Path) -> None:
    """Single decisive hook 'chess' wins on its own — the May 21 chess
    trace failure mode is fixed by this pick."""
    hit = _gm(tmp_path).retrieve_skeleton("Game of chess human vs computer")
    assert hit.name == "canvas_board_turn_basic.html"


def test_checkers_picks_board_turn(tmp_path: Path) -> None:
    hit = _gm(tmp_path).retrieve_skeleton("build checkers vs computer")
    assert hit.name == "canvas_board_turn_basic.html"


def test_tic_tac_toe_picks_board_turn(tmp_path: Path) -> None:
    """Three-word 'tic tac toe' joins to 'tictactoe' via _modality_tokens
    and matches the board strong-hook set. Also matches the DOM hook —
    board wins because 3D is checked first then board (DOM is checked
    last). Both scaffolds would be acceptable; board is the canonical
    choice for turn-based games."""
    hit = _gm(tmp_path).retrieve_skeleton("tic tac toe game")
    assert hit.name == "canvas_board_turn_basic.html"


def test_doom_picks_3d(tmp_path: Path) -> None:
    """'doom' is a 3D strong hook (rendering modality, not the game
    title — the matched tokens describe scanline raycaster perspective
    that doom-likes share)."""
    hit = _gm(tmp_path).retrieve_skeleton("first person doom shooter")
    assert hit.name == "canvas_3d_basic.html"


def test_first_person_compound_picks_3d(tmp_path: Path) -> None:
    """Two-word 'first person' joins to 'firstperson' and matches 3D
    hooks."""
    hit = _gm(tmp_path).retrieve_skeleton("a first person webgl game")
    assert hit.name == "canvas_3d_basic.html"


def test_calculator_picks_dom(tmp_path: Path) -> None:
    """'calculator' is a DOM strong hook; no canvas keyword present so
    DOM scaffold wins."""
    hit = _gm(tmp_path).retrieve_skeleton("build a calculator")
    assert hit.name == "canvas_dom_basic.html"


def test_dom_modality_detector_skips_on_canvas_hint(tmp_path: Path) -> None:
    """The detector itself must NOT match DOM intent when canvas / sprite
    keywords are present (rendering-modality contradicts DOM). Whether
    the downstream Jaccard path still surfaces DOM via sidecar overlap
    is acceptable — the DETECTOR contract is what we're pinning."""
    from memory import _detect_dom_intent
    assert _detect_dom_intent("an animated canvas calculator with sprites") == []
    # Sanity: pure DOM goal still matches.
    assert _detect_dom_intent("a calculator app") != []


# ---------------------------------------------------------------------------
# Jaccard fallback (bundled sidecar threshold relaxation)
# ---------------------------------------------------------------------------


def test_pacman_picks_grid_scaffold(tmp_path: Path) -> None:
    """'pac man with ghosts' tokenizes to ['pac','man','ghosts']. After
    the May 21 sidecar update (pac/man/ghost/ghosts added) the Jaccard
    score is ~0.23 — below the 0.30 threshold that was previously applied
    blanket. The 2026-05-21 fix exempts BUNDLED skeletons from that
    threshold (only past-win files keep it), so grid scaffold wins."""
    hit = _gm(tmp_path).retrieve_skeleton("pac man with ghosts")
    assert hit.name == "canvas_grid_basic.html"


def test_sokoban_picks_grid_scaffold(tmp_path: Path) -> None:
    hit = _gm(tmp_path).retrieve_skeleton("sokoban puzzle")
    assert hit.name == "canvas_grid_basic.html"


# ---------------------------------------------------------------------------
# v2 fallback (replaces v1 canvas_basic.html as the universal default)
# ---------------------------------------------------------------------------


def test_asteroids_falls_to_v2(tmp_path: Path) -> None:
    """Asteroids has no modality or sidecar match. Locked 2026-05-21:
    the new fallback is canvas_basic_v2.html (was canvas_basic.html).
    v2 pre-empts the focus-blur / dt-cap / restart-cleanup failures that
    the May 20-21 traces hit repeatedly."""
    hit = _gm(tmp_path).retrieve_skeleton("asteroids")
    assert hit.name == "canvas_basic_v2.html"


def test_snake_falls_to_v2(tmp_path: Path) -> None:
    """Snake goal has no modality match; falls to v2."""
    hit = _gm(tmp_path).retrieve_skeleton("build me a snake game")
    assert hit.name == "canvas_basic_v2.html"


def test_empty_goal_falls_to_v2(tmp_path: Path) -> None:
    hit = _gm(tmp_path).retrieve_skeleton("")
    assert hit.name == "canvas_basic_v2.html"


# ---------------------------------------------------------------------------
# v1 still reachable as `default` skeleton_mode (tune baseline)
# ---------------------------------------------------------------------------


def test_v1_bootstrapped_to_disk(tmp_path: Path) -> None:
    """v1 stays on disk so skeleton_mode='default' (in tune) can read it
    even though retrieve_skeleton no longer surfaces it."""
    gm = _gm(tmp_path)
    v1 = gm.base_skeletons_dir / "canvas_basic.html"
    assert v1.exists(), "v1 must remain bootstrapped for tune --skeleton-mode default"


def test_v2_bootstrapped_with_sidecar(tmp_path: Path) -> None:
    gm = _gm(tmp_path)
    v2 = gm.base_skeletons_dir / "canvas_basic_v2.html"
    sidecar = v2.with_suffix(".json")
    assert v2.exists()
    assert sidecar.exists()


def test_board_and_dom_bootstrapped(tmp_path: Path) -> None:
    gm = _gm(tmp_path)
    assert (gm.base_skeletons_dir / "canvas_board_turn_basic.html").exists()
    assert (gm.base_skeletons_dir / "canvas_dom_basic.html").exists()
    assert (gm.base_skeletons_dir / "canvas_board_turn_basic.json").exists()
    assert (gm.base_skeletons_dir / "canvas_dom_basic.json").exists()
