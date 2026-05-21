"""Tests for Playbook.retrieve(..., modality_tokens=...) — the token
expansion path added 2026-05-21.

Evidence: May 21 FPS trace retrieved `pathfinding-bfs-grid` and
`tetris-matrix-rotation` for a Doom goal because Jaccard over short goals
is too sparse to discriminate. Token expansion via the modality detector
appends rendering-shape keywords to the query so modality-tagged bullets
clear the noise floor and irrelevant bullets do not win on a single
coincidental token match.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import Playbook  # noqa: E402


def _pb(tmp_path: Path) -> Playbook:
    pb = Playbook(root=str(tmp_path / "memory"))
    pb.ensure()
    return pb


def test_pacman_modality_expansion_boosts_corner_sliding(tmp_path: Path) -> None:
    """Pac-man goal alone barely matches corner-sliding-alignment.
    With modality tokens grid/tile/maze/pacman, the bullet's tags
    (which include 'pacman', 'grid', 'corridor') match a much higher
    share of the expanded query and the bullet jumps to the top."""
    pb = _pb(tmp_path)

    baseline = pb.retrieve("pac man with ghosts", stage="plan", k=5)
    baseline_ids = [h.bullet.id for h in baseline]
    baseline_corner_score = next(
        (h.score for h in baseline if h.bullet.id == "corner-sliding-alignment"),
        0.0,
    )

    expanded = pb.retrieve(
        "pac man with ghosts", stage="plan", k=5,
        modality_tokens=["grid", "tile", "maze", "pacman", "sokoban", "corridor"],
    )
    expanded_ids = [h.bullet.id for h in expanded]
    expanded_corner_score = next(
        (h.score for h in expanded if h.bullet.id == "corner-sliding-alignment"),
        0.0,
    )

    # Token expansion must lift, not depress.
    assert expanded_corner_score > baseline_corner_score, (
        f"corner-sliding score should rise; got "
        f"{baseline_corner_score:.4f} -> {expanded_corner_score:.4f}"
    )
    # And it should now retrieve (was at noise floor before).
    assert "corner-sliding-alignment" in expanded_ids


def test_doom_modality_expansion_drops_tetris_noise(tmp_path: Path) -> None:
    """The May 21 FPS trace's biggest indicator of retrieval being
    broken was `tetris-matrix-rotation` retrieved for a Doom goal —
    pure Jaccard noise (one-token coincidence on 'rotation' / 'matrix').
    With 3D modality tokens appended, more 3D-tagged bullets compete
    and tetris loses its top-K slot."""
    pb = _pb(tmp_path)

    expanded = pb.retrieve(
        "first person doom shooter", stage="plan", k=5,
        modality_tokens=[
            "3d", "firstperson", "perspective", "raycaster", "wolfenstein",
        ],
    )
    ids = [h.bullet.id for h in expanded]
    assert "tetris-matrix-rotation" not in ids, (
        f"tetris should not retrieve for Doom; got {ids}"
    )


def test_empty_modality_tokens_is_noop(tmp_path: Path) -> None:
    """Passing modality_tokens=[] (or None) must NOT change scoring vs
    the default code path — backwards-compatible for callers that don't
    supply tokens."""
    pb = _pb(tmp_path)
    baseline = pb.retrieve("asteroids ship", stage="code", k=3)
    no_tokens = pb.retrieve("asteroids ship", stage="code", k=3, modality_tokens=[])
    none_tokens = pb.retrieve("asteroids ship", stage="code", k=3, modality_tokens=None)

    base_ids = [h.bullet.id for h in baseline]
    assert [h.bullet.id for h in no_tokens] == base_ids
    assert [h.bullet.id for h in none_tokens] == base_ids


def test_seed_bullets_include_new_2026_05_21_additions(tmp_path: Path) -> None:
    """Sanity: the 5 new seed bullets (4 mechanic + 1 promoted) are
    present and retrieve for relevant goals."""
    pb = _pb(tmp_path)
    all_ids = {b.id for b in pb.load_all()}
    assert "turn-based-select-move" in all_ids
    assert "board-grid-indexing" in all_ids
    assert "click-cell-from-pointer" in all_ids
    assert "expose-state-on-window" in all_ids
    # Promoted from live -> seed.
    assert "probe-warmup-state-exposure" in all_ids


def test_window_state_bullet_retrieves_for_probe_signature(tmp_path: Path) -> None:
    """The expose-state-on-window bullet must retrieve when the goal /
    code includes the probe phrasing. This is the highest-frequency
    failure across the May 20-21 traces."""
    pb = _pb(tmp_path)
    hits = pb.retrieve(
        "game where probes need window.gameState exposed",
        stage="plan", k=5,
    )
    ids = [h.bullet.id for h in hits]
    assert "expose-state-on-window" in ids
