"""Tests for _resolve_seed_target — the chat.py helper that maps a
seed path back to the canonical games/<basename>.html so /seed of a
snapshot or a .best.html still reuses the original game's folders.

Without this, /seed games/snapshots/snake_x/iter_03.html would set
_session_id to "iter_03" and new asset generation would land in
games/snapshots/snake_x/iter_03_assets/ — the wrong place at the
wrong level. Same for .best.html (stem = "snake_x.best" ≠ basename).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import _resolve_seed_target  # noqa: E402


def test_live_html_passes_through() -> None:
    seed = Path("games/snake_20260101_120000.html")
    assert _resolve_seed_target(seed) == seed


def test_best_html_drops_suffix() -> None:
    seed = Path("games/snake_20260101_120000.best.html")
    assert _resolve_seed_target(seed) == Path(
        "games/snake_20260101_120000.html"
    )


def test_snapshot_resolves_to_canonical() -> None:
    seed = Path("games/snapshots/snake_20260101_120000/iter_03.html")
    assert _resolve_seed_target(seed) == Path(
        "games/snake_20260101_120000.html"
    )


def test_snapshot_with_double_digit_iter() -> None:
    seed = Path("games/snapshots/snake_20260101_120000/iter_12.html")
    assert _resolve_seed_target(seed) == Path(
        "games/snake_20260101_120000.html"
    )


def test_unrelated_html_passes_through() -> None:
    # A seed from outside the games/ tree (e.g. ~/Downloads/foo.html)
    # has no canonical pair to map to, so we return it unchanged.
    seed = Path("/Users/jr/Downloads/random.html")
    assert _resolve_seed_target(seed) == seed


def test_non_snapshot_dir_named_snapshots_does_not_misfire() -> None:
    # "snapshots" must be the grandparent literally AND the stem must
    # start with "iter_" — a coincidentally-named dir won't trigger.
    seed = Path("games/snake_x/iter_03.html")
    # Grandparent is "games", not "snapshots" → pass through.
    assert _resolve_seed_target(seed) == seed


def test_snapshot_without_iter_prefix_passes_through() -> None:
    # If someone manually saved a snapshot with a non-iter_ name,
    # we don't claim to know its basename — pass through.
    seed = Path("games/snapshots/snake_x/manual_save.html")
    assert _resolve_seed_target(seed) == seed
