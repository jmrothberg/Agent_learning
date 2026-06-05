"""Cross-session asset library — retrieval + admission + LRU."""
from __future__ import annotations

from pathlib import Path

import pytest

from asset_library import AssetLibrary, _tokenize, _jaccard


def _make_png(path: Path, color=(255, 0, 0)) -> Path:
    """Smallest viable PNG (1x1) for tests that just need a real file."""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color)
    img.save(path, format="PNG")
    return path


def _make_ogg(path: Path) -> Path:
    """Smallest viable OGG. We never play it — admission only checks
    bytes exist on disk."""
    path.write_bytes(b"OggS\x00" + b"\x00" * 16)
    return path


def test_admit_then_retrieve_returns_hit(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    src = _make_png(tmp_path / "ship.png")
    entry = lib.admit(
        prompt="pixel art space ship, blue",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=src,
    )
    assert entry is not None
    # Same prompt + size hits.
    hit = lib.retrieve(
        prompt="pixel art space ship, blue",
        modality="sprite",
        size_or_duration=(64, 64),
    )
    assert hit is not None
    assert hit.entry.id == entry.id
    assert hit.absolute_path.exists()
    # Different size misses.
    miss = lib.retrieve(
        prompt="pixel art space ship, blue",
        modality="sprite",
        size_or_duration=(32, 32),
    )
    assert miss is None
    # Different modality misses.
    miss2 = lib.retrieve(
        prompt="pixel art space ship, blue",
        modality="sound",
        size_or_duration=(64, 64),
    )
    assert miss2 is None


def test_retrieve_jaccard_threshold(tmp_path: Path):
    """A near-but-not-identical prompt with shared tokens should still
    hit; a fully disjoint prompt should miss."""
    lib = AssetLibrary(root=tmp_path)
    _make_png(tmp_path / "src.png")
    lib.admit(
        prompt="pixel space ship",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=tmp_path / "src.png",
    )
    near = lib.retrieve(
        prompt="space ship pixel",
        modality="sprite",
        size_or_duration=(64, 64),
    )
    assert near is not None
    assert near.score >= 0.5
    far = lib.retrieve(
        prompt="walnut tree forest",
        modality="sprite",
        size_or_duration=(64, 64),
    )
    assert far is None


def test_admit_idempotent_on_same_sha(tmp_path: Path):
    """Re-admitting the same bytes does NOT duplicate library files."""
    lib = AssetLibrary(root=tmp_path)
    src = _make_png(tmp_path / "ship.png")
    e1 = lib.admit(
        prompt="pixel ship",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=src,
    )
    e2 = lib.admit(
        prompt="pixel ship",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=src,
    )
    assert e1 is not None and e2 is not None
    assert e1.id == e2.id
    assert len(lib.load_all()) == 1


def test_lru_eviction_caps_library(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path, max_entries=3)
    for i, color in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]):
        src = _make_png(tmp_path / f"src{i}.png", color=color)
        lib.admit(
            prompt=f"ship variant {i}",
            modality="sprite",
            size_or_duration=(64, 64),
            source_path=src,
        )
    entries = lib.load_all()
    assert len(entries) == 3
    # Oldest entry (i=0) was evicted.
    prompts = {e.prompt for e in entries}
    assert "ship variant 0" not in prompts


def test_touch_updates_last_used(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    src = _make_png(tmp_path / "ship.png")
    e = lib.admit(
        prompt="pixel ship",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=src,
    )
    assert e is not None
    before = e.last_used
    # Forge a slight delay then touch.
    import time
    time.sleep(0.01)
    lib.touch(e.id)
    after = lib.load_all()[0].last_used
    assert after > before


def test_sound_duration_tolerance(tmp_path: Path):
    """Sounds match within 10% duration."""
    lib = AssetLibrary(root=tmp_path)
    src = _make_ogg(tmp_path / "boom.ogg")
    lib.admit(
        prompt="explosion crunch",
        modality="sound",
        size_or_duration=1.00,
        source_path=src,
    )
    near = lib.retrieve(
        prompt="explosion crunch",
        modality="sound",
        size_or_duration=1.05,  # within 10%
    )
    assert near is not None
    far = lib.retrieve(
        prompt="explosion crunch",
        modality="sound",
        size_or_duration=2.00,  # outside
    )
    assert far is None


def test_stale_path_returns_none(tmp_path: Path):
    """If the library file has been removed externally, retrieve()
    must NOT return a hit pointing to a missing file."""
    lib = AssetLibrary(root=tmp_path)
    src = _make_png(tmp_path / "ship.png")
    e = lib.admit(
        prompt="pixel ship",
        modality="sprite",
        size_or_duration=(64, 64),
        source_path=src,
    )
    assert e is not None
    # Yank the library file out from under the index.
    (lib.root / e.path).unlink()
    hit = lib.retrieve(
        prompt="pixel ship",
        modality="sprite",
        size_or_duration=(64, 64),
    )
    assert hit is None


def test_tokenize_drops_stopwords_and_lowercases():
    toks = _tokenize("PIXEL Sprite OF a SHIP, transparent background")
    # "sprite", "transparent", "background", "a", "of" are stopwords.
    assert "ship" in toks
    assert "pixel" in toks
    assert "sprite" not in toks
    assert "a" not in toks
    assert all(t == t.lower() for t in toks)


def test_jaccard_disjoint_is_zero_identical_is_one():
    assert _jaccard([], []) == 0.0
    assert _jaccard(["a"], ["b"]) == 0.0
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0


# ---- #5: cross-session pose-prompt recipes ---------------------------------

def test_admit_pose_recipe_records_moved_pose_and_retrieves(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    assert lib.admit_pose_recipe(
        stem="fighter", pose="punch",
        prompt="fighter, arm fully extended forward in a punch", delta=0.21,
    ) is True
    hits = lib.retrieve_pose_recipes("a fighting game with punching", k=4)
    assert any(h["pose"] == "punch" for h in hits)
    assert hits[0]["prompt"].startswith("fighter")


def test_admit_pose_recipe_skips_clones_below_floor(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    # A near-idle clone (delta below the move floor) is not worth keeping.
    assert lib.admit_pose_recipe(
        stem="fighter", pose="punch", prompt="x", delta=0.01,
    ) is False
    assert lib.retrieve_pose_recipes("punch") == []


def test_admit_pose_recipe_dedups_keeping_higher_delta(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    assert lib.admit_pose_recipe(stem="hero", pose="kick", prompt="a", delta=0.10) is True
    # Lower delta for the same (stem,pose) is dropped.
    assert lib.admit_pose_recipe(stem="hero", pose="kick", prompt="b", delta=0.05) is False
    # Higher delta is admitted (a better example of the pose).
    assert lib.admit_pose_recipe(stem="hero", pose="kick", prompt="c", delta=0.30) is True
    all_rows = lib.retrieve_pose_recipes("")  # empty text -> all
    assert len([r for r in all_rows if r["pose"] == "kick"]) == 2


def test_retrieve_pose_recipes_ranks_by_overlap_then_delta(tmp_path: Path):
    lib = AssetLibrary(root=tmp_path)
    lib.admit_pose_recipe(stem="ninja", pose="kick", prompt="kick", delta=0.5)
    lib.admit_pose_recipe(stem="ninja", pose="jump", prompt="jump", delta=0.9)
    hits = lib.retrieve_pose_recipes("ninja kick attack", k=4)
    # "kick" shares 2 tokens (ninja+kick) vs jump's 1 (ninja) -> kick first.
    assert hits[0]["pose"] == "kick"
