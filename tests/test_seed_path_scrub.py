"""Seed-skeleton asset/sound path scrubbing in `first_build_instruction`.

When the skeleton retriever picks a `won_<other_session>.html` to seed a
new build, that skeleton hard-codes the previous session's asset/sound
directory names as JS constants (e.g.
`const ASSET_DIR = './my-game_20260511_1234_assets'`). Any model — small
or large — will copy those constants verbatim, and the new session's
sprites/sounds silently fail to load because the directory doesn't
exist.

The fix rewrites the path literals BEFORE the seed reaches the model:
- If we know the current session's directory names, substitute them
  (so the seed remains copy-pasteable and just works).
- Otherwise replace with a self-describing sentinel that fails LOUDLY
  at runtime (so a leaked copy is obvious in console errors, not
  silently broken).

Traced to classic-doom 20260512_153449 (DeepSeek-V4-Flash) but the
failure shape generalizes — every model exhibits it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts_v1 import _scrub_seed_paths, first_build_instruction  # noqa: E402


# ---------------------------------------------------------------------------
# _scrub_seed_paths — the pure regex helper.
# ---------------------------------------------------------------------------


def test_scrub_replaces_stale_assets_with_current_dir():
    seed = "const ASSET_DIR = './old-game_20260101_assets';"
    out = _scrub_seed_paths(
        seed,
        current_asset_dir="new-game_20260512_assets",
        current_sound_dir=None,
    )
    assert "./new-game_20260512_assets" in out
    assert "old-game_20260101_assets" not in out


def test_scrub_replaces_stale_sounds_with_current_dir():
    seed = "const SOUND_DIR = './old-game_20260101_sounds';"
    out = _scrub_seed_paths(
        seed,
        current_asset_dir=None,
        current_sound_dir="new-game_20260512_sounds",
    )
    assert "./new-game_20260512_sounds" in out
    assert "old-game_20260101_sounds" not in out


def test_scrub_substitutes_sentinel_when_no_current_dir():
    """When current_asset_dir is None, a loud sentinel goes in so a
    leaked path fails fast at runtime instead of silently rendering
    nothing."""
    seed = "const ASSET_DIR = './doom_20260101_assets';"
    out = _scrub_seed_paths(
        seed, current_asset_dir=None, current_sound_dir=None,
    )
    assert "STALE_PATH" in out
    assert "GENERATED_ASSETS_BLOCK_ABOVE" in out
    assert "doom_20260101_assets" not in out


def test_scrub_handles_both_kinds_in_one_pass():
    seed = (
        "const ASSET_DIR = './foo_assets';\n"
        "const SOUND_DIR = './foo_sounds';\n"
    )
    out = _scrub_seed_paths(
        seed,
        current_asset_dir="bar_assets",
        current_sound_dir="bar_sounds",
    )
    assert "./bar_assets" in out
    assert "./bar_sounds" in out
    assert "foo_assets" not in out
    assert "foo_sounds" not in out


def test_scrub_leaves_unrelated_paths_alone():
    """Other path-like strings must not be touched."""
    seed = (
        "const CDN = './lib/three.min.js';\n"
        "const CSS = './styles/main.css';\n"
        "const ICONS = './icons_set/foo.png';\n"  # ends with _set, not _assets/_sounds
    )
    out = _scrub_seed_paths(
        seed,
        current_asset_dir="x_assets",
        current_sound_dir="x_sounds",
    )
    assert out == seed


def test_scrub_handles_empty_and_pathless_input():
    assert _scrub_seed_paths("", current_asset_dir="x", current_sound_dir="y") == ""
    seed = "const COLOR = '#ff0000'; // no paths here"
    assert _scrub_seed_paths(seed, current_asset_dir="a", current_sound_dir="b") == seed


def test_scrub_only_consumes_directory_name_not_trailing_filename():
    """`./foo_assets/sprite.png` → `./current_assets/sprite.png` (the
    filename inside the directory stays intact)."""
    seed = "img.src = './old_assets/demon.png';"
    out = _scrub_seed_paths(
        seed, current_asset_dir="new_assets", current_sound_dir=None,
    )
    assert "./new_assets/demon.png" in out
    assert "old_assets" not in out


# ---------------------------------------------------------------------------
# first_build_instruction — end-to-end: the seed inside the prompt is
# scrubbed before injection.
# ---------------------------------------------------------------------------


def test_first_build_instruction_scrubs_seed_paths():
    seed = (
        "<script>\n"
        "const ASSET_DIR = './doom_20260101_assets';\n"
        "const SOUND_DIR = './doom_20260101_sounds';\n"
        "</script>"
    )
    out = first_build_instruction(
        seed, seed_source="some past game",
        current_asset_dir="newgame_20260512_assets",
        current_sound_dir="newgame_20260512_sounds",
    )
    assert "newgame_20260512_assets" in out
    assert "newgame_20260512_sounds" in out
    assert "doom_20260101_assets" not in out
    assert "doom_20260101_sounds" not in out


def test_first_build_instruction_uses_sentinel_when_no_current_dirs():
    """No current paths known → leaked copies should fail loudly."""
    seed = "const ASSET_DIR = './doom_20260101_assets';"
    out = first_build_instruction(seed, seed_source="x")
    assert "STALE_PATH" in out
    assert "doom_20260101_assets" not in out


def test_first_build_instruction_unchanged_when_seed_has_no_stale_paths():
    """The bundled default skeleton has no session-specific asset paths.
    Scrubbing should be a no-op in that case."""
    seed = (
        "<canvas id='c'></canvas>\n"
        "<script>const W=800, H=600;</script>\n"
    )
    out_with = first_build_instruction(
        seed, seed_source="default",
        current_asset_dir="any_assets",
        current_sound_dir="any_sounds",
    )
    out_without = first_build_instruction(seed, seed_source="default")
    # Body identical except for the absence of substitutions.
    assert seed in out_with
    assert seed in out_without
