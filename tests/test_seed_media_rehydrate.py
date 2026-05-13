"""Tests for seed-file media rehydration.

When a session is started from a seed HTML that already references
`./<prefix>_assets/<name>.png` and/or `./<prefix>_sounds/<name>.ogg`,
the agent should:

  1. Pre-populate `_session_assets` / `_session_sounds` from any of
     those files that exist on disk.
  2. If all references share a single prefix, redirect mid-session
     generation into that EXISTING folder so new sprites/sounds merge
     in alongside the originals instead of creating a sibling
     `<new_basename>_assets/` dir.

Pure-function tests cover the scanner directly. An end-to-end test
exercises the GameAgent constructor + the run-loop seed branch with
mocked diffuser/browser so we never load weights or talk to a model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent, _scan_seed_media  # noqa: E402


# ---------------------------------------------------------------------------
# _scan_seed_media — pure scanner
# ---------------------------------------------------------------------------

def _write_pngs(dirpath: Path, names: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dirpath / n).write_bytes(b"\x89PNG fake")


def test_scan_picks_up_existing_asset_refs(tmp_path: Path) -> None:
    seed = tmp_path / "snake_20260101_120000.html"
    assets_dir = tmp_path / "snake_20260101_120000_assets"
    _write_pngs(assets_dir, ["player.png", "food.png"])
    seed.write_text(
        '<html><body><script>'
        'const ASSETS={'
        '"player":"./snake_20260101_120000_assets/player.png",'
        '"food":"./snake_20260101_120000_assets/food.png"'
        '};</script></body></html>'
    )
    a, s, adir, sdir = _scan_seed_media(seed.read_text(), seed)
    assert set(a.keys()) == {"player", "food"}
    assert adir == assets_dir.resolve()
    assert s == {}
    assert sdir is None


def test_scan_ignores_missing_files(tmp_path: Path) -> None:
    seed = tmp_path / "g.html"
    # Reference an asset that doesn't exist on disk.
    seed.write_text('<img src="./ghost_assets/missing.png">')
    a, s, adir, sdir = _scan_seed_media(seed.read_text(), seed)
    assert a == {}
    assert adir is None


def test_scan_picks_up_sounds(tmp_path: Path) -> None:
    seed = tmp_path / "bg.html"
    sounds_dir = tmp_path / "bg_arcade_sounds"
    sounds_dir.mkdir(parents=True)
    (sounds_dir / "laser.ogg").write_bytes(b"OggS fake")
    (sounds_dir / "music.ogg").write_bytes(b"OggS fake")
    seed.write_text(
        "<script>"
        'const SOUNDS={'
        '"laser": new Audio("./bg_arcade_sounds/laser.ogg"),'
        '"music": new Audio("./bg_arcade_sounds/music.ogg")'
        '};'
        "</script>"
    )
    a, s, adir, sdir = _scan_seed_media(seed.read_text(), seed)
    assert set(s.keys()) == {"laser", "music"}
    assert sdir == sounds_dir.resolve()


def test_scan_does_not_override_dir_when_prefixes_split(tmp_path: Path) -> None:
    # A seed manually merged from two sessions — two different
    # `_assets` prefixes appear. We refuse to pick one arbitrarily.
    seed = tmp_path / "merged.html"
    d1 = tmp_path / "first_session_assets"
    d2 = tmp_path / "second_session_assets"
    _write_pngs(d1, ["alpha.png"])
    _write_pngs(d2, ["beta.png"])
    seed.write_text(
        '<img src="./first_session_assets/alpha.png">'
        '<img src="./second_session_assets/beta.png">'
    )
    a, s, adir, sdir = _scan_seed_media(seed.read_text(), seed)
    assert set(a.keys()) == {"alpha", "beta"}
    # Two prefixes → no override.
    assert adir is None


def test_scan_picks_up_unreferenced_disk_assets(tmp_path: Path) -> None:
    """Motivating case (donkey-kong session): the assets folder has
    20+ PNGs from prior sessions (mario_idle, mario_walk1, dk_throw,
    jumpman_jump, …) but the seed HTML only loads 5 (hero, barrel,
    girder, ladder, princess). Without folder discovery, the model
    re-invents names; with it, the model sees the full roster and
    reuses what's there."""
    seed_basename = "donkey_kong_20260512_201139"
    seed = tmp_path / f"{seed_basename}.html"
    assets_dir = tmp_path / f"{seed_basename}_assets"
    _write_pngs(assets_dir, [
        # Referenced in the seed HTML (currently wired up):
        "hero.png", "barrel.png", "girder.png", "ladder.png", "princess.png",
        # NOT referenced — sitting on disk from prior sessions:
        "mario_idle.png", "mario_walk1.png", "mario_walk2.png",
        "mario_climb1.png", "mario_climb2.png", "mario_jump.png",
        "donkey_kong_idle.png", "donkey_kong_throw1.png",
        "donkey_kong_throw2.png",
        "jumpman_idle.png", "jumpman_run1.png", "jumpman_run2.png",
    ])
    seed.write_text(
        '<script>const ASSETS={};const E=[["hero","hero.png"],'
        '["barrel","barrel.png"],["girder","girder.png"],'
        '["ladder","ladder.png"],["princess","princess.png"]];'
        f'const AP="./{seed_basename}_assets/";</script>'
    )
    # Use a HTML scan that finds the 5 refs (they include the prefix
    # in JS strings); plus folder scan adds the other 12.
    seed_html = (
        f'<script>const AP="./{seed_basename}_assets/";\n'
        f'const HERO="./{seed_basename}_assets/hero.png";'
        f'const BARREL="./{seed_basename}_assets/barrel.png";'
        f'const GIRDER="./{seed_basename}_assets/girder.png";'
        f'const LADDER="./{seed_basename}_assets/ladder.png";'
        f'const PRINCESS="./{seed_basename}_assets/princess.png";'
        '</script>'
    )
    a, _, _, _ = _scan_seed_media(seed_html, seed)
    # All 17 PNGs (5 referenced + 12 unreferenced) must be in the roster.
    expected = {
        "hero", "barrel", "girder", "ladder", "princess",
        "mario_idle", "mario_walk1", "mario_walk2",
        "mario_climb1", "mario_climb2", "mario_jump",
        "donkey_kong_idle", "donkey_kong_throw1", "donkey_kong_throw2",
        "jumpman_idle", "jumpman_run1", "jumpman_run2",
    }
    assert set(a.keys()) == expected, (
        f"missing assets: {expected - set(a.keys())}; "
        f"extra: {set(a.keys()) - expected}"
    )


def test_scan_ignores_non_image_files_in_folder(tmp_path: Path) -> None:
    """Folder may contain .DS_Store / .json / stray files; only
    real images get rolled into the asset roster."""
    seed_basename = "g_20260101"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text("<html></html>")
    adir = tmp_path / f"{seed_basename}_assets"
    adir.mkdir()
    (adir / "real.png").write_bytes(b"\x89PNG")
    (adir / ".DS_Store").write_bytes(b"junk")
    (adir / "manifest.json").write_text("{}")
    (adir / "subdir").mkdir()  # nested dirs are skipped, not crawled

    a, _, _, _ = _scan_seed_media("<html></html>", seed)
    assert set(a.keys()) == {"real"}


def test_scan_picks_up_unreferenced_sounds_in_folder(tmp_path: Path) -> None:
    """Mirror of the asset case for the sounds folder."""
    seed_basename = "g_20260101"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text("<html></html>")
    sdir = tmp_path / f"{seed_basename}_sounds"
    sdir.mkdir()
    for f in ("jump.ogg", "hit.ogg", "music.ogg", "win.ogg"):
        (sdir / f).write_bytes(b"OggS")
    _, s, _, _ = _scan_seed_media("<html></html>", seed)
    assert set(s.keys()) == {"jump", "hit", "music", "win"}


def test_scan_tolerates_leading_dot_slash_and_quoting(tmp_path: Path) -> None:
    seed = tmp_path / "q.html"
    d = tmp_path / "q_assets"
    _write_pngs(d, ["a.png", "b.png"])
    # Mix single quotes, double quotes, with/without "./".
    seed.write_text(
        "<script>"
        "const A = './q_assets/a.png';"
        'const B = "q_assets/b.png";'
        "</script>"
    )
    a, _, adir, _ = _scan_seed_media(seed.read_text(), seed)
    assert set(a.keys()) == {"a", "b"}
    assert adir == d.resolve()


# ---------------------------------------------------------------------------
# Integration: with the seed reused as out_path (the chat.py behavior),
# _session_id naturally matches the seed's basename and mid-session
# <assets> blocks generate into the seed's existing folder.
#
# No override mechanism is needed — _session_id does the work.
# ---------------------------------------------------------------------------

class _StubImageGenerator:
    def __init__(self) -> None:
        self.last_stats: list[dict] = []

    def generate(self, prompt: str) -> str:
        import tempfile
        from PIL import Image
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        Image.new("RGB", (768, 768), (123, 123, 123)).save(f.name)
        return f.name


def _make_agent(tmp_path: Path, out_name: str = "fresh.html") -> GameAgent:
    out = tmp_path / out_name
    out.write_text("<html></html>")
    return GameAgent(
        model="stub",
        out_path=out,
        browser=MagicMock(),
        max_iters=1,
        memory_root=str(tmp_path / "memory"),
    )


async def _drain(agen) -> list:
    return [ev async for ev in agen]


def test_seeded_session_writes_into_original_folder(tmp_path: Path) -> None:
    """When chat.py points out_path at the seed (the simple flow we
    want), _session_id matches the seed's basename and new <assets>
    land in the seed's existing `<basename>_assets/` folder — no new
    sibling folder is created."""
    # Simulate the chat.py decision: out_path IS the seed.
    seed_basename = "snake_20260101_120000"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text(
        '<html><body><script>const ASSETS={'
        f'"player":"./{seed_basename}_assets/player.png"'
        "};</script></body></html>"
    )
    existing_dir = tmp_path / f"{seed_basename}_assets"
    existing_dir.mkdir()
    (existing_dir / "player.png").write_bytes(b"\x89PNG seed")

    a = _make_agent(tmp_path, out_name=f"{seed_basename}.html")
    a._asset_generator = _StubImageGenerator()
    a._session_assets["player"] = (existing_dir / "player.png").resolve()

    reply = (
        '<assets>[{"name":"boss","prompt":"giant red boss sprite"}]</assets>'
    )
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="mid_session",
    )))

    assert "boss" in a._session_assets
    boss_path = Path(a._session_assets["boss"]).resolve()
    # Same folder as the seed's existing player.png — no new folder.
    assert boss_path.parent == existing_dir.resolve(), (
        f"boss landed in {boss_path.parent}, not the seed's existing "
        f"{existing_dir}"
    )
    # No sibling folder was created.
    siblings = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert siblings.count(f"{seed_basename}_assets") == 1
    # The seed's original asset is intact (merge, not overwrite).
    assert "player" in a._session_assets


def test_fresh_session_creates_basename_folder(tmp_path: Path) -> None:
    """Control: a fresh session (no seed) still gets a new
    `<basename>_assets/` folder — only the seeded path reuses."""
    a = _make_agent(tmp_path, out_name="brand_new_session.html")
    a._asset_generator = _StubImageGenerator()

    reply = '<assets>[{"name":"alien","prompt":"green alien"}]</assets>'
    asyncio.run(_drain(a._maybe_generate_assets_and_sounds(
        reply, trigger="phase_a",
    )))

    expected = (tmp_path / "brand_new_session_assets").resolve()
    alien_path = Path(a._session_assets["alien"]).resolve()
    assert alien_path.parent == expected


def test_seed_scan_anchors_at_out_path_not_seed_file(tmp_path: Path) -> None:
    """When the seed is a snapshot deep under games/snapshots/<basename>/,
    chat.py resolves out_path back to games/<basename>.html — and the
    scanner must anchor on out_path.parent (the games dir, where
    <basename>_assets/ lives) instead of seed_file.parent (the
    snapshot dir, which doesn't contain the assets folder).
    """
    # Layout mirroring a real session:
    #   tmp/games/snake_x.html               <- canonical (becomes out_path)
    #   tmp/games/snake_x_assets/player.png  <- the real assets
    #   tmp/games/snapshots/snake_x/iter_03.html  <- the snapshot we seed
    games = tmp_path / "games"
    snapshots = games / "snapshots" / "snake_x"
    assets_dir = games / "snake_x_assets"
    games.mkdir()
    snapshots.mkdir(parents=True)
    assets_dir.mkdir()
    (assets_dir / "player.png").write_bytes(b"\x89PNG seed")

    snapshot_html = (
        '<html><body><script>'
        'const ASSETS={"player":"./snake_x_assets/player.png"};'
        '</script></body></html>'
    )
    snapshot_path = snapshots / "iter_03.html"
    snapshot_path.write_text(snapshot_html)

    # Simulate what chat.py does post-resolve: out_path is the
    # CANONICAL games/snake_x.html (even though we never touched it
    # yet — it's where the seed branch will write the HTML).
    from agent import _scan_seed_media
    out_path = games / "snake_x.html"

    # Anchor on out_path (canonical). Picks up the assets.
    a, s, adir, sdir = _scan_seed_media(snapshot_html, out_path)
    assert set(a.keys()) == {"player"}
    assert adir == assets_dir.resolve()

    # For contrast: anchoring on the snapshot path would miss it.
    a2, _, _, _ = _scan_seed_media(snapshot_html, snapshot_path)
    assert a2 == {}, (
        "scanner anchored at snapshot dir should not find assets that "
        "live one level up — this guards against regressing the anchor"
    )
