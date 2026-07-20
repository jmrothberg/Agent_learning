"""Regression tests for P1 seed asset rehydration (MK trace 20260528).

When a session is started from a seed HTML that already references
`./<prefix>_assets/<name>.png` and/or `./<prefix>_sounds/<name>.ogg`,
the agent MUST:

  1. Rehydrate `_session_assets` / `_session_sounds` BEFORE Phase A
     asset/sound generation runs (so the next check has data).
  2. SKIP `_maybe_generate_assets_and_sounds` when `trigger="phase_a"`
     and seed media is already on disk — no diffuser call, no GPU
     warm-up, no asset regeneration that would wipe the user's art.
  3. Tell the planner via `from_seed=True` to omit the
     "MUST emit <assets>/<sounds>" directives.

Pure-function tests cover each layer directly. No GPU, no model calls.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
from prompts_v1 import plan_instruction  # noqa: E402


def _write_pngs(dirpath: Path, names: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dirpath / n).write_bytes(b"\x89PNG fake")


def _write_oggs(dirpath: Path, names: list[str]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dirpath / n).write_bytes(b"OggS fake")


def _make_seed_agent(tmp_path: Path, seed_basename: str) -> GameAgent:
    """Build an agent that has a `seed_file` pointing at `<basename>.html`
    in tmp_path, mirroring the chat.py path where out_path IS the seed."""
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text(
        '<html><body><script>const ASSETS={'
        f'"player":"./{seed_basename}_assets/player.png"'
        "};</script></body></html>",
        encoding="utf-8",
    )
    a = GameAgent(
        model="stub",
        out_path=seed,
        browser=MagicMock(),
        max_iters=1,
        memory_root=str(tmp_path / "memory"),
        seed_file=seed,
    )
    return a


# ---------------------------------------------------------------------------
# Layer 1 — _early_rehydrate_seed_media populates session before phase_a
# ---------------------------------------------------------------------------

def test_early_rehydrate_populates_session_from_disk(tmp_path: Path) -> None:
    seed_basename = "mortal_kombat_20260524_101226"
    a = _make_seed_agent(tmp_path, seed_basename)
    _write_pngs(
        tmp_path / f"{seed_basename}_assets",
        ["player.png", "boss.png", "fireball.png"],
    )
    _write_oggs(
        tmp_path / f"{seed_basename}_sounds",
        ["hit.ogg", "shoot.ogg"],
    )

    n_a, n_s = a._early_rehydrate_seed_media()

    assert n_a == 3
    assert n_s == 2
    assert set(a._session_assets.keys()) == {"player", "boss", "fireball"}
    assert set(a._session_sounds.keys()) == {"hit", "shoot"}


def test_early_rehydrate_is_noop_without_seed(tmp_path: Path) -> None:
    """A fresh (non-seeded) session has no seed_file — helper short-circuits."""
    out = tmp_path / "fresh.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub", out_path=out, browser=MagicMock(),
        max_iters=1, memory_root=str(tmp_path / "memory"),
    )
    n_a, n_s = a._early_rehydrate_seed_media()
    assert n_a == 0
    assert n_s == 0
    assert a._session_assets == {}
    assert a._session_sounds == {}


def test_early_rehydrate_is_idempotent(tmp_path: Path) -> None:
    """The later seed-branch rehydrate must be a safe no-op merge so we
    don't double-yield the 'rehydrated from seed' info event."""
    seed_basename = "mortal_kombat_20260524_101226"
    a = _make_seed_agent(tmp_path, seed_basename)
    _write_pngs(
        tmp_path / f"{seed_basename}_assets",
        ["player.png"],
    )

    a._early_rehydrate_seed_media()
    first = dict(a._session_assets)
    a._early_rehydrate_seed_media()
    second = dict(a._session_assets)

    assert first == second
    assert len(a._session_assets) == 1


# ---------------------------------------------------------------------------
# Layer 2 — _maybe_generate_assets_and_sounds skips phase_a when seed media
# ---------------------------------------------------------------------------

async def _drain(agen) -> list:
    return [ev async for ev in agen]


def test_phase_a_skipped_when_seed_media_present(tmp_path: Path) -> None:
    """The MK-trace symptom: model emits <assets> in plan, harness used to
    regenerate them all. With the skip guard, the generator is NEVER
    called and a `seed_phase_a_media_skipped` trace fires instead."""
    seed_basename = "mortal_kombat_20260524_101226"
    a = _make_seed_agent(tmp_path, seed_basename)
    _write_pngs(
        tmp_path / f"{seed_basename}_assets",
        ["player.png", "boss.png"],
    )
    a._early_rehydrate_seed_media()

    # Wire a sentinel so we can assert the generator was NOT touched.
    sentinel = MagicMock()
    sentinel.generate.side_effect = AssertionError(
        "image generator must not be called on seeded phase_a"
    )
    a._asset_generator = sentinel

    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = (
        '<assets>['
        '{"name":"player","prompt":"new player"},'
        '{"name":"boss","prompt":"new boss"}'
        ']</assets>'
    )
    events = asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"),
    ))

    assert any(
        t.get("kind") == "seed_phase_a_media_skipped" for t in traces
    ), "must trace seed_phase_a_media_skipped"
    # Visible info event so the user can see why generation didn't run.
    assert any(
        "phase_a asset/sound generation skipped" in str(ev)
        for ev in events
    )
    # Generator was not called.
    assert sentinel.generate.call_count == 0


def test_mid_session_generation_unaffected_by_skip_guard(tmp_path: Path) -> None:
    """The skip guard must apply ONLY to trigger='phase_a'. Explicit
    user requests like 'add a new boss sprite' (which arrive as
    trigger='mid_session') still flow through to the generator."""
    seed_basename = "mortal_kombat_20260524_101226"
    a = _make_seed_agent(tmp_path, seed_basename)
    _write_pngs(
        tmp_path / f"{seed_basename}_assets",
        ["player.png"],
    )
    a._early_rehydrate_seed_media()

    # Use the same stub generator pattern as test_seed_media_rehydrate.
    import tempfile
    from PIL import Image

    class _StubImageGenerator:
        last_stats: list = []

        def generate(self, prompt: str) -> str:
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            Image.new("RGB", (768, 768), (12, 34, 56)).save(f.name)
            return f.name

    a._asset_generator = _StubImageGenerator()

    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = '<assets>[{"name":"new_boss","prompt":"giant red boss"}]</assets>'
    asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="mid_session"),
    ))

    # Skip MUST NOT fire on mid_session.
    assert not any(
        t.get("kind") == "seed_phase_a_media_skipped" for t in traces
    )
    # The new asset DID generate.
    assert "new_boss" in a._session_assets


def test_phase_a_runs_normally_for_fresh_session(tmp_path: Path) -> None:
    """Control: no seed_file → skip guard must NEVER fire."""
    import tempfile
    from PIL import Image

    out = tmp_path / "brand_new.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub", out_path=out, browser=MagicMock(),
        max_iters=1, memory_root=str(tmp_path / "memory"),
    )

    class _StubImageGenerator:
        last_stats: list = []

        def generate(self, prompt: str) -> str:
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            Image.new("RGB", (768, 768), (12, 34, 56)).save(f.name)
            return f.name

    a._asset_generator = _StubImageGenerator()
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = '<assets>[{"name":"alien","prompt":"green alien"}]</assets>'
    asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"),
    ))

    assert not any(
        t.get("kind") == "seed_phase_a_media_skipped" for t in traces
    )
    assert "alien" in a._session_assets


def test_phase_a_runs_when_seed_file_set_but_no_media_on_disk(
    tmp_path: Path,
) -> None:
    """Edge case: someone passes a seed_file but the corresponding
    assets/sounds dirs are empty. We should NOT skip — there's no
    on-disk media to preserve, so phase_a generation must run normally."""
    import tempfile
    from PIL import Image

    seed_basename = "empty_seed"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text("<html><body>no media refs</body></html>", encoding="utf-8")
    a = GameAgent(
        model="stub", out_path=seed, browser=MagicMock(),
        max_iters=1, memory_root=str(tmp_path / "memory"),
        seed_file=seed,
    )
    a._early_rehydrate_seed_media()
    assert a._session_assets == {}
    assert a._session_sounds == {}

    class _StubImageGenerator:
        last_stats: list = []

        def generate(self, prompt: str) -> str:
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            Image.new("RGB", (768, 768), (12, 34, 56)).save(f.name)
            return f.name

    a._asset_generator = _StubImageGenerator()
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = '<assets>[{"name":"hero","prompt":"hero sprite"}]</assets>'
    asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"),
    ))

    # No skip — empty seed has nothing to preserve.
    assert not any(
        t.get("kind") == "seed_phase_a_media_skipped" for t in traces
    )
    assert "hero" in a._session_assets


def test_malformed_phase_a_assets_on_seed_does_not_queue_reemit_feedback(
    tmp_path: Path,
) -> None:
    """TD seed trace 20260630_114658: malformed plan <assets> must not queue
    'NO art was generated / re-emit <assets>' — mid_session build still
    generates sprites on the same iter."""
    seed_basename = "tower_defense_seed"
    a = _make_seed_agent(tmp_path, seed_basename)
    _write_pngs(
        tmp_path / f"{seed_basename}_assets",
        ["tower_gun_idle.png"],
    )
    a._early_rehydrate_seed_media()

    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = '<assets>[{"name":"bad"}]</assets>'
    asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"),
    ))

    assert any(
        t.get("kind") == "assets_parse_failed_seed_ignored" for t in traces
    )
    assert not any(t.get("kind") == "assets_parse_failed" for t in traces)
    assert not a._pending_feedback
    assert not any(
        "ASSET FORMAT ERROR" in fb for fb in a._pending_feedback
    )


def test_malformed_phase_a_assets_fresh_session_still_queues_feedback(
    tmp_path: Path,
) -> None:
    """Control: non-seed phase_a parse failure still coaches the model."""
    out = tmp_path / "brand_new.html"
    out.write_text("<html></html>")
    a = GameAgent(
        model="stub", out_path=out, browser=MagicMock(),
        max_iters=1, memory_root=str(tmp_path / "memory"),
    )
    sentinel = MagicMock()
    sentinel.generate.side_effect = AssertionError(
        "image generator must not be called for parse-failure coaching test"
    )
    a._asset_generator = sentinel
    traces: list[dict] = []
    a._trace = lambda obj: traces.append(obj)

    reply = '<assets>[{"name":"bad"}]</assets>'
    asyncio.run(_drain(
        a._maybe_generate_assets_and_sounds(reply, trigger="phase_a"),
    ))

    assert any(t.get("kind") == "assets_parse_failed" for t in traces)
    assert not any(
        t.get("kind") == "assets_parse_failed_seed_ignored" for t in traces
    )
    assert any("ASSET FORMAT ERROR" in fb for fb in a._pending_feedback)


# ---------------------------------------------------------------------------
# Layer 3 — plan_instruction(from_seed=True) suppresses MUST-emit nudges
# ---------------------------------------------------------------------------

def test_plan_instruction_from_seed_suppresses_art_directive() -> None:
    """A goal with art keywords normally triggers 'ART INTENT DETECTED'.
    On a seed continuation it must NOT — the model should reuse existing
    art, not request fresh generation."""
    fresh = plan_instruction(
        reference_block="", goal="make a sprite-heavy fighting game"
    )
    seeded = plan_instruction(
        reference_block="",
        goal="make a sprite-heavy fighting game",
        from_seed=True,
        seed_asset_names=["player_idle", "boss_idle"],
        seed_sound_names=["punch"],
    )
    assert "ART INTENT DETECTED" in fresh
    assert "ART INTENT DETECTED" not in seeded


def test_plan_instruction_from_seed_suppresses_audio_directive() -> None:
    fresh = plan_instruction(
        reference_block="", goal="game with sfx and music"
    )
    seeded = plan_instruction(
        reference_block="",
        goal="game with sfx and music",
        from_seed=True,
        seed_asset_names=[],
        seed_sound_names=["punch", "music"],
    )
    assert "AUDIO INTENT DETECTED" in fresh
    assert "AUDIO INTENT DETECTED" not in seeded


def test_plan_instruction_from_seed_contains_continuation_block() -> None:
    """The seed-continuation directive must appear with the asset/sound
    names rendered as a 'reuse these by name' contract."""
    out = plan_instruction(
        reference_block="",
        goal="fix the punch animation",
        from_seed=True,
        seed_asset_names=["player_idle", "cpu_idle", "fireball"],
        seed_sound_names=["punch", "kick"],
    )
    assert "SEED CONTINUATION" in out
    assert "DO NOT emit <assets>" in out
    assert "DO NOT emit <sounds>" in out
    # Names rendered alphabetically (helps the model spot duplicates).
    assert "cpu_idle, fireball, player_idle" in out
    assert "Existing sounds on disk: kick, punch" in out


def test_plan_instruction_non_seed_unchanged() -> None:
    """Backwards compatibility: a regular (non-seed) call must not contain
    SEED CONTINUATION text or behave differently."""
    out = plan_instruction(reference_block="", goal="make a snake game")
    assert "SEED CONTINUATION" not in out
    # Backward-compat: the existing PLAN_INSTRUCTION body is still there.
    assert "PROBES are real code" in out or "<probes>" in out


# ---------------------------------------------------------------------------
# Assets-only seed regen — declared PATHS, no genre invent
# ---------------------------------------------------------------------------

def test_plan_instruction_seed_regen_requires_declared_roster() -> None:
    out = plan_instruction(
        reference_block="",
        goal="create all new assets, keep code identical",
        from_seed=True,
        seed_asset_names=["hero_idle_down", "bomb", "soft_block"],
        seed_sound_names=[],
        seed_regen_assets=True,
    )
    assert "SEED ASSETS-ONLY REGEN" in out
    assert "DO NOT emit <assets>" not in out
    assert "EXACTLY the asset names" in out
    assert "hero_idle_down" in out
    assert "bomb" in out
    assert "soft_block" in out


def test_detect_assets_only_intent() -> None:
    from prompts_v1 import (
        _detect_art_intent,
        _detect_assets_only_intent,
        _detect_seed_media_replace_intent,
    )
    assert _detect_assets_only_intent(
        "creat all new assets, keep code identical, no changes, "
        "just create the correct assets in the correct folders"
    )
    assert _detect_assets_only_intent(
        "regenerate all sprites, keep code identical"
    )
    assert not _detect_assets_only_intent("build a bomberman game")
    assert not _detect_assets_only_intent("add a jump mechanic")
    # Frustrating re-run wording — replace intent must fire without keep-code.
    g = (
        "Make new assets for bomberman, do NOT change the code, "
        "just generate the assets"
    )
    assert _detect_art_intent(g)
    assert _detect_seed_media_replace_intent(g)
    assert _detect_seed_media_replace_intent("sprites please, generate them")
    assert not _detect_seed_media_replace_intent("fix the jump collision")


def test_declared_stems_complete_ignores_orphans() -> None:
    from agent_helpers import (
        _declared_stems_complete,
        _missing_declared_stems,
    )
    declared = ["hero_idle_down", "bomb", "soft_block"]
    orphans = {"ship": Path("/x/ship.png"), "asteroid": Path("/x/a.png")}
    assert not _declared_stems_complete(declared, orphans)
    assert _missing_declared_stems(declared, orphans) == declared
    present = {
        "hero_idle_down": Path("/x/h.png"),
        "bomb": Path("/x/b.png"),
        "soft_block": Path("/x/s.png"),
        "ship": Path("/x/ship.png"),
    }
    assert _declared_stems_complete(declared, present)


def test_coerce_specs_to_declared_seed_roster_drops_invented(tmp_path: Path) -> None:
    seed = tmp_path / "g.html"
    seed.write_text("<html></html>")
    a = GameAgent(
        model="stub",
        out_path=seed,
        browser=MagicMock(),
        max_iters=1,
        memory_root=str(tmp_path / "memory"),
        seed_file=seed,
    )
    a._seed_media_regen = True
    a._seed_media_regen_all = True
    a._assets_only_goal = True
    a._seed_declared_asset_names = ["hero_idle_down", "bomb"]
    a._seed_declared_sound_names = []
    specs, sounds = a._coerce_specs_to_declared_seed_roster(
        [
            {"name": "ship", "prompt": "spaceship"},
            {"name": "asteroid", "prompt": "rock"},
            {"name": "bomb", "prompt": "round bomb"},
        ],
        [],
    )
    assert [s["name"] for s in specs] == ["hero_idle_down", "bomb"]
    assert specs[1]["prompt"] == "round bomb"
    assert "hero" in specs[0]["prompt"].lower() or "idle" in specs[0]["prompt"].lower()
    # Trace 20260720_135103: synthesized specs must carry size or
    # generate_assets KeyError zeros the whole batch.
    for s in specs:
        assert s.get("size"), f"missing size on coerced spec {s.get('name')}"
        assert isinstance(s["size"], tuple) and len(s["size"]) == 2
    assert sounds == []


def test_phase_a_orphans_do_not_skip_when_declared_incomplete(
    tmp_path: Path, monkeypatch,
) -> None:
    """Leftover ship/asteroid PNGs must not count as declared coverage."""
    seed_basename = "grid_bomber_orphans"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text(
        "<html><body><script>"
        f"const PATHS=[['hero_idle_down','./{seed_basename}_assets/hero_idle_down.png'],"
        f"['bomb','./{seed_basename}_assets/bomb.png']];"
        "</script></body></html>",
        encoding="utf-8",
    )
    adir = tmp_path / f"{seed_basename}_assets"
    adir.mkdir()
    (adir / "ship.png").write_bytes(b"\x89PNG fake")
    (adir / "asteroid.png").write_bytes(b"\x89PNG fake")
    a = GameAgent(
        model="stub",
        out_path=seed,
        browser=MagicMock(),
        max_iters=1,
        memory_root=str(tmp_path / "memory"),
        seed_file=seed,
    )
    a._early_rehydrate_seed_media()
    assert set(a._session_assets) == {"ship", "asteroid"}
    assert set(a._seed_declared_asset_names) == {"hero_idle_down", "bomb"}
    a._seed_media_regen = True
    a._seed_media_regen_all = True
    a._assets_only_goal = True

    captured: list[list[str]] = []

    class _StubGen:
        def generate(self, prompt: str) -> str:
            return ""

    def _fake_generate(specs, *args, **kwargs):
        captured.append([str(s.get("name")) for s in specs])
        out = {}
        for s in specs:
            p = adir / f"{s['name']}.png"
            p.write_bytes(b"\x89PNG fake")
            out[s["name"]] = p
        return out

    monkeypatch.setattr(
        "agent_assets.try_load_image_generator", lambda: _StubGen(),
    )
    monkeypatch.setattr("agent_assets.generate_assets", _fake_generate)

    reply = (
        "<assets>["
        '{"name":"ship","prompt":"spaceship"},'
        '{"name":"asteroid","prompt":"rock"}'
        "]</assets>"
    )

    async def _drain():
        async for _ev in a._maybe_generate_assets_and_sounds(
            reply, trigger="phase_a",
        ):
            pass

    asyncio.run(_drain())
    assert captured, "must generate despite orphan PNGs on disk"
    assert set(captured[0]) == {"hero_idle_down", "bomb"}
    assert "ship" not in captured[0]


def test_phase_a_generates_declared_names_when_assets_only(
    tmp_path: Path, monkeypatch,
) -> None:
    """Empty disk + media replace intent: generator gets declared stems."""
    seed_basename = "grid_bomber_seed"
    seed = tmp_path / f"{seed_basename}.html"
    seed.write_text(
        "<html><body><script>"
        f"const PATHS=[['hero_idle_down','./{seed_basename}_assets/hero_idle_down.png'],"
        f"['bomb','./{seed_basename}_assets/bomb.png'],"
        f"['soft_block','./{seed_basename}_assets/soft_block.png']];"
        "</script></body></html>",
        encoding="utf-8",
    )
    (tmp_path / f"{seed_basename}_assets").mkdir()
    a = GameAgent(
        model="stub",
        out_path=seed,
        browser=MagicMock(),
        max_iters=1,
        memory_root=str(tmp_path / "memory"),
        seed_file=seed,
    )
    a._early_rehydrate_seed_media()
    assert a._session_assets == {}
    assert set(a._seed_declared_asset_names) == {
        "hero_idle_down", "bomb", "soft_block",
    }
    a._seed_media_regen = True
    a._seed_media_regen_all = True
    a._assets_only_goal = True

    captured: list[list[str]] = []

    class _StubGen:
        def generate(self, prompt: str) -> str:
            return ""

    def _fake_generate(specs, *args, **kwargs):
        captured.append([str(s.get("name")) for s in specs])
        out = {}
        for s in specs:
            p = tmp_path / f"{seed_basename}_assets" / f"{s['name']}.png"
            p.write_bytes(b"\x89PNG fake")
            out[s["name"]] = p
        return out

    monkeypatch.setattr(
        "agent_assets.try_load_image_generator", lambda: _StubGen(),
    )
    monkeypatch.setattr("agent_assets.generate_assets", _fake_generate)

    reply = (
        "<assets>["
        '{"name":"ship","prompt":"spaceship"},'
        '{"name":"asteroid","prompt":"rock"},'
        '{"name":"bullet","prompt":"bullet"}'
        "]</assets>"
    )

    async def _drain():
        events = []
        async for ev in a._maybe_generate_assets_and_sounds(
            reply, trigger="phase_a",
        ):
            events.append(ev)
        return events

    asyncio.run(_drain())
    assert captured, "generate_assets should have been called"
    assert set(captured[0]) == {"hero_idle_down", "bomb", "soft_block"}
    assert "ship" not in captured[0]
