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
