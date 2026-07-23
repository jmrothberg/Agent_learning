"""Phase 1 — non-blocking critic + diffuser pre-warm + live throughput.

All three subphases must:
  - Keep GPU assignment identical to today (no slot moves).
  - Work gracefully on single-GPU systems (fall back to blocking /
    skip pre-warm to avoid VRAM contention).
  - Surface progress in the TUI status panel.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402
import gpu_status as _gs  # noqa: E402


def _stub_agent(tmp_path: Path) -> GameAgent:
    return GameAgent(
        model="stub:1b",
        out_path=tmp_path / "game.html",
        browser=MagicMock(),
        max_iters=3,
        memory_root=str(tmp_path / "memory"),
    )


# ---------------------------------------------------------------------------
# Phase 1A — critic runs on independent slot detection
# ---------------------------------------------------------------------------


def test_critic_independent_slot_returns_false_when_no_critic(tmp_path):
    a = _stub_agent(tmp_path)
    assert a._critic_runs_on_independent_slot(None) is False


def test_critic_independent_slot_returns_false_when_critic_is_coder(tmp_path):
    # Single-slot fallback — critic backend IS the coder backend.
    a = _stub_agent(tmp_path)
    assert a._critic_runs_on_independent_slot(a._backend) is False


def test_critic_independent_slot_returns_true_for_distinct_backend(tmp_path):
    a = _stub_agent(tmp_path)
    distinct = MagicMock()
    # Make sure the endpoint check doesn't false-pair them.
    distinct.info.endpoint = "http://127.0.0.1:11435"
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    assert a._critic_runs_on_independent_slot(distinct) is True


def test_critic_independent_slot_returns_false_for_same_endpoint(tmp_path):
    # Two backend objects but pointing at the same Ollama endpoint —
    # concurrent runs would queue at the same daemon. Must fall back to
    # blocking behavior.
    a = _stub_agent(tmp_path)
    same_endpoint = MagicMock()
    same_endpoint.info.endpoint = "http://127.0.0.1:11434"
    a._backend.info.endpoint = "http://127.0.0.1:11434"
    assert a._critic_runs_on_independent_slot(same_endpoint) is False


def test_critic_task_initialized_none(tmp_path):
    a = _stub_agent(tmp_path)
    assert a._critic_task is None


def test_drain_pending_critic_task_no_op_when_no_task(tmp_path):
    a = _stub_agent(tmp_path)

    async def _drive():
        return await a._drain_pending_critic_task(wait=True)

    assert asyncio.run(_drive()) is False


# ---------------------------------------------------------------------------
# Phase 1B — pre-warm gating
# ---------------------------------------------------------------------------


def test_diffuser_has_dedicated_gpu_returns_false_on_single_gpu():
    """Synthetic snapshot with one GPU must return False — pre-warm
    would compete with the LLM for VRAM."""
    snap = _gs.GpuSnapshot(
        gpus=[
            _gs.GpuInfo(
                0,
                "NVIDIA RTX 4080",
                memory_used_mib=8000,
                memory_total_mib=16384,
            ),
        ],
        processes=[],
    )
    assert _gs.diffuser_has_dedicated_gpu(snap) is False


def test_diffuser_has_dedicated_gpu_returns_true_on_workstation():
    """The 4×48 GB workstation pattern — diffuser pins to GPU 0,
    LLMs to GPUs 1-3. Pre-warm fires."""
    name = "NVIDIA RTX 6000 Ada Generation"
    snap = _gs.GpuSnapshot(
        gpus=[
            _gs.GpuInfo(
                i, name, memory_used_mib=used, memory_total_mib=49140,
            )
            for i, used in enumerate([1000, 2000, 2000, 2000])
        ],
        processes=[
            # Ollama on slots 1-3 only.
            _gs.GpuProcess(1, 100, "ollama", 18000),
            _gs.GpuProcess(2, 200, "ollama", 18000),
            _gs.GpuProcess(3, 300, "ollama", 18000),
        ],
    )
    assert _gs.diffuser_has_dedicated_gpu(snap) is True


def test_diffuser_has_dedicated_gpu_returns_false_on_no_snapshot():
    # Conservative fallback — when GPU info is unavailable, treat as
    # single-GPU shared (skip pre-warm).
    assert _gs.diffuser_has_dedicated_gpu(None) is False


def test_prewarm_helper_skips_when_single_gpu(tmp_path, monkeypatch):
    a = _stub_agent(tmp_path)
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)

    # Force the gating helper to say "single GPU".
    monkeypatch.setattr(_gs, "diffuser_has_dedicated_gpu", lambda snap=None: False)

    a._maybe_prewarm_diffusers_during_phase_a()

    assert any(
        e.get("kind") == "diffuser_prewarm_skipped"
        and e.get("reason") == "no_dedicated_gpu"
        for e in a._trace_events
    )


def test_prewarm_helper_spawns_tasks_when_dedicated_gpu(tmp_path, monkeypatch):
    """Verify the helper schedules the prewarm tasks when GPU gate passes.
    Does NOT actually load diffusers (requires CUDA + ~14GB VRAM); we
    monkeypatch the generator classes to no-op."""
    a = _stub_agent(tmp_path)
    a._trace_events = []
    a._trace = lambda obj: a._trace_events.append(obj)

    monkeypatch.setattr(_gs, "diffuser_has_dedicated_gpu", lambda snap=None: True)

    class _NoOpGen:
        def _lazy_init(self):
            return True

    import assets
    import sounds
    monkeypatch.setattr(assets, "try_load_image_generator", lambda: _NoOpGen())
    monkeypatch.setattr(sounds, "StableAudioGenerator", _NoOpGen)

    async def _drive():
        a._maybe_prewarm_diffusers_during_phase_a()
        # Let the spawned tasks run.
        await asyncio.sleep(0.1)

    asyncio.run(_drive())

    # Two prefill_warm events with target=z_image / stable_audio.
    targets = {
        e.get("target")
        for e in a._trace_events
        if e.get("kind") == "prefill_warm"
    }
    assert "z_image" in targets
    assert "stable_audio" in targets


# ---------------------------------------------------------------------------
# Phase 1C — live throughput renderer
# ---------------------------------------------------------------------------


def test_assets_live_progress_shows_nothing_when_no_gen_in_flight():
    from chat import CodingBoxApp
    app = CodingBoxApp()
    app.agent = None
    assert app._format_assets_live_progress() == ""


def test_assets_live_progress_shows_rate_and_eta_mid_flight():
    from chat import CodingBoxApp
    app = CodingBoxApp()
    app.agent = MagicMock()
    # 12 requested, 4 produced — mid-flight.
    app._assets_in_flight_total = 12
    gen = MagicMock()
    gen.last_stats = [
        {"gen_seconds": 2.8},
        {"gen_seconds": 3.0},
        {"gen_seconds": 2.9},
        {"gen_seconds": 3.1},
    ]
    app.agent._asset_generator = gen

    line = app._format_assets_live_progress()
    assert "Sprites:" in line
    assert "4/12" in line
    # avg ≈ 2.95s, rate ≈ 0.34/s, ETA ≈ 23.6s
    assert "2.9s" in line or "3.0s" in line  # tolerate rounding
    assert "ETA" in line


def test_assets_live_progress_hides_after_completion():
    from chat import CodingBoxApp
    app = CodingBoxApp()
    app.agent = MagicMock()
    # 12 requested, 12 produced — done.
    app._assets_in_flight_total = 12
    gen = MagicMock()
    gen.last_stats = [{"gen_seconds": 2.8}] * 12
    app.agent._asset_generator = gen

    # While in_flight_total > produced, the live row renders. Once they
    # equalise (or in_flight_total is cleared by the `assets` event),
    # this method must return "" so the sticky summary takes over.
    assert app._format_assets_live_progress() == ""
