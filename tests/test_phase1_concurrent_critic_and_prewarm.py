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


# ---------------------------------------------------------------------------
# Sept 2026 — oMLX (continuous batching) counts as concurrency-capable, and
# the code-critic sidecar that rides on it.
# ---------------------------------------------------------------------------


class _Info:
    def __init__(self, name: str, endpoint: str, model: str = "m"):
        self.name = name
        self.endpoint = endpoint
        self.model = model


def _mlx_server_backend(endpoint: str = "http://127.0.0.1:8000"):
    bk = MagicMock()
    bk.info = _Info("mlx-server", endpoint)
    return bk


def test_loopback_omlx_endpoint_supports_concurrency(monkeypatch):
    """A loopback oMLX server batches streams; Ollama loopback still serializes."""
    import backend as backend_mod
    monkeypatch.setattr(backend_mod, "endpoint_is_omlx", lambda ep: ":8000" in (ep or ""))
    assert GameAgent._endpoint_supports_concurrency("http://127.0.0.1:8000") is True
    assert GameAgent._endpoint_supports_concurrency("http://127.0.0.1:11434") is False


def test_backend_supports_concurrency_by_backend_kind(tmp_path):
    a = _stub_agent(tmp_path)
    assert a._backend_supports_concurrency(_mlx_server_backend()) is True
    ollama = MagicMock()
    ollama.info = _Info("ollama", "http://127.0.0.1:11434")
    assert a._backend_supports_concurrency(ollama) is False
    assert a._backend_supports_concurrency(None) is False


def test_inprocess_mlx_is_never_concurrent(tmp_path, monkeypatch):
    """BATTLEZ3 20260905_144613: in-process MLX (`MLXBackend`, endpoint
    sentinel "in-process") was classed as concurrency-capable because the
    sentinel is not a loopback URL. The code critic then queued in the
    single Metal executor ahead of the feedback router (60 s parse
    timeout) and the iter-2 coder turn. auto must resolve OFF here."""
    import backend as backend_mod
    monkeypatch.delenv("AGENT_CODE_CRITIC", raising=False)
    assert GameAgent._endpoint_supports_concurrency("in-process") is False
    a = _stub_agent(tmp_path)
    info = backend_mod.BackendInfo(
        name="mlx", model="/Users/x/MLX_Models/Qwen3.8-27B-mxfp8",
        source="test", endpoint="in-process",
    )
    a._backend = backend_mod.MLXBackend(info)
    assert a._backend_supports_concurrency(a._backend) is False
    assert a._critic_runs_on_independent_slot(a._backend) is False
    assert a._code_critic_enabled() is False
    assert [lbl for _bk, lbl in a._available_sampler_slots()] == ["slot1"]


def test_forced_critic_on_serial_backend_runs_inline(tmp_path):
    """`/critic on` on a serial backend must finish the review inside
    _spawn_code_critic (no background task left competing for the GPU)."""
    a = _stub_agent(tmp_path)  # legacy loopback Ollama → serial
    a._code_critic_mode = "on"
    seen: list[int] = []

    async def _fake_run(html, iteration):
        seen.append(iteration)
        return "LGTM"

    a.run_code_critic = _fake_run  # type: ignore[method-assign]

    async def _drive():
        await a._spawn_code_critic("<html></html>", 3)
        return a._code_critic_task.done()

    assert asyncio.run(_drive()) is True
    assert seen == [3]


def test_critic_on_same_omlx_backend_runs_concurrently(tmp_path):
    """Same backend object as the coder used to force the inline path;
    on oMLX the second stream overlaps the coder instead."""
    a = _stub_agent(tmp_path)
    a._backend = _mlx_server_backend()
    assert a._critic_runs_on_independent_slot(a._backend) is True


def test_single_omlx_slot_offers_two_sampler_slots(tmp_path):
    """Best-of-2 stuck escalation fans out in parallel on one oMLX slot."""
    a = _stub_agent(tmp_path)
    a._backend = _mlx_server_backend()
    slots = a._available_sampler_slots()
    assert [lbl for _bk, lbl in slots] == ["slot1", "slot1b"]
    assert slots[0][0] is slots[1][0]
    # Serial backend: unchanged single slot.
    b = _stub_agent(tmp_path)
    assert [lbl for _bk, lbl in b._available_sampler_slots()] == ["slot1"]


def test_code_critic_mode_resolution(tmp_path, monkeypatch):
    """auto = ON only on a parallel backend; on/off/env/allroles override."""
    monkeypatch.delenv("AGENT_CODE_CRITIC", raising=False)
    a = _stub_agent(tmp_path)  # legacy Ollama loopback → serial
    assert a._code_critic_mode == "auto"
    assert a._code_critic_enabled() is False
    assert "off" in a._code_critic_status_label()
    a._backend = _mlx_server_backend()
    assert a._code_critic_enabled() is True
    assert a._code_critic_status_label() == "ON (parallel)"
    a._code_critic_mode = "off"
    assert a._code_critic_enabled() is False
    b = _stub_agent(tmp_path)
    b._all_roles_enabled = True
    assert b._code_critic_enabled() is True
    assert "inline" in b._code_critic_status_label()
    monkeypatch.setenv("AGENT_CODE_CRITIC", "1")
    c = _stub_agent(tmp_path)
    assert c._code_critic_mode == "on" and c._code_critic_enabled() is True
    d = GameAgent(model="stub:1b", out_path=tmp_path / "g.html", browser=MagicMock(),
                  max_iters=3, memory_root=str(tmp_path / "memory"), code_critic_mode="off")
    assert d._code_critic_mode == "off"  # explicit arg beats env


def test_code_critic_bullet_parser():
    text = (
        "blocker | goal says first-person, code draws a top-down grid; add a horizon projection | `ctx.fillRect(tx, ty, 8, 8)`\n"
        "- missing | no ArrowUp handler though goal lists it | `if (e.key === 'ArrowLeft')`\n"
        "style | rename foo | `foo`\n"
        "teach | Math.sin is unavailable on the FPGA; use a LUT | `Math.sin(a)`\n"
        "not a bullet at all\n"
    )
    out = GameAgent._parse_code_critic_bullets(text)
    assert [b["severity"] for b in out] == ["blocker", "missing", "teach"]
    assert out[0]["anchor"] == "ctx.fillRect(tx, ty, 8, 8)"
    assert GameAgent._parse_code_critic_bullets("LGTM") == []
    assert GameAgent._parse_code_critic_bullets("") == []


def test_code_critic_harvest_drops_stale_anchors_and_folds_into_next_turn(tmp_path):
    """Bullets anchored to code the coder already changed are dropped;
    survivors are appended to the queued next-user turn, labeled [CODE
    CRITIC] so clean-pass coaching suppression keeps them."""
    a = _stub_agent(tmp_path)
    a._backend = _mlx_server_backend()
    reviewed = "<html><script>ctx.fillRect(tx, ty, 8, 8); Math.sin(a); if (e.key === 'ArrowLeft') {}</script></html>"
    # Coder changed the fillRect line since the review was spawned.
    current = reviewed.replace("ctx.fillRect(tx, ty, 8, 8)", "drawHorizon()")
    a._current_file = current
    a._messages = [{"role": "user", "content": "FIX TURN base"}]
    reply = (
        "blocker | top-down, not first-person | `ctx.fillRect(tx, ty, 8, 8)`\n"
        "teach | Math.sin not on FPGA; LUT | `Math.sin(a)`\n"
    )

    async def _fake_run(html, iteration):
        return reply

    a.run_code_critic = _fake_run  # type: ignore[method-assign]

    async def _drive():
        await a._spawn_code_critic(reviewed, 2)
        assert a._code_critic_task is not None
        await a._code_critic_task
        return await a._fold_code_critic_into_next_turn()

    assert asyncio.run(_drive()) is True
    content = a._messages[-1]["content"]
    assert "[CODE CRITIC] review of iter 2" in content
    assert "Math.sin(a)" in content
    assert "top-down, not first-person" not in content  # stale anchor dropped
    assert a._pending_coaching == []  # moved out of the queue
    assert a._code_critic_task is None


def test_code_critic_not_spawned_when_disabled_and_cancel_is_safe(tmp_path):
    a = _stub_agent(tmp_path)
    a._code_critic_mode = "off"

    async def _drive():
        await a._spawn_code_critic("<html></html>", 1)
        assert a._code_critic_task is None
        await a._cancel_code_critic()
        return await a._harvest_code_critic("<html></html>")

    assert asyncio.run(_drive()) is None


def test_code_critic_coaching_survives_clean_pass(tmp_path):
    """A green probe report used to drop [VLM-CRITIQUE] notes (prefix not
    in the must-keep list). Both critic prefixes must survive."""
    a = _stub_agent(tmp_path)
    a._previous_report_ok = True
    a._previous_report = {"probes": [{"ok": True}], "errors": [], "soft_warnings": [],
                          "page_errors": [], "console_errors": []}
    a._pending_coaching = [
        "[CODE CRITIC] review of iter 1:\n  - missing: no fire key",
        "[VLM-CRITIQUE] ship is off-canvas",
        "generic nudge that should be suppressed",
    ]
    out = a._flush_user_injections("base")
    assert "[CODE CRITIC]" in out and "[VLM-CRITIQUE]" in out
    assert "[CRITIC]" in out
    assert "generic nudge" not in out
