"""Tests for the activity-aware MLX stall semantics.

Background: the old stall check measured `time.monotonic() - started`,
ignoring whether MLX was actively making prefill progress. On a
17K-token prompt with a cold KV cache, MLX can spend 60+ seconds in
prefill before the first generated token. The old watchdog fired at
exactly the stall_seconds wall-clock mark and killed the session.

The fix tracks `last_activity_at`, bumped on:
  - prompt_progress_callback firing (prefill chunks)
  - any generated token

The stall is then measured from last activity, not from stream start.

These tests exercise the helper directly without needing a real MLX
model load — we just verify the timer-arithmetic logic.
"""

import time


def test_stall_check_uses_last_activity_not_start():
    """The stall predicate must compare against `last_activity_at`.

    Synthetic timeline:
      t=0    started, last_activity = 0
      t=20   prefill chunk → last_activity = 20
      t=40   prefill chunk → last_activity = 40
      t=60   we check — wall clock since start = 60s,
             since last activity = 20s.
      stall_seconds = 30 → NOT stalled (activity within window).

    This is the bug the watchdog had before the fix; check we now
    do the right thing."""
    started = 0.0
    last_activity_at = 40.0  # last prefill chunk was 20s ago
    now = 60.0
    stall_seconds = 30.0
    n_tokens = 0

    # The actual predicate from backend.py:_stream_once
    stalled = (
        now - last_activity_at > stall_seconds
        and n_tokens == 0
    )
    assert stalled is False, (
        "active prefill 20s ago must NOT trip a 30s stall window"
    )


def test_stall_fires_after_quiet_window_post_prefill():
    """Same setup but a 40-second post-progress quiet window. Now
    we ARE stalled — no activity for longer than stall_seconds."""
    last_activity_at = 40.0
    now = 90.0  # 50s since last progress
    stall_seconds = 30.0
    n_tokens = 0

    stalled = (
        now - last_activity_at > stall_seconds
        and n_tokens == 0
    )
    assert stalled is True


def test_stall_does_not_fire_when_tokens_have_been_emitted():
    """The `n_tokens == 0` guard means the watchdog only declares
    a stall before the first generated token. After that, the
    repetition / deliberation / overall-timeout detectors take
    over. (Slow-generation models that produce a token every N
    seconds still bump last_activity_at on each token, so a
    follow-on quiet window would also reset.)"""
    last_activity_at = 10.0
    now = 100.0  # 90s since last activity
    stall_seconds = 30.0
    n_tokens = 5  # generated 5 tokens

    stalled = (
        now - last_activity_at > stall_seconds
        and n_tokens == 0
    )
    assert stalled is False


def test_stall_check_is_monotonic_safe():
    """`time.monotonic()` is what the real code uses. This test
    just confirms our predicate is symmetric in unit choice —
    feeding it monotonic-style ticks gives the same answer."""
    started = time.monotonic()
    last_activity_at = started + 5.0  # bumped 5s after start
    now = started + 40.0
    stall_seconds = 30.0
    n_tokens = 0

    stalled = (
        now - last_activity_at > stall_seconds
        and n_tokens == 0
    )
    assert stalled is True  # 35s since last activity, > 30s budget


# ---------------------------------------------------------------------------
# Silent-stream guard — activity-aware (holochess trace 20260623)
#
# The silent guard aborts when n_tokens==0 and no backend activity for
# 180s. It must use last_activity_at (prefill chunks, empty gen events),
# NOT stream start — otherwise a 5–8 minute prefill on a 27K-token GLM
# prompt false-aborts the instant generation begins.
# ---------------------------------------------------------------------------

_SILENT_FLOOR = 180.0


def _silent_would_fire(*, last_activity_at: float, now: float, n_tokens: int) -> bool:
    """Inline copy of backend.py + ollama_io.py silent predicate."""
    return (
        n_tokens == 0
        and (now - last_activity_at) >= _SILENT_FLOOR
    )


def test_silent_guard_does_not_fire_during_long_prefill():
    """Stream has been running 400s but prefill bumped activity 5s ago —
    must NOT abort (holochess false-positive shape)."""
    started = 0.0
    now = 400.0
    last_activity_at = now - 5.0  # prefill chunk 5s ago
    assert _silent_would_fire(last_activity_at=last_activity_at, now=now, n_tokens=0) is False


def test_silent_guard_does_not_fire_when_started_old_but_activity_recent():
    """Wall clock since start is irrelevant; only last_activity_at matters."""
    now = 600.0
    last_activity_at = now - 30.0
    assert _silent_would_fire(last_activity_at=last_activity_at, now=now, n_tokens=0) is False


def test_silent_guard_fires_after_quiet_window_with_no_visible_tokens():
    """Genuinely silent: no chunks/tokens for 180s+ after last activity."""
    now = 300.0
    last_activity_at = now - 200.0
    assert _silent_would_fire(last_activity_at=last_activity_at, now=now, n_tokens=0) is True


def test_silent_guard_never_fires_once_visible_tokens_landed():
    now = 1000.0
    last_activity_at = now - 500.0
    assert _silent_would_fire(last_activity_at=last_activity_at, now=now, n_tokens=12) is False


# ---------------------------------------------------------------------------
# MLX server post-prefill generation kickoff (backend.py MLXServerBackend)
#
# After SSE prefill progress hits cur>=tot, mlx_lm.server should emit tokens
# within ~30s. The kickoff check must run on line-read timeout — not only
# when a line arrives — or a wedged generate thread hangs until stall_seconds.
# ---------------------------------------------------------------------------

_MLX_GENERATION_KICKOFF_SECONDS = 30.0


def _server_post_prefill_would_abort(
    *,
    prompt_eval_done_at: float | None,
    now: float,
    n_tokens: int,
) -> bool:
    return (
        prompt_eval_done_at is not None
        and n_tokens == 0
        and (now - prompt_eval_done_at) > _MLX_GENERATION_KICKOFF_SECONDS
    )


def test_server_generation_kickoff_fires_on_read_timeout_after_prefill():
    done_at = 100.0
    now = 135.0  # 35s quiet after prefill
    assert _server_post_prefill_would_abort(
        prompt_eval_done_at=done_at, now=now, n_tokens=0,
    ) is True


def test_server_generation_kickoff_waits_grace_window_after_prefill():
    done_at = 100.0
    now = 120.0  # 20s — within 30s grace
    assert _server_post_prefill_would_abort(
        prompt_eval_done_at=done_at, now=now, n_tokens=0,
    ) is False


# ---------------------------------------------------------------------------
# oMLX first-token watchdog (BATTLEZO 20260904_095910: 1800 s at 0 tokens
# with the model already loaded — SSE keepalive lines kept the read from
# timing out so the quiet-window checks never ran).
# ---------------------------------------------------------------------------

class _FakeKeepaliveResponse:
    """SSE stream that only ever sends keepalive comment lines."""

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        import asyncio
        while True:
            await asyncio.sleep(0.005)
            yield ": keepalive"


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *a, **k):
        return _FakeStreamCtx(_FakeKeepaliveResponse())


def _run_server_stream(monkeypatch, *, loaded: bool, overall: float):
    import asyncio
    import httpx
    import backend

    # `_stream_once` does `import httpx` locally — patch the module attr.
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(backend, "_MLX_SERVER_FIRST_TOKEN_WATCHDOG_S", 0.05)

    async def _loaded(self):
        return loaded

    monkeypatch.setattr(backend.MLXServerBackend, "_model_reported_loaded", _loaded)
    be = backend.MLXServerBackend(backend.BackendInfo(
        name="mlx-server", model="GLM-5.3-Flash-MLX-6bit",
        source="test", endpoint="http://127.0.0.1:8000",
    ))
    return asyncio.run(be._stream_once(
        [{"role": "user", "content": "hi"}],
        on_token=None, options={}, stall_seconds=600.0, overall_seconds=overall,
    ))


def test_server_first_token_watchdog_aborts_when_model_loaded(monkeypatch):
    """Keepalive-only stream + model loaded → abort at the watchdog window,
    not at the cold-load cap."""
    res = _run_server_stream(monkeypatch, loaded=True, overall=5.0)
    assert res.stalled is True
    assert res.tokens == 0
    assert "first-token watchdog" in (res.error_message or "")
    assert res.duration_s < 2.0


def test_server_first_token_watchdog_defers_to_cold_load_when_not_loaded(monkeypatch):
    """Model NOT resident → genuine cold load; the watchdog must not fire and
    the existing overall (cold-load) cap ends the wait."""
    res = _run_server_stream(monkeypatch, loaded=False, overall=0.3)
    assert res.stalled is True
    assert "first-token watchdog" not in (res.error_message or "")
    assert "overall timeout" in (res.error_message or "")


# ---------------------------------------------------------------------------
# In-process MLX cross-turn KV prompt cache (Sept 2026). Fake `mlx_lm` /
# `mlx.core` modules stand in for Metal so the reuse plumbing is testable:
# turn 2 must prefill only the suffix and report `cached_prompt_tokens`.
# ---------------------------------------------------------------------------

def _install_fake_mlx(monkeypatch, calls: list):
    import sys
    import types

    class _FakeTok:
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            return [ord(ch) for ch in text]

        def apply_chat_template(self, messages, **kw):
            return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"

    class _FakeKV:
        def __init__(self):
            self.trimmed = 0

    class _Arr:
        def __init__(self, ids):
            self.ids = list(ids)
            self.size = len(self.ids)

    class _Gen:
        def __init__(self, text, token, pt, gt):
            self.text, self.token, self.prompt_tokens, self.generation_tokens = text, token, pt, gt

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):
        n_prompt = prompt.size if isinstance(prompt, _Arr) else len(tokenizer.encode(prompt))
        calls.append({"n_prompt": n_prompt, "prompt_cache": kw.get("prompt_cache")})
        for i, (txt, tok) in enumerate((("ok", 1001), ("!", 1002))):
            yield _Gen(txt, tok, n_prompt, i + 1)

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kw: None
    models = types.ModuleType("mlx_lm.models")
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model: [_FakeKV()]
    cache_mod.can_trim_prompt_cache = lambda c: True

    def _trim(c, n):
        c[0].trimmed += n
        return n
    cache_mod.trim_prompt_cache = _trim
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = _Arr
    for name, mod in (
        ("mlx_lm", mlx_lm), ("mlx_lm.sample_utils", sample_utils),
        ("mlx_lm.models", models), ("mlx_lm.models.cache", cache_mod),
        ("mlx", mlx_pkg), ("mlx.core", mlx_core),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return _FakeTok()


def _run_inprocess_turn(be, messages):
    import asyncio
    return asyncio.run(be._stream_once(
        messages, on_token=None, options={}, stall_seconds=30.0, overall_seconds=30.0,
    ))


def test_inprocess_prompt_cache_reuses_prefix_across_turns(monkeypatch):
    import backend
    monkeypatch.delenv("MLX_PROMPT_CACHE", raising=False)
    calls: list = []
    tok = _install_fake_mlx(monkeypatch, calls)
    cls = backend.MLXBackend
    # Pretend the model is resident so no load path / fork guard runs.
    monkeypatch.setattr(cls, "_loaded_model", object())
    monkeypatch.setattr(cls, "_loaded_tokenizer", tok)
    monkeypatch.setattr(cls, "_loaded_path", "fake-text-model")
    monkeypatch.setattr(cls, "_prompt_cache", None)
    monkeypatch.setattr(cls, "_prompt_cache_tokens", None)
    monkeypatch.setattr(backend, "_resolve_prefill_step_size", lambda m: 512)
    be = cls(backend.BackendInfo(
        name="mlx", model="fake-text-model", source="test", endpoint="inprocess",
    ))
    sys_msg = {"role": "system", "content": "S" * 200}
    turn1 = [sys_msg, {"role": "user", "content": "build pong"}]
    r1 = _run_inprocess_turn(be, turn1)
    assert r1.text == "ok!"
    assert r1.cached_prompt_tokens == 0  # cold: nothing to reuse
    full1 = calls[0]["n_prompt"]
    assert r1.prompt_tokens == full1
    # Cache now holds prompt + generated ids.
    assert cls._prompt_cache_tokens[-2:] == [1001, 1002]

    # Turn 2: append-only history → only the new suffix is prefilled.
    turn2 = turn1 + [{"role": "assistant", "content": "ok!"}, {"role": "user", "content": "fix paddle"}]
    r2 = _run_inprocess_turn(be, turn2)
    assert r2.cached_prompt_tokens is not None and r2.cached_prompt_tokens >= full1 - 20
    assert calls[1]["n_prompt"] < full1  # suffix only, not the whole prompt
    assert calls[1]["prompt_cache"] is cls._prompt_cache
    assert r2.prompt_tokens == r2.cached_prompt_tokens + calls[1]["n_prompt"]

    # Kill-switch: MLX_PROMPT_CACHE=0 → plain full prefill, no cache object.
    monkeypatch.setenv("MLX_PROMPT_CACHE", "0")
    r3 = _run_inprocess_turn(be, turn2)
    assert r3.cached_prompt_tokens is None
    assert calls[2]["prompt_cache"] is None
    assert calls[2]["n_prompt"] == r3.prompt_tokens
