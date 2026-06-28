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
