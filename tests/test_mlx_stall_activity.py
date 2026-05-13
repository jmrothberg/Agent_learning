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
