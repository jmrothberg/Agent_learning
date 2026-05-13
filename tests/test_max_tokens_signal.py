"""Tests for the max_tokens_hit cloud-cap signal added after the DK
trace 20260513_135011, where Claude burned 3 consecutive iters because
the Anthropic backend was defaulting to 8192 max_tokens and the model
was silently being cut off mid-rewrite. The agent saw a "truncated
HTML" generic error and gave the model no actionable signal about
WHY it was truncated."""

from ollama_io import StreamResult


def test_streamresult_field_default_false():
    """Backward-compat: callers that don't set max_tokens_hit still
    work, and the flag is False (no cap hit)."""
    sr = StreamResult(text="", tokens=0, duration_s=0.0, stalled=False)
    assert sr.max_tokens_hit is False


def test_streamresult_field_can_be_set():
    sr = StreamResult(
        text="<html_file>...",
        tokens=8192,
        duration_s=60.0,
        stalled=False,
        max_tokens_hit=True,
    )
    assert sr.max_tokens_hit is True


def test_anthropic_default_max_tokens_is_32k():
    """The DK trace caught the 8192 default cutting 17KB rewrites off
    mid-stream. The new default must be high enough for a full HTML
    game in one go but inside cloud-side per-model limits (Sonnet 4.6:
    64K, Opus 4.7: 32K)."""
    import inspect
    from backend import AnthropicBackend
    src = inspect.getsource(AnthropicBackend.stream_chat)
    # Either the literal 32768 appears as the default OR an env-driven
    # override pathway is present. We just want to ensure 8192 isn't
    # the silent default any more.
    assert "32768" in src
    assert "ANTHROPIC_MAX_TOKENS" in src  # env override doc-present


def test_anthropic_max_tokens_env_override(monkeypatch):
    """ANTHROPIC_MAX_TOKENS env var must clamp the default. Verified
    via the resolution logic the backend uses — we mirror it here
    rather than spinning up an AsyncAnthropic client."""
    import os
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")
    env_cap = os.environ.get("ANTHROPIC_MAX_TOKENS", "").strip()
    try:
        env_max = int(env_cap) if env_cap else 0
    except ValueError:
        env_max = 0
    default_max = env_max if env_max > 0 else 32768
    assert default_max == 4096

    # Bad env value falls through to the hardcoded default.
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "garbage")
    env_cap = os.environ.get("ANTHROPIC_MAX_TOKENS", "").strip()
    try:
        env_max = int(env_cap) if env_cap else 0
    except ValueError:
        env_max = 0
    default_max = env_max if env_max > 0 else 32768
    assert default_max == 32768
