"""Tests for the model-agnostic session-timeout policy.

The earlier bracket-table approach was the wrong design — it violated
the project's standing rule (CLAUDE.md: "we do NOT inspect the model
name") and rotted whenever a new model didn't fit its substring
table. The replacement is one generous timeout for every model,
paired with the activity-aware stall watchdog (see
test_mlx_stall_activity.py) which prevents false stalls during
prefill.

These tests pin the policy: same numbers for every model, fail-open
floor, no model-name parsing involved.
"""

from chat import resolve_session_timeouts


def test_returns_same_values_for_every_model():
    """The whole point: model name doesn't change the answer."""
    cases = [
        "",
        "qwen3:4b",
        "qwen3.6:27b-mlx-bf16",
        "llama3:70b",
        "gpt-5",
        "claude-opus-4-8",
        "/Users/anyone/MLX_Models/Some-Future-Model-mxfp8",
        "/Users/anyone/MLX_Models/DeepSeek-V4-Flash-mxfp8",
        "this-model-doesnt-exist-yet",
    ]
    first = resolve_session_timeouts(cases[0])
    for name in cases[1:]:
        assert resolve_session_timeouts(name) == first, (
            f"timeout policy should be model-agnostic; got different "
            f"result for {name!r} vs empty string"
        )


def test_floor_is_generous():
    """A 10-minute quiet-window stall is plenty for any realistic
    prompt under the activity-aware watchdog. The earlier policy's
    60s default was the bug that killed multiple DK traces."""
    stall, overall = resolve_session_timeouts("")
    assert stall >= 600.0
    assert overall >= 1800.0
    # And overall is comfortably more than stall, so a stream that
    # uses its full stall window on prefill still has time to
    # generate.
    assert overall >= stall * 2.0


def test_argument_unused_signature_compat():
    """The function still accepts a model argument (for back-compat
    with callers that pass it), but the value is ignored."""
    # Pass weird types — should never raise.
    assert resolve_session_timeouts("") == resolve_session_timeouts("X")
    assert resolve_session_timeouts(None) == resolve_session_timeouts("any")  # type: ignore[arg-type]
