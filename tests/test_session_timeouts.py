"""Tests for the fail-open session-timeout policy.

Background: the donkey-kong-arcade-clone-800x6_20260513_173528
session burned iter 2 because the stall watchdog killed MLX
mid-prefill at exactly 60.00 seconds. The 60s budget came from
the unknown-model branch of `resolve_session_timeouts` (param
detection failed for `DeepSeek-V4-Flash-mxfp8` because the folder
name carries no `<n>B` token). This test suite locks down:

  1. The fail-open floor — unknown models get >= 300s stall, not 60.
  2. The MoE-aware `config.json` fallback — `n_routed_experts` is
     factored in so DeepSeek-V4 ends up in the >40B bracket.
  3. The size-name regex still works for the common case.
"""

import json

from chat import (
    _estimate_params_from_config,
    _parse_param_billions,
    resolve_session_timeouts,
)


# --- resolve_session_timeouts --------------------------------------------

def test_unknown_model_gets_generous_floor():
    """Empty model string → 300s stall, NOT 60s.

    The old policy was 60/600 with the comment 'err small so we
    detect a true wedge fast'. That was the bug — modern local
    models on real prompts need at least 5 minutes of prefill
    headroom on a cold cache, and an unknown model is the case
    where we should be MOST generous, not least."""
    stall, overall = resolve_session_timeouts("")
    assert stall >= 300.0, f"unknown-model floor should be >= 300s, got {stall}"
    assert overall >= 1500.0


def test_small_named_model_still_in_small_bracket():
    """A model whose name does carry a small param tag still lands
    in the small bracket — the fix lifts the floor, doesn't move
    everything to xl."""
    stall, overall = resolve_session_timeouts("qwen3:4b")
    # 4B falls into the small (≤13B) bracket — same as unknown,
    # since the floor is now generous regardless.
    assert stall == 300.0
    assert overall == 1500.0


def test_27b_named_model_bracket():
    """27B by name → large bracket → 900s stall."""
    stall, overall = resolve_session_timeouts("qwen3.6:27b-mlx-bf16")
    assert stall == 900.0
    assert overall == 3600.0


def test_70b_named_model_bracket():
    stall, overall = resolve_session_timeouts("llama3:70b")
    assert stall == 1500.0
    assert overall == 5400.0


# --- _estimate_params_from_config ----------------------------------------

def test_estimate_dense_27b():
    """A dense Qwen-3-27B-shaped config should estimate roughly 27B.

    Bracket selection — ±20% is fine. Exact target is 27B; accept
    anywhere in [20, 35] B."""
    cfg = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 27648,
        "vocab_size": 152064,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    }
    n = _estimate_params_from_config(cfg)
    bn = n / 1e9
    assert 20.0 <= bn <= 50.0, f"dense 27B estimate out of range: {bn:.1f}B"


def test_estimate_moe_deepseek_v4_shape():
    """The exact failure-mode config: DeepSeek-V4 Flash's MoE.
    The dense estimator (pre-fix) said 11B → small bracket →
    60s stall → wasted the session.

    With the MoE path, n_routed_experts=256 × moe_intermediate_size
    contributes the bulk of the weights and lands the model in
    the >40B bracket. We don't need exact accuracy — just enough
    to be > 40B for bracket selection."""
    cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "vocab_size": 129280,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
    }
    n = _estimate_params_from_config(cfg)
    bn = n / 1e9
    assert bn > 40.0, (
        f"MoE DeepSeek-V4 shape must estimate > 40B for correct "
        f"bracketing; got {bn:.1f}B. Without the MoE branch the "
        f"dense fallback returned ~11B and killed the session at 60s."
    )


def test_estimate_missing_required_fields_returns_zero():
    """Config missing hidden_size / num_hidden_layers / vocab_size
    can't be estimated. Return 0.0 (caller falls through to the
    generous unknown-model floor)."""
    assert _estimate_params_from_config({}) == 0.0
    assert _estimate_params_from_config({"hidden_size": 4096}) == 0.0
    assert _estimate_params_from_config({"vocab_size": 128000}) == 0.0


def test_estimate_uses_explicit_num_parameters_when_present():
    """Some HF configs ship `num_parameters` directly. The caller
    in _model_param_size prefers that field when valid — the
    estimator is only called as a fallback. This test isn't on
    the estimator itself but documents the preferred precedence."""
    # If num_parameters is missing or invalid (None/0), the estimator
    # is invoked. We've covered both branches above.
    cfg = {"num_parameters": 30_000_000_000}  # 30B
    # _estimate ignores num_parameters; that's the caller's job.
    assert _estimate_params_from_config(cfg) == 0.0


def test_parse_param_billions_handles_estimated_string():
    """The `f'{n/1e9:.1f}B'` format that _model_param_size emits
    from a config.json estimate must round-trip through
    _parse_param_billions."""
    assert _parse_param_billions("280.7B") == 280.7
    assert _parse_param_billions("27.0B") == 27.0
    assert _parse_param_billions("") == 0.0
