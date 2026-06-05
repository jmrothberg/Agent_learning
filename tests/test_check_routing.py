"""Tests for `/check` model routing.

User report (2026-05-16): `/check with claude` said "needs a local
model." Root cause analysis: chat.py's old branching only knew about
Claude; any other cloud name (gpt-5, gpt-4o, gemini-...) fell into
the local-MLX-VLM resolver and errored. The fix added:

  1. `vision_judge._cloud_vendor()` — explicit vendor mapping.
  2. `vision_judge._openai_judge()` — OpenAI Responses-API call,
     mirrors the Anthropic helper.
  3. chat.py routes via `_cloud_vendor()` BEFORE attempting the
     local-MLX resolver, and recognizes more aliases (`gpt`, `gpt-5`,
     `openai`, etc.). With no arg, uses the active session model if
     it's a VLM (no API call).

These tests cover the routing — not the actual API calls, which need
network keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_judge import _cloud_vendor, _looks_like_local_mlx  # noqa: E402


# ---------------------------------------------------------------------------
# _cloud_vendor — explicit vendor mapping
# ---------------------------------------------------------------------------


def test_cloud_vendor_recognizes_claude_family():
    assert _cloud_vendor("claude") == "anthropic"
    assert _cloud_vendor("claude-sonnet-4-6") == "anthropic"
    assert _cloud_vendor("claude-opus-4-8") == "anthropic"
    assert _cloud_vendor("claude-haiku-4-5") == "anthropic"
    assert _cloud_vendor("anthropic-experimental") == "anthropic"


def test_cloud_vendor_recognizes_openai_family():
    assert _cloud_vendor("gpt-5") == "openai"
    assert _cloud_vendor("gpt-5-mini") == "openai"
    assert _cloud_vendor("gpt-4o") == "openai"
    assert _cloud_vendor("gpt-4.1-vision") == "openai"
    assert _cloud_vendor("openai-anything") == "openai"
    # Reasoning-model series.
    assert _cloud_vendor("o1-pro") == "openai"
    assert _cloud_vendor("o3-mini") == "openai"
    assert _cloud_vendor("o4-experimental") == "openai"


def test_cloud_vendor_returns_none_for_local():
    """Anything that isn't a known cloud prefix is treated as local
    and the caller routes to the MLX-VLM resolver."""
    assert _cloud_vendor("qwen3.6-27b-mxfp8") is None
    assert _cloud_vendor("Qwen3.6-27B") is None
    assert _cloud_vendor("/Users/me/MLX_Models/Qwen3.6-27B-mxfp8") is None
    assert _cloud_vendor("llava-1.6") is None
    assert _cloud_vendor("") is None


def test_cloud_vendor_case_insensitive():
    """Real-world: users will Title-Case or UPPER-CASE the vendor."""
    assert _cloud_vendor("Claude") == "anthropic"
    assert _cloud_vendor("GPT-5") == "openai"
    assert _cloud_vendor("Openai") == "openai"


# ---------------------------------------------------------------------------
# _looks_like_local_mlx — negation of the cloud check, stays consistent
# ---------------------------------------------------------------------------


def test_looks_like_local_mlx_is_negation_of_cloud_vendor():
    """Either a name is a cloud vendor, or it's a local-MLX query —
    never both. Catches drift if someone tweaks one helper but not
    the other."""
    samples = [
        "claude", "claude-sonnet-4-6", "anthropic-foo",
        "gpt-5", "gpt-4o", "openai", "o1-pro", "o3-mini",
        "qwen3.6", "Qwen3.6-27B-mxfp8", "llava", "minicpm-v",
    ]
    for s in samples:
        local = _looks_like_local_mlx(s)
        cloud = _cloud_vendor(s)
        # Exactly one must be true: local OR cloud, never both.
        assert (cloud is None) == local, (
            f"{s!r}: cloud={cloud!r} but local={local}"
        )


# ---------------------------------------------------------------------------
# Imports — surface failures so a syntax error in vision_judge.py
# fails THIS test loudly instead of hiding in `/check` runtime.
# ---------------------------------------------------------------------------


def test_openai_judge_is_importable():
    """`_openai_judge` exists and is async. We don't call it (would
    need OPENAI_API_KEY + network); we just confirm the symbol is
    wired so future imports won't break."""
    import vision_judge
    import inspect
    assert hasattr(vision_judge, "_openai_judge")
    assert inspect.iscoroutinefunction(vision_judge._openai_judge)


def test_judge_visual_progress_routes_via_cloud_vendor():
    """Smoke-check the routing inside judge_visual_progress reads
    _cloud_vendor — guards against someone removing the call when
    refactoring."""
    import vision_judge
    src = Path(vision_judge.__file__).read_text()
    assert "_cloud_vendor(use_model)" in src
    assert "_openai_judge" in src
    assert "_anthropic_judge" in src
