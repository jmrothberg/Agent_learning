"""Tests for `backend.classify_model_modality`.

The classifier returns "vlm" (can read images) or "text" based on a
substring match against the model NAME. Used by chat.py /list to
badge each row, and by README docs to explain when to pick a VLM
(visual debugging via screenshots) vs a text-only model (fast iter
on a small local LLM).

Name-based; the agent's runtime `_detect_vlm` probe is the
authoritative source for an actual session. This classifier just
labels the picker UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import classify_model_modality  # noqa: E402


# ---------------------------------------------------------------------------
# VLM matches (must classify as "vlm")
# ---------------------------------------------------------------------------


def test_qwen_vl_variants():
    """Qwen family — VL and Omni multimodal variants."""
    for name in (
        "qwen-vl-7b", "qwen2-vl-7b", "qwen2.5-vl-7b", "qwen3-vl-32b",
        "qwen3.6-vl-27b", "qwen-omni-7b", "qwen2.5-omni-7b",
    ):
        assert classify_model_modality(name) == "vlm", name


def test_llava_variants():
    for name in ("llava:13b", "llava-1.5-7b", "llava-1.6-mistral-7b",
                 "bakllava"):
        assert classify_model_modality(name) == "vlm", name


def test_deepseek_vl():
    assert classify_model_modality("deepseek-vl-7b") == "vlm"
    assert classify_model_modality("deepseek-vl2") == "vlm"


def test_internvl():
    assert classify_model_modality("internvl-26b") == "vlm"
    assert classify_model_modality("internvl-chat-v1.5") == "vlm"


def test_minicpm_v():
    assert classify_model_modality("minicpm-v:8b") == "vlm"
    assert classify_model_modality("minicpm-llama3-v-2_5") == "vlm"


def test_pixtral():
    assert classify_model_modality("pixtral-12b") == "vlm"


def test_gemma3_multimodal():
    """Google Gemma 3 family — multimodal by default."""
    assert classify_model_modality("gemma3:12b") == "vlm"
    assert classify_model_modality("gemma-3-27b") == "vlm"


def test_phi_multimodal():
    assert classify_model_modality("phi-3-vision") == "vlm"
    assert classify_model_modality("phi-4-multimodal") == "vlm"


def test_misc_vlm_families():
    """Smaller / less-common VLM families that still get the badge."""
    for name in (
        "cogvlm-17b", "cogagent-vqa", "bunny-v1.0-3b",
        "moondream:1.8b", "idefics-9b", "florence-2-base",
        "mplug-owl3-7b", "paligemma-3b",
    ):
        assert classify_model_modality(name) == "vlm", name


def test_anthropic_claude_models():
    """All current Claude 3 / 4 models accept images via API."""
    for name in (
        "claude-3-opus", "claude-3-5-sonnet", "claude-3-haiku",
        "claude-opus-4-1", "claude-opus-4-7", "claude-sonnet-4-6",
        "claude-haiku-3-5", "claude-haiku-4-5-20251001",
    ):
        assert classify_model_modality(name) == "vlm", name


def test_openai_vision_models():
    """GPT-4o family + reasoning o-series all accept images."""
    for name in (
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5",
        "o1-mini", "o3-mini", "o4-mini",
    ):
        assert classify_model_modality(name) == "vlm", name


# ---------------------------------------------------------------------------
# Text-only matches (must classify as "text")
# ---------------------------------------------------------------------------


def test_qwen_non_vl_is_text():
    """Plain Qwen / Qwen-Coder without -vl- in the name = text-only."""
    for name in ("qwen3.6:27b", "qwen2.5-coder-32b", "qwen2.5:7b",
                 "qwen3.6-35b"):
        assert classify_model_modality(name) == "text", name


def test_deepseek_non_vl_is_text():
    """DeepSeek-Coder / DeepSeek-V4 / DeepSeek-V3 are text-only.
    Only deepseek-vl* is multimodal."""
    for name in (
        "deepseek-coder-33b", "deepseek-v3.5", "deepseek-v4-flash",
        "deepseek-r1-distill-32b",
    ):
        assert classify_model_modality(name) == "text", name


def test_llama_is_text():
    for name in ("llama-3.1-70b", "llama-3.2-3b", "codellama-13b",
                 "tinyllama:1.1b"):
        assert classify_model_modality(name) == "text", name


def test_mistral_codestral_is_text():
    for name in ("mistral:7b", "mistral-large", "codestral-22b",
                 "mixtral-8x7b"):
        assert classify_model_modality(name) == "text", name


def test_misc_text_only_families():
    for name in (
        "phi-3-mini", "phi-3-medium",  # non-vision phi-3 variants
        "gemma-2-9b", "gemma-2-27b",   # gemma 2 was text-only
        "yi-coder-9b", "starcoder2-15b", "wizardlm-2-22b",
    ):
        assert classify_model_modality(name) == "text", name


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_inputs_return_text():
    """Defensive — `None` or `""` returns "text" so the UI shows a
    benign badge rather than crashing."""
    assert classify_model_modality(None) == "text"
    assert classify_model_modality("") == "text"


def test_case_insensitive():
    assert classify_model_modality("QWEN2.5-VL-7B") == "vlm"
    assert classify_model_modality("LLaVa") == "vlm"
    assert classify_model_modality("GPT-4o") == "vlm"


def test_mlx_path_form_also_matches():
    """MLX models can be passed as full disk paths; classifier should
    still match on the basename."""
    # Qwen3.6 family unified vision into the base 27B — see
    # mlx-community/Qwen3.6-27B-bf16 HF card (pipeline_tag:
    # image-text-to-text). Earlier this test asserted "text" for
    # this path; that was wrong and the user caught it 2026-05-15.
    path = "/Users/jmr/MLX_Models/Qwen3.6-27B-mxfp8"
    assert classify_model_modality(path) == "vlm"
    path = "/opt/mlx/qwen2.5-vl-7b-mxfp4"
    assert classify_model_modality(path) == "vlm"
    # Plain Qwen3 (no .6) is still text-only — keep the false-positive
    # guard so the name-prefix check doesn't bleed beyond the 3.6 family.
    path = "/Users/jmr/MLX_Models/Qwen3-30B-A3B-8bit"
    assert classify_model_modality(path) == "text"
