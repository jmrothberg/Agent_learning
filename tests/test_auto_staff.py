"""Tests for auto-staff behavior added 2026-05-21.

When the user assigns --role architect on a sidecar slot, architect-split
auto-enables in one step. Critic screenshot review is NOT auto-enabled —
use /vlm-critique on explicitly (or /allroles).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


def _app_stub() -> CodingBoxApp:
    """Minimal app stub matching the patterns in test_wait_mode_defaults."""
    app = CodingBoxApp.__new__(CodingBoxApp)
    app.agent = None
    app._run_profile = "local_manual"
    app._last_listing = [
        ("mlx", "Qwen3.6-27B-mxfp8"),
        ("mlx", "Llama-VLM-3.3-8B"),
        ("mlx", "Mistral-Large-2025"),
    ]
    app._next_backend = None
    app._next_model = None
    app._next_backend2 = None
    app._next_model2 = None
    app._next_role2 = None
    app._next_backend3 = None
    app._next_model3 = None
    app._next_role3 = None
    app._session_role2 = None
    app._session_role3 = None
    app._session_done = True
    app._session_backend_info = None
    app._session_model = None
    app._is_streaming = False
    app._profile_review_model = None
    app._profile_review_auto_apply = False
    app._use_prefill = True
    app._use_vlm_critique = False
    app._vlm_critique_auto = False
    app._use_double_screenshot = False
    app._use_architect_split = False
    app._architect_split_auto = False
    app._log_info = lambda *args, **kwargs: None
    app._log_error = lambda *args, **kwargs: None
    app._update_status = lambda *args, **kwargs: None
    app._update_mode_bar = lambda *args, **kwargs: None
    return app


def test_model2_role_only_inherits_staged_model1() -> None:
    """README_forMac: /model2 --role critic must inherit model 1 when no N."""
    app = _app_stub()
    app._next_model = "Qwen3.6-27B-mxfp8"
    app._next_backend = "mlx"

    app._cmd_set_model2("--role critic")

    assert app._next_model2 == "Qwen3.6-27B-mxfp8"
    assert app._next_backend2 == "mlx"
    assert app._next_role2 == "critic"


def test_model2_shorthand_critic_inherits_model1() -> None:
    app = _app_stub()
    app._next_model = "Qwen3.6-27B-mxfp8"
    app._next_backend = "mlx"

    app._cmd_set_model2("--critic")

    assert app._next_model2 == "Qwen3.6-27B-mxfp8"
    assert app._next_role2 == "critic"


def test_model3_role_only_inherits_staged_model1() -> None:
    app = _app_stub()
    app._next_model = "Qwen3.6-27B-mxfp8"
    app._next_backend = "mlx"

    app._cmd_set_model3("--role architect")

    assert app._next_model3 == "Qwen3.6-27B-mxfp8"
    assert app._next_backend3 == "mlx"
    assert app._next_role3 == "architect"


def test_model3_shorthand_architect_inherits_model1() -> None:
    app = _app_stub()
    app._next_model = "Qwen3.6-27B-mxfp8"
    app._next_backend = "mlx"

    app._cmd_set_model3("--architect")

    assert app._next_model3 == "Qwen3.6-27B-mxfp8"
    assert app._next_role3 == "architect"


def test_model2_role_critic_does_not_auto_enable_vlm_critique() -> None:
    app = _app_stub()
    assert app._use_vlm_critique is False
    assert app._vlm_critique_auto is False

    app._cmd_set_model2("2 --role critic")

    assert app._use_vlm_critique is False, (
        "staging critic must not flip /vlm-critique — user opts in explicitly"
    )
    assert app._vlm_critique_auto is False


def test_model3_role_critic_does_not_auto_enable_vlm_critique() -> None:
    app = _app_stub()
    app._cmd_set_model3("2 --role critic")
    assert app._use_vlm_critique is False
    assert app._vlm_critique_auto is False


def test_model2_role_critic_then_explicit_vlm_critique_on() -> None:
    app = _app_stub()
    app._cmd_set_model2("1 --role critic")
    assert app._use_vlm_critique is False
    app._cmd_toggle_vlm_critique("on")
    assert app._use_vlm_critique is True
    assert app._vlm_critique_auto is False


def test_model2_role_architect_auto_enables_architect_split() -> None:
    app = _app_stub()
    assert app._use_architect_split is False
    assert app._architect_split_auto is False

    app._cmd_set_model2("3 --role architect")

    assert app._use_architect_split is True
    assert app._architect_split_auto is True


def test_explicit_vlm_critique_off_clears_auto_flag() -> None:
    app = _app_stub()
    app._cmd_toggle_vlm_critique("on")
    assert app._use_vlm_critique is True

    app._cmd_toggle_vlm_critique("off")

    assert app._use_vlm_critique is False
    assert app._vlm_critique_auto is False


def test_explicit_architect_off_clears_auto_flag() -> None:
    app = _app_stub()
    app._cmd_set_model2("3 --role architect")
    assert app._use_architect_split is True
    assert app._architect_split_auto is True

    app._cmd_toggle_architect("off")

    assert app._use_architect_split is False
    assert app._architect_split_auto is False


def test_already_on_vlm_critique_unchanged_when_staging_critic() -> None:
    app = _app_stub()
    app._cmd_toggle_vlm_critique("on")
    assert app._use_vlm_critique is True
    assert app._vlm_critique_auto is False

    app._cmd_set_model2("2 --role critic")

    assert app._use_vlm_critique is True
    assert app._vlm_critique_auto is False
