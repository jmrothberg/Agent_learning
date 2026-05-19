"""Pins for WAIT mode being the default manual safety mode."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


def _app_stub() -> CodingBoxApp:
    app = CodingBoxApp.__new__(CodingBoxApp)
    app.agent = None
    app._run_profile = "local_manual"
    app._last_listing = [("mlx", "/tmp/TestModel")]
    app._next_backend = None
    app._next_model = None
    app._session_done = True
    app._session_backend_info = None
    app._session_model = None
    app._profile_review_model = None
    app._profile_review_auto_apply = False
    app._log_info = lambda *args, **kwargs: None
    app._log_error = lambda *args, **kwargs: None
    app._update_status = lambda *args, **kwargs: None
    app._update_mode_bar = lambda *args, **kwargs: None
    return app


def test_wait_mode_is_effective_before_agent_exists() -> None:
    app = _app_stub()

    assert app._effective_step_mode() is True


def test_loading_model_stages_next_session_in_wait_mode() -> None:
    app = _app_stub()
    app._run_profile = "local_auto"

    app._cmd_set_model("1")

    assert app._run_profile == "local_manual"
    assert app._effective_step_mode() is True
