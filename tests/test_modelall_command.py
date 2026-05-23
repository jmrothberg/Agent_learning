"""/modelall <N|name> — shorthand to stage the same model on all 3 slots.

The 4-GPU workstation auto-pins 11434→GPU1, 11435→GPU2, 11436→GPU3
(see four_gpu_workstation_topology). Phase 1 / 2 features (non-blocking
critic on slot 3, best-of-N fan-out across slots 1/2/3) work best when
all three slots host identical capacity — /modelall is the one-command
setup for that.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402


def _bare_app() -> CodingBoxApp:
    """Construct without spinning up the Textual driver — we only need
    the slash-command machinery, not the rendered UI."""
    app = CodingBoxApp()
    # Listing the dispatcher consults — fake an Ollama tag so the
    # _cmd_set_model_slot resolver picks it up by name.
    app._last_listing = [("ollama", "qwen3.6:27b")]
    # No active agent — staging-only path.
    app.agent = None
    # Stub out anything that touches the UI directly. The slot setter
    # calls _update_status and _log_info; both are safe to no-op for
    # this unit test.
    app._update_status = MagicMock()  # type: ignore[assignment]
    app._log_info = MagicMock()  # type: ignore[assignment]
    app._log_error = MagicMock()  # type: ignore[assignment]
    return app


def test_modelall_stages_same_model_on_all_three_slots():
    app = _bare_app()

    app._cmd_set_model_all("qwen3.6:27b")

    # Slot 1 — no slot suffix, default role for coder.
    assert app._next_model == "qwen3.6:27b"
    assert app._next_backend == "ollama"
    # Slot 2 — role critic.
    assert app._next_model2 == "qwen3.6:27b"
    assert app._next_backend2 == "ollama"
    assert app._next_role2 == "critic"
    # Slot 3 — role architect.
    assert app._next_model3 == "qwen3.6:27b"
    assert app._next_backend3 == "ollama"
    assert app._next_role3 == "architect"


def test_modelall_clears_all_slots_when_bare():
    app = _bare_app()
    # Pre-populate as if the user previously staged something.
    app._next_model = "qwen3.6:27b"
    app._next_backend = "ollama"
    app._next_role = None
    app._next_model2 = "qwen3.6:27b"
    app._next_backend2 = "ollama"
    app._next_role2 = "critic"
    app._next_model3 = "qwen3.6:27b"
    app._next_backend3 = "ollama"
    app._next_role3 = "architect"

    app._cmd_set_model_all("")

    for slot_str in ("", "2", "3"):
        assert getattr(app, f"_next_model{slot_str}") is None
        assert getattr(app, f"_next_backend{slot_str}") is None
        assert getattr(app, f"_next_role{slot_str}") is None


def test_modelall_works_by_index_from_list():
    app = _bare_app()
    app._last_listing = [
        ("ollama", "qwen3.6:27b"),
        ("ollama", "deepseek-v4-flash"),
    ]
    # Pick #2 from /list.
    app._cmd_set_model_all("2")

    assert app._next_model == "deepseek-v4-flash"
    assert app._next_model2 == "deepseek-v4-flash"
    assert app._next_model3 == "deepseek-v4-flash"
    assert app._next_role2 == "critic"
    assert app._next_role3 == "architect"


def test_modelall_help_text_lists_the_command():
    """/help must surface /modelall so the user can discover it."""
    app = _bare_app()
    # _cmd_help writes via self._log(...) which is itself a no-op stub
    # path. We just need to ensure the help-line table includes the
    # expected text; cheapest check is the source of _cmd_help itself
    # which we already grep at build time — but a runtime check that
    # nothing throws is also worth keeping.
    rendered: list[str] = []
    app._log = lambda *args, **kwargs: rendered.append(" ".join(str(a) for a in args))  # type: ignore[assignment]
    app._status_manual_body = None
    # _cmd_help also sets _status_manual_body and calls _update_status;
    # both are safe with the stubs above.
    app._cmd_help()
    assert any("/modelall" in line for line in rendered), (
        "/help output must mention /modelall"
    )
