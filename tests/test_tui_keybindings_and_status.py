"""TUI keybinding + status-panel correctness.

Two pre-existing bugs surfaced today:
  1. Ctrl+D never shipped — Textual's `Input` widget has a built-in
     `Binding('delete,ctrl+d', 'delete_right')` that captured the
     key whenever the input box had focus (which is ~all of session
     time in a chat-style TUI). Fix: priority=True on the App's
     ship_it binding.
  2. /help replaced the entire status panel — live activity, mode,
     iter, GPU placement all disappeared while /help text was on
     screen, so a session in flight became invisible. Fix: prepend
     the live activity + mode rows above the manual body.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from textual.binding import Binding  # noqa: E402

from chat import CodingBoxApp  # noqa: E402


def test_ctrl_d_binding_has_priority_to_beat_input_widget():
    """Without priority=True, Textual's Input widget eats Ctrl+D for
    delete_right before the App's ship_it can fire."""
    ctrl_d_bindings = [
        b for b in CodingBoxApp.BINDINGS
        if (
            (isinstance(b, Binding) and b.key == "ctrl+d")
            or (isinstance(b, tuple) and b[0] == "ctrl+d")
        )
    ]
    assert ctrl_d_bindings, "App must have a Ctrl+D binding"
    b = ctrl_d_bindings[0]
    if isinstance(b, Binding):
        assert b.priority is True, (
            "Ctrl+D must be priority=True so it fires before the "
            "focused Input widget's built-in delete_right binding"
        )
        assert b.action == "ship_it"


def test_status_manual_body_does_not_swallow_live_activity():
    """When /help sets _status_manual_body, _update_status must still
    show the live activity line and mode row above the help text so the
    user can see what the agent is doing."""
    from unittest.mock import MagicMock

    app = CodingBoxApp()

    # Stub out the widget-update path so we can inspect what got rendered.
    rendered: list[str] = []
    fake_status_static = MagicMock()
    fake_status_static.update = lambda content: rendered.append(content)
    fake_mode_bar = MagicMock()
    fake_mode_bar.update = lambda content: None
    fake_mode_bar.add_class = lambda *a, **kw: None
    fake_mode_bar.remove_class = lambda *a, **kw: None

    def _fake_query_one(selector, *args, **kwargs):
        if selector == "#status-body":
            return fake_status_static
        return fake_mode_bar

    app.query_one = _fake_query_one  # type: ignore[assignment]

    # Inject the live activity content. The exact rendering of these
    # depends on a lot of session state — for the test we just stub
    # them to return distinct sentinel strings.
    app._render_activity_line = lambda: "[ACTIVITY-LINE]"  # type: ignore[assignment]
    app._render_mode_row = lambda: "[MODE-ROW]"  # type: ignore[assignment]
    app._update_mode_bar = lambda: None  # type: ignore[assignment]

    # Manual body simulates /help being on screen.
    app._status_manual_body = "[HELP-BODY] type /list for models"

    app._update_status()

    assert rendered, "status panel was never updated"
    body = rendered[-1]
    # All three components must be present — the live activity must NOT
    # be swallowed by the manual body.
    assert "[ACTIVITY-LINE]" in body, (
        "live activity line must appear in status even when /help body "
        "is set"
    )
    assert "[MODE-ROW]" in body
    assert "[HELP-BODY]" in body
    # And the activity line must come BEFORE the help body so the user
    # sees it first.
    assert body.index("[ACTIVITY-LINE]") < body.index("[HELP-BODY]")
