"""Stream-visibility tests (2026-06-12, trace 20260612_132314).

A coder stream ran 24+ minutes at ~12 tok/s (18K+ tokens) while the TUI
console printed NOTHING for 9+ minutes — a healthy long generation was
indistinguishable from a hang, and the agent's runaway_stream_warning
only went to the trace. These tests pin the display-only fixes:

  * mega-line flush — a huge no-newline buffer flushes as a partial line
  * [stream alive] console line — tokens flowing, nothing printable
  * one-shot console mirror of the 15K runaway floor
  * brevity nudge in the [AUTONOMOUS PLAYTEST] feedback text

Nothing here aborts or truncates a stream (standing no-cutoff rule).
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import CodingBoxApp  # noqa: E402
from agent import GameAgent  # noqa: E402


def _app_stub() -> CodingBoxApp:
    """CodingBoxApp without Textual init — matches test_auto_staff pattern."""
    app = CodingBoxApp.__new__(CodingBoxApp)
    app.agent = None  # _emit_token resolves role "coder" without an agent
    app._stream_buf = ""
    app._stream_tokens = 0
    app._stream_started_at = 0.0
    app._last_token_at = 0.0
    app._is_streaming = True
    app._last_console_flush_at = 0.0
    app._last_stream_alive_note_at = 0.0
    app._runaway_console_warned = False
    app._raw_lines: list[str] = []
    app._info_lines: list[str] = []
    app._log_raw = lambda line: app._raw_lines.append(line)
    app._log_info = lambda line: app._info_lines.append(line)
    return app


# ---------------------------------------------------------------------------
# Mega-line flush
# ---------------------------------------------------------------------------


def test_megaline_flushes_without_newline():
    app = _app_stub()
    app._emit_token("x" * (CodingBoxApp._STREAM_PARTIAL_FLUSH_CHARS + 100))
    assert app._raw_lines, "a huge no-newline buffer must flush as a partial line"
    assert app._stream_buf == ""
    assert app._last_console_flush_at > 0.0


def test_normal_short_pieces_buffer_until_newline():
    app = _app_stub()
    app._emit_token("const x = 1; ")
    assert app._raw_lines == []
    assert app._last_console_flush_at == 0.0
    app._emit_token("const y = 2;\n")
    assert app._raw_lines == ["const x = 1; const y = 2;"]
    assert app._last_console_flush_at > 0.0


# ---------------------------------------------------------------------------
# Runaway console mirror
# ---------------------------------------------------------------------------


def test_runaway_console_warning_fires_once():
    app = _app_stub()
    app._stream_tokens = CodingBoxApp._RUNAWAY_CONSOLE_FLOOR - 1
    app._emit_token("a")  # crosses the floor
    warnings = [l for l in app._info_lines if "long stream" in l]
    assert len(warnings) == 1
    app._emit_token("b")  # past the floor — must NOT repeat
    warnings = [l for l in app._info_lines if "long stream" in l]
    assert len(warnings) == 1
    assert app._runaway_console_warned is True


# ---------------------------------------------------------------------------
# [stream alive] line
# ---------------------------------------------------------------------------


def test_stream_alive_fires_when_tokens_flow_but_nothing_prints():
    app = _app_stub()
    now = time.monotonic()
    app._stream_tokens = 18_000
    app._stream_started_at = now - 1400.0
    app._last_token_at = now - 1.0          # tokens flowing
    app._last_console_flush_at = now - 600.0  # 10 min of silence
    app._maybe_note_stream_alive()
    alive = [l for l in app._info_lines if "stream alive" in l]
    assert len(alive) == 1
    assert "18,000" in alive[0]


def test_stream_alive_rate_limited():
    app = _app_stub()
    now = time.monotonic()
    app._stream_tokens = 18_000
    app._stream_started_at = now - 1400.0
    app._last_token_at = now - 1.0
    app._last_console_flush_at = now - 600.0
    app._maybe_note_stream_alive()
    app._maybe_note_stream_alive()  # immediate second tick — suppressed
    alive = [l for l in app._info_lines if "stream alive" in l]
    assert len(alive) == 1


def test_stream_alive_silent_when_output_is_recent():
    app = _app_stub()
    now = time.monotonic()
    app._stream_tokens = 2_000
    app._stream_started_at = now - 120.0
    app._last_token_at = now - 1.0
    app._last_console_flush_at = now - 5.0  # console is advancing fine
    app._maybe_note_stream_alive()
    assert not any("stream alive" in l for l in app._info_lines)


def test_stream_alive_silent_when_tokens_stalled():
    """Tokens NOT flowing is a stall — the status panel covers that;
    the alive line must stay quiet so the two states are distinct."""
    app = _app_stub()
    now = time.monotonic()
    app._stream_tokens = 18_000
    app._stream_started_at = now - 1400.0
    app._last_token_at = now - 60.0          # no tokens for a minute
    app._last_console_flush_at = now - 600.0
    app._maybe_note_stream_alive()
    assert not any("stream alive" in l for l in app._info_lines)


def test_stream_alive_silent_when_not_streaming():
    app = _app_stub()
    app._is_streaming = False
    app._maybe_note_stream_alive()
    assert app._info_lines == []


# ---------------------------------------------------------------------------
# Brevity nudge in the autonomous-playtest ask
# ---------------------------------------------------------------------------


def test_autonomous_feedback_contains_brevity_nudge():
    src = inspect.getsource(GameAgent._run_autonomous_playtest)
    assert "Keep your reply brief" in src
    assert "no extended analysis" in src
    # The nudge must live INSIDE the feedback text, before the queue call.
    i_nudge = src.index("Keep your reply brief")
    i_queue = src.index("_queue_internal_feedback(feedback_text)")
    assert i_nudge < i_queue
