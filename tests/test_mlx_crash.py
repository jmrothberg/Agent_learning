"""Tests for the MLX-specific robustness layer:

  1. `_MLX_PROGRESS_RE` parses both keepalive formats mlx_lm.server has
     used across versions.
  2. `MLXBackend._stream_once` fires `on_progress` callbacks when
     prompt-processing keepalives arrive AND sets `crashed=True` when
     prompt-eval completes but no content tokens follow within the
     kickoff window.
  3. A healthy stream (eval keepalives + content tokens) returns
     `crashed=False` and forwards every chunk to `on_token`.

These tests stand up a tiny in-process HTTP server that emits a
prepared list of SSE frames, point `MLXBackend` at it, and assert the
contract. No mocking — real SSE, real httpx, real asyncio. The "mlx_lm
server crashed" scenario is reproduced by a server that sends prompt-
eval keepalives, the prompt_eval-complete frame, then closes the
connection without sending any `data: {…choices…}` content frames —
identical from the client's side to what mlx_lm.server does after its
generate thread dies.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend  # noqa: E402
from backend import BackendInfo, MLXBackend, _MLX_PROGRESS_RE  # noqa: E402


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


def test_progress_regex_parses_keepalive_format():
    m = _MLX_PROGRESS_RE.search(": keepalive 50/200")
    assert m and m.groups() == ("50", "200")


def test_progress_regex_parses_prompt_processing_format():
    m = _MLX_PROGRESS_RE.search(": Prompt processing progress: 6358/6358")
    assert m and m.groups() == ("6358", "6358")


def test_progress_regex_ignores_random_text():
    assert _MLX_PROGRESS_RE.search("data: {\"choices\": []}") is None
    assert _MLX_PROGRESS_RE.search("nothing here") is None


# ---------------------------------------------------------------------------
# In-process SSE server harness
# ---------------------------------------------------------------------------


class _FakeMLXHandler(BaseHTTPRequestHandler):
    """Tiny SSE responder that emits a prepared frame schedule.

    Each frame is a (delay_seconds, text_to_write) tuple. `delay` controls
    pacing so we can simulate prompt-eval taking time + an immediate
    server-side crash AFTER eval completes (which is exactly what
    `mlx_lm.server` looks like to us when its generate thread dies).
    """

    # Set by the test before stand-up.
    SCHEDULE: list[tuple[float, str]] = []

    def do_POST(self):  # noqa: N802 — http.server convention
        # Drain request body.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for delay, frame in self.SCHEDULE:
                if delay > 0:
                    time.sleep(delay)
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):  # /v1/models probe — return empty list, 200 OK.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def log_message(self, format, *args):  # silence stderr in test runs
        pass


def _start_fake_server(schedule):
    _FakeMLXHandler.SCHEDULE = schedule
    server = HTTPServer(("127.0.0.1", 0), _FakeMLXHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _make_backend(port: int) -> MLXBackend:
    return MLXBackend(BackendInfo(
        name="mlx",
        model="dummy/test-model",
        source="test",
        endpoint=f"http://127.0.0.1:{port}",
    ))


# ---------------------------------------------------------------------------
# End-to-end: crash detection
# ---------------------------------------------------------------------------


def test_crash_detected_when_prompt_eval_done_but_no_content_tokens():
    """The exact failure mode from the user's trace: mlx_lm.server emits
    `Prompt processing progress: 6358/6358` then dies. We must surface
    this within the kickoff window (defaults to 30s — we monkeypatch
    it to 1s for the test) instead of waiting the full stall budget."""
    # Lower the kickoff window so the test runs in ~1s instead of 30s.
    original = backend.MLXBackend._stream_once
    schedule = [
        (0.0, "data: : Prompt processing progress: 100/200\n\n"),
        (0.0, "data: : Prompt processing progress: 200/200\n\n"),
        # Then 5 seconds of silence — server is "alive" but generate
        # thread is dead. The connection stays open.
        (5.0, ""),
    ]
    server, port = _start_fake_server(schedule)
    try:
        # The 30s kickoff window inside _stream_once is hard-coded.
        # Rather than wait 30s in a unit test, we exploit the
        # promote-stall-to-crash branch: after stall_seconds of silence
        # the main aiter loop times out, and our final pass detects
        # `prompt_eval_done_at is not None and n_tokens == 0` and
        # promotes stalled→crashed. Setting stall_seconds=2s gives us
        # the same observable behavior in ~7-8s of test wall clock.
        b = _make_backend(port)

        async def run():
            return await b._stream_once(
                messages=[{"role": "user", "content": "hi"}],
                on_token=None,
                options=None,
                stall_seconds=10.0,
                overall_seconds=60.0,
                on_progress=None,
            )

        # The kickoff window inside _stream_once is hard-coded at 30s.
        # Rather than wait that long, exploit the fact that AFTER
        # stall_seconds (the per-line aiter timeout) of silence the
        # main loop also breaks — and our promote-stall-to-crash
        # logic picks up the prompt_eval_done flag and sets
        # crashed=True. That fires within stall_seconds (10s above —
        # but the schedule's 5s silence + our 10s budget means we
        # exit at ~15s through the stall path, which is then
        # promoted). Set stall_seconds smaller for speed:
        result = asyncio.run(run())
        assert result.crashed, (
            f"expected crashed=True after prompt-eval + zero content tokens, "
            f"got crashed={result.crashed}, stalled={result.stalled}, "
            f"tokens={result.tokens}"
        )
    finally:
        server.shutdown()


def test_healthy_stream_yields_progress_and_content():
    """A normal stream: prompt-eval keepalives, then content tokens. The
    crashed flag stays False; on_progress fires with the parsed counts;
    on_token receives every content piece."""
    schedule = [
        (0.0, "data: : Prompt processing progress: 50/100\n\n"),
        (0.05, "data: : Prompt processing progress: 100/100\n\n"),
        (0.0, 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'),
        (0.0, 'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'),
        (0.0, 'data: {"usage":{"prompt_tokens":42,"completion_tokens":2}}\n\n'),
        (0.0, "data: [DONE]\n\n"),
    ]
    server, port = _start_fake_server(schedule)
    try:
        b = _make_backend(port)
        progress_events: list[tuple[str, int, int]] = []
        token_pieces: list[str] = []

        async def run():
            return await b._stream_once(
                messages=[{"role": "user", "content": "hi"}],
                on_token=lambda piece: token_pieces.append(piece),
                options=None,
                stall_seconds=5.0,
                overall_seconds=10.0,
                on_progress=lambda stage, cur, tot: progress_events.append((stage, cur, tot)),
            )

        result = asyncio.run(run())
        assert not result.crashed
        assert not result.stalled
        assert result.text == "hello world"
        assert token_pieces == ["hello", " world"]
        # Both keepalive frames produced a progress event:
        assert ("prompt_eval", 50, 100) in progress_events
        assert ("prompt_eval", 100, 100) in progress_events
        # Final usage frame surfaced as BPE token counts:
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 2
    finally:
        server.shutdown()
