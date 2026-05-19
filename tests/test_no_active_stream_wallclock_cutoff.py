"""Active streams must not be cut off by an absolute wall-clock cap."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_io import stream_chat  # noqa: E402


class _SlowStream:
    def __init__(self) -> None:
        self._chunks = iter([
            {"message": {"content": "<html_file>"}},
            {"message": {"content": "<html>"}},
            {"message": {"content": "</html></html_file>"}},
            {"done": True, "prompt_eval_count": 1, "eval_count": 3},
        ])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration
        await asyncio.sleep(0.02)
        return chunk

    async def aclose(self) -> None:
        return None


class _FakeClient:
    async def chat(self, **kwargs):  # noqa: D401
        return _SlowStream()


def test_slow_active_stream_ignores_overall_wallclock_cap() -> None:
    async def run():
        return await stream_chat(
            _FakeClient(),
            "fake",
            [{"role": "user", "content": "make html"}],
            stall_seconds=1.0,
            overall_seconds=0.01,
        )

    result = asyncio.run(run())

    assert result.stalled is False
    assert result.text.endswith("</html></html_file>")
    assert result.completion_tokens == 3
