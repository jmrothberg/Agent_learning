"""Focused tests for low-level Ollama streaming recovery."""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama_io import stream_chat_with_retry  # noqa: E402


class _FakeStream:
    def __init__(self, chunks: list[dict]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeClient:
    def __init__(self):
        self.options_seen: list[dict] = []
        self.keep_alive_seen: list[float | str | None] = []

    async def chat(self, *, model, messages, stream, options, keep_alive=None):
        self.options_seen.append(dict(options or {}))
        self.keep_alive_seen.append(keep_alive)
        if len(self.options_seen) == 1:
            raise ollama.ResponseError(
                "memory layout cannot be allocated with num_gpu = 999",
                500,
            )
        return _FakeStream([
            {"message": {"content": "ok"}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 1},
        ])


def test_stream_retry_drops_num_gpu_but_keeps_num_ctx() -> None:
    client = _FakeClient()
    result = asyncio.run(
        stream_chat_with_retry(
            client,
            "qwen3.6:27b-q8_0",
            [{"role": "user", "content": "hi"}],
            options={"num_ctx": 262144, "num_gpu": 999, "temperature": 0.1},
            keep_alive=-1,
            max_retries=0,
        )
    )

    assert result.text == "ok"
    assert client.options_seen == [
        {"num_ctx": 262144, "num_gpu": 999, "temperature": 0.1},
        {"num_ctx": 262144, "temperature": 0.1},
    ]
    assert client.keep_alive_seen == [-1, -1]
