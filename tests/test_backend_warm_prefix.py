"""Phase 0.4 — Backend.warm_prefix.

Validates the default implementation that ships with the `Backend` ABC:
  - sends a 1-token cap stream_chat
  - returns a dict with ok / elapsed_s / tokens / error
  - swallows exceptions (treated as advisory)
  - respects timeout_s
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import Backend, BackendInfo, StreamResult  # noqa: E402


class _StubBackend(Backend):
    """Minimal Backend implementation for unit-testing warm_prefix.

    Records the last stream_chat call so the test can assert options/
    cap behavior.
    """

    def __init__(self, *, behavior: str = "ok") -> None:
        self.info = BackendInfo(
            name="ollama",
            model="stub-model",
            source="test",
            endpoint="http://127.0.0.1:0",
        )
        self._behavior = behavior
        self.last_options: dict[str, Any] | None = None
        self.last_keep_alive: Any = None

    async def stream_chat(
        self,
        messages,
        *,
        on_token=None,
        options=None,
        keep_alive=None,
        stall_seconds=600.0,
        overall_seconds=1800.0,
        max_retries=1,
        on_stall=None,
        on_progress=None,
        cancel_event=None,
    ) -> StreamResult:
        self.last_options = dict(options or {})
        self.last_keep_alive = keep_alive
        if self._behavior == "raise":
            raise RuntimeError("stub exploded")
        if self._behavior == "timeout":
            await asyncio.sleep(1.0)  # exceeds our timeout below
        return StreamResult(text="x", tokens=1, duration_s=0.01, stalled=False)

    async def is_vlm(self) -> bool:
        return False


def test_warm_prefix_caps_output_and_returns_ok():
    bk = _StubBackend(behavior="ok")
    res = asyncio.run(bk.warm_prefix([{"role": "user", "content": "hi"}]))
    assert res["ok"] is True
    assert res.get("tokens") == 1
    # Caps the model: num_predict (Ollama) and max_tokens (other backends).
    assert bk.last_options is not None
    assert bk.last_options.get("num_predict") == 1
    assert bk.last_options.get("max_tokens") == 1
    assert bk.last_options.get("temperature") == 0.0


def test_warm_prefix_swallows_exceptions():
    bk = _StubBackend(behavior="raise")
    res = asyncio.run(bk.warm_prefix([{"role": "user", "content": "hi"}]))
    assert res["ok"] is False
    assert "stub exploded" in res["error"]


def test_warm_prefix_honors_timeout():
    bk = _StubBackend(behavior="timeout")
    res = asyncio.run(
        bk.warm_prefix(
            [{"role": "user", "content": "hi"}],
            timeout_s=0.05,
        )
    )
    assert res["ok"] is False
    assert res["error"] == "timeout"


def test_warm_prefix_caller_options_win():
    # Caller can override num_predict if they explicitly want more
    # (rare — but the merge must respect their value, not clobber it).
    bk = _StubBackend(behavior="ok")
    asyncio.run(
        bk.warm_prefix(
            [{"role": "user", "content": "hi"}],
            options={"num_predict": 4, "num_ctx": 16384},
        )
    )
    assert bk.last_options is not None
    assert bk.last_options["num_predict"] == 4
    # And our own defaults still arrive for keys the caller didn't set.
    assert bk.last_options.get("max_tokens") == 1
    assert bk.last_options.get("num_ctx") == 16384
