"""Honest exit / stall messages — Mr. Do! 20260723_135057 regression.

Trace showed WAIT at iter 2/6 while the UI screamed:
  - "MLX call failed … Try lowering MLX_MAX_TOKENS / iogpu.wired_limit"
    (actual cause: silent thinking-channel, empty visible content)
  - "iter cap reached with failing build"
    (false — max_iters=6; loop returned early after stream stall)

These tests lock the honest wording.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent_stream as stream_mod  # noqa: E402
from agent import GameAgent  # noqa: E402


def test_exit_finalize_source_distinguishes_early_stop_from_iter_cap() -> None:
    src = inspect.getsource(GameAgent._run_exit_and_finalize)
    assert "_stopped_early_reason" in src
    assert "stopped early" in src
    assert "iter cap reached with failing build" in src  # true-cap path kept


def test_iterate_loop_records_early_stop_before_return() -> None:
    src = inspect.getsource(GameAgent._run_build_iterate_loop)
    assert "iterate_stopped_early" in src
    assert "_stopped_early_reason" in src


def test_silent_stall_hint_does_not_blame_metal_vram() -> None:
    """Silent/thinking stalls must not recommend iogpu.wired_limit_mb."""
    src = Path(stream_mod.__file__).read_text(encoding="utf-8")
    needle = 'if getattr(result, "silent", False):\n                hint = ('
    start = src.find(needle)
    assert start >= 0, "silent hint branch missing"
    end = src.find('elif backend_name == "mlx":', start)
    assert end > start
    silent_branch = src[start:end]
    assert "thinking/reasoning channel" in silent_branch
    assert "iogpu.wired_limit_mb" not in silent_branch
    assert "MLX_MAX_TOKENS" not in silent_branch


def test_classify_stall_still_matches_silent_wording() -> None:
    info = GameAgent._classify_stall(
        "Model produced no visible tokens (silent/thinking stall) "
        "at 253.05s on backend=mlx. Model emitted no visible tokens…"
    )
    assert info is not None
    assert info["kind"] == "no_tokens_stall"
    assert info["stall_seconds"] == 253.05
