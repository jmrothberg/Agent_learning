"""MLX stall recovery: compact + light stream before retry."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_mlx_stall_recovery_prunes_before_continue():
    agent_src = (Path(__file__).parent.parent / "agent.py").read_text()
    idx = agent_src.index('"kind": "mlx_stall_recovery"')
    block = agent_src[max(0, idx - 1200):idx + 1200]
    assert "_force_compact_after_stall = True" in block
    assert "self._prune_messages()" in block
    assert "_mlx_stall_light_stream" not in block
    prune_pos = block.index("self._prune_messages()")
    continue_pos = block.index("\n                            continue", prune_pos)
    assert prune_pos < continue_pos


def test_mlx_stall_light_prefill_skips_patch_first():
    src = GameAgent.class_inspect_source()
    idx = src.index('"kind": "mlx_stall_light_prefill"')
    block = src[max(0, idx - 900):idx + 400]
    assert "_stall_light" in block
    assert '"<diagnose>"' in block
    assert "patch_first_prefill" not in block.split("mlx_stall_light_prefill")[0][-200:]