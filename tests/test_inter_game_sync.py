"""Inter-game handoff: loop waits for watcher release, not Terminal Enter."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.inter_game_sync import (  # noqa: E402
    pending_awaiting_fix,
    wait_for_ready,
    write_pending,
    write_ready,
)


def test_pending_then_ready_releases(tmp_path: Path) -> None:
    write_pending(tmp_path, {"label": "01_foo", "outcome": "fresh_fail"})
    assert pending_awaiting_fix(tmp_path) is True

    def _release() -> None:
        time.sleep(0.2)
        write_ready(tmp_path, note="test fix")

    import threading
    t = threading.Thread(target=_release, daemon=True)
    t.start()
    assert wait_for_ready(tmp_path, timeout_s=5.0, poll_s=0.05) is True
    assert pending_awaiting_fix(tmp_path) is False


def test_ready_older_than_pending_still_awaiting(tmp_path: Path) -> None:
    write_ready(tmp_path, note="stale")
    time.sleep(0.05)
    write_pending(tmp_path, {"label": "02_bar"})
    assert pending_awaiting_fix(tmp_path) is True
