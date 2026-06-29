"""File-based handoff between tune_serial_loop and the overnight monitor/agent.

Terminal batch runs all night WITHOUT stdin/Enter. After each game the loop
writes inter_game_pending.json and blocks until inter_game_ready.json appears
(written by the Cursor watcher after triage + fixes, or by the monitor on timeout).

Usage (agent, after applying fixes):
  .venv/bin/python eval/tune_inter_game_ready.py --out-dir games/tune_serial10/run_07_big \\
      --note "playbook: grid-chase tag bump"
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PENDING_NAME = "inter_game_pending.json"
READY_NAME = "inter_game_ready.json"


def pending_path(out_dir: Path) -> Path:
    return out_dir / PENDING_NAME


def ready_path(out_dir: Path) -> Path:
    return out_dir / READY_NAME


def write_pending(out_dir: Path, payload: dict[str, Any]) -> Path:
    """Loop calls after each game — signals watcher to triage before next game."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    path = pending_path(out_dir)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def write_ready(out_dir: Path, *, note: str = "", released_by: str = "agent") -> Path:
    """Release the loop to start the next game."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "released_by": released_by,
        "note": note,
    }
    path = ready_path(out_dir)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def load_pending(out_dir: Path) -> dict[str, Any]:
    path = pending_path(out_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_ready(out_dir: Path) -> dict[str, Any]:
    path = ready_path(out_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def pending_awaiting_fix(out_dir: Path) -> bool:
    """True when pending exists and is newer than the last ready (or no ready yet)."""
    p = pending_path(out_dir)
    r = ready_path(out_dir)
    if not p.is_file():
        return False
    if not r.is_file():
        return True
    return p.stat().st_mtime > r.stat().st_mtime


def pending_age_seconds(out_dir: Path) -> float:
    p = pending_path(out_dir)
    if not p.is_file():
        return 0.0
    return max(0.0, time.time() - p.stat().st_mtime)


def wait_for_ready(
    out_dir: Path,
    *,
    timeout_s: float,
    poll_s: float = 5.0,
) -> bool:
    """Block until ready file is newer than pending. Returns False on timeout."""
    p = pending_path(out_dir)
    if not p.is_file():
        return True
    pending_mtime = p.stat().st_mtime
    deadline = time.time() + timeout_s if timeout_s > 0 else float("inf")
    while time.time() < deadline:
        r = ready_path(out_dir)
        if r.is_file() and r.stat().st_mtime >= pending_mtime:
            return True
        time.sleep(poll_s)
    return False


def sync_status(out_dir: Path) -> dict[str, Any]:
    pending = load_pending(out_dir)
    ready = load_ready(out_dir)
    awaiting = pending_awaiting_fix(out_dir)
    return {
        "awaiting_agent_fix": awaiting,
        "pending": pending or None,
        "ready": ready or None,
        "pending_age_s": round(pending_age_seconds(out_dir), 1) if awaiting else 0,
    }
