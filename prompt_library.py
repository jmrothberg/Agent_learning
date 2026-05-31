"""Curated cross-session library of game-build prompts.

`memory/prompt_library.jsonl` holds one JSON object per line:

    {"n": 1, "name": "street-fighter", "title": "Street Fighter",
     "tags": [...], "prompt": "Build a single-screen 2D fighting game ..."}

The TUI `/games` command lists them by number; `/games <N>` loads prompt #N
into the input box (press Enter to build). Hand-curated like
`memory/playbook.jsonl` — edited by humans, not learned. Each prompt is
written to encode the agent's planning lessons: it names the entity AND every
action pose the game needs (so the multi-frame planner generates a frame per
named action up front), plus controls, win/lose, and visual style.
"""
from __future__ import annotations

import json
from pathlib import Path

_REL = Path("memory") / "prompt_library.jsonl"


def _resolve_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    # Match memory.py's convention: cwd-relative `memory/` for a normal run.
    cwd_path = _REL
    if cwd_path.exists():
        return cwd_path
    # Fallback: alongside this module, so it loads regardless of cwd.
    return Path(__file__).resolve().parent / _REL


def load_prompt_library(path: str | Path | None = None) -> list[dict]:
    """Return the curated prompts sorted by number.

    Each entry is a dict with at least `n` (int) and `prompt` (str); `title`
    defaults to `name` then `#n`. Malformed lines are skipped, not fatal —
    a broken line never blocks the rest of the library.
    """
    p = _resolve_path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        if not isinstance(rec.get("n"), int) or not str(rec.get("prompt", "")).strip():
            continue
        rec.setdefault("title", rec.get("name") or f"#{rec['n']}")
        out.append(rec)
    out.sort(key=lambda r: r["n"])
    return out


def get_prompt(n: int, path: str | Path | None = None) -> dict | None:
    """Return the library entry numbered `n`, or None if absent."""
    for rec in load_prompt_library(path):
        if rec["n"] == n:
            return rec
    return None
