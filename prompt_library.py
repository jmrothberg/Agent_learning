"""Curated cross-session library of game-build prompts.

`memory/prompt_library.jsonl` holds one JSON object per line:

    {"n": 1, "name": "street-fighter", "title": "Street Fighter",
     "tags": [...], "prompt": "Build a …", "prompt_640": "Build a … TARGET=/640 …"}

`prompt` = full media pipeline (sprites/sounds/video when the goal asks).
`prompt_640` = JMR native single-file /640 path (pixel maps, no sidecars).
`/640png` = same JMR walls as `/640` (no three.js/CDN/WebGL) but art is
generated STEM-N.png sheets via `jmr:spr:N` — derived from `prompt_640`.

The TUI `/games` command lists them by number; `/games <N>` loads prompt #N
into the input box (press Enter to build). When `/640` or `/640png` is on,
`/games N` prefers the JMR-native wording. Hand-curated like
`memory/playbook.jsonl`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_REL = Path("memory") / "prompt_library.jsonl"

# Shared TARGET footers on every prompt_640 line. /640png keeps the same
# JMR walls (no CDN/WebGL/three.js) but swaps pixel-map art for STEM-N sheets.
_TARGET_640_RE = re.compile(r"TARGET=/640\b.*$", re.S)
_TARGET_640PNG = (
    "TARGET=/640png JMR native: ONE 640×480 HTML file. No CDN, no fetch, "
    "no WebGL, no three.js. Emit <assets> (one name per pose; harness packs "
    "related frames onto STEM-N.png strips, ≤16 sheets); paint with "
    "img.src=\"jmr:spr:N\" + 9-arg crop / blitSpr — not sprite()/ASSETS, not "
    "fillRect-as-sprite placeholders, not solid colored boxes for playfield "
    "units. src must be a quoted literal (not \"jmr:spr:\"+i). Dest x,y>=0 "
    "(crop dest AND source if off-glass). No 1px drawImage columns, no "
    "full-glass black wipe then splash. Do NOT emit <sounds> or <videos>. "
    "imageSmoothingEnabled=false. HUD via canvas fillText."
)


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


def _prompt_640_to_640png(text: str) -> str:
    """Same /640 game rules; rewrite art contract to STEM-N.png sheets.

    Starts from `prompt_640` so Doom/Minecraft never pull the media
    three.js CDN wording. Softens in-body pixel-map language, then applies
    the sheet TARGET footer last so body rewrites cannot mangle it.
    """
    body = (text or "").strip()
    if not body:
        return body
    # Drop old /640 TARGET before body rewrites.
    body = _TARGET_640_RE.sub("", body).rstrip()
    replacements = (
        (r"\bpixel-map(?:s)?\b", "PNG-sheet"),
        (r"\bpixel maps?\b", "PNG sheets"),
        (r"\bPIXEL MAPS\b", "PNG SHEETS"),
        (r"\bpixel sprites?\b", "PNG sprites"),
        (r"\bpixel art\b", "sheet art"),
        (r"\bpixel poses?\b", "sheet poses"),
        (r"\bpixel tiles?\b", "sheet tiles"),
        (
            r"Do NOT emit assets, sounds, or videos tags",
            "Emit <assets> for sheets; do NOT emit sounds or videos tags",
        ),
        (
            r"do NOT emit assets, sounds, or videos tags",
            "emit <assets> for sheets; do NOT emit sounds or videos tags",
        ),
    )
    for pat, repl in replacements:
        body = re.sub(pat, repl, body, flags=re.I)
    return (body + " " + _TARGET_640PNG).strip()


def effective_prompt(
    rec: dict, *, simulator_mode: bool = False, jmr_png_mode: bool = False,
) -> str:
    """Goal text for a library entry.

    `/640` (simulator): prefer `prompt_640` when present so arcade pixel-map
    goals replace video/CDN/asset-pipeline wording.
    `/640png`: same `prompt_640` base (no three.js / CDN / WebGL) with the
    TARGET/art lines rewritten for STEM-N.png + jmr:spr:N sheets.
    Media mode: always `prompt`.
    """
    if jmr_png_mode:
        alt = rec.get("prompt_640")
        if isinstance(alt, str) and alt.strip():
            return _prompt_640_to_640png(alt)
        # No prompt_640: fall back to media prompt (system still enforces
        # JMR walls); prefer not inventing sheet goals from CDN wording.
        return str(rec.get("prompt") or "").strip()
    if simulator_mode:
        alt = rec.get("prompt_640")
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    return str(rec.get("prompt") or "").strip()
