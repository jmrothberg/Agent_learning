#!/usr/bin/env python3
"""Maintainer helper: fill or refresh `prompt_640` on library entries.

Run from repo root:

  .venv/bin/python scripts/gen_prompt_640_library.py

Game-specific wording (mechanics, on-screen sizes, franchise art, ply/search)
lives ONLY in memory/prompt_library.jsonl. This script never hardcodes a
title. If `prompt_640` already exists, keep its body and refresh the shared
TARGET=/640 footer. If missing, derive a first draft from the media `prompt`
(strip CDN/video sentences, inject pixel-map art language).

Keeps media-mode `prompt` untouched. `/640` + `/games N` uses prompt_640.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "memory" / "prompt_library.jsonl"

FOOTER = (
    " TARGET=/640 JMR native: ONE 640×480 HTML file. Do NOT emit assets, "
    "sounds, or videos tags; no CDN, no fetch, no WebGL. Art MUST be classic "
    "arcade animated PIXEL MAPS (compact 2D 0/1 or palette-index arrays, blit "
    "with fillRect per texel) or at most 16 inline data:image sheets — include "
    "at least 2 frames for idle/walk/attack wherever motion matters so "
    "animation reads clearly. imageSmoothingEnabled=false. HUD via canvas "
    "fillText. Never ship solid colored squares/circles as the final look "
    "for playfield units."
)

_TARGET_640_RE = re.compile(r"\s*TARGET=/640\b.*$", re.S)


def _strip_media_sentences(text: str) -> str:
    # Drop video / diffuser pipeline sentences; keep mechanics.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    keep: list[str] = []
    skip_re = re.compile(
        r"generate\s+video|image-to-video|<videos>|<assets>|<sounds>|"
        r"muted\s+skippable|full-screen\s+overlay|z-image|stable\s+audio|"
        r"from\s+cdn|three\.js\s+from\s+cdn",
        re.I,
    )
    for p in parts:
        if skip_re.search(p):
            continue
        keep.append(p)
    return " ".join(keep).strip()


def _inject_art(text: str) -> str:
    t = text
    t = re.sub(
        r"Generated PNGs must be drawn with drawImage \(not fillRect placeholders\)\.?",
        "Draw units with animated pixel maps (not fillRect placeholders).",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Generated PNGs\s*→\s*drawImage\.?",
        "Use animated pixel-map sprites.",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"high-resolution(?: detailed)? sprites",
        "classic arcade pixel-map sprites",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Colorful high-resolution",
        "Colorful classic pixel-map",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"generate high-resolution transparent sprite art",
        "author compact animated pixel-map sprite art",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"Generate <assets>[^.]*\.",
        "Author inline pixel-map sprites for each unit. ",
        t,
        flags=re.I,
    )
    if "pixel map" not in t.lower() and "pixel-map" not in t.lower():
        t = t.rstrip(".") + (
            ". Animate key actions with ≥2 pixel-map frames so motion reads."
        )
    return t


def derive_prompt_640(rec: dict) -> str:
    """Body from jsonl when present; else a draft from media `prompt`.

    Never reads a Python title table — hand-tuned craft stays in the JSON.
    """
    existing = str(rec.get("prompt_640") or "").strip()
    if existing:
        body = _TARGET_640_RE.sub("", existing).rstrip()
    else:
        body = _inject_art(_strip_media_sentences(str(rec.get("prompt") or "")))
    body = body.strip()
    if not body.endswith("."):
        body += "."
    return body + FOOTER


def main() -> None:
    rows = []
    for line in LIB.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    out_lines = []
    for rec in rows:
        rec["prompt_640"] = derive_prompt_640(rec)
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    LIB.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    n640 = sum(1 for r in rows if r.get("prompt_640"))
    print(f"wrote {n640}/{len(rows)} prompt_640 entries → {LIB}")


if __name__ == "__main__":
    main()
