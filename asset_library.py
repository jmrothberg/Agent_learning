"""Cross-session asset library — sprites + sounds compound across runs.

The per-project `_asset_cache/` already caches assets by exact
(model, prompt, size) hash. That helps when a user asks for the
literal same thing twice; it does NOT help when session 7's
"explosion sprite, 64px" could reuse the asset session 3 generated
under "small explosion, 64px transparent".

This module sits one layer above the exact cache. It indexes every
admitted asset by its tokenized prompt and a small set of metadata
(modality, size or duration, sha). Retrieval is a Jaccard token
match — same scoring shape as `memory.Playbook.retrieve`. A hit lets
`generate_assets` skip the GPU call entirely and copy the library
file into the session dir instead.

Storage layout under `memory/`:

    asset_library/
        sprites/<id>.png
        sounds/<id>.ogg
    asset_index.jsonl   # one JSON line per entry

The library writes are append-only with idempotent admission (same
sha → no-op). Cleanup is opt-in via scripts/clean_artifacts.sh.

Design rules (per the project's standing memory):
- No genre lists. Tokenization is content-agnostic.
- Local-only. Never reaches out to remote services.
- Honest "compounding only" — a hit is a hit; we do not auto-resize
  or re-prompt. If the recorded size doesn't match the request, we
  return None and let generation happen normally.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = [
    "AssetLibrary",
    "LibraryHit",
    "AssetEntry",
    "default_library_root",
]


# Reuse the same tokenizer shape as memory.Playbook so the two index
# scoring schemes stay aligned — we import lazily to keep this module
# importable from contexts where memory.py isn't yet loaded.

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "for", "in", "on", "with", "to", "by",
    "is", "it", "or", "as", "at", "be", "this", "that",
    "sprite", "sound", "asset", "image", "audio",
    "transparent", "background", "png", "ogg",
})


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOPWORDS]


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def default_library_root() -> Path:
    """Project-relative default. Sits beside other memory artifacts."""
    return Path(__file__).resolve().parent / "memory"


@dataclass
class AssetEntry:
    """One indexed asset. `size_or_duration` is the size tuple for
    sprites and the float duration_s for sounds."""

    id: str
    modality: str            # "sprite" | "sound"
    prompt: str
    tokens: list[str]
    size_or_duration: tuple[int, int] | float
    sha: str
    path: str                # relative to library root
    helpful: int = 0
    harmful: int = 0
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "modality": self.modality,
            "prompt": self.prompt,
            "tokens": list(self.tokens),
            "size_or_duration": (
                list(self.size_or_duration)
                if isinstance(self.size_or_duration, tuple)
                else self.size_or_duration
            ),
            "sha": self.sha,
            "path": self.path,
            "helpful": int(self.helpful),
            "harmful": int(self.harmful),
            "last_used": float(self.last_used),
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_json(cls, obj: dict) -> "AssetEntry":
        raw = obj.get("size_or_duration")
        size_or_duration: tuple[int, int] | float
        if isinstance(raw, list) and len(raw) == 2:
            size_or_duration = (int(raw[0]), int(raw[1]))
        else:
            try:
                size_or_duration = float(raw)
            except (TypeError, ValueError):
                size_or_duration = 0.0
        return cls(
            id=str(obj.get("id") or ""),
            modality=str(obj.get("modality") or "sprite"),
            prompt=str(obj.get("prompt") or ""),
            tokens=[str(t) for t in (obj.get("tokens") or [])],
            size_or_duration=size_or_duration,
            sha=str(obj.get("sha") or ""),
            path=str(obj.get("path") or ""),
            helpful=int(obj.get("helpful", 0)),
            harmful=int(obj.get("harmful", 0)),
            last_used=float(obj.get("last_used", time.time())),
            created_at=float(obj.get("created_at", time.time())),
        )


@dataclass
class LibraryHit:
    entry: AssetEntry
    score: float
    absolute_path: Path


class AssetLibrary:
    """JSONL-backed asset index. Thread/process safe enough for the
    serial agent loop — we read the file before each query and append
    on admission. Concurrent admissions from multiple agents would race
    the index file; that's outside our use case.
    """

    DEFAULT_MAX_ENTRIES = 2000
    MIN_SCORE = 0.5

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.root = Path(root) if root else default_library_root()
        self.index_path = self.root / "asset_index.jsonl"
        self.sprite_dir = self.root / "asset_library" / "sprites"
        self.sound_dir = self.root / "asset_library" / "sounds"
        self.max_entries = int(max_entries)

    # -- io ------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.sprite_dir.mkdir(parents=True, exist_ok=True)
        self.sound_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[AssetEntry]:
        if not self.index_path.exists():
            return []
        entries: list[AssetEntry] = []
        try:
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(AssetEntry.from_json(json.loads(line)))
                except Exception:
                    continue
        except OSError:
            return []
        return entries

    def _write_all(self, entries: Iterable[AssetEntry]) -> None:
        self._ensure_dirs()
        tmp = self.index_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e.to_json(), separators=(",", ":")) + "\n")
        tmp.replace(self.index_path)

    # -- public api ----------------------------------------------------

    def retrieve(
        self,
        *,
        prompt: str,
        modality: str,
        size_or_duration: tuple[int, int] | float,
        threshold: float | None = None,
    ) -> LibraryHit | None:
        """Best Jaccard-token match against the index, scoped to the
        requested modality and matching size/duration exactly.

        Returns None when no entry scores above `threshold`.
        """
        thr = self.MIN_SCORE if threshold is None else float(threshold)
        entries = self.load_all()
        if not entries:
            return None
        q_toks = _tokenize(prompt)
        if not q_toks:
            return None
        best: LibraryHit | None = None
        for e in entries:
            if e.modality != modality:
                continue
            if not self._size_matches(e.size_or_duration, size_or_duration):
                continue
            score = _jaccard(q_toks, e.tokens)
            if score < thr:
                continue
            abs_path = self.root / e.path
            if not abs_path.exists():
                continue  # stale index entry, skip
            if best is None or score > best.score:
                best = LibraryHit(entry=e, score=score, absolute_path=abs_path)
        return best

    @staticmethod
    def _size_matches(
        recorded: tuple[int, int] | float,
        requested: tuple[int, int] | float,
    ) -> bool:
        # Sprites: exact (w, h). Sounds: duration within 10% tolerance.
        if isinstance(recorded, tuple) and isinstance(requested, tuple):
            return recorded == requested
        if not isinstance(recorded, tuple) and not isinstance(requested, tuple):
            try:
                r = float(recorded)
                q = float(requested)
            except (TypeError, ValueError):
                return False
            if r <= 0 or q <= 0:
                return False
            return abs(r - q) / max(r, q) <= 0.1
        return False

    def admit(
        self,
        *,
        prompt: str,
        modality: str,
        size_or_duration: tuple[int, int] | float,
        source_path: Path | str,
    ) -> AssetEntry | None:
        """Copy `source_path` into the library and index it. Idempotent
        on (sha, modality, size_or_duration) — duplicate admissions are
        no-ops that bump `last_used` instead of duplicating files.

        Returns the entry on success, None on failure.
        """
        src = Path(source_path)
        if not src.is_file():
            return None
        try:
            sha = hashlib.sha256(src.read_bytes()).hexdigest()
        except OSError:
            return None
        self._ensure_dirs()
        entries = self.load_all()
        # Existing entry with same sha — bump last_used and return.
        for e in entries:
            if e.sha == sha and e.modality == modality:
                e.last_used = time.time()
                self._write_all(entries)
                return e
        # New admission.
        suffix = src.suffix.lower()
        if not suffix:
            suffix = ".png" if modality == "sprite" else ".ogg"
        target_dir = self.sprite_dir if modality == "sprite" else self.sound_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        entry_id = sha[:16]
        target_path = target_dir / f"{entry_id}{suffix}"
        try:
            shutil.copy2(src, target_path)
        except OSError:
            return None
        try:
            rel = target_path.relative_to(self.root).as_posix()
        except ValueError:
            rel = str(target_path)
        entry = AssetEntry(
            id=entry_id,
            modality=modality,
            prompt=prompt,
            tokens=_tokenize(prompt),
            size_or_duration=size_or_duration,
            sha=sha,
            path=rel,
        )
        entries.append(entry)
        # LRU evict if over cap.
        if len(entries) > self.max_entries:
            entries.sort(key=lambda e: e.last_used, reverse=True)
            for victim in entries[self.max_entries:]:
                try:
                    (self.root / victim.path).unlink(missing_ok=True)
                except OSError:
                    pass
            entries = entries[: self.max_entries]
        self._write_all(entries)
        return entry

    def touch(self, entry_id: str) -> None:
        """Mark an entry as recently used (called on a library hit)."""
        entries = self.load_all()
        changed = False
        for e in entries:
            if e.id == entry_id:
                e.last_used = time.time()
                changed = True
                break
        if changed:
            self._write_all(entries)

    # -- pose-prompt recipes (cross-session animation reuse) -----------
    # A from_image pose frame whose delta-vs-idle shows it actually MOVED is
    # worth remembering: the exact prompt that produced a real distinct pose.
    # Future animation/fighting games reuse it for the same pose (faster, and
    # consistent). Stored in a separate JSONL so it never collides with the
    # image index. Keyed by (stem, pose); genre/model-agnostic.
    @property
    def pose_recipes_path(self) -> Path:
        return self.root / "pose_recipes.jsonl"

    def _load_pose_recipes(self) -> list[dict]:
        p = self.pose_recipes_path
        if not p.exists():
            return []
        out: list[dict] = []
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        continue
        except OSError:
            return []
        return out

    def admit_pose_recipe(
        self, *, stem: str, pose: str, prompt: str, delta: float,
        min_delta: float = 0.04,
    ) -> bool:
        """Record a verified-moved pose prompt. Skips clones (delta < min_delta)
        and (stem,pose) duplicates that already have an equal/higher delta."""
        if not (stem and pose and prompt) or delta is None or delta < min_delta:
            return False
        key = (stem.strip().lower(), pose.strip().lower())
        for r in self._load_pose_recipes():
            if (str(r.get("stem", "")).lower(), str(r.get("pose", "")).lower()) == key \
                    and float(r.get("delta", 0)) >= delta:
                return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            rec = {"stem": stem, "pose": pose,
                   "prompt": prompt, "delta": round(float(delta), 4)}
            with self.pose_recipes_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            return True
        except OSError:
            return False

    @staticmethod
    def _stem_tokens(s: str) -> set:
        # Light suffix strip so goal gerunds/plurals ("punching", "kicks")
        # match stored pose words ("punch", "kick"). Genre-free, no table.
        out = set()
        for w in re.findall(r"[a-z]+", s.lower()):
            for suf in ("ing", "ed", "es", "s"):
                if len(w) > len(suf) + 2 and w.endswith(suf):
                    w = w[: -len(suf)]
                    break
            out.add(w)
        return out

    def retrieve_pose_recipes(self, text: str, k: int = 4) -> list[dict]:
        """Proven pose prompts whose pose/stem words overlap `text`, best (most
        overlap, then highest delta) first. Empty `text` returns all rows."""
        rows = self._load_pose_recipes()
        if not text:
            return rows
        toks = self._stem_tokens(text)
        scored = []
        for r in rows:
            rt = self._stem_tokens(f"{r.get('pose', '')} {r.get('stem', '')}")
            ov = len(toks & rt)
            if ov:
                scored.append((ov, float(r.get("delta", 0)), r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r for _, _, r in scored[:k]]
