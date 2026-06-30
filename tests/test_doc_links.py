"""Doc path guards — referenced scripts/paths in maintainer docs must exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
_DOC_FILES = [
    _REPO / "AGENTS.md",
    _REPO / "DEV.md",
    _REPO / "TEST.md",
    _REPO / "eval" / "OPERATIONS.md",
    _REPO / "FOR_NEXT_LLM.md",
    _REPO / "HARNESS_DEBUG.md",
]

# `eval/foo.py`, `scripts/bar.sh`, `memory/baz.jsonl`, `tests/test_x.py`
_PATH_RE = re.compile(
    r"`((?:eval|scripts|memory|tests)/[A-Za-z0-9_./\-]+(?:\.(?:py|sh|jsonl|txt|md))?)`"
)


def _collect_doc_paths() -> set[str]:
    found: set[str] = set()
    for doc in _DOC_FILES:
        if not doc.is_file():
            continue
        text = doc.read_text(encoding="utf-8")
        for m in _PATH_RE.finditer(text):
            found.add(m.group(1).split("#")[0].rstrip("/"))
    return found


@pytest.mark.parametrize("rel_path", sorted(_collect_doc_paths()))
def test_doc_referenced_path_exists(rel_path: str) -> None:
    p = _REPO / rel_path
    if rel_path.endswith("/"):
        assert p.is_dir(), rel_path
    else:
        assert p.exists(), rel_path
