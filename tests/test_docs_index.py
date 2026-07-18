"""TEST.md must index every tests/test_*.py (organization guard)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TESTS = _REPO / "tests"
_TEST_MD = _REPO / "TEST.md"

_BEGIN = "<!-- BEGIN AUTO-TEST-INDEX -->"
_END = "<!-- END AUTO-TEST-INDEX -->"


def test_test_md_indexes_every_unit_test_file():
    """No orphaned test files — keep the suite findable without deletions."""
    assert _TEST_MD.is_file(), "TEST.md missing"
    body = _TEST_MD.read_text(encoding="utf-8")
    assert _BEGIN in body and _END in body, (
        "TEST.md missing AUTO-TEST-INDEX markers; regenerate the index section"
    )
    files = sorted(p.name for p in _TESTS.glob("test_*.py"))
    assert files, "no test_*.py files found"
    missing = [f for f in files if f not in body]
    assert not missing, (
        f"{len(missing)} test file(s) not mentioned in TEST.md: "
        f"{missing[:12]}{'…' if len(missing) > 12 else ''}"
    )


def test_auto_index_section_lists_all_files():
    """The auto-generated block itself must list every test_*.py basename."""
    body = _TEST_MD.read_text(encoding="utf-8")
    start = body.index(_BEGIN)
    end = body.index(_END)
    section = body[start:end]
    files = sorted(p.name for p in _TESTS.glob("test_*.py"))
    missing = [f for f in files if f"`{f}`" not in section]
    assert not missing, (
        f"AUTO-TEST-INDEX incomplete ({len(missing)} missing); regenerate"
    )
