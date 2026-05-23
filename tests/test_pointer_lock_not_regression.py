"""Tests for the harness-environment error filter on the pageerror feed.

Motivating trace:
  games/traces/make-a-first-person-shooter-ga_20260523_152317.log

Bug: the auto-input smoke test triggers `element.requestPointerLock()`
from a non-active document, which fires an UNCAUGHT pageerror
("The root document of this element is not valid for pointer lock").
This error is harmless for real users (pointer-lock works once they
click) but the regression detector counts it as a "new page error"
and auto-reverts otherwise-good patches. In the DOOM trace iters
5/7/11 all reverted on this — the model's CSS patches were correct.

Fix: `LiveBrowser._on_pageerror` filters pointer-lock errors into
`_warnings` instead of `_page_errors`, so the model still sees them
(under soft warnings) but the regression detector doesn't trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import LiveBrowser  # noqa: E402


def _make_lb() -> LiveBrowser:
    lb = LiveBrowser.__new__(LiveBrowser)
    # Mirror LiveBrowser.__init__ buffers without launching a browser.
    lb._errors = []
    lb._console_errors = []
    lb._page_errors = []
    lb._warnings = []
    lb._logs = []
    return lb


def test_pointer_lock_error_routed_to_warnings_not_page_errors() -> None:
    lb = _make_lb()
    lb._on_pageerror("Error: The root document of this element is not valid for pointer lock.")

    assert lb._page_errors == []
    assert lb._errors == []
    assert len(lb._warnings) == 1
    assert "pointer lock" in lb._warnings[0].lower()
    # The marker lets the model and the trace reader know why this
    # didn't fail the test.
    assert "harness-env" in lb._warnings[0]


def test_real_uncaught_still_routes_to_page_errors() -> None:
    """Sanity: actual bugs must still register as page_errors so the
    fix loop kicks in. Don't accidentally swallow real exceptions."""
    lb = _make_lb()
    lb._on_pageerror("TypeError: Cannot read properties of null (reading 'x')")

    assert len(lb._page_errors) == 1
    assert "TypeError" in lb._page_errors[0]
    assert len(lb._errors) == 1
    assert lb._warnings == []


def test_pointer_lock_difference_does_not_count_as_new_page_error() -> None:
    """End-to-end-ish: a previous report with 0 page_errors and a new
    report whose only pageerror is the pointer-lock artifact must
    compare equal under the regression detector's len-based check."""
    prev_lb = _make_lb()
    new_lb = _make_lb()
    new_lb._on_pageerror("Error: The root document of this element is not valid for pointer lock.")

    prev_page_errors = list(prev_lb._page_errors)
    new_page_errors = list(new_lb._page_errors)
    # This is the literal expression used in agent.py auto-revert
    # (line ~11158): `len(report.page_errors) > len(prev.page_errors)`.
    assert not (len(new_page_errors) > len(prev_page_errors))
