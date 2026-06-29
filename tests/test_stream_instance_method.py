"""Regression: orphan @classmethod on _stream broke get_backend(role) (run_06)."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import GameAgent  # noqa: E402


def test_stream_is_instance_method_not_classmethod():
    attr = inspect.getattr_static(GameAgent, "_stream")
    assert not isinstance(attr, classmethod), (
        "_stream must be an instance method; @classmethod makes get_backend(role) fail"
    )
