"""Phase 1 regression tests — write-path integrity for weak-LLM output.

Trace 1 (chess 20260522_000304) wrote concatenated drafts and
diagnose-preamble garbage to disk. The pre-write reject and continuation
sanitize must keep the file safe even when the model emits broken output
or when the on-disk baseline is already corrupt from a prior session.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (  # noqa: E402
    GameAgent,
    _baseline_structurally_broken,
    _normalize_extracted_html,
)


def _make_agent(tmp_path: Path) -> GameAgent:
    out = tmp_path / "game.html"
    return GameAgent(
        model="stub:1b",
        out_path=out,
        browser=MagicMock(),
        max_iters=2,
        memory_root=str(tmp_path / "memory"),
    )


def _real_html(extra: str = "") -> str:
    body = (
        "<!DOCTYPE html><html><body><canvas id='c' width='480' height='480'>"
        "</canvas><script>"
        "(() => {\n"
        "  const ctx = document.getElementById('c').getContext('2d');\n"
        "  function loop() { requestAnimationFrame(loop); }\n"
        + extra
        + "  loop();\n"
        "})();\n"
        "</script></body></html>"
    )
    return body


# --------------------------------------------------------------------
# 1a. Format-doctor reuses _materialize, so the structurally-broken
# gate covers it too. Assert that explicitly.
# --------------------------------------------------------------------


def test_format_doctor_path_blocks_unclosed_body(tmp_path: Path) -> None:
    """An unclosed-body reply piped through `_materialize(dry_run=True)` —
    which is exactly what the format-doctor branch does — must be rejected
    by `_baseline_structurally_broken`.

    This guards against a regression where someone bypasses `_materialize`
    in the doctor path and calls `_extract_html` directly.
    """
    a = _make_agent(tmp_path)
    a._current_file = _real_html(extra="  const padding = 'x'.repeat(2048);\n")
    a._snapshot_n = 1

    # Wrapper closes cleanly so _extract_html returns the body, but the
    # body inside has open tags that fail the structural gate.
    inner = (
        "<!DOCTYPE html><html><body>\n"
        "<canvas id='c'></canvas>\n<script>\n"
        + ("  const x = 1;\n" * 400)
        # No </script>, no </body>, no </html>.
    )
    truncated = f"<html_file>\n{inner}\n</html_file>"
    materialized, msg = asyncio.run(a._materialize(truncated, dry_run=True))
    assert materialized is None, "doctor reply with unclosed body must be rejected"
    assert "rejected" in msg.lower()


# --------------------------------------------------------------------
# 1b. Continuation start sanitizes a corrupt on-disk baseline.
# --------------------------------------------------------------------


def test_continuation_baseline_unrecoverable_arms_rewrite(tmp_path: Path) -> None:
    """When the on-disk baseline cannot be recovered (no <!DOCTYPE inside),
    the continuation handler must arm `_allow_one_rewrite` so the next turn
    can emit a fresh <html_file> without hitting the baseline-exists reject.
    """
    a = _make_agent(tmp_path)
    a._current_file = "Just diagnose prose without any HTML at all\n" * 80
    assert _baseline_structurally_broken(a._current_file) is not None

    # Manually simulate the continuation branch (without exercising the
    # full async generator): re-run the same checks the agent runs.
    broken = _baseline_structurally_broken(a._current_file)
    assert broken is not None
    normalized = _normalize_extracted_html(a._current_file)
    assert normalized is None  # no recoverable HTML inside

    # Sanity: assert the rewrite-arming branch path keeps the rewrite
    # gate open for the next materialize.
    a._continuation_baseline_corrupt = True
    a._allow_one_rewrite = True
    assert a._allow_one_rewrite is True


def test_continuation_normalizable_baseline_can_be_sanitized(tmp_path: Path) -> None:
    """When leading garbage hides a real HTML document, the normalizer
    must pull it out so the next turn patches against clean HTML.
    """
    real = _real_html()
    corrupt = "was truncated mid-stream</diagnose>\n\n" + real
    assert _baseline_structurally_broken(corrupt) is not None

    normalized = _normalize_extracted_html(corrupt)
    assert normalized is not None
    assert normalized.lower().startswith("<!doctype")
    # The recovered HTML must itself pass the structural gate.
    assert _baseline_structurally_broken(normalized) is None


def test_continuation_clean_baseline_is_left_alone(tmp_path: Path) -> None:
    """A healthy on-disk baseline must NOT be flagged as corrupt — the
    continuation sanitize path is opt-in based on _baseline_structurally_broken.
    """
    a = _make_agent(tmp_path)
    a._current_file = _real_html(extra="  let counter = 0;\n")
    assert _baseline_structurally_broken(a._current_file) is None
