"""Tests for format-doctor output validation.

Trace 20260518_220003 (street-fighter): the original first-build stream
hit the wall-clock cap and emitted an unclosed `<html_file>`. The
format-doctor was invoked, its own stream was also cut off, and the
materialized recovery file was 183 bytes. The agent emitted
`format_doctor_recovered` anyway — only the downstream micro-probe
pre-flight caught the empty file. The wall-clock cutoff has since been
removed (tests/test_no_active_stream_wallclock_cutoff.py), but the
doctor's success contract still needs to validate its OWN output before
declaring recovery: a doctor reply whose materialized HTML would fail
micro-probes must not produce a `format_doctor_recovered` trace event.

These tests pin the validation contract to `tools.run_micro_probes`,
the same function the regular pre-flight uses, so future doctor reply
shapes are checked exactly the way real iter materialize is checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import run_micro_probes  # noqa: E402


# Reconstructed from the trace's truncation_recovery event:
#   broken_file_bytes: 183, reason: "unclosed <html>".
# The doctor's materialized output looked like the canvas-skeleton
# wrapper without the script body. The exact bytes don't matter — what
# matters is that the validator rejects "looks like an HTML stub but
# the body is empty" the same way the live pre-flight does.
_DOCTOR_STUB_183B = (
    "<!DOCTYPE html>\n"
    "<html lang=\"en\"><head>\n"
    "<meta charset=\"utf-8\"><title>Street Fighter</title>\n"
    "</head>\n"
    "<body>\n"
    "<canvas id=\"c\"></canvas>\n"
    "<script>\n"
    "</script>\n"
    "</body></html>"
)


def test_doctor_stub_under_size_floor_fails_validation() -> None:
    """A 183-byte stub trips the size floor — exactly the trace failure."""
    short_stub = "<!DOCTYPE html><html><body></body></html>"
    assert len(short_stub) < 200
    report = run_micro_probes(short_stub)
    assert report["ok"] is False
    err_blob = "\n".join(report["errors"]).lower()
    assert "essentially empty" in err_blob


def test_doctor_stub_with_empty_script_fails_validation() -> None:
    """The reconstructed 183-byte stub also fails validation: the
    `<script>` tag is empty so the structural checks reject it as
    a non-functional file. Regardless of which structural error
    fires, the validator must NOT return `ok=True` on this shape.
    """
    report = run_micro_probes(_DOCTOR_STUB_183B)
    assert report["ok"] is False, (
        f"empty-script doctor stub must fail validation; got: {report}"
    )


def test_doctor_recovery_with_complete_minimal_file_passes() -> None:
    """The validator must NOT block recovery on a small but complete
    file — only on structurally broken / essentially-empty output. A
    minimal canvas+RAF page should pass micro-probes cleanly, so the
    doctor's `format_doctor_recovered` path stays available for legit
    recoveries.
    """
    minimal_complete = (
        "<!DOCTYPE html><html lang=\"en\"><head>"
        "<meta charset=\"utf-8\"><title>t</title></head>"
        "<body><canvas id=\"c\" width=\"320\" height=\"240\"></canvas>"
        "<script>"
        "(function(){var c=document.getElementById('c');"
        "var ctx=c.getContext('2d');"
        "function loop(){ctx.fillStyle='#000';ctx.fillRect(0,0,320,240);"
        "requestAnimationFrame(loop);} loop();})();"
        "</script></body></html>"
    )
    assert len(minimal_complete) >= 200
    report = run_micro_probes(minimal_complete)
    assert report["ok"] is True, (
        f"complete minimal file must pass validation; got: {report}"
    )


def test_doctor_validation_uses_same_rules_as_iter_preflight() -> None:
    """Pin: the doctor branch in agent.py uses run_micro_probes for
    its self-check. If this contract changes (e.g. someone adds a
    custom doctor-only validator), this test should fail loudly so
    the regression-trace evidence stays linked to the right entrypoint.
    """
    import agent  # noqa: PLC0415
    # The agent's run() method imports run_micro_probes at module load
    # (agent.py line ~94) and uses it on both the regular pre-flight
    # AND the doctor's dry-run. Confirm the symbol is still wired so
    # the doctor validation cannot silently regress to "no check".
    assert agent.run_micro_probes is run_micro_probes
