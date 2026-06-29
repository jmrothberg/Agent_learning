"""Tests for the rejected-reply stub + JS-source-in-body blocker round
(plan rejected_reply_stub_and_script_leak_blocker).

Trace-backed fixes from run build-a-first-person-3d-shoote_20260611_213744
(DeepSeek-V4-Flash):

  1. Rejected-reply stub — three format-rejected replies (~86 KB, crashed/
     looped streams with unclosed <html_file>) stayed verbatim in
     `_messages`, ballooning the prompt 9K → 47K tokens. A rejected reply
     with nothing usable is replaced in history by a 400-char head + an
     explicit elision marker; the full text stays in the trace/.log.
  2. JS-SOURCE-IN-BODY blocker — iter 5's patch broke a script boundary
     and 2,517 chars of JavaScript rendered as visible page text. The
     report showed the raw body sample but never named the failure.

Pure-function / source-pinned tests — no model, no Chromium.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools as tools_module  # noqa: E402
from agent import GameAgent  # noqa: E402
from tools import _js_source_in_body_warning  # noqa: E402


# ---------------------------------------------------------------------------
# 1. _stub_rejected_reply — pure helper
# ---------------------------------------------------------------------------

def test_stub_keeps_head_and_marker():
    reply = "<html_file>\n<!DOCTYPE html>\n" + ("x" * 30000)
    stub = GameAgent._stub_rejected_reply(reply, "unclosed_html_file")
    assert stub is not None
    # Head preserved so the model recognizes what it sent.
    assert stub.startswith("<html_file>\n<!DOCTYPE html>")
    assert reply[:400] == stub[:400]
    # Marker names the rejection and forbids repeating.
    assert "unclosed_html_file" in stub
    assert "elided" in stub
    assert "do not repeat" in stub
    # Massive size reduction (the whole point).
    assert len(stub) < 700
    assert len(stub) < len(reply) // 10


def test_stub_reports_elided_chars():
    reply = "y" * 5000
    stub = GameAgent._stub_rejected_reply(reply, "tags_in_fence")
    assert stub is not None
    assert f"{5000 - 400} chars elided" in stub


def test_stub_skips_short_replies():
    # Stubbing a short reply would save nothing — return None.
    assert GameAgent._stub_rejected_reply("short reply", "tags_in_fence") is None
    assert GameAgent._stub_rejected_reply("x" * 600, "tags_in_fence") is None
    assert GameAgent._stub_rejected_reply("", "tags_in_fence") is None


def test_stub_fires_just_above_threshold():
    head = GameAgent._REJECTED_REPLY_STUB_HEAD
    assert GameAgent._stub_rejected_reply("z" * (head + 201), "k") is not None


# ---------------------------------------------------------------------------
# 1b. Stub wiring in the no-usable-code branch (source-pinned)
# ---------------------------------------------------------------------------

def _run_src() -> str:
    return GameAgent.run_loop_inspect_source()


def test_run_wires_stub_after_no_usable_code_trace():
    src = _run_src()
    idx = src.index('"kind": "no_usable_code"')
    # Window widened (Phase 0F): the no_usable_code trace now carries
    # content-shape + stall fields, so the stub wiring sits a bit further down.
    after = src[idx:idx + 3600]
    assert "_stub_rejected_reply" in after
    assert '"kind": "rejected_reply_stubbed"' in after
    assert '"chars_elided"' in after


def test_run_stub_requires_format_rejection_and_nothing_usable():
    # plan-only / probes-only / media-only replies carry value and must
    # NOT be stubbed; prose without a format rejection is also exempt.
    src = _run_src()
    idx = src.index("_stub_rejected_reply")
    guard = src[max(0, idx - 900):idx]
    assert "format_rejection is not None" in guard
    assert "plan_only or probes_only or media_only" in guard


def test_run_stub_replaces_last_assistant_message_defensively():
    # The replacement only touches the message it appended: role must be
    # assistant and content must equal the rejected reply.
    src = _run_src()
    idx = src.index("_stub_rejected_reply")
    guard = src[max(0, idx - 900):idx]
    assert '.get("role") == "assistant"' in guard
    assert '.get("content") == reply' in guard


def test_fingerprint_runs_before_stub():
    # Identical-repeat detection fingerprints the FULL reply; the stub
    # must come after it in the branch so detection is unaffected.
    src = _run_src()
    fp_idx = src.index("current_fp = GameAgent._reply_fingerprint(reply)")
    stub_idx = src.index("_stub_rejected_reply")
    assert fp_idx < stub_idx


# ---------------------------------------------------------------------------
# 2. _js_source_in_body_warning — pure helper
# ---------------------------------------------------------------------------

def _trace_body() -> str:
    # Reconstructed from the actual trace report (iter 5, 2,517 chars).
    return (
        "Health: \xa0 Ammo: \xa0 Score:\n"
        "const enemies=[]; function spawnEnemies(n){ for(let i=0;i<n;i++){ "
        "const e={x:0,z:0,hp:30}; enemies.push(e); } }\n"
        "if(tx>=0&&tx<16&&tz>=0&&tz<16&&maze[tz][tx]===0){"
        "camera.position.x=nx;camera.position.z=nz;} else{ "
        "const t2x=Math.floor(camera.position.x); }\n"
    ) * 8


def test_js_in_body_fires_on_trace_body():
    warn = _js_source_in_body_warning(_trace_body())
    assert warn is not None
    assert warn.startswith("JS-SOURCE-IN-BODY")
    # Names the two usual causes so the model can act.
    assert "</script>" in warn
    assert "outside the <script> tag" in warn


def test_js_in_body_silent_on_hud_text():
    assert _js_source_in_body_warning("Health: 100  Ammo: 50  Score: 0") is None
    # Long HUD/story prose with no JS signatures stays silent.
    prose = (
        "You awaken in a dungeon. Your function here is simple: survive. "
        "Collect the key and let the gate variable guide you onward. "
    ) * 10
    assert _js_source_in_body_warning(prose) is None


def test_js_in_body_silent_on_short_or_empty_bodies():
    assert _js_source_in_body_warning("") is None
    assert _js_source_in_body_warning(None) is None
    # JS-looking but under the length floor (a tiny inline snippet shown
    # in a HUD tooltip shouldn't gate).
    assert _js_source_in_body_warning("const x=1; function f(){}") is None


def test_js_in_body_requires_two_signature_kinds():
    # One signature kind alone (here: const-assignment), padded long —
    # not enough evidence of leaked source.
    one_kind = ("const score = 12345 " * 30)
    assert len(one_kind) > 200
    assert _js_source_in_body_warning(one_kind) is None


def test_load_and_test_wires_js_in_body_gate():
    src = inspect.getsource(tools_module.LiveBrowser.load_and_test)
    assert "_js_source_in_body_warning(body_text)" in src
    idx = src.index("_js_source_in_body_warning")
    after = src[idx:idx + 300]
    assert "soft_warnings" in after  # gating channel
