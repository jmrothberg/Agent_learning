"""Tests for run_micro_probes / format_micro_probes_for_model in tools.py.

These run with NO browser — they're the OpenCoder #4 pre-flight checks
that gate Chromium round-trips. Cheap, deterministic, fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import run_micro_probes, format_micro_probes_for_model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap(body: str = "", *, with_canvas: bool = True) -> str:
    """Realistic-size well-formed page with optional <canvas>.

    Padded past 200 bytes so the "essentially empty" check doesn't fire
    on tiny test fixtures.
    """
    canvas = "<canvas id='c' width='400' height='300'></canvas>" if with_canvas else ""
    style = (
        "<style>body{margin:0;padding:0;background:#111;color:#fff;"
        "font-family:system-ui;}canvas{display:block;}</style>"
    )
    return (
        "<!DOCTYPE html>\n<html><head><title>t</title>" + style + "</head>\n"
        f"<body>{canvas}\n<script>{body}</script>\n</body></html>"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_minimal_page_passes():
    html = _wrap("var x = 1; function loop(){requestAnimationFrame(loop);} loop();")
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert r["errors"] == []
    assert r["stats"]["scripts_inline"] == 1


def test_dom_only_with_inline_handlers_passes():
    """Open-domain games can have NO <canvas> and use DOM elements +
    onclick handlers — must not be flagged as broken."""
    html = (
        "<!DOCTYPE html>\n<html><head><title>todo</title>"
        "<style>body{font-family:system-ui;background:#fafafa;}"
        "button{padding:8px;margin:4px;}</style></head>\n"
        "<body><h1>Todo App</h1>\n"
        "<button onclick='add()'>add</button>\n"
        "<button onclick='clear_all()'>clear</button>\n"
        "<ul id='list'></ul>\n"
        "<script>function add(){console.log('a');}\n"
        "function clear_all(){document.getElementById('list').innerHTML='';}\n"
        "</script>\n"
        "</body></html>"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True, f"expected ok, got errors: {r['errors']}"
    assert r["stats"]["inline_event_handlers"] is True


# ---------------------------------------------------------------------------
# Empty / truncation
# ---------------------------------------------------------------------------


def test_empty_file_errors():
    r = run_micro_probes("")
    assert r["ok"] is False
    assert any("essentially empty" in e for e in r["errors"])


def test_tiny_file_errors():
    r = run_micro_probes("<html></html>")
    assert r["ok"] is False


def test_unclosed_html_errors_truncation():
    html = (
        "<!DOCTYPE html>\n<html><head><title>t</title></head>\n<body>\n"
        "<canvas></canvas>\n<script>\nvar x = 1;\n"
        + ("// padding " * 20)
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("truncat" in e.lower() or "never closed" in e for e in r["errors"])


def test_unclosed_body_errors():
    html = (
        "<!DOCTYPE html>\n<html><head><title>t</title></head>\n<body>\n"
        "<canvas></canvas>\n<script>\nvar x = 1;\n"
        + ("// padding " * 20)
        + "</html>"
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("body" in e.lower() and ("never closed" in e or "truncat" in e.lower()) for e in r["errors"])


# ---------------------------------------------------------------------------
# Script presence
# ---------------------------------------------------------------------------


def test_no_script_errors():
    html = (
        "<!DOCTYPE html>\n<html><head><title>t</title></head>\n"
        "<body><canvas></canvas></body></html>" + (" " * 200)
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("script" in e.lower() and "no" in e.lower() for e in r["errors"])


def test_external_script_only_passes():
    html = (
        "<!DOCTYPE html>\n<html><head><title>phaser game</title>"
        "<style>body{margin:0;padding:0;background:#000;}canvas{display:block;}</style>"
        "</head>\n"
        "<body><canvas id='game'></canvas>\n"
        "<script src='https://cdn.example.com/phaser.min.js'></script>\n"
        "<script>const game = new Phaser.Game({width:800, height:600});</script>\n"
        "</body></html>"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True, f"expected ok, got errors: {r['errors']}"


# ---------------------------------------------------------------------------
# Bracket balance
# ---------------------------------------------------------------------------


def test_unbalanced_braces_errors():
    """Two extra opening braces with no close — clear syntax error."""
    body = "function f() { if (true) { return 1; "
    html = _wrap(body)
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("unbalanced" in e.lower() and "{}" in e for e in r["errors"])


def test_balanced_braces_with_strings_passes():
    """Strings/regex with brackets inside should NOT trip the balance check
    after the strip-strings step."""
    body = "var s = \"{not real}\"; var t = '[also not]'; var x = 1;"
    html = _wrap(body)
    r = run_micro_probes(html)
    assert r["ok"] is True


def test_balanced_braces_with_template_literals_passes():
    body = "var s = `${foo} {bar}`; function f(){ return 1; }"
    html = _wrap(body)
    r = run_micro_probes(html)
    assert r["ok"] is True


def test_balanced_braces_with_block_comments_passes():
    body = "/* if (true) { return; } */ function f(){ return 1; }"
    html = _wrap(body)
    r = run_micro_probes(html)
    assert r["ok"] is True


def test_off_by_one_brace_warns_not_errors():
    """A single missing/extra brace might be a regex-literal false positive,
    so it should warn, not error."""
    body = "function f(){ return 1; "  # missing one closing {
    html = _wrap(body)
    r = run_micro_probes(html)
    # Off-by-one is a warning, not error. Off-by-2+ is an error.
    # In this case it's off-by-one so it should warn.
    assert any("possibly unbalanced" in w.lower() for w in r["warnings"])


# ---------------------------------------------------------------------------
# Elision markers
# ---------------------------------------------------------------------------


def test_elision_marker_errors():
    html = _wrap("// ... rest unchanged ...\nfunction draw(){}")
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("elision" in e.lower() or "incomplete" in e.lower() for e in r["errors"])


def test_existing_code_marker_errors():
    html = _wrap("function init(){}\n// (existing code)\nfunction draw(){}")
    r = run_micro_probes(html)
    assert r["ok"] is False


def test_dotted_elision_marker_errors():
    """Donkey-kong trace 20260516_124628 iter 2 shipped
    `// ...rest of seed code stays same...` past the detector because
    the previous literal-list match looked for `// rest of` (space) and
    the model emitted `// ...rest of` (no space, ellipsis glued). The
    regex variant now catches this dotted shape."""
    html = _wrap(
        "function init(){}\n  // ...rest of seed code stays same...\nfunction draw(){}"
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("elision" in e.lower() for e in r["errors"]), r["errors"]


def test_dotted_elision_rest_unchanged_variant():
    """Same regex catches the `// .. rest unchanged` form too."""
    html = _wrap("var x = 1;\n// .. rest unchanged\nvar y = 2;")
    r = run_micro_probes(html)
    assert r["ok"] is False
    assert any("elision" in e.lower() for e in r["errors"]), r["errors"]


def test_duplicate_top_level_const_detected():
    """Donkey-kong trace 20260516_124628 iter 2 had `const ctx`,
    `const state`, and `function buildLevels` each declared twice
    inside the IIFE — the concatenated-two-drafts shape. Chromium
    catches it post-load; this micro-probe catches it pre-Chromium and
    names the duplicated identifier(s) so the model can fix it in one
    turn."""
    html = _wrap(
        "(() => {\n"
        "  const ctx = canvas.getContext('2d');\n"
        "  const state = { score: 0 };\n"
        "  function buildLevels() { return []; }\n"
        "  // ... draft 1 ends here, draft 2 starts below ...\n"
        "  const ctx = canvas.getContext('2d');\n"
        "  const state = { score: 0 };\n"
        "  function buildLevels() { return []; }\n"
        "})();"
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    err_blob = "\n".join(r["errors"]).lower()
    assert "duplicate top-level declaration" in err_blob, r["errors"]
    # The error names the actual duplicated identifiers.
    assert "ctx" in err_blob and "state" in err_blob, r["errors"]


def test_duplicate_const_inside_nested_function_allowed():
    """Shadowing in nested scopes is legal JS; the probe only flags
    duplicates at the IIFE / top level. Without this guard the probe
    would false-positive on the common `for (const x of …)` pattern
    used inside multiple sibling functions."""
    html = _wrap(
        "(() => {\n"
        "  function a() { const x = 1; return x; }\n"
        "  function b() { const x = 2; return x; }\n"
        "  a(); b();\n"
        "})();"
    )
    r = run_micro_probes(html)
    err_blob = "\n".join(r.get("errors", [])).lower()
    assert "duplicate top-level declaration" not in err_blob, r["errors"]


def test_bare_script_sibling_function_locals_allowed():
    """Castlevania 20260720_175910: bare (non-IIFE) script with
    `const p` / `const mat` inside sibling functions must NOT be treated
    as concatenated drafts. Canned IIFE games already sit locals at
    depth >= 2; bare scripts need outer_depth=0."""
    html = _wrap(
        "function buildCastle() {\n"
        "  const mat = 1;\n"
        "  const p = 2;\n"
        "  return mat + p;\n"
        "}\n"
        "function animate() {\n"
        "  const mat = 3;\n"
        "  const p = 4;\n"
        "  const sinF = 0;\n"
        "  return mat + p + sinF;\n"
        "}\n"
        "buildCastle(); animate();\n"
    )
    r = run_micro_probes(html)
    err_blob = "\n".join(r.get("errors", [])).lower()
    assert "duplicate top-level declaration" not in err_blob, r["errors"]
    assert not (r.get("stats") or {}).get("duplicate_declarations")


def test_bare_script_true_top_level_concat_still_detected():
    """Bare script with two top-level `const state` is still a real
    concatenated-draft failure — outer_depth=0 must still catch it."""
    html = _wrap(
        "const state = { score: 0 };\n"
        "function tick() { return state.score; }\n"
        "const state = { score: 1 };\n"
        "tick();\n"
    )
    r = run_micro_probes(html)
    assert r["ok"] is False
    err_blob = "\n".join(r["errors"]).lower()
    assert "duplicate top-level declaration" in err_blob, r["errors"]
    assert "state" in err_blob, r["errors"]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_format_includes_stats_and_errors():
    r = run_micro_probes("")
    out = format_micro_probes_for_model(r)
    assert "MICRO-PROBE" in out
    assert "OK: False" in out
    assert "ERRORS" in out


def test_format_clean_report_has_no_errors_section():
    html = _wrap("var x = 1; function loop(){requestAnimationFrame(loop);} loop();")
    r = run_micro_probes(html)
    out = format_micro_probes_for_model(r)
    assert "OK: True" in out, f"got: {out}"
    assert "ERRORS" not in out


# ---------------------------------------------------------------------------
# API allowlist (roadmap item #2)
# ---------------------------------------------------------------------------


def test_api_allowlist_flags_canvas2d_hallucination():
    """`ctx.drawCircle(...)` is a known hallucination — flag as warning."""
    html = _wrap(
        "const ctx = document.querySelector('canvas').getContext('2d');\n"
        "ctx.fillRect(0,0,100,100);\n"   # real
        "ctx.drawCircle(50,50,20);\n"     # hallucinated
    )
    r = run_micro_probes(html)
    assert r["ok"] is True  # warnings, not errors
    assert any("drawCircle" in w and "CanvasRenderingContext2D" in w for w in r["warnings"])
    assert r["stats"].get("api_hallucinations", 0) >= 1


def test_api_allowlist_does_not_flag_real_canvas2d_method():
    """Real ctx methods (fillRect, arc, beginPath, etc.) must not warn."""
    html = _wrap(
        "const ctx = document.querySelector('canvas').getContext('2d');\n"
        "ctx.fillRect(0,0,100,100);\n"
        "ctx.arc(50,50,20,0,2*Math.PI);\n"
        "ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(100,100); ctx.stroke();\n"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert r["stats"].get("api_hallucinations", 0) == 0


def test_api_allowlist_does_not_flag_unknown_receiver():
    """User-defined object with a custom method must NOT trigger a warning —
    we only check known receiver-name conventions (`ctx`, `audioCtx`, etc)."""
    html = _wrap(
        "const myThing = { drawCircle: () => {} };\n"
        "myThing.drawCircle();\n"
        "const game = { spawn: () => {} };\n"
        "game.spawn();\n"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert r["stats"].get("api_hallucinations", 0) == 0


def test_api_allowlist_flags_audio_context_hallucination():
    """audioCtx is the strict convention — flag unknown methods on it."""
    html = _wrap(
        "const audioCtx = new AudioContext();\n"
        "const osc = audioCtx.createOscillator();\n"   # real
        "audioCtx.playSound();\n"                       # hallucinated
    )
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert any("playSound" in w and "AudioContext" in w for w in r["warnings"])


def test_api_allowlist_does_not_double_flag():
    """Same hallucination called twice should warn ONCE (deduped)."""
    html = _wrap(
        "const ctx = document.querySelector('canvas').getContext('2d');\n"
        "ctx.drawCircle(0,0,10);\n"
        "ctx.drawCircle(50,50,20);\n"
        "ctx.drawCircle(100,100,30);\n"
    )
    r = run_micro_probes(html)
    matching = [w for w in r["warnings"] if "drawCircle" in w]
    assert len(matching) == 1


def test_api_allowlist_strips_strings_and_comments():
    """`ctx.drawCircle()` inside a string or comment must NOT trigger."""
    html = _wrap(
        "const ctx = document.querySelector('canvas').getContext('2d');\n"
        "// ctx.drawCircle() — not really called\n"
        'const note = "ctx.drawCircle is not real";\n'
        "ctx.fillRect(0,0,100,100);\n"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert r["stats"].get("api_hallucinations", 0) == 0


# ---------------------------------------------------------------------------
# Asset path existence check (added after the doom-game trace where the
# 27B corrupted file paths and Chromium reported generic ERR_FILE_NOT_FOUND
# with no URL).
# ---------------------------------------------------------------------------


def test_asset_path_missing_with_close_match(tmp_path):
    """Path the model wrote is wrong; close match exists on disk."""
    assets_dir = tmp_path / "game_assets"
    assets_dir.mkdir()
    (assets_dir / "wall_stone.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out_path = tmp_path / "game.html"
    html = _wrap(
        "const ASSETS = {wall: './game_assets/wood_wall.png'};\n"
        "const img = new Image(); img.src = ASSETS.wall;\n"
    )
    r = run_micro_probes(html, out_path=out_path)
    misses = [w for w in r["warnings"] if "wood_wall.png" in w]
    assert len(misses) == 1
    assert "wall_stone.png" in misses[0]
    assert r["stats"].get("missing_asset_paths") == 1


def test_asset_path_existing_no_warning(tmp_path):
    """Path matches a real file — no warning."""
    assets_dir = tmp_path / "g_assets"
    assets_dir.mkdir()
    (assets_dir / "imp.png").write_bytes(b"PNG")
    out_path = tmp_path / "game.html"
    html = _wrap("const x = './g_assets/imp.png';\n")
    r = run_micro_probes(html, out_path=out_path)
    assert r["stats"].get("missing_asset_paths") is None
    assert all("imp.png" not in w for w in r["warnings"])


def test_asset_path_skips_when_out_path_absent(tmp_path):
    """Without out_path, no filesystem check happens (back-compat)."""
    html = _wrap("const x = './nonexistent/foo.png';\n")
    r = run_micro_probes(html)
    assert r["stats"].get("missing_asset_paths") is None


def test_asset_path_skips_cdn_urls(tmp_path):
    """Absolute https URLs are CDNs; ignore them."""
    out_path = tmp_path / "game.html"
    html = _wrap('const x = "https://cdn.example.com/img.png";\n')
    r = run_micro_probes(html, out_path=out_path)
    assert r["stats"].get("missing_asset_paths") is None


def test_asset_path_skips_data_uris(tmp_path):
    """data: URIs don't reference filesystem."""
    out_path = tmp_path / "game.html"
    html = _wrap('const x = "data:image/png;base64,iVBOR...";\n')
    r = run_micro_probes(html, out_path=out_path)
    assert r["stats"].get("missing_asset_paths") is None


# ---------------------------------------------------------------------------
# Canvas-element API allowlist (classic-doom 20260512_111015 regression)
#
# Every iter of that trace flagged `cvs.requestPointerLock()` as a
# hallucination across two extension sessions, and the mouse-look bug
# persisted plausibly because the harness kept disagreeing with the
# correct API. requestPointerLock IS a real method on Element →
# HTMLCanvasElement.
# ---------------------------------------------------------------------------


def test_canvas_request_pointer_lock_is_allowed():
    html = _wrap(
        "const cvs = document.getElementById('c');\n"
        "cvs.addEventListener('click', () => cvs.requestPointerLock());\n"
    )
    r = run_micro_probes(html)
    assert r["ok"] is True
    assert r["stats"].get("api_hallucinations", 0) == 0, (
        f"requestPointerLock incorrectly flagged: {r['warnings']}"
    )


def test_canvas_exit_pointer_lock_is_allowed():
    html = _wrap(
        "const cvs = document.getElementById('c');\n"
        "cvs.exitPointerLock && cvs.exitPointerLock();\n"
    )
    r = run_micro_probes(html)
    assert r["stats"].get("api_hallucinations", 0) == 0


def test_canvas_request_fullscreen_is_allowed():
    html = _wrap(
        "const cvs = document.getElementById('c');\n"
        "cvs.requestFullscreen();\n"
    )
    r = run_micro_probes(html)
    assert r["stats"].get("api_hallucinations", 0) == 0


def test_canvas_matches_and_closest_are_allowed():
    """Element selector API methods used in event-routing patterns."""
    html = _wrap(
        "const cvs = document.getElementById('c');\n"
        "if (cvs.matches('canvas')) console.log('ok');\n"
        "const root = cvs.closest('body');\n"
    )
    r = run_micro_probes(html)
    assert r["stats"].get("api_hallucinations", 0) == 0


def test_canvas_truly_hallucinated_method_still_flagged():
    """Regression guard: the allowlist additions must not be so broad
    that a genuinely fake method slips through."""
    html = _wrap(
        "const cvs = document.getElementById('c');\n"
        "cvs.drawScene();\n"  # not a real method
    )
    r = run_micro_probes(html)
    assert r["stats"].get("api_hallucinations", 0) >= 1
    assert any("drawScene" in w and "HTMLCanvasElement" in w
               for w in r["warnings"])


def test_extra_close_paren_promoted_to_error_via_node():
    """run_16 1942: extra `)` was only ±1 WARNING; Chromium then failed and
    flooded the local model. node confirm must make micro-probes ok=False."""
    html = _wrap(
        "const state={bullets:[],player:{x:1,y:1}};\n"
        "function fire(){\n"
        "  state.bullets.push({x:1,y:2,vx:0,vy:-1}));\n"  # extra )
        "}\n"
        "function loop(){requestAnimationFrame(loop);} loop();\n"
    )
    r = run_micro_probes(html)
    assert r["stats"]["bracket_imbalance"]["()"] == -1
    assert r["ok"] is False
    assert any("syntax error" in e.lower() for e in r["errors"])
