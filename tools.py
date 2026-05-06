"""Browser test harness for the coding-box agent.

Single public function: `test_html_file(path, run_seconds=3.0) -> dict`.

It launches a headless Chromium via Playwright, loads the file, lets it
animate for a few seconds (so requestAnimationFrame loops actually tick),
then returns a SHORT report. Keeping the report short matters: a small
model gets confused by huge logs, so we cap and truncate aggressively.

Also exposes `run_micro_probes(html: str) -> dict` — fast pre-flight
checks (HTML structure, script presence, bracket balance) that run
BEFORE the Chromium round-trip. OpenCoder's "Educational-Instruct" lesson:
cheap execution filters, often. A micro-probe failure is structurally
unrecoverable (truncated stream, syntactically broken script) and
doesn't need a 3+ second browser load to confirm.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


# Cap how much we feed back to the model. Smaller is better for small models.
_MAX_MSGS = 12          # at most this many console lines forwarded
_MAX_MSG_LEN = 240      # truncate each line to this many chars
_MAX_BODY_TEXT = 200    # tiny snippet of body text


def _truncate(s: str, n: int) -> str:
    """Truncate long strings with a clear marker so the model knows it was cut."""
    if len(s) <= n:
        return s
    return s[:n] + f"...[+{len(s) - n} chars]"


# JS injected via add_init_script BEFORE any of the page's own scripts run.
# Hooks requestAnimationFrame (so we know if the loop fires) and wraps
# addEventListener (so we can count input handlers). Shared by both
# test_html_file (sync, headless) and LiveBrowser (async, visible).
_INSTRUMENTATION_JS = """
window.__rafRan = false;
window.__listenerCount = { document: 0, window: 0, body: 0, other: 0 };

const _origRAF = window.requestAnimationFrame;
window.requestAnimationFrame = function(cb) {
    window.__rafRan = true;
    return _origRAF.call(window, cb);
};

const _origAdd = EventTarget.prototype.addEventListener;
EventTarget.prototype.addEventListener = function(type, ...rest) {
    try {
        if (this === document) window.__listenerCount.document++;
        else if (this === window) window.__listenerCount.window++;
        else if (document.body && this === document.body) window.__listenerCount.body++;
        else window.__listenerCount.other++;
    } catch (e) { /* ignore - some targets are exotic */ }
    return _origAdd.call(this, type, ...rest);
};
"""

# Downsampled canvas hash. We sample a 32x32 grid spread across the canvas
# and concatenate the RGBA bytes. Cheap (~1KB string) but catches per-frame
# motion anywhere on the playfield — the older 9-pixel fingerprint missed
# slowly-moving objects in the middle. Used by the input smoke test AND the
# frozen-canvas check.
_CANVAS_HASH_JS = """
() => {
    const c = document.querySelector('canvas');
    if (!c) return null;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (!ctx || c.width < 4 || c.height < 4) return null;
    try {
        const w = c.width, h = c.height;
        const N = 32;
        const out = [];
        for (let iy = 0; iy < N; iy++) {
            const y = ((iy + 0.5) * h / N) | 0;
            for (let ix = 0; ix < N; ix++) {
                const x = ((ix + 0.5) * w / N) | 0;
                const d = ctx.getImageData(x, y, 1, 1).data;
                // Pack into base36 so the resulting string stays compact.
                out.push(((d[0] << 16) | (d[1] << 8) | d[2]).toString(36));
            }
        }
        return out.join(',');
    } catch (e) { return null; }
}
"""

# JS run AFTER the game has had a chance to animate. Returns the canvas info
# (size, RAF flag, blank-pixel heuristic). Shared between sync + async paths.
#
# Blank detection samples a 32x32 grid (1024 samples) instead of 9 corner
# pixels. The 9-corner version was producing FALSE POSITIVES on any game
# with a centered "Press Space to Start" / score-only-in-middle screen
# because the four corner samples + four edge midpoints + one center
# sample would all hit the uniform background and miss the centered
# content. With 1024 samples, even small centered text / a single
# rendered sprite produces colors.size >= 2 → not blank.
_CANVAS_PROBE_JS = """
() => {
    const c = document.querySelector('canvas');
    if (!c) return null;
    const out = {
        width: c.width,
        height: c.height,
        raf_ran: !!window.__rafRan,
        blank: null,
        sampled_colors: null,
    };
    const ctx = c.getContext('2d', { willReadFrequently: true });
    if (!ctx || c.width < 4 || c.height < 4) return out;
    try {
        const w = c.width, h = c.height;
        const N = 32;
        const colors = new Set();
        for (let iy = 0; iy < N; iy++) {
            const y = ((iy + 0.5) * h / N) | 0;
            for (let ix = 0; ix < N; ix++) {
                const x = ((ix + 0.5) * w / N) | 0;
                const d = ctx.getImageData(x, y, 1, 1).data;
                colors.add((d[0] << 16 | d[1] << 8 | d[2]) | 0);
                if (colors.size > 4) break; // early-out: clearly not blank
            }
            if (colors.size > 4) break;
        }
        out.sampled_colors = colors.size;
        out.blank = colors.size <= 1;
    } catch (e) { /* keep blank: null */ }
    return out;
}
"""


def _build_report(
    errors: list[str],
    warnings: list[str],
    logs: list[str],
    title: str,
    canvas_info: dict | None,
    listener_info: dict,
    body_text: str,
) -> dict[str, Any]:
    """Assemble the final report dict + heuristic soft warnings.

    Pulled out so test_html_file (sync) and LiveBrowser.load_and_test (async)
    produce IDENTICAL report shapes from identical inputs.
    """
    errors = errors[:_MAX_MSGS]
    warnings = warnings[:_MAX_MSGS]
    logs = logs[:_MAX_MSGS]

    soft_warnings: list[str] = []
    if canvas_info is not None:
        if canvas_info.get("raf_ran") is False:
            soft_warnings.append(
                "HEURISTIC: <canvas> exists but requestAnimationFrame never fired - "
                "your animation loop is not running."
            )
        # NOTE: BLANK canvas alone is NOT a fail signal. Many legitimate
        # apps start blank (drawing canvas, paint, charting tools where
        # data is fed via input). LiveBrowser appends a stronger
        # "blank AND input did nothing" warning later in load_and_test
        # once it knows the input-smoke result.
    if listener_info["total"] == 0:
        soft_warnings.append(
            "HEURISTIC: zero addEventListener calls detected - the game probably "
            "ignores all input."
        )

    return {
        "ok": len(errors) == 0 and len(soft_warnings) == 0,
        "errors": errors,
        "warnings": warnings,
        "soft_warnings": soft_warnings,
        "logs": logs,
        "title": title,
        "canvas": canvas_info,
        "input_listeners": listener_info,
        "body_chars": len(body_text),
        "body_sample": _truncate(body_text.strip(), _MAX_BODY_TEXT),
    }


# ---------------------------------------------------------------------------
# Micro-probes: fast pre-flight checks (no browser)
# ---------------------------------------------------------------------------
#
# Run BEFORE the Chromium round-trip to catch structurally-broken output
# (truncated streams, empty scripts, badly-unbalanced braces). Cheap; runs
# in <1 ms on a typical 5KB game file. OpenCoder's Educational-Instruct
# pattern: discard samples cheaply and often.
#
# Conservative on errors. Only flags as ERROR what is almost certainly
# unrecoverable without re-prompting; everything fuzzy goes to WARNINGS
# so the Chromium load still gets a chance.

# --- API allowlist (browser API hallucination guard) ----------------------
#
# Models hallucinate methods that don't exist (`ctx.drawCircle`,
# `ctx.fillCircle`, `audioCtx.playSound`, etc). These crash at runtime
# and we eventually catch them via Chromium's console.error — but slow.
# A small allowlist of real method names per known receiver convention
# lets us flag the hallucinated call at micro-probe time, before the
# Chromium round-trip.
#
# Conservative philosophy:
#   - Only check receivers whose variable name matches a STRICT convention
#     (`ctx` for canvas2d, `audioCtx` for AudioContext, `cvs` for the
#     element). If the user named their canvas context `myThing`, we
#     don't try — false negatives are fine, false positives are not.
#   - Output as a WARNING, not an ERROR. Chromium has the final word;
#     this is a fast preview so the model can preempt obvious mistakes.
#   - Allowlist values are method names commonly used in games. NOT
#     exhaustive — the goal is to catch hallucinations, not to gate
#     legitimate calls. We bias toward false negatives.

# Receivers we treat as canvas2d. Variable names from the games
# literature (Aider/Cline/Bolt prompts, Mozilla MDN, JS13k post-mortems).
_CANVAS2D_RECEIVERS = {"ctx", "c2d", "ctx2d", "context", "gfx", "g2", "g2d"}

# Receivers we treat as AudioContext. We use audioCtx, not bare `audio`,
# because `audio` is overloaded (HTMLAudioElement OR AudioContext); if
# the user wrote `audio`, we can't know which they meant. Strict here.
_AUDIOCTX_RECEIVERS = {"audioctx", "audiocontext", "actx", "audctx"}

# Receivers we treat as HTMLCanvasElement (the <canvas> DOM element,
# distinct from its 2D rendering context).
_CANVAS_ELT_RECEIVERS = {"cvs", "canvas", "canvasel", "canvaselt"}

# Real CanvasRenderingContext2D methods (subset useful for games).
# Source: MDN. Hallucinations the model often emits (drawCircle,
# fillCircle, drawLine, line, point) are NOT in this set, so flagged.
_CANVAS2D_METHODS = frozenset({
    "arc", "arcto", "beginpath", "beziercurveto", "clearrect", "clip",
    "closepath", "createimagedata", "createlineargradient", "createpattern",
    "createradialgradient", "createconicgradient", "drawimage",
    "drawfocusifneeded", "ellipse", "fill", "fillrect", "filltext",
    "getcontextattributes", "getimagedata", "getlinedash", "gettransform",
    "ispointinpath", "ispointinstroke", "lineto", "measuretext", "moveto",
    "putimagedata", "quadraticcurveto", "rect", "resettransform", "restore",
    "rotate", "roundrect", "save", "scale", "setlinedash", "settransform",
    "stroke", "strokerect", "stroketext", "transform", "translate",
    # Properties accessed as `.foo` are stripped before we check, but
    # methods called as e.g. `.translate()` ARE checked.
})

_AUDIOCTX_METHODS = frozenset({
    "createoscillator", "creategain", "createmediastreamsource",
    "createmediaelementsource", "createanalyser", "createbiquadfilter",
    "createbuffer", "createbuffersource", "createchannelmerger",
    "createchannelsplitter", "createconstantsource", "createconvolver",
    "createdelay", "createdynamicscompressor", "createiirfilter",
    "createpanner", "createperiodicwave", "createscriptprocessor",
    "createstereopanner", "createwaveshaper", "decodeaudiodata",
    "resume", "suspend", "close", "getoutputtimestamp",
    # Modern AudioContext additions
    "audioworklet",
})

_CANVAS_ELT_METHODS = frozenset({
    "getcontext", "todataurl", "toblob", "capturestream",
    "transfercontroltoOffscreen", "transfercontroltooffscreen",
    "addeventlistener", "removeeventlistener", "dispatchevent",
    "focus", "blur", "click", "getboundingclientrect",
    "queryselector", "queryselectorall",
    # Generic Element methods worth keeping
    "appendchild", "removechild", "setattribute", "getattribute",
    "remove", "contains",
})

# Map of (lowered receiver name) -> (allowlist set, friendly label).
_RECEIVER_TYPES: dict[str, tuple[frozenset[str], str]] = {}
for r in _CANVAS2D_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS2D_METHODS, "CanvasRenderingContext2D")
for r in _AUDIOCTX_RECEIVERS:
    _RECEIVER_TYPES[r] = (_AUDIOCTX_METHODS, "AudioContext")
for r in _CANVAS_ELT_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS_ELT_METHODS, "HTMLCanvasElement")

# Match `<word>.<word>(` in JS source. The receiver word must be on the
# left of the dot, the method on the right, with a `(` immediately after
# (i.e. it's CALLED, not just read). Greedy-tolerant of whitespace
# between tokens.
_METHOD_CALL_RE = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\("
)


def _check_api_allowlist(js: str) -> list[tuple[str, str, str]]:
    """Find unknown-method calls on known-receiver variable names.

    Returns a list of (receiver_var, method, type_label) tuples — one
    per distinct hallucination found. Comments and string literals are
    stripped first via _strip_js_noise. Caller decides severity.
    """
    stripped = _strip_js_noise(js)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for m in _METHOD_CALL_RE.finditer(stripped):
        recv, method = m.group(1), m.group(2)
        recv_key = recv.lower()
        if recv_key not in _RECEIVER_TYPES:
            continue
        allowlist, type_label = _RECEIVER_TYPES[recv_key]
        if method.lower() in allowlist:
            continue
        key = (recv, method)
        if key in seen:
            continue
        seen.add(key)
        out.append((recv, method, type_label))
    return out


# Used for cheap brace-balance checks. Strips line + block comments and
# string literals so a `for (;;)` inside a comment doesn't trip us.
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'        # double-quoted
    r"|'(?:[^'\\]|\\.)*'"       # single-quoted
    r"|`(?:[^`\\]|\\.)*`",      # template literal (good-enough)
    re.DOTALL,
)
_SCRIPT_BLOCK_RE = re.compile(
    r"<script\b([^>]*)>(.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_js_noise(js: str) -> str:
    """Remove comments and string literals so brace-counting is reliable."""
    js = _BLOCK_COMMENT_RE.sub("", js)
    js = _LINE_COMMENT_RE.sub("", js)
    js = _STRING_LITERAL_RE.sub("''", js)
    return js


def _bracket_imbalance(js: str) -> dict[str, int]:
    """Return |open - close| count per bracket type after stripping strings
    and comments. Zero = balanced.
    """
    stripped = _strip_js_noise(js)
    return {
        "{}": stripped.count("{") - stripped.count("}"),
        "()": stripped.count("(") - stripped.count(")"),
        "[]": stripped.count("[") - stripped.count("]"),
    }


def run_micro_probes(html: str) -> dict[str, Any]:
    """Pre-Chromium structural sanity check.

    Report shape:
      ok:        bool          - True if no errors (warnings allowed).
      errors:    list[str]     - structurally-broken; Chromium will fail.
      warnings:  list[str]     - suspicious but maybe ok; Chromium continues.
      stats:     dict          - small numeric snapshot of what we measured.

    The agent uses this between materialize and Chromium: an `ok=False`
    report skips the browser round-trip and feeds errors back to the
    model on the next turn.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "size_bytes": len(html or ""),
    }

    if not html or len(html) < 200:
        errors.append(
            f"file is essentially empty ({len(html or '')} bytes) — likely "
            "the patch left the document near-empty or the model produced "
            "no usable content."
        )
        return {"ok": False, "errors": errors, "warnings": warnings, "stats": stats}

    low = html.lower()

    # --- structural completeness ---------------------------------------
    if "<!doctype" not in low and "<html" not in low:
        errors.append(
            "no <!DOCTYPE> or <html> root tag — the file does not look "
            "like an HTML document."
        )
    if "<html" in low and "</html" not in low:
        errors.append(
            "<html> opened but never closed — likely the stream was "
            "truncated. Re-emit the rest of the file (or a full <html_file> "
            "rewrite if patches can't recover)."
        )
    if "<body" in low and "</body" not in low:
        errors.append(
            "<body> opened but never closed — truncation indicator. "
            "Close the </body> and </html> tags."
        )

    # --- script presence -----------------------------------------------
    scripts = _SCRIPT_BLOCK_RE.findall(html)
    has_inline_handlers = bool(re.search(r"\bon[a-z]+\s*=", html, re.IGNORECASE))
    n_inline = sum(1 for (_attrs, body) in scripts if body.strip())
    n_external = sum(1 for (attrs, body) in scripts if "src=" in attrs.lower())
    stats["scripts_inline"] = n_inline
    stats["scripts_external"] = n_external
    stats["inline_event_handlers"] = has_inline_handlers

    if n_inline == 0 and n_external == 0 and not has_inline_handlers:
        errors.append(
            "no <script> blocks (inline or external) and no inline event "
            "handlers — the file has no game logic. Add a <script> with "
            "the game implementation."
        )

    # --- bracket balance per inline script -----------------------------
    # Per pi-mono's prescriptive-error pattern: tell the model EXACTLY
    # which kind is unbalanced and by how much.
    total_imbalance = {"{}": 0, "()": 0, "[]": 0}
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        imb = _bracket_imbalance(body)
        for k, v in imb.items():
            total_imbalance[k] += v
    stats["bracket_imbalance"] = total_imbalance

    for kind, delta in total_imbalance.items():
        # Heuristic strips can over-count by ~1 in pathological corner
        # cases (regex literals look like division, etc). Allow ±1 as
        # WARNING; ±2 as ERROR.
        if abs(delta) >= 2:
            sign = "extra opening" if delta > 0 else "extra closing"
            errors.append(
                f"unbalanced {kind} brackets in <script>: {sign} "
                f"{kind[0] if delta>0 else kind[1]} by {abs(delta)} "
                "(after stripping comments and string literals). "
                "Almost certainly a syntax error — close the missing "
                "brace before re-running."
            )
        elif delta != 0:
            warnings.append(
                f"possibly unbalanced {kind} in <script> by {delta} "
                "(could be a regex-literal false-positive — Chromium "
                "will confirm)."
            )

    # --- API allowlist (hallucinated method calls) --------------------
    # Scan inline scripts for `<known-receiver>.<method>(` patterns where
    # the method is not on the canonical allowlist for that receiver
    # type. Reported as warnings, not errors — Chromium has the final
    # word and the allowlist is intentionally incomplete (better to miss
    # a hallucination than to flag a real method we forgot).
    api_warnings: list[str] = []
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        for recv, method, type_label in _check_api_allowlist(body):
            api_warnings.append(
                f"`{recv}.{method}(...)` called but '{method}' is NOT a "
                f"known method on {type_label}. If `{recv}` is a "
                f"{type_label}, this is a hallucination — pick the "
                "real method name (Chromium will throw a TypeError "
                "otherwise). If `{recv}` is your own object, this "
                "warning is a false positive; ignore."
            )
    if api_warnings:
        warnings.extend(api_warnings)
        stats["api_hallucinations"] = len(api_warnings)

    # --- elision sentinels ---------------------------------------------
    # Models occasionally slip "// ... rest of code unchanged ..." into
    # a patch even after we tell them not to. Catch it here so we don't
    # ship a half-implemented file.
    elision_markers = [
        "// ... rest unchanged",
        "// ... rest of code",
        "// rest of",
        "// (existing code)",
        "/* existing code */",
        "<- leave original",
    ]
    for m in elision_markers:
        if m.lower() in low:
            errors.append(
                f"elision marker found in source: {m!r} — the file is "
                "incomplete. Re-emit the patch with the EXACT lines, no "
                "shortcuts."
            )
            break

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def format_micro_probes_for_model(report: dict[str, Any]) -> str:
    """Compact text version of a micro-probe report for the user-turn
    feedback message. Mirrors `format_report_for_model`'s shape so the
    model sees a familiar structure.
    """
    lines = ["MICRO-PROBE PRE-FLIGHT (structural sanity, no browser yet):"]
    lines.append(f"OK: {report.get('ok', False)}")
    stats = report.get("stats") or {}
    if stats:
        bits = [f"size={stats.get('size_bytes', 0)}b"]
        if "scripts_inline" in stats:
            bits.append(
                f"scripts(inline/external)={stats['scripts_inline']}/"
                f"{stats.get('scripts_external', 0)}"
            )
        if "bracket_imbalance" in stats:
            imb = stats["bracket_imbalance"]
            non_zero = {k: v for k, v in imb.items() if v != 0}
            if non_zero:
                bits.append(f"bracket_imbalance={non_zero}")
        lines.append("Stats: " + ", ".join(bits))
    if report.get("errors"):
        lines.append("ERRORS (must fix):")
        for e in report["errors"]:
            lines.append(f"  - {e}")
    if report.get("warnings"):
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def test_html_file(path: str | Path, run_seconds: float = 3.0) -> dict[str, Any]:
    """Run an HTML file in headless Chromium and return a small report dict.

    Report shape (always these keys, so the agent can rely on it):
      ok:          bool   - True if zero errors AND zero exceptions
      errors:      list[str]  - console.error lines + page errors
      warnings:    list[str]  - console.warn lines
      logs:        list[str]  - first few console.log lines (debug aid only)
      title:       str
      canvas:      dict | None  - {width, height, raf_ran: bool} if a <canvas> exists
      body_chars:  int    - length of body innerText (rough "is anything there?" check)
      body_sample: str    - first chars of body innerText
    """
    path = Path(path).resolve()
    file_url = f"file://{path}"

    # Buffers we fill from event handlers.
    errors: list[str] = []
    warnings: list[str] = []
    logs: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        # New context per run so localStorage / cookies don't leak between iterations.
        context = browser.new_context(viewport={"width": 800, "height": 600})
        page = context.new_page()

        # --- console capture ---
        # Playwright fires "console" for log/info/warn/error and "pageerror" for
        # uncaught JS exceptions. We split them into our three buckets.
        def on_console(msg):
            text = _truncate(msg.text, _MAX_MSG_LEN)
            t = msg.type
            if t == "error":
                errors.append(text)
            elif t == "warning":
                warnings.append(text)
            else:
                logs.append(text)

        def on_pageerror(exc):
            # `exc` stringifies to the JS error message + stack head.
            errors.append(_truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN))

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)

        # --- pre-load instrumentation ---
        # add_init_script runs in EVERY frame BEFORE any of the page's own
        # scripts. The actual JS lives in _INSTRUMENTATION_JS at module level
        # so LiveBrowser (async) can use the SAME hooks.
        page.add_init_script(_INSTRUMENTATION_JS)

        # --- load ---
        try:
            page.goto(file_url, wait_until="load", timeout=10_000)
        except Exception as e:
            # Page failed to even load - return early with a clear error.
            browser.close()
            return {
                "ok": False,
                "errors": [f"PAGE FAILED TO LOAD: {e}"],
                "warnings": [],
                "logs": [],
                "title": "",
                "canvas": None,
                "body_chars": 0,
                "body_sample": "",
            }

        # NOTE: RAF hook used to be injected here via page.evaluate. It was
        # moved into add_init_script above so it runs BEFORE the game's own
        # scripts; otherwise the game grabs the original RAF before we wrap it.

        # Let the game animate for a few seconds.
        time.sleep(run_seconds)

        # --- collect post-run facts ---
        title = page.title() or ""
        try:
            body_text = page.evaluate("document.body ? document.body.innerText : ''") or ""
        except Exception:
            body_text = ""

        canvas_info = None
        try:
            # Heuristic blank-canvas detector + RAF flag readout. The JS lives
            # in _CANVAS_PROBE_JS at module level so LiveBrowser uses the same.
            canvas_info = page.evaluate(_CANVAS_PROBE_JS)
        except Exception:
            canvas_info = None

        # Pull the listener counts injected by add_init_script. Tells us whether
        # the game actually wired up input - a real game has at least 1 keyboard
        # or mouse listener somewhere on document/window.
        listener_info = {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}
        try:
            counts = page.evaluate("window.__listenerCount || null") or {}
            for k in ("document", "window", "body", "other"):
                listener_info[k] = int(counts.get(k, 0) or 0)
            listener_info["total"] = sum(listener_info[k] for k in ("document", "window", "body", "other"))
        except Exception:
            pass

        browser.close()

    return _build_report(errors, warnings, logs, title, canvas_info, listener_info, body_text)


def score_test_report(report: dict[str, Any]) -> float:
    """Continuous quality score in [0, 100] for a test report.

    Used by best-of-N candidate selection (and `tune why` postmortem
    rendering) so partial-credit candidates win over completely-broken
    ones, instead of a binary pass/fail. Designed to be monotone in
    "how close the candidate is to passing", so picking the max gives a
    sensible candidate even when none pass.

    Scoring (max 100):
      * test passes outright           → 100
      * else, weighted demerits / bonuses on:
        - errors (real JS exceptions)  : -8 per, capped at -40
        - soft_warnings (heuristics)   : -5 per, capped at -20
        - frozen_canvas True           : -10
        - canvas blank True            : -10
        - listener_total > 0           : +5
        - raf_ran True                 : +5
        - input_test passes            : +10
    """
    if report is None:
        return 0.0
    if report.get("ok") is True:
        return 100.0
    s = 50.0
    n_err = len(report.get("errors") or [])
    s -= min(n_err, 5) * 8
    n_iss = len(report.get("soft_warnings") or [])
    s -= min(n_iss, 4) * 5
    if report.get("frozen_canvas") is True:
        s -= 10
    canv = report.get("canvas") or {}
    if canv.get("blank") is True:
        s -= 10
    li = report.get("input_listeners") or {}
    if int(li.get("total") or 0) > 0:
        s += 5
    if canv.get("raf_ran") is True:
        s += 5
    it = report.get("input_test") or {}
    if it.get("ran") and it.get("any_change") is True:
        s += 10
    # Model-proposed acceptance probes (when present): each contributes a
    # small bonus, so a candidate that passes its own acceptance criteria
    # outranks one that doesn't.
    probes = report.get("probes") or []
    if probes:
        n_pass = sum(1 for p in probes if p.get("ok"))
        s += min(15, n_pass * 3)
    return max(0.0, min(100.0, s))


def format_report_for_model(report: dict[str, Any]) -> str:
    """Turn the report dict into the SHORT plain-text block we feed to the model.

    Keeping this terse is intentional - per the project's debug-minimum rule we
    only send what the model needs to fix the next issue.
    """
    lines = []
    lines.append(f"OK: {report['ok']}")
    lines.append(f"Title: {report['title']!r}")
    if report["canvas"] is not None:
        c = report["canvas"]
        blank_str = "unknown" if c.get("blank") is None else str(c["blank"])
        lines.append(
            f"Canvas: {c['width']}x{c['height']}, RAF ran: {c['raf_ran']}, "
            f"blank: {blank_str}"
        )
    else:
        lines.append("Canvas: none")
    li = report.get("input_listeners", {})
    lines.append(
        f"Input listeners: total={li.get('total', 0)} "
        f"(doc={li.get('document', 0)}, win={li.get('window', 0)}, "
        f"body={li.get('body', 0)}, other={li.get('other', 0)})"
    )
    # New: results of the auto-input smoke test. Tells the model whether
    # the game actually responded to keys, not just whether listeners exist.
    it = report.get("input_test") or {}
    if it.get("ran"):
        if it.get("any_change"):
            lines.append(
                f"Input test: PASS — pressed keys, canvas changed on "
                f"{it.get('first_responsive_key')!r}."
            )
        else:
            lines.append(
                f"Input test: FAIL — pressed {it.get('keys_tried', [])} "
                "and canvas pixels never changed."
            )
    # New: did the canvas freeze (drawing same frame) between two samples?
    fz = report.get("frozen_canvas")
    if fz is True:
        lines.append("Frozen canvas: YES (same pixels at t=half and t=full)")
    elif fz is False:
        lines.append("Frozen canvas: no (pixels changed during run)")
    lines.append(f"Body text length: {report['body_chars']} chars")
    if report["body_sample"]:
        lines.append(f"Body sample: {report['body_sample']!r}")
    if report["errors"]:
        lines.append("ERRORS (must fix):")
        for e in report["errors"]:
            lines.append(f"  - {e}")
    if report.get("soft_warnings"):
        # Heuristic broken-but-no-exception findings. Listed as ISSUES so the
        # model treats them with the same urgency as real errors.
        lines.append("ISSUES (must fix):")
        for s in report["soft_warnings"]:
            lines.append(f"  - {s}")
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report.get("probes"):
        n_pass = sum(1 for p in report["probes"] if p.get("ok"))
        n = len(report["probes"])
        lines.append(f"Acceptance probes: {n_pass}/{n} pass")
        for p in report["probes"]:
            tag = "ok " if p.get("ok") else "FAIL"
            err = f"  ({p['err']})" if p.get("err") else ""
            lines.append(f"  {tag} {p.get('name','probe')}: {p.get('expr','')[:80]}{err}")
    if report["logs"] and not report["errors"]:
        # Only show logs when there are no errors - otherwise the model focuses
        # on the wrong thing.
        lines.append("Console logs (info only):")
        for l in report["logs"][:4]:
            lines.append(f"  - {l}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LiveBrowser: VISIBLE, persistent browser used by the TUI.
#
# Different from test_html_file in three ways:
#   1. headless=False - you actually SEE the game in a real Chromium window.
#   2. The browser stays open across iterations (the TUI keeps a single
#      LiveBrowser instance for its whole session). Reload between iterations
#      is fast and visually shows the game updating.
#   3. Async API (so it cooperates with Textual's asyncio event loop instead
#      of blocking the TUI). Returns the SAME report shape as test_html_file.
#
# We use playwright.async_api here. Mixing sync_playwright and async_playwright
# in the same process is fine as long as you don't use both AT THE SAME TIME -
# the TUI uses async only, the CLI uses sync only.
# ---------------------------------------------------------------------------

# Imported lazily inside __init__ so people running the CLI never pay the
# async-playwright import cost (and so a missing async install doesn't break
# the headless path).


class LiveBrowser:
    """Persistent visible Chromium for the TUI. Async, single page, reusable.

    Usage:
        lb = LiveBrowser()
        await lb.start()
        report = await lb.load_and_test("games/game.html")
        ...
        await lb.close()
    """

    def __init__(
        self,
        viewport: tuple[int, int] = (800, 600),
        run_seconds: float = 3.0,
        headless: bool = False,
    ):
        self._viewport = viewport
        self._run_seconds = run_seconds
        self._headless = headless
        # Buffers reset on every load_and_test call.
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._logs: list[str] = []
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    async def start(self) -> None:
        """Launch the browser. Call once before load_and_test."""
        # Lazy import: keeps the sync CLI lightweight.
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        # headless=False is the whole point - the user wants to SEE the game.
        # --window-position keeps the window from landing on top of the terminal
        # by default (X/Wayland may ignore this; tile-WM users will arrange it
        # themselves anyway).
        # headless=False is the normal interactive case (TUI lets the user
        # SEE the game). headless=True is for the test driver.
        launch_args: list[str] = []
        if not self._headless:
            launch_args = [
                f"--window-position=850,50",
                f"--window-size={self._viewport[0]},{self._viewport[1]}",
            ]
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=launch_args,
        )
        self._context = await self._browser.new_context(
            viewport={"width": self._viewport[0], "height": self._viewport[1]}
        )
        # Init script runs on every navigation - critical for the listener
        # counter to reset between iterations.
        await self._context.add_init_script(_INSTRUMENTATION_JS)
        self._page = await self._context.new_page()

        # Wire console + pageerror handlers ONCE; the buffers themselves are
        # reset per-test (see load_and_test).
        self._page.on("console", self._on_console)
        self._page.on("pageerror", self._on_pageerror)

    def _on_console(self, msg) -> None:
        text = _truncate(msg.text, _MAX_MSG_LEN)
        t = msg.type
        if t == "error":
            self._errors.append(text)
        elif t == "warning":
            self._warnings.append(text)
        else:
            self._logs.append(text)

    def _on_pageerror(self, exc) -> None:
        self._errors.append(_truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN))

    async def load_and_test(
        self,
        path: str | Path,
        screenshot_path: str | Path | None = None,
        *,
        probes: list[dict] | None = None,
        screenshot_before_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Navigate to the file, let it run, return the report.

        New (this version) compared to the old text-only test:
          - Takes a screenshot at the end (saved to screenshot_path if given).
          - Detects FROZEN canvas by sampling pixels twice (~1s apart).
          - Runs an input smoke test (arrow keys, WASD, space) and checks
            whether any of them produced a canvas pixel change. If none did,
            the game probably ignores input.

        All of that ends up as additional report fields and soft_warnings the
        model treats with the same urgency as crashes.
        """
        import asyncio

        if self._page is None:
            raise RuntimeError("LiveBrowser.start() must be awaited before load_and_test()")

        # Reset per-test buffers. The handlers stay attached.
        self._errors.clear()
        self._warnings.clear()
        self._logs.clear()

        path = Path(path).resolve()
        file_url = f"file://{path}"
        try:
            await self._page.goto(file_url, wait_until="load", timeout=10_000)
        except Exception as e:
            return _build_report(
                [f"PAGE FAILED TO LOAD: {e}"], [], [], "", None,
                {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}, "",
            )

        # Sleep half the budget; sample canvas; sleep rest; sample again.
        # If both samples are byte-identical the game is FROZEN even if
        # requestAnimationFrame fired (it's drawing the same frame).
        # We also take a 32x32 hash at each moment — the boolean equality of
        # the two hashes is what drives the FROZEN heuristic (it catches
        # slowly-moving content the 9-pixel probe misses).
        half = max(self._run_seconds / 2.0, 0.5)
        await asyncio.sleep(half)
        canvas_first = await self._safe_eval(_CANVAS_PROBE_JS)
        hash_first = await self._safe_eval(_CANVAS_HASH_JS)
        # Optional "before-input" screenshot — captures the t=startup
        # state so a VLM can later see motion as a before/after pair.
        screenshot_before_saved: str | None = None
        if screenshot_before_path is not None:
            try:
                bp = Path(screenshot_before_path)
                bp.parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=str(bp), full_page=False)
                screenshot_before_saved = str(bp)
            except Exception:
                screenshot_before_saved = None
        await asyncio.sleep(half)
        canvas_info = await self._safe_eval(_CANVAS_PROBE_JS)
        hash_last = await self._safe_eval(_CANVAS_HASH_JS)

        # ---- input smoke test ---------------------------------------------
        # Most small-model bugs we miss are "controls don't work". Fire a few
        # standard inputs and check if pixels change. Captured pre/post
        # snapshots are compared via a key-set hash from the same probe.
        input_test = await self._input_smoke_test()

        # ---- model-proposed probes ----------------------------------------
        # Agent emits <probes> in Phase A — JSON list of {name, expr} where
        # expr is a JS expression that should evaluate truthy on the running
        # game. We run each in the page context. Per-probe results join the
        # report so the model sees its own assertions checked.
        probe_results: list[dict[str, Any]] = []
        if probes:
            for p in probes:
                pname = str(p.get("name") or "probe")[:60]
                pexpr = str(p.get("expr") or "true")[:600]
                ok, err = await self._run_probe(pexpr)
                probe_results.append({
                    "name": pname, "expr": pexpr[:200], "ok": ok, "err": err,
                })

        title = (await self._page.title()) or ""
        try:
            body_text = await self._page.evaluate(
                "document.body ? document.body.innerText : ''"
            ) or ""
        except Exception:
            body_text = ""

        listener_info = {"document": 0, "window": 0, "body": 0, "other": 0, "total": 0}
        try:
            counts = await self._page.evaluate("window.__listenerCount || null") or {}
            for k in ("document", "window", "body", "other"):
                listener_info[k] = int(counts.get(k, 0) or 0)
            listener_info["total"] = sum(listener_info[k] for k in ("document", "window", "body", "other"))
        except Exception:
            pass

        # ---- screenshot (always taken; saved if path given) ---------------
        screenshot_saved: str | None = None
        if screenshot_path is not None:
            try:
                p = Path(screenshot_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                await self._page.screenshot(path=str(p), full_page=False)
                screenshot_saved = str(p)
            except Exception:
                screenshot_saved = None

        # ---- detect frozen canvas: 32x32 hash identical at t=half and t=full
        # The hash covers the whole playfield, so even one pixel of motion
        # somewhere flips the result. Only flag FROZEN when we have content
        # (not blank) and the loop is actually running (RAF fired).
        frozen = None
        if hash_first is not None and hash_last is not None and canvas_info:
            same = (hash_first == hash_last)
            if same and canvas_info.get("blank") is False and canvas_info.get("raf_ran"):
                frozen = True
            else:
                frozen = False

        report = _build_report(
            list(self._errors), list(self._warnings), list(self._logs),
            title, canvas_info, listener_info, body_text,
        )
        # Attach the new fields. The model never sees raw bytes - just paths
        # and small booleans / counts via format_report_for_model.
        report["screenshot"] = screenshot_saved
        report["screenshot_before"] = screenshot_before_saved
        report["frozen_canvas"] = frozen
        report["input_test"] = input_test
        report["probes"] = probe_results

        # Promote the new findings into soft_warnings so they get the same
        # "must fix" treatment as RAF/blank in the report formatter.
        # Important nuance: FROZEN-without-input-change is bad; FROZEN-but-
        # input-causes-change is just "input-driven, not auto-animated" which
        # is normal for many games (turn-based, click counter, tic-tac-toe).
        # We only flag FROZEN when input ALSO doesn't move pixels.
        input_responsive = bool(
            input_test.get("ran") and input_test.get("any_change") is True
        )
        input_dead = bool(
            input_test.get("ran") and input_test.get("any_change") is False
        )
        # Blank canvas + dead keyboard = real failure. Blank with
        # responsive input (drawing canvas after a stroke) is fine.
        if (canvas_info and canvas_info.get("blank") is True
                and input_dead and not input_responsive):
            report["soft_warnings"].append(
                f"HEURISTIC: canvas pixels are uniform AND keyboard input "
                f"didn't change anything either — the game is not "
                f"rendering / not interactive."
            )
        if frozen is True and not input_responsive:
            report["soft_warnings"].append(
                "HEURISTIC: canvas drew SOMETHING but did not change between two "
                "samples 1s apart AND no key press changed anything either - "
                "the game is frozen / stuck on one frame."
            )
        if input_dead:
            keys_str = ", ".join(input_test.get("keys_tried", []))
            # Don't double-fire if the page only has a button (no canvas
            # input expected). We check for an interactive element below.
            has_clickable = await self._safe_eval(
                "document.querySelectorAll('button, [onclick]').length > 0"
            )
            if not has_clickable:
                report["soft_warnings"].append(
                    f"HEURISTIC: pressed {keys_str} - canvas pixels never changed. "
                    "Controls are not wired up (or input handler is broken)."
                )
            else:
                report["warnings"].append(
                    "Note: keyboard test produced no canvas change, but the "
                    "page has clickable elements; treating as DOM-driven."
                )
        # ok must reflect the fresh soft_warnings count after we appended.
        report["ok"] = len(report["errors"]) == 0 and len(report["soft_warnings"]) == 0
        return report

    async def _safe_eval(self, js: str):
        """page.evaluate that swallows errors and returns None on failure."""
        try:
            return await self._page.evaluate(js)
        except Exception:
            return None

    async def _run_probe(self, expr: str) -> tuple[bool, str]:
        """Run one model-proposed probe expression in the page; return
        (ok, err). Wraps in (() => Boolean(EXPR))() so the model can write
        either a boolean expression or a IIFE.
        """
        wrapped = (
            "(() => { try { return Boolean(" + expr + "); } "
            "catch (e) { return { __probe_err: String(e && e.message || e) }; } })()"
        )
        try:
            res = await self._page.evaluate(wrapped)
            if isinstance(res, dict) and res.get("__probe_err"):
                return False, str(res["__probe_err"])[:200]
            return bool(res), ""
        except Exception as e:
            return False, str(e)[:200]

    async def _input_smoke_test(self) -> dict[str, Any]:
        """Hold each test key for a few frames; report whether the canvas changed.

        Two big differences vs the original 9-pixel version:

          1. We HOLD each key (down/up) for ~250ms instead of press(). Held-key
             games like thrust-controlled ships only move while a key is
             actively down; press() releases instantly and the next sample
             catches the post-release frame.

          2. We use a ~32x32 downsampled hash of the FULL canvas instead of
             nine corner pixels, so slowly-moving objects in the middle of the
             playfield register as a change.
        """
        import asyncio

        has_canvas = await self._safe_eval(
            "!!document.querySelector('canvas')"
        )
        if not has_canvas:
            return {"ran": False, "reason": "no canvas"}

        keys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "KeyW", "KeyA", "KeyS", "KeyD"]
        tried: list[str] = []
        any_change = False
        first_responsive_key: str | None = None

        try:
            await self._page.bring_to_front()
            await self._page.evaluate("if (document.body) document.body.focus();")
        except Exception:
            pass

        for k in keys:
            before = await self._safe_eval(_CANVAS_HASH_JS)
            if before is None:
                return {"ran": False, "reason": "canvas not sampleable", "keys_tried": tried}
            try:
                await self._page.keyboard.down(k)
                await asyncio.sleep(0.25)  # hold long enough for thrust to move ship
                after_held = await self._safe_eval(_CANVAS_HASH_JS)
                await self._page.keyboard.up(k)
            except Exception:
                continue
            tried.append(k)
            # Wait one more frame for any post-release tween / momentum.
            await asyncio.sleep(0.05)
            after_release = await self._safe_eval(_CANVAS_HASH_JS)
            if (after_held is not None and after_held != before) or \
               (after_release is not None and after_release != before):
                any_change = True
                first_responsive_key = k
                break

        return {
            "ran": True,
            "any_change": any_change,
            "keys_tried": tried,
            "first_responsive_key": first_responsive_key,
        }

    async def show_status(self, title: str, message: str = "") -> None:
        """Replace whatever the browser is currently displaying with a small
        status page. Used between sessions so a failed new session can't
        masquerade as the previous successful one (the browser otherwise
        keeps the last-loaded game on screen).
        """
        if self._page is None:
            return
        # Inline HTML via data: URL — no temp file needed.
        from html import escape
        from urllib.parse import quote
        body = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{escape(title)}</title>"
            "<style>html,body{margin:0;height:100%;background:#0b1020;"
            "color:#e7ecff;font:16px/1.4 system-ui,sans-serif;}"
            "div{position:fixed;inset:0;display:grid;place-items:center;"
            "text-align:center;padding:20px;}"
            "h1{font-size:22px;color:#79a;margin:0 0 12px;}"
            "p{opacity:.75;max-width:600px;}</style></head>"
            f"<body><div><div><h1>{escape(title)}</h1>"
            f"<p>{escape(message)}</p></div></div></body></html>"
        )
        try:
            await self._page.goto("data:text/html;charset=utf-8," + quote(body),
                                  wait_until="load", timeout=5_000)
        except Exception:
            # Best-effort; never let a status update crash the TUI.
            pass

    async def close(self) -> None:
        """Tear down. Safe to call multiple times."""
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            if self._pw is not None:
                await self._pw.stop()
                self._pw = None
