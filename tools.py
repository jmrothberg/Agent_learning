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
from collections import Counter as _Counter
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


# Cap how much we feed back to the model. Smaller is better for small models.
_MAX_MSGS = 12          # at most this many console lines forwarded
_MAX_MSG_LEN = 240      # truncate each line to this many chars
_MAX_BODY_TEXT = 200    # tiny snippet of body text


def screenshot_delta(prev_png: bytes | None, curr_png: bytes | None) -> float | None:
    """Mean per-pixel RGB delta between two PNGs, normalized to [0, 1].

    Returns None when either input is missing or PIL is unavailable.
    Used by the loop's visual-regression detector: a high delta paired
    with a "no progress" verdict from the vision judge implies the
    iter visibly changed the canvas without making it better — i.e.,
    a regression. The threshold lives in the caller.

    Implementation note: both screenshots are resized to a small fixed
    resolution before differencing so the comparison is fast and
    independent of the original game's canvas size. We use 128×128
    because that's enough to capture overall composition while keeping
    the diff under 1 ms on the harness machine.
    """
    if not prev_png or not curr_png:
        return None
    try:
        from io import BytesIO

        from PIL import Image
    except Exception:
        return None
    try:
        a = Image.open(BytesIO(prev_png)).convert("RGB").resize((128, 128))
        b = Image.open(BytesIO(curr_png)).convert("RGB").resize((128, 128))
    except Exception:
        return None
    pa = a.tobytes()
    pb = b.tobytes()
    if len(pa) != len(pb) or not pa:
        return None
    total = 0
    for x, y in zip(pa, pb):
        d = x - y
        total += d if d >= 0 else -d
    return total / (len(pa) * 255.0)


# Stop-Losing-To-OneShot todo #2 — criteria coverage helper.
# Stop-words drop generic prose ("the player should be able to move")
# and keep meaningful tokens ("rotate", "thrust", "shoot"). Genre-free:
# we never compile a list of subject matter; we just remove function
# words and short tokens. Matches a probe to a criterion when at least
# one meaningful word appears in BOTH (the probe's `name` or `expr` and
# the criterion line). Conservative: a missing match flags the line,
# which biases toward asking the model to add probes.
_COVERAGE_STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "from", "into", "onto", "for",
    "are", "was", "were", "have", "has", "had", "been", "being", "must",
    "should", "could", "would", "will", "shall", "may", "might", "can",
    "do", "does", "did", "done", "is", "be", "an", "at", "by", "on",
    "in", "of", "to", "or", "no", "if", "as", "it", "its", "after",
    "when", "while", "than", "then", "but", "all", "any", "some",
    "press", "click", "type", "key", "keys", "test", "edge", "stress",
    "basic", "case", "value", "values", "true", "false", "render",
    "renders", "rendered", "show", "shown", "user", "game",
})
# Underscores are word boundaries (so 'ship_rotates_left' contributes
# 'ship', 'rotates', 'left' to the bag of words rather than a single
# compound token that almost never matches a criterion line).
_COVERAGE_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def _coverage_words(text: str) -> set[str]:
    """Return lowercased meaningful tokens (>=4 chars, non-stop)."""
    if not text:
        return set()
    return {
        w for w in (m.group(0).lower() for m in _COVERAGE_WORD_RE.finditer(text))
        if len(w) >= 4 and w not in _COVERAGE_STOPWORDS
    }


# Words in <criteria>, <plan>, or the goal text that signal the game
# expects keyboard / pointer input. When ANY of these appear, an
# input_test FAIL should be a hard signal (controls not wired), NOT
# auto-rationalized as "DOM-driven". DK trace 20260514_104131 shipped
# a game whose <criteria> said "ArrowRight moves the player right" and
# "climb ladders with ArrowUp/ArrowDown" while keyboard input was
# silently broken; the harness still reported `ok=True` because the
# page had a clickable restart button, which triggered the
# "treating as DOM-driven" carve-out.
#
# Model-agnostic, game-agnostic: describes input MODALITY, not subject
# matter or genre. Calculators / todo lists / drawing apps avoid these
# words and keep the existing DOM-driven carve-out.
_GAME_CONTROL_KEYWORDS = frozenset({
    "arrow", "arrows",
    "arrowleft", "arrowright", "arrowup", "arrowdown",
    "wasd", "key", "keys", "keyboard", "keypress",
    "press", "pressed", "control", "controls", "controller",
    "joystick", "gamepad",
    "move", "moves", "movement", "moving",
    "walk", "walks", "walking",
    "run", "runs", "running",
    "climb", "climbs", "climbing",
    "jump", "jumps", "jumping",
    "shoot", "shoots", "shooting",
    "fire", "fires", "firing",
    "spacebar", "space",
    "dpad", "stick", "thumbstick",
})


def expects_game_controls(*texts: str) -> bool:
    """True when any of the supplied texts mention keyboard / pointer
    controls. Used by load_and_test to decide whether an input_test
    failure should be a hard signal or a soft DOM-driven warning.

    Tokenizes on word boundaries so substrings inside other words
    (e.g. "monkey" matching "key") don't false-positive. Case-
    insensitive.
    """
    pattern = re.compile(r"[a-zA-Z]+")
    for text in texts:
        if not text:
            continue
        for m in pattern.finditer(text):
            if m.group(0).lower() in _GAME_CONTROL_KEYWORDS:
                return True
    return False


def _slugify_criterion(text: str) -> str:
    """Compact identifier-safe slug for a criterion line, used as the
    suffix of a synthetic coverage-gap probe name. Keeps the slug short
    (≤32 chars) and limited to lowercase ASCII / digit / underscore so
    downstream regex matching (the agent's Phase B re-parse) can rely
    on it. Drops the leading category label `Basic:` / `Edge:` /
    `Stress:` if present so the slug describes the criterion itself.
    """
    s = (text or "").strip()
    # Strip leading category label like "Basic:", "Edge:", "Stress:".
    if ":" in s:
        head, rest = s.split(":", 1)
        if 2 <= len(head.strip()) <= 12 and head.strip().isalpha():
            s = rest.strip()
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
        if len(out) >= 32:
            break
    return ("".join(out).strip("_")) or "criterion"


def _criteria_coverage_gaps(criteria_text: str, probes: list[dict]) -> list[str]:
    """For each non-empty criterion line, return the line if no probe's
    name+expr shares any meaningful word with it. Generic, behavioral,
    no genre awareness. Only used when both criteria and probes are
    present — empty inputs short-circuit to no gaps reported."""
    if not criteria_text or not probes:
        return []
    probe_words: set[str] = set()
    for p in probes:
        probe_words |= _coverage_words(str(p.get("name") or ""))
        probe_words |= _coverage_words(str(p.get("expr") or ""))
    if not probe_words:
        return []
    uncovered: list[str] = []
    for raw in criteria_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        words = _coverage_words(line)
        if not words:
            continue
        overlap = words & probe_words
        # Existence-only probes ("ship_exists", "asteroids_exist")
        # typically share exactly one noun with a behavioral criterion
        # ("ship rotates with arrows; asteroids drift"). Require >=2
        # overlapping tokens for criteria with >=3 meaningful words so
        # an existence probe can't satisfy a multi-verb claim by accident.
        # Terse criteria (1-2 tokens) keep the lenient >=1 rule.
        threshold = 2 if len(words) >= 3 else 1
        if len(overlap) < threshold:
            uncovered.append(line[:140])
    return uncovered


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

// Audio-events shim — records every Audio playback start so probes can
// assert "an SFX fired in response to <event>". We wrap both the plain
// HTMLAudioElement.play() path AND the WebAudio source.start() path
// because games use either. The record shape is intentionally flat
// — {t: epoch-ms, src: source identifier} — so probes can filter with
// trivial JS without parsing nested events.
window.__audioEvents = [];
const _origAudioPlay = HTMLAudioElement && HTMLAudioElement.prototype && HTMLAudioElement.prototype.play;
if (_origAudioPlay) {
    HTMLAudioElement.prototype.play = function(...rest) {
        try {
            window.__audioEvents.push({
                t: Date.now(),
                src: (this && this.src) ? this.src : "<inline>",
                kind: "html_audio",
            });
        } catch (e) { /* ignore */ }
        return _origAudioPlay.apply(this, rest);
    };
}
if (typeof AudioBufferSourceNode !== "undefined" && AudioBufferSourceNode.prototype) {
    const _origStart = AudioBufferSourceNode.prototype.start;
    if (_origStart) {
        AudioBufferSourceNode.prototype.start = function(...rest) {
            try {
                window.__audioEvents.push({
                    t: Date.now(),
                    src: "<webaudio_buffer>",
                    kind: "webaudio",
                });
            } catch (e) { /* ignore */ }
            return _origStart.apply(this, rest);
        };
    }
}
if (typeof OscillatorNode !== "undefined" && OscillatorNode.prototype) {
    const _origOscStart = OscillatorNode.prototype.start;
    if (_origOscStart) {
        OscillatorNode.prototype.start = function(...rest) {
            try {
                window.__audioEvents.push({
                    t: Date.now(),
                    src: "<oscillator>",
                    kind: "oscillator",
                });
            } catch (e) { /* ignore */ }
            return _origOscStart.apply(this, rest);
        };
    }
}
"""

# Downsampled canvas hash. We sample a 32x32 grid spread across the canvas
# and concatenate the RGBA bytes. Cheap (~1KB string) but catches per-frame
# motion anywhere on the playfield — the older 9-pixel fingerprint missed
# slowly-moving objects in the middle. Used by the input smoke test AND the
# frozen-canvas check.
_CANVAS_HASH_JS = """
() => {
    const c = document.querySelector('canvas');
    if (!c || c.width < 4 || c.height < 4) return null;
    const N = 32;
    const w = c.width, h = c.height;
    // 2D context: getImageData per sample.
    const ctx2d = c.getContext('2d', { willReadFrequently: true });
    if (ctx2d) {
        try {
            const out = [];
            for (let iy = 0; iy < N; iy++) {
                const y = ((iy + 0.5) * h / N) | 0;
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const d = ctx2d.getImageData(x, y, 1, 1).data;
                    out.push(((d[0] << 16) | (d[1] << 8) | d[2]).toString(36));
                }
            }
            return out.join(',');
        } catch (e) { /* fall through to WebGL */ }
    }
    // WebGL context: read full backbuffer then sample at the same grid.
    // gl.readPixels has to read the whole framebuffer once because the
    // back-buffer is invalidated after a single readPixels call on most
    // drivers. 32x32 samples = 1024 pixels, still tiny.
    const gl = c.getContext('webgl2', { preserveDrawingBuffer: true })
            || c.getContext('webgl', { preserveDrawingBuffer: true });
    if (gl) {
        try {
            const buf = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            const out = [];
            for (let iy = 0; iy < N; iy++) {
                // WebGL Y is flipped relative to canvas Y.
                const y = h - 1 - (((iy + 0.5) * h / N) | 0);
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const i = (y * w + x) * 4;
                    out.push(((buf[i] << 16) | (buf[i+1] << 8) | buf[i+2]).toString(36));
                }
            }
            return out.join(',');
        } catch (e) { return null; }
    }
    return null;
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
        ctx_kind: null,  // "2d" | "webgl2" | "webgl" | null
    };
    if (c.width < 4 || c.height < 4) return out;
    const N = 32;
    const w = c.width, h = c.height;
    // Pass 1: try 2D.
    const ctx2d = c.getContext('2d', { willReadFrequently: true });
    if (ctx2d) {
        out.ctx_kind = "2d";
        try {
            const colors = new Set();
            for (let iy = 0; iy < N; iy++) {
                const y = ((iy + 0.5) * h / N) | 0;
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const d = ctx2d.getImageData(x, y, 1, 1).data;
                    colors.add((d[0] << 16 | d[1] << 8 | d[2]) | 0);
                }
            }
            out.sampled_colors = colors.size;
            out.blank = colors.size <= 1;
            return out;
        } catch (e) { /* fall through */ }
    }
    // Pass 2: WebGL. Reads full backbuffer (preserveDrawingBuffer required
    // to survive past the implicit swap that follows a paint).
    const gl = c.getContext('webgl2', { preserveDrawingBuffer: true })
            || c.getContext('webgl', { preserveDrawingBuffer: true });
    if (gl) {
        out.ctx_kind = gl instanceof WebGL2RenderingContext ? "webgl2" : "webgl";
        try {
            const buf = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            const colors = new Set();
            for (let iy = 0; iy < N; iy++) {
                const y = h - 1 - (((iy + 0.5) * h / N) | 0);
                for (let ix = 0; ix < N; ix++) {
                    const x = ((ix + 0.5) * w / N) | 0;
                    const i = (y * w + x) * 4;
                    colors.add((buf[i] << 16 | buf[i+1] << 8 | buf[i+2]) | 0);
                }
            }
            out.sampled_colors = colors.size;
            out.blank = colors.size <= 1;
        } catch (e) { /* keep blank: null */ }
    }
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

# Receivers we treat as HTMLAudioElement (`new Audio(...)` instances).
# Distinct from AudioContext above. Common variable names for sound-
# effect handles in browser games. We're conservative: bare `audio` is
# overloaded with AudioContext patterns in some codebases, so we skip
# it for AudioContext and treat it as HTMLAudioElement only.
_AUDIO_ELT_RECEIVERS = {"audio", "snd", "sfx", "sound", "clip", "track"}

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
    # Pointer Lock / Fullscreen — first-person games use both. The
    # classic-doom 20260512_111015 trace flagged every single
    # `cvs.requestPointerLock()` call as a hallucination across two
    # extension sessions, and the model's mouse-look bug persisted
    # plausibly because the harness kept saying the real API name was
    # wrong. Pointer Lock is on Element since ~2020; canvases inherit
    # it. Fullscreen the same.
    "requestpointerlock", "exitpointerlock",
    "requestfullscreen", "exitfullscreen",
    "webkitrequestpointerlock", "webkitexitpointerlock",
    "webkitrequestfullscreen", "webkitexitfullscreen",
    "mozrequestpointerlock", "mozrequestfullscreen",
    # Element / Selector API methods used in routing patterns
    "matches", "closest", "getrootnode",
    # Web Animations + ScrollIntoView — both legit, both occasionally
    # used in games.
    "animate", "scrollintoview",
})

# Real HTMLAudioElement / HTMLMediaElement methods (subset useful for
# games). Hallucinations the model often emits — `playWithFade`,
# `start3d`, `loopOnce` — are NOT here, so flagged.
_AUDIO_ELT_METHODS = frozenset({
    "play", "pause", "load", "canplaytype", "fastseek",
    "addtexttrack", "captureStream".lower(), "capturestream",
    # Inherited Element methods used in audio-handling patterns.
    "addeventlistener", "removeeventlistener", "dispatchevent",
    "cloneNode".lower(), "clonenode",
    "setattribute", "getattribute", "remove",
})

# Map of (lowered receiver name) -> (allowlist set, friendly label).
_RECEIVER_TYPES: dict[str, tuple[frozenset[str], str]] = {}
for r in _CANVAS2D_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS2D_METHODS, "CanvasRenderingContext2D")
for r in _AUDIOCTX_RECEIVERS:
    _RECEIVER_TYPES[r] = (_AUDIOCTX_METHODS, "AudioContext")
for r in _CANVAS_ELT_RECEIVERS:
    _RECEIVER_TYPES[r] = (_CANVAS_ELT_METHODS, "HTMLCanvasElement")
for r in _AUDIO_ELT_RECEIVERS:
    _RECEIVER_TYPES[r] = (_AUDIO_ELT_METHODS, "HTMLAudioElement")

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


_ASSET_REF_RE = re.compile(
    # Match relative paths inside string literals (single or double quote
    # or backtick). Examples we catch:
    #   './foo_assets/wall.png', "./bar_sounds/shoot.ogg"
    # We deliberately skip CDN URLs (https://...) and data: URIs.
    # Path must end in a known media extension to avoid false positives
    # on arbitrary string literals.
    r"""['"`]\s*(\./[^'"`\s]+\.(?:png|jpe?g|gif|webp|svg|ogg|mp3|wav|m4a))\s*['"`]""",
    re.IGNORECASE,
)


def _check_asset_paths(
    html: str, out_path: "Path | None"
) -> list[str]:
    """Find relative asset paths in the HTML that reference files which
    don't exist on disk. For each missing path, suggest the closest
    real file using difflib.

    Returns a list of warning strings (never errors — Chromium has the
    final word; this is just a useful soft signal).

    Bench traces from `first-person-shooter-doom-game_20260511_160924`
    showed the 27B model corrupting paths in generated HTML: numeric
    tokenizer artifacts (`20260511` → `20260_511`), inconsistent
    underscores in slugs, file renames between attempts. Chromium
    surfaced these as generic `Failed to load resource: net::
    ERR_FILE_NOT_FOUND` lines with NO URL attached, so the model
    couldn't fix them on the next turn. This check tells the model
    EXACTLY which path missed and what the closest match is.
    """
    if out_path is None:
        return []
    try:
        from pathlib import Path
        out = Path(out_path)
        base = out.parent
        if not base.is_dir():
            return []
    except Exception:
        return []

    # Collect candidate files on disk under the session base dir
    # (assets/, sounds/, anywhere ~2 levels deep). Stored as relative
    # POSIX paths so suggestions are pasteable.
    candidates: list[str] = []
    try:
        for ext in ("png", "jpg", "jpeg", "gif", "webp", "svg",
                    "ogg", "mp3", "wav", "m4a"):
            for p in base.rglob(f"*.{ext}"):
                try:
                    rel = p.relative_to(base).as_posix()
                except Exception:
                    continue
                candidates.append("./" + rel)
    except Exception:
        return []

    if not candidates:
        return []  # nothing generated to compare against

    import difflib
    seen: set[str] = set()
    out_warnings: list[str] = []
    for m in _ASSET_REF_RE.finditer(html or ""):
        ref = m.group(1)
        if ref in seen:
            continue
        seen.add(ref)
        full = base / ref[2:]  # strip leading "./"
        try:
            if full.is_file():
                continue
        except Exception:
            continue
        # Missing — find closest match by basename, then by full rel path.
        suggestion: str | None = None
        cand_basenames = [c.rsplit("/", 1)[-1] for c in candidates]
        bn = ref.rsplit("/", 1)[-1]
        close = difflib.get_close_matches(bn, cand_basenames, n=1, cutoff=0.5)
        if close:
            # Look up the full candidate path that ends with this basename.
            for c in candidates:
                if c.endswith("/" + close[0]):
                    suggestion = c
                    break
        out_warnings.append(
            f"asset reference {ref!r} does not exist on disk"
            + (f"; did you mean {suggestion!r}?" if suggestion else
               f" and no close match found among {len(candidates)} "
               "generated files. Use one of the paths from the "
               "GENERATED ASSETS block in the user-turn message verbatim.")
        )
    return out_warnings


def run_micro_probes(
    html: str, out_path: "Path | None" = None
) -> dict[str, Any]:
    """Pre-Chromium structural sanity check.

    Report shape:
      ok:        bool          - True if no errors (warnings allowed).
      errors:    list[str]     - structurally-broken; Chromium will fail.
      warnings:  list[str]     - suspicious but maybe ok; Chromium continues.
      stats:     dict          - small numeric snapshot of what we measured.

    The agent uses this between materialize and Chromium: an `ok=False`
    report skips the browser round-trip and feeds errors back to the
    model on the next turn.

    `out_path` is optional. When provided, asset-path checks scan for
    relative file references in the HTML and flag any that don't exist
    on disk — useful when the model corrupts generated asset paths.
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

    # --- duplicate top-level declarations -----------------------------
    # Catches the "concatenated two drafts" failure mode where the model
    # writes a first draft, starts over partway through, and emits a
    # second draft below the first WITHOUT deleting it. Result: duplicate
    # `const NAME = …` / `function NAME(…)` declarations at the same
    # scope, which Chromium correctly rejects with "Identifier '<name>'
    # has already been declared." Observed in donkey-kong trace
    # 20260516_124628 iter 2 (duplicate `const ctx`, `const state`,
    # `function buildLevels`) — Chromium reported the issue, but only
    # after a 3-second browser load. This probe catches it pre-Chromium
    # and tells the model EXACTLY which name is duplicated so the next
    # turn can target the right delete.
    _DECL_RE = re.compile(
        r"^(?P<indent>[ \t]*)(?P<kind>const|let|function)\s+"
        r"(?P<name>[A-Za-z_$][\w$]*)\b"
    )
    dup_names: list[str] = []
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        stripped = _strip_js_noise(body)
        depth = 0
        # Names declared at depth ≤ 1. Depth 0 is the bare script body;
        # depth 1 is "inside the IIFE wrapper that most agent games use"
        # — `(() => { ... })()`. Nested function bodies live at depth ≥ 2
        # and are excluded because shadowing there is legal.
        seen: dict[str, int] = {}
        for raw_line in stripped.splitlines():
            line = raw_line.rstrip()
            # Track brace depth BEFORE matching this line so a declaration
            # on the same line as an opening `{` is still attributed to
            # the enclosing scope.
            opens = line.count("{")
            closes = line.count("}")
            if depth <= 1:
                m = _DECL_RE.match(line)
                if m:
                    name = m.group("name")
                    seen[name] = seen.get(name, 0) + 1
            depth += opens - closes
            if depth < 0:
                # Unbalanced; bail out — bracket-imbalance probe already
                # reported this. Counting from negative depth would
                # produce noise.
                break
        for name, count in seen.items():
            if count >= 2:
                dup_names.append(name)
    if dup_names:
        # De-dup the message across multiple scripts; keep order stable.
        seen_msg: set[str] = set()
        unique = [n for n in dup_names if not (n in seen_msg or seen_msg.add(n))]
        sample = ", ".join(f"`{n}`" for n in unique[:5])
        errors.append(
            f"duplicate top-level declaration(s) in <script>: {sample} "
            "declared 2+ times at the same scope — looks like two drafts "
            "got concatenated. Re-emit with one body; delete the older "
            "duplicate."
        )
        stats["duplicate_declarations"] = unique

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

    # --- repetition-collapse loop --------------------------------------
    # Mid-size models occasionally degenerate into token-repeat loops
    # (e.g. `ENEMY_HISS_CHANCE_PER_SEC_ACTUAL_ACTUAL_ACTUAL...`) or emit
    # the same line dozens of times before stalling. Catch it here so we
    # surface a specific actionable error instead of forcing the user to
    # decode a 200 KB stream-stall trace.
    rep_warnings: list[str] = []
    for (_attrs, body) in scripts:
        if not body.strip():
            continue
        # Same line repeated > 10× verbatim (ignoring blank lines).
        line_counts: dict[str, int] = {}
        for ln in body.splitlines():
            stripped = ln.strip()
            if len(stripped) < 4:
                continue
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
        for ln, n in line_counts.items():
            if n > 10:
                rep_warnings.append(
                    f"line repeated {n}× verbatim in <script>: "
                    f"{ln[:80]!r}{'…' if len(ln) > 80 else ''}. "
                    "This is a token-repeat loop — emit a focused "
                    "<patch> instead of rewriting the whole file."
                )
                break  # one example per script body is enough
        # Single 4+-char identifier appearing > 30× in one script body.
        # Threshold is intentionally high so legit names like `ctx`/`x`/
        # `i` don't trip; the failure pattern is identifier copies like
        # `_ACTUAL_ACTUAL_ACTUAL_…`.
        for tok, n in _Counter(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", body)).items():
            if n > 30 and "_" in tok and tok.count("_") >= 2:
                rep_warnings.append(
                    f"identifier `{tok}` appears {n}× in one <script> "
                    "body — almost certainly a repeat-loop degeneration. "
                    "Restart the change with a small <patch>."
                )
                break
        # Suffix-loop: a 5+-char substring repeated > 25× anywhere in
        # the body. Catches the `_ACTUAL_ACTUAL_ACTUAL_…` family where
        # each full identifier is unique (so the token counter above
        # doesn't fire) but the suffix repeats.
        if len(body) > 2000:
            for substr in ("_ACTUAL", "_FINAL", "_REAL", "_TRUE"):
                n = body.count(substr)
                if n > 25:
                    rep_warnings.append(
                        f"suffix `{substr}` appears {n}× in one "
                        "<script> body — token-repeat loop. Send a "
                        "focused <patch>, not a rewrite."
                    )
                    break
    if rep_warnings:
        # Promote to errors only when 2+ scripts agree (very likely real).
        if len(rep_warnings) >= 2:
            errors.extend(rep_warnings[:3])
        else:
            warnings.extend(rep_warnings[:3])
        stats["repetition_signals"] = len(rep_warnings)

    # --- elision sentinels ---------------------------------------------
    # Models occasionally slip "// ... rest of code unchanged ..." into
    # a patch even after we tell them not to. Catch it here so we don't
    # ship a half-implemented file. The regex tolerates dotted variants
    # ("// ...rest of", "// .. rest of seed code stays same") which the
    # plain-substring list missed — donkey-kong trace 20260516_124628
    # iter 2 shipped "// ...rest of seed code stays same..." past the
    # detector because the literal had a space between "//" and "rest"
    # while the model emitted "// ..." (no space) instead.
    elision_markers = [
        "// ... rest unchanged",
        "// ... rest of code",
        "// rest of",
        "// (existing code)",
        "/* existing code */",
        "<- leave original",
    ]
    matched_marker: str | None = None
    for m in elision_markers:
        if m.lower() in low:
            matched_marker = m
            break
    if matched_marker is None:
        m_re = re.search(
            r"//\s*\.{2,}\s*rest\b\s+(?:of|unchanged)",
            html,
            re.IGNORECASE,
        )
        if m_re:
            matched_marker = m_re.group(0)
    if matched_marker is not None:
        errors.append(
            f"elision marker found in source: {matched_marker!r} — the file is "
            "incomplete. Re-emit the patch with the EXACT lines, no "
            "shortcuts."
        )

    # --- asset path existence check ------------------------------------
    # Catches model-corrupted file paths before Chromium does (which
    # only reports ERR_FILE_NOT_FOUND without telling the model WHICH
    # file). Soft warnings only — never errors.
    if out_path is not None:
        path_warnings = _check_asset_paths(html, out_path)
        if path_warnings:
            warnings.extend(path_warnings)
            stats["missing_asset_paths"] = len(path_warnings)

        # Inverse check: assets generated on disk but NEVER referenced
        # in the HTML. Common silent failure — model declared 12 sprites
        # in <assets>, generated them all, then forgot to wire them in.
        # Probes pass, visual judge sees a blank canvas, iter "succeeds".
        # Soft warnings only; the model decides whether to wire or drop.
        unused_warnings = _check_unused_assets(html, out_path)
        if unused_warnings:
            warnings.extend(unused_warnings)
            stats["unused_assets"] = len(unused_warnings)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def _check_unused_assets(
    html: str, out_path: "Path | None"
) -> list[str]:
    """Find generated assets (PNG/OGG) on disk under the session dir
    that are NOT referenced by the HTML. Returns warning strings.

    Why this matters: a session can pay 30s+ to generate a dozen sprites
    and a half-dozen sound clips, then ship an HTML file that never
    references any of them. The visible game looks empty/silent; the
    harness probes still pass on the structural side. This check makes
    the omission explicit so the next iter's fix prompt is specific
    instead of "the game looks blank, fix it".
    """
    if out_path is None:
        return []
    try:
        from pathlib import Path
        out = Path(out_path)
        base = out.parent
        if not base.is_dir():
            return []
    except Exception:
        return []
    # Only flag files under the session's generated dirs — we ignore
    # arbitrary art the user dropped into the workspace.
    sprite_dirs: list[Path] = []
    sound_dirs: list[Path] = []
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            n = child.name.lower()
            if n.endswith("_assets"):
                sprite_dirs.append(child)
            elif n.endswith("_sounds"):
                sound_dirs.append(child)
    except OSError:
        return []
    if not sprite_dirs and not sound_dirs:
        return []
    html_text = html or ""
    out_warnings: list[str] = []
    for kind, dirs in (("sprite", sprite_dirs), ("sound", sound_dirs)):
        for d in dirs:
            try:
                files = sorted(d.iterdir())
            except OSError:
                continue
            for f in files:
                if not f.is_file():
                    continue
                # Cheap presence test: either the basename or the full
                # relative path appears verbatim in the HTML. Avoids
                # false positives from arbitrary path-mangling.
                name = f.name
                if name in html_text:
                    continue
                try:
                    rel = f.relative_to(base).as_posix()
                except Exception:
                    rel = name
                if rel in html_text:
                    continue
                out_warnings.append(
                    f"{kind} {name!r} was generated to {rel!r} but is "
                    "NEVER referenced in the HTML. Either wire it in "
                    "(use the GENERATED ASSETS path verbatim) or drop "
                    "it from the next <assets>/<sounds> request to "
                    "save generation time."
                )
                if len(out_warnings) >= 8:
                    return out_warnings
    return out_warnings


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
        # CORS-taint fix (1.1): file:// pages that drawImage from a
        # local PNG taint the canvas, breaking getImageData and any
        # probe that relies on it. These flags re-enable cross-origin
        # reads + give file:// pages full access to local files. We're
        # already running untrusted-by-design HTML inside Playwright;
        # the flags don't widen the threat model meaningfully.
        cors_flags = [
            "--allow-file-access-from-files",
            "--disable-web-security",
        ]
        browser = pw.chromium.launch(headless=True, args=cors_flags)
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

    report = _build_report(errors, warnings, logs, title, canvas_info, listener_info, body_text)
    # A1: synchronous path doesn't separate page vs console errors, so just
    # scan the union for stack frames into the file we just ran.
    report["path"] = str(path)
    report["crash_source_slices"] = extract_crash_source_slices(
        list(errors), file_filter=path,
    )
    return report


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
    # "Feels like a game" bonuses — differentiate "compiles + dark canvas"
    # from "actually runs". These are deliberately small (max +9) so they
    # don't overpower the structural signals above; their job is to break
    # ties between candidates that are otherwise equivalent on errors /
    # listeners / RAF.
    if not (report.get("console_errors") or []):
        s += 3                                  # genuinely silent console
    if report.get("frozen_canvas") is False:
        s += 3                                  # canvas is being repainted
    if it.get("ran") and it.get("any_change") is True:
        s += 3                                  # input_test caused real motion
    return max(0.0, min(100.0, s))


# A1: when a JS error names a file:LINE:COL, splice the actual source
# region into the report so a 27B doesn't have to reverse-line-count
# a 700-line file from a raw stack trace. Stops "let me think about
# which .y access this could be" deliberation chains cold.
_STACK_FRAME_RE = re.compile(r"file://(?P<path>[^\s:)]+):(?P<line>\d+):(?P<col>\d+)")


def extract_crash_source_slices(
    error_strs: list[str],
    *,
    file_filter: str | Path | None = None,
    radius: int = 3,
    max_slices: int = 2,
) -> list[dict[str, Any]]:
    """Parse `file://...:LINE:COL` frames out of each error string and read the
    surrounding source lines from disk. Returns a list of dicts with `path`,
    `line`, `col`, `snippet` (multi-line string with `>` on the offending line).
    Dedups by (path, line) so a 5-deep stack trace doesn't produce 5 copies of
    the same neighborhood. Cap at `max_slices` so a noisy multi-error report
    doesn't blow up the prompt.
    """
    filter_resolved: str | None = None
    if file_filter is not None:
        try:
            filter_resolved = str(Path(file_filter).resolve())
        except Exception:
            filter_resolved = str(file_filter)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for err in error_strs:
        if not err:
            continue
        for m in _STACK_FRAME_RE.finditer(err):
            path_str = m.group("path")
            try:
                line_no = int(m.group("line"))
                col_no = int(m.group("col"))
            except ValueError:
                continue
            if filter_resolved:
                try:
                    cand_resolved = str(Path(path_str).resolve())
                except Exception:
                    cand_resolved = path_str
                if cand_resolved != filter_resolved:
                    continue
            key = (path_str, line_no)
            if key in seen:
                continue
            try:
                src = Path(path_str).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            src_lines = src.splitlines()
            if not src_lines or line_no < 1 or line_no > len(src_lines):
                continue
            lo = max(1, line_no - radius)
            hi = min(len(src_lines), line_no + radius)
            width = len(str(hi))
            rendered = []
            for i in range(lo, hi + 1):
                arrow = ">" if i == line_no else " "
                rendered.append(f"  {arrow} {i:>{width}}: {src_lines[i - 1]}")
            seen.add(key)
            out.append({
                "path": path_str,
                "line": line_no,
                "col": col_no,
                "snippet": "\n".join(rendered),
            })
            if len(out) >= max_slices:
                return out
    return out


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
    # Auto-input smoke test. With the gameplay-state-global fix
    # (2026-05-16), the harness now samples window.state (the
    # documented convention) and names exactly which fields moved on
    # which key. PASS reads like "ArrowRight→[player.x, player.facing];
    # Space→[bullets.length]" — names the wiring path that works. FAIL
    # distinguishes "state exposed but no field moved" (data-flow bug
    # downstream of the listener) from "no state global at all" (the
    # game never exposed window.state — your probes can't run).
    it = report.get("input_test") or {}
    if it.get("ran"):
        summary = it.get("summary") or ""
        if it.get("any_change"):
            lines.append(f"Input test: PASS — {summary}")
        else:
            lines.append(f"Input test: FAIL — {summary}")
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
        # 2.4: split rendering when the harness exposed kinds. Page
        # errors (uncaught exceptions) are usually game bugs the model
        # can fix; console.error lines are often informational logs.
        # Probe errors are reported in their own probes section below;
        # we exclude them here so the model doesn't double-count.
        page_errs = report.get("page_errors") or []
        cons_errs = report.get("console_errors") or []
        if page_errs:
            lines.append("PAGE ERRORS (uncaught exceptions — must fix):")
            for e in page_errs:
                lines.append(f"  - {e}")
        if cons_errs:
            lines.append("CONSOLE ERRORS:")
            for e in cons_errs:
                lines.append(f"  - {e}")
        if not (page_errs or cons_errs):
            # Older path / sync test_html_file: fall back to the union.
            lines.append("ERRORS (must fix):")
            for e in report["errors"]:
                lines.append(f"  - {e}")
        # A1: prepend source context. If the test runner already attached
        # `crash_source_slices`, use those; otherwise compute on the fly so
        # callers that bypass LiveBrowser (sync test path, unit tests) still
        # benefit.
        slices = report.get("crash_source_slices")
        if slices is None:
            slices = extract_crash_source_slices(
                list(page_errs) + list(cons_errs),
                file_filter=report.get("path"),
            )
        for sl in slices or []:
            lines.append(f"SOURCE NEAR ERROR ({Path(sl['path']).name}:{sl['line']}):")
            lines.append(sl["snippet"])
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
        # 2.4: split error sources so format_report_for_model can show
        # them differently. _errors is kept as the combined view for
        # backward compat (downstream code reads report["errors"]).
        self._errors: list[str] = []
        self._console_errors: list[str] = []
        self._page_errors: list[str] = []
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
        # CORS-taint fix (1.1): see comment in test_html_file. Without
        # these flags, drawImage(<file:// PNG>) taints the canvas and
        # any getImageData probe (including the harness's frozen check
        # and the model's non_blank example) throws SecurityError. The
        # cure used to be the model writing crossOrigin="anonymous" +
        # try/catch — error-prone and frequently broke working games.
        # Set the flags at launch and the issue goes away entirely.
        launch_args: list[str] = [
            "--allow-file-access-from-files",
            "--disable-web-security",
        ]
        if not self._headless:
            launch_args.extend([
                f"--window-position=850,50",
                f"--window-size={self._viewport[0]},{self._viewport[1]}",
            ])
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
            # 2.4: split feed for the report. _errors stays as the
            # combined view for backward compat with downstream code.
            self._console_errors.append(text)
            self._errors.append(text)
        elif t == "warning":
            self._warnings.append(text)
        else:
            self._logs.append(text)

    def _on_pageerror(self, exc) -> None:
        text = _truncate(f"UNCAUGHT: {exc}", _MAX_MSG_LEN)
        self._page_errors.append(text)
        self._errors.append(text)

    async def load_and_test(
        self,
        path: str | Path,
        screenshot_path: str | Path | None = None,
        *,
        probes: list[dict] | None = None,
        screenshot_before_path: str | Path | None = None,
        criteria: str | None = None,
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
        self._console_errors.clear()
        self._page_errors.clear()
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
                # 1.2: tainted-canvas / cross-origin / SecurityError on
                # getImageData are HARNESS-side limitations, not game
                # bugs — the actual canvas might be perfectly rendered.
                # Don't let a brittle probe block shipping a working
                # game; classify it as warning, not failure. The launch
                # flags from 1.1 should prevent this in practice but
                # defense-in-depth.
                taint_signal = bool(err and any(
                    s in err.lower()
                    for s in ("tainted", "cross-origin", "securityerror")
                ))
                entry: dict[str, Any] = {
                    "name": pname, "expr": pexpr[:200], "ok": ok, "err": err,
                }
                if taint_signal and not ok:
                    entry["ok"] = True
                    entry["downgraded"] = (
                        "harness-side CORS/taint (probe relies on "
                        "getImageData/toDataURL) — treating as pass"
                    )
                probe_results.append(entry)

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
        # 2.4: split error sources so the model + trace can tell apart
        # "console.error('...')" (game-logged, often informational) from
        # "UNCAUGHT TypeError" (real crash). Existing report["errors"]
        # is the union — kept for backward compat with downstream code.
        report["console_errors"] = list(self._console_errors)
        report["page_errors"] = list(self._page_errors)
        # Probe errors are derived from probe_results (entries with
        # ok=False AND non-empty err).
        report["probe_errors"] = [
            f"{p.get('name','?')}: {p.get('err','')[:160]}"
            for p in probe_results
            if not p.get("ok") and p.get("err")
        ]
        # A1: precompute source slices once so format_report_for_model
        # doesn't re-read the file on every render. file_filter biases the
        # extractor to frames inside the current game file (deep frameworks
        # in CDN scripts won't pollute the report).
        report["path"] = str(path)
        report["crash_source_slices"] = extract_crash_source_slices(
            list(self._page_errors) + list(self._console_errors),
            file_filter=path,
        )

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
        # Low-color-diversity check: 1024 sample points across the canvas
        # but only a handful of unique colors → game is barely rendering
        # (typical: WebGL "Three.js stub" with one cube on solid bg, or a
        # 2D canvas drawn once with a flat fill). The model can't game
        # this because the threshold runs harness-side, not as a
        # model-authored probe. Skip the check on tiny canvases (< 256
        # pixels) where low diversity is legitimate, and on the input-
        # responsive case where the canvas may be intentionally minimal
        # before user input (drawing app, tic-tac-toe blank board).
        if (canvas_info
                and canvas_info.get("sampled_colors") is not None
                and canvas_info.get("blank") is False
                and canvas_info.get("raf_ran")
                and canvas_info.get("width", 0) * canvas_info.get("height", 0) >= 256
                and not input_responsive):
            n_colors = int(canvas_info["sampled_colors"])
            if n_colors < 8:
                report["soft_warnings"].append(
                    f"HEURISTIC: only {n_colors} unique colors across 1024 "
                    f"canvas sample points - the scene is trivial (likely a "
                    f"stub/placeholder, not a finished game). Real games "
                    f"render at least 8-16 distinct colors even early in "
                    f"the first frame. Add real sprites, varied geometry, "
                    f"or proper textures."
                )
        if input_dead:
            keys_str = ", ".join(input_test.get("keys_tried", []))
            # Don't double-fire if the page only has a button (no canvas
            # input expected). We check for an interactive element below.
            has_clickable = await self._safe_eval(
                "document.querySelectorAll('button, [onclick]').length > 0"
            )
            # DK trace 20260514_104131 fix: when criteria explicitly
            # mention keyboard / pointer controls, override the
            # "treating as DOM-driven" carve-out — the game promised
            # input and the input is broken, regardless of whether a
            # restart button happens to exist.
            controls_expected = expects_game_controls(criteria or "")
            # Item 3, trace build-a-donkey-kong-clone-in-o_20260514_214747:
            # the model's code had window.addEventListener + e.code +
            # e.preventDefault, and report showed `Input listeners:
            # total=5 (win=5)` — listeners WERE registered, RAF ran,
            # canvas was not blank. But the input_test still failed,
            # and the prior message ("Controls are not wired up") was
            # misleading because controls ARE wired. The actual cause
            # is downstream: handler writes to a `keys` object that
            # draw() doesn't read (closure-scope mismatch), or draw()
            # reads from a stale snapshot, or RAF is started before
            # the listener is registered. Branch the message so the
            # model sees a precise diagnostic instead of a wrong one.
            listeners_present = (
                bool(listener_info)
                and listener_info.get("total", 0) > 0
            )
            raf_ran = bool(canvas_info and canvas_info.get("raf_ran"))
            handler_present_but_no_visible_change = (
                listeners_present and raf_ran
                and canvas_info and canvas_info.get("blank") is False
            )
            if not has_clickable or controls_expected:
                if handler_present_but_no_visible_change:
                    # Listeners + RAF + non-blank canvas, but keys
                    # don't move pixels: the wiring exists, the bug
                    # is in the data flow between handler and draw().
                    report["soft_warnings"].append(
                        f"HEURISTIC: pressed {keys_str} - "
                        f"{listener_info.get('total', 0)} key listener(s) "
                        "registered on window AND the RAF loop is "
                        "rendering, but pressing keys did not produce "
                        "any visible pixel change. The wiring exists "
                        "but the data flow between your keydown "
                        "handler and your draw() is broken. CHECK: "
                        "(1) your keydown handler writes to the SAME "
                        "`keys` object that updatePlayer/draw() reads "
                        "from (closure-scope mismatch is the #1 cause "
                        "on small models); (2) you use `e.code` (not "
                        "`e.key`) so layout-dependent values like "
                        "'ArrowLeft' actually match your KEYMAP; "
                        "(3) RAF is started AFTER the listener is "
                        "registered so the first frame can read "
                        "state set by held keys; (4) update() is "
                        "actually being called (no early-return on "
                        "state.over=true at startup)."
                    )
                else:
                    # Listeners absent or RAF dead → original message.
                    report["soft_warnings"].append(
                        f"HEURISTIC: pressed {keys_str} - canvas pixels never changed. "
                        "Controls are not wired up (or input handler is broken)."
                    )
                # Synthesize an `input_responsive` probe entry so the
                # failure also surfaces in the per-probe display (model
                # can't rationalize it as "just a warning"). The
                # existing probe-failure gate below promotes it to a
                # second PROBE FAILED soft_warning, which the `ok`
                # formula already catches. Only fires when controls are
                # affirmatively expected — for genuinely DOM-driven
                # pages we still want the lighter signal.
                if controls_expected:
                    if handler_present_but_no_visible_change:
                        probe_err = (
                            f"pressed {keys_str}; "
                            f"{listener_info.get('total', 0)} listener(s) "
                            "registered AND RAF is rendering, but keys "
                            "do not produce visible state change. "
                            "Verify your keydown handler writes to the "
                            "same `keys` object draw() reads (closure-"
                            "scope mismatch), and that update() runs "
                            "(no early-return on game-state flags)."
                        )
                    else:
                        probe_err = (
                            f"pressed {keys_str} and canvas pixels "
                            f"never changed — controls promised in "
                            f"<criteria> but not wired. Confirm the "
                            f"key handler runs and updates state."
                        )
                    probe_results.append({
                        "name": "input_responsive",
                        "expr": (
                            "(harness) keyboard input must produce a "
                            "visible canvas change"
                        ),
                        "ok": False,
                        "err": probe_err,
                        "synthetic": True,
                    })
            else:
                report["warnings"].append(
                    "Note: keyboard test produced no canvas change, but the "
                    "page has clickable elements; treating as DOM-driven."
                )
        # A1 — probe failures gate the test. Until this commit, probes
        # were advisory: the harness reported probe results but `ok`
        # ignored them. So a game where the model's own probes
        # (`player.health > 0`, `monsters.length > 0`) failed would
        # still get `ok=True` and ship as "passed". Now any probe
        # whose .ok is False appends a PROBE FAILED soft_warning,
        # which the existing `ok` formula catches. See games/traces/
        # game-of-doom-a-first-person-sh_20260506_230058.jsonl iter 1
        # for the failure case this fixes.
        for p in (report.get("probes") or []):
            if p.get("ok"):
                continue
            name = p.get("name", "probe")
            expr = (p.get("expr") or "")[:80]
            err = p.get("err") or "evaluated falsy"
            report["soft_warnings"].append(
                f"PROBE FAILED [{name}]: `{expr}` — {err}. "
                "Your Phase A acceptance criterion is unmet; fix the "
                "game so it evaluates truthy."
            )
        # Stop-Losing-To-OneShot todo #2 — criteria/probe coverage gate.
        # The model authors its own <probes>; mid-tier models often write
        # existence-only probes (`!!player`) that pass even when the
        # promised behavior is broken. Flag any <criteria> line whose
        # meaningful words do not appear in any probe's name/expr — the
        # model must add a probe that actually tests it. Genre-free:
        # uses simple word overlap, no semantic parsing.
        coverage_gaps = _criteria_coverage_gaps(criteria or "", probes or [])
        if coverage_gaps:
            report["criteria_uncovered"] = coverage_gaps
            # Synthesize a failing probe per gap so the coverage hole shows
            # up in the same probe_errors list the model already responds
            # to. Local LLMs ignore "soft_warnings" but fix failing probes
            # — proven across the asteroid trace, where the soft-warning
            # form was visible for 4 iters and never addressed. The
            # synthetic probes carry name `coverage_gap__<slug>` so the
            # agent's Phase B re-parse can detect closure (model emits a
            # new <probes> block whose entries reference the gap → agent
            # replaces self._probes with the new set).
            for gap in coverage_gaps:
                slug = _slugify_criterion(gap)
                probe_results.append({
                    "name": f"coverage_gap__{slug}",
                    "expr": "false  /* synthetic - no model-authored probe for this criterion */",
                    "ok": False,
                    "err": (
                        f"criterion has no probe: {gap[:160]}. "
                        f"In your NEXT reply, keep doing your normal "
                        f"fix work (<patch> or <html_file>) AND include "
                        f"an updated <probes>...</probes> block that "
                        f"adds one entry referencing this criterion "
                        f"(name OR expr should share words with the "
                        f"criterion text). The probes block is "
                        f"re-parsed only when accompanied by code — "
                        f"don't emit probes alone."
                    ),
                    "synthetic": True,
                })
            # Make sure report["probes"] and report["probe_errors"] reflect
            # the just-appended synthetic entries — they were built earlier
            # in this method, so we refresh them here.
            report["probes"] = probe_results
            report["probe_errors"] = [
                f"{p.get('name','?')}: {p.get('err','')[:160]}"
                for p in probe_results
                if not p.get("ok") and p.get("err")
            ]
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

        Stop-Losing-To-OneShot todo #2 — ambient baseline + gameState delta.
        Games with continuous animation (asteroids drifting, particles,
        scrolling parallax) made the old before/after canvas-hash test
        ALWAYS report change → false-positive "TEST OK" on broken
        controls. We now:
          (a) take an ambient baseline (two hashes 250ms apart, no key
              held) and capture window.gameState if exposed;
          (b) for each held key, prefer a `gameState` numeric-leaf delta
              that did NOT change in ambient (input-attributable; works
              even when the canvas is also drifting), and only fall back
              to canvas-hash change when ambient was stable.
        Generic and behavioral — works for any HTML/JS game.
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

        # Flatten the game's exposed state into a {dotted-path: number}
        # map of numeric leaves, depth-capped and fanout-capped so giant
        # entity arrays don't explode the snapshot. Returns null when
        # nothing is exposed — input-test then falls back to canvas hash
        # only.
        #
        # NAME-MATCHING BUG FIX (2026-05-16): for the entire history of
        # this code, the snapshot looked at `window.gameState` while the
        # system prompt and all won-skeletons expose `window.state`. The
        # net effect was that the gameplay verification path was BLIND
        # to the actual state of agent-generated games and silently
        # fell back to canvas-hash (which is degenerate for any auto-
        # animating game). This is the single biggest reason "input
        # smoke test passed" did not correlate with "controls work."
        # Now we walk a small ordered list of plausible globals and
        # take the first that's an object. `state` is the documented
        # convention; the others are back-compat.
        _GAMESTATE_SNAPSHOT_JS = """
        () => {
            const candidates = ['state', 'gameState', 'game', 'GAME', 'world'];
            let gs = null;
            for (const name of candidates) {
                const v = window[name];
                if (v != null && typeof v === 'object') { gs = v; break; }
            }
            if (gs == null) return null;
            const out = {};
            const visit = (obj, path, depth) => {
                if (depth > 4 || obj == null) return;
                if (typeof obj === 'number' && isFinite(obj)) { out[path] = obj; return; }
                if (typeof obj === 'boolean') { out[path] = obj ? 1 : 0; return; }
                if (typeof obj !== 'object') return;
                const ks = Array.isArray(obj)
                    ? obj.slice(0, 32).map((_, i) => String(i))
                    : Object.keys(obj).slice(0, 64);
                for (const k of ks) {
                    try { visit(obj[k], path ? path + '.' + k : k, depth + 1); }
                    catch (e) { /* getters etc */ }
                }
            };
            visit(gs, '', 0);
            // Also surface array lengths so "bullets fired" registers as a delta.
            const lenVisit = (obj, path, depth) => {
                if (depth > 4 || obj == null || typeof obj !== 'object') return;
                if (Array.isArray(obj)) { out[(path || '_root') + '.length'] = obj.length; return; }
                const ks = Object.keys(obj).slice(0, 64);
                for (const k of ks) {
                    try { lenVisit(obj[k], path ? path + '.' + k : k, depth + 1); }
                    catch (e) {}
                }
            };
            lenVisit(gs, '', 0);
            return out;
        }
        """

        def _gs_changed_leaves(before: Any, after: Any) -> set[str]:
            """Set of leaf paths whose value differs (epsilon for floats)
            or that appeared/disappeared between the two snapshots."""
            if not isinstance(before, dict) or not isinstance(after, dict):
                return set()
            EPS = 1e-3
            changed: set[str] = set()
            for k, b in before.items():
                a = after.get(k)
                if a is None and k not in after:
                    changed.add(k)
                    continue
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > EPS:
                        changed.add(k)
            for k in after:
                if k not in before:
                    changed.add(k)
            return changed

        # ---- ambient baseline (no key held) -------------------------------
        ambient_a = await self._safe_eval(_CANVAS_HASH_JS)
        ambient_gs_a = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
        await asyncio.sleep(0.25)
        ambient_b = await self._safe_eval(_CANVAS_HASH_JS)
        ambient_gs_b = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
        ambient_canvas_changed = bool(
            ambient_a is not None and ambient_b is not None and ambient_a != ambient_b
        )
        ambient_gs_changes = _gs_changed_leaves(ambient_gs_a, ambient_gs_b)
        has_gamestate = ambient_gs_a is not None

        # Track WHICH state fields move on WHICH key — names the
        # exact wiring path that works, so the report can say
        # "ArrowRight → state.player.x changed" instead of just
        # "input test passed." When nothing moves we get the dual:
        # "ArrowRight: zero numeric fields on window.state changed."
        responsive_evidence: dict[str, list[str]] = {}

        for k in keys:
            before = await self._safe_eval(_CANVAS_HASH_JS)
            before_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
            if before is None:
                return {"ran": False, "reason": "canvas not sampleable", "keys_tried": tried}
            try:
                await self._page.keyboard.down(k)
                await asyncio.sleep(0.25)  # hold long enough for thrust to move ship
                after_held = await self._safe_eval(_CANVAS_HASH_JS)
                after_gs = await self._safe_eval(_GAMESTATE_SNAPSHOT_JS)
                await self._page.keyboard.up(k)
            except Exception:
                continue
            tried.append(k)
            # Wait one more frame for any post-release tween / momentum.
            await asyncio.sleep(0.05)
            after_release = await self._safe_eval(_CANVAS_HASH_JS)

            # Decide responsiveness with the strongest available signal.
            # (1) state input-only delta: leaves that changed during
            # the held window but did NOT change during the ambient
            # window are attributable to the key press.
            input_only_leaves: set[str] = set()
            if has_gamestate:
                held_changes = _gs_changed_leaves(before_gs, after_gs)
                input_only_leaves = held_changes - ambient_gs_changes
            # (2) canvas-hash fallback: only credit when ambient was
            # stable (so the held-window change was input-induced).
            canvas_input_changed = (
                (after_held is not None and after_held != before)
                or (after_release is not None and after_release != before)
            ) and not ambient_canvas_changed

            if input_only_leaves:
                # Sort + cap so the report stays bounded.
                responsive_evidence[k] = sorted(input_only_leaves)[:5]
                any_change = True
                if first_responsive_key is None:
                    first_responsive_key = k
                # Don't break — collect evidence for ALL keys that
                # work so the report can name which control wires
                # which field. Cheap; ~250ms per key, bounded.
            elif canvas_input_changed:
                responsive_evidence[k] = ["<canvas-pixel-change>"]
                any_change = True
                if first_responsive_key is None:
                    first_responsive_key = k

        # Concise summary line for the report. Two shapes:
        #   PASS — "ArrowRight→[player.x, player.facing], Space→[bullets.length]"
        #   FAIL — "had window.state but zero fields moved across [keys]"
        if any_change:
            parts = []
            for k, leaves in responsive_evidence.items():
                parts.append(f"{k}→[{', '.join(leaves)}]")
            summary = "; ".join(parts[:4])
        elif has_gamestate:
            summary = (
                f"window.state IS exposed but zero numeric fields "
                f"changed across {tried} — the listeners may be wired "
                f"but the data flow into state is broken."
            )
        else:
            summary = (
                f"no game state global exposed (tried "
                f"window.state/.gameState/.game) AND canvas pixels did "
                f"not change for {tried}."
            )

        return {
            "ran": True,
            "any_change": any_change,
            "keys_tried": tried,
            "first_responsive_key": first_responsive_key,
            "responsive_evidence": responsive_evidence,
            "summary": summary,
            "ambient_canvas_motion": ambient_canvas_changed,
            "ambient_gs_motion": bool(ambient_gs_changes),
            "had_gamestate": has_gamestate,
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
