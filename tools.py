"""Browser test harness for the coding-box agent.

Single public function: `test_html_file(path, run_seconds=3.0) -> dict`.

It launches a headless Chromium via Playwright, loads the file, lets it
animate for a few seconds (so requestAnimationFrame loops actually tick),
then returns a SHORT report. Keeping the report short matters: a small
model gets confused by huge logs, so we cap and truncate aggressively.
"""

from __future__ import annotations

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
    const ctx = c.getContext('2d');
    if (!ctx || c.width < 2 || c.height < 2) return out;
    try {
        const w = c.width, h = c.height;
        const pts = [
            [0,0],[w/2|0,0],[w-1,0],
            [0,h/2|0],[w/2|0,h/2|0],[w-1,h/2|0],
            [0,h-1],[w/2|0,h-1],[w-1,h-1],
        ];
        const colors = new Set();
        for (const [x,y] of pts) {
            const d = ctx.getImageData(x,y,1,1).data;
            colors.add(d[0]+','+d[1]+','+d[2]+','+d[3]);
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
        if canvas_info.get("blank") is True:
            soft_warnings.append(
                f"HEURISTIC: canvas appears blank (all {canvas_info.get('sampled_colors')} sampled "
                "pixels are identical) - nothing is being drawn."
            )
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
        await asyncio.sleep(half)
        canvas_info = await self._safe_eval(_CANVAS_PROBE_JS)
        hash_last = await self._safe_eval(_CANVAS_HASH_JS)

        # ---- input smoke test ---------------------------------------------
        # Most small-model bugs we miss are "controls don't work". Fire a few
        # standard inputs and check if pixels change. Captured pre/post
        # snapshots are compared via a key-set hash from the same probe.
        input_test = await self._input_smoke_test()

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
        report["frozen_canvas"] = frozen
        report["input_test"] = input_test

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
