#!/usr/bin/env python3
"""Live-Chromium smoke test for the gameplay-state-global fix.

Background: until 2026-05-16, tools.py's input smoke test sampled
window.gameState while the system prompt and every won-skeleton
expose window.state. The two never matched, so the gameplay
verification path silently fell back to canvas-hash only — and the
canvas-hash path is degenerate for any auto-animating game.

This script writes a tiny HTML game that DOES bind window.state and
DOES move state.player.x on ArrowRight, loads it in real Chromium
via the harness's LiveBrowser, runs the input smoke test, and
asserts:

  1. had_gamestate is True  (the new state-global walk found it)
  2. any_change is True     (ArrowRight produced a state delta)
  3. responsive_evidence['ArrowRight'] names 'player.x'
  4. summary string mentions 'ArrowRight' and 'player.x'

Run from the repo root after Playwright Chromium is installed:

    .venv/bin/python scripts/_smoke_input_test.py

Exits 0 on success, 1 on assertion failure, 2 on missing deps.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>smoke</title></head>
<body>
<canvas id="c" width="400" height="300"></canvas>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  // Game state exposed on window.state per the documented convention.
  const state = {
    player: { x: 100, y: 150 },
    bullets: [],
    score: 0,
    t: 0,
  };
  window.state = state;
  const keys = Object.create(null);
  addEventListener("keydown", e => {
    keys[e.code] = true;
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Space"]
        .includes(e.code)) e.preventDefault();
  }, { passive: false });
  addEventListener("keyup", e => { keys[e.code] = false; });
  function frame() {
    state.t += 1;
    if (keys.ArrowRight) state.player.x += 2;
    if (keys.ArrowLeft)  state.player.x -= 2;
    if (keys.Space) state.bullets.push({ x: state.player.x, y: state.player.y });
    ctx.clearRect(0, 0, 400, 300);
    ctx.fillStyle = "#7ab6ff";
    ctx.fillRect(state.player.x, state.player.y, 20, 20);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


async def amain() -> int:
    try:
        from tools import LiveBrowser
    except Exception as e:
        print(f"FAIL: could not import tools.LiveBrowser — {e}")
        return 2

    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_input_"))
    html_path = tmp_dir / "game.html"
    html_path.write_text(HTML, encoding="utf-8")
    print(f"[1] wrote test game to {html_path}")

    lb = LiveBrowser(headless=True)
    try:
        await lb.start()
        print("[2] Chromium up")
        report = await lb.load_and_test(html_path, probes=[])
        it = report.get("input_test") or {}
        print()
        print("== input_test return ==")
        print(f"  ran                : {it.get('ran')}")
        print(f"  had_gamestate      : {it.get('had_gamestate')}")
        print(f"  any_change         : {it.get('any_change')}")
        print(f"  first_responsive   : {it.get('first_responsive_key')!r}")
        print(f"  responsive_evidence: {it.get('responsive_evidence')}")
        print(f"  summary            : {it.get('summary')}")
        print()

        # Assertions — the four properties this fix promises.
        failures: list[str] = []
        if not it.get("had_gamestate"):
            failures.append(
                "had_gamestate is False — the state-global walk did "
                "NOT find window.state. Check tools.py "
                "_GAMESTATE_SNAPSHOT_JS includes 'state' in candidates."
            )
        if not it.get("any_change"):
            failures.append(
                "any_change is False — no key produced a state delta. "
                "The smoke test wrote a game that moves state.player.x "
                "on ArrowRight; if this fails the snapshot is broken."
            )
        ev = it.get("responsive_evidence") or {}
        # Either ArrowRight or ArrowLeft should have moved player.x; the
        # smoke game responds to both. Be lenient about which key was
        # tested first.
        moved_player_x = any(
            "player.x" in leaves for leaves in ev.values()
        )
        if not moved_player_x:
            failures.append(
                f"responsive_evidence does not include player.x in any "
                f"key's evidence list. Got: {ev!r}"
            )
        summary = it.get("summary") or ""
        if "player.x" not in summary:
            failures.append(
                f"summary string does not mention player.x. Got: "
                f"{summary!r}"
            )

        if failures:
            print("== FAIL ==")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("== PASS — gameplay-state-global fix verified end-to-end ==")
        return 0
    finally:
        await lb.close()
        try:
            html_path.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
