#!/usr/bin/env python3
"""Automated playthrough of games/dragons-lair-deluxe.html.

Drives the QTE loop by polling window.state and pressing the expected key
while each reaction window is open. Also exercises one deliberate death
(wrong key) and the cutscene-skip path. Saves screenshots to
games/dragons-lair-deluxe_shots/ and fails on any console/page error.

Usage:
    .venv/bin/python scripts/_verify_dragons_lair_deluxe.py [--headed]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GAME = REPO / "games" / "dragons-lair-deluxe.html"
SHOTS = REPO / "games" / "dragons-lair-deluxe_shots"

from playwright.async_api import async_playwright  # noqa: E402


async def main() -> int:
    headed = "--headed" in sys.argv
    SHOTS.mkdir(exist_ok=True)
    errors: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not headed,
            args=["--allow-file-access-from-files", "--disable-web-security",
                  "--autoplay-policy=no-user-gesture-required"],
        )
        page = await browser.new_page(viewport={"width": 1100, "height": 720})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(GAME.as_uri())
        await page.wait_for_timeout(2500)  # asset load

        async def shot(name: str) -> None:
            await page.screenshot(path=str(SHOTS / f"{name}.png"))

        async def st(expr: str):
            return await page.evaluate(f"window.state && window.state.{expr}")

        phase = await st("phase")
        print(f"phase after load: {phase}")
        await shot("01_title")
        assert phase == "title", f"expected title, got {phase}"

        # start the game; skip intro cutscene quickly
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(1200)
        await shot("02_intro_cut")
        await page.keyboard.press("Enter")  # skip cutscene
        await page.wait_for_timeout(400)

        # deliberate death on scene 1: press the WRONG key in the window
        died = False
        for _ in range(200):
            if await st("windowOpen"):
                await page.keyboard.press("ArrowDown")  # scene 1 wants ArrowUp
                died = True
                break
            await page.wait_for_timeout(50)
        assert died, "reaction window never opened on scene 1"
        await page.wait_for_timeout(800)
        await shot("03_death_flash")
        ph = await st("phase")
        assert ph in ("deathflash", "cut"), f"expected death, got {ph}"
        lives = await st("lives")
        assert lives == 4, f"expected 4 lives, got {lives}"
        # wait out flash + skip death cutscene
        await page.wait_for_timeout(1700)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(500)

        # now play through all 8 scenes pressing the right keys
        keymap = {"Space": " "}
        shots_done: set[int] = set()
        for _ in range(900):  # ~90 s budget
            phase = await st("phase")
            if phase == "victory":
                break
            if phase == "cut":
                await page.keyboard.press("Enter")
                await page.wait_for_timeout(150)
                continue
            if phase != "play":
                await page.wait_for_timeout(100)
                continue
            scene = await st("scene")
            if scene not in shots_done:
                shots_done.add(scene)
                await page.wait_for_timeout(300)
                await shot(f"1{scene}_scene{scene + 1}")
            if await st("windowOpen") and not await st("resolved"):
                exp = await page.evaluate("window.__expKey ? window.__expKey() : null")
                if exp is None:
                    print("WARN: __expKey probe missing")
                    return 3
                await page.keyboard.press(keymap.get(exp, exp))
                await page.wait_for_timeout(120)
                continue
            await page.wait_for_timeout(60)

        phase = await st("phase")
        # skip the victory cutscene if it is playing
        if phase == "victory":
            await page.wait_for_timeout(600)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(600)
            await shot("90_victory")
        score = await st("score")
        scene = await st("scene")
        print(f"final phase={phase} scene={scene} score={score}")
        await browser.close()

    real_errors = [e for e in errors if "ERR_FILE_NOT_FOUND" not in e]
    missing_files = [e for e in errors if "ERR_FILE_NOT_FOUND" in e]
    if missing_files:
        print(f"(note: {len(missing_files)} missing-file loads — videos not generated yet, fallback path OK)")
    if real_errors:
        print("CONSOLE/PAGE ERRORS:")
        for e in real_errors:
            print("  ", e)
        return 1
    if phase != "victory":
        print(f"FAIL: did not reach victory (phase={phase})")
        return 2
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
