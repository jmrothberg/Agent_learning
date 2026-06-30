"""Shared VLM facing sanity + Q4 helpers for opt-in facing eval.

Sanity: screenshot the intentional seed bug (fighters not facing each other)
and require the local VLM to answer NO on the facing question. If it says YES,
the model cannot be trusted as a facing judge — skip or fail the full eval.

`/vlm-critique` in the main agent is unchanged; this module is only for eval
preflight and smoke scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

from agent import GameAgent
from tools import LiveBrowser

FACING_Q = "Do both characters appear to face toward each other?"

_RAF_SETTLE_JS = (
    "(async()=>{for(let i=0;i<4;i++)await new Promise(r=>requestAnimationFrame(r));})()"
)


def facing_q_prompt(*, context: str = "") -> str:
    """Single-question facing prompt (matches eval post-run Q4)."""
    lead = context.strip()
    if lead:
        lead = lead.rstrip() + "\n"
    return (
        f"{lead}"
        "Answer with exactly one line: Q1: YES, NO, or UNCLEAR.\n\n"
        f"Q1: {FACING_Q}\n"
    )


def facing_sanity_passes(q_answer: str | None) -> bool:
    """True when VLM says NO on the known-bad seed (can spot facing bugs)."""
    return (q_answer or "").lower() == "no"


def _resolve_vlm_path() -> str | None:
    from vision_judge import _resolve_local_mlx_vlm, is_enabled

    if not is_enabled():
        return None
    query = os.environ.get(
        "MLX_MODEL",
        os.environ.get("SMOKE_VLM_MODEL", "qwen3.6-27b-mxfp8"),
    )
    return _resolve_local_mlx_vlm(query)


def parse_facing_answer(raw: str) -> dict:
    """Parse a one-line facing VLM reply into {q_answer, parse_rate, raw_preview}."""
    import memory as memory_mod

    recipe = memory_mod.VisualPlaytestRecipe(
        id="canvas-two-actors-facing",
        kind="visual_playtest",
        content="facing q",
        tags=[],
        source_tier="root",
        verified=True,
        helpful=0,
        harmful=0,
        recipe={"checklist": [FACING_Q]},
    )
    text = (raw or "").strip()
    if text and not text.upper().startswith("Q"):
        text = f"Q1: {text}"
    parsed = GameAgent._parse_visual_playtest_response(text, recipe)
    ans = (parsed.get("answers") or {}).get(1, ("", ""))[0]
    return {
        "q_answer": ans or None,
        "parse_rate": parsed.get("parse_rate"),
        "raw_preview": (raw or "")[:300],
    }


async def ask_facing_vlm(
    png: bytes,
    *,
    model_path: str,
    context: str = "",
) -> dict:
    """Run facing Q on one screenshot. ``ok`` = model answered YES."""
    from vision_judge import run_local_vlm_prompt

    prompt = facing_q_prompt(context=context)
    raw = await run_local_vlm_prompt(
        prompt=prompt,
        images=[png],
        model_path=model_path,
        max_tokens=128,
    )
    if not raw:
        return {
            "ok": False,
            "q_answer": None,
            "parse_rate": 0.0,
            "raw_preview": "",
            "error": "VLM empty response",
        }
    out = parse_facing_answer(raw)
    out["ok"] = out.get("q_answer") == "yes"
    out["error"] = None
    return out


async def capture_html_screenshot(
    html_path: Path,
    *,
    headless: bool,
    run_seconds: float = 1.5,
) -> tuple[bytes | None, str | None]:
    """Load HTML in Chromium, wait for assets, return (PNG bytes, error)."""
    if not html_path.is_file():
        return None, f"file not found: {html_path}"
    browser = LiveBrowser(run_seconds=run_seconds, headless=headless)
    try:
        await browser.start()
        await browser._page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=15_000)
        settle = await browser._wait_for_session_assets_ready()
        if settle.get("need") and not settle.get("ready"):
            return None, f"asset_decode_settle not ready: {settle}"
        await browser._page.evaluate(_RAF_SETTLE_JS)
        return await browser._page.screenshot(type="png", full_page=False), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:240]
    finally:
        try:
            await browser.close()
        except Exception:
            pass


async def run_vlm_facing_sanity(
    fixture_html: Path,
    *,
    headless: bool = True,
) -> dict:
    """Preflight: VLM must say NO on the seeded facing bug screenshot."""
    model_path = _resolve_vlm_path()
    if not model_path:
        return {
            "sanity_ok": False,
            "error": "VISION_JUDGE disabled or no local VLM resolved",
            "model_path": None,
        }
    png, cap_err = await capture_html_screenshot(fixture_html, headless=headless)
    if not png:
        return {
            "sanity_ok": False,
            "error": cap_err or f"screenshot failed for {fixture_html}",
            "model_path": model_path,
        }
    vlm = await ask_facing_vlm(
        png,
        model_path=model_path,
        context=(
            "You are reviewing one screenshot of a 2-player fighting game seed "
            "where fighters may be facing the wrong direction."
        ),
    )
    q = vlm.get("q_answer")
    return {
        "sanity_ok": facing_sanity_passes(q),
        "q_answer": q,
        "parse_rate": vlm.get("parse_rate"),
        "raw_preview": vlm.get("raw_preview"),
        "model_path": model_path,
        "error": vlm.get("error"),
    }
