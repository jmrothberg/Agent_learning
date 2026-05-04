"""coder.py — non-interactive CLI driver for the GameAgent.

Usage:
    python coder.py "Make a Snake game with a wraparound board and a score counter"
    python coder.py "snake" --max-iters 4 --best-of-n 1 --headless

Optional flags:
    --model NAME        Override the Ollama model tag (default: see MODEL below)
    --max-iters N       Cap iterations (default 6)
    --out PATH          Where to save the final game (default games/game.html)
    --best-of-n N       Sample N candidates per fix turn (default 2)
    --num-ctx N         Ollama context window (default 16384)
    --stall-seconds N   Per-chunk stream watchdog (default 60)
    --headless          Run Chromium headless (no visible window). Use this
                        for unattended runs and CI; the TUI uses visible.
    --open              After finishing, open the result in your real browser.

This driver shares its agent core (agent.py) with the TUI in chat.py, so
behavior matches what you'd see interactively.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from agent import AgentEvent, GameAgent
from tools import LiveBrowser


# Hard fallback model. chat.py prefers a different resolution path
# (env var → installed). Keeping this constant here so the CLI is
# usable on its own without resolve_chat_model.
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:latest")

# Sentinel: --out was not supplied → derive a unique meaningful name from
# the goal (instead of clobbering games/game.html every run).
_DEFAULT_OUT = "games/game.html"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 30) -> str:
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    if not s:
        s = "game"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-") or "game"
    return s


def _resolve_out_path(arg_out: str, goal: str) -> Path:
    """If the user accepted the default, give the file a unique meaningful
    stem so successive runs don't overwrite each other. If they passed
    --out explicitly, respect their choice verbatim."""
    if arg_out != _DEFAULT_OUT:
        return Path(arg_out)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("games") / f"{_slugify(goal)}_{ts}.html"


def _print_event(ev: AgentEvent) -> None:
    """Single line per event, matched by `tail -f games/traces/*.log`."""
    if ev.kind == "phase":
        print(f"\n── {ev.text} ──", flush=True)
    elif ev.kind == "memory":
        print(f"  memory: {ev.text}", flush=True)
    elif ev.kind == "best_of_n":
        print(f"  best-of-N: {ev.text}", flush=True)
    elif ev.kind == "diagnose":
        print(f"  diagnose: {ev.text[:200]}", flush=True)
    elif ev.kind == "code":
        d = ev.data
        print(
            f"  wrote {ev.text} ({d.get('size', 0)} bytes; "
            f"{d.get('materialize', 'n/a')})",
            flush=True,
        )
    elif ev.kind == "test":
        ok = ev.data.get("ok", False)
        n_err = len(ev.data.get("errors", []))
        n_iss = len(ev.data.get("soft_warnings", []))
        tag = "TEST OK" if ok else "TEST FAILED"
        print(f"  {tag} ({n_err} error(s), {n_iss} issue(s))", flush=True)
        if not ok:
            print(f"  --- report ---\n{ev.text}\n  --- /report ---", flush=True)
    elif ev.kind == "question":
        print(f"\n? Model asks: {ev.text}", flush=True)
    elif ev.kind == "done":
        print(f"\nDONE — {ev.text}", flush=True)
    elif ev.kind == "error":
        print(f"\n! ERROR: {ev.text}", flush=True)
    elif ev.kind == "info":
        print(f"  i {ev.text}", flush=True)
    elif ev.kind == "plan":
        # Plan tokens already streamed via on_token; no-op here.
        pass


async def _run(
    goal: str,
    model: str,
    max_iters: int,
    out_path: Path,
    best_of_n: int,
    num_ctx: int,
    stall_seconds: float,
    headless: bool,
    open_when_done: bool,
    seed_file: Path | None,
) -> int:
    browser = LiveBrowser(viewport=(800, 600), run_seconds=3.0, headless=headless)
    try:
        await browser.start()
    except Exception as e:
        print(f"Could not launch Chromium: {e}", file=sys.stderr)
        print(
            "Hint: run `playwright install chromium` (and ensure a display "
            "for non-headless mode).",
            file=sys.stderr,
        )
        return 2

    agent = GameAgent(
        model=model,
        out_path=out_path,
        browser=browser,
        max_iters=max_iters,
        best_of_n=best_of_n,
        num_ctx=num_ctx,
        stall_seconds=stall_seconds,
        seed_file=seed_file,
    )

    # Stream tokens to stdout, one chunk at a time. Newlines flush.
    def on_token(piece: str) -> None:
        sys.stdout.write(piece)
        sys.stdout.flush()
    agent.set_token_callback(on_token)

    print(f"== Coding Box CLI · model={model} · headless={headless} · best-of-N={best_of_n}")
    print(f"== Goal: {goal}\n")
    rc = 0
    try:
        async for ev in agent.run(goal):
            _print_event(ev)
    except KeyboardInterrupt:
        print("\n^C — stopping.", file=sys.stderr)
        rc = 130
    except Exception as e:
        import traceback
        print(f"\nAgent crashed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        rc = 2
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    print(f"\nFinal game saved to: {out_path}", flush=True)
    if open_when_done and out_path.exists():
        webbrowser.open(f"file://{out_path.resolve()}")
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="Coding-box CLI driver (Ollama).")
    p.add_argument("goal", help="What game to build, in plain English.")
    p.add_argument("--model", default=MODEL, help=f"Ollama model tag (default: {MODEL})")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help="Output path. Default: games/<goal-slug>_<timestamp>.html "
             "(unique per run). Pass an explicit path to override.",
    )
    p.add_argument("--best-of-n", type=int, default=1,
                   help="Sample N candidates per fix, sequentially with early exit. "
                        "Default 1 (off). Set 2-3 to retry harder when local model is weak.")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="Ollama context window. Default 8192 matches the "
                        "default load size — anything different forces a "
                        "model reload on every request.")
    p.add_argument("--stall-seconds", type=float, default=90.0)
    p.add_argument("--headless", action="store_true", help="Run Chromium without a visible window.")
    p.add_argument("--open", action="store_true", help="Open final game in your browser.")
    p.add_argument(
        "--seed",
        default=None,
        help="Path to an existing .html file to start from. The agent will "
             "ADAPT it to your goal via patches instead of generating from "
             "scratch (memory skeleton is skipped).",
    )
    args = p.parse_args()

    seed_path: Path | None = None
    if args.seed:
        seed_path = Path(args.seed).expanduser()
        if not seed_path.is_file():
            print(f"--seed path is not a file: {seed_path}", file=sys.stderr)
            return 2

    return asyncio.run(_run(
        goal=args.goal,
        model=args.model,
        max_iters=args.max_iters,
        out_path=_resolve_out_path(args.out, args.goal),
        best_of_n=args.best_of_n,
        num_ctx=args.num_ctx,
        stall_seconds=args.stall_seconds,
        headless=args.headless,
        open_when_done=args.open,
        seed_file=seed_path,
    ))


if __name__ == "__main__":
    sys.exit(main())
