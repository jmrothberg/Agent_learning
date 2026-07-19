"""coder.py — non-interactive CLI driver for the GameAgent.

Usage:
    python coder.py "Make a Snake game with a wraparound board and a score counter"
    python coder.py "snake" --max-iters 4 --best-of-n 3 --headless

Optional flags:
    --backend BACKEND   Pick LLM daemon: mlx (default on macOS) | auto |
                        ollama. Default follows LLM_BACKEND if set, else
                        MLX on Mac else auto. 'auto' probes both; MLX wins
                        ties when both have a model loaded.
    --model NAME        Override the model id resolved by backend detection.
                        (Ollama tag like 'qwen3.6:27b', or MLX local/HF path like
                        '/Users/jonathanrothberg_1/MLX_Models/Qwen3.6-27B-mxfp8'.)
    --max-iters N       Cap iterations (default 6)
    --out PATH          Where to save the final game (default games/game.html)
    --best-of-n N       Sample N candidates per fix turn (default 1)
    --stuck-bon         Enable automatic stuck best-of-2 escalation (default off)
    --num-ctx N         Ollama context window (default 100000; env CODING_BOX_NUM_CTX)
    --stall-seconds N   Per-stream no-activity watchdog (default 300; fail-open floor)
    --headless          Run Chromium headless (no visible window). Use this
                        for unattended runs and CI; the TUI uses visible.
    --open              After finishing, open the result in your real browser.

This driver shares its agent core (agent.py) with the TUI in chat.py, so
behavior matches what you'd see interactively.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import os
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

# Load .env (gitignored, chmod 600) so cloud-backend keys are visible
# without manual shell export. Mirrors the loader in chat.py.
try:
    from dotenv import load_dotenv
    # override=True so .env wins over empty/stale shell vars; matches chat.py.
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    if (Path(__file__).resolve().parent / ".env").exists():
        print(
            "note: .env present but python-dotenv not installed; "
            "run `.venv/bin/pip install python-dotenv` or export keys "
            "manually in your shell.",
            file=sys.stderr,
        )

import backend as backend_mod
from agent import AgentEvent, GameAgent, default_num_ctx
from overlay_identity import PRODUCT_NAME
from tools import LiveBrowser

# Sentinel: --out was not supplied → derive a unique meaningful name from
# the goal (instead of clobbering games/game.html every run).
_DEFAULT_OUT = "games/game.html"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_stdout_broken = False


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


def _stdout_print(*args: object, **kwargs: object) -> None:
    """Print unless the stdout reader has closed its pipe.

    CLI output is observational: a detached batch reader must not abort the
    agent before an already-generated reply can be materialized to disk.
    """
    global _stdout_broken
    if _stdout_broken:
        return
    try:
        print(*args, **kwargs)
    except OSError as exc:
        if exc.errno != errno.EPIPE:
            raise
        _stdout_broken = True


def _stdout_write(piece: str) -> None:
    """Write streamed tokens with the same closed-pipe handling as events."""
    global _stdout_broken
    if _stdout_broken:
        return
    try:
        sys.stdout.write(piece)
        sys.stdout.flush()
    except OSError as exc:
        if exc.errno != errno.EPIPE:
            raise
        _stdout_broken = True


def _print_event(ev: AgentEvent) -> None:
    """Single line per event, matched by `tail -f games/traces/*.log`."""
    if ev.kind == "phase":
        _stdout_print(f"\n── {ev.text} ──", flush=True)
    elif ev.kind == "memory":
        _stdout_print(f"  memory: {ev.text}", flush=True)
    elif ev.kind == "best_of_n":
        _stdout_print(f"  best-of-N: {ev.text}", flush=True)
    elif ev.kind == "diagnose":
        _stdout_print(f"  diagnose: {ev.text[:200]}", flush=True)
    elif ev.kind == "code":
        d = ev.data
        _stdout_print(
            f"  wrote {ev.text} ({d.get('size', 0)} bytes; "
            f"{d.get('materialize', 'n/a')})",
            flush=True,
        )
    elif ev.kind == "test":
        ok = ev.data.get("ok", False)
        n_err = len(ev.data.get("errors", []))
        n_iss = len(ev.data.get("soft_warnings", []))
        tag = "TEST OK" if ok else "TEST FAILED"
        _stdout_print(f"  {tag} ({n_err} error(s), {n_iss} issue(s))", flush=True)
        if not ok:
            _stdout_print(
                f"  --- report ---\n{ev.text}\n  --- /report ---", flush=True
            )
    elif ev.kind == "question":
        _stdout_print(f"\n? Model asks: {ev.text}", flush=True)
    elif ev.kind == "done":
        _stdout_print(f"\nDONE — {ev.text}", flush=True)
    elif ev.kind == "error":
        _stdout_print(f"\n! ERROR: {ev.text}", flush=True)
    elif ev.kind == "info":
        _stdout_print(f"  i {ev.text}", flush=True)
    elif ev.kind == "restart":
        _stdout_print(f"  ↻ {ev.text}", flush=True)
    elif ev.kind == "mlx_stall":
        _stdout_print(f"\n! STALL: {ev.text}", flush=True)
    elif ev.kind == "plan":
        # Plan tokens already streamed via on_token; no-op here.
        pass
    elif ev.kind == "await_user":
        # Step-mode pause banner. Actual stdin read happens in the event
        # consumer below so it doesn't block the streaming printer here.
        _stdout_print(f"\n⏸  {ev.text}", flush=True)


async def _run(
    goal: str,
    backend_pref: str,
    model_override: str | None,
    max_iters: int,
    out_path: Path,
    best_of_n: int,
    num_ctx: int,
    stall_seconds: float,
    headless: bool,
    open_when_done: bool,
    seed_file: Path | None,
    step: bool = False,
    model_class: str = "auto",
    restart_n: int = 1,
    restart_threshold: float = 60.0,
    playbook_on: bool = True,
    playbook_writeback: bool = True,
    use_vlm_critique: bool = False,
    stuck_bon_enabled: bool = False,
    no_auto_step: bool = False,
) -> int:
    # Resolve which LLM daemon we'll talk to. --backend overrides the
    # LLM_BACKEND env. If --model was given, build a BackendInfo
    # directly with that model (still resolved against the chosen
    # daemon's endpoint).
    try:
        info = backend_mod.detect_backend(backend_pref)
    except RuntimeError as e:
        print(f"backend resolution failed: {e}", file=sys.stderr)
        return 2
    if model_override:
        info = backend_mod.BackendInfo(
            name=info.name, model=model_override,
            source=f"--model {model_override!r}",
            endpoint=info.endpoint,
        )
    backend_inst = backend_mod.make_backend(info)

    # Preload diffusers (Stable Audio; Z-Image when not on FLUX2 klein)
    # BEFORE Playwright opens IPC pipes — same fds_to_keep trap as chat.py.
    if os.environ.get("SKIP_DIFFUSER_PRELOAD", "").strip() not in ("", "0", "false", "False"):
        pass
    else:
        try:
            import assets as _assets
            _assets.preload()
        except Exception:
            pass

    browser = LiveBrowser(run_seconds=3.0, headless=headless)
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
        backend=backend_inst,
        out_path=out_path,
        browser=browser,
        max_iters=max_iters,
        best_of_n=best_of_n,
        num_ctx=num_ctx,
        stall_seconds=stall_seconds,
        seed_file=seed_file,
        # Default to v1 prompt: <playbook> + <criteria> + <probes>.
        prompt_version="v1",
        # todo #6 — mid-tier prompt trim. "auto" infers from model
        # tag; pass --model-class mid|large to override.
        model_class=model_class,
        restart_n=restart_n,
        restart_score_threshold=restart_threshold,
        # K=0 disables retrieval entirely. Default is ON (TUI parity) —
        # opt out via --no-playbook. Writeback stays on so session
        # outcomes keep updating the bullet counters.
        playbook_top_k=6 if playbook_on else 0,
        playbook_writeback=playbook_writeback,
        use_vlm_critique=use_vlm_critique,
        stuck_bon_enabled=stuck_bon_enabled,
    )

    # Stream tokens to stdout, one chunk at a time. Newlines flush.
    def on_token(piece: str) -> None:
        _stdout_write(piece)
    agent.set_token_callback(on_token)
    # Step-mode (Stop-Losing-To-OneShot todo #1): pause after each iter
    # and read stdin before continuing. Off by default — autonomous
    # behavior is unchanged unless --step is passed.
    if step:
        agent.set_step_mode(True)
    # Unattended runs (headless batch, tune_serial_loop) must not auto-arm
    # step-mode on first test failure — await_user would block on stdin.
    if headless or no_auto_step:
        agent.set_auto_step_on_failure(False)

    _stdout_print(
        f"== {PRODUCT_NAME} CLI · {info.name.upper()}={info.model} "
        f"[{info.source}] · headless={headless} · "
        f"vlm-critique={use_vlm_critique} · "
        f"best-of-N={best_of_n} · stuck-bon={stuck_bon_enabled} · step={step}"
    )
    _stdout_print(f"== Goal: {goal}\n")
    rc = 0
    try:
        async for ev in agent.run_with_restarts(goal):
            _print_event(ev)
            # Step-mode: when the agent emits await_user it has parked
            # in an asyncio.sleep loop waiting for either
            # signal_step_continue() or new feedback. Read one line
            # from stdin in a thread so the asyncio loop keeps ticking,
            # then either signal continue (empty) or queue feedback.
            if ev.kind == "await_user":
                loop = asyncio.get_running_loop()
                try:
                    line = await loop.run_in_executor(
                        None, input,
                        "[step-mode] Enter to continue, or type feedback: ",
                    )
                except EOFError:
                    line = ""
                line = (line or "").strip()
                if line:
                    agent.add_user_feedback(line)
                else:
                    agent.signal_step_continue()
    except KeyboardInterrupt:
        print("\n^C — stopping.", file=sys.stderr)
        rc = 130
    except Exception as e:
        import traceback
        try:
            agent._trace_agent_crash(e, source="coder_cli")
        except Exception:
            pass
        print(f"\nAgent crashed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        rc = 2
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    best_path = out_path.with_name(out_path.stem + ".best.html")
    shipped = best_path.is_file() and best_path.stat().st_size > 500
    if rc == 0 and not shipped:
        rc = 1
    if shipped or out_path.is_file():
        _stdout_print(f"\nFinal game saved to: {out_path}", flush=True)
    else:
        _stdout_print(
            f"\nNo game shipped (missing {best_path.name})", flush=True
        )
    if open_when_done and out_path.exists():
        webbrowser.open(f"file://{out_path.resolve()}")
    # Unattended serial/batch (--no-auto-step): skip interpreter shutdown.
    # torch/MLX/diffusers atexit can SIGSEGV after an otherwise clean run
    # (tune_serial10 run_03 game 1: exit=-11, best.html OK).
    if no_auto_step:
        os._exit(rc)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(
        description="Coding-box CLI driver (defaults to MLX on macOS, else auto)."
    )
    p.add_argument("goal", help="What game to build, in plain English.")
    p.add_argument(
        "--backend",
        choices=["auto", "ollama", "mlx", "mlx-server"],
        default=os.environ.get("LLM_BACKEND")
        or ("mlx" if sys.platform == "darwin" else "auto"),
        help="LLM daemon. Default: LLM_BACKEND env if set, else mlx on macOS "
             "else auto. 'mlx-server' / MLX_SERVER_URL → batched mlx_lm.server; "
             "'mlx' → in-process (TUI default). 'auto' probes both (MLX wins ties).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the model id resolved by backend detection. "
             "Ollama tag (e.g. 'qwen3.6:27b') or MLX HF id "
             "(e.g. local dir '/Users/jonathanrothberg_1/MLX_Models/Qwen3.6-27B-mxfp8' "
             "or an HF model id). "
             "When omitted, backend.detect_backend picks for you.",
    )
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help="Output path. Default: games/<goal-slug>_<timestamp>.html "
             "(unique per run). Pass an explicit path to override.",
    )
    p.add_argument(
        "--playbook",
        action="store_true",
        default=True,
        help="Playbook bullet retrieval. ON by default so CLI/headless "
             "builds get the same accumulated rules the TUI uses (driver "
             "parity). Pass --no-playbook to disable.",
    )
    p.add_argument(
        "--no-playbook",
        action="store_false",
        dest="playbook",
        # Historical note (kept for the next LLM): an early bench on a 27B
        # local model had OFF beating ON 5/6 on short-loop runs. That was
        # before the playbook was curated and before plan/first-build/fix
        # injection was tuned; the TUI has defaulted ON since, so CLI now
        # matches it. Use --no-playbook for A/B baselines or very tight loops.
        help="Disable playbook retrieval (A/B baselines or tight loops).",
    )
    p.add_argument(
        "--playbook-writeback",
        action="store_true",
        default=True,
        help="Update bullet helpful/harmful counters based on pass / "
             "stuck-streak outcomes. ON by default — sessions teach "
             "the playbook over time. Pass --no-playbook-writeback to "
             "freeze scores (e.g. for tune-battery A/B baselines).",
    )
    p.add_argument(
        "--no-playbook-writeback",
        dest="playbook_writeback",
        action="store_false",
        help="Freeze playbook scores (for A/B baseline comparisons).",
    )
    p.add_argument("--best-of-n", type=int, default=1,
                   help="Sample N candidates per fix, sequentially with early exit "
                        "(score=100 ships immediately). Default 1 — multi-candidate "
                        "fan-out oversubscribed the 4-GPU box on first build "
                        "(2026-05-23). Opt in with --best-of-n 2 or 3 when you have "
                        "spare staged slots.")
    p.add_argument(
        "--stuck-bon",
        action="store_true",
        help="Enable automatic stuck best-of-2 escalation after 2+ failed "
             "iters (cap 2/session). Default off — same as TUI without "
             "/bestof on. Separate from --best-of-n.",
    )
    p.add_argument("--num-ctx", type=int, default=None,
                   help="Ollama context window. Default 100000 (or "
                        "CODING_BOX_NUM_CTX env: supports 100k, 262k, full). "
                        "KV-cache scales linearly with ctx — raise to 262144 "
                        "only for long extension chats. Changing forces an "
                        "Ollama reload; preload with "
                        "`ollama run --ctx-size N <model>`.")
    p.add_argument("--stall-seconds", type=float, default=600.0,
                   help="Per-stream no-activity stall budget (default 600s = "
                        "10 min). Activity-aware: prefill progress chunks and "
                        "every emitted token reset the timer, so this is a "
                        "quiet-window budget, not a cold-start budget. Same "
                        "value for every model — no bracket table.")
    p.add_argument(
        "--vlm-critique",
        action="store_true",
        help="Enable structured visual critic (/vlm-critique parity with TUI). "
             "Requires a VLM-capable backend (in-process --backend mlx with mlx_vlm).",
    )
    p.add_argument(
        "--no-auto-step",
        action="store_true",
        help="Do not auto-arm step-mode on first test failure (unattended "
             "serial/batch runs with visible browser).",
    )
    p.add_argument("--headless", action="store_true", help="Run Chromium without a visible window.")
    p.add_argument("--open", action="store_true", help="Open final game in your browser.")
    p.add_argument(
        "--seed",
        default=None,
        help="Path to an existing .html file to start from. The agent will "
             "ADAPT it to your goal via patches instead of generating from "
             "scratch (memory skeleton is skipped).",
    )
    # Step-mode (Stop-Losing-To-OneShot todo #1): pause between iters.
    p.add_argument(
        "--step",
        action="store_true",
        help="Step-mode: pause after each iteration and wait for stdin "
             "input (Enter to continue, or type feedback) before the next "
             "model turn. Use to manually verify each iter on mid-tier "
             "models that the autonomous harness can't fully grade.",
    )
    # Model-class override (Stop-Losing-To-OneShot todo #6).
    p.add_argument(
        "--model-class",
        choices=["auto", "small", "mid", "large"],
        default="auto",
        help="System-prompt trim. 'auto' (default) = 'small': lean "
             "~5 KB schema, drops <assets>/<sounds>/<lookup_bullet> — "
             "biased for mid-size local LLMs and one-shot strength. "
             "Pass 'large' only when running a frontier-tier model. "
             "Model names are NEVER inspected.",
    )
    # Restart-N (Stop-Losing-To-OneShot Track A): when iter 1 ends with
    # a low score, restart from scratch instead of polishing a stinker.
    p.add_argument(
        "--restart-n",
        type=int,
        default=2,
        help="Independent full restarts. When iter 1 of a session "
             "produces a score below --restart-threshold, throw it away "
             "and try again from scratch (Z-Image asset cache is reused). "
             "Best-by-score wins. Default 2 (simple games pass iter 1 "
             "and never restart; hard games get a clean second attempt). "
             "Set to 1 to disable.",
    )
    p.add_argument(
        "--restart-threshold",
        type=float,
        default=60.0,
        help="Score floor (0-100) below which iter 1 triggers a restart. "
             "Default 60: 'applied but several errors' restarts; 'applied "
             "and partial probes pass' continues iterating.",
    )
    args = p.parse_args()

    seed_path: Path | None = None
    if args.seed:
        seed_path = Path(args.seed).expanduser()
        if not seed_path.is_file():
            print(f"--seed path is not a file: {seed_path}", file=sys.stderr)
            return 2

    resolved_ctx = args.num_ctx if args.num_ctx is not None else default_num_ctx()

    return asyncio.run(_run(
        goal=args.goal,
        backend_pref=args.backend,
        model_override=args.model,
        max_iters=args.max_iters,
        out_path=_resolve_out_path(args.out, args.goal),
        best_of_n=args.best_of_n,
        num_ctx=resolved_ctx,
        stall_seconds=args.stall_seconds,
        headless=args.headless,
        open_when_done=args.open,
        seed_file=seed_path,
        step=args.step,
        model_class=args.model_class,
        restart_n=args.restart_n,
        restart_threshold=args.restart_threshold,
        playbook_on=args.playbook,
        playbook_writeback=args.playbook_writeback,
        use_vlm_critique=args.vlm_critique,
        stuck_bon_enabled=args.stuck_bon,
        no_auto_step=args.no_auto_step,
    ))


if __name__ == "__main__":
    sys.exit(main())
