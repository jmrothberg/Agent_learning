"""chat.py - the interactive Textual TUI for the coding-box agent.

Run with NO arguments:
    .venv/bin/python chat.py

Layout (Claude-Code-style):
  ┌─ header (model, phase, iteration) ─────────────────────────────┐
  │┌── Agent log (60%) ─────────┐┌── Status (40%) ─────────────────┐│
  ││ streaming agent output     ││ current phase, last test report ││
  │└────────────────────────────┘└─────────────────────────────────┘│
  │ ┌── your message ──────────────────────────────────────────────┐│
  │ │> _                                                            ││
  │ └───────────────────────────────────────────────────────────────┘│
  └─ footer (key bindings) ────────────────────────────────────────┘

The actual playable game opens in a real Chromium window beside the terminal
(LiveBrowser, headless=False). You arrange the windows side by side.

Model selection (when you press Enter on your game idea):
  1. If env OLLAMA_MODEL (or CHAT_OLLAMA_MODEL) is set → use that tag.
  2. Else detect a loaded model, in order: Python `ollama.ps()`, then raw
     GET /api/ps, then the `ollama ps` shell command (first non-empty wins).
     That matches whatever you have in memory from `ollama run ...`.
  3. Else → fall back to coder.MODEL (the CLI default in coder.py).

Workflow:
  1. App launches, asks "What game do you want to build?".
  2. You type a description, press Enter.
  3. Agent starts. You see plan + code + test reports in the agent log.
  4. Type into the input box ANY time to give feedback - it's queued and
     injected at the next agent turn.
  5. If the model asks a <question>, the input box shows the question and
     waits for your reply.
  6. Press Ctrl+D when you're satisfied -> agent finishes the current turn
     cleanly and exits.
  7. Press Ctrl+Q to quit. (Avoid rebinding Ctrl+C — it can leave the shell
     without echo after exit; if that ever happens, run `reset` or `stty sane`.)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ollama
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from agent import AgentEvent, GameAgent
from coder import MODEL  # CLI / last-resort default when ps is empty and no env
from tools import LiveBrowser


# Where the game lives on disk. Same default as the CLI.
DEFAULT_OUT_PATH = Path("games/game.html")


def _ollama_ps_base_urls() -> list[str]:
    """Base URLs to try for GET /api/ps (deduped).

    IDE-launched Python often has a different view than your login shell: the
    daemon may still only listen on loopback, but OLLAMA_HOST can differ, and
    some setups bind IPv4 only or IPv6 only — so we probe several loopback URLs.
    """
    bases: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        raw = raw.strip().rstrip("/")
        if not raw:
            return
        if not raw.startswith("http"):
            raw = "http://" + raw
        if raw not in seen:
            seen.add(raw)
            bases.append(raw)

    env_host = (os.environ.get("OLLAMA_HOST") or "").strip()
    if env_host:
        add(env_host)
    # Always try these too (many users have no OLLAMA_HOST; ::1 vs 127.0.0.1 matters).
    add("127.0.0.1:11434")
    add("localhost:11434")
    add("[::1]:11434")
    return bases


def _http_get_models(base: str, endpoint: str) -> tuple[list[str], str | None]:
    """GET {base}{endpoint} → (names, err). Used for both /api/ps and /api/tags."""
    import json
    import urllib.error
    import urllib.request

    url = base.rstrip("/") + endpoint
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        return [], f"{url}: {e!r}"

    out: list[str] = []
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            continue
        tag = (m.get("name") or m.get("model") or "").strip()
        if tag:
            out.append(tag)
    return out, None


def _running_models_via_http_one(base: str) -> tuple[list[str], str | None]:
    """Currently-loaded (in-memory) models from /api/ps."""
    return _http_get_models(base, "/api/ps")


def _installed_models_via_http() -> tuple[list[str], str | None]:
    """Installed-on-disk models from /api/tags. Tries every loopback URL."""
    last_err: str | None = None
    for base in _ollama_ps_base_urls():
        names, err = _http_get_models(base, "/api/tags")
        if err:
            last_err = err
            continue
        if names:
            return names, None
    return [], last_err


def _ollama_cli_candidates() -> list[str]:
    """Resolve `ollama` executable — PATH alone fails inside Cursor for many users."""
    import shutil

    out: list[str] = []
    seen: set[str] = set()
    for c in (
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/snap/bin/ollama",
        os.path.expanduser("~/.local/bin/ollama"),
    ):
        if not c or c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            out.append(c)
    return out


def _running_models_via_cli() -> tuple[list[str], str | None]:
    """Parse `ollama ps` table via a real ollama binary. Returns (names, diag_or_None)."""
    import subprocess

    diags: list[str] = []
    for exe in _ollama_cli_candidates():
        try:
            r = subprocess.run(
                [exe, "ps"],
                capture_output=True,
                text=True,
                timeout=15,
                env=os.environ.copy(),
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            diags.append(f"{exe}: {e!r}")
            continue

        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()[:200]
            diags.append(f"{exe} exit {r.returncode}: {err!r}")
            continue

        lines = (r.stdout or "").strip().splitlines()
        if len(lines) < 2:
            diags.append(
                f"{exe}: no table rows (first line: {(lines[0] if lines else '')!r})"
            )
            continue

        names: list[str] = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            if name.upper() == "NAME":
                continue
            names.append(name)
        if names:
            return names, None
        diags.append(f"{exe}: table header only (no loaded models in ps)")

    if not diags:
        return [], "no `ollama` binary found (PATH + /usr/bin + /usr/local/bin + snap)"
    return [], " | ".join(diags[:5])


# Model tags that we know are broken on this machine (Ollama returns 500 for
# them). When falling back to "first installed", we skip these so we never pick
# a model that we already know cannot load.
_KNOWN_BROKEN_TAGS: set[str] = {"qwen3.6:27b"}


def _pick_first_workable(names: list[str]) -> str | None:
    """Return the first installed model that isn't in the broken blacklist."""
    for n in names:
        if n not in _KNOWN_BROKEN_TAGS:
            return n
    return None


def _running_models_all_sources() -> tuple[list[str], str, str]:
    """Return (tags, how_we_got_them, diagnostic_if_empty).

    We merge HTTP tries + CLI + Python client because IDE terminals often miss
    `ollama` on PATH, and loopback host can differ from the shell that ran
    `ollama run`.
    """
    diag_bits: list[str] = []

    # 1) Official Python client (uses OLLAMA_HOST internally — same as httpx default)
    try:
        ps = ollama.ps()
        rows = list(ps.models or [])
        names_py: list[str] = []
        for row in rows:
            tag = (getattr(row, "name", None) or getattr(row, "model", None) or "").strip()
            if tag:
                names_py.append(tag)
        if names_py:
            return names_py, "ollama Python client ps()", ""
    except Exception as e:
        diag_bits.append(f"Python ollama.ps(): {e!r}")

    # 2) Raw HTTP — try every loopback base (Ollama returns {"models":[...]})
    for base in _ollama_ps_base_urls():
        http_names, err = _running_models_via_http_one(base)
        if err:
            diag_bits.append(err)
        if http_names:
            return http_names, f"GET {base}/api/ps", ""

    # 3) Subprocess `ollama ps` — full path to binary, not only PATH
    cli_names, cli_err = _running_models_via_cli()
    if cli_err:
        diag_bits.append(cli_err)
    if cli_names:
        return cli_names, "`ollama ps` CLI (absolute path)", ""

    hint = ""
    if diag_bits:
        hint = " | ".join(diag_bits[:4])
        if len(hint) > 400:
            hint = hint[:400] + "..."
    return [], "no running models detected", hint


def resolve_chat_model(fallback: str) -> tuple[str, str]:
    """Pick which Ollama tag chat.py should use. Simple two-step:

      1. OLLAMA_MODEL env var wins, full stop.
      2. Else: ask Ollama which models are INSTALLED (/api/tags) and pick the
         first one that isn't on _KNOWN_BROKEN_TAGS. Ollama auto-loads it on
         the first chat request - we don't need to probe what's "currently in
         memory" or do anything clever.
      3. Else: fallback (coder.MODEL).

    Returns (model_name, source_label) for the TUI to log.
    """
    for key in ("OLLAMA_MODEL", "CHAT_OLLAMA_MODEL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw, f"{key} env"

    installed, err = _installed_models_via_http()
    if installed:
        chosen = _pick_first_workable(installed) or installed[0]
        return chosen, f"installed: [{', '.join(installed)}] — using {chosen!r}"

    fb = fallback.strip() or "llama3.2"
    if err:
        return fb, f"fallback {fallback!r} (could not reach Ollama: {err})"
    return fb, f"fallback {fallback!r}"


def _restore_terminal_state() -> None:
    """Best-effort tty cleanup after Textual exits (especially abrupt Ctrl+C).

    If Ctrl+C was bound to `exit()` the driver sometimes skipped restoring
    canonical mode — the shell then accepts keys but does not echo them.
    ANSI resets + `stty sane` fix that for most Linux terminals.
    """
    try:
        sys.stdout.write(
            "\x1b[?1049l"  # leave alternate screen
            "\x1b[?25h"  # show cursor
            "\x1b[?2004l"  # bracketed paste off
            "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l"  # mouse reporting off
            "\x1b[0m\r\n"
        )
        sys.stdout.flush()
    except Exception:
        pass
    try:
        if sys.stdin.isatty():
            subprocess.run(
                ["stty", "sane"],
                stdin=sys.stdin,
                capture_output=True,
                timeout=2,
                check=False,
            )
        else:
            with open("/dev/tty", "r") as tty:
                subprocess.run(
                    ["stty", "sane"],
                    stdin=tty,
                    capture_output=True,
                    timeout=2,
                    check=False,
                )
    except Exception:
        pass


class CodingBoxApp(App):
    """The TUI app. One instance == one session."""

    CSS = """
    Screen {
        background: $background;
    }

    #main {
        height: 1fr;
    }

    #log-pane {
        width: 60%;
        border: round $primary;
        padding: 0 1;
    }

    #status-pane {
        width: 40%;
        border: round $secondary;
        padding: 0 1;
    }

    #status-title {
        color: $accent;
        text-style: bold;
        padding: 0 0 1 0;
    }

    #input-row {
        height: 3;
        dock: bottom;
        border: round $accent;
        padding: 0 1;
    }

    Input {
        background: $surface;
    }
    """

    # Key bindings shown in the footer. action_xxx methods below implement them.
    # Do NOT bind ctrl+c to quit: that intercepts SIGINT-style exit and Textual
    # can skip full driver teardown, leaving the tty with echo disabled. Use
    # ctrl+q instead (common TUI convention).
    BINDINGS = [
        Binding("ctrl+d", "ship_it", "Ship game / done"),
        Binding("ctrl+q", "quit_app", "Quit"),
        # Ctrl+L: re-print where the FULL log files live. Useful when you
        # want to `cat` them from another terminal to share with an LLM.
        Binding("ctrl+l", "show_log_paths", "Show log paths"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: GameAgent | None = None
        self.browser: LiveBrowser | None = None
        # awaiting_kind tells us how to interpret the next Input submission:
        #   "goal"     -> the very first message (the game description)
        #   "answer"   -> a reply to the model's <question>
        #   "feedback" -> free-form mid-run feedback
        self._awaiting_kind: str = "goal"
        self._goal: str | None = None
        self._iteration_label = "—"
        self._phase_label = "waiting for goal"
        # Filled in _start_session after resolve_chat_model().
        self._session_model: str | None = None
        # Plain-text mirror of the agent log pane. Opened lazily in
        # _start_session so we know the games/ folder exists. None means
        # "not yet open" - _log() handles that gracefully.
        self._log_file_handle = None
        self._log_file_path: Path | None = None

    # ----------------------------- layout ---------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            with Horizontal():
                yield RichLog(id="log-pane", wrap=True, markup=True, highlight=False)
                with Vertical(id="status-pane"):
                    yield Static("Status", id="status-title")
                    yield Static("", id="status-body")
        # Input row docked to the bottom. We start it empty with a goal prompt.
        yield Input(placeholder="What game do you want to build?", id="user-input")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Coding Box"
        self.sub_title = "type your game idea below, then Enter"
        self._update_status()
        # Show what model we'll use when the user submits a goal. One call,
        # one line. Override with OLLAMA_MODEL env var if you want.
        preview_model, preview_src = resolve_chat_model(MODEL)
        self._log_info(f"Will use model: [b]{preview_model}[/b] [dim]({preview_src})[/dim]")
        if _KNOWN_BROKEN_TAGS:
            self._log_info(
                f"[dim]Skipping known-broken tags: {sorted(_KNOWN_BROKEN_TAGS)}. "
                "Set OLLAMA_MODEL=<tag> to override.[/dim]"
            )
        self._log_info("Type your game idea in the input box below and press Enter.")
        self._log_info(
            "[dim]Keys: Ctrl+D ship · Ctrl+L log paths · Ctrl+Q quit · "
            "if the shell stops echoing after exit, run `reset`.[/dim]"
        )
        self._log_info(
            "[dim]Cut/paste: hold [b]Shift[/b] while click-dragging in the agent "
            "log pane (this bypasses Textual's mouse capture so your terminal can "
            "select text). Then Ctrl+Shift+C to copy. Or just `cat` the .log file "
            "from another terminal - path appears in the right pane.[/dim]"
        )
        self.query_one(Input).focus()

    # ----------------------------- helpers --------------------------------

    # Strip Rich/Textual markup from a string so the file mirror is plain text.
    # Pattern matches `[tag]`, `[/tag]`, `[tag=value]`, etc - same syntax Rich
    # uses for inline styling. We only strip from the FILE copy; the TUI still
    # gets the colored version.
    _MARKUP_RE = __import__("re").compile(r"\[/?[a-zA-Z][^\[\]]*\]")

    def _log(self, text: str) -> None:
        """Append a line to the agent log pane AND mirror to the .log file."""
        self.query_one("#log-pane", RichLog).write(text)
        if self._log_file_handle is not None:
            try:
                plain = self._MARKUP_RE.sub("", text)
                self._log_file_handle.write(plain.rstrip() + "\n")
                self._log_file_handle.flush()  # so `tail -f` sees it live
            except Exception:
                # Mirror must never crash the TUI.
                pass

    def _log_info(self, text: str) -> None:
        self._log(f"[cyan]i[/cyan] {text}")

    def _log_error(self, text: str) -> None:
        self._log(f"[red]![/red] {text}")

    # Streaming tokens from the model arrive one-piece-at-a-time. Textual's
    # RichLog appends per-call as a new line, which would break sentences.
    # We accumulate into a buffer and flush on newlines (and also on the
    # final post-stream "settle" tick).
    _stream_buf: str = ""

    def _emit_token(self, piece: str) -> None:
        # Called from inside the agent's async loop (same event loop as the
        # TUI), so this is safe without explicit thread-marshaling.
        self._stream_buf += piece
        # Flush at any newline boundary so the user sees lines as they arrive.
        if "\n" in self._stream_buf:
            *complete, self._stream_buf = self._stream_buf.split("\n")
            for line in complete:
                if line.strip():
                    self._log(line)

    def _flush_stream(self) -> None:
        """Push any remaining buffered tokens (no trailing newline)."""
        if self._stream_buf.strip():
            self._log(self._stream_buf)
        self._stream_buf = ""

    def _update_status(self, extra: str = "") -> None:
        """Render the right-hand status panel."""
        body = (
            f"[b]Phase:[/b] {self._phase_label}\n"
            f"[b]Iteration:[/b] {self._iteration_label}\n"
        )
        if self._session_model:
            body += f"[b]Model:[/b] {self._session_model}\n"
        body += f"[b]Goal:[/b] {self._goal or '—'}\n"
        # Show the FULL log path persistently so the user always knows what
        # file to `cat` / share when something goes wrong.
        if self._log_file_path is not None:
            body += f"\n[b]Log:[/b] [dim]{self._log_file_path}[/dim]\n"
            body += "[dim]Ctrl+L to reprint paths[/dim]\n"
        if extra:
            body += "\n" + extra
        self.query_one("#status-body", Static).update(body)

    # ----------------------------- actions --------------------------------

    async def action_ship_it(self) -> None:
        """Ctrl+D - tell the agent the human is satisfied."""
        if self.agent is None:
            return
        self.agent.request_done()
        self._log_info("[yellow]Ship requested.[/yellow] Agent will finish current turn and exit.")

    async def action_quit_app(self) -> None:
        """Ctrl+Q — quit (browser cleanup happens in on_unmount)."""
        self.exit()

    async def action_show_log_paths(self) -> None:
        """Ctrl+L - print every artifact path so the user can `cat` them."""
        if self._log_file_path is None:
            self._log_info("[dim]no session active yet - paths appear after you submit a goal[/dim]")
            return
        ts_stem = self._log_file_path.stem  # e.g. 20260503_160301
        traces = self._log_file_path.parent
        snaps = DEFAULT_OUT_PATH.parent / "snapshots" / ts_stem
        self._log("[bold cyan]── log artifacts ──[/bold cyan]")
        self._log(f"  full log:     {self._log_file_path}")
        self._log(f"  jsonl trace:  {traces / (ts_stem + '.jsonl')}")
        self._log(f"  conversation: {traces / (ts_stem + '.conversation.md')}")
        self._log(f"  snapshots:    {snaps}")
        self._log(f"  best clean:   {DEFAULT_OUT_PATH.parent / 'best.html'}")
        self._log("[dim]Tip: paste the full log above into your AI assistant to debug.[/dim]")

    # ----------------------------- input handler --------------------------

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Single dispatch point for the user's bottom input box."""
        text = (message.value or "").strip()
        message.input.value = ""
        if not text:
            return

        if self._awaiting_kind == "goal":
            # First message: the game description. Spin up the agent + browser.
            self._goal = text
            self._log(f"[bold green]>[/bold green] {text}")
            message.input.placeholder = "type feedback any time, or Ctrl+D when done"
            self.sub_title = "agent is working"
            await self._start_session(text)
            self._awaiting_kind = "feedback"

        elif self._awaiting_kind == "answer":
            self._log(f"[bold magenta]> answer:[/bold magenta] {text}")
            if self.agent is not None:
                self.agent.add_user_answer(text)
            message.input.placeholder = "type feedback any time, or Ctrl+D when done"
            self._awaiting_kind = "feedback"

        else:  # "feedback"
            self._log(f"[bold blue]> feedback:[/bold blue] {text}")
            if self.agent is not None:
                self.agent.add_user_feedback(text)
                # Tell the user CLEARLY their words were captured. Otherwise
                # they assume the input field swallowed the text.
                pending = len(self.agent._pending_feedback)
                self._log(
                    f"[dim cyan]  ✓ queued (pending: {pending}). "
                    f"Will be applied at the next user-turn boundary - watch "
                    f"for an [italic]→ applying your input[/italic] line.[/dim cyan]"
                )
            else:
                self._log("[dim red]  (no active agent - feedback ignored)[/dim red]")

    # ----------------------------- session --------------------------------

    async def _start_session(self, goal: str) -> None:
        """Boot the LiveBrowser + GameAgent and start consuming events."""
        self._phase_label = "starting browser"
        self._update_status()

        self.browser = LiveBrowser(viewport=(800, 600), run_seconds=3.0)
        try:
            await self.browser.start()
        except Exception as e:
            self._log_error(
                f"Could not launch Chromium: {e}\n"
                "Make sure you ran `playwright install chromium` and that you "
                "have a graphical display (this needs headless=False)."
            )
            return

        # Resolve tag here (not at app launch) so you can `ollama run mymodel`
        # in another terminal while the TUI is open, then submit your goal.
        model_name, model_src = resolve_chat_model(MODEL)
        self._session_model = model_name
        self.title = f"Coding Box - {model_name}"
        self._log_info(f"Using model [b]{model_name}[/b] [dim]({model_src})[/dim]")
        self._update_status()

        self.agent = GameAgent(
            model=model_name,
            out_path=DEFAULT_OUT_PATH,
            browser=self.browser,
            max_iters=6,
        )
        self.agent.set_token_callback(self._emit_token)

        # Open the plain-text log mirror. We use the same timestamp dir the
        # agent uses so .jsonl + .log live side by side and are easy to pair.
        try:
            from datetime import datetime as _dt

            log_dir = DEFAULT_OUT_PATH.parent / "traces"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            self._log_file_path = log_dir / f"{ts}.log"
            self._log_file_handle = self._log_file_path.open("w", encoding="utf-8")
            self._log_info(f"Mirroring log to: {self._log_file_path}")
        except Exception as e:
            self._log_info(f"[dim]could not open log mirror: {e}[/dim]")

        # Spawn the agent loop as a background task so the TUI stays responsive.
        self.run_worker(self._consume_events(goal), exclusive=True)

    async def _consume_events(self, goal: str) -> None:
        """Drain the AgentEvent stream and update widgets accordingly."""
        assert self.agent is not None
        try:
            async for ev in self.agent.run(goal):
                self._handle_event(ev)
        except Exception as e:
            # Include the FULL traceback so the .log file has enough info to
            # debug the crash without re-running. Plain string only - the
            # markup-stripping mirror handles either way.
            import traceback
            tb = traceback.format_exc()
            self._log_error(f"Agent crashed: {e}")
            self._log(f"[dim red]{tb}[/dim red]")
        finally:
            self._flush_stream()
            self._phase_label = "finished"
            self._update_status()
            self.sub_title = "session ended - Ctrl+Q to quit"
            # Always end with a clear footer pointing at the log files so the
            # user can paste them to an AI for debugging.
            await self.action_show_log_paths()

    def _handle_event(self, ev: AgentEvent) -> None:
        """Pattern-match on event kind and update the UI."""
        # Always flush any half-streamed line before logging a new event header.
        self._flush_stream()

        if ev.kind == "phase":
            # "planning", "iteration N/M", "self-critique"
            self._phase_label = ev.text
            if ev.text.startswith("iteration"):
                self._iteration_label = ev.text.replace("iteration ", "")
            self._update_status()
            self._log(f"\n[bold yellow]── {ev.text} ──[/bold yellow]")

        elif ev.kind == "plan":
            # The full plan is already in the log via streaming tokens; just
            # mark the boundary so the user knows phase A is done.
            self._log("[dim](plan complete)[/dim]")

        elif ev.kind == "code":
            self._log(f"[green]wrote {ev.text}[/green] ({ev.data.get('size', 0)} bytes)")

        elif ev.kind == "test":
            # ev.text is the human-readable report, ev.data is the dict.
            ok = ev.data.get("ok", False)
            tag = "[green]TEST OK[/green]" if ok else "[red]TEST FAILED[/red]"
            n_err = len(ev.data.get("errors", []))
            n_iss = len(ev.data.get("soft_warnings", []))
            self._log(f"{tag} ({n_err} error(s), {n_iss} issue(s))")
            # Also drop the full report into the right-hand status panel.
            self._update_status(extra=f"[b]Last test:[/b]\n{ev.text}")

        elif ev.kind == "question":
            self._log(f"\n[bold magenta]?[/bold magenta] [bold]Model asks:[/bold] {ev.text}")
            self._awaiting_kind = "answer"
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "type your answer and press Enter"
            inp.focus()

        elif ev.kind == "done":
            self._log(f"\n[bold green]DONE[/bold green] - {ev.text}")
            self._log(f"[dim]Final game: {DEFAULT_OUT_PATH.resolve()}[/dim]")
            # Surface the auto-saved best version + trace location so the user
            # knows where to look if they want to inspect or recover.
            best = DEFAULT_OUT_PATH.parent / "best.html"
            if best.exists():
                self._log(f"[dim]Best clean version: {best.resolve()}[/dim]")
            traces_dir = DEFAULT_OUT_PATH.parent / "traces"
            if traces_dir.exists():
                self._log(f"[dim]Trace logs: {traces_dir.resolve()}[/dim]")

        elif ev.kind == "error":
            self._log_error(ev.text)

        elif ev.kind == "info":
            self._log_info(ev.text)

    # ----------------------------- shutdown -------------------------------

    async def on_unmount(self) -> None:
        """Tear down the browser cleanly when the app exits (Ctrl+C / quit)."""
        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception:
                # Best-effort - the process is exiting anyway.
                pass
        # Close the plain-text log mirror, if open.
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None


def main() -> int:
    try:
        CodingBoxApp().run()
    finally:
        _restore_terminal_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
