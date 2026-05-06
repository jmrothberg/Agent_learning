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
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import ollama
from rich.markup import escape as _esc
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from agent import AgentEvent, GameAgent
from coder import MODEL  # CLI / last-resort default when ps is empty and no env
from tools import LiveBrowser


# Parent directory for all generated artifacts. Each session writes a unique
# file inside here named "<goal-slug>_<timestamp>.html" so prior runs are
# never overwritten. Snapshots, traces, logs and best.html all derive from
# the same stem (see agent.py).
GAMES_DIR = Path("games")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Short, unambiguous "I'm satisfied" phrases — typed at the input box these
# trigger a ship (= same effect as Ctrl+D / /ship) instead of being queued
# as feedback for the model. Match is case-insensitive, strips trailing
# punctuation/whitespace, and requires an EXACT full-string match — so
# "done" ships but "almost done, add sound" goes through as feedback.
_SHIP_PHRASES: frozenset[str] = frozenset({
    "done", "ok", "okay", "ok done", "okay done", "ok, done", "okay, done",
    "im done", "i'm done", "we're done", "we are done", "all done",
    "ship", "ship it", "ship it!", "ship now", "ship them", "deploy",
    "looks good", "lgtm", "looks great", "looks fine",
    "perfect", "great", "nice", "good", "good enough", "fine",
    "stop", "stop it", "finish", "finished",
    "yes", "yes ship", "yep", "yep done",
})


def _looks_like_ship(text: str) -> bool:
    """Detect a clear ship-now intent without false-positives on real feedback.

    Conservative on purpose: only matches *exact* (after normalizing case,
    whitespace, and trailing .!?) full-string matches in `_SHIP_PHRASES`.
    A user who types "ok done now add sound" still gets treated as feedback;
    "ok done" by itself ships.
    """
    s = " ".join(text.strip().lower().rstrip(".!?").split())
    return s in _SHIP_PHRASES


def _slugify(text: str, max_len: int = 30) -> str:
    """Compact, filename-safe stem from a free-form goal."""
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    if not s:
        s = "game"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-") or "game"
    return s


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


def _running_models_with_meta() -> list[dict]:
    """Currently-loaded models from /api/ps WITH metadata.

    Ollama renews each loaded model's `expires_at` (TTL) on every use, so the
    record with the latest `expires_at` is the one you most recently talked
    to — which is what people mean when they say "I have ollama running X".
    Tries every loopback base; first reachable one wins.

    Records also include parameter_size when Ollama reports it — used to
    auto-scale the streaming budget in resolve_session_timeouts().
    """
    import json
    import urllib.error
    import urllib.request

    for base in _ollama_ps_base_urls():
        url = base.rstrip("/") + "/api/ps"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="GET"), timeout=5
            ) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            continue
        out: list[dict] = []
        for m in data.get("models") or []:
            if not isinstance(m, dict):
                continue
            name = (m.get("name") or m.get("model") or "").strip()
            if not name:
                continue
            details = m.get("details") or {}
            out.append({
                "name": name,
                "expires_at": m.get("expires_at") or "",
                "parameter_size": (details.get("parameter_size") or "").strip(),
                "context_length": m.get("context_length") or 0,
            })
        if out:
            return out
    return []


def _parse_param_billions(p: str) -> float:
    """'20.9B' -> 20.9; '7B' -> 7.0; '36.0B' -> 36.0; '' -> 0.0."""
    s = (p or "").strip().upper().rstrip("B")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _model_param_size(model: str) -> str:
    """Best-effort parameter_size string for a model (e.g. '36.0B').

    Tries /api/ps first (loaded models, fast). Falls back to ollama.show()
    (works for installed-but-not-loaded models — that's the common case
    when the user hasn't run the model yet). Returns '' if both fail.
    """
    for m in _running_models_with_meta():
        if m["name"] == model and m.get("parameter_size"):
            return m["parameter_size"]
    try:
        info = ollama.show(model=model)
        details = getattr(info, "details", None)
        if details is not None:
            return (getattr(details, "parameter_size", "") or "").strip()
    except Exception:
        pass
    return ""


def resolve_session_timeouts(model: str) -> tuple[float, float]:
    """Pick (stall_seconds, overall_seconds) for a given model.

    Scaling is by parameter count (queried from /api/ps then /api/show).
    Larger models take longer per token AND tend to write more verbose
    output, so we bump BOTH timeouts:

        params      stall    overall
        ─────────   ─────    ───────
        ≤ 13B       60       600    (small/fast, default-ish)
        14–25B      90       900    (gpt-oss 20B ballpark)
        26–40B      150      1800   (qwen3.6:35b — Space Invaders takes 25+ min)
        > 40B       240      2700   (70B class)

    These are wall-clock budgets PER STREAM, not total session.
    """
    b = _parse_param_billions(_model_param_size(model))
    if b > 40:
        return 240.0, 2700.0
    if b > 25:
        return 150.0, 1800.0
    if b > 13:
        return 90.0, 900.0
    # Default / unknown: err small so we detect a true wedge fast.
    return 60.0, 600.0


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


# Model tags we want auto-detection to skip when nothing is loaded in
# /api/ps and we have to guess from the installed list. Empty by default —
# the previous entries (qwen3.6:27b/35b) are now the user's actual working
# models and excluding them caused chat.py to fall through to gpt-oss
# every fresh launch. Keep this here so a future broken tag can be
# blacklisted without touching the rest of the resolver.
_KNOWN_BROKEN_TAGS: set[str] = set()


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
    """Pick which Ollama tag chat.py should use.

    Order of preference (matches what users actually expect):
      1. OLLAMA_MODEL / CHAT_OLLAMA_MODEL env var — explicit override.
      2. Models currently LOADED IN MEMORY (/api/ps), preferring the one with
         the latest `expires_at`. Ollama bumps that TTL on every use, so the
         freshest entry is the model the user most recently ran. The
         broken-tag blacklist is NOT applied here — if the model is in ps it
         was loaded successfully, and silently overriding the user's explicit
         `ollama run` would be infuriating.
      3. First INSTALLED model (/api/tags) skipping _KNOWN_BROKEN_TAGS — only
         a guess, used when nothing is loaded yet.
      4. Hard fallback (coder.MODEL).

    Returns (model_name, source_label) for the TUI to log.
    """
    for key in ("OLLAMA_MODEL", "CHAT_OLLAMA_MODEL"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw, f"{key} env"

    running = _running_models_with_meta()
    if running:
        # Sort by expires_at descending — ISO-8601 strings sort lexically the
        # right way. Ties (or missing values) fall back to ps order.
        running.sort(key=lambda m: m.get("expires_at") or "", reverse=True)
        chosen = running[0]["name"]
        names = [m["name"] for m in running]
        if len(names) == 1:
            return chosen, f"loaded in ollama (/api/ps): {chosen!r}"
        return chosen, (
            f"loaded in ollama: {names} — picking most-recently-used "
            f"{chosen!r} (latest expires_at)"
        )

    installed, err = _installed_models_via_http()
    if installed:
        chosen = _pick_first_workable(installed) or installed[0]
        return chosen, (
            f"nothing running; first installed (skipping broken): {chosen!r} "
            f"of {installed}"
        )

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
        # Per-session paths assigned in _start_session. None until then.
        self._out_path: Path | None = None
        self._best_path: Path | None = None
        # True between session-end and session-start. Used so feedback typed
        # after <done/> automatically triggers a continuation extension
        # instead of being silently queued forever.
        self._session_done: bool = True
        # /model stages a tag for the NEXT session; the running session keeps
        # whatever it was constructed with. None = use resolve_chat_model().
        self._next_model: str | None = None
        # /iters lets the user change the max-iters cap before starting a
        # session or extending. Default matches GameAgent's default.
        self._max_iters: int = 6
        # /seed stages an existing HTML file as the baseline for the next
        # /new session. Cleared once consumed.
        self._next_seed: Path | None = None

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
        self.title = "JMR's Coding Box"
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
            "[dim]Slash commands available — type [b]/help[/b] for the full list "
            "(/list, /model, /new, /open, /clear, /iters, /status, /ship, /quit).[/dim]"
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
    _MARKUP_RE = re.compile(r"\[/?[a-zA-Z][^\[\]]*\]")

    def _log(self, text: str) -> None:
        """Append a Rich-markup line to the agent log pane AND mirror to file.

        Use this for OUR annotation text (headers, status lines, prefixes) —
        anything you want Rich to color. For raw model output that may contain
        bracket-y code (e.g. `KEYMAP[e.code]`, `bullets[i]`), use _log_raw
        instead — Rich would otherwise eat those brackets as fake markup tags.
        """
        self.query_one("#log-pane", RichLog).write(text)
        if self._log_file_handle is not None:
            try:
                plain = self._MARKUP_RE.sub("", text)
                self._log_file_handle.write(plain.rstrip() + "\n")
                self._log_file_handle.flush()  # so `tail -f` sees it live
            except Exception:
                # Mirror must never crash the TUI.
                pass

    def _log_raw(self, text: str) -> None:
        """Append text VERBATIM — no Rich markup parsing, no regex stripping.

        Streamed model tokens go through here. JS code legitimately contains
        `[i]`, `[k]`, `[e.code]`, etc.; the regex used in _log would eat those
        in the file mirror, and `RichLog(markup=True)` would eat them in the
        pane. Wrapping in `Text` bypasses Rich's parser entirely.
        """
        self.query_one("#log-pane", RichLog).write(Text(text))
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.write(text.rstrip("\n") + "\n")
                self._log_file_handle.flush()
            except Exception:
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
                    # Raw: model output contains JS bracket indexing that
                    # would otherwise be eaten by Rich's markup parser.
                    self._log_raw(line)

    def _flush_stream(self) -> None:
        """Push any remaining buffered tokens (no trailing newline)."""
        if self._stream_buf.strip():
            self._log_raw(self._stream_buf)
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
        if self._log_file_path is None or self._out_path is None:
            self._log_info("[dim]no session active yet - paths appear after you submit a goal[/dim]")
            return
        stem = self._log_file_path.stem  # e.g. asteroids_20260503_175727
        traces = self._log_file_path.parent
        snaps = GAMES_DIR / "snapshots" / stem
        self._log("[bold cyan]── log artifacts ──[/bold cyan]")
        self._log(f"  game file:    {self._out_path}")
        self._log(f"  full log:     {self._log_file_path}")
        self._log(f"  jsonl trace:  {traces / (stem + '.jsonl')}")
        self._log(f"  conversation: {traces / (stem + '.conversation.md')}")
        self._log(f"  snapshots:    {snaps}")
        if self._best_path is not None:
            self._log(f"  best clean:   {self._best_path}")
        self._log("[dim]Tip: paste the full log above into your AI assistant to debug.[/dim]")

    # ----------------------------- input handler --------------------------

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Single dispatch point for the user's bottom input box.

        Dispatch order:
          1. Empty → ignore.
          2. Starts with `/` → slash command.
          3. awaiting_kind == "goal"   → start fresh session.
          4. awaiting_kind == "answer" → reply to model's <question>.
          5. awaiting_kind == "feedback":
                - session running → queue as mid-run feedback.
                - session done    → auto-extend (continuation mode).
        """
        text = (message.value or "").strip()
        message.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            await self._handle_slash(text)
            return

        if self._awaiting_kind == "goal":
            self._goal = text
            self._log(f"[bold green]>[/bold green] {text}")
            message.input.placeholder = "feedback · 'done' or Ctrl+D to ship · /help"
            self.sub_title = "agent is working"
            await self._start_session(text)
            self._awaiting_kind = "feedback"

        elif self._awaiting_kind == "answer":
            self._log(f"[bold magenta]> answer:[/bold magenta] {text}")
            if self.agent is not None:
                self.agent.add_user_answer(text)
            message.input.placeholder = "feedback · 'done' or Ctrl+D to ship · /help"
            self._awaiting_kind = "feedback"

        else:  # "feedback"
            self._log(f"[bold blue]> feedback:[/bold blue] {text}")
            if self.agent is None:
                self._log("[dim red]  (no active agent - feedback ignored)[/dim red]")
                return
            # Natural-language ship detection: "done", "ok", "looks good", etc.
            # ALWAYS shippable, whether the session is mid-run or already done.
            if _looks_like_ship(text):
                if self._session_done:
                    self._log_info(
                        f"[yellow]'{_esc(text)}' is a ship phrase but the "
                        "session is already finished — nothing to ship. "
                        "Type a new request to extend, or /new <goal> for fresh.[/yellow]"
                    )
                    return
                self.agent.request_done()
                self._log_info(
                    f"[yellow]'{_esc(text)}' interpreted as SHIP IT.[/yellow] "
                    "Agent will finish the current turn and stop. "
                    "[dim](To force more iteration on a 'done'-ish phrase, "
                    "rephrase as a request, e.g. 'looks good, but add sound'.)[/dim]"
                )
                return
            if self._session_done:
                # Session ended after <done/> — feedback is no longer queued
                # and forgotten. Restart the agent in continuation mode so
                # the new request is applied as patches against the existing
                # game file.
                await self._extend_session(text)
                return
            self.agent.add_user_feedback(text)
            pending = len(self.agent._pending_feedback)
            self._log(
                f"[dim cyan]  ✓ queued (pending: {pending}). "
                f"Will be applied at the next user-turn boundary - watch "
                f"for an [italic]→ applying your input[/italic] line.[/dim cyan]"
            )

    # ----------------------------- slash commands -------------------------

    async def _handle_slash(self, text: str) -> None:
        """Parse `/cmd args...` and dispatch. Unknown commands log help hint."""
        parts = text[1:].strip().split(maxsplit=1)
        if not parts:
            self._log_info("type /help to see available commands")
            return
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        # Echo the command so the trace shows what was typed.
        self._log(f"[bold cyan]>[/bold cyan] /{cmd}{(' ' + arg) if arg else ''}")
        try:
            if cmd in ("help", "h", "?"):
                self._cmd_help()
            elif cmd in ("list", "models"):
                self._cmd_list_models()
            elif cmd == "model":
                self._cmd_set_model(arg)
            elif cmd == "new":
                await self._cmd_new(arg)
            elif cmd == "ship":
                await self.action_ship_it()
            elif cmd == "quit":
                await self.action_quit_app()
            elif cmd in ("log", "paths", "files"):
                await self.action_show_log_paths()
            elif cmd == "open":
                self._cmd_open()
            elif cmd == "clear":
                self.query_one("#log-pane", RichLog).clear()
            elif cmd == "iters":
                self._cmd_set_iters(arg)
            elif cmd == "seed":
                self._cmd_set_seed(arg)
            elif cmd == "reset":
                self._cmd_reset()
            elif cmd == "status":
                self._cmd_status()
            else:
                self._log_info(f"unknown command /{cmd} — type /help")
        except Exception as e:
            self._log_error(f"/{cmd} failed: {e}")

    def _cmd_help(self) -> None:
        lines = [
            "[bold cyan]── what to type when ──[/bold cyan]",
            "  [b]first run[/b]                  describe the game you want, press Enter",
            "  [b]small change to what shipped[/b]  just type it — no slash needed",
            "                                  e.g. [italic]ship is too slow, double the thrust[/italic]",
            "  [b]ship as-is, stop[/b]              type [b]done[/b] / [b]looks good[/b] / [b]ship[/b] (or Ctrl+D)",
            "  [b]brand-new unrelated game[/b]      [b]/new <goal>[/b]",
            "  [b]start from an existing .html[/b]  [b]/seed <path>[/b]  then  [b]/new <goal>[/b]",
            "",
            "[dim]After <done/> the input box reads [b]'feedback to extend, /new <goal>"
            " for a fresh game'[/b] — that's the trigger for auto-extend.[/dim]",
            "",
            "[bold cyan]── slash commands ──[/bold cyan]",
            "  [b]/help[/b]                    show this help (also /h, /?)",
            "  [b]/list[/b]                    list installed Ollama models with numbers (also /models)",
            "  [b]/model <name|N>[/b]          stage model (STICKY across /new) · /model alone clears",
            "  [b]/seed <path>[/b]             stage a baseline .html (STICKY across /new) · /seed alone clears",
            "  [b]/iters <N>[/b]               set max iterations (sticky)",
            "  [b]/reset[/b]                   wipe ALL staged state (seed + model + iters → defaults)",
            "  [b]/new <goal>[/b]              end current session, start a fresh one (uses staged seed/model)",
            "  [b]/ship[/b]                    ship current build (= Ctrl+D, or type 'done')",
            "  [b]/open[/b]                    open the current game in your default browser",
            "  [b]/log[/b]                     print all session artifact paths (= Ctrl+L; also /paths, /files)",
            "  [b]/clear[/b]                   clear the agent log pane (does not affect staged state)",
            "  [b]/status[/b]                  print model, phase, iteration, paths, what's staged",
            "  [b]/quit[/b]                    quit (= Ctrl+Q)",
            "",
            "[bold cyan]── sticky staging ──[/bold cyan]",
            "  /seed, /model, /iters PERSIST across multiple /new calls. Set once,",
            "  reuse forever. Clear individually with the bare command "
            "(e.g. [b]/seed[/b] alone),",
            "  or wipe all of them with [b]/reset[/b].",
            "",
            "[dim]Example: /seed games/asteroids.html  →  /new add multiplayer  "
            "→  /new add boss  ▸ both use asteroids.html[/dim]",
        ]
        for line in lines:
            self._log(line)

    def _cmd_list_models(self) -> None:
        installed, err = _installed_models_via_http()
        if not installed:
            self._log_error(f"no installed models reachable: {err or 'unknown'}")
            return
        running = {m["name"] for m in _running_models_with_meta()}
        self._log("[bold cyan]── installed models ──[/bold cyan]")
        self._log("[dim]  * = currently loaded in ollama  · ← active = this session is using it[/dim]")
        for i, name in enumerate(installed, 1):
            loaded = "*" if name in running else " "
            active = "  [yellow]← active[/yellow]" if name == self._session_model else ""
            staged = "  [magenta]← staged for next /new[/magenta]" if name == self._next_model else ""
            self._log(f"  [{i:>2}] {loaded} {_esc(name)}{active}{staged}")
        self._log("[dim]Use /model <number-or-name> to switch.[/dim]")

    def _cmd_set_model(self, arg: str) -> None:
        if not arg:
            if self._next_model is None:
                self._log_info("no staged model (usage: /model <name-or-number>)")
            else:
                self._log_info(f"cleared staged model (was: {self._next_model})")
                self._next_model = None
            return
        installed, _ = _installed_models_via_http()
        if not installed:
            self._log_error("no installed models to choose from")
            return
        chosen: str | None = None
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(installed):
                chosen = installed[idx]
        if chosen is None and arg in installed:
            chosen = arg
        if chosen is None:
            matches = [n for n in installed if arg.lower() in n.lower()]
            if len(matches) == 1:
                chosen = matches[0]
            elif len(matches) > 1:
                self._log_error(f"ambiguous: {matches}")
                return
        if chosen is None:
            self._log_error(f"no match for {arg!r} — try /list")
            return
        self._next_model = chosen
        self._log_info(
            f"staged [b]{_esc(chosen)}[/b] for next /new session "
            "[dim](current session keeps its model)[/dim]"
        )

    async def _cmd_new(self, arg: str) -> None:
        if not arg:
            self._log_info("usage: /new <game description>")
            return
        if self.agent is not None and not self._session_done:
            self._log_error(
                "a session is currently running — press Ctrl+D to ship it "
                "first, then /new <goal>"
            )
            return
        await self._new_session(arg)

    def _cmd_open(self) -> None:
        if self._out_path is None or not self._out_path.exists():
            self._log_error("no game file to open yet")
            return
        import webbrowser
        url = f"file://{self._out_path.resolve()}"
        try:
            webbrowser.open(url)
            self._log_info(f"opened {url}")
        except Exception as e:
            self._log_error(f"could not open browser: {e}")

    def _cmd_set_iters(self, arg: str) -> None:
        if not arg.isdigit() or int(arg) <= 0:
            self._log_info(f"usage: /iters <positive int>  (current: {self._max_iters})")
            return
        self._max_iters = int(arg)
        self._log_info(
            f"max iterations set to [b]{self._max_iters}[/b] for next session/extension"
        )

    def _cmd_set_seed(self, arg: str) -> None:
        """/seed <path> stages an existing HTML file as the baseline for the
        next /new session. /seed with no argument clears the staged file.

        The file is NOT copied yet — it's just remembered. Path is checked
        for existence, .html-ness, and a sane size; we error early instead
        of letting the agent fail mid-run on a bad path.
        """
        if not arg:
            if self._next_seed is None:
                self._log_info("no staged seed file (usage: /seed <path>)")
            else:
                self._log_info(f"cleared staged seed file (was: {self._next_seed})")
                self._next_seed = None
            return
        # Allow shell-style ~ expansion and quoted paths.
        candidate = Path(arg.strip().strip("'\"")).expanduser()
        if not candidate.exists():
            self._log_error(f"seed file does not exist: {candidate}")
            return
        if not candidate.is_file():
            self._log_error(f"seed path is not a file: {candidate}")
            return
        if candidate.suffix.lower() not in {".html", ".htm"}:
            self._log_info(
                f"[yellow]warning:[/yellow] {candidate.suffix!r} is not .html — "
                "staging anyway, but the harness expects HTML"
            )
        size = candidate.stat().st_size
        self._next_seed = candidate.resolve()
        self._log_info(
            f"staged seed for next /new: [b]{_esc(str(self._next_seed))}[/b] "
            f"[dim]({size:,} bytes)[/dim]"
        )

    def _cmd_reset(self) -> None:
        """Wipe ALL staged state in one shot.

        After /reset:
          - no /seed staged → next /new starts from a memory skeleton
          - no /model staged → next /new uses /api/ps detection
          - max-iters back to default (6)

        Does NOT touch the currently-running session (if any), the browser,
        or anything on disk. To also start a fresh session, follow with
        /new <goal>.
        """
        had_seed = self._next_seed
        had_model = self._next_model
        had_iters = self._max_iters
        self._next_seed = None
        self._next_model = None
        self._max_iters = 6
        bits: list[str] = []
        if had_seed is not None:
            bits.append(f"seed={had_seed}")
        if had_model is not None:
            bits.append(f"model={had_model}")
        if had_iters != 6:
            bits.append(f"iters={had_iters}→6")
        if not bits:
            self._log_info("nothing to reset (no staged seed/model, iters at default)")
            return
        self._log_info(
            f"[yellow]reset:[/yellow] cleared {', '.join(bits)}. "
            "Next /new starts from defaults. [dim](Run /new <goal> to start fresh.)[/dim]"
        )

    def _cmd_status(self) -> None:
        lines = [
            "[bold cyan]── status ──[/bold cyan]",
            f"  model (active):    {_esc(self._session_model or '—')}",
            f"  model (next /new): {_esc(self._next_model or '(auto-detect)')}",
            f"  goal:              {_esc(self._goal or '—')}",
            f"  phase:             {_esc(self._phase_label)}",
            f"  iteration:         {_esc(self._iteration_label)}",
            f"  max iters:         {self._max_iters}",
            f"  staged seed:       {_esc(str(self._next_seed) if self._next_seed else '—')}",
            f"  session done:      {self._session_done}",
            f"  game file:         {self._out_path or '—'}",
            f"  log file:          {self._log_file_path or '—'}",
        ]
        for line in lines:
            self._log(line)

    # ----------------------------- session --------------------------------

    async def _start_session(self, goal: str) -> None:
        """Boot the LiveBrowser + GameAgent and start consuming events."""
        self._phase_label = "starting browser"
        self._session_done = False
        self._update_status()

        if self.browser is None:
            self.browser = LiveBrowser(viewport=(800, 600), run_seconds=3.0)
            try:
                await self.browser.start()
            except Exception as e:
                self._log_error(
                    f"Could not launch Chromium: {e}\n"
                    "Make sure you ran `playwright install chromium` and that you "
                    "have a graphical display (this needs headless=False)."
                )
                self._session_done = True
                return

        # /model staged tag wins; otherwise resolve from env / ps / installed.
        # Staging is STICKY across /new calls — clear with /model (no arg)
        # or replace with another /model <tag>.
        if self._next_model:
            model_name, model_src = self._next_model, "/model staged (sticky)"
        else:
            model_name, model_src = resolve_chat_model(MODEL)
        self._session_model = model_name
        self.title = f"JMR's Coding Box — {model_name}"
        self._log_info(f"Using model [b]{_esc(model_name)}[/b] [dim]({_esc(model_src)})[/dim]")
        self._update_status()

        # Build a unique, meaningful basename for every session artifact:
        # "<goal-slug>_<timestamp>". The agent derives trace/snapshots/best
        # paths from out_path.stem, so they all share this basename.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = f"{_slugify(goal)}_{ts}"
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        self._out_path = GAMES_DIR / f"{basename}.html"
        self._best_path = GAMES_DIR / f"{basename}.best.html"
        self._log_info(f"Game file: [b]{self._out_path}[/b]")

        # Use the staged seed file (if any). Staging is STICKY — every /new
        # uses the same seed until you /seed (no arg) to clear or /seed
        # <other> to replace. Matches /model's sticky behavior.
        seed = self._next_seed
        if seed is not None:
            self._log_info(
                f"using staged seed file (sticky): [b]{_esc(str(seed))}[/b] "
                "[dim](/seed (no arg) to clear)[/dim]"
            )

        # Auto-bump streaming timeouts for larger models — qwen3.6:35b
        # writing a full Space Invaders takes 25+ minutes per stream and
        # the default 600s budget kills it mid-output every time.
        stall_s, overall_s = resolve_session_timeouts(model_name)
        if overall_s > 600:
            self._log_info(
                f"[dim]large model detected — using stall={stall_s:.0f}s "
                f"overall={overall_s:.0f}s per stream[/dim]"
            )

        self.agent = GameAgent(
            model=model_name,
            out_path=self._out_path,
            browser=self.browser,
            max_iters=self._max_iters,
            seed_file=seed,
            stall_seconds=stall_s,
            overall_seconds=overall_s,
            # v1 prompt: includes <playbook> retrieval, <criteria>,
            # <probes>, stuck-loop ladder. Real sessions need this on
            # so the offline learner has rich traces to reflect over.
            prompt_version="v1",
        )
        self.agent.set_token_callback(self._emit_token)

        self._open_log_mirror(basename)

        # Spawn the agent loop as a background task so the TUI stays responsive.
        self.run_worker(self._consume_events(goal, continuation=False), exclusive=True)

    async def _extend_session(self, feedback: str) -> None:
        """Continuation: re-run the agent on the existing file with new feedback.

        Triggered when the user types plain text after the agent declared
        <done/>. Reuses agent, browser, model, and out_path — only kicks off
        another iteration loop with the feedback as a fix prompt.

        Graceful fallback: if the previous session never produced a working
        file (model exhausted iterations without emitting valid <patch> /
        <html_file>), there's nothing to extend — silently switch to a
        fresh session with the original goal + feedback combined, instead
        of dying with the misleading "no current file" error.
        """
        if self.agent is None or self.browser is None:
            self._log_error("can't extend — no active agent/browser")
            return
        # No file on disk means the previous run produced nothing to patch.
        if self._out_path is None or not self._out_path.exists():
            self._log_info(
                "[yellow]previous session produced no working file[/yellow] — "
                "starting a fresh session with your feedback as a refinement "
                "of the original goal"
            )
            combined = f"{self._goal} — {feedback}" if self._goal else feedback
            await self._new_session(combined)
            return
        # Apply any /iters change before extending.
        self.agent.max_iters = self._max_iters
        self._session_done = False
        self._phase_label = "extending"
        self.sub_title = "agent is working (extension)"
        self._update_status()
        self._log_info(
            f"[yellow]extending session[/yellow] with feedback: {_esc(feedback[:160])}"
        )
        self.run_worker(self._consume_events(feedback, continuation=True), exclusive=True)

    async def _new_session(self, goal: str) -> None:
        """End the current session (if any) and start a fresh one.

        Browser is reused but cleared to a status page so a failed new
        session can't masquerade as the previous one's output. Log mirror
        is rotated to the new basename.
        """
        # Drop the old agent reference; the worker (if any) has already
        # finished — we checked _session_done in _cmd_new.
        self.agent = None
        # Close the previous log mirror.
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
            self._log_file_path = None
        # Clear the browser to a status page so the previous game isn't
        # still visible — otherwise a failed new session looks like the
        # old one is still working, and any feedback gets routed to the
        # WRONG file path.
        if self.browser is not None:
            try:
                await self.browser.show_status(
                    "Starting new session…",
                    f"Goal: {goal[:200]}",
                )
            except Exception:
                pass
        self.query_one("#log-pane", RichLog).clear()
        self._goal = goal
        self._iteration_label = "—"
        self._awaiting_kind = "feedback"
        self._log(f"[bold green]>[/bold green] /new {_esc(goal)}")
        await self._start_session(goal)

    def _open_log_mirror(self, basename: str) -> None:
        """Open or rotate the plain-text .log mirror for the new session."""
        # Close prior handle if rotating.
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
        try:
            log_dir = GAMES_DIR / "traces"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file_path = log_dir / f"{basename}.log"
            self._log_file_handle = self._log_file_path.open("w", encoding="utf-8")
            self._log_info(f"Mirroring log to: {self._log_file_path}")
        except Exception as e:
            self._log_info(f"[dim]could not open log mirror: {e}[/dim]")

    async def _consume_events(self, goal: str, *, continuation: bool = False) -> None:
        """Drain the AgentEvent stream and update widgets accordingly."""
        assert self.agent is not None
        try:
            async for ev in self.agent.run(goal, continuation=continuation):
                self._handle_event(ev)
        except Exception as e:
            # Include the FULL traceback so the .log file has enough info to
            # debug the crash without re-running. Plain string only - the
            # markup-stripping mirror handles either way.
            import traceback
            tb = traceback.format_exc()
            self._log_error(f"Agent crashed: {e}")
            self._log(f"[dim red]{_esc(tb)}[/dim red]")
        finally:
            self._flush_stream()
            self._phase_label = "finished"
            self._session_done = True
            # Reset input mode in case the session ended mid-question; if we
            # left it on "answer", post-done text would be routed to a defunct
            # add_user_answer instead of triggering a continuation.
            self._awaiting_kind = "feedback"
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "type feedback to extend, /new <goal> for a fresh game"
            self._update_status()
            self.sub_title = "session ended - type more feedback to extend, or /new <goal>"
            # Always end with a clear footer pointing at the log files so the
            # user can paste them to an AI for debugging.
            await self.action_show_log_paths()
            self._log(
                "[bold yellow]Done.[/bold yellow] Type more feedback to "
                "[b]extend this game[/b] (auto-continuation), [b]/new <goal>[/b] "
                "for a fresh session, [b]/help[/b] for all commands."
            )

    def _handle_event(self, ev: AgentEvent) -> None:
        """Pattern-match on event kind and update the UI."""
        # Always flush any half-streamed line before logging a new event header.
        self._flush_stream()
        # ev.text often contains bracket characters (test reports list keys
        # like ['ArrowUp'], notes may quote code). Escape before interpolating
        # into a Rich markup format string so '[' isn't read as a fake tag.
        text_safe = _esc(ev.text or "")

        if ev.kind == "phase":
            # "planning", "iteration N/M", "self-critique"
            self._phase_label = ev.text
            if ev.text.startswith("iteration"):
                self._iteration_label = ev.text.replace("iteration ", "")
            self._update_status()
            self._log(f"\n[bold yellow]── {text_safe} ──[/bold yellow]")

        elif ev.kind == "plan":
            # The full plan is already in the log via streaming tokens; just
            # mark the boundary so the user knows phase A is done.
            self._log("[dim](plan complete)[/dim]")

        elif ev.kind == "code":
            self._log(f"[green]wrote {text_safe}[/green] ({ev.data.get('size', 0)} bytes)")

        elif ev.kind == "test":
            # ev.text is the human-readable report, ev.data is the dict.
            ok = ev.data.get("ok", False)
            tag = "[green]TEST OK[/green]" if ok else "[red]TEST FAILED[/red]"
            n_err = len(ev.data.get("errors", []))
            n_iss = len(ev.data.get("soft_warnings", []))
            self._log(f"{tag} ({n_err} error(s), {n_iss} issue(s))")
            # Also drop the full report into the right-hand status panel.
            self._update_status(extra=f"[b]Last test:[/b]\n{text_safe}")

        elif ev.kind == "question":
            self._log(f"\n[bold magenta]?[/bold magenta] [bold]Model asks:[/bold] {text_safe}")
            self._awaiting_kind = "answer"
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "type your answer and press Enter"
            inp.focus()

        elif ev.kind == "done":
            self._log(f"\n[bold green]DONE[/bold green] - {text_safe}")
            if self._out_path is not None:
                self._log(f"[dim]Final game: {self._out_path.resolve()}[/dim]")
            # Surface the auto-saved best version + trace location so the user
            # knows where to look if they want to inspect or recover.
            if self._best_path is not None and self._best_path.exists():
                self._log(f"[dim]Best clean version: {self._best_path.resolve()}[/dim]")
            traces_dir = GAMES_DIR / "traces"
            if traces_dir.exists():
                self._log(f"[dim]Trace logs: {traces_dir.resolve()}[/dim]")

        elif ev.kind == "error":
            self._log_error(text_safe)

        elif ev.kind == "info":
            self._log_info(text_safe)

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
