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
  1. backend.detect_backend() defaults to MLX on macOS (Apple GPU) unless
     you set LLM_BACKEND or use /backend. Otherwise it follows the same
     rules as coder.py --backend.
       Ollama (port 11434) — loaded model from /api/ps, or OLLAMA_MODEL /
       CHAT_OLLAMA_MODEL overrides.
       MLX (in-process) — MLX_MODEL env, else single auto-discovered model
       under ~/MLX_Models / HF cache. No mlx_lm.server, no HTTP.
  2. With LLM_BACKEND=auto (or explicit --backend auto), if both daemons
     have a loaded model, MLX wins. Force one with LLM_BACKEND=ollama /
     LLM_BACKEND=mlx or /backend in the TUI.
  3. In full auto-probe mode only: if neither daemon has a model loaded
     but Ollama is reachable, falls back to first installed Ollama tag.

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
import time
from datetime import datetime
from pathlib import Path

import ollama
from rich.markup import escape as _esc
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

import backend as backend_mod
from agent import AgentEvent, GameAgent
from tools import LiveBrowser


class MultilinePasteInput(Input):
    """Single-line Input that accepts multi-line paste by flattening
    newlines to spaces.

    Textual's stock Input._on_paste does
    `event.text.splitlines()[0]`, silently discarding everything after
    the first newline. That's a real footgun when the user pastes a
    multi-line game-design prompt: only the first line reaches the
    agent. Override the paste handler so the full pasted text lands in
    the field, with whitespace collapsed.

    For the agent, newlines vs spaces in the goal text are
    indistinguishable — the model receives the goal as part of a
    user-turn string, so flattening is lossless. If you ever need true
    multi-line semantics, swap Input for TextArea instead (different
    submit ergonomics — Ctrl+Enter to submit, Enter inserts newline).
    """

    def _on_paste(self, event: events.Paste) -> None:  # type: ignore[override]
        text = event.text or ""
        # Collapse all runs of whitespace (including newlines and tabs)
        # to a single space. Strip leading/trailing whitespace so a
        # paste that starts with a blank line doesn't drop a leading
        # space into the field at the cursor.
        flat = " ".join(text.split())
        if flat:
            selection = self.selection
            if selection.is_empty:
                self.insert_text_at_cursor(flat)
            else:
                self.replace(flat, *selection)
        event.stop()


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


_PARAM_SIZE_IN_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*B(?![a-z0-9])",
    re.IGNORECASE,
)


def _model_param_size(model: str) -> str:
    """Best-effort parameter_size string for a model (e.g. '36.0B').

    Tries /api/ps first (loaded models, fast). Falls back to ollama.show()
    (works for installed-but-not-loaded models — that's the common case
    when the user hasn't run the model yet). For MLX-served models Ollama
    knows nothing, so as a last resort we scan the model name/path for a
    "<n>B" token (e.g. 'Qwen3.6-27B-mxfp8' -> '27B'). Without this,
    timeouts default to small-model values and the MLX stream watchdog
    kills big-prompt requests before the first token arrives.
    Returns '' if all paths fail.
    """
    for m in _running_models_with_meta():
        if m["name"] == model and m.get("parameter_size"):
            return m["parameter_size"]
    try:
        info = ollama.show(model=model)
        details = getattr(info, "details", None)
        if details is not None:
            size = (getattr(details, "parameter_size", "") or "").strip()
            if size:
                return size
    except Exception:
        pass
    matches = _PARAM_SIZE_IN_NAME_RE.findall(model or "")
    if matches:
        return f"{max(float(m) for m in matches)}B"
    return ""


def resolve_session_timeouts(model: str) -> tuple[float, float]:
    """Pick (stall_seconds, overall_seconds) for a given model.

    Scaling is by parameter count (queried from /api/ps then /api/show,
    or parsed from the model name for MLX). Larger models take longer
    per token AND tend to write more verbose output, so we bump BOTH
    timeouts:

        params      stall    overall
        ─────────   ─────    ───────
        ≤ 13B       60       600    (small/fast, default-ish)
        14–25B      90       1200   (gpt-oss 20B ballpark)
        26–40B      900      3600   (27B/35B MLX — first-token can be
                                     15 min on big Doom-class prompts;
                                     mlx_lm.server keepalives are not
                                     reliably sub-stall during heavy
                                     prompt processing)
        > 40B       1200     5400   (70B class)

    These are wall-clock budgets PER STREAM, not total session.
    """
    b = _parse_param_billions(_model_param_size(model))
    if b > 40:
        return 1200.0, 5400.0
    if b > 25:
        return 900.0, 3600.0
    if b > 13:
        return 90.0, 1200.0
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

# Tags that are clearly NOT chat models — diffusers (Z-Image-Turbo,
# Stable Diffusion), embedding models, etc. Excluded from auto-pick in
# both /api/ps and /api/tags paths because Ollama lists them alongside
# real chat tags and the resolver would otherwise grab whichever is
# freshest by expires_at. The user can still force one via
# OLLAMA_MODEL=<tag> if they really mean it. Match is case-insensitive
# substring on the tag, so `x/z-image-turbo:latest` and
# `stabilityai/stable-diffusion-3:latest` are both filtered.
_NON_CHAT_TAG_FRAGMENTS: tuple[str, ...] = (
    "z-image",          # Z-Image-Turbo (used in-process via diffusers)
    "stable-diffusion",
    "sdxl",
    "flux",             # Black Forest Labs FLUX — image gen
    "embed",            # nomic-embed-text, bge-*, etc.
    "embedding",
    "minilm",           # sentence-transformers / embedding models
    "bge-",
    "rerank",           # cross-encoder rerankers
    "whisper",          # speech-to-text
    "tts-",             # text-to-speech
)


def _is_chat_capable_tag(name: str) -> bool:
    """True if the tag is plausibly a chat model. False for known
    image / embed / speech model families. Defensive: returns True for
    unknown tags so we don't accidentally exclude a working chat
    model — only excludes tags we've seen cause `does not support
    chat` (status code: 400) responses."""
    n = (name or "").lower()
    return not any(frag in n for frag in _NON_CHAT_TAG_FRAGMENTS)


def _pick_first_workable(names: list[str]) -> str | None:
    """Return the first installed model that's chat-capable AND not in
    the broken blacklist. Filters out diffusers / embed / etc. so a
    fresh launch with no chat model loaded doesn't grab Z-Image-Turbo."""
    for n in names:
        if n in _KNOWN_BROKEN_TAGS:
            continue
        if not _is_chat_capable_tag(n):
            continue
        return n
    return None


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
        # Drop diffusers / embed / etc. before picking the freshest. After
        # in-process Z-Image-Turbo loads its weights via the Ollama-pulled
        # `x/z-image-turbo:latest` tag, that tag becomes the most-recently-
        # used entry in /api/ps; without this filter the next /new session
        # would try to chat with it and Ollama returns 400.
        chat_running = [m for m in running if _is_chat_capable_tag(m["name"])]
        if chat_running:
            chosen = chat_running[0]["name"]
            names = [m["name"] for m in chat_running]
            skipped = [m["name"] for m in running if not _is_chat_capable_tag(m["name"])]
            tail = f" (skipped non-chat: {skipped})" if skipped else ""
            if len(names) == 1:
                return chosen, f"loaded in ollama (/api/ps): {chosen!r}{tail}"
            return chosen, (
                f"loaded in ollama: {names} — picking most-recently-used "
                f"{chosen!r} (latest expires_at){tail}"
            )
        # All running models were filtered out as non-chat — fall through
        # to /api/tags so we can pick an installed-but-not-loaded chat
        # model. Don't return an unusable tag.

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

    #mode-bar {
        height: 1;
        dock: bottom;
        padding: 0 1;
        color: $accent;
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
        # Ctrl+S: toggle "selection mode" — releases Textual's mouse
        # capture so the terminal handles drag-select natively. Press
        # again to resume normal TUI mouse handling. Without this, the
        # left log pane is unselectable while the agent is running.
        Binding("ctrl+s", "toggle_selection_mode", "Select text"),
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
        self._assets_dir: Path | None = None
        # Status-panel state — kept here so _update_status() can render
        # without re-derived state. Reset in _new_session via _reset_status_state.
        self._activity_label: str = ""        # what's happening right now
        self._activity_started_at: float = 0.0  # monotonic; for "Ns" age
        self._stream_tokens: int = 0          # tokens this stream
        self._stream_started_at: float = 0.0  # monotonic; for tok/s
        self._last_token_at: float = 0.0      # monotonic; for stall age
        self._is_streaming: bool = False
        self._assets_summary: str = ""        # sticky summary of last batch
        # Sticky sounds summary — same pattern as assets. Populated from
        # the `sounds` event payload; cleared on session reset. Looping
        # entries surface a `(loop)` suffix in the rendered list.
        self._sounds_summary: str = ""
        self._sounds_dir: Path | None = None
        # Probe pass/fail counts updated on each `test` event. None
        # before any test fires — same display pattern as streak (the
        # iteration line stays clean when there's nothing to report).
        self._probes_passed: int | None = None
        self._probes_total: int | None = None
        # Sticky one-line preview of the most recent <diagnose> text.
        # Helps the user see what the model is currently working on
        # without scrolling the log. Truncated to ~140 chars at set time.
        self._last_diagnose: str | None = None
        # Context-window display. `_ctx_max` is read once at session
        # start from BackendInfo.context_length (Ollama) or the MLX
        # config.json. `_ctx_fill_chars` is recomputed each
        # _update_status() tick by summing message lengths on the agent.
        self._ctx_max: int | None = None
        self._streak_clean: int = 0
        self._streak_min: int = 2
        self._streak_stuck: int = 0
        # Trace JSONL path for the current session. Surfaced in status.
        self._trace_path: Path | None = None
        # True between session-end and session-start. Used so feedback typed
        # after <done/> automatically triggers a continuation extension
        # instead of being silently queued forever.
        self._session_done: bool = True
        # /model stages an Ollama tag for the NEXT session; the running
        # session keeps whatever it was constructed with. None = let
        # backend.detect_backend pick.
        self._next_model: str | None = None
        # /backend stages a backend preference for the NEXT session.
        # None / "auto" = probe both daemons and pick whichever has a
        # model loaded (MLX wins ties). "ollama" / "mlx" = force.
        self._next_backend: str | None = None
        # Snapshot of the last /list output: list of (backend_name, model_id)
        # pairs in display order, so /load N or /model N can pick by number
        # across BOTH backends. Empty until the user runs /list once; the
        # /load handler refreshes it on demand if empty.
        self._last_listing: list[tuple[str, str]] = []
        # Resolved Backend + BackendInfo for the running session.
        # Constructed in _start_session.
        self._session_backend = None
        self._session_backend_info: backend_mod.BackendInfo | None = None
        # /iters lets the user change the max-iters cap before starting a
        # session or extending. Default matches GameAgent's default.
        self._max_iters: int = 6
        # /seed stages an existing HTML file as the baseline for the next
        # /new session. Cleared once consumed.
        self._next_seed: Path | None = None
        # Stop-Losing-To-OneShot Track A — restart-N. The threshold gate
        # makes 2 essentially free for simple games (they pass iter 1
        # with score > 60 and never trigger a restart) while giving hard
        # games (DOOM, pac-man) a second chance from a clean slate.
        # /restarts <N> overrides per session.
        self._restart_n: int = 2
        self._restart_threshold: float = 60.0
        # System-prompt trim level. None = "auto" → resolves to "small"
        # in GameAgent (lean ~5 KB schema). Override via /model-class
        # large when running a frontier-tier model. We do NOT inspect
        # the model name — the user rotates local LLMs constantly.
        self._model_class: str | None = None

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
        yield MultilinePasteInput(
            placeholder="What game do you want to build?", id="user-input"
        )
        # Single-row mode indicator just above the Footer. Tells the
        # user at a glance whether the agent is RUNNING or WAITING for
        # them. Sits in the same visual band as the binding hints so
        # both are scannable without moving the eye.
        yield Static("", id="mode-bar")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "JMR's Coding Box"
        self.sub_title = "type your game idea below, then Enter"
        self._update_status()
        # Periodic refresh so the activity line ages naturally — tok/s,
        # "last token Ns ago", and the pre-first-token wait counter all
        # need to advance even when no new event arrives. 1s cadence is
        # cheap (Static.update diffs Rich content) and is what makes a
        # stalled stream visible without a fresh event.
        self.set_interval(1.0, self._tick_status)
        # Show what model we'll use when the user submits a goal. One call,
        # one line. Override with OLLAMA_MODEL env var if you want.
        try:
            preview = backend_mod.detect_backend(self._next_backend)
            self._log_info(
                f"Will use [b]{preview.name.upper()}[/b] · "
                f"[b]{_esc(preview.model)}[/b] [dim]({_esc(preview.source)})[/dim]"
            )
            # MLX-on-Mac caveat — Apple Silicon's Metal wired-memory cap.
            # Big models + long context routinely cross the default cap
            # and OOM mid-generation. README has the sysctl + recovery
            # steps. Fires only when MLX is actually selected.
            import sys as _sys
            if preview.name == "mlx" and _sys.platform == "darwin":
                self._log_info(
                    "[dim]MLX tip: model runs in-process. If prompt eval "
                    "OOMs on a 27B+ model, raise the Metal cap via "
                    "[b]sudo sysctl iogpu.wired_limit_mb=$N[/b] (README "
                    "§MLX memory limit on Apple Silicon), or drop "
                    "MLX_PREFILL_STEP_SIZE to 512.[/dim]"
                )
        except RuntimeError as e:
            self._log_error(str(e))
        if _KNOWN_BROKEN_TAGS:
            self._log_info(
                f"[dim]Skipping known-broken tags: {sorted(_KNOWN_BROKEN_TAGS)}. "
                "Set OLLAMA_MODEL=<tag> to override.[/dim]"
            )
        self._log_info("Type your game idea in the input box below and press Enter.")
        self._log_info(
            "[dim]Keys: Ctrl+D ship · Ctrl+L log paths · Ctrl+S select-text · "
            "Ctrl+Q quit · if the shell stops echoing after exit, run `reset`.[/dim]"
        )
        self._log_info(
            "[dim]Slash commands available — type [b]/help[/b] for the full list "
            "(/list, /model, /new, /open, /clear, /iters, /status, /ship, /quit).[/dim]"
        )
        self._log_info(
            "[dim]Cut/paste: press [b]Ctrl+S[/b] to enable selection mode, then "
            "click-drag to select (Ctrl+Shift+C to copy). Or hold [b]Shift[/b] "
            "(or [b]Option[/b] on iTerm2) while dragging — the modifier bypasses "
            "Textual's mouse capture without toggling. Or [b]tail -f[/b] the "
            ".jsonl trace from another terminal (path via Ctrl+L) for live "
            "progress including [b]stream_heartbeat[/b] events every 30 s.[/dim]"
        )
        # Short prompt-engineering tips for medium-skilled local models
        # (qwen3.6:27b/35b). Long-form guidance lives in the README;
        # these are the four lines that move the success rate the most.
        self._log("")
        self._log("[bold]── how to write a prompt that ships a playable game ──[/bold]")
        self._log(
            "  • [b]Be specific about controls + win/lose:[/b] "
            "\"WASD to move, mouse to aim, space to shoot, lose at 0 HP, "
            "restart with R\". Vague goals → vague games."
        )
        self._log(
            "  • [b]Name what's on screen:[/b] enemies, projectiles, terrain. "
            "The agent uses these to request sprite art automatically."
        )
        self._log(
            "  • [b]Ask for art directly:[/b] words like [italic]"
            "\"sprite art\", \"pixel-art\", \"cool graphics\"[/italic] "
            "trigger the Z-Image-Turbo pipeline. Skip them for DOM-only "
            "apps (todo, calculator)."
        )
        self._log(
            "  • [b]Mark mixed graphics explicitly:[/b] "
            "\"sprites for X, but procedural for Y because Y gets destroyed "
            "brick-by-brick\" — the model honors this and keeps state-rich "
            "entities procedural."
        )
        self._log(
            "  • [b]For 3D, just say \"3D\" or \"first-person\":[/b] the agent "
            "detects this and switches to three.js via CDN. Don't ask for a "
            "raycaster from scratch unless you really mean it."
        )
        self._log(
            "  • [b]Iterate via plain text:[/b] after [b]<done/>[/b] just "
            "type changes ([italic]\"the gun is sideways, rotate 90°\"[/italic]) "
            "— it auto-extends. Use [b]/new <goal>[/b] only for unrelated games."
        )
        self._log("")
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
        # Track token-rate stats so the status panel can display tok/s and
        # detect a wedge (last_token_at growing without bound).
        now = time.monotonic()
        self._stream_tokens += 1
        self._last_token_at = now
        if self._stream_started_at == 0.0:
            self._stream_started_at = now
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
        """Render the right-hand status panel.

        Sections (in order):
          1. Activity — what's happening right now (streaming / assets /
             browser / idle), with tok/s and stall-age while streaming.
          2. Iteration — phase + clean-streak + queued user feedback.
          3. Assets — sticky summary of the most recent generation batch
             (paths, per-asset cache hits, generation times).
          4. Files — paths to game.html, best.html, trace JSONL, assets
             dir, plain-text log mirror. Always visible so the user
             knows what to `cat` / share.
          5. Last test (`extra`) — full report shown when a test event
             fires; passed in by the caller.
        """
        body = self._render_activity_line()
        body += self._render_iteration_block()
        body += self._render_assets_block()
        body += self._render_sounds_block()
        body += self._render_playbook_block()
        body += self._render_files_block()
        if extra:
            body += "\n" + extra
        self.query_one("#status-body", Static).update(body)
        # Mode bar gets a free refresh on every status tick. It's a
        # single-line Static update — cheap.
        self._update_mode_bar()
        # Persist a structured snapshot to the trace .jsonl so the
        # right-hand panel is reconstructable from logs alone (de-duped
        # inside agent.trace_status — successive ticks that don't
        # change anything meaningful are skipped). Guarded because
        # _update_status fires before the agent is wired during early
        # init.
        agent = getattr(self, "agent", None)
        if agent is not None and hasattr(agent, "trace_status"):
            try:
                # Sample ctx fill in chars (cheap method on the agent).
                # Token count derived in `_format_ctx_row`; trace stores
                # the raw char count so future analyses can apply any
                # tokenizer they like.
                ctx_fill_chars = 0
                if hasattr(agent, "_estimate_ctx_fill"):
                    ctx_fill_chars = int(agent._estimate_ctx_fill())
                agent.trace_status({
                    "activity": self._activity_label or "idle",
                    "is_streaming": bool(self._is_streaming),
                    "phase": self._phase_label,
                    "iteration": self._iteration_label,
                    "streak_clean": int(self._streak_clean or 0),
                    "streak_stuck": int(self._streak_stuck or 0),
                    "probes_passed": self._probes_passed,
                    "probes_total": self._probes_total,
                    "last_diagnose": self._last_diagnose,
                    "ctx_max": self._ctx_max,
                    "ctx_fill_chars": ctx_fill_chars,
                    "backend": getattr(agent, "_backend_label", None)
                        or type(getattr(agent, "_backend", None)).__name__,
                    "model": getattr(agent, "model", None),
                    "goal": getattr(self, "_current_goal", None),
                    "files": {
                        "game": str(self._out_path) if self._out_path else None,
                        "best": str(self._best_path) if self._best_path else None,
                        "log": str(self._log_file_path) if self._log_file_path else None,
                        "sounds": str(self._sounds_dir) if self._sounds_dir else None,
                    },
                })
            except Exception:
                pass

    def _render_activity_line(self) -> str:
        """Top line: the heartbeat. Always rendered, even when idle."""
        now = time.monotonic()
        if self._is_streaming:
            elapsed = max(0.001, now - self._stream_started_at) if self._stream_started_at else 0.001
            tok_per_s = self._stream_tokens / elapsed if elapsed > 0 else 0.0
            since_last = now - self._last_token_at if self._last_token_at else 0.0
            # Stall threshold: 30s without a token while streaming = warn.
            stalled = self._stream_tokens > 0 and since_last > 30.0
            label = self._activity_label or "streaming reply"
            if self._stream_tokens == 0:
                # Pre-first-token: show how long we've been waiting. This
                # is the case the user complained about — Ollama not
                # responding looks identical to "thinking" without this.
                wait = now - self._stream_started_at if self._stream_started_at else 0.0
                # MLX path: if the in-process backend is feeding us
                # prompt-eval progress (see backend.MLXBackend._stream_once),
                # show that instead of a generic wait counter — turns the
                # 6358-token blank wait into "prompt eval 6358/6358".
                # Also handles the "mlx_load" stage which fires once on
                # cold start while the weights stream into VRAM.
                progress_total = getattr(self.agent, "_stream_progress_total", 0) or 0
                progress_current = getattr(self.agent, "_stream_progress_current", 0) or 0
                progress_stage = getattr(self.agent, "_stream_progress_stage", None)
                if progress_stage == "mlx_load":
                    return (
                        f"[bold yellow]Activity:[/bold yellow] {label} — "
                        f"[cyan]loading MLX weights into VRAM[/cyan] "
                        f"[dim]— {wait:.0f}s (cold-start, ~30-60s)[/dim]\n"
                    )
                if progress_total > 0:
                    pct = (100.0 * progress_current / progress_total) if progress_total else 0.0
                    return (
                        f"[bold yellow]Activity:[/bold yellow] {label} — "
                        f"[cyan]prompt eval {progress_current:,}/{progress_total:,} "
                        f"({pct:.0f}%)[/cyan] [dim]— {wait:.0f}s[/dim]\n"
                    )
                if wait > 30.0:
                    return (
                        f"[bold yellow]Activity:[/bold yellow] {label} — "
                        f"[red]waiting {wait:.0f}s for first token[/red]\n"
                    )
                return (
                    f"[bold yellow]Activity:[/bold yellow] {label} — "
                    f"[dim]waiting for first token ({wait:.0f}s)[/dim]\n"
                )
            tag = "[red]STALLED[/red]" if stalled else "[green]live[/green]"
            return (
                f"[bold yellow]Activity:[/bold yellow] {label} — "
                f"{self._stream_tokens:,} tok, {tok_per_s:.1f} tok/s, "
                f"last {since_last:.1f}s ago {tag}\n"
            )
        if self._activity_label:
            age = now - self._activity_started_at if self._activity_started_at else 0.0
            return (
                f"[bold yellow]Activity:[/bold yellow] {self._activity_label} "
                f"[dim]({age:.0f}s)[/dim]\n"
            )
        return "[bold yellow]Activity:[/bold yellow] [dim]idle[/dim]\n"

    def _render_iteration_block(self) -> str:
        """Phase / iteration / streak / probes / ctx / model / goal / queued."""
        out = (
            f"[b]Phase:[/b] {self._phase_label}\n"
            f"[b]Iteration:[/b] {self._iteration_label}"
        )
        if self._streak_clean or self._streak_stuck:
            out += (
                f" [dim](streak {self._streak_clean}/"
                f"{self._streak_min} clean"
            )
            if self._streak_stuck:
                out += f", {self._streak_stuck} stuck"
            out += ")[/dim]"
        # Probe pass/fail counts, sticky after the first `test` event.
        # Shows green when all probes pass; otherwise red/yellow accent
        # so the user spots regressions at a glance.
        if self._probes_total:
            passed = self._probes_passed or 0
            total = self._probes_total
            if passed == total:
                tag = f"[green]{passed}/{total} passed[/green]"
            else:
                tag = f"[yellow]{passed}/{total} passed[/yellow]"
            out += f" — Probes: {tag}"
        out += "\n"
        if self._session_backend_info is not None:
            out += f"[b]Backend:[/b] {self._session_backend_info.name.upper()}\n"
        if self._session_model:
            out += f"[b]Model:[/b] {self._session_model}\n"
        # Context window row: hide entirely when neither max nor a
        # running agent is available (avoids a sad-looking "0 / 0" on
        # backends that don't expose context_length).
        ctx_row = self._format_ctx_row()
        if ctx_row:
            out += ctx_row
        out += f"[b]Goal:[/b] {self._goal or '—'}\n"
        if self._last_diagnose:
            out += f"[b]Last fix:[/b] [dim]{_esc(self._last_diagnose)}[/dim]\n"
        if self.agent is not None:
            pending_fb = list(getattr(self.agent, "_pending_feedback", []) or [])
            pending_ans = getattr(self.agent, "_pending_answer", None)
            queue_lines: list[str] = []
            if pending_ans:
                queue_lines.append(
                    f"  [magenta]answer:[/magenta] {_esc(pending_ans[:80])}"
                )
            for i, fb in enumerate(pending_fb, 1):
                preview = fb[:80] + ("…" if len(fb) > 80 else "")
                queue_lines.append(f"  [blue]{i}.[/blue] {_esc(preview)}")
            if queue_lines:
                out += f"\n[b]Queued ({len(queue_lines)}):[/b]\n"
                out += "\n".join(queue_lines) + "\n"
                out += "[dim]Applied at the next user-turn boundary.[/dim]\n"
        return out

    def _update_mode_bar(self) -> None:
        """Refresh the single-line mode indicator above the Footer.

        Spells out at-a-glance whether the agent is RUNNING (the
        user can keep typing feedback, it queues for the next turn)
        or WAITING for them (Enter to continue step-mode, or type an
        answer to a model question). Idle = between sessions.

        Cheap; called from `_update_status` and any event handler that
        flips `_awaiting_kind` / `_session_done` / `_is_streaming`.
        """
        try:
            bar = self.query_one("#mode-bar", Static)
        except Exception:
            return
        # Sticky badges that don't depend on _awaiting_kind: step-mode
        # is ON for the whole session once /wait toggles it, even while
        # an iter is mid-stream. Selection mode is independent of the
        # session. Both render as small prefix badges so the user can
        # see the mode "in the bar with the commands" rather than only
        # at iter-boundary pause prompts.
        prefix_badges: list[str] = []
        if getattr(self.agent, "_step_mode", False):
            prefix_badges.append("[black on yellow] WAIT MODE [/]")
        if getattr(self, "_selection_mode_on", False):
            prefix_badges.append("[black on cyan] SELECT [/]")
        badge_prefix = " ".join(prefix_badges) + (" " if prefix_badges else "")

        if self._awaiting_kind == "step":
            body = (
                "[bold red]WAITING (step):[/bold red] "
                "press Enter to continue, or type feedback first"
            )
        elif self._awaiting_kind == "answer":
            body = (
                "[bold yellow]WAITING (answer):[/bold yellow] "
                "type your reply to the model's question"
            )
        elif self._session_done and self._awaiting_kind == "goal":
            body = "[dim]idle — type a new goal or /help[/dim]"
        elif self._session_done:
            body = "[dim]session ended — type feedback to extend, or /new[/dim]"
        elif self._is_streaming:
            body = "[bold green]RUNNING:[/bold green] streaming — feedback queues for next turn"
        else:
            body = "[bold green]RUNNING:[/bold green] feedback queues for next turn"
        bar.update(badge_prefix + body)

    def _format_ctx_row(self) -> str:
        """Compose the `Ctx: X / Y (Z%)` status row.

        Hidden when neither a max nor an active agent is available.
        Max is read once at session start from BackendInfo.context_length
        (Ollama populates this; MLX populates it via the new config.json
        sniff in backend.py). Fill estimate sums message chars on the
        agent and divides by 3.5 — middle ground between English prose
        (~4 cpt) and dense code (~3 cpt). Approximate; flagged with a
        yellow tint and `approx` label when above 80%.
        """
        if self._ctx_max is None and self.agent is None:
            return ""
        fill_chars = 0
        try:
            if self.agent is not None and hasattr(self.agent, "_estimate_ctx_fill"):
                fill_chars = int(self.agent._estimate_ctx_fill())
        except Exception:
            fill_chars = 0
        fill_tokens = int(fill_chars / 3.5) if fill_chars else 0
        # If we have no max AND no fill, hide.
        if self._ctx_max is None and fill_tokens == 0:
            return ""
        def _fmt(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            if n >= 1000:
                return f"{n / 1000:.1f}K"
            return str(n)
        if self._ctx_max is None:
            return f"[b]Ctx:[/b] {_fmt(fill_tokens)} [dim](max unknown)[/dim]\n"
        pct = (100.0 * fill_tokens / self._ctx_max) if self._ctx_max else 0.0
        # Approx label kicks in above 80% to draw attention to the
        # imprecision exactly where it matters most (you're about to
        # blow context and want to know whether the estimate is solid).
        suffix = f"({pct:.1f}%)" if pct < 80 else f"({pct:.0f}% approx)"
        body = f"{_fmt(fill_tokens)} / {_fmt(self._ctx_max)}  {suffix}"
        if pct >= 80:
            body = f"[yellow]{body}[/yellow]"
        return f"[b]Ctx:[/b] {body}\n"

    def _render_assets_block(self) -> str:
        """Sticky multi-line summary of the most recent asset batch.
        Empty when no assets have been generated this session."""
        if not self._assets_summary:
            return ""
        return f"\n{self._assets_summary}\n"

    def _render_sounds_block(self) -> str:
        """Sticky compact summary of the most recent sound batch.
        Empty when no sounds have been generated this session."""
        if not self._sounds_summary:
            return ""
        return f"\n{self._sounds_summary}\n"

    def _render_playbook_block(self) -> str:
        """Show which playbook bullets are currently injected in prompts,
        plus an honesty note about whether the writeback loop is live.

        The user explicitly asked for this — without seeing what's being
        injected, the playbook is invisible prompt bloat. Now they can
        watch retrieval per-turn and decide whether the bullets are
        helping or noise, and toggle the whole thing off with /playbook.
        """
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        if not getattr(agent, "_playbook_top_k", 0):
            return "\n[b]Playbook:[/b] [dim]disabled[/dim]\n"
        ids = list(getattr(agent, "_active_bullet_ids", []) or [])
        if not ids:
            return (
                "\n[b]Playbook:[/b] [dim]none retrieved this turn[/dim]\n"
            )
        # Pull scores from the playbook so the user can see whether a
        # bullet has accumulated any track record. All seed bullets are
        # at score 0 until writeback runs — surface that honestly so the
        # user knows they're looking at unvalidated rules.
        try:
            all_b = {b.id: b for b in agent._playbook.load_all()}
        except Exception:
            all_b = {}
        writeback_on = bool(getattr(agent, "_playbook_writeback", False))
        rows: list[str] = []
        for bid in ids:
            b = all_b.get(bid)
            if b is None:
                rows.append(f"  [dim]·[/dim] {bid} [dim](missing)[/dim]")
                continue
            sc = b.score()
            if sc > 0:
                tag = f"[green]+{sc}[/green]"
            elif sc < 0:
                tag = f"[red]{sc}[/red]"
            else:
                tag = "[dim]0[/dim]"
            rows.append(f"  [dim]·[/dim] {bid} {tag}")
        footer = (
            "[dim]writeback ON — scores update on pass/stuck[/dim]"
            if writeback_on
            else "[yellow]writeback OFF — scores never update; "
            "all bullets are unvalidated seeds[/yellow]"
        )
        return (
            f"\n[b]Playbook:[/b] {len(ids)} injected\n"
            + "\n".join(rows)
            + f"\n{footer}\n[dim]/playbook off | on | toggle[/dim]\n"
        )

    def _render_files_block(self) -> str:
        """Per-session file paths. Always shown when available so the
        user can `cat` / inspect / share without scrolling logs."""
        rows: list[str] = []
        if self._out_path is not None:
            rows.append(f"  [dim]game[/dim]    {self._out_path}")
        if self._best_path is not None and self._best_path.exists():
            rows.append(f"  [dim]best[/dim]    {self._best_path}")
        if self._trace_path is not None:
            rows.append(f"  [dim]trace[/dim]   {self._trace_path}")
        if self._assets_dir is not None and self._assets_dir.exists():
            rows.append(f"  [dim]assets[/dim]  {self._assets_dir}")
        if self._sounds_dir is not None and self._sounds_dir.exists():
            rows.append(f"  [dim]sounds[/dim]  {self._sounds_dir}")
        if self._log_file_path is not None:
            rows.append(f"  [dim]log[/dim]     {self._log_file_path}")
        if not rows:
            return ""
        return (
            "\n[b]Files:[/b]\n"
            + "\n".join(rows)
            + "\n[dim]Ctrl+L to reprint paths[/dim]\n"
        )

    # ----------------------------- actions --------------------------------

    async def action_ship_it(self) -> None:
        """Ctrl+D - ship the current build.

        Two-tap escalation:
          1. First tap: graceful ship — request_done(), iter-boundary
             check in agent.run() breaks out before the next stream.
          2. Second tap within 2s: force quit the whole TUI. The current
             stream may still be in-flight; Ctrl+Q (or this second tap)
             is the user's emergency exit.
        """
        now = time.monotonic()
        last = getattr(self, "_last_ship_request_at", 0.0) or 0.0
        if last and (now - last) < 2.0:
            self._log_info(
                "[red]Second Ctrl+D within 2s — force-quitting.[/red] "
                "(In-flight stream may take a moment to release the model.)"
            )
            self.exit()
            return
        self._last_ship_request_at = now
        if self.agent is None:
            return
        self.agent.request_done()
        self._log_info(
            "[yellow]Ship requested.[/yellow] Agent will break out at the next "
            "iteration boundary (current stream finishes first). "
            "[dim]Press Ctrl+D again within 2s to force-quit.[/dim]"
        )

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
        jsonl = traces / (stem + ".jsonl")
        self._log("[bold cyan]── log artifacts ──[/bold cyan]")
        self._log(f"  game file:    {self._out_path}")
        self._log(f"  full log:     {self._log_file_path}")
        self._log(f"  jsonl trace:  {jsonl}")
        self._log(f"  conversation: {traces / (stem + '.conversation.md')}")
        self._log(f"  snapshots:    {snaps}")
        if self._best_path is not None:
            self._log(f"  best clean:   {self._best_path}")
        self._log("[dim]Tip: paste the full log above into your AI assistant to debug.[/dim]")
        # The jsonl trace gets `stream_heartbeat` entries every 30s
        # during a long stream — invaluable when the model goes off
        # the rails (e.g. emits 200 duplicate sprite specs and stalls
        # 25 minutes later). Run from another terminal:
        self._log("[dim]Live progress (run from another terminal):[/dim]")
        self._log(f"[dim]  tail -f {jsonl}[/dim]")
        self._log("[dim]Press Ctrl+S to enable mouse selection in this pane.[/dim]")

    async def action_toggle_selection_mode(self) -> None:
        """Ctrl+S - toggle Textual's mouse tracking so the terminal can
        handle drag-select. Useful for copying log content while the
        agent is running. Press Ctrl+S again to resume normal TUI mouse.
        On terminals that natively bypass app mouse capture with a
        modifier (iTerm2: hold Option; most Linux terms: hold Shift),
        you can also drag-select without toggling — but Ctrl+S works
        everywhere."""
        # The earlier `set_mouse_capture` approach was a no-op on
        # Textual 8.x — that method doesn't exist, and `capture_mouse`
        # only re-routes events between widgets; the terminal still
        # consumes the mouse-tracking escape sequence, so drag-select
        # never reached the terminal. The driver-level
        # _enable_mouse_support / _disable_mouse_support pair emits the
        # actual `CSI ?1000l` (off) / `CSI ?1000h` (on) sequences that
        # toggle whether the terminal sees the mouse at all. Private
        # API by underscore convention, but stable across recent
        # Textual releases; guarded with hasattr so we degrade
        # gracefully on future versions.
        new_state = not getattr(self, "_selection_mode_on", False)
        driver = getattr(self, "_driver", None)
        applied = False
        try:
            if new_state:
                if driver is not None and hasattr(driver, "_disable_mouse_support"):
                    driver._disable_mouse_support()
                    applied = True
            else:
                if driver is not None and hasattr(driver, "_enable_mouse_support"):
                    driver._enable_mouse_support()
                    applied = True
        except Exception:
            # Even if the API path fails, surfacing the hint is
            # valuable — modifier-key drag-select still works.
            applied = False
        self._selection_mode_on = new_state
        if new_state:
            if applied:
                self._log_info(
                    "[bold yellow]selection mode ON[/bold yellow] — "
                    "drag-select with the mouse to copy. "
                    "[dim]Ctrl+S again to resume normal TUI mouse.[/dim]"
                )
            else:
                # API path unavailable — fall back to the modifier hint.
                self._log_info(
                    "[yellow]selection toggle unavailable on this "
                    "Textual build.[/yellow] Hold [b]Option[/b] (iTerm2) "
                    "or [b]Shift[/b] (most Linux terms) while dragging "
                    "to bypass mouse capture without toggling."
                )
        else:
            self._log_info(
                "[dim]selection mode OFF — mouse handed back to TUI[/dim]"
            )
        self._update_mode_bar()

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

        # Step-mode (Stop-Losing-To-OneShot todo #1): when the agent has
        # paused between iters, empty Enter means "continue" and any
        # non-empty text falls through to the normal feedback path
        # (which also unblocks the agent-side wait via
        # has_pending_user_input becoming True). Either way we exit
        # step-state so the next event resets routing.
        if self._awaiting_kind == "step":
            self._awaiting_kind = "feedback"
            message.input.placeholder = "feedback · 'done' or Ctrl+D to ship · /help"
            self._update_mode_bar()
            if not text:
                if self.agent is not None:
                    self.agent.signal_step_continue()
                    self._log_info("[dim]→ continuing iteration[/dim]")
                return
            # Non-empty: fall through to the regular dispatch below.

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
            self._update_status()

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
            # Refresh the right-hand status panel so the new entry shows up
            # under "Queued (N)" immediately. Without this the panel only
            # refreshes on the next agent event, which can be 30+s away.
            self._update_status()

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
            elif cmd in ("model", "load"):
                self._cmd_set_model(arg)
            elif cmd == "backend":
                self._cmd_set_backend(arg)
            elif cmd == "unload":
                self._cmd_unload(arg)
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
            elif cmd == "wait":
                self._cmd_toggle_wait(arg)
            elif cmd in ("playbook", "memory"):
                self._cmd_toggle_playbook(arg)
            elif cmd == "audit":
                self._cmd_audit_playbook()
            elif cmd == "restarts":
                self._cmd_set_restarts(arg)
            elif cmd in ("model-class", "modelclass"):
                self._cmd_set_model_class(arg)
            elif cmd == "launch":
                self._cmd_launch_mlx(arg)
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
            "[bold cyan]── redraw ONE asset (no code change) ──[/bold cyan]",
            "  [dim]Use the asset name + a media verb + a code-lock phrase.[/dim]",
            "  [b]template[/b]  redraw the [italic]<name>[/italic] asset as [italic]<new visual>[/italic], no code changes",
            "  [b]examples[/b]  redraw the [italic]player_ship[/italic] asset as a pink heart with sparkles, only the asset",
            "                  remake the [italic]centipede_tail[/italic] sprite — rounder, two animated legs, no code changes",
            "                  regenerate the [italic]mushroom[/italic] art — taller, deep red cap, just the asset",
            "  [dim]Triggers MEDIA-CHANGE DIRECTIVE; agent emits <assets> with the same name —[/dim]",
            "  [dim]PNG is replaced in place, drawSprite() is left alone, code-rewrite gate is closed.[/dim]",
            "",
            "[bold cyan]── remake ONE sound (no code change) ──[/bold cyan]",
            "  [b]template[/b]  remake the [italic]<name>[/italic] sound as [italic]<new audio>[/italic], no code changes",
            "  [b]examples[/b]  remake the [italic]laser[/italic] sound — deeper, punchier 8-bit, only the audio",
            "                  regenerate the [italic]explosion[/italic] sfx as a wet thud, just the sound",
            "                  redo the [italic]music[/italic] track — slower chiptune, no code changes",
            "  [dim]Triggers <sounds> re-render with same name; new Audio() call already in the file picks it up.[/dim]",
            "",
            "[dim]After <done/> the input box reads [b]'feedback to extend, /new <goal>"
            " for a fresh game'[/b] — that's the trigger for auto-extend.[/dim]",
            "",
            "[bold cyan]── images, animation, sound ──[/bold cyan]",
            "  [dim]The agent decides per session; you nudge by what you write in the goal.[/dim]",
            "  [b]sprites (txt2img)[/b]        model emits [b]<assets>[/b] in Phase A → Z-Image-Turbo PNGs",
            "                                  saved next to the .html. Encourage with [italic]sprite[/italic],",
            "                                  [italic]pixel-art[/italic], [italic]icon[/italic], [italic]texture[/italic], [italic]cool art[/italic] in your goal.",
            "  [b]animation frames (img2img)[/b]  model adds [b]from_image[/b] + [b]strength[/b] to an asset",
            "                                  → SD-Turbo seeds frame N from frame N-1. Encourage with",
            "                                  [italic]walk cycle[/italic], [italic]animated[/italic], [italic]two-frame[/italic], [italic]flap[/italic] in your goal.",
            "  [b]sound effects (txt2audio)[/b] model emits [b]<sounds>[/b] in Phase A → Stable Audio Open",
            "                                  OGGs saved next to the .html. Encourage with [italic]sound[/italic],",
            "                                  [italic]audio[/italic], [italic]sfx[/italic], [italic]music[/italic], [italic]chiptune[/italic] in your goal.",
            "  [b]opt out[/b]                  launch with [b]SKIP_DIFFUSER_PRELOAD=1[/b] env var to skip",
            "                                  the ~15-30 s diffuser preload on startup; sessions that",
            "                                  don't request assets/sounds are unaffected either way.",
            "  [b]smoke tests[/b]              [b]scripts/_smoke_doom.py[/b] (sprite), [b]_smoke_img2img.py[/b]",
            "                                  (animation), [b]_smoke_audio.py[/b] (sound) — run after install.",
            "",
            "[bold cyan]── slash commands ──[/bold cyan]",
            "  [b]/help[/b]                    show this help (also /h, /?)",
            "  [b]/list[/b]                    unified Ollama + MLX list with numbers (also /models)",
            "  [b]/load <N|name>[/b]           pick model #N from /list (any backend) · STICKY across /new (also /model)",
            "  [b]/launch <N|name|path>[/b]   stage an MLX model for next /new (loads in-process on first request)",
            "  [b]/backend <auto|ollama|mlx>[/b]  stage default backend when no specific model is staged",
            "  [b]/unload [N|name|all|mlx][/b]  free VRAM · #N from /list · bare = active session · all = every Ollama · mlx = drop the in-process MLX model",
            "  [b]/seed <path>[/b]             stage a baseline .html (STICKY across /new) · /seed alone clears",
            "  [b]/iters <N>[/b]               set max iterations (sticky)",
            "  [b]/restarts <N>[/b]            independent full restarts when iter-1 score < 60 (sticky · default 2 · 1=off)",
            "  [b]/model-class <auto|small|mid|large>[/b]  override prompt-size trim (sticky · default 'small' = lean ~5KB)",
            "  [b]/reset[/b]                   wipe ALL staged state (seed + model + iters → defaults)",
            "  [b]/new <goal>[/b]              end current session, start a fresh one (uses staged seed/model)",
            "  [b]/ship[/b]                    ship current build (= Ctrl+D, or type 'done')",
            "  [b]/open[/b]                    open the current game in your default browser",
            "  [b]/log[/b]                     print all session artifact paths (= Ctrl+L; also /paths, /files)",
            "  [b]/clear[/b]                   clear the agent log pane (does not affect staged state)",
            "  [b]/status[/b]                  print model, phase, iteration, paths, what's staged",
            "  [b]/wait[/b] [on|off]            toggle step-mode: pause after each iter; Enter or feedback to continue",
            "  [b]/playbook[/b] [on|off]        toggle playbook bullet injection (alias /memory) - A/B vs one-shot when iters feel worse than no agent",
            "  [b]/audit[/b]                     print per-bullet earnings (fires, pass-rate, avg-iter) from trace history",
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

    def _refresh_listing(self) -> list[tuple[str, str]]:
        """Build a unified (backend, model) list across both daemons.

        Order: every Ollama installed tag first (loaded + unloaded), then
        every MLX downloaded model. Stable across calls so the numbers
        the user just saw in /list mean the same thing in /load N.
        Stored on self._last_listing for the /load handler to consume.
        """
        listing: list[tuple[str, str]] = []
        ollama_installed, _ = backend_mod.list_ollama_inventory()
        for name in ollama_installed:
            listing.append(("ollama", name))
        mlx_downloaded, _ = backend_mod.list_mlx_inventory()
        for name in mlx_downloaded:
            listing.append(("mlx", name))
        self._last_listing = listing
        return listing

    def _cmd_list_models(self) -> None:
        listing = self._refresh_listing()
        ollama_installed, ollama_loaded = backend_mod.list_ollama_inventory()
        mlx_downloaded, mlx_active = backend_mod.list_mlx_inventory()

        if not listing:
            self._log_error(
                "no LLM backend reachable — start ollama "
                "(`ollama run <model>`) or set MLX_MODEL / drop an MLX "
                "model under ~/MLX_Models so the in-process backend can "
                "find it"
            )
            return

        self._log("[bold cyan]── available models ──[/bold cyan]")
        self._log(
            "[dim]  [O]/[M] = backend  ·  * = loaded right now  ·  "
            "← active = this session  ·  ← staged = next /new[/dim]"
        )
        # Track which model the next /new will resolve to so it gets the
        # ← staged marker even when the user hasn't typed /load yet.
        staged_backend = self._next_backend or "ollama"
        for i, (b, name) in enumerate(listing, 1):
            if b == "ollama":
                tag = "O"
                loaded = "*" if name in ollama_loaded else " "
                is_active = (
                    self._session_backend_info is not None
                    and self._session_backend_info.name == "ollama"
                    and name == self._session_model
                )
                is_staged = name == self._next_model and staged_backend == "ollama"
            else:
                tag = "M"
                loaded = "*" if name == mlx_active else " "
                is_active = (
                    self._session_backend_info is not None
                    and self._session_backend_info.name == "mlx"
                    and name == self._session_model
                )
                is_staged = name == self._next_model and staged_backend == "mlx"
            mark_active = "  [yellow]← active[/yellow]" if is_active else ""
            mark_staged = "  [magenta]← staged[/magenta]" if is_staged else ""
            # Show MLX entries by short basename when they're disk paths
            # (avoids screen-eating absolute paths for the common case).
            # The full path is still what /load N picks for launching.
            if b == "mlx" and "/" in name:
                display = Path(name).name
                hint = f"  [dim]({Path(name).parent})[/dim]"
            else:
                display = name
                hint = ""
            self._log(
                f"  [{i:>2}] [b]{tag}[/b] {loaded} {_esc(display)}"
                f"{mark_active}{mark_staged}{hint}"
            )
        self._log(
            "[dim]Use [b]/load N[/b] (or /model N) to stage by number, or "
            "[b]/load <name>[/b] for a substring match. /backend toggles the "
            "default daemon when no specific model is staged.[/dim]"
        )

    def _cmd_unload(self, arg: str) -> None:
        """/unload [N|name|all|mlx] — free VRAM held by an LLM daemon.

          /unload                     unload the active session's model
                                      (Ollama API only — for MLX use /unload mlx)
          /unload <N>                 unload entry #N from /list (any backend)
          /unload <name>              unload by exact tag or substring match
          /unload all                 unload every model loaded in Ollama
                                      (does NOT touch MLX — use /unload mlx)
          /unload mlx                 drop the in-process MLX model from VRAM
                                      (next /new will reload on first request)

        Models stay installed on disk; only the VRAM allocation is released.
        """
        norm = arg.strip()
        norm_lc = norm.lower()

        if norm_lc == "mlx":
            self._print_mlx_kill_hint()
            return

        if norm_lc == "all":
            results = backend_mod.unload_all_ollama_models()
            if not results:
                self._log_info("[dim]no Ollama models currently loaded[/dim]")
                return
            for name, ok, msg in results:
                tag = "[green]✓[/green]" if ok else "[red]✗[/red]"
                self._log_info(f"  {tag} {_esc(name)} — {_esc(msg)}")
            self._log_info("[dim](MLX untouched — /unload mlx for that.)[/dim]")
            return

        # Default (no arg): unload the active session's Ollama model.
        if not norm:
            if (
                self._session_backend_info is None
                or self._session_backend_info.name != "ollama"
            ):
                self._log_info(
                    "no active Ollama session to unload. Try [b]/unload <N>[/b] "
                    "(number from /list), [b]/unload all[/b] to evict every "
                    "loaded Ollama model, or [b]/unload mlx[/b] for MLX hints."
                )
                return
            self._unload_ollama_named(self._session_backend_info.model)
            return

        # Resolve the argument against the unified /list. Same matching
        # rules as /load so the user can index into the same numbered
        # list they just saw.
        backend_name, model_name = self._resolve_listing_arg(norm)
        if model_name is None:
            return  # error already logged
        if backend_name == "mlx":
            self._log_info(
                f"[yellow]{_esc(model_name)}[/yellow] is an MLX model — "
                "MLX is single-process-per-model and has no unload API."
            )
            self._print_mlx_kill_hint()
            return
        # Ollama tag — issue the unload.
        self._unload_ollama_named(model_name)

    def _resolve_listing_arg(self, arg: str) -> tuple[str | None, str | None]:
        """Match `arg` (number, exact tag, or substring) against /list.

        Refreshes the listing on demand if /list hasn't been run. Returns
        (backend_name, model_name) on hit, (None, None) on miss/ambiguity
        AND logs the appropriate error so the caller can simply return.
        """
        if not self._last_listing:
            self._refresh_listing()
        listing = self._last_listing
        if not listing:
            self._log_error(
                "no LLM backend reachable — start ollama, or set MLX_MODEL / drop a model under ~/MLX_Models"
            )
            return None, None

        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(listing):
                b, n = listing[idx]
                return b, n
            self._log_error(
                f"out of range: /list has {len(listing)} entries (1-{len(listing)})"
            )
            return None, None

        for b, n in listing:
            if arg == n:
                return b, n

        needle = arg.lower()
        matches = [(b, n) for b, n in listing if needle in n.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = [f"{b.upper()}:{n}" for b, n in matches]
            self._log_error(f"ambiguous {arg!r} — matches: {names}")
            return None, None
        self._log_error(f"no match for {arg!r} — try /list")
        return None, None

    def _unload_ollama_named(self, name: str) -> None:
        ok, msg = backend_mod.unload_ollama_model(name)
        tag = "[green]✓[/green]" if ok else "[red]✗[/red]"
        self._log_info(f"{tag} {_esc(name)} — {_esc(msg)}")

    def _print_mlx_kill_hint(self) -> None:
        """/unload mlx — MLX now runs in-process. Drop the cached model
        from VRAM by clearing MLXBackend's class-level cache and
        forcing a GC pass. Next /new will reload (or pick a different
        model if MLX_MODEL changed)."""
        loaded = backend_mod.MLXBackend._loaded_path
        if not loaded:
            self._log_info("[dim]no MLX model is currently loaded[/dim]")
            return
        backend_mod.MLXBackend._loaded_model = None
        backend_mod.MLXBackend._loaded_tokenizer = None
        backend_mod.MLXBackend._loaded_path = None
        import gc
        gc.collect()
        self._log_info(
            f"[green]✓[/green] released [b]{_esc(loaded)}[/b] from VRAM. "
            "Next /new will reload on first request."
        )
        # Legacy hint for users still running the old mlx_lm.server.
        pids = backend_mod.mlx_server_pids()
        if pids:
            self._log(
                f"[dim]Also detected stale mlx_lm.server pid(s) "
                f"{' '.join(str(p) for p in pids)} — those are no longer "
                f"used; kill them with `pkill -f mlx_lm.server` to free "
                f"server-side VRAM.[/dim]"
            )

    def _cmd_set_backend(self, arg: str) -> None:
        """/backend [auto|ollama|mlx] — pick the LLM daemon for the next /new.

        Sticky across /new, like /model. Bare /backend prints the current
        staged choice and what would resolve right now. Useful when both
        Ollama and mlx_lm.server are running and you want to force one.
        """
        norm = arg.strip().lower()
        if not norm:
            current = self._next_backend or "auto"
            self._log_info(
                f"staged backend (next /new): [b]{current}[/b]. "
                "Pass /backend ollama, /backend mlx, or /backend auto."
            )
            try:
                preview = backend_mod.detect_backend(self._next_backend)
                self._log_info(
                    f"would resolve to → [b]{preview.name.upper()}[/b] · "
                    f"[b]{_esc(preview.model)}[/b] [dim]({_esc(preview.source)})[/dim]"
                )
            except RuntimeError as e:
                self._log_info(f"[dim]({e})[/dim]")
            return
        if norm in ("auto", "any", "default"):
            self._next_backend = None
            self._log_info("backend → [b]auto[/b] (probe both, MLX wins ties)")
            return
        if norm in ("ollama", "ol", "o"):
            self._next_backend = "ollama"
            self._log_info("backend → [b]ollama[/b] (sticky)")
            return
        if norm in ("mlx", "m"):
            self._next_backend = "mlx"
            self._log_info("backend → [b]mlx[/b] (sticky)")
            return
        self._log_error(
            f"unknown backend {arg!r} — pick one of: auto, ollama, mlx"
        )

    def _cmd_set_model(self, arg: str) -> None:
        """/model <N|name> — pick by global number from /list, or by substring.

        N is the unified-list index across both Ollama and MLX (the
        number printed in /list). Substring matches against any
        installed Ollama tag or downloaded MLX id; ambiguous matches
        require disambiguation. Bare /model clears the staged model.
        """
        if not arg:
            if self._next_model is None:
                self._log_info(
                    "no staged model (usage: /model <number-from-/list-or-name>)"
                )
            else:
                self._log_info(f"cleared staged model (was: {self._next_model})")
                self._next_model = None
                # Don't clear _next_backend — user may still want to force a
                # specific daemon for the next /new.
            return

        # Refresh the unified listing if /list hasn't been run yet.
        if not self._last_listing:
            self._refresh_listing()
        listing = self._last_listing
        if not listing:
            self._log_error(
                "no LLM backend reachable — start ollama, or set MLX_MODEL / drop a model under ~/MLX_Models"
            )
            return

        chosen_backend: str | None = None
        chosen_name: str | None = None

        # 1) Numeric → unified-list index.
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(listing):
                chosen_backend, chosen_name = listing[idx]
            else:
                self._log_error(
                    f"out of range: /list has {len(listing)} entries (1-{len(listing)})"
                )
                return

        # 2) Exact full-string match (e.g. user pasted a tag).
        if chosen_name is None:
            for b, name in listing:
                if arg == name:
                    chosen_backend, chosen_name = b, name
                    break

        # 3) Case-insensitive substring match. Ambiguity is an error so we
        #    don't silently pick the wrong model.
        if chosen_name is None:
            needle = arg.lower()
            matches = [(b, n) for b, n in listing if needle in n.lower()]
            if len(matches) == 1:
                chosen_backend, chosen_name = matches[0]
            elif len(matches) > 1:
                names = [f"{b.upper()}:{n}" for b, n in matches]
                self._log_error(f"ambiguous {arg!r} — matches: {names}")
                return

        if chosen_name is None or chosen_backend is None:
            self._log_error(f"no match for {arg!r} — try /list")
            return

        # Stage backend + model together. _start_session honors this pair
        # over plain detect_backend so the user's pick wins even when both
        # daemons are running.
        self._next_backend = chosen_backend
        self._next_model = chosen_name
        backend_label = chosen_backend.upper()
        self._log_info(
            f"staged [b]{backend_label}[/b] · [b]{_esc(chosen_name)}[/b] "
            "for next /new session [dim](current session keeps its model)[/dim]"
        )

        # MLX-specific: warn if the staged model differs from the
        # currently in-VRAM model. The in-process loader will swap on
        # first request (~30-60s pause); the user should know.
        if chosen_backend == "mlx":
            mlx_active = backend_mod.MLXBackend._loaded_path
            if mlx_active is None:
                self._log_info(
                    f"[dim]MLX runs in-process; weights for "
                    f"[b]{_esc(chosen_name)}[/b] will load on the first "
                    f"request of /new (~30-60s the first time).[/dim]"
                )
            elif mlx_active != chosen_name:
                self._log_info(
                    f"[yellow]heads-up:[/yellow] [b]{_esc(mlx_active)}[/b] "
                    f"is currently loaded in VRAM. The first request of /new "
                    f"will swap to the staged model (~30-60s pause). To "
                    f"preload immediately, run [b]/unload mlx[/b] then "
                    f"trigger a generation."
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

    def _cmd_set_restarts(self, arg: str) -> None:
        """/restarts N — when iter 1 of a session ends below the score
        threshold (60/100), throw it away and try again from scratch up
        to N total times. Best-by-score wins. Default 2 (cheap insurance
        — simple games pass iter 1 and never restart; hard games get a
        second clean attempt). Set to 1 to disable.
        """
        if not arg.isdigit() or int(arg) <= 0:
            self._log_info(
                f"usage: /restarts <positive int>  (current: {self._restart_n}). "
                "Default 2; set to 1 to disable, 3+ for harder games."
            )
            return
        self._restart_n = int(arg)
        self._log_info(
            f"restart-N set to [b]{self._restart_n}[/b] for next session"
        )

    def _cmd_launch_mlx(self, arg: str) -> None:
        """/launch <N|name|path> — stage an MLX model for the next /new.

        MLX now runs in-process (no separate mlx_lm.server). "Launching"
        means selecting which model the in-process backend will load on
        the next /new — the actual weight load happens lazily on the
        first model interaction. To swap models mid-session, first run
        /unload mlx to free the currently-loaded weights.
        """
        if not arg.strip():
            self._log_info(
                "usage: /launch <N|name|path>  — pick an MLX entry from "
                "/list to stage for the next /new (loads in-process)"
            )
            return

        backend_name, model_name = self._resolve_listing_arg(arg.strip())
        if model_name is None:
            return  # error already logged
        if backend_name != "mlx":
            self._log_error(
                f"{model_name!r} is an Ollama tag — /launch is for MLX. "
                "Ollama loads on demand; just /load it and run /new."
            )
            return

        currently_loaded = backend_mod.MLXBackend._loaded_path
        self._next_backend = "mlx"
        self._next_model = model_name
        msg = (
            f"[green]✓[/green] staged MLX model [b]{_esc(model_name)}[/b] "
            "for next /new"
        )
        if currently_loaded and currently_loaded != model_name:
            msg += (
                f" · [yellow]note:[/yellow] [b]{_esc(currently_loaded)}[/b] "
                f"is still resident in VRAM — run [b]/unload mlx[/b] before "
                f"/new if you want to free it first (otherwise the in-process "
                f"loader swaps weights at first request, ~30-60s)"
            )
        self._log_info(msg)

    def _cmd_set_model_class(self, arg: str) -> None:
        """/model-class auto|small|mid|large — override the system-prompt
        trim. Default 'auto' = 'small' (lean ~5 KB schema, drops
        <assets>/<sounds>/<lookup_bullet>) — biased for mid-size local
        LLMs and one-shot strength. Pass 'large' only when running a
        frontier-tier model. We never inspect model names.
        """
        choices = {"auto", "small", "mid", "large"}
        a = (arg or "").strip().lower()
        if a not in choices:
            cur = self._model_class or "auto"
            self._log_info(
                f"usage: /model-class <auto|small|mid|large>  (current: {cur})"
            )
            return
        self._model_class = None if a == "auto" else a
        self._log_info(
            f"model-class set to [b]{a}[/b] for next session"
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
        had_backend = self._next_backend
        had_iters = self._max_iters
        had_restarts = self._restart_n
        had_class = self._model_class
        self._next_seed = None
        self._next_model = None
        self._next_backend = None
        self._max_iters = 6
        self._restart_n = 2
        self._model_class = None
        bits: list[str] = []
        if had_seed is not None:
            bits.append(f"seed={had_seed}")
        if had_model is not None:
            bits.append(f"model={had_model}")
        if had_backend is not None:
            bits.append(f"backend={had_backend}")
        if had_iters != 6:
            bits.append(f"iters={had_iters}→6")
        if had_restarts != 2:
            bits.append(f"restarts={had_restarts}→2")
        if had_class:
            bits.append(f"model-class={had_class}→auto")
        if not bits:
            self._log_info("nothing to reset (no staged seed/model, iters at default)")
            return
        self._log_info(
            f"[yellow]reset:[/yellow] cleared {', '.join(bits)}. "
            "Next /new starts from defaults. [dim](Run /new <goal> to start fresh.)[/dim]"
        )

    def _cmd_status(self) -> None:
        # Step-mode shows up here so users can confirm whether the
        # agent will pause between iters or run continuously.
        step_label = "—"
        if self.agent is not None:
            step_label = "ON" if getattr(self.agent, "_step_mode", False) else "off"
        lines = [
            "[bold cyan]── status ──[/bold cyan]",
            f"  backend (active):  {_esc(self._session_backend_info.name if self._session_backend_info else '—')}",
            f"  backend (next /new): {_esc(self._next_backend or '(auto)')}",
            f"  model (active):    {_esc(self._session_model or '—')}",
            f"  model (next /new): {_esc(self._next_model or '(auto-detect)')}",
            f"  goal:              {_esc(self._goal or '—')}",
            f"  phase:             {_esc(self._phase_label)}",
            f"  iteration:         {_esc(self._iteration_label)}",
            f"  max iters:         {self._max_iters}",
            f"  restart-N:         {self._restart_n if self._restart_n > 1 else '1 (off)'}",
            f"  model-class:       {self._model_class or 'auto (= small, lean ~5KB schema)'}",
            f"  step-mode (/wait): {step_label}",
            f"  staged seed:       {_esc(str(self._next_seed) if self._next_seed else '—')}",
            f"  session done:      {self._session_done}",
            f"  game file:         {self._out_path or '—'}",
            f"  log file:          {self._log_file_path or '—'}",
        ]
        for line in lines:
            self._log(line)

    def _cmd_toggle_wait(self, arg: str) -> None:
        """/wait — toggle step-mode (pause after each iter and wait for
        explicit user input before continuing). /wait on or /wait off
        for explicit set. Stop-Losing-To-OneShot todo #1: makes the user
        the verifier between iters, the strongest defense against
        'iter 2 wrecks iter 1' for mid-tier models."""
        if self.agent is None:
            self._log_info(
                "no active session — start one and try /wait again "
                "(it applies once an agent is running)"
            )
            return
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1"):
            new_state = True
        elif arg_lc in ("off", "false", "0"):
            new_state = False
        else:
            new_state = not getattr(self.agent, "_step_mode", False)
        self.agent.set_step_mode(new_state)
        if new_state:
            self._log_info(
                "[yellow]step-mode ON[/yellow] — agent will pause after each "
                "iter. Press Enter to continue, or type feedback to inject "
                "before the next turn."
            )
        else:
            self._log_info("step-mode off — agent will run iterations continuously.")
        self._update_status()
        # Surface the new mode in the bottom bar immediately, not only
        # at the next iter-pause event.
        self._update_mode_bar()

    def _cmd_audit_playbook(self) -> None:
        """/audit — shell out to scripts/audit_playbook.py and print
        the table inline so the user can judge bullet earnings without
        leaving the TUI."""
        import subprocess
        try:
            out = subprocess.run(
                [".venv/bin/python", "scripts/audit_playbook.py"],
                capture_output=True, text=True, timeout=10,
            )
            text = (out.stdout or "").strip() or "(no output)"
            self._log(f"[bold cyan]── playbook audit ──[/bold cyan]")
            for line in text.splitlines():
                self._log_raw(line)
        except FileNotFoundError:
            self._log_info("audit script not found — pull latest main")
        except Exception as e:
            self._log_info(f"audit failed: {e}")

    def _cmd_toggle_playbook(self, arg: str) -> None:
        """/playbook (or /memory) — toggle playbook injection.

        The playbook injects rule-of-thumb bullets into each prompt at
        plan + code stages. They retrieve by weighted Jaccard against
        the goal; relevance matters less than precision of tags. If a
        run is performing worse than one-shot without the agent, the
        most likely culprit is the playbook injecting low-relevance
        bullets that distract a mid-tier local model. Disable with
        /playbook off and re-run the same goal to A/B compare.

        Persists for the active session: flipping top-K to 0 stops
        retrieval entirely on the current GameAgent. A future /new will
        inherit the agent's defaults again unless you flip it before
        the next start.
        """
        if self.agent is None:
            self._log_info(
                "no active session — playbook setting applies once an "
                "agent is running. Start a session, then /playbook off "
                "to A/B against one-shot."
            )
            return
        arg_lc = arg.strip().lower()
        currently_on = bool(getattr(self.agent, "_playbook_top_k", 0))
        if arg_lc in ("off", "false", "0", "disable"):
            new_on = False
        elif arg_lc in ("on", "true", "1", "enable"):
            new_on = True
        else:
            new_on = not currently_on
        if new_on:
            # Default the K back to 6 (the constructor default in
            # GameAgent.__init__) so re-enabling matches a fresh run.
            self.agent._playbook_top_k = 6
            self._log_info(
                "[green]playbook ON[/green] — bullets will inject on "
                "the next prompt. Watch the [b]Playbook[/b] section in "
                "the status panel to see what's being added."
            )
        else:
            self.agent._playbook_top_k = 0
            # Also clear the currently-active list so the status panel
            # immediately reflects "none retrieved" instead of stale ids.
            self.agent._active_bullet_ids = []
            self._log_info(
                "[yellow]playbook OFF[/yellow] — no bullets will inject "
                "on subsequent prompts. Run the same goal with this off "
                "and on to A/B whether it helps or hurts."
            )
        self._update_status()

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

        # Resolve the LLM backend (Ollama or MLX) and the model id within
        # it. Three sticky-staging tiers, in order of specificity:
        #   1. /model <N> or /load <N>  → both _next_backend AND _next_model
        #      were set to a specific (backend, model) from /list. Use it
        #      directly; this is the user's most explicit pick.
        #   2. /model <name> alone (legacy) → _next_model only, treat as
        #      Ollama tag.
        #   3. /backend <auto|ollama|mlx> → run detect_backend with the
        #      preference applied.
        # Clear with the bare /model and /backend commands.
        if self._next_backend in ("ollama", "mlx") and self._next_model:
            endpoint = (
                backend_mod.mlx_endpoint_url() if self._next_backend == "mlx"
                else backend_mod.ollama_endpoint_url()
            )
            info = backend_mod.BackendInfo(
                name=self._next_backend, model=self._next_model,
                source=f"/{'load' if self._next_backend == 'mlx' else 'model'} staged (sticky)",
                endpoint=endpoint,
            )
        elif self._next_model:
            info = backend_mod.BackendInfo(
                name="ollama", model=self._next_model,
                source="/model staged (sticky)",
                endpoint=backend_mod.ollama_endpoint_url(),
            )
        else:
            try:
                info = backend_mod.detect_backend(self._next_backend)
            except RuntimeError as e:
                self._log_error(str(e))
                self._session_done = True
                return
        try:
            self._session_backend = backend_mod.make_backend(info)
        except Exception as e:
            self._log_error(f"could not initialize backend: {e}")
            self._session_done = True
            return
        self._session_backend_info = info
        model_name = info.model
        self._session_model = model_name
        self.title = f"JMR's Coding Box — {info.name.upper()} · {model_name}"
        self._log_info(
            f"Using [b]{info.name.upper()}[/b] · [b]{_esc(model_name)}[/b] "
            f"[dim]({_esc(info.source)})[/dim]"
        )
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
            backend=self._session_backend,
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
            model_class=self._model_class or "auto",
            restart_n=self._restart_n,
            restart_score_threshold=self._restart_threshold,
            # Playbook injection OFF by default — across 6 ON/OFF
            # bench pairs on a 27B local model, OFF beat ON 5/6.
            # The "kitchen sink" effect (8 bullets of advice at plan
            # stage) hurt more than it helped. Users can /playbook on
            # to enable; status panel shows the current state.
            playbook_top_k=0,
            # Writeback still on so that when the user enables the
            # playbook (/playbook on), session outcomes update the
            # counters. Safe even when top_k=0 — writeback only fires
            # when bullets actually retrieved.
            playbook_writeback=True,
        )
        self.agent.set_token_callback(self._emit_token)

        # Surface per-session paths in the status panel. The agent owns
        # the canonical paths; we mirror them here so the panel stays
        # accurate even if the user typed /open or /new mid-flight.
        self._trace_path = self.agent.trace_path
        self._assets_dir = self._out_path.parent / f"{basename}_assets"
        self._sounds_dir = self._out_path.parent / f"{basename}_sounds"
        # Reset rolling status state for the new session — sticky values
        # from a prior session would mislead the user about THIS one.
        self._reset_status_state()
        # Stash the backend's reported context window for the new
        # status row. None = backend didn't expose it; the row hides.
        try:
            backend = getattr(self.agent, "_backend", None)
            info = getattr(backend, "info", None) if backend else None
            self._ctx_max = getattr(info, "context_length", None) if info else None
        except Exception:
            self._ctx_max = None

        self._open_log_mirror(basename)

        # Spawn the agent loop as a background task so the TUI stays responsive.
        self.run_worker(self._consume_events(goal, continuation=False), exclusive=True)

    def _reset_status_state(self) -> None:
        """Clear rolling status fields between sessions."""
        self._activity_label = ""
        self._activity_started_at = 0.0
        self._stream_tokens = 0
        self._stream_started_at = 0.0
        self._last_token_at = 0.0
        self._is_streaming = False
        self._assets_summary = ""
        self._sounds_summary = ""
        self._sounds_dir = None
        self._probes_passed = None
        self._probes_total = None
        self._last_diagnose = None
        self._ctx_max = None
        self._streak_clean = 0
        self._streak_stuck = 0

    def _tick_status(self) -> None:
        """Periodic refresh of the status panel — only repaints when
        there's something time-sensitive to update (active stream,
        non-idle activity). Idle UI doesn't need re-renders."""
        if self._is_streaming or self._activity_label:
            try:
                self._update_status()
            except Exception:
                pass

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
            async for ev in self.agent.run_with_restarts(goal, continuation=continuation):
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
        # Refresh the status panel on EVERY event so the queued-feedback
        # list disappears the moment the agent drains it (the agent's
        # internal flush happens between events, not via a callback).
        # Cheap render — Textual diffs the Static content.
        try:
            self._update_status()
        except Exception:
            pass
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
            # Capture probe pass/fail counts for the iteration line. The
            # agent's test event exposes `probes` as a list of
            # {name, expr, ok, err, ...} dicts (see tools.py). Counting at
            # consume-time keeps the UI insulated from payload-shape drift.
            probes = ev.data.get("probes") or []
            if isinstance(probes, list) and probes:
                passed = sum(1 for p in probes if isinstance(p, dict) and p.get("ok"))
                self._probes_passed = passed
                self._probes_total = len(probes)
            # Also drop the full report into the right-hand status panel.
            self._update_status(extra=f"[b]Last test:[/b]\n{text_safe}")

        elif ev.kind == "question":
            self._log(f"\n[bold magenta]?[/bold magenta] [bold]Model asks:[/bold] {text_safe}")
            self._awaiting_kind = "answer"
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "type your answer and press Enter"
            inp.focus()
            self._update_mode_bar()

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

        elif ev.kind == "restart":
            # Structured restart event (attempt comparison or winner).
            # Render same as info but the data payload also lands in
            # the trace .jsonl for offline filtering (`jq
            # 'select(.event=="restart")' games/traces/<stem>.jsonl`).
            self._log_info(text_safe)

        elif ev.kind == "mlx_stall":
            # Structured no-tokens-stall event. Render as an error in
            # the TUI; the .jsonl data carries stall_seconds + iter.
            self._log_error(text_safe)

        elif ev.kind == "await_user":
            # Step-mode pause (Stop-Losing-To-OneShot todo #1). Switch
            # the input box into "step" routing — empty Enter signals
            # the agent to continue; non-empty falls through to the
            # normal feedback path which also unblocks the wait.
            self._awaiting_kind = "step"
            self._log(f"\n[bold magenta]\u23f8[/bold magenta]  [bold]{text_safe}[/bold]")
            inp = self.query_one("#user-input", Input)
            inp.placeholder = "step-mode · Enter to continue, or type feedback"
            inp.focus()
            self._update_mode_bar()

        elif ev.kind == "activity":
            # ev.text is the state name: streaming | generating_assets |
            # browser | idle. ev.data may carry a human-readable "label".
            state = ev.text or ""
            label = (ev.data or {}).get("label", "")
            now = time.monotonic()
            if state == "idle":
                self._activity_label = ""
                self._activity_started_at = 0.0
                self._is_streaming = False
            else:
                self._activity_label = label or state.replace("_", " ")
                self._activity_started_at = now
                if state == "streaming":
                    # New stream begins; clear per-stream counters so
                    # tok/s reflects this stream only, not last one.
                    self._is_streaming = True
                    self._stream_tokens = 0
                    self._stream_started_at = now
                    self._last_token_at = 0.0
                else:
                    self._is_streaming = False
            self._update_status()

        elif ev.kind == "assets":
            self._assets_summary = self._format_assets_summary(ev.data or {})
            session_dir = (ev.data or {}).get("session_dir")
            if session_dir:
                try:
                    self._assets_dir = Path(session_dir)
                except Exception:
                    pass
            produced = (ev.data or {}).get("produced", 0)
            requested = (ev.data or {}).get("requested", 0)
            self._log_info(
                f"[green]assets:[/green] {produced}/{requested} generated"
                + (f" at [b]{session_dir}[/b]" if session_dir else "")
            )
            self._update_status()

        elif ev.kind == "sounds":
            # Parallel to the assets handler, but terser display — sound
            # names are what you reference in feedback ("make shoot less
            # harsh"); per-sound timing stays in .log / .jsonl.
            self._sounds_summary = self._format_sounds_summary(ev.data or {})
            session_dir = (ev.data or {}).get("session_dir")
            if session_dir:
                try:
                    self._sounds_dir = Path(session_dir)
                except Exception:
                    pass
            produced = (ev.data or {}).get("produced", 0)
            requested = (ev.data or {}).get("requested", 0)
            self._log_info(
                f"[green]sounds:[/green] {produced}/{requested} generated"
                + (f" at [b]{session_dir}[/b]" if session_dir else "")
            )
            self._update_status()

        elif ev.kind == "diagnose":
            # Sticky one-line preview of the model's most recent
            # diagnosis. Helps the user see what failure mode the agent
            # is currently chasing without scrolling the log scroll.
            txt = (ev.text or "").strip()
            if txt:
                # Collapse internal whitespace so a multi-line diagnosis
                # renders on one line in the status panel.
                preview = " ".join(txt.split())
                if len(preview) > 140:
                    preview = preview[:137] + "…"
                self._last_diagnose = preview
            self._update_status()

        elif ev.kind == "streak":
            self._streak_clean = int((ev.data or {}).get("consecutive_clean_iters", 0))
            self._streak_stuck = int((ev.data or {}).get("stuck_streak", 0))
            self._streak_min = int((ev.data or {}).get("min_to_ship", 2))
            self._update_status()

    def _format_sounds_summary(self, data: dict) -> str:
        """Render a compact sticky summary for a `sounds` event.

        Format mirrors the Assets block header but the list is just
        a comma-joined name list — no per-sound timing rows. Looping
        sounds get a `(loop)` suffix. Soft-wraps to a second line when
        the joined names exceed ~80 chars; caps at ~12 names with
        `(+N more)` to keep the panel scannable.

        Failures collapse to a single red line rather than per-row
        expansion (you can grep the `.log` for per-sound errors).
        """
        requested = data.get("requested", 0)
        produced = data.get("produced", 0)
        session_dir = data.get("session_dir") or ""
        paths = data.get("paths") or {}
        looping = set(data.get("looping") or [])
        per_sound = list(data.get("per_sound") or [])

        head = f"[b]Sounds:[/b] {produced}/{requested} generated"
        if session_dir:
            head += f" [dim]→ {session_dir}[/dim]"

        # Build name list from the produced paths (ordering matches the
        # agent's emission order). Cap at 12; rest as "(+N more)".
        names = list(paths.keys())
        max_show = 12
        overflow = max(0, len(names) - max_show)
        names = names[:max_show]
        labeled = [
            f"{n} (loop)" if n in looping else n
            for n in names
        ]
        # Soft-wrap: chunk so each line stays under ~80 chars when joined.
        rows: list[str] = []
        line: list[str] = []
        line_len = 0
        for lbl in labeled:
            add = len(lbl) + (2 if line else 0)  # ", " separator
            if line and line_len + add > 80:
                rows.append("  " + ", ".join(line))
                line = [lbl]
                line_len = len(lbl)
            else:
                line.append(lbl)
                line_len += add
        if line:
            rows.append("  " + ", ".join(line))
        if overflow:
            rows.append(f"  [dim](+{overflow} more)[/dim]")

        # Failure summary — collapsed one-liner.
        failed = [
            s for s in per_sound
            if isinstance(s, dict) and s.get("error")
        ]
        if failed:
            failed_names = ", ".join(str(s.get("name", "?"))[:24] for s in failed[:6])
            extra = f" [+{len(failed)-6} more]" if len(failed) > 6 else ""
            rows.append(f"  [red]{len(failed)} failed:[/red] {_esc(failed_names)}{extra}")

        return head + ("\n" + "\n".join(rows) if rows else "")

    def _format_assets_summary(self, data: dict) -> str:
        """Render the structured per-asset stats from an `assets` event
        into a few lines for the status panel. Truncates the per-asset
        list to 6 entries with a "+N more" suffix to keep the panel
        scannable even on big batches."""
        requested = data.get("requested", 0)
        produced = data.get("produced", 0)
        session_dir = data.get("session_dir") or ""
        per_asset = list(data.get("per_asset") or [])
        head = f"[b]Assets:[/b] {produced}/{requested} generated"
        if session_dir:
            head += f" -> [dim]{session_dir}[/dim]"
        rows: list[str] = []
        for stat in per_asset[:6]:
            if not isinstance(stat, dict):
                continue
            name = str(stat.get("name", "?"))[:32]
            cache = "cached" if stat.get("cache_hit") else "fresh "
            secs = stat.get("gen_seconds")
            secs_str = f"{secs:5.1f}s" if isinstance(secs, (int, float)) else "  -  "
            err = stat.get("error")
            if err:
                rows.append(f"  - {name:<20} [red]error[/red] {_esc(str(err)[:40])}")
                continue
            extra = ""
            alpha = stat.get("alpha_pixel_ratio")
            if isinstance(alpha, (int, float)):
                extra = f"  alpha={alpha:.2f}"
            rows.append(f"  - {name:<20} {cache} {secs_str}{extra}")
        if len(per_asset) > 6:
            rows.append(f"  [dim]+{len(per_asset) - 6} more[/dim]")
        if rows:
            return head + "\n" + "\n".join(rows)
        return head

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
    # Pre-load the Z-Image-Turbo pipeline NOW, before Textual mounts
    # and before Playwright/Chromium opens its IPC pipes. The
    # diffusers from_pretrained path forks a subprocess (via
    # huggingface_hub / safetensors); doing that fork after Playwright
    # is up makes _posixsubprocess.fork_exec reject the inherited fd
    # table with "bad value(s) in fds_to_keep" — which is why the
    # smoke test (clean process) succeeds and chat.py (Playwright
    # already running) fails for every asset. SKIP_DIFFUSER_PRELOAD=1
    # opts out for users who never want art and don't want to wait
    # ~15-30s on launch.
    if os.environ.get("SKIP_DIFFUSER_PRELOAD", "").strip() not in ("", "0", "false", "False"):
        pass
    else:
        try:
            import assets as _assets
            _assets.preload()
        except Exception:
            # preload() captures its own errors on the wrapper's
            # _last_error; a bare exception here would only fire if
            # the import itself failed, which we handle silently
            # (the agent will skip assets and the user can read the
            # real reason from the .log on their next session).
            pass
    try:
        CodingBoxApp().run()
    finally:
        _restore_terminal_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
