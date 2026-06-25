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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Load .env from the project root BEFORE any backend code reads
# os.environ. python-dotenv is import-light and the only side effect
# is populating os.environ for THIS process (it never writes back to
# disk or to the parent shell). The .env file is gitignored and chmod
# 600 — see .gitignore + scripts/setup.sh.
try:
    from dotenv import load_dotenv
    # override=True so a project-level .env wins over stale empty
    # shell vars (e.g. an unset-but-exported ANTHROPIC_API_KEY= line
    # in ~/.zshrc would otherwise block the load with override=False).
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    # python-dotenv not installed — chat.py still works for users who
    # export their keys directly in the shell. Print a quiet hint only
    # if a .env exists but couldn't be loaded.
    if (Path(__file__).resolve().parent / ".env").exists():
        print(
            "note: .env present but python-dotenv not installed; "
            "run `.venv/bin/pip install python-dotenv` or export keys "
            "manually in your shell.",
            file=sys.stderr,
        )

import ollama
def _esc(s: str) -> str:
    """Escape text so it is safe to embed inside Rich/Textual markup.

    Stricter than rich.markup.escape: that function only escapes balanced
    [...] pairs (its regex requires a literal closing ']'), so a truncated
    bracket like '[id*=hu' — which tools.format_report_for_model produces
    via `probe.expr[:80]` — slips through and crashes Textual's strict
    markup parser inside Static.update. Both Rich and Textual treat '\\['
    as a literal '['; only '[' needs escaping (']' is harmless on its own).
    """
    return s.replace("\\", "\\\\").replace("[", "\\[")
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.actions import SkipAction
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets._option_list import Option

import backend as backend_mod
from agent import (
    AgentEvent,
    DEFAULT_NUM_CTX,
    MAX_NUM_CTX,
    MIN_NUM_CTX,
    GameAgent,
    default_num_ctx,
    parse_num_ctx_arg,
)
from tools import LiveBrowser

# Shown in the Textual window header (top bar). Bump when verifying a fresh git pull.
CHAT_APP_VERSION = "1.1"
CHAT_APP_TITLE = f"Coding Agent v{CHAT_APP_VERSION}"

# xterm SGR uses 2 for right press; some stacks report 3.
_RIGHT_CLICK_BUTTONS = frozenset({2, 3})
_CONTEXT_LOG_TAIL_LINES = 200


class ContextMenuOverlay(Vertical):
    """Small floating menu (Cut/Copy/Paste or log/status copy actions)."""

    DEFAULT_CSS = """
    ContextMenuOverlay {
        layer: overlay;
        width: 36;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 0;
    }
    ContextMenuOverlay OptionList {
        width: 36;
        height: auto;
        max-height: 14;
        border: none;
        padding: 0 1;
    }
    """

    class Closed(Message):
        """Posted when the user picks an item or presses Escape."""

        def __init__(self, action_id: str | None) -> None:
            super().__init__()
            self.action_id = action_id

    def __init__(
        self,
        menu_items: list[tuple[str, str, bool]],
        *,
        screen_x: int,
        screen_y: int,
    ) -> None:
        """menu_items: (label, action_id, disabled)."""
        super().__init__(id="context-menu-overlay")
        self._menu_items = menu_items
        self._screen_x = screen_x
        self._screen_y = screen_y

    def compose(self) -> ComposeResult:
        options = [
            Option(label, id=action_id, disabled=disabled)
            for label, action_id, disabled in self._menu_items
        ]
        yield OptionList(*options, id="context-menu-list", compact=True)

    def on_mount(self) -> None:
        self.styles.offset = (self._screen_x, self._screen_y)
        try:
            self.query_one(OptionList).focus()
        except Exception:
            self.focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected,
    ) -> None:
        action_id = event.option_id or ""
        self.post_message(self.Closed(action_id))
        event.stop()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.post_message(self.Closed(None))
            event.stop()


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

    class RightClickRequest(Message):
        """Request the app to open the input context menu."""

        def __init__(self, *, screen_x: int, screen_y: int) -> None:
            super().__init__()
            self.screen_x = screen_x
            self.screen_y = screen_y

    def paste_flattened_text(self, text: str) -> bool:
        """Insert pasted text using the same newline->space policy as _on_paste."""
        flat = " ".join((text or "").split())
        if not flat:
            return False
        selection = self.selection
        if selection.is_empty:
            self.insert_text_at_cursor(flat)
        else:
            self.replace(flat, *selection)
        return True

    def _on_paste(self, event: events.Paste) -> None:  # type: ignore[override]
        self.paste_flattened_text(event.text or "")
        # prevent_default() is REQUIRED, not just event.stop().
        # Textual's MessagePump dispatches _on_paste to every class in
        # the MRO (textual/message_pump.py:_get_dispatch_methods walks
        # the full chain), so without prevent_default the parent
        # Input._on_paste ALSO runs after ours and inserts
        # `event.text.splitlines()[0]` at the current cursor — which
        # is right after the flat paste we just made. Net result:
        # full paste + duplicate of the first line tacked on.
        # event.stop() alone only blocks BUBBLING to ancestors; it
        # doesn't stop the MRO walk on the same widget.
        event.prevent_default()
        event.stop()

    def action_paste(self) -> None:
        """Paste from the OS/app clipboard, flattening any newlines to spaces."""
        self.paste_flattened_text(self.app.clipboard)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Forward input right-clicks even if Input consumes mouse down."""
        if event.button not in _RIGHT_CLICK_BUTTONS:
            return
        sx = getattr(event, "screen_x", event.x)
        sy = getattr(event, "screen_y", event.y)
        self.post_message(self.RightClickRequest(
            screen_x=int(sx),
            screen_y=int(sy),
        ))
        event.stop()


def _widget_has_id(widget: Widget | None, widget_id: str) -> bool:
    """True if `widget` is or is inside a node with the given id."""
    node: Widget | None = widget
    while node is not None:
        if getattr(node, "id", None) == widget_id:
            return True
        node = node.parent  # type: ignore[assignment]
    return False


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


def _resolve_seed_target(seed: Path) -> Path:
    """Map a seed path back to the canonical games/<basename>.html.

    Three shapes need to normalize:
      games/snake_x.html               → games/snake_x.html         (no-op)
      games/snake_x.best.html          → games/snake_x.html         (drop .best)
      games/snapshots/snake_x/iter_3.html → games/snake_x.html      (snapshot)

    The seed's HTML CONTENT is still what gets loaded into out_path
    by the agent's seed branch — we're only normalizing the PATH so
    out_path.stem (and therefore agent._session_id) ends up as the
    original game's basename. Every downstream artifact (assets
    folder, sounds folder, traces, snapshots, .best.html) then keys
    off that basename and reuses the original folders instead of
    spawning siblings.

    A seed shape we don't recognize (e.g. /seed ~/Downloads/foo.html)
    falls through unchanged — there's no canonical pair to map it
    onto, so the seed's own path is the basename source.
    """
    seed = Path(seed)
    parent = seed.parent
    stem = seed.stem

    # Case 1: games/snapshots/<basename>/iter_*.html — strip both the
    # snapshot subdir and the iter_* suffix. Only matches when the
    # immediate grandparent is literally named "snapshots".
    if parent.parent.name == "snapshots" and stem.startswith("iter_"):
        basename = parent.name
        return parent.parent.parent / f"{basename}.html"

    # Case 2: games/<basename>.best.html — drop the .best suffix.
    if stem.endswith(".best"):
        return parent / f"{stem[:-len('.best')]}.html"

    # Case 3: anything else — use as-is.
    return seed


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
    """Return (stall_seconds, overall_seconds) for the session.

    Deliberately ignores the model argument. The agent's standing
    rule (CLAUDE.md) is "we do NOT inspect the model name — a
    model-name table would go stale every release." Earlier
    versions of this function broke that rule with a four-bucket
    bracket table and an MoE-aware config.json parser; both were
    model-specific gunk that rotted as new models shipped.

    The right design is one timeout policy that works for every
    model. It works because:

      - The MLXBackend stall watchdog is **activity-aware**: every
        prefill-progress chunk and every emitted token bumps the
        timer. A model doing real work — even slowly — never trips
        the stall window.
      - The numbers below are no-activity quiet windows, NOT
        cold-start budgets.
      - 600 s of total silence (no progress chunks, no tokens) is
        a genuine hang. No realistic prompt produces a 10-minute
        quiet window between MLX prefill chunks; the watchdog is
        there to catch driver wedges and dead servers, not slow
        prefills.
      - 1800 s overall lets even an enormous output stream finish,
        and pathological runaway generation still gets capped.

    Argument is kept in the signature for back-compat with callers
    that pass it — but unused.
    """
    del model  # signature compat; not used by policy
    return 600.0, 1800.0


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

# Ollama load failures (bad GGUF blob, corrupt manifest). Used to auto-skip
# tags in resolve_chat_model and to surface /model escape hints in the TUI.
_OLLAMA_LOAD_FAIL_MARKERS: tuple[str, ...] = (
    "unable to load model",
    "status code: 500",
    "status code: 502",
)


def _is_ollama_load_failure(err_text: str) -> bool:
    low = (err_text or "").lower()
    return any(m in low for m in _OLLAMA_LOAD_FAIL_MARKERS)


def mark_broken_ollama_tag(tag: str, *, reason: str = "") -> None:
    """Session-scoped blacklist: skip on auto-detect; still forceable via /model."""
    name = (tag or "").strip()
    if not name:
        return
    if name in _KNOWN_BROKEN_TAGS:
        return
    _KNOWN_BROKEN_TAGS.add(name)

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
        chat_running = [
            m for m in running
            if _is_chat_capable_tag(m["name"])
            and m["name"] not in _KNOWN_BROKEN_TAGS
        ]
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
        height: auto;
    }

    #status-scroll {
        height: 1fr;
        scrollbar-gutter: stable;
    }

    #status-body {
        height: auto;
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

    #footer-row {
        dock: bottom;
        height: 1;
        layout: horizontal;
    }

    /* Inside the row, override Footer's own dock so it lays out next
       to the badge instead of detaching to the screen bottom. */
    #footer-row Footer {
        dock: initial;
        width: 1fr;
    }

    #footer-badge {
        width: auto;
        height: 1;
        padding: 0 1;
        background: $warning;
        color: black;
        text-style: bold;
        display: none;
    }

    #footer-badge.-on {
        display: block;
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
        # priority=True is REQUIRED — Textual's Input widget has its own
        # `Binding('delete,ctrl+d', 'delete_right')` that captures Ctrl+D
        # whenever focus is on the input box (which is ~all of the time
        # in a chat-style TUI). Without `priority=True`, Ctrl+D silently
        # deletes a character from the input instead of shipping the
        # build. User-visible symptom before this fix: "Ctrl+D never
        # works." Ctrl+Q escapes the same way (Input has no Ctrl+Q
        # binding) so we keep that one normal.
        Binding("ctrl+d", "ship_it", "Ship game / done", priority=True),
        Binding("ctrl+q", "quit_app", "Quit"),
        # Ctrl+L: re-print where the FULL log files live. Useful when you
        # want to `cat` them from another terminal to share with an LLM.
        Binding("ctrl+l", "show_log_paths", "Show log paths"),
        # Ctrl+S: toggle "selection mode" — releases Textual's mouse
        # capture so the terminal handles drag-select natively. Press
        # again to resume normal TUI mouse handling. Without this, the
        # left log pane is unselectable while the agent is running.
        Binding("ctrl+s", "toggle_selection_mode", "Select text"),
        # Route keyboard paste through the same OS-aware path as the
        # input context menu, so external clipboard content works.
        Binding("ctrl+v", "paste_input", "Paste into input", show=False),
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
        # Plain lines mirrored from the log pane for right-click copy.
        self._log_mirror_lines: list[str] = []
        self._status_plain: str = ""
        # Optional manual status body (e.g. /help). When set, it is shown
        # in the status panel until the next user input.
        self._status_manual_body: str | None = None
        self._context_menu: ContextMenuOverlay | None = None
        self._context_menu_origin: str = ""
        # Per-session paths assigned in _start_session. None until then.
        self._out_path: Path | None = None
        self._best_path: Path | None = None
        self._assets_dir: Path | None = None
        # Status-panel state — kept here so _update_status() can render
        # without re-derived state. Reset in _new_session via _reset_status_state.
        self._activity_label: str = ""        # what's happening right now
        self._activity_role: str = "coder"    # coder | critic | architect
        self._activity_started_at: float = 0.0  # monotonic; for "Ns" age
        self._stream_tokens: int = 0          # tokens this stream
        self._stream_started_at: float = 0.0  # monotonic; for tok/s
        self._last_token_at: float = 0.0      # monotonic; for stall age
        self._is_streaming: bool = False
        # Stream-visibility bookkeeping (2026-06-12, trace 20260612_132314):
        # a coder stream ran 24+ min at ~12 tok/s with NOTHING printed to
        # the console for 9+ min (long reasoning, no newline) — the user
        # couldn't tell a healthy generation from a hang. These drive the
        # [stream alive] console line and the one-shot runaway mirror.
        # Display-only; nothing here aborts or truncates a stream.
        self._last_console_flush_at: float = 0.0   # last stream line printed
        self._last_stream_alive_note_at: float = 0.0
        self._runaway_console_warned: bool = False
        # Model 2 / Model 3 sidecar stream stats (same shape as coder Activity).
        self._model2_stream_tokens: int = 0
        self._model2_stream_started_at: float = 0.0
        self._model2_last_token_at: float = 0.0
        self._model2_is_streaming: bool = False
        self._model3_stream_tokens: int = 0
        self._model3_stream_started_at: float = 0.0
        self._model3_last_token_at: float = 0.0
        self._model3_is_streaming: bool = False
        self._assets_summary: str = ""        # sticky summary of last batch
        # Phase 1C — in-flight totals so the status panel can render
        # live "Sprites: 4/12 · 2.9s avg · ~24s ETA" rows. Set when
        # the agent emits `activity:generating_assets`, cleared when
        # the `assets` completion event fires.
        self._assets_in_flight_total: int = 0
        self._sounds_in_flight_total: int = 0
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
        # Sticky one-line note when the LAST stream was aborted by a guard
        # (repetition / deliberation loop). Without this the reason only
        # appears as a single scrolled-past log line (minecraft 20260621
        # trace: a 7-min stream was killed and the status just flipped to
        # "idle" with no hint why). Cleared once a clean iteration flows.
        self._last_stall_reason: str | None = None
        # Context-window display. `_ctx_max` is read once at session
        # start from BackendInfo.context_length (Ollama) or the MLX
        # config.json. `_ctx_fill_chars` is recomputed each
        # _update_status() tick by summing message lengths on the agent.
        self._ctx_max: int | None = None
        self._streak_clean: int = 0
        self._streak_min: int = 2
        self._streak_stuck: int = 0
        # Sticky test report. Set by the "test" event handler via
        # _update_status(extra=...); persists across subsequent status
        # ticks so it doesn't flash and disappear on the next 1Hz refresh.
        # Replaced on each new test, cleared on session reset.
        self._last_test_block: str = ""
        # Feedback ledger (2026-06-11 FPS trace): a session-scoped record of
        # every user feedback item with status queued → applying → applied.
        # The old "Queued (N)" panel section only shows items still sitting
        # in agent._pending_feedback — most feedback is consumed within
        # milliseconds (idle boundary / extension restart), so the user
        # never saw any acknowledgment. The ledger persists for the whole
        # session and is rendered as a "Feedback:" section in the status
        # panel. Entries: {text, status, iter, ok}.
        self._feedback_ledger: list[dict] = []
        self._last_gpu_summary_plain: str = ""
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
        # Secondary and tertiary model slots with configurable roles
        self._next_model2: str | None = None
        self._next_backend2: str | None = None
        self._next_role2: str | None = None
        self._next_model3: str | None = None
        self._next_backend3: str | None = None
        self._next_role3: str | None = None
        # Snapshot of the last /list output: list of (backend_name, model_id)
        # pairs in display order, so /load N or /model N can pick by number
        # across BOTH backends. Empty until the user runs /list once; the
        # /load handler refreshes it on demand if empty.
        self._last_listing: list[tuple[str, str]] = []
        # Resolved Backend + BackendInfo for the running session.
        # Constructed in _start_session.
        self._session_backend = None
        self._session_backend_info: backend_mod.BackendInfo | None = None
        self._session_backend2 = None
        self._session_backend_info2: backend_mod.BackendInfo | None = None
        self._session_model2: str | None = None
        self._session_role2: str | None = None
        self._session_backend3 = None
        self._session_backend_info3: backend_mod.BackendInfo | None = None
        self._session_model3: str | None = None
        self._session_role3: str | None = None
        self._ollama_placement_status: str = ""
        # /iters lets the user change the max-iters cap before starting a
        # session or extending. Default matches GameAgent's default.
        self._max_iters: int = 6
        # /ctx sets Ollama num_ctx (KV reservation). Sticky across /new.
        self._num_ctx: int = default_num_ctx()
        # /seed stages an existing HTML file as the baseline for the next
        # /new session. Cleared once consumed.
        self._next_seed: Path | None = None
        # Seed file latched into the CURRENT session at /new time. This is
        # separate from _next_seed so status can still show "seed in use"
        # even if the staged seed gets changed/cleared mid-session.
        self._session_seed: Path | None = None
        # /ref can be staged before a session exists. On session start we
        # hand these bytes to GameAgent._next_image_bytes so the FIRST user
        # turn (planning/build) gets the reference image.
        self._staged_ref_image_bytes: bytes | None = None
        self._staged_ref_image_name: str | None = None
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
        # Lean system-prompt override. None = auto (the agent renders the
        # compact `small` schema for LOCAL backends so a local VLM like
        # qwen3.6:27b isn't buried under a 20KB schema + 6KB project doc).
        # /leanprompt on|off|auto sets this explicitly.
        self._lean_prompt: bool | None = None
        # Run-profile contract shown in status/mode bar. Default to
        # local_manual so new sessions start in wait mode: the user can
        # inspect each iter before the next model call. Override with
        # `/mode local_auto` or `/wait off` for unattended runs.
        self._run_profile: str = "local_manual"
        # Optional reviewer model used by the local_plus_review profile.
        # Explicitly user-configured via /mode local_plus_review with <model>.
        self._profile_review_model: str | None = None
        # When true, local_plus_review in AUTO mode can run `/check <model>`
        # and auto-queue its guidance after failed tests. OFF by default to avoid surprise
        # paid API calls.
        self._profile_review_auto_apply: bool = False
        # Guard against overlapping auto-review workers.
        self._auto_review_running: bool = False
        # Optional expanded per-iter diagnostics line. OFF by default so
        # normal runs stay concise; toggle via /iter-detail on|off.
        self._iter_decision_verbose: bool = False
        # Advanced agent behavior bundles (toggled dynamically via slash commands)
        self._use_prefill: bool = True
        self._use_vlm_critique: bool = False
        # Phase 1.5 — autonomous self-feedback loop. After each clean
        # iter, runs a short scripted playtest, captures multi-window
        # screenshots, evaluates genre-free behavior recipes, and queues
        # user-style feedback into _pending_feedback. Default ON; one
        # slash command (/playtest off, alias /feedback off) disables if it ever causes
        # regressions. Existing test reports + visual critic + patch
        # diagnostics are NOT gated by this — only the extra autonomous
        # direction is.
        self._use_autonomous_feedback: bool = True
        # /rawfeedback toggle — DEFAULT FALSE (raw mode is ON).
        # When False (default): the model sees your typed feedback under
        # only the basic USER FEEDBACK (HIGHEST PRIORITY) wrapper and
        # decides for itself whether to patch or re-render. Machine bug
        # feedback (Chromium test reports, console errors, frozen-canvas
        # detector, probes), playbook retrieval, and autonomous playtest
        # are UNAFFECTED — they all still run.
        # When True: the harness wraps your typed text with classifier-
        # driven directives (MEDIA-CHANGE, ORIENTATION-CHANGE, SCOPE
        # ARBITRATION, asset stem mapping). Opt in with /rawfeedback off
        # when the classifier reads your phrasing correctly.
        # Default flipped 2026-05-23 — the wrapping classifier had over-
        # ridden user guidance in the doom session ("down key moves you
        # forward" got tagged as ART/SOUND feedback). Per the user's
        # standing rule: the agent must beat zero-shot, not lose value
        # to regex misroutes. Sticky across /new.
        self._use_feedback_directives: bool = False
        # Remembered vlm-critique state captured when /wait turns ON.
        # Rationale (2026-05-24 mortal kombat trace): when the user is
        # actively reviewing each iter and typing their own feedback,
        # the auto visual critic emits ~400 chars of paraphrased
        # observations the user already saw — pure prompt noise. On
        # /wait on we save the current state and force vlm-critique
        # off; on /wait off we restore it. Explicit /vlm-critique
        # while in wait mode clears this so the auto-restore doesn't
        # override the user's deliberate choice.
        self._vlm_critique_pre_wait: bool | None = None
        # Auto-staff flags (2026-05-21): True when the matching feature
        # was flipped by auto-enable (sidecar role or local-VLM detect)
        # rather than by an explicit toggle. Surfaced in the status panel
        # so the user sees "[auto]" and knows they can override.
        self._vlm_critique_auto: bool = False
        self._use_double_screenshot: bool = False
        self._use_architect_split: bool = False
        self._architect_split_auto: bool = False
        # Bundle toggle: /allroles flips architect-split + vlm-critique together
        # so a single loaded LLM covers coder + critic + architect roles
        # without needing /model2 / /model3 staging or extra GPUs.
        self._all_roles_enabled: bool = False

    # ----------------------------- layout ---------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            with Horizontal():
                yield RichLog(id="log-pane", wrap=True, markup=True, highlight=False)
                with Vertical(id="status-pane"):
                    yield Static("Status", id="status-title")
                    with VerticalScroll(id="status-scroll"):
                        yield Static("", id="status-body")
        # Input row docked to the bottom. We start it empty with a goal prompt.
        yield MultilinePasteInput(
            placeholder="What game do you want to build?",
            id="user-input",
            select_on_focus=False,
        )
        # Single-row mode indicator just above the Footer. Tells the
        # user at a glance whether the agent is RUNNING or WAITING for
        # them. Sits in the same visual band as the binding hints so
        # both are scannable without moving the eye.
        yield Static("", id="mode-bar")
        # Bottom row: leading WAIT-mode badge + the standard Footer
        # (key bindings). The badge is hidden by default and toggled on
        # by `_update_mode_bar`. Putting it on the SAME row as Footer
        # is intentional — when step-mode is on, the user needs that
        # cue right where they look for the control commands.
        with Horizontal(id="footer-row"):
            yield Static("", id="footer-badge")
            yield Footer()

    async def on_mount(self) -> None:
        self.title = CHAT_APP_TITLE
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
            "[dim]Keys: Ctrl+D ship · Ctrl+L show log file paths · Ctrl+S select "
            "in log · Ctrl+Q quit · if the shell stops echoing after exit, run "
            "`reset`.[/dim]"
        )
        self._log_info(
            "[dim]Slash commands available — type [b]/help[/b] for the full list "
            "(/list, /model, /new, /open, /clear, /iters, /status, /ship, /quit).[/dim]"
        )
        self._log_info(
            "[dim]Right-click the input for Cut/Copy/Paste; right-click the log "
            "or status panel to copy text or enable drag-select (same as "
            "Ctrl+S). Paste in input: Ctrl+V (or terminal Edit→Paste / Cmd+V on macOS).[/dim]"
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
        plain = self._MARKUP_RE.sub("", text).rstrip()
        if plain:
            self._append_log_mirror_line(plain)
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.write(plain + "\n")
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
        line = text.rstrip("\n")
        if line:
            self._append_log_mirror_line(line)
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.write(line + "\n")
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

    # Mega-line flush threshold (2026-06-12, trace 20260612_132314): the
    # console only flushed on newline, so one enormous line (a long
    # reasoning paragraph) stayed invisible for minutes while the stream
    # was healthy. Past this many buffered chars we flush a partial line.
    # High enough that normal code lines are unaffected. Display-only.
    _STREAM_PARTIAL_FLUSH_CHARS = 2000
    # Mirror of agent.py's _RUNAWAY_TOKEN_FLOOR (15K) — the agent's
    # runaway_stream_warning is trace-only; the console needs its own.
    _RUNAWAY_CONSOLE_FLOOR = 15000
    # [stream alive] line: fires after this much token-flow with no
    # printable output, repeats at most every _STREAM_ALIVE_REPEAT_S.
    _STREAM_ALIVE_SILENCE_S = 90.0
    _STREAM_ALIVE_REPEAT_S = 120.0

    def _activity_header(self, role: str) -> str:
        return f"[bold yellow]Activity ({role}):[/bold yellow]"

    def _role_slot_for_stream(self, role: str | None) -> int | None:
        """Map agent stream role to Model 2 or 3 slot; None = Model 1 (coder)."""
        if not role or role == "coder":
            return None
        if self._session_backend2 and (self._session_role2 or "") == role:
            return 2
        if self._session_backend3 and (self._session_role3 or "") == role:
            return 3
        return None

    def _emit_token(self, piece: str) -> None:
        # Called from inside the agent's async loop (same event loop as the
        # TUI), so this is safe without explicit thread-marshaling.
        self._stream_buf += piece
        now = time.monotonic()
        role = (
            getattr(self.agent, "_last_stream_role", None) if self.agent else None
        ) or "coder"
        slot = self._role_slot_for_stream(role)
        if slot == 2:
            self._model2_stream_tokens += 1
            self._model2_last_token_at = now
            if self._model2_stream_started_at == 0.0:
                self._model2_stream_started_at = now
        elif slot == 3:
            self._model3_stream_tokens += 1
            self._model3_last_token_at = now
            if self._model3_stream_started_at == 0.0:
                self._model3_stream_started_at = now
        else:
            self._stream_tokens += 1
            self._last_token_at = now
            if self._stream_started_at == 0.0:
                self._stream_started_at = now
            # One-shot console mirror of the agent's trace-only
            # runaway_stream_warning (which never reaches the TUI —
            # _record without yield is trace-only).
            if (
                not self._runaway_console_warned
                and self._stream_tokens >= self._RUNAWAY_CONSOLE_FLOOR
            ):
                self._runaway_console_warned = True
                self._log_info(
                    f"[yellow]long stream: {self._stream_tokens:,} completion "
                    "tokens — typical iter is 1-4K. Not aborting (no-cutoff "
                    "rule); watch the [stream alive] line below, or /done "
                    "ships the last clean build.[/yellow]"
                )
        # Flush at any newline boundary so the user sees lines as they arrive.
        if "\n" in self._stream_buf:
            *complete, self._stream_buf = self._stream_buf.split("\n")
            for line in complete:
                if line.strip():
                    # Raw: model output contains JS bracket indexing that
                    # would otherwise be eaten by Rich's markup parser.
                    self._log_raw(line)
            self._last_console_flush_at = now
        elif len(self._stream_buf) >= self._STREAM_PARTIAL_FLUSH_CHARS:
            # Mega-line flush: one enormous line must not look like a
            # hang — push what we have as a partial line.
            self._log_raw(self._stream_buf)
            self._stream_buf = ""
            self._last_console_flush_at = now

    def _flush_stream(self) -> None:
        """Push any remaining buffered tokens (no trailing newline)."""
        if self._stream_buf.strip():
            self._log_raw(self._stream_buf)
        self._stream_buf = ""

    def _update_status(self, extra: str | None = None) -> None:
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
          5. Last test — full report from the most recent test event.
             Sticky: callers pass `extra=` once when a new test fires;
             subsequent status ticks (1Hz timer, other events) reuse the
             stored block instead of wiping it. New test replaces it;
             session reset clears it.
        """
        if self._status_manual_body is not None:
            # Always prepend live activity + mode so the user can still
            # see what the agent is doing while /help is on screen. The
            # pre-fix behavior replaced the entire panel with help text
            # — a session in flight became invisible and the user
            # couldn't tell whether to wait, ship, or quit.
            live_prefix = self._render_activity_line() + self._render_mode_row()
            body = live_prefix + "\n" + self._status_manual_body
            if extra is not None:
                self._last_test_block = extra
            self._status_plain = self._MARKUP_RE.sub("", body)
            self.query_one("#status-body", Static).update(body)
            self._update_mode_bar()
            return
        if extra is not None:
            self._last_test_block = extra
        body = self._render_activity_line()
        body += self._render_mode_row()
        body += self._render_iteration_block()
        body += self._render_gpu_placement_block()
        body += self._render_assets_block()
        body += self._render_sounds_block()
        body += self._render_memory_block()
        body += self._render_playbook_block()
        body += self._render_files_block()
        if self._last_test_block:
            body += "\n" + self._last_test_block
        self._status_plain = self._MARKUP_RE.sub("", body)
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
                    "model2_streaming": bool(self._model2_is_streaming),
                    "model3_streaming": bool(self._model3_is_streaming),
                    "model2_tokens": self._model2_stream_tokens,
                    "model3_tokens": self._model3_stream_tokens,
                    "phase": self._phase_label,
                    "iteration": self._iteration_label,
                    "streak_clean": int(self._streak_clean or 0),
                    "streak_stuck": int(self._streak_stuck or 0),
                    "probes_passed": self._probes_passed,
                    "probes_total": self._probes_total,
                    "last_diagnose": self._last_diagnose,
                    "last_stall_reason": self._last_stall_reason,
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
                    "gpu_summary": getattr(self, "_last_gpu_summary_plain", None) or None,
                    # Queue observability (2026-06-11 FPS trace): the trace
                    # couldn't answer "was my feedback queued/visible?" —
                    # record the pending count + last ledger status.
                    "pending_feedback": len(
                        getattr(agent, "_pending_feedback", []) or []
                    ),
                    "ledger_tail": (
                        self._feedback_ledger[-1]["status"]
                        if self._feedback_ledger else None
                    ),
                })
            except Exception:
                pass

    def _format_stream_activity_detail(
        self,
        *,
        label: str,
        tokens: int,
        stream_started_at: float,
        last_token_at: float,
        is_streaming: bool,
    ) -> str:
        """tok/s line matching the main Activity row (for any model slot)."""
        if not is_streaming:
            return label
        elapsed = max(0.001, time.monotonic() - stream_started_at) if stream_started_at else 0.001
        tok_per_s = tokens / elapsed if elapsed > 0 else 0.0
        since_last = time.monotonic() - last_token_at if last_token_at else 0.0
        stalled = tokens > 0 and since_last > 30.0
        if tokens == 0:
            wait = time.monotonic() - stream_started_at if stream_started_at else 0.0
            progress_total = getattr(self.agent, "_stream_progress_total", 0) or 0
            progress_current = getattr(self.agent, "_stream_progress_current", 0) or 0
            progress_stage = getattr(self.agent, "_stream_progress_stage", None)
            if progress_stage == "mlx_load":
                return (
                    f"{label} — [cyan]loading MLX weights[/cyan] "
                    f"[dim]({wait:.0f}s)[/dim]"
                )
            if progress_total > 0:
                pct = (100.0 * progress_current / progress_total) if progress_total else 0.0
                return (
                    f"{label} — [cyan]prompt eval {progress_current:,}/"
                    f"{progress_total:,} ({pct:.0f}%)[/cyan] [dim]{wait:.0f}s[/dim]"
                )
            if wait > 30.0:
                return f"{label} — [red]waiting {wait:.0f}s for first token[/red]"
            return f"{label} — [dim]waiting for first token ({wait:.0f}s)[/dim]"
        tag = "[red]STALLED[/red]" if stalled else "[green]live[/green]"
        return (
            f"{label} — {tokens:,} tok, {tok_per_s:.1f} tok/s, "
            f"last {since_last:.1f}s ago {tag}"
        )

    def _slot_stream_state(
        self,
        slot: int | None,
    ) -> tuple[bool, int, float, float, str]:
        """(is_streaming, tokens, started_at, last_token_at, idle_label) per slot."""
        if slot == 2:
            return (
                self._model2_is_streaming,
                self._model2_stream_tokens,
                self._model2_stream_started_at,
                self._model2_last_token_at,
                (
                    getattr(self.agent, "_model2_activity", None)
                    if self.agent else None
                ) or "idle",
            )
        if slot == 3:
            return (
                self._model3_is_streaming,
                self._model3_stream_tokens,
                self._model3_stream_started_at,
                self._model3_last_token_at,
                (
                    getattr(self.agent, "_model3_activity", None)
                    if self.agent else None
                ) or "idle",
            )
        return (
            self._is_streaming,
            self._stream_tokens,
            self._stream_started_at,
            self._last_token_at,
            self._activity_label or "idle",
        )

    def _render_role_activity_line(
        self,
        role: str,
        *,
        slot: int | None = None,
        color: str = "cyan",
        model_name: str | None = None,
    ) -> str:
        """One Activity row per role — same tok/tok/s display as the coder line."""
        streaming, tokens, started, last_tok, idle_tag = self._slot_stream_state(slot)
        hdr = self._activity_header(role)
        if streaming:
            if slot is None:
                label = self._activity_label or f"streaming {role}"
            elif (self._activity_role or "") == role:
                label = self._activity_label or idle_tag
            else:
                label = idle_tag if idle_tag not in ("idle", role) else f"streaming {role}"
            if label in ("idle", role):
                label = f"streaming {role}"
            body = self._format_stream_activity_detail(
                label=label,
                tokens=tokens,
                stream_started_at=started,
                last_token_at=last_tok,
                is_streaming=True,
            )
        elif idle_tag and idle_tag != "idle":
            age = (
                time.monotonic() - self._activity_started_at
                if self._activity_started_at and slot is None
                else 0.0
            )
            if idle_tag in ("crashed",) or str(idle_tag).startswith("failed"):
                body = f"[red]{_esc(idle_tag)}[/red]"
            else:
                suffix = f" [dim]({age:.0f}s)[/dim]" if age > 0 else ""
                body = f"{_esc(idle_tag)}{suffix}"
        else:
            body = "[dim]idle[/dim]"
        model_tag = f" · {_esc(model_name)}" if model_name else ""
        return f"{hdr}{model_tag} {body}\n"

    def _render_activity_line(self) -> str:
        """Per-role activity rows: coder + model2 + model3 when configured."""
        lines: list[str] = []
        coder_model = None
        if self.agent is not None:
            coder_model = getattr(self.agent, "model", None)
        elif self._session_model:
            coder_model = self._session_model
        # In /allroles mode (or any time no dedicated slot 2/3 is staged
        # for the active role), architect / critic streams are multiplexed
        # onto the coder backend. Swap the slot-1 header to show the
        # currently-active role so the user can SEE the architect drive
        # Phase A planning and the critic fire after a clean iter — the
        # row used to be hardcoded "coder" and made those roles invisible.
        slot1_role = "coder"
        active = self._activity_role or "coder"
        if active != "coder" and self._role_slot_for_stream(active) is None:
            slot1_role = active
        lines.append(self._render_role_activity_line(
            slot1_role, slot=None, color="cyan", model_name=coder_model,
        ))
        if self._session_backend2 or (
            self.agent is not None and getattr(self.agent, "_backend2", None)
        ):
            lines.append(self._render_role_activity_line(
                self._session_role2 or "critic",
                slot=2,
                color="green",
                model_name=self._session_model2,
            ))
        if self._session_backend3 or (
            self.agent is not None and getattr(self.agent, "_backend3", None)
        ):
            lines.append(self._render_role_activity_line(
                self._session_role3 or "architect",
                slot=3,
                color="magenta",
                model_name=self._session_model3,
            ))
        # Phase 0.12 — queued-feedback banner right under the activity
        # rows so users can see "my feedback IS pending" during a long
        # stream. The 2026-05-22 trace had 25 minutes of silent waiting
        # while a stream finished and the queued feedback was invisible
        # at the top of the panel — the existing `Queued (N):` section
        # only appears further down. This banner is high-visibility
        # (bold yellow) and shows when ANY input is queued for the next
        # user-turn boundary.
        if self.agent is not None:
            pending_count = len(getattr(self.agent, "_pending_feedback", []) or [])
            pending_ans = getattr(self.agent, "_pending_answer", None)
            if pending_ans or pending_count:
                bits: list[str] = []
                if pending_ans:
                    bits.append("1 answer")
                if pending_count:
                    bits.append(
                        f"{pending_count} feedback item{'s' if pending_count != 1 else ''}"
                    )
                summary = " + ".join(bits)
                lines.append(
                    f"[bold yellow]⚠ Queued for next user-turn:[/bold yellow] "
                    f"{summary} [dim](inject at iter boundary — current stream "
                    f"finishes first)[/dim]\n"
                )
        return "".join(lines)

    def _render_mode_row(self) -> str:
        """Render the colored Mode row in the right-hand status panel.

        Item 5 (request 2026-05-15): the user wants explicit visual
        feedback for the `/wait` step-mode state. Previously the only
        signal was a black-on-yellow "WAIT MODE" badge in the bottom
        mode-bar — visible only when step-mode was ON, invisible when
        off (the user had to remember which state they last toggled).
        The status panel now always renders a colored Mode line:

          - step-mode ON:  black-on-yellow "WAIT" badge — pause-after-iter
          - step-mode OFF: green "AUTO" — continuous run

        VLM models also get a small badge here when the active session's
        model is image-capable (the agent's runtime _detect_vlm latched
        positive). Lets the user know at a glance whether the model can
        consume screenshots for visual debugging or is text-only.
        """
        step_on = self._effective_step_mode()
        if step_on:
            mode_badge = (
                "[black on yellow] WAIT [/]  "
                "[yellow]pause after each iter — Enter or feedback to continue[/yellow]"
            )
        else:
            mode_badge = (
                "[black on green] AUTO [/]  "
                "[green dim]continuous run — /wait on to pause per-iter[/green dim]"
            )
        # VLM hint — agent's runtime probe sets self.agent._is_vlm to
        # True/False after the first stream call. None = unknown (not
        # probed yet). When True, the model can read screenshots
        # (which the VLM-critique path attaches to feedback turns when
        # /vlm is enabled). Mention it so the user knows their model
        # can be helped by visual feedback.
        is_vlm = getattr(self.agent, "_is_vlm", None)
        if is_vlm is True:
            vlm_hint = "  [magenta]\\[VLM][/magenta]"
        elif is_vlm is False:
            vlm_hint = "  [dim]\\[text-only][/dim]"
        else:
            vlm_hint = ""  # unprobed — stay silent
        profile = self._format_run_profile()
        review_hint = ""
        if self._run_profile == "local_plus_review" and self._profile_review_model:
            apply_hint = "auto-apply" if self._profile_review_auto_apply else "manual-apply"
            review_hint = (
                "  "
                f"[dim]review: {_esc(self._profile_review_model)} "
                f"({apply_hint})[/dim]"
            )
        # /allroles indicator — when architect-split AND vlm-critique are
        # both enabled (the /allroles bundle), tell the user the multiplex
        # is wired so they know the architect + critic ARE running, even
        # though everything routes through the coder backend. Prefer the
        # live agent's feature flags when a session is running; fall back
        # to the chat-level toggle otherwise.
        if self.agent is not None:
            arch_on = bool(getattr(self.agent, "_use_architect_split", False))
            crit_on = bool(getattr(self.agent, "_use_vlm_critique", False))
        else:
            arch_on = self._use_architect_split
            crit_on = self._use_vlm_critique
        roles_line = ""
        slot2_active = bool(self._session_backend2)
        slot3_active = bool(self._session_backend3)
        if arch_on and crit_on:
            if slot2_active or slot3_active:
                roles_line = (
                    "\n[bold]Roles:[/bold] "
                    "[green]/allroles ON[/green] "
                    "[dim]— architect + critic enabled (using staged slot 2/3 where assigned, "
                    "else multiplexed onto the coder backend)[/dim]"
                )
            else:
                roles_line = (
                    "\n[bold]Roles:[/bold] "
                    "[green]/allroles ON[/green] "
                    "[dim]— coder + architect + critic all on the one loaded LLM "
                    "(slot-1 Activity header reflects which role is streaming)[/dim]"
                )
        elif arch_on:
            roles_line = (
                "\n[bold]Roles:[/bold] [green]architect-split ON[/green] "
                "[dim]— architect drives planning (critic off)[/dim]"
            )
        elif crit_on:
            roles_line = (
                "\n[bold]Roles:[/bold] [green]vlm-critique ON[/green] "
                "[dim]— critic reviews each clean iter (architect-split off)[/dim]"
            )
        return (
            f"[bold]Mode:[/bold] {mode_badge}{vlm_hint}\n"
            f"[bold]Profile:[/bold] {profile}{review_hint}"
            f"{roles_line}\n"
        )

    def _format_run_profile(self) -> str:
        """Human-readable label for the active run profile."""
        labels = {
            "custom": "custom",
            "local_manual": "local_manual",
            "local_auto": "local_auto",
            "local_plus_review": "local_plus_review",
        }
        return labels.get(self._run_profile, self._run_profile)

    def _effective_step_mode(self) -> bool:
        """UI truth for WAIT mode before and after a GameAgent exists."""
        if self.agent is not None:
            return bool(getattr(self.agent, "_step_mode", False))
        return self._run_profile == "local_manual"

    def _wants_three_ollama_slots(self) -> bool:
        """True only when the user explicitly staged slots 2 AND 3.

        Default-on autopin for 4-GPU boxes was reverted on 2026-05-23
        after it crashed the workstation: three Ollama daemons plus the
        iter-1 best-of-N fan-out hammered every GPU at once. Multi-slot
        is now strictly opt-in via /model2 /model3 /modelall.
        """
        primary_backend = (self._next_backend or "auto").lower()
        if primary_backend in ("mlx", "openai", "anthropic", "claude"):
            return False
        for backend_name, model_name in (
            (self._next_backend2, self._next_model2),
            (self._next_backend3, self._next_model3),
        ):
            if not model_name:
                return False
            if (backend_name or "ollama").lower() != "ollama":
                return False
        return True

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
        if self._session_backend_info2 is not None:
            out += f"[b]Model 2 Backend:[/b] {self._session_backend_info2.name.upper()} [dim]({self._session_role2})[/dim]\n"
        if self._session_model2:
            out += f"[b]Model 2:[/b] {self._session_model2}\n"
        if self._session_backend_info3 is not None:
            out += f"[b]Model 3 Backend:[/b] {self._session_backend_info3.name.upper()} [dim]({self._session_role3})[/dim]\n"
        if self._session_model3:
            out += f"[b]Model 3:[/b] {self._session_model3}\n"
        out += f"[b]Run profile:[/b] {self._format_run_profile()}\n"
        if self._ollama_placement_status:
            out += f"[b]Ollama placement:[/b] {_esc(self._ollama_placement_status)}\n"
        if self._run_profile == "local_plus_review" and self._profile_review_model:
            apply_hint = "auto-apply in AUTO mode" if self._profile_review_auto_apply else "manual apply"
            out += (
                f"[b]Review hook:[/b] {_esc(self._profile_review_model)} "
                f"[dim]({apply_hint})[/dim]\n"
            )
        # Surface the staged-for-next-/new pair when it differs from the
        # running session. /load and /backend only stage — the current
        # session keeps its model — but the status panel used to be
        # silent about staging, so users would /load claude, see
        # "Backend: MLX", and think the swap had been lost.
        cur_backend = (
            self._session_backend_info.name if self._session_backend_info else None
        )
        staged_backend = self._next_backend
        staged_model = self._next_model
        differs_b = staged_backend is not None and staged_backend != cur_backend
        differs_m = staged_model is not None and staged_model != self._session_model
        if differs_b or differs_m:
            label_b = (staged_backend or cur_backend or "auto").upper()
            label_m = staged_model or self._session_model or "—"
            out += (
                f"[b]Staged for /new:[/b] [yellow]{label_b}[/yellow] · "
                f"{_esc(label_m)}\n"
            )

        # Stage model2
        cur_backend2 = (
            self._session_backend_info2.name if self._session_backend_info2 else None
        )
        staged_backend2 = self._next_backend2
        staged_model2 = self._next_model2
        differs_b2 = staged_backend2 is not None and staged_backend2 != cur_backend2
        differs_m2 = staged_model2 is not None and staged_model2 != self._session_model2
        if differs_b2 or differs_m2:
            label_b2 = (staged_backend2 or cur_backend2 or "auto").upper()
            label_m2 = staged_model2 or self._session_model2 or "—"
            role2 = self._next_role2 or self._session_role2 or ""
            role_part = f" [dim]({role2})[/dim]" if role2 else ""
            out += (
                f"[b]Staged Model 2:[/b] [yellow]{label_b2}[/yellow] · "
                f"{_esc(label_m2)}{role_part}\n"
            )

        # Stage model3
        cur_backend3 = (
            self._session_backend_info3.name if self._session_backend_info3 else None
        )
        staged_backend3 = self._next_backend3
        staged_model3 = self._next_model3
        differs_b3 = staged_backend3 is not None and staged_backend3 != cur_backend3
        differs_m3 = staged_model3 is not None and staged_model3 != self._session_model3
        if differs_b3 or differs_m3:
            label_b3 = (staged_backend3 or cur_backend3 or "auto").upper()
            label_m3 = staged_model3 or self._session_model3 or "—"
            role3 = self._next_role3 or self._session_role3 or ""
            role_part = f" [dim]({role3})[/dim]" if role3 else ""
            out += (
                f"[b]Staged Model 3:[/b] [yellow]{label_b3}[/yellow] · "
                f"{_esc(label_m3)}{role_part}\n"
            )
        if self._session_seed is not None:
            out += f"[b]Seed in use:[/b] [green]{_esc(str(self._session_seed))}[/green]\n"
        if self._next_seed is not None:
            out += f"[b]Staged seed:[/b] [dim]{_esc(str(self._next_seed))}[/dim]\n"
        # /ref visibility in the status panel:
        #   - staged before /new (no active agent yet)
        #   - queued for the next user turn on an active session
        if self._staged_ref_image_name:
            out += (
                f"[b]Ref image:[/b] [yellow]{_esc(self._staged_ref_image_name)}[/yellow] "
                "[dim](staged for first turn of next session)[/dim]\n"
            )
        elif (
            self.agent is not None
            and bool(getattr(self.agent, "_next_image_bytes", None))
        ):
            model_is_vlm = getattr(self.agent, "_is_vlm", None)
            if model_is_vlm is False:
                hint = " [yellow](queued, but active model is text-only)[/yellow]"
            else:
                hint = " [dim](queued for next user turn)[/dim]"
            out += f"[b]Ref image:[/b] [green]queued[/green]{hint}\n"
        # Context window row: hide entirely when neither max nor a
        # running agent is available (avoids a sad-looking "0 / 0" on
        # backends that don't expose context_length).
        ctx_row = self._format_ctx_row()
        if ctx_row:
            out += ctx_row
        out += f"[b]Goal:[/b] {self._goal or '—'}\n"
        if self.agent is not None and getattr(self.agent, "_active_skeleton", None) is not None:
            out += f"[b]Skeleton:[/b] [cyan]{_esc(self.agent._active_skeleton)}[/cyan]\n"
        if self._last_diagnose:
            out += f"[b]Last fix:[/b] [dim]{_esc(self._last_diagnose)}[/dim]\n"
        if self._last_stall_reason:
            out += f"[b]Last stall:[/b] [yellow]{_esc(self._last_stall_reason)}[/yellow]\n"
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
        # Feedback ledger — persistent receipt of every feedback item this
        # session. Most items are consumed within milliseconds, so the
        # transient "Queued (N)" section alone left the user wondering
        # whether their prompts were received at all (2026-06-11 FPS trace).
        if self._feedback_ledger:
            out += f"\n[b]Feedback ({len(self._feedback_ledger)}):[/b]\n"
            for entry in self._feedback_ledger[-4:]:
                txt = str(entry.get("text") or "")
                preview = txt[:60] + ("…" if len(txt) > 60 else "")
                status = entry.get("status")
                if status == "applied":
                    it = entry.get("iter") or "?"
                    if entry.get("ok"):
                        badge = f"[green]✓ applied (iter {it})[/green]"
                    else:
                        badge = f"[yellow]△ applied (iter {it}, checks failing)[/yellow]"
                elif status == "wrote":
                    it = entry.get("iter") or "?"
                    badge = f"[cyan]↦ wrote code (iter {it})[/cyan]"
                elif status == "no_usable":
                    badge = "[red]✗ no usable code[/red]"
                elif status == "applying":
                    badge = "[blue]→ applying[/blue]"
                else:
                    badge = "[yellow]⏳ queued[/yellow]"
                out += f"  {badge} [dim]{_esc(preview)}[/dim]\n"
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
        #
        # Item 5: also show an AUTO-mode badge when step-mode is OFF,
        # so the user always sees which mode is active rather than
        # inferring from absence-of-badge. Yellow = WAIT (pause per
        # iter); green-dim = AUTO (continuous).
        prefix_badges: list[str] = []
        step_on = self._effective_step_mode()
        if step_on:
            prefix_badges.append("[black on yellow] WAIT MODE [/]")
        else:
            prefix_badges.append("[black on green] AUTO [/]")
        if self._run_profile != "custom":
            profile_badge = self._format_run_profile().replace("_", " ").upper()
            prefix_badges.append(f"[black on blue] {profile_badge} [/]")
        if getattr(self, "_selection_mode_on", False):
            prefix_badges.append("[black on cyan] SELECT [/]")
        badge_prefix = " ".join(prefix_badges) + (" " if prefix_badges else "")
        # Mirror the WAIT badge onto the Footer row so the user sees it
        # right next to the keybinding hints, not just on the mode-bar
        # one line above.
        try:
            footer_badge = self.query_one("#footer-badge", Static)
            if step_on:
                footer_badge.update("WAIT MODE")
                footer_badge.add_class("-on")
            else:
                footer_badge.remove_class("-on")
        except Exception:
            pass

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
        elif (
            self._is_streaming
            or self._model2_is_streaming
            or self._model3_is_streaming
        ):
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

    def _render_gpu_placement_block(self) -> str:
        """GPU map: Model 1/2/3 (tag + GPU each) and Diffusers."""
        try:
            import gpu_status as gs
        except Exception:
            return ""
        snap = gs.snapshot_gpus()
        agent = getattr(self, "agent", None)
        rows: list[str] = []
        my_pid = os.getpid()

        def _collect_slots() -> list[tuple[str | None, str | None, backend_mod.BackendInfo | None]]:
            out: list[tuple[str | None, str | None, backend_mod.BackendInfo | None]] = []
            if agent is not None:
                b1 = getattr(agent, "_backend", None)
                out.append((
                    getattr(agent, "model", None),
                    "coder",
                    b1.info if b1 else None,
                ))
                if getattr(agent, "_backend2", None) is not None:
                    out.append((
                        self._session_model2,
                        getattr(agent, "_model2_role", None) or self._session_role2,
                        agent._backend2.info,
                    ))
                if getattr(agent, "_backend3", None) is not None:
                    out.append((
                        self._session_model3,
                        getattr(agent, "_model3_role", None) or self._session_role3,
                        agent._backend3.info,
                    ))
            elif self._session_model:
                out.append((self._session_model, "coder", self._session_backend_info))
                if self._session_backend_info2:
                    out.append((
                        self._session_model2, self._session_role2,
                        self._session_backend_info2,
                    ))
                if self._session_backend_info3:
                    out.append((
                        self._session_model3, self._session_role3,
                        self._session_backend_info3,
                    ))
            return out

        def _ollama_ps_entry(
            tag: str | None,
            endpoint: str | None = None,
        ) -> dict | None:
            if not tag:
                return None
            want = tag.strip()
            ep = (endpoint or "").strip().rstrip("/")
            for m in ollama_ps:
                if (m.get("name") or "").strip() != want:
                    continue
                row_ep = (m.get("endpoint") or "").strip().rstrip("/")
                if ep and row_ep and row_ep != ep:
                    continue
                return m
            return None

        def _slot_gpu_line(
            model: str | None,
            bi: backend_mod.BackendInfo | None,
        ) -> str:
            if bi is None and not model:
                return "not loaded"
            if bi and bi.name == "ollama":
                endpoint_gpu = gs.ollama_endpoint_gpu_index(bi.endpoint or "")
                entry = _ollama_ps_entry(
                    model,
                    bi.endpoint if bi else None,
                )
                vram_gib = entry.get("vram_gib") if entry else None
                vram_bytes = entry.get("size_vram_bytes") if entry else None
                if entry is None:
                    if endpoint_gpu is not None:
                        placed = gs.format_model_gpu_placement([endpoint_gpu], snap)
                        return f"{placed} [dim](pinned, not loaded)[/dim]"
                    return gs.format_model_gpu_placement([], snap, not_loaded=True)
                slot_gpus = (
                    [endpoint_gpu] if endpoint_gpu is not None
                    else gs.gpu_indices_for_ollama_loaded_model(
                        snap,
                        vram_bytes=vram_bytes,
                        vram_gib=vram_gib,
                    )
                )
                return gs.format_model_gpu_placement(
                    slot_gpus, snap, vram_gib=vram_gib,
                )
            if bi and bi.name == "mlx":
                gpus = gs.pids_on_gpus(snap, pid=my_pid) or gs.large_python_gpu_indices(
                    snap, exclude_pid=my_pid,
                )
                if not gpus:
                    return gs.format_model_gpu_placement([], snap, not_loaded=True)
                return gs.format_model_gpu_placement(gpus, snap)
            gpus = gs.pids_on_gpus(snap, pid=my_pid)
            if not gpus:
                return gs.format_model_gpu_placement([], snap, not_loaded=True)
            return gs.format_model_gpu_placement(gpus, snap)

        def _same_physical_as_slot1(
            slot_i: int,
            model: str | None,
            bi: backend_mod.BackendInfo | None,
            m1: str | None,
            b1: backend_mod.BackendInfo | None,
        ) -> bool:
            if slot_i <= 1 or bi is None or b1 is None:
                return False
            return (
                bi.name == b1.name
                and (model or "").strip() == (m1 or "").strip()
                and (
                    bi.name != "ollama"
                    or (bi.endpoint or "").rstrip("/") == (b1.endpoint or "").rstrip("/")
                )
            )

        def _slot_is_streaming(slot_i: int) -> bool:
            if slot_i == 1:
                return self._is_streaming
            if slot_i == 2:
                return self._model2_is_streaming
            if slot_i == 3:
                return self._model3_is_streaming
            return False

        slots = _collect_slots()
        ollama_ps = gs.ollama_loaded_models()

        if slots:
            rows.append("  [b]LLM[/b]")
            m1_model, _, m1_bi = slots[0]
            m1_gpu = _slot_gpu_line(m1_model, m1_bi)
            live_gpus: list[str] = []
            for slot_i, (model, role, bi) in enumerate(slots, start=1):
                if bi is None and not model:
                    continue
                bname = (bi.name if bi else "?").upper()
                role_s = f" ({role})" if role else ""
                gpu_line = m1_gpu if _same_physical_as_slot1(slot_i, model, bi, m1_model, m1_bi) else _slot_gpu_line(model, bi)
                suffix = ""
                if _same_physical_as_slot1(slot_i, model, bi, m1_model, m1_bi):
                    suffix = " · [dim]same VRAM[/dim]"
                live = ""
                if _slot_is_streaming(slot_i):
                    live = " · [green]live[/green]"
                    m = re.search(r"GPU\s*(\d+)", gpu_line)
                    if m:
                        live_gpus.append(f"GPU {m.group(1)} ({role or '?'})")
                    elif "not loaded" not in gpu_line.lower():
                        live_gpus.append(role or f"slot{slot_i}")
                rows.append(
                    f"  Model {slot_i}{role_s} · {_esc(model or '—')} · {bname} · "
                    f"{gpu_line}{suffix}{live}"
                )
            if live_gpus:
                rows.append(
                    "  [green]busy now:[/green] " + ", ".join(live_gpus)
                )
            split_tip = gs.ollama_split_tip_short(snap)
            if split_tip:
                rows.append(f"  [yellow]{_esc(split_tip)}[/yellow]")

        rows.append("  [b]Diffusers[/b]")
        zgen = getattr(agent, "_asset_generator", None) if agent else None
        sgen = getattr(agent, "_sound_generator", None) if agent else None
        z_line = gs.format_diffuser_line("Z-Image-Turbo", None)
        sd_line = gs.format_diffuser_line("SD-Turbo img2img", None)
        if zgen is not None:
            kind = gs.diffuser_kind(zgen)
            if kind == "Z-Image-Turbo":
                z_line = gs.format_diffuser_line(kind, zgen)
            elif kind == "SD-Turbo img2img":
                sd_line = gs.format_diffuser_line(kind, zgen)
        rows.append(f"  {z_line}")
        rows.append(f"  {sd_line}")
        rows.append(f"  {gs.format_diffuser_line('Stable-Audio', sgen)}")
        # nvidia-smi may show large chat-process VRAM (e.g. GPU 0 ~20 GB)
        # before/without _asset_generator set — not the same as "loaded" above.
        if zgen is None and sgen is None:
            chat_vram = gs.chat_process_gpu_vram(snap, my_pid, min_mib=8000)
            if chat_vram:
                parts = [
                    f"GPU {gi} ~{mem / 1024:.0f} GB" for gi, mem in chat_vram
                ]
                rows.append(
                    "  [dim]chat process · "
                    + ", ".join(parts)
                    + " — not Ollama; diffusers say "
                    "'not loaded' until first sprite/sound gen "
                    "this session (or restart chat if VRAM is stale)[/dim]"
                )

        vram_footer = gs.format_vram_footer(snap)
        if vram_footer:
            rows.append(f"  [dim]VRAM · {vram_footer}[/dim]")

        if not slots and not vram_footer:
            return ""
        block = "\n[b]GPU map:[/b]\n" + "\n".join(rows) + "\n"
        self._last_gpu_summary_plain = self._MARKUP_RE.sub("", block).strip()
        return block

    def _render_assets_block(self) -> str:
        """Sticky multi-line summary of the most recent asset batch.

        Phase 1C — when a gen is IN-FLIGHT (the `last_stats` count on
        the generator is rising but no `assets_generated` event has
        fired yet), render a live progress row instead of the sticky
        summary so the user sees the rate + ETA in real time.
        Empty when no assets have been generated this session.
        """
        live = self._format_assets_live_progress()
        if live:
            return f"\n{live}\n"
        if not self._assets_summary:
            return ""
        return f"\n{self._assets_summary}\n"

    def _render_sounds_block(self) -> str:
        """Sticky compact summary of the most recent sound batch.

        Phase 1C — same live-progress treatment as the assets block.
        Empty when no sounds have been generated this session.
        """
        live = self._format_sounds_live_progress()
        if live:
            return f"\n{live}\n"
        if not self._sounds_summary:
            return ""
        return f"\n{self._sounds_summary}\n"

    def _format_assets_live_progress(self) -> str:
        """Phase 1C — render `Sprites: 4/12 · 2.9s avg · 0.34/s · ~24s ETA`
        when an assets gen is mid-flight. Detection: the generator has
        `last_stats` but the in-flight `_assets_in_flight_total` is set
        and > len(last_stats). Returns "" when no gen is in flight."""
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        total = getattr(self, "_assets_in_flight_total", 0) or 0
        if total <= 0:
            return ""
        gen = getattr(agent, "_asset_generator", None)
        stats = list(getattr(gen, "last_stats", None) or []) if gen else []
        produced = len(stats)
        if produced >= total:
            # Done — let the sticky summary take over on next tick.
            return ""
        avg_s = 0.0
        if stats:
            secs = [
                s.get("gen_seconds", 0.0)
                for s in stats
                if isinstance(s.get("gen_seconds"), (int, float))
            ]
            if secs:
                avg_s = sum(secs) / len(secs)
        rate = (1.0 / avg_s) if avg_s > 0.0 else 0.0
        remaining = max(0, total - produced)
        eta_s = remaining * avg_s if avg_s > 0.0 else 0.0
        return (
            f"[b]Sprites:[/b] {produced}/{total} "
            f"[dim]· {avg_s:.1f}s avg · {rate:.2f}/s · "
            f"~{eta_s:.0f}s ETA[/dim]"
        )

    def _format_sounds_live_progress(self) -> str:
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        total = getattr(self, "_sounds_in_flight_total", 0) or 0
        if total <= 0:
            return ""
        gen = getattr(agent, "_sound_generator", None)
        stats = list(getattr(gen, "last_stats", None) or []) if gen else []
        produced = len(stats)
        if produced >= total:
            return ""
        avg_s = 0.0
        if stats:
            secs = [
                s.get("gen_seconds", 0.0)
                for s in stats
                if isinstance(s.get("gen_seconds"), (int, float))
            ]
            if secs:
                avg_s = sum(secs) / len(secs)
        rate = (1.0 / avg_s) if avg_s > 0.0 else 0.0
        remaining = max(0, total - produced)
        eta_s = remaining * avg_s if avg_s > 0.0 else 0.0
        return (
            f"[b]Sounds:[/b] {produced}/{total} "
            f"[dim]· {avg_s:.1f}s avg · {rate:.2f}/s · "
            f"~{eta_s:.0f}s ETA[/dim]"
        )

    def _render_memory_block(self) -> str:
        """Show which memory items the agent has active for this session.

        Memory layers surfaced (in order, only when non-empty):
          - Skeleton (selected at session start, used by Phase A)
          - Visual playtest recipe (matched at end of Phase A; drives
            the structured VLM checklist + auto-injected probes)
          - Opening-book hits this turn (outline + playtest +
            asset_audit + animation_audit IDs retrieved per turn,
            injected into the model's prompt as <opening_book>)

        The (existing) `_render_playbook_block` shows playbook bullets
        separately — leaves the visual organization clean.

        Without these rows the user can't tell whether the memory
        layers are actually firing on their goal; with them, they see
        which mechanism recipe the agent picked, which probes were
        added on top of the model's own <probes>, and which recipes
        the opening book pulled for the current turn.
        """
        agent = getattr(self, "agent", None)
        if agent is None:
            return ""
        parts: list[str] = []

        # Skeleton — set once at session start.
        skel = getattr(agent, "_active_skeleton", None)
        if skel:
            parts.append(f"  skeleton: [b]{skel}[/b]")

        # Visual playtest recipe (mechanism-keyed checklist for the
        # VLM critic). Surface even when no auto-probes — the recipe
        # still steers the critic prompt.
        vp_id = getattr(agent, "_active_visual_playtest_recipe_id", None)
        if vp_id:
            ap_names = list(
                getattr(agent, "_active_visual_playtest_auto_probes", []) or []
            )
            row = f"  vlm-critique checklist: [b]{vp_id}[/b]"
            if ap_names:
                row += (
                    f"  [dim]+ {len(ap_names)} auto-probe(s): "
                    + ", ".join(ap_names)
                    + "[/dim]"
                )
            parts.append(row)

        # Opening-book hits — these change per turn. Group by kind so
        # the list reads cleanly when N is small.
        hits = list(getattr(agent, "_active_opening_book_recipes", []) or [])
        if hits:
            by_kind: dict[str, list[str]] = {}
            for h in hits:
                kind = str(h.get("kind", "?"))
                hid = str(h.get("id", "?"))
                by_kind.setdefault(kind, []).append(hid)
            kind_rows = []
            for kind in ("outline", "playtest", "asset_audit", "animation_audit"):
                ids = by_kind.get(kind)
                if ids:
                    kind_rows.append(
                        f"    {kind}: " + ", ".join(ids)
                    )
            if kind_rows:
                parts.append("  opening book (this turn):")
                parts.extend(kind_rows)

        if not parts:
            return ""
        return "\n[b]Memory in use:[/b]\n" + "\n".join(parts) + "\n"

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
        # One-complete-trace (2026-06-14): the .jsonl is the single complete
        # trace now (.log / .conversation.md / .summary.md siblings retired).
        self._log(f"  jsonl trace:  {jsonl}")
        self._log(f"  snapshots:    {snaps}")
        if self._best_path is not None and self._best_path.exists():
            self._log(f"  best clean:   {self._best_path}")
        else:
            self._log("  best clean:   none saved")
        self._log("[dim]Tip: paste the jsonl trace above into your AI assistant to debug.[/dim]")
        # The jsonl trace gets `stream_heartbeat` entries every 30s
        # during a long stream — invaluable when the model goes off
        # the rails (e.g. emits 200 duplicate sprite specs and stalls
        # 25 minutes later). Run from another terminal:
        self._log("[dim]Live progress (run from another terminal):[/dim]")
        self._log(f"[dim]  tail -f {jsonl}[/dim]")
        self._log("[dim]Press Ctrl+S to enable mouse selection in this pane.[/dim]")

    def action_paste_input(self) -> None:
        """Ctrl+V - paste into the bottom input using OS/local clipboard fallback."""
        try:
            inp = self.query_one("#user-input", MultilinePasteInput)
        except Exception:
            return
        if self._paste_into_input(inp):
            return
        self._log_info(
            "[dim]Paste unavailable — clipboard appears empty or inaccessible.[/dim]"
        )

    def _append_log_mirror_line(self, plain: str) -> None:
        """Keep a bounded plain-text mirror for context-menu copy."""
        self._log_mirror_lines.append(plain)
        if len(self._log_mirror_lines) > 5000:
            self._log_mirror_lines = self._log_mirror_lines[-3000:]

    def _log_text_for_copy(self, *, tail: int | None) -> str:
        """Plain log text — prefer on-disk mirror, else in-memory lines."""
        if tail is None:
            if self._log_file_path is not None and self._log_file_path.exists():
                try:
                    return self._log_file_path.read_text(
                        encoding="utf-8", errors="replace",
                    )
                except Exception:
                    pass
            return "\n".join(self._log_mirror_lines)
        n = max(1, tail)
        if self._log_file_path is not None and self._log_file_path.exists():
            try:
                lines = self._log_file_path.read_text(
                    encoding="utf-8", errors="replace",
                ).splitlines()
                return "\n".join(lines[-n:])
            except Exception:
                pass
        return "\n".join(self._log_mirror_lines[-n:])

    def _write_system_clipboard_text(self, text: str) -> bool:
        """Best-effort write to OS clipboard text (outside Textual's local mirror)

        using system tools like pbcopy on macOS and wl-copy/xclip/xsel on Linux.
        """
        if not text:
            return False
        commands: list[list[str]]
        if sys.platform == "darwin":
            commands = [["pbcopy"], ["/usr/bin/pbcopy"]]
        else:
            commands = [
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ]
        for cmd in commands:
            if shutil.which(cmd[0]) is None:
                continue
            try:
                proc = subprocess.run(
                    cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                    check=False,
                )
                if proc.returncode == 0:
                    return True
            except Exception:
                continue
        return False

    def copy_to_clipboard(self, text: str) -> None:
        """Override Textual's copy_to_clipboard to also write to the system OS clipboard.

        This ensures both TUI actions (Cut, Copy, copying logs/status) and standard
        keyboard triggers like Ctrl+C work correctly on macOS and Linux.
        """
        # Update Textual's local clipboard state and output OSC 52
        super().copy_to_clipboard(text)
        # Attempt system clipboard write
        self._write_system_clipboard_text(text)

    @property
    def clipboard(self) -> str:
        """Get the clipboard value, prioritizing the OS clipboard, then Textual's local fallback."""
        sys_text = self._read_system_clipboard_text()
        if sys_text is not None:
            return sys_text
        return getattr(self, "_clipboard", "")

    def _read_system_clipboard_text(self) -> str | None:
        """Best-effort read of OS clipboard text (outside Textual's local mirror)."""
        commands: list[list[str]]
        if sys.platform == "darwin":
            commands = [["pbpaste"]]
        else:
            commands = [
                ["wl-paste", "--no-newline"],
                ["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "--clipboard", "--output"],
            ]
        for cmd in commands:
            if shutil.which(cmd[0]) is None:
                continue
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                    check=False,
                )
            except Exception:
                continue
            if proc.returncode == 0:
                return proc.stdout or ""
        return None

    def _paste_into_input(self, inp: MultilinePasteInput) -> bool:
        """Paste into input using the app's OS/local clipboard."""
        return inp.paste_flattened_text(self.clipboard)

    def _input_context_menu_items(
        self, inp: MultilinePasteInput,
    ) -> list[tuple[str, str, bool]]:
        has_sel = not inp.selection.is_empty
        return [
            ("Cut", "cut", not has_sel),
            ("Copy", "copy", not has_sel),
            # Keep paste enabled even when Textual's clipboard mirror is
            # empty; the terminal/system clipboard can still have content.
            ("Paste", "paste", False),
            ("Select all", "select_all", False),
        ]

    def _log_context_menu_items(self) -> list[tuple[str, str, bool]]:
        has_log = bool(self._log_mirror_lines) or (
            self._log_file_path is not None and self._log_file_path.exists()
        )
        return [
            (
                f"Copy last {_CONTEXT_LOG_TAIL_LINES} lines",
                "copy_log_tail",
                not has_log,
            ),
            ("Copy full log", "copy_log_all", not has_log),
            ("Enable selection mode (drag to copy)", "selection_mode", False),
        ]

    def _status_context_menu_items(self) -> list[tuple[str, str, bool]]:
        has_body = bool((self._status_plain or "").strip())
        return [
            ("Copy status panel", "copy_status", not has_body),
            ("Enable selection mode (drag to copy)", "selection_mode", False),
        ]

    async def _dismiss_context_menu(self) -> None:
        menu = self._context_menu
        self._context_menu = None
        self._context_menu_origin = ""
        if menu is not None:
            try:
                await menu.remove()
            except Exception:
                pass

    async def _show_context_menu(
        self,
        *,
        screen_x: int,
        screen_y: int,
        items: list[tuple[str, str, bool]],
        origin: str,
    ) -> None:
        await self._dismiss_context_menu()
        sw = max(1, self.size.width)
        sh = max(1, self.size.height)
        # Keep the whole popup on screen; opening near the bottom used to
        # place the menu off-screen so it couldn't be navigated.
        menu_w = 36
        menu_h = min(14, max(1, len(items))) + 2
        x = max(0, min(screen_x, max(0, sw - menu_w)))
        y = max(0, min(screen_y, max(0, sh - menu_h)))
        menu = ContextMenuOverlay(items, screen_x=x, screen_y=y)
        self._context_menu = menu
        self._context_menu_origin = origin
        await self.screen.mount(menu)

    async def _run_context_menu_action(self, action_id: str | None) -> None:
        if not action_id:
            return
        origin = self._context_menu_origin
        try:
            if origin == "input":
                inp = self.query_one("#user-input", MultilinePasteInput)
                # Focus the input back so cursor state is active and visible
                inp.focus()
                if action_id == "cut":
                    inp.action_cut()
                elif action_id == "copy":
                    inp.action_copy()
                elif action_id == "paste":
                    if not self._paste_into_input(inp):
                        raise SkipAction()
                elif action_id == "select_all":
                    inp.action_select_all()
                else:
                    return
                self._log_info("[dim]Clipboard action applied to input.[/dim]")
            elif origin == "log":
                if action_id == "copy_log_tail":
                    text = self._log_text_for_copy(
                        tail=_CONTEXT_LOG_TAIL_LINES,
                    )
                    if text.strip():
                        self.copy_to_clipboard(text)
                        self._log_info(
                            f"[dim]Copied last {_CONTEXT_LOG_TAIL_LINES} "
                            "log lines to clipboard.[/dim]"
                        )
                elif action_id == "copy_log_all":
                    text = self._log_text_for_copy(tail=None)
                    if text.strip():
                        self.copy_to_clipboard(text)
                        self._log_info("[dim]Copied full log to clipboard.[/dim]")
                elif action_id == "selection_mode":
                    if not getattr(self, "_selection_mode_on", False):
                        await self.action_toggle_selection_mode()
                    else:
                        self._log_info("[dim]Selection mode is already on.[/dim]")
                else:
                    return
            elif origin == "status":
                if action_id == "copy_status":
                    text = (self._status_plain or "").strip()
                    if text:
                        self.copy_to_clipboard(text)
                        self._log_info(
                            "[dim]Copied status panel to clipboard.[/dim]"
                        )
                elif action_id == "selection_mode":
                    if not getattr(self, "_selection_mode_on", False):
                        await self.action_toggle_selection_mode()
                    else:
                        self._log_info("[dim]Selection mode is already on.[/dim]")
                else:
                    return
        except SkipAction:
            self._log_info("[dim]That action is not available right now.[/dim]")
        except Exception as e:
            self._log_info(f"[dim]Clipboard action failed: {e!r}[/dim]")

    async def on_context_menu_overlay_closed(
        self, message: ContextMenuOverlay.Closed,
    ) -> None:
        origin = self._context_menu_origin
        action_id = message.action_id
        await self._dismiss_context_menu()
        self._context_menu_origin = origin
        await self._run_context_menu_action(action_id)

    async def on_multiline_paste_input_right_click_request(
        self, message: MultilinePasteInput.RightClickRequest,
    ) -> None:
        """Open input menu for right-clicks consumed by Input internals."""
        inp = self.query_one("#user-input", MultilinePasteInput)
        await self._show_context_menu(
            screen_x=message.screen_x,
            screen_y=message.screen_y,
            items=self._input_context_menu_items(inp),
            origin="input",
        )

    async def on_mouse_down(self, event: events.MouseDown) -> None:
        """Right-click context menus on input, log, and status panes."""
        target = event.widget
        if _widget_has_id(target, "context-menu-overlay"):
            return
        menu = self._context_menu
        if menu is not None:
            await self._dismiss_context_menu()
        if event.button not in _RIGHT_CLICK_BUTTONS:
            return
        event.stop()
        sx = getattr(event, "screen_x", None)
        sy = getattr(event, "screen_y", None)
        if sx is None or sy is None:
            sx, sy = event.x, event.y
        if _widget_has_id(target, "user-input"):
            inp = self.query_one("#user-input", MultilinePasteInput)
            await self._show_context_menu(
                screen_x=int(sx),
                screen_y=int(sy),
                items=self._input_context_menu_items(inp),
                origin="input",
            )
        elif _widget_has_id(target, "log-pane"):
            await self._show_context_menu(
                screen_x=int(sx),
                screen_y=int(sy),
                items=self._log_context_menu_items(),
                origin="log",
            )
        elif _widget_has_id(target, "status-pane") or _widget_has_id(
            target, "status-body",
        ) or _widget_has_id(target, "status-scroll"):
            await self._show_context_menu(
                screen_x=int(sx),
                screen_y=int(sy),
                items=self._status_context_menu_items(),
                origin="status",
            )

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
        # Clear manual /help status content on the next non-help input.
        if self._status_manual_body is not None:
            lower = text.lower()
            if not lower.startswith("/help") and lower not in {"/h", "/?"}:
                self._status_manual_body = None

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
            message.input.placeholder = "feedback · 'done' or Ctrl+D to ship · /help"
            self.sub_title = "agent is working"
            # A prior session that already finished (e.g. a prompt loaded via
            # /games after shipping the last game) needs the full re-init
            # _new_session does — agent reset, log rotate, browser clear —
            # not a bare _start_session. The first game of a session has
            # agent is None and takes the simple path.
            if self.agent is not None and self._session_done:
                await self._new_session(text)
            else:
                self._goal = text
                self._log(f"[bold green]>[/bold green] {text}")
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
            # Feedback ledger: record receipt. Status flips to "applying"
            # when the agent drains it (reconciled in _handle_event) and
            # to "applied" on the next test event.
            self._feedback_ledger.append(
                {"text": text, "status": "queued", "iter": None, "ok": None}
            )
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
                self._cmd_help(arg)
            elif cmd in ("list", "models"):
                self._cmd_list_models()
            elif cmd in ("model", "load"):
                self._cmd_set_model(arg)
            elif cmd == "model2":
                self._cmd_set_model2(arg)
            elif cmd == "model3":
                self._cmd_set_model3(arg)
            elif cmd in ("modelall", "loadall"):
                self._cmd_set_model_all(arg)
            elif cmd == "retry":
                await self._cmd_retry(arg)
            elif cmd == "backend":
                self._cmd_set_backend(arg)
            elif cmd == "unload":
                self._cmd_unload(arg)
            elif cmd == "new":
                await self._cmd_new(arg)
            elif cmd in ("games", "game", "library", "prompts"):
                self._cmd_games(arg)
            elif cmd in ("goodgame", "good"):
                self._cmd_goodgame()
            elif cmd == "ship":
                await self.action_ship_it()
            elif cmd in ("revert", "rewind"):
                self._cmd_revert(arg)
            elif cmd in ("unqueue", "dequeue", "clearqueue"):
                self._cmd_unqueue(arg)
            elif cmd == "quit":
                await self.action_quit_app()
            elif cmd in ("log", "paths", "files"):
                await self.action_show_log_paths()
            elif cmd == "open":
                self._cmd_open()
            elif cmd == "edit":
                self._cmd_edit()
            elif cmd == "clear":
                self.query_one("#log-pane", RichLog).clear()
                self._log_mirror_lines = []
            elif cmd == "iters":
                self._cmd_set_iters(arg)
            elif cmd == "ctx":
                self._cmd_set_ctx(arg)
            elif cmd == "seed":
                self._cmd_set_seed(arg)
            elif cmd == "reset":
                await self._cmd_reset()
            elif cmd == "status":
                self._cmd_status()
            elif cmd == "wait":
                self._cmd_toggle_wait(arg)
            elif cmd in ("iter-detail", "iterdetail"):
                self._cmd_iter_detail(arg)
            elif cmd == "mode":
                self._cmd_set_mode(arg)
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
            elif cmd == "check":
                await self._cmd_check(arg)
            elif cmd == "ask":
                await self._cmd_ask(arg)
            elif cmd == "ref":
                self._cmd_attach_ref_image(arg)
            elif cmd == "prefill":
                self._cmd_toggle_prefill(arg)
            elif cmd in ("architect", "arch"):
                self._cmd_toggle_architect(arg)
            elif cmd in ("double-screenshot", "doublescreenshot", "ds"):
                self._cmd_toggle_double_screenshot(arg)
            elif cmd in ("vlm-critique", "vlmcritique", "vc", "judge"):
                self._cmd_toggle_vlm_critique(arg)
            elif cmd in ("allroles", "all-roles"):
                self._cmd_toggle_allroles(arg)
            elif cmd in ("leanprompt", "lean-prompt", "lean"):
                self._cmd_set_leanprompt(arg)
            elif cmd in ("critique", "playtest", "feedback"):
                self._cmd_toggle_autonomous_feedback(arg)
            elif cmd in ("rawfeedback", "raw-feedback", "raw"):
                self._cmd_toggle_raw_feedback(arg)
            else:
                self._log_info(f"unknown command /{cmd} — type /help")
        except Exception as e:
            self._log_error(f"/{cmd} failed: {e}")

    def _cmd_help(self, arg: str = "") -> None:
        import tui_help as _tui_help

        topic_id = _tui_help.normalize_help_topic(arg)
        if arg.strip() and topic_id is None:
            for line in _tui_help.format_unknown_topic_message(arg):
                self._log_info(line)
            return
        if topic_id is not None:
            lines = _tui_help.help_topic_lines(topic_id) or []
            self._render_help_lines(lines)
            return

        lines = [
            "[bold cyan]── what to type when ──[/bold cyan]",
            "  [b]first run[/b]                  describe the game you want, press Enter",
            "  [b]small change to shipped game[/b]  just type it — no slash needed",
            "                                  e.g. [italic]ship is too slow, double the thrust[/italic]",
            "  [b]ship as-is, stop[/b]              type [b]done[/b] / [b]looks good[/b] / [b]ship[/b] (or Ctrl+D)",
            "                                  [dim]Ctrl+D wins: any queued feedback (incl. autonomous playtest) is dropped — re-send after ship if still wanted[/dim]",
            "  [b]brand-new unrelated game[/b]      [b]/new <goal>[/b]",
            "  [b]start from an existing .html[/b]  [b]/seed <path>[/b]  then type your request (or [b]/new <goal>[/b] for a new game)",
            "                                  [dim]existing sprites/sounds are reused — the planner won't regenerate them[/dim]",
            "  [b]paste in input[/b]                [b]Ctrl+V[/b] or terminal Edit→Paste ([b]Cmd+V[/b] on macOS)",
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
            # =====================================================================
            # SLASH COMMANDS — grouped by purpose so the user can scan one section
            # instead of hunting through a flat list. Aliases shown in [dim] after
            # the canonical name; bare command (no args) usually shows current
            # state or clears, where relevant.
            # =====================================================================
            "[bold cyan]── session lifecycle ──[/bold cyan]",
            "  [b]/new[/b]                       reset to a clean slate — type your game idea next",
            "  [b]/new <goal>[/b]                start a fresh game (uses staged seed/model if any)",
            "  [b]/goodgame[/b]                  copy best.html + *_assets/ + *_sounds/ → goodgame/ [dim](not gitignored)[/dim]",
            "  [b]/ship[/b]                      ship current build [dim](= Ctrl+D, or type 'done' / 'looks good')[/dim]",
            "  [b]/revert [N][/b]               roll the game file back to the last clean iter [dim](or iter N specifically; aliases /rewind)[/dim]",
            "                                  [dim]use this when the model breaks something — one keystroke beats typing 'undo that'[/dim]",
            "  [b]/unqueue[/b]                   drop your last typed feedback only (accidental queue) [dim](/dequeue)[/dim]",
            "                                  [dim]older queued lines stay; /unqueue all or /unqueue N for more[/dim]",
            "  [b]/retry[/b]                     re-run after a bad model (keeps game file + trace)",
            "  [b]/reset[/b]                     same as bare [b]/new[/b] — clear staging + wait for a goal",
            "  [b]/open[/b]                      open the current game in your default browser",
            "  [b]/edit[/b]                      open Asset Studio in your default browser",
            "  [b]/clear[/b]                     clear the agent log pane (no effect on staged state)",
            "  [b]/quit[/b]                      quit the TUI [dim](= Ctrl+Q)[/dim]",
            "",
            "[bold cyan]── models ──[/bold cyan]",
            "  [b]/list[/b]                      unified Ollama + MLX (+ cloud if keys set) list with numbers [dim](alias /models)[/dim]",
            "  [b]/load <N|name>[/b]             pick model #N from /list (any backend); sticky across /new [dim](alias /model)[/dim]",
            "  [b]/model2 <N|name> [--role critic|architect][/b]   stage sidecar slot 2",
            "                                  [dim]omit N to inherit staged model 1: /model2 --role critic or /model2 --critic[/dim]",
            "  [b]/model3 <N|name> [--role critic|architect][/b]   stage sidecar slot 3",
            "                                  [dim]omit N to inherit model 1: /model3 --role architect or /model3 --architect[/dim]",
            "  [b]/modelall <N|name>[/b]         stage SAME model on all 3 slots (coder + critic + architect) [dim](alias /loadall)[/dim]",
            "  [b]/backend <auto|ollama|mlx|openai|anthropic>[/b]  default backend when no specific model is staged",
            "                                  [dim]cloud backends require OPENAI_API_KEY / ANTHROPIC_API_KEY in shell env[/dim]",
            "  [b]/launch <N|name|path>[/b]     stage an MLX model for next /new (loads in-process on first request)",
            "                                  [dim]MLX stalls don't auto-fall-back to Ollama — use /backend ollama + /load to switch[/dim]",
            "  [b]/unload [N|name|all|mlx][/b]  free VRAM · bare = active session · all = every Ollama · mlx = drop in-process MLX",
            "",
            "[bold cyan]── run knobs (all sticky across /new) ──[/bold cyan]",
            "  [b]/seed <path>[/b]               stage a baseline .html · bare = clear",
            "  [b]/iters <N>[/b]                 max iterations per session",
            "  [b]/ctx [N|100k|131k|262k|full][/b]   Ollama context window · default 100k · bare = show current",
            "                                  [dim]raises KV VRAM; Ollama reloads on next request — preload with[/dim]",
            "                                  [dim]`ollama run --ctx-size N <model>` to avoid a stall[/dim]",
            "  [b]/restarts <N>[/b]              independent full restarts when iter-1 score < 60 · default 2 · 1=off",
            "  [b]/model-class <auto|small|mid|large>[/b]   prompt-size trim · default 'small' = lean ~5 KB",
            "  [b]/mode <local_manual|local_auto|local_plus_review with <model> [--auto-apply]|custom>[/b]",
            "                                  run contract preset · [dim]reviewer auto-apply runs only with /wait off[/dim]",
            "",
            "[bold cyan]── feature toggles ──[/bold cyan]",
            "  [b]/wait [on|off][/b]             step-mode: pause after each iter; Enter or feedback to continue",
            "                                  [dim]/wait on auto-disables /vlm-critique (you're the reviewer); restored on /wait off[/dim]",
            "  [b]/vlm-critique [on|off][/b]    review WITH vision: looks at the screen, tells the agent \u00b7 default off",
            "                                  [dim]alias: /judge · uses a memory checklist when one fits the game[/dim]",
            "                                  [dim]uses model 2 to look when your main model can't see[/dim]",
            "                                  [dim]RECOMMENDED: OFF when YOU review each iter \u2014 auto review adds paraphrased noise[/dim]",
            "  [b]/critique [on|off][/b]        review WITHOUT vision: plays it + reads the report, tells the agent \u00b7 default ON",
            "                                  [dim]aliases: /playtest /feedback · catches frozen loops / dead controls[/dim]",
            "                                  [dim]sends problems to the coder so it fixes them next round[/dim]",
            "                                  [dim]test reports, patch diagnostics, and vlm-critique still run when off[/dim]",
            "  [b]/rawfeedback [on|off][/b]     YOUR typed feedback goes to the model verbatim · default ON",
            "                                  [dim]flip OFF to opt-in to MEDIA-CHANGE / ORIENTATION / SCOPE wrappers on your feedback[/dim]",
            "                                  [dim]machine bug feedback (test reports, console errors, probes) is UNAFFECTED[/dim]",
            "  [b]/playbook [on|off][/b]        playbook bullet injection · default ON [dim](alias /memory)[/dim]",
            "                                  [dim]A/B vs one-shot when iters feel worse than no agent[/dim]",
            "  [b]/architect [on|off][/b]       architect/editor split on complex first-builds · default off [dim](alias /arch)[/dim]",
            "                                  [dim]Phase B split — for the slot use /model2 … --role architect instead[/dim]",
            "  [b]/allroles[/b]                  bundle: /architect on + /vlm-critique on, all on your one loaded LLM",
            "                                  [dim]no extra GPUs; staged /model2 / /model3 slots still win[/dim]",
            "  [b]/prefill [on|off][/b]         force assistant prefill tags · default ON [dim](XML syntax compliance)[/dim]",
            "  [b]/double-screenshot [on|off][/b]  capture startup + after-input screenshots · default off [dim](alias /ds)[/dim]",
            "                                  [dim]helps debug movement; needs /vlm-critique on to be useful[/dim]",
            "  [b]/iter-detail [on|off][/b]    extra blocker details after each iter decision · default off",
            "",
            "[bold cyan]── inspection ──[/bold cyan]",
            "  [b]/status[/b]                    model, phase, iteration, paths, what's staged",
            "  [b]/log[/b]                       print all session artifact paths [dim](= Ctrl+L; aliases /paths /files)[/dim]",
            "  [b]/audit[/b]                     per-playbook-bullet earnings (fires, pass-rate, avg-iter) from trace history",
            "  [b]/check [<N|name>][/b]         visual review on latest screenshot · bare uses active session VLM",
            "                                  [dim]WAIT ON: loads suggested feedback into input for edit/Enter[/dim]",
            "                                  [dim]WAIT OFF: auto-queues guidance into the next coding turn[/dim]",
            "  [b]/ask <question>[/b]            read-only Q&A about the current game — no code changes",
            "                                  [dim]e.g. /ask how does digging at skull beach work?[/dim]",
            "  [b]/ref <path>[/b]               attach a reference image (PNG/JPEG/WebP) to the NEXT user turn",
            "                                  [dim]works before /new too; drag a file from Finder into the terminal to fill the path[/dim]",
            "                                  [dim]VLM-only — say 'make the game look like this' on the next line[/dim]",
            "  [b]/help[/b]                     command list [dim](aliases /h /?)[/dim]",
            "  [b]/help <topic>[/b]             detail: [b]feedback[/b], [b]vlm-critique[/b], [b]rawfeedback[/b], [b]ask[/b]",
            "",
            "[bold cyan]── visual playtest recipes (auto-applied) ──[/bold cyan]",
            "  Hand-curated MECHANISM-keyed checklists the VLM critic uses. The matcher",
            "  picks a recipe from your goal + the model's <plan> text + asset names so",
            "  it works even when you don't name the game ([italic]'collect dots while avoiding[/italic]",
            "  [italic]ghosts in corridors'[/italic] → canvas-grid-navigation). ~11 mechanism recipes",
            "  cover the top-100 games — Pacman, Doom, fighters, Mario, chess, Tetris, etc.",
            "  Three of them also carry [b]auto-probes[/b] — deterministic JS assertions that",
            "  catch state-shape regressions even if the VLM misses them (two-actor",
            "  facing flip, player-in-wall, player off-screen). Trace events:",
            "  [b]visual_playtest_recipe_used[/b], [b]visual_playtest_auto_probes_injected[/b].",
            "  Library file: [b]memory/visual_playtests.jsonl[/b] (hand-edited data file, tracked in git).",
            "  Adding a new recipe = append one JSONL line — no Python code change, matches next session.",
            "",
            "[bold cyan]── sticky staging ──[/bold cyan]",
            "  Run-knob commands (/seed, /load, /iters, /ctx, /restarts, /model-class, /mode)",
            "  PERSIST across multiple /new calls. Set once, reuse forever. Clear individually",
            "  with the bare command (e.g. [b]/seed[/b] alone), or wipe everything with [b]/reset[/b].",
            "",
            "[dim]Example: /seed games/asteroids.html  →  type "
            "[italic]add multiplayer[/italic]  →  type [italic]add boss[/italic]  "
            "▸ both use asteroids.html and its existing assets/sounds[/dim]",
        ]
        lines.extend(_tui_help.help_topics_index_lines())
        self._render_help_lines(lines)

    def _render_help_lines(self, lines: list[str]) -> None:
        for line in lines:
            self._log(line)
        # Mirror /help in the right status panel so command guidance is
        # visible there immediately.
        self._status_manual_body = "\n".join(lines)
        self._update_status()

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
        # Cloud backends — only listed when the corresponding env-var
        # key is set, so /list doesn't dangle picks that would fail
        # immediately on /load. Keys are read fresh on every refresh so
        # users who set them mid-session see entries appear on the next
        # /list without restarting.
        openai_models, _ = backend_mod.list_openai_inventory()
        for name in openai_models:
            listing.append(("openai", name))
        anthropic_models, _ = backend_mod.list_anthropic_inventory()
        for name in anthropic_models:
            listing.append(("anthropic", name))
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
            "[dim]  [O]llama / [M]LX / open[X]AI / [C]laude  ·  "
            "* = loaded in Ollama VRAM now  ·  ← active = bound to this session  ·  "
            "← staged = next /new  ·  "
            "[magenta]VLM[/magenta] = can read screenshots (vision-language) · "
            "[dim]text[/dim] = text-only[/dim]"
        )
        # Track active/staged model configurations across all slots
        staged_backend_1 = self._next_backend or "ollama"
        staged_backend_2 = self._next_backend2 or "ollama"
        staged_backend_3 = self._next_backend3 or "ollama"
        
        for i, (b, name) in enumerate(listing, 1):
            if b == "ollama":
                tag = "O"
                loaded = "*" if name in ollama_loaded else " "
            elif b == "mlx":
                tag = "M"
                loaded = "*" if name == mlx_active else " "
            elif b == "openai":
                tag = "X"
                loaded = " "
            else:  # anthropic
                tag = "C"
                loaded = " "

            is_active_1 = (
                self._session_backend_info is not None
                and self._session_backend_info.name == b
                and name == self._session_model
            )
            is_active_2 = (
                self._session_backend_info2 is not None
                and self._session_backend_info2.name == b
                and name == self._session_model2
            )
            is_active_3 = (
                self._session_backend_info3 is not None
                and self._session_backend_info3.name == b
                and name == self._session_model3
            )

            is_staged_1 = name == self._next_model and staged_backend_1 == b
            is_staged_2 = name == self._next_model2 and staged_backend_2 == b
            is_staged_3 = name == self._next_model3 and staged_backend_3 == b

            active_roles = []
            if is_active_1:
                active_roles.append("coder")
            if is_active_2:
                active_roles.append(f"model2: {self._session_role2 or 'critic'}")
            if is_active_3:
                active_roles.append(f"model3: {self._session_role3 or 'architect'}")

            staged_roles = []
            if is_staged_1:
                staged_roles.append("coder")
            if is_staged_2:
                staged_roles.append(f"model2: {self._next_role2 or 'critic'}")
            if is_staged_3:
                staged_roles.append(f"model3: {self._next_role3 or 'architect'}")

            mark_active = ""
            if active_roles:
                roles_str = ", ".join(active_roles)
                mark_active = f"  [yellow]← active ({roles_str})[/yellow]"

            mark_staged = ""
            if staged_roles:
                roles_str = ", ".join(staged_roles)
                mark_staged = f"  [magenta]← staged ({roles_str})[/magenta]"

            # Show MLX entries by short basename when they're disk paths
            if b == "mlx" and "/" in name:
                display = Path(name).name
                hint = f"  [dim]({Path(name).parent})[/dim]"
            else:
                display = name
                hint = ""

            modality = backend_mod.classify_model_modality(name)
            if modality == "vlm":
                badge = " [magenta]\\[VLM][/magenta]"
            else:
                badge = " [dim]\\[text][/dim]"

            self._log(
                f"  [{i:>2}] [b]{tag}[/b] {loaded} {_esc(display)}"
                f"{badge}{mark_active}{mark_staged}{hint}"
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
            bases = backend_mod.ollama_unload_probe_bases()
            ports = ", ".join(b.rsplit(":", 1)[-1] for b in bases)
            results = backend_mod.unload_all_ollama_models()
            if not results:
                self._log_info(
                    f"[dim]probed Ollama on port(s) {ports}; "
                    "no model was resident in VRAM (/api/ps empty on each).[/dim]"
                )
            else:
                for name, ok, msg in results:
                    tag = "[green]✓[/green]" if ok else "[red]✗[/red]"
                    self._log_info(f"  {tag} {_esc(name)} — {_esc(msg)}")
            freed_diff = self._unload_diffusers_vram()
            if freed_diff:
                self._log_info(
                    f"[green]✓[/green] released diffuser VRAM: {_esc(', '.join(freed_diff))}"
                )
            self._log_info("[dim](MLX untouched — /unload mlx for that.)[/dim]")
            self._log_info(
                "[dim]← active in /list = this session's model picks (coder/model2/model3); "
                "that does not change after unload. Use * to see VRAM residency; "
                "run /list again.[/dim]"
            )
            self._log_post_unload_vram_hint()
            return

        if norm_lc == "model2":
            if (
                self._session_backend_info2 is None
                or self._session_backend_info2.name != "ollama"
            ):
                self._log_info("no active Ollama model2 to unload.")
                return
            self._unload_ollama_named(self._session_backend_info2.model)
            return

        if norm_lc == "model3":
            if (
                self._session_backend_info3 is None
                or self._session_backend_info3.name != "ollama"
            ):
                self._log_info("no active Ollama model3 to unload.")
                return
            self._unload_ollama_named(self._session_backend_info3.model)
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
        if backend_name in ("openai", "anthropic"):
            self._log_info(
                f"[dim]{_esc(model_name)} is a cloud model "
                f"({backend_name}); nothing to unload — no in-process "
                "weights are held. The API key stays in your shell env.[/dim]"
            )
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
        if ok and (
            self._session_model == name
            or (
                self._session_backend_info is not None
                and self._session_backend_info.model == name
            )
        ):
            self._log_info(
                "[dim]Session is still bound to that tag in VRAM terms — "
                "use [b]/model <N>[/b] to point the agent at a working model "
                "(your game file and trace are kept).[/dim]"
            )

    def _unload_diffusers_vram(self) -> list[str]:
        """Drop in-process Z-Image / img2img pipelines (chat preload + agent)."""
        freed: list[str] = []
        try:
            from assets import release_preloaded_diffusers
            freed.extend(release_preloaded_diffusers())
        except Exception:
            pass
        agent = self.agent
        if agent is not None:
            gen = getattr(agent, "_asset_generator", None)
            if gen is not None and hasattr(gen, "cleanup"):
                try:
                    gen.cleanup()
                    if "Z-Image-Turbo (session)" not in freed:
                        freed.append("Z-Image-Turbo (session)")
                except Exception:
                    pass
        return freed

    def _log_post_unload_vram_hint(self) -> None:
        """After /unload all, show remaining GPU use so 'still full' is actionable."""
        try:
            import gpu_status as gs
            snap = gs.snapshot_gpus(force=True)
        except Exception:
            return
        if snap is None:
            return
        footer = gs.format_vram_footer(snap)
        if footer:
            self._log_info(f"[dim]GPU VRAM now: {footer}[/dim]")
        split = gs.ollama_tensor_split_gpu_indices(snap)
        if split and not gs.ollama_multi_daemon_setup():
            self._log_info(
                f"[yellow]Ollama still split across GPU "
                f"{'+'.join(str(i) for i in split)} — "
                "try again or stop stray `ollama serve` processes.[/yellow]"
            )
        elif gs.ollama_multi_daemon_setup():
            loaded = gs.ollama_loaded_models()
            if loaded:
                rows = ", ".join(
                    f"{m['name']}@{m['endpoint'].rsplit(':', 1)[-1]}"
                    for m in loaded[:4]
                )
                self._log_info(
                    f"[yellow]Still loaded in Ollama: {rows} — "
                    "re-run /unload all[/yellow]"
                )
        # chat.py / diffusers often hold a separate Python allocation (not /api/ps).
        py_vram = gs.chat_process_gpu_vram(snap, os.getpid(), min_mib=4000)
        if py_vram:
            gpus = "+".join(str(g) for g, _ in py_vram)
            self._log_info(
                f"[dim]This chat process still uses GPU {gpus} "
                f"(diffusers/MLX) — /unload mlx or quit chat to drop that.[/dim]"
            )

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
        """/backend [auto|ollama|mlx|openai|anthropic] — pick the LLM
        host for the next /new.

        Sticky across /new, like /model. Bare /backend prints the current
        staged choice and what would resolve right now. Useful when both
        Ollama and mlx_lm.server are running and you want to force one,
        or when you want to switch the active session to a cloud model.

        Cloud backends (openai, anthropic) require the matching
        API-key env var set in your shell — OPENAI_API_KEY or
        ANTHROPIC_API_KEY. The TUI never reads keys from disk; the
        SDK reads them directly from os.environ at request time.
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
        if norm in ("openai", "oai", "x", "gpt"):
            if not os.environ.get("OPENAI_API_KEY"):
                self._log_error(
                    "OPENAI_API_KEY is not set — export it in your shell "
                    "before /backend openai. Key text never enters this "
                    "process; the SDK reads it from env."
                )
                return
            self._next_backend = "openai"
            self._run_profile = "custom"
            self._profile_review_model = None
            self._profile_review_auto_apply = False
            self._log_info(
                "backend → [b]openai[/b] (sticky) — "
                "[yellow]cloud calls cost real money[/yellow]"
            )
            return
        if norm in ("anthropic", "claude", "c"):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._log_error(
                    "ANTHROPIC_API_KEY is not set — export it in your shell "
                    "before /backend anthropic. Key text never enters this "
                    "process; the SDK reads it from env."
                )
                return
            self._next_backend = "anthropic"
            self._run_profile = "custom"
            self._profile_review_model = None
            self._profile_review_auto_apply = False
            self._log_info(
                "backend → [b]anthropic[/b] (sticky) — "
                "[yellow]cloud calls cost real money[/yellow]"
            )
            return
        self._log_error(
            f"unknown backend {arg!r} — pick one of: "
            "auto, ollama, mlx, openai, anthropic"
        )

    def _ollama_escape_hint(self) -> str:
        """One-line hint after a load failure or /unload — how to switch model."""
        return (
            "[yellow]Escape:[/yellow] [b]/list[/b] then [b]/model <N>[/b] "
            "(works even after the session ends — keeps your game file). "
            "Then type feedback to continue, or [b]/retry[/b] to re-run planning."
        )

    def _apply_model_to_active_session_slot(
        self, chosen_backend: str, chosen_name: str, slot: int, role: str | None, *, source: str,
    ) -> bool:
        """Point the live agent's slots at a new backend. Returns False on init error."""
        if chosen_backend == "mlx":
            desired_endpoint = backend_mod.mlx_endpoint_url()
        elif chosen_backend == "openai":
            desired_endpoint = backend_mod.openai_endpoint_url()
        elif chosen_backend == "anthropic":
            desired_endpoint = backend_mod.anthropic_endpoint_url()
        else:
            desired_endpoint = backend_mod.ollama_endpoint_url(slot)

        # Check if the requested model name & backend match what is already loaded in any other slot,
        # and reuse the Backend and BackendInfo instances directly. This avoids double-allocations,
        # redundant in-VRAM loading, and cold-start pauses on Apple Silicon. For Ollama,
        # the endpoint must also match so auto-pinned slots do not collapse to one daemon.
        reused_backend = None
        reused_info = None
        # Use getattr with a default of None in case attributes are not yet initialized (e.g. mock objects or partial setups)
        slots_to_check = [
            ("_session_backend_info", "_session_backend"),
            ("_session_backend_info2", "_session_backend2"),
            ("_session_backend_info3", "_session_backend3")
        ]
        for info_attr, inst_attr in slots_to_check:
            b_info = getattr(self, info_attr, None)
            b_inst = getattr(self, inst_attr, None)
            if b_info is not None and b_inst is not None:
                same_endpoint = (
                    chosen_backend != "ollama"
                    or (b_info.endpoint or "").rstrip("/") == desired_endpoint.rstrip("/")
                )
                if (
                    b_info.name == chosen_backend
                    and b_info.model == chosen_name
                    and same_endpoint
                ):
                    reused_backend = b_inst
                    reused_info = b_info
                    break

        if reused_backend is not None and reused_info is not None:
            new_backend = reused_backend
            new_info = reused_info
        else:
            try:
                new_info = backend_mod.BackendInfo(
                    name=chosen_backend, model=chosen_name,
                    source=source,
                    endpoint=desired_endpoint,
                )
                new_backend = backend_mod.make_backend(new_info)
            except Exception as e:
                self._log_error(f"model swap failed: {e}")
                return False

        if slot == 1:
            return self._apply_model_to_active_session(chosen_backend, chosen_name, source=source)
        elif slot == 2:
            if self.agent is not None:
                self.agent._backend2 = new_backend
                self.agent._model2_role = role
            self._session_backend2 = new_backend
            self._session_backend_info2 = new_info
            self._session_model2 = chosen_name
            self._session_role2 = role
        elif slot == 3:
            if self.agent is not None:
                self.agent._backend3 = new_backend
                self.agent._model3_role = role
            self._session_backend3 = new_backend
            self._session_backend_info3 = new_info
            self._session_model3 = chosen_name
            self._session_role3 = role

        return True

    def _apply_model_to_active_session(
        self, chosen_backend: str, chosen_name: str, *, source: str,
    ) -> bool:
        """Point the live agent at a new backend. Returns False on init error."""
        try:
            if chosen_backend == "mlx":
                endpoint = backend_mod.mlx_endpoint_url()
            elif chosen_backend == "openai":
                endpoint = backend_mod.openai_endpoint_url()
            elif chosen_backend == "anthropic":
                endpoint = backend_mod.anthropic_endpoint_url()
            else:
                endpoint = backend_mod.ollama_endpoint_url(1)
            new_info = backend_mod.BackendInfo(
                name=chosen_backend, model=chosen_name,
                source=source,
                endpoint=endpoint,
            )
            new_backend = backend_mod.make_backend(new_info)
        except Exception as e:
            self._log_error(f"model swap failed: {e}")
            return False
        if self.agent is not None:
            self.agent._backend = new_backend
        self._session_backend = new_backend
        self._session_backend_info = new_info
        self._session_model = chosen_name
        if getattr(self, "_id", None) is not None:
            self.title = (
                f"{CHAT_APP_TITLE} — {new_info.name.upper()} · {chosen_name}"
            )
        return True

    def _parse_model_and_role(self, arg: str) -> tuple[str, str | None]:
        role = None
        # `--role critic` may follow the model token ("6 --role critic") or
        # stand alone at the start ("--role critic") when inheriting model 1.
        match = re.search(r"(?:^|\s+)--(?:role|r)\s+(\w+)", arg, re.IGNORECASE)
        if match:
            role = match.group(1).lower()
            arg = arg[:match.start()].strip()
        else:
            match_short = re.search(r"(?:^|\s+)-r\s+(\w+)", arg, re.IGNORECASE)
            if match_short:
                role = match_short.group(1).lower()
                arg = arg[:match_short.start()].strip()
            else:
                # Shorthand: /model2 --critic | /model3 --architect (inherit model 1)
                match_role_flag = re.search(
                    r"(?:^|\s+)--(critic|architect)\b", arg, re.IGNORECASE
                )
                if match_role_flag:
                    role = match_role_flag.group(1).lower()
                    arg = arg[:match_role_flag.start()].strip()
        return arg, role

    def _cmd_set_model(self, arg: str) -> None:
        """/model <N|name> — pick by global number from /list, or by substring.

        N is the unified-list index across both Ollama and MLX (the
        number printed in /list). Substring matches against any
        installed Ollama tag or downloaded MLX id; ambiguous matches
        require disambiguation. Bare /model clears the staged model.
        """
        self._cmd_set_model_slot(arg, 1)

    def _cmd_set_model2(self, arg: str) -> None:
        """/model2 <N|name> [--role critic|architect] — pick and configure role for Model 2."""
        self._cmd_set_model_slot(arg, 2)

    def _cmd_set_model3(self, arg: str) -> None:
        """/model3 <N|name> [--role critic|architect] — pick and configure role for Model 3."""
        self._cmd_set_model_slot(arg, 3)

    def _cmd_set_model_all(self, arg: str) -> None:
        """/modelall <N|name> — stage the SAME model into slots 1, 2, and 3.

        Convenience for the multi-slot Ollama topology (auto-pin on the
        4×48 GB workstation: 11434→GPU1, 11435→GPU2, 11436→GPU3). Useful
        when you want three identical model instances ready for:
          - parallel best-of-N candidate sampling
          - non-blocking critic on the architect's idle slot
          - any future fan-out work that benefits from identical capacity
            across all three GPUs

        Roles default to coder / critic / architect on slots 1 / 2 / 3
        via the same smart-default logic /model2 and /model3 use; pass
        per-slot `/model2` or `/model3 --role X` afterward to override.
        Bare /modelall clears all three staged slots.
        """
        if not arg.strip():
            # Clear all three slots — matches the bare /model behavior.
            for slot in (1, 2, 3):
                slot_str = "" if slot == 1 else str(slot)
                next_model_attr = f"_next_model{slot_str}"
                next_backend_attr = f"_next_backend{slot_str}"
                next_role_attr = f"_next_role{slot_str}"
                setattr(self, next_model_attr, None)
                setattr(self, next_backend_attr, None)
                setattr(self, next_role_attr, None)
            self._log_info(
                "cleared staged model on all 3 slots "
                "(usage: /modelall <number-from-/list-or-name>)"
            )
            self._update_status()
            return
        # Stage slot 1 first (no role flag — coder is the default).
        self._cmd_set_model_slot(arg, 1)
        # Slots 2 and 3 get explicit critic/architect roles so the
        # auto-staff side-effects (vlm-critique on, architect-split on)
        # fire deterministically regardless of how the smart-default
        # logic would have picked them.
        self._cmd_set_model_slot(f"{arg} --role critic", 2)
        self._cmd_set_model_slot(f"{arg} --role architect", 3)
        self._log_info(
            "[green]/modelall[/green] — same model staged on all 3 slots "
            "(slot 1: coder · slot 2: critic · slot 3: architect)"
        )

    def _cmd_set_model_slot(self, arg: str, slot: int) -> None:
        slot_str = "" if slot == 1 else str(slot)
        next_model_attr = f"_next_model{slot_str}"
        next_backend_attr = f"_next_backend{slot_str}"
        next_role_attr = f"_next_role{slot_str}"

        if not arg:
            current_val = getattr(self, next_model_attr)
            if current_val is None:
                self._log_info(
                    f"no staged model{slot_str} (usage: /model{slot_str} <number-from-/list-or-name> [--role critic|architect])"
                )
            else:
                self._log_info(f"cleared staged model{slot_str} (was: {current_val})")
                setattr(self, next_model_attr, None)
                setattr(self, next_backend_attr, None)
                setattr(self, next_role_attr, None)
            self._update_status()
            return

        arg, role = self._parse_model_and_role(arg)

        # Smart inheritance of model and backend from Model 1 if arg is empty but role is specified!
        if not arg and role is not None and slot > 1:
            chosen_name = self._next_model or self._session_model
            chosen_backend = self._next_backend or (self._session_backend_info.name if self._session_backend_info else None)
            if not chosen_name:
                self._log_error("cannot stage role without a primary model. Please stage/load Model 1 first.")
                return
        else:
            # Refresh the unified listing if /list hasn't been run yet.
            if not self._last_listing:
                self._refresh_listing()
            listing = self._last_listing
            if not listing:
                self._log_error(
                    "no LLM backend reachable — start ollama, or set MLX_MODEL / drop a model under ~/MLX_Models"
                )
                return

            chosen_backend = None
            chosen_name = None

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

        # Smart role default if not explicitly specified
        if slot > 1 and role is None:
            modality = backend_mod.classify_model_modality(chosen_name)
            base_default = "critic" if modality == "vlm" else "architect"
            if slot == 3:
                model2_role = self._next_role2 or self._session_role2
                if model2_role == "critic":
                    role = "architect"
                elif model2_role == "architect":
                    role = "critic" if modality == "vlm" else "architect"
                else:
                    role = base_default
            elif slot == 2:
                model3_role = self._next_role3 or self._session_role3
                if model3_role == "critic":
                    role = "architect"
                elif model3_role == "architect":
                    role = "critic" if modality == "vlm" else "architect"
                else:
                    role = base_default
            else:
                role = base_default

        if role is not None and role not in ("critic", "architect"):
            self._log_error(f"invalid role {role!r} — must be 'critic' or 'architect'")
            return

        setattr(self, next_backend_attr, chosen_backend)
        setattr(self, next_model_attr, chosen_name)
        setattr(self, next_role_attr, role)

        # Auto-staff architect only — critic screenshot review is explicit
        # via /vlm-critique (or /allroles), not silently enabled here.
        if slot > 1 and role == "architect" and not self._use_architect_split:
            self._use_architect_split = True
            self._architect_split_auto = True
            if self.agent is not None:
                self.agent._use_architect_split = True
            self._log_info(
                f"[dim]auto-enabled[/dim] [b]architect-split[/b] "
                f"[dim](role=architect on model{slot_str}; override: /architect off)[/dim]"
            )

        backend_label = chosen_backend.upper()
        role_suffix = f" [--role {role}]" if role else ""

        if self.agent is not None:
            if not self._apply_model_to_active_session_slot(
                chosen_backend, chosen_name, slot, role=role, source=f"/model{slot_str} hot-swap",
            ):
                return
            self.agent.set_step_mode(True)
            self.agent.set_auto_step_on_failure(True)
            self._run_profile = "local_manual"
            if self._is_streaming:
                self._log_info(
                    f"[green]switched session model{slot_str} to[/green] "
                    f"[b]{backend_label}[/b] · [b]{_esc(chosen_name)}[/b]{role_suffix} "
                    f"[dim](in-flight call finishes on the OLD model; "
                    f"next LLM call uses the new one; WAIT mode ON)[/dim]"
                )
            elif self._session_done:
                has_file = (
                    self._out_path is not None
                    and self._out_path.exists()
                    and self._out_path.stat().st_size > 200
                )
                cont = (
                    "Type feedback to continue this game (file + trace kept)."
                    if has_file
                    else f"Run [b]/retry[/b] to re-run planning with this model{slot_str}."
                )
                self._log_info(
                    f"[green]switched session model{slot_str} to[/green] "
                    f"[b]{backend_label}[/b] · [b]{_esc(chosen_name)}[/b]{role_suffix} "
                    f"[dim](session ended — {cont})[/dim]"
                )
            else:
                self._log_info(
                    f"[green]switched session model{slot_str} to[/green] "
                    f"[b]{backend_label}[/b] · [b]{_esc(chosen_name)}[/b]{role_suffix} "
                    f"[dim](next iter uses it; WAIT mode ON)[/dim]"
                )
            self._update_status()
            return

        self._run_profile = "local_manual"
        self._log_info(
            f"staged model{slot_str} [b]{backend_label}[/b] · [b]{_esc(chosen_name)}[/b]{role_suffix} "
            "for next /new session [dim](WAIT mode ON; no agent yet)[/dim]"
        )

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
        self._update_status()
        self._update_mode_bar()

    async def _cmd_retry(self, arg: str) -> None:
        """/retry — re-run after swapping off a bad model without /new.

        Keeps out_path, trace, snapshots, and (when present) the HTML on
        disk. Uses continuation when a working file exists; otherwise
        re-runs planning with the stored goal.
        """
        if self.agent is None:
            self._log_error("no agent — type a goal to start first")
            return
        if not self._session_done:
            self._log_error(
                "session still running — wait for it to finish, or "
                "[b]/model <N>[/b] mid-run to swap for the next LLM call"
            )
            return
        has_file = (
            self._out_path is not None
            and self._out_path.exists()
            and self._out_path.stat().st_size > 200
        )
        if has_file:
            feedback = (arg or "Continue from the current game file.").strip()
            await self._extend_session(feedback)
            return
        goal = (arg or self._goal or "").strip()
        if not goal:
            self._log_info(
                "usage: /retry [goal] — no stored goal; pass the game description"
            )
            return
        self._goal = goal
        self._session_done = False
        self._phase_label = "retry"
        self.sub_title = "agent is working (retry)"
        self._log_info(
            f"[yellow]retrying[/yellow] planning + build with "
            f"[b]{_esc(self._session_model or '?')}[/b] "
            f"[dim](trace + paths unchanged)[/dim]"
        )
        self.run_worker(self._consume_events(goal, continuation=False), exclusive=True)

    async def _cmd_new(self, arg: str) -> None:
        if self.agent is not None and not self._session_done:
            self._log_error(
                "a session is currently running — press Ctrl+D to ship it "
                "first, then /new <goal>"
            )
            return
        if not arg:
            await self._fresh_start()
            return
        await self._new_session(arg)

    def _cmd_games(self, arg: str) -> None:
        """/games — list curated prompts; /games <N> loads prompt #N into input.

        Loaded prompts go into the input box for review/edit; pressing Enter
        starts the build (the goal path handles a clean relaunch after a
        finished session). A running session is refused, mirroring /new.
        """
        from prompt_library import load_prompt_library

        try:
            games = load_prompt_library()
        except Exception as e:
            self._log_error(f"could not load prompt library: {e}")
            return
        if not games:
            self._log_info(
                "prompt library is empty — expected memory/prompt_library.jsonl"
            )
            return
        arg = (arg or "").strip()
        if not arg:
            self._log_info("[bold]Curated game prompts[/bold] — /games <N> to load:")
            for g in games:
                self._log(f"  [cyan]{g['n']:>2}[/cyan]  {_esc(g['title'])}")
            self._log_info(
                f"e.g. [b]/games 1[/b] loads prompt #1 into the input — "
                "press Enter to build, or edit it first."
            )
            return
        try:
            n = int(arg.split()[0])
        except ValueError:
            self._log_error(f"usage: /games <number 1-{len(games)}>")
            return
        match = next((g for g in games if g["n"] == n), None)
        if match is None:
            self._log_error(
                f"no prompt #{n} — /games lists the {len(games)} available"
            )
            return
        if self.agent is not None and not self._session_done:
            self._log_error(
                "a session is currently running — press Ctrl+D to ship it "
                f"first, then /games {n}"
            )
            return
        try:
            inp = self.query_one("#user-input", Input)
            inp.value = match["prompt"]
            inp.placeholder = (
                f"prompt #{n} ({match['title']}) loaded · press Enter to "
                "build, or edit first"
            )
            inp.focus()
            # Route the next Enter to a fresh build, not feedback/continuation.
            self._awaiting_kind = "goal"
            self._update_mode_bar()
            self._log_info(
                f"[green]loaded prompt #{n}: {_esc(match['title'])}[/green] — "
                "press Enter to build, or edit it first."
            )
        except Exception as e:
            self._log_error(f"could not load prompt into input: {e}")

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

    def _cmd_edit(self) -> None:
        """/edit — Asset Studio in the system browser (same as /open)."""
        import webbrowser
        from urllib.parse import quote

        try:
            _scripts = Path(__file__).resolve().parent / "scripts"
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            import asset_studio as _asset_studio  # noqa: WPS433

            url = _asset_studio.ensure_server(open_browser=False)
        except Exception as e:
            self._log_error(f"could not start Asset Studio server: {e}")
            return

        if self._assets_dir is not None and self._assets_dir.is_dir():
            url = f"{url.rstrip('/')}?dir={quote(str(self._assets_dir.resolve()))}"

        try:
            webbrowser.open(url)
            self._log_info(f"Asset Studio → {url}")
        except Exception as e:
            self._log_error(f"could not open browser: {e}")

    def _cmd_goodgame(self) -> None:
        """/goodgame — copy current session HTML + asset dirs into goodgame/."""
        if self._out_path is None:
            self._log_info("no session yet — build a game first, then /goodgame")
            return
        from goodgame import promote_session_game

        try:
            copied = promote_session_game(
                out_path=self._out_path,
                best_path=self._best_path,
                assets_dir=self._assets_dir,
                sounds_dir=self._sounds_dir,
            )
        except FileNotFoundError as e:
            self._log_error(str(e))
            return
        stem = self._out_path.stem
        self._log(f"[bold green]goodgame[/bold green] saved [b]{stem}[/b] → {copied['html']}")
        if not copied.get("assets") and not copied.get("sounds"):
            self._log_info(
                "[dim]no *_assets/ or *_sounds/ on disk (HTML only is fine)[/dim]"
            )

    def _cmd_revert(self, arg: str) -> None:
        """/revert [N] — roll the on-disk game file back to a previous iter.

        No-arg: most-recent clean iter (last `iter_summary` with ok=True),
                falling back to best.html if no clean iter snapshot exists.
        With N: revert to iter N specifically (snapshot file iter_NN.html).

        The iter counter does NOT reset — this rewinds the FILE, not the
        conversation. The model's next turn will see the reverted bytes
        as CURRENT FILE ON DISK and any feedback you type.

        Audit context: built 2026-05-25 after a 10-trace audit showed
        harness gates catch only ~5-10% of "model takes liberty beyond
        user scope" failures. Cheaper to give the user a one-keystroke
        rollback than to build more clever gates that don't catch most
        of the failure shape.
        """
        if self.agent is None:
            self._log_info("no session yet — start one before /revert")
            return
        requested = None
        if arg.strip():
            try:
                requested = int(arg.strip())
                if requested < 1:
                    raise ValueError
            except ValueError:
                self._log_info(
                    f"/revert takes a positive iter number; got {arg!r}. "
                    "Use /revert with no arg for the most-recent clean iter."
                )
                return
        try:
            result = self.agent.revert_to_iter(requested)
        except Exception as e:
            self._log_error(f"/revert failed: {e}")
            return
        if not result.get("ok"):
            self._log_error(f"/revert: {result.get('error', 'unknown error')}")
            return
        to_iter = result.get("to_iter")
        source = result.get("source", "?")
        from_iter = result.get("from_iter") or 0
        file_bytes = result.get("file_bytes", 0)
        if source == "snapshot":
            target_label = f"iter {to_iter}"
        else:
            target_label = "best.html"
        self._log(
            f"[bold green]reverted[/bold green] from iter {from_iter} "
            f"→ [b]{target_label}[/b] ({file_bytes:,} bytes). "
            "The on-disk file is now the working version. Type new "
            "feedback to continue."
        )

    def _cmd_unqueue(self, arg: str) -> None:
        """/unqueue — drop your last typed feedback before it is applied.

        Bare ``/unqueue`` removes ONLY the most recently typed feedback
        line (the accidental keystroke). Older queued feedback and any
        pending model-question answer are left alone. See also
        ``/unqueue all``, ``/unqueue answer``, ``/unqueue N``.
        """
        if self.agent is None:
            self._log_info("no session yet — nothing queued to remove")
            return
        which = arg.strip()
        try:
            result = self.agent.unqueue_pending_input(which)
        except Exception as e:
            self._log_error(f"/unqueue failed: {e}")
            return
        if not result.get("ok"):
            self._log_info(result.get("error", "nothing to unqueue"))
            return
        removed = result.get("removed") or []
        if not removed:
            self._log_info("queue already empty")
            return
        for item in removed:
            kind = item.get("kind", "feedback")
            preview = item.get("preview", "")
            label = "answer" if kind == "answer" else "feedback"
            self._log(
                f"[bold yellow]unqueued[/bold yellow] {label}: "
                f"[dim]{_esc(preview)}[/dim]"
            )
        remain_fb = int(result.get("remaining_feedback") or 0)
        remain_ans = bool(result.get("remaining_answer"))
        which = str(result.get("which") or "")
        if which == "last_feedback" and remain_fb:
            self._log_info(
                f"{remain_fb} older feedback item{'s' if remain_fb != 1 else ''} "
                "still queued for next user-turn"
            )
        elif which == "last_feedback" and remain_ans:
            self._log_info(
                "queued model-question answer left unchanged "
                "(use /unqueue answer to drop it)"
            )
        elif which == "last_feedback":
            self._log_info("no feedback queued for next user-turn")
        else:
            bits: list[str] = []
            if remain_fb:
                bits.append(
                    f"{remain_fb} feedback item{'s' if remain_fb != 1 else ''}"
                )
            if remain_ans:
                bits.append("1 answer")
            if bits:
                self._log_info(f"still queued: {' + '.join(bits)}")
            else:
                self._log_info("queue empty — nothing pending for next user-turn")
        self._update_status()

    def _cmd_attach_ref_image(self, arg: str) -> None:
        """/ref <path> — attach a reference image to the NEXT user turn.

        Use case: "make the game look like this." The image is loaded
        from disk, validated as PNG/JPEG/WebP, and stashed on the agent
        as `_next_image_bytes`. When the next user message goes out
        AND the active model is a VLM, the existing image-attachment
        path ([agent.py:_stream](agent.py) ~line 2785) will pair them.

        Notes:
          - This only works when the active model is a VLM. On a
            text-only model the image is dropped and the user sees a
            warning.
          - Pasting binary into a terminal Input field isn't possible —
            the user provides a path (drag the file from Finder into
            the terminal, or copy a path via Cmd+Option+C).
          - The image clears after one use (single-shot). Re-run /ref
            for each turn that needs a reference.
        """
        path_str = (arg or "").strip().strip('"\'')
        if not path_str:
            self._log_info(
                "usage: /ref <path/to/image.png>  "
                "(then type 'make the game look like this' on the next line)"
            )
            return
        path = Path(path_str).expanduser()
        if not path.exists():
            self._log_error(f"/ref: file not found: {path}")
            return
        if not path.is_file():
            self._log_error(f"/ref: not a regular file: {path}")
            return
        try:
            data = path.read_bytes()
        except Exception as e:
            self._log_error(f"/ref: could not read file: {e}")
            return
        # Cheap magic-byte sniff. PNG = 89 50 4E 47, JPEG = FF D8 FF,
        # WebP = "RIFF....WEBP". Reject anything else so we don't
        # waste a VLM turn on a corrupt or non-image file.
        head = data[:12]
        is_png = head.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = head.startswith(b"\xff\xd8\xff")
        is_webp = head[:4] == b"RIFF" and head[8:12] == b"WEBP"
        if not (is_png or is_jpeg or is_webp):
            self._log_error(
                f"/ref: {path.name} is not PNG / JPEG / WebP (got "
                f"magic bytes {head[:4].hex()}). Convert and retry."
            )
            return
        # Cap at 4 MB so a 50 MB scan doesn't blow up the prompt
        # tokens (the backend will resize, but the bytes still travel
        # through the chat-template render).
        if len(data) > 4 * 1024 * 1024:
            self._log_error(
                f"/ref: {path.name} is {len(data) // 1024} KB — too large "
                "(cap is 4 MB). Resize to ~1024px max and retry."
            )
            return
        kind = (
            "PNG" if is_png else ("JPEG" if is_jpeg else "WebP")
        )
        # Active session -> attach directly to the next user turn.
        if self.agent is not None:
            self.agent._next_image_bytes = data
            # Surface a hint if the active model is text-only — the bytes
            # will be ignored in that case.
            is_vlm = bool(getattr(self.agent, "_is_vlm", False))
            vlm_hint = (
                "" if is_vlm
                else " [yellow](active model is text-only — image may be ignored; "
                     "/load a VLM first)[/yellow]"
            )
            self._log_info(
                f"/ref: attached {path.name} ({kind}, {len(data) // 1024} KB) "
                f"to the next user turn.{vlm_hint}"
            )
            return
        # No active agent yet -> stage for the first turn of the next session.
        self._staged_ref_image_bytes = data
        self._staged_ref_image_name = path.name
        self._log_info(
            f"/ref: staged {path.name} ({kind}, {len(data) // 1024} KB) for "
            "the first turn of the next session. Start with /new <goal> (or type a goal)."
        )

    def _cmd_set_iters(self, arg: str) -> None:
        if not arg.isdigit() or int(arg) <= 0:
            self._log_info(f"usage: /iters <positive int>  (current: {self._max_iters})")
            return
        self._max_iters = int(arg)
        self._log_info(
            f"max iterations set to [b]{self._max_iters}[/b] for next session/extension"
        )

    def _cmd_set_ctx(self, arg: str) -> None:
        """/ctx [N|100k|131k|262k|full] — Ollama num_ctx (KV reservation)."""
        if not arg:
            self._log_info(
                f"Ollama context: [b]{self._num_ctx:,}[/b] tokens "
                f"(default {DEFAULT_NUM_CTX:,}). "
                "usage: /ctx <N|100k|131k|262k|full|native>"
            )
            if self.agent is not None:
                self._log_info(
                    f"[dim]active session agent.num_ctx={self.agent.num_ctx:,}[/dim]"
                )
            return
        try:
            new_ctx = parse_num_ctx_arg(arg)
        except ValueError as e:
            self._log_error(str(e))
            self._log_info(
                "usage: /ctx <N|100k|131k|262k|full>  "
                f"(range {MIN_NUM_CTX:,}–{MAX_NUM_CTX:,})"
            )
            return
        old = self._num_ctx
        self._num_ctx = new_ctx
        self._ctx_max = new_ctx
        if self.agent is not None:
            self.agent.num_ctx = new_ctx
        self._log_info(
            f"Ollama context set to [b]{new_ctx:,}[/b] "
            f"(was {old:,}) for next turns"
        )
        if (
            self._session_backend_info is not None
            and self._session_backend_info.name == "ollama"
        ):
            self._log_info(
                "[dim]Ollama will reload the model on the next request when "
                "num_ctx changes. Preload to avoid a stall: "
                f"`ollama run --ctx-size {new_ctx} "
                f"{self._session_model or '<model>'}`[/dim]"
            )
        elif self.agent is not None:
            self._log_info(
                "[dim]num_ctx applies to Ollama backends; MLX ignores it.[/dim]"
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

    async def _cmd_check(self, arg: str) -> None:
        """/check [<N|model>] — visual check + guidance injection.

        Looks at the most recent screenshot of your game with a vision
        model and answers two things: did the last iteration make
        VISIBLE progress toward your goal, and what's still visibly
        wrong. This is the "third signal" the harness otherwise lacks
        (probes only check structure, not whether the game LOOKS like
        what you asked for).

        Calls a reviewer model exactly when you type the command. This
        does NOT change the coding model; it only produces guidance.

        Simpler default behavior:
          - wait OFF: verdict is auto-queued into the next coding turn.
          - wait ON: suggested feedback is inserted into the input box
            so you can edit it or press Enter as-is.

        Back-compat: `--apply` still forces direct queueing.

        Vendor routing (any vision-capable model works):
          - `claude` / `sonnet` / `opus` / `haiku` / `claude-*` →
            Anthropic (needs ANTHROPIC_API_KEY)
          - `gpt` / `gpt-5` / `gpt-4o` / `o3` / `o4-mini` →
            OpenAI (needs OPENAI_API_KEY)
          - anything else → resolved against your local MLX VLMs
        With no argument, uses the active session model IF it's a VLM.
        """
        # Parse argument forms:
        #   (no arg)       -> active VLM session model
        #   "<N>"          -> /list row number (e.g. /check 17)
        #   "with <model>" -> tolerated legacy form
        #   "<model>"      -> explicit model string
        # Optional flag (back-compat):
        #   --apply        -> force queueing even when wait is ON
        apply_requested = False
        stripped_full = (arg or "").strip()
        if stripped_full:
            tokens = stripped_full.split()
            kept: list[str] = []
            for tok in tokens:
                if tok == "--apply":
                    apply_requested = True
                else:
                    kept.append(tok)
            arg = " ".join(kept).strip()
        selected_backend: str | None = None
        agent = getattr(self, "agent", None)
        if not arg:
            # No arg: prefer the active session model when it's a VLM.
            # Falls back to a usage hint when there's nothing usable.
            if agent is not None and bool(getattr(agent, "_is_vlm", False)):
                model = getattr(agent, "model", None) or ""
                if not model:
                    self._log_info(
                        "usage: /check [<N|model>]   (no active model)"
                    )
                    return
                self._log_info(
                    f"[magenta]/check[/magenta] using active session "
                    f"VLM: [b]{_esc(model)}[/b]"
                )
            else:
                self._log_info("usage: /check <N|model>")
                self._log_info(
                    "  by number: /check 17   (uses model #17 from /list)"
                )
                self._log_info(
                    "  by name: /check claude   /check gpt-5   /check <mlx-vlm-substring>"
                )
                return
        else:
            # Accept both `with <model>` and bare `<model>` shorthand.
            stripped = arg.strip()
            if stripped.lower().startswith("with "):
                model = stripped[5:].strip()
            else:
                model = stripped
            # Numeric shorthand: resolve via /list snapshot.
            if model.isdigit():
                model_num = model
                selected_backend, resolved = self._resolve_listing_arg(model)
                if resolved is None:
                    return
                model = resolved
                if selected_backend == "ollama":
                    self._log_error(
                        "/check by number only supports vision-capable reviewer "
                        "entries (MLX/OpenAI/Claude). The selected /list row is "
                        "an Ollama entry."
                    )
                    return
                self._log_info(
                    f"[magenta]/check[/magenta] model #{_esc(model_num)} → "
                    f"[b]{_esc(model)}[/b]"
                )
        aliases = {
            "claude": "claude-sonnet-4-6",
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-8",
            "fable": "claude-fable-5",
            "haiku": "claude-haiku-4-5",
            "gpt": "gpt-5",
            "openai": "gpt-5",
            "gpt5": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt5-mini": "gpt-5-mini",
        }
        model = aliases.get(model.lower(), model)
        # Vendor routing — explicit and ordered. Anything not matched
        # falls through to the local-MLX-VLM resolver.
        try:
            from vision_judge import _cloud_vendor, _resolve_local_mlx_vlm
        except Exception as e:
            self._log_error(f"/check: vision_judge unavailable — {e}")
            return
        vendor = _cloud_vendor(model)
        if vendor == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                self._log_error(
                    "/check: ANTHROPIC_API_KEY not set. Add it to .env "
                    "or your shell env, OR use a local VLM "
                    "([b]/check <name-or-number>[/b] — run /list and look "
                    "for [magenta]\\[VLM][/magenta]), OR use OpenAI "
                    "([b]/check gpt-5[/b]) if OPENAI_API_KEY is set."
                )
                return
        elif vendor == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                self._log_error(
                    "/check: OPENAI_API_KEY not set. Add it to .env "
                    "or your shell env, OR use a local VLM "
                    "([b]/check <name-or-number>[/b] — run /list and look "
                    "for [magenta]\\[VLM][/magenta]), OR use Anthropic "
                    "([b]/check claude[/b]) if ANTHROPIC_API_KEY "
                    "is set."
                )
                return
        else:
            # Local-MLX-VLM resolver. If the user typed the EXACT name
            # of the active session model and it's a VLM, accept it
            # verbatim (covers the case where the user wants to use
            # their currently-loaded VLM by full name).
            active_model = (
                getattr(agent, "model", None) if agent is not None else None
            )
            if (
                active_model
                and model == active_model
                and bool(getattr(agent, "_is_vlm", False))
            ):
                self._log_info(
                    f"[magenta]/check[/magenta] using active session "
                    f"VLM: [b]{_esc(model)}[/b]"
                )
            else:
                resolved = _resolve_local_mlx_vlm(model)
                if resolved is None:
                    self._log_error(
                        f"/check: {_esc(model)!r} didn't match a cloud "
                        "vendor (claude / gpt / o*) or any local MLX "
                        "VLM. Run /list to see local VLMs (look for "
                        "[magenta]\\[VLM][/magenta]). Cloud examples: "
                        "[b]/check claude[/b]  or  "
                        "[b]/check gpt-5[/b]."
                    )
                    return
                self._log_info(
                    f"[magenta]/check[/magenta] resolved local VLM: "
                    f"[b]{_esc(resolved)}[/b]"
                )
                model = resolved  # vision_judge picks the path back up
        if backend_mod.classify_model_modality(model) != "vlm":
            self._log_error(
                f"/check: {_esc(model)!r} is not a vision model. Pick a "
                "[magenta]\\[VLM][/magenta] entry from /list."
            )
            return
        agent = getattr(self, "agent", None)
        if agent is None:
            self._log_error(
                "/check needs an active agent — start a session first "
                "with /new <goal>."
            )
            return
        png = (
            getattr(agent, "_last_screenshot_after", None)
            or getattr(agent, "_prev_judge_png", None)
        )
        if not png:
            self._log_error(
                "/check: no screenshot yet. Run at least one iteration "
                "first so there's something to look at."
            )
            return
        goal = getattr(agent, "_goal", None)
        if not goal:
            self._log_error(
                "/check: no active goal. Start a session with /new <goal>."
            )
            return
        step_on = bool(getattr(agent, "_step_mode", False))
        apply_verdict = apply_requested or (not step_on)
        offer_input_suggestion = (not apply_requested) and step_on
        self._log_info(
            f"[magenta]/check[/magenta] calling [b]{_esc(model)}[/b] (one API call)…"
        )
        await self._run_visual_check(
            model=model,
            goal=goal,
            png=png,
            agent=agent,
            apply_verdict=apply_verdict,
            offer_input_suggestion=offer_input_suggestion,
            source="slash_check",
        )

    def _agent_is_streaming(self) -> bool:
        """True when any model slot is mid-stream."""
        return bool(
            self._is_streaming
            or self._model2_is_streaming
            or self._model3_is_streaming
        )

    async def _cmd_ask(self, arg: str) -> None:
        """/ask <question> — read-only Q&A; does not queue feedback or patch code."""
        question = (arg or "").strip()
        if not question:
            self._log_info(
                "usage: /ask <question>   "
                "(e.g. /ask how does digging at skull beach work?)"
            )
            return
        agent = self.agent
        if agent is None:
            self._log_error(
                "/ask needs an active session — start with a goal or /new <goal>"
            )
            return
        if self._agent_is_streaming():
            self._log_error(
                "/ask: wait for the current model turn to finish, then retry"
            )
            return
        self._log_info(f"[cyan]/ask[/cyan] {_esc(question)}")
        try:
            async for ev in agent.run_ask_turn(question):
                self._handle_event(ev)
        except Exception as e:
            self._log_error(f"/ask failed: {e}")

    async def _run_visual_check(
        self,
        *,
        model: str,
        goal: str,
        png: bytes,
        agent: GameAgent,
        apply_verdict: bool,
        offer_input_suggestion: bool = False,
        source: str,
    ) -> bool:
        """Run one visual check and optionally queue coaching."""
        try:
            from vision_judge import judge_visual_progress, _cloud_vendor
        except Exception as e:
            self._log_error(f"/check: vision_judge unavailable — {e}")
            return False
        vendor = _cloud_vendor(model)
        verdict = await judge_visual_progress(
            goal=goal,
            current_png=png,
            previous_png=None,
            model=model,
        )
        if verdict is None:
            self._log_error(
                "/check: model returned nothing (API down, timeout, or rejected). "
                "No state was changed."
            )
            return False
        if verdict.progress is True:
            prog = "[green]progress[/green]"
            progress_label = "yes"
        elif verdict.progress is False:
            prog = "[red]no progress[/red]"
            progress_label = "no"
        else:
            prog = "[yellow]unclear[/yellow]"
            progress_label = "unclear"
        self._log(f"[magenta]{_esc(model)}:[/magenta] {prog}")
        note = (verdict.note or "").strip()
        self._log(f"  missing: {_esc(note) if note else '(no note)'}")
        coach = (
            f"EXTERNAL VISUAL REVIEW ({model}): progress={progress_label}. "
            f"Most important visible gap: {note or 'unspecified by reviewer'}. "
            "Address this in the next iteration before shipping."
        )
        applied = False
        if apply_verdict:
            pending = getattr(agent, "_pending_coaching", None)
            if isinstance(pending, list):
                pending.append(coach)
                applied = True
            else:
                agent.add_user_feedback(coach)
                applied = True
            self._log_info(
                "[green]review verdict queued[/green] for the next coding turn "
                "(via agent coaching)"
            )
        elif offer_input_suggestion:
            try:
                inp = self.query_one("#user-input", Input)
                inp.value = coach
                inp.placeholder = "review suggestion loaded · edit it or press Enter to submit"
                inp.focus()
                self._awaiting_kind = "feedback"
                self._log_info(
                    "[green]review suggestion loaded into input[/green] — "
                    "edit it, or press Enter to send as-is."
                )
            except Exception:
                self._log_info(
                    "[yellow]review suggestion ready[/yellow] but input focus "
                    "failed; copy from log and send manually."
                )
        else:
            self._log_info(
                "(verdict shown only; no injection requested)"
            )
        # Trace the explicit reviewer action so tune/forensics can measure
        # cloud/local review impact on iteration outcomes.
        trace_fn = getattr(agent, "_trace", None)
        if callable(trace_fn):
            try:
                trace_fn({
                    "kind": "manual_visual_check",
                    "source": source,
                    "model": model,
                    "vendor": vendor or "local",
                    "applied": applied,
                    "progress": verdict.progress,
                    "note": note,
                })
            except Exception:
                pass
        return True

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
            f"staged seed for next session: [b]{_esc(str(self._next_seed))}[/b] "
            f"[dim]({size:,} bytes) — type your request, or /new <goal>[/dim]"
        )

    def _clear_staged_state(self) -> list[str]:
        """Wipe game staging for bare /new; model/backend slots stay sticky."""
        had_seed = self._next_seed
        had_iters = self._max_iters
        had_ctx = self._num_ctx
        had_restarts = self._restart_n
        had_class = self._model_class
        had_ref = self._staged_ref_image_name
        had_allroles = self._all_roles_enabled
        self._next_seed = None
        self._staged_ref_image_bytes = None
        self._staged_ref_image_name = None
        self._max_iters = 6
        self._num_ctx = default_num_ctx()
        self._ctx_max = self._num_ctx
        self._restart_n = 2
        self._model_class = None
        self._run_profile = "custom"
        self._profile_review_model = None
        self._profile_review_auto_apply = False
        if had_allroles:
            self._all_roles_enabled = False
            self._use_architect_split = False
            self._use_vlm_critique = False
            self._architect_split_auto = False
            self._vlm_critique_auto = False
            if self.agent is not None:
                self.agent._use_architect_split = False
                self.agent._use_vlm_critique = False
                self.agent._architect_split_auto = False
                self.agent._vlm_critique_auto = False
        bits: list[str] = []
        if had_seed is not None:
            bits.append(f"seed={had_seed}")
        if had_iters != 6:
            bits.append(f"iters={had_iters}→6")
        if had_ctx != default_num_ctx():
            bits.append(f"ctx={had_ctx}→{self._num_ctx}")
        if had_restarts != 2:
            bits.append(f"restarts={had_restarts}→2")
        if had_class:
            bits.append(f"model-class={had_class}→auto")
        if had_ref:
            bits.append(f"ref={had_ref}→cleared")
        if had_allroles:
            bits.append("allroles=on→off")
        return bits

    async def _fresh_start(self) -> None:
        """Drop the finished session, clear game staging, wait for the next goal."""
        bits = self._clear_staged_state()
        self.agent = None
        self._session_done = True
        self._session_seed = None
        self._goal = ""
        self._out_path = None
        self._best_path = None
        self._trace_path = None
        self._feedback_ledger = []
        self._last_test_block = ""
        self._phase_label = "ready"
        self._iteration_label = "—"
        self._awaiting_kind = "goal"
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
            self._log_file_path = None
        if self.browser is not None:
            try:
                await self.browser.show_status(
                    "Ready for a new game",
                    "Type your game idea below and press Enter.",
                )
            except Exception:
                pass
        self.query_one("#log-pane", RichLog).clear()
        self._log_mirror_lines = []
        self._status_plain = ""
        inp = self.query_one("#user-input", Input)
        inp.placeholder = "describe your game and press Enter"
        inp.focus()
        self.sub_title = "ready — type a game goal"
        self._update_mode_bar()
        self._update_status()
        if bits:
            self._log_info(
                f"[yellow]cleared staging:[/yellow] {', '.join(bits)}"
            )
        self._log_info(
            "[bold green]fresh start[/bold green] — type your game idea and press Enter. "
            "[dim](/games lists curated prompts; /seed only if continuing from old HTML)[/dim]"
        )

    async def _cmd_reset(self) -> None:
        """Full clean slate — same as bare /new."""
        if self.agent is not None and not self._session_done:
            self._log_error(
                "a session is currently running — press Ctrl+D to ship it first"
            )
            return
        await self._fresh_start()

    def _cmd_status(self) -> None:
        # Step-mode shows up here so users can confirm whether the
        # agent will pause between iters or run continuously.
        step_label = "ON" if self._effective_step_mode() else "off"
        # Prefer the live agent's feature flags when a session is
        # running — the agent can AUTO-staff vlm-critique on local-VLM
        # detection (agent.py:5355) and architect-split on complex
        # first-builds without touching the App's flags. Without this
        # `/status` shows the stale chat-level value while the right-
        # side status panel (which already reads from the agent)
        # correctly shows "vlm-critique ON".
        if self.agent is not None:
            eff_vlm_critique = bool(getattr(self.agent, "_use_vlm_critique", self._use_vlm_critique))
            eff_vlm_auto = bool(getattr(self.agent, "_vlm_critique_auto", self._vlm_critique_auto))
            eff_arch_split = bool(getattr(self.agent, "_use_architect_split", self._use_architect_split))
            eff_arch_auto = bool(getattr(self.agent, "_architect_split_auto", self._architect_split_auto))
        else:
            eff_vlm_critique = self._use_vlm_critique
            eff_vlm_auto = self._vlm_critique_auto
            eff_arch_split = self._use_architect_split
            eff_arch_auto = self._architect_split_auto
        lines = [
            "[bold cyan]── status ──[/bold cyan]",
            f"  backend (active):     {_esc(self._session_backend_info.name if self._session_backend_info else '—')}",
            f"  backend (next /new):  {_esc(self._next_backend or '(auto)')}",
            f"  model (active):       {_esc(self._session_model or '—')}",
            f"  model (next /new):    {_esc(self._next_model or '(auto-detect)')}",
        ]
        
        if self._session_backend_info2 or self._next_model2:
            lines.append(f"  model2 (active):      {_esc(self._session_model2 or '—')} [dim]({self._session_role2 or 'critic'})[/dim]")
            lines.append(f"  model2 (next /new):   {_esc(self._next_model2 or '—')} [dim]({self._next_role2 or 'critic'})[/dim]")
            
        if self._session_backend_info3 or self._next_model3:
            lines.append(f"  model3 (active):      {_esc(self._session_model3 or '—')} [dim]({self._session_role3 or 'architect'})[/dim]")
            lines.append(f"  model3 (next /new):   {_esc(self._next_model3 or '—')} [dim]({self._next_role3 or 'architect'})[/dim]")

        lines.extend([
            f"  goal:                 {_esc(self._goal or '—')}",
            f"  phase:                {_esc(self._phase_label)}",
            f"  iteration:            {_esc(self._iteration_label)}",
            f"  max iters:            {self._max_iters}",
            f"  Ollama ctx (next):    {self._num_ctx:,}",
            f"  restart-N:            {self._restart_n if self._restart_n > 1 else '1 (off)'}",
            f"  model-class:          {self._model_class or 'auto (= small, lean ~5KB schema)'}",
            f"  step-mode (/wait):    {step_label}",
            f"  prefill:              {'ON' if self._use_prefill else 'off'}",
            f"  architect-split:      {'ON' if eff_arch_split else 'off'}{' [auto]' if eff_arch_auto and eff_arch_split else ''}",
            f"  double-screenshot:    {'ON' if self._use_double_screenshot else 'off'}",
            f"  vlm-critique (vision):{'ON' if eff_vlm_critique else 'off'}{' [auto]' if eff_vlm_auto and eff_vlm_critique else ''}  [dim](looks at the screen; sends problems to the agent)[/dim]",
            f"  /allroles bundle:     {'ON' if self._all_roles_enabled else 'off'}",
            f"  critique (no vision): {'ON' if self._use_autonomous_feedback else 'OFF'}  [dim](reviews without looking; sends problems to the agent; /critique off to disable)[/dim]",
            f"  raw user feedback:    {'ON · directives suppressed (default)' if not self._use_feedback_directives else 'off · classifier wrapping ACTIVE'}  [dim](/rawfeedback on|off — bypass MEDIA-CHANGE / ORIENTATION / SCOPE wrappers; machine bug feedback always on)[/dim]",
            f"  iter detail:          {self._iter_decision_verbose}",
            f"  run profile:          {self._format_run_profile()}",
            f"  review hook:          {self._profile_review_model or '—'}",
            f"  review auto-apply:    {self._profile_review_auto_apply}",
            f"  seed in use:          {_esc(str(self._session_seed) if self._session_seed else '—')}",
            f"  staged seed:          {_esc(str(self._next_seed) if self._next_seed else '—')}",
            f"  staged /ref image:    {_esc(self._staged_ref_image_name or '—')}",
            f"  session done:         {self._session_done}",
            f"  game file:            {self._out_path or '—'}",
            f"  log file:             {self._log_file_path or '—'}",
        ])
        for line in lines:
            self._log(line)

    def _cmd_iter_detail(self, arg: str) -> None:
        """/iter-detail [on|off] — toggle optional expanded iter blocker info."""
        a = (arg or "").strip().lower()
        if not a:
            self._iter_decision_verbose = not self._iter_decision_verbose
        elif a in {"on", "true", "1"}:
            self._iter_decision_verbose = True
        elif a in {"off", "false", "0"}:
            self._iter_decision_verbose = False
        else:
            self._log_info("usage: /iter-detail [on|off]")
            return
        state = "ON" if self._iter_decision_verbose else "off"
        self._log_info(f"iter decision detail → [b]{state}[/b]")
        self._update_status()

    def _cmd_set_mode(self, arg: str) -> None:
        """/mode — apply a run profile.

        Profiles:
          - local_manual: local-first + wait mode ON
          - local_auto: local-first + wait mode OFF
          - local_plus_review with <model> [--auto-apply]:
                local-first + wait mode OFF + reviewer hook model
                (auto-apply only runs when NOT in wait mode)
          - custom: clear profile and keep manual command control
        """
        raw = (arg or "").strip()
        if not raw:
            self._log_info(
                "usage: /mode <local_manual|local_auto|local_plus_review with <model> [--auto-apply]|custom>"
            )
            self._log_info(
                f"current profile: [b]{self._format_run_profile()}[/b] · "
                f"review model: [b]{_esc(self._profile_review_model or '—')}[/b] · "
                f"auto-apply: [b]{self._profile_review_auto_apply}[/b]"
            )
            return
        parts = raw.split()
        profile = parts[0].lower()
        rest = parts[1:]
        if profile == "custom":
            self._run_profile = "custom"
            self._profile_review_model = None
            self._profile_review_auto_apply = False
            self._log_info("run profile → [b]custom[/b] (manual /wait, /backend, /check workflow)")
            self._update_status()
            self._update_mode_bar()
            return
        if profile not in {"local_manual", "local_auto", "local_plus_review"}:
            self._log_error(
                "unknown /mode profile. Use local_manual, local_auto, local_plus_review, or custom."
            )
            return

        # Local-first contract: clear staged cloud backend so next /new
        # resolves through the local auto policy.
        if self._next_backend in {"openai", "anthropic"}:
            self._next_backend = None
            self._next_model = None
            self._log_info(
                "[yellow]cleared staged cloud backend/model[/yellow] to honor local-first mode"
            )

        if profile == "local_plus_review":
            auto_apply = False
            cleaned: list[str] = []
            for tok in rest:
                if tok == "--auto-apply":
                    auto_apply = True
                else:
                    cleaned.append(tok)
            if cleaned and cleaned[0].lower() == "with":
                cleaned = cleaned[1:]
            review_model = " ".join(cleaned).strip()
            if not review_model:
                self._log_error(
                    "/mode local_plus_review needs a model. Example: "
                    "/mode local_plus_review with gpt-5 --auto-apply"
                )
                return
            self._profile_review_model = review_model
            self._profile_review_auto_apply = auto_apply
        else:
            self._profile_review_model = None
            self._profile_review_auto_apply = False

        self._run_profile = profile
        if self.agent is not None:
            if profile == "local_manual":
                self.agent.set_step_mode(True)
                self.agent.set_auto_step_on_failure(True)
            else:
                self.agent.set_step_mode(False)
                # Keep AUTO profiles uninterrupted: no surprise
                # auto-pause when an iteration fails.
                self.agent.set_auto_step_on_failure(False)
        mode_bits = [f"profile → [b]{profile}[/b]"]
        if profile == "local_plus_review":
            mode_bits.append(f"review model: [b]{_esc(self._profile_review_model or '')}[/b]")
            mode_bits.append(
                "auto-apply ON (AUTO mode only)" if self._profile_review_auto_apply
                else "auto-apply OFF (manual apply)"
            )
            try:
                from vision_judge import _cloud_vendor
                vendor = _cloud_vendor(self._profile_review_model or "")
            except Exception:
                vendor = ""
            if vendor:
                mode_bits.append("[yellow]explicit cloud review can incur cost[/yellow]")
        elif profile == "local_manual":
            mode_bits.append("wait mode ON")
        elif profile == "local_auto":
            mode_bits.append("wait mode OFF")
        self._log_info(" · ".join(mode_bits))
        self._update_status()
        self._update_mode_bar()

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
        if self._run_profile == "local_manual" and not new_state:
            self._run_profile = "custom"
            self._log_info("[dim]run profile moved to custom (manual /wait override)[/dim]")
        elif self._run_profile == "local_auto" and new_state:
            self._run_profile = "custom"
            self._log_info("[dim]run profile moved to custom (manual /wait override)[/dim]")
        self.agent.set_step_mode(new_state)
        # Mirror explicit /wait intent for auto-step behavior: when the
        # user turns wait OFF, don't auto-re-enable pauses later.
        self.agent.set_auto_step_on_failure(new_state)
        # Auto-toggle vlm-critique with wait mode. When the user is
        # reviewing each iter themselves, the auto critic is pure
        # noise (mortal-kombat 2026-05-24 trace: 6 paraphrased
        # complaints in 6 iters about the same visual issue). Save
        # the current state on /wait on and restore on /wait off.
        # User can still flip /vlm-critique explicitly mid-wait —
        # that clears the saved state so the auto-restore doesn't
        # override their choice.
        if new_state:
            # Entering wait mode.
            if self._vlm_critique_pre_wait is None and self._use_vlm_critique:
                self._vlm_critique_pre_wait = True
                self._use_vlm_critique = False
                self._vlm_critique_auto = False
                if self.agent is not None:
                    self.agent._use_vlm_critique = False
                    self.agent._vlm_critique_auto = False
                self._log_info(
                    "[dim]vlm-critique auto-off while in wait mode — "
                    "you're the visual critic now. Toggle /vlm-critique "
                    "explicitly if you want the auto critic back; it'll "
                    "restore to ON when you /wait off.[/dim]"
                )
        else:
            # Leaving wait mode — restore prior vlm-critique state.
            if self._vlm_critique_pre_wait is not None:
                prior = self._vlm_critique_pre_wait
                self._vlm_critique_pre_wait = None
                if self._use_vlm_critique != prior:
                    self._use_vlm_critique = prior
                    if self.agent is not None:
                        self.agent._use_vlm_critique = prior
                    state_word = "ON" if prior else "off"
                    self._log_info(
                        f"[dim]vlm-critique restored to {state_word} "
                        "(was saved when you entered wait mode).[/dim]"
                    )
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

    def _cmd_toggle_prefill(self, arg: str) -> None:
        """/prefill [on|off] — toggle assistant prefill force (<plan> / <diagnose> tags)."""
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        else:
            new_state = not self._use_prefill
        self._use_prefill = new_state
        if self.agent is not None:
            self.agent._use_prefill = new_state
        status = "[green]ON[/green]" if new_state else "[yellow]OFF[/yellow]"
        self._log_info(f"assistant prefill set to {status}")
        self._update_status()

    def _cmd_toggle_architect(self, arg: str) -> None:
        """/architect [on|off] — toggle architect/editor split (Aider's 2-call pattern)."""
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        else:
            new_state = not self._use_architect_split
        self._use_architect_split = new_state
        # Explicit toggle clears auto-staff so a user "off" sticks.
        self._architect_split_auto = False
        if self.agent is not None:
            self.agent._use_architect_split = new_state
            self.agent._architect_split_auto = False
        status = "[green]ON[/green]" if new_state else "[yellow]OFF[/yellow]"
        self._log_info(f"architect split set to {status} (doubles plan-turn time)")
        self._update_status()

    def _cmd_toggle_double_screenshot(self, arg: str) -> None:
        """/double-screenshot [on|off] — toggle dual-screenshot capturing (startup and after input)."""
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        else:
            new_state = not self._use_double_screenshot
        self._use_double_screenshot = new_state
        if self.agent is not None:
            self.agent._use_double_screenshot = new_state
        status = "[green]ON[/green]" if new_state else "[yellow]OFF[/yellow]"
        self._log_info(f"double screenshot capturing set to {status}")
        self._update_status()

    def _cmd_toggle_autonomous_feedback(self, arg: str) -> None:
        """/critique [on|off] — the review that does NOT look at the screen.

        Primary name: /critique. Silent aliases: /playtest, /feedback.
        When ON (default): after each clean round the agent reviews the game
        WITHOUT vision — it plays it (scripted browser input from
        memory/playtests.jsonl) and reads the test report, then queues any
        problems back to the coder so it fixes them next round.

        For the review that DOES look at the screen, use /vlm-critique.
        When OFF: harness, your notes, and the vision critique are unchanged.
        Sticky across /new.
        """
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        elif arg_lc == "":
            # Show current state instead of flipping when called bare.
            cur = "[green]ON[/green]" if self._use_autonomous_feedback else "[yellow]OFF[/yellow]"
            self._log_info(
                f"critique (no vision) is {cur} "
                "(usage: /critique on  ·  /critique off)"
            )
            return
        else:
            new_state = not self._use_autonomous_feedback
        self._use_autonomous_feedback = new_state
        if self.agent is not None:
            self.agent._use_autonomous_feedback = new_state
        status = "[green]ON[/green]" if new_state else "[yellow]OFF[/yellow]"
        self._log_info(
            f"critique (no vision) set to {status} "
            "(reviews without looking at the screen; sends problems to the agent — "
            "use /vlm-critique to also look)"
        )
        self._update_status()

    def _cmd_toggle_raw_feedback(self, arg: str) -> None:
        """/rawfeedback [on|off] — your typed feedback goes to the model verbatim.

        Default ON: every directive block is suppressed. The model
        sees ONLY the basic USER FEEDBACK (HIGHEST PRIORITY) wrapper
        around your literal text and decides for itself whether to
        emit <patch> or <assets>. This is the default because the
        classifier-driven wrappers actively misrouted typed feedback
        in past sessions ("down key moves you forward" got wrapped
        with "the feedback above is about ART/SOUND, not code").

        OFF: opt-in to classifier wrapping. The harness reads your
        typed text, classifies it (art change / orientation change /
        scope lock / etc.), and adds directives like MEDIA-CHANGE,
        ORIENTATION-CHANGE, SCOPE ARBITRATION, asset stem mapping.
        Useful when the classifier reads your phrasing correctly and
        the extra coaching helps the model.

        UNAFFECTED by this toggle — always runs regardless:
          - Chromium browser test reports each iter (console errors,
            page errors, frozen-canvas check, RAF firing, probes,
            input smoke test). This is the load-bearing bug signal.
          - Patch failure diagnostics.
          - Playbook retrieval (/playbook off to disable separately).
          - Autonomous behavior playtest after clean iters (/playtest
            off to disable separately).
          - Visual critic on screenshots (/vlm-critique off to
            disable separately).

        Sticky across /new. Safe to toggle mid-session. Equivalent to
        /raw or /raw-feedback.
        """
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            raw_on = True
        elif arg_lc in ("off", "false", "0", "disable"):
            raw_on = False
        elif arg_lc == "":
            cur = (
                "[yellow]ON[/yellow]" if not self._use_feedback_directives
                else "[green]OFF[/green]"
            )
            self._log_info(
                f"raw-feedback mode is {cur} "
                "(usage: /rawfeedback on  ·  /rawfeedback off)"
            )
            return
        else:
            raw_on = self._use_feedback_directives  # currently directives-on → flip to raw-on
        # raw_on=True  → suppress directives → _use_feedback_directives=False
        # raw_on=False → keep directives      → _use_feedback_directives=True
        new_state = not raw_on
        self._use_feedback_directives = new_state
        if self.agent is not None:
            self.agent._use_feedback_directives = new_state
        status = (
            "[yellow]ON[/yellow] — directives suppressed, model sees your literal text"
            if not new_state else
            "[green]OFF[/green] — directives active (default)"
        )
        self._log_info(f"raw-feedback mode set to {status}")
        self._update_status()

    def _cmd_toggle_vlm_critique(self, arg: str) -> None:
        """/vlm-critique [on|off] — the review that LOOKS at the screen.

        When ON, after each round a vision model looks at the screenshot,
        uses a memory checklist when one fits the game, and sends any problems
        back to the coder so it fixes them next round. Which model looks:
          • model 2 staged as critic (model2/3 --role critic) — the fallback
            when your main model can't see
          • else the main model, if it is a VLM
          • else skip (a blind model never pretends to see)

        Default OFF. Alias: /judge. Sticky across /new.
        For the review that does NOT look at the screen, use /critique.
        """
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        else:
            new_state = not self._use_vlm_critique
        self._use_vlm_critique = new_state
        # Explicit toggle clears auto-staff so a user "off" sticks even if
        # _detect_vlm runs again and would otherwise re-enable.
        self._vlm_critique_auto = False
        # Explicit toggle while in wait mode clears the auto-restore-on-
        # wait-off memory. Without this, /vlm-critique on during wait
        # mode would be overridden when the user later runs /wait off
        # (we'd restore the saved-at-wait-on state, ignoring the new
        # explicit one). Same applies to explicit off. getattr() so
        # test fixtures that use App.__new__ (bypassing __init__) don't
        # AttributeError when the field hasn't been initialized.
        if getattr(self, "_vlm_critique_pre_wait", None) is not None:
            self._vlm_critique_pre_wait = None
        if self.agent is not None:
            self.agent._use_vlm_critique = new_state
            self.agent._vlm_critique_auto = False
            self.agent._all_roles_enabled = self._all_roles_enabled
        status = "[green]ON[/green]" if new_state else "[yellow]OFF[/yellow]"
        self._log_info(
            f"vlm-critique (vision) set to {status} "
            "(looks at the screen, uses a memory checklist when one fits, sends problems "
            "to the agent; uses model 2 if the main model can't see)"
        )
        self._update_status()

    def _cmd_set_leanprompt(self, arg: str) -> None:
        """/leanprompt on|off|auto — control the compact system-prompt schema.

        auto (default): the agent renders the lean `small` schema for LOCAL
        backends (MLX/Ollama) so a local VLM spends attention on the game,
        not a ~20KB schema; `large`/cloud keep the full schema. `on` forces
        lean; `off` forces the full schema for the current tier.
        """
        a = arg.strip().lower()
        if a in ("on", "true", "1", "enable"):
            self._lean_prompt = True
        elif a in ("off", "false", "0", "disable"):
            self._lean_prompt = False
        elif a in ("", "auto", "default"):
            self._lean_prompt = None
        else:
            self._log_info("usage: /leanprompt on|off|auto")
            return
        if self.agent is not None:
            self.agent.set_lean_prompt(self._lean_prompt)
        label = {True: "ON (forced)", False: "OFF (forced)", None: "auto (on for local)"}[self._lean_prompt]
        self._log_info(f"/leanprompt {label} — takes effect on the next /new session")
        self._update_status()

    def _cmd_toggle_allroles(self, arg: str) -> None:
        """/allroles — toggle ON/OFF: run coder + critic + architect on the single loaded LLM.

        Bundles architect-split and vlm-critique so one bare command covers
        every role without /model2 / /model3 staging or a second GPU. If a
        critic- or architect-tagged slot 2/3 IS staged, the router still
        prefers it — this toggle just turns the role-using features on.
        """
        arg_lc = arg.strip().lower()
        if arg_lc in ("on", "true", "1", "enable"):
            new_state = True
        elif arg_lc in ("off", "false", "0", "disable"):
            new_state = False
        else:
            new_state = not self._all_roles_enabled
        self._all_roles_enabled = new_state
        self._use_architect_split = new_state
        self._use_vlm_critique = new_state
        # Explicit user toggle — clear auto-staff flags so the state sticks.
        self._architect_split_auto = False
        self._vlm_critique_auto = False
        if self.agent is not None:
            self.agent._use_architect_split = new_state
            self.agent._use_vlm_critique = new_state
            self.agent._architect_split_auto = False
            self.agent._vlm_critique_auto = False
            self.agent._all_roles_enabled = new_state
        if new_state:
            self._log_info(
                "[green]/allroles ON[/green] — coder + critic + architect "
                "all running on the loaded LLM (architect-split + vlm-critique on)"
            )
        else:
            self._log_info(
                "[yellow]/allroles OFF[/yellow] — architect-split and vlm-critique disabled"
            )
        self._update_status()

    # ----------------------------- session --------------------------------

    async def _start_session(self, goal: str) -> None:
        """Boot the LiveBrowser + GameAgent and start consuming events."""
        self._phase_label = "starting browser"
        self._session_done = False
        self._session_seed = (
            Path(self._next_seed).resolve() if self._next_seed is not None else None
        )
        try:
            placement = backend_mod.ensure_ollama_slot_daemons_for_chat(
                enabled=self._wants_three_ollama_slots(),
                prefer=self._next_backend,
            )
            if placement.mode == "auto-pinned":
                self._ollama_placement_status = (
                    "auto-pinned · 11434→GPU1 · 11435→GPU2 · 11436→GPU3"
                )
                self._log_info(f"[dim]Ollama placement: {_esc(self._ollama_placement_status)}[/dim]")
            elif placement.mode == "manual":
                self._ollama_placement_status = "manual HOST2/HOST3"
            elif placement.mode == "fallback" and self._wants_three_ollama_slots():
                self._ollama_placement_status = f"single daemon fallback · {placement.message}"
                self._log_info(f"[yellow]Ollama placement:[/yellow] {_esc(self._ollama_placement_status)}")
            elif self._wants_three_ollama_slots():
                self._ollama_placement_status = f"single daemon fallback · {placement.message}"
        except Exception as e:
            self._ollama_placement_status = f"single daemon fallback · autopin error: {e}"
            self._log_info(f"[yellow]Ollama placement:[/yellow] {_esc(self._ollama_placement_status)}")
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
        if self._next_backend in ("ollama", "mlx", "openai", "anthropic") and self._next_model:
            if self._next_backend == "mlx":
                endpoint = backend_mod.mlx_endpoint_url()
            elif self._next_backend == "openai":
                endpoint = backend_mod.openai_endpoint_url()
            elif self._next_backend == "anthropic":
                endpoint = backend_mod.anthropic_endpoint_url()
            else:
                endpoint = backend_mod.ollama_endpoint_url(1)
            info = backend_mod.BackendInfo(
                name=self._next_backend, model=self._next_model,
                source=f"/load staged: {self._next_backend} (sticky)",
                endpoint=endpoint,
            )
        elif self._next_model:
            info = backend_mod.BackendInfo(
                name="ollama", model=self._next_model,
                source="/model staged (sticky)",
                endpoint=backend_mod.ollama_endpoint_url(1),
            )
        elif self._session_model and self._session_backend_info:
            prev = self._session_backend_info
            info = backend_mod.BackendInfo(
                name=prev.name,
                model=self._session_model,
                source="previous session (sticky)",
                endpoint=prev.endpoint,
                context_length=prev.context_length,
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
        # Latch for the next /new — keeps the backend+model unless /load clears it.
        self._next_backend = info.name
        self._next_model = info.model

        # Resolve staged model2 and model3
        self._session_backend2 = None
        self._session_backend_info2 = None
        self._session_model2 = None
        self._session_role2 = self._next_role2
        if self._next_backend2 and self._next_model2:
            endpoint2 = (
                backend_mod.ollama_endpoint_url(2)
                if self._next_backend2 == "ollama"
                else None
            )
            can_reuse_slot2 = (
                self._next_backend2 == info.name
                and self._next_model2 == info.model
                and (
                    self._next_backend2 != "ollama"
                    or (endpoint2 or "").rstrip("/") == (info.endpoint or "").rstrip("/")
                )
            )
            if can_reuse_slot2:
                self._session_backend2 = self._session_backend
                self._session_backend_info2 = info
                self._session_model2 = info.model
            else:
                if self._next_backend2 == "mlx":
                    endpoint = backend_mod.mlx_endpoint_url()
                elif self._next_backend2 == "openai":
                    endpoint = backend_mod.openai_endpoint_url()
                elif self._next_backend2 == "anthropic":
                    endpoint = backend_mod.anthropic_endpoint_url()
                else:
                    endpoint = backend_mod.ollama_endpoint_url(2)
                info2 = backend_mod.BackendInfo(
                    name=self._next_backend2, model=self._next_model2,
                    source=f"/model2 staged: {self._next_backend2} (sticky)",
                    endpoint=endpoint,
                )
                try:
                    self._session_backend2 = backend_mod.make_backend(info2)
                    self._session_backend_info2 = info2
                    self._session_model2 = info2.model
                except Exception as e:
                    self._log_error(f"could not initialize backend2: {e}")

        self._session_backend3 = None
        self._session_backend_info3 = None
        self._session_model3 = None
        self._session_role3 = self._next_role3
        if self._next_backend3 and self._next_model3:
            endpoint3 = (
                backend_mod.ollama_endpoint_url(3)
                if self._next_backend3 == "ollama"
                else None
            )
            can_reuse_primary_for_slot3 = (
                self._next_backend3 == info.name
                and self._next_model3 == info.model
                and (
                    self._next_backend3 != "ollama"
                    or (endpoint3 or "").rstrip("/") == (info.endpoint or "").rstrip("/")
                )
            )
            can_reuse_slot2_for_slot3 = (
                self._session_backend_info2
                and self._next_backend3 == self._session_backend_info2.name
                and self._next_model3 == self._session_backend_info2.model
                and (
                    self._next_backend3 != "ollama"
                    or (endpoint3 or "").rstrip("/") == (
                        self._session_backend_info2.endpoint or ""
                    ).rstrip("/")
                )
            )
            if can_reuse_primary_for_slot3:
                self._session_backend3 = self._session_backend
                self._session_backend_info3 = info
                self._session_model3 = info.model
            elif can_reuse_slot2_for_slot3:
                self._session_backend3 = self._session_backend2
                self._session_backend_info3 = self._session_backend_info2
                self._session_model3 = self._session_backend_info2.model
            else:
                if self._next_backend3 == "mlx":
                    endpoint = backend_mod.mlx_endpoint_url()
                elif self._next_backend3 == "openai":
                    endpoint = backend_mod.openai_endpoint_url()
                elif self._next_backend3 == "anthropic":
                    endpoint = backend_mod.anthropic_endpoint_url()
                else:
                    endpoint = backend_mod.ollama_endpoint_url(3)
                info3 = backend_mod.BackendInfo(
                    name=self._next_backend3, model=self._next_model3,
                    source=f"/model3 staged: {self._next_backend3} (sticky)",
                    endpoint=endpoint,
                )
                try:
                    self._session_backend3 = backend_mod.make_backend(info3)
                    self._session_backend_info3 = info3
                    self._session_model3 = info3.model
                except Exception as e:
                    self._log_error(f"could not initialize backend3: {e}")

        self.title = f"{CHAT_APP_TITLE} — {info.name.upper()} · {model_name}"
        self._log_info(
            f"Using [b]{info.name.upper()}[/b] · [b]{_esc(model_name)}[/b] "
            f"[dim]({_esc(info.source)})[/dim]"
        )
        if self._session_backend2:
            self._log_info(
                f"Using Model 2 [b]{self._session_backend_info2.name.upper()}[/b] · [b]{_esc(self._session_model2)}[/b] "
                f"as [cyan]{self._session_role2}[/cyan]"
            )
        if self._session_backend3:
            self._log_info(
                f"Using Model 3 [b]{self._session_backend_info3.name.upper()}[/b] · [b]{_esc(self._session_model3)}[/b] "
                f"as [cyan]{self._session_role3}[/cyan]"
            )

        for bi in (
            self._session_backend_info,
            self._session_backend_info2,
            self._session_backend_info3,
        ):
            if (
                bi is not None
                and bi.name == "ollama"
                and not bi.context_length
            ):
                cl = backend_mod.ollama_context_length(bi.endpoint, bi.model)
                if cl:
                    bi.context_length = cl

        # Large workstation: auto-unload tensor-split Ollama VRAM on /new so
        # the user does not need manual /unload or Ollama service env edits.
        # Probe each distinct Ollama endpoint (3-slot runs use HOST/HOST2/HOST3).
        seen_endpoints: set[str] = set()
        for bi in (
            self._session_backend_info,
            self._session_backend_info2,
            self._session_backend_info3,
        ):
            if bi is None or bi.name != "ollama":
                continue
            ep = (bi.endpoint or "").rstrip("/")
            if not ep or ep in seen_endpoints:
                continue
            seen_endpoints.add(ep)
            try:
                fixed, fix_msg = backend_mod.auto_fix_ollama_tensor_split(ep)
                if fixed and fix_msg:
                    self._log_info(f"[dim]{_esc(fix_msg)}[/dim]")
            except Exception:
                pass

        self._update_status()

        # Use the staged seed file (if any). Staging is STICKY — every /new
        # uses the same seed until you /seed (no arg) to clear or /seed
        # <other> to replace. Matches /model's sticky behavior.
        seed = self._next_seed
        # Pick the working file. With a seed: REUSE the original game's
        # canonical path (games/<basename>.html) so _session_id —
        # derived from out_path.stem in agent.py — inherits the
        # ORIGINAL basename. Every downstream artifact (assets dir,
        # sounds dir, traces, snapshots, .best.html) then lives in the
        # original folders. No new files, no new folders.
        #
        # The seed can be the canonical live file OR a snapshot like
        # games/snapshots/<basename>/iter_05.html OR a .best.html
        # sibling — _resolve_seed_target normalizes all three back to
        # games/<basename>.html. The seed's HTML content is still
        # what gets written into out_path; only the BASENAME is taken
        # from the canonical path so _session_id is right.
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        if seed is not None:
            self._out_path = _resolve_seed_target(Path(seed))
            self._best_path = self._out_path.with_suffix(".best.html")
            if self._out_path.resolve() != Path(seed).resolve():
                self._log_info(
                    f"continuing seed via canonical path: "
                    f"[b]{_esc(str(self._out_path))}[/b] "
                    f"[dim](seed: {_esc(str(seed))})[/dim]"
                )
            else:
                self._log_info(
                    f"continuing seed in place: [b]{_esc(str(seed))}[/b] "
                    "[dim](traces, snapshots, assets reuse the seed's "
                    "basename — /seed (no arg) to clear)[/dim]"
                )
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._out_path = GAMES_DIR / f"{_slugify(goal)}_{ts}.html"
            self._best_path = self._out_path.with_suffix(".best.html")
        # Single source of truth for the basename downstream code uses —
        # works for both seeded (canonical from _resolve_seed_target)
        # and fresh paths.
        basename = self._out_path.stem
        self._log_info(f"Game file: [b]{self._out_path}[/b]")

        # One timeout policy for every model — activity-aware stall
        # plus a generous quiet-window budget means we don't have to
        # guess model size to set the watchdog. See
        # resolve_session_timeouts for rationale.
        stall_s, overall_s = resolve_session_timeouts(model_name)
        # `overall_seconds` no longer caps active generation — that
        # cutoff was removed from ollama_io.stream_chat and
        # MLXBackend._stream_once after street-fighter trace
        # 20260518_220003 cut a still-producing 1800s MLX stream. The
        # value is now used only as the MLX cold-load timeout (waiting
        # for the model to land in VRAM before any tokens arrive). Log
        # message and trace event updated to reflect actual behavior.
        self._log_info(
            f"[dim]stream guards: no-activity stall={stall_s:.0f}s, "
            f"mlx cold-load cap={overall_s:.0f}s "
            f"(active streams not capped by wall-clock)[/dim]"
        )
        # Trace emission is deferred until just after GameAgent is
        # constructed — see the timeouts_resolved trace event below.
        self._pending_timeout_trace = {
            "kind": "timeouts_resolved",
            "stall_seconds": stall_s,
            "overall_seconds": overall_s,
            "stall_seconds_role": "no_activity_abort",
            "overall_seconds_role": "mlx_cold_load_cap_only",
            "active_stream_wallclock_cap": False,
        }

        self.agent = GameAgent(
            backend=self._session_backend,
            out_path=self._out_path,
            browser=self.browser,
            max_iters=self._max_iters,
            seed_file=seed,
            stall_seconds=stall_s,
            overall_seconds=overall_s,
            num_ctx=self._num_ctx,
            # v1 prompt: includes <playbook> retrieval, <criteria>,
            # <probes>, stuck-loop ladder.
            prompt_version="v1",
            # Pass the resolved chat model name so GameAgent._classify_model
            # can map it to "small"/"mid"/"large". Without this, model=None
            # falls through to "small" and silently strips <assets>/<sounds>/
            # <todos>/<lookup_bullet> from the system prompt for every local
            # 14B-35B coder, hiding the diffuser pipeline from the model.
            model=self._session_model,
            model_class=self._model_class or "auto",
            restart_n=self._restart_n,
            restart_score_threshold=self._restart_threshold,
            # Playbook injection is now ON by default, because we regulated
            # retrieval sizes for mid/small models to avoid attention pollution
            # and filled it with elite classic-game math & physics bullets.
            playbook_top_k=6,
            # Writeback still on so that when the user enables the
            # playbook (/playbook on), session outcomes update the
            # counters. Safe even when top_k=0 — writeback only fires
            # when bullets actually retrieved.
            playbook_writeback=True,
            use_prefill=self._use_prefill,
            use_vlm_critique=self._use_vlm_critique,
            use_double_screenshot=self._use_double_screenshot,
            use_architect_split=self._use_architect_split,
            backend2=self._session_backend2,
            model2_role=self._session_role2,
            backend3=self._session_backend3,
            model3_role=self._session_role3,
        )
        # Phase 1.5 — autonomous-feedback flag is set on the agent
        # AFTER construction so we don't have to thread it through
        # GameAgent.__init__'s long kwargs list (every existing test
        # that constructs GameAgent would otherwise need updating).
        # Default on the agent is True; the App's runtime toggle wins.
        self.agent._use_autonomous_feedback = self._use_autonomous_feedback
        self.agent._use_feedback_directives = self._use_feedback_directives
        self.agent._all_roles_enabled = self._all_roles_enabled
        # Lean system-prompt override (None = agent auto-decides: on for
        # local backends). Only set when the user explicitly toggled it.
        if self._lean_prompt is not None:
            self.agent.set_lean_prompt(self._lean_prompt)
        # Apply run-profile step policy on session start.
        if self._run_profile == "local_manual":
            self.agent.set_step_mode(True)
            self.agent.set_auto_step_on_failure(True)
        else:
            self.agent.set_step_mode(False)
            # Default for non-manual flows: keep running without forced
            # checkpoints; user can always /wait on when desired.
            self.agent.set_auto_step_on_failure(False)
        self.agent.set_token_callback(self._emit_token)
        # Pre-session /ref staging: if the user attached an image before
        # starting, feed it into the very first user turn of this run.
        if self._staged_ref_image_bytes is not None:
            self.agent._next_image_bytes = self._staged_ref_image_bytes
            staged_name = self._staged_ref_image_name or "reference image"
            self._log_info(
                f"/ref: using staged {staged_name} on the first model turn."
            )
            self._staged_ref_image_bytes = None
            self._staged_ref_image_name = None

        # Persist the resolved-timeouts info to the session trace now
        # that the agent (and its _trace sink) exist. Lets future
        # debugging answer "why did this session stall at Xs?" with
        # a single `jq` against the .jsonl.
        if getattr(self, "_pending_timeout_trace", None):
            try:
                self.agent._trace(self._pending_timeout_trace)
            except Exception:
                pass
            self._pending_timeout_trace = None

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
            if self._ctx_max is None and self.agent is not None:
                self._ctx_max = int(getattr(self.agent, "num_ctx", 0) or 0) or None
        except Exception:
            self._ctx_max = None

        # Use the agent-owned artifact stem, not the reusable game basename.
        # Seeded runs intentionally reuse games/<basename>.html and assets,
        # but logs/conversations/traces must be per-run to avoid mixed goals.
        self._open_log_mirror(self.agent.trace_path.stem)

        # Spawn the agent loop as a background task so the TUI stays responsive.
        self.run_worker(self._consume_events(goal, continuation=False), exclusive=True)

    def _reset_status_state(self) -> None:
        """Clear rolling status fields between sessions."""
        self._activity_label = ""
        self._activity_role = "coder"
        self._activity_started_at = 0.0
        self._stream_tokens = 0
        self._stream_started_at = 0.0
        self._last_token_at = 0.0
        self._is_streaming = False
        self._last_console_flush_at = 0.0
        self._last_stream_alive_note_at = 0.0
        self._runaway_console_warned = False
        self._model2_stream_tokens = 0
        self._model2_stream_started_at = 0.0
        self._model2_last_token_at = 0.0
        self._model2_is_streaming = False
        self._model3_stream_tokens = 0
        self._model3_stream_started_at = 0.0
        self._model3_last_token_at = 0.0
        self._model3_is_streaming = False
        self._assets_summary = ""
        self._sounds_summary = ""
        self._sounds_dir = None
        self._probes_passed = None
        self._probes_total = None
        self._last_diagnose = None
        self._last_stall_reason = None
        self._ctx_max = None
        self._streak_clean = 0
        self._streak_stuck = 0
        self._last_test_block = ""
        # Fresh session = fresh feedback ledger. Extensions skip this
        # method (they reuse the session), so the ledger survives them.
        self._feedback_ledger = []

    def _tick_status(self) -> None:
        """Periodic refresh of the status panel — only repaints when
        there's something time-sensitive to update (active stream,
        non-idle activity). Idle UI doesn't need re-renders."""
        if self._is_streaming or self._activity_label:
            try:
                self._maybe_note_stream_alive()
            except Exception:
                pass
            try:
                self._update_status()
            except Exception:
                pass

    def _maybe_note_stream_alive(self) -> None:
        """Print a dim [stream alive] console line when tokens are
        flowing but nothing printable has appeared for a while.

        Evidence (trace 20260612_132314): a 24-minute reasoning episode
        produced 18K+ tokens with zero console output for 9+ minutes —
        indistinguishable from a hang. Display-only, rate-limited;
        never touches the stream itself.
        """
        if not self._is_streaming or self._last_token_at == 0.0:
            return
        now = time.monotonic()
        if now - self._last_token_at > 10.0:
            # Tokens NOT flowing — that's a stall, which the status
            # panel's stall-age display already covers.
            return
        last_flush = self._last_console_flush_at or self._stream_started_at
        if not last_flush:
            return
        silent_for = now - last_flush
        if silent_for < self._STREAM_ALIVE_SILENCE_S:
            return
        if now - self._last_stream_alive_note_at < self._STREAM_ALIVE_REPEAT_S:
            return
        self._last_stream_alive_note_at = now
        elapsed = (
            now - self._stream_started_at if self._stream_started_at else 0.0
        )
        tps = self._stream_tokens / elapsed if elapsed > 0 else 0.0
        mins = int(silent_for // 60)
        silent_label = f"{mins}m" if mins else f"{int(silent_for)}s"
        self._log_info(
            f"[dim][stream alive] {self._stream_tokens:,} tokens · "
            f"{tps:.0f} tok/s · no printable output for {silent_label} — "
            "model is mid-reasoning; /done ships the last clean build[/dim]"
        )

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
        # Apply any /iters or /ctx change before extending.
        self.agent.max_iters = self._max_iters
        self.agent.num_ctx = self._num_ctx
        self._session_done = False
        self._phase_label = "extending"
        self.sub_title = "agent is working (extension)"
        # Feedback ledger: extension feedback never enters
        # agent._pending_feedback (it restarts the loop directly), so it
        # was invisible in the panel — record it as already applying.
        self._feedback_ledger.append(
            {"text": feedback, "status": "queued", "iter": None, "ok": None,
             "continuation": True}
        )
        self._update_status()
        self._log(
            "[dim cyan]  ✓ received — queued as extension of the "
            "finished session.[/dim cyan]"
        )
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
        self._log_mirror_lines = []
        self._status_plain = ""
        self._goal = goal
        self._iteration_label = "—"
        self._awaiting_kind = "feedback"
        self._log(f"[bold green]>[/bold green] /new {_esc(goal)}")
        await self._start_session(goal)

    def _open_log_mirror(self, basename: str) -> None:
        """Record the per-session artifact stem. The on-disk .log mirror is
        retired (2026-06-14): the .jsonl is the single complete trace, and the
        in-memory ``_log_mirror_lines`` buffer still backs the copy-log feature.
        We still set ``_log_file_path`` because it is the stem anchor other UI
        code uses to derive the jsonl / snapshots paths."""
        # Close any prior handle if rotating (older sessions may have opened one).
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
            # NOTE: intentionally NOT opening the file for writing — the disk
            # .log mirror duplicated the .jsonl. _log_file_handle stays None, so
            # the per-line writers (which no-op when None) skip disk writes and
            # _log_text_for_copy falls back to the in-memory mirror.
        except Exception as e:
            self._log_info(f"[dim]could not resolve log dir: {e}[/dim]")

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

    def _classify_test_blocker(self, report: dict) -> tuple[str, str]:
        """Return (category, short detail) for the top failing signal."""
        if report.get("ok"):
            return ("clean", "all checks passed")
        probes = report.get("probes") or []
        if isinstance(probes, list):
            failed = [p for p in probes if isinstance(p, dict) and not p.get("ok")]
            if failed:
                name = str(failed[0].get("name") or "probe")
                return ("probe", name[:60])
        errors = [str(e) for e in (report.get("errors") or []) if e]
        joined = " | ".join(errors).lower()
        if "page failed to load" in joined:
            return ("browser", errors[0][:80] if errors else "page failed to load")
        asset_tokens = (
            "err_file_not_found", "failed to load resource", "404",
            "no such file", "could not decode", "decode()",
        )
        if any(tok in joined for tok in asset_tokens):
            return ("assets", errors[0][:80] if errors else "asset load failure")
        if errors:
            return ("runtime", errors[0][:80])
        warns = [str(s) for s in (report.get("soft_warnings") or []) if s]
        if warns:
            return ("runtime", warns[0][:80])
        return ("format", "non-actionable output shape")

    async def _run_profile_review_hook(self, model: str) -> None:
        """Auto-run /check for local_plus_review when AUTO mode is active."""
        try:
            agent = self.agent
            if agent is None:
                return
            try:
                from vision_judge import _cloud_vendor
                vendor = _cloud_vendor(model)
            except Exception:
                vendor = ""
            if vendor == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
                self._log_info(
                    "[magenta]review hook[/magenta] skipped — ANTHROPIC_API_KEY not set"
                )
                return
            if vendor == "openai" and not os.environ.get("OPENAI_API_KEY"):
                self._log_info(
                    "[magenta]review hook[/magenta] skipped — OPENAI_API_KEY not set"
                )
                return
            goal = getattr(agent, "_goal", None)
            png = (
                getattr(agent, "_last_screenshot_after", None)
                or getattr(agent, "_prev_judge_png", None)
            )
            if not goal or not png:
                return
            self._log_info(
                f"[magenta]review hook[/magenta] running /check "
                f"[b]{_esc(model)}[/b] (auto-queue)"
            )
            await self._run_visual_check(
                model=model,
                goal=goal,
                png=png,
                agent=agent,
                apply_verdict=True,
                source="profile_hook_auto",
            )
        finally:
            self._auto_review_running = False

    def _maybe_trigger_profile_review(self, report: dict, blocker_category: str) -> None:
        """Schedule or suggest reviewer checks for local_plus_review."""
        if self._run_profile != "local_plus_review":
            return
        if report.get("ok"):
            return
        model = (self._profile_review_model or "").strip()
        if not model:
            return
        step_on = bool(getattr(self.agent, "_step_mode", False))
        base_cmd = f"/check {model}"
        if step_on:
            self._log_info(
                f"[magenta]review hook[/magenta] ({blocker_category}) ready: "
                f"run [b]{_esc(base_cmd)}[/b] before continuing"
            )
            return
        if self._profile_review_auto_apply:
            if self._auto_review_running:
                return
            self._auto_review_running = True
            self.run_worker(self._run_profile_review_hook(model), exclusive=False)
            return
        self._log_info(
            f"[magenta]review hook[/magenta] ({blocker_category}) suggestion: "
            f"[b]{_esc(base_cmd)}[/b]"
        )

    def _reconcile_feedback_ledger(self) -> None:
        """Flip ledger entries queued → applying once the agent drains them.

        The agent's `_flush_user_injections` consumes `_pending_feedback`
        between events (no callback), so we reconcile by set-difference:
        a "queued" ledger entry whose text is no longer pending has been
        injected into the current turn.
        """
        agent = getattr(self, "agent", None)
        if agent is None or not self._feedback_ledger:
            return
        pending = set(getattr(agent, "_pending_feedback", []) or [])
        for entry in self._feedback_ledger:
            if entry.get("status") == "queued" and entry.get("text") not in pending:
                entry["status"] = "applying"
                entry["iter"] = self._iteration_label

    def _handle_event(self, ev: AgentEvent) -> None:
        """Pattern-match on event kind and update the UI."""
        # Always flush any half-streamed line before logging a new event header.
        self._flush_stream()
        # Feedback ledger: detect drained items before rendering so the
        # status panel shows "applying" the moment injection happens.
        try:
            self._reconcile_feedback_ledger()
        except Exception:
            pass
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
            # Continuation honesty: a user feedback item is not "applied"
            # merely because the agent started a turn. Mark it as written
            # only after a real code materialization event; the following
            # test event will attach the harness result.
            for _entry in self._feedback_ledger:
                if _entry.get("status") in {"queued", "applying"}:
                    _entry["status"] = "wrote"
                    _entry["iter"] = self._iteration_label

        elif ev.kind == "test":
            # ev.text is the human-readable report, ev.data is the dict.
            ok = ev.data.get("ok", False)
            # Feedback ledger: only a real prior code write flips to
            # applied. Rejected/no-usable continuation turns can still
            # produce info events but must not claim the file changed.
            for _entry in self._feedback_ledger:
                if _entry.get("status") == "wrote":
                    _entry["status"] = "applied"
                    _entry["iter"] = self._iteration_label
                    _entry["ok"] = ok
            tag = "[green]TEST OK[/green]" if ok else "[red]TEST FAILED[/red]"
            n_err = len(ev.data.get("errors", []))
            n_iss = len(ev.data.get("soft_warnings", []))
            self._log(f"{tag} ({n_err} error(s), {n_iss} issue(s))")
            # Capture probe pass/fail counts for the iteration line. The
            # agent's test event exposes `probes` as a list of
            # {name, expr, ok, err, ...} dicts (see tools.py). Counting at
            # consume-time keeps the UI insulated from payload-shape drift.
            probes = ev.data.get("probes") or []
            probe_text = "—"
            if isinstance(probes, list) and probes:
                passed = sum(1 for p in probes if isinstance(p, dict) and p.get("ok"))
                self._probes_passed = passed
                self._probes_total = len(probes)
                probe_text = f"{passed}/{len(probes)}"
            blocker_category, blocker_detail = self._classify_test_blocker(ev.data or {})
            decision = "pass" if ok else "blocked"
            self._log(
                "[dim]iter decision:[/dim] "
                f"{decision} · probes {probe_text} · blocker {blocker_category}"
                + (f" ({_esc(blocker_detail)})" if blocker_detail else "")
            )
            if self._iter_decision_verbose and not ok:
                page_err = (ev.data.get("page_errors") or [])
                console_err = (ev.data.get("console_errors") or [])
                soft = (ev.data.get("soft_warnings") or [])
                probe_fail = [
                    str(p.get("name") or "probe")
                    for p in probes
                    if isinstance(p, dict) and not p.get("ok")
                ]
                detail_bits: list[str] = []
                if probe_fail:
                    detail_bits.append("probe_fails=" + ",".join(probe_fail[:3]))
                if page_err:
                    detail_bits.append("page_error=" + str(page_err[0])[:80])
                if console_err:
                    detail_bits.append("console_error=" + str(console_err[0])[:80])
                if soft:
                    detail_bits.append("soft=" + str(soft[0])[:80])
                if detail_bits:
                    self._log("[dim]iter detail:[/dim] " + _esc(" | ".join(detail_bits)))
            self._maybe_trigger_profile_review(ev.data or {}, blocker_category)
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
            if _is_ollama_load_failure(ev.text or ""):
                bad = self._session_model
                if bad:
                    mark_broken_ollama_tag(bad, reason=ev.text or "")
                self._log_info(self._ollama_escape_hint())

        elif ev.kind == "info":
            if ev.data.get("ask"):
                self._log(f"\n[bold cyan]── ask ──[/bold cyan]\n{text_safe}")
            else:
                self._log_info(text_safe)
            # Capture guard aborts (repetition / deliberation) into a
            # sticky status field so the reason doesn't just scroll past.
            stall_reason = (ev.data or {}).get("stall_reason")
            if stall_reason:
                if stall_reason == "repetition_loop":
                    kind = (ev.data or {}).get("loop_kind") or "repeat"
                    label = f"repetition loop ({kind})"
                else:
                    label = "deliberation loop (pre-code rambling)"
                toks = (ev.data or {}).get("tokens")
                tail = f" after {toks} tok" if toks else ""
                self._last_stall_reason = f"{label}{tail} — kept partial output"
                self._update_status()
            if (ev.text or "").startswith("no usable code:"):
                for _entry in self._feedback_ledger:
                    if _entry.get("status") in {"queued", "applying"}:
                        _entry["status"] = "no_usable"
                        _entry["ok"] = False
            # Info events can carry state changes the user needs to see
            # reflected in the bottom bar immediately — e.g. the agent
            # auto-arming step-mode after a first iter failure. Cheap;
            # Static.update diffs internally.
            self._update_mode_bar()

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
            data = ev.data or {}
            label = data.get("label", "")
            stream_role = data.get("role", "coder")
            now = time.monotonic()
            slot = self._role_slot_for_stream(stream_role)

            def _any_slot_streaming() -> bool:
                return (
                    self._is_streaming
                    or self._model2_is_streaming
                    or self._model3_is_streaming
                )

            if state == "idle":
                if "role" not in data:
                    # Legacy bare idle — end every slot's stream display.
                    self._is_streaming = False
                    self._model2_is_streaming = False
                    self._model3_is_streaming = False
                elif slot == 2:
                    self._model2_is_streaming = False
                elif slot == 3:
                    self._model3_is_streaming = False
                else:
                    self._is_streaming = False
                if not _any_slot_streaming():
                    self._activity_label = ""
                    self._activity_role = "coder"
                    self._activity_started_at = 0.0
            else:
                display = label or state.replace("_", " ")
                self._activity_role = stream_role
                self._activity_label = display
                self._activity_started_at = now
                # Phase 1C — record in-flight totals so the live
                # progress row in the status panel can show "Sprites:
                # N/total · X.Ys avg · ~Y ETA". The agent emits the
                # `requested` count alongside the activity event.
                if state == "generating_assets":
                    req = data.get("requested", 0)
                    if isinstance(req, int) and req > 0:
                        self._assets_in_flight_total = req
                elif state == "generating_sounds":
                    req = data.get("requested", 0)
                    if isinstance(req, int) and req > 0:
                        self._sounds_in_flight_total = req
                if state == "streaming":
                    if slot == 2:
                        self._model2_is_streaming = True
                        self._model2_stream_tokens = 0
                        self._model2_stream_started_at = now
                        self._model2_last_token_at = 0.0
                    elif slot == 3:
                        self._model3_is_streaming = True
                        self._model3_stream_tokens = 0
                        self._model3_stream_started_at = now
                        self._model3_last_token_at = 0.0
                    else:
                        self._is_streaming = True
                        self._stream_tokens = 0
                        self._stream_started_at = now
                        self._last_token_at = 0.0
                        # Per-stream visibility bookkeeping resets.
                        self._last_console_flush_at = 0.0
                        self._last_stream_alive_note_at = 0.0
                        self._runaway_console_warned = False
                elif slot == 2:
                    self._model2_is_streaming = False
                elif slot == 3:
                    self._model3_is_streaming = False
                else:
                    self._is_streaming = False
            self._update_status()

        elif ev.kind == "assets":
            self._assets_summary = self._format_assets_summary(ev.data or {})
            # Phase 1C — completion clears the in-flight total so the
            # live progress row stops rendering on the next tick.
            self._assets_in_flight_total = 0
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
            # Phase 1C — same clear pattern for sounds.
            self._sounds_in_flight_total = 0
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
            if self._streak_clean >= 1 and self._last_diagnose:
                # Clear stale diagnose text once clean iterations are flowing.
                self._last_diagnose = None
            if self._streak_clean >= 1 and self._last_stall_reason:
                # A clean iteration means the stream recovered — drop the
                # sticky stall note so it doesn't linger as stale.
                self._last_stall_reason = None
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
        scannable even on big batches.

        Item 6 (request 2026-05-15): the user asked for a clear visual
        marker showing whether each generated image is:
          - txt2img (text-prompt only — most assets)
          - img2img (animation frame chained from a prior asset via
            `from_image` — e.g. mario_walk2 from mario_walk1 for a
            walk-cycle frame pair)

        The pipeline tracks this distinction in the per-asset stat dict
        (`from_image` field set when img2img). We surface it as a small
        colored badge so the user can visually verify that animation
        chains rendered correctly (a img2img-chain failure usually
        shows up as a child frame that looks unrelated to its parent).
        """
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
            # Source badge — Item 6. `from_image` field in the stat
            # dict (set by assets.py:929-961) names the PARENT asset
            # this one was chained from via img2img. Absent / None =
            # plain txt2img.
            from_image = stat.get("from_image")
            if from_image:
                # img2img animation chain — show source asset name.
                # Truncate to keep the row scannable.
                parent = str(from_image)[:18]
                src = f"  [cyan]img2img←{_esc(parent)}[/cyan]"
            else:
                # Plain text-prompt generation. Dim so the eye is drawn
                # to the chained-frame entries.
                src = "  [dim]txt2img[/dim]"
            # If img2img fell back to txt2img (parent failed to
            # render), assets.py sets fallback_to_txt2img=True. Make
            # that visible because the asset's visual identity may
            # diverge from its intended parent.
            if stat.get("fallback_to_txt2img"):
                src = "  [yellow]txt2img (img2img fallback)[/yellow]"
            diffuser = stat.get("diffuser")
            gpu = stat.get("gpu")
            if diffuser:
                src += f"  [bold]{_esc(str(diffuser))}[/bold]"
                if gpu:
                    src += f" @ {_esc(str(gpu))}"
            rows.append(
                f"  - {name:<20} {cache} {secs_str}{src}{extra}"
            )
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
    # Asset Studio (browser UI for Z-Image txt2img/img2img) — same GPU
    # stack, always at http://127.0.0.1:8765/ while chat.py runs.
    # SKIP_ASSET_STUDIO=1 to opt out. Or double-click
    # scripts/Asset Studio.command when chat.py isn't running.
    if os.environ.get("SKIP_ASSET_STUDIO", "").strip() not in ("", "0", "false", "False"):
        pass
    else:
        try:
            _scripts = Path(__file__).resolve().parent / "scripts"
            if str(_scripts) not in sys.path:
                sys.path.insert(0, str(_scripts))
            import asset_studio as _asset_studio  # noqa: WPS433

            _asset_studio.ensure_server(open_browser=False)
        except Exception:
            pass
    try:
        CodingBoxApp().run()
    finally:
        _restore_terminal_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
