"""Build a short-prompt HTML training set with Qwen3.8-27B on oMLX.

Each prompt is one chat completion. A page is kept only when it is a full
HTML document and tools.test_html_file reports no console or page errors.
Twelve completions run at once on this M2 Ultra. A page at
http://127.0.0.1:8768/ shows aggregate decode tokens per second.

The script does not start oMLX and does not load a second copy of the model.
Load and pin Qwen3.8-27B-mxfp8 on the server first, and stop the LoRA trainer
so the GPU is free.

Rows match fine_tunning/rows.py kind "html": system, the short user line,
then <think> plus <html_file>.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import os
import random
import re
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

# Same string as train_lora._SHORT_SYS. kind "html" rows are trained under it.
SHORT_SYS = (
    "You are an expert HTML5 game programmer. Write the complete game as one "
    "self-contained HTML file (inline CSS and JavaScript, canvas) that runs in "
    "Chrome. Reply with brief <think> notes, then the file inside "
    "<html_file>...</html_file>."
)
ASSISTANT_NOTE = "Complete HTML game file with canvas/script."
# Generation only. pack_row still stores SHORT_SYS and a canned <think> line.
# Asking for notes here uses up the token cap before </html>, which shows up
# on the monitor as "no html".
GEN_SYS = (
    "Reply with one complete single-file browser game and nothing else. "
    "No thinking, no explanation, no markdown fences. "
    "The first characters are <html_file> and the last characters are </html_file>. "
    "Inside those tags, one document from <!DOCTYPE html> through </html>, "
    "with inline CSS and a script. "
    "Make it playable: a start, a score or a goal, a way to win or lose, "
    "and controls that work. "
    "Draw a full scene. Use several colors, a background, and shapes that "
    "read as the subject. Do not leave the canvas as one rectangle. "
    "Put the script in clear parts: state, input with addEventListener, "
    "update, draw, and a requestAnimationFrame loop. "
    "A famous title is one playable scene of that game, not the whole product. "
    "Finish that scene, then stop. Do not add another level, a second mode, or a long story. "
    "Close </html> and </html_file>."
)
# r128 has no CapsuleGeometry. The model mixes that old CDN with newer classes.
_THREE_SRC = "https://unpkg.com/three@0.160.0/build/three.min.js"
_THREE_RIDER = (
    " For three.js, include this script and no other three.js file: "
    f'<script src="{_THREE_SRC}"></script> '
    "Put it before your own script. Do not load r128. "
    "Add a scene, a camera, an ambient light, a directional light, and a ground plane. "
    "Use MeshStandardMaterial in a few distinct colors. "
    "Build objects from BoxGeometry, SphereGeometry, CylinderGeometry, "
    "ConeGeometry, or PlaneGeometry. Do not use CapsuleGeometry."
)
_EIGHT_RIDER = (
    " Draw with fillRect in a chunky pixel style. "
    "Paint a background, a player, and at least three other objects. "
    "Use about six solid colors."
)
_PHASER_RIDER = (
    " Load only https://cdn.jsdelivr.net/npm/phaser@3.80.1/dist/phaser.min.js . "
    "One Phaser.Scene with create and update. "
    "Draw the playfield with graphics or generated textures."
)
_PIXI_RIDER = (
    " Load only https://cdn.jsdelivr.net/npm/pixi.js@7.4.2/dist/pixi.min.js . "
    "Build a stage of Graphics or sprites, a ticker, and several colors."
)
_HOWLER_RIDER = (
    " Load only https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.4/howler.min.js . "
    "Use at least two Howl sounds, a visible control for each, and a score or a goal."
)
# rows.TOKEN_BUDGET. A row over this does not fit the trainer window.
TOKEN_BUDGET = 8128
# Generation cap. The packed training row must stay under TOKEN_BUDGET.
MAX_TOKENS = 6144
DEFAULT_JOBS = 12
DEFAULT_PORT = 8768
DEFAULT_MODEL = "Qwen3.8-27B-mxfp8"
CHROME_WORKERS = 2
ROLL_SECONDS = 10.0
PREVIEW_CHARS = 80_000
JOB_CHOICES = (4, 8, 12, 16, 24, 32, 48, 64)

_HTML_FILE = re.compile(r"<html_file>\s*(.*?)\s*</html_file>", re.I | re.S)
_DOCUMENT = re.compile(
    r"(<!doctype\s+html\b[\s\S]*?</html\s*>|<html\b[\s\S]*?</html\s*>)",
    re.I,
)
_SCRIPT = re.compile(r"<script\b", re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_FENCE = re.compile(r"^```(?:html)?\s*|\s*```$", re.I)

# Shared with the monitor thread. The event loop is the only writer of
# counters except jobs, which the page sets.
STATS: "Stats | None" = None
GATE: "Gate | None" = None
LOOP: asyncio.AbstractEventLoop | None = None


def estimate_tokens(text: str) -> int:
    """Rough token count so a row can be rejected without loading the tokenizer."""
    return max(1, (len(text) + 3) // 4)


def normalize_prompt(text: str) -> str:
    return " ".join(text.lower().split())


# Longest first so "make me" is not cut down to "make".
_OPENER_PREFIXES = (
    "could you make ",
    "can you build ",
    "can you make ",
    "please build ",
    "please make ",
    "put together ",
    "let's make ",
    "build me ",
    "make me ",
    "how about ",
    "i'd like ",
    "show me ",
    "write me ",
    "give me ",
    "i want ",
    "create ",
    "build ",
    "make ",
    "code ",
)
# One game or graphic, ignoring the opening and the "small version" wrapper.
_SUBJECT_FILLERS = (
    "a complete working version of ",
    "a small working version of ",
    "a timed one-minute game of ",
    "a kid-friendly game of ",
    "a little browser toy that ",
    "a single-screen tool that ",
    "a two-player game of ",
    "a one-screen game of ",
    "a game of ",
    "a page that ",
)


def subject_key(text: str) -> str:
    """Identity of the game or graphic, so a reworded prompt is still the same one."""
    s = normalize_prompt(text).rstrip(".?!").strip()
    for prefix in _OPENER_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    for filler in _SUBJECT_FILLERS:
        if s.startswith(filler):
            s = s[len(filler):].strip()
            break
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def load_ideas(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def prompts_to_run(seed: list[str], already: set[str], limit: int) -> tuple[list[str], int]:
    """Seed prompts still to generate, and how many further prompts to invent.

    `already` holds normalized prompts already stored in the output file.
    limit 0 means keep inventing after the seed (the extra count is -1).
    """
    seen = set(already)
    subjects = {subject_key(item) for item in already}
    subjects.discard("")
    out: list[str] = []
    for prompt in seed:
        key = normalize_prompt(prompt)
        subject = subject_key(prompt)
        if key in seen or (subject and subject in subjects):
            continue
        if limit and len(already) + len(out) >= limit:
            break
        out.append(prompt)
        seen.add(key)
        if subject:
            subjects.add(subject)
    if limit == 0:
        return out, -1
    extra = max(0, limit - len(already) - len(out))
    return out, extra


def parse_idea_lines(text: str, seen: set[str]) -> list[str]:
    """Pull new one-line prompts out of a model reply. Skips ones in `seen`."""
    out: list[str] = []
    local = set(seen)
    for item in seen:
        subject = subject_key(item)
        if subject:
            local.add(subject)
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\).\]]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith(("here ", "sure", "these ", "note", "okay", "ok ")):
            continue
        if len(line.split()) < 4 or len(line) > 240:
            continue
        if not line.endswith("."):
            line += "."
        key = normalize_prompt(line)
        subject = subject_key(line)
        if key in local or (subject and subject in local):
            continue
        local.add(key)
        if subject:
            local.add(subject)
        out.append(line)
    return out


def extract_html(text: str) -> str | None:
    """Return the HTML document inside a model reply, or None."""
    if not text or not text.strip():
        return None
    wrapped = _HTML_FILE.search(text)
    chunk = wrapped.group(1) if wrapped else text
    chunk = _FENCE.sub("", chunk.strip()).strip()
    doc = _DOCUMENT.search(chunk)
    if doc:
        return doc.group(1).strip()
    if re.search(r"<html\b", chunk, re.I) and re.search(r"</html\s*>", chunk, re.I):
        return chunk.strip()
    return None


def structure_error(html: str | None) -> str:
    if not html or not html.strip():
        return "no html"
    low = html.lower()
    if "<html" not in low or "</html" not in low:
        return "not a full document"
    if not _SCRIPT.search(html):
        return "no script"
    return ""


def assistant_text(html: str) -> str:
    page = html.strip()
    return (
        f"<think>\n{ASSISTANT_NOTE}\n</think>\n"
        f"<html_file>\n{page}\n</html_file>\n"
    )


def title_of(html: str, prompt: str) -> str:
    m = _TITLE.search(html)
    if m:
        t = re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip()
        if t:
            return t[:120]
    return prompt[:120]


def pack_row(prompt: str, html: str) -> dict[str, Any]:
    """One trainer row. Same shape as rows._pack for kind html."""
    page = html.strip()
    digest = hashlib.sha256(
        b"synth\n" + prompt.encode("utf-8") + b"\n" + page.encode("utf-8")
    ).hexdigest()
    source = "synth:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return {
        "sha": digest,
        "kind": "html",
        "title": title_of(page, prompt),
        "source": source,
        "weight": 1,
        "messages": [
            {"role": "system", "content": SHORT_SYS},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_text(page)},
        ],
    }


def row_tokens(prompt: str, html: str) -> int:
    row = pack_row(prompt, html)
    blob = "\n".join(m["content"] for m in row["messages"])
    return estimate_tokens(blob)


def bug_report(reason: str, report: dict | None) -> dict[str, Any]:
    """Chrome's own errors, including a line or stack when the browser sent one."""
    errors = [str(item) for item in ((report or {}).get("errors") or [])]
    warnings = [str(item) for item in ((report or {}).get("warnings") or [])]
    soft = [str(item) for item in ((report or {}).get("soft_warnings") or [])]
    parts: list[str] = []
    if reason and reason.strip():
        parts.append(reason.strip())
    if errors:
        parts.append("Chrome errors:\n" + "\n".join(errors))
    if warnings:
        parts.append("Warnings:\n" + "\n".join(warnings))
    if soft:
        parts.append("Checks:\n" + "\n".join(soft))
    text = "\n\n".join(parts).strip() or "failed"
    return {"report": text, "errors": errors, "warnings": warnings}


def pack_buggy_row(prompt: str, code: str, reason: str, report: dict | None) -> dict[str, Any]:
    """Trainer-shaped row plus a bug entry. One shot: nothing here is repaired."""
    page = extract_html(code or "") or ""
    if page:
        row = pack_row(prompt, page)
    else:
        raw = (code or "").strip()
        row = pack_row(prompt, "")
        row["title"] = prompt[:120]
        row["messages"][2]["content"] = raw
        row["sha"] = hashlib.sha256(
            b"buggy\n" + prompt.encode("utf-8") + b"\n" + raw.encode("utf-8")
        ).hexdigest()
    bug = bug_report(reason, report)
    row["kind"] = "buggy"
    row["source"] = "buggy:" + str(row["source"]).split(":", 1)[-1]
    row["bug"] = bug
    row["messages"].append({"role": "bug", "content": bug["report"]})
    return row


def decide_keep(html: str | None, report: dict | None, prompt: str) -> tuple[bool, str]:
    """Keep only a full scripted document that fits the window and passed Chrome."""
    err = structure_error(html)
    if err:
        return False, err
    assert html is not None
    if row_tokens(prompt, html) > TOKEN_BUDGET:
        return False, "over token budget"
    if not report or not report.get("ok"):
        errs = (report or {}).get("errors") or ["chrome error"]
        return False, "; ".join(str(e) for e in errs)[:500]
    return True, ""


def prompts_in_jsonl(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.is_file():
        return seen
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for msg in row.get("messages") or []:
            if msg.get("role") == "user" and msg.get("content"):
                seen.add(normalize_prompt(str(msg["content"])))
    return seen


def take_delta(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def take_usage_completion(obj: dict) -> int | None:
    usage = obj.get("usage") or {}
    n = usage.get("completion_tokens")
    if isinstance(n, int) and n > 0:
        return n
    return None


class Stats:
    """Numbers the monitor page reads. A lock covers the HTTP thread."""

    def __init__(self, jobs: int, limit: int) -> None:
        self.lock = threading.Lock()
        self.jobs = jobs
        self.limit = limit
        self.inflight = 0
        self.testing = 0
        self.kept = 0
        self.dropped = 0
        self.resumed = 0
        self.queue = 0
        self.extra_left = 0
        self.prefill_sum = 0.0
        self.prefill_n = 0
        self.roll: deque[tuple[float, int]] = deque()
        self.examples: deque[dict[str, Any]] = deque(maxlen=3)
        self.working: deque[dict[str, Any]] = deque(maxlen=80)
        self.buggy: deque[dict[str, Any]] = deque(maxlen=80)
        self.pages: dict[str, str] = {}
        self.out_path: Path | None = None
        self.buggy_path: Path | None = None
        self.t0 = time.time()
        self.last_error = ""
        self.model = ""
        self.running = True

    def _trim(self, now: float) -> None:
        cutoff = now - ROLL_SECONDS
        while self.roll and self.roll[0][0] < cutoff:
            self.roll.popleft()

    def add_tokens(self, n: int, now: float | None = None) -> None:
        if n == 0:
            return
        now = time.time() if now is None else now
        with self.lock:
            self.roll.append((now, n))
            self._trim(now)

    def note_prefill(self, seconds: float) -> None:
        if seconds < 0:
            return
        with self.lock:
            self.prefill_sum += seconds
            self.prefill_n += 1

    def tok_s(self, now: float | None = None) -> float:
        """Aggregate decode tokens per second over the last 10 seconds."""
        now = time.time() if now is None else now
        with self.lock:
            self._trim(now)
            total = sum(n for _, n in self.roll)
            elapsed = now - self.t0
            denom = ROLL_SECONDS if elapsed >= ROLL_SECONDS else max(elapsed, 0.25)
            return max(0.0, total / denom)

    def push_feed(
        self,
        prompt: str,
        status: str,
        error: str,
        shown: str,
        sha: str,
        weight: int = 1,
        page: str = "",
    ) -> None:
        snippet = (shown or "").strip()[:1500]
        item = {
            "sha": sha,
            "prompt": prompt,
            "status": status,
            "error": error[:1500],
            "weight": weight,
            "code": snippet,
        }
        with self.lock:
            if page and len(page) <= PREVIEW_CHARS:
                self.pages[sha] = page
            if status == "kept":
                self.working.appendleft(item)
            else:
                self.buggy.appendleft(item)

    def set_feed_weight(self, sha: str, weight: int) -> None:
        with self.lock:
            for row in list(self.working) + list(self.buggy):
                if row.get("sha") == sha:
                    row["weight"] = weight

    def push_example(self, prompt: str, html: str) -> None:
        preview = html if len(html) <= PREVIEW_CHARS else ""
        note = "" if preview else "too big to preview"
        with self.lock:
            self.examples.appendleft({
                "prompt": prompt,
                "html": preview,
                "note": note,
            })

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            self._trim(now)
            total = sum(n for _, n in self.roll)
            elapsed = max(now - self.t0, 0.25)
            denom = ROLL_SECONDS if elapsed >= ROLL_SECONDS else elapsed
            tok_s = max(0.0, total / denom)
            done = self.kept + self.dropped
            pass_rate = (self.kept / done) if done else 0.0
            ppm = done / (elapsed / 60.0)
            prefill = (self.prefill_sum / self.prefill_n) if self.prefill_n else 0.0
            remaining = self.queue + self.inflight + max(0, self.extra_left)
            eta = (remaining / (done / elapsed)) / 60.0 if done else None
            if self.limit == 0:
                eta = None
            return {
                "tok_s": round(tok_s, 1),
                "prefill_s": round(prefill, 2),
                "inflight": self.inflight,
                "testing": self.testing,
                "jobs": self.jobs,
                "kept": self.kept,
                "dropped": self.dropped,
                "resumed": self.resumed,
                "pass_rate": round(pass_rate, 3),
                "pages_per_min": round(ppm, 2),
                "queue": self.queue,
                "eta_min": None if eta is None else round(eta, 1),
                "limit": self.limit,
                "elapsed_s": round(elapsed, 1),
                "last_error": self.last_error,
                "model": self.model,
                "running": self.running,
                "examples": list(self.examples),
                "working": list(self.working),
                "buggy": list(self.buggy),
            }


class Gate:
    """Caps how many generations run at once. The monitor can change `n`."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.inflight = 0
        self._cv = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cv:
            while self.inflight >= self.n:
                await self._cv.wait()
            self.inflight += 1

    async def release(self) -> None:
        async with self._cv:
            self.inflight = max(0, self.inflight - 1)
            self._cv.notify_all()

    async def set_n(self, n: int) -> None:
        async with self._cv:
            self.n = n
            self._cv.notify_all()


def chrome_smoke(path: str) -> dict:
    """Process-pool entry. Imports tools inside the child so tests stay light."""
    import sys
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools import test_html_file
    # Headless only. One second is enough to catch a console or page error.
    return test_html_file(path, run_seconds=1.0)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HTML synth</title>
<style>
  html, body { height: 100%; }
  body { margin: 0; font: 16px/1.4 ui-sans-serif, system-ui, sans-serif;
         background: #111; color: #eee; display: flex; flex-direction: column;
         overflow: hidden; }
  header { padding: 10px 16px 4px; flex: none; }
  .tok { font-size: 42px; font-weight: 700; letter-spacing: -1px; line-height: 1; }
  .sub { color: #9aa; margin-top: 6px; }
  .grid { display: flex; flex-wrap: wrap; gap: 10px 18px; padding: 6px 16px 8px; flex: none; }
  .grid b { display: block; font-size: 22px; }
  .grid span { color: #9aa; font-size: 13px; }
  button { background: #222; color: #eee; border: 1px solid #444; border-radius: 8px;
           padding: 8px 14px; font-size: 16px; cursor: pointer; }
  button.on { background: #246; border-color: #48a; }
  .row { padding: 0 28px 12px; display: flex; gap: 8px; align-items: center; }
  .err { color: #f88; padding: 0 28px 8px; min-height: 1.2em; }
  #cols { display: flex; gap: 8px; flex: 0 1 30vh; min-height: 120px; margin: 0 16px 8px; }
  .col { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .col h2 { margin: 0 0 4px; font-size: 13px; font-weight: 600; color: #9ab; }
  .log { flex: 1; min-height: 0; overflow: auto; background: #1b1b1b; border-radius: 10px; }
  .log .item { padding: 8px 10px; border-bottom: 1px solid #333; cursor: pointer; }
  .log .item.on { background: #243044; }
  .log .meta { color: #9ab; font-size: 12px; white-space: pre-wrap; }
  .log .notes { color: #fc8; font-size: 12px; white-space: pre-wrap; margin-top: 4px; }
  .log pre { margin: 4px 0 0; white-space: pre-wrap; color: #cde; font-size: 12px; max-height: 72px; overflow: hidden; }
  #stage { flex: 1 1 auto; min-height: 0; width: calc(100% - 32px); height: auto; margin: 0 16px 12px; border: 0; background: #111; border-radius: 10px; }
  @media (max-width: 900px) { .cards { grid-template-columns: 1fr; } .tok { font-size: 64px; } }
</style>
</head>
<body>
<header>
  <div class="tok" id="tok">–</div>
  <div class="sub" id="sub">decode tok/s</div>
</header>
<div class="grid" id="grid"></div>
<div class="row">
  <span>streams</span>
  <button data-n="4">4</button>
  <button data-n="8">8</button>
  <button data-n="12">12</button>
  <button data-n="16">16</button>
  <button data-n="24">24</button>
  <button data-n="32">32</button>
  <button data-n="48">48</button>
  <button data-n="64">64</button>
</div>
<div class="err" id="err"></div>
<div class="row">
  <button id="run">Run</button>
  <button id="up">Up</button>
  <button id="down">Down</button>
  <span id="pick">Select a row. Down sets training weight to 0.</span>
</div>
<div id="cols">
  <div class="col"><h2>Working</h2><div id="ok" class="log"></div></div>
  <div class="col"><h2>Buggy</h2><div id="bad" class="log"></div></div>
</div>
<iframe id="stage" title="selected page"></iframe>
<script>
const tok = document.getElementById("tok");
const sub = document.getElementById("sub");
const grid = document.getElementById("grid");
const err = document.getElementById("err");
const okLog = document.getElementById("ok");
const badLog = document.getElementById("bad");
const stage = document.getElementById("stage");
const pick = document.getElementById("pick");
let selected = "";
function cell(n, label) {
  const d = document.createElement("div");
  const b = document.createElement("b");
  b.textContent = n;
  const s = document.createElement("span");
  s.textContent = label;
  d.append(b, s);
  return d;
}
async function setJobs(n) {
  await fetch("/jobs", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({jobs: n})});
}
document.querySelectorAll("button[data-n]").forEach(btn => {
  btn.addEventListener("click", () => setJobs(Number(btn.dataset.n)));
});
function show(s) {
  tok.textContent = s.tok_s.toFixed(0);
  const pf = s.prefill_s ? s.prefill_s.toFixed(1) + "s avg prefill" : "prefill —";
  sub.textContent = "decode tok/s over the last 10s · " + pf;
  const eta = s.eta_min == null ? "—" : s.eta_min + " min";
  const rate = Math.round(s.pass_rate * 100) + "%";
  grid.replaceChildren(
    cell(s.inflight + " / " + s.jobs, "in flight / cap"),
    cell(s.testing, "in Chrome"),
    cell(s.kept, "kept"),
    cell(s.dropped, "dropped"),
    cell(rate, "pass rate"),
    cell(s.pages_per_min, "pages / min"),
    cell(s.queue, "queued"),
    cell(eta, "time left"),
    cell(s.resumed, "saved earlier")
  );
  document.querySelectorAll("button[data-n]").forEach(btn => {
    btn.classList.toggle("on", Number(btn.dataset.n) === s.jobs);
  });
  err.textContent = s.last_error || "";
  const working = s.working || [];
  const buggy = s.buggy || [];
  const sig = working.concat(buggy).map(r => r.sha + ":" + r.weight + ":" + r.status).join("|");
  if (sig === show.sig) return;
  show.sig = sig;
  fillLog(okLog, working);
  fillLog(badLog, buggy);
}
function fillLog(log, rows) {
  const top = log.scrollTop;
  log.replaceChildren();
  rows.forEach(row => {
    const item = document.createElement("div");
    item.className = "item" + (row.sha === selected ? " on" : "");
    const title = document.createElement("div");
    title.textContent = row.prompt;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = (row.weight === 0 ? "weight 0" : "weight " + row.weight);
    item.append(title, meta);
    if (row.error) {
      const notes = document.createElement("div");
      notes.className = "notes";
      notes.textContent = row.error;
      item.append(notes);
    }
    if (row.code) {
      const pre = document.createElement("pre");
      pre.textContent = row.code;
      item.append(pre);
    }
    item.addEventListener("click", () => {
      selected = row.sha;
      pick.textContent = row.prompt;
      document.querySelectorAll(".log .item").forEach(el => el.classList.remove("on"));
      item.classList.add("on");
    });
    log.append(item);
  });
  log.scrollTop = top;
}
show.sig = "";
document.getElementById("run").addEventListener("click", () => {
  if (!selected) return;
  stage.src = "/html?sha=" + encodeURIComponent(selected);
});
async function vote(weight) {
  if (!selected) return;
  await fetch("/vote", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({sha: selected, weight: weight})});
}
document.getElementById("up").addEventListener("click", () => vote(1));
document.getElementById("down").addEventListener("click", () => vote(0));
async function tick() {
  try {
    const r = await fetch("/stats", {cache: "no-store"});
    show(await r.json());
  } catch (e) {}
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/stats"):
            snap = STATS.snapshot() if STATS else {}
            body = json.dumps(snap).encode("utf-8")
            self._send(200, "application/json", body)
            return
        if self.path.startswith("/html"):
            sha = ""
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse
                sha = (parse_qs(urlparse(self.path).query).get("sha") or [""])[0]
            page = ""
            if STATS is not None:
                with STATS.lock:
                    page = STATS.pages.get(sha, "")
            if not page:
                self._send(404, "text/plain", b"no page")
                return
            self._send(200, "text/html; charset=utf-8", fit_html(page).encode("utf-8"))
            return
        self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        if not (self.path.startswith("/jobs") or self.path.startswith("/vote")):
            self._send(404, "text/plain", b"no")
            return
        n = 0
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            n = int(json.loads(raw.decode("utf-8") or "{}").get("jobs") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            n = 0
        if self.path.startswith("/vote"):
            weight = -1
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
                weight = int(body.get("weight"))
                sha = str(body.get("sha") or "")
            except (json.JSONDecodeError, TypeError, ValueError):
                sha = ""
            if weight not in (0, 1) or not sha or STATS is None:
                self._send(400, "text/plain", b"need sha and weight 0 or 1")
                return
            STATS.set_feed_weight(sha, weight)
            if STATS.out_path is not None:
                rewrite_weight(STATS.out_path, sha, weight)
            if STATS.buggy_path is not None:
                rewrite_weight(STATS.buggy_path, sha, weight)
            self._send(204, "text/plain", b"")
            return
        if n not in JOB_CHOICES or GATE is None or LOOP is None or STATS is None:
            self._send(400, "text/plain", b"jobs must be 4, 8, 12, 16, 24, 32, 48, or 64")
            return
        with STATS.lock:
            STATS.jobs = n
        asyncio.run_coroutine_threadsafe(GATE.set_n(n), LOOP)
        self._send(204, "text/plain", b"")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve_monitor(port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, name="synth-monitor", daemon=True)
    thread.start()
    return httpd


def rewrite_weight(path: Path, sha: str, weight: int) -> bool:
    """Set weight on the matching synth.jsonl row. Down vote uses 0."""
    if not path.is_file():
        return False
    changed = False
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        if row.get("sha") == sha:
            row["weight"] = weight
            changed = True
            line = json.dumps(row, ensure_ascii=False)
        out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def seed_feed(stats: "Stats", path: Path, status: str = "kept") -> None:
    """Show saved rows so either set can be run and voted."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = ""
        page = ""
        shown = ""
        error = ""
        saw_user = False
        for message in row.get("messages") or []:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user" and not saw_user:
                prompt = content
                saw_user = True
            elif role == "assistant":
                shown = content
                page = extract_html(content) or ""
            elif role == "bug":
                error = content
        bug = row.get("bug") or {}
        if not error and isinstance(bug, dict):
            error = str(bug.get("report") or "")
        weight = row.get("weight", 1)
        stats.push_feed(
            prompt, status, error, page or shown, str(row.get("sha") or ""),
            0 if weight == 0 else 1, page=page,
        )


def fit_html(page: str) -> str:
    """Show the whole game in the preview. Pages use a tall canvas plus a
    100vh column with overflow hidden, which clips the paddle and the caption.
    Lay the game out at 960x900, then scale that frame to the preview.
    """
    payload = json.dumps(page).replace("<", "\\u003c")
    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
        "html,body{margin:0;height:100%;background:#111;overflow:hidden}"
        "#view{border:0;width:960px;height:900px;transform-origin:0 0}"
        "</style></head><body>"
        "<iframe id=\"view\" sandbox=\"allow-scripts\"></iframe><script>"
        "const view=document.getElementById('view');"
        f"view.srcdoc={payload};"
        "function fit(){const s=Math.min(window.innerWidth/960,window.innerHeight/900);"
        "view.style.transform='scale('+s+')';}"
        "addEventListener('resize',fit);fit();"
        "</script></body></html>"
    )


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


def gen_messages(prompt: str) -> list[dict[str, str]]:
    system = GEN_SYS
    low = prompt.lower()
    if "three.js" in low:
        system += _THREE_RIDER
    elif "8-bit" in low:
        system += _EIGHT_RIDER
    elif " in phaser" in low:
        system += _PHASER_RIDER
    elif " in pixijs" in low:
        system += _PIXI_RIDER
    elif " in howler.js" in low:
        system += _HOWLER_RIDER
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def repair_messages(prompt: str, reason: str) -> list[dict[str, str]]:
    note = (reason or "it failed").strip()[:800]
    return [
        {"role": "system", "content": SHORT_SYS},
        {"role": "user", "content": (
            prompt
            + "\n\nThat page failed ("
            + note
            + "). Fix it. Reply with brief <think> notes, then the full file "
            "inside <html_file>."
        )},
    ]


def invent_messages(sample: list[str]) -> list[dict[str, str]]:
    block = "\n".join(sample)
    return [
        {"role": "system", "content": "You invent short user requests. Reply with lines only."},
        {"role": "user", "content": (
            "Propose 40 new one-line requests a person would type to get a "
            "single-file browser program. Each line is only the request. "
            "Vary the opening. People say Write me, Make, Build, Create, "
            "Code, Give me, I want, Can you make, and Please build. "
            "Do not start every line the same way. No numbers and no explanations. "
            "Do not repeat these:\n" + block
        )},
    ]


async def complete(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    stats: Stats,
    max_tokens: int,
) -> str:
    """Stream one completion. Decode tokens land in the rolling tok/s window."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "enable_thinking": False,
    }
    t0 = time.time()
    t_first: float | None = None
    parts: list[str] = []
    counted = 0
    usage_tokens: int | None = None
    async with client.stream(
        "POST", url, json=body, headers={"accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = take_delta(obj)
            if delta:
                if t_first is None:
                    t_first = time.time()
                    stats.note_prefill(t_first - t0)
                parts.append(delta)
                n = max(1, len(delta) // 4)
                stats.add_tokens(n)
                counted += n
            got = take_usage_completion(obj)
            if got:
                usage_tokens = got
    if usage_tokens is not None and usage_tokens != counted:
        stats.add_tokens(usage_tokens - counted)
    return "".join(parts)


def model_id_from_list(payload: dict, want: str) -> str | None:
    want_base = Path(want).name
    ids = [str(m.get("id", "")) for m in payload.get("data") or []]
    for mid in ids:
        if mid == want or mid == want_base or mid.endswith("/" + want_base) or want_base in mid:
            return mid
    return None


async def ensure_model(client: httpx.AsyncClient, base: str, want: str) -> str:
    url = base.rstrip("/") + "/v1/models"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"oMLX is not reachable at {base} ({exc}). "
            "Start it, load Qwen3.8-27B-mxfp8, and rerun. This script does not start the server."
        ) from exc
    found = model_id_from_list(payload, want)
    if found:
        return found
    ids = [str(m.get("id", "")) for m in payload.get("data") or []]
    loaded = ", ".join(ids) if ids else "nothing"
    raise SystemExit(
        f"{Path(want).name} is not loaded at {base}. Loaded: {loaded}. "
        "Pin Qwen3.8-27B-mxfp8 in oMLX and rerun. Do not load a second copy."
    )


async def run_chrome(pool: ProcessPoolExecutor, html: str) -> dict:
    fd, name = tempfile.mkstemp(suffix=".html", prefix="synth_")
    try:
        os.write(fd, html.encode("utf-8"))
    finally:
        os.close(fd)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(pool, chrome_smoke, name)
    finally:
        # Scratch file for the smoke test only. The kept page lives in the jsonl.
        try:
            os.unlink(name)
        except OSError:
            pass


async def amain(args: argparse.Namespace) -> None:
    global STATS, GATE, LOOP
    out = args.out
    buggy = out.with_name("buggy.jsonl")
    added = out.with_name("added.jsonl")
    ideas = load_ideas(args.ideas)
    kept_done = prompts_in_jsonl(out)
    # A failed game stays failed. Do not generate it again.
    already = kept_done | prompts_in_jsonl(buggy)
    queued, extra = prompts_to_run(ideas, already, args.limit)
    if not queued and extra == 0:
        raise SystemExit(f"Nothing to do. {len(already)} prompts already in {out}.")

    stats = Stats(args.jobs, args.limit)
    stats.out_path = out
    stats.buggy_path = buggy
    seed_feed(stats, out, "kept")
    seed_feed(stats, buggy, "buggy")
    stats.resumed = len(kept_done)
    stats.extra_left = 0 if extra < 0 else extra
    STATS = stats
    gate = Gate(args.jobs)
    GATE = gate
    LOOP = asyncio.get_running_loop()
    try:
        serve_monitor(args.port)
    except OSError as exc:
        raise SystemExit(f"Could not listen on 127.0.0.1:{args.port} ({exc}).") from exc
    print(f"monitor http://127.0.0.1:{args.port}/", flush=True)

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for prompt in queued:
        queue.put_nowait(prompt)
    seen = set(already)
    seen.update(subject_key(item) for item in already if subject_key(item))
    seen.update(normalize_prompt(p) for p in queued)
    seen.update(subject_key(p) for p in queued if subject_key(p))
    extra_left = extra
    expand_lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=timeout) as client:
        model = await ensure_model(client, args.server, args.model)
        stats.model = model
        url = args.server.rstrip("/") + "/v1/chat/completions"
        print(f"model {model}  jobs {args.jobs}  queued {len(queued)}  invent {extra}", flush=True)

        pool = ProcessPoolExecutor(max_workers=CHROME_WORKERS)

        async def write_keep(prompt: str, html: str) -> None:
            row = pack_row(prompt, html)
            append_jsonl(out, row)
            if args.into_added:
                append_jsonl(added, row)
            with stats.lock:
                stats.kept += 1
            stats.push_example(prompt, html)
            stats.push_feed(prompt, "kept", "", html, row["sha"], 1, page=html)

        async def write_drop(
            prompt: str,
            reason: str,
            shown: str = "",
            page: str = "",
            report: dict | None = None,
        ) -> None:
            # One shot. The buggy set is a separate trainer file. No repair.
            code = page or shown or ""
            row = pack_buggy_row(prompt, code, reason, report)
            append_jsonl(buggy, row)
            with stats.lock:
                stats.dropped += 1
                stats.last_error = reason[:240]
            stats.push_feed(
                prompt, "buggy", row["bug"]["report"], code, row["sha"], 1, page=page,
            )

        async def generate(messages: list[dict[str, str]], max_tokens: int) -> str:
            return await complete(client, url, model, messages, stats, max_tokens)

        async def chrome_one(html: str) -> dict:
            with stats.lock:
                stats.testing += 1
            try:
                return await run_chrome(pool, html)
            finally:
                with stats.lock:
                    stats.testing = max(0, stats.testing - 1)

        async def generate_held(messages: list[dict[str, str]], max_tokens: int = MAX_TOKENS) -> str:
            # The cap covers model calls only, so Chrome does not hold a stream slot.
            await gate.acquire()
            with stats.lock:
                stats.inflight = gate.inflight
            try:
                return await generate(messages, max_tokens)
            finally:
                await gate.release()
                with stats.lock:
                    stats.inflight = gate.inflight

        async def produce(
            prompt: str, messages: list[dict[str, str]],
        ) -> tuple[str | None, str, str, dict | None]:
            try:
                text = await generate_held(messages)
            except httpx.HTTPError as exc:
                detail = str(exc).strip() or type(exc).__name__
                return None, f"request failed: {detail}", "", None
            html = extract_html(text)
            err = structure_error(html)
            if err:
                return html, err, text, None
            assert html is not None
            if row_tokens(prompt, html) > TOKEN_BUDGET:
                return html, "over token budget", text, None
            report = await chrome_one(html)
            ok, reason = decide_keep(html, report, prompt)
            return html, "" if ok else reason, text, report

        async def handle(prompt: str) -> None:
            # One shot. A failed page is stored in the buggy set. No repair call.
            html, reason, raw, report = await produce(prompt, gen_messages(prompt))
            if not reason and html:
                await write_keep(prompt, html)
            else:
                await write_drop(
                    prompt,
                    reason or "dropped",
                    shown=html or raw,
                    page=html or "",
                    report=report,
                )

        async def expand_once() -> int:
            nonlocal extra_left
            if extra_left == 0:
                return 0
            async with expand_lock:
                if extra_left == 0:
                    return 0
                sample = random.sample(list(seen), k=min(30, len(seen))) if seen else []
                try:
                    text = await generate_held(invent_messages(sample), 600)
                except httpx.HTTPError as exc:
                    with stats.lock:
                        stats.last_error = f"idea request failed: {exc}"[:240]
                    return 0
                fresh = parse_idea_lines(text, seen)
                room = extra_left if extra_left > 0 else len(fresh)
                took = 0
                for prompt in fresh:
                    if took >= room:
                        break
                    seen.add(normalize_prompt(prompt))
                    subject = subject_key(prompt)
                    if subject:
                        seen.add(subject)
                    queue.put_nowait(prompt)
                    took += 1
                if extra_left > 0:
                    extra_left -= took
                    with stats.lock:
                        stats.extra_left = extra_left
                return took

        busy = 0

        async def worker() -> None:
            nonlocal busy
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                busy += 1
                try:
                    try:
                        await handle(item)
                    except Exception as exc:
                        await write_drop(item, f"failed: {exc}")
                finally:
                    busy -= 1
                    queue.task_done()
                    with stats.lock:
                        stats.queue = queue.qsize()

        workers = [asyncio.create_task(worker()) for _ in range(max(max(JOB_CHOICES), args.jobs))]
        empty_invents = 0
        try:
            while True:
                with stats.lock:
                    stats.queue = queue.qsize()
                if extra_left != 0 and queue.qsize() < gate.n:
                    took = await expand_once()
                    if took == 0:
                        empty_invents += 1
                        # Three empty replies means stop asking and finish what is queued.
                        if empty_invents >= 3 and extra_left > 0:
                            extra_left = 0
                            with stats.lock:
                                stats.extra_left = 0
                                stats.last_error = "model returned no new prompts"
                        await asyncio.sleep(2)
                        continue
                    empty_invents = 0
                if extra_left == 0 and queue.empty() and busy == 0 and gate.inflight == 0:
                    break
                await asyncio.sleep(0.3)
        finally:
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers, return_exceptions=True)
            pool.shutdown(wait=False, cancel_futures=True)
            stats.running = False
            print(f"done kept {stats.kept} dropped {stats.dropped} -> {out}", flush=True)


def default_out() -> Path:
    root = Path(os.environ.get("LORA_ROOT", "~/MLX_Models/html_game_sft")).expanduser()
    return root / "jsonl" / "synth.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic HTML training rows via oMLX.")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="Generations in flight (default 12).")
    p.add_argument("--limit", type=int, default=3000, help="Prompts to attempt. 0 keeps inventing.")
    p.add_argument("--ideas", type=Path, default=Path(__file__).with_name("ideas_3000.txt"))
    p.add_argument("--server", default=os.environ.get("MLX_SERVER_URL", "http://127.0.0.1:8000"))
    p.add_argument("--model", default=os.environ.get("MLX_MODEL", DEFAULT_MODEL))
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out", type=Path, default=None, help="JSONL path. Default $LORA_ROOT/jsonl/synth.jsonl.")
    p.add_argument("--into-added", action="store_true", help="Also append kept rows to added.jsonl.")
    args = p.parse_args(argv)
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    if args.out is None:
        args.out = default_out()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
