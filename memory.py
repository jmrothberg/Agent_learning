"""Persistent cross-session memory for the coding agent.

Layout under games/memory/:

    skeletons/                 hand-curated and accumulated skeletons
        canvas_basic.html      the default RAF + DPR + input loop
        ...                    (winning past games get added with goal-derived names)
    mistakes.jsonl             append-only log of {error_signature, fix_summary}
    goals/                     index of past sessions
        20260503_161305/
            goal.txt
            best.html          a copy of the session's best.html if it ever passed
            outcome.json       {goal, model, iters, ok, last_report_summary, ...}

What this module does NOT do: vectors, embeddings, ML. The retrieval here is
deliberately keyword/Jaccard-based — local-only, zero deps, fast, and good
enough for "find me past games that look like 'snake'". You can always
upgrade to a sentence-transformer later by replacing `_score_similarity`.

The agent uses two memory operations per session:

  1. `retrieve_skeleton(goal)` — returns ONE HTML skeleton to seed iter 1
     with. Either the bundled default or a similar past winning game.

  2. `retrieve_mistakes(error_signature, k=3)` — small bullet list of
     "model has done this before, the fix was X". Goes into the diagnose
     prompt so the model sees its own past hindsight.

After each session:

  3. `record_outcome(...)` — copies best.html, writes outcome.json.
  4. `record_mistake(error_signature, fix_summary)` — append to mistakes.jsonl.

All file I/O is best-effort: memory must NEVER crash the agent.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Local alias so the retrieve() formula reads cleanly. Keeps the function's
# scoring expression close to its docstring.
_tanh = math.tanh


# Bundled default skeletons. Kept inline (not sibling files) so a fresh
# checkout has a working seed even before any sessions ran.
#
# DEFAULT_SKELETON  — the original ~80-line scaffold. Stable for v0.
# CANVAS_SKELETON_V2 — a denser ~250-line scaffold that pre-empts many
#                     playbook bullets in the seed itself: focus-aware
#                     key state with blur clearing; dt with 0.1s cap and
#                     resume-time reset; DPR with resize handler; layered
#                     draw (bg/entities/fx/hud); pointer-events HUD;
#                     try/catch frame body; lazy AudioContext on first
#                     gesture; restart that cancels pending RAF and zeroes
#                     state. Switch via skeleton_mode="default_v2" when
#                     you want to test "what if every game starts from
#                     a near-bug-free scaffold?".
DEFAULT_SKELETON_NAME = "canvas_basic.html"
CANVAS_SKELETON_V2_NAME = "canvas_basic_v2.html"
DEFAULT_SKELETON = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Game</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#0b1020; color:#e7ecff;
    font:16px/1.4 system-ui,sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#10162e; border-radius:12px; box-shadow:0 10px 40px #0008;
    max-width:96vw; max-height:90vh; touch-action:none; }
  #hud { position:fixed; top:12px; left:12px; background:#0008; padding:8px 12px;
    border-radius:8px; pointer-events:none; }
  #help { position:fixed; bottom:12px; left:12px; opacity:.75; font-size:13px; }
  #modal { position:fixed; inset:0; display:none; place-items:center;
    background:#000a; backdrop-filter:blur(4px); }
  #modal .card { background:#1a2348; padding:24px 28px; border-radius:14px;
    text-align:center; box-shadow:0 20px 60px #000a; }
  button { background:#3b62ff; color:#fff; border:0; padding:10px 18px;
    border-radius:8px; font-size:15px; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<div id="hud">Score: <span id="score">0</span></div>
<div id="help">Arrows/WASD to move, Space to act, P to pause</div>
<div id="modal"><div class="card">
  <h2 id="endTitle">Game Over</h2>
  <p id="endMsg">Final score: <span id="endScore">0</span></p>
  <button id="restart">Play again</button>
</div></div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const scoreEl = document.getElementById("score");
  const modal = document.getElementById("modal");
  const endScoreEl = document.getElementById("endScore");
  function fit() {
    const dpr = Math.min(window.devicePixelRatio||1, 2);
    const w = 800, h = 600;
    cvs.width = w*dpr; cvs.height = h*dpr;
    cvs.style.width = w+"px"; cvs.style.height = h+"px";
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  fit(); addEventListener("resize", fit);

  const keys = Object.create(null), pressed = Object.create(null);
  const KEYMAP = { ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right",
    KeyW:"up", KeyS:"down", KeyA:"left", KeyD:"right", Space:"act", KeyP:"pause" };
  addEventListener("keydown", e => {
    const k = KEYMAP[e.code]; if (!k) return;
    if (!keys[k]) pressed[k] = true;
    keys[k] = true;
    if (["up","down","left","right","act"].includes(k)) e.preventDefault();
  }, { passive:false });
  addEventListener("keyup", e => { const k = KEYMAP[e.code]; if (k) keys[k] = false; });
  cvs.addEventListener("pointerdown", e => { pressed.act = true; });

  const state = { score:0, paused:false, over:false, t:0, x:400, y:300 };
  function reset() {
    state.score = 0; state.paused = false; state.over = false; state.t = 0;
    state.x = 400; state.y = 300;
    modal.style.display = "none";
  }
  document.getElementById("restart").onclick = reset;

  function update(dt) {
    if (state.over || state.paused) return;
    state.t += dt;
    const speed = 220;
    if (keys.left)  state.x -= speed*dt;
    if (keys.right) state.x += speed*dt;
    if (keys.up)    state.y -= speed*dt;
    if (keys.down)  state.y += speed*dt;
    state.x = Math.max(20, Math.min(780, state.x));
    state.y = Math.max(20, Math.min(580, state.y));
    if (pressed.act)   state.score += 1;
    if (pressed.pause) state.paused = !state.paused;
  }
  function draw() {
    ctx.clearRect(0,0,800,600);
    ctx.fillStyle = "#7ab6ff";
    ctx.beginPath(); ctx.arc(state.x, state.y, 18, 0, Math.PI*2); ctx.fill();
    if (state.paused) {
      ctx.fillStyle = "#fff"; ctx.font = "32px system-ui";
      ctx.textAlign = "center"; ctx.fillText("PAUSED", 400, 300);
    }
  }
  function gameOver() {
    state.over = true;
    endScoreEl.textContent = state.score;
    modal.style.display = "grid";
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    try {
      update(dt);
      draw();
      scoreEl.textContent = state.score;
    } catch (err) {
      console.error("game crashed:", err);
      gameOver();
    }
    for (const k in pressed) pressed[k] = false;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Larger, bug-hardened scaffold. Pre-empts ~half of the playbook bullets
# by baking the right pattern into the seed itself, so the model only has
# to fill in update/draw/state — not re-discover correct DPR scaling,
# focus-loss handling, or restart cleanup every time. Use via
# skeleton_mode="default_v2" once the v0/v1 baseline is in.
CANVAS_SKELETON_V2 = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Game</title>
<style>
  :root { color-scheme: dark; --bg:#0b1020; --fg:#e7ecff; --panel:#1a2348;
          --accent:#3b62ff; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font:16px/1.4 system-ui,sans-serif; overflow:hidden;
    -webkit-tap-highlight-color:transparent; user-select:none; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#10162e; border-radius:12px; box-shadow:0 10px 40px #0008;
    max-width:96vw; max-height:90vh; touch-action:none; }
  #hud { position:fixed; top:12px; left:12px; background:#0008; padding:8px 12px;
    border-radius:8px; pointer-events:none; font-variant-numeric:tabular-nums; }
  #hud span { margin-right:14px; }
  #help { position:fixed; bottom:12px; left:12px; opacity:.75; font-size:13px;
    pointer-events:none; }
  #modal { position:fixed; inset:0; display:none; place-items:center;
    background:#000a; backdrop-filter:blur(4px); }
  #modal .card { background:var(--panel); padding:24px 28px; border-radius:14px;
    text-align:center; box-shadow:0 20px 60px #000a; min-width:240px; }
  #modal h2 { margin:0 0 8px; }
  #pauseTag { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    background:#000c; padding:14px 22px; border-radius:10px; display:none;
    font-size:24px; pointer-events:none; }
  #errOverlay { position:fixed; inset:auto 12px 12px 12px; max-height:30%;
    overflow:auto; background:#400; color:#fdd; padding:8px 12px; font:12px/1.3 monospace;
    border-radius:6px; display:none; pointer-events:none; }
  button { background:var(--accent); color:#fff; border:0; padding:10px 18px;
    border-radius:8px; font-size:15px; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<div id="hud">
  <span>Score: <b id="score">0</b></span>
  <span>Lives: <b id="lives">3</b></span>
</div>
<div id="help">Arrows / WASD to move · Space to act · P to pause</div>
<div id="pauseTag">PAUSED</div>
<div id="modal"><div class="card">
  <h2 id="endTitle">Game Over</h2>
  <p id="endMsg">Final score: <span id="endScore">0</span></p>
  <button id="restart">Play again</button>
</div></div>
<div id="errOverlay"></div>
<script>
(() => {
  "use strict";

  // ---- DOM refs (cached once) ------------------------------------------
  const cvs   = document.getElementById("c");
  const ctx   = cvs.getContext("2d");
  const W = 800, H = 600;
  const scoreEl = document.getElementById("score");
  const livesEl = document.getElementById("lives");
  const modal   = document.getElementById("modal");
  const endTitle = document.getElementById("endTitle");
  const endScoreEl = document.getElementById("endScore");
  const pauseTag = document.getElementById("pauseTag");
  const errBox  = document.getElementById("errOverlay");

  // ---- DPR scaling — set on init, re-apply on resize -------------------
  function fit() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cvs.width  = W * dpr;
    cvs.height = H * dpr;
    cvs.style.width  = W + "px";
    cvs.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  fit();
  window.addEventListener("resize", fit);

  // ---- Input — held + pressed + released, on window, with preventDefault
  // pressed/released fire ONCE per key transition; cleared at end of frame.
  const KEYMAP = {
    ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right",
    KeyW:"up",    KeyS:"down",     KeyA:"left",     KeyD:"right",
    Space:"act",  Enter:"act",     KeyP:"pause",    Escape:"pause",
  };
  const keys     = Object.create(null);
  const pressed  = Object.create(null);
  const released = Object.create(null);
  function onKey(e, down) {
    const k = KEYMAP[e.code];
    if (!k) return;
    if (down) {
      if (!keys[k]) pressed[k] = true;
      keys[k] = true;
      // Stop arrow keys from scrolling, space from page-jumping etc.
      e.preventDefault();
    } else {
      if (keys[k]) released[k] = true;
      keys[k] = false;
    }
  }
  window.addEventListener("keydown", e => onKey(e, true),  { passive:false });
  window.addEventListener("keyup",   e => onKey(e, false), { passive:false });
  // Mobile / touch — same `act` event as Space.
  cvs.addEventListener("pointerdown", () => { pressed.act = true; }, { passive:true });

  // ---- Focus / visibility — pause on tab-out, clear keys, reset dt -----
  let paused = false;
  function clearKeys() {
    for (const k in keys) keys[k] = false;
    for (const k in pressed) pressed[k] = false;
  }
  window.addEventListener("blur",  () => { paused = true;  clearKeys(); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { paused = true; clearKeys(); }
    else                 { last = performance.now(); }   // reset dt anchor
  });

  // ---- Lazy audio — first user gesture creates / resumes the context ---
  let audio = null;
  function ensureAudio() {
    if (audio) return audio;
    try { audio = new (window.AudioContext || window.webkitAudioContext)(); }
    catch (e) { return null; }
    return audio;
  }
  function unlockAudio() { const a = ensureAudio(); if (a && a.state === "suspended") a.resume(); }
  window.addEventListener("keydown",     unlockAudio, { once:true });
  window.addEventListener("pointerdown", unlockAudio, { once:true });

  // ---- State + restart cleanup -----------------------------------------
  // game-specific globals go here. Replace freely.
  const state = { score: 0, lives: 3, over: false, t: 0,
                  player: { x: W/2, y: H/2, r: 12 } };
  let rafHandle = 0;
  function reset() {
    state.score = 0; state.lives = 3; state.over = false; state.t = 0;
    state.player.x = W/2; state.player.y = H/2;
    paused = false;
    modal.style.display = "none";
    errBox.style.display = "none"; errBox.textContent = "";
    if (rafHandle) cancelAnimationFrame(rafHandle);
    last = performance.now();
    rafHandle = requestAnimationFrame(frame);
  }
  document.getElementById("restart").addEventListener("click", reset);

  // ---- Game logic — replace these with your update/draw ----------------
  function update(dt) {
    if (state.over) return;
    if (pressed.pause) paused = !paused;
    if (paused) return;
    state.t += dt;
    const speed = 220;
    const p = state.player;
    if (keys.left)  p.x -= speed * dt;
    if (keys.right) p.x += speed * dt;
    if (keys.up)    p.y -= speed * dt;
    if (keys.down)  p.y += speed * dt;
    p.x = Math.max(p.r, Math.min(W - p.r, p.x));
    p.y = Math.max(p.r, Math.min(H - p.r, p.y));
    if (pressed.act) state.score += 1;
  }
  function draw() {
    // Layer 0 — clear / background
    ctx.clearRect(0, 0, W, H);
    // Layer 1 — entities (drawn first so HUD goes on top)
    const p = state.player;
    ctx.fillStyle = "#7ab6ff";
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
    // Layer 2 — effects (particles, etc.) — none in skeleton
    // Layer 3 — HUD overlays (in-canvas; outer DOM HUD is its own layer)
  }
  function gameOver(reason) {
    state.over = true;
    endTitle.textContent = reason || "Game Over";
    endScoreEl.textContent = state.score;
    modal.style.display = "grid";
  }

  // ---- Frame loop with try/catch + dt cap + resume reset ---------------
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    try {
      update(dt);
      draw();
      // Update DOM HUD (cheap; tabular-nums prevents reflow jitter).
      scoreEl.textContent = state.score;
      livesEl.textContent = state.lives;
      pauseTag.style.display = paused && !state.over ? "block" : "none";
    } catch (err) {
      console.error("game crashed:", err);
      errBox.textContent = "crash: " + (err && err.message || err);
      errBox.style.display = "block";
      // Continue the loop so the user can see the error overlay.
    } finally {
      // Clear one-shot inputs at frame end.
      for (const k in pressed)  pressed[k]  = false;
      for (const k in released) released[k] = false;
    }
    rafHandle = requestAnimationFrame(frame);
  }
  rafHandle = requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Tokens we strip when computing similarity. Mostly stop-words plus generic
# game-domain words that don't help discriminate (e.g. "game" matches every
# past entry and so adds no signal).
_STOPWORDS: set[str] = {
    "a", "an", "and", "the", "of", "for", "in", "on", "with", "to", "by",
    "is", "it", "or", "as", "at", "be", "this", "that",
    "game", "make", "build", "create", "simple", "small",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s) if t.lower() not in _STOPWORDS]


def _score_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Weighted Jaccard on tokens. Identical = 1.0, disjoint = 0.0.

    Counter intersection rather than set intersection so "snake snake game"
    matches another snake reference more than a single mention.
    """
    if not a_tokens or not b_tokens:
        return 0.0
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


@dataclass
class SkeletonHit:
    """A skeleton candidate returned by retrieval."""

    name: str
    html: str
    score: float
    source_goal: str | None     # the original goal this skeleton came from, if any


@dataclass
class MistakeHit:
    """A past mistake / fix pairing surfaced for the diagnose prompt."""

    error_signature: str
    fix_summary: str
    score: float


class GameMemory:
    """Filesystem-backed memory for the agent.

    All paths are computed from `root` (default games/memory/). The directory
    is created lazily on first write so a fresh checkout works without setup.
    """

    def __init__(self, root: str | Path = "games/memory"):
        self.root = Path(root)
        self.skeletons_dir = self.root / "skeletons"
        self.goals_dir = self.root / "goals"
        self.mistakes_path = self.root / "mistakes.jsonl"

    # --- bootstrap ---------------------------------------------------------

    def ensure(self) -> None:
        """Create directory layout and seed the default skeleton if missing.

        Cheap to call repeatedly; agent constructor calls it once.
        """
        try:
            self.skeletons_dir.mkdir(parents=True, exist_ok=True)
            self.goals_dir.mkdir(parents=True, exist_ok=True)
            default = self.skeletons_dir / DEFAULT_SKELETON_NAME
            if not default.exists():
                default.write_text(DEFAULT_SKELETON, encoding="utf-8")
        except Exception:
            # If memory is broken (read-only fs etc) the agent should still
            # run — retrieval just returns empty results.
            pass

    # --- skeleton retrieval ------------------------------------------------

    def retrieve_skeleton(self, goal: str) -> SkeletonHit:
        """Pick the best skeleton for a new goal.

        Strategy: among all skeletons (default + past winning games), pick
        the one whose tagged source goal has the highest token-Jaccard with
        the current goal. Falls back to the bundled default with score 0.0.
        """
        self.ensure()
        goal_toks = _tokenize(goal)

        best: SkeletonHit | None = None
        for path in sorted(self.skeletons_dir.glob("*.html")):
            try:
                html = path.read_text(encoding="utf-8")
            except Exception:
                continue
            # Past-game skeletons get a sidecar .json with the source goal.
            sidecar = path.with_suffix(".json")
            source_goal: str | None = None
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    source_goal = meta.get("goal")
                except Exception:
                    source_goal = None

            if source_goal:
                score = _score_similarity(goal_toks, _tokenize(source_goal))
            elif path.name == DEFAULT_SKELETON_NAME:
                # Default is a no-op match — only picked if nothing else hits.
                score = 0.0
            else:
                # Manually-added skeleton with no sidecar — give it tiny
                # bonus over the default so curated extras win ties.
                score = 0.05

            hit = SkeletonHit(
                name=path.name, html=html, score=score, source_goal=source_goal
            )
            if best is None or hit.score > best.score:
                best = hit

        if best is not None:
            return best
        # Fully empty memory dir — return DEFAULT_SKELETON in-memory.
        return SkeletonHit(
            name=DEFAULT_SKELETON_NAME, html=DEFAULT_SKELETON, score=0.0,
            source_goal=None,
        )

    # --- mistake retrieval -------------------------------------------------

    def retrieve_mistakes(self, error_signature: str, k: int = 3) -> list[MistakeHit]:
        """Look up past mistakes whose signature is similar to this one.

        Used by the diagnose prompt to give the model "you've seen this
        before, here's what worked" hints. Empty list if no memory yet.
        """
        if not self.mistakes_path.exists():
            return []
        sig_toks = _tokenize(error_signature)
        if not sig_toks:
            return []

        hits: list[MistakeHit] = []
        try:
            with self.mistakes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    sig = rec.get("error_signature", "")
                    fix = rec.get("fix_summary", "")
                    if not sig or not fix:
                        continue
                    s = _score_similarity(sig_toks, _tokenize(sig))
                    if s > 0.15:  # threshold to avoid noise
                        hits.append(MistakeHit(
                            error_signature=sig, fix_summary=fix, score=s,
                        ))
        except Exception:
            return []

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # --- recording ---------------------------------------------------------

    def record_mistake(self, error_signature: str, fix_summary: str) -> None:
        """Append one mistake/fix pair. Idempotent at the user level (we'll
        get duplicates on retries; that's intentional — frequency is signal).
        """
        if not error_signature.strip() or not fix_summary.strip():
            return
        self.ensure()
        try:
            with self.mistakes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "error_signature": error_signature[:500],
                    "fix_summary": fix_summary[:500],
                }) + "\n")
        except Exception:
            pass

    def record_outcome(
        self,
        session_id: str,
        goal: str,
        model: str,
        iterations: int,
        ok: bool,
        best_html_path: Path | None,
        last_report_summary: str,
    ) -> Path | None:
        """Snapshot the outcome of a session under goals/<session_id>/.

        If the session ended OK and produced a best.html, we ALSO copy that
        HTML into skeletons/ with a sidecar so future similar goals can
        retrieve it as a starting point.
        """
        self.ensure()
        out_dir = self.goals_dir / session_id
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "goal.txt").write_text(goal, encoding="utf-8")
            (out_dir / "outcome.json").write_text(
                json.dumps({
                    "session_id": session_id,
                    "goal": goal,
                    "model": model,
                    "iterations": iterations,
                    "ok": ok,
                    "last_report_summary": last_report_summary[:1000],
                    "ts": datetime.utcnow().isoformat() + "Z",
                }, indent=2),
                encoding="utf-8",
            )
            if ok and best_html_path is not None and best_html_path.exists():
                copy_path = out_dir / "best.html"
                try:
                    shutil.copy2(best_html_path, copy_path)
                except Exception:
                    pass
                # Promote to a skeleton — sidecar carries the goal so future
                # retrieval can match on it. Name uses session_id to avoid
                # collisions; goals_dir keeps the human-readable copy too.
                skel_name = f"won_{session_id}.html"
                skel_path = self.skeletons_dir / skel_name
                try:
                    shutil.copy2(best_html_path, skel_path)
                    skel_path.with_suffix(".json").write_text(
                        json.dumps({"goal": goal, "session_id": session_id}, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            return out_dir
        except Exception:
            return None


def signature_for_report(report: dict[str, Any]) -> str:
    """Build a short, stable signature from a test report for memory keying.

    We deliberately ignore line numbers, IDs, and other variable parts so
    "the same bug" produces the same signature across runs.
    """
    bits: list[str] = []
    for e in (report.get("errors") or [])[:3]:
        bits.append(_strip_volatile(str(e)))
    for s in (report.get("soft_warnings") or [])[:3]:
        bits.append(_strip_volatile(str(s)))
    if report.get("frozen_canvas") is True:
        bits.append("FROZEN")
    it = report.get("input_test") or {}
    if it.get("ran") and it.get("any_change") is False:
        bits.append("INPUT_DEAD")
    return " | ".join(bits)[:500]


_VOLATILE_RE = re.compile(
    r"\b\d+\b|"                         # numbers (line numbers, byte counts)
    r"file://\S+|"                      # absolute file urls
    r"@\d+:\d+|"                        # source positions like @123:45
    r"0x[0-9a-fA-F]+"                   # hex addresses
)


def _strip_volatile(s: str) -> str:
    return _VOLATILE_RE.sub("N", s)


# ===========================================================================
# Playbook — accumulated, structured rules of thumb the agent retrieves at
# inference time. Separate file from mistakes.jsonl: mistakes are
# error-signature keyed retrospective fix notes; the playbook is forward-
# looking guidance the agent uses while planning and building.
#
# Inspired by ACE (arXiv 2510.04618): bullets carry helpful/harmful counters
# so the offline learner can prune rules that fire on failed runs and reward
# rules that fire on passing ones, without context-collapsing rewrites.
# ===========================================================================


PLAYBOOK_FILENAME = "playbook.jsonl"


@dataclass
class Bullet:
    """A single playbook entry: a one-paragraph rule with metadata.

    `tags` are short strings used both for retrieval (match against goal +
    code) and for organization. `helpful`/`harmful` are running counters
    updated by the offline learner.
    """

    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    helpful: int = 0
    harmful: int = 0
    source: str = "seed"            # 'seed' | 'learned' | session id
    created_at: str = ""

    def score(self) -> int:
        """Net usefulness — used as a tiebreaker in retrieval and for pruning."""
        return self.helpful - self.harmful

    def to_jsonl(self) -> str:
        return json.dumps({
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "source": self.source,
            "created_at": self.created_at,
        }, ensure_ascii=False)


@dataclass
class BulletHit:
    """A retrieval result — bullet plus the retrieval score."""

    bullet: Bullet
    score: float


# Hand-curated seed bullets, distilled from the OpenGame paper, the
# Macklon canvas-bug taxonomy (arXiv 2201.07351), JS13k post-mortems, and
# the SOTA-prompt mining of Aider/Cline/Bolt. Each bullet is one paragraph
# the model can drop directly into an iteration's reasoning. Tags drive
# retrieval, so they're chosen to match the words a goal/code is likely
# to use ("ship", "thrust", "rotation" all on the asteroids bullet).
SEED_BULLETS: list[Bullet] = [
    Bullet(
        id="rotation-thrust-vector",
        content=(
            "When applying thrust to a rotatable ship/character, compute "
            "velocity from its facing angle: vx = Math.cos(angle) * speed, "
            "vy = Math.sin(angle) * speed. NEVER use plain world-axis dx/dy. "
            "Canvas y grows downward, so 'forward at angle 0' means +x with "
            "no y change; rotating CCW by π/2 (-Math.PI/2 from +x) points up."
        ),
        tags=["ship", "thrust", "rotation", "asteroids", "angle", "spaceship",
              "movement", "physics"],
    ),
    Bullet(
        id="asteroid-irregular-polygons",
        content=(
            "Asteroids must be irregular polygons, not perfect circles. "
            "Pre-generate ≥8 vertices at angles 0..2π with per-vertex jitter "
            "of radius * (0.7..1.3); store the offsets per asteroid and draw "
            "with ctx.beginPath / lineTo / closePath. Drawing arc(0,0,r,0,2π) "
            "ships a circle and feels wrong."
        ),
        tags=["asteroids", "shape", "polygon", "drawing", "rocks"],
    ),
    Bullet(
        id="raf-must-start",
        content=(
            "Animation requires BOTH (a) a frame() function whose final "
            "statement is requestAnimationFrame(frame), and (b) at least one "
            "initial requestAnimationFrame(frame) call to kick the loop off. "
            "Missing either leaves a static canvas — a common silent failure."
        ),
        tags=["raf", "animation", "loop", "render"],
    ),
    Bullet(
        id="ecode-not-ekey",
        content=(
            "Keyboard input MUST use e.code values like 'ArrowLeft', 'KeyW', "
            "'Space'. Don't use e.key (varies with keyboard layout) or "
            "e.keyCode (deprecated, missing on some browsers). Mapping: "
            "{ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', "
            "ArrowRight:'right', KeyW:'up', KeyS:'down', KeyA:'left', "
            "KeyD:'right', Space:'fire'}."
        ),
        tags=["input", "keyboard", "keys", "controls"],
    ),
    Bullet(
        id="preventdefault-game-keys",
        content=(
            "Arrow keys and Space scroll the page by default and steal focus. "
            "Inside keydown, call e.preventDefault() for any code in your "
            "game key set. Attach the listener to window with "
            "{passive:false} so preventDefault actually takes effect."
        ),
        tags=["input", "keyboard", "preventdefault", "scroll", "controls"],
    ),
    Bullet(
        id="window-listener-not-canvas",
        content=(
            "Attach keydown/keyup listeners to window (or document), NOT to "
            "the <canvas> element. Canvas isn't focused by default and won't "
            "receive keys. If you really need canvas focus, set tabIndex=0 "
            "and call .focus() after creation."
        ),
        tags=["input", "keyboard", "focus", "canvas"],
    ),
    Bullet(
        id="clear-keys-on-blur",
        content=(
            "Add window.addEventListener('blur', () => for (k in keys) "
            "keys[k]=false) to clear held keys on focus loss; otherwise "
            "Alt-Tab leaves the player thrusting forever. Same on "
            "document.addEventListener('visibilitychange', ...) when hidden."
        ),
        tags=["input", "keyboard", "focus", "blur", "stuck"],
    ),
    Bullet(
        id="dt-physics-with-cap",
        content=(
            "Use delta-time physics: dt = Math.min(0.05, (now - last)/1000); "
            "last = now. Multiply movement by dt so speed is frame-rate "
            "independent. Cap dt to ~0.05s before stepping — without the cap, "
            "tab-switch triggers a 5+ second dt and objects tunnel through "
            "everything (the 'spiral of death')."
        ),
        tags=["physics", "frame-rate", "dt", "delta-time", "loop", "movement"],
    ),
    Bullet(
        id="canvas-dpr-scaling",
        content=(
            "On retina/HiDPI displays, the canvas is blurry without DPR "
            "scaling. Set: const dpr = Math.min(window.devicePixelRatio||1, "
            "2); cvs.width = cssW * dpr; cvs.height = cssH * dpr; "
            "cvs.style.width = cssW + 'px'; cvs.style.height = cssH + 'px'; "
            "ctx.setTransform(dpr,0,0,dpr,0,0). Re-apply on 'resize'."
        ),
        tags=["canvas", "dpr", "retina", "scaling", "blurry"],
    ),
    Bullet(
        id="visible-hud-score",
        content=(
            "Score and controls must be visible at all times. Either render "
            "a #hud div with position:fixed (top-left) and update its text "
            "from the game state, or fillText('Score: '+score, 12, 24) every "
            "frame inside draw(). A game without a visible score is judged "
            "broken even if it works."
        ),
        tags=["hud", "score", "ui", "instructions", "visible"],
    ),
    Bullet(
        id="game-over-reachable",
        content=(
            "There MUST be a way to lose AND a way to restart. Use a state "
            "machine with state.over flag; show a modal on game over; bind "
            "Space or a button to reset the state and restart the loop. "
            "Cancel any pending animation frames on restart."
        ),
        tags=["game-state", "game-over", "restart", "modal", "lose"],
    ),
    Bullet(
        id="image-load-race",
        content=(
            "drawImage on an Image whose .src was just set paints nothing — "
            "loading is async. Either: (a) wait for img.decode() / onload "
            "before starting the loop, or (b) generate sprites procedurally "
            "via offscreen canvas at boot. Don't drawImage in the same tick "
            "you assign .src."
        ),
        tags=["image", "load", "drawimage", "race", "async", "sprite"],
    ),
    Bullet(
        id="touch-pointer-events",
        content=(
            "For mobile/touch, bind pointerdown/pointermove/pointerup OR "
            "touchstart/touchmove/touchend (with preventDefault) to the same "
            "action handlers as keys. Use Pointer Events when possible — "
            "they unify mouse+touch+pen and avoid the touchstart→mousedown "
            "double-fire."
        ),
        tags=["input", "mobile", "touch", "pointer", "controls"],
    ),
    Bullet(
        id="visibility-pause",
        content=(
            "On document.visibilitychange when document.hidden, pause the "
            "loop AND reset 'last' time on resume. Without this, dt explodes "
            "to seconds when returning to the tab and physics fly off."
        ),
        tags=["pause", "visibility", "tab", "focus"],
    ),
    Bullet(
        id="js-modulo-negative",
        content=(
            "JavaScript modulo on negative numbers returns a negative result: "
            "(-1) % 800 === -1, not 799. For wrap-around use "
            "((x % w) + w) % w."
        ),
        tags=["math", "modulo", "wrap", "toroidal"],
    ),
    Bullet(
        id="z-order-layers",
        content=(
            "Drawing order matters. Inside draw(): clear → background → "
            "entities → effects/particles → HUD overlay. Don't draw HUD "
            "inside the entity loop or it will be obscured by sprites drawn "
            "later in the same loop."
        ),
        tags=["render", "draw", "z-order", "layers", "hud"],
    ),
    Bullet(
        id="frame-trycatch",
        content=(
            "Wrap the frame() body in try/catch that logs to console.error. "
            "One uncaught exception inside requestAnimationFrame silently "
            "stops the loop with no visible error in the page — the game "
            "looks frozen with no diagnosis."
        ),
        tags=["error-handling", "trycatch", "raf", "loop"],
    ),
    Bullet(
        id="restart-cleanup",
        content=(
            "On restart, cancel pending animation frames "
            "(cancelAnimationFrame) and don't re-add the same event "
            "listeners — without cleanup, listeners stack and each restart "
            "doubles input speed. Prefer mutating in-place over re-binding."
        ),
        tags=["restart", "cleanup", "listeners", "leak"],
    ),
    Bullet(
        id="snake-grid-tick",
        content=(
            "Snake/grid games move in fixed-cell ticks, not free RAF motion. "
            "Use an accumulator: tickAccum += dt; if (tickAccum >= TICK) { "
            "step(); tickAccum -= TICK; }. The head must advance EXACTLY one "
            "cell per step; turning sets the next direction, applied at the "
            "next step."
        ),
        tags=["snake", "grid", "tick", "discrete", "step"],
    ),
    Bullet(
        id="breakout-ball-launch",
        content=(
            "Breakout's ball must launch with NON-ZERO dy. A common bug is "
            "dy=0 on launch — ball travels horizontally forever, paddle "
            "never sees it. Initial state: dx = ±2, dy = -3 (negative = up). "
            "On paddle hit, mirror dy and bias dx by hit-position offset."
        ),
        tags=["breakout", "ball", "physics", "paddle"],
    ),
    Bullet(
        id="single-html-file-cdn",
        content=(
            "Output is a SINGLE .html file containing inline <style> and "
            "<script>. CDN libraries via <script src='https://...'> are "
            "allowed (Phaser, three.js, kontra). No bundlers, no node_modules, "
            "no separate .js or .css files."
        ),
        tags=["build", "single-file", "html", "cdn", "structure"],
    ),
    Bullet(
        id="no-localstorage-init",
        content=(
            "Don't read/write localStorage on first load without a "
            "feature-detect — some headless contexts throw "
            "SecurityError. try { localStorage.getItem('x') } catch (_) {} "
            "before relying on it."
        ),
        tags=["localstorage", "persistence", "init"],
    ),
    Bullet(
        id="hud-pointer-events",
        content=(
            "If the HUD is a positioned div over the canvas, set "
            "pointer-events:none on it so clicks/touches pass through to "
            "the game. Otherwise the HUD silently swallows input."
        ),
        tags=["hud", "ui", "pointer-events", "input"],
    ),
    Bullet(
        id="audio-autoplay-gesture",
        content=(
            "Browsers block autoplay until a user gesture. If using "
            "AudioContext, create it on first keydown/click: "
            "if (!audio) { audio = new (window.AudioContext||...)(); } "
            "audio.resume()."
        ),
        tags=["audio", "sound", "autoplay", "gesture"],
    ),
    Bullet(
        id="phaser-three-cdn-defer",
        content=(
            "When loading Phaser/three.js via CDN, place the <script "
            "src='...'> in <head> with `defer`, and put your game code in a "
            "<script> AFTER the body — otherwise the library may not exist "
            "when your script runs."
        ),
        tags=["phaser", "three", "cdn", "load", "library"],
    ),
    Bullet(
        id="ui-driven-no-canvas",
        content=(
            "For UI-style requests (todo list, calculator, tic-tac-toe, word "
            "games) prefer DOM elements over canvas. Use <button>, <input>, "
            "<ul>, addEventListener('click', ...). Reserve canvas for games "
            "that genuinely need per-pixel rendering or smooth animation."
        ),
        tags=["ui", "dom", "todo", "calculator", "click", "non-canvas"],
    ),
    Bullet(
        id="aabb-collision",
        content=(
            "For axis-aligned bounding boxes, collision is: "
            "a.x < b.x+b.w && a.x+a.w > b.x && a.y < b.y+b.h && a.y+a.h > b.y. "
            "Always check ALL FOUR conditions; getting one direction wrong "
            "produces ghost-collisions (objects pass through one side only)."
        ),
        tags=["collision", "aabb", "physics", "bbox"],
    ),
    Bullet(
        id="circle-collision",
        content=(
            "Circle-circle collision: const dx=a.x-b.x, dy=a.y-b.y, "
            "rsum=a.r+b.r; if (dx*dx + dy*dy < rsum*rsum) { hit }. "
            "Use squared distance — avoids sqrt in the hot loop."
        ),
        tags=["collision", "circle", "physics"],
    ),
    Bullet(
        id="bullet-pool",
        content=(
            "Bullets/projectiles: pre-allocate a pool and reuse via an "
            "active flag, instead of new Bullet()/array.push every fire. "
            "Garbage collection in the middle of a frame causes hitches "
            "visible as stutter."
        ),
        tags=["bullets", "pool", "performance", "gc", "shooter"],
    ),
    Bullet(
        id="seed-respect",
        content=(
            "When the user provides a seed file, PREFER patches over a full "
            "rewrite. A wholesale rewrite loses structural choices the user "
            "may care about (variable names, layout, styling). Patches keep "
            "their work intact and only change what the goal demands."
        ),
        tags=["seed", "patches", "rewrite", "preserve"],
    ),
    Bullet(
        id="place-entities-at-runtime",
        content=(
            "When a level uses a tile grid (maze, dungeon, room), PLACE "
            "spawn/exit/enemies/pickups at runtime by SCANNING empty cells "
            "in code — do not hand-verify coordinates in your reply. "
            "Snippet: function pickEmpty(map){ while(true){ const "
            "x=1+Math.floor(Math.random()*(W-2)), "
            "y=1+Math.floor(Math.random()*(H-2)); if(map[y][x]===0) "
            "return {x:x+0.5,y:y+0.5}; } }. The trap: enumerating "
            "'(3.5, 9.5): row 9 col 3 = 1 (wall!)' in the reply or in "
            "<think> burns thousands of tokens on validation the runtime "
            "would do in microseconds, and often never reaches the "
            "<html_file> tag at all. If the maze is fixed, also write "
            "spawn/exit lookups as scans, not as literal coordinates."
        ),
        tags=[
            "maze", "grid", "spawn", "placement", "level", "entity",
            "tile", "dungeon", "tilemap", "enemy", "pickup",
            "thinktag-thrash",
        ],
    ),
]


class Playbook:
    """Filesystem-backed playbook of structured rules.

    On first use the file is seeded with SEED_BULLETS; subsequent saves
    persist the merged set. Retrieval is keyword/Jaccard against the goal
    and (optionally) the in-progress code.
    """

    def __init__(self, root: str | Path = "games/memory"):
        self.root = Path(root)
        self.path = self.root / PLAYBOOK_FILENAME

    # --- bootstrap ---------------------------------------------------------

    def ensure(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._save_all(SEED_BULLETS)
        except Exception:
            pass

    def _save_all(self, bullets: list[Bullet]) -> None:
        try:
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for b in bullets:
                    if not b.created_at:
                        b.created_at = datetime.utcnow().isoformat() + "Z"
                    f.write(b.to_jsonl() + "\n")
            tmp.replace(self.path)
        except Exception:
            pass

    def load_all(self) -> list[Bullet]:
        self.ensure()
        out: list[Bullet] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        out.append(Bullet(
                            id=str(rec["id"]),
                            content=str(rec["content"]),
                            tags=list(rec.get("tags") or []),
                            helpful=int(rec.get("helpful") or 0),
                            harmful=int(rec.get("harmful") or 0),
                            source=str(rec.get("source") or "seed"),
                            created_at=str(rec.get("created_at") or ""),
                        ))
                    except Exception:
                        continue
        except Exception:
            return []
        return out

    # --- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        goal: str,
        *,
        code: str = "",
        k: int = 8,
        stage: str = "code",
    ) -> list[BulletHit]:
        """Return up to k bullets most relevant to goal (+ optional code).

        Scoring is a relevance × quality product:

          relevance = weighted Jaccard between (goal+code) tokens and bullet
                      (tags+content) tokens, with tag matches weighted 2x.

          quality  = 1.0 + 0.10 * tanh(b.score() / 5)
                      → bounded in [~0.90, ~1.10], so a +5 bullet gets
                      ~7% boost, a -5 bullet gets ~7% penalty.

        OpenCoder lesson — when collapsing duplicates, prefer the WINNER:
        the quality multiplier is multiplicative on relevance so duplicate-
        relevance bullets reorder by net helpfulness without overwhelming
        the relevance signal.

        `stage` controls how strict we are about quality:
          - "plan" (broad, OpenCoder Stage-1): include all positive-relevance
            hits, even net-harmful ones — exposure to history.
          - "code" (narrow, OpenCoder Stage-2): drop bullets with net score
            ≤ -2, since at code-time we want only validated patterns. This
            mirrors OpenCoder's two-stage SFT (broad first, narrow second).

        Caller is responsible for any further dedup / budget capping
        (see `dedup_hits` and `cap_hits_by_budget`).
        """
        bullets = self.load_all()
        if not bullets:
            return []

        query_toks = _tokenize(goal) + _tokenize(code)
        if not query_toks:
            ordered = sorted(bullets, key=lambda b: (-b.score(), b.id))
            return [BulletHit(b, 0.0) for b in ordered[:k]]

        q_counter = Counter(query_toks)
        hits: list[BulletHit] = []
        for b in bullets:
            # Code-stage filter: drop persistently-harmful bullets so the
            # model isn't exposed to known-bad patterns at coding time.
            if stage == "code" and b.score() <= -2:
                continue

            tag_toks = _tokenize(" ".join(b.tags))
            content_toks = _tokenize(b.content)
            t_counter: Counter = Counter()
            for t in tag_toks:
                t_counter[t] += 2
            for t in content_toks:
                t_counter[t] += 1
            inter = sum((q_counter & t_counter).values())
            union = sum((q_counter | t_counter).values())
            if union <= 0:
                continue
            sim = inter / union
            # Noise floor: random matches (single shared common word
            # like "canvas") cluster at sim < 0.02. Keep this floor
            # low — bench traces show genre-fit bullets like
            # snake-grid-tick scoring ~0.02 against "snake game with
            # arrow keys", so a higher floor drops genuine matches.
            # The bigger lever for noise is the playbook-off default
            # in chat.py/coder.py; this is the second line of defense
            # for users who opt back in via /playbook on.
            if sim < 0.02:
                continue

            # Quality multiplier: bounded ±10% of relevance.
            quality = 1.0 + 0.10 * _tanh(b.score() / 5.0)
            hits.append(BulletHit(b, sim * quality))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # --- delta ops (used by offline learner) -------------------------------

    def add(self, bullet: Bullet) -> None:
        all_b = self.load_all()
        # Replace if id already exists (idempotent ADD).
        all_b = [b for b in all_b if b.id != bullet.id] + [bullet]
        self._save_all(all_b)

    def update_counters(
        self,
        bullet_ids: list[str],
        *,
        helpful_delta: int = 0,
        harmful_delta: int = 0,
    ) -> None:
        all_b = self.load_all()
        idset = set(bullet_ids)
        for b in all_b:
            if b.id in idset:
                b.helpful = max(0, b.helpful + helpful_delta)
                b.harmful = max(0, b.harmful + harmful_delta)
        self._save_all(all_b)

    def remove(self, bullet_id: str) -> None:
        all_b = self.load_all()
        all_b = [b for b in all_b if b.id != bullet_id]
        self._save_all(all_b)


# ===========================================================================
# Shingle dedup + context-budget cap (OpenCoder #5, #2)
# ===========================================================================
#
# OpenCoder's file-level dedup beat repo-level dedup: 75% of files were
# duplicates, and the smaller file-level corpus trained a *better* model.
# At inference the analog is dedup-before-concatenate: when two retrieved
# bullets cover the same idea (smart-quote drift, fuzzy tagging) only the
# higher-quality one belongs in the prompt.
#
# Annealing-mix lesson (OpenCoder): ~16% of training tokens were the
# "high-signal" curated set; removing it collapsed scores. The inference
# analog is a context budget — the high-signal exemplars get a CAPPED
# share of the prompt rather than being allowed to bloat indefinitely.

# 5-gram word shingles, lowercased + alphanum-normalized, dedup pairs
# whose Jaccard similarity exceeds this threshold. 0.85 ≈ "they say
# essentially the same thing"; tuned to be conservative — false positives
# are worse than false negatives because we lose information.
_SHINGLE_DEDUP_THRESHOLD = 0.85
_SHINGLE_N = 5

# Default char budget for a rendered <playbook> block. ~3.6KB ≈ 900 tokens
# at the typical 4 chars/token. With an 8K-16K total context and an HTML
# file inline at fix turns, this lands the playbook at ~5-12% of total —
# OpenCoder's 16% rule taken with a safety margin since the model also
# needs room for its own reply.
_DEFAULT_PLAYBOOK_CHAR_BUDGET = 3600

_NONALNUM = re.compile(r"[^a-z0-9 ]+")


def _shingles(text: str, n: int = _SHINGLE_N) -> set[str]:
    """Return the set of n-gram word shingles for `text`, lowercased.

    Used for dedup similarity. Cheap; allocates O(words) strings.
    """
    words = _NONALNUM.sub(" ", text.lower()).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_hits(
    hits: list[BulletHit],
    *,
    threshold: float = _SHINGLE_DEDUP_THRESHOLD,
) -> list[BulletHit]:
    """Drop near-duplicate bullets, keeping the higher-scoring of each pair.

    Walks `hits` in order (callers usually pass them already sorted by
    retrieval score). For each candidate, compares its content shingles
    against every already-kept bullet; if Jaccard similarity exceeds
    threshold, drop the candidate (it's a near-duplicate of something
    already kept and ranked higher).

    Result preserves the input ordering of the kept bullets.

    OpenCoder pattern: when picking among duplicates, pick by quality.
    The input-ordering convention here means callers control "quality"
    (typically retrieval score, which already folds in helpful-harmful);
    we just enforce the dedup discipline.
    """
    if len(hits) <= 1:
        return list(hits)
    kept: list[BulletHit] = []
    kept_shingles: list[set[str]] = []
    for h in hits:
        sh = _shingles(h.bullet.content)
        if any(_jaccard(sh, k) >= threshold for k in kept_shingles):
            continue
        kept.append(h)
        kept_shingles.append(sh)
    return kept


def cap_hits_by_budget(
    hits: list[BulletHit],
    *,
    char_budget: int = _DEFAULT_PLAYBOOK_CHAR_BUDGET,
) -> list[BulletHit]:
    """Truncate `hits` so the rendered block fits inside `char_budget`.

    OpenCoder's annealing-mix lesson: cap high-signal exemplars at a fixed
    fraction of total context. We approximate that here by char count
    (≈4 chars/token) on the rendered output. Lower-ranked tail is dropped
    first; a single bullet that already exceeds the budget still gets
    included (we never return an empty list when the input was non-empty
    and the first bullet has signal).
    """
    if not hits:
        return hits
    out: list[BulletHit] = []
    used = 0
    # Budget overhead for the wrapping <playbook> tags + header — keep a
    # 200-char headroom so the rendered version really fits.
    header_overhead = 200
    for h in hits:
        # `score=±N` meta + bullet markup + content + newline ≈ len(content) + 30
        line_cost = len(h.bullet.content) + 30
        if used + line_cost > char_budget - header_overhead and out:
            break
        out.append(h)
        used += line_cost
    return out


def render_playbook_block(
    hits: list[BulletHit],
    header: str | None = None,
    *,
    dedup: bool = True,
    char_budget: int = _DEFAULT_PLAYBOOK_CHAR_BUDGET,
    mode: str = "full",
    full_top_n: int = 3,
) -> str:
    """Format retrieved bullets as a compact <playbook> block for the prompt.

    Pipeline: optional dedup → budget cap → render. Both filters are
    on-by-default per the OpenCoder findings (#5 dedup, #2 budget).

    `mode` selects the rendering shape:
      - "full" (default): every bullet kept after cap renders with full
        body. Best when retrieval is already narrow (code stage, K ≤ 3).
      - "hybrid": the top `full_top_n` bullets get full body; the rest
        render as ID + tags only with a hint that the model can emit
        <lookup_bullet>id</lookup_bullet> to fetch the body in the next
        turn. Pi-mono "skills" pattern — advertise breadth, expand on
        demand. Best when retrieval is broad (plan stage, K ≥ 6) so the
        model sees more options without paying for them in tokens.

    Empty / fully-filtered list → empty string (caller should detect and
    skip injection).
    """
    work = list(hits)
    if dedup:
        work = dedup_hits(work)
    work = cap_hits_by_budget(work, char_budget=char_budget)
    if not work:
        return ""
    head = header or (
        "RELEVANT PLAYBOOK ENTRIES — these are accumulated lessons from past "
        "runs, ordered by relevance to your goal. Apply when applicable; "
        "ignore the ones that don't fit."
    )
    lines = ["<playbook>", head, ""]

    if mode == "hybrid":
        full_set = work[:full_top_n]
        summary_set = work[full_top_n:]
    else:
        full_set = work
        summary_set = []

    for h in full_set:
        b = h.bullet
        meta = f" (score={b.score():+d})" if (b.helpful or b.harmful) else ""
        lines.append(f"- [{b.id}]{meta} {b.content}")

    if summary_set:
        lines.append("")
        lines.append(
            "ADDITIONAL PLAYBOOK INDEX (body NOT included to save context — "
            "emit <lookup_bullet>id</lookup_bullet> in your reply to have "
            "any of these expanded into your next turn):"
        )
        for h in summary_set:
            b = h.bullet
            tags = ",".join(b.tags[:5]) if b.tags else "untagged"
            meta = f" score={b.score():+d}" if (b.helpful or b.harmful) else ""
            lines.append(f"- [{b.id}]{meta} tags=[{tags}]")

    lines.append("</playbook>")
    return "\n".join(lines)


def lookup_bullet(playbook: "Playbook", bullet_id: str) -> Bullet | None:
    """Fetch a single bullet from `playbook` by exact ID match.

    Used by the agent's `<lookup_bullet>` handler to resolve on-demand
    skill-style lookups. Returns None if the ID doesn't exist (rather
    than raising, so the agent can fall through gracefully).
    """
    for b in playbook.load_all():
        if b.id == bullet_id:
            return b
    return None
