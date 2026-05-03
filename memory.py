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
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Bundled default skeleton. Same one the system prompt used to embed —
# moved here so the prompt itself can stay short. Kept inline (not a sibling
# file) so a fresh checkout has a working seed even before any sessions ran.
DEFAULT_SKELETON_NAME = "canvas_basic.html"
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
