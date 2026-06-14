"""Persistent cross-session memory for the coding agent.

Layout under memory/:

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
# Below this Jaccard similarity, a past-win skeleton is more likely to
# bias the model toward the wrong structure than to provide a useful
# starting point (see 2026-05-15 DK trace where sim=0.23 forced a
# DK-arcade scaffold onto a DK-with-ramps goal and burned 6 iters).
# Hits below this threshold fall through to the bundled empty
# `canvas_basic.html` template — the model builds clean rather than
# fighting a mismatched scaffold. 0.3 leaves real overlap matches
# (shared mechanic words) untouched while filtering coincidental
# token bleed.
# Recipe→skeleton routing map (added 2026-06-02). The visual-playtest layer
# (find_best_visual_playtest: strong_hooks → applies_keywords overlap) already
# routes 15/21 of the curated prompts with ZERO misroutes, while the skeleton
# Jaccard picker surfaced only ~5/18 scaffolds (24/27 prompts fell to the
# generic v2 canvas). Rather than build a SECOND scorer for skeletons — which
# risks re-deriving the 2D-arcade misroute we just fixed — `retrieve_skeleton`
# reuses the already-correct recipe match and maps it to a skeleton via this
# table. ONLY mechanism-aligned, non-arcade-forbidden pairs are listed; a recipe
# with no safe skeleton here falls through to the existing modality→Jaccard→v2
# logic unchanged. Empirically verified: 0 arcade violations, specialized
# coverage 5→12 of 27 prompts. Keys are visual_playtest ids; values are skeleton
# filenames (without .html).
_RECIPE_TO_SKELETON: dict[str, str] = {
    "canvas-side-scroll-platformer": "canvas_platformer_basic",
    "canvas-vertical-platformer": "canvas_platformer_basic",
    "canvas-racing-perspective": "canvas_mode7_basic",
    "canvas-grid-navigation": "canvas_grid_basic",
    "canvas-vfx-fluid": "canvas_vfx_particles_basic",
    "canvas-3d-first-person": "canvas_3d_basic",
    "canvas-board-game": "canvas_board_turn_basic",
    "canvas-cutscene-qte": "canvas_cutscene_qte_basic",
    # Phase-2 additions (2026-06-02): recipes added for skeleton-only classes
    # that previously mis-routed (cards→vfx outline, physics→top-down) or had no
    # recipe (lit-dungeon, mobile). Each maps to its existing dedicated skeleton.
    "canvas-card-tabletop": "canvas_cards_basic",
    "canvas-physics-projectile": "canvas_physics_basic",
    "canvas-lit-dungeon": "canvas_lit_dungeon_basic",
    "canvas-mobile-touch": "canvas_mobile_basic",
    # NOTE deliberately ABSENT (no dedicated/safe skeleton — fall through to v2):
    # top-down-action, paddle-ball, lane-crossing, puzzle-grid, isometric-tile,
    # overworld-rpg, two-actors-facing, point-and-click, city-builder,
    # space-trading, single-fighter, controllable-player, generic-baseline.
}

_SKELETON_MIN_SIM = 0.3
# Generic skeleton-match tokens (added 2026-05-31). BUNDLED specialized
# scaffolds (canvas_3d / canvas_lit_dungeon / canvas_crawler / canvas_cards /
# canvas_rpg / canvas_board_turn …) used to be EXEMPT from any floor — only
# past-win "won_" files were thresholded. That let a SINGLE incidental shared
# token route a plain 2D arcade goal to a wrong specialized scaffold (measured:
# asteroids/galaga/centipede/qbert -> canvas_3d on "space"/"vector"/"game"/
# "projection"; pong/missile-command -> canvas_lit_dungeon on "light"; breakout
# -> canvas_crawler on "slide"; snake -> canvas_rpg on "grid"; tetris -> cards
# on "grid"/"puzzle"; frogger/monkey-island -> board_turn on "move"/"player"/
# "select"). A raw Jaccard *score* floor can't separate these: a genuine 1-token
# pick (sokoban -> grid on the DISTINCTIVE token "sokoban") scores the same as a
# coincidental 1-token pick (pong -> lit_dungeon on the GENERIC token "light").
# So we gate on token *distinctiveness* instead — a bundled scaffold may only
# win if the goal shares at least one NON-generic token with its sidecar. The
# good picks survive (sokoban/pacman -> grid on "sokoban"/"ghost", 1942 ->
# scrolling on "shooter"); the coincidences fall back to the safe generic v2
# canvas. Modality picks (chess/doom/minecraft/FPS) bypass this entirely via
# _modality_skeleton. Genre-free: these are common rendering / mechanic /
# English words, not subject-matter category names.
_SKELETON_GENERIC_TOKENS: frozenset[str] = frozenset({
    "game", "games", "2d", "3d", "top", "down", "side", "move", "moves",
    "movement", "player", "players", "click", "select", "grid", "light",
    "lights", "lighting", "space", "vector", "puzzle", "slide", "action",
    "arcade", "level", "levels", "score", "simple", "basic", "screen",
    "projection", "coordinate", "control", "controls", "play", "adventure",
    "multi", "mouse", "look", "pointer", "build", "break",
})
# Specialized scaffolds that must clear a HIGHER bar — >= 2 distinctive shared
# tokens — to win via the Jaccard fallback. One distinctive token is too weak to
# commit a flat 2D arcade goal to a 3D / board / dungeon / card scaffold (frogger
# -> board_turn on "cell"; qbert -> voxel on "cube"; tetris -> cards on "drop";
# missile-command -> cards on "mouse"; pong -> 3d on "first"; monkey-island ->
# board_turn on "go"). Their legitimate users either route through
# _modality_skeleton (doom/chess/minecraft) or share several tokens (zelda -> rpg
# on rpg+tile+based). The safe 2D scaffolds (grid/scrolling/platformer/physics/
# mode7/…) keep the >= 1 distinctive bar, so pac-man/sokoban -> grid and 1942 ->
# scrolling still win.
_SKELETON_SPECIALIZED_STRICT: frozenset[str] = frozenset({
    "canvas_3d_basic.html", "canvas_voxel_minecraft_basic.html",
    "canvas_board_turn_basic.html", "canvas_lit_dungeon_basic.html",
    "canvas_crawler_basic.html", "canvas_cards_basic.html",
    "canvas_rpg_basic.html",
})
CANVAS_SKELETON_V2_NAME = "canvas_basic_v2.html"
# v2 sidecar: deliberately generic tokens so it wins as the FALLBACK when no
# modality skeleton matches, NOT so it competes with modality scaffolds. 4/4
# May 20-21 traces (chess, pac, doom, FPS) fell through to the bare v1 default
# at score 0.0 — v2 pre-empts the focus-blur / dt-cap / restart-cleanup /
# DPR / lazy-audio / HUD pointer-events failures those sessions repeatedly hit.
CANVAS_BASIC_V2_SIDECAR = '{"goal": "canvas 2d generic action arcade default fallback"}'
CANVAS_3D_SKELETON_NAME = "canvas_3d_basic.html"
CANVAS_3D_SKELETON_SIDECAR = '{"goal": "3D space vector WebGL three.js coordinate projection game first person perspective"}'

CANVAS_GRID_SKELETON_NAME = "canvas_grid_basic.html"
# Grid sidecar — pac/man/ghost/ghosts added 2026-05-21 after May 20-21
# pac-man trace fell through to canvas_basic.html at score 0.0 because the
# goal "pac man with ghosts" tokenized to ["pac","man","ghosts"], none of
# which matched the existing single-token "pacman". Splitting both spellings
# in the sidecar covers either user phrasing without committing to a genre.
CANVAS_GRID_SKELETON_SIDECAR = '{"goal": "grid continuous tile corridor snap slide sokoban pacman pac man ghost ghosts maze"}'

CANVAS_PLATFORMER_SKELETON_NAME = "canvas_platformer_basic.html"
CANVAS_PLATFORMER_SKELETON_SIDECAR = '{"goal": "platformer gravity jump climbing ladders oneway donkey kong lode runner"}'

CANVAS_SCROLLING_SKELETON_NAME = "canvas_scrolling_basic.html"
CANVAS_SCROLLING_SKELETON_SIDECAR = '{"goal": "scrolling camera viewport parallax offsets scroll horizontal defender scramble side scroller shooter"}'

CANVAS_MODE7_SKELETON_NAME = "canvas_mode7_basic.html"
CANVAS_MODE7_SKELETON_SIDECAR = '{"goal": "mode7 mode 7 perspective ground texture projection 3D rotating kart retro racer fzero f-zero"}'

CANVAS_CRAWLER_SKELETON_NAME = "canvas_crawler_basic.html"
CANVAS_CRAWLER_SKELETON_SIDECAR = '{"goal": "crawler dungeon procedural hack and slash gauntlet multi player wall slide screen clamp boundaries split"}'

CANVAS_MOBILE_SKELETON_NAME = "canvas_mobile_basic.html"
CANVAS_MOBILE_SKELETON_SIDECAR = '{"goal": "mobile touch tablet phone ipad iphone virtual joystick d-pad responsive letterbox pointer events"}'

CANVAS_RPG_SKELETON_NAME = "canvas_rpg_basic.html"
CANVAS_RPG_SKELETON_SIDECAR = '{"goal": "rpg grid tile-based discrete step movement pokemon adventure explorer"}'

CANVAS_CARDS_SKELETON_NAME = "canvas_cards_basic.html"
CANVAS_CARDS_SKELETON_SIDECAR = '{"goal": "cards drag and drop mouse touch board game chess checker solitaire puzzle match snap grid stacking"}'

CANVAS_PHYSICS_SKELETON_NAME = "canvas_physics_basic.html"
CANVAS_PHYSICS_SKELETON_SIDECAR = '{"goal": "physics puzzle bubble shooter angry birds projectile trajectory launch gravity reflection collision stick"}'

CANVAS_VOXEL_MINECRAFT_SKELETON_NAME = "canvas_voxel_minecraft_basic.html"
CANVAS_VOXEL_MINECRAFT_SKELETON_SIDECAR = '{"goal": "voxel 3D grid minecraft cube block chunk terrain procedurally mouse look pointer lock build break Three.js"}'

CANVAS_AR_FLICK_SKELETON_NAME = "canvas_ar_flick_basic.html"
CANVAS_AR_FLICK_SKELETON_SIDECAR = '{"goal": "flick mobile pointer swipe curveball projectile spin gravity throw capture target poke pokeman go ar camera"}'

CANVAS_LIT_DUNGEON_SKELETON_NAME = "canvas_lit_dungeon_basic.html"
CANVAS_LIT_DUNGEON_SIDECAR = '{"goal": "lighting light composite dynamic dungeon visual shadow darkness gradient vision crawler top down"}'

CANVAS_VFX_PARTICLES_SKELETON_NAME = "canvas_vfx_particles_basic.html"
CANVAS_VFX_PARTICLES_SIDECAR = '{"goal": "vfx visual effects particles shake screenshake pooling explosion magic juicy float feedback damage impact text"}'

# Turn-based board scaffold: click-to-select / click-to-move on a grid with
# alternating players. Added 2026-05-21 after chess-trace evidence — bare
# canvas_basic.html was being used for chess/checkers/go and the model burned
# iters re-discovering board indexing + click handling. Genre-free sidecar
# tokens (turn-based / board / select / move + a few canonical examples).
CANVAS_BOARD_TURN_SKELETON_NAME = "canvas_board_turn_basic.html"
CANVAS_BOARD_TURN_SKELETON_SIDECAR = '{"goal": "turn based board click cell select move alternate player hotseat chess checkers go reversi tic tac toe othello"}'

# DOM-only scaffold for UI-style apps where canvas is overkill: calculator,
# tic-tac-toe, todo lists, simple word games. The existing ui-driven-no-canvas
# playbook bullet pointed in this direction but provided no starter shape.
CANVAS_DOM_SKELETON_NAME = "canvas_dom_basic.html"
CANVAS_DOM_SKELETON_SIDECAR = '{"goal": "dom html buttons table calculator todo word puzzle text form input click no canvas tic tac toe simple"}'

# Opening-book memory: root `memory/` stores trusted, precomputed recipes;
# live `games/game-memory/` stores learned candidates that must earn trust.
PLAYTESTS_FILENAME = "playtests.jsonl"
ASSET_AUDITS_FILENAME = "asset_audits.jsonl"
ANIMATION_AUDITS_FILENAME = "animation_audits.jsonl"
IMPLEMENTATION_OUTLINES_FILENAME = "implementation_outlines.jsonl"
VERIFIED_FINDINGS_FILENAME = "verified_findings.jsonl"
# 2026-05-24: VLM-driven visual playtest checklists, mechanism-keyed
# (NOT per-game). 10-12 mechanism recipes cover the top-100 games via
# keyword overlap on goal + plan + asset names. Same loader / scoring
# infrastructure as playtests.jsonl. Designed alongside the
# vlm-critic-memory-driven-checklist plan.
VISUAL_PLAYTESTS_FILENAME = "visual_playtests.jsonl"
# 2026-06-10: component skill library — tested, mechanics-level JS snippets
# (game loop, input buffer, particles, hit-pause, ...) the coder can paste
# and adapt. Code body lives in recipe["code"]; `content` stays the one-line
# description used for Jaccard matching. Hand-edited data file like
# visual_playtests.jsonl. See plan local-model_game_agent_capability_round.
COMPONENTS_FILENAME = "components.jsonl"

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


# 3D/WebGL three.js template. Serves as a robust, generic seed when you build 3D
# games (first person, space shooters, 3D arenas).
CANVAS_3D_SKELETON = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Game</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#0b1020; color:#e7ecff;
    font:16px/1.4 system-ui,sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { display:block; max-width:100vw; max-height:100vh; touch-action:none; }
  #hud { position:fixed; top:12px; left:12px; background:#0008; padding:8px 12px;
    border-radius:8px; pointer-events:none; }
  #help { position:fixed; bottom:12px; left:12px; opacity:.75; font-size:13px; }
  #modal { position:fixed; inset:0; display:none; place-items:center;
    background:#000a; backdrop-filter:blur(4px); z-index:100; }
  #modal .card { background:#1a2348; padding:24px 28px; border-radius:14px;
    text-align:center; box-shadow:0 20px 60px #000a; }
  button { background:#3b62ff; color:#fff; border:0; padding:10px 18px;
    border-radius:8px; font-size:15px; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="hud">Score: <span id="score">0</span></div>
<div id="help">WASD/Arrows to move, Space to act, Mouse to look</div>
<div id="modal"><div class="card">
  <h2 id="endTitle">Game Over</h2>
  <p id="endMsg">Final score: <span id="endScore">0</span></p>
  <button id="restart">Play again</button>
</div></div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const scoreEl = document.getElementById("score");
  const modal = document.getElementById("modal");
  const endScoreEl = document.getElementById("endScore");

  // Create Scene, Camera, Renderer
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  scene.fog = new THREE.FogExp2(0x0b1020, 0.015);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ canvas: cvs, antialias: true });
  renderer.shadowMap.enabled = true;

  function fit() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  fit(); window.addEventListener("resize", fit);

  // Add simple lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(20, 40, 20);
  dirLight.castShadow = true;
  scene.add(dirLight);

  // Inputs
  const keys = Object.create(null);
  const KEYMAP = { ArrowUp:"up", ArrowDown:"down", ArrowLeft:"left", ArrowRight:"right",
    KeyW:"up", KeyS:"down", KeyA:"left", KeyD:"right", Space:"act" };
  addEventListener("keydown", e => {
    const k = KEYMAP[e.code]; if (!k) return;
    keys[k] = true;
    if (["up","down","left","right","act"].includes(k)) e.preventDefault();
  }, { passive:false });
  addEventListener("keyup", e => { const k = KEYMAP[e.code]; if (k) keys[k] = false; });

  const state = { score:0, over:false };
  function reset() {
    state.score = 0; state.over = false;
    modal.style.display = "none";
    camera.position.set(0, 5, 15);
    camera.lookAt(0, 0, 0);
  }
  document.getElementById("restart").onclick = reset;
  reset();

  function update(dt) {
    if (state.over) return;
    const speed = 10;
    if (keys.left)  camera.position.x -= speed*dt;
    if (keys.right) camera.position.x += speed*dt;
    if (keys.up)    camera.position.z -= speed*dt;
    if (keys.down)  camera.position.z += speed*dt;
  }

  function draw() {
    renderer.render(scene, camera);
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
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Corridor movement and corner snapping (Pac-man, Sokoban).
CANVAS_GRID_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Grid Game</title>
<style>
  html,body { margin:0; background:#0b1020; color:#fff; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#10162e; border-radius:8px; max-width:96vw; max-height:90vh; }
  #hud { position:fixed; top:12px; left:12px; background:#0008; padding:8px 12px; border-radius:8px; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="640" height="480"></canvas></div>
<div id="hud">Score: <span id="score">0</span></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const TILE_SIZE = 32;
  const state = { player: { x: 32, y: 32, vx: 0, vy: 0 }, score: 0 };
  const keys = {};

  addEventListener("keydown", e => { keys[e.code] = true; });
  addEventListener("keyup", e => { keys[e.code] = false; });

  function update(dt) {
    const p = state.player;
    const speed = 120;
    if (keys.ArrowLeft || keys.KeyA) { p.vx = -speed; p.vy = 0; }
    else if (keys.ArrowRight || keys.KeyD) { p.vx = speed; p.vy = 0; }
    else if (keys.ArrowUp || keys.KeyW) { p.vy = -speed; p.vx = 0; }
    else if (keys.ArrowDown || keys.KeyS) { p.vy = speed; p.vx = 0; }

    p.x += p.vx * dt;
    p.y += p.vy * dt;
  }

  function draw() {
    ctx.clearRect(0,0,640,480);
    ctx.fillStyle = "#4a6cff";
    ctx.fillRect(state.player.x, state.player.y, TILE_SIZE, TILE_SIZE);
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Platformer climbing and gravity jumping mechanics. Includes input buffering.
CANVAS_PLATFORMER_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Platformer Game</title>
<style>
  html,body { margin:0; background:#0b1020; color:#fff; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#10162e; border-radius:8px; }
  #hud { position:fixed; top:12px; left:12px; background:#0008; padding:8px 12px; border-radius:8px; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<div id="hud">Score: <span id="score">0</span> &nbsp; Health: <span id="health">100</span></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const p = { x: 100, y: 400, vx: 0, vy: 0, w: 20, h: 32, climbing: false, grounded: false, health: 100, visibleHealth: 100 };
  const gravity = 800, speed = 200, jumpForce = -350;
  const keys = {};

  // Simple input buffer
  const inputBuffer = [];
  const BUFFER_WINDOW = 0.2; // 200ms

  addEventListener("keydown", e => {
    keys[e.code] = true;
    inputBuffer.push({ code: e.code, t: performance.now() / 1000 });
  });
  addEventListener("keyup", e => { keys[e.code] = false; });

  function checkBuffer(code) {
    const now = performance.now() / 1000;
    // Clean old
    while (inputBuffer.length > 0 && now - inputBuffer[0].t > BUFFER_WINDOW) {
      inputBuffer.shift();
    }
    const idx = inputBuffer.findIndex(item => item.code === code);
    if (idx >= 0) {
      inputBuffer.splice(idx, 1); // consume
      return true;
    }
    return false;
  }

  function update(dt) {
    p.vx = 0;
    if (keys.ArrowLeft || keys.KeyA) p.vx = -speed;
    if (keys.ArrowRight || keys.KeyD) p.vx = speed;

    if (p.climbing) {
      p.vy = 0;
      if (keys.ArrowUp || keys.KeyW) p.vy = -speed / 2;
      if (keys.ArrowDown || keys.KeyS) p.vy = speed / 2;
    } else {
      p.vy += gravity * dt;
      
      // Buffer check for jump
      const wantsJump = keys.ArrowUp || keys.KeyW || checkBuffer("Space");
      if (wantsJump && p.grounded) {
        p.vy = jumpForce;
        p.grounded = false;
      }
    }

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.y > 450) { p.y = 450; p.vy = 0; p.grounded = true; }

    // Smooth HUD health bar transition (LERP)
    p.visibleHealth += (p.health - p.visibleHealth) * 0.1;
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    ctx.fillStyle = "#00ff88";
    ctx.fillRect(p.x, p.y, p.w, p.h);

    // Draw HUD Health Bar smoothly
    ctx.fillStyle = "#300";
    ctx.fillRect(100, 20, 200, 16);
    ctx.fillStyle = "#ff3333";
    ctx.fillRect(100, 20, (p.visibleHealth / 100) * 200, 16);
    ctx.strokeStyle = "#fff";
    ctx.strokeRect(100, 20, 200, 16);
  }

  function frame(now) {
    const dt = 0.016;
    update(dt); draw();
    document.getElementById("health").textContent = Math.round(p.health);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Camera viewport horizontal scroll and multi-layer parallax backdrop.
CANVAS_SCROLLING_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Scrolling Game</title>
<style>
  html,body { margin:0; background:#010206; color:#fff; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#040613; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="400"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const cam = { x: 0, y: 0 };
  const p = { x: 100, y: 200, w: 24, h: 24 };
  const keys = {};

  addEventListener("keydown", e => { keys[e.code] = true; });
  addEventListener("keyup", e => { keys[e.code] = false; });

  function update(dt) {
    const speed = 250;
    if (keys.ArrowLeft || keys.KeyA) p.x -= speed * dt;
    if (keys.ArrowRight || keys.KeyD) p.x += speed * dt;
    if (keys.ArrowUp || keys.KeyW) p.y -= speed * dt;
    if (keys.ArrowDown || keys.KeyS) p.y += speed * dt;

    cam.x += (p.x - cam.x - 400) * 0.1;
  }

  function draw() {
    ctx.clearRect(0,0,800,400);
    ctx.save();
    ctx.translate(-cam.x, -cam.y);
    
    ctx.fillStyle = "rgba(255,255,255,0.2)";
    for (let i = 0; i < 20; i++) {
      let sx = (i * 120 - cam.x * 0.3) % 1200;
      ctx.fillRect(sx, 50 + (i % 3) * 40, 2, 2);
    }

    ctx.fillStyle = "#ff4a8b";
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.restore();
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Mode 7 texture coordinate perspective ground projection mapping.
CANVAS_MODE7_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mode 7 Racer</title>
<style>
  html,body { margin:0; background:#000; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#111; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="320" height="240"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const p = { x: 50, y: 50, angle: 0, speed: 0 };
  const keys = {};

  addEventListener("keydown", e => { keys[e.code] = true; });
  addEventListener("keyup", e => { keys[e.code] = false; });

  const texSize = 64;
  const tex = ctx.createImageData(texSize, texSize);
  for (let y = 0; y < texSize; y++) {
    for (let x = 0; x < texSize; x++) {
      const isAlt = ((x >> 3) + (y >> 3)) % 2 === 0;
      const idx = (y * texSize + x) * 4;
      tex.data[idx] = isAlt ? 40 : 100;
      tex.data[idx+1] = isAlt ? 140 : 40;
      tex.data[idx+2] = isAlt ? 40 : 20;
      tex.data[idx+3] = 255;
    }
  }

  function update(dt) {
    if (keys.ArrowUp || keys.KeyW) p.speed = 40;
    else if (keys.ArrowDown || keys.KeyS) p.speed = -20;
    else p.speed *= 0.95;

    if (keys.ArrowLeft || keys.KeyA) p.angle -= 2.5 * dt;
    if (keys.ArrowRight || keys.KeyD) p.angle += 2.5 * dt;

    p.x += Math.cos(p.angle) * p.speed * dt;
    p.y += Math.sin(p.angle) * p.speed * dt;
  }

  function draw() {
    ctx.clearRect(0,0,320,240);
    ctx.fillStyle = "#87ceeb"; ctx.fillRect(0, 0, 320, 100);
    
    const screenData = ctx.getImageData(0, 100, 320, 140);
    const horizon = 0, fov = 120;
    for (let screenY = 0; screenY < 140; screenY++) {
      const distance = fov / (screenY + 1);
      const scaleX = distance / fov;
      const stepX = Math.sin(p.angle) * scaleX;
      const stepY = -Math.cos(p.angle) * scaleX;
      let worldX = p.x + Math.cos(p.angle) * distance - 160 * stepX;
      let worldY = p.y + Math.sin(p.angle) * distance - 160 * stepY;

      for (let screenX = 0; screenX < 320; screenX++) {
        const tx = Math.floor(worldX) & (texSize - 1);
        const ty = Math.floor(worldY) & (texSize - 1);
        const texIdx = (ty * texSize + tx) * 4;
        const screenIdx = (screenY * 320 + screenX) * 4;

        screenData.data[screenIdx] = tex.data[texIdx];
        screenData.data[screenIdx+1] = tex.data[texIdx+1];
        screenData.data[screenIdx+2] = tex.data[texIdx+2];
        screenData.data[screenIdx+3] = 255;

        worldX += stepX; worldY += stepY;
      }
    }
    ctx.putImageData(screenData, 0, 100);
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Top-down dungeon crawler with wall-sliding diagonal velocity response, pathfinding and FSM.
CANVAS_CRAWLER_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dungeon Crawler</title>
<style>
  html,body { margin:0; background:#0a0705; color:#fff; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#15100e; border: 2px solid #5a3c28; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const p = { x: 400, y: 300, vx: 0, vy: 0, r: 16 };
  const keys = {};

  // Procedural maze grid (1 = wall, 0 = path)
  const COLS = 25, ROWS = 19;
  const grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(1));
  for (let r = 1; r < ROWS - 1; r++) {
    for (let c = 1; c < COLS - 1; c++) {
      if (Math.random() > 0.28 || r % 2 === 1 && c % 2 === 1) {
        grid[r][c] = 0; // path
      }
    }
  }

  // BFS Pathfinding toward player grid position
  function findPathBFS(startX, startY, targetX, targetY) {
    const queue = [[[startX, startY]]];
    const visited = new Set([`${startX},${startY}`]);

    while (queue.length > 0) {
      const path = queue.shift();
      const [cx, cy] = path[path.length - 1];

      if (cx === targetX && cy === targetY) {
        return path[1] || path[0]; // Next step
      }

      const neighbors = [
        [cx, cy - 1], [cx, cy + 1], [cx - 1, cy], [cx + 1, cy]
      ];

      for (const [nx, ny] of neighbors) {
        if (nx >= 0 && nx < COLS && ny >= 0 && ny < ROWS && grid[ny][nx] === 0) {
          const key = `${nx},${ny}`;
          if (!visited.has(key)) {
            visited.add(key);
            queue.push([...path, [nx, ny]]);
          }
        }
      }
    }
    return [startX, startY]; // stay
  }

  // Enemy FSM State representation
  const enemy = {
    x: 100, y: 100, r: 12, speed: 80,
    state: "patrol", // "patrol", "alert", "chase"
    targetNode: { x: 100, y: 100 },
    timer: 0
  };

  addEventListener("keydown", e => { keys[e.code] = true; });
  addEventListener("keyup", e => { keys[e.code] = false; });

  function checkCollision(x, y, r) {
    const gridX = Math.floor(x / 32);
    const gridY = Math.floor(y / 32);
    if (gridX < 0 || gridX >= COLS || gridY < 0 || gridY >= ROWS) return true;
    return grid[gridY][gridX] === 1;
  }

  function update(dt) {
    const speed = 180;
    
    // 1. Decomposed Wall-Sliding movement
    let targetX = p.x;
    let targetY = p.y;
    
    if (keys.ArrowLeft || keys.KeyA) targetX -= speed * dt;
    if (keys.ArrowRight || keys.KeyD) targetX += speed * dt;
    if (!checkCollision(targetX, p.y, p.r)) {
      p.x = targetX;
    }
    
    if (keys.ArrowUp || keys.KeyW) targetY -= speed * dt;
    if (keys.ArrowDown || keys.KeyS) targetY += speed * dt;
    if (!checkCollision(p.x, targetY, p.r)) {
      p.y = targetY;
    }

    // 2. Enemy AI State Loop & BFS pathfinding
    const pGridX = Math.floor(p.x / 32);
    const pGridY = Math.floor(p.y / 32);
    const eGridX = Math.floor(enemy.x / 32);
    const eGridY = Math.floor(enemy.y / 32);
    
    const distToPlayer = Math.hypot(p.x - enemy.x, p.y - enemy.y);
    
    if (enemy.state === "patrol") {
      if (distToPlayer < 180) {
        enemy.state = "chase";
      } else {
        // Simple random node walk
        enemy.timer -= dt;
        if (enemy.timer <= 0) {
          enemy.timer = 2 + Math.random() * 2;
          enemy.targetNode = {
            x: (1 + Math.floor(Math.random() * (COLS - 2))) * 32 + 16,
            y: (1 + Math.floor(Math.random() * (ROWS - 2))) * 32 + 16
          };
        }
        const ang = Math.atan2(enemy.targetNode.y - enemy.y, enemy.targetNode.x - enemy.x);
        enemy.x += Math.cos(ang) * (enemy.speed * 0.6) * dt;
        enemy.y += Math.sin(ang) * (enemy.speed * 0.6) * dt;
      }
    } else if (enemy.state === "chase") {
      if (distToPlayer > 300) {
        enemy.state = "patrol";
      } else {
        const nextStep = findPathBFS(eGridX, eGridY, pGridX, pGridY);
        const targetX = nextStep[0] * 32 + 16;
        const targetY = nextStep[1] * 32 + 16;
        const ang = Math.atan2(targetY - enemy.y, targetX - enemy.x);
        enemy.x += Math.cos(ang) * enemy.speed * dt;
        enemy.y += Math.sin(ang) * enemy.speed * dt;
      }
    }
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    
    // Draw grid map
    ctx.fillStyle = "#2a1e12";
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        if (grid[r][c] === 1) {
          ctx.fillRect(c * 32, r * 32, 32, 32);
        }
      }
    }

    // Player
    ctx.fillStyle = "#bf935a";
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2); ctx.fill();

    // Enemy
    ctx.fillStyle = enemy.state === "chase" ? "#ff3333" : "#ff9933";
    ctx.beginPath(); ctx.arc(enemy.x, enemy.y, enemy.r, 0, Math.PI*2); ctx.fill();
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Responsive mobile/iPad letterbox scaling with on-screen virtual controls.
CANVAS_MOBILE_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Mobile Joystick Game</title>
<style>
  html,body { margin:0; height:100%; background:#0b1020; color:#fff; overflow:hidden; touch-action:none; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#10162e; max-width:100%; max-height:100%; touch-action:none; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const p = { x: 400, y: 300, vx: 0, vy: 0, r: 16 };
  const joystick = { active: false, startX: 0, startY: 0, curX: 0, curY: 0, maxDist: 60 };

  function fit() {
    const dpr = Math.min(window.devicePixelRatio||1, 2);
    cvs.width = 800*dpr; cvs.height = 600*dpr;
    cvs.style.width = "100%"; cvs.style.height = "100%";
    ctx.setTransform(dpr,0,0,dpr,0,0);
  }
  fit(); window.addEventListener("resize", fit);

  addEventListener("pointerdown", e => {
    if (e.clientX < window.innerWidth / 2) {
      joystick.active = true;
      joystick.startX = e.clientX; joystick.startY = e.clientY;
      joystick.curX = e.clientX; joystick.curY = e.clientY;
    }
  });

  addEventListener("pointermove", e => {
    if (joystick.active) {
      joystick.curX = e.clientX; joystick.curY = e.clientY;
      const dx = joystick.curX - joystick.startX;
      const dy = joystick.curY - joystick.startY;
      const dist = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);
      const finalDist = Math.min(dist, joystick.maxDist);
      p.vx = Math.cos(angle) * (finalDist / joystick.maxDist) * 200;
      p.vy = Math.sin(angle) * (finalDist / joystick.maxDist) * 200;
    }
  });

  addEventListener("pointerup", e => {
    joystick.active = false; p.vx = 0; p.vy = 0;
  });

  function update(dt) {
    p.x += p.vx * dt; p.y += p.vy * dt;
    p.x = Math.max(p.r, Math.min(800 - p.r, p.x));
    p.y = Math.max(p.r, Math.min(600 - p.r, p.y));
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    ctx.fillStyle = "#ff5555";
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2); ctx.fill();

    if (joystick.active) {
      const localX = 100, localY = 500;
      ctx.fillStyle = "rgba(255,255,255,0.1)";
      ctx.beginPath(); ctx.arc(localX, localY, joystick.maxDist, 0, Math.PI*2); ctx.fill();
      const dx = joystick.curX - joystick.startX;
      const dy = joystick.curY - joystick.startY;
      const d = Math.hypot(dx, dy);
      const a = Math.atan2(dy, dx);
      const fd = Math.min(d, joystick.maxDist);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.beginPath(); ctx.arc(localX + Math.cos(a)*fd, localY + Math.sin(a)*fd, 20, 0, Math.PI*2); ctx.fill();
    }
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Pokemon-style continuous-lerp tile-discrete steps (Pokemon/RPG style explorers).
CANVAS_RPG_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Tile RPG</title>
<style>
  html,body { margin:0; background:#000; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#1e2f15; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="480" height="480"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const TILE_SIZE = 32;
  const p = { gridX: 7, gridY: 7, animX: 7, animY: 7, moving: false, t: 0 };
  const keys = {};

  addEventListener("keydown", e => { keys[e.code] = true; });
  addEventListener("keyup", e => { keys[e.code] = false; });

  function update(dt) {
    if (p.moving) {
      p.t += dt * 6;
      if (p.t >= 1) {
        p.animX = p.gridX; p.animY = p.gridY;
        p.moving = false; p.t = 0;
      } else {
        p.animX = p.animX + (p.gridX - p.animX) * p.t;
        p.animY = p.animY + (p.gridY - p.animY) * p.t;
      }
    } else {
      let dx = 0, dy = 0;
      if (keys.ArrowLeft || keys.KeyA) dx = -1;
      else if (keys.ArrowRight || keys.KeyD) dx = 1;
      else if (keys.ArrowUp || keys.KeyW) dy = -1;
      else if (keys.ArrowDown || keys.KeyS) dy = 1;

      if (dx !== 0 || dy !== 0) {
        p.gridX += dx; p.gridY += dy;
        p.moving = true; p.t = 0;
      }
    }
  }

  function draw() {
    ctx.clearRect(0,0,480,480);
    ctx.fillStyle = "#ffd84a";
    ctx.fillRect(p.animX * TILE_SIZE, p.animY * TILE_SIZE, TILE_SIZE, TILE_SIZE);
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Card/Board drag-and-drop coordinate overlaps and target card snaps.
CANVAS_CARDS_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Card Snapper</title>
<style>
  html,body { margin:0; background:#072517; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#0a3a25; border-radius:8px; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const cards = [{ id: 1, x: 100, y: 100, w: 80, h: 120, targetX: 100, targetY: 100, dragging: false }];
  const slots = [{ x: 400, y: 300, w: 90, h: 130 }];
  let activeCard = null, dragOffset = { x: 0, y: 0 };

  cvs.addEventListener("pointerdown", e => {
    const rect = cvs.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    for (const c of cards) {
      if (mx > c.x && mx < c.x+c.w && my > c.y && my < c.y+c.h) {
        activeCard = c; c.dragging = true;
        dragOffset.x = mx - c.x; dragOffset.y = my - c.y;
        break;
      }
    }
  });

  addEventListener("pointermove", e => {
    if (activeCard) {
      const rect = cvs.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      activeCard.x = mx - dragOffset.x; activeCard.y = my - dragOffset.y;
    }
  });

  addEventListener("pointerup", e => {
    if (activeCard) {
      activeCard.dragging = false;
      for (const s of slots) {
        const dx = (activeCard.x + activeCard.w/2) - (s.x + s.w/2);
        const dy = (activeCard.y + activeCard.h/2) - (s.y + s.h/2);
        if (Math.hypot(dx, dy) < 80) {
          activeCard.targetX = s.x + 5; activeCard.targetY = s.y + 5;
          break;
        }
      }
      activeCard = null;
    }
  });

  function update(dt) {
    for (const c of cards) {
      if (!c.dragging) {
        c.x += (c.targetX - c.x) * 0.2;
        c.y += (c.targetY - c.y) * 0.2;
      }
    }
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    ctx.strokeStyle = "rgba(255,255,255,0.2)"; ctx.lineWidth = 2;
    for (const s of slots) ctx.strokeRect(s.x, s.y, s.w, s.h);
    ctx.fillStyle = "#ffffff";
    for (const c of cards) ctx.fillRect(c.x, c.y, c.w, c.h);
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Physics launch with gravity trajectories and elastic boundary wall bounces.
CANVAS_PHYSICS_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Physics Launch</title>
<style>
  html,body { margin:0; background:#0c0d14; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#141622; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const b = { x: 100, y: 450, vx: 0, vy: 0, r: 12, launched: false };
  const gravity = 400;
  let dragStart = null, mouse = { x: 0, y: 0 };

  cvs.addEventListener("pointerdown", e => {
    const rect = cvs.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    if (Math.hypot(mx - b.x, my - b.y) < 30) { dragStart = { x: b.x, y: b.y }; }
  });

  addEventListener("pointermove", e => {
    const rect = cvs.getBoundingClientRect();
    mouse.x = e.clientX - rect.left; mouse.y = e.clientY - rect.top;
  });

  addEventListener("pointerup", e => {
    if (dragStart) {
      const dx = dragStart.x - mouse.x, dy = dragStart.y - mouse.y;
      b.vx = dx * 4; b.vy = dy * 4;
      b.launched = true; dragStart = null;
    }
  });

  function update(dt) {
    if (b.launched) {
      b.vy += gravity * dt;
      b.x += b.vx * dt; b.y += b.vy * dt;
      if (b.x < b.r || b.x > 800 - b.r) { b.vx = -b.vx * 0.8; b.x = b.x < b.r ? b.r : 800 - b.r; }
      if (b.y > 600 - b.r) { b.vy = -b.vy * 0.6; b.y = 600 - b.r; }
    }
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    if (dragStart) {
      ctx.strokeStyle = "rgba(255,255,255,0.4)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(dragStart.x, dragStart.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
    }
    ctx.fillStyle = "#ffd54f";
    ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, Math.PI*2); ctx.fill();
  }

  function frame() {
    update(0.016); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


CANVAS_VOXEL_MINECRAFT_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voxel Minecraft basic</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#101118; font-family:sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; }
  #hud { position:fixed; top:12px; left:12px; background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:8px; pointer-events:none; }
  #help { position:fixed; bottom:12px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:8px; pointer-events:none; text-align:center; font-size:14px; }
  #crosshair { position:fixed; top:50%; left:50%; width:10px; height:10px; transform:translate(-50%,-50%); pointer-events:none; }
  #crosshair::before, #crosshair::after { content:''; position:absolute; background:#fff; }
  #crosshair::before { top:4px; left:0; width:10px; height:2px; }
  #crosshair::after { top:0; left:4px; width:2px; height:10px; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="wrap"></div>
<div id="hud">Blocks Broken: <span id="score">0</span></div>
<div id="help">Click to lock mouse. WASD to move, Space to Jump.<br>Left-Click to break, Right-Click to place.</div>
<div id="crosshair"></div>
<script>
(() => {
  "use strict";
  let scene, camera, renderer, score = 0;
  const blocks = new Map();
  const keys = { KeyW:0, KeyS:0, KeyA:0, KeyD:0, Space:0 };
  let py = 1, vy = 0, isGrounded = true;
  const moveSpeed = 6, gravity = 20, jumpForce = 8;
  const cameraRotation = { x: 0, y: 0 };
  
  // Pointer lock setup
  const wrap = document.getElementById("wrap");
  wrap.addEventListener("click", () => wrap.requestPointerLock());
  document.addEventListener("mousemove", e => {
    if (document.pointerLockElement !== wrap) return;
    cameraRotation.y -= e.movementX * 0.002;
    cameraRotation.x -= e.movementY * 0.002;
    cameraRotation.x = Math.max(-Math.PI/2 + 0.05, Math.min(Math.PI/2 - 0.05, cameraRotation.x));
  });

  addEventListener("keydown", e => { if (e.code in keys) keys[e.code] = 1; });
  addEventListener("keyup", e => { if (e.code in keys) keys[e.code] = 0; });

  function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x80a0e0);
    scene.fog = new THREE.FogExp2(0x80a0e0, 0.03);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    wrap.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xcccccc); scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5); dirLight.position.set(10, 20, 10); scene.add(dirLight);

    // Procedural voxel texture
    const cvs = document.createElement("canvas"); cvs.width = 16; cvs.height = 16;
    const ctx = cvs.getContext("2d");
    ctx.fillStyle = "#557a2b"; ctx.fillRect(0,0,16,16);
    ctx.fillStyle = "#8d5e3a"; ctx.fillRect(0,4,16,12);
    const tex = new THREE.CanvasTexture(cvs); tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshLambertMaterial({ map: tex });

    // Seed terrain
    for (let x = -10; x <= 10; x++) {
      for (let z = -10; z <= 10; z++) {
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, 0, z); scene.add(mesh);
        blocks.set(`${x},0,${z}`, mesh);
      }
    }

    camera.position.set(0, py + 1.6, 5);
    
    // Build/break logic
    const raycaster = new THREE.Raycaster();
    const mouseCenter = new THREE.Vector2(0, 0);

    wrap.addEventListener("pointerdown", e => {
      if (document.pointerLockElement !== wrap) return;
      raycaster.setFromCamera(mouseCenter, camera);
      const intersects = raycaster.intersectObjects(Array.from(blocks.values()));
      if (intersects.length > 0 && intersects[0].distance < 6) {
        const hit = intersects[0];
        if (e.button === 0) { // Break
          scene.remove(hit.object);
          for (let [k, v] of blocks.entries()) { if (v === hit.object) { blocks.delete(k); break; } }
          score++; document.getElementById("score").textContent = score;
        } else if (e.button === 2) { // Place
          const p = hit.point.clone().add(hit.face.normal.clone().multiplyScalar(0.5));
          const bx = Math.round(p.x), by = Math.round(p.y), bz = Math.round(p.z);
          const key = `${bx},${by},${bz}`;
          if (!blocks.has(key)) {
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(bx, by, bz); scene.add(mesh);
            blocks.set(key, mesh);
          }
        }
      }
    });

    addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    
    // Physics / Controls
    camera.rotation.set(cameraRotation.x, cameraRotation.y, 0, 'YXZ');
    
    const moveVector = new THREE.Vector3();
    if (keys.KeyW) moveVector.z -= 1; if (keys.KeyS) moveVector.z += 1;
    if (keys.KeyA) moveVector.x -= 1; if (keys.KeyD) moveVector.x += 1;
    moveVector.normalize().multiplyScalar(moveSpeed * dt).applyQuaternion(camera.quaternion);
    moveVector.y = 0;
    
    camera.position.add(moveVector);

    // Gravity & Jump
    vy -= gravity * dt;
    camera.position.y += vy * dt;
    const floorY = 0 + 1.6;
    if (camera.position.y <= floorY) { camera.position.y = floorY; vy = 0; isGrounded = true; }
    if (keys.Space && isGrounded) { vy = jumpForce; isGrounded = false; }

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  init();
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


CANVAS_AR_FLICK_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AR Capture basic</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#101118; font-family:sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#141622; touch-action:none; max-width:100vw; max-height:100vh; }
  #hud { position:fixed; top:12px; left:12px; background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:8px; pointer-events:none; }
  #help { position:fixed; bottom:12px; left:50%; transform:translateX(-50%); background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:8px; pointer-events:none; font-size:12px; text-align:center; }
  #camView { position:fixed; inset:0; object-fit:cover; z-index:-1; display:none; }
</style></head>
<body>
<video id="camView" autoplay playsinline></video>
<div id="wrap"><canvas id="c" width="400" height="600"></canvas></div>
<div id="hud">Caught: <span id="score">0</span></div>
<div id="help">Swipe up rapidly to throw the ball at the target! Add horizontal curve for bonus.</div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const scoreEl = document.getElementById("score");
  const cam = document.getElementById("camView");
  
  let score = 0;
  
  // Try AR Camera
  navigator.mediaDevices?.getUserMedia({ video: { facingMode: "environment" } })
    .then(s => { cam.srcObject = s; cam.style.display = "block"; })
    .catch(() => {});

  const monster = { x: 200, y: 200, r: 35, baseR: 35, t: 0 };
  const ball = { x: 200, y: 520, z: 0, vx: 0, vy: 0, vz: 0, r: 20, active: false, drag: false, prev: [] };
  const target = { pulse: 0, size: 40 };

  cvs.addEventListener("pointerdown", e => {
    const r = cvs.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    if (Math.hypot(mx - ball.x, my - ball.y) < 40 && !ball.active) {
      ball.drag = true; ball.prev = [{ x: mx, y: my, t: performance.now() }];
    }
  });

  cvs.addEventListener("pointermove", e => {
    const r = cvs.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    if (ball.drag) {
      ball.x = mx; ball.y = my;
      ball.prev.push({ x: mx, y: my, t: performance.now() });
      if (ball.prev.length > 5) ball.prev.shift();
    }
  });

  cvs.addEventListener("pointerup", e => {
    if (ball.drag) {
      ball.drag = false;
      if (ball.prev.length >= 2) {
        const p1 = ball.prev[0], p2 = ball.prev[ball.prev.length - 1];
        const dt = (p2.t - p1.t) / 1000;
        if (dt > 0.05) {
          ball.vx = (p2.x - p1.x) / dt;
          ball.vy = (p2.y - p1.y) / dt;
          ball.vz = -Math.abs(ball.vy) * 1.5; // Simulate forward depth
          if (ball.vy < -200) { ball.active = true; }
        }
      }
      if (!ball.active) { ball.x = 200; ball.y = 520; }
    }
  });

  function drawBackground() {
    if (cam.style.display !== "block") {
      const grad = ctx.createLinearGradient(0,0,0,600);
      grad.addColorStop(0, "#2c3e50"); grad.addColorStop(1, "#1e272c");
      ctx.fillStyle = grad; ctx.fillRect(0,0,400,600);

      // Draw landscape hills
      ctx.fillStyle = "#27ae60";
      ctx.beginPath(); ctx.ellipse(200, 480, 300, 150, 0, 0, Math.PI*2); ctx.fill();
      ctx.fillStyle = "#2ecc71";
      ctx.beginPath(); ctx.ellipse(100, 500, 250, 120, 0, 0, Math.PI*2); ctx.fill();
    } else {
      ctx.clearRect(0,0,400,600);
    }
  }

  function update(dt) {
    monster.t += dt;
    monster.y = 200 + Math.sin(monster.t * 2) * 40;
    monster.r = monster.baseR + Math.sin(monster.t * 4) * 2;

    target.pulse = (target.pulse + dt) % 1.5;
    target.size = 50 * (1 - (target.pulse / 1.5));

    if (ball.active) {
      ball.vz += 800 * dt; // Gravity
      ball.x += ball.vx * dt;
      ball.y += ball.vy * dt + ball.vz * 0.05 * dt;
      ball.r = Math.max(8, 20 * (1 - (Math.abs(ball.vz) / 3000)));

      if (ball.y < -100 || ball.y > 650 || ball.x < -100 || ball.x > 500) {
        ball.active = false; ball.x = 200; ball.y = 520; ball.vz = 0;
      }

      if (ball.y <= monster.y + 10 && ball.y >= monster.y - 10) {
        if (Math.hypot(ball.x - monster.x, ball.y - monster.y) < monster.r + 5) {
          score++; scoreEl.textContent = score;
          ball.active = false; ball.x = 200; ball.y = 520; ball.vz = 0;
        }
      }
    }
  }

  function draw() {
    drawBackground();

    ctx.fillStyle = "#e74c3c";
    ctx.beginPath(); ctx.arc(monster.x, monster.y, monster.r, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(monster.x - 10, monster.y - 5, 8, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(monster.x + 10, monster.y - 5, 8, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#000";
    ctx.beginPath(); ctx.arc(monster.x - 10, monster.y - 5, 3, 0, Math.PI*2); ctx.fill();
    ctx.beginPath(); ctx.arc(monster.x + 10, monster.y - 5, 3, 0, Math.PI*2); ctx.fill();

    ctx.strokeStyle = "#2ecc71"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(monster.x, monster.y, target.size, 0, Math.PI*2); ctx.stroke();

    ctx.save();
    ctx.translate(ball.x, ball.y);
    ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = 8;
    ctx.fillStyle = "#f1c40f";
    ctx.beginPath(); ctx.arc(0, 0, ball.r, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(-ball.r/4, -ball.r/4, ball.r/4, 0, Math.PI*2); ctx.fill();
    ctx.restore();
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    update(dt); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


CANVAS_LIT_DUNGEON_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lit Dungeon basic</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#050508; font-family:sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#08080c; border-radius:12px; max-width:96vw; max-height:90vh; touch-action:none; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  
  const lightCvs = document.createElement("canvas");
  const lightCtx = lightCvs.getContext("2d");
  lightCvs.width = 800; lightCvs.height = 600;

  const player = { x: 400, y: 300, speed: 200, r: 15, lightRadius: 180 };
  const keys = { KeyW: 0, KeyS: 0, KeyA: 0, KeyD: 0 };
  const torches = [
    { x: 200, y: 150, r: 80, pulse: 0 },
    { x: 600, y: 150, r: 80, pulse: 0.5 },
    { x: 400, y: 450, r: 100, pulse: 0.2 }
  ];

  addEventListener("keydown", e => { if (e.code in keys) keys[e.code] = 1; });
  addEventListener("keyup", e => { if (e.code in keys) keys[e.code] = 0; });

  function update(dt) {
    let dx = 0, dy = 0;
    if (keys.KeyW) dy -= 1; if (keys.KeyS) dy += 1;
    if (keys.KeyA) dx -= 1; if (keys.KeyD) dx += 1;
    if (dx !== 0 && dy !== 0) { dx *= 0.707; dy *= 0.707; }
    player.x = Math.max(20, Math.min(780, player.x + dx * player.speed * dt));
    player.y = Math.max(20, Math.min(580, player.y + dy * player.speed * dt));

    torches.forEach(t => {
      t.pulse = (t.pulse + dt) % (Math.PI * 2);
    });
  }

  function draw() {
    ctx.clearRect(0,0,800,600);
    
    ctx.fillStyle = "#1e1f29";
    for(let y=0; y<600; y+=40) {
      for(let x=0; x<800; x+=40) {
        ctx.fillRect(x+1, y+1, 38, 38);
      }
    }

    torches.forEach(t => {
      ctx.fillStyle = "#d35400";
      ctx.fillRect(t.x-4, t.y-4, 8, 16);
      ctx.fillStyle = "#f1c40f";
      ctx.beginPath(); ctx.arc(t.x, t.y-6, 6 + Math.sin(t.pulse*6)*2, 0, Math.PI*2); ctx.fill();
    });

    ctx.fillStyle = "#3498db";
    ctx.beginPath(); ctx.arc(player.x, player.y, player.r, 0, Math.PI*2); ctx.fill();

    lightCtx.fillStyle = "#0c0d12";
    lightCtx.fillRect(0,0,800,600);
    lightCtx.globalCompositeOperation = "screen";

    const playerGrad = lightCtx.createRadialGradient(player.x, player.y, 0, player.x, player.y, player.lightRadius);
    playerGrad.addColorStop(0, "rgba(255,255,255,1.0)");
    playerGrad.addColorStop(0.5, "rgba(255,255,255,0.4)");
    playerGrad.addColorStop(1, "rgba(255,255,255,0.0)");
    lightCtx.fillStyle = playerGrad;
    lightCtx.beginPath(); lightCtx.arc(player.x, player.y, player.lightRadius, 0, Math.PI*2); lightCtx.fill();

    torches.forEach(t => {
      const tr = t.r + Math.sin(t.pulse*8)*5;
      const torchGrad = lightCtx.createRadialGradient(t.x, t.y-6, 0, t.x, t.y-6, tr);
      torchGrad.addColorStop(0, "rgba(230,126,34,1.0)");
      torchGrad.addColorStop(0.4, "rgba(230,126,34,0.5)");
      torchGrad.addColorStop(1, "rgba(230,126,34,0.0)");
      lightCtx.fillStyle = torchGrad;
      lightCtx.beginPath(); lightCtx.arc(t.x, t.y-6, tr, 0, Math.PI*2); lightCtx.fill();
    });

    lightCtx.globalCompositeOperation = "source-over";

    ctx.globalCompositeOperation = "multiply";
    ctx.drawImage(lightCvs, 0, 0);
    ctx.globalCompositeOperation = "source-over";
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    update(dt); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


CANVAS_VFX_PARTICLES_SKELETON = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VFX Particles sandbox</title>
<style>
  :root { color-scheme: dark; }
  html,body { margin:0; height:100%; background:#0a0b10; font-family:sans-serif; overflow:hidden; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; }
  canvas { background:#0e1017; border-radius:12px; max-width:96vw; max-height:90vh; touch-action:none; }
</style></head>
<body>
<div id="wrap"><canvas id="c" width="800" height="600"></canvas></div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");

  const player = { x: 400, y: 300, r: 15 };
  const numbers = [];
  const particlePool = [];
  
  for(let i=0; i<150; i++) {
    particlePool.push({ x:0, y:0, vx:0, vy:0, color:"#fff", size:1, life:0, maxLife:0, active:false });
  }

  let shake = { x: 0, y: 0, amt: 0 };

  cvs.addEventListener("pointerdown", e => {
    const r = cvs.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    triggerExplosion(mx, my);
    triggerScreenShake(12);
    triggerDamageNumber(mx, my - 20, Math.floor(Math.random() * 50) + 10);
  });

  function spawnParticle(x, y, color, size) {
    const p = particlePool.find(item => !item.active);
    if (!p) return;
    const angle = Math.random() * Math.PI * 2;
    const speed = 50 + Math.random() * 150;
    p.x = x; p.y = y;
    p.vx = Math.cos(angle) * speed; p.vy = Math.sin(angle) * speed;
    p.color = color; p.size = size;
    p.life = p.maxLife = 0.4 + Math.random() * 0.4;
    p.active = true;
  }

  function triggerExplosion(x, y) {
    const colors = ["#ff5722", "#ffc107", "#ffeb3b", "#e91e63"];
    for(let i=0; i<30; i++) {
      spawnParticle(x, y, colors[Math.floor(Math.random() * colors.length)], 2 + Math.random() * 4);
    }
  }

  function triggerScreenShake(power) {
    shake.amt = power;
  }

  function triggerDamageNumber(x, y, num) {
    numbers.push({ x, y, val: num, life: 1.0, vy: -50 });
  }

  function update(dt) {
    if (shake.amt > 0.1) {
      shake.x = (Math.random() - 0.5) * shake.amt;
      shake.y = (Math.random() - 0.5) * shake.amt;
      shake.amt *= Math.exp(-6 * dt);
    } else {
      shake.x = 0; shake.y = 0; shake.amt = 0;
    }

    particlePool.forEach(p => {
      if (!p.active) return;
      p.life -= dt;
      if (p.life <= 0) { p.active = false; return; }
      p.x += p.vx * dt; p.y += p.vy * dt;
    });

    for(let i = numbers.length - 1; i >= 0; i--) {
      const n = numbers[i];
      n.life -= dt;
      if (n.life <= 0) { numbers.splice(i, 1); continue; }
      n.y += n.vy * dt;
    }
  }

  function draw() {
    ctx.save();
    ctx.translate(shake.x, shake.y);
    ctx.clearRect(0,0,800,600);

    ctx.strokeStyle = "#1a1e2a"; ctx.lineWidth = 1;
    for(let x=0; x<800; x+=50) {
      ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,600); ctx.stroke();
    }
    for(let y=0; y<600; y+=50) {
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(800,y); ctx.stroke();
    }

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    particlePool.forEach(p => {
      if (!p.active) return;
      ctx.globalAlpha = p.life / p.maxLife;
      ctx.fillStyle = p.color;
      ctx.fillRect(p.x - p.size/2, p.y - p.size/2, p.size, p.size);
    });
    ctx.restore();

    ctx.fillStyle = "#a855f7";
    ctx.beginPath(); ctx.arc(player.x, player.y, player.r, 0, Math.PI*2); ctx.fill();

    ctx.textAlign = "center"; ctx.font = "bold 20px system-ui";
    numbers.forEach(n => {
      ctx.fillStyle = `rgba(239, 68, 68, ${n.life})`;
      ctx.fillText(n.val, n.x, n.y);
    });

    ctx.restore();
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    update(dt); draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>
"""


# Turn-based board scaffold. Added 2026-05-21 to stop chess/checkers/go goals
# falling through to canvas_basic.html. Provides: 8x8 grid, currentPlayer,
# selected cell, click-to-select / click-to-move state machine, HUD showing
# whose turn it is, and a stub legalMoves(from) the model REPLACES with real
# rules. Window.gameState exposed so probes can see it (addresses the
# recurring window.state-undefined failure across May 20-21 traces).
CANVAS_BOARD_TURN_SKELETON = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Board Game</title>
<style>
  html,body { margin:0; height:100%; background:#1a1a2e; color:#e7ecff;
    font:16px/1.4 system-ui,sans-serif; overflow:hidden; user-select:none; }
  #wrap { position:fixed; inset:0; display:grid; place-items:center; gap:12px;
    grid-template-rows:auto auto auto; }
  canvas { background:#2a2a3e; border-radius:8px; touch-action:none;
    box-shadow:0 10px 40px #0008; image-rendering:pixelated; }
  #hud { font-size:18px; opacity:.9; pointer-events:none; }
  #ctrls { display:flex; gap:8px; }
  button { background:#3b62ff; color:#fff; border:0; padding:8px 14px;
    border-radius:6px; font-size:14px; cursor:pointer; }
  button:hover { filter:brightness(1.1); }
</style></head>
<body>
<div id="wrap">
  <div id="hud">Turn: <span id="turn">White</span> &nbsp; <span id="msg"></span></div>
  <canvas id="c" width="480" height="480"></canvas>
  <div id="ctrls"><button id="restart">Restart</button></div>
</div>
<script>
(() => {
  "use strict";
  const cvs = document.getElementById("c");
  const ctx = cvs.getContext("2d");
  const turnEl = document.getElementById("turn");
  const msgEl  = document.getElementById("msg");
  const SIZE = 8, TILE = 60;

  // ---- State (exposed on window so probes can read it) ------------------
  const state = {
    board: [],            // board[r][c] = null | { kind, owner: 'W'|'B' }
    currentPlayer: "W",   // 'W' or 'B'
    selected: null,       // { r, c } or null
    legalTargets: [],     // [{r,c}, ...] for the selected piece
    over: false,
    winner: null,
  };
  window.gameState = state;
  window.game = { reset, currentPlayer: () => state.currentPlayer };

  // ---- Init / reset -----------------------------------------------------
  function reset() {
    state.board = Array.from({ length: SIZE }, () => Array(SIZE).fill(null));
    // Demo placement: model REPLACES this with real piece setup.
    state.board[0][0] = { kind: "piece", owner: "B" };
    state.board[7][7] = { kind: "piece", owner: "W" };
    state.currentPlayer = "W";
    state.selected = null;
    state.legalTargets = [];
    state.over = false;
    state.winner = null;
    msgEl.textContent = "";
    turnEl.textContent = state.currentPlayer === "W" ? "White" : "Black";
    draw();
  }
  document.getElementById("restart").addEventListener("click", reset);

  // ---- Rules stub (model REPLACES with real logic) ----------------------
  function legalMoves(from) {
    // Demo: any empty cell is "legal". Real games override this entirely.
    const out = [];
    for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
      if (!state.board[r][c]) out.push({ r, c });
    }
    return out;
  }

  function applyMove(from, to) {
    state.board[to.r][to.c] = state.board[from.r][from.c];
    state.board[from.r][from.c] = null;
    state.currentPlayer = state.currentPlayer === "W" ? "B" : "W";
    turnEl.textContent = state.currentPlayer === "W" ? "White" : "Black";
  }

  // ---- Pointer -> cell --------------------------------------------------
  function cellFromPointer(e) {
    const rect = cvs.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const c = Math.floor(mx / TILE), r = Math.floor(my / TILE);
    if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return null;
    return { r, c };
  }

  cvs.addEventListener("pointerdown", e => {
    if (state.over) return;
    const cell = cellFromPointer(e);
    if (!cell) return;

    if (state.selected) {
      const ok = state.legalTargets.some(t => t.r === cell.r && t.c === cell.c);
      if (ok) {
        applyMove(state.selected, cell);
        state.selected = null;
        state.legalTargets = [];
      } else {
        // Clicking own piece reselects; otherwise deselect.
        const p = state.board[cell.r][cell.c];
        if (p && p.owner === state.currentPlayer) {
          state.selected = cell;
          state.legalTargets = legalMoves(cell);
        } else {
          state.selected = null;
          state.legalTargets = [];
        }
      }
    } else {
      const p = state.board[cell.r][cell.c];
      if (p && p.owner === state.currentPlayer) {
        state.selected = cell;
        state.legalTargets = legalMoves(cell);
      }
    }
    draw();
  });

  // ---- Draw -------------------------------------------------------------
  function draw() {
    try {
      for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
        ctx.fillStyle = ((r + c) % 2 === 0) ? "#e6d3a8" : "#7a5a3a";
        ctx.fillRect(c * TILE, r * TILE, TILE, TILE);
      }
      if (state.selected) {
        ctx.strokeStyle = "#ffd84a"; ctx.lineWidth = 3;
        ctx.strokeRect(state.selected.c * TILE + 2, state.selected.r * TILE + 2,
                       TILE - 4, TILE - 4);
      }
      for (const t of state.legalTargets) {
        ctx.fillStyle = "rgba(80,200,120,0.45)";
        ctx.beginPath();
        ctx.arc(t.c * TILE + TILE / 2, t.r * TILE + TILE / 2, TILE * 0.18, 0, Math.PI * 2);
        ctx.fill();
      }
      for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
        const p = state.board[r][c];
        if (!p) continue;
        ctx.fillStyle = p.owner === "W" ? "#fafafa" : "#1a1a1a";
        ctx.beginPath();
        ctx.arc(c * TILE + TILE / 2, r * TILE + TILE / 2, TILE * 0.35, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#444"; ctx.lineWidth = 2; ctx.stroke();
      }
    } catch (err) { console.error("draw failed:", err); }
  }

  reset();
})();
</script>
</body></html>
"""


# DOM-only scaffold. For UI-style apps where canvas would be over-engineering:
# tic-tac-toe, calculators, todo lists, word games. Uses <table> with
# data-r/data-c attributes + event delegation. Window.gameState exposed for
# probes. Added 2026-05-21 to complement the ui-driven-no-canvas playbook
# bullet which had no scaffold backing it.
CANVAS_DOM_SKELETON = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOM App</title>
<style>
  html,body { margin:0; min-height:100%; background:#1a1a2e; color:#e7ecff;
    font:16px/1.4 system-ui,sans-serif; }
  #app { max-width:520px; margin:32px auto; padding:24px;
    background:#2a2a3e; border-radius:12px; box-shadow:0 10px 40px #0008; }
  h1 { margin:0 0 16px; font-size:22px; }
  #hud { display:flex; justify-content:space-between; margin-bottom:12px;
    font-size:16px; opacity:.9; }
  table { border-collapse:collapse; margin:8px auto; }
  td button { width:64px; height:64px; font-size:28px; cursor:pointer;
    background:#1a1a2e; color:#e7ecff; border:2px solid #3b62ff;
    border-radius:8px; user-select:none; }
  td button:hover:not(:disabled) { filter:brightness(1.2); }
  td button:disabled { opacity:.7; cursor:default; }
  #ctrls { display:flex; gap:8px; justify-content:center; margin-top:12px; }
  button.action { background:#3b62ff; color:#fff; border:0; padding:8px 14px;
    border-radius:6px; font-size:14px; cursor:pointer; }
</style></head>
<body>
<div id="app">
  <h1>App</h1>
  <div id="hud"><span id="turn">Turn: X</span><span id="msg"></span></div>
  <table id="board"></table>
  <div id="ctrls"><button class="action" id="restart">Restart</button></div>
</div>
<script>
(() => {
  "use strict";
  const SIZE = 3;
  const boardEl = document.getElementById("board");
  const turnEl  = document.getElementById("turn");
  const msgEl   = document.getElementById("msg");

  // ---- State (exposed on window so probes can read it) ------------------
  const state = {
    cells: [],           // flat length SIZE*SIZE; "" | "X" | "O"
    currentPlayer: "X",  // "X" or "O"
    over: false,
    winner: null,
    score: 0,            // common probe target
  };
  window.gameState = state;
  window.game = { reset, currentPlayer: () => state.currentPlayer };

  function buildBoard() {
    boardEl.innerHTML = "";
    for (let r = 0; r < SIZE; r++) {
      const tr = document.createElement("tr");
      for (let c = 0; c < SIZE; c++) {
        const td = document.createElement("td");
        const b = document.createElement("button");
        b.dataset.r = String(r); b.dataset.c = String(c);
        td.appendChild(b); tr.appendChild(td);
      }
      boardEl.appendChild(tr);
    }
  }

  function reset() {
    state.cells = Array(SIZE * SIZE).fill("");
    state.currentPlayer = "X";
    state.over = false;
    state.winner = null;
    state.score = 0;
    msgEl.textContent = "";
    turnEl.textContent = "Turn: " + state.currentPlayer;
    render();
  }
  document.getElementById("restart").addEventListener("click", reset);

  // Event delegation on the table — one listener for the whole board.
  boardEl.addEventListener("click", e => {
    const btn = e.target.closest("button");
    if (!btn || state.over) return;
    const r = +btn.dataset.r, c = +btn.dataset.c;
    const idx = r * SIZE + c;
    if (state.cells[idx]) return;
    state.cells[idx] = state.currentPlayer;
    state.score += 1;
    if (checkWin(state.currentPlayer)) {
      state.over = true; state.winner = state.currentPlayer;
      msgEl.textContent = state.currentPlayer + " wins!";
    } else if (state.cells.every(x => x)) {
      state.over = true;
      msgEl.textContent = "Draw";
    } else {
      state.currentPlayer = state.currentPlayer === "X" ? "O" : "X";
      turnEl.textContent = "Turn: " + state.currentPlayer;
    }
    render();
  });

  function checkWin(p) {
    const lines = [
      [0,1,2],[3,4,5],[6,7,8],          // rows
      [0,3,6],[1,4,7],[2,5,8],          // cols
      [0,4,8],[2,4,6],                  // diagonals
    ];
    return lines.some(L => L.every(i => state.cells[i] === p));
  }

  function render() {
    try {
      const btns = boardEl.querySelectorAll("button");
      btns.forEach(b => {
        const idx = (+b.dataset.r) * SIZE + (+b.dataset.c);
        b.textContent = state.cells[idx];
        b.disabled = !!state.cells[idx] || state.over;
      });
    } catch (err) { console.error("render failed:", err); }
  }

  buildBoard();
  reset();
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
    "game", "make", "build", "create", "simple", "small", "making",
    "canvas", "html", "css", "javascript", "js", "app", "application",
    "using", "use", "uses", "play", "player", "screen", "implementation",
    "implement", "code", "file", "single",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s) if t.lower() not in _STOPWORDS]


# Modality keyword sets — genre-free per the project rule. Pure rendering /
# UI shape (turn-based-board, DOM-only, 3D), NOT subject matter (chess,
# tic-tac-toe, doom). Canonical game names appear only as token hooks the
# user is likely to type, not as game-specific scaffolds. Added 2026-05-21
# after May 20-21 trace evidence: chess / pac / doom / FPS all fell through
# to canvas_basic.html at score 0.0 because Jaccard on short goals (1-2
# non-stopword tokens) cannot clear _SKELETON_MIN_SIM = 0.3.
_BOARD_KEYWORDS: frozenset[str] = frozenset({
    "board", "chess", "checkers", "checker", "go", "reversi", "othello",
    "hotseat", "turn-based", "turnbased",
    "tic", "tac", "toe", "tic-tac-toe", "tictactoe",
    "two-player", "twoplayer", "hot-seat",
})

_DOM_KEYWORDS: frozenset[str] = frozenset({
    "calculator", "todo", "form", "word", "puzzle",
    "tic-tac-toe", "tictactoe", "button", "buttons", "table", "input",
    "spreadsheet", "checklist",
})

# Hint: if the goal explicitly asks for canvas/animation, DOM scaffold is
# the wrong fit even if it has DOM-keyword overlap. Used as a NEGATIVE
# filter inside _detect_dom_intent.
_CANVAS_HINT_KEYWORDS: frozenset[str] = frozenset({
    "canvas", "animation", "animated", "sprite", "sprites", "render",
    "rendered", "draw", "particle", "particles", "physics", "shader",
    "webgl",
})

# 3D rendering modality. Mirrors prompts_v1._3D_KEYWORDS to avoid a new
# cross-module import — keep in sync if either side changes.
_3D_MODALITY_KEYWORDS: frozenset[str] = frozenset({
    "3d", "three", "threejs",
    "first-person", "firstperson", "fps",
    "raycaster", "raycasting", "raycast",
    "voxel", "voxels",
    "wolfenstein", "doom", "doom-like", "doomlike",
    "minecraft", "minecraft-like", "minecraftlike",
    "perspective",
})

# When ≥ this many modality tokens hit, the modality skeleton wins.
# Threshold of 2 balances "needs more than coincidence" against the trace
# evidence that chess/doom/FPS goals have only 1-2 modality tokens at most.
_MODALITY_MIN_HITS = 2

# Lone-hook tokens that are SO specific they justify a modality match on
# their own (1 hit is enough). Keeps "chess" / "doom" / "tic-tac-toe" from
# needing a second keyword to find their scaffold.
_BOARD_STRONG_HOOKS: frozenset[str] = frozenset({
    "chess", "checkers", "reversi", "othello", "tictactoe", "tic-tac-toe",
})
_DOM_STRONG_HOOKS: frozenset[str] = frozenset({
    "calculator", "todo", "tictactoe", "tic-tac-toe",
})
_3D_STRONG_HOOKS: frozenset[str] = frozenset({
    "doom", "wolfenstein", "minecraft", "raycaster", "raycasting",
    "first-person", "firstperson", "voxel",
})


def _modality_tokens(goal: str) -> list[str]:
    """Lowercased word tokens including digits + hyphenated forms.

    Different from `_tokenize`: keeps digits (so "3d" survives) and does
    not strip stopwords (modality words are rarely stopwords; the safety
    net just adds noise here). Also emits two- and three-word joined
    forms ("first person" -> "firstperson", "tic-tac-toe" stays as-is).
    """
    if not goal:
        return []
    s = goal.lower()
    raw = [w for w in re.findall(r"[a-z0-9-]+", s)]
    if not raw:
        return raw
    out: list[str] = list(raw)
    for i in range(len(raw) - 1):
        joined = raw[i] + raw[i + 1]
        out.append(joined)
        out.append(raw[i] + "-" + raw[i + 1])
    for i in range(len(raw) - 2):
        out.append(raw[i] + raw[i + 1] + raw[i + 2])
        out.append(raw[i] + "-" + raw[i + 1] + "-" + raw[i + 2])
    return out


def _detect_board_intent(goal: str) -> list[str]:
    """Return list of board-modality keywords found in `goal`. Empty means
    the board scaffold is NOT a match. Genre-free per project rule —
    these are UI-shape words (board, turn-based) + canonical token hooks
    a user is likely to type.
    """
    toks = _modality_tokens(goal)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in _BOARD_KEYWORDS and t not in seen:
            seen.add(t); out.append(t)
    return out


def _detect_dom_intent(goal: str) -> list[str]:
    """Return DOM-modality keywords. Empty if the goal mentions canvas /
    animation / sprites (those override toward a canvas scaffold even if
    DOM keywords also appear).
    """
    toks = _modality_tokens(goal)
    if any(t in _CANVAS_HINT_KEYWORDS for t in toks):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in _DOM_KEYWORDS and t not in seen:
            seen.add(t); out.append(t)
    return out


def _detect_3d_intent(goal: str) -> list[str]:
    """Return 3D-modality keywords. Mirrors the prompts_v1 detector but
    kept local so memory.py doesn't import the prompt module."""
    toks = _modality_tokens(goal)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in _3D_MODALITY_KEYWORDS and t not in seen:
            seen.add(t); out.append(t)
    return out


# Stopwords for visual-playtest matching context tokenization. These
# words appear in almost every game goal/plan and would dilute the
# overlap signal. Genre-free — pure English stop words + game-meta
# words that don't disambiguate mechanism (`game`, `build`, `make`,
# `want`, etc.).
_VISUAL_MATCH_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "in", "on", "at",
    "for", "with", "of", "is", "are", "be", "by", "as", "it", "its",
    "this", "that", "their", "they", "them", "we", "you", "i",
    "about", "into", "from", "between", "through", "during", "after",
    "before", "above", "below", "than", "then", "when", "where", "why",
    "how", "what", "which", "who", "some", "any", "every", "all",
    "make", "makes", "making", "build", "builds", "building",
    "create", "creates", "creating", "want", "wants", "wanting",
    "need", "needs", "needing", "should", "would", "could", "can",
    "do", "does", "doing", "use", "using", "used", "uses",
    "have", "has", "had", "get", "gets", "got",
    "game", "games", "like", "type", "kind", "version", "clone",
    "style", "sort", "way", "ways", "thing", "things", "stuff",
    "good", "great", "nice", "cool", "fun", "best", "better",
    "small", "large", "big", "tiny", "huge",
    "really", "very", "just", "also", "even", "only", "still",
    "please", "ok", "yes", "now", "here", "there",
})


def _visual_match_tokens(text: str) -> set[str]:
    """Tokenize matching context into a keyword set for visual-playtest
    recipe retrieval.

    Aggressive shape (more permissive than `_modality_tokens` because
    we WANT to catch mechanic verbs like "navigate" / "shoot" /
    "stack" / "race" that game descriptions use without naming the
    game). Behavior:

      - Lowercase, extract `[a-z0-9]+` words.
      - Drop stopwords (see `_VISUAL_MATCH_STOPWORDS`) and tokens <3 chars.
      - Emit 2-gram and 3-gram joined forms (without and with hyphens)
        so phrases like "first person" / "tile based" / "side scroll"
        match recipe keywords like `firstperson` / `tile-based`.

    Used against the goal text, the model's <plan> text (when
    available), and asset names — combined into one string before
    tokenization.
    """
    if not text:
        return set()
    raw = re.findall(r"[a-z0-9]+", text.lower())
    if not raw:
        return set()
    toks: set[str] = set()
    for w in raw:
        if len(w) >= 3 and w not in _VISUAL_MATCH_STOPWORDS:
            toks.add(w)
    # 2-grams (joined + hyphenated). Length-6 floor avoids "of-the" cruft.
    for i in range(len(raw) - 1):
        a, b = raw[i], raw[i + 1]
        joined = a + b
        if len(joined) >= 6 and joined not in _VISUAL_MATCH_STOPWORDS:
            toks.add(joined)
        toks.add(a + "-" + b)
    # 3-grams (joined + hyphenated). Length-9 floor.
    for i in range(len(raw) - 2):
        a, b, c = raw[i], raw[i + 1], raw[i + 2]
        joined3 = a + b + c
        if len(joined3) >= 9:
            toks.add(joined3)
        toks.add(a + "-" + b + "-" + c)
    return toks


def find_best_visual_playtest(
    recipes: list[VisualPlaytestRecipe],
    *,
    goal: str = "",
    plan_text: str = "",
    asset_names: list[str] | None = None,
    default_min_matches: int = 2,
) -> tuple[VisualPlaytestRecipe | None, dict]:
    """Pick the best-matching visual playtest recipe for the current
    session, or (None, diag) when no recipe scores above its floor.

    Matching context = goal + plan_text + asset_names joined by
    spaces. By the time the visual critic fires (after Phase A), all
    three are populated. Even a vague goal like "collect dots while
    avoiding ghosts in corridors" gets a strong match via plan-text
    + asset-name keyword overlap.

    Two-stage matcher:

      1. **Strong-hook bypass.** If the recipe declares any tokens in
         its `strong_hooks` (game-name level: "doom", "chess",
         "pacman") AND any of those appear in the matching context,
         return immediately with score = len(hits). Decisive single
         token wins (mirrors the existing `_modality_skeleton`
         strong-hook pattern).
      2. **Overlap count.** For each remaining recipe, count overlap
         between its `applies_keywords` and the matching token set.
         Recipe with the highest overlap wins, must be
         >= `applies_min_matches` (default 2).

    Returns (recipe_or_None, diag_dict). `diag_dict` carries:
      - `top_candidates`: list of (recipe_id, score) for the top 3
        scorers — visible in the trace so it's clear WHY a recipe
        was/wasn't chosen.
      - `match_tokens_sample`: first 20 tokens from the matching
        context for postmortem analysis.
    """
    asset_names = asset_names or []
    context = " ".join([
        goal or "",
        plan_text or "",
        " ".join(asset_names),
    ])
    ctx_toks = _visual_match_tokens(context)
    diag: dict = {
        "top_candidates": [],
        "match_tokens_sample": sorted(ctx_toks)[:20],
        "context_token_count": len(ctx_toks),
    }
    if not recipes or not ctx_toks:
        return None, diag

    scored: list[tuple[float, int, VisualPlaytestRecipe, str]] = []
    for r in recipes:
        rec_dict = r.recipe if isinstance(r.recipe, dict) else {}
        kws = rec_dict.get("applies_keywords") or []
        strong = set(str(x).lower() for x in (rec_dict.get("strong_hooks") or []))
        min_matches = int(
            rec_dict.get("applies_min_matches", default_min_matches)
        )
        # Strong-hook bypass.
        if strong:
            strong_hits = strong & ctx_toks
            if strong_hits:
                # Score strong hits very high (1000+) so they beat
                # any overlap-based match.
                scored.append(
                    (1000.0 + len(strong_hits), min_matches, r, "strong_hook")
                )
                continue
        # Overlap count.
        kw_set = set(str(k).lower() for k in kws)
        overlap = kw_set & ctx_toks
        score = len(overlap)
        if score >= min_matches:
            scored.append((float(score), min_matches, r, "overlap"))

    if not scored:
        return None, diag
    scored.sort(key=lambda t: (-t[0], t[2].id))  # highest score, then stable
    diag["top_candidates"] = [
        {"id": r.id, "score": round(s, 2), "via": via}
        for (s, _m, r, via) in scored[:3]
    ]
    return scored[0][2], diag


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


@dataclass
class OpeningBookItem:
    """Trusted root or lower-confidence live recipe/outline memory."""

    id: str
    kind: str
    content: str
    tags: list[str] = field(default_factory=list)
    source_tier: str = "root"  # root | live
    verified: bool = False
    helpful: int = 0
    harmful: int = 0
    recipe: dict[str, Any] = field(default_factory=dict)
    trace_ids: list[str] = field(default_factory=list)
    pass_count: int = 0
    false_positive_count: int = 0
    last_verified_at: str = ""

    def score(self) -> int:
        return int(self.helpful) - int(self.harmful)

    def evidence_score(self) -> int:
        return self.score() + int(self.pass_count) - int(self.false_positive_count)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "tags": list(self.tags),
            "source_tier": self.source_tier,
            "verified": bool(self.verified),
            "helpful": int(self.helpful),
            "harmful": int(self.harmful),
            "recipe": dict(self.recipe or {}),
            "trace_ids": list(self.trace_ids),
            "pass_count": int(self.pass_count),
            "false_positive_count": int(self.false_positive_count),
            "last_verified_at": self.last_verified_at,
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any], *, source_tier: str) -> "OpeningBookItem":
        return cls(
            id=str(rec.get("id") or "").strip(),
            kind=str(rec.get("kind") or "").strip(),
            content=str(rec.get("content") or "").strip(),
            tags=[str(t).strip() for t in (rec.get("tags") or []) if str(t).strip()],
            source_tier=str(rec.get("source_tier") or source_tier),
            verified=bool(rec.get("verified", source_tier == "root")),
            helpful=int(rec.get("helpful") or 0),
            harmful=int(rec.get("harmful") or 0),
            recipe=dict(rec.get("recipe") or {}),
            trace_ids=[str(t) for t in (rec.get("trace_ids") or [])],
            pass_count=int(rec.get("pass_count") or 0),
            false_positive_count=int(rec.get("false_positive_count") or 0),
            last_verified_at=str(rec.get("last_verified_at") or ""),
        )


@dataclass
class PlaytestRecipe(OpeningBookItem):
    pass


@dataclass
class AssetAuditRecipe(OpeningBookItem):
    pass


@dataclass
class AnimationAuditRecipe(OpeningBookItem):
    pass


@dataclass
class VisualPlaytestRecipe(OpeningBookItem):
    """Hand-curated structured checklist a VLM can answer about a
    screenshot. Each recipe targets a MECHANISM (grid navigation,
    two-actor facing, first-person perspective) — NOT a specific
    game — so ~10 recipes cover the top-100 games via keyword overlap.

    Matched against (goal + plan + asset names) at critic time. By
    then all three are populated, so even goals that don't name a
    game ("collect dots while avoiding ghosts in corridors") still
    resolve via plan-text + asset-name keyword hits.

    The `recipe` dict carries:
      - applies_keywords: list[str]  — vocabulary the matcher overlaps
        against. Include game names AND mechanic nouns AND action
        verbs so phrasings without genre names still match.
      - strong_hooks: list[str]      — game-name-level tokens that
        win on 1 hit (chess, doom, pacman). Optional.
      - applies_min_matches: int     — keyword-overlap threshold for
        non-strong-hook matches. Default 2.
      - checklist: list[str]         — yes/no questions the VLM
        answers. 6-10 high-signal questions per recipe.
      - format: str                  — output shape, currently
        "yes_no_per_line".
    """
    pass


@dataclass
class ImplementationOutline(OpeningBookItem):
    pass


@dataclass
class OpeningBookHit:
    item: OpeningBookItem
    score: float


def _opening_book_seed_items() -> dict[str, list[OpeningBookItem]]:
    """Small universal root opening book. No generated artifacts or traces."""
    return {
        PLAYTESTS_FILENAME: [
            PlaytestRecipe(
                id="controllable-movement-delta",
                kind="playtest",
                content=(
                    "For controllable canvas games, hold movement keys and assert "
                    "that an exposed player position field or canvas pixels change "
                    "because of input, not ambient animation."
                ),
                tags=["input", "movement", "controls", "player", "canvas"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "input_delta",
                    "keys": ["ArrowRight", "ArrowLeft", "KeyD", "KeyA"],
                    "duration_ms": 250,
                    "expect": "state_or_canvas_delta",
                },
            ),
            PlaytestRecipe(
                id="turn-based-select-commit",
                kind="playtest",
                content=(
                    "For board or card games, test that selecting a legal object "
                    "creates visible selection state, then committing a legal move "
                    "changes board/current-player state."
                ),
                tags=["board", "turn-based", "click", "select", "move", "commit"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "state_delta",
                    "actions": ["click_select", "click_commit"],
                    "expect": "selection_then_board_delta",
                },
            ),
            PlaytestRecipe(
                id="projectile-action-spawns",
                kind="playtest",
                content=(
                    "For action/shooter games, press the primary action key and "
                    "assert that projectile/action state or visible pixels change."
                ),
                tags=["action", "projectile", "fire", "space", "attack", "spawn"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "input_delta",
                    "keys": ["Space", "KeyX", "Enter"],
                    "duration_ms": 150,
                    "expect": "entity_count_or_canvas_delta",
                },
            ),
            PlaytestRecipe(
                id="restart-resets-state",
                kind="playtest",
                content=(
                    "For every game, trigger restart and assert score, timers, "
                    "game-over flags, pending effects, and active entities return "
                    "to a clean initial state."
                ),
                tags=["restart", "reset", "state", "timer", "game-over"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "restart_reset",
                    "selectors": ["#restart", "#restartBtn", "button"],
                    "expect": "state_returns_initial",
                },
            ),
            # Phase 1.5 — autonomous-mode recipes. Driven by the agent's
            # _run_autonomous_playtest hook, NOT the standard harness
            # check pass. Each recipe carries a `type: "behavior_playtest"`
            # marker so the existing opening-book pipeline ignores it
            # (it only handles the legacy `input_delta` / `state_delta`
            # / `restart_reset` types). Every recipe is GENRE-FREE —
            # applicability gates filter on observable structure (state
            # exposure, canvas presence) and never on subject matter.
            PlaytestRecipe(
                id="entity-progress-over-time",
                kind="playtest",
                content=(
                    "For any game with self-driven motion (animations, AI, "
                    "projectiles, timers), with no user input, the game "
                    "state or canvas pixels should advance over a 10-second "
                    "observation window. If nothing changes the game is "
                    "stuck and the user can't tell because the page didn't "
                    "crash. Catches the 'agent shipped a frozen game' bug. "
                    "Skips when no self-driven-motion signal is present "
                    "(score, moving enemies, projectiles, particles, timer) — "
                    "otherwise it false-positives on input-driven arcade "
                    "games waiting for the player to act. Doom 2026-05-23 "
                    "trace burned 2 iters fixing this false positive."
                ),
                tags=["progress", "motion", "stuck", "frozen", "ai", "timer"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "behavior_playtest",
                    # Gate requires: canvas exists AND there's at least one
                    # observable self-driven-motion signal in window.state /
                    # window.gameState. When the gate returns false the
                    # recipe is skipped (trace event:
                    # autonomous_recipe_skipped, reason
                    # applicability_gate_falsy) so input-driven games where
                    # nothing is supposed to move without user input don't
                    # false-positive as "frozen". Genre-free — keyed on
                    # observable state shape, not subject matter.
                    "applies_when": (
                        "(()=>{"
                        "const c=document.querySelector('canvas');"
                        "if(!c||c.width<=0||c.height<=0)return false;"
                        "const s=window.state||window.gameState;"
                        # No exposed state → run conservatively (don't skip).
                        # The "agent shipped a frozen game" failure mode
                        # commonly hides behind unstructured state.
                        "if(!s)return true;"
                        # Enemies / NPCs / opponents with non-zero velocity.
                        "const enemies=s.enemies||s.npcs||s.opponents;"
                        "if(Array.isArray(enemies)){"
                        "for(const e of enemies){if(!e)continue;"
                        "const vx=+e.vx||+e.velocityX||+e.dx||0;"
                        "const vy=+e.vy||+e.velocityY||+e.vz||+e.dy||0;"
                        "if(vx!==0||vy!==0)return true;}}"
                        # Live projectiles / bullets / shots.
                        "const proj=s.projectiles||s.bullets||s.shots;"
                        "if(Array.isArray(proj)&&proj.length>0)return true;"
                        # Live particles / fx.
                        "const parts=s.particles||s.fx;"
                        "if(Array.isArray(parts)&&parts.length>0)return true;"
                        # Score already moving.
                        "if(typeof s.score==='number'&&s.score>0)return true;"
                        # Game timer / clock.
                        "if(typeof s.time==='number'&&s.time>0)return true;"
                        "if(typeof s.timer==='number'&&s.timer>0)return true;"
                        # No self-driven-motion signal → input-driven; skip.
                        "return false;"
                        "})()"
                    ),
                    "input_script": [{"type": "wait", "ms": 10000}],
                    "sample_times_s": [0.0, 2.0, 5.0, 9.5],
                    "check_kind": "any_progress",
                    "finding_label": (
                        "After 10 seconds with no user input, neither the "
                        "canvas pixels nor any exposed game-state field "
                        "changed. The game appears frozen — likely no RAF "
                        "loop, no AI tick, or a stuck game-state."
                    ),
                },
            ),
            PlaytestRecipe(
                id="input-axis-matches-facing",
                kind="playtest",
                content=(
                    "For controllable games that expose a player position "
                    "AND a facing/angle, holding the 'forward' control "
                    "should produce a position delta whose angle matches "
                    "the rendered facing (±15°). Catches the bug where "
                    "the forward key is mapped to a world axis instead of "
                    "the facing-vector projection (cos/sin of facing)."
                ),
                tags=["controls", "facing", "direction", "movement", "forward"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "behavior_playtest",
                    "applies_when": (
                        "(()=>{const s=window.state||window.gameState;"
                        "if(!s)return false;"
                        "const p=s.player||s.ship||s.hero||s;"
                        "const hasPos=typeof p.x==='number'&&typeof p.y==='number';"
                        "const hasFace=typeof p.facing==='number'||"
                        "typeof p.angle==='number'||typeof p.heading==='number'||"
                        "typeof p.rot==='number'||typeof p.rotation==='number';"
                        "return hasPos&&hasFace;})()"
                    ),
                    "input_script": [
                        {"type": "wait", "ms": 200},
                        {"type": "keydown", "key": "ArrowUp", "duration_ms": 1000},
                    ],
                    "sample_times_s": [0.05, 1.30],
                    "check_kind": "facing_matches_movement",
                    "finding_label": (
                        "Held the forward control for 1 s. The position "
                        "moved, but in a direction that doesn't match the "
                        "facing/angle exposed on the player. The forward "
                        "control is likely mapped to a fixed world axis "
                        "(world-X or world-Y) instead of cos/sin of the "
                        "player's facing. Project velocity through the "
                        "facing in the movement update."
                    ),
                },
            ),
            PlaytestRecipe(
                id="held-key-stays-in-bounds",
                kind="playtest",
                content=(
                    "For controllable games that expose a player position "
                    "and a canvas, holding any directional control for "
                    "3 seconds should NOT carry the player outside the "
                    "canvas viewport. Out-of-bounds means the boundary or "
                    "wall logic is missing — the entity flies off-map. "
                    "Generic version of the 'ghosts walk through walls' "
                    "shape."
                ),
                tags=["bounds", "wall", "collision", "boundary", "off-screen"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={
                    "type": "behavior_playtest",
                    "applies_when": (
                        "(()=>{const c=document.querySelector('canvas');"
                        "if(!c||!c.width||!c.height)return false;"
                        "const s=window.state||window.gameState;if(!s)return false;"
                        "const p=s.player||s.ship||s.hero||s;"
                        "return typeof p.x==='number'&&typeof p.y==='number';})()"
                    ),
                    "input_script": [
                        {"type": "wait", "ms": 200},
                        {"type": "keydown", "key": "ArrowRight", "duration_ms": 3000},
                    ],
                    "sample_times_s": [0.05, 3.20],
                    "check_kind": "stays_in_canvas",
                    "finding_label": (
                        "Held ArrowRight for 3 s. The player position "
                        "left the canvas bounds entirely. Either there's "
                        "no boundary / wall logic in the movement update, "
                        "or the entity clamp is missing. Add a clamp or "
                        "wall-collision test in the update step."
                    ),
                },
            ),
        ],
        ASSET_AUDITS_FILENAME: [
            AssetAuditRecipe(
                id="generated-assets-loaded-and-drawn",
                kind="asset_audit",
                content=(
                    "When assets are generated, audit that each requested asset "
                    "appears in a loader table, decodes into an Image, and is "
                    "eventually used by drawImage rather than replaced by a "
                    "procedural placeholder."
                ),
                tags=["assets", "sprites", "drawImage", "loader", "generated"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={"type": "asset_usage", "expect": "loader_and_drawimage"},
            ),
            AssetAuditRecipe(
                id="sprite-alpha-and-distinctness",
                kind="asset_audit",
                content=(
                    "Audit generated sprites for sane alpha coverage and visual "
                    "distinctness between entity classes; flag blank/opaque cards "
                    "and accidental same-looking role sprites."
                ),
                tags=["assets", "alpha", "distinct", "sprites", "visual"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={"type": "asset_stats", "expect": "alpha_and_distinct"},
            ),
        ],
        ANIMATION_AUDITS_FILENAME: [
            AnimationAuditRecipe(
                id="movement-has-midframe",
                kind="animation_audit",
                content=(
                    "For promised movement animation, capture before/mid/after "
                    "frames and verify the object has a real intermediate frame "
                    "instead of teleporting from source to destination."
                ),
                tags=["animation", "movement", "midframe", "lerp", "teleport"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={"type": "before_mid_after", "expect": "intermediate_delta"},
            ),
            AnimationAuditRecipe(
                id="hit-effect-visible-window",
                kind="animation_audit",
                content=(
                    "For capture/hit/explosion effects, verify the effect appears "
                    "during the short event window and disappears afterward."
                ),
                tags=["animation", "effect", "hit", "capture", "particles"],
                verified=True,
                helpful=1,
                pass_count=1,
                recipe={"type": "event_window", "expect": "effect_transient"},
            ),
        ],
        IMPLEMENTATION_OUTLINES_FILENAME: [
            ImplementationOutline(
                id="outline-controllable-canvas-game",
                kind="implementation_outline",
                content=(
                    "For a controllable canvas game: define state first, expose it "
                    "on window, wire input to a single keys/pressed object, update "
                    "movement with capped dt, draw layers in order, then add restart "
                    "and game-over. Keep input/update/draw reading the same state."
                ),
                tags=["canvas", "input", "movement", "state", "arcade"],
                verified=True,
                helpful=1,
                pass_count=1,
            ),
            ImplementationOutline(
                id="outline-turn-based-board",
                kind="implementation_outline",
                content=(
                    "For a turn-based board game: store board[r][c], render row/col "
                    "only at draw boundaries, implement select then legal-target "
                    "commit, block input while animating/AI thinking, and expose a "
                    "small game API for probes."
                ),
                tags=["board", "turn-based", "grid", "select", "legal", "ai"],
                verified=True,
                helpful=1,
                pass_count=1,
            ),
            ImplementationOutline(
                id="outline-asset-backed-animation",
                kind="implementation_outline",
                content=(
                    "For an asset-backed animated game: request base sprites early, "
                    "load/decode them before first draw, use drawImage for generated "
                    "entities, track animation timers in state, and provide procedural "
                    "fallbacks only when asset generation is unavailable."
                ),
                tags=["assets", "animation", "sprites", "drawImage", "loader"],
                verified=True,
                helpful=1,
                pass_count=1,
            ),
        ],
        VERIFIED_FINDINGS_FILENAME: [],
        # Visual playtest recipes live in `memory/visual_playtests.jsonl`
        # as a hand-edited data file (same pattern as memory/playbook.jsonl
        # and memory/skeletons/). NOT seeded from Python here — editing
        # the .jsonl directly is the supported workflow. The loader at
        # `_load_opening_book(VISUAL_PLAYTESTS_FILENAME)` reads from disk
        # and falls back to empty list when the file is missing (which
        # means the matcher returns None for every goal and the critic
        # uses its legacy open-ended prompt — safe degradation).
        VISUAL_PLAYTESTS_FILENAME: [],
        # Component skill library is a hand-edited data file
        # (memory/components.jsonl) like visual_playtests — not seeded here.
        COMPONENTS_FILENAME: [],
    }


def _render_outline_recipe(recipe: dict) -> str:
    """Deep 'book line' render for ONE matched outline (plan stage only).

    Terse imperative lines, no code fences — the same shape playbook
    bullets use, so local models read it as reference data rather than
    new instructions. Returns "" for legacy entries with empty recipes.
    """
    if not isinstance(recipe, dict) or not recipe:
        return ""
    lines: list[str] = []
    state = str(recipe.get("state") or "").strip()
    if state:
        lines.append(f"state: {state}")
    order = [str(s).strip() for s in (recipe.get("order") or []) if str(s).strip()]
    if order:
        lines.append("order: " + " -> ".join(order))
    traps = [str(s).strip() for s in (recipe.get("traps") or []) if str(s).strip()]
    if traps:
        lines.append("traps:")
        lines.extend(f"- {t}" for t in traps)
    tuning = [str(s).strip() for s in (recipe.get("tuning") or []) if str(s).strip()]
    if tuning:
        lines.append("tuning: " + "; ".join(tuning))
    probes = [str(s).strip() for s in (recipe.get("probes") or []) if str(s).strip()]
    if probes:
        lines.append("probes: " + "; ".join(probes))
    return "\n".join(lines)


# Generic visual recipes — too broad to inject as plan-time checklists.
VLM_CHECKLIST_SKIP_IDS = frozenset({
    "canvas-controllable-player",
    "generic-canvas-game-baseline",
})


def render_vlm_checklist_section(
    recipe: VisualPlaytestRecipe,
    *,
    max_items: int = 5,
) -> str:
    """Compact per-mechanism vision checklist for plan-stage opening book.

    Surfaces what `/vlm-critique` will judge so the coder can build
  correctly on iter 1 even when vision review is still off.
    """
    rec = recipe.recipe if isinstance(recipe.recipe, dict) else {}
    checklist = list(rec.get("checklist") or [])[:max_items]
    if not checklist:
        return ""
    lines = [
        f"VLM_CHECKLIST [{recipe.id}] (checked when /vlm-critique is on):",
    ]
    for i, q in enumerate(checklist, start=1):
        q_short = q if len(q) <= 120 else q[:117] + "..."
        lines.append(f"- Q{i}: {q_short}")
    return "\n".join(lines)


def render_opening_book_block(
    outline: OpeningBookHit | None,
    playtests: list[OpeningBookHit],
    asset_audits: list[OpeningBookHit],
    animation_audits: list[OpeningBookHit],
    *,
    char_budget: int = 2600,
    deep: bool = False,
    vlm_checklist: str | None = None,
) -> str:
    """Compact trusted/lower-confidence recipes for prompts.

    `deep=True` (plan stage) additionally renders the matched outline's
    structured recipe fields (state/order/traps/tuning/probes). Code-stage
    callers keep the default shallow render so iterate-loop prompts never
    grow with book depth.
    """
    sections: list[str] = []
    if outline:
        item = outline.item
        outline_text = (
            f"OUTLINE [{item.id}] tier={item.source_tier} score={outline.score:.3f}\n"
            f"{item.content}"
        )
        if deep:
            recipe_text = _render_outline_recipe(getattr(item, "recipe", None))
            if recipe_text:
                outline_text += "\n" + recipe_text
        sections.append(outline_text)

    def _line(label: str, hits: list[OpeningBookHit]) -> str:
        rows = []
        for h in hits:
            item = h.item
            rows.append(
                f"- {label} [{item.id}] tier={item.source_tier} "
                f"score={h.score:.3f}: {item.content}"
            )
        return "\n".join(rows)

    for label, hits in (
        ("PLAYTEST", playtests),
        ("ASSET_AUDIT", asset_audits),
        ("ANIMATION_AUDIT", animation_audits),
    ):
        if hits:
            sections.append(_line(label, hits))
    if vlm_checklist:
        sections.append(vlm_checklist)
    if not sections:
        return ""
    body = "\n\n".join(sections)
    if len(body) > char_budget:
        body = body[:char_budget].rstrip() + "\n[opening book truncated by budget]"
    return (
        "<opening_book>\n"
        "Root `memory/` entries are trusted opening-book recipes. Live "
        "`games/game-memory/` entries are lower-confidence and should only be "
        "used when they directly match the goal. Do not copy traces; use these "
        "as compact implementation/test guidance.\n\n"
        f"{body}\n"
        "</opening_book>"
    )


def render_components_block(
    hits: list[OpeningBookHit],
    *,
    char_budget: int = 2200,
) -> str:
    """Render component-library hits as a <components> block with fenced JS.

    Tested, mechanics-level snippets the model should paste and ADAPT (not
    treat as a library import). Whole entries are dropped — never truncated
    mid-code — when the budget is exceeded, so emitted JS is always complete.
    """
    if not hits:
        return ""
    sections: list[str] = []
    used = 0
    for h in hits:
        item = h.item
        code = str((item.recipe or {}).get("code") or "").strip()
        if not code:
            continue
        section = (
            f"COMPONENT [{item.id}]: {item.content}\n"
            f"```js\n{code}\n```"
        )
        # Drop whole entries past the budget (always keep at least one).
        if sections and used + len(section) > char_budget:
            break
        sections.append(section)
        used += len(section)
    if not sections:
        return ""
    body = "\n\n".join(sections)
    return (
        "<components>\n"
        "Tested, working snippets relevant to this goal. Adapt, don't "
        "import — paste the pattern into your code and rename/modify to fit "
        "your structures. They are reference implementations, not a library.\n\n"
        f"{body}\n"
        "</components>"
    )


class GameMemory:
    """Filesystem-backed memory for the agent.

    All paths are computed from `root` (default memory/). The directory
    is created lazily on first write so a fresh checkout works without setup.
    """

    def __init__(self, root: str | Path = "memory"):
        self._input_root = Path(root)
        
        # Detect standard run vs custom test run
        if self._input_root.name == "memory" and self._input_root.parent == Path("."):
            self.base_root = self._input_root
            self.live_root = Path("games/game-memory")
            self.short_term_root = Path("games")
        else:
            self.base_root = self._input_root
            self.live_root = self._input_root
            self.short_term_root = self._input_root

        # Keep root pointing to live_root for all legacy/writing lookups
        self.root = self.live_root

        self.base_skeletons_dir = self.base_root / "skeletons"
        self.live_skeletons_dir = self.live_root / "skeletons"
        self.goals_dir = self.short_term_root / "goals"
        self.base_mistakes_path = self.base_root / "mistakes.jsonl"
        self.live_mistakes_path = self.live_root / "mistakes.jsonl"
        self.base_opening_book_paths = {
            PLAYTESTS_FILENAME: self.base_root / PLAYTESTS_FILENAME,
            ASSET_AUDITS_FILENAME: self.base_root / ASSET_AUDITS_FILENAME,
            ANIMATION_AUDITS_FILENAME: self.base_root / ANIMATION_AUDITS_FILENAME,
            IMPLEMENTATION_OUTLINES_FILENAME: self.base_root / IMPLEMENTATION_OUTLINES_FILENAME,
            VERIFIED_FINDINGS_FILENAME: self.base_root / VERIFIED_FINDINGS_FILENAME,
            VISUAL_PLAYTESTS_FILENAME: self.base_root / VISUAL_PLAYTESTS_FILENAME,
            COMPONENTS_FILENAME: self.base_root / COMPONENTS_FILENAME,
        }
        self.live_opening_book_paths = {
            PLAYTESTS_FILENAME: self.live_root / PLAYTESTS_FILENAME,
            ASSET_AUDITS_FILENAME: self.live_root / ASSET_AUDITS_FILENAME,
            ANIMATION_AUDITS_FILENAME: self.live_root / ANIMATION_AUDITS_FILENAME,
            IMPLEMENTATION_OUTLINES_FILENAME: self.live_root / IMPLEMENTATION_OUTLINES_FILENAME,
            VERIFIED_FINDINGS_FILENAME: self.live_root / VERIFIED_FINDINGS_FILENAME,
            VISUAL_PLAYTESTS_FILENAME: self.live_root / VISUAL_PLAYTESTS_FILENAME,
            COMPONENTS_FILENAME: self.live_root / COMPONENTS_FILENAME,
        }

        # Compatibility properties for legacy accesses
        self.skeletons_dir = self.live_skeletons_dir
        self.mistakes_path = self.live_mistakes_path

    # --- bootstrap ---------------------------------------------------------

    def ensure(self) -> None:
        """Create directory layout and seed default skeletons if missing.

        Cheap to call repeatedly; agent constructor calls it once.
        """
        try:
            self.base_skeletons_dir.mkdir(parents=True, exist_ok=True)
            self.live_skeletons_dir.mkdir(parents=True, exist_ok=True)
            self.goals_dir.mkdir(parents=True, exist_ok=True)

            # List of all default templates to bootstrap.
            # v2 + board + dom added 2026-05-21 — see plan
            # memory_completeness_review for evidence.
            templates = [
                (DEFAULT_SKELETON_NAME, DEFAULT_SKELETON, None),
                (CANVAS_SKELETON_V2_NAME, CANVAS_SKELETON_V2, CANVAS_BASIC_V2_SIDECAR),
                (CANVAS_3D_SKELETON_NAME, CANVAS_3D_SKELETON, CANVAS_3D_SKELETON_SIDECAR),
                (CANVAS_GRID_SKELETON_NAME, CANVAS_GRID_SKELETON, CANVAS_GRID_SKELETON_SIDECAR),
                (CANVAS_PLATFORMER_SKELETON_NAME, CANVAS_PLATFORMER_SKELETON, CANVAS_PLATFORMER_SKELETON_SIDECAR),
                (CANVAS_SCROLLING_SKELETON_NAME, CANVAS_SCROLLING_SKELETON, CANVAS_SCROLLING_SKELETON_SIDECAR),
                (CANVAS_MODE7_SKELETON_NAME, CANVAS_MODE7_SKELETON, CANVAS_MODE7_SKELETON_SIDECAR),
                (CANVAS_CRAWLER_SKELETON_NAME, CANVAS_CRAWLER_SKELETON, CANVAS_CRAWLER_SKELETON_SIDECAR),
                (CANVAS_MOBILE_SKELETON_NAME, CANVAS_MOBILE_SKELETON, CANVAS_MOBILE_SKELETON_SIDECAR),
                (CANVAS_RPG_SKELETON_NAME, CANVAS_RPG_SKELETON, CANVAS_RPG_SKELETON_SIDECAR),
                (CANVAS_CARDS_SKELETON_NAME, CANVAS_CARDS_SKELETON, CANVAS_CARDS_SKELETON_SIDECAR),
                (CANVAS_PHYSICS_SKELETON_NAME, CANVAS_PHYSICS_SKELETON, CANVAS_PHYSICS_SKELETON_SIDECAR),
                (CANVAS_VOXEL_MINECRAFT_SKELETON_NAME, CANVAS_VOXEL_MINECRAFT_SKELETON, CANVAS_VOXEL_MINECRAFT_SKELETON_SIDECAR),
                (CANVAS_AR_FLICK_SKELETON_NAME, CANVAS_AR_FLICK_SKELETON, CANVAS_AR_FLICK_SKELETON_SIDECAR),
                (CANVAS_LIT_DUNGEON_SKELETON_NAME, CANVAS_LIT_DUNGEON_SKELETON, CANVAS_LIT_DUNGEON_SIDECAR),
                (CANVAS_VFX_PARTICLES_SKELETON_NAME, CANVAS_VFX_PARTICLES_SKELETON, CANVAS_VFX_PARTICLES_SIDECAR),
                (CANVAS_BOARD_TURN_SKELETON_NAME, CANVAS_BOARD_TURN_SKELETON, CANVAS_BOARD_TURN_SKELETON_SIDECAR),
                (CANVAS_DOM_SKELETON_NAME, CANVAS_DOM_SKELETON, CANVAS_DOM_SKELETON_SIDECAR),
            ]

            for name, html, sidecar in templates:
                html_file = self.base_skeletons_dir / name
                if not html_file.exists():
                    html_file.write_text(html, encoding="utf-8")
                if sidecar is not None:
                    sidecar_file = html_file.with_suffix(".json")
                    if not sidecar_file.exists():
                        sidecar_file.write_text(sidecar, encoding="utf-8")
            self._ensure_opening_book_seed_files()
        except Exception:
            # If memory is broken (read-only fs etc) the agent should still
            # run — retrieval just returns empty results.
            pass

    def _ensure_opening_book_seed_files(self) -> None:
        seeds = _opening_book_seed_items()
        for filename, items in seeds.items():
            path = self.base_opening_book_paths[filename]
            if path.exists():
                continue
            with path.open("w", encoding="utf-8") as f:
                for item in items:
                    rec = item.to_record()
                    rec["source_tier"] = "root"
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- opening-book retrieval -------------------------------------------

    def _opening_book_path_pair(self, filename: str) -> tuple[Path, Path]:
        return self.base_opening_book_paths[filename], self.live_opening_book_paths[filename]

    def _load_opening_book_file(
        self,
        path: Path,
        *,
        source_tier: str,
        cls: type[OpeningBookItem] = OpeningBookItem,
    ) -> list[OpeningBookItem]:
        if not path.exists():
            return []
        out: list[OpeningBookItem] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        item = cls.from_record(rec, source_tier=source_tier)
                    except Exception:
                        continue
                    if item.id and item.content:
                        out.append(item)
        except Exception:
            return []
        return out

    def _load_opening_book(
        self,
        filename: str,
        *,
        cls: type[OpeningBookItem] = OpeningBookItem,
    ) -> list[OpeningBookItem]:
        self.ensure()
        base_path, live_path = self._opening_book_path_pair(filename)
        same_path = False
        try:
            same_path = base_path.resolve() == live_path.resolve()
        except Exception:
            same_path = str(base_path) == str(live_path)
        merged: dict[str, OpeningBookItem] = {}
        ordered: list[str] = []
        for item in self._load_opening_book_file(base_path, source_tier="root", cls=cls):
            if same_path and item.source_tier == "live":
                continue
            merged[item.id] = item
            ordered.append(item.id)
        live_items = [] if same_path else self._load_opening_book_file(
            live_path, source_tier="live", cls=cls,
        )
        for item in live_items:
            if item.id not in merged:
                ordered.append(item.id)
            else:
                # Keep root as the trusted opening book; live items with the
                # same id must earn promotion rather than shadowing root.
                item.id = f"live-{item.id}"
                ordered.append(item.id)
            merged[item.id] = item
        return [merged[i] for i in ordered if i in merged]

    @staticmethod
    def _opening_book_similarity(item: OpeningBookItem, query_tokens: list[str]) -> float:
        toks = _tokenize(" ".join([item.id, item.kind, item.content, " ".join(item.tags)]))
        return _score_similarity(query_tokens, toks)

    def _retrieve_opening_book(
        self,
        filename: str,
        *,
        goal: str,
        modality: str | list[str] | None = None,
        k: int = 3,
        cls: type[OpeningBookItem] = OpeningBookItem,
    ) -> list[OpeningBookHit]:
        mod_text = " ".join(modality) if isinstance(modality, list) else (modality or "")
        q = _tokenize(goal) + _tokenize(mod_text)
        if not q:
            return []
        hits: list[OpeningBookHit] = []
        for item in self._load_opening_book(filename, cls=cls):
            sim = self._opening_book_similarity(item, q)
            if sim <= 0:
                continue
            if item.source_tier == "live":
                if sim < 0.08:
                    continue
                if not item.verified or item.evidence_score() <= 0:
                    continue
            quality = 1.0 + 0.10 * _tanh(item.evidence_score() / 5.0)
            tier_bonus = 0.02 if item.source_tier == "root" else 0.0
            hits.append(OpeningBookHit(item=item, score=sim * quality + tier_bonus))
        hits.sort(key=lambda h: (h.score, 1 if h.item.source_tier == "root" else 0), reverse=True)
        return hits[:k]

    def load_playtests(self) -> list[OpeningBookItem]:
        return self._load_opening_book(PLAYTESTS_FILENAME, cls=PlaytestRecipe)

    def retrieve_playtests(
        self, goal: str, modality: str | list[str] | None = None, k: int = 3
    ) -> list[OpeningBookHit]:
        return self._retrieve_opening_book(
            PLAYTESTS_FILENAME, goal=goal, modality=modality, k=k, cls=PlaytestRecipe,
        )

    def load_components(self) -> list[OpeningBookItem]:
        """Component skill library: tested mechanics-level JS snippets."""
        return self._load_opening_book(COMPONENTS_FILENAME)

    def retrieve_components(
        self, goal: str, modality: str | list[str] | None = None, k: int = 3
    ) -> list[OpeningBookHit]:
        """Top-k components by goal/modality similarity (opening-book scoring)."""
        return self._retrieve_opening_book(
            COMPONENTS_FILENAME, goal=goal, modality=modality, k=k,
        )

    def load_asset_audits(self) -> list[OpeningBookItem]:
        return self._load_opening_book(ASSET_AUDITS_FILENAME, cls=AssetAuditRecipe)

    def retrieve_asset_audits(
        self, goal: str, modality: str | list[str] | None = None, k: int = 2
    ) -> list[OpeningBookHit]:
        return self._retrieve_opening_book(
            ASSET_AUDITS_FILENAME, goal=goal, modality=modality, k=k, cls=AssetAuditRecipe,
        )

    def load_animation_audits(self) -> list[OpeningBookItem]:
        return self._load_opening_book(ANIMATION_AUDITS_FILENAME, cls=AnimationAuditRecipe)

    def retrieve_animation_audits(
        self, goal: str, modality: str | list[str] | None = None, k: int = 2
    ) -> list[OpeningBookHit]:
        return self._retrieve_opening_book(
            ANIMATION_AUDITS_FILENAME, goal=goal, modality=modality, k=k, cls=AnimationAuditRecipe,
        )

    def load_visual_playtests(self) -> list[OpeningBookItem]:
        """Load all visual-playtest recipes (root + live tiers)."""
        return self._load_opening_book(
            VISUAL_PLAYTESTS_FILENAME, cls=VisualPlaytestRecipe,
        )

    def find_visual_playtest_for(
        self,
        *,
        goal: str = "",
        plan_text: str = "",
        asset_names: list[str] | None = None,
        default_min_matches: int = 2,
    ) -> tuple[VisualPlaytestRecipe | None, dict]:
        """Pick the best-matching visual playtest recipe for the
        current session. Wrapper around the module-level
        `find_best_visual_playtest` that pulls the recipe list from
        disk (root + live).

        Returns (recipe_or_None, diag) — `diag` is a dict surfacing
        the top-3 candidate scores so the trace can record WHY a
        recipe was/wasn't chosen.
        """
        all_items = self.load_visual_playtests()
        recipes = [r for r in all_items if isinstance(r, VisualPlaytestRecipe)]
        return find_best_visual_playtest(
            recipes,
            goal=goal,
            plan_text=plan_text,
            asset_names=asset_names,
            default_min_matches=default_min_matches,
        )

    def retrieve_implementation_outline(
        self, goal: str, modality: str | list[str] | None = None
    ) -> OpeningBookHit | None:
        hits = self._retrieve_opening_book(
            IMPLEMENTATION_OUTLINES_FILENAME,
            goal=goal,
            modality=modality,
            k=1,
            cls=ImplementationOutline,
        )
        return hits[0] if hits else None

    def append_live_opening_book_item(self, filename: str, item: OpeningBookItem) -> bool:
        """Append a verified candidate to live memory only."""
        try:
            self.ensure()
            path = self.live_opening_book_paths[filename]
            path.parent.mkdir(parents=True, exist_ok=True)
            item.source_tier = "live"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

    # --- skeleton retrieval ------------------------------------------------

    def retrieve_skeleton(self, goal: str) -> SkeletonHit:
        """Pick the best skeleton for a new goal.

        Strategy:
        1. MODALITY DETECTOR (added 2026-05-21) — pure keyword match on
           board/DOM/3D intent. Short goals like "chess game" or "doom"
           tokenize to 1-2 non-stopword tokens; weighted Jaccard cannot
           clear the 0.30 threshold for them. 4/4 May 20-21 traces
           (chess, pac, doom, FPS) hit this failure mode. Strong-hook
           lookup (single decisive token like "chess" / "doom" / "tictactoe")
           catches the 1-keyword case; otherwise ≥ _MODALITY_MIN_HITS
           wins. Genre-free per project rule (UI shape, not subject).
        2. JACCARD over all skeletons + past wins (existing behavior).
        3. FALLBACK is now canvas_basic_v2 (was canvas_basic). v2 pre-
           empts focus-blur / dt-cap / restart-cleanup / DPR-resize /
           lazy-audio / HUD pointer-events failures that the medium 27B
           model keeps reinventing wrong. v1 stays reachable via
           skeleton_mode="default" for tune A/B baselines.
        """
        self.ensure()
        goal_toks = _tokenize(goal)

        # ---- 1. Modality detector (before Jaccard) -----------------------
        modality_pick = self._modality_skeleton(goal)
        if modality_pick is not None:
            return modality_pick

        # ---- 1b. Recipe-routed skeleton (added 2026-06-02) ---------------
        # Reuse the already-correct visual-playtest matcher to pick a skeleton
        # for classes the modality detector doesn't cover (platformer, racing,
        # grid, vfx, …). This lifts specialized-skeleton coverage from ~5 to 12
        # of the 27 curated prompts WITHOUT a second bespoke scorer. The map
        # (_RECIPE_TO_SKELETON) only contains mechanism-aligned pairs; a recipe
        # with no safe skeleton falls through to the Jaccard logic below. The
        # recipe matcher already routes arcade games to top-down/paddle/lane
        # recipes (which are NOT in the map), so this cannot reintroduce the
        # 2D-arcade→3D/board misroute — verified by the regression tests.
        recipe_pick = self._recipe_routed_skeleton(goal)
        if recipe_pick is not None:
            return recipe_pick

        best: SkeletonHit | None = None
        paths: dict[str, Path] = {}
        # Base skeletons
        if self.base_skeletons_dir.exists():
            for p in self.base_skeletons_dir.glob("*.html"):
                paths[p.name] = p
        # Live skeletons (override base)
        if self.live_skeletons_dir.exists():
            for p in self.live_skeletons_dir.glob("*.html"):
                paths[p.name] = p

        for name, path in sorted(paths.items()):
            # v1 (canvas_basic.html) is bootstrapped to disk only so
            # skeleton_mode="default" tune baseline has a file to read.
            # It is NOT a retrieval candidate — v2 is the new fallback.
            # (Locked 2026-05-21 — see retrieve_skeleton docstring.)
            if name == DEFAULT_SKELETON_NAME:
                continue
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
                src_toks = _tokenize(source_goal)
                score = _score_similarity(goal_toks, src_toks)
                # A 0.0 sidecar match means NO token overlap whatsoever — not
                # a real candidate. Skip so we don't accidentally win ties
                # against the v2 fallback by being alphabetically first.
                # (Caught 2026-05-21 when "asteroids" picked canvas_3d_basic
                # at 0.0 because 3d was first in sorted order.)
                if score <= 0.0:
                    continue
                # A BUNDLED specialized scaffold may only win on a DISTINCTIVE
                # shared token — a generic filler word (game/grid/light/space/
                # move/select/…) is not enough. Otherwise one incidental token
                # routes a plain 2D arcade goal to a wrong specialized scaffold
                # (see _SKELETON_GENERIC_TOKENS). Past-win "won_" skeletons keep
                # the score-floor gate below instead of this distinctiveness one.
                if not name.startswith("won_"):
                    distinct = (set(goal_toks) & set(src_toks)) - _SKELETON_GENERIC_TOKENS
                    need = 2 if name in _SKELETON_SPECIALIZED_STRICT else 1
                    if len(distinct) < need:
                        continue
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
            # Low-similarity past-win skeletons hurt more than help (the
            # "KEEP its structure" instruction in prompts_v1 forces a
            # bad scaffold on a mismatched goal). If the winner is a
            # past-win match below the threshold, fall back to the
            # bundled empty template — the model builds fresh instead
            # of fighting wrong structure. BUNDLED skeletons (canvas_*) are
            # gated up in the scoring loop instead (they must share a DISTINCTIVE
            # token to be a candidate at all — see _SKELETON_GENERIC_TOKENS), so
            # here we only apply the stricter past-win score floor. Past-win
            # files start with "won_" so the prefix-check distinguishes them
            # deterministically. Added 2026-05-21 after a pac-man trace showed
            # Jaccard at 0.23 (< 0.30) discarding the correct grid scaffold pick.
            is_past_win = best.name.startswith("won_")
            below_threshold = (
                is_past_win
                and best.source_goal is not None
                and best.score < _SKELETON_MIN_SIM
            )
            if below_threshold:
                # v2 fallback (was DEFAULT_SKELETON_NAME). Locked 2026-05-21
                # — 4/4 newest traces hit this branch and v1's bare scaffold
                # forced the model to re-invent boilerplate. v1 still wins
                # for skeleton_mode="default" (tune baseline).
                fallback_path = self.live_skeletons_dir / CANVAS_SKELETON_V2_NAME
                if not fallback_path.exists():
                    fallback_path = self.base_skeletons_dir / CANVAS_SKELETON_V2_NAME
                try:
                    fallback_html = fallback_path.read_text(encoding="utf-8")
                except Exception:
                    fallback_html = CANVAS_SKELETON_V2
                return SkeletonHit(
                    name=CANVAS_SKELETON_V2_NAME,
                    html=fallback_html,
                    score=0.0,
                    source_goal=None,
                )
            return best
        # Fully empty memory dir — return v2 in-memory (was DEFAULT_SKELETON).
        return SkeletonHit(
            name=CANVAS_SKELETON_V2_NAME, html=CANVAS_SKELETON_V2, score=0.0,
            source_goal=None,
        )

    def _modality_skeleton(self, goal: str) -> SkeletonHit | None:
        """Return a SkeletonHit if the goal cleanly matches a modality
        (board / DOM / 3D), else None. Single STRONG-HOOK token (e.g.
        "chess", "doom", "tictactoe") wins on its own; otherwise we need
        ≥ _MODALITY_MIN_HITS keyword matches to commit. See class docstring
        on retrieve_skeleton for the trace evidence motivating this.
        """
        # 3D is checked first because some 3D goals also mention "board"
        # ("3D chess") but should pick the 3D scaffold, not the board one.
        threed_hits = _detect_3d_intent(goal)
        if any(h in _3D_STRONG_HOOKS for h in threed_hits) or len(threed_hits) >= _MODALITY_MIN_HITS:
            return self._load_modality(CANVAS_3D_SKELETON_NAME, CANVAS_3D_SKELETON,
                                       source_goal_tokens=threed_hits)

        board_hits = _detect_board_intent(goal)
        if any(h in _BOARD_STRONG_HOOKS for h in board_hits) or len(board_hits) >= _MODALITY_MIN_HITS:
            return self._load_modality(CANVAS_BOARD_TURN_SKELETON_NAME, CANVAS_BOARD_TURN_SKELETON,
                                       source_goal_tokens=board_hits)

        dom_hits = _detect_dom_intent(goal)
        if any(h in _DOM_STRONG_HOOKS for h in dom_hits) or len(dom_hits) >= _MODALITY_MIN_HITS:
            return self._load_modality(CANVAS_DOM_SKELETON_NAME, CANVAS_DOM_SKELETON,
                                       source_goal_tokens=dom_hits)

        return None

    def _load_modality(self, name: str, fallback_html: str,
                       *, source_goal_tokens: list[str]) -> SkeletonHit:
        """Helper for `_modality_skeleton`: load the on-disk template (so
        the agent sees the exact bytes the user can edit), falling back to
        the bundled constant if disk read fails.
        """
        path = self.live_skeletons_dir / name
        if not path.exists():
            path = self.base_skeletons_dir / name
        try:
            html = path.read_text(encoding="utf-8")
        except Exception:
            html = fallback_html
        # Score signals "modality match"; > _SKELETON_MIN_SIM so callers
        # downstream don't think it's a low-confidence pick.
        return SkeletonHit(
            name=name, html=html, score=1.0,
            source_goal=" ".join(source_goal_tokens) if source_goal_tokens else None,
        )

    def _recipe_routed_skeleton(self, goal: str) -> SkeletonHit | None:
        """Pick a skeleton by reusing the visual-playtest matcher (added
        2026-06-02). The recipe layer routes accurately (strong_hooks →
        applies_keywords overlap); we map its result to a skeleton via
        `_RECIPE_TO_SKELETON`. Returns None when no recipe matches or the
        matched recipe has no mapped skeleton — caller then falls through to
        the Jaccard logic. See `_RECIPE_TO_SKELETON` for why this can't
        reintroduce the 2D-arcade misroute.
        """
        try:
            recipe, _diag = self.find_visual_playtest_for(goal=goal)
        except Exception:
            return None
        if recipe is None:
            return None
        skel_stem = _RECIPE_TO_SKELETON.get(recipe.id)
        if not skel_stem:
            return None
        name = skel_stem + ".html"
        path = self.live_skeletons_dir / name
        if not path.exists():
            path = self.base_skeletons_dir / name
        if not path.exists():
            return None
        try:
            html = path.read_text(encoding="utf-8")
        except Exception:
            return None
        # Score 1.0 (same confidence band as a modality pick) — this is a
        # deliberate, recipe-backed match, not a low-confidence Jaccard hit.
        return SkeletonHit(name=name, html=html, score=1.0, source_goal=recipe.id)

    # --- mistake retrieval -------------------------------------------------

    def retrieve_mistakes(self, error_signature: str, k: int = 3) -> list[MistakeHit]:
        """Look up past mistakes whose signature is similar to this one.

        Used by the diagnose prompt to give the model "you've seen this
        before, here's what worked" hints. Empty list if no memory yet.
        """
        sig_toks = _tokenize(error_signature)
        if not sig_toks:
            return []

        hits: list[MistakeHit] = []

        def load_from_file(path: Path):
            if not path.exists():
                return
            try:
                with path.open("r", encoding="utf-8") as f:
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
                pass

        load_from_file(self.base_mistakes_path)
        load_from_file(self.live_mistakes_path)

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
            with self.live_mistakes_path.open("a", encoding="utf-8") as f:
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
                skel_path = self.live_skeletons_dir / skel_name
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
# so you can manually prune rules that fire on failed runs and reward
# rules that fire on passing ones, without context-collapsing rewrites.
# ===========================================================================


PLAYBOOK_FILENAME = "playbook.jsonl"


@dataclass
class Bullet:
    """A single playbook entry: a one-paragraph rule with metadata.

    `tags` are short strings used both for retrieval (match against goal +
    code) and for organization. `helpful`/`harmful` are running counters
    you can hand-update when a bullet proves itself or misfires.
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
            "For tile-grid levels, PLACE spawn/exit/enemies/pickups by "
            "SCANNING empty cells in code — never hand-verify coordinates "
            "in your reply. Snippet: function pickEmpty(map){ while(true){"
            " const x=1+Math.floor(Math.random()*(W-2)), "
            "y=1+Math.floor(Math.random()*(H-2)); if(map[y][x]===0) "
            "return {x:x+0.5,y:y+0.5}; } }. Enumerating 'row 9 col 3 = "
            "wall!' in <think> burns thousands of tokens and often never "
            "reaches <html_file>."
        ),
        tags=[
            "maze", "grid", "spawn", "placement", "level", "entity",
            "tile", "dungeon", "tilemap", "enemy", "pickup",
            "thinktag-thrash",
        ],
    ),
    Bullet(
        id="projection-3d-wireframe",
        content=(
            "3D wireframe perspective on 2D canvas (Battlezone-style): "
            "(1) Rotate to camera space: dx=x-px, dz=z-pz; rx=dx*cos(-θ)-"
            "dz*sin(-θ); rz=dx*sin(-θ)+dz*cos(-θ). "
            "(2) Project: if(rz<=0.1) return; screenX=cx+(rx/rz)*fov; "
            "screenY=cy-(y/rz)*fov. "
            "(3) Sort objects by descending rz (painter's algorithm) so "
            "closer geometry draws on top; connect with beginPath/moveTo/"
            "lineTo/stroke."
        ),
        tags=["3d", "projection", "wireframe", "battlezone", "perspective", "vector", "camera", "rotation"],
    ),
    Bullet(
        id="platformer-ladders-and-one-way",
        content=(
            "For platformers with climbing (e.g. Donkey Kong) and one-way platforms: "
            "(1) Ladders: Allow climbing only when entity's center is closely aligned horizontally "
            "with the ladder's x position. Set gravity to zero and allow vertical velocity while "
            "climbing. (2) One-way platforms: Allow entities to jump up through platforms by checking "
            "collisions ONLY when velocity vy is positive (falling) AND the entity's feet (bottom) "
            "are above the platform's top edge before the step. Check: if (vy > 0 && feet <= platformTop "
            "&& feet + vy >= platformTop) { land_on_platform }."
        ),
        tags=["ladder", "climbing", "one-way", "platformer", "gravity", "jump", "donkey-kong"],
    ),
    Bullet(
        id="fighting-states-and-hitboxes",
        content=(
            "For fighting/action games (e.g. Street Fighter): (1) State Machine: "
            "Separate characters into strict states: idle, walking, jumping, crouching, "
            "attack-startup, attack-active, attack-recovery, hitstun, knockback. Block input "
            "during attacks, hitstun, and knockback. (2) Combat Collision: Do NOT check plain player "
            "bounding-box overlaps for attacks. Instead, define 'hurtboxes' (body regions that can "
            "be hit) and 'hitboxes' (attack regions, e.g. fist/foot). Check overlap of attacker's "
            "active hitbox with defender's hurtbox ONLY during 'attack-active' animation frames."
        ),
        tags=["fighting", "state-machine", "hitbox", "hurtbox", "combat", "collision", "melee", "street-fighter"],
    ),
    Bullet(
        id="isometric-grid-math",
        content=(
            "To project a 3D grid onto an isometric screen coordinate system (e.g. Q-bert style): "
            "For a grid coordinate (row, col, height): screenX = centerX + (col - row) * (tileWidth / 2); "
            "screenY = centerY + (col + row) * (tileHeight / 2) - height * heightScale. "
            "To sort rendering order to prevent overlaps (painter's algorithm), draw entities "
            "by sorting from lowest (row + col) to highest. For isometric jumping, animate "
            "entity's height with a parabolic curve: height = jumpApex * (4 * t * (1 - t)) where t goes 0..1."
        ),
        tags=["isometric", "grid", "q-bert", "math", "projection", "render-order", "isometric-jump"],
    ),
    Bullet(
        id="corner-sliding-alignment",
        content=(
            "To implement smooth tile-based corner sliding (e.g. Pac-Man) in continuous movement: "
            "Allow the player to buffer turns before aligning perfectly with a tile axis. If key is pressed, "
            "check if coordinate orthogonal to movement is close to a tile boundary (e.g., within a snap tolerance of "
            "8px). If close, snap coordinate directly to tile center (p.y = Math.floor(p.y/TILE)*TILE) and change "
            "velocity. This prevents players from sticking to walls when turning corridors."
        ),
        tags=["grid", "movement", "corner", "snap", "alignment", "corridor", "pacman", "pac-man"],
    ),
    Bullet(
        id="grid-wave-movement",
        content=(
            "For Space Invaders alien wave formations: Store coordinates in a grid matrix or array. "
            "Move all aliens horizontally together. Keep track of left-most and right-most active bounds. "
            "When either bound touches screen margins, reverse horizontal direction for all aliens and "
            "increment their y position downward by one step. To keep tension, increase movement update "
            "speed proportionally as the number of remaining aliens decreases (e.g., delay = baseDelay * (activeCount / total))."
        ),
        tags=["wave", "grid", "formation", "movement", "space-invaders", "aliens", "shooter"],
    ),
    Bullet(
        id="segmented-entity-follow",
        content=(
            "To make segment links cleanly follow a leader (e.g. Centipede): Store a history array of the leader's "
            "coordinates (e.g., up to 200 points). Each follower segment is assigned a specific history index offset "
            "(e.g., segment i reads index i * segmentSpacing from history). On update, push leader's latest coordinate "
            "to the front of the history array and trim the tail. This prevents segments from overlapping or separating during turns."
        ),
        tags=["follow", "segmented", "centipede", "tail", "history", "movement", "snake"],
    ),
    Bullet(
        id="ball-paddle-angle-bias",
        content=(
            "To prevent endless horizontal/vertical bouncing in Breakout: Modify ball bounce angle based on "
            "where it strikes the paddle. Calculate offset = (ball.x - paddle.x) / paddle.width (valued -0.5 to +0.5). "
            "Determine bounce direction angle: angle = (offset * maxBiasAngle) - Math.PI/2. Set new ball velocities: "
            "vx = Math.cos(angle) * speed; vy = Math.sin(angle) * speed (making vy negative/up). This rewards precision hits."
        ),
        tags=["breakout", "bounce", "paddle", "physics", "angle", "ball", "collision", "arkanoid"],
    ),
    Bullet(
        id="parallax-coordinate-camera",
        content=(
            "For side-scrolling worlds (e.g. Defender): Create a camera object holding a single horizontal "
            "offset: camera.x = player.x - centerX. When rendering, wrap context drawings inside ctx.save() / "
            "ctx.translate(-camera.x, -camera.y). Render background details with an offset multiplier (e.g., "
            "star.x - camera.x * 0.3) to achieve parallax depth. Automatically clear/recycle bullets and "
            "enemies once they move beyond the viewport boundaries (e.g., x < camera.x - 100)."
        ),
        tags=["scrolling", "camera", "parallax", "viewport", "defender", "wrap", "sidescroller"],
    ),
    Bullet(
        id="mode7-ground-projection",
        content=(
            "SNES Mode 7 ground projection on 2D canvas (Mario Kart style): "
            "for each scanline y below horizon, distance = fov / (y + 1); "
            "scaleX = distance / fov; stepX = sin(θ)*scaleX, stepY = "
            "-cos(θ)*scaleX. Per pixel along the line: worldX = player.x "
            "+ cos(θ)*distance + (screenX - cx)*stepX (and matching "
            "worldY). Sample track texture, clamp bounds, write to "
            "getImageData buffer."
        ),
        tags=["mode7", "mode-7", "projection", "perspective", "racer", "3d", "retro", "mario-kart"],
    ),
    Bullet(
        id="tetris-matrix-rotation",
        content=(
            "To rotate 2D piece matrices in Tetris: Transpose and reverse the grid array: "
            "rotated = matrix[0].map((_, i) => matrix.map(row => row[i]).reverse()). "
            "To prevent clipping walls on rotation, implement a 'wall-kick' guard: "
            "If the rotated shape collides with side boundaries or locked cells, try shifting "
            "its position left or right by 1 or 2 cells. If all shifts still collide, revert rotation. "
            "When rows are cleared, slice them out and unshift empty rows of width W to the top of the board."
        ),
        tags=["matrix", "rotation", "tetris", "blocks", "grid", "wall-kick", "row-clear"],
    ),
    Bullet(
        id="crawler-wall-sliding",
        content=(
            "To prevent players from sticking to walls during diagonal movement (e.g. Gauntlet): "
            "Decompose player movement into separate horizontal and vertical steps. "
            "Step 1: Apply x movement `p.x += p.vx * dt`, and check collisions. If overlapping a wall boundary, "
            "snap x position back and zero out `vx`. Step 2: Apply y movement `p.y += p.vy * dt`, and check "
            "collisions. If overlapping, snap y back. This allows the player to slide smoothly along walls."
        ),
        tags=["crawler", "collision", "slide", "wall-sliding", "diagonal", "gauntlet", "movement"],
    ),
    Bullet(
        id="shared-camera-bounds",
        content=(
            "For shared shared-screen multiplayer viewports (e.g. Gauntlet): "
            "Calculate camera center as the average coordinates of all active players: "
            "centerX = sum(players.x) / count. Clamp camera coordinate boundaries to global "
            "level bounds. Clamping players: Prevent players from leaving screen edges by checking "
            "their distance to the clamped camera bounds. If a player coordinates `p.x` leaves "
            "`centerX - halfScreenWidth`, clamp their coordinates to the screen margin."
        ),
        tags=["multiplayer", "camera", "shared-screen", "clamp", "viewport", "gauntlet", "bounds"],
    ),
    Bullet(
        id="mobile-joystick-theta",
        content=(
            "To calculate player velocity from virtual joystick touches: "
            "On pointermove, compute offsets: dx = touchX - joystickStartX, dy = touchY - joystickStartY. "
            "Find absolute distance and angle: dist = Math.hypot(dx, dy), theta = Math.atan2(dy, dx). "
            "If distance exceeds max joystick radius, clamp touch visual coordinates to `maxRadius`. "
            "Calculate player velocities: vx = Math.cos(theta) * speed * Math.min(1, dist/maxRadius), "
            "and matching vy. This ensures responsive and precise multi-directional mobile steering."
        ),
        tags=["mobile", "touch", "joystick", "theta", "angle", "velocity", "trig", "controls"],
    ),
    Bullet(
        id="mobile-letterbox-scaling",
        content=(
            "To letterbox a standard canvas for mobile viewports (e.g. iPad/iPhone): "
            "On resize, fetch window aspect ratio and scale canvas container styles. "
            "Check: scale = Math.min(window.innerWidth / baseW, window.innerHeight / baseH). "
            "Set canvas styled width/height: cvs.style.width = (baseW * scale) + 'px'; "
            "cvs.style.height = (baseH * scale) + 'px'. Center canvas container with flexbox "
            "or fixed absolute positioning. This prevents screen distortion on rotated devices."
        ),
        tags=["mobile", "scaling", "responsive", "letterbox", "aspect-ratio", "resize", "ios"],
    ),
    Bullet(
        id="drag-and-drop-snapping",
        content=(
            "For card/board drag-and-drop mechanics (e.g. Solitaire): "
            "On pointerdown, test bounding box of all items from top-drawn to bottom. "
            "On match, record active item and calculate dragging offset relative to cursor: "
            "offsetX = mouseX - item.x. On pointermove, set `item.x = mouseX - offsetX`. "
            "On pointerup, check overlaps with valid slots using circle-distance or AABB. "
            "If overlapping, snap item coordinate directly to target coordinates, else lerp back to original slot."
        ),
        tags=["drag-and-drop", "mouse", "touch", "snap", "board", "puzzle", "cards", "solitaire"],
    ),
    Bullet(
        id="discrete-tile-stepping",
        content=(
            "To implement discrete grid step movement (e.g. Pokemon/Ultima): "
            "Do NOT move player sprites continuously on arrow holds. Store separate coordinates: "
            "gridX/gridY (logical grid) and animX/animY (visual coordinates). When a direction key is pressed "
            "and player is idle: set target cells, block input, and flag player as moving. "
            "Increment animation offset: t += dt * speed; visually interpolate: animX = animX + (gridX - animX) * t. "
            "When t >= 1, snap visual coordinates to grid, unblock input, and reset t = 0."
        ),
        tags=["grid", "rpg", "discrete", "tile", "step", "interpolation", "lerp", "movement"],
    ),
    Bullet(
        id="gravity-trajectory-bounce",
        content=(
            "To bounce projectile circles off wall boundaries (e.g. Bubble Shooter): "
            "Upon edge collision check, mirror the horizontal velocity: vx = -vx. "
            "To reflect projectile circles off line segments: find normal vector of segment. "
            "Velocity reflection formula: R = V - 2 * (V . N) * N. For gravity path forecasting: "
            "Plot trajectory coordinates inside a simple loop: nextX = x + vx * t; nextY = y + vy * t + 0.5 * gravity * t*t. "
            "Draw forecasted points on canvas as a guide line."
        ),
        tags=["physics", "trajectory", "bounce", "vector", "reflection", "gravity", "launch", "shooter"],
    ),
    Bullet(
        id="pathfinding-bfs-grid",
        content=(
            "To implement simple grid pathfinding for enemy chasers (e.g. Pac-Man ghosts, Gauntlet ghosts) avoiding walls: "
            "Use Breadth-First Search (BFS) to find the shortest path in a 2D tile map. "
            "Algorithm: Let queue = [[start]]; let visited = set(start). While queue is not empty, pop first path. "
            "Get last cell in path. If it equals target, return first step of path. "
            "Otherwise, for each neighboring cell (up, down, left, right), if inside map, not a wall, and not visited: "
            "visited.add(neighbor) and push path + [neighbor] to queue. Falls back to straight-line direction vector when target is unreached."
        ),
        tags=["pathfinding", "bfs", "ghosts", "enemy-ai", "maze", "chase", "grid", "map", "path"],
    ),
    Bullet(
        id="enemy-fsm-states",
        content=(
            "To manage complex enemy behaviors in arcade games: Implement a Finite State Machine (FSM) "
            "with states: 'patrol' (walk between spawn nodes), 'alert' (stop and look around when player is near), "
            "'chase' (pathfind towards player coordinates), and 'attack' (trigger attack-frames and pause movement). "
            "Transition triggers: distance to player < alertRadius transitions to 'alert', distance < chaseRadius transitions "
            "to 'chase', and distance < attackRadius transitions to 'attack'. Use cooldown-ticks or timers to enforce state durations."
        ),
        tags=["fsm", "state-machine", "enemy-ai", "combat", "states", "behavior", "ai", "arcade"],
    ),
    Bullet(
        id="pseudo3d-curved-road",
        content=(
            "To project an Out Run/retro style curved 3D road onto a 2D canvas: "
            "Use a scanline camera projection. For each screen line y below horizon (e.g., from top to bottom): "
            "Calculate normalized depth: z = fov / (y - horizon). Compute horizontal road center shift: "
            "roadX = baseCenterX + curveAccumulator * (z * z) + Math.sin(z * frequency) * amplitude. "
            "Project road width: screenW = baseRoadWidth / z. Draw road segment from `roadX - screenW` to "
            "`roadX + screenW`. Scale obstacles and cars proportionally: scale = spriteSize / z, and draw "
            "with x offset centered relative to roadX."
        ),
        tags=["pseudo-3d", "pseudo3d", "road", "racer", "scanline", "curve", "projection", "outrun", "pole-position"],
    ),
    Bullet(
        id="animation-frame-timing",
        content=(
            "To manage smooth sprite animations (running, swinging, hitting) with variable frame rates: "
            "Track animation state holding: animTimer, currentFrameIndex, and activeSpriteSheet. "
            "On update, increment: `animTimer += dt`. If `animTimer >= 1 / fps`: reset timer `animTimer -= 1 / fps` "
            "and advance frame `currentFrameIndex = (currentFrameIndex + 1) % totalFrames`. "
            "When rendering, draw image using clipping coordinates: `ctx.drawImage(sheet, currentFrameIndex * frameW, 0, "
            "frameW, frameH, x, y, w, h)`. This keeps visual cycles consistent and speed-independent."
        ),
        tags=["animation", "sprite-sheet", "sprites", "frame-rate", "clipping", "timing", "frames", "loop"],
    ),
    Bullet(
        id="input-buffering-queue",
        content=(
            "To implement responsive input buffering for fighting or action games (e.g. Street Fighter, platformers): "
            "Create a buffer queue array to store recent inputs with timestamps: `inputBuffer.push({ key, t: now })`. "
            "On update, filter out old actions: `inputBuffer = inputBuffer.filter(item => now - item.t < bufferWindowSeconds)` "
            "(window is typically 0.15 to 0.25 seconds). When a player exits recovery states, hitstun, or lands "
            "on the ground: check if any valid command key (e.g., 'jump' or 'attack') exists in the active buffer. "
            "If found, consume it immediately: execute action and clear buffer."
        ),
        tags=["input", "buffer", "buffering", "queue", "responsive", "fighting", "action", "combos"],
    ),
    Bullet(
        id="resource-meters-hud",
        content=(
            "To render smooth, glowing, or animated health and ammo meters on canvas: "
            "Draw a background bar: `ctx.fillStyle = '#300'; ctx.fillRect(x, y, maxW, h);`. "
            "Linearly interpolate visible value to actual value for a smooth animated fill effect: "
            "`visibleVal += (actualVal - visibleVal) * 0.1` (or `dt * speed`). "
            "Draw filled bar: `ctx.fillStyle = '#f33'; ctx.fillRect(x, y, (visibleVal / maxVal) * maxW, h);`. "
            "Add a glossy stroke outline and thin vertical interval lines to make it look clean and highly readable."
        ),
        tags=["hud", "health", "ammo", "meter", "ui", "lerp", "canvas-draw", "glossy"],
    ),
    # ---- Added 2026-05-21 from May 20-21 trace evidence ------------------
    # Highest-frequency probe failure across pac/dk/sf/doom/FPS traces was
    # missing window.gameState / window.game.reset. Promotes the recurring
    # diagnose hint into a proactive rule.
    Bullet(
        id="expose-state-on-window",
        content=(
            "Acceptance probes look up state via window.gameState, "
            "window.state, or window.game.reset() — un-exposed state fails "
            "probes even when the game works. Always end init with: "
            "window.gameState = state; window.game = { reset: () => "
            "resetGame() }. Score, player position, currentPlayer, and any "
            "reset entry point must be reachable from a probe in ≤3 lines."
        ),
        tags=["state", "window", "probe", "gameState", "exposure", "reset",
              "testing", "acceptance"],
    ),
    # Promoted from learned -> seed 2026-05-21 (helpful=1 in live).
    # Documents the 3s probe warmup window so the model designs init
    # accordingly instead of paying it in lost iters.
    Bullet(
        id="probe-warmup-state-exposure",
        content=(
            "Automated acceptance probes typically evaluate window.gameState "
            "after a ~3-second warmup. Expose the state object synchronously "
            "before any async asset loading or initialization callbacks. "
            "Remove or minimize startup delays (ready timers, intro screens, "
            "loading gates) so gameplay begins immediately and probes can "
            "verify score increases or entity positions within the evaluation "
            "window."
        ),
        tags=["probe", "testing", "warmup", "state", "async", "ready-timer"],
    ),
    # Turn-based board mechanics. Genre-free (no chess/checkers rules) —
    # just the select-then-move state machine that all of them share.
    Bullet(
        id="turn-based-select-move",
        content=(
            "Turn-based board UIs use a two-click commit: state.selected = "
            "null at start; clicking your own piece sets selected and "
            "computes legalMoves(); clicking a legal target calls "
            "applyMove() then state.currentPlayer = (currentPlayer === 'W' "
            "? 'B' : 'W'). Block input during animations or AI thinking. "
            "Highlight the selected cell + legal targets so the user knows "
            "what's clickable."
        ),
        tags=["turn-based", "board", "select", "move", "alternate", "hotseat",
              "click", "two-click", "chess", "checkers"],
    ),
    Bullet(
        id="board-grid-indexing",
        content=(
            "Store a 2D board as board[r][c] with r=row=y-index, c=col=x-"
            "index — NEVER mix (x,y) and (r,c) in the same function. "
            "Render with: for(let r=0;r<SIZE;r++) for(let c=0;c<SIZE;c++) "
            "{ const piece = board[r][c]; drawAt(c*TILE, r*TILE, piece); }. "
            "Always (row, col) order in code; (col*TILE, row*TILE) only at "
            "the draw boundary."
        ),
        tags=["board", "grid", "row", "col", "index", "tile", "indexing"],
    ),
    Bullet(
        id="click-cell-from-pointer",
        content=(
            "Pointer -> board cell: const rect = cvs.getBoundingClientRect(); "
            "const mx = e.clientX - rect.left, my = e.clientY - rect.top; "
            "const c = Math.floor(mx / TILE), r = Math.floor(my / TILE); if "
            "(r < 0 || r >= SIZE || c < 0 || c >= SIZE) return; — rejects "
            "clicks outside the board. Use pointerdown (unified mouse + "
            "touch + pen) and call e.preventDefault() if the page scrolls."
        ),
        tags=["pointer", "click", "cell", "board", "getBoundingClientRect",
              "tile", "input"],
    ),
]


class Playbook:
    """Filesystem-backed playbook of structured rules.

    On first use the file is seeded with SEED_BULLETS; subsequent saves
    persist the merged set. Retrieval is keyword/Jaccard against the goal
    and (optionally) the in-progress code.
    """

    def __init__(self, base_root: str | Path = "memory", live_root: str | Path | None = None, root: str | Path | None = None):
        if root is not None:
            self.base_root = Path(root)
        else:
            self.base_root = Path(base_root)
        
        if live_root is None:
            if self.base_root.name == "memory" and self.base_root.parent == Path("."):
                self.live_root = Path("games/game-memory")
            else:
                self.live_root = self.base_root
        else:
            self.live_root = Path(live_root)

        self.base_path = self.base_root / PLAYBOOK_FILENAME
        self.live_path = self.live_root / PLAYBOOK_FILENAME

        # Learner / curator / writeback persist to the root playbook (tracked in
        # git). games/game-memory/playbook.jsonl is optional local overlay on read
        # only — not committed.
        self.path = self.base_path
        # Compatibility properties for legacy calls
        self.root = self.base_root

    # --- bootstrap ---------------------------------------------------------

    def ensure(self) -> None:
        try:
            self.base_root.mkdir(parents=True, exist_ok=True)
            self.live_root.mkdir(parents=True, exist_ok=True)
            if not self.base_path.exists():
                self._save_all_to_path(SEED_BULLETS, self.base_path)
        except Exception:
            pass

    def _save_all_to_path(self, bullets: list[Bullet], path: Path) -> None:
        try:
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for b in bullets:
                    if not b.created_at:
                        b.created_at = datetime.utcnow().isoformat() + "Z"
                    f.write(b.to_jsonl() + "\n")
            tmp.replace(path)
        except Exception:
            pass

    def _save_all(self, bullets: list[Bullet]) -> None:
        self._save_all_to_path(bullets, self.path)

    def _load_from_path(self, path: Path) -> list[Bullet]:
        if not path.exists():
            return []
        out: list[Bullet] = []
        try:
            with path.open("r", encoding="utf-8") as f:
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

    def load_all(self) -> list[Bullet]:
        self.ensure()
        
        # Load base bullets
        base_bullets = self._load_from_path(self.base_path)
        
        # Load live bullets
        live_bullets = self._load_from_path(self.live_path)
        
        # Merge them by ID (live overrides base)
        merged: dict[str, Bullet] = {b.id: b for b in base_bullets}
        for b in live_bullets:
            merged[b.id] = b
            
        # Preserve original base/seed bullet ordering, followed by any new live bullets
        ordered_merged: list[Bullet] = []
        seen = set()
        for b in base_bullets:
            ordered_merged.append(merged[b.id])
            seen.add(b.id)
        for b in live_bullets:
            if b.id not in seen:
                ordered_merged.append(b)
                seen.add(b.id)
                
        return ordered_merged

    # --- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        goal: str,
        *,
        code: str = "",
        k: int = 8,
        stage: str = "code",
        modality_tokens: list[str] | None = None,
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

        `modality_tokens` (optional, added 2026-05-21): caller-supplied
        keyword list from the modality detector (e.g. ["grid","tile","maze",
        "pacman"] for a pac-man goal). Appended to the query so bullets
        tagged with those modality words retrieve above the 0.05 Jaccard
        noise floor. Evidence: May 21 FPS trace retrieved `tetris-matrix-
        rotation` for a Doom goal because its single token-overlap was
        enough to top the noise; pac-man trace got `corner-sliding` at
        0.05 instead of the much higher score it deserves.

        Caller is responsible for any further dedup / budget capping
        (see `dedup_hits` and `cap_hits_by_budget`).
        """
        bullets = self.load_all()
        if not bullets:
            return []

        # Modality tokens go through the same stopword stripper as the
        # goal so "3d" / "tic-tac-toe" tokenize consistently.
        mod_toks: list[str] = []
        if modality_tokens:
            mod_toks = _tokenize(" ".join(modality_tokens))

        query_toks = _tokenize(goal) + _tokenize(code) + mod_toks
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
            # Code-stage floor (2026-06-10 dojo-fight trace): fix turns
            # injected `fps-camera-and-movement-vectors` into a 2D-fighter
            # session at 0.0319 — pure token noise, 1.8KB of misleading
            # content per fix prompt, and the bullet earned harmful+1 yet
            # kept re-injecting within the session. Code-stage is the
            # narrow "validated patterns only" stage, so it gets a higher
            # floor than plan-stage (which stays broad by design).
            # 0.035 calibrated against BOTH trace sets: the fight-trace
            # noise hit scored 0.0319, while genuine code-stage hits
            # against the real playbook (doom goal + arrows feedback,
            # pinned by test_doom_trace_fixes fix_b) score 0.0373-0.0517.
            # Do not raise toward 0.05+ without re-measuring — long goals
            # dilute Jaccard, so genuine hits sit lower than you'd guess.
            if stage == "code" and sim < 0.035:
                continue

            # Quality multiplier: bounded ±10% of relevance.
            quality = 1.0 + 0.10 * _tanh(b.score() / 5.0)
            hits.append(BulletHit(b, sim * quality))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # --- delta ops (hand-edit helpers) -------------------------------------

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
