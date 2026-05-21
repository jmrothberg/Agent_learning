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
_SKELETON_MIN_SIM = 0.3
CANVAS_SKELETON_V2_NAME = "canvas_basic_v2.html"
CANVAS_3D_SKELETON_NAME = "canvas_3d_basic.html"
CANVAS_3D_SKELETON_SIDECAR = '{"goal": "3D space vector WebGL three.js coordinate projection game first person perspective"}'

CANVAS_GRID_SKELETON_NAME = "canvas_grid_basic.html"
CANVAS_GRID_SKELETON_SIDECAR = '{"goal": "grid continuous tile corridor snap slide sokoban pacman maze"}'

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

    All paths are computed from `root` (default memory/). The directory
    is created lazily on first write so a fresh checkout works without setup.
    """

    def __init__(self, root: str | Path = "memory"):
        self.root = Path(root)
        self.skeletons_dir = self.root / "skeletons"
        self.goals_dir = self.root / "goals"
        self.mistakes_path = self.root / "mistakes.jsonl"

    # --- bootstrap ---------------------------------------------------------

    def ensure(self) -> None:
        """Create directory layout and seed default skeletons if missing.

        Cheap to call repeatedly; agent constructor calls it once.
        """
        try:
            self.skeletons_dir.mkdir(parents=True, exist_ok=True)
            self.goals_dir.mkdir(parents=True, exist_ok=True)

            # List of all default templates to bootstrap
            templates = [
                (DEFAULT_SKELETON_NAME, DEFAULT_SKELETON, None),
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
            ]

            for name, html, sidecar in templates:
                html_file = self.skeletons_dir / name
                if not html_file.exists():
                    html_file.write_text(html, encoding="utf-8")
                if sidecar is not None:
                    sidecar_file = html_file.with_suffix(".json")
                    if not sidecar_file.exists():
                        sidecar_file.write_text(sidecar, encoding="utf-8")
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
            # Low-similarity past-win skeletons hurt more than help (the
            # "KEEP its structure" instruction in prompts_v1 forces a
            # bad scaffold on a mismatched goal). If the winner is a
            # past-win match below the threshold, fall back to the
            # bundled empty template — the model builds fresh instead
            # of fighting wrong structure. Manually-added skeletons
            # (score=0.05, no sidecar) also fall through; the default
            # itself (score=0.0, name match) is exempt so it can still
            # win when it IS the best.
            below_threshold = (
                best.source_goal is not None
                and best.score < _SKELETON_MIN_SIM
            )
            if below_threshold:
                default_path = self.skeletons_dir / DEFAULT_SKELETON_NAME
                try:
                    default_html = default_path.read_text(encoding="utf-8")
                except Exception:
                    default_html = DEFAULT_SKELETON
                return SkeletonHit(
                    name=DEFAULT_SKELETON_NAME,
                    html=default_html,
                    score=0.0,
                    source_goal=None,
                )
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
    Bullet(
        id="projection-3d-wireframe",
        content=(
            "To render 3D wireframe perspective projection (e.g. Battlezone-style vector "
            "graphics) on a 2D canvas without heavy external libraries: "
            "(1) Translate/rotate coordinates relative to player position (px, pz) and player "
            "angle (theta): const dx = x - px, dz = z - pz; const rx = dx * Math.cos(-theta) - "
            "dz * Math.sin(-theta); const rz = dx * Math.sin(-theta) + dz * Math.cos(-theta). "
            "(2) Perspective Projection: Projected screen coordinates are: if (rz <= 0.1) return; "
            "const screenX = centerX + (rx / rz) * fov; const screenY = centerY - (y / rz) * fov "
            "(y-axis inverted on canvas). (3) Rendering: Sort objects by depth (painter's algorithm) "
            "from farthest to closest (descending rz) so overlapping elements render correctly; connect "
            "projected points with ctx.beginPath(), ctx.moveTo(x1, y1), ctx.lineTo(x2, y2), ctx.stroke()."
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
            "To render a SNES-style Mode 7 3D ground projection in 2D canvas (e.g. Mario Kart): "
            "For each screen scanline y below horizon, calculate physical distance: distance = fov / (y + 1). "
            "Compute texture scale factor: scaleX = distance / fov. For rotation angle theta, steps are: "
            "stepX = Math.sin(theta) * scaleX, stepY = -Math.cos(theta) * scaleX. Sample pixels along scanline using: "
            "worldX = player.x + Math.cos(theta)*distance + (screenX - centerX)*stepX, and matching worldY. Read "
            "color from track texture, clamp coordinate bounds, and write into ctx.getImageData screen buffer."
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
]


class Playbook:
    """Filesystem-backed playbook of structured rules.

    On first use the file is seeded with SEED_BULLETS; subsequent saves
    persist the merged set. Retrieval is keyword/Jaccard against the goal
    and (optionally) the in-progress code.
    """

    def __init__(self, root: str | Path = "memory"):
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
