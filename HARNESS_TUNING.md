# HARNESS_TUNING.md — improving the verification harness

You are fixing a coding agent that drives a **local ~27–35B model** (qwen3.6 via MLX/Ollama) to
build single-file HTML5 games verified in real Chromium. Read **`DEV.md`** next (commands, env,
binding rules). Use **`HARNESS_DEBUG.md`** when debugging a bad session trace.

---

## New agent — harness improvement (read this first)

You are **not** building games in `games/*.html`. You are improving the **loop that builds and
verifies** games. A fresh Cursor agent should follow this path before editing code.

### 1. Read order (≈30 min)

| Step | File | Why |
|------|------|-----|
| 1 | **`AGENTS.md`** | Source vs artifacts; mixin map; trace paths; **never commit `games/`** |
| 2 | **This file** (`HARNESS_TUNING.md`) | Standing rules + traps |
| 3 | **`HARNESS_DEBUG.md`** | Gates, `failure_class`, trace timeline |
| 4 | **`TEST.md`** | What pytest guards; which file to extend per failure class |
| 5 | **`eval/OPERATIONS.md`** | Commands for batch eval / overnight tune |
| 6 | **`tools.py`** → `LiveBrowser.load_and_test` | Highest-leverage verifier (browser + gates) |
| 7 | **`assets.py`** → `render_asset_paths_block` | Injected `sprite()` / loader JS copied into every sprite game |

Optional deep dive: one bad trace via `scripts/enrich_trace.py <id> --timeline`, then open the
matching `.html` and play it — **do not trust TEST OK alone**.

### 2. Harness vs memory — where to put a fix

| Question | Put the fix in… |
|----------|------------------|
| Bug in **injected JS** every sprite game copies (`sprite()`, `loadAssets`, probes)? | **`assets.py`** (or `tools.py` if browser-side only) |
| Bug in **browser test / gates / ok scoring**? | **`tools.py`** + tests in `tests/test_*gate*.py`, `test_drawn_asset_detector.py`, `test_fix_round.py` |
| Bug in **agent loop** (compaction, feedback routing, phase order)? | Matching **`agent_*.py`** mixin — see **`AGENTS.md` §1b** |
| **Genre / game-type** convention (versus fighters, TD waves, chess CPU)? | **`memory/playbook.jsonl`** or `visual_playtests.jsonl` — retrieval-gated, not `if "mortal" in goal` |
| Model keeps **mis-wiring one game** but harness is correct? | Playbook + optional user feedback; **not** a one-game hardcode in harness |

**Recent example (parallel roster sprites):** `f2_walk` resolving to `f1_walk` was a **harness**
bug in injected `sprite()` token tie-breaking + load-race cache → fixed in **`assets.py`**. Clearing
`_spriteCache` on `reset()` and using `prefix + '_' + phase` in `drawFighter` is **memory**
(`versus-fighter-sprite-prefix` playbook bullet).

### 3. Canonical fix loop (every harness change)

```text
trace or failing pytest
  → classify failure_class (harness_bug | memory_gap | local_llm_limit)
  → smallest edit in the right layer (table above)
  → targeted pytest (TEST.md) then full suite: .venv/bin/python -m pytest tests/ -q
  → optional: re-run one eval goal from memory/prompt_library.jsonl
  → durable trap → add row below or bullet in playbook; ephemeral → triage.md only
```

**Suite must stay green.** Failing tests are regressions — update tests when behavior intentionally
changes (see `tests/test_fix_round.py` source-grep guards, `tests/test_assets.py` sprite resolver mirror).

### 4. Do not

- Patch **`games/*.html`** as source (artifacts only).
- Add **game-title or genre `if` branches** in Python (`tools.py`, `agent.py`).
- Weaken fuzzy **`sprite()`** matching for one game without a general tie-break / test (`tests/test_assets.py`).
- Gate **`ok=False`** on cosmetic sprite warnings (dead-frame pose delta, etc.).
- Create new top-level markdown files — extend **`HARNESS_TUNING.md`**, **`HARNESS_DEBUG.md`**, **`TEST.md`**, **`DEV.md`**, **`README.md`**.

### 5. High-leverage files (symptom → first open)

| Symptom | First file(s) |
|---------|----------------|
| Art on disk, colored boxes / wrong fighter sprite | `assets.py` (injected resolver), `tools.py` (`ASSETS_LOADED_BUT_UNDRAWN`) |
| Keys wired but no pixel change / input_responsive | `tools.py` `_input_smoke_test`, game `keys` object — often **memory** + probe |
| Patch SEARCH not found / feedback ignored | `agent_compaction.py`, `agent_feedback.py`, trace `structured_compaction` |
| Wrong coaching / asset regen when user wanted wire-only | `agent_feedback.py` routing |
| Probe false pass (state ok, pixels wrong) | `tools.py` gates, `tests/test_drawn_asset_detector.py` |
| Plan missing probes / wrong skeleton | `memory.py`, `prompts_v1.py`, `memory/skeletons/` |

---

## The 5 rules

1. **Tune the agent, not the model.** Fix prompts / retrieval / harness / scoring / memory — never
   “try a bigger model.”
2. **General fix → code. Game craft → memory.** Mechanisms that help many game shapes live in
   `tools.py` / `agent.py` / `assets.py`. Genre-specific guidance goes in `memory/*.jsonl`
   (retrieval-gated). No `if "pacman" in goal` in code.
3. **Genre-free in code.** Detect by observable *shape* (state paths, canvas, recipe gates), not
   subject matter.
4. **User feedback is authoritative.** `/rawfeedback` defaults ON. Do not override the user’s words
   with regex routing.
5. **Length ≠ failure once code is streaming.** Latch abort guards on code emission (`<html_file>`,
   `<!DOCTYPE`, `function`, `const`), not token count or wall clock alone.

## Mental model: verification is the lever

Most “agent failures” are **verifier failures** — `ok=True` while the game is broken, so the fix loop
never runs. Before new loop machinery: *does a gate flip `ok=False` on this failure?* Probes often
check **state** and pass while **pixels/behavior** are wrong.

When you debug or claim success: **open the game, drive it, read PNGs** — never trust “TEST OK”
alone.

## Where things live

| Area | Files |
|------|--------|
| **Source vs artifacts map** | **`AGENTS.md`** — what to edit vs read for triage (trace paths, logs) |
| Verifier (highest leverage) | `tools.py` — `load_and_test`, `_input_smoke_test`, gates |
| Agent loop | `agent.py` (orchestrator) + mixins — see **`AGENTS.md` §1b** (`agent_feedback`, `agent_prompts`, `agent_stream`, `agent_gates`, `agent_critic`, `agent_assets`, …) |
| Assets / audio | `assets.py`, `sounds.py`, `videos.py` |
| Prompts | `prompts_v1.py` — `FormatSpec` list; don’t hand-edit rendered blob |
| Memory (one JSONL line, no restart) | `memory/playbook.jsonl`, `visual_playtests.jsonl`, `implementation_outlines.jsonl`, `playtests.jsonl`, `skeletons/` |

Playbook retrieval uses weighted Jaccard on goal tags (tags weigh 2×). Below the ~0.02 floor a
bullet never reaches the prompt — broaden tags if a good bullet doesn’t fire.

## Traps — don’t repeat these

**Animation / sprites**

- **Consistency is the hard constraint.** Same character across frames; img2img cannot change pose
  (`guidance_scale=0` locks idle). Fresh txt2img replacement breaks consistency with art already in
  the game. **Never regenerate a pose frame to “fix” a dead one.** Near-identical frames are
  **cosmetic** — advisory `warning` only; must not flip `ok=False` or defer user gameplay feedback.
- Plan-time poses: txt2img with one shared character description + fixed seed. In-session: cycle
  frames; convey action with the **sprite**, not code-drawn limbs (`ACTION_DRAWN_NOT_SPRITED`,
  `CODE_DRAWN_OVER_SPRITE`).
- Wrong facing → flip in code (`ctx.scale(-1,1)`), don’t regenerate art.
- Sprite-key drift (`left_idle` vs `left_fighter_idle`) → silent colored boxes. Use the injected
  `sprite()` resolver; gate `ASSETS_LOADED_BUT_UNDRAWN` catches misses.
- **Parallel roster cross-wiring** (`f1_walk` / `f2_walk`, `blue_*` / `red_*`): harness `sprite()`
  must not tie-break on action token alone (`walk`) — entity prefix must win. LLM must still clear
  `_spriteCache` on rematch (playbook `versus-fighter-sprite-prefix`).

**Compaction / context**

- Pressure = `prompt_tokens / num_ctx`. A too-small `num_ctx` denominator (e.g. treating 32K as the
  window on a 100K+ session) triggers lossy compaction every turn — shredding playbook, user
  feedback, and file view (“patches don’t stick”). Default `num_ctx` is **100000**; compact only
  near a genuinely full window (~70% pressure), not on message count alone.
- **Do NOT add a `warm_prefix` after compaction** (Phase 4B investigation). MLX (`backend.py`
  `stream_generate`) is called fresh each turn with **no `prompt_cache`** → zero cross-call KV reuse,
  so a warm just re-prefills on the next real call (dead overhead). On Ollama, compaction rewrites
  the prefix (state-anchor replaces msgs 1..cutoff) so the cached KV is invalid at the divergence
  point, and there is no idle window right after compaction to hide prefill in. The existing
  `warm_prefix` is correctly gated to the **cross-slot** case only (coder slot ≠ architect slot, the
  multi-GPU Ollama box, where asset/sound gen on another GPU IS the idle window) — keep it there.

**Sampling**

- MLX must pass `top_p` / `top_k` (vendor coding preset). Untruncated sampling causes degenerate
  line-repeat loops on large first builds. Repetition penalty stays off for code.

**Visual critic**

- Prefill assistant with `"Q1: "` so the VLM emits parseable yes/no lines.
- Abstain only when the model can’t see the *image* — real findings (“no projectile visible”) are
  valid.
- Per-question `fix_hints` only for failed checks — blanket hints caused sprite oscillation.

**Other**

- Movement: tile `moveProgress` is 0→1 fraction — don’t divide by tile size twice.
- Weak models can’t author big mazes inline — give a seeded generator or skeleton to extend.
- Never call cloud models without explicit user opt-in.
- `<videos>` for a build must be generated in **that same build** — untestable otherwise.
- **3D FPS navigation X-axis mismatch** (Doom trace `build-a-doom-game-first-person_20260630_164114`): `fx=+Math.sin(yaw)` with `camera.rotation.y=yaw` walks opposite the gun on world X; Up/Down feels fine because Z matches. Do **not** fix by flipping minimap `lineTo` alone or tweaking only strafe `rz`. Fix: `applyQuaternion(camera.quaternion)` (preferred) or `fx=-Math.sin(yaw)` everywhere (movement, fire, minimap). Playbook: `3d-navigation-modality-invariants`, `fps-camera-and-movement-vectors`. Wireframe Battlezone uses `+cos(z)` — never paste three.js `-cos` into wireframe code.

## Debug workflow

See **`HARNESS_DEBUG.md`**. Batch score snapshots → **`eval/OPERATIONS.md`**.

## failure_class → where to edit

| `failure_class` | First place to edit |
|-----------------|---------------------|
| `harness_bug` | `tools.py` gates · `agent_*.py` loop |
| `memory_gap` | `memory/playbook.jsonl`, skeletons, outlines |
| `local_llm_limit` | `prompts_v1.py`, `agent_compaction.py`, `backend.py` |

## Serial tune learnings (durable)

Per-run scores live in **`eval/OPERATIONS.md`** (run_06 snapshot). Mid-batch harness fixes already in repo:

| Symptom | Fix | File(s) |
|---------|-----|---------|
| Iter-1 first build rambled 37–50k tok before `<html_file>` | Force prefill opening on local MLX/Ollama iter-1 (`<!DOCTYPE html>` constrained decode) | `agent.py` |
| Holochess 33 KB `<html_file>` rejected — micro_probe false positives on `} else {` branches | Benign script-repeat filter; promote repetition only when multiple scripts signal | `tools.py` |
| Art-intent builds: assets loaded but procedural boxes drawn | drawImage contract last on first build; playbook bullet; skeletons teach drawImage; undrawn advisory | `prompts_v1.py`, `agent.py`, `memory/playbook.jsonl`, skeletons, `tools.py` |
| Probes false-fail on `window.state` | Require `window.state = state` in HARD_RULES + first build | `prompts_v1.py` |
| ENTITY-NOT-RENDERED gates ok when all probes pass (thin crosshair) | Bbox sample + advisory when probes green | `tools.py` |
| Board games: pointerdown vs mousedown, frozen idle board | Board probe pointerdown+pointerup; turn-based frozen-canvas exemption | `tools.py` |
| Stuck best-of-2 silently doubled fix time on single MLX | Default **off**; opt in `/bestof on` or `coder.py --stuck-bon`; candidates under `candidates/iter_NN/` | `agent.py`, `chat.py`, `coder.py` |
| Kung-Fu: movement gated on idle/walk only | Playbook: include crouch/duck in movement branch or reset action before move | `memory/playbook.jsonl` |
| Video intro: orphan setTimeout, enemies never spawn | Playbook: call `reset()`/`startGame()` — same path as R restart | `memory/playbook.jsonl` |
| Parallel roster: `f2_walk` shows `f1` art (MK, Street Fighter, chess colors) | Harness: `sprite()` token tie-break + flush `_spriteCache` on `_assetsReady`; memory: `versus-fighter-sprite-prefix` | `assets.py`, `tests/test_assets.py`, `memory/playbook.jsonl` |
| Game looks perfect to human but trace shows 2 `soft_warnings` | Often probe timing or partial patch — not always a visual bug; read `iter_summary.soft_warnings` | trace + `HARNESS_DEBUG.md` § “looks fine” |

### Recurring patterns (watch in new traces)

- **memory_gap: assets loaded but undrawn** — self-recovers but burns iters (Joust 5×). Often state-gated sprites (enemies not spawned yet) or drawImage not wired — playbook `draw-generated-sprites-not-boxes` + `td-enemies-follow-waypoints` / spawn timers; not new agent machinery.
- **input_moves_player false-fail** — harness dispatches ArrowRight; if prior keydown left player in crouch/block, movement gated on `idle||walk` fails probe while game feels fine to a human.
- **ENTITY-NOT-RENDERED soft_warning** — fix only if 2+ games show same pattern (Missile Command iter 2).
- **OOM at game 12** — verify media/MLX freed at serial game boundaries; run heaviest media game earlier or isolated.
- **Stuck BoN** — only helps sampling noise on multi-slot parallel backends; on single MLX it is ~2× wall time. Keep default off for tune batches.

## Feedback channels (user vs harness)

| Source | Queue / path | Prompt wrapper |
|--------|----------------|----------------|
| User typed (TUI) | `_pending_feedback` | `USER FEEDBACK (HIGHEST PRIORITY)` |
| Harness auto | `_queue_internal_feedback` | `HARNESS NOTICE` in raw mode; same queue |
| Test failures | `_build_fix_prompt` + report | Fix-turn prompt (not feedback queue) |
| Stall / repeat errors | `_pending_coaching` | `AGENT COACHING` block |
| `/critique` playtest | `_queue_internal_feedback` | After **clean** iter only |
| `/vlm-critique` | `_pending_coaching` + `[CRITIC]` | Needs VLM model or local vision judge + toggle ON |

**Triage traps (TD seed trace `20260630_114658`):**

- `assets_parse_failed` at **phase_a** on a seed run ≠ “no art this session.” Build-turn **mid_session** `<assets>` can still generate sprites (`tower_tesla_idle`, `tower_flame_idle` in that trace).
- `ASSETS_LOADED_BUT_UNDRAWN` = sprites on disk but not drawn — **wiring** issue, not Z-Image failure. Playbook `draw-generated-sprites-not-boxes` is the right lever.
- **Safari vs Chromium trap:** sprites visible in Safari but `ASSETS_LOADED_BUT_UNDRAWN` in Playwright often means the harness sampled `__drawImageEvents` before async `loadAssets()` finished (placeholder `fillRect` frames first). Check trace `asset_decode_settle.ready`; re-run `scripts/_smoke_asset_decode_settle.py` before chasing model wiring.
- User feedback naming sprites **already in `_session_assets`** should get wire/draw coaching, not `ASSET GENERATION REQUIRED`.
- **Per-entity unique art** ("each tower its own unique head sprite") must stay armed through vague retry nudges — do not fuzzy-match an existing `*_head_*` asset and clear `_unhonored_asset_request`.

## Read order

**New harness agent:** start with **§ “New agent — harness improvement”** at the top of this file,
then `DEV.md` → **`eval/OPERATIONS.md`** (run batch / pytest) → `TEST.md` (suite map) →
`tools.py` (`load_and_test`) → trace `.jsonl` → relevant `memory/*.jsonl`. Agent comparison vs
Cursor/Aider: **`README.md#how-this-compares-to-other-coding-agents`**.

Quick test: `.venv/bin/python -m pytest tests/ -q` (pure-function; no GPU/model). **Full suite must
pass before push** (~2258 tests, ~1 min).
