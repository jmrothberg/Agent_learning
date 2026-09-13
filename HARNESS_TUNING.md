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
| 1 | **`AGENTS.md`** | Source vs artifacts; mixin map; trace paths; prefer not committing generated `games/` trees |
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

#### Standing rule — game titles vs game *classes* (do not violate)

| Layer | What belongs there |
|-------|--------------------|
| **Saved test prompts only** — `memory/prompt_library.jsonl`, `eval/*_goals.txt` | **Game-specific** wording is allowed and expected (e.g. Centipede head/body/tail, named controls, one title’s quirks, per-title `On-screen sizes:` px). |
| **Memory / retrieval / pins** — `playbook.jsonl`, outlines, nudges, skeletons, `visual_playtests`, `ensure_ids` / first-build pins | Teach **classes of games** (fixed-shooter, pinball, segmented-follower, versus). Pin by recipe id or class phrases (`fixed shooter`, `body segments`, `flipper`) — **never** by a single title (`if "centipede" in goal`, `ensure_ids` for `"galaga"` only). Class size/readability OK (`jmr-png-sheets`, `draw-fighters-large`) — no title→px tables. |
| **Harness Python** — `tools.py`, `agent_*.py`, `assets.py`, `prompts_v1.py` detectors | Mechanism / structural only — **no** game-title branches or franchise art hint tables. Generator `scripts/gen_prompt_640_library.py` must not hardcode titles (preserves jsonl bodies). |

**Mistake to never repeat:** stuffing Centipede/Galaga/… title logic into memory pins or harness “to fix one HTML smell.” Put the named-game sentence in the **library/eval prompt**; put the reusable mechanism in a **class** playbook/outline/skeleton bullet that any matching goal retrieves.

#### Standing rule — two product tracks (do not collapse)

Local models are getting good enough to write **full** HTML5 games (`sprite()`, audio, video). `/640` and `/640png` are a **different product**: FPGA-legal 640×480 with packed `STEM-N.png` / `jmr:spr:N`. Keep both sharp.

| Track | What “good” means | Do not |
|-------|-------------------|--------|
| Full HTML (`/media on`) | Rich first-build: `sprite(key)`, sounds, video; keep a **crisp tagged** `<plan>` in context when the model stayed inside tags | Weaken this path to unstick FPGA (don’t strip sprite() contracts, don’t cap a well-formed plan) |
| `/640` / `/640png` | JMR walls, packed sheets, `tx`/`ty` probes, no three.js, no leftover pose-filename coaching | Leak `sprite()` / 30k plan essays / Pac-Man chase autos into a dig/tunnel first-build |

Other coding agents keep **structured state** and drop narrative before the write turn. Do that here too: elide untagged Phase-A essays; never dumb down the Dig Dug *prompt* because a false-positive gate fired.

**Prompt / memory style (local LLMs):** library goals and playbook bullets must state **one** best
practice, not a menu (`raycaster or three.js` → prefer three.js). Prefer extending an existing
bullet over adding a new ID. Keep bullets short (~250–500 chars); put mechanics in playbook /
outlines, not long goal appendices.

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

### 3b. Overnight parallel improvement (same every night)

**Full HARD RULES (canonical):** [`eval/OPERATIONS.md` § HARD RULES](eval/OPERATIONS.md). Short form: Terminal batch (`Overnight.command` / `overnight.sh`) + Cursor Shell watcher (`block_until_ms=0`); never `nohup` the watcher; never halt the batch; never ask the human to paste.

**When a game finishes (or `agent_monitor.json` moves):**

1. Timeline: `.venv/bin/python scripts/enrich_trace.py games/tune_serial10/run_XX/traces/<label>__run_*.jsonl --timeline`
2. Optionally open the matching snapshot / shipped HTML under that run dir — **read-only evidence**
3. Classify `failure_class` on failed `iter_summary` (`HARNESS_DEBUG.md`)
4. Edit the **right layer** (§2 table) — surgical; no title hardcodes; no `games/*.html` as source
5. `.venv/bin/python -m pytest tests/ -q` (or targeted file first) — must stay green
6. Record durable trap in this file’s table; class craft → `memory/playbook.jsonl` / outline / skeleton
7. **Do not** pause the batch, ask the user to restart, or wait until morning to start fixing

**Burned on run_18:** (1) batch in Cursor → `chrome-mac-x64` Playwright fail ×11; (2) asked human to paste; (3) watcher via `nohup` so Cursor showed nothing. Fix = Terminal launcher + visible Cursor Shell watcher.

### 4. Do not

- Patch **`games/*.html`** as source (artifacts only).
- Add **game-title or genre `if` branches** in Python (`tools.py`, `agent.py`).
- Pin playbook / `ensure_ids` / skeletons by a **single game title** — use **classes** (see §2 standing rule). Game-specific lines go only in `prompt_library` / eval goals.
- Weaken fuzzy **`sprite()`** matching for one game without a general tie-break / test (`tests/test_assets.py`).
- Gate **`ok=False`** on cosmetic sprite warnings (dead-frame pose delta, etc.).
- Create new top-level markdown files — extend **`HARNESS_TUNING.md`**, **`HARNESS_DEBUG.md`**, **`TEST.md`**, **`DEV.md`**, **`README.md`**.
- **Start overnight batch inside Cursor’s integrated terminal** — use `eval/overnight.sh` (opens Terminal.app).
- **Invent a new `tune_runXX.sh` / goals file** for a normal night — use `overnight.sh --prompts … --model … --vlm …`.
- **Ask the human to paste** the overnight batch command — you launch Terminal yourself.
- **Hide the watcher** with `nohup`/`disown` — it must appear in the Cursor terminals panel.
- **Halt the overnight batch** between games to land a harness fix unless the user explicitly requested `TUNE_WAIT_FOR_MONITOR`.

### 5. High-leverage files (symptom → first open)

| Symptom | First file(s) |
|---------|----------------|
| Art on disk, colored boxes / wrong fighter sprite | `assets.py` (injected resolver), `tools.py` (`ASSETS_LOADED_BUT_UNDRAWN`) |
| Keys wired but no pixel change / input_responsive | `tools.py` `_input_smoke_test`, game `keys` object — often **memory** + probe |
| Patch SEARCH not found / feedback ignored | `agent_compaction.py`, `agent_feedback.py`, trace `structured_compaction` |
| Feedback fix → nested SEARCH / “no usable code” | `patches.py` (`_salvage_contaminated_search`), `agent_stream.py` recovery |
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
- **Media regen “already in HTML”** (invaders `20260723_162903`): `ASSETS['name']` /
  `drawSprite('name')` without a `*_assets/*.png` or `PATHS` URL is **not** loadable. Mid-session
  must emit the full loader block — never “No JS patch is required”. Use
  `_scan_html_for_loadable_asset_refs` for that split; keep the broader scan for alignment coaching.
- **Parallel roster cross-wiring** (`f1_walk` / `f2_walk`, `blue_*` / `red_*`): harness `sprite()`
  must not tie-break on action token alone (`walk`) — entity prefix must win. LLM must still clear
  `_spriteCache` on rematch (playbook `versus-fighter-sprite-prefix`).

**Backend / oMLX**

- **GLM-5.3 already loaded ≠ TUI session** (BATTLEZ2 `20260904_094928`): typing a goal without
  `/load` still went through in-process `mlx_lm` → `ValueError: Model type glm5_next not supported`,
  0 tokens. `detect_backend` now uses oMLX **currently loaded** models (`loaded=true` on
  `/v1/models/status`); unloaded/discovered rows do not get a `*` and are not auto-picked.
  `glm5_next` never uses in-process mlx_lm.
- **GLM thinking looks dead in TUI** (BATTLEZO `20260904_095910`): hidden CoT
  (`reasoning_content`) already reset the stall clock but Activity stayed at
  0 tokens / "waiting for first token". TUI now counts thinking chunks live
  and does **not** print the chain-of-thought.
- **oMLX 1800s overall must not kill a live think** (BATTLEZO `20260904_095910`): TUI already
  says active streams are not wall-clock capped; ollama + in-process MLX honor that. `MLXServerBackend`
  still aborted GLM CoT at 1800s (GPU 98%). `overall_seconds` now fires only when **both** content
  and thinking token counts are still 0. Runaway is repetition/max_tokens, not wall-clock. The
  MacBook first-token watchdog (`_model_reported_loaded`) sits alongside it: it aborts only when
  oMLX says the model is loaded yet nothing at all has streamed.
- **Qwen3.8 / in-process MLX `fds_to_keep` after GLM/oMLX** (BATTLEZ2/4/5/6/7
  `20260904`): switching to Qwen3.8-27B after GLM left Chromium running, then
  `mlx_vlm.load` forked and died at 0 tokens
  (`ValueError: bad value(s) in fds_to_keep`). Warm-load before a *new*
  browser is not enough — `/new` reuses the live Playwright. Close Chromium
  before in-process `warm_load`, then start it after; also filter closed FDs
  out of `multiprocessing.util.spawnv_passfds`. GLM/oMLX HTTP is unchanged.

**Compaction / context**

- Pressure = `prompt_tokens / num_ctx`. A too-small `num_ctx` denominator (e.g. treating 32K as the
  window on a 100K+ session) triggers lossy compaction every turn — shredding playbook, user
  feedback, and file view (“patches don’t stick”). Default `num_ctx` is **100000**; compact only
  near a genuinely full window (~70% pressure), not on message count alone.
- **KV prefix cache (Sept 2026) — keep history append-only.** Local backends reuse the prefill of the
  previous turn only for the **byte-identical prefix**: oMLX tiered cache (`hot_cache_max_size` > 0),
  in-process `mlx_lm` via the harness's cross-turn `prompt_cache` (`MLX_PROMPT_CACHE`), Ollama slot
  cache. Anything that rewrites an earlier message (per-turn HTML elision, report collapse, plan
  elision, stage-effort change in the system prompt) invalidates everything after it. So: on
  `mlx`/`mlx-server` the default elision is **deferred** until projected prompt ≥ 80% of the compaction
  ceiling and then runs as a batch (`AGENT_PREFIX_CACHE_FRIENDLY`); the Qwen `medium→low` effort switch
  costs exactly one miss at the first fix turn; structured compaction is still a full miss (rare, under
  pressure). Triage: `stream_done.ttft_s` vs `prompt_tokens` — ~10 s on a 30k prompt = hit, 60-120 s =
  miss; in-process also reports `cached_prompt_tokens`. **Never** add per-turn edits to messages other
  than the newest user turn.
- **Do NOT add a `warm_prefix` after compaction** (Phase 4B investigation). In-process MLX now keeps a
  cross-turn `prompt_cache`, but compaction rewrites the prefix, so a warm still just re-prefills
  what the next real call would prefill anyway (dead overhead). On Ollama, compaction rewrites
  the prefix (state-anchor replaces msgs 1..cutoff) so the cached KV is invalid at the divergence
  point, and there is no idle window right after compaction to hide prefill in. The existing
  `warm_prefix` is correctly gated to the **cross-slot** case only (coder slot ≠ architect slot, the
  multi-GPU Ollama box, where asset/sound gen on another GPU IS the idle window) — keep it there.

**/640 simulator (JMR native)**

- **Fieldrunners `20260829_083734`:** `/640` + retrieved media-era won skeleton (`sim=1.00`) + lean
  budget keeping ~4KB opening-book while **dropping** `bfs-grid-pathfinding` → plan said “emit
  assets”, first builds hit `inline_data_bloat` / `unclosed_html_file`, `no_usable_code`. Fixes:
  simulator skeleton fallback when HTML refs sidecars / `data:image` / is >28KB; lean budget
  priority **components > playbook > opening**; Phase A uses `PLAN_INSTRUCTION_SIMULATOR` (no
  EXPECTED `<assets>`); LOOK + HARD_RULES_SIMULATOR ask for classic pixel maps on
  turn 1 (no fail-and-retry art gate — teach in the prompt, do not force a second pass).
  Pin playbook `classic-arcade-pixel-maps` on `/640` first build.
  **`/640png`:** same JMR V1 640×480 / JS walls, but the art pipeline writes
  `STEM-0.png` … `STEM-15.png` next to the HTML (`jmr:spr:N` + `window.JMR_SPR`).
  Pin `jmr-png-sheets` instead of pixel-maps. `/games N` derives the goal from
  `prompt_640` (STEM-N.png / `jmr:spr:N` rewrite) — never the media `prompt`.
  **Vector-stroke class** (2D wireframe, not sprites): keep JMR walls but do
  not rewrite to Emit `<assets>` / `jmr:spr`. The sprite TARGET's
  "1px drawImage columns" strong-hooked `canvas-puzzle-grid` over
  `canvas-vector-wireframe` (BATTLE10).

  **Atlas packing — the clear rules** (`assets.py`: `jmr_atlas_group_key`,
  `jmr_atlas_layout`, `materialize_jmr_png_sheets`; full rationale in that
  file's module docstring):
  1. **16 is a FILE cap, not a per-sheet frame cap.** A strip with 8 frames
     is still one file. Adding frames only makes that one PNG wider.
  2. **64 poses generated per session** (`JMR_PNG_MAX_FRAMES`, raised from 48
     on 2026-09-03 — a 6-type/2-color chess roster at idle+walk+lift+slam
     needs exactly 48, zero slack), then folded onto ≤16 sheets — that's
     the real `<assets>` cap under `/640png`.
  3. **Grouping = subject prefix (before the first `_`), NEVER a pose-word
     list.** 2026-09-03: the old `idle|walk|attack|…` vocabulary split every
     unlisted word onto its own sheet (ANIMATIO: `big_grab`/`small_impact`/
     `small_over` → 2 characters became 8 PNGs). Any game invents any pose
     word, so only the prefix decides: `hero_idle` + `hero_walk1` → one strip
     (stem `hero`); `big_*` and `small_*` → exactly two strips. `hero` and
     `creep` → separate sheets (no shared prefix). Two chess pieces (`pawn`,
     `king`) never share a sheet just because both are "chess art" — grouping
     is on the *name*, not the genre. Name assets `<subject>_<pose>`.
     Each prefix batch is then **split by source pixel size** — cells pad to
     the largest frame, so a 512px asset never rides a 64px sprite strip
     (bloats every cell and draws small art at the padded size).
  4. **Cost driver is cell size (px), not frame count.** Strip width =
     `cellW × frameCount`. In `/640png`, `<assets>` `"size"` **is** on-screen
     px (`blitSpr` draws 1:1 on 640×480) — pick how many fit across the
     playfield. Per-title numbers live in each library `prompt_640`
     (`On-screen sizes: …`); class rule in playbook `jmr-png-sheets`. A big
     cell costs on every frame (8×512 is a 4096px-wide strip).
     **Scale (Sept 2026, FROGGERC/DIGDUGD3 "4× too small"):** the first
     library pass used arcade-*native* px (16 px frog on a 224 px screen →
     `~24x24`), which is ~1/27 of the 640 glass instead of ~1/14. Library is
     now ~2× / arcade-faithful (16 px arcade ≈ 40 px; grid games capped so the
     arena still fits: Pac-Man/Dig Dug/Zelda tiles 40, Tetris well unchanged,
     fighters/Doom/laserdisc unchanged). Harness floor: `apply_jmr_size_floor`
     scales any **animated** subject (prefix with ≥2 poses) whose short side is
     < `JMR_PNG_MIN_ANIMATED_PX` (32) up to 32, aspect kept; singles (bullets,
     dots, balls, HUD icons) untouched; trace `jmr_size_floor_applied`.
  5. **Draw contract:** 9-arg `drawImage` / injected `blitSpr` helper crops
     `sx = frameIndex * cellW`. `render_jmr_png_paths_block` emits the exact
     frame-index table so the model never invents `sx`. **cellW/cellH must
     match that table**, not `TILE`, when TILE is a different size (ZELDATOP
     blit 32 on 40px sheets cropped neighboring frames). **fillText is 8×8
     ASCII 32–126** (`♦` / `\u25C6` → chip paints `b`; `textAlign` center/right
     uses UTF-8 **byte** length so WAVE/LIVES overlap). HUD: `textAlign=left`
     at `x+n*(8*k)`; lives via `fillRect` or `*`. Playbook `jmr-filltext-ascii-hud`.
     **Chip walls are prompt/playbook only — never microprobe-fail Chromium.**
     Also: do not use `arr.splice`’s return (FPGA `undefined`); copy-down.
     Playbook `jmr-splice-return-undefined`.
  **Library:** every `memory/prompt_library.jsonl` entry has `prompt_640` (pixel-map /
  animated arcade goal + `On-screen sizes:`). `/640` then `/games N` loads that
  variant; `/640png` rewrites it for STEM-N sheets; media mode still uses
  `prompt`. Hand-edit game craft in the jsonl only; `scripts/gen_prompt_640_library.py`
  refreshes the shared TARGET footer and fills missing `prompt_640` — it does not
  hardcode titles.
- **Fieldrunners `20260829_124033`:** `/640` first build aborted with
  `inline_data_bloat` → `unclosed_html_file` / `no_usable_code` after ~2.4KB.
  Partial reply was looping on near-identical `const PAL` / `PAL2` / `PAL3`…
  color arrays (playbook + heavy `prompt_640` asked for every tower’s idle/fire/
  angle sheets). Not a harness false fail — local model hit the art-data wall.
  Softened TD `prompt_640` + playbook + `SIMULATOR_TARGET_BLOCK` /
  `HARD_RULES_SIMULATOR`: **one shared palette**, tiny ≤12×12 maps, no PAL2/PAL3
  clones; extra frames via later `<patch>`. Recovery coaching for
  `inline_data_bloat` now names the palette-clone trap.

**Sampling**


- MLX must pass `top_p` / `top_k` (vendor coding preset). Untruncated sampling causes degenerate
  line-repeat loops on large first builds. Repetition penalty stays off for code.
- **Thinking effort (`/thinking`)** — user levels `low|medium|high|max|off` (default **medium**; shortcuts `/low` `/medium` `/high` `/max`). Env: `REASONING_EFFORT` (preferred) or `QWEN_REASONING_EFFORT`. Display dump is `/showthinking` (unrelated). **Qwen3.8** `chat_template.jinja`: **`xhigh`, `medium`, `low` only — there is no `high`**. Passing `high` jinja-raises. Native template default is `xhigh`; **harness default is `medium`** (Aug 15 DK/SF: xhigh plan turns ran 15k–22k tokens / 40 min before tags). User `high`/`max` → native `xhigh`. DK `20260815_085321` also failed on xhigh because first-build `<html_file>` prefill sat *inside* the open `<think>`. Close think before code prefill; `_extract_html` salvages a complete `<html_file>` trapped before `</think>`. Incomplete plans: stub the huge assistant blob **before** `plan_incomplete_retry` so retry is not a 100k+ CoT prefill. `QWEN_ENABLE_THINKING=0` = `/thinking off`. **oMLX HTTP:** keep thinking ON (`medium`). Do not disable thinking to “fix” stalls — trace `20260829_165958` was HTML trapped in CoT plus a 16384 `max_tokens` cap.
- **oMLX assistant prefill = `partial: true`** (DIGDUGDI `20260910_162103`, 21 min → 0 files). oMLX renders the template server-side; a trailing assistant message **without** `partial` is a *finished* turn followed by a fresh `<|assistant|><think>` — the model thinks again and re-emits the opener (reply began `<html_file>\n<!DOCTYPE html>\n<html_file>\n<!DOCTYPE html>`). `omlx_messages_close_think_prefill` now sets `partial: True` (oMLX → `continue_final_message`) and sends **no `</think>` prefix**: both GLM-5.3 and Qwen3.8 templates emit a closed empty `<think></think>` before a trailing assistant message, and a prefix breaks GLM (HF “final message does not appear”). `_stream` also skips the local prepend when the reply already starts with the prefill (`prefill_echo_detected`). **Prefill anchor rules (GLM on oMLX, verified by curl):** GLM `strip()`s the final message; a prompt ending in **any `>`** (`<!DOCTYPE html>`, `<html>`, `<html lang="en">`) makes GLM emit EOS with 0 tokens; and oMLX drops the leading space of the first continued token (`<html` + ` lang=` → `<htmllang=` in DOOM3DF2 `20260911_170915`). The first-build prefill is `<html_file>\n<!DOCTYPE html>\n<html lang` — continues as `="en">`, no `>` boundary, no leading-space token.
- **First-build parse-error salvage** (DOOM3DF2 `20260911_170915`): a complete 26 KB three.js build was discarded for ONE hallucinated token (`{ map: per-pixel ceilTex }`) and the retry was a full 7800-token regeneration — 26 min at 5 tok/s for a one-line fix. `_materialize` now writes a first build whose only defect is a JS parse / bracket error in a complete document (`_first_build_parse_error_salvageable`), traces `first_build_syntax_salvaged`, and coaches ONE `<patch>`. Prose-before-DOCTYPE, elisions, placeholders, duplicated drafts and truncation are still rejected. The one-token slip is the model's; throwing away the other 7799 tokens was the harness's.
- **GLM-5.3 thinking** (`chat_template.jinja`): legal `low`/`high`; **omitting the field is `max`**. BATTLEZO `20260904_095910` omitted it → 30 min CoT, 0 visible tokens. Harness always sends `reasoning_effort`. User `medium` → GLM `high` (the middle rung). `/thinking max` restores native max. `/thinking off` still opens `<think>` at `low` (template cannot disable).
- **Negated modality words** (DIGDUGDI `20260910_162103`): the `/640png` suffix says “no WebGL, no three.js”; `detect_3d_intent` tokenized that to `three`/`threejs` and routed a 2D Dig Dug to `canvas_3d_basic.html` at sim=1.00, tagging every `/640png` retrieval with 3D tokens. `modality.py` now skips a 3D keyword directly after `no|not|without|never|non`.
- **HARD RULE (Mac) — oMLX must run with native Metal kernels.** `pip install git+…omlx` / `pip install -e .` builds **no** `_ext` (needs `OMLX_WITH_CUSTOM_KERNEL=1` + full Xcode); the affected families silently fall back to generic paths. This repo ran that way from Aug 4 to Sep 11 2026 (DIGDUGDI, DOOM3DF2) — never again:
  - **Install** into `~/MLX_Models/.omlx-venv` from the release's prebuilt wheel, never from source: `pip install --no-deps --force-reinstall omlx-<ver>-cp312-cp312-macosx_15_0_universal2.whl` (asset on github.com/jundot/omlx/releases; ships `_ext.cpython-312-darwin.so` + `.metallib` for bonsai / decode_fast / glm_moe_dsa / minimax_m3 / qwen35_prefill). Verify sha256 against the release digest. oMLX.app also ships them.
  - **Verify**: `curl -s :8000/api/status | jq .custom_kernels` — every entry `available: true`.
  - **Enforced by the harness**: `backend.ensure_omlx_server()` → `omlx_require_metal_kernels()` raises with the fix text if any kernel is missing; `OMLX_ALLOW_NO_KERNELS=1` downgrades to a stderr warning for deliberate experiments only.
  - **What the kernels did NOT fix (measured Sep 11 after install):** GLM-5.3-Flash-MLX-**6bit** (affine, 285 GB, 320B-A18B) still decodes at **5.1 tok/s** (300 tok, prompt 25) and prefills at ~300 tok/s (13.3K prompt, 44 s TTFT). That is the known speed of MLX-affine GLM-5.3 quants under oMLX on M3 Ultra (community: ~6.2 tok/s); oMLX's **oQ4e** conversion of the same model is reported at ~24 tok/s TG / 480 tok/s PP. A raw `gather_qmm` microbench of the 288-expert layers at 6-bit allows ~45 tok/s, so the gap is oMLX's glm5_next affine path, not the quant width. Options: oQ4e GLM-5.3 (~170 GB, 4-bit-class quality), or build with Qwen3.8-Flash-Next 8-bit-MTP (33–40 tok/s measured here).

**Visual critic**

- Prefill assistant with `"Q1: "` so the VLM emits parseable yes/no lines.
- Abstain only when the model can’t see the *image* — real findings (“no projectile visible”) are
  valid.
- Per-question `fix_hints` only for failed checks — blanket hints caused sprite oscillation.

**Other**

- **Duplicate-decl microprobe vs bare scripts** (Castlevania trace
  `20260720_175910`): materialize rejected a complete `<html_file>` as
  “concatenated drafts” because `depth <= 1` treated function-local
  `const mat` / `const p` as top-level. Canned/serial/skeleton games wrap
  JS in an IIFE, so locals sit at depth ≥ 2 and never hit this FP —
  winners reuse short names freely inside the IIFE. Fix: count decls only
  at `_script_outer_decl_depth` (IIFE body = 1, bare = 0). Detect IIFE as
  whole-script wrap at the start, not “contains IIFE somewhere.”
- Movement: tile `moveProgress` is 0→1 fraction — don’t divide by tile size twice.
- Weak models can’t author big mazes inline — give a seeded generator or skeleton to extend.
- Never call cloud models without explicit user opt-in.
- `<videos>` for a build must be generated in **that same build** — untestable otherwise.
- **3D FPS navigation X-axis mismatch** (Doom trace `build-a-doom-game-first-person_20260630_164114`): `fx=+Math.sin(yaw)` with `camera.rotation.y=yaw` walks opposite the gun on world X; Up/Down feels fine because Z matches. Do **not** fix by flipping minimap `lineTo` alone or tweaking only strafe `rz`. Fix: `applyQuaternion(camera.quaternion)` (preferred) or `fx=-Math.sin(yaw)` everywhere (movement, fire, minimap). Playbook: `3d-navigation-modality-invariants`, `fps-camera-and-movement-vectors`. Wireframe Battlezone uses `+cos(z)` — never paste three.js `-cos` into wireframe code.
- **run_18 empty 3D / dim vector / trench stall / WebGL undrawn FP:** (1) Minecraft sky-only still `ok=true` because WebGL `readPixels` blank was advisory when input moved state — gate on **Playwright screenshot** dominant hue (`EMPTY-3D-VIEW`), not `canvas_info.blank`. (2) Battlezone near-black strokes (`#0a330a`) → `DIM-VECTOR-SCENE`. (3) Star Wars obstacles never decrease `z` while `distance` advances → `OBSTACLE-DEPTH-STALL` + sticky `trench-depth-vector-spawn`. (4) Doom listed all wall textures `ASSETS_LOADED_BUT_UNDRAWN` via drawImage audit while PNG showed walls — **skip undrawn nag on WebGL/three.js**. (5) Frozen-canvas pins must **merge** with wireframe/FPS/voxel `ensure_ids`, not replace them. (6) Rampage wall-on-monster → `OPAQUE-SPRITE-SCENERY` + `character-sprite-isolation`. Tests: `tests/test_run18_quality_gates.py`.
- **OPAQUE-SPRITE-SCENERY keyart name collision** (Doom trace `build-a-doom-game-first-person_20260721_132716`, `failure_class=harness_bug`): gate correctly flags character PNGs with baked wall edges, but stems like `keyart_boss` match the character token `boss` and hard-fail `ok` while probes are green (opaque scenery on title/cutscene plates is intentional). **Fix (role skip, not Doom-specific):** in `opaque_scenery_soft_warning_for_png`, skip stems whose path tokens are `keyart` / `title` / `intro` / `cutscene` (same idea as existing `bg_` / `sky` skips). Keep the gate for real character stems (`monster_idle`, `boss_idle`, …). Do **not** strip `boss` from the character regex and do **not** demote the whole gate. Test: `test_opaque_scenery_skips_keyart_even_when_boss_in_name`. Note: a later iter freeze on that run was LLM (`muzzleDiv` undeclared) — separate from this harness FP.
- **Seed floor/ceil-only full rewrite** (`DOOM3DFI__run_20260902_121459`, `failure_class=harness_bug`): user said walls were perfect and only floor/ceiling were static; `_goal_is_small_scope_edit` missed preserve/static language → first build allowed `<html_file>` → sheet-index scramble → walls gone. Not a three.js/prompt issue. **Fix:** `_PRESERVE_WORKING_PATTERNS` + `_LOCALIZED_RENDER_FIX_PATTERNS` in `agent_feedback.py` arm seed patch-lock. Tests: `eval/seed_edit_scenarios.jsonl` + `test_seed_preserve_floor_ceil_goal_locks_patch_only`.
- **Nested diagnose inside patch-first SEARCH** (`DOOM3DFI__run_20260902_131715` iter 2): iter 1 patch applied (wrong projection math). User feedback + `patch_first_prefill` (`<patch>\\n<<<<<<< SEARCH\\n`) → model put `<diagnose>` then a second SEARCH/REPLACE *inside* SEARCH → `embedded SEARCH/REPLACE marker` → **no usable code** even though the inner edit was correct. **Fix (general):** `patches._salvage_contaminated_search` keeps the innermost SEARCH body; recovery coaching if salvage cannot apply. Tests: `test_salvage_nested_search_after_diagnose_inside_prefill`.
- **Raycaster floor/ceil “small tiles + black gaps”** (`DOOM3DFI__run_20260902_134411`): projection + `0xFF000000|texel` passed probes, but `floor_tile.png` / `ceil_tile.png` still ~30% transparent with inset art. `putImageData` keeps RGB 0 in those margins → seams; tiles look too small and not continuous to wall tops. **Memory:** playbook `raycaster-floor-ceil-opaque-tiles` — bake getTileBuf fully opaque edge-to-edge (fill a<128 from opaque average / nearest). Not a probe false-fail.
- **Auto-revert undid correct user fix** (same trace **iter 3**): model baked opaque tiles as asked; brittle `floor_not_solid` (`mx-mn>30`) failed → fewer probes (7/8 vs 8/8) → auto-revert restored gappy iter-2 file. **Harness:** skip probe-count-only auto-revert when this iter honored user feedback and there are no new page errors / coverage gaps (`GameAgent._auto_revert_should_fire`, trace `auto_revert_skipped`). Hard regress (page errors, coverage gaps) still reverts. Tests: `tests/test_auto_revert_user_feedback.py`.
- **run_19 probe dual-dispatch SyntaxError (all 8 games + Doom):** `_patch_probe_keyboard_dispatch` / `_patch_probe_pointer_board_clicks` prepended `window.__X=...;` *statements*, then `_run_probe` wrapped as `Boolean(await (EXPR))` → SyntaxError → instant quarantine of every KeyboardEvent/MouseEvent probe. Verification silently degraded since those helpers landed. **Fix:** wrap helper + body in one async IIFE (`_probe_expr_as_async_iife`). Tests must parse through `_wrap_probe_expr_for_eval` (same string as `_run_probe`). Soft-fail ≠ broken game; Qwen+VLM 4/8 ≈ run_14, not a regression vs GLM no-VLM run_18.
- **run_19 sibling `_assets` OPAQUE contamination:** serial overnight co-locates games; scanning `parent.glob("*_assets/*.png")` made Rampage fail on Prince’s `hero_pullup`. **Fix:** `opaque_scenery_soft_warning_for_html_assets` → only `{html_stem}_assets/`. Test: `test_opaque_scenery_scan_ignores_sibling_game_assets`.
- **run_19 `undrawn_present` advisory lie:** counting demoted `ASSETS_LOADED_BUT_UNDRAWN` in `warnings` forced `failure_class=memory_gap` while OPAQUE/STATIC-ACTION were the real soft blockers. **Fix:** `_undrawn_present` iterates **`soft_warnings` only**. Deferred: DK jump craft, Centipede STATIC-ACTION threshold, OPAQUE asset-regen — re-triage next batch once probes actually run.
- **DOOM3DF3 mouse-look ×4** (`DOOM3DF3__run_20260911_184919`, Qwen3.8-Flash-Next on oMLX): user asked four turns in a row for the mouse to turn the player; every turn re-patched the `mousemove` handler. Game bug was ONE line in `update()`: `p.yaw=camera.rotation.y; p.pitch=camera.rotation.x;` (stale write-back that discards the handler's write every frame) — and the playbook bullet `fps-camera-and-movement-vectors` literally taught it (“Sync yaw for probes: `state.player.yaw = camera.rotation.y`”). Harness gaps, all fixed generically: (1) smoke test never **moved** the mouse → `_input_smoke_test` now does a real CDP press-drag-release + bare move and always prints `Drag→[player.yaw]` / `Drag→NO state change` (`input_test.pointer_drag`); (2) model copied the focused-slice header `// --- function \`update\` (focused slice) ---` into SEARCH → partial patch → `patches._strip_focused_slice_markers` + marker text says NOT in file; (3) no writer audit → router JSON gains `state_fields`, `tools.static_state_writer_lines` (alias-aware, `const p=state.player`) lists every writer with its function, `LiveBrowser.state_field_stickiness` pokes the field and reports REVERTED → `STATE-FIELD EVIDENCE` block under the user note (`_build_state_field_evidence`); (4) raw feedback mode (default) never ran repeat detection → second identical complaint adds `REPEAT COMPLAINT — CHANGE APPROACH` naming prior applied SEARCH heads (`_fix_attempt_ledger`) and swaps `<patch>`-first prefill for `<diagnose>` that one turn. Memory: recipe probe `auto_fp_turn_changes_yaw` (dispatch drag + ArrowLeft/KeyQ, wait frames, yaw must change), playbook/outline now say state is the single source of truth, camera derives FROM state. Tests: `test_focused_slice_marker_in_search_is_stripped_before_matching`, `test_smoke_test_runs_a_pointer_drag_and_always_names_its_verdict`, `test_static_state_writer_lines_resolves_alias_and_names_frame_writer`, `test_raw_mode_repeat_complaint_escalates_and_attaches_evidence`.
- **DOOM3DF3 OPAQUE on i2v keyframe** (same trace, every iter): `OPAQUE-SPRITE-SCENERY [boss_key]` hard-failed `ok` with 8/8 probes green — `boss_key`/`intro_key` are LTX i2v seed plates, not in-world sprites. **Fix:** role-skip tokens `key|keyframe|plate|splash|poster` in `_CUTSCENE_OR_KEYART_NAME_RE`; `opaque_scenery_soft_warning_for_html_assets(..., skip_names=)` for `<videos image:>` stems (`_video_i2v_source_names`). Soft-warning-only + green probes no longer `harmful++` playbook (`_failure_blames_code`); reverted wrongful `character-sprite-isolation` harmful bump. Tests: `test_opaque_scenery_skips_boss_key_and_keyframe_role_tokens`, `test_soft_warning_only_with_green_probes_does_not_blame_code`.
- **DOOM3DF3 first-fix cache miss** (same trace, TTFT 41.7 s on 33k prompt, `cached_prompt_tokens: 0`): `<patch>` prefill closes think (`thinking_tokens: 0`) but `medium→low` stage effort still rewrote the system prompt. **Fix:** `chat_template_thinking_kwargs(..., thinking_closed_prefill=True)` skips the stage→low demotion; `_stream` sets `_thinking_closed_prefill` when an assistant prefill is used. Trace `stage_effort_held_for_cache`. Test: `test_thinking_closed_prefill_holds_effort_for_prefix_cache`.
- **DOOM3DF3 probe-lint CSSOM FP** (same trace, 3×): `weapon_overlay_visible reads visibility` — `_probes_referencing_unassigned_props` flagged `getComputedStyle(w).visibility`. **Fix:** skip receivers from `getComputedStyle` / `getBoundingClientRect` / `.style.`; ignore `visibility|display|opacity|transform`. Test: `test_probe_lint_skips_getComputedStyle_cssom_reads`.

## Debug workflow

See **`HARNESS_DEBUG.md`**. Batch score snapshots → **`eval/OPERATIONS.md`**.

## failure_class → where to edit

| `failure_class` | First place to edit |
|-----------------|---------------------|
| `harness_bug` | `tools.py` gates · `agent_*.py` loop |
| `memory_gap` | `memory/playbook.jsonl`, skeletons, outlines |
| `local_llm_limit` | `prompts_v1.py`, `agent_compaction.py`, `backend.py` |

## Measure before/after (scoreboard)

Before claiming a harness/memory change helped, run:

```bash
.venv/bin/python eval/compare_runs.py run_15 run_16
```

Compare `fresh_pass`, `avg wasted_iters`, `avg first_clean`, and `failure_class` histograms on real batch traces — not anecdotes. Snapshot durable scoreboards into `eval/OPERATIONS.md`.

---

## Serial tune learnings (durable)

Per-run scores live in **`eval/OPERATIONS.md`** (run_06 snapshot). Mid-batch harness fixes already in repo:

| Symptom | Fix | File(s) |
|---------|-----|---------|
| Iter-1 first build rambled 37–50k tok before `<html_file>` | Force prefill opening on local MLX/Ollama iter-1 (`<!DOCTYPE html>` constrained decode) | `agent.py` |
| Holochess 33 KB `<html_file>` rejected — micro_probe false positives on `} else {` branches | Benign script-repeat filter; promote repetition only when multiple scripts signal | `tools.py` |
| Art-intent builds: assets loaded but procedural boxes drawn | drawImage contract last on first build; playbook bullet; skeletons teach drawImage; undrawn advisory | `prompts_v1.py`, `agent.py`, `memory/playbook.jsonl`, skeletons, `tools.py` |
| Probes false-fail on `window.state` | Require `window.state = state` in HARD_RULES + first build | `prompts_v1.py` |
| ENTITY-NOT-RENDERED gates ok when all probes pass (thin crosshair) | Bbox sample + advisory when probes green | `tools.py` |
| ENTITY-NOT-RENDERED on fog-hidden stairs/exits (run_16 roguelike) | Skip sample when `seen`/`explored`[y][x]===false | `tools.py` |
| SyntaxError → cascade soft_warnings flood fix prompt (run_16 1942) | On page SyntaxError, suppress ISSUES/probe dump noise | `tools.py` |
| Partial quarantine blocks green probes under max-iters 3 (Holochess) | `_PARTIAL_QUARANTINE_GATE_CAP` 2→1 | `agent_probes.py` |
| Pinball `auto_body_enters_playfield` after mutating probes (run_16) | Reseat via reset/`R` before Space launch check | `memory/visual_playtests.jsonl` |
| Bullet-hell `bullets_spawn` length-grows at steady-state (run_16) | Outline + Phase-A lint `fragile_length_growth_probe` | `implementation_outlines.jsonl`, `agent.py` |
| QTE auto_probe `const d=…` flagged undefined helper (run_14) | Skip locally declared idents; inline Math.hypot in recipe | `tools.py`, `visual_playtests.jsonl` |
| Typing probe `document.dispatchEvent` vs `window` listener (run_14) | Dual-dispatch KeyboardEvent to window+document | `tools.py` `_patch_probe_keyboard_dispatch` |
| ASSETS_UNDRAWN on intro/title while probes green (run_15 OutRun) | Demote when `state.mode` is intro/title/menu | `tools.py` |
| DK jumpOverSet declared never scored (run_15) | Vertical-platformer trap + probe: award score once airborne | `implementation_outlines.jsonl` |
| Board games: pointerdown vs mousedown, frozen idle board | Board probe pointerdown+pointerup; turn-based frozen-canvas exemption | `tools.py` |
| Stuck best-of-2 silently doubled fix time on single MLX | Default **off**; opt in `/bestof on` or `coder.py --stuck-bon`; candidates under `candidates/iter_NN/` | `agent.py`, `chat.py`, `coder.py` |
| Kung-Fu: movement gated on idle/walk only | Playbook: include crouch/duck in movement branch or reset action before move | `memory/playbook.jsonl` |
| Video intro: orphan setTimeout, enemies never spawn | Playbook: call `reset()`/`startGame()` — same path as R restart | `memory/playbook.jsonl` |
| Parallel roster: `f2_walk` shows `f1` art (MK, Street Fighter, chess colors) | Harness: `sprite()` token tie-break + flush `_spriteCache` on `_assetsReady`; memory: `versus-fighter-sprite-prefix` | `assets.py`, `tests/test_assets.py`, `memory/playbook.jsonl` |
| Game looks perfect to human but trace shows 2 `soft_warnings` | Often probe timing or partial patch — not always a visual bug; read `iter_summary.soft_warnings` | trace + `HARNESS_DEBUG.md` § “looks fine” |
| Cascade hazards roll uphill / skip mid-span tumble | INITIAL vx from slopeDir + ladder gaps (`ramp-hazard-roll-then-tumble`) | `memory/playbook.jsonl`, outline trap |
| Ladder mid-climb stuck then thrash-revert (DK 20260722) | Full-span `findLadderAt` + top-exit; pin ladder craft for vertical-platformer; drop bare `"climb"`→Rampage pin | `memory/playbook.jsonl`, `outline-vertical-platformer`, `agent_memory.py` |
| `/640png` appendix "1px drawImage columns" strong-hooked puzzle-grid over vector-wireframe (BATTLE10) | Strip `TARGET=/640*` before recipe match; drop generic `columns` strong_hook; wireframe vs puzzle disambiguate; `/640png` + vector-stroke uses /640 plan (no `<assets>`) | `memory.py`, `visual_playtests.jsonl`, `prompts_v1.py`, `prompt_library.py` |
| `/640png` footer `no three.js` tokenized as 3D (`three`+`threejs`, MIN_HITS=2) → Dig Dug seeded `canvas_3d_basic` + Jaccard mixed `jmr-png-sheets` with `classic-arcade-pixel-maps` → first-build `inline_data_bloat` (DIGDUGD2 `20260905_153855`; morning DIGDUGDI same wrong skeleton but still emitted HTML) | Strip `no/not/never/without three.js` and `no WebGL` before 3D detect; JMR `/640` `/640png` never seeds WebGL skeleton (recipe still picks grid); `/640png` suppresses inline pixel-maps; `ensure_ids` ignores negated three.js as WebGL intent. Do **not** dumb down `/critic auto` or the Dig Dug prompt — this is a false-positive, not "defaults too rich" | `modality.py`, `memory.py` `_modality_skeleton`, `agent_memory.py`; tests `test_b1_3d_negated_threejs_is_not_3d_intent`, `test_640png_arcade_does_not_inherit_threejs_skeleton`, `test_jmr_png_no_threejs_footer_still_pins_sheets_not_webgl` |
| `/640png` sprites "4× too small" (FROGGERC `20260905_191823`, DIGDUGD3): 24 px frog/digger on 640×480 — art was exactly the library's arcade-native `On-screen sizes:` | Library `prompt_640` sizes ~2× (arcade 16 px ≈ 40 px on glass; grid arenas capped); `jmr-png-sheets` + Phase-A blocks teach 32-48 px characters; harness floors animated subjects at 32 px (`apply_jmr_size_floor`, singles untouched) | `memory/prompt_library.jsonl`, `memory/playbook.jsonl`, `prompts_v1.py`, `assets.py`, `agent_assets.py`; tests `test_apply_jmr_size_floor_scales_animated_subjects_only`, `test_jmr_png_size_floor_reaches_generator_and_traces` |
| ZELDATOP (`20260905_200400`): TEST OK 6/6 but `unused_assets` nagged packed singles `npc.png`/`enemy.png`/`heart.png` (DIGDUGD3 leftover skip required `_` in the name); first-build skeleton `canvas_pinball_basic` at Jaccard 0.05 because `canvas-overworld-rpg` was ABSENT from the recipe→skeleton map; skip diagnostics `has_player_xy=false` on `state.hero`; self-`from_image` (`hero_idle`←`hero_idle`) burned pose-retry GPU; blitSpr used TILE=32 on 40px cells | Skip **all** non-`STEM-N.png` leftovers when HTML paints `jmr:spr` (not only underscore poses). Map `canvas-overworld-rpg` → `canvas_grid_basic`. Playtest diagnostics read `player\|\|hero\|\|ship\|\|digger\|\|avatar` (string facing counts). Drop `from_image` when it equals own name. Playbook/prompt: blitSpr cw,ch = sheet table, not TILE | `tools.py`, `memory.py`, `agent_critic.py`, `assets.py`, `prompts_v1.py`, `memory/playbook.jsonl`; tests `test_unused_assets_skips_jmr_packed_pose_leftovers`, `test_overworld_rpg_gets_grid_not_pinball`, `test_parse_drops_self_from_image` |
| DIGDUGD3 (`20260905_161144`): 30k untagged plan essay → 68k first-build history → silent 0-token stall; restart then failed `input_moves_player` after Pac-Man `auto_chaser_moves_autonomously`; `unused_assets=8` on leftover `digger_idle.png` | Keep Phase-A **tags**, drop untagged essay before first-build (`_planning_keep_structured_tags`); FPGA elides modest prose, full HTML keeps a crisp tagged plan. Skip chase auto-probes unless the goal has chase/pellet/pursuer class words. Run `input_moves_player` before other effectful autos. Default movement probe snaps `tx`/`ty` as well as `x`. Skip pose-filename unused-asset nags when HTML uses `jmr:spr`. `/640png` first-build nudge says `jmr:spr`, not `sprite()` | `agent.py`, `agent_critic.py`, `agent_helpers.py`, `tools.py`, `prompts_v1.py`; tests `test_lean_compact_drops_untagged_plan_essay`, `test_dig_tunnel_grid_skips_chase_auto_probes`, `test_unused_assets_skips_jmr_packed_pose_leftovers` |
| `adjacent_line_spam` killed a progressing 10 kB first build on `const c1..c4 = {x:0,y:0}` (BATTLE10) | Window 4 fires at 4 only for RAW-identical lines (digit-collapsed runs need 8); spam gets the same open-`<html_file>`/`<patch>` grace as `inline_data_bloat` (both call sites) | `ollama_io.py`, `backend.py`, `tests/test_repetition.py` |
| Plan streamed 50k completion / 18k visible tokens over 80 min (BATTLE10) | `stage="plan"` → `max_tokens` cap (`PLAN_MAX_TOKENS`, default 12000; 0 disables); cap hit → existing `plan_incomplete_retry`. Build/fix turns stay uncapped | `agent_stream.py` `_plan_stage_max_tokens`, `agent.py` plan `_stream(...)` calls |
| 36% of wall clock on polish turns after 8/8 probes; polish regressed + auto-reverted (CENTIPED, DOOM3DFI r4) | Polish cap **0** in simulator mode (`/640`, `/640png`); media mode keeps 2 | `agent_prompts.py` `_effective_polish_turn_cap`, `tests/test_capability_round.py` |
| 7/7 probes green but `partial patch apply` forced `ok=False` → two 0/1 patch turns (DOOM3DFI r1) | Advisory `warning` + recovery block on next prompt when ok, all probes green, no page errors; else forced retry as before | `agent_gates.py` `_partial_patch_is_advisory`, `agent.py` |
| oMLX 1800 s at 0 tokens with model already resident — keepalive lines kept the read from timing out (BATTLEZO) | First-token watchdog every loop pass: `min(stall, 300 s)` quiet + `/v1/models/status loaded=true` → abort with clear message; cold load still gets the full cap | `backend.py` `MLXServerBackend._stream_once`, `_model_reported_loaded` |
| `dropped_opening=True` on every `/640` first build — class outline never reached the model (CENTIPED, ANIMATIO, BATTLE10) | Simulator lean budget falls back to traps-only outline slice before dropping (`kept_opening_mode: traps_only`) | `agent_memory.py` `_outline_traps_only_for_goal` |
| 91-98% of Qwen completion tokens hidden CoT; patch turns 60-105 s for ≤67 visible tokens | Stage-aware effort: `fix`/`patch` → `low`; plan / first build keep `medium`; explicit `QWEN_REASONING_EFFORT` wins. `_stage` rides `options`, stripped before the wire | `backend.py` `chat_template_thinking_kwargs(stage=)`, `agent_stream.py` |
| `/640` Battlezone shipped a **top-down** tank game with 5/6 probes green (BATTLEZ2 `20260905_112313`) — the planner asked "2D... top-down?" and no gate disagreed | `prompt_640` said only "2D wireframe vector tank"; the full prompt's class-defining "First-person… recede toward a horizon" was dropped by the /640 shortening. Fixed #28 battlezone / #29 star-wars; test guards that first-person wireframe prompts keep the view words in `prompt_640`. This is a **memory_gap**, not something a critic fixes — the critic sees the same shortened goal | `memory/prompt_library.jsonl`, `tests/test_prompt_library.py::test_prompt_640_keeps_first_person_view_for_wireframe_games` |
| Code critic spawned "ON (parallel)" on **in-process MLX** (BATTLEZ3 `20260905_144613`): feedback router timed out (`feedback_router_parse_failed` after 60 s), iter-2 coder turn queued behind the review | In-process `BackendInfo.endpoint` is the sentinel `"in-process"` — not a loopback URL, so `_endpoint_supports_concurrency` fell through to "non-loopback ⇒ concurrent". Now: no `://` ⇒ serial; `MLXBackend` instances are never concurrent; a forced-on critic on a serial backend runs **inline** inside `_spawn_code_critic` (`code_critic_inline` trace, 600-token cap) so nothing shares the single Metal executor with the coder. TUI **`/server on`** or **`/model N server`** is how you put Qwen3.8-27B on oMLX so `/critic auto` is actually parallel — no `LLM_BACKEND=mlx-server` | `agent.py`, `agent_critic.py`, `chat.py` `/server`; tests `test_inprocess_mlx_is_never_concurrent`, `test_tui_server_command.py` |
| Every fix turn re-prefilled 24-35k tokens (60-120 s TTFT) on oMLX / in-process MLX although only the newest user turn changed | In-process: cross-turn `prompt_cache` (trim to shared prefix, prefill suffix only). All local: defer per-turn elision so history is append-only until 80% of ceiling (`prune_deferred_prefix_cache`). oMLX hot-cache off → one-shot TUI warning + `prefix_cache_status` trace | `backend.py` `plan_prompt_cache_reuse`, `agent_compaction.py` `_prefix_cache_friendly`, `agent_stream.py` `_maybe_report_prefix_cache_status` |
| `unused_assets=8` on JMR pages using `jmr:spr:N`; false `ENTITY-NOT-RENDERED [player]` on first-person/wireframe builds | `STEM-N.png` referenced via `jmr:spr:N` / `"jmr:spr:"+i` / `window.JMR_SPR`; skip viewpoint entity via recipe `viewpoint_entity` | `tools.py` `_jmr_sheet_referenced`, `memory.recipe_viewpoint_entities` |
| Phase 3: game-title literals (`doom`/`wolfenstein`/`quake`/`tetris`/`battlezone`/`pinball`) still hardcoded in Python detectors | Move class tokens into `visual_playtests.jsonl` / `plan_nudges.jsonl` (`auto_probe_requires_words`, `viewpoint_entity`, `disambiguation_signals`, `ensure_ids`, `gate_skips`, `applies_keywords`); Python only loads via `memory.py`; guard with `tests/test_no_game_title_literals.py` | `memory.py`, recipes JSONL, `agent_*.py`, `tools.py`, `prompts_v1.py`, `modality.py` |
| Campaign mix needs 8 `/640png` + 4 full-HTML in one serial batch | `TARGET=/640png` in goal → per-child `AGENT_JMR_PNG=1` (`_child_env_for_goal`); full-HTML clears simulator flags | `eval/tune_serial_loop.py`, `eval/tune_run20.sh`, `eval/tune_campaign_qwen38_goals.txt` |
| Phase 5 media A/B (optional) | `AGENT_SPRITE_SHEET_LORA` / `AGENT_SA3_SMALL_SFX` — off by default; missing weights fall back | `assets.py`, `sounds.py`, `DEV.md` |
| run_20 playtest: Dig Dug too fast; Frogger HUD 3/state 1; Zelda no enemies/left flip/Controls; Centipede top-only mushrooms + solid-red death; Battlezone no fire / reversed move; Pac-Man maze+ghosts ignore walls / fancy ghosts; DK start at top / flat floors | Class playbook+outline traps + recipe auto_probes/`ensure_ids` (`when_any` so Pac-Man maze pins do not hit Dig Dug); title sentences only in `prompt_library` | `memory/playbook.jsonl`, outlines, `visual_playtests.jsonl`, `prompt_library.jsonl`; `tests/test_run20_playtest_memory.py` |
| Policy pin: FPGA-only rule violations (`Object.keys`, `performance.now`, `"jmr:spr:"+i`, splice return, unicode `fillText`) must never fail Chrome-working code | Regression test only — harness already teach-only | `tests/test_simulator_mode.py::test_fpga_only_rule_violations_never_fail_micro_probes` |
| Attack limb points away from opponent | Code EXTRA flip — no VLM (`attack-sprite-wrong-direction-flip-in-code`) | `memory/playbook.jsonl` |
| Versus P2 incomplete pose roster → MISSING boxes | Same pose suffixes both prefixes (`versus-fighter-sprite-prefix`) | `memory/playbook.jsonl` |
| Sprite opaque when figure touches image edge | Chroma: near-white 5/8 + border-majority fallback | `assets.py`, `tests/test_tier1_2.py` |
| /640 HUD `♦` lives paint as `b`; WAVE overlaps LIVES | Teach in prompts/playbook only (do **not** microprobe-fail Chromium). ASCII fillText; `textAlign=left`; lives via `fillRect`/`*` | `prompts_v1.py`, playbook `jmr-filltext-ascii-hud` |
| /640 `splice` return used as the tail array | Teach only: FPGA return is undefined (ghost chain, wave 2 never). Copy-down. Never gate Chromium | `prompts_v1.py`, playbook `jmr-splice-return-undefined` |
| Pseudo-3D rivals stack / car undrawn | Discrete lanes + z-sort + drawImage (`pseudo3d-curved-road`) | `memory/playbook.jsonl`, racing outline |
| Maze chase looks frozen | Don’t gate mouth cycle on dir; chasers must move (`maze-chase-sprite-chomp-cycle`) | `memory/playbook.jsonl` |
| Seed empty `_assets/` + assets-only goal → Asteroids roster into Bomberman PATHS folder | Declared PATHS = roster; skip only when **every** declared stem is on disk; orphans never count; art/replace intent (not keep-code phrases) forces declared regen + media_only | `agent_helpers.py`, `agent_assets.py`, `agent.py`, `prompts_v1.py` |

### Recurring patterns (watch in new traces)

- **memory_gap: assets loaded but undrawn** — self-recovers but burns iters (Joust 5×). Often state-gated sprites (enemies not spawned yet) or drawImage not wired — playbook `draw-generated-sprites-not-boxes` + `td-enemies-follow-waypoints` / spawn timers; not new agent machinery.
- **input_moves_player false-fail** — harness dispatches ArrowRight; if prior keydown left player in crouch/block, movement gated on `idle||walk` fails probe while game feels fine to a human.
- **ENTITY-NOT-RENDERED soft_warning** — fix only if 2+ games show same pattern; fog-hidden tiles are now skipped (`seen`/`explored`).
- **OOM at game 12** — verify media/MLX freed at serial game boundaries; run heaviest media game earlier or isolated.
- **BrokenPipe before materialize (run_16 Centipede)** — the full first-build HTML reached `assistant_reply`, but an overnight relaunch closed stdout; printing cosmetic notes raised before `_materialize`, losing the artifact (`rc=2`). `coder.py` stdout is now best-effort so a detached reader cannot abort the agent loop.
- **SIGKILL / exit=-9 with no HTML (run_15/16)** — separate from BrokenPipe: jetsam / early restart can leave Centipede/Galaga at planning or `stream_start` with no reply or `iter_summary`. Crash-bonus retry exists; optional dedicated re-run for infra-only fails.
- **Repetition / think thrash (run_16 pinball, torch)** — plunger kinematics or asset-loader loops burn 15–40+ min; outline traps + harness abort help but still waste wall clock under max-iters 3.
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
| `/critic` (code critic, Sept 2026) | spawned after materialize → folded into the queued next user turn (`[CODE CRITIC]`), else `_pending_coaching` | `auto` = ON on oMLX / cloud (parallel stream), off on serial backends; `/allroles` or `AGENT_CODE_CRITIC=on` force it. Works with or without `/wait` |

**Three-role reality check (Sept 2026):** architect = Phase A planner + exit decision; coder = every build/fix turn; critic = **three independent reviewers** that all end in the coder's next prompt — `/critique` (scripted playtest, no LLM), `/vlm-critique` (screenshot, needs a VLM), `/critic` (source review, any text model). Before Sept 2026 a loopback oMLX endpoint was classed as a serial daemon (`_endpoint_supports_concurrency`), so any same-instance critic ran inline and stuck best-of-2 ran sequentially — on a continuous-batching server both now run in parallel (`_backend_supports_concurrency`; `_available_sampler_slots` offers `slot1b`). `OMLX_SESSION_KEEP_MODELS` stops one role's pre-stream unload from evicting another role's weights. Critic notes were also silently dropped on a clean probe report because the real prefix `[VLM-CRITIQUE]` was missing from `must_keep_keywords` in `agent_feedback.py` — fixed alongside `[CODE CRITIC]`.

**Triage traps (TD seed trace `20260630_114658`):**

- `assets_parse_failed` at **phase_a** on a seed run ≠ “no art this session.” Build-turn **mid_session** `<assets>` can still generate sprites (`tower_tesla_idle`, `tower_flame_idle` in that trace).
- `ASSETS_LOADED_BUT_UNDRAWN` = sprites on disk but not drawn — **wiring** issue, not Z-Image failure. Playbook `draw-generated-sprites-not-boxes` is the right lever.
- **Safari vs Chromium trap:** sprites visible in Safari but `ASSETS_LOADED_BUT_UNDRAWN` in Playwright often means the harness sampled `__drawImageEvents` before async `loadAssets()` finished (placeholder `fillRect` frames first). Check trace `asset_decode_settle.ready`; re-run `scripts/_smoke_asset_decode_settle.py` before chasing model wiring.
- User feedback naming sprites **already in `_session_assets`** should get wire/draw coaching, not `ASSET GENERATION REQUIRED`.
- **Per-entity unique art** ("each tower its own unique head sprite") must stay armed through vague retry nudges — do not fuzzy-match an existing `*_head_*` asset and clear `_unhonored_asset_request`.
- **Seed empty disk + assets-only** (Bomberman PATHS, zero PNGs, goal “create assets / keep code identical”): do **not** treat as fresh game. Force declared PATHS stems; status label must match active diffuser (`FLUX2` vs Z-Image).
- **Seed declared PATHS vs disk leftovers:** coverage = every declared stem present; orphan `ship.png` in the folder does not unlock skip or become `allowed_asset_names`. Art/replace intent is wording-agnostic (`assets`/`sprites`/`generate`/`new`), not a keep-code phrase list.

### run_vlm10 batch (Jul 2026) — durable learnings

10-game Qwen 27B + VLM batch: **5 fresh pass**, **2 artifact pass**, **3 fresh fail**
(PoP, Monkey Island, Dragon's Lair).

| Pattern | Fix (harness/memory) |
|---------|----------------------|
| **7/10 iter-1 `ASSETS_LOADED_BUT_UNDRAWN`** | First-build `generated_sprite_draw_contract()` inline + pin `draw-generated-sprites-not-boxes` at plan stage when assets exist |
| **`ASSETS_DROPPED_PENDING` (Dragon's Lair)** | Persist dropped specs; `_maybe_autogen_pending_dropped_assets()` at iter start; exclude from `_failure_blames_code` writeback |
| **Malformed Phase-A probes (OutRun)** | `_lint_probe_syntax()` + one-shot plan re-stream before first build |
| **`HOTSPOT_ALIGNMENT_MISS` (Monkey Island)** | Actionable coords in warning; pin `pointclick-hotspot-from-source-art`; gbox fix_hint on `canvas-point-and-click` |
| **Post-clean shrink (OutRun / castle courtyard)** | `post_clean_shrink_rollback`: reject materialize that shrinks a clean build >20% AND is structurally truncated; keep baseline; arm `<html_file>` rewrite. Well-formed shrinks still write + `post_clean_shrink_detected` coach |
| **Chess audit misrouted to PoP** | Narrow `chess-path-walk-no-teleport` tags — drop generic `walk`/`path`/`teleport` (matched platformer goals) |
| **`auto_*` probe lint noise** | Skip recipe-injected `auto_*` probes in `_probes_referencing_unassigned_props` — model cannot fix alias-tolerant harness probes |
| **Platformer `jump_works` races landing** | `outline-side-scroll-platformer` probes: assert `vy<0` or `onGround` flip within ~300ms, not `y!==y0` after landing |

**Deferred (noted, not fixed):** QTE `no_action_frame_captured` — harness never captures an action frame during the QTE window for `canvas-cutscene-qte`, so the VLM pose-change question is always skipped; needs dispatch-during-window plumbing in a separate task.

Re-validate failures only: `eval/tune_run_vlm10_failed3.sh` (3 goals, fresh Python process).

## Read order

**New harness agent:** start with **§ “New agent — harness improvement”** at the top of this file,
then `DEV.md` → **`eval/OPERATIONS.md`** (run batch / pytest) → `TEST.md` (suite map) →
`tools.py` (`load_and_test`) → trace `.jsonl` → relevant `memory/*.jsonl`. Agent comparison vs
Cursor/Aider: **`README.md#how-this-compares-to-other-coding-agents`**.

Quick test: `.venv/bin/python -m pytest tests/ -q` (pure-function; no GPU/model). **Full suite must
pass before push** (~2343 tests / 195 files, ~1 min; see `TEST.md`).
