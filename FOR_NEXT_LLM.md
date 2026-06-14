# FOR_NEXT_LLM.md — fixing this agent

You're tuning a coding agent that drives a **local ~27-35B model** (qwen3.6 via MLX/Ollama)
to build single-file HTML5 games, verified in real Chromium. Your job is to make *that* class
of model ship working games. Read this, then `CLAUDE.md`. ~3 min.

## The 5 rules

1. **Tune the agent, not the model.** Never "try a bigger model." Fix prompts / retrieval /
   harness checks / scoring / memory.
2. **General fix → code. Specific guidance → memory.** A mechanism that helps 100s of game
   types goes in `tools.py`/`agent.py`/`assets.py`. Genre/game craft goes in `memory/*.jsonl`
   (retrieval-gated). Never put a genre string (`if "pacman" in goal`) in code.
3. **Genre-free in code.** Detect by observable *shape* (exposed `state.player.x`, canvas size,
   keyword/recipe gates), never by subject matter.
4. **Listening beats speed.** The user's typed feedback is authoritative. `/rawfeedback` defaults
   ON (no classifier wrapping). Don't add a regex that overrides the user's words.
5. **Length ≠ failure once a stream is producing code.** Don't tighten an abort threshold to kill
   a long-but-legitimate first build. Latch on code-emission (`<html_file>`, `<!DOCTYPE`,
   `function`, `const x =`), not length.

## The one mental model: verification is the lever

**Most "agent failures" are verifier failures** — the harness said `ok=True` while the game was
broken, so the fix-loop never engaged. Before adding agent-loop machinery, ask: *does the gate
flip `ok=False` on this failure?* If yes, the existing loop usually fixes it. The biggest wins in
this project are all gate fixes. The trap: probes check **state** (`state==='punch'`,
`facing===1`, "a key changed a field") and pass while the **pixels/behavior** are wrong.

**When you debug or claim success: LOOK and DRIVE.** `Read` renders PNGs — generate an asset and
open it (a delta metric will say "1.6% changed = consistent" while the punch is visibly just
idle). Load the built game, press keys, read the player's position, screenshot it. Never trust
"TEST OK."

## Where things live

**Code (general mechanisms):**
- `tools.py` — the verifier. `LiveBrowser.load_and_test` (Chromium), `_input_smoke_test`
  (presses keys, detects movement/actions), micro-probes, the gates. **This is the highest-leverage
  file.**
- `agent.py` — the async `GameAgent.run` loop, prompt assembly, compaction, memory retrieval.
- `assets.py` / `sounds.py` — in-process Z-Image-Turbo / Stable Audio (lazy GPU load).
- `prompts_v1.py` — data-driven system prompt (`FormatSpec` list; don't hand-edit the rendered blob).

**Memory (genre/game craft — one JSONL line to add, no restart):**
| File | Holds |
|---|---|
| `memory/playbook.jsonl` | code rules-of-thumb, retrieved by Jaccard on the goal each turn |
| `memory/visual_playtests.jsonl` | mechanism-keyed yes/no VLM checklists + `auto_probes` |
| `memory/skeletons/` | first-build HTML templates per mechanism |
| `memory/playtests.jsonl` | autonomous behavioral playtest recipes (seeded in `memory.py`), used by the `/critique` (no-vision) review |

Retrieval gotcha: a bullet needs tag overlap with goal text to clear the **~0.02 Jaccard floor**,
or it never reaches the prompt. If a relevant bullet doesn't fire, broaden its tags (tags weigh 2×).

## Traps that have cost real time (don't repeat these)

- **CONSISTENT ANIMATION NEEDS AN IMAGE — never a series of txt-only frames.** The whole reason the
  asset pipeline exists: a character that kicks/punches must be the *same* character across frames.
  - **img2img can't change a pose** (Z-Image/SD-Turbo at `guidance_scale=0` stay locked to idle at
    every strength — proven A/B `animation_ab/`). `from_image` is recolor/restyle only.
  - **A fresh txt2img *replacement* frame breaks consistency** with the character already in the
    running game — and consistency is the hard constraint. So **don't regenerate a pose frame to
    "fix" a dead/wrong one** — both routes are dead ends. A near-identical/dead frame is COSMETIC:
    surfaced as an advisory `warning`, it must NOT flip `ok=False` or defer the user's gameplay
    feedback. (See `feedback_sprite_animation_from_image.md`; `_apply_dead_animation_check_to_report`.)
  - At PLAN time, request all named pose frames as **txt2img with one shared character description +
    fixed seed**. In-session, cycle the frames you have; convey the action with the SPRITE, not code.
- **Don't draw the action in CODE over the sprite.** A kick = swap to the `*_kick` sprite, never a
  `ctx.lineTo`/`fillRect` limb or a "motion line + flash" bolted on top. Two gates enforce this:
  `ACTION_DRAWN_NOT_SPRITED` (action changed the canvas by code-drawing but drew no new sprite) and
  `CODE_DRAWN_OVER_SPRITE` (drew the sprite AND stroke/arc junk on top).
- **Sprite-key drift silently shows colored boxes.** The model builds `'left_idle'` but the asset is
  `'left_fighter_idle'`, so `ASSETS['left_idle']` is undefined → silent `fillRect` block. The injected
  loader now provides a `sprite(key)` resolver (exact→normalized→token match) + a loud `MISSING`
  marker; `ASSETS_LOADED_BUT_UNDRAWN` names the mismatch. Don't index `ASSETS` directly.
- **The visual critic must SEE and must ABSTAIN-not-lie.** qwen3.6 is a real VLM and sees screenshots,
  but it answers in reasoning prose and never emits `Qn: yes/no` → parse_rate 0 → dropped. Fix: the
  critic call **prefills the assistant turn with `"Q1: "`** to force the format. And abstain detection
  is anchored to "can't see the *image*" — a real finding ("I don't see a projectile") must NOT be
  treated as blindness. Per-question `fix_hints` surface ONLY advice for checks that failed (a
  blanket fix_hint once made the model flip a correctly-facing sprite → oscillation).
- **Wrong sprite direction → flip in CODE, don't regenerate.** Per-state `ctx.scale(-1,1)` beats a
  model call. Prefer code transforms (flip/rotate/offset/scale) over asset regen for any wrong
  direction/size/position.
- **Compaction denominator.** Pressure = `prompt_tokens / num_ctx`. A wrong `num_ctx` (it once
  defaulted to the 32K Ollama output cap for *every* backend) makes a 40K prompt read as >100% on a
  200K model → lossy compaction every turn, shredding playbook + user instructions + file view →
  "feedback doesn't stick." Default is now 100K; only compact near a genuinely full window.
- **Movement units.** Tile-step `moveProgress` is a 0→1 fraction → `+= speed*dt` (tiles/sec); a
  stray `/TILE` makes everything ~20× too slow.
- **Weak models can't author a maze** (repetition loop → truncation). Give them one to *copy*.
  Sampling matters too: the MLX path must pass `top_p`/`top_k` (vendor coding preset
  temp 0.6/top_p 0.95/top_k 20) — untruncated sampling causes the degenerate line-repeat loop.
- **Never call cloud silently.** Cloud confirms need the user's API key and explicit opt-in.
  Iterate on free local; expect qwen to truncate/loop on big builds — judge harness signals from
  the trace, not always a finished game.

## The gates today (in `tools.py`, all genre-free, all flip `ok=False`)

- **PLAYER-STUCK** — a movement key registers but no *position* leaf changes → stuck (in a wall /
  collision). (`_is_position_leaf`, `_MOVEMENT_KEYS`.)
- **ACTION_DRAWN_NOT_SPRITED / CODE_DRAWN_OVER_SPRITE** — per action key, snapshots drawImage
  sources + fillRect/stroke counts around the hold: a real action draws a NEW sprite src; a faked
  one only code-draws (no sprite), or draws the sprite AND scribbles stroke/arc on top.
- **ASSETS_LOADED_BUT_UNDRAWN / PROCEDURAL_REGRESSION_SUSPECTED** — sprite loaded but never drawn
  (key mismatch / fillRect blocks instead of `drawImage`).
- **Action frame + STATIC-ACTION** — presses the model's `<criteria>` keys, captures the peak
  *transient* frame per key (all saved to the trace as `iter_NN_action_<Key>.png`), feeds them to
  the VLM critic; a held-static "animation" is flagged.
- **ENTITY-NOT-RENDERED** — entity in `state` with x/y but not drawn on canvas.
- **Frozen-canvas, asset-path-exists, audio-events-fire** — see `load_and_test`.
- **Advisory (NOT gating):** dead/near-identical sprite frames — cosmetic, never block shipping.

## Debug a bad session in 5 greps

Trace is `games/traces/<slug>__run_*.jsonl`. Grep for: `iter_summary` (ok, action_frame_captured,
static_action, player_stuck), `structured_compaction` (reason/pressure — premature?),
`playbook_injected` (did your bullet reach the model?), `visual_critic_*` (image_count — did it see
an action?), `patch_outcome` (SEARCH-not-found = stale view, usually compaction). Then **open the
game and drive it.**

## Read these first
`CLAUDE.md` (operational summary) → `tools.py` `load_and_test` + `_input_smoke_test` → the most
recent trace `.jsonl` for the failure you're chasing → the relevant `memory/*.jsonl` bullets.
Tests are pure-function, ~12s: `.venv/bin/python -m pytest tests/ -q`.
