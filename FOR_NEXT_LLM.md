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
| `memory/playtests.jsonl` | autonomous behavioral playtest recipes (seeded in `memory.py`) |

Retrieval gotcha: a bullet needs tag overlap with goal text to clear the **~0.02 Jaccard floor**,
or it never reaches the prompt. If a relevant bullet doesn't fire, broaden its tags (tags weigh 2×).

## Traps that have cost real time (don't repeat these)

- **Compaction denominator.** Compaction pressure = `prompt_tokens / num_ctx`. If `num_ctx` is
  wrong (it once defaulted to the 32K Ollama output cap for *every* backend), a 40K prompt reads
  as >100% on a 200K model and the lossy compaction fires **every turn**, shredding the playbook +
  the user's instructions + the file view → "feedback doesn't stick," patches fail. Default is now
  100K; only compact near a genuinely full window.
- **img2img can't change a pose.** For animation frames (punch/kick), use **txt2img per pose with
  one shared character description** (fixed seed → consistent character). img2img from idle returns
  idle at low strength, a different character at high strength. (`from_image` is for recolor/restyle.)
- **Wrong sprite direction → flip in CODE, don't regenerate.** An extra per-state `ctx.scale(-1,1)`
  beats a model call. General rule: prefer code transforms (flip/rotate/offset/scale) over asset
  regen for any wrong direction/size/position.
- **Don't let the harness suggest the rejected pattern.** A warning that said "render the limb in
  code" caused the procedural-limb relapse users reject. Watch your own coaching text.
- **Movement units.** Tile-step `moveProgress` is a 0→1 fraction → `+= speed*dt` (tiles/sec); a
  stray `/TILE` makes everything ~20× too slow.
- **Weak models can't author a maze** (repetition-loop → truncation). Give them one to *copy*
  (`pacman-maze-copy-dont-generate`).
- **Never call cloud silently.** Cloud confirms need the user's API key and explicit opt-in.
  Iterate on free local; expect qwen to truncate/loop on big builds — judge harness signals from
  the trace, not always a finished game.

## The gates today (in `tools.py`, all genre-free, all flip `ok=False`)

- **PLAYER-STUCK** — a movement key registers but no *position* leaf changes → stuck (in a wall /
  collision). (`_is_position_leaf`, `_MOVEMENT_KEYS`.)
- **Action frame + STATIC-ACTION** — presses the model's `<criteria>` keys, captures the peak
  *transient* frame (restart/persistent keys excluded), feeds it to the VLM critic; a held-static
  "animation" is flagged.
- **State-vs-render** — entity in `state` with x/y but not drawn on canvas (`ENTITY-NOT-RENDERED`).
- **Frozen-canvas, asset-path-exists, unused-asset, audio-events-fire** — see `load_and_test`.

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
