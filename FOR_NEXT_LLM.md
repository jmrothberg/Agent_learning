# FOR_NEXT_LLM.md — tuning this agent

You are fixing a coding agent that drives a **local ~27–35B model** (qwen3.6 via MLX/Ollama) to
build single-file HTML5 games verified in real Chromium. Read **`CLAUDE.md`** next (commands, env,
binding rules). Use **`HARNESS_DEBUG.md`** when debugging a bad session trace.

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
| Agent loop | `agent.py` — `GameAgent.run`, compaction, memory injection |
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

## Debug workflow

See **`HARNESS_DEBUG.md`** (5-grep recipe on `games/traces/*__run_*.jsonl` or batch
`games/tune_serial10/run_XX/traces/*__run_*.jsonl`).

## Serial tune learnings (run_05 → run_06)

Persisted from `games/tune_serial10/run_05/triage.md` before artifact cleanup. Per-run scratch
pads stay gitignored under `games/tune_serial10/run_XX/triage.md`.

**run_05 score:** 11/12 PASS (GLM-5.2-MLX-4bit). Sole FAIL: Dragon's Lair — exit=-9 SIGKILL ×3
(OOM after Holochess 48-sprite session). **run_06:** partial re-run list in
`eval/tune_serial10_round2_rerun.txt` — run Dragon's Lair first / in isolation.

### Mid-batch harness fixes (applied in repo)

| Symptom | Fix | File(s) |
|---------|-----|---------|
| Iter-1 first build rambled 37–50k tok before `<html_file>` | Force prefill opening on local MLX/Ollama iter-1 (`<!DOCTYPE html>` constrained decode) | `agent.py` |
| Holochess 33 KB `<html_file>` rejected — micro_probe false positives on `} else {` branches | Benign script-repeat filter; promote repetition only when multiple scripts signal | `tools.py` |
| Art-intent builds: assets loaded but procedural boxes drawn | drawImage contract last on first build; playbook bullet; skeletons teach drawImage; undrawn advisory | `prompts_v1.py`, `agent.py`, `memory/playbook.jsonl`, skeletons, `tools.py` |
| Probes false-fail on `window.state` | Require `window.state = state` in HARD_RULES + first build | `prompts_v1.py` |
| ENTITY-NOT-RENDERED gates ok when all probes pass (thin crosshair) | Bbox sample + advisory when probes green | `tools.py` |
| Board games: pointerdown vs mousedown, frozen idle board | Board probe pointerdown+pointerup; turn-based frozen-canvas exemption | `tools.py` |

### Recurring patterns (watch in new traces)

- **memory_gap: assets loaded but undrawn** — self-recovers but burns iters (Joust 5×). Playbook/outline pre-empt, not new agent machinery.
- **ENTITY-NOT-RENDERED soft_warning** — fix only if 2+ games show same pattern (Missile Command iter 2).
- **OOM at game 12** — verify media/MLX freed at serial game boundaries; run heaviest media game earlier or isolated.

## Read order

`CLAUDE.md` → this file → `tools.py` (`load_and_test`) → trace `.jsonl` for the failure → relevant
`memory/*.jsonl`. Agent comparison vs Cursor/Aider: **`README.md#how-this-compares-to-other-coding-agents`**.
Tests: `.venv/bin/python -m pytest tests/ -q` (pure-function, no GPU/model).
