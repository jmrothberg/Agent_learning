# HARNESS_DEBUG.md — what the harness does, what it can't, and how to debug a bad session

For future-me, after a session goes poorly. Blunt about the limits.
**The harness detects, coaches, and learns from failure. It does not make the model better at
writing code in a single session.** Most first-iter quality comes from the model + playbook +
prompts, not the detectors. If sessions flat-line, the question is *which bottleneck* (below), not
"add another detector."

## The one mental model
**Most "agent failures" are verifier failures** — the harness said `ok=True` while the game was
broken, so the fix-loop never engaged. Before adding loop machinery: *does the gate flip
`ok=False` on this failure?* If yes, the existing loop usually fixes it. The classic trap: probes
check **state** (`state==='punch'`, a field changed) and pass while the **pixels/behavior** are
wrong. When you debug or claim success: **open the game, drive it, look at pixels** — never trust
"TEST OK."

## The gates today (all in `tools.py` `load_and_test` / `_input_smoke_test`, genre-free)

Flip `ok=False`:
- **PLAYER-STUCK** — movement key registers but no *position* leaf moves (spawned in a wall /
  collision blocks every move).
- **ACTION_DRAWN_NOT_SPRITED** — an action key changed the canvas by code-drawing
  (`fillRect`/`lineTo`/`stroke`) but drew NO new sprite source = a faked action (e.g. a kick drawn
  as lines over idle instead of swapping to the kick sprite).
- **CODE_DRAWN_OVER_SPRITE** — the action DID draw its sprite but ALSO code-drew stroke/arc shapes
  on top (a "motion line + flash"/reach-bar/limb). The sprite conveys the move; the overlay is junk.
- **ASSETS_LOADED_BUT_UNDRAWN** — a sprite PNG loaded but never `drawImage`'d this run. Usual cause
  is a sprite-KEY mismatch (`'left_idle'` built vs `'left_fighter_idle'` generated) → silent
  fillRect fallback. The injected loader's `sprite(key)` resolver tolerates the drift.
- **PROCEDURAL_REGRESSION_SUSPECTED** — ≥3 sprites declared but the canvas is mostly big fillRects,
  drawImage outnumbered 5:1 (entities rendered as colored boxes).
- **ENTITY-NOT-RENDERED** — entity in `state` with x/y but not drawn.
- **STATIC-ACTION** — the responsive action key renders one held pose while the game animates
  elsewhere (a frozen "animation").
- **Frozen-canvas, elision/duplicate-decl/brace micro-probes** (pre-Chromium), unclosed/truncated
  `<html_file>`.

Advisory (surfaced as `warnings`, NEVER gate): **dead / near-identical sprite frames** — cosmetic.
Do not regenerate to "fix" them and do not defer the user's gameplay feedback behind them.

## The visual critic (the human-eyes check for bugs probes can't see)
The critic is the only thing that catches "fighters facing the wrong way / punch renders backward."
It was effectively useless until 2026-06-03 — three real bugs, all fixed:
1. qwen3.6 (a real VLM) **sees** the screenshot but answers in reasoning prose, never emitting
   `Qn: yes/no` → parse_rate 0 → dropped. **Fix:** prefill the critic's assistant turn with
   `"Q1: "` so generation starts inside the format; parser tolerates a doubled ordinal.
2. The abstain guard false-fired on real findings ("I don't see a projectile"). **Fix:** abstain
   only when the model says it can't see the *image/screenshot* itself.
3. A blanket `fix_hint` preached facing-flip on every critique → the model flipped a correctly-
   facing sprite (oscillation). **Fix:** per-question `fix_hints` surface ONLY failed checks'
   advice. The critic also skips entirely when `backend.is_vlm()` is False (text models hallucinate
   about images). All per-action frames are saved to the trace (`iter_NN_action_<Key>.png`).

## Sampling (MLX) — a non-obvious quality lever
The MLX sampler must pass `top_p`/`top_k`. Untruncated temperature sampling (top_p=0, top_k=0)
causes the degenerate identical-line repeat loop that truncates big builds. Defaults follow the
Qwen3.6 coding preset: temp 0.6, top_p 0.95, top_k 20 (env-overridable). Repetition penalty stays
off — it hurts code, which legitimately repeats `}`/`const`/`ctx.`.

## Debug a bad session in ~5 greps
Trace: `games/traces/<slug>__run_*.jsonl`. The first non-empty answer is the bottleneck.
1. **`iter_summary`** — `ok`, `fail_reason`, which soft_warnings repeat across iters. Same FAIL
   every time → a playbook bullet is missing; different → base-model quality.
2. **`structured_compaction`** — fired early (pressure near 1.0 on a small prompt)? wrong num_ctx.
3. **`playbook_injected` / `opening_book_retrieved`** — did your bullet/recipe actually reach the
   model? (Jaccard floor ~0.02; broaden tags if not.)
4. **`visual_playtest_parsed` vs `visual_critic_abstained`/`unparseable`** — is the critic giving a
   real verdict, or being dropped?
5. **`patch_outcome`** — SEARCH-not-found = stale file view (usually compaction shredded it).
Then **open the game and drive it.**

## Why dozens of tries can show no improvement
Detectors abort faster and coach clearer; they don't add HTML/JS skill. Headroom lives in:
1. the base model's one-shot architecture ceiling, 2. playbook coverage (token-Jaccard retrieval
misses synonyms), 3. the visual critic actually gating quality (now functional — see above),
4. a single model being both implementer and reviewer. If 30 runs are flat, it's one of these, not
a missing detector.

## Standing rules (don't break)
1. Tune the agent, not the model. 2. No hardcoded genre lists (detect by rendering/interaction
shape). 3. All code self-contained in `Agent_learning/`. 4. Visible Chromium by default (TUI).
5. Asteroids is the canonical regression check. 6. Never silently call a cloud model.

## Files to look at
`tools.py` (`load_and_test`, `_input_smoke_test`, the draw shims + gates) ·
`agent.py` (`run`, `run_visual_critic`, `_apply_dead_animation_check_to_report`, fix-prompt) ·
`assets.py` (`render_asset_paths_block` → the `sprite()` loader, txt2img pose frames) ·
`backend.py` (MLX sampler, VLM path) · `memory/*.jsonl` (playbook, visual_playtests, outlines,
skeletons, system_battery) · `prompts_v1.py` (data-driven system prompt).
Tests are pure-function (~1 min): `.venv/bin/python -m pytest tests/ -q` (1377 passing).
