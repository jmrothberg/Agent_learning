# HARNESS_DEBUG.md — debug a bad session

The harness **detects and coaches**; it does not make the model write better code in one session.
If quality flat-lines, find the bottleneck (below), not “add another detector.”

Tuning rules and mistake traps: **`FOR_NEXT_LLM.md`**. Commands and env: **`CLAUDE.md`**.

## Mental model

Most “agent failures” are **verifier failures** — `ok=True` while broken, so the fix loop never runs.
Probes check **state**; gates check **behavior and pixels**. When debugging: open the game, drive
it, read screenshots — never trust “TEST OK” alone.

## Gates (`tools.py` — all genre-free; hard gates flip `ok=False`)

| Gate | What it catches |
|------|-----------------|
| **PLAYER-STUCK** | Movement key registers but no position leaf changes (wall spawn, blocked collision) |
| **ACTION_DRAWN_NOT_SPRITED** | Action changed canvas via code-draw only — no new sprite |
| **CODE_DRAWN_OVER_SPRITE** | Sprite drawn but stroke/arc “motion lines” on top |
| **ASSETS_LOADED_BUT_UNDRAWN** | PNG loaded, never `drawImage` / texture bind (key mismatch) |
| **PROCEDURAL_REGRESSION_SUSPECTED** | Many sprites declared; canvas mostly big fillRects |
| **ENTITY-NOT-RENDERED** | Entity in `state` with x/y not drawn |
| **STATIC-ACTION** | Action key holds one pose while game animates elsewhere |
| **Frozen-canvas** | RAF runs but pixels unchanged (and input didn’t explain it) |
| **Micro-probes** | Truncated HTML, bracket imbalance, elision sentinels (pre-Chromium) |
| **Model `<probes>`** | Phase-A acceptance checks evaluated each iter |
| **Opening-book checks** | Retrieved memory recipes (e.g. P&C chain only when goal is P&C) |

**Advisory only (never gate):** dead / near-identical sprite frames; harness-env pointer-lock noise.

Per-action frames save to the trace as `iter_NN_action_<Key>.png`.

## Visual critic (`/vlm-critique`)

Mechanism-keyed yes/no checklist vs screenshots — catches facing/pose bugs probes miss. Requires a
VLM backend; skips on text-only models. See **`FOR_NEXT_LLM.md`** for prefill/abstain/fix_hint traps.

## Debug: start with the timeline

The `.jsonl` is an **LLM-only artifact** (humans never read it) — it carries only
milestone + diagnostic events. Live-monitoring chatter (`stream_heartbeat`,
`stream_progress`) is NOT persisted; its signal (tokens / tok·s / duration /
prefill) is folded onto `iter_summary`.

**One command for the whole session** (reuses `render_run_summary`):

```bash
.venv/bin/python scripts/enrich_trace.py <session-id> --timeline
```

One row per iter: `ok · probes · patch N/M · router intent · tok/s · failure_class · blocker`
(`test_skipped:no_browser` when Chromium was intentionally absent — seed-edit eval).

### `failure_class` — which layer to fix (on `iter_summary` AND `no_usable_code`)

| Class | Means | Fix where |
|-------|-------|-----------|
| **harness_bug** | harness contradicted a correct model/router turn (coaching suppressed, art request cleared, false-positive gate) | agent/harness **code** |
| **memory_gap** | clean materialize but an avoidable mistake canned guidance should pre-empt (undrawn art on art intent, missed build-order) | add expert guidance to **`memory/`** (not the local model) |
| **local_llm_limit** | model-side stall, no harness contradiction (deliberation / repetition / silent / wrong envelope) | **local model** limit — mitigate via prompt/format forcing |
| **none** | nothing actionable to triage | — |

Precedence: `harness_bug` > `memory_gap` > `local_llm_limit`. The canonical
Fieldrunners iters 4–5 emit `no_usable_code` (not `iter_summary`) — `failure_class`
lives on both so those failures are one-grep visible.

### Then ~5 follow-up greps

1. **`iter_summary` / `no_usable_code`** — `failure_class` + `failure_reason` first; then
   `fail_reason`, repeating soft_warnings.
2. **`structured_compaction`** — early fire + high `pressure` → wrong `num_ctx` or denominator.
3. **`playbook_injected` / `opening_book_retrieved`** — did memory actually reach the model?
4. **`visual_critic_*`** — parsed vs abstained/unparseable; `image_count` for action frames.
5. **`patch_outcome`** — SEARCH-not-found → stale file view (often compaction).

Then **open the game and drive it.**

## When dozens of tries show no improvement

1. Base model one-shot ceiling  
2. Playbook / opening-book coverage (Jaccard misses synonyms)  
3. Visual critic not running or not gating usefully  
4. Same model as both implementer and reviewer  

## Key files

`tools.py` · `agent.py` (`run`, `run_visual_critic`, fix prompts) · `assets.py` (loader block) ·
`backend.py` (MLX sampler, VLM) · `memory/*.jsonl` · `prompts_v1.py`

Tests (no GPU/model): `.venv/bin/python -m pytest tests/ -q`
