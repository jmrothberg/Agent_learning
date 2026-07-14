# AGENTS.md — source vs artifacts

Map for maintainers and LLMs: **edit source, read artifacts for triage, never commit generated games.**

---

## 0. Who reads what

| File | Cursor | Game LLM | Purpose |
|------|:------:|:--------:|---------|
| `AGENTS.md` | yes | **no** | Router, edit vs artifacts, mixin map |
| `DEV.md` | yes | **no** | Commands, env vars, architecture |
| `HARNESS_TUNING.md` | yes | no | Harness traps, onboarding, trace→fix patterns |
| `FOR_NEXT_LLM.md` | yes | no | Legacy redirect → `HARNESS_TUNING.md` |
| `HARNESS_DEBUG.md` | yes | no | Gates, `failure_class`, enrich_trace |
| `eval/OPERATIONS.md` | yes | no | Natural-language → shell commands |
| `TEST.md` | yes | no | What each pytest guards |
| `README.md` | optional | no | Human onboarding |
| `prompts_v1.py` + `memory/` | no | **yes** | Canonical game-codegen rules + retrieval |

Legacy `CLAUDE.md` is a redirect stub → `DEV.md`.

### New harness maintainer (fresh Cursor agent)

Improve the **verification loop and agent**, not generated `games/*.html`. Start here:

1. **[`HARNESS_TUNING.md`](HARNESS_TUNING.md) § “New agent — harness improvement”** — read order, harness vs memory, fix loop, do-not list
2. **[`HARNESS_DEBUG.md`](HARNESS_DEBUG.md)** — gates, traces, when TEST OK lies
3. **[`TEST.md`](TEST.md)** — pytest map; extend existing test file for your failure class
4. Run **`.venv/bin/python -m pytest tests/ -q`** after every change (must stay green)

| Layer | Edit when… |
|-------|------------|
| **Harness** | Injected loader (`assets.py`), browser gates (`tools.py`), loop mixins (`agent_*.py`) |
| **Memory** | Game-type craft the LLM should learn via retrieval (`memory/*.jsonl`, skeletons) |

Trace → fix loop: **§4** below. Batch commands: **`eval/OPERATIONS.md`**.

---

## 1. Source tree — safe to edit (commit to git)

| Area | Paths |
|------|--------|
| Agent loop | `agent.py` + mixins (`agent_helpers.py`, `agent_feedback.py`, `agent_prompts.py`, `agent_compaction.py`, `agent_stream.py`, `agent_gates.py`, `agent_critic.py`, `agent_assets.py`, `agent_probes.py`, `agent_memory.py`), `chat.py`, `coder.py`, `patches.py`, `prompts_v1.py`, `memory.py` |
| Verification | `tools.py`, `vlm_critic.py`, `modality.py` (genre-free rendering-shape detectors) |
| Assets / media | `assets.py`, `sounds.py`, `videos.py` |
| Memory data | `memory/*.jsonl`, `memory/skeletons/` |
| Eval | `eval/*.py`, `eval/*.jsonl`, `eval/fixtures/`, `eval/*.txt` goal lists |
| Tests / scripts | `tests/`, `scripts/`, `system_tests.py` |
| Curated samples | `goodgame/` (promoted via TUI `/goodgame`) |

Do **not** patch random `games/*.html` to fix the agent — change harness/memory and re-run.

---

## 1b. Agent module map (incremental split)

**Rule:** add new logic to the matching mixin — do not grow `run()` inline.

`agent.py` still holds phase orchestration (`_run_phase_a_and_first_build`, `_run_build_iterate_loop`,
`_run_exit_and_finalize`) — ~9.5K lines. Mixins own prefixed method groups below.

| Module | Concern |
|--------|---------|
| [`agent.py`](agent.py) | `GameAgent` shell, `run()` orchestration, public API, re-exports |
| [`agent_helpers.py`](agent_helpers.py) | Pure helpers: seed media scan, HTML normalize/extract regexes, compaction constants |
| [`agent_feedback.py`](agent_feedback.py) | Feedback routing, scope locks, `_apply_scoped_check_to_report` |
| [`agent_prompts.py`](agent_prompts.py) | `_build_fix_prompt`, structured summary, seed HTML for prompts |
| [`agent_compaction.py`](agent_compaction.py) | `_prune_messages`, continuation context reset |
| [`agent_stream.py`](agent_stream.py) | HTML extract, format doctor, stall/repetition recovery |
| [`agent_probes.py`](agent_probes.py) | Probe quarantine, impossible-probe downgrade |
| [`agent_memory.py`](agent_memory.py) | Opening-book / components / lean budget retrieval |
| [`agent_gates.py`](agent_gates.py) | Report post-processing gates (`_apply_*_to_report` except scoped check) |
| [`agent_critic.py`](agent_critic.py) | VLM / visual playtest / autonomous playtest |
| [`agent_assets.py`](agent_assets.py) | Mid-session asset/sound generation and alignment |

### `run()` section index

Thin `run()` delegates to phase methods. Grep helpers: `GameAgent.run_loop_inspect_source()`.

| Method | Section |
|--------|---------|
| `_run_phase_a_and_first_build` | Phase A planning, seed/skeleton, first-build assets |
| `_run_build_iterate_loop` | Phase B iter: stream → materialize → test → next turn |
| `_run_exit_and_finalize` | Exit decision, critique, final iter test, outcome |

### Method groups on `GameAgent`

| Prefix | Examples | Target mixin |
|--------|----------|--------------|
| `_build_*` | `_build_fix_prompt`, `_build_structured_summary` | `agent_prompts.py` |
| `_apply_*` (gates) | `_apply_undrawn_art_intent_gate` | `agent_gates.py` |
| `_apply_scoped_check_to_report` | scope-lock on test report | `agent_feedback.py` |
| `_feedback_*` / `_route_*` | `_parse_feedback_route_json`, `_route_user_feedback_llm` | `agent_feedback.py` |
| `_prune_*` / `_maybe_reset_*` | `_prune_messages`, `_maybe_reset_continuation_context` | `agent_compaction.py` |
| `_extract_*` / `_run_format_*` | `_extract_html`, `_run_format_doctor` | `agent_stream.py` |
| `_run_autonomous_*` / `_queue_visual_*` | visual playtest, VLM critic | `agent_critic.py` |
| `_maybe_generate_*` | mid-session assets/sounds | `agent_assets.py` |

---

## 2. Artifact tree — read for triage, never edit as source, never commit

Everything under `games/` is **generated at runtime**. Gitignored locally; stays on disk for debugging.

### Trace and log paths

| Run type | Trace JSONL | Batch log | Monitor |
|----------|-------------|-----------|---------|
| TUI / one-shot `coder.py` | `games/traces/<artifact_id>.jsonl` | optional `games/traces/*.log` | — |
| Serial tune (`run_XX`) | `games/tune_serial10/run_XX/traces/<label>__run_*.jsonl` | `games/tune_serial10/run_XX/overnight.log` | `games/tune_serial10/run_XX/agent_monitor.json` |
| Checkpoints | — | — | `tune_summary.json`, `tune_checkpoint.json` in same run dir |

**Rule:** traces live at `{html_out_dir}/traces/`.

Per-run scratch notes: `games/tune_serial10/run_XX/triage.md` (gitignored). Copy durable learnings into `HARNESS_TUNING.md` before wiping runs.

### Triage tools

```bash
# TUI / one-shot — substring id under games/traces/ only:
.venv/bin/python scripts/enrich_trace.py <session-id> --timeline

# Tune batch — full path required (or label stem; see enrich_trace --help):
.venv/bin/python scripts/enrich_trace.py games/tune_serial10/run_XX/traces/01_label__run_....jsonl --timeline

.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_XX
tail -f games/tune_serial10/run_XX/overnight.log
```

---

## 3. Curated games — read-only samples

- **`goodgame/`** only — hand-promoted wins, tracked in git.
- Session output under `games/` is disposable; use `/goodgame` to preserve a winner.

---

## 4. Which doc when + trace → fix loop

| Task | Read |
|------|------|
| Commands, env, architecture | `DEV.md` |
| **Tests, scripts, what each suite guards** | **`TEST.md`** |
| Tune harness / traps / batch learnings | `HARNESS_TUNING.md` |
| Trace grep workflow | `HARNESS_DEBUG.md` |
| Serial / overnight eval workflow | **`eval/OPERATIONS.md`** · `eval/PARALLEL_MLX_TESTING.md` |
| Human onboarding | `README.md` |

### Trace evidence → code (canonical loop)

1. **Timeline:** `enrich_trace.py <full-path-or-stem> --timeline`
2. **Classify:** `failure_class` on failed `iter_summary` (`HARNESS_DEBUG.md`)
3. **Route:**

| `failure_class` | First place to edit |
|-----------------|---------------------|
| `harness_bug` | `tools.py` gates · `agent_*.py` loop wiring |
| `memory_gap` | `memory/playbook.jsonl`, skeletons, outlines |
| `local_llm_limit` | `prompts_v1.py`, `agent_compaction.py`, `backend.py` sampling |

4. **Verify:** targeted pytest from `TEST.md` / `eval/OPERATIONS.md`
5. **Persist:** durable trap → `HARNESS_TUNING.md` + optional playbook bullet; ephemeral notes → `triage.md` then OPERATIONS snapshot

**Never** edit `games/*.html` as source — tune harness/memory and re-run.

---

## 5. Cleanup

`./scripts/clean_artifacts.sh` wipes stale `games/` output (including `games/tune_serial10/`). **Never** touches `goodgame/` or `memory/playbook.jsonl`. Do not run during an active tune batch.
