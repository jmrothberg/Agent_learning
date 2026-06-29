# AGENTS.md — source vs artifacts

Map for maintainers and LLMs: **edit source, read artifacts for triage, never commit generated games.**

Full commands and env vars → [`CLAUDE.md`](CLAUDE.md). Harness tuning traps → [`FOR_NEXT_LLM.md`](FOR_NEXT_LLM.md). Trace debug → [`HARNESS_DEBUG.md`](HARNESS_DEBUG.md).

---

## 1. Source tree — safe to edit (commit to git)

| Area | Paths |
|------|--------|
| Agent loop | `agent.py` + mixins (`agent_helpers.py`, `agent_feedback.py`, `agent_prompts.py`, `agent_compaction.py`, `agent_stream.py`, `agent_gates.py`, `agent_critic.py`, `agent_assets.py`, `agent_probes.py`, `agent_memory.py`), `chat.py`, `coder.py`, `patches.py`, `prompts_v1.py`, `memory.py` |
| Verification | `tools.py`, `vlm_critic.py` |
| Assets / media | `assets.py`, `sounds.py`, `videos.py` |
| Memory data | `memory/*.jsonl`, `memory/skeletons/` |
| Eval | `eval/*.py`, `eval/*.jsonl`, `eval/fixtures/`, `eval/*.txt` goal lists |
| Tests / scripts | `tests/`, `scripts/`, `system_tests.py` |
| Curated samples | `goodgame/` (promoted via TUI `/goodgame`) |

Do **not** patch random `games/*.html` to fix the agent — change harness/memory and re-run.

---

## 1b. Agent module map (incremental split)

**Rule:** add new logic to the matching mixin — do not grow `run()` inline.

| Module | Concern |
|--------|---------|
| [`agent.py`](agent.py) | `GameAgent` shell, `run()` orchestration, public API, re-exports |
| [`agent_helpers.py`](agent_helpers.py) | Pure helpers: seed media scan, HTML normalize/extract regexes, compaction constants |
| [`agent_feedback.py`](agent_feedback.py) | Feedback routing classifiers, scope locks, blocker-first deferral |
| [`agent_prompts.py`](agent_prompts.py) | `_build_fix_prompt`, structured summary, seed HTML for prompts |
| [`agent_compaction.py`](agent_compaction.py) | `_prune_messages`, continuation context reset |
| [`agent_stream.py`](agent_stream.py) | HTML extract, format doctor, stall/repetition recovery |
| [`agent_probes.py`](agent_probes.py) | Probe quarantine, impossible-probe downgrade |
| [`agent_memory.py`](agent_memory.py) | Opening-book / components / lean budget retrieval |
| [`agent_gates.py`](agent_gates.py) | Report post-processing gates (`_apply_*_to_report`) |
| [`agent_critic.py`](agent_critic.py) | VLM / visual playtest / autonomous playtest |
| [`agent_assets.py`](agent_assets.py) | Mid-session asset/sound generation and alignment |

### `run()` section index

Thin `run()` delegates to phase methods. Grep helpers: `GameAgent.run_loop_inspect_source()`.

| Method | Section |
|--------|---------|
| `_run_phase_a_and_first_build` | Phase A planning, seed/skeleton, first-build assets |
| `_run_build_iterate_loop` | Phase B iter: stream → materialize → test → next turn |
| `_run_exit_and_finalize` | Exit decision, critique, final iter test, outcome |

Section markers inside `_run_build_iterate_loop` match the `# ----` comments in `agent.py` (mid-session assets, diagnose, materialize, test, self-critique, etc.).

### Method groups on `GameAgent`

| Prefix | Examples | Target mixin |
|--------|----------|--------------|
| `_build_*` | `_build_fix_prompt`, `_build_structured_summary` | `agent_prompts.py` |
| `_apply_*` | `_apply_undrawn_art_intent_gate`, `_apply_scoped_check_to_report` | `agent_gates.py` |
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

**Rule:** traces live at `{html_out_dir}/traces/` — for tune batches, HTML is `games/tune_serial10/run_XX/<label>.html`, so traces nest under that run folder (not top-level `games/traces/`).

Per-run scratch notes: `games/tune_serial10/run_XX/triage.md` (gitignored). Copy durable learnings into `FOR_NEXT_LLM.md` before wiping runs.

### Triage tools

```bash
.venv/bin/python scripts/enrich_trace.py <session-id> --timeline
.venv/bin/python eval/tune_overnight_monitor.py --out-dir games/tune_serial10/run_XX
tail -f games/tune_serial10/run_XX/overnight.log
```

In Cursor: `@games/tune_serial10/run_XX/traces/01_...jsonl` — traces are **not** in `.cursorignore`.

---

## 3. Curated games — read-only samples

- **`goodgame/`** only — hand-promoted wins, tracked in git.
- Session output under `games/` is disposable; use `/goodgame` to preserve a winner.

---

## 4. Which doc when

| Task | Read |
|------|------|
| Commands, env, architecture | `CLAUDE.md` |
| **Tests, scripts, what each suite guards** | **`TEST.md`** |
| Tune harness / traps / batch learnings | `FOR_NEXT_LLM.md` |
| Trace grep workflow | `HARNESS_DEBUG.md` |
| Serial / overnight eval workflow | **`eval/OPERATIONS.md`** (start here) · `eval/PARALLEL_MLX_TESTING.md` |
| Human onboarding | `README.md` |

---

## 5. Cleanup

`./scripts/clean_artifacts.sh` wipes stale `games/` output (including `games/tune_serial10/`). **Never** touches `goodgame/` or `memory/playbook.jsonl`. Do not run during an active tune batch.
