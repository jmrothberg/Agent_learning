# AGENTS.md — source vs artifacts

Map for maintainers and LLMs: **edit source, read artifacts for triage, never commit generated games.**

Full commands and env vars → [`CLAUDE.md`](CLAUDE.md). Harness tuning traps → [`FOR_NEXT_LLM.md`](FOR_NEXT_LLM.md). Trace debug → [`HARNESS_DEBUG.md`](HARNESS_DEBUG.md).

---

## 1. Source tree — safe to edit (commit to git)

| Area | Paths |
|------|--------|
| Agent loop | `agent.py`, `chat.py`, `coder.py`, `patches.py`, `prompts_v1.py`, `memory.py` |
| Verification | `tools.py`, `vlm_critic.py` |
| Assets / media | `assets.py`, `sounds.py`, `videos.py` |
| Memory data | `memory/*.jsonl`, `memory/skeletons/` |
| Eval | `eval/*.py`, `eval/*.jsonl`, `eval/fixtures/`, `eval/*.txt` goal lists |
| Tests / scripts | `tests/`, `scripts/`, `system_tests.py` |
| Curated samples | `goodgame/` (promoted via TUI `/goodgame`) |

Do **not** patch random `games/*.html` to fix the agent — change harness/memory and re-run.

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
| Tune harness / traps / batch learnings | `FOR_NEXT_LLM.md` |
| Trace grep workflow | `HARNESS_DEBUG.md` |
| Serial / overnight eval workflow | `eval/PARALLEL_MLX_TESTING.md` |
| Human onboarding | `README.md` |

---

## 5. Cleanup

`./scripts/clean_artifacts.sh` wipes stale `games/` output (including `games/tune_serial10/`). **Never** touches `goodgame/` or `memory/playbook.jsonl`. Do not run during an active tune batch.
