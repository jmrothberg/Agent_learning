# fine_tunning — LoRA training system

This folder has everything needed to train, monitor, and use a LoRA adapter on a local MLX model: the code, the start script, and this guide. The first project is the **HTML-game LoRA** on `Qwen3.8-27B-mxfp8`. The same code trains other LoRAs by pointing it at a different project folder, as described below.

- **Code (in git):** this folder, `Agent_learning/fine_tunning/`.
- **Data, weights, and logs (not in git, large):** one project folder per LoRA under `~/MLX_Models/`, e.g. `~/MLX_Models/html_game_sft/`.
- **Base models (never modified):** `~/MLX_Models/<model>/`, e.g. `~/MLX_Models/Qwen3.8-27B-mxfp8/`.

> `sft/` in the repo root is the older copy of these scripts. `fine_tunning/` is the current one: it has the HOLD/RESUME dashboard and the env-var settings. The Sep 21–23 run was started from `~/MLX_Models/html_game_sft/scripts/`, which is the same code without the env vars. A restart with `./start.sh` uses this folder.

---

## 1. Quick start (no agent needed)

Run this in **Terminal.app**, not in a Cursor chat. The jobs run with `nohup`, so closing Terminal or quitting Cursor leaves them running.

```bash
cd ~/Agent_learning/fine_tunning
./start.sh            # progress page + trainer (HTML-game LoRA defaults)
```

Then open **http://127.0.0.1:8766/**. That page is the monitor.

| Command | Starts |
|---|---|
| `./start.sh` | monitor + trainer |
| `./start.sh all` | monitor + GitHub game downloader + trainer (HTML-game LoRA only) |
| `./start.sh monitor` | the progress page only. This is safe while a trainer is already running. |
| `./start.sh train` | the trainer only |
| `./start.sh ingest` | the downloader only |

A second trainer for the same project refuses to start (`trainer already running pid N` in `logs/supervisor.log`), so running `./start.sh` twice is harmless.

**Before you start:** quit `chat.py` if it has the 27B loaded. Training loads its own copy (about 98 GB peak), and two copies do not fit in 192 GB. Only one trainer can use the GPU at a time, even for different projects.

**Watch the raw log:**

```bash
tail -f ~/MLX_Models/html_game_sft/logs/train.log
```

**Stop:** press **HOLD** on the progress page. It stops `train_lora.py` and `run_slices.py` and frees the GPU. **RESUME** starts the trainer again, and it continues the saved adapter. Without the page:

```bash
pkill -f fine_tunning/run_slices.py; pkill -f fine_tunning/train_lora.py
pkill -f fine_tunning/serve_progress.py     # the monitor, if wanted
```

(`pkill -f` stops processes whose command line matches that path. Nothing else is touched.)

---

## 2. What each file does

| File | Job |
|---|---|
| `start.sh` | Sets the env vars and launches the jobs in the background |
| `run_slices.py` | Supervisor loop. It builds `jsonl/train.jsonl`, runs `train_lora.py` for one slice (30 min), snapshots the adapter, and repeats. It always resumes the existing adapter. |
| `train_lora.py` | One training slice: loads the base with mlx-vlm, applies the LoRA, trains, and saves `adapters/adapters.safetensors` |
| `serve_progress.py` + `progress.html` | Monitor page: loss plot, rows, checkpoints, HOLD/RESUME |
| `rows.py` | Turns one HTML file into one training row in agent format (HTML-game project) |
| `ingest.py` | Overnight downloader. It searches GitHub for HTML/JS games, clones them, and appends rows to `jsonl/added.jsonl` (HTML-game project) |
| `rebuild_corpus.py` | Rebuilds `jsonl/html_corpus.jsonl` from every HTML file on disk. Run it only while the trainer is stopped. |

---

## 3. Project folder layout (`$LORA_ROOT`)

Every LoRA project uses the same layout. The scripts create missing folders.

| Path | What |
|---|---|
| `jsonl/html_corpus.jsonl` | Main row file (rebuilt by `rebuild_corpus.py`). For a new project, put your rows here. |
| `jsonl/added.jsonl` | Rows appended while training runs (`ingest.py`, or your own appends). They join at the next slice. |
| `jsonl/train.jsonl` | Written by `run_slices.py` before every slice (corpus + added, deduplicated by `sha`). Do not edit it. |
| `adapters/adapters.safetensors` | The live LoRA. `00000NN_adapters.safetensors` files are step checkpoints. |
| `adapters/adapter_config.json` | mlx-vlm adapter config (rank, scale, layer keys) |
| `snapshots/<YYYYMMDDTHHMMSSZ>/` | Copy of the adapter after each slice, plus a `READY` file (date, base model, path). The agent's `/lora` lists these, and nothing deletes them. |
| `logs/train.log` | Trainer output (`Iter N: Train loss …`) |
| `logs/state.json` | Live step / loss / it/s / peak GB (read by the monitor) |
| `logs/hold.json` | HOLD / RESUME flag written by the monitor |
| `logs/trainer.lock` | PID of the running supervisor |
| `logs/ingest.json`, `logs/rebuild.json`, `logs/seen_repos.txt` | Downloader / rebuild status (HTML project) |
| `games.sqlite` | sha256 index of files already turned into rows (HTML project) |
| `raw/` | Downloaded source files (HTML project) |

---

## 4. Training data — HTML-game LoRA (`~/MLX_Models/html_game_sft/`)

### Where it comes from

| Source | Path | How it got there |
|---|---|---|
| Canonical `/640png` games | `~/JMR-JS-CSS-FPGA-COMPUTER/storage/` | Hand-made 640×480 JMR games. These are the only **gold** rows. |
| Curated agent wins | `Agent_learning/goodgame/` | Promoted with TUI `/goodgame` |
| GitHub | `raw/github/<org>/<repo>/` (~5,200 repos) | `ingest.py`: shallow `git clone` of repos found by `gh search repos` (js13k, html5-game, canvas-game, three.js game, vanilla JS canvas game, one query per year 2013–2026), the `js13kGames` org, and a seed list (phaser examples, LittleJS, 2048, BrowserQuest, javascript-racer/breakout/pong, …). The repo list is in `logs/seen_repos.txt`. |
| Hugging Face | `raw/huggingface/html_game_gen/`, `raw/huggingface/the-stack-smol/` | Downloaded by hand before `ingest.py` existed. They are walked like any other folder. |

There is no license filter and no count cap. A game is unique by the sha256 of its file bytes, indexed in `games.sqlite`.

### How a file becomes a row (`rows.py`)

1. **Walk** every `.html`/`.htm` file (skipping `.git`, `node_modules`, `dist`, `vendor`).
2. **Is it a game?** (`looks_like_game`) The file must have a canvas, `requestAnimationFrame`, an engine name, or a script tag. Typedoc/JSDoc pages are skipped.
3. **Gold `/640png` rows** come from files under `JMR_STORAGE`. System prompt = `prompts_v1.build_system_prompt(goal, jmr_png_mode=True)` (~2,200 tokens). User goal = `Build a one-screen game: <title>. <TARGET=/640png footer>`.
4. **General HTML rows** are every other file. They must be one vanilla HTML+JS page (`chrome_vanilla_html`), and pages using Phaser, Pixi, Kaboom, melonJS, Babylon, A-Frame, PlayCanvas, Crafty, Kontra, LittleJS, React, Vue, Angular, PHP, or Python are rejected. three.js is allowed. Local `<script src="x.js">` files (<400 KB) are inlined, so the label is one file. System prompt = `build_system_prompt(goal)` (the full agent prompt). User goal = `Build a browser HTML5 game: <title>.`
5. **Assistant** = `<think>\n<one-line note>\n</think>\n<html_file>\n<the page>\n</html_file>\n`, which is exactly the format the agent parses.
6. **Token budget:** 8,128 tokens per row (Qwen tokenizer from the base folder). Longer pages are split into pieces. A file that cannot fit is recorded as `unfitted` and gets no row.

At training time, `train_lora.py` changes the rows further (the defaults, see §6):
- `--system short` swaps the ~6,100-token agent prompt on `html` rows for a 60-token prompt. That prompt gets no loss anyway, so each step is ~4× cheaper. Gold rows keep their contract.
- `--whole-games 1` glues a source's pieces back together and keeps only complete pages (`<!doctype…</html>`) that fit in 8k tokens. Mid-file fragments are dropped.
- Only the assistant turn is trained (`train_on_completions=True`).

`run_slices.py` puts rows it recognizes as games first (gold, goodgame, then canvas+loop or three.js pages) and holds back the rest (`LORA_GAME_FILTER=1`).

### Size (Sep 23, 2026)

| File | Size | Notes |
|---|---|---|
| `jsonl/html_corpus.jsonl` | 2.9 GB, 92,411 rows | Rebuild Sep 21: 16,587 files walked → 246 gold + 92,165 html rows, 8,497 unfitted |
| `jsonl/added.jsonl` | 4.5 GB | Overnight `ingest.py` additions |
| `jsonl/train.jsonl` | 0.8 GB | Assembled each slice (~235K rows counted by the monitor) |
| `games.sqlite` | 79 MB | ~246K unique hashes |
| `jsonl/seed.jsonl`, `also.jsonl`, `incoming.jsonl` | small | Older plan-card rows (`kind: plan`). The current trainer does not read them. |

### Row format (JSONL, one object per line)

```json
{
  "sha": "sha256 of path:part:page",
  "kind": "gold | html",
  "title": "Centipede Arcade",
  "source": "/abs/path/to/the/file.html",
  "messages": [
    {"role": "system", "content": "…agent system prompt…"},
    {"role": "user", "content": "Build a browser HTML5 game: Centipede Arcade."},
    {"role": "assistant", "content": "<think>\n…\n</think>\n<html_file>\n<!doctype html>…</html>\n</html_file>\n"}
  ]
}
```

Only `messages` is required for training. `sha` deduplicates rows (last write wins). `kind` and `source` are used by the HTML-only options `--system short` and `--whole-games`.

### Rebuild the corpus (trainer stopped)

```bash
cd ~/MLX_Models/html_game_sft
~/Agents/.venv/bin/python ~/Agent_learning/fine_tunning/rebuild_corpus.py
# walks storage/, goodgame/, raw/ → jsonl/html_corpus.jsonl (and resets games.sqlite)
```

---

## 5. Training on any (Qwen-style) model / a new LoRA

Settings are env vars, all read by `start.sh`, `run_slices.py`, `train_lora.py`, `rows.py`, and `serve_progress.py`:

| Env var | Default | Meaning |
|---|---|---|
| `LORA_ROOT` | `~/MLX_Models/html_game_sft` | Project folder (data, adapters, snapshots, logs) |
| `LORA_BASE` | `~/MLX_Models/Qwen3.8-27B-mxfp8` | Base model folder. Opened read-only. |
| `LORA_PY` | `~/Agents/.venv/bin/python` | Python with `mlx`, `mlx-vlm`, `datasets`, `tokenizers` |
| `LORA_PORT` | `8766` | Monitor port. Give each project its own. (8765 is chat.py Asset Studio.) |
| `LORA_TRAIN_ARGS` | *(empty)* | Extra `train_lora.py` flags, appended to every slice |
| `LORA_GAME_FILTER` | `1` | `0` = train on every row (use this for non-game data) |
| `LORA_SLICE_MIN` | `30` | Minutes per slice. A snapshot is saved after each one. |

### What base models work

`train_lora.py` loads the base with **mlx-vlm** (`mlx_vlm.utils.load`), and the agent applies `MLX_ADAPTER` only on its mlx-vlm path. So the base must be a model that **mlx-vlm can load**: Qwen3.x / Qwen2.5-VL style multimodal checkpoints in MLX format (quantized is fine; `--dequantize 1` unpacks the frozen base to bf16 in memory for speed). The vision tower is frozen and not trained.

- `--lora-top-layers N` expects Qwen-VL layer naming (`model.language_model.model.layers`). For other layouts, pass `--lora-top-layers 0` (train all layers).
- `--dequantize 1` needs about 2 bytes per parameter of free memory (27B ≈ 54 GB). For a big base on a smaller machine, use `--dequantize 0`.
- **Text-only models** (loaded by mlx-lm, not mlx-vlm) cannot use `train_lora.py`. Train them with `mlx_lm` directly: `python -m mlx_lm lora --model <base> --train --data <folder with train.jsonl> --iters N --adapter-path <out>`. The row format above (`messages`) works there too. The agent's `/lora` does not load text-only adapters.

### Recipe: a new LoRA project

```bash
# 1) project folder with your rows (format in §4; only "messages" is required)
mkdir -p ~/MLX_Models/my_lora/jsonl
cp my_rows.jsonl ~/MLX_Models/my_lora/jsonl/html_corpus.jsonl

# 2) start it. For non-HTML data turn off the game filter and the HTML-only row changes.
cd ~/Agent_learning/fine_tunning
LORA_ROOT=~/MLX_Models/my_lora \
LORA_BASE=~/MLX_Models/Qwen3.8-27B-mxfp8 \
LORA_PORT=8767 \
LORA_GAME_FILTER=0 \
LORA_TRAIN_ARGS="--system full --whole-games 0" \
./start.sh
# monitor: http://127.0.0.1:8767/
```

To add rows while it runs, append lines to `~/MLX_Models/my_lora/jsonl/added.jsonl`. They join at the next slice. The HTML-only counters on the page (games, gold files) stay at 0 for other projects, but loss, steps, and checkpoints work.

**Starting over vs continuing:** the trainer always resumes `adapters/adapters.safetensors` if that file exists. To start a fresh LoRA, use a new `LORA_ROOT`. Do not delete the old adapter.

**Changing base models:** an adapter only fits the base it was trained on. Each snapshot's `READY` file records `MLX_MODEL=`. Use a new `LORA_ROOT` for a new base.

---

## 6. `train_lora.py` flags

| Flag | Default | Effect |
|---|---|---|
| `--jsonl PATH` | (required; `run_slices.py` passes `jsonl/train.jsonl`) | Rows |
| `--iters N` | 20 (`run_slices.py` sizes it to the slice length) | Steps this slice |
| `--max-seq-length` | 8192 | Longer rows are truncated |
| `--adapter-path DIR` | `adapters/` when resuming | Resume this adapter |
| `--learning-rate` | 5e-5 | Adam LR (1e-5 before Sep 23) |
| `--system short\|full` | short | `short` = 60-token prompt on `html` rows. Use `full` for non-HTML data. |
| `--whole-games 1\|0` | 1 | Rejoin split HTML pieces, whole pages only. Use `0` for non-HTML data. |
| `--dequantize 1\|0` | 1 | bf16 base in memory for dense GEMM (same values) |
| `--lora-top-layers N` | 32 | Train only the top N decoder layers. Lower LoRA is kept and saved. `0` = all. |
| `--grad-checkpoint 1\|0` | 1 | Needed on the 27B (off runs out of memory) |
| `--batch-size` | 1 | 2 gave no speedup on the 27B |
| `--steps-per-save` | 10 | Step checkpoints in `adapters/` |

The LoRA shape is fixed in code: rank 16, alpha 32, dropout 0, language model only. For the old slow behavior: `--system full --whole-games 0 --dequantize 0 --lora-top-layers 0 --learning-rate 1e-5`.

Speed on the 27B (M-series, 192 GB): ~15 s per game with the defaults (was 143 s), peak ~98 GB.

---

## 7. Using a trained LoRA in the agent (`chat.py`)

Stop training first (HOLD on the monitor) so the GPU is free. Then in the TUI:

```text
/model          pick the base the LoRA was trained on (e.g. Qwen3.8-27B-mxfp8)
/server off     the adapter loads in-process only
/lora           list every snapshot of every project as project/stamp
/lora latest                    newest dated snapshot across all projects
/lora latest html_game_sft      newest snapshot of one project
/lora 12        pick by number     /lora html_game_sft/20260923T181017Z   by name
/lora off       base model only
/640png         only for a 640×480 sheet game (HTML-game LoRA)
/new your game idea
```

`/lora` finds every `~/MLX_Models/<project>/snapshots/` folder, so a new project shows up without code changes. On selection it prints the base model from the snapshot's `READY` file. `/status` and the header show the LoRA as `project/stamp`. Press RESUME on the monitor afterwards. It quits the agent's model and continues training.

Scripting/eval without the TUI: `MLX_MODEL=<base> MLX_ADAPTER=<snapshot dir>` (the `READY` file has both lines).

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Page says `waiting_for_seed` forever | The project has no rows. Put rows in `jsonl/html_corpus.jsonl` or `jsonl/added.jsonl`. |
| `trainer already running pid N` | One is already up. Check the page, or `cat $LORA_ROOT/logs/trainer.lock`. |
| Out of memory at load | `chat.py` or another trainer holds a model. HOLD / quit it. |
| `KeyError: 'source'` or `whole games 0 of N sources` in `train.log` | Non-HTML data with `--whole-games 1`. Pass `LORA_TRAIN_ARGS="--system full --whole-games 0"`. |
| `AttributeError … language_model` | The base is not Qwen-VL shaped. Pass `--lora-top-layers 0`. |
| `No Metal device available` | Running inside a sandbox / headless shell. Use Terminal.app. |
