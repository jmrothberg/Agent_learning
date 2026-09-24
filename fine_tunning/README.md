# fine_tunning — LoRA training system

This folder has everything needed to train, monitor, and use a LoRA adapter on a local MLX model: the code, the start script, and this guide. The first project is the **HTML-game LoRA** on `Qwen3.8-27B-mxfp8`. The same code trains other LoRAs by pointing it at a different project folder, as described below.

- **Code (in git):** this folder, `Agent_learning/fine_tunning/`.
- **Data, weights, and logs (not in git, large):** one project folder per LoRA under `~/MLX_Models/`, e.g. `~/MLX_Models/html_game_sft/`.
- **Base models (never modified):** `~/MLX_Models/<model>/`, e.g. `~/MLX_Models/Qwen3.8-27B-mxfp8/`.

> **Small HTML/JS model** (full fine-tune of a ~1B base, no LoRA): code in `small/`, data in `~/MLX_Models/html_js_small/`, start with `./start.sh small`. Everything about it, including moving it to another Mac and trying another base model, is in **§9**.

> `sft/` in the repo root is the older copy of these scripts. `fine_tunning/` is the current one: it has the HOLD/RESUME dashboard and the env-var settings. The Sep 21–23 run was started from `~/MLX_Models/html_game_sft/scripts/`, which is the same code without the env vars. A restart with `./start.sh` uses this folder.

---

## 1. Quick start (no agent needed)

`./start.sh` detaches each job to launchd (PPID 1), so closing Terminal or quitting Cursor leaves it running.

```bash
cd ~/Agent_learning/fine_tunning
./start.sh            # progress page + trainer (HTML-game LoRA defaults)
# Small HTML/JS full fine-tune (MiniCPM5-1B) — data must already be on disk (§9):
./start.sh small      # → http://127.0.0.1:8767/
```

Then open **http://127.0.0.1:8766/** (LoRA) or **http://127.0.0.1:8767/** (small model). That page is the monitor.

| Command | Starts |
|---|---|
| `./start.sh` | monitor + trainer |
| `./start.sh all` | monitor + GitHub game downloader + trainer (HTML-game LoRA only) |
| `./start.sh monitor` | the progress page only. This is safe while a trainer is already running. |
| `./start.sh train` | the trainer only |
| `./start.sh ingest` | the downloader only |
| `./start.sh small` | **small HTML/JS model:** monitor + full fine-tune (§9) → :8767 |

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

---

## 9. Small HTML/JS model (full fine-tune, `small/`)

A ~1B model trained only on HTML and JavaScript. **Every weight is trained (not LoRA).** Stage 1 is continued pretraining on packed 4,096-token blocks, with loss on every token. Stage 2 (spec → whole game, planned) teaches it the short spec prompt the agent will send.

| | |
|---|---|
| Base (first run) | `openbmb/MiniCPM5-1B-Base` (Apache 2.0, 2026), loaded by **mlx-lm** |
| Base (M3 Ultra run) | `openbmb/MiniCPM5-2B-Base` — same tokenizer as the 1B shards, so no retok. See §9f |
| Speed, M2 Ultra 192 GB | ~2,080 tokens/s, 16× the 27B LoRA (130 tokens/s), peak 57 GB |
| Speed, M3 Ultra 512 GB, 2B base | ~1,120 tokens/s at batch 2, peak 103 GB. Same 2B-token budget ≈ 22 days |
| Budget | 2B tokens = 61,035 updates ≈ 11 days on the M2 Ultra (1B) |
| Code (git) | `fine_tunning/small/` — **no training data in git** |
| Data + checkpoints (not in git) | `$SMALL_ROOT`, default `~/MLX_Models/html_js_small/` |

**Quick path after a git update:** fill a drive on the source Mac (§9a) → on the new Mac run the five commands in §9b → open http://127.0.0.1:8767/. To try a 0.5B instead, use §9d.

### 9a. Fill the drive (on the Mac that already has the data)

The training set is **not** in GitHub. Copy these from `~/MLX_Models/html_js_small/` onto an external drive (about **11.6 GB**). Replace `/Volumes/DRIVE` with your disk's mount name.

```bash
mkdir -p /Volumes/DRIVE/html_js_small
# Required — this is the training set:
rsync -a --progress ~/MLX_Models/html_js_small/shards /Volumes/DRIVE/html_js_small/
rsync -a --progress ~/MLX_Models/html_js_small/quality.sqlite /Volumes/DRIVE/html_js_small/
# Optional — only if you want to CONTINUE this exact run on the other Mac:
# rsync -a --progress ~/MLX_Models/html_js_small/checkpoints /Volumes/DRIVE/html_js_small/
```

| Copy? | Path | Size | Why |
|---|---|---|---|
| **yes** | `shards/` | 11 GB | Token ids + one JSON line per doc (`src`, `path`, `rank`, `norm`, …). Includes `shards/tokenizer.json`. |
| **yes** | `quality.sqlite` | 0.6 GB | Quality weights keyed by text hash (`norm`). Works with any tokenizer. |
| only to continue | `checkpoints/latest/` | 6.1 GB | Weights + optimizer + counters for this run |
| optional | `snapshots/` | 2 GB each | Dated weights-only copies every 6 hours |
| **no** | `~/MLX_Models/html_game_sft/raw/` (138 GB) | — | Source HTML/JS files. **Not needed** — the shards already hold the text. |
| **no** | `logs/` | — | Rebuilds on the new Mac |

### 9b. Start on another Mac (e.g. M3 Ultra 512 GB)

`./start.sh` detaches each job to launchd (PPID 1). Closing Terminal or quitting Cursor leaves it running.

```bash
# 1) code (clone once; later visits: cd ~/Agent_learning && git pull)
git clone https://github.com/jmrothberg/Agent_learning.git ~/Agent_learning
#    if the repo is already there:
#    cd ~/Agent_learning && git pull

# 2) python (scripts default to ~/Agents/.venv/bin/python — or set LORA_PY=...)
python3.12 -m venv ~/Agents/.venv
~/Agents/.venv/bin/pip install mlx mlx-lm tokenizers numpy pyarrow huggingface_hub torch transformers
# browser quality pass only (optional):
~/Agents/.venv/bin/pip install playwright && ~/Agents/.venv/bin/python -m playwright install chromium
export BROWSER_PY=~/Agents/.venv/bin/python   # else start.sh looks for ~/Agent_learning/.venv/bin/python

# 3) base model (~2 GB) — never modified by training
mkdir -p ~/MLX_Models
~/Agents/.venv/bin/hf download openbmb/MiniCPM5-1B-Base --local-dir ~/MLX_Models/MiniCPM5-1B-Base

# 4) training set from the drive (DRIVE = your disk mount name)
mkdir -p ~/MLX_Models/html_js_small
rsync -a --progress /Volumes/DRIVE/html_js_small/shards /Volumes/DRIVE/html_js_small/quality.sqlite \
  ~/MLX_Models/html_js_small/
#    to CONTINUE the other Mac's run instead of starting fresh, also copy checkpoints/:
#    rsync -a --progress /Volumes/DRIVE/html_js_small/checkpoints ~/MLX_Models/html_js_small/

# 5) start trainer + progress page
cd ~/Agent_learning/fine_tunning
chmod +x start.sh          # once, if needed
./start.sh small
```

Then open **http://127.0.0.1:8767/**. You should see tokens/s (~2,000+ on an Ultra), updates, and loss. Or:

```bash
tail -f ~/MLX_Models/html_js_small/logs/train.log
# expect lines like: Iter 10: Train loss … Tokens/sec 2080 …
```

**Train straight off a fast external SSD** (skip the copy into `~/MLX_Models`):

```bash
SMALL_ROOT=/Volumes/DRIVE/html_js_small ./start.sh small
```

**Do not continue the same run on both Macs at once.** They would diverge from the same checkpoint. To compare machines or settings, use a fresh `SMALL_ROOT` (or a different base — §9d).

| Command | Does |
|---|---|
| `./start.sh small` | monitor + trainer (starts, or resumes `checkpoints/latest/`) |
| `./start.sh small-train` / `small-monitor` | one of the two |
| `./start.sh small-data` | build shards from scratch (own games need `html_game_sft/games.sqlite` + `raw/`; github re-streams, ~20 min) |
| `./start.sh small-retok DIR` | re-tokenize `DIR/shards` for `SMALL_BASE` into `SMALL_ROOT/shards` |
| `./start.sh small-quality` | all quality passes, niced (score ~2 min; edu ~11 h; browser ~2 h) |
| `./start.sh small-stack` | The Stack v2 download. Resumes `logs/stack_state.json`. 32 workers, stop after 8B new tokens |

Env vars: `SMALL_ROOT` (data/checkpoints), `SMALL_BASE` (model), `SMALL_PORT` (page, default 8767), `SMALL_TRAIN_ARGS` (extra trainer flags), `LORA_PY`, `BROWSER_PY`.

**Stop:** `pkill -f train_small.py`. Up to 30 minutes since the last save are lost. The next `./start.sh small` resumes from `checkpoints/latest/`.

### 9c. Confirm it is running / common failures

| Check | Expect |
|---|---|
| http://127.0.0.1:8767/ | Title names the base; green tokens/s; updates counting |
| `tail ~/MLX_Models/html_js_small/logs/train.log` | `Iter N: … Tokens/sec …` every 10 updates |
| `pgrep -fl train_small` | one Python process |
| `ls ~/MLX_Models/html_js_small/shards/*.jsonl \| wc -l` | many shard files (not 0) |

| Symptom | Fix |
|---|---|
| `no model at SMALL_BASE` | Step 3 did not finish; re-run `hf download …` |
| `no shards in …/shards` | Step 4 missed `shards/`; re-rsync from the drive |
| `train_small.py already running` | A trainer is up; open the page or `pkill -f train_small.py` first |
| `No Metal device available` | Run in Terminal.app, not a sandboxed Cursor shell |
| Page blank / old LoRA curves | Wrong port — small model is **8767**, LoRA is 8766 |
| Out of memory | Quit any other GPU model (`chat.py`, another trainer). Or `SMALL_TRAIN_ARGS="--batch 1 --accum 8" ./start.sh small` |

### 9d. Train a different base model (e.g. a 0.5B) and compare

A smaller base (about 0.5B) trains about twice as fast and can learn this HTML/JS set well: short pages, canvas games, the patterns in the data. It will not match a 7B+ coder at long games, new mechanics, or fixing its own bugs. Use it as a fast HTML writer that the 27B agent prompts with a short spec. Compare it to the 1B with the same data and the same `--total-tokens`, then judge games in the browser, not by loss.

The shards are token ids of one tokenizer, so a new base gets its **own** `SMALL_ROOT` and a re-tokenized copy of the same data. Retok decodes and re-encodes (~15 min for 11 GB). Documents keep their original `norm`, so the same `quality.sqlite` applies. Use a **different port** so both progress pages can run.

```bash
M=Qwen-X-0.5B-Base                                   # any text model mlx-lm can load
~/Agents/.venv/bin/hf download <org>/$M --local-dir ~/MLX_Models/$M

# Smoke-load before spending hours:
~/Agents/.venv/bin/python -c "from mlx_lm import load; load('$HOME/MLX_Models/$M')"

cd ~/Agent_learning/fine_tunning
export SMALL_ROOT=~/MLX_Models/html_js_$M SMALL_BASE=~/MLX_Models/$M SMALL_PORT=8768
./start.sh small-retok ~/MLX_Models/html_js_small    # log: $SMALL_ROOT/logs/data.out
cp ~/MLX_Models/html_js_small/quality.sqlite $SMALL_ROOT/

# Optional speed check (saves nothing):
~/Agents/.venv/bin/python small/train_small.py --bench 20

./start.sh small                                     # monitor: http://127.0.0.1:8768/
```

- BOS/EOS come from the base's `config.json` (no BOS → EOS on both sides of each doc).
- Speed scales roughly with 1/parameters: a 0.5B should be about 2× the 1B. A large vocabulary (150k+) costs extra memory for the logits.
- **One GPU trainer at a time per Mac.** `start.sh` refuses a second `train_small.py`.
- **Comparing runs:** same data and same `--total-tokens`. **Do not compare raw loss across different tokenizers** (a token covers a different amount of text). Compare with the held-out game eval (`small/eval_small.py`, planned: 30 specs → headless Chromium), or at matched tokens on the same base.

### 9e. What the files do / how the data was built

| File | Job |
|---|---|
| `small/data.py` | Builds `shards/`. `--source own` = games from `html_game_sft/games.sqlite`; `--source gcc` = [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) HTML + JavaScript (streamed); `--source retok --from DIR` = re-tokenize for a new base. Filters size / minified / base64 / symbol soup / HTML without `<script>`. Resumes. |
| `small/train_small.py` | Trainer: weighted, deduplicated packing; compiled step; AdamW, warmup + cosine. Every 30 min saves `checkpoints/latest/` and re-reads `shards/` + `quality.sqlite`. Resumes automatically. |
| `small/quality_worker.py` | CPU quality scoring (niced). Writes `quality.sqlite` → `quality(norm, weight)`. |
| `start.sh small*` | One-line starts (§9b) |
| `serve_progress.py` + `progress.html` | Monitor; small-model root shows speed first |

Where the data came from: own games (`html_game_sft/raw/`) → 14,470 unique files (~33M tokens) after dedup; github-code-clean (49 parquet files) → 916,493 docs, 2.22B tokens. Sampling: own canvas/three.js with a loop ×3, other own ×2/×1; github ×1.5/×1/×0.7. `quality.sqlite` multiplies:

| Pass | Weight |
|---|---|
| Near-duplicate (MinHash, ~0.77 Jaccard), not the keeper | 0 (25% of docs) |
| JS syntax error (node `vm` parse, never run) | 0.3 |
| Repeated lines / "generated, do not edit" | 0.3 / 0.2 |
| Stack-Edu JavaScript classifier ([SmolLM2](https://arxiv.org/abs/2502.02737)), github only | score <1.5 → 0.3, <2.5 → 0.7, <3.5 → 1.3, else 2.0 |
| Headless Chromium, 2 s, network blocked | self-contained page with errors 0.5, canvas drew 1.5 |

### `train_small.py` flags

| Flag | Default | Effect |
|---|---|---|
| `--block` | 4096 | Tokens per packed sequence |
| `--batch` | 2 | Sequences per micro-step. On the M2, batch 1/2/4 were the same tokens/s (2 = 57 GB). On the M3 Ultra with the 2B, batch 4 and 8 were no faster and used 180 GB and 343 GB |
| `--accum` | 4 | Micro-steps per optimizer update (update = 32,768 tokens) |
| `--lr` / `--warmup` | 5e-5 / 200 | Peak LR, warmup updates; cosine to 10% after |
| `--total-tokens` | 2e9 | Stop after this many trained tokens |
| `--save-minutes` | 30 | Save `checkpoints/latest/` + re-read data and quality |
| `--bench N` | 0 | Time N micro-steps, save nothing |

### Use a checkpoint

`checkpoints/latest/` (and every `snapshots/<stamp>/`) is a normal mlx-lm model folder. Stage 1 is a base model, so use a raw prompt, not a chat template:

```bash
~/Agents/.venv/bin/python -m mlx_lm generate --model ~/MLX_Models/html_js_small/checkpoints/latest \
  --ignore-chat-template --max-tokens 400 --prompt '<!DOCTYPE html>
<html><head><title>Snake</title>'
```

### 9f. MiniCPM5-2B on the M3 Ultra (Sep 24, 2026)

`openbmb/MiniCPM5-2B-Base` is the newer base (Sep 7, 2026). It is a normal Llama, 2.52B parameters, bf16 weights about 4.7 GB. The 1B shards already use this tokenizer (same `tokenizer.json`), so train them in place. Do not retokenize.

| | |
|---|---|
| Weights | `~/MLX_Models/MiniCPM5-2B-Base` (never modified) |
| Data + checkpoints | `/Users/jonathanrothberg/Data/html_js_small` (`SMALL_ROOT`) |
| Page | http://127.0.0.1:8767/ |
| Measured speed | batch 2 → ~1,060–1,120 tokens/s, 103 GB. Batch 4 → ~1,110 tok/s, 180 GB. Batch 8 → ~990 tok/s, 343 GB. Stay at batch 2 |
| 2B-token budget | about 22 days at that speed |

8-bit and MXFP8 are not faster. MLX dequantizes them and then runs the bf16 GEMM, and those kernels are tuned for decoding. The 27B LoRA trainer already unpacks MXFP8 to bf16 for the same reason (`train_lora.py --dequantize`).

A throwaway speed test lives in `~/MLX_Models/html_js_MiniCPM5-2B-Base-bench`. Those shards are random tokens. Do not train on that folder.

`./start.sh` puts each job under launchd (PPID 1), in its own session. Quitting Cursor does not stop it. `nohup` alone does not, because a Cursor shell kills its process group.

```bash
cd ~/Agent_learning/fine_tunning
export SMALL_ROOT=/Users/jonathanrothberg/Data/html_js_small
export SMALL_BASE=~/MLX_Models/MiniCPM5-2B-Base
./start.sh small-monitor    # page first
./start.sh small-train
# After tokens/s is moving. This script uses 16 score workers, then edu 12 + browser 6.
# The Sep 24 run used 8, then edu 4 + browser 2, so the GPU stayed at ~1,120 tokens/s.
./start.sh small-quality
```

Node 25 aborts the syntax checker on some bad scripts (`Assertion failed: (end) >= (start)`), which used to kill the whole score pass. `quality_worker.py` restarts node and counts that script as bad JS. A checkpoint of this 2B run is much larger than the 1B's 6.1 GB: weights plus AdamW state are on the order of 25 GB.
