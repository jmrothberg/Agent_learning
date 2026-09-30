# fine_tunning — LoRA training system

> **Note to the other computers (Sep 30, 2026): remove Vulcan. Do not use it again.**
> Vulcan (`xlelords/vulcan`, request-to-page lessons) did not help. It is gone from this repo, code included, so nothing blocks it any more: a set that is not in `SET_WEIGHT` trains at weight 1. After `git pull`, press Stop training on the progress page, then run this. Press Resume when step 1 prints nothing.

```bash
export SMALL_ROOT=~/MLX_Models/html_js_small      # or that machine's data folder
# 1) list Vulcan files. Nothing printed = nothing to do.
find "$SMALL_ROOT/shards" "$SMALL_ROOT/sets" -name 'vulcan_*'
# 2) delete them (the .jsonl and .bin files listed above)
find "$SMALL_ROOT/shards" "$SMALL_ROOT/sets" -name 'vulcan_*' -delete
# 3) drop their rows from the quality database
sqlite3 "$SMALL_ROOT/quality.sqlite" ".timeout 60000" "BEGIN IMMEDIATE; \
  DELETE FROM browser WHERE norm IN (SELECT norm FROM feat WHERE src='vulcan'); \
  DELETE FROM done WHERE shard LIKE '%vulcan%'; \
  DELETE FROM feat WHERE src='vulcan'; COMMIT;"
rm -f "$SMALL_ROOT/logs/data_vulcan.json"
```

> **Synthetic games are in git.** `small/synth_games/synth.jsonl` holds every game that passed Chrome (660 on Sep 30, 2026). Git carries the games, not tokens. `small-synth` tokenizes them with `SMALL_BASE`'s `tokenizer.json` into `sets/synth/original`, `train`, and `tokens`. The 1B and the 2B share one tokenizer, so both Macs get the same token ids. Games already in `tokens/` (by `sha`) are skipped. Training can keep running; the next 30-minute save draws them. Details in §9b, "Synthetic games from git".

```bash
cd ~/Agent_learning/fine_tunning

# This Mac (runs synth_html.py, trains the 1B). Defaults: ~/MLX_Models/html_js_small + MiniCPM5-1B-Base.
./start.sh small-synth-share      # tokenize here, then commit + push the games

# The other Mac (trains the 2B), after git pull:
git pull
export SMALL_ROOT=/Users/jonathanrothberg/Data/html_js_small   # the 2B run (§9f)
export SMALL_BASE=~/MLX_Models/MiniCPM5-2B-Base
./start.sh small-synth
```

**Other Mac — small HTML/JS model.** One command. Copy `shards/` and `quality.sqlite` into `~/MLX_Models/html_js_small` first (not the raw HTML). If `~/Agents/.venv` or `~/MLX_Models/MiniCPM5-1B-Base` is missing, the command creates them. Jobs stay running if you quit Cursor.

```bash
cd ~/Agent_learning/fine_tunning
./start.sh small             # page, trainer, file check, code quality, browser. No Stack download.
./start.sh small continue    # the same, plus the Stack download (resumes logs/stack_state.json)
```

Run either line again any time. Anything already running is left alone, and anything that stopped is started. File check stays up and scores new shards as they appear. `small` does not start the download. A download that is already running keeps going until you stop it:

```bash
pkill -f "data.py --source stack"
```

**Other computers — clean the training data after you pull.** Good HTML and JavaScript only. The documents below are taken out of the shard files. They are not kept at a low weight. Checkpoints are not touched. Training can keep running. The next 30-minute save opens the cleaned shards. The loss will step up. That is the junk leaving, not a learning-rate blow-up. The page keeps the old `docs=` and `sampled_tokens=` numbers until the trainer is started again.

Set `SMALL_ROOT` to the data folder on that machine if it is not `~/MLX_Models/html_js_small`.

```bash
cd ~/Agent_learning/fine_tunning
git pull
export SMALL_ROOT=~/MLX_Models/html_js_small
~/Agents/.venv/bin/python small/quality_worker.py --strip
```

`--strip` rewrites `quality.sqlite` so those files are weight 0, then rewrites `shards/*.jsonl` and `sets/*/tokens/*.jsonl` without them.

| Throw out | How to recognize it | Old weight |
| --- | --- | --- |
| Near-duplicate, not the keeper | `quality.weight = 0` already | 0 |
| JavaScript that does not parse | `feat.syntax = 0` | 0.3 |
| Repeated lines, or "generated" / "do not edit" | `feat.junk < 1` | 0.2–0.3 |
| GitHub boilerplate | `edu.score < 1.5` (github only) | 0.3 |
| Page that errors in the headless browser | `browser.weight = 0.5` | 0.5 |

Keep a file when the script parses, it is not a repeated or generated dump, its Edu score is not under 1.5, the browser did not flag an error, and it is the keeper of its near-duplicate cluster. Plain HTML and JavaScript that passes those checks stays. A canvas game is not required.

**Try a prompt on the progress page.** Press Stop training. That frees the GPU and stays stopped until you press Resume (the 30-minute watcher will not start the trainer again). Type a prompt, pick how many new tokens, then press Generate on one side only. The left side is a model to compare against (this run's base, or another local model such as Qwen3.8-27B). The right side is a checkpoint (`checkpoints/latest` or a dated snapshot).

`./start.sh small` reloads the page when `serve_progress.py` or `progress.html` is newer than the process. It does not start the trainer while `logs/hold.json` says held. Resume on the page continues from `checkpoints/latest` at the learning rate already in `logs/train.log`.

This folder has everything needed to train, monitor, and use a LoRA adapter on a local MLX model: the code, the start script, and this guide. The first project is the **HTML-game LoRA** on `Qwen3.8-27B-mxfp8`. The same code trains other LoRAs by pointing it at a different project folder, as described below.

- **Code (in git):** this folder, `Agent_learning/fine_tunning/`.
- **Data, weights, and logs (not in git, large):** one project folder per LoRA under `~/MLX_Models/`, e.g. `~/MLX_Models/html_game_sft/`.
- **Base models (never modified):** `~/MLX_Models/<model>/`, e.g. `~/MLX_Models/Qwen3.8-27B-mxfp8/`.

> **Small HTML/JS model** (full fine-tune of a ~1B base, no LoRA): code in `small/`, data in `~/MLX_Models/html_js_small/`, start with `./start.sh small continue` (top of this file). Everything about it, including moving it to another Mac and trying another base model, is in **§9**.

> `fine_tunning/` is the trainer: it has the HOLD/RESUME dashboard and the env-var settings. The Sep 21–23 run was started from `~/MLX_Models/html_game_sft/scripts/`, which is the same code without the env vars. A restart with `./start.sh` uses this folder.

---



## 1. Quick start (no agent needed)

`./start.sh` detaches each job to launchd (PPID 1), so closing Terminal or quitting Cursor leaves it running.

Monitor: [http://127.0.0.1:8767/](http://127.0.0.1:8767/) (small HTML/JS model) · [http://127.0.0.1:8766/](http://127.0.0.1:8766/) (27B LoRA).

```bash
cd ~/Agent_learning/fine_tunning
./start.sh small
```


| Command                     | Starts                                                                   |
| --------------------------- | ------------------------------------------------------------------------ |
| `./start.sh`                | monitor + trainer                                                        |
| `./start.sh all`            | monitor + GitHub game downloader + trainer (HTML-game LoRA only)         |
| `./start.sh monitor`        | the progress page only. This is safe while a trainer is already running. |
| `./start.sh train`          | the trainer only                                                         |
| `./start.sh ingest`         | the downloader only                                                      |
| `./start.sh small`          | **small HTML/JS model:** page + trainer + file check + code quality + browser → :8767. No Stack download. |
| `./start.sh small continue` | same, plus the Stack download                                                                            |


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


| File                                  | Job                                                                                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `start.sh`                            | Sets the env vars and launches the jobs in the background                                                                                                                |
| `run_slices.py`                       | Supervisor loop. It builds `jsonl/train.jsonl`, runs `train_lora.py` for one slice (30 min), snapshots the adapter, and repeats. It always resumes the existing adapter. |
| `train_lora.py`                       | One training slice: loads the base with mlx-vlm, applies the LoRA, trains, and saves `adapters/adapters.safetensors`                                                     |
| `serve_progress.py` + `progress.html` | Monitor page: loss plot, rows, checkpoints, HOLD/RESUME                                                                                                                  |
| `rows.py`                             | Turns one HTML file into one training row in agent format (HTML-game project)                                                                                            |
| `ingest.py`                           | Overnight downloader. It searches GitHub for HTML/JS games, clones them, and appends rows to `jsonl/added.jsonl` (HTML-game project)                                     |
| `rebuild_corpus.py`                   | Rebuilds `jsonl/html_corpus.jsonl` from every HTML file on disk. Run it only while the trainer is stopped.                                                               |
| `synth_html.py` + `ideas_3000.txt`    | Asks Qwen3.8-27B on oMLX for one HTML file per prompt, keeps pages that pass headless Chrome, writes `jsonl/synth.jsonl`. Live stats at <http://127.0.0.1:8768/>. `expand_ideas.py` builds one prompt per game: the lines in `ideas_1000.txt`, plus one complete working version of each top title that is not already there. |


---



## 2b. Synthetic HTML set (Qwen3.8-27B on oMLX)

One short prompt in, one HTML file out. `--jobs` is how many streams start. The page can set 4 through 64, and the server cap is 64. This script does not start oMLX and does not load a second copy of the model. Stop the LoRA trainer first so the GPU is free, then load and pin `Qwen3.8-27B-mxfp8`.

```bash
cd ~/Agent_learning
MLX_SERVER_URL=http://127.0.0.1:8000 \
  .venv/bin/python fine_tunning/synth_html.py --jobs 48 --limit 3000
```

Restart with this. It stops only the generator. The model server on port 8000 keeps running. Saved rows stay. A reply still being written is dropped.

```bash
pkill -f '[f]ine_tunning/synth_html.py'
cd ~/Agent_learning
MLX_SERVER_URL=http://127.0.0.1:8000 \
  .venv/bin/python fine_tunning/synth_html.py --jobs 48 --limit 3000
```

That command is the start. Run the restart block to load prompt changes. `--jobs 48` is the starting stream count.

If the model server on port 8000 exits, this starts it again. If something is already listening there, it only watches. Log: `~/MLX_Models/html_game_sft/logs/mlx_server.log`.

```bash
cd ~/Agent_learning/fine_tunning
./start.sh server
```
 It reads the current `ideas_3000.txt` from the top. A game already stored in `synth.jsonl` or `buggy.jsonl` is skipped, including when the wording differs, so the run continues at the next line that is not done. A request timeout is not one of those. It goes back on the queue. The file is ordered with the longest instructions first: three.js, then 8-bit graphics, then Phaser, PixiJS, and Howler.js, then the plain lines.

Open <http://127.0.0.1:8768/>. The big number is aggregate decode tokens per second over the last 30 seconds. Prefill time is shown apart from that. Buttons 4, 8, 12, 16, 24, 32, 48, and 64 change how many streams run without a restart. The model server decodes that many only up to `MLX_VLM_MAX_NUM_SEQS` (started at 64, the top button). A button above that number waits in line until the server is restarted with a higher cap. The page has two scrolling lists, Working and Buggy. Click a row in either list, then Run. Up sets `weight` to 1. Down sets `weight` to 0 on that row. `train_lora.py` skips weight 0.

The model is told to skip thinking and explanations and to return one playable scene: a goal, controls, a drawn background in several colors, and a script split into state, input, update, and draw. It is told to finish that scene and then stop, with no second level. Generation ends at 7800 tokens. A saved row must fit in 8128. It is told not to write long tile maps or level strings. The model server runs with `MLX_VLM_TOKEN_QUEUE_TIMEOUT=10800` (default 600 s). At 600 s, about 47 streams made the server kill queued requests with an empty reply, which was more than half of `buggy.jsonl`. A server `{"error": …}` event is a request failure: the prompt goes back on the queue. A `buggy.jsonl` row with "no html" and an empty reply is retried on the next start. A reply that is still going at the cutoff is dropped, because the file never closes. The reply must start with `<html_file>` and end with `</html_file>`. The saved row still uses the short training system prompt, and `pack_row` adds the canned `<think>` line. A page is kept only when `tools.test_html_file` reports no console or page errors. That check waits three seconds for the game to boot, clicks the canvas, holds Enter and Space, then holds each arrow and WASD long enough for the update loop to see it, then waits two more seconds. An error after those inputs drops the game. Three.js requests load `https://unpkg.com/three@0.160.0/build/three.min.js` only, with a light and a ground, and they do not use `CapsuleGeometry` or r128. A three.js scene is at most twelve meshes, then the controls and the closing tags, so the file finishes inside the token cap. A stream that sends no bytes for 3 hours is stuck: that prompt goes back on the queue and is not written to `buggy.jsonl`. Silence shorter than that, including 48 streams waiting on other prefills, is not a timeout. A page that comes back and fails Chrome is not repaired and is not run again. It is written once to `$LORA_ROOT/jsonl/buggy.jsonl` in the same row shape as a training example, with an extra `bug` entry: the prompt, the code, and the Chrome report (the error text, and the line or stack when Chrome sent one). Kept rows go to `$LORA_ROOT/jsonl/synth.jsonl` (default `~/MLX_Models/html_game_sft`). They are not added to `added.jsonl` unless you pass `--into-added`. The prompt list is `ideas_3000.txt`. Each game or graphic is on one line. One hundred 3D games (Doom, Minecraft, and the others) also have a second line that says to use three.js. Games that are a better fit for another library get one extra line instead: Phaser for 2D action, PixiJS for simple sprite games, and Howler.js for rhythm and music games. A game is in at most one of those. One hundred pixel-art games say to use nice 8-bit graphics, as close to the original as possible. Openings vary (Make, Build, Create, and the others). The list includes one small working version of each of the top 1,000 acclaimed games, and it does not repeat a game already in `ideas_1000.txt`. `--limit 0` asks the 27B for still more prompts. A rerun skips a game that is already in `synth.jsonl` or `buggy.jsonl`, including when the wording differs. On the monitor, "saved earlier" is that count.

---



## 3. Project folder layout (`$LORA_ROOT`)

Every LoRA project uses the same layout. The scripts create missing folders.


| Path                                                           | What                                                                                                                                           |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `jsonl/html_corpus.jsonl`                                      | Main row file (rebuilt by `rebuild_corpus.py`). For a new project, put your rows here.                                                         |
| `jsonl/added.jsonl`                                            | Rows appended while training runs (`ingest.py`, or your own appends). They join at the next slice.                                             |
| `jsonl/train.jsonl`                                            | Written by `run_slices.py` before every slice (corpus + added, deduplicated by `sha`). Do not edit it.                                         |
| `adapters/adapters.safetensors`                                | The live LoRA. `00000NN_adapters.safetensors` files are step checkpoints.                                                                      |
| `adapters/adapter_config.json`                                 | mlx-vlm adapter config (rank, scale, layer keys)                                                                                               |
| `snapshots/<YYYYMMDDTHHMMSSZ>/`                                | Copy of the adapter after each slice, plus a `READY` file (date, base model, path). The agent's `/lora` lists these, and nothing deletes them. |
| `logs/train.log`                                               | Trainer output (`Iter N: Train loss …`)                                                                                                        |
| `logs/state.json`                                              | Live step / loss / it/s / peak GB (read by the monitor)                                                                                        |
| `logs/hold.json`                                               | HOLD / RESUME flag written by the monitor                                                                                                      |
| `logs/trainer.lock`                                            | PID of the running supervisor                                                                                                                  |
| `logs/ingest.json`, `logs/rebuild.json`, `logs/seen_repos.txt` | Downloader / rebuild status (HTML project)                                                                                                     |
| `games.sqlite`                                                 | sha256 index of files already turned into rows (HTML project)                                                                                  |
| `raw/`                                                         | Downloaded source files (HTML project)                                                                                                         |


---



## 4. Training data — HTML-game LoRA (`~/MLX_Models/html_game_sft/`)



### Where it comes from


| Source                    | Path                                                                | How it got there                                                                                                                                                                                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Canonical `/640png` games | `~/JMR-JS-CSS-FPGA-COMPUTER/storage/`                               | Hand-made 640×480 JMR games. These are the only **gold** rows.                                                                                                                                                                                                                                                                                   |
| Curated agent wins        | `Agent_learning/goodgame/`                                          | Promoted with TUI `/goodgame`                                                                                                                                                                                                                                                                                                                    |
| GitHub                    | `raw/github/<org>/<repo>/` (~5,200 repos)                           | `ingest.py`: shallow `git clone` of repos found by `gh search repos` (js13k, html5-game, canvas-game, three.js game, vanilla JS canvas game, one query per year 2013–2026), the `js13kGames` org, and a seed list (phaser examples, LittleJS, 2048, BrowserQuest, javascript-racer/breakout/pong, …). The repo list is in `logs/seen_repos.txt`. |
| Hugging Face              | `raw/huggingface/html_game_gen/`, `raw/huggingface/the-stack-smol/` | Downloaded by hand before `ingest.py` existed. They are walked like any other folder.                                                                                                                                                                                                                                                            |


There is no license filter and no count cap. A game is unique by the sha256 of its file bytes, indexed in `games.sqlite`.

### How a file becomes a row (`rows.py`)

1. **Walk** every `.html`/`.htm` file (skipping `.git`, `node_modules`, `dist`, `vendor`).
2. **Is it a game?** (`looks_like_game`) The file must have a canvas, `requestAnimationFrame`, an engine name, or a script tag. Typedoc/JSDoc pages are skipped.
3. **Gold** `/640png` **rows** come from files under `JMR_STORAGE`. System prompt = `prompts_v1.build_system_prompt(goal, jmr_png_mode=True)` (~2,200 tokens). User goal = `Build a one-screen game: <title>. <TARGET=/640png footer>`.
4. **General HTML rows** are every other file. They must be one vanilla HTML+JS page (`chrome_vanilla_html`), and pages using Phaser, Pixi, Kaboom, melonJS, Babylon, A-Frame, PlayCanvas, Crafty, Kontra, LittleJS, React, Vue, Angular, PHP, or Python are rejected. three.js is allowed. Local `<script src="x.js">` files (<400 KB) are inlined, so the label is one file. System prompt = `build_system_prompt(goal)` (the full agent prompt). User goal = `Build a browser HTML5 game: <title>.`
5. **Assistant** = `<think>\n<one-line note>\n</think>\n<html_file>\n<the page>\n</html_file>\n`, which is exactly the format the agent parses.
6. **Token budget:** 8,128 tokens per row (Qwen tokenizer from the base folder). Longer pages are split into pieces. A file that cannot fit is recorded as `unfitted` and gets no row.

At training time, `train_lora.py` changes the rows further (the defaults, see §6):

- `--system short` swaps the ~6,100-token agent prompt on `html` rows for a 60-token prompt. That prompt gets no loss anyway, so each step is ~4× cheaper. Gold rows keep their contract.
- `--whole-games 1` glues a source's pieces back together and keeps only complete pages (`<!doctype…</html>`) that fit in 8k tokens. Mid-file fragments are dropped.
- Only the assistant turn is trained (`train_on_completions=True`).

`run_slices.py` puts rows it recognizes as games first (gold, goodgame, then canvas+loop or three.js pages) and holds back the rest (`LORA_GAME_FILTER=1`).

### Size (Sep 23, 2026)


| File                                               | Size                | Notes                                                                             |
| -------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------- |
| `jsonl/html_corpus.jsonl`                          | 2.9 GB, 92,411 rows | Rebuild Sep 21: 16,587 files walked → 246 gold + 92,165 html rows, 8,497 unfitted |
| `jsonl/added.jsonl`                                | 4.5 GB              | Overnight `ingest.py` additions                                                   |
| `jsonl/train.jsonl`                                | 0.8 GB              | Assembled each slice (~235K rows counted by the monitor)                          |
| `games.sqlite`                                     | 79 MB               | ~246K unique hashes                                                               |
| `jsonl/seed.jsonl`, `also.jsonl`, `incoming.jsonl` | small               | Older plan-card rows (`kind: plan`). The current trainer does not read them.      |




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


| Env var            | Default                          | Meaning                                                                  |
| ------------------ | -------------------------------- | ------------------------------------------------------------------------ |
| `LORA_ROOT`        | `~/MLX_Models/html_game_sft`     | Project folder (data, adapters, snapshots, logs)                         |
| `LORA_BASE`        | `~/MLX_Models/Qwen3.8-27B-mxfp8` | Base model folder. Opened read-only.                                     |
| `LORA_PY`          | `~/Agents/.venv/bin/python`      | Python with `mlx`, `mlx-vlm`, `datasets`, `tokenizers`                   |
| `LORA_PORT`        | `8766`                           | Monitor port. Give each project its own. (8765 is chat.py Asset Studio.) |
| `LORA_TRAIN_ARGS`  | *(empty)*                        | Extra `train_lora.py` flags, appended to every slice                     |
| `LORA_GAME_FILTER` | `1`                              | `0` = train on every row (use this for non-game data)                    |
| `LORA_SLICE_MIN`   | `30`                             | Minutes per slice. A snapshot is saved after each one.                   |




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


| Flag                    | Default                                                | Effect                                                                        |
| ----------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `--jsonl PATH`          | (required; `run_slices.py` passes `jsonl/train.jsonl`) | Rows                                                                          |
| `--iters N`             | 20 (`run_slices.py` sizes it to the slice length)      | Steps this slice                                                              |
| `--max-seq-length`      | 8192                                                   | Longer rows are truncated                                                     |
| `--adapter-path DIR`    | `adapters/` when resuming                              | Resume this adapter                                                           |
| `--learning-rate`       | 5e-5                                                   | Adam LR (1e-5 before Sep 23)                                                  |
| `--system short|full`   | short                                                  | `short` = 60-token prompt on `html` rows. Use `full` for non-HTML data.       |
| `--whole-games 1|0`     | 1                                                      | Rejoin split HTML pieces, whole pages only. Use `0` for non-HTML data.        |
| `--dequantize 1|0`      | 1                                                      | bf16 base in memory for dense GEMM (same values)                              |
| `--lora-top-layers N`   | 32                                                     | Train only the top N decoder layers. Lower LoRA is kept and saved. `0` = all. |
| `--grad-checkpoint 1|0` | 1                                                      | Needed on the 27B (off runs out of memory)                                    |
| `--batch-size`          | 1                                                      | 2 gave no speedup on the 27B                                                  |
| `--steps-per-save`      | 10                                                     | Step checkpoints in `adapters/`                                               |


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


| Symptom                                                             | Fix                                                                                           |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Page says `waiting_for_seed` forever                                | The project has no rows. Put rows in `jsonl/html_corpus.jsonl` or `jsonl/added.jsonl`.        |
| `trainer already running pid N`                                     | One is already up. Check the page, or `cat $LORA_ROOT/logs/trainer.lock`.                     |
| Out of memory at load                                               | `chat.py` or another trainer holds a model. HOLD / quit it.                                   |
| `KeyError: 'source'` or `whole games 0 of N sources` in `train.log` | Non-HTML data with `--whole-games 1`. Pass `LORA_TRAIN_ARGS="--system full --whole-games 0"`. |
| `AttributeError … language_model`                                   | The base is not Qwen-VL shaped. Pass `--lora-top-layers 0`.                                   |
| `No Metal device available`                                         | Running inside a sandbox / headless shell. Use Terminal.app.                                  |


---



## 9. Small HTML/JS model (full fine-tune, `small/`)

A ~1B model trained only on HTML and JavaScript. **Every weight is trained (not LoRA).** Stage 1 is continued pretraining on packed 4,096-token blocks, with loss on every token. Stage 2 (spec → whole game, planned) teaches it the short spec prompt the agent will send.


|                                 |                                                                                    |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| Base (first run)                | `openbmb/MiniCPM5-1B-Base` (Apache 2.0, 2026), loaded by **mlx-lm**                |
| Base (M3 Ultra run)             | `openbmb/MiniCPM5-2B-Base` — same tokenizer as the 1B shards, so no retok. See §9f |
| Speed, M2 Ultra 192 GB          | ~2,080 tokens/s, 16× the 27B LoRA (130 tokens/s), peak 57 GB                       |
| Speed, M3 Ultra 512 GB, 2B base | ~1,120 tokens/s at batch 2, peak 103 GB. Same 2B-token budget ≈ 22 days            |
| Budget                          | 2B tokens = 61,035 updates ≈ 11 days on the M2 Ultra (1B)                          |
| Code (git)                      | `fine_tunning/small/` — **no training data in git**                                |
| Data + checkpoints (not in git) | `$SMALL_ROOT`, default `~/MLX_Models/html_js_small/`                               |


**Quick path after a git update:** fill a drive on the source Mac (§9a) → on the new Mac, `./start.sh small` (top of this file). That starts the page, the trainer, and the review. `./start.sh small continue` also resumes the Stack download. To try a 0.5B instead, use §9d.

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


| Copy?            | Path                                       | Size            | Why                                                                                                     |
| ---------------- | ------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------- |
| **yes**          | `shards/`                                  | 11 GB           | Token ids + one JSON line per doc (`src`, `path`, `rank`, `norm`, …). Includes `shards/tokenizer.json`. |
| **yes**          | `quality.sqlite`                           | 0.6 GB          | Quality weights keyed by text hash (`norm`). Works with any tokenizer.                                  |
| only to continue | `checkpoints/latest/`                      | about 6 GB      | Newest weights + optimizer. Overwritten every 30 minutes. Training resumes from here.                   |
| kept             | `snapshots/<date>/`                        | about 2 GB each | Weights-only copy every 6 hours. Nothing deletes these.                                                 |
| **no**           | `~/MLX_Models/html_game_sft/raw/` (138 GB) | —               | Source HTML/JS files. **Not needed** — the shards already hold the text.                                |
| **no**           | `logs/`                                    | —               | Rebuilds on the new Mac                                                                                 |




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

# 5) start page, trainer, and review. Add `continue` only if you also want the Stack download.
cd ~/Agent_learning/fine_tunning
chmod +x start.sh          # once, if needed
./start.sh small
# ./start.sh small continue
```

Then open **[http://127.0.0.1:8767/](http://127.0.0.1:8767/)**. You should see tokens/s (~2,000+ on an Ultra), updates, and loss. Or:

```bash
tail -f ~/MLX_Models/html_js_small/logs/train.log
# expect lines like: Iter 10: Train loss … Tokens/sec 2080 …
```

**Train straight off a fast external SSD** (skip the copy into `~/MLX_Models`):

```bash
SMALL_ROOT=/Volumes/DRIVE/html_js_small ./start.sh small
# same, plus the Stack download:
# SMALL_ROOT=/Volumes/DRIVE/html_js_small ./start.sh small continue
```

**Do not continue the same run on both Macs at once.** They would diverge from the same checkpoint. To compare machines or settings, use a fresh `SMALL_ROOT` (or a different base — §9d).


| Command                                    | Does                                                                                                         |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `./start.sh small`                         | page + trainer + file check + code quality + browser. No Stack download.                                    |
| `./start.sh small continue`                | same, plus the Stack download. File check keeps scoring new shards.                                         |
| `./start.sh small-train` / `small-monitor` | one of the two                                                                                               |
| `./start.sh small-data`                    | build shards from scratch (own games need `html_game_sft/games.sqlite` + `raw/`; github re-streams, ~20 min) |
| `./start.sh small-retok DIR`               | re-tokenize `DIR/shards` for `SMALL_BASE` into `SMALL_ROOT/shards`                                           |
| `./start.sh small-quality`                 | file check (stays up), code quality, and browser. Starts only the ones that are down.                        |
| `./start.sh small-stack`                   | The Stack v2 download. Resumes `logs/stack_state.json`. 32 workers, stop after 8B new tokens                 |
| `./start.sh small-synth`                   | Refresh the synthetic-game set (original, train text, tokens). Safe to run again as `synth.jsonl` grows. |
| `./start.sh small-synth-share`             | Generator Mac only: `small-synth`, then copy the games to `small/synth_games/synth.jsonl`, commit that one file, and push. |
| `./start.sh small-export`                  | Rebuild `train/` text for own, GitHub, and Stack from `tokens/`. Skips shards already exported.         |


**Set weights.** The files stay in separate folders. One training pass draws them at these weights (canvas game with a loop; a weaker file in the same set is 0.6× or 0.35×). The numbers are sized for about 3,000 synthetic games against the tokens already on disk.

| Set | What it is | On disk | Weight | Share of one pass |
| --- | --- | --- | --- | --- |
| Synthetic games | prompt, newline, page, from `synth.jsonl` | about 7M tokens at 3,000 games | 1500 | about 20% |
| Own games | collected HTML games | 0.53B tokens | 40 | about 40% |
| GitHub (`gcc`) | github-code-clean HTML and JavaScript | 2.2B tokens | 6 | about 25% |
| The Stack | public HTML and JavaScript | 8.0B tokens | 1 | about 15% |

Rebuild the synthetic shards after more games land. Rows already tokenized are skipped. The chat tags are removed. Each document is the user prompt, a newline, and the raw page.

```bash
cd ~/Agent_learning/fine_tunning
./start.sh small-synth
```

**Synthetic games from git.** The games are small (8 MB at 660), so the kept games travel in git as `fine_tunning/small/synth_games/synth.jsonl`, the same chat rows `synth_html.py` writes. Token files do not go in git: each Mac tokenizes for its own base.

On the Mac that runs `synth_html.py`, after more games land:

```bash
cd ~/Agent_learning/fine_tunning
./start.sh small-synth-share    # tokenize here, copy the games into git, commit that file, push
```

On any other Mac:

```bash
cd ~/Agent_learning && git pull && cd fine_tunning
./start.sh small-synth          # reads small/synth_games/synth.jsonl when there is no local generator file
```

`small-synth` reads `$LORA_ROOT/jsonl/synth.jsonl` when that file exists (the generator Mac), else the git copy. `SYNTH_JSONL=path` overrides both. It writes the three folders:

| Folder | What `small-synth` writes |
| --- | --- |
| `sets/synth/original/rows.jsonl` | every kept chat row, unchanged (rewritten each run) |
| `sets/synth/train/docs.jsonl` | `{"sha", "text"}` per game. The text is the user prompt, a newline, and the page, with the chat tags removed |
| `sets/synth/tokens/synth_NNNNN.bin` + `.jsonl` | token ids, and one line per game (`sha`, `offset`, `ntok`, `src=synth`, `rank`). Only games whose `sha` is not already here are added, as a new `synth_NNNNN` pair |

A row with `weight` 0 (thumbs-down on the synth page) is left out. The trainer reads only `tokens/`. It rebuilds its draw list at start and at every 30-minute save, so new games join without a restart.

**Where the sets live.** One folder per set, under `~/MLX_Models/html_js_small/sets/`. Move a set by moving that folder. A short map sits beside them in `sets/README.md`.

| Folder | What is in it |
| --- | --- |
| `original/` | The source, unchanged. Synthetic games: the chat JSONL row. Own games: the HTML file path and its bytes. GitHub and The Stack were streamed and not kept as files; `original/SOURCE.txt` names the dataset. |
| `train/` | The exact string that is tokenized. Own, GitHub, and Stack: the file text. Synthetic games: the prompt, a newline, then the page, with the chat tags removed. |
| `tokens/` | Token ids (`.bin`) plus one sidecar `.jsonl` per shard (`sha`, `path`, `rank`, `ntok`). This is what `train_small.py` reads. |

```text
sets/own/    original/  train/  tokens/
sets/gcc/    original/  train/  tokens/
sets/stack/  original/  train/  tokens/
sets/synth/  original/  train/  tokens/
```

**What training streams.** There is no single training file, and the weights are not stored inside the data. `train_small.py` reads every `sets/<name>/tokens/*.bin`. The `.jsonl` beside each `.bin` is an index: which document, the byte offset, the token count, `src`, and `rank`. `train/` and `original/` are not read.

The learning rate is one number for every token (`--lr`, default `5e-5` on the 1B). A higher set weight does not take a bigger step. It only puts that document into the draw list more times, so its tokens are seen more often and the same-sized step happens more often on them.

On each start, and again every 30 minutes, the trainer builds a draw list:

1. Open each sidecar under `sets/*/tokens/`.
2. Look up that document's weight: set weight × rank factor × the quality score.
3. Put that many copies of the document into one long list (weight 40 puts in 40 copies; weight 0.6 puts in one copy on about 60% of passes). Weight 0 is left out.
4. Shuffle the whole list once, so the copies land in random places and are not shown back to back. Then read token ids from the `.bin` files and pack them into 4,096-token blocks. The loss on each token in the block is the same.

The set weights are in `fine_tunning/small/train_small.py` (`SET_WEIGHT`). They travel with the code, not with the data folder. The per-file quality scores are `~/MLX_Models/html_js_small/quality.sqlite` (a multiplier, often 0 for a near-duplicate). A document's draw rate is those two numbers multiplied.

To train on another machine, copy the code, the base model, `sets/*/tokens/`, and `quality.sqlite`. Copy `checkpoints/latest/` only to continue this run. `train/` is the readable text if you need to tokenize again for a different model.

These weights apply the next time `train_small.py` starts. A trainer that is already running keeps the weights it loaded at start.

Env vars: `SMALL_ROOT` (data/checkpoints), `SMALL_BASE` (model), `SMALL_PORT` (page, default 8767), `SMALL_TRAIN_ARGS` (extra trainer flags), `LORA_PY`, `BROWSER_PY`.

**Stop:** On the page, press Stop training. That writes `logs/hold.json` and ends `train_small.py`. The watcher will not start it again until you press Resume. Up to 30 minutes since the last save are lost. `./start.sh small` reloads the page if this code is newer, and it leaves the trainer stopped while that file says held. Resume continues from `checkpoints/latest/` at the `lr=` in `logs/train.log`. If the loss has blown up, do not resume that checkpoint — follow [Restart when the loss blows up](#restart-when-the-loss-blows-up).

### 9c. Confirm it is running / common failures


| Check                                                  | Expect                                                 |
| ------------------------------------------------------ | ------------------------------------------------------ |
| [http://127.0.0.1:8767/](http://127.0.0.1:8767/)       | Title names the base; green tokens/s; updates counting |
| `tail ~/MLX_Models/html_js_small/logs/train.log`       | `Iter N: … Tokens/sec …` every 10 updates              |
| `pgrep -fl train_small`                                | one Python process                                     |
| `ls ~/MLX_Models/html_js_small/sets/*/tokens/*.bin | wc -l` | thousands of token shards (not 0)                     |



| Symptom                          | Fix                                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `no model at SMALL_BASE`         | Step 3 did not finish; re-run `hf download …`                                                                       |
| `no token shards`                | `sets/*/tokens/` was not copied. Re-rsync `sets/` and `quality.sqlite`.                                        |
| `train_small.py already running` | A trainer is up; open the page or `pkill -f train_small.py` first                                                   |
| `No Metal device available`      | Run in Terminal.app, not a sandboxed Cursor shell                                                                   |
| Page blank / old LoRA curves     | Wrong port — small model is **8767**, LoRA is 8766                                                                  |
| Out of memory                    | Quit any other GPU model (`chat.py`, another trainer). Or `SMALL_TRAIN_ARGS="--batch 1 --accum 8" ./start.sh small` |
| Loss near 7–10 after a start near 1 | The learning rate wrecked the weights. Shards are fine. [Restart when the loss blows up](#restart-when-the-loss-blows-up). |




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


| File                                  | Job                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `small/data.py`                       | Builds `sets/<name>/{original,train,tokens}`. `--source own` = games from `html_game_sft/games.sqlite`; `--source gcc` = [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) HTML + JavaScript (streamed); `--source stack` = The Stack; `--source synth` = `synth.jsonl` (chat row, then prompt-newline-page, then tokens); `--source retok --from DIR` = re-tokenize for a new base. Filters size / minified / base64 / symbol soup / HTML without `<script>`. Resumes. |
| `small/train_small.py`                | Trainer: weighted, deduplicated packing; compiled step; AdamW, warmup + cosine. Every 30 min overwrites `checkpoints/latest/` and re-reads `shards/` + `quality.sqlite`. Every 6 hours keeps a dated weights-only copy in `snapshots/`. Those copies are never deleted. Resumes from `checkpoints/latest/`.                                                          |
| `small/quality_worker.py`             | CPU quality scoring (niced). Writes `quality.sqlite` → `quality(norm, weight)`.                                                                                                                                                                                                                                                                                      |
| `start.sh small*`                     | One-line starts (§9b)                                                                                                                                                                                                                                                                                                                                                |
| `serve_progress.py` + `progress.html` | Monitor; small-model root shows speed first                                                                                                                                                                                                                                                                                                                          |


Where the data came from: own games (`html_game_sft/raw/`) → 14,470 unique files (~33M tokens) after dedup; github-code-clean (49 parquet files) → 916,493 docs, 2.22B tokens. The Stack added about 8.0B tokens. Sampling weights are in the table in §9b (synthetic ×1500, own ×40, GitHub ×6, Stack ×1 for a canvas game with a loop). `quality.sqlite` multiplies:


The scorer writes 0 for the rows marked "no" below. A score under 1.5 is boilerplate and is removed. Scores from 1.5 up stay, including the 0.7 band.

| Pass                                                                                       | Weight the scorer writes                              | Train it? |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------- | --------- |
| Near-duplicate (MinHash, ~0.77 Jaccard), not the keeper                                    | 0                                                     | no        |
| JS syntax error (node `vm` parse, never run)                                               | 0                                                     | no        |
| Repeated lines / "generated, do not edit"                                                  | 0                                                     | no        |
| Stack-Edu JavaScript classifier ([SmolLM2](https://arxiv.org/abs/2502.02737)), github only | score <1.5 → 0, <2.5 → 0.7, <3.5 → 1.3, else 2.0      | score <1.5: no. The rest stays |
| Headless Chromium, 2 s, network blocked                                                    | page error 0, canvas that draws 1.5                   | errors: no. A canvas that draws stays |



### `train_small.py` flags


| Flag                | Default    | Effect                                                                                                                                                                    |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--block`           | 4096       | Tokens per packed sequence                                                                                                                                                |
| `--batch`           | 2          | Sequences per micro-step. On the M2, batch 1/2/4 were the same tokens/s (2 = 57 GB). On the M3 Ultra with the 2B, batch 4 and 8 were no faster and used 180 GB and 343 GB |
| `--accum`           | 4          | Micro-steps per optimizer update (update = 32,768 tokens)                                                                                                                 |
| `--lr` / `--warmup` | 5e-5 / 200 | Peak LR, warmup updates; cosine to 10% after. Set with `SMALL_TRAIN_ARGS="--lr 1e-5"`. Default `5e-5` fits the 1B. The 2B diverged at that peak — use `1e-5`. |
| `--total-tokens`    | 2e9        | Stop after this many trained tokens                                                                                                                                       |
| `--save-minutes`    | 30         | Overwrite `checkpoints/latest/` and re-read data and quality. A dated copy is kept every 6 hours.                                                                         |
| `--bench N`         | 0          | Time N micro-steps, save nothing                                                                                                                                          |


### Restart when the loss blows up

A healthy continued-pretrain loss on these shards starts near **1.0** (the base already uses this tokenizer). A wrong tokenizer starts near `ln(vocab)` (~11.8 for MiniCPM5, vocab 130560) on the first logged update. A learning-rate blow-up looks different: loss stays near 1 through warmup, then jumps to about 7–10 within a few updates of the peak and stays there. The shards are fine. Resume the last checkpoint that still had loss near 1, at a lower peak. Do not resume a checkpoint written after the jump, and do not throw away a healthy run to start over at the base.

`checkpoints/latest/` is overwritten every 30 minutes, and the dated snapshot is only every 6 hours. On Sep 24, 2026 the 2B hit `5e-5` at update 200, and the loss went from 1.08 to 10.1 at update 220. The only files still on disk after that were update 728 (snapshot `20260925T011623Z`) and update 850. Both are past the jump. There was no earlier save left to resume, so the restart had to load `MiniCPM5-2B-Base`. While a run is healthy, copy `checkpoints/latest/` to `checkpoints/last_good/` after each save so the next restart has that point.

`5e-5` is the `--lr` default and is what the 1B run uses. Restart the 2B at `1e-5`. If `1e-5` also jumps, resume `last_good` at `3e-6`.

```bash
# 1) stop the run that diverged
pkill -f train_small.py

# 2) park the dead latest. Do not delete it. Do not copy it back.
mv "$SMALL_ROOT/checkpoints/latest" \
   "$SMALL_ROOT/checkpoints/diverged_$(date -u +%Y%m%dT%H%M%SZ)"

# 3) resume the last healthy save. This is the whole point.
cp -cR "$SMALL_ROOT/checkpoints/last_good" "$SMALL_ROOT/checkpoints/latest"

# 4) hide the dead curve. The page plots every Iter line in this file.
mv "$SMALL_ROOT/logs/train.log" \
   "$SMALL_ROOT/logs/train_diverged_$(date -u +%Y%m%dT%H%M%SZ).log"

# 5) lower peak. resume=True loads last_good, including its update count.
cd ~/Agent_learning/fine_tunning
export SMALL_TRAIN_ARGS="--lr 1e-5"   # 2B. Use 3e-6 if 1e-5 already blew up.
# New shell: also export SMALL_ROOT and SMALL_BASE (§9b, or §9f for the 2B).
./start.sh small
```

The first line of `logs/train.log` must say the new `lr=` and `resume=True`, and the `Iter` loss should be near 1.0, not 7–10. `resume=False` means `checkpoints/latest/` was missing and the run started at the base again.

Set the rate on any start, not only a restart: `SMALL_TRAIN_ARGS="--lr 1e-5" ./start.sh small`. Other trainer flags go in that same variable (`--lr 1e-5 --warmup 400`).

`./start.sh small-watch` does the restart by itself, every 30 minutes. `./start.sh small` starts it too. On a healthy save it copies `checkpoints/latest/` to `checkpoints/last_good/` (older copies move to `checkpoints/kept/`, nothing is deleted). If the loss goes above 3, it resumes `last_good` at the next lower peak: `1e-5`, then `3e-6`, then `1e-6`, then it stops restarting. If the trainer dies while the loss is still near 1, it resumes `checkpoints/latest/` at the same peak. It will not start from the base when a checkpoint exists. If `logs/hold.json` says held, it leaves the trainer stopped — that is a Stop from the page, not a crash. One line per check: `$SMALL_ROOT/logs/watch.log`.


### Use a checkpoint

On the progress page, Stop training. The left Generate runs the model you picked there (the base, or another folder under `~/MLX_Models` such as Qwen3.8-27B). Qwen and the other vision models load with `mlx_vlm` and their chat template, the same way `chat.py` does. The right Generate runs the checkpoint you picked (latest, or a snapshot) as plain text. Each button runs only that side.

The same folders are normal mlx-lm models. Stage 1 is a base model, so use a raw prompt, not a chat template:

```bash
"$LORA_PY" -m mlx_lm generate --model "$SMALL_ROOT/checkpoints/latest" \
  --ignore-chat-template --max-tokens 400 --prompt '<!DOCTYPE html>
<html><head><title>Snake</title>'
```



### 9f. MiniCPM5-2B on the M3 Ultra (Sep 24, 2026)

`openbmb/MiniCPM5-2B-Base` is the newer base (Sep 7, 2026). It is a normal Llama, 2.52B parameters, bf16 weights about 4.7 GB. The 1B shards already use this tokenizer (same `tokenizer.json`), so train them in place. Do not retokenize.


|                    |                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Weights            | `~/MLX_Models/MiniCPM5-2B-Base` (never modified)                                                                       |
| Data + checkpoints | `/Users/jonathanrothberg/Data/html_js_small` (`SMALL_ROOT`)                                                            |
| Page               | [http://127.0.0.1:8767/](http://127.0.0.1:8767/)                                                                       |
| Measured speed     | batch 2 → ~1,060–1,120 tokens/s, 103 GB. Batch 4 → ~1,110 tok/s, 180 GB. Batch 8 → ~990 tok/s, 343 GB. Stay at batch 2 |
| 2B-token budget    | about 22 days at that speed                                                                                            |


8-bit and MXFP8 are not faster. MLX dequantizes them and then runs the bf16 GEMM, and those kernels are tuned for decoding. The 27B LoRA trainer already unpacks MXFP8 to bf16 for the same reason (`train_lora.py --dequantize`).

A throwaway speed test lives in `~/MLX_Models/html_js_MiniCPM5-2B-Base-bench`. Those shards are random tokens. Do not train on that folder.

`./start.sh` puts each job under launchd (PPID 1), in its own session. Quitting Cursor does not stop it. `nohup` alone does not, because a Cursor shell kills its process group.

The 2B does not use the default `5e-5` learning rate. That peak wrecked a fresh start on Sep 24, 2026 (loss ~1 through warmup, then ~10). Use `1e-5`. If a later run does the same, follow [Restart when the loss blows up](#restart-when-the-loss-blows-up) and resume `checkpoints/last_good/`, not the base and not the save from after the jump.

Stop and Generate are on the page (http://127.0.0.1:8767/). Left Generate runs the comparison model you picked. Right Generate runs the checkpoint you picked. Resume there keeps the `lr=` from `logs/train.log`. `./start.sh small-monitor` reloads the page when the code is newer and does not touch the trainer. While `logs/hold.json` says held, `./start.sh small-train` does not start the trainer.

```bash
cd ~/Agent_learning/fine_tunning
export SMALL_ROOT=/Users/jonathanrothberg/Data/html_js_small
export SMALL_BASE=~/MLX_Models/MiniCPM5-2B-Base
export SMALL_TRAIN_ARGS="--lr 1e-5"
./start.sh small-monitor    # page first
./start.sh small-train
# After tokens/s is moving. This script uses 16 score workers, then edu 12 + browser 6.
# The Sep 24 run used 8, then edu 4 + browser 2, so the GPU stayed at ~1,120 tokens/s.
./start.sh small-quality
```

Node 25 aborts the syntax checker on some bad scripts (`Assertion failed: (end) >= (start)`), which used to kill the whole score pass. `quality_worker.py` restarts node and counts that script as bad JS. A checkpoint of this 2B run is much larger than the 1B's 6.1 GB: weights plus AdamW state are on the order of 25 GB.