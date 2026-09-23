"""Language-only LoRA on Qwen3.8-27B-mxfp8. Vision tower is not trained.

Writes adapters under html_game_sft/adapters/. The base model directory
is opened read-only by mlx-vlm.

MULTI-LORA: LORA_ROOT (project folder) and LORA_BASE (base model folder)
pick a different LoRA / base. Defaults are the HTML-game run. See README.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from datasets import load_dataset
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
from mlx_vlm.quant_utils import dequantize_model
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.utils import load

# MULTI-LORA: env overrides; defaults = the HTML-game LoRA on Qwen3.8-27B.
ROOT = Path(os.environ.get("LORA_ROOT", "/Users/jonathanrothberg/MLX_Models/html_game_sft")).expanduser()
BASE = os.path.expanduser(os.environ.get("LORA_BASE", "/Users/jonathanrothberg/MLX_Models/Qwen3.8-27B-mxfp8"))


# SPEED: general "html" rows carried the full ~6,100-token agent prompt, which was
# ~78% of every step's tokens but gets no loss (train_on_completions). Swapping it
# for this short prompt makes each step ~4x cheaper. "gold" /640png rows keep
# their ~2,200-token contract so the chip rules are still learned in context.
_SHORT_SYS = (
    "You are an expert HTML5 game programmer. Write the complete game as one "
    "self-contained HTML file (inline CSS and JavaScript, canvas) that runs in "
    "Chrome. Reply with brief <think> notes, then the file inside "
    "<html_file>...</html_file>."
)


def _short_system(row: dict) -> dict:
    if row.get("kind") == "html" and row["messages"] and row["messages"][0]["role"] == "system":
        row["messages"][0]["content"] = _SHORT_SYS
    return row


# SPEED/QUALITY: rows.py split long games into ~2k-token pieces to fit the old
# 6k prompt, so most html rows were mid-file fragments wrapped in <html_file>.
# With _SHORT_SYS the room is ~8k, so glue a source's pieces back together and
# keep only whole pages (<!doctype/<html ... </html>) that fit. Gold rows as-is.
def _whole_games(path: Path, tokenizer, max_tokens: int) -> list[dict]:
    import json
    groups: dict[str, list[dict]] = {}
    gold = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "gold":
                gold.append(row)
            else:
                groups.setdefault(row["source"], []).append(row)
    out = []
    for rows in groups.values():
        bodies = [r["messages"][-1]["content"].split("<html_file>\n", 1)[-1]
                  .rsplit("\n</html_file>", 1)[0] for r in rows]
        page = "".join(bodies).strip()
        low = page.lower()
        if not (low.startswith(("<!doctype", "<html")) and "</html>" in low[-200:]):
            continue
        if len(tokenizer.encode(page)) > max_tokens:
            continue
        row = rows[0]
        head = row["messages"][-1]["content"].split("<html_file>\n", 1)[0]
        row["messages"][-1]["content"] = f"{head}<html_file>\n{page}\n</html_file>\n"
        out.append(row)
    print(f"whole games {len(out)} of {len(groups)} sources, gold rows {len(gold)}", flush=True)
    return out + gold


def _ns(**kw):
    class NS:
        pass
    o = NS()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--steps-per-report", type=int, default=1)
    p.add_argument("--steps-per-save", type=int, default=10)
    p.add_argument("--adapter-path", type=Path, default=None,
                   help="Resume from this adapter directory")
    # SPEED flags. Defaults are the fast settings so run_slices needs no change.
    p.add_argument("--system", choices=("short", "full"), default="short",
                   help="short = swap the agent prompt on html rows for _SHORT_SYS")
    p.add_argument("--grad-checkpoint", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    # SPEED: mxfp8 quantized matmuls are tuned for decoding, not the long
    # prompts of training. Unpacking the frozen base to bf16 (~54 GB) uses the
    # dense GEMM. The values are the same, so the LoRA still fits the mxfp8 model.
    p.add_argument("--dequantize", type=int, default=1)
    # 1e-5 is ~10x below the usual LoRA rate; each game taught very little.
    p.add_argument("--learning-rate", type=float, default=5e-5)
    # SPEED: only the top N decoder layers train. Backward stops at layer 64-N,
    # so each step is ~1.6x cheaper at N=32. Lower-layer LoRA stays applied and
    # is still written to the adapter file (see _save_all below). 0 = all layers.
    p.add_argument("--lora-top-layers", type=int, default=32)
    p.add_argument("--whole-games", type=int, default=1)
    args = p.parse_args()

    out_file = ROOT / "adapters" / "adapters.safetensors"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base (vision tower included, frozen): {BASE}", flush=True)
    model, processor = load(BASE, processor_config={"trust_remote_code": True})
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    print(f"model_type={model_type}", flush=True)
    if args.dequantize:
        model = dequantize_model(model)
        for _, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                if m.weight.dtype == mx.float32:
                    m.weight = m.weight.astype(mx.bfloat16)
                # Resume (apply_lora_layers) never freezes the base. Quantized
                # weights were frozen by type; unpacked ones must be frozen here
                # or all 27B params train and land in the adapter file.
                m.freeze()
        mx.eval(model.parameters())
        print("dequantized base to bf16", flush=True)

    if args.whole_games and args.system == "short":
        from datasets import Dataset
        tok = getattr(processor, "tokenizer", processor)
        dataset = Dataset.from_list(_whole_games(args.jsonl, tok, args.max_seq_length - 300))
    else:
        dataset = load_dataset("json", data_files=str(args.jsonl), split="train")
    if args.system == "short":
        dataset = dataset.map(_short_system)
    dataset = transform_dataset_to_messages(dataset, model_type, None)
    config = model.config.__dict__ if not isinstance(model.config, dict) else model.config
    train_dataset = VisionDataset(
        dataset,
        config,
        processor,
        image_resize_shape=None,
        train_on_completions=True,
    )
    train_args_ns = _ns(
        full_finetune=False,
        train_vision=False,
        lora_rank=16,
        lora_alpha=32.0,
        lora_dropout=0.0,
    )
    resume = str(args.adapter_path) if args.adapter_path else None
    model = setup_model_for_training(model, train_args_ns, resume)
    if args.lora_top_layers:
        # MULTI-LORA: layer path is Qwen-VL style (mlx-vlm). Other VLM layouts: --lora-top-layers 0.
        layers = model.language_model.model.layers
        for layer in layers[: max(0, len(layers) - args.lora_top_layers)]:
            layer.freeze()
        # The trainer saves trainable params only. Frozen lower-layer LoRA (and
        # anything else already in the resumed file) must stay in the file or the
        # next slice resumes with those layers reset.
        import mlx_vlm.trainer.sft_trainer as sft
        keep = set(mx.load(str(Path(resume) / "adapters.safetensors"))) if resume else set()
        orig_save = sft.save_adapter

        def _save_all(m, adapter_file):
            orig_save(m, adapter_file)
            params = dict(tree_flatten(m.parameters()))
            out = dict(tree_flatten(m.trainable_parameters()))
            out.update({k: params[k] for k in keep if k in params and k not in out})
            mx.save_safetensors(str(adapter_file), out)

        sft.save_adapter = _save_all
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params {n_train / 1e6:.0f}M", flush=True)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=10**9,
        steps_per_save=args.steps_per_save,
        val_batches=0,
        max_seq_length=args.max_seq_length,
        adapter_file=str(out_file),
        grad_checkpoint=bool(args.grad_checkpoint),
        learning_rate=args.learning_rate,
        grad_clip=None,
        gradient_accumulation_steps=1,
        full_finetune=False,
    )
    print(
        f"train iters={args.iters} seq={args.max_seq_length} "
        f"rows={len(dataset)} resume={resume} system={args.system} "
        f"ckpt={args.grad_checkpoint} bs={args.batch_size} lr={args.learning_rate} "
        f"top_layers={args.lora_top_layers}",
        flush=True,
    )
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        train_on_completions=True,
    )
    print(f"Saved adapter {out_file}", flush=True)


if __name__ == "__main__":
    main()
