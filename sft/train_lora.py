"""Language-only LoRA on Qwen3.8-27B-mxfp8. Vision tower is not trained.

Writes adapters under html_game_sft/adapters/. The base model directory
is opened read-only by mlx-vlm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.optimizers as optim
from datasets import load_dataset
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.utils import load

ROOT = Path("/Users/jonathanrothberg/MLX_Models/html_game_sft")
BASE = "/Users/jonathanrothberg/MLX_Models/Qwen3.8-27B-mxfp8"


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
    args = p.parse_args()

    out_file = ROOT / "adapters" / "adapters.safetensors"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base (vision tower included, frozen): {BASE}", flush=True)
    model, processor = load(BASE, processor_config={"trust_remote_code": True})
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    print(f"model_type={model_type}", flush=True)

    dataset = load_dataset("json", data_files=str(args.jsonl), split="train")
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
    optimizer = optim.Adam(learning_rate=1e-5)
    training_args = TrainingArgs(
        batch_size=1,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=10**9,
        steps_per_save=args.steps_per_save,
        val_batches=0,
        max_seq_length=args.max_seq_length,
        adapter_file=str(out_file),
        grad_checkpoint=True,
        learning_rate=1e-5,
        grad_clip=None,
        gradient_accumulation_steps=1,
        full_finetune=False,
    )
    print(
        f"train iters={args.iters} seq={args.max_seq_length} "
        f"rows={len(dataset)} resume={resume}",
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
