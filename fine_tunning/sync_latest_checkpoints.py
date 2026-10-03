"""Copy the newest LoRA and the newest full finetune into the repo.

`git commit` runs this from .git/hooks/pre-commit. One folder of each type
is updated, and only when that save is newer than the copy already here.
Older snapshots stay in ~/MLX_Models. The optimizer file is not copied.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST_ROOT = HERE / "checkpoints"
LORA_ROOT = Path(os.environ.get("LORA_ROOT", "~/MLX_Models/html_game_sft")).expanduser()
SMALL_ROOT = Path(os.environ.get("SMALL_ROOT", "~/MLX_Models/html_js_small")).expanduser()
SKIP = {"optimizer.safetensors"}
# GitHub LFS rejects a single file bigger than 2 GiB.
MAX_LFS_BYTES = 2147483648


def newest_dir(folders: list[Path], filename: str) -> Path | None:
    best: Path | None = None
    best_m = -1.0
    for folder in folders:
        if not folder.is_dir():
            continue
        for child in folder.iterdir():
            target = child / filename if child.is_dir() else None
            if target is None or not target.is_file():
                continue
            mtime = target.stat().st_mtime
            if mtime > best_m:
                best, best_m = child, mtime
    return best


def source_stamp(folder: Path, filename: str) -> str:
    target = folder / filename
    return f"{folder}\n{int(target.stat().st_mtime)}"


def already_current(dest: Path, stamp: str) -> bool:
    source = dest / "SOURCE.txt"
    if not source.is_file():
        return False
    return source.read_text(encoding="utf-8").strip() == stamp.strip()


def split_oversize(path: Path) -> None:
    """Replace one over-2-GiB file with two parts. The source checkpoint is untouched."""
    if not path.is_file() or path.stat().st_size <= MAX_LFS_BYTES:
        return
    first = path.with_name(path.name + ".part-aa")
    second = path.with_name(path.name + ".part-ab")
    half = path.stat().st_size // 2
    with path.open("rb") as src, first.open("wb") as out_a, second.open("wb") as out_b:
        remaining = half
        while remaining:
            block = src.read(min(remaining, 8 * 1024 * 1024))
            if not block:
                break
            out_a.write(block)
            remaining -= len(block)
        while True:
            block = src.read(8 * 1024 * 1024)
            if not block:
                break
            out_b.write(block)
    path.unlink()


def copy_folder(src: Path, dest: Path, stamp: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file() or item.name in SKIP or item.name == "SOURCE.txt":
            continue
        target = dest / item.name
        # Clone on APFS when the copy is new. Fall back to a normal copy.
        result = subprocess.run(["cp", "-c", "-f", str(item), str(target)], check=False)
        if result.returncode != 0:
            subprocess.run(["cp", "-f", str(item), str(target)], check=True)
        split_oversize(target)
    (dest / "SOURCE.txt").write_text(stamp + "\n", encoding="utf-8")


def sync_one(label: str, src: Path | None, filename: str) -> None:
    dest = DEST_ROOT / label
    if src is None:
        print(f"{label}: no checkpoint", flush=True)
        return
    stamp = source_stamp(src, filename)
    if already_current(dest, stamp):
        print(f"{label}: already current ({src.name})", flush=True)
        return
    copy_folder(src, dest, stamp)
    print(f"{label}: copied {src}", flush=True)


def main() -> None:
    lora = newest_dir([LORA_ROOT / "snapshots"], "adapters.safetensors")
    small_folders = [SMALL_ROOT / "checkpoints", SMALL_ROOT / "snapshots"]
    small = newest_dir(small_folders, "model.safetensors")
    sync_one("lora", lora, "adapters.safetensors")
    sync_one("small", small, "model.safetensors")


if __name__ == "__main__":
    main()
