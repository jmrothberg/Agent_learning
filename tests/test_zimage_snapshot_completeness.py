"""Z-Image snapshot selection should skip incomplete HF cache dirs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import assets  # noqa: E402


def _snap_dir(fake_home: Path) -> Path:
    return (
        fake_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Tongyi-MAI--Z-Image-Turbo"
        / "snapshots"
        / "deadbeef"
    )


def test_resolve_zimage_path_skips_empty_text_encoder(tmp_path: Path, monkeypatch) -> None:
    # text_encoder exists but has no weights at all => incomplete.
    fake_home = tmp_path / "home"
    (_snap_dir(fake_home) / "text_encoder").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(os.path, "expanduser", lambda *_: str(fake_home))
    assert assets._resolve_zimage_path() == "Tongyi-MAI/Z-Image-Turbo"


def test_resolve_zimage_path_skips_partial_sharded_snapshot(tmp_path: Path, monkeypatch) -> None:
    # Reproduce the observed failure: index lists 3 shards but only shard 3
    # is present on disk (shards 1 and 2 missing). Must be rejected.
    fake_home = tmp_path / "home"
    te = _snap_dir(fake_home) / "text_encoder"
    te.mkdir(parents=True, exist_ok=True)
    index = {
        "weight_map": {
            "a": "model-00001-of-00003.safetensors",
            "b": "model-00002-of-00003.safetensors",
            "c": "model-00003-of-00003.safetensors",
        }
    }
    (te / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (te / "model-00003-of-00003.safetensors").write_text("x", encoding="utf-8")

    monkeypatch.setattr(os.path, "expanduser", lambda *_: str(fake_home))
    assert assets._resolve_zimage_path() == "Tongyi-MAI/Z-Image-Turbo"


def test_resolve_zimage_path_accepts_complete_sharded_snapshot(tmp_path: Path, monkeypatch) -> None:
    # All shards present => the local snapshot is accepted (no network fallback).
    fake_home = tmp_path / "home"
    snap = _snap_dir(fake_home)
    te = snap / "text_encoder"
    te.mkdir(parents=True, exist_ok=True)
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (te / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (te / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")
    (te / "model-00002-of-00002.safetensors").write_text("y", encoding="utf-8")

    monkeypatch.setattr(os.path, "expanduser", lambda *_: str(fake_home))
    assert assets._resolve_zimage_path() == str(snap)

