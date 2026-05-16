"""Visual-progress judge — the missing "third signal".

The agent's built-in verifier (tools.py) checks structural things: the
HTML loads, the canvas animates, pressing keys changes state, the
model's own probes return truthy. It does NOT check whether the running
game LOOKS like what the user asked for. A donkey-kong session can pass
every probe while showing a blank screen, the wrong character, or
mario floating in space.

This module fills that gap by taking the per-iter screenshot the agent
already captures and asking a vision-capable model "is this getting
closer to the user's goal, and what's still visibly missing?". The
answer is fed into the next iteration's prompt so the code-writing
model knows what to fix — without the user having to play through.

Local-first policy: if the active building backend is itself a VLM
(rare for local — most coding models are text-only), use it. Otherwise
fall back to Anthropic Claude (cheap, fast, vision-capable). Skip
silently if neither is available — the run still works, just without
this signal.

Disable entirely with env VISION_JUDGE=0.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["VisionVerdict", "judge_visual_progress", "is_enabled"]


# Claude Sonnet 4.6 — current default vision model. Cheap, fast,
# accurate enough for the "did this iter make visible progress" check.
# Override via env VISION_JUDGE_MODEL if needed.
_DEFAULT_VISION_MODEL = "claude-sonnet-4-6"
# Hard ceiling — the judge's reply is two short lines, never more.
_JUDGE_MAX_TOKENS = 200
# Per-call timeout. The judge shouldn't block the agent loop.
_JUDGE_TIMEOUT_S = 30.0


@dataclass
class VisionVerdict:
    """One judge call's verdict."""

    progress: bool | None   # True = closer, False = not closer, None = couldn't judge
    note: str               # one-sentence "what's still missing"
    raw: str                # full model reply (kept for trace + debugging)
    model: str              # which model produced the verdict


def is_enabled() -> bool:
    """Honor the kill-switch. `VISION_JUDGE=0` disables this entirely."""
    val = (os.environ.get("VISION_JUDGE") or "").strip().lower()
    if val in {"0", "false", "off", "no"}:
        return False
    return True


def _judge_prompt(goal: str, has_prev: bool) -> str:
    """Build the text portion of the judge prompt. Kept tiny on purpose
    — the model only needs to see what to compare and how to answer."""
    if has_prev:
        compare = (
            "Two screenshots are attached. The first is the previous "
            "iteration; the second is the current iteration. Decide "
            "whether the current iteration is visibly closer to the "
            "user's goal than the previous one."
        )
    else:
        compare = (
            "One screenshot is attached — the current iteration (no "
            "previous to compare against). Decide whether what you "
            "see is plausibly on track to become the user's goal."
        )
    return (
        f"USER'S GOAL: {goal.strip()}\n\n{compare}\n\n"
        "Answer in exactly this format, no extra prose:\n"
        "PROGRESS: yes | no | unclear\n"
        "MISSING: <one short sentence naming the most important "
        "thing still visibly missing or wrong, OR 'nothing obvious' "
        "if the game looks like the goal>\n"
    )


def _parse(raw: str) -> tuple[bool | None, str]:
    """Pull PROGRESS and MISSING out of the model reply. Tolerant of
    minor whitespace/case variation."""
    progress: bool | None = None
    note = ""
    m = re.search(r"PROGRESS\s*:\s*(\w+)", raw, re.IGNORECASE)
    if m:
        v = m.group(1).strip().lower()
        if v == "yes":
            progress = True
        elif v == "no":
            progress = False
        # "unclear" stays None
    m = re.search(r"MISSING\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        note = m.group(1).strip()
        # Collapse internal whitespace.
        note = re.sub(r"\s+", " ", note)
    return progress, note


def _png_block(image_bytes: bytes) -> dict:
    """Anthropic vision message-content block for an inline PNG."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(image_bytes).decode("ascii"),
        },
    }


async def _anthropic_judge(
    *,
    goal: str,
    current_png: bytes,
    previous_png: bytes | None,
    model: str,
) -> VisionVerdict:
    """One-shot vision call against Anthropic. Caller wraps with timeout
    + try/except, so this can raise freely."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed (pip install 'anthropic>=0.40')"
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = AsyncAnthropic()
    content: list[dict] = []
    if previous_png is not None:
        content.append(_png_block(previous_png))
    content.append(_png_block(current_png))
    content.append({"type": "text", "text": _judge_prompt(goal, previous_png is not None)})
    msg = await client.messages.create(
        model=model,
        max_tokens=_JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": content}],
    )
    # Anthropic returns a list of content blocks; the text we want
    # lives in any blocks with type=="text". Concat to be safe.
    parts: list[str] = []
    for block in (msg.content or []):
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    raw = "".join(parts).strip()
    progress, note = _parse(raw)
    return VisionVerdict(progress=progress, note=note, raw=raw, model=model)


# ---- Local MLX VLM path (added 2026-05-15) ---------------------------
# When the user types `/check with <local-vlm>`, the model name is the
# MLX path or substring of one (e.g. "qwen3.6-27b" or
# "/Users/.../Qwen3.6-27B-mxfp8"). We load via mlx_vlm.load on first
# call, cache (model, processor) for subsequent calls, and run a one-
# shot generate with the screenshot(s) on disk.
#
# Memory note: this loads the FULL VLM weights into Metal VRAM, in
# addition to whatever the main MLXBackend has loaded. A 27B mxfp8
# model is ~27 GB of weights. If your Mac is tight on unified memory,
# `/unload mlx` the main session model before calling /check, or use
# a smaller VLM (Qwen2.5-VL-7B etc.).

_MLX_VLM_CACHE: dict[str, tuple] = {}  # path -> (model, processor, config)


def _resolve_local_mlx_vlm(query: str) -> str | None:
    """Resolve a `/check with <name>` query to a local MLX VLM path.

    Accepts:
      - an absolute path to a directory (returns it verbatim)
      - a basename or substring — we scan the same dirs MLXBackend
        scans (`~/MLX_Models`, `MLX_MODELS_DIR`, HF cache) and
        return the first match whose name classifies as VLM.
    Returns None if no match.
    """
    import os as _os
    from pathlib import Path as _Path
    try:
        from backend import classify_model_modality, list_mlx_inventory  # type: ignore
    except Exception:
        return None
    # Direct path.
    if _os.sep in query and _Path(query).is_dir():
        return query
    q = query.lower()
    try:
        downloaded, _loaded = list_mlx_inventory()
    except Exception:
        downloaded = []
    for entry in downloaded:
        name = entry.split("/")[-1] if "/" in entry else entry
        if q in entry.lower() and classify_model_modality(name) == "vlm":
            return entry
    return None


def _mlx_vlm_judge_sync(
    *, goal: str, current_png: bytes, previous_png: bytes | None,
    model_path: str,
) -> VisionVerdict:
    """Blocking — caller wraps in `run_in_executor` so it doesn't
    block the asyncio event loop. Loads (or reuses cached) mlx_vlm
    pipeline, writes screenshots to temp files, runs `generate`,
    parses the verdict.
    """
    import tempfile
    from pathlib import Path
    # Lazy import — mlx_vlm is optional. If it's not installed, the
    # caller sees the ImportError and treats this judge backend as
    # unavailable (returns None).
    from mlx_vlm import generate as _vlm_generate, load as _vlm_load  # type: ignore
    from mlx_vlm.prompt_utils import apply_chat_template as _vlm_template  # type: ignore
    from mlx_vlm.utils import load_config as _vlm_load_config  # type: ignore

    cached = _MLX_VLM_CACHE.get(model_path)
    if cached is None:
        model_obj, processor = _vlm_load(model_path)
        config = _vlm_load_config(model_path)
        _MLX_VLM_CACHE[model_path] = (model_obj, processor, config)
    else:
        model_obj, processor, config = cached

    image_paths: list[str] = []
    tmp_dir = tempfile.mkdtemp(prefix="vlm_judge_")
    try:
        if previous_png is not None:
            p1 = Path(tmp_dir) / "prev.png"
            p1.write_bytes(previous_png)
            image_paths.append(str(p1))
        p2 = Path(tmp_dir) / "current.png"
        p2.write_bytes(current_png)
        image_paths.append(str(p2))

        text_prompt = _judge_prompt(goal, has_prev=previous_png is not None)
        templated = _vlm_template(
            processor, config, text_prompt,
            num_images=len(image_paths),
        )
        result = _vlm_generate(
            model_obj, processor, templated,
            image=image_paths if len(image_paths) > 1 else image_paths[0],
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=0.0,
            verbose=False,
        )
        # mlx_vlm.generate returns a GenerationResult; the text is on
        # `.text` (current 0.5.0 API). Fall back to str() for forward
        # compat if the field name shifts.
        raw = (getattr(result, "text", None) or str(result) or "").strip()
        progress, note = _parse(raw)
        return VisionVerdict(
            progress=progress, note=note, raw=raw, model=model_path,
        )
    finally:
        # Best-effort cleanup of temp images. If it fails the OS will
        # reclaim on reboot — not worth crashing the judge over.
        try:
            for p in image_paths:
                try:
                    Path(p).unlink()
                except Exception:
                    pass
            Path(tmp_dir).rmdir()
        except Exception:
            pass


async def _mlx_vlm_judge(
    *, goal: str, current_png: bytes, previous_png: bytes | None,
    model_path: str,
) -> VisionVerdict:
    """Async wrapper — runs the blocking mlx_vlm call on a thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _mlx_vlm_judge_sync(
            goal=goal, current_png=current_png,
            previous_png=previous_png, model_path=model_path,
        ),
    )


def _looks_like_local_mlx(model: str) -> bool:
    """Heuristic: anything that isn't obviously a cloud model name
    is treated as a local MLX query. `/check with claude-...` /
    `/check with gpt-...` stay on the cloud paths.
    """
    low = model.lower()
    if low.startswith("claude") or low.startswith("anthropic"):
        return False
    if low.startswith("gpt") or low.startswith("openai"):
        return False
    if low.startswith("o1-") or low.startswith("o3-") or low.startswith("o4-"):
        return False
    return True


async def judge_visual_progress(
    *,
    goal: str,
    current_png: bytes,
    previous_png: bytes | None = None,
    model: str | None = None,
) -> VisionVerdict | None:
    """Run the vision judge. Returns None when judging isn't available
    or the call failed — caller should treat that as "no signal", NOT
    as a regression.

    Routes by `model`:
      - cloud (claude-*, gpt-*, o*-) → Anthropic / (future OpenAI)
      - anything else → local MLX VLM via mlx_vlm. Substring resolved
        against `list_mlx_inventory()` and only used if the resolved
        model classifies as a VLM.
    """
    if not is_enabled():
        return None
    if not current_png:
        return None
    use_model = model or os.environ.get("VISION_JUDGE_MODEL") or _DEFAULT_VISION_MODEL

    # Local MLX VLM path.
    if _looks_like_local_mlx(use_model):
        resolved = _resolve_local_mlx_vlm(use_model)
        if resolved is None:
            return None
        try:
            return await asyncio.wait_for(
                _mlx_vlm_judge(
                    goal=goal,
                    current_png=current_png,
                    previous_png=previous_png,
                    model_path=resolved,
                ),
                # Local VLM load is slow on first call (~30-60s cold);
                # generate itself runs ~2-10s. Bump the overall ceiling
                # so cold-load doesn't get killed mid-load.
                timeout=180.0,
            )
        except Exception:
            return None

    # Cloud path (default — Anthropic).
    try:
        return await asyncio.wait_for(
            _anthropic_judge(
                goal=goal,
                current_png=current_png,
                previous_png=previous_png,
                model=use_model,
            ),
            timeout=_JUDGE_TIMEOUT_S,
        )
    except Exception:
        # Never let the judge crash the run. Network blip, missing key,
        # API rate limit — the agent must keep working without us.
        return None
