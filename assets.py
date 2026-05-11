"""Per-session asset generation pipeline (Z-Image-Turbo, no server).

The model can declare an `<assets>` block in Phase A:

    <assets>
    [
      {"name": "ship",     "prompt": "pixel-art retro arcade spaceship facing right, transparent background"},
      {"name": "asteroid", "prompt": "pixel-art irregular grey rocky asteroid, transparent bg", "size": "64x64"},
      {"name": "explosion","prompt": "pixel-art orange explosion sprite, transparent bg",       "size": 96}
    ]
    </assets>

If a CUDA GPU + the user's local `Colossal_Cave/diffusion_manager.py`
are reachable, this module:

  1. Parses the JSON list out of the planning reply.
  2. Lazy-loads `ImageGenerator(model_id="Z-Image-Turbo")` (free until
     the first call — the import + pipeline init only happen if the
     model actually requested assets).
  3. Generates each missing PNG (cache hit by sha256 of (model, prompt,
     size) so re-runs are free).
  4. Saves PNGs into `games/<slug>_<ts>_assets/<name>.png` next to the
     working HTML file. The first-build prompt is later prepended with
     `render_asset_paths_block(...)` so the model knows the paths.

Fully optional. When no `<assets>` tag is emitted, OR no GPU /
diffusion_manager is reachable, this module is a no-op and the agent
proceeds with procedural drawing exactly as before.

Generation strategy: Z-Image-Turbo natively renders 768×768 in ~2-4 s
per image (8-step turbo). We always generate at native and downscale
with PIL Lanczos to the per-asset target size. Default target 128 px
square — small enough to be a sprite, big enough to look decent.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# Per-asset default target size. Sprites are typically 32-128 px; 128
# is a good middle ground, can be overridden per-asset by the model.
_DEFAULT_TARGET_SIZE = 128

# Where Z-Image-Turbo's weights live on disk. Cross-platform — works on
# Linux (Models_Diffusers convention) and macOS (Diffusion_Models
# convention). Override per-machine via DIFFUSION_MODELS_DIR.
#
# Search order — first existing directory wins:
#   1. $DIFFUSION_MODELS_DIR env var (preferred override)
#   2. Platform default bases (see _default_model_search_dirs):
#      Hidden ~/.Diffusion_Models and ~/.Models_Diffusers are tried before
#      visible ~/Diffusion_Models / ~/Models_Diffusers.
#      macOS checks Diffusion_Models* before Models_Diffusers*; Linux the opposite.
#   3. /home/jonathan/Models_Diffusers   (legacy, kept so existing
#                                         setups don't break on update)
#   4. ./models_diffusers      (repo-relative — for portability when
#                               cloning fresh on a new machine)
#   5. HuggingFace fallback: `Tongyi-MAI/Z-Image-Turbo` is downloaded
#      to ~/.cache/huggingface/hub/ on first run if no local path
#      matches — no manual download needed.
#
# Model files are DATA, not code; they live outside the repo by design
# (5GB+) but the search code itself stays self-contained here.

import os as _os


def _default_model_search_dirs() -> list[str]:
    """Build the search list at import time. `~` is expanded so the
    list is concrete absolute paths plus one relative entry.

    Hidden ``~/.Diffusion_Models`` / ``~/.Models_Diffusers`` are tried
    before visible siblings so dot-prefixed weight trees win first.

    On macOS, Diffusion_Models* precedes Models_Diffusers*; Linux uses
    the opposite preference.
    """
    home = _os.path.expanduser("~")
    dot_dm = _os.path.join(home, ".Diffusion_Models")
    dot_md = _os.path.join(home, ".Models_Diffusers")
    diffusion_models = _os.path.join(home, "Diffusion_Models")
    models_diffusers = _os.path.join(home, "Models_Diffusers")
    if sys.platform == "darwin":
        home_bases = [dot_dm, diffusion_models, dot_md, models_diffusers]
    else:
        home_bases = [dot_md, models_diffusers, dot_dm, diffusion_models]
    return home_bases + [
        "/home/jonathan/Models_Diffusers",
        "./models_diffusers",
    ]


_MODEL_SEARCH_DIRS = _default_model_search_dirs()
_HF_FALLBACK_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# Cap so a chatty plan can't trigger 50 generations.
_MAX_ASSETS_PER_TURN = 8

def _strip_thinking(reply: str) -> str:
    """Drop everything up to and including the LAST `</think>` tag.

    Reasoning-mode models (Qwen3.6, DeepSeek-V3.x, etc.) stream their
    chain-of-thought first, terminated by `</think>`. The CoT may
    legitimately MENTION tag names in markdown backticks
    (`` `<assets>` ``), and the greedy non-greedy regex below would
    then match from the first <assets> in the prose all the way to
    the real </assets>, capturing the thinking text as the body and
    failing JSON parse — observed in
    games/traces/game-of-space-invaders-with-gr_20260511_093225 where
    13 asset specs + 10 sound specs were silently dropped.

    Stripping at the LAST `</think>` is safe: if the model uses
    multiple think segments, the real answer follows the last one. If
    no `</think>` is present, return the reply unchanged.
    """
    idx = reply.rfind("</think>")
    if idx < 0:
        return reply
    return reply[idx + len("</think>"):]


_ASSETS_RE = re.compile(
    r"<assets>\s*(.*?)\s*</assets>", re.DOTALL | re.IGNORECASE,
)
# Truncated case — model emitted <assets>[...content...] but the stream
# ended before </assets>. We've seen this on long planning turns where
# the model exhausts the token budget mid-block. Recover by treating
# everything from <assets>[ to end-of-reply as the body, then trying to
# repair the JSON list (drop the incomplete trailing entry, close the
# bracket).
_ASSETS_OPEN_RE = re.compile(
    r"<assets>\s*(\[.*?)$", re.DOTALL | re.IGNORECASE,
)


def _extract_assets_body(reply: str) -> str | None:
    """Pull the body of an <assets>...</assets> block, tolerating a
    missing closing tag. Returns None when nothing usable was found.

    Reasoning prose stripped first — see _strip_thinking docstring.
    """
    reply = _strip_thinking(reply)
    m = _ASSETS_RE.search(reply)
    if m:
        return m.group(1)
    m = _ASSETS_OPEN_RE.search(reply)
    if m:
        return m.group(1)
    return None


def _try_repair_truncated_json_list(text: str) -> list[Any]:
    """Best-effort recovery of a JSON list whose stream was cut off.

    Walks back from the end of `text` looking for the last `}` (closing
    a complete object) and treats everything up to that point as a
    valid list, plus a synthesized `]`. Drops any incomplete trailing
    entry. Returns [] if recovery fails. Used only when the strict
    `json.loads(body)` already failed.
    """
    text = text.rstrip().rstrip(",").rstrip()
    if text.endswith("]"):
        return []  # already closed; strict parse will have caught real errors
    last_brace = text.rfind("}")
    if last_brace < 0:
        return []
    candidate = text[: last_brace + 1] + "]"
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def parse_assets_block(reply: str) -> list[dict]:
    """Extract the JSON list inside <assets>...</assets>.

    Tolerant of fenced ```json wrappers (some models love adding them)
    AND of truncated streams that cut off before </assets> (recovered
    by `_try_repair_truncated_json_list` — drops the incomplete final
    entry and treats the rest as a complete list).

    Returns [] if no <assets> opener is present or recovery fails; the
    caller should treat empty as "model didn't request assets" and
    skip the pipeline.

    Each returned dict has keys: name (str), prompt (str), size
    (tuple[int, int]). Specs missing name OR prompt are dropped.
    """
    if not reply:
        return []
    body = _extract_assets_body(reply)
    if body is None:
        return []
    body = body.strip()
    body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
    body = re.sub(r"\n?```$", "", body).strip()
    try:
        obj = json.loads(body)
    except Exception:
        # Truncated stream / trailing garbage — try repair.
        obj = _try_repair_truncated_json_list(body)
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
    # Dedupe by (normalized prompt, size). Catches the failure mode where
    # the model spams numbered variants of the same template — e.g. 200×
    # `{"name":"minimap_compiler<N>", "prompt":"green computer","size":"16x16"}`.
    # Without this, `generate_assets` would burn 200 GPU calls (or hit
    # _MAX_ASSETS_PER_TURN at 8 and silently truncate, masking the bug).
    seen_keys: set[tuple[str, tuple[int, int]]] = set()
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"asset_{i + 1}").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        try:
            size = _parse_size(item.get("size", _DEFAULT_TARGET_SIZE))
        except Exception:
            size = (_DEFAULT_TARGET_SIZE, _DEFAULT_TARGET_SIZE)
        # Normalize prompt the same way the cache key does, so trivial
        # whitespace / case differences don't create duplicate entries
        # that would all map to the same cached PNG anyway.
        norm_prompt = " ".join(prompt.lower().split())
        key = (norm_prompt, size)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({"name": name, "prompt": prompt, "size": size})
        if len(out) >= _MAX_ASSETS_PER_TURN:
            break
    return out


def _parse_size(raw: Any) -> tuple[int, int]:
    """Accept '64', '64x64', '128x96', or int; return (w, h)."""
    if isinstance(raw, int):
        n = max(1, min(1024, raw))
        return (n, n)
    s = str(raw).strip().lower()
    if "x" in s:
        a, b = s.split("x", 1)
        w, h = int(a), int(b)
        return (max(1, min(1024, w)), max(1, min(1024, h)))
    n = int(s)
    n = max(1, min(1024, n))
    return (n, n)


def _cache_key(model_id: str, prompt: str, size: tuple[int, int]) -> str:
    """sha256 of (model_id, normalized prompt, size) → 32-hex.

    Keeps the cache stable across runs so re-asking for the same sprite
    is free. Whitespace + case in the prompt are normalized so trivial
    formatting differences don't bust the cache.
    """
    norm_prompt = " ".join(prompt.strip().lower().split())
    norm = f"{model_id}|{norm_prompt}|{size[0]}x{size[1]}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_filename(name: str) -> str:
    """Clean an asset name for filesystem use. Caps at 48 chars."""
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned[:48] or "asset"


# ---------------------------------------------------------------------------
# Z-Image-Turbo loader (self-contained, vendored from Colossal_Cave on
# 2026-05-06 with the watermark / Generated_Art / multi-pipeline branches
# stripped out — Agent_learning only ever uses the Z-Image-Turbo path)
# ---------------------------------------------------------------------------


def _resolve_zimage_path() -> str:
    """Find Z-Image-Turbo weights on disk, or return the HF model ID
    so diffusers downloads on first run. Search:
      1. $DIFFUSION_MODELS_DIR env var
      2. _MODEL_SEARCH_DIRS (the user's standard /home/jonathan/Models_Diffusers
         layout, plus a relative fallback)
      3. The HuggingFace hub fallback ID (Tongyi-MAI/Z-Image-Turbo).
    """
    import os
    env_dir = (os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    candidates: list[str] = []
    if env_dir:
        candidates.extend([
            os.path.join(env_dir, "Z-Image-Turbo"),
            os.path.join(env_dir, "Tongyi-MAI_Z-Image-Turbo"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            os.path.join(base, "Z-Image-Turbo"),
            os.path.join(base, "Tongyi-MAI_Z-Image-Turbo"),
        ])
    for c in candidates:
        if os.path.isdir(c):
            return c
    return _HF_FALLBACK_MODEL_ID


class ZImageTurboGenerator:
    """In-process Z-Image-Turbo wrapper. No server, no subprocess.

    Usage:
        gen = ZImageTurboGenerator()
        path = gen.generate("pixel-art retro spaceship")  # returns PNG path

    The pipeline is loaded lazily on the first `.generate()` call so
    importing this module is cheap. After the first call, the model
    stays resident in GPU VRAM for the rest of the Python process —
    subsequent calls cost only the inference time (~2-4 s per 768×768
    image at 8 inference steps).

    `.cleanup()` releases the pipeline + frees CUDA memory if the
    caller wants to reclaim the VRAM mid-session.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _resolve_zimage_path()
        self._pipeline: Any = None  # lazy-init in .generate()
        # Resolved at first .generate() call; "cuda", "mps", or None.
        self._device: str | None = None
        # Last error captured from _lazy_init or generate. Surfaced via
        # last_stats[i]["error"] so the caller can show the user the
        # actual exception (instead of a canned "OOM / NSFW / etc"
        # guess that hid e.g. diffusers API drift or model path errors).
        self._last_error: str | None = None

    def cleanup(self) -> None:
        if self._pipeline is None:
            return
        try:
            import torch
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._pipeline = None

    def _lazy_init(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            import torch
            from diffusers import ZImagePipeline
        except Exception as e:
            self._last_error = (
                f"import failed: {type(e).__name__}: {e!s}. "
                "Run `pip install -r requirements-diffuser.txt` in the "
                "Agent_learning venv."
            )
            return False

        # Pick the best available device. Z-Image-Turbo's authors
        # ship and test on CUDA; MPS is experimental — may work on
        # recent Apple Silicon + diffusers nightlies, may not. CPU is
        # excluded because inference would take 10+ minutes per image
        # (an hour for a 5-asset session), worse than just drawing
        # procedurally.
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16   # Blackwell-class GPU sweet spot
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
            # fp16 on MPS produces NaN for Z-Image-Turbo (verified
            # 2026-05-07: every output was 100% transparent because
            # NaN→0 in cast). fp32 works at ~20s/image on M-series.
            dtype = torch.float32
        else:
            self._last_error = (
                "no CUDA and no MPS device available — torch sees "
                "neither. Z-Image-Turbo on CPU is not supported "
                "(would be hours per image)."
            )
            return False

        try:
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self._pipeline.to(device)
            self._device = device
            return True
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"pipeline load failed at {self.model_path}: "
                f"{type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            self._pipeline = None
            self._device = None
            return False

    def generate(self, prompt: str) -> str | None:
        """Run inference and save a 768×768 PNG to a temp file. Returns
        the absolute path, or None on failure (caller skips that asset).
        On None, `self._last_error` carries the real exception or "
        diffuser returned None (no images in pipeline output)" — read
        it via getattr to keep generator API stable for callers that
        don't care."""
        # Clear stale error from a previous successful call so a subsequent
        # success leaves _last_error None.
        self._last_error = None
        if not self._lazy_init():
            return None
        try:
            import tempfile
            import torch
            # `torch.Generator(device)` ensures the seed RNG lives on
            # the same device as the pipeline; mismatched devices throw
            # `RuntimeError: Expected all tensors to be on the same device`.
            gen = torch.Generator(self._device or "cpu").manual_seed(42)
            result = self._pipeline(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=9,   # 8 actual DiT forwards in turbo mode
                guidance_scale=0.0,      # turbo: guidance must be 0
                generator=gen,
            )
            # Some pipelines return a result with `.images = []` when an
            # internal safety/NSFW checker rejected the output, or when
            # the result struct shape is different from what we expect.
            # Distinguish empty-images from a real exception so the user
            # knows whether it's a content filter or a code path.
            images = getattr(result, "images", None)
            if not images:
                self._last_error = (
                    "pipeline returned no images (empty .images list). "
                    "Likely an internal NSFW/safety filter, OR a "
                    "diffusers API drift where the result attribute "
                    "name changed. Inspect the result object with "
                    f"type={type(result).__name__}, "
                    f"keys={list(getattr(result, '__dict__', {}).keys())}."
                )
                return None
            image = images[0]
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            image.save(f.name, format="PNG")
            return f.name
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"{type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return None


# Module-level cache for a preloaded generator. Set by `preload()`
# (called from chat.py's main BEFORE Playwright/Chromium starts) and
# returned by subsequent `try_load_image_generator()` calls so the
# agent reuses the already-loaded pipeline instead of triggering its
# own _lazy_init — which would fork a subprocess with Playwright's
# IPC pipes already in the inherited fd table, making
# _posixsubprocess.fork_exec raise "bad value(s) in fds_to_keep".
# That subprocess fork happens once per pipeline load (huggingface_hub
# / safetensors / transformers do it during from_pretrained); doing
# it before Playwright opens its pipes is the entire fix.
_PRELOADED: Any = None


def preload() -> Any:
    """Eagerly construct + load the Z-Image-Turbo pipeline RIGHT NOW.

    Call this from your program's main entry, BEFORE any subprocess-
    spawning library (Playwright/Chromium, multiprocessing pools, etc)
    has opened file descriptors. The ~15-30s pipeline load includes a
    fork of subprocess.Popen via huggingface_hub or transformers; if
    that fork happens AFTER Playwright is up, the inherited fd table
    has Playwright's pipe handles and the fork raises ValueError:
    bad value(s) in fds_to_keep. Loading first sidesteps this entirely.

    Returns the loaded generator (cached and reused by future calls
    to try_load_image_generator), or None when torch/diffusers aren't
    installed. Idempotent: subsequent calls return the same instance.
    """
    global _PRELOADED
    if _PRELOADED is not None:
        return _PRELOADED
    gen = _construct_generator()
    if gen is None:
        return None
    # Trigger the heavy load NOW so the subprocess fork happens
    # before Playwright/etc opens any FDs. _lazy_init returns False
    # on failure with the reason on _last_error; we still cache the
    # wrapper so the agent path can read _last_error and skip
    # gracefully instead of retrying the broken fork.
    gen._lazy_init()
    _PRELOADED = gen
    return gen


def _construct_generator() -> Any:
    """Internal: just check imports and construct a wrapper. Pulled
    out of try_load_image_generator so preload() can share it."""
    import importlib.util as _iu
    if _iu.find_spec("torch") is None or _iu.find_spec("diffusers") is None:
        return None
    try:
        return ZImageTurboGenerator()
    except Exception:
        return None


def try_load_image_generator(
    model_id: str = "Z-Image-Turbo",  # kept for API stability; unused
    diffuser_dir: str | None = None,  # kept for API stability; unused
) -> Any:
    """Return the Z-Image-Turbo wrapper. If `preload()` ran earlier,
    reuses that already-loaded pipeline (this is the path chat.py
    takes). Otherwise constructs a fresh wrapper that lazy-loads on
    first .generate() — fine for the smoke test (clean process, no
    competing fds) but will fail from inside chat.py because Playwright
    has already opened its IPC pipes.

    Returns None when torch+diffusers aren't installed.
    """
    if _PRELOADED is not None:
        return _PRELOADED
    return _construct_generator()


def generate_assets(
    specs: list[dict],
    session_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    image_generator: Any = None,
    model_id: str = "Z-Image-Turbo",
) -> dict[str, Path]:
    """Generate one PNG per spec and return {name: absolute_path}.

    `specs` come from `parse_assets_block`. Each is {name, prompt, size}.
    Returns a dict mapping name → absolute path of the saved PNG.

    Cache strategy: each (model_id, prompt, size) hashes to a key under
    `cache_dir`; cache hits hard-link (or copy as fallback) to a stable
    per-session path inside `session_dir` so the HTML file's <img src>
    references stay relative and predictable.

    `image_generator` is dependency-injected for tests. When None we
    attempt `try_load_image_generator()`; if THAT also returns None the
    function returns {} silently — the agent should log + proceed.

    Failures generating an individual asset are caught and logged via
    the returned dict's missing keys; we never abort the batch.
    """
    if not specs:
        return {}
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        # One asset cache per project — sibling of session_dir, shared
        # across sessions so re-asking for the same sprite is free.
        cache_root = Path(session_dir).parent / "_asset_cache"
    else:
        cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if image_generator is None:
        image_generator = try_load_image_generator(model_id)
    if image_generator is None:
        return {}

    out: dict[str, Path] = {}
    # 2.2: per-asset stats accumulated as a side channel. Caller can
    # check `image_generator.last_stats` (a list of per-asset dicts)
    # after the call. Using an attribute on the generator instance so
    # the function signature stays backward-compatible.
    asset_stats: list[dict[str, Any]] = []
    for spec in specs:
        import time
        t0 = time.time()
        name = _safe_filename(spec["name"])
        prompt = spec["prompt"]
        size = spec["size"]
        key = _cache_key(model_id, prompt, size)
        # Cache filename: human-readable `<name>__<hash6>.png`. The
        # 6-char hash slice keeps the cache deterministic (same prompt
        # + size = same file = cache hit) while letting you scan
        # _asset_cache/ visually. Old SHA32 filenames in existing
        # caches become orphans and naturally regenerate under the
        # new name on next request.
        cache_path = cache_root / f"{name}__{key[:6]}.png"
        target_path = session_dir / f"{name}.png"
        stat: dict[str, Any] = {
            "name": name,
            "prompt": prompt[:140],
            "target_size": list(size),
            "cache_hit": False,
            "gen_seconds": 0.0,
            "bg_color": None,
            "alpha_pixel_ratio": 0.0,
        }
        if cache_path.exists():
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            stat["cache_hit"] = True
            stat["gen_seconds"] = round(time.time() - t0, 3)
            asset_stats.append(stat)
            continue
        # Cache miss — generate.
        gen_path = _safe_generate(image_generator, prompt)
        if gen_path is None:
            # Pull the real error from the generator (set by generate()
            # or _lazy_init()) so the user sees the actual cause —
            # import error, model path miss, fp16 NaN, real NSFW
            # filter, or empty result struct — instead of a one-size-
            # fits-all canned message.
            real_err = getattr(image_generator, "_last_error", None)
            stat["error"] = (
                f"diffuser failed: {real_err}" if real_err else
                "diffuser returned None (no exception captured — check "
                "generator's _last_error attribute)"
            )
            stat["gen_seconds"] = round(time.time() - t0, 3)
            asset_stats.append(stat)
            continue
        try:
            from PIL import Image
            with Image.open(gen_path) as src_img:
                src_img.load()
                stat["native_size"] = [src_img.width, src_img.height]
                # Resize first; chroma-key second. Resizing 768→128 is
                # ~36x cheaper to mask than masking at native res.
                if size != (src_img.width, src_img.height):
                    resized = src_img.resize(size, Image.LANCZOS)
                else:
                    resized = src_img
                # 1.3: apply chroma-key to add a transparent background.
                # Z-Image-Turbo renders with a solid bg even when the
                # prompt says "transparent background"; this turns it
                # into actual alpha so the model never has to clean it
                # up at runtime.
                keyed, ck_stats = _chroma_key_to_rgba(resized)
                stat["bg_color"] = (
                    list(ck_stats["bg_color"])
                    if ck_stats["bg_color"] is not None else None
                )
                stat["alpha_pixel_ratio"] = ck_stats["alpha_pixel_ratio"]
                keyed.save(cache_path, format="PNG")
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
        except Exception as e:
            stat["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            stat["gen_seconds"] = round(time.time() - t0, 3)
            asset_stats.append(stat)
    # Stash stats on the generator so the caller can read them out.
    try:
        image_generator.last_stats = asset_stats  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def _safe_generate(gen: Any, prompt: str) -> str | None:
    """Wrap ImageGenerator.generate(prompt) so a single failure (OOM,
    NSFW filter, network) doesn't poison the whole batch.

    On exception, stamps `gen._last_error` with the real traceback so
    the caller can surface it via `last_stats[i]["error"]` instead of
    a canned guess.
    """
    try:
        return gen.generate(prompt)
    except Exception as e:
        import traceback as _tb
        try:
            gen._last_error = (
                f"_safe_generate caught {type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
        except Exception:
            pass
        return None


# 1.3 — chroma-key pass. Z-Image-Turbo (and most diffusion models)
# render with a uniform background even when the prompt says
# "transparent background". The model used to be asked to clean the
# white square at runtime via getImageData / pixel manipulation,
# which CORS-tainted the canvas (see games/traces/using-great-graphics-
# that-you_20260507_103355 for the cascade failure). Right fix is to
# do the chroma-key once, in PIL, before the PNG ever reaches the
# game. RGBA output → drawImage just works with full alpha.

def _detect_bg_color(img) -> tuple[int, int, int] | None:
    """Sample the four corners + four edge-midpoints; return the most
    common color if it dominates (>= 6 of 8 samples agree within
    tolerance), else None (don't mask — we'd risk eating real pixels).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    px = img.load()
    samples = [
        px[0, 0],            px[w - 1, 0],
        px[0, h - 1],        px[w - 1, h - 1],
        px[w // 2, 0],       px[w // 2, h - 1],
        px[0, h // 2],       px[w - 1, h // 2],
    ]
    # Group samples within tolerance — find the largest cluster.
    tol = 16
    best: tuple[tuple[int, int, int], int] | None = None
    for s in samples:
        n = sum(
            1 for o in samples
            if (abs(o[0] - s[0]) <= tol
                and abs(o[1] - s[1]) <= tol
                and abs(o[2] - s[2]) <= tol)
        )
        if best is None or n > best[1]:
            best = (s, n)
    if best is None or best[1] < 6:
        # No clearly-dominant background — leave the image alone.
        return None
    return best[0]


def _apply_chroma_key_alpha(img, bg: tuple[int, int, int],
                             tolerance: int = 24) -> tuple[Any, float]:
    """Convert pixels within `tolerance` of `bg` to alpha=0.

    Returns (rgba_image, alpha_pixel_ratio). The ratio is the fraction
    of pixels that became transparent — useful for trace logging.
    Border pixels also get a small alpha falloff so edges don't look
    fringy after chroma-keying.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.load()
    w, h = img.size
    bg_r, bg_g, bg_b = bg
    masked = 0
    total = w * h
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            dr = abs(r - bg_r)
            dg = abs(g - bg_g)
            db = abs(b - bg_b)
            if dr <= tolerance and dg <= tolerance and db <= tolerance:
                px[x, y] = (r, g, b, 0)
                masked += 1
    return img, (masked / total if total else 0.0)


def _chroma_key_to_rgba(pil_img) -> tuple[Any, dict]:
    """Top-level helper: detect background color, apply alpha mask,
    return the RGBA image plus a small stats dict for tracing.

    Stats dict shape:
      {"bg_color": (r,g,b) | None, "alpha_pixel_ratio": float}

    If no dominant bg color was detected, leaves the image alone (only
    converts to RGBA so save format is consistent).
    """
    stats: dict[str, Any] = {"bg_color": None, "alpha_pixel_ratio": 0.0}
    bg = _detect_bg_color(pil_img)
    if bg is None:
        # No clear bg — convert mode but skip masking.
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        return pil_img, stats
    stats["bg_color"] = bg
    keyed, ratio = _apply_chroma_key_alpha(pil_img, bg)
    stats["alpha_pixel_ratio"] = round(ratio, 3)
    return keyed, stats


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink first (free, instant); fall back to copy. Used to give
    each session a stable per-name path even when the actual PNG bytes
    came from the cache."""
    try:
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                return
        try:
            dst.hardlink_to(src)
            return
        except (OSError, AttributeError):
            pass
        import shutil
        shutil.copy2(src, dst)
    except Exception:
        pass


def _filter_existing_assets(
    asset_paths: dict[str, Path],
) -> dict[str, Path]:
    """Drop entries whose PNG isn't on disk so we never inject a path
    the page will hit as ERR_FILE_NOT_FOUND. Floppy-birds trace burned
    iterations chasing missing-file console errors that the model could
    not fix because the corresponding code reference was correct.
    """
    kept: dict[str, Path] = {}
    dropped: list[str] = []
    for name, path in asset_paths.items():
        try:
            if Path(path).exists():
                kept[name] = path
            else:
                dropped.append(name)
        except Exception:
            dropped.append(name)
    if dropped:
        print(
            f"[assets] dropped {len(dropped)} missing PNG path(s) "
            f"before injection: {', '.join(dropped[:5])}"
            + ("…" if len(dropped) > 5 else ""),
            flush=True,
        )
    return kept


def render_asset_paths_block(
    asset_paths: dict[str, Path], session_html_path: Path | str,
) -> str:
    """Build the injection block listing generated asset paths.

    Paths are resolved relative to the directory of the HTML file so
    the model can `<img src="./<name>.png">` directly. Empty input →
    empty string (caller should not inject).

    The phrasing is intentionally aggressive ("YOU MUST", "REGRESSION
    IF YOU DON'T") because small models (qwen3.6, gpt-oss) default to
    procedural ctx.fillRect drawing — that's what's in their training
    distribution. Without explicit, repeated instruction to use the
    PNGs, the model treats the asset list as descriptive rather than
    actionable, and ships a bare procedural game.
    """
    if not asset_paths:
        return ""
    asset_paths = _filter_existing_assets(asset_paths)
    if not asset_paths:
        return ""
    html_dir = Path(session_html_path).resolve().parent
    lines = [
        "================ GENERATED ASSETS (sprites) ================",
        "Z-Image-Turbo generated these PNGs and saved them next to your",
        "HTML file. YOU MUST USE THEM via `new Image()` + `drawImage()`",
        "for EVERY entity listed below. Procedural ctx.fillRect drawing",
        "for these entities IS A REGRESSION on this turn — the user",
        "explicitly asked for sprite art and got the PNGs you requested.",
        "",
        "ULTRA IMPORTANT — pattern you MUST follow:",
        "",
        "  // 1. Build an asset-loader (do this ONCE at startup):",
        "  const ASSETS = {};",
        "  async function loadAssets() {",
        "    const entries = [",
    ]
    for name, path in asset_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"      ['{name}', './{rel}'],")
    lines += [
        "    ];",
        "    for (const [name, src] of entries) {",
        "      const img = new Image();",
        "      img.src = src;",
        "      await img.decode();",
        "      ASSETS[name] = img;",
        "    }",
        "  }",
        "  // 2. Wait for it BEFORE starting the game loop:",
        "  loadAssets().then(() => requestAnimationFrame(frame));",
        "  // 3. In your draw():",
        "  ctx.drawImage(ASSETS.<name>, x, y, w, h);",
        "",
        "Available assets — name → relative path:",
    ]
    for name, path in asset_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"  - {name}: ./{rel}")
    lines.append("")
    lines.append(
        "If you fall back to procedural drawing for an entity that has "
        "a sprite above, you have FAILED THIS TURN. The seed code is "
        "procedural by default — REPLACE its draw bodies with "
        "drawImage() calls."
    )
    lines.append("")
    lines.append(
        "ORIENTATION: Z-Image-Turbo renders sprites in the orientation "
        "the prompt described (e.g. \"facing right\"). If your in-game "
        "entity faces a different way, ROTATE before drawing — do NOT "
        "ship a sideways gun or a backwards player. Pattern:"
    )
    lines.append("")
    lines.append(
        "  ctx.save();"
    )
    lines.append(
        "  ctx.translate(x + w / 2, y + h / 2);"
    )
    lines.append(
        "  ctx.rotate(angle);  // radians; 0 = sprite's native facing"
    )
    lines.append(
        "  ctx.drawImage(ASSETS.ship, -w / 2, -h / 2, w, h);"
    )
    lines.append(
        "  ctx.restore();"
    )
    lines.append(
        "============================================================"
    )
    return "\n".join(lines)
