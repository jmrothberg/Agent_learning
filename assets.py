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

# Where the user's diffusion_manager.py lives. Adding to sys.path
# (instead of vendoring) means upstream improvements to that script
# carry over automatically — single source of truth for the diffuser.
_DEFAULT_DIFFUSER_DIR = "/home/jonathan/Colossal_Cave"

# Per-asset default target size. Sprites are typically 32-128 px; 128
# is a good middle ground, can be overridden per-asset by the model.
_DEFAULT_TARGET_SIZE = 128

# Cap so a chatty plan can't trigger 50 generations.
_MAX_ASSETS_PER_TURN = 8

_ASSETS_RE = re.compile(
    r"<assets>\s*(.*?)\s*</assets>", re.DOTALL | re.IGNORECASE,
)


def parse_assets_block(reply: str) -> list[dict]:
    """Extract the JSON list inside <assets>...</assets>.

    Tolerant of fenced ```json wrappers (some models love adding them).
    Returns [] if the tag is missing or the JSON is malformed; the
    caller should treat empty as "model didn't request assets" and
    skip the pipeline.

    Each returned dict has keys: name (str), prompt (str), size
    (tuple[int, int]). Specs missing name OR prompt are dropped.
    """
    if not reply:
        return []
    m = _ASSETS_RE.search(reply)
    if not m:
        return []
    body = m.group(1).strip()
    body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
    body = re.sub(r"\n?```$", "", body).strip()
    try:
        obj = json.loads(body)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
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


def try_load_image_generator(model_id: str = "Z-Image-Turbo",
                             diffuser_dir: str = _DEFAULT_DIFFUSER_DIR):
    """Import the user's diffusion_manager.py and instantiate
    ImageGenerator. Returns None if anything is unavailable — the
    caller MUST treat None as "skip asset generation, proceed without."

    Failure modes (all returning None silently): diffuser_dir not on
    disk, the import fails (no diffusers / no torch), no CUDA, or
    pipeline construction itself raises.
    """
    if diffuser_dir and diffuser_dir not in sys.path:
        sys.path.insert(0, diffuser_dir)
    try:
        from diffusion_manager import ImageGenerator  # type: ignore
    except Exception:
        return None
    try:
        return ImageGenerator(model_id=model_id)
    except Exception:
        return None


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
    for spec in specs:
        name = _safe_filename(spec["name"])
        prompt = spec["prompt"]
        size = spec["size"]
        key = _cache_key(model_id, prompt, size)
        cache_path = cache_root / f"{key}.png"
        target_path = session_dir / f"{name}.png"
        if cache_path.exists():
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            continue
        # Cache miss — generate.
        gen_path = _safe_generate(image_generator, prompt)
        if gen_path is None:
            continue
        try:
            from PIL import Image
            with Image.open(gen_path) as src_img:
                src_img.load()
                if size != (src_img.width, src_img.height):
                    resized = src_img.resize(size, Image.LANCZOS)
                else:
                    resized = src_img
                resized.save(cache_path, format="PNG")
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
        except Exception:
            # Couldn't post-process this one; skip it. Other assets in
            # the batch still proceed.
            continue
    return out


def _safe_generate(gen: Any, prompt: str) -> str | None:
    """Wrap ImageGenerator.generate(prompt) so a single failure (OOM,
    NSFW filter, network) doesn't poison the whole batch."""
    try:
        return gen.generate(prompt)
    except Exception:
        return None


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


def render_asset_paths_block(
    asset_paths: dict[str, Path], session_html_path: Path | str,
) -> str:
    """Build the injection block listing generated asset paths.

    Paths are resolved relative to the directory of the HTML file so
    the model can `<img src="./<name>.png">` directly. Empty input →
    empty string (caller should not inject).
    """
    if not asset_paths:
        return ""
    html_dir = Path(session_html_path).resolve().parent
    lines = [
        "================ GENERATED ASSETS (sprites) ================",
        "Z-Image-Turbo generated these PNGs and saved them next to your",
        "HTML file. Reference them via <img> or `new Image()`. ALWAYS",
        "wait for `await img.decode()` (or onload) before drawing — see",
        "playbook bullet image-load-race.",
        "",
    ]
    for name, path in asset_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"  - {name}: ./{rel}")
    lines.append(
        "============================================================"
    )
    return "\n".join(lines)
