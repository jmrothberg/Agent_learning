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

# Per-asset default target size, chosen PER PROMPT (2026-05-31): a prompt that
# asks for high-resolution / detailed art defaults to 768 (the diffuser's
# native size — no downscaling, maximum detail); everything else defaults to
# 512 (plenty for most sprites, and ~2.25× faster to generate than 768 — a
# blanket 768 made every project's asset gen slow, e.g. 27 sprites took 7 min).
# A 128-px postage-stamp default was the original mistake (2026-05-23). Always
# overridable per-asset via "size":"32x32" / "size":256 for HUD icons etc.
_DEFAULT_TARGET_SIZE = 512   # back-compat alias; the lo-res default
_LORES_DEFAULT_SIZE = 512
_HIRES_DEFAULT_SIZE = 768
_HIRES_CUES = (
    "high-res", "high res", "hi-res", "hires", "high resolution",
    "high-resolution", "high fidelity", "detailed", "hd", "4k",
    "realistic", "cinematic", "photoreal",
)


def _default_size_for_prompt(prompt: str) -> int:
    """768 when the prompt asks for high-res/detailed art, else 512. Lets a
    'highest resolution' goal get native-size sprites while a plain arcade
    sprite stays fast — without a blanket slowdown for every project."""
    lo = (prompt or "").lower()
    return _HIRES_DEFAULT_SIZE if any(c in lo for c in _HIRES_CUES) else _LORES_DEFAULT_SIZE

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
#   3. /data/Diffusion_Models   (Linux beast — large local weight tree;
#                                 skipped on macOS where this path absent)
#   4. /home/jonathan/Models_Diffusers   (legacy, kept so existing
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
        # Linux workstation: weights live on /data or external data mount,
        # not under ~/. Only prepended on Linux — macOS order unchanged.
        for _linux_dm in (
            "/run/media/jonathan/data/Diffusion_Models",
            "/data/Diffusion_Models",
        ):
            if _os.path.isdir(_linux_dm):
                home_bases = [_linux_dm] + home_bases
    return home_bases + [
        "/home/jonathan/Models_Diffusers",
        "./models_diffusers",
    ]


_MODEL_SEARCH_DIRS = _default_model_search_dirs()
_HF_FALLBACK_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# B1: img2img backbone. SD-Turbo (512×512, 1-4 step) chosen because:
#   - smallest VRAM footprint (~2 GB) of any txt2img+img2img model
#   - fastest on MPS — runs in fp16 with no NaN issues (unlike Z-Image-Turbo)
#   - 512 px is comfortably above our 64-128 px asset target sizes
#   - exact same diffusers API on CUDA and MPS — Linux + Mac with one path
_HF_IMG2IMG_FALLBACK_MODEL_ID = "stabilityai/sd-turbo"

# Cap so a chatty plan can't trigger 50 generations. Bumped from 8
# to 24 after four DK traces (20260513_*) all hit the same failure
# mode: model requested 14 sprites (mario walk/jump/climb x6 + DK x3
# + Pauline x2 + barrel/girder/ladder), only the first 8 were
# generated, the rest were silently dropped. The truncated 6 then
# returned net::ERR_FILE_NOT_FOUND in the browser, and the model
# spent 2-3 iters patching drawImage symptoms before realizing
# files were missing. The dedup-by-(prompt,size) check below already
# protects against the spam pattern (200 numbered variants → 1
# distinct entry), so the cap is purely a runaway guard. 24 covers
# every real-game roster we've seen (DK = 14, Asteroids = 4,
# Centipede = 12, Galaga = 18) with headroom.
_MAX_ASSETS_PER_TURN = 24

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


def parse_assets_block(reply: str, *, max_assets: int | None = None) -> list[dict]:
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

    Use `parse_assets_block_with_meta` if you also need the names of
    entries dropped due to the per-turn cap — the agent uses that to
    coach the model on the overflow instead of silently swallowing it.

    Phase 0.10 — `max_assets` defaults to `_MAX_ASSETS_PER_TURN` (24)
    but can be raised per-session when the user's goal explicitly asks
    for multi-frame rosters (see `prompts_v1._detect_multi_frame_intent`).
    """
    specs, _dropped, _dropped_specs = parse_assets_block_with_meta(
        reply, max_assets=max_assets,
    )
    return specs


def parse_assets_block_with_meta(
    reply: str, *, max_assets: int | None = None,
) -> tuple[list[dict], list[str], list[dict]]:
    """Same as `parse_assets_block` but also returns the names (and full
    specs) of any asset entries parsed-but-dropped due to the per-turn cap.

    Why a separate API: the agent needs to tell the model (and the user)
    when a plan asked for more assets than we'll generate, so the model
    can either split the request across turns or use img2img chaining
    to reduce sprite count. Previously silent — the model thought all
    14 of its requested sprites would exist; only 8 did; the rest 404'd
    in the browser and triggered a 3-iter debugging cascade (4 DK
    traces with this exact pattern).

    Phase 0.10 — `max_assets` is the effective cap for this call. When
    None (default), falls back to module-level `_MAX_ASSETS_PER_TURN`.
    The agent passes a raised cap when the goal contains explicit
    multi-frame language (`prompts_v1._detect_multi_frame_intent`); the
    raise lets a user-requested 12 entities × 3 frames = 36 roster
    land in one turn instead of getting silently truncated to 24.
    """
    effective_cap = _MAX_ASSETS_PER_TURN if max_assets is None else max(1, int(max_assets))
    if not reply:
        return [], [], []
    body = _extract_assets_body(reply)
    if body is None:
        return [], [], []
    body = body.strip()
    body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
    body = re.sub(r"\n?```$", "", body).strip()
    try:
        obj = json.loads(body)
    except Exception:
        # Truncated stream / trailing garbage — try repair.
        obj = _try_repair_truncated_json_list(body)
    # Wrapper tolerance (GLM-5.2 trace 20260625_124038): some models wrap the
    # spec array in a single-key object — `{"sprites":[...]}`, `{"assets":[...]}`,
    # `{"images":[...]}` — instead of emitting a bare top-level list. The old
    # `isinstance(obj, list)` guard silently dropped ALL specs in that case
    # (16 tower-defense sprites lost; Z-Image never ran). Unwrap a dict whose
    # only/first list value looks like the spec array so the request is honored.
    if isinstance(obj, dict):
        _inner = None
        for _k in ("sprites", "assets", "images", "items", "list"):
            _v = obj.get(_k)
            if isinstance(_v, list):
                _inner = _v
                break
        if _inner is None:
            # Fall back to the first list value under any key.
            for _v in obj.values():
                if isinstance(_v, list):
                    _inner = _v
                    break
        if _inner is not None:
            obj = _inner
    if not isinstance(obj, list):
        return [], [], []
    out: list[dict] = []
    dropped: list[str] = []
    dropped_specs: list[dict] = []
    # Dedupe by (normalized prompt, size). Catches the failure mode where
    # the model spams numbered variants of the same template — e.g. 200×
    # `{"name":"minimap_compiler<N>", "prompt":"green computer","size":"16x16"}`.
    # Without this, `generate_assets` would burn 200 GPU calls (or hit
    # _MAX_ASSETS_PER_TURN and silently truncate, masking the bug).
    seen_keys: set[tuple[str, tuple[int, int]]] = set()
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"asset_{i + 1}").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        _size_default = _default_size_for_prompt(prompt)
        try:
            size = _parse_size(item.get("size") or _size_default)
        except Exception:
            size = (_size_default, _size_default)
        # Normalize prompt the same way the cache key does, so trivial
        # whitespace / case differences don't create duplicate entries
        # that would all map to the same cached PNG anyway.
        norm_prompt = " ".join(prompt.lower().split())
        key = (norm_prompt, size)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        spec: dict[str, Any] = {"name": name, "prompt": prompt, "size": size}
        # B2: optional img2img chaining. `from_image` names another asset
        # already declared in this same <assets> block (or generated in a
        # prior turn — the generator falls back to txt2img when the
        # reference is missing). `strength` controls how much the prompt
        # moves away from the source frame; default 0.45 preserves the
        # silhouette while allowing pose changes.
        from_image = item.get("from_image")
        if isinstance(from_image, str) and from_image.strip():
            spec["from_image"] = from_image.strip()
            try:
                strength = float(item.get("strength", 0.45))
            except (TypeError, ValueError):
                strength = 0.45
            spec["strength"] = max(0.05, min(1.0, strength))
        if len(out) >= effective_cap:
            dropped.append(name)
            dropped_specs.append(dict(spec))
            continue
        out.append(spec)
    return out, dropped, dropped_specs


def prefer_video_seed_assets(
    kept: list[dict],
    dropped_names: list[str],
    dropped_specs: list[dict],
    video_specs: list[dict] | None,
) -> tuple[list[dict], list[str], list[dict]]:
    """When the per-turn cap drops an i2v seed still, swap it back into `kept`.

    run_14 Dragon's Lair / golden 20260626: FIFO cap dropped `key_victory`
    while `<videos>` still pointed at it — cutscene fell back to t2v / 404
    and lost the locked character look. Coaching alone did not stop the
    model from referencing the dropped name. Prefer keeping video `image`
    seeds; evict a non-seed, non-bg, non-key, non-from_image-base sprite.
    Cap size unchanged.
    """
    if not kept or not dropped_specs or not video_specs:
        return kept, dropped_names, dropped_specs
    seed_names = {
        str(s.get("image") or "").strip()
        for s in video_specs
        if isinstance(s, dict) and str(s.get("image") or "").strip()
    }
    if not seed_names:
        return kept, dropped_names, dropped_specs

    kept = [dict(s) for s in kept]
    dropped_specs = [dict(s) for s in dropped_specs]
    dropped_names = list(dropped_names)
    kept_names = {str(s.get("name") or "") for s in kept}
    # Bases referenced by kept from_image frames — do not evict those.
    from_image_bases = {
        str(s.get("from_image") or "").strip()
        for s in kept
        if str(s.get("from_image") or "").strip()
    }

    def _eviction_rank(spec: dict) -> int | None:
        """Lower = safer to drop. None = never evict."""
        name = str(spec.get("name") or "").strip()
        if not name or name in seed_names:
            return None
        if name in from_image_bases:
            return None
        if name.startswith("key_"):
            return None
        if name.startswith("bg_"):
            return None
        # Prefer evicting derived frames, then ordinary sprites.
        if spec.get("from_image"):
            return 0
        return 1

    for rescue in list(dropped_specs):
        rname = str(rescue.get("name") or "").strip()
        if rname not in seed_names or rname in kept_names:
            continue
        candidates: list[tuple[int, int, dict]] = []
        for i, spec in enumerate(kept):
            rank = _eviction_rank(spec)
            if rank is None:
                continue
            candidates.append((rank, i, spec))
        if not candidates:
            break
        candidates.sort(key=lambda t: (t[0], -t[1]))  # safest, prefer later
        _rank, idx, victim = candidates[0]
        victim_name = str(victim.get("name") or "")
        kept[idx] = rescue
        kept_names.discard(victim_name)
        kept_names.add(rname)
        dropped_specs = [s for s in dropped_specs if str(s.get("name") or "") != rname]
        dropped_specs.append(victim)
        dropped_names = [n for n in dropped_names if n != rname]
        if victim_name:
            dropped_names.append(victim_name)
        # Victim may have been a from_image base — refresh set for next rescue.
        from_image_bases = {
            str(s.get("from_image") or "").strip()
            for s in kept
            if str(s.get("from_image") or "").strip()
        }
    return kept, dropped_names, dropped_specs


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
    # Allow multiple base dirs (portable across machines):
    #   DIFFUSION_MODELS_DIR="/path/one:/path/two"
    raw_env = (os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    env_bases = [p.strip() for p in raw_env.split(":") if p.strip()]
    candidates: list[str] = []
    for env_dir in env_bases:
        candidates.extend([
            os.path.join(env_dir, "Z-Image-Turbo"),
            os.path.join(env_dir, "Tongyi-MAI_Z-Image-Turbo"),
            # Common "downloaded diffusers" layout:
            #   Diffusion_Models/Tongyi-MAI/Z-Image-Turbo
            os.path.join(env_dir, "Tongyi-MAI", "Z-Image-Turbo"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            os.path.join(base, "Z-Image-Turbo"),
            os.path.join(base, "Tongyi-MAI_Z-Image-Turbo"),
            # Common "downloaded diffusers" layout:
            #   Diffusion_Models/Tongyi-MAI/Z-Image-Turbo
            os.path.join(base, "Tongyi-MAI", "Z-Image-Turbo"),
        ])
    for c in candidates:
        if os.path.isdir(c):
            return c
    # If the user already has the weights in the HuggingFace hub cache,
    # prefer a local snapshot directory (no network) over the hub ID.
    #
    # IMPORTANT: snapshots can be incomplete (interrupted download / partial
    # cleanup). Only accept a snapshot that is actually COMPLETE — otherwise
    # diffusers raises FileNotFoundError mid-load and every asset fails.
    # We validate the sharded text_encoder against its index (all shards must
    # exist), because a snapshot can contain shard 3-of-3 but be missing 1,2.
    try:
        home = os.path.expanduser("~")
        hub_root = os.path.join(home, ".cache", "huggingface", "hub")
        model_root = os.path.join(hub_root, "models--Tongyi-MAI--Z-Image-Turbo", "snapshots")
        if os.path.isdir(model_root):
            snaps = [
                os.path.join(model_root, d)
                for d in os.listdir(model_root)
                if os.path.isdir(os.path.join(model_root, d))
            ]
            snaps.sort(reverse=True)
            for snap in snaps:
                if _zimage_snapshot_is_complete(snap):
                    return snap
    except Exception:
        pass
    return _HF_FALLBACK_MODEL_ID


def _zimage_snapshot_is_complete(snapshot_dir: str) -> bool:
    """True when a HF-cache Z-Image snapshot has all files needed to load.

    Guards against partial downloads. We check the text_encoder because that
    is where the observed failure occurred (a snapshot with only
    `model-00003-of-00003.safetensors` present, shards 1-2 missing). When a
    sharded index is present we require EVERY referenced shard to exist on
    disk (following symlinks); otherwise we require at least one non-sharded
    `*.safetensors` weight file.
    """
    import os
    te = os.path.join(snapshot_dir, "text_encoder")
    if not os.path.isdir(te):
        return False
    index_path = os.path.join(te, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            import json
            with open(index_path, "r", encoding="utf-8") as fh:
                idx = json.load(fh)
            shard_files = set((idx.get("weight_map") or {}).values())
            if not shard_files:
                return False
            for shard in shard_files:
                # os.path.exists follows symlinks, so a broken HF blob link => False.
                if not os.path.exists(os.path.join(te, shard)):
                    return False
            return True
        except Exception:
            return False
    # No index: accept only if a real (non-broken) safetensors weight exists.
    try:
        for fn in os.listdir(te):
            if fn.endswith(".safetensors") and os.path.exists(os.path.join(te, fn)):
                return True
    except OSError:
        pass
    return False


def _resolve_flux2_path() -> str | None:
    """Find a locally-downloaded FLUX2 model directory for diffusers.

    Intended for users who store diffusion weights outside the HF hub
    cache (or in custom trees). Unlike Z-Image-Turbo, we do NOT return
    a hub ID fallback here — selecting FLUX2 is an explicit opt-in and
    should never start a surprise 10s-of-GB download.
    """
    raw_env = (_os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    env_bases = [p.strip() for p in raw_env.split(":") if p.strip()]
    candidates: list[str] = []
    for env_dir in env_bases:
        candidates.extend([
            _os.path.join(env_dir, "FLUX2-klein-9B-mlx-8bit"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            _os.path.join(base, "FLUX2-klein-9B-mlx-8bit"),
        ])
    for c in candidates:
        if _os.path.isdir(c):
            return c
    return None


def _resolve_mflux_generate_flux2() -> str | None:
    """Return the mflux FLUX2 generator binary path, or None.

    Cursor/non-interactive shells may not see the same PATH as the user's
    login shell. Allow an explicit override path via env var.
    """
    override = (_os.environ.get("MFLUX_GENERATE_FLUX2") or "").strip()
    if override:
        try:
            if _os.path.isfile(override):
                return override
        except Exception:
            return None
        return None
    # Look next to the running interpreter FIRST: when mflux is pip-installed
    # into the same venv as chat.py, the CLI lands in that venv's bin/ dir,
    # which is the most reliable location regardless of the ambient PATH.
    try:
        venv_bin = _os.path.dirname(sys.executable or "")
        if venv_bin:
            cand = _os.path.join(venv_bin, "mflux-generate-flux2")
            if _os.path.isfile(cand):
                return cand
    except Exception:
        pass
    try:
        import shutil
        hit = shutil.which("mflux-generate-flux2")
        if hit:
            return hit
    except Exception:
        pass
    # Common `uv tool install` location (outside PATH for non-login shells):
    #   ~/.local/share/uv/tools/<tool>/bin/mflux-generate-flux2
    try:
        home = _os.path.expanduser("~")
        uv_tools = _os.path.join(home, ".local", "share", "uv", "tools")
        if _os.path.isdir(uv_tools):
            # Keep this shallow and cheap: only scan 2 levels.
            for tool_dir in _os.listdir(uv_tools):
                cand = _os.path.join(uv_tools, tool_dir, "bin", "mflux-generate-flux2")
                if _os.path.isfile(cand):
                    return cand
    except Exception:
        pass
    return None


class Flux2KleinMfluxGenerator:
    """FLUX2 klein sprite generator via mflux CLI (local, no diffusers).

    The user's `FLUX2-klein-9B-mlx-8bit` weights are in mflux format and do not
    include a diffusers `model_index.json`, so we generate by invoking the CLI.
    """

    def __init__(self, *, model_path: str | None = None) -> None:
        self.model_path = model_path or (_resolve_flux2_path() or "")
        self._mflux_bin = _resolve_mflux_generate_flux2() or ""
        self._last_error: str | None = None

    def _ensure_ready(self) -> bool:
        if not self.model_path:
            self._last_error = (
                "FLUX2 klein selected but no local model directory was found. "
                "Put `FLUX2-klein-9B-mlx-8bit/` under `~/Diffusion_Models` or "
                "set DIFFUSION_MODELS_DIR to include the parent directory."
            )
            return False
        if not self._mflux_bin:
            self._last_error = (
                "FLUX2 klein selected but `mflux-generate-flux2` was not found. "
                "Put it on PATH, or set MFLUX_GENERATE_FLUX2=/full/path/to/mflux-generate-flux2."
            )
            return False
        return True

    def _fresh_out_path(self) -> str:
        """Return a unique .png path that does NOT yet exist.

        mflux refuses to overwrite an existing --output file and instead
        writes a de-duplicated `<name>_1.png`, which would leave us returning
        an empty placeholder. So we hand it a brand-new path in a temp dir.
        """
        import tempfile
        import uuid
        d = tempfile.mkdtemp(prefix="flux2klein_")
        return _os.path.join(d, f"{uuid.uuid4().hex}.png")

    def _steps(self) -> str:
        """FLUX2 klein is distilled for very few steps. Default 4 (verified
        ~8.5s incl. load on this Mac); override with MFLUX_STEPS."""
        raw = (_os.environ.get("MFLUX_STEPS") or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return raw
        return "4"

    def _base_cmd(self, prompt: str, out_path: str) -> list[str]:
        """Shared CLI args for txt2img and img2img.

        NOTE: klein (distilled) rejects `--guidance` (mflux: "only supported
        for FLUX.2 base models"), so we do NOT pass it. `--base-model` names
        the klein family so a local pre-quantized mflux path loads correctly.
        """
        return [
            self._mflux_bin,
            "--model", str(self.model_path),
            "--base-model", "flux2-klein-9b",
            "--prompt", str(prompt),
            # NOTE: mflux rejects --negative-prompt for FLUX.2 klein (exit 2).
            # Steer away from checkerboard via _ensure_sprite_bg_prompt() instead.
            "--width", "768",
            "--height", "768",
            "--steps", self._steps(),
            "--seed", "42",
            "--output", out_path,
        ]

    def generate(self, prompt: str) -> str | None:
        """txt2img via `mflux-generate-flux2`."""
        self._last_error = None
        if not self._ensure_ready():
            return None
        try:
            import subprocess

            out_path = self._fresh_out_path()
            cmd = self._base_cmd(prompt, out_path)
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                self._last_error = (
                    f"mflux txt2img failed (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
                )
                return None
            return out_path
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"mflux txt2img invoke failed: {type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return None

    def generate_img2img(
        self,
        prompt: str,
        init_image_path: str,
        *,
        strength: float = 0.5,
    ) -> str | None:
        """Init-image guidance via mflux (img2img).

        mflux 0.18 uses `--image-path` + `--image-strength` (higher strength =
        result resembles the init image more closely).
        """
        self._last_error = None
        if not self._ensure_ready():
            return None
        try:
            import subprocess

            out_path = self._fresh_out_path()
            strength = max(0.0, min(1.0, float(strength)))
            cmd = self._base_cmd(prompt, out_path) + [
                "--image-path", str(init_image_path),
                "--image-strength", str(strength),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                self._last_error = (
                    f"mflux img2img failed (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
                )
                return None
            return out_path
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"mflux img2img invoke failed: {type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return None


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
        # Default stays Z-Image-Turbo. Opt-in override via env var:
        #   DIFFUSER_TXT2IMG_BACKBONE=flux2
        # This only affects the txt2img pipeline; img2img remains SD-Turbo.
        self._txt2img_backbone = (
            (_os.environ.get("DIFFUSER_TXT2IMG_BACKBONE") or "").strip().lower()
        )
        self._use_mflux = self._txt2img_backbone in ("flux2", "flux")
        if self._use_mflux:
            self.model_path = model_path or (_resolve_flux2_path() or "")
        else:
            self.model_path = model_path or _resolve_zimage_path()
        self._pipeline: Any = None  # lazy-init in .generate()
        # Img2img pipeline built lazily from _pipeline.components (shared VRAM)
        # in .generate_img2img(); used for from_image animation frames.
        self._img2img_pipeline: Any = None
        # Resolved at first .generate() call; "cuda", "mps", or None.
        self._device: str | None = None
        # Physical/logical CUDA index after .to("cuda"); status panel only.
        self._cuda_device_index: int | None = None
        # Last error captured from _lazy_init or generate. Surfaced via
        # last_stats[i]["error"] so the caller can show the user the
        # actual exception (instead of a canned "OOM / NSFW / etc"
        # guess that hid e.g. diffusers API drift or model path errors).
        self._last_error: str | None = None

    def cleanup(self) -> None:
        if self._pipeline is None and self._img2img_pipeline is None:
            return
        try:
            import torch
            # The img2img pipeline shares _pipeline's components — drop the
            # wrapper reference before freeing the underlying modules.
            self._img2img_pipeline = None
            del self._pipeline
            self._pipeline = None
            _torch_empty_device_cache()
        except Exception:
            self._pipeline = None
            self._img2img_pipeline = None

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
            try:
                import gpu_status as _gs
                snap = _gs.snapshot_gpus()
                pick = _gs.pick_diffuser_cuda_index(snap)
                if pick is not None:
                    _gs.activate_cuda_device(pick)
            except Exception:
                pass
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
            self._cuda_device_index = (
                int(torch.cuda.current_device())
                if device == "cuda" else None
            )
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
            self._cuda_device_index = None
            return False

    def generate_batch(self, prompts: list[str]) -> list[str | None]:
        """Phase 2C — batched txt2img. Same pipeline, one forward pass
        for N prompts. Returns one path per prompt (None on failure).

        Falls back to per-prompt `generate()` calls on any exception
        — preserves the existing behavior on hardware that can't fit
        the batch in VRAM, on diffusers API drift, or on pipelines
        that don't accept a list prompt argument.

        Caller decides batch size. Stable for batches of 2-4 on a
        48 GB card with Z-Image-Turbo (~14 GB resident, ~3 GB per
        batched forward). Larger batches risk OOM; the caller should
        chunk.

        Wire-in status (2026-05-22): this method is **available** for
        callers but `generate_assets` is NOT yet refactored to use it
        — that refactor touches the per-spec cache/chroma-key/library
        code path with several side effects per asset, and the win
        (~10s on a 12-asset batch) doesn't justify destabilising the
        existing per-call flow until we have live coverage. Use this
        method directly from new code paths; the existing
        generate_assets continues to call .generate() per spec.
        """
        self._last_error = None
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0])]
        if not self._lazy_init():
            return [None] * len(prompts)
        try:
            import tempfile
            import torch
            gen = torch.Generator(self._device or "cpu").manual_seed(42)
            result = self._pipeline(
                prompt=list(prompts),
                height=768,
                width=768,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=gen,
            )
            images = getattr(result, "images", None)
            if not images or len(images) != len(prompts):
                self._last_error = (
                    f"batched pipeline returned {len(images) if images else 0} "
                    f"images for {len(prompts)} prompts — falling back to "
                    "per-prompt generation."
                )
                # Fall back per-prompt so the user still gets all
                # requested PNGs even if batching is wonky on this
                # pipeline version.
                return [self.generate(p) for p in prompts]
            out: list[str | None] = []
            for image in images:
                try:
                    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    f.close()
                    image.save(f.name, format="PNG")
                    out.append(f.name)
                except Exception as e:
                    self._last_error = f"save failed: {type(e).__name__}: {e!s}"
                    out.append(None)
            return out
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"batched gen failed, falling back to per-prompt: "
                f"{type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return [self.generate(p) for p in prompts]

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

    def generate_img2img(
        self,
        prompt: str,
        init_image_path: str,
        *,
        strength: float = 0.5,
    ) -> str | None:
        """Img2img using the SAME Z-Image-Turbo model as txt2img.

        Animation frames MUST be drawn by the same model as the base/idle
        sprite, or they won't match it (the old pipeline used a foreign
        SD-Turbo model for `from_image`, so derived frames were visibly
        inconsistent with the character — 2026-05-29 trace). We build a
        `ZImageImg2ImgPipeline` from the already-loaded txt2img pipeline's
        components, so it shares the transformer/VAE/text-encoder and costs
        NO extra VRAM. Returns a temp PNG path, or None (sets _last_error).
        """
        self._last_error = None
        if not self._lazy_init():
            return None
        try:
            import tempfile
            import torch
            from PIL import Image
            from diffusers import ZImageImg2ImgPipeline

            if getattr(self, "_img2img_pipeline", None) is None:
                # Reuse the loaded components — no second model in VRAM.
                self._img2img_pipeline = ZImageImg2ImgPipeline(
                    **self._pipeline.components
                )
                self._img2img_pipeline.to(self._device or "cpu")

            init_img = Image.open(init_image_path).convert("RGB")
            if init_img.size != (768, 768):
                init_img = init_img.resize((768, 768), Image.LANCZOS)
            strength = max(0.05, min(1.0, float(strength)))
            # Turbo DiT: actual denoising steps ≈ num_inference_steps * strength.
            # Aim for ~8 actual steps so the pose moves while the character holds.
            steps = max(9, int(round(8.0 / strength)))
            gen = torch.Generator(self._device or "cpu").manual_seed(42)
            result = self._img2img_pipeline(
                prompt=prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen,
            )
            images = getattr(result, "images", None)
            if not images:
                self._last_error = (
                    "Z-Image img2img returned no images "
                    f"(type={type(result).__name__})."
                )
                return None
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            images[0].save(f.name, format="PNG")
            return f.name
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"Z-Image img2img failed: {type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return None


# ---------------------------------------------------------------------------
# B1: SD-Turbo img2img wrapper. Used to chain animation frames where
# frame N is generated FROM frame N-1 — the standard fix for "two
# different aliens" output when the model asked for "alien walk1" and
# "alien walk2" as independent txt2img calls.
# ---------------------------------------------------------------------------


def _resolve_sd_turbo_path() -> str:
    """Find SD-Turbo weights on disk, or return the HF model id so
    diffusers downloads on first run. Same search order as
    `_resolve_zimage_path` — env var, then known model dirs, then HF.
    """
    env_dir = (_os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    candidates: list[str] = []
    if env_dir:
        candidates.extend([
            _os.path.join(env_dir, "sd-turbo"),
            _os.path.join(env_dir, "stabilityai_sd-turbo"),
            # Common "downloaded diffusers" layout:
            #   Diffusion_Models/stabilityai/sd-turbo
            _os.path.join(env_dir, "stabilityai", "sd-turbo"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            _os.path.join(base, "sd-turbo"),
            _os.path.join(base, "stabilityai_sd-turbo"),
            # Common "downloaded diffusers" layout:
            #   Diffusion_Models/stabilityai/sd-turbo
            _os.path.join(base, "stabilityai", "sd-turbo"),
        ])
    for c in candidates:
        if _os.path.isdir(c):
            return c
    # Same local HF-cache preference as Z-Image-Turbo (no network):
    #   ~/.cache/huggingface/hub/models--stabilityai--sd-turbo/snapshots/<sha>/
    try:
        home = _os.path.expanduser("~")
        hub_root = _os.path.join(home, ".cache", "huggingface", "hub")
        model_root = _os.path.join(hub_root, "models--stabilityai--sd-turbo", "snapshots")
        if _os.path.isdir(model_root):
            snaps = [
                _os.path.join(model_root, d)
                for d in _os.listdir(model_root)
                if _os.path.isdir(_os.path.join(model_root, d))
            ]
            snaps.sort(reverse=True)
            if snaps:
                return snaps[0]
    except Exception:
        pass
    return _HF_IMG2IMG_FALLBACK_MODEL_ID


class Img2ImgGenerator:
    """In-process SD-Turbo img2img wrapper.

    Usage:
        gen = Img2ImgGenerator()
        path = gen.generate(prompt, init_image_path, strength=0.45)

    The pipeline lazy-loads on the first `.generate()` call. SD-Turbo
    runs at 512×512 in fp16 on CUDA + fp16 on MPS (numerically stable,
    unlike Z-Image-Turbo). 1-4 inference steps; we use 2 by default
    which is the standard turbo recipe for img2img.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _resolve_sd_turbo_path()
        self._pipeline: Any = None
        self._device: str | None = None
        self._cuda_device_index: int | None = None
        self._last_error: str | None = None

    def cleanup(self) -> None:
        if self._pipeline is None:
            return
        try:
            import torch
            del self._pipeline
            self._pipeline = None
            _torch_empty_device_cache()
        except Exception:
            self._pipeline = None

    def _lazy_init(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except Exception as e:
            self._last_error = (
                f"import failed: {type(e).__name__}: {e!s}. "
                "Run `pip install -r requirements-diffuser.txt` in the "
                "Agent_learning venv (diffusers >= 0.30)."
            )
            return False
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            try:
                import gpu_status as _gs
                snap = _gs.snapshot_gpus()
                pick = _gs.pick_diffuser_cuda_index(snap)
                if pick is not None:
                    _gs.activate_cuda_device(pick)
            except Exception:
                pass
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
            # SD-Turbo is reported numerically stable on MPS fp16
            # (unlike Z-Image-Turbo which produces NaN). Caller can
            # bump dtype via env if needed.
            dtype = torch.float16
        else:
            self._last_error = (
                "no CUDA and no MPS device available — torch sees neither. "
                "SD-Turbo on CPU would be ~30s per image; we refuse to "
                "fall back to it. Install accelerator-aware torch."
            )
            return False
        try:
            self._pipeline = AutoPipelineForImage2Image.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self._pipeline.to(device)
            # Disable the safety checker if present — these are tiny
            # game sprites, not user-uploaded photos, and the checker
            # occasionally false-positives on cartoony explosions and
            # then returns black images with no error.
            sc = getattr(self._pipeline, "safety_checker", None)
            if sc is not None:
                self._pipeline.safety_checker = None
            self._device = device
            self._cuda_device_index = (
                int(torch.cuda.current_device())
                if device == "cuda" else None
            )
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
            self._cuda_device_index = None
            return False

    def generate(
        self,
        prompt: str,
        init_image_path: str,
        *,
        strength: float = 0.45,
        num_inference_steps: int = 2,
    ) -> str | None:
        """Run img2img and save a 512×512 PNG to a temp file. Returns
        the absolute path, or None on failure (set self._last_error).

        SD-Turbo requires `num_inference_steps * strength >= 1` for
        useful output. We default strength=0.45 (preserves silhouette
        + palette while letting the prompt move the pose) at 2 steps,
        which lands exactly at the recommended floor.
        """
        self._last_error = None
        if not self._lazy_init():
            return None
        try:
            import tempfile
            import torch
            from PIL import Image
            init_img = Image.open(init_image_path).convert("RGB")
            if init_img.size != (512, 512):
                init_img = init_img.resize((512, 512), Image.LANCZOS)
            # SD-Turbo: num_inference_steps * strength must be >= 1.
            strength = max(0.05, min(1.0, float(strength)))
            steps = max(num_inference_steps, max(1, int(round(1.0 / strength))))
            gen = torch.Generator(self._device or "cpu").manual_seed(42)
            result = self._pipeline(
                prompt=prompt,
                image=init_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=gen,
            )
            images = getattr(result, "images", None)
            if not images:
                self._last_error = (
                    "pipeline returned no images. Possible NSFW safety "
                    "rejection (the safety_checker was disabled at load, "
                    "but a wrapper may still filter) OR a diffusers API "
                    f"drift: type={type(result).__name__}."
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


def _preload_stable_audio() -> None:
    """Load Stable Audio Open before Playwright opens IPC pipes.

    Same fds_to_keep fork trap as Z-Image-Turbo. When macOS uses FLUX2
    klein (mflux CLI) we skip the diffusers image preload, but audio
    still uses diffusers from_pretrained and MUST preload here.
    """
    try:
        import sounds as _sounds
        _sounds.preload()
    except Exception:
        pass


def diffuser_cuda_reuse_index() -> int | None:
    """Physical CUDA index of a loaded image pipeline (for co-locating audio)."""
    gen = _PRELOADED
    if gen is None:
        return None
    idx = getattr(gen, "_cuda_device_index", None)
    return int(idx) if idx is not None else None


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
        _preload_stable_audio()
        return _PRELOADED
    # If FLUX2 klein (mflux CLI) is active, do NOT preload the diffusers
    # Z-Image-Turbo pipeline. FLUX2 generation is handled by an external
    # CLI and doesn't need the Playwright-safe preload — but Stable Audio
    # still does, so call _preload_stable_audio() on every exit path.
    backbone = (_os.environ.get("DIFFUSER_TXT2IMG_BACKBONE") or "").strip().lower()
    if backbone in ("flux2", "flux"):
        _preload_stable_audio()
        return None
    if sys.platform == "darwin":
        # macOS default: if FLUX2 is available (model + binary), skip preload.
        if _resolve_flux2_path() and _resolve_mflux_generate_flux2():
            _preload_stable_audio()
            return None
    gen = _construct_generator()
    if gen is None:
        _preload_stable_audio()
        return None
    # Trigger the heavy load NOW so the subprocess fork happens
    # before Playwright/etc opens any FDs. _lazy_init returns False
    # on failure with the reason on _last_error; we still cache the
    # wrapper so the agent path can read _last_error and skip
    # gracefully instead of retrying the broken fork.
    gen._lazy_init()
    _PRELOADED = gen
    _preload_stable_audio()
    return gen


def _construct_generator() -> Any:
    """Internal: just check imports and construct a wrapper. Pulled
    out of try_load_image_generator so preload() can share it."""
    try:
        # Selection order:
        # 1) Explicit env override: DIFFUSER_TXT2IMG_BACKBONE=flux2
        # 2) macOS default: FLUX2 klein when model+binary are present
        # 3) fallback: Z-Image-Turbo (diffusers)
        backbone = (_os.environ.get("DIFFUSER_TXT2IMG_BACKBONE") or "").strip().lower()
        want_flux2 = backbone in ("flux2", "flux")
        if (want_flux2 or sys.platform == "darwin"):
            m = _resolve_flux2_path()
            b = _resolve_mflux_generate_flux2()
            if m and b and (want_flux2 or sys.platform == "darwin"):
                return Flux2KleinMfluxGenerator(model_path=m)
        # diffusers fallback
        import importlib.util as _iu
        if _iu.find_spec("torch") is None or _iu.find_spec("diffusers") is None:
            return None
        return ZImageTurboGenerator()
    except Exception:
        return None


def _torch_empty_device_cache() -> None:
    """Best-effort GPU allocator flush after dropping diffuser pipelines."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def release_preloaded_diffusers() -> list[str]:
    """Release module-level diffuser pipelines and free CUDA cache.

    Called from chat ``/unload all`` so sprite/img2img VRAM is dropped
    alongside Ollama models. Returns human labels for what was cleared.
    """
    global _PRELOADED, _PRELOADED_IMG2IMG
    freed: list[str] = []
    for label, attr in (
        ("Z-Image-Turbo", "_PRELOADED"),
        ("SD-Turbo img2img", "_PRELOADED_IMG2IMG"),
    ):
        gen = globals().get(attr)
        if gen is None:
            continue
        try:
            if hasattr(gen, "cleanup"):
                gen.cleanup()
        except Exception:
            pass
        globals()[attr] = None
        freed.append(label)
    _torch_empty_device_cache()
    return freed


def try_load_image_generator(
    model_id: str = "Z-Image-Turbo",  # kept for API stability; unused
    diffuser_dir: str | None = None,  # kept for API stability; unused
) -> Any:
    """Return the active sprite generator (FLUX2 klein on macOS when available,
    else Z-Image-Turbo via diffusers).

    If `preload()` ran earlier, reuses that already-loaded pipeline (this is the
    path chat.py takes for diffusers). Otherwise constructs a fresh wrapper that
    lazy-loads on first .generate().

    Returns None when torch+diffusers aren't installed.
    """
    if _PRELOADED is not None:
        return _PRELOADED
    return _construct_generator()


# B1: parallel preload/loader pair for the img2img pipeline. Kept
# separate from the Z-Image-Turbo cache so each pipeline only loads
# when the session actually needs it.
_PRELOADED_IMG2IMG: Any = None


def _construct_img2img_generator() -> Any:
    """Internal: just check imports and construct a wrapper."""
    import importlib.util as _iu
    if _iu.find_spec("torch") is None or _iu.find_spec("diffusers") is None:
        return None
    try:
        return Img2ImgGenerator()
    except Exception:
        return None


def preload_img2img() -> Any:
    """Eagerly load SD-Turbo before any subprocess-spawning library
    (Playwright, multiprocessing) is up. Same fds-to-keep workaround
    as `preload()`. Idempotent.
    """
    global _PRELOADED_IMG2IMG
    if _PRELOADED_IMG2IMG is not None:
        return _PRELOADED_IMG2IMG
    gen = _construct_img2img_generator()
    if gen is None:
        return None
    gen._lazy_init()
    _PRELOADED_IMG2IMG = gen
    return gen


def try_load_img2img_generator() -> Any:
    """Return the SD-Turbo wrapper, reusing a preloaded instance if one
    was constructed earlier. None when torch+diffusers aren't installed.
    """
    if _PRELOADED_IMG2IMG is not None:
        return _PRELOADED_IMG2IMG
    return _construct_img2img_generator()


# B2: topologically sort specs so `from_image` references resolve in
# order. Roots (no from_image) come first; chained children come after
# the asset they depend on. Cycles fall back to the original order
# with a stat marker on the offending entries.
def _topo_sort_specs(specs: list[dict]) -> list[dict]:
    by_name = {s["name"]: s for s in specs}
    in_order: list[dict] = []
    visited: dict[str, str] = {}  # name -> "tmp" | "done"
    cycle = False

    def visit(s: dict) -> None:
        nonlocal cycle
        nm = s["name"]
        state = visited.get(nm)
        if state == "done":
            return
        if state == "tmp":
            cycle = True
            return
        visited[nm] = "tmp"
        parent = s.get("from_image")
        if parent and parent in by_name and parent != nm:
            visit(by_name[parent])
        visited[nm] = "done"
        in_order.append(s)

    for s in specs:
        visit(s)
    if cycle or len(in_order) != len(specs):
        # Don't try to be clever — return original order, let the per-spec
        # generation code mark unresolved parents as errors.
        return specs
    return in_order


_UNSET: Any = object()  # sentinel: "argument not provided"

# Below this mean per-pixel RGB delta (0..1), a `from_image` pose frame is
# treated as "near-identical to its parent" — the pose didn't change at all.
# NOTE (2026-05-30): this is only a CHEAP FLOOR for a literally-unchanged frame;
# delta does NOT measure whether the pose is correct (a style-drifted idle can
# score 0.3+ while still showing no punch — see animation_ab/). Real pose checks
# are the VLM's job. Pose frames now render as TXT2IMG from the shared character
# description (img2img stayed locked to idle at guidance_scale=0), so a genuine
# pose scores high (~0.3-0.5); 0.04 only catches a frame that came back as the
# idle. Heuristic warning, never a hard error.
_DERIVED_FRAME_MIN_DELTA = 0.04

# Idle prompts often lock orientation ("facing straight up"). When merged with
# a bank/tilt pose that should change orientation, those locks cancel the pose
# (1942 run_13). Strip only clear orientation locks from the parent string.
_IDLE_ORIENTATION_LOCK_RE = re.compile(
    r",?\s*(?:facing\s+(?:straight\s+)?(?:up|forward|front)|"
    r"viewed\s+from\s+(?:above|front)|"
    r"front[- ]view|"
    r"looking\s+(?:straight\s+)?(?:up|forward))\b",
    re.IGNORECASE,
)


def _strip_idle_orientation_locks(prompt: str) -> str:
    """Remove idle orientation clamps so pose clauses can change facing/tilt."""
    if not prompt:
        return prompt
    cleaned = _IDLE_ORIENTATION_LOCK_RE.sub("", prompt)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned).strip(" ,")
    return cleaned or prompt


def _derived_frame_delta(new_path: Path | str, parent_path: Path | str) -> float | None:
    """Mean per-pixel RGB delta between a derived frame and its parent.

    Both images are composited over a neutral gray before differencing so
    transparent (chroma-keyed) regions don't register as spurious change, then
    resized to 128×128 — the same comparison shape as tools.screenshot_delta.
    Returns None if PIL is unavailable or either image can't be read.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        def _flat(p: Path | str):
            im = Image.open(p).convert("RGBA")
            bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
            return Image.alpha_composite(bg, im).convert("RGB").resize((128, 128))
        a = _flat(new_path)
        b = _flat(parent_path)
    except Exception:
        return None
    pa = a.tobytes()
    pb = b.tobytes()
    if len(pa) != len(pb) or not pa:
        return None
    total = sum(x - y if x >= y else y - x for x, y in zip(pa, pb))
    return total / (len(pa) * 255.0)




def generate_assets(
    specs: list[dict],
    session_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    image_generator: Any = None,
    img2img_generator: Any = _UNSET,
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

    # Tests inject a stub `image_generator` to bypass the heavy
    # diffusers stack; that's our signal to also bypass the shared
    # cross-session library (which would otherwise leak production
    # state into hermetic tests).
    _caller_provided_generator = image_generator is not None
    if image_generator is None:
        image_generator = try_load_image_generator(model_id)
    if image_generator is None:
        return {}

    # B2: order so img2img children are generated AFTER their parents.
    # Falls back to original order on cycles.
    specs = _topo_sort_specs(list(specs))
    # Lazy-load img2img only if any spec actually requested it AND the
    # caller didn't explicitly disable it (img2img_generator=None vs
    # the _UNSET sentinel that means "not provided"). Saves ~2 GB of
    # VRAM in the common case of all-root assets, and lets tests inject
    # None to assert the fallback path.
    # Pose/animation frames are now generated as TXT2IMG (merged character +
    # pose prompt, fixed seed) — NOT img2img. Proven 2026-05-30 (A/B in
    # animation_ab/): img2img at guidance_scale=0 stays locked to the idle pose
    # at every strength/model; txt2img from the shared character description
    # produces the real pose AND the same character. So the SD-Turbo img2img
    # generator is no longer loaded.
    needs_img2img = False
    if img2img_generator is _UNSET:
        img2img_generator = (
            try_load_img2img_generator() if needs_img2img else None
        )
    # name -> prompt, so a `from_image` pose frame can be regenerated as
    # txt2img with the PARENT's character+style description prepended to this
    # frame's pose clause (keyed by both raw and filesafe name).
    prompt_by_name: dict[str, str] = {}
    for s in specs:
        prompt_by_name[s["name"]] = s["prompt"]
        prompt_by_name[_safe_filename(s["name"])] = s["prompt"]

    out: dict[str, Path] = {}
    # 2.2: per-asset stats accumulated as a side channel. Caller can
    # check `image_generator.last_stats` (a list of per-asset dicts)
    # after the call. Using an attribute on the generator instance so
    # the function signature stays backward-compatible.
    asset_stats: list[dict[str, Any]] = []
    # Cross-session asset library: lazily-instantiated, lets sessions
    # reuse semantically-similar sprites from prior wins without paying
    # the GPU cost again. Library lookups are skipped for img2img
    # children (they depend on a session-local parent that doesn't
    # exist in the library yet). We only enable the library when the
    # caller did NOT pass an explicit `cache_dir` — that's the test
    # opt-out signal: tests want hermetic state, production uses the
    # default cache_dir and gets the shared library too.
    library: Any = None
    if cache_dir is None and not _caller_provided_generator:
        try:
            from asset_library import AssetLibrary
            library = AssetLibrary()
        except Exception:
            library = None
    for spec in specs:
        import time
        t0 = time.time()
        name = _safe_filename(spec["name"])
        prompt = spec["prompt"]
        size = spec["size"]
        from_image = spec.get("from_image")
        strength = spec.get("strength")
        # Append background suffix for sprite assets (not full-bleed backgrounds).
        # FLUX2 klein: solid white (not "transparent" — it draws checkerboard).
        # Z-Image: transparent hint + chroma-key safety net below.
        gen_prompt = prompt
        if not _is_full_bleed_asset_name(name):
            gen_prompt = _ensure_sprite_bg_prompt(gen_prompt, image_generator)
        # Sprite facing/orientation is NOT pinned or rewritten here — that
        # convention now lives in the playbook (directional-art-faces-right),
        # which teaches the model to author one right-facing pose set and flip
        # in code. The pipeline stays genre/art-policy free.
        # B2: include parent file's mtime in the cache key for img2img so
        # regenerating a parent invalidates downstream frames.
        if from_image and from_image in out:
            parent_path = out[from_image]
            try:
                parent_sig = f"{parent_path.stat().st_size}:{int(parent_path.stat().st_mtime)}"
            except OSError:
                parent_sig = "missing"
            cache_basis = f"{model_id}|img2img|{from_image}|{parent_sig}|{strength}|{prompt}|{size}"
            key = hashlib.sha256(cache_basis.encode("utf-8")).hexdigest()[:32]
        else:
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
            "from_image": from_image,
            "strength": strength,
        }
        if cache_path.exists():
            # Re-chroma cached sprites so older white/checkerboard caches
            # become RGBA transparent (pipeline upgrade must not require
            # manual cache wipes).
            if not _is_full_bleed_asset_name(name):
                try:
                    from PIL import Image
                    with Image.open(cache_path) as src_img:
                        src_img.load()
                        keyed, ck_stats = _resize_and_chroma_sprite(src_img, size)
                        stat["bg_color"] = (
                            list(ck_stats["bg_color"])
                            if ck_stats["bg_color"] is not None else None
                        )
                        stat["alpha_pixel_ratio"] = ck_stats["alpha_pixel_ratio"]
                        if ck_stats.get("checkerboard_bg"):
                            stat["checkerboard_bg"] = ck_stats["checkerboard_bg"]
                        keyed.save(cache_path, format="PNG")
                except Exception:
                    pass
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            stat["cache_hit"] = True
            stat["gen_seconds"] = round(time.time() - t0, 3)
            _attach_diffuser_stat(stat, image_generator)
            asset_stats.append(stat)
            # Phase 1C — make progress visible LIVE by publishing
            # the partial stats list after each asset, so the TUI
            # poller can render "Sprites: 4/12 · 2.9s avg" while gen
            # is still running. Caller polls `image_generator.last_stats`
            # at the existing 1 Hz status tick.
            try:
                image_generator.last_stats = list(asset_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue
        # Cross-session library lookup — only for root assets (img2img
        # children depend on session-local parents). Returns the path
        # of a semantically-similar prior-session asset, or None.
        if library is not None and not from_image:
            try:
                hit = library.retrieve(
                    prompt=prompt,
                    modality="sprite",
                    size_or_duration=size,
                )
            except Exception:
                hit = None
            if hit is not None:
                try:
                    _link_or_copy(hit.absolute_path, target_path)
                    # Also seed the per-project _asset_cache so the
                    # in-session exact cache benefits too.
                    _link_or_copy(hit.absolute_path, cache_path)
                    library.touch(hit.entry.id)
                    out[name] = target_path.resolve()
                    stat["library_hit"] = True
                    stat["library_score"] = round(hit.score, 3)
                    stat["library_source_prompt"] = hit.entry.prompt[:80]
                    stat["gen_seconds"] = round(time.time() - t0, 3)
                    _attach_diffuser_stat(stat, image_generator)
                    asset_stats.append(stat)
                    continue
                except Exception:
                    # Fall through to generation on copy failures.
                    pass
        # Cache miss — generate. img2img path when from_image resolves;
        # txt2img otherwise.
        gen_path: str | None = None
        if from_image:
            # POSE / ANIMATION FRAME → TXT2IMG, not img2img. img2img (any model,
            # any strength) stays locked to the idle init at guidance_scale=0
            # (proven 2026-05-30, animation_ab/). We regenerate the pose from
            # the SHARED character description (the parent/idle prompt) + this
            # frame's pose clause, on the FIXED seed — same character, real
            # pose. parent prompt carries the character + style so the frame
            # matches; this frame's prompt carries the pose.
            parent_prompt = (
                prompt_by_name.get(from_image)
                or prompt_by_name.get(_safe_filename(from_image))
                or ""
            )
            # Pose clause FIRST. Prepending parent then pose (old order) left
            # idle orientation locks like "facing straight up" dominating the
            # prompt — 1942 run_13 bank/roll frames came back ~97% identical
            # to idle (deltas 0.02–0.04) despite strength 0.55–0.6. Lead with
            # the NEW pose, keep parent as character/style trailing context.
            # Also strip idle orientation phrases from the parent so they
            # cannot cancel bank/tilt/roll wording in the pose clause.
            if parent_prompt:
                parent_for_merge = _strip_idle_orientation_locks(parent_prompt)
                merged = f"{prompt}, {parent_for_merge}" if prompt else parent_for_merge
            else:
                merged = prompt
            if from_image not in out and from_image not in prompt_by_name:
                stat["parent_missing"] = from_image
            stat["pose_txt2img"] = True
            stat["merged_prompt"] = merged[:240]
            if not _is_full_bleed_asset_name(name):
                merged = _ensure_sprite_bg_prompt(merged, image_generator)
            gen_path = _safe_generate(image_generator, merged)
        else:
            gen_path = _safe_generate(image_generator, gen_prompt)
        if gen_path is None:
            # Pull the real error from the most recently-used generator
            # so the user sees the actual cause — import error, model
            # path miss, fp16 NaN, real NSFW filter, or empty result
            # struct — instead of a one-size-fits-all canned message.
            err_txt2img = getattr(image_generator, "_last_error", None)
            err_img2img = (
                getattr(img2img_generator, "_last_error", None)
                if img2img_generator is not None else None
            )
            real_err = err_img2img or err_txt2img
            stat["error"] = (
                f"diffuser failed: {real_err}" if real_err else
                "diffuser returned None (no exception captured — check "
                "generator's _last_error attribute)"
            )
            stat["gen_seconds"] = round(time.time() - t0, 3)
            _attach_diffuser_stat(
                stat,
                img2img_generator
                if from_image and not stat.get("fallback_to_txt2img")
                else image_generator,
            )
            asset_stats.append(stat)
            # Phase 1C — make progress visible LIVE by publishing
            # the partial stats list after each asset, so the TUI
            # poller can render "Sprites: 4/12 · 2.9s avg" while gen
            # is still running. Caller polls `image_generator.last_stats`
            # at the existing 1 Hz status tick.
            try:
                image_generator.last_stats = list(asset_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue
        try:
            from PIL import Image
            with Image.open(gen_path) as src_img:
                src_img.load()
                stat["native_size"] = [src_img.width, src_img.height]
                keyed, ck_stats = _resize_and_chroma_sprite(src_img, size)
                # 1.3: chroma-key → real RGBA alpha (transparent background).
                stat["bg_color"] = (
                    list(ck_stats["bg_color"])
                    if ck_stats["bg_color"] is not None else None
                )
                stat["alpha_pixel_ratio"] = ck_stats["alpha_pixel_ratio"]
                if ck_stats.get("checkerboard_bg"):
                    stat["checkerboard_bg"] = ck_stats["checkerboard_bg"]
                if (
                    not _is_full_bleed_asset_name(name)
                    and ck_stats["alpha_pixel_ratio"] < 0.15
                ):
                    stat["checkerboard_bg_suspect"] = True
                keyed.save(cache_path, format="PNG")
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            # Derived-frame sanity: if this asset was chained from a parent
            # (from_image) and looks near-identical to it, the requested pose
            # change probably didn't render. Record the delta; the caller
            # turns a small value into a model-facing warning.
            if from_image and from_image in out:
                pdelta = _derived_frame_delta(target_path, out[from_image])
                if pdelta is not None:
                    stat["parent_delta"] = round(pdelta, 4)
                    # run_13 fighter showcase: 5/8 derived frames stayed
                    # near-identical to idle after the first txt2img pass.
                    # One-shot retry with amplified pose wording (pose frames
                    # are txt2img — img2img locks to idle at any strength).
                    # Plan "strength bump" is recorded for cache/stats only.
                    if pdelta < _DERIVED_FRAME_MIN_DELTA:
                        retry_strength = min(
                            0.9, float(strength or 0.55) + 0.2
                        )
                        amplified = (
                            "EXTREME visible pose asymmetry, clear "
                            "silhouette change from idle: " + merged
                        )
                        if not _is_full_bleed_asset_name(name):
                            amplified = _ensure_sprite_bg_prompt(
                                amplified, image_generator
                            )
                        retry_gen = _safe_generate(
                            image_generator, amplified
                        )
                        if retry_gen is not None:
                            try:
                                with Image.open(retry_gen) as retry_img:
                                    retry_img.load()
                                    retry_keyed, retry_ck = (
                                        _resize_and_chroma_sprite(
                                            retry_img, size
                                        )
                                    )
                                    # Write to a side file so we can compare
                                    # without clobbering until we know better.
                                    retry_tmp = (
                                        session_dir
                                        / f"{name}__pose_retry.png"
                                    )
                                    retry_keyed.save(
                                        retry_tmp, format="PNG"
                                    )
                                retry_delta = _derived_frame_delta(
                                    retry_tmp, out[from_image]
                                )
                                if (
                                    retry_delta is not None
                                    and retry_delta > pdelta
                                ):
                                    # Prefer the on-disk retry PNG (safer than
                                    # re-saving a PIL image after the with-block).
                                    _link_or_copy(retry_tmp, cache_path)
                                    _link_or_copy(cache_path, target_path)
                                    out[name] = target_path.resolve()
                                    stat["parent_delta"] = round(
                                        retry_delta, 4
                                    )
                                    stat["pose_retry"] = True
                                    stat["pose_retry_strength"] = (
                                        retry_strength
                                    )
                                    stat["pose_retry_delta"] = round(
                                        retry_delta, 4
                                    )
                                    pdelta = retry_delta
                                    # Refresh chroma stats from the kept frame.
                                    stat["bg_color"] = (
                                        list(retry_ck["bg_color"])
                                        if retry_ck["bg_color"] is not None
                                        else None
                                    )
                                    stat["alpha_pixel_ratio"] = (
                                        retry_ck["alpha_pixel_ratio"]
                                    )
                                else:
                                    stat["pose_retry"] = True
                                    stat["pose_retry_kept_first"] = True
                                    if retry_delta is not None:
                                        stat["pose_retry_delta"] = round(
                                            retry_delta, 4
                                        )
                            except Exception as _retry_err:
                                stat["pose_retry_error"] = (
                                    f"{type(_retry_err).__name__}: "
                                    f"{str(_retry_err)[:80]}"
                                )
                    # #5: a pose frame that actually MOVED (delta over the
                    # clone floor) is worth remembering — record the exact
                    # prompt so future animation games reuse the proven recipe.
                    # Keyed by entity stem (from the parent) + pose (this
                    # frame's distinguishing tokens). Library is production-only.
                    if library is not None:
                        try:
                            _stem = (from_image.split("_", 1)[0] or from_image).lower()
                            _toks = [p for p in name.split("_")
                                     if p and not p.isdigit() and p.lower() != _stem]
                            _pose = "_".join(_toks) or name
                            if library.admit_pose_recipe(
                                stem=_stem, pose=_pose,
                                prompt=prompt, delta=pdelta,
                            ):
                                stat["pose_recipe_saved"] = True
                        except Exception:
                            pass
            # Admit root-prompt sprites to the cross-session library.
            # We skip img2img children because their value is tied to a
            # session-specific parent that isn't admitted.
            if library is not None and not from_image:
                try:
                    library.admit(
                        prompt=prompt,
                        modality="sprite",
                        size_or_duration=size,
                        source_path=target_path,
                    )
                    stat["library_admitted"] = True
                except Exception:
                    pass
        except Exception as e:
            stat["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            stat["gen_seconds"] = round(time.time() - t0, 3)
            _attach_diffuser_stat(
                stat,
                img2img_generator
                if from_image and not stat.get("fallback_to_txt2img")
                else image_generator,
            )
            asset_stats.append(stat)
            # Phase 1C — same live-publish for the success branch.
            try:
                image_generator.last_stats = list(asset_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
    # Stash stats on the generator so the caller can read them out.
    try:
        image_generator.last_stats = asset_stats  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def _attach_diffuser_stat(stat: dict[str, Any], gen: Any | None) -> None:
    """Status-panel fields: which diffuser ran and on which GPU."""
    if gen is None:
        return
    try:
        import gpu_status as _gs
    except Exception:
        return
    stat["diffuser"] = _gs.diffuser_kind(gen)
    stat["gpu"] = (
        _gs.diffuser_placement(gen)
        if getattr(gen, "_pipeline", None) is not None
        else "not loaded"
    )


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


def _safe_call_img2img(
    fn: Any,
    prompt: str,
    init_image_path: str,
    strength: float,
) -> str | None:
    """Like _safe_img2img but for a bound img2img METHOD (e.g. the main
    Z-Image generator's `generate_img2img`), which takes the same args but
    is not the generator's `.generate`. Never propagates."""
    try:
        return fn(prompt, init_image_path, strength=strength)
    except Exception:
        return None


def _safe_img2img(
    gen: Any,
    prompt: str,
    init_image_path: str,
    strength: float,
) -> str | None:
    """B2: parallel to _safe_generate for the img2img path. Same exception
    discipline — never propagate; stash the trace on _last_error so the
    asset stats line tells the user exactly why a frame failed.
    """
    try:
        return gen.generate(prompt, init_image_path, strength=strength)
    except Exception as e:
        import traceback as _tb
        try:
            gen._last_error = (
                f"_safe_img2img caught {type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
        except Exception:
            pass
        return None


def _ensure_transparent_prompt(prompt: str) -> str:
    """Append transparent-background hint when the model omitted it."""
    if "transparent" in prompt.lower():
        return prompt
    return f"{prompt}, transparent background"


# FLUX2-klein often renders a fake Photoshop checkerboard when asked for
# "transparent background". Ask for solid white instead; chroma-key below
# removes it. (mflux does NOT support --negative-prompt on FLUX.2 klein.)
_FLUX2_BG_SUFFIX = "solid pure white background, no checkerboard, no grid"


def _is_flux2_mflux_generator(generator: Any) -> bool:
    return type(generator).__name__ == "Flux2KleinMfluxGenerator"


def _ensure_sprite_bg_prompt(prompt: str, generator: Any) -> str:
    """Background wording matched to the active sprite generator.

    Deliverable is always an RGBA PNG with real alpha (via chroma-key below).
    FLUX2 cannot paint literal transparency — we ask for solid white, then
    key it out. Z-Image gets 'transparent background' in the prompt; same key.
    """
    if not _is_flux2_mflux_generator(generator):
        return _ensure_transparent_prompt(prompt)
    p = prompt.strip()
    if "transparent" in p.lower():
        # FLUX interprets this as "draw the transparency grid" — strip it.
        p = re.sub(r",?\s*transparent\s+background\b", "", p, flags=re.I).strip(" ,")
    low = p.lower()
    if "solid pure white" in low or "plain white background" in low:
        return p
    return f"{p}, {_FLUX2_BG_SUFFIX}" if p else _FLUX2_BG_SUFFIX


def _is_full_bleed_asset_name(name: str) -> bool:
    """Skip transparency suffix for full-screen background assets."""
    n = name.lower()
    return n.startswith("bg_") or n.endswith("_background") or n == "background"


def _resize_and_chroma_sprite(pil_img, size: tuple[int, int]) -> tuple[Any, dict]:
    """Resize to target game size, then chroma-key to RGBA transparent PNG."""
    from PIL import Image as _Image
    if size != (pil_img.width, pil_img.height):
        resized = pil_img.resize(size, _Image.LANCZOS)
    else:
        resized = pil_img
    return _chroma_key_to_rgba(resized)


# 1.3 — chroma-key pass. Z-Image-Turbo (and most diffusion models)
# render with a uniform background even when the prompt says
# "transparent background". The model used to be asked to clean the
# white square at runtime via getImageData / pixel manipulation,
# which CORS-tainted the canvas (see games/traces/using-great-graphics-
# that-you_20260507_103355 for the cascade failure). Right fix is to
# do the chroma-key once, in PIL, before the PNG ever reaches the
# game. RGBA output → drawImage just works with full alpha.

def _is_neutral_bg_pixel(r: int, g: int, b: int) -> bool:
    """True for light gray/white pixels suitable as sprite backdrop."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 180:
        return False
    return (mx - mn) <= 40


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


def _detect_checkerboard_bg_colors(img) -> list[tuple[int, int, int]] | None:
    """When FLUX2 draws a fake transparency grid, return its two neutral tones.

    Samples the full border strip. Requires two distinct neutral clusters
    covering >= 60% of border pixels — avoids masking multicolor scenes.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    px = img.load()
    border: list[tuple[int, int, int]] = []
    for x in range(w):
        border.append(px[x, 0])
        border.append(px[x, h - 1])
    for y in range(h):
        border.append(px[0, y])
        border.append(px[w - 1, y])
    if not border:
        return None

    tol = 16
    clusters: list[tuple[tuple[int, int, int], int]] = []
    for s in border:
        if not _is_neutral_bg_pixel(s[0], s[1], s[2]):
            continue
        matched = False
        for i, (center, count) in enumerate(clusters):
            if all(abs(s[j] - center[j]) <= tol for j in range(3)):
                clusters[i] = (center, count + 1)
                matched = True
                break
        if not matched:
            clusters.append((s, 1))
    if len(clusters) < 2:
        return None
    clusters.sort(key=lambda item: item[1], reverse=True)
    c1, n1 = clusters[0]
    c2, n2 = clusters[1]
    if (n1 + n2) / len(border) < 0.6:
        return None
    if sum(abs(c1[i] - c2[i]) for i in range(3)) < 20:
        return None
    return [c1, c2]


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


def _apply_chroma_key_alpha_multi(
    img,
    bgs: list[tuple[int, int, int]],
    tolerance: int = 24,
) -> tuple[Any, float]:
    """Key multiple backdrop colors (e.g. FLUX2 checkerboard white + gray)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    px = img.load()
    w, h = img.size
    masked = 0
    total = w * h
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            for bg_r, bg_g, bg_b in bgs:
                if (
                    abs(r - bg_r) <= tolerance
                    and abs(g - bg_g) <= tolerance
                    and abs(b - bg_b) <= tolerance
                ):
                    px[x, y] = (r, g, b, 0)
                    masked += 1
                    break
    return img, (masked / total if total else 0.0)


def _chroma_key_to_rgba(pil_img) -> tuple[Any, dict]:
    """Top-level helper: detect background color, apply alpha mask,
    return the RGBA image plus a small stats dict for tracing.

    Stats dict shape:
      {"bg_color": (r,g,b) | None, "alpha_pixel_ratio": float,
       "checkerboard_bg": [(r,g,b), ...] | omitted}

    If no dominant bg color was detected, tries a two-tone checkerboard
    fallback (FLUX2-klein fake transparency grid). If that also fails,
    converts to RGBA without masking.
    """
    stats: dict[str, Any] = {"bg_color": None, "alpha_pixel_ratio": 0.0}
    bg = _detect_bg_color(pil_img)
    if bg is not None:
        stats["bg_color"] = bg
        keyed, ratio = _apply_chroma_key_alpha(pil_img, bg)
        stats["alpha_pixel_ratio"] = round(ratio, 3)
        return keyed, stats
    checker = _detect_checkerboard_bg_colors(pil_img)
    if checker:
        stats["bg_color"] = checker[0]
        stats["checkerboard_bg"] = [list(c) for c in checker]
        keyed, ratio = _apply_chroma_key_alpha_multi(pil_img, checker)
        stats["alpha_pixel_ratio"] = round(ratio, 3)
        return keyed, stats
    # No clear bg — convert mode but skip masking.
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    return pil_img, stats


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
        "  let _assetsReady = false;",
        "  const PATHS = {",
    ]
    for name, path in asset_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"    '{name}': './{rel}',")
    lines += [
        "  };",
        "  async function loadAssets() {",
        "    for (const [name, src] of Object.entries(PATHS)) {",
        "      const img = new Image();",
        "      img.src = src;",
        "      ASSETS[name] = img;",
        "      try { await img.decode(); } catch (e) { console.warn('asset failed', name, e); }",
        "    }",
        "    for (const k in _spriteCache) delete _spriteCache[k];",
        "    _assetsReady = true;",
        "  }",
        "  // COPY the exact PATHS keys above — do NOT invent alternate key names.",
        "  // 2. Start RAF IMMEDIATELY; load assets in the background:",
        "  //    Never gate requestAnimationFrame behind every image decode.",
        "  //    If one PNG fails, the game must still render fallbacks and respond.",
        "  requestAnimationFrame(frame);",
        "  loadAssets();",
        "",
        "  // 3. ALWAYS fetch sprites through this resolver — NEVER index ASSETS",
        "  //    directly. It tolerates small key-naming differences, the #1 cause",
        "  //    of 'I generated art but the game shows colored boxes': you build",
        "  //    key 'left_idle' but the asset is 'left_fighter_idle', so",
        "  //    ASSETS['left_idle'] is undefined and you silently draw a rectangle.",
        "  //    Resolves by exact, then normalized, then token match; caches result.",
        "  const _spriteCache = {};",
        "  function sprite(key) {",
        "    if (_spriteCache[key] !== undefined) {",
        "      // Retry while loadAssets() is still populating ASSETS — Chrome often",
        "      // draws before the dict is full; an early null cache is permanent",
        "      // until reload (Safari usually wins the race so art 'just works').",
        "      if (_spriteCache[key] === null && !_assetsReady) delete _spriteCache[key];",
        "      else return _spriteCache[key];",
        "    }",
        "    let img = ASSETS[key];",
        "    if (!img) {",
        "      const norm = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');",
        "      const want = norm(key), names = Object.keys(ASSETS);",
        "      let hit = names.find(n => norm(n) === want)",
        "             || names.find(n => norm(n).includes(want) || want.includes(norm(n)));",
        "      if (!hit) {",
        "        const toks = want.match(/[a-z]+|[0-9]+/g) || []; let best = 0;",
        "        let ties = [];",
        "        for (const n of names) { const nn = norm(n); let sc = 0;",
        "          for (const t of toks) if (t.length >= 3 && nn.includes(t)) sc += t.length;",
        "          if (sc > best) { best = sc; ties = [n]; }",
        "          else if (sc === best && sc > 0) ties.push(n); }",
        "        if (ties.length === 1) hit = ties[0];",
        "        else if (ties.length > 1) {",
        "          const etoks = want.match(/[a-z]+\\d+|\\d+|[a-z]+/g) || [];",
        "          let tb = -1, tbHit = ties[0];",
        "          for (const n of ties) {",
        "            const nn = norm(n); let sc2 = 0;",
        "            for (const t of etoks) if (t.length >= 2 && nn.includes(t)) sc2 += t.length;",
        "            if (sc2 > tb) { tb = sc2; tbHit = n; }",
        "            else if (sc2 === tb) {",
        "              let pref = 0;",
        "              for (let i = 0; i < want.length && i < nn.length && want[i] === nn[i]; i++) pref++;",
        "              const oh = norm(tbHit); let oldPref = 0;",
        "              for (let i = 0; i < want.length && i < oh.length && want[i] === oh[i]; i++) oldPref++;",
        "              if (pref > oldPref) tbHit = n;",
        "            }",
        "          }",
        "          hit = tbHit;",
        "        }",
        "      }",
        "      img = hit ? ASSETS[hit] : null;",
        "    }",
        "    // Cache the resolved IMAGE OBJECT, NOT a readiness verdict. A key that",
        "    // matches an asset name is cached even while its PNG is still decoding",
        "    // — the live Image's naturalWidth flips to >0 once decode finishes, so",
        "    // the next frame draws it (self-healing). Do the naturalWidth check at",
        "    // DRAW time (step 4), never here: caching `null` for a still-decoding",
        "    // image makes the miss permanent and is the #1 'reload a few times and",
        "    // the art appears' bug. Cache null ONLY when the key matches no asset",
        "    // name at all (a true, permanent miss), and record THAT in",
        "    // window.__assetMisses so the harness can fail a real key-drift bug.",
        "    // Never cache null before _assetsReady — ASSETS may still be filling.",
        "    if (img) {",
        "      _spriteCache[key] = img;",
        "    } else if (_assetsReady) {",
        "      (window.__assetMisses = window.__assetMisses || {})[key] = 1;",
        "      _spriteCache[key] = null;",
        "    }",
        "    return img || null;",
        "  }",
        "  function drawEntity(key, x, y, w, h) {",
        "    const img = sprite(key);",
        "    if (img && img.complete && img.naturalWidth > 0) {",
        "      ctx.drawImage(img, x - w / 2, y - h / 2, w, h);",
        "    } else {",
        "      ctx.fillStyle = '#f0f'; ctx.fillRect(x - w / 2, y - h / 2, w, h);",
        "      ctx.fillStyle = '#000'; ctx.font = '12px monospace';",
        "      ctx.fillText('MISSING ' + key, x - w / 2 + 2, y - h / 2 + 14);",
        "    }",
        "  }",
        "",
        "  // 4. In draw(): call drawEntity(PATHS_KEY, ...) for EVERY entity.",
        "  //    Use the exact PATHS key strings — never invent aliases.",
        "  drawEntity('example_key', cx, cy, 64, 64);",
        "",
        "  // 5. Readiness check at DRAW time (same as drawEntity):",
        "  const img = sprite(key);",
        "  if (img && img.complete && img.naturalWidth > 0) {",
        "    ctx.drawImage(img, x, y, w, h);",
        "  } else {",
        "    ctx.fillStyle = '#f0f'; ctx.fillRect(x, y, w, h);",
        "    ctx.fillStyle = '#000'; ctx.font = '12px monospace';",
        "    ctx.fillText('MISSING ' + key, x + 2, y + 14);",
        "  }",
        "",
        "MANDATORY ITER-1 WIRING — every PATHS key MUST appear in draw() via",
        "drawEntity() or ctx.drawImage(sprite('KEY'), ...) on this turn:",
    ]
    names_checklist = ", ".join(asset_paths.keys())
    lines.append(f"  {names_checklist}")
    lines += [
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
        "ANIMATION — sprites only, no code-drawn limbs: to animate a move "
        "(punch/kick/etc), CYCLE its sprite frames with drawImage over the "
        "active window. NEVER draw a character's arm/leg/fist/body with "
        "ctx.fillRect/arc/lineTo on top of the sprite — those code-drawn "
        "limbs are exactly what users reject. To make NEW pose frames, emit "
        "<assets> as txt2img with the SAME detailed character description "
        "(same hair/gi/headband/build/style) and only change the pose clause "
        "('arm fully extended', 'leg raised high') — do NOT use `from_image` "
        "for a pose change: it returns the idle pose at low strength and a "
        "different character at high strength."
    )
    lines.append("")
    # Sprite orientation/facing guidance lives in the playbook
    # (directional-art-faces-right), not hardcoded here — keeps the asset
    # pipeline free of art-direction policy.
    lines.append(
        "============================================================"
    )
    return "\n".join(lines)
