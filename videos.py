"""Per-session video cutscene generation pipeline (Wan2.2-TI2V-5B).

Sibling of assets.py / sounds.py — same parse + cache + per-session-dir
pattern, but for short MP4 cutscene clips. The model can declare a
`<videos>` block in Phase A (or mid-session):

    <videos>
    [
      {"name": "intro",   "prompt": "slow push-in toward a dark castle at sunset, bats circling", "image": "key_castle", "seconds": 4},
      {"name": "victory", "prompt": "the hero raises the trophy, confetti falls, camera orbits",  "seconds": 4}
    ]
    </videos>

The optional `image` field names a generated ASSET from the same session
(key art) — the clip is then image-to-video seeded from that PNG, which
keeps cutscene art consistent with in-game sprites. Without `image` the
clip is text-to-video.

Backend: `scripts/generate_video.py` run as a SUBPROCESS (never imported)
so neither mlx nor torch video deps leak into the agent process:
  - macOS (Apple Silicon): mlx-gen CLI in the dedicated `.venv-video`
    virtualenv, model `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`.
  - Linux (NVIDIA CUDA):  diffusers WanPipeline in the main `.venv`
    (torch + diffusers are already installed by install_diffuser.sh),
    model `Wan-AI/Wan2.2-TI2V-5B-Diffusers`.

Fully optional. When no `<videos>` tag is emitted, OR no usable backend
is present, this module is a no-op and the agent proceeds without
cutscenes exactly as before.

Cost note: a 4-second 832x480 clip takes ~3 minutes on an M3 Ultra —
two orders of magnitude more than a sprite. The per-turn cap is small
(4) and the prompt guidelines steer the model toward 2-4 clips per
session, cutscenes only.
"""

from __future__ import annotations

import hashlib
import json
import os as _os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent

# Per-clip duration in seconds (mapped to Wan's 4k+1 frame counts at
# 12 fps by scripts/generate_video.py). Cutscenes shorter than 2s feel
# like a glitch; longer than 8s costs >6 min of GPU per clip.
_DEFAULT_SECONDS = 4.0
_MIN_SECONDS = 2.0
_MAX_SECONDS = 8.0

# Hard cap so a chatty plan can't burn an hour of GPU. A full game
# (intro / death / victory / boss-reveal) fits in 4.
_MAX_VIDEOS_PER_TURN = 4

# Default model ids per backend — must match scripts/generate_video.py.
_MLX_MODEL_ID = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
_DIFFUSERS_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Generation can legitimately take minutes per clip; kill runaways at 30.
_GEN_TIMEOUT_S = 1800

_VIDEOS_RE = re.compile(
    r"<videos>\s*(.*?)\s*</videos>", re.DOTALL | re.IGNORECASE,
)
# Truncated case — same tolerance pattern as assets.py / sounds.py.
_VIDEOS_OPEN_RE = re.compile(
    r"<videos>\s*(\[.*?)$", re.DOTALL | re.IGNORECASE,
)


def _extract_videos_body(reply: str) -> str | None:
    """Pull the body of a <videos>...</videos> block, tolerating a
    missing closing tag. Reasoning prose stripped first (a model that
    mentions `<videos>` in CoT would otherwise corrupt the parse)."""
    from assets import _strip_thinking
    reply = _strip_thinking(reply)
    m = _VIDEOS_RE.search(reply)
    if m:
        return m.group(1)
    m = _VIDEOS_OPEN_RE.search(reply)
    if m:
        return m.group(1)
    return None


def _try_repair_truncated_json_list(text: str) -> list[Any]:
    """Best-effort recovery of a JSON list whose stream was cut off.
    Same shape as sounds.py's helper."""
    text = text.rstrip().rstrip(",").rstrip()
    if text.endswith("]"):
        return []
    last_brace = text.rfind("}")
    if last_brace < 0:
        return []
    candidate = text[: last_brace + 1] + "]"
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _parse_seconds(raw: Any) -> float:
    """Accept int, float, or numeric string; clamp to [_MIN, _MAX]."""
    try:
        s = float(raw)
    except Exception:
        return _DEFAULT_SECONDS
    if s != s:  # NaN
        return _DEFAULT_SECONDS
    return max(_MIN_SECONDS, min(_MAX_SECONDS, s))


def parse_videos_block(reply: str) -> list[dict]:
    """Extract the JSON list inside <videos>...</videos>.

    Tolerant of fenced ```json wrappers AND truncated streams.

    Returns [] if no <videos> opener is present or recovery fails.
    Each returned dict has keys: name (str), prompt (str),
    seconds (float), image (str | None — session asset name to seed
    image-to-video). Specs missing a prompt are dropped; a missing
    name auto-fills as `video_<i>`.
    """
    if not reply:
        return []
    body = _extract_videos_body(reply)
    if body is None:
        return []
    body = body.strip()
    body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
    body = re.sub(r"\n?```$", "", body).strip()
    try:
        obj = json.loads(body)
    except Exception:
        obj = _try_repair_truncated_json_list(body)
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
    # Dedupe by (normalized prompt, seconds, image) — protects against
    # numbered-variant template spam, mirroring assets/sounds.
    seen_keys: set[tuple[str, float, str]] = set()
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"video_{i + 1}").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        # "duration" accepted as an alias for "seconds" — models copy the
        # <sounds> schema key (observed live, 20260611 snake-cutscene run).
        seconds = _parse_seconds(
            item.get("seconds", item.get("duration", _DEFAULT_SECONDS))
        )
        image = str(item.get("image") or "").strip() or None
        norm_prompt = " ".join(prompt.lower().split())
        key = (norm_prompt, round(seconds, 1), image or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "name": name, "prompt": prompt,
            "seconds": seconds, "image": image,
        })
        if len(out) >= _MAX_VIDEOS_PER_TURN:
            break
    return out


def _cache_key(model_id: str, prompt: str, seconds: float, image_sig: str) -> str:
    """sha256 of (model, normalized prompt, seconds, image signature)."""
    norm_prompt = " ".join(prompt.strip().lower().split())
    norm = f"{model_id}|{norm_prompt}|{seconds:.1f}|{image_sig}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_filename(name: str) -> str:
    """Clean a clip name for filesystem use. Caps at 48 chars."""
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned[:48] or "video"


# ---------------------------------------------------------------------------
# Backend probe + subprocess generator
# ---------------------------------------------------------------------------


def _mlxgen_path() -> Path:
    """Location of the mlxgen CLI in the dedicated video venv."""
    venv = Path(_os.environ.get("VIDEO_VENV", _REPO_ROOT / ".venv-video"))
    return venv / "bin" / "mlxgen"


def default_video_model_id() -> str:
    """Model id used by the active backend (env override wins)."""
    env = (_os.environ.get("VIDEO_MODEL") or "").strip()
    if env:
        return env
    return _MLX_MODEL_ID if sys.platform == "darwin" else _DIFFUSERS_MODEL_ID


class VideoGenerator:
    """Subprocess wrapper around scripts/generate_video.py.

    Unlike the image/audio generators (in-process diffusers), video
    generation always runs in a child process: on macOS the actual
    model lives in `.venv-video` (mlx-gen pins its own mlx), and on
    Linux keeping Wan's ~10 GB of VRAM out of the agent process means
    the weights are released the moment a clip finishes.
    """

    def __init__(self) -> None:
        self._last_error: str | None = None
        self.last_stats: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        out_path: Path,
        *,
        seconds: float = _DEFAULT_SECONDS,
        image_path: Path | None = None,
    ) -> str | None:
        """Run one clip. Returns str(out_path) on success, None on
        failure (with `_last_error` set)."""
        self._last_error = None
        # 12 fps container; frames must be 4k+1 for Wan's VAE.
        frames = int(round(seconds * 12 / 4)) * 4 + 1
        frames = max(17, min(97, frames))
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "generate_video.py"),
            "--prompt", prompt,
            "--out", str(out_path),
            "--frames", str(frames),
        ]
        if image_path is not None:
            cmd += ["--image", str(image_path)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_GEN_TIMEOUT_S,
                cwd=str(_REPO_ROOT),
            )
        except subprocess.TimeoutExpired:
            self._last_error = f"video generation timed out after {_GEN_TIMEOUT_S}s"
            return None
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e!s}"
            return None
        if proc.returncode != 0 or not Path(out_path).exists():
            tail = "\n".join(
                (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
            )
            self._last_error = (
                f"generate_video.py rc={proc.returncode}: {tail[:400]}"
            )
            return None
        return str(out_path)


def video_backend_status() -> tuple[bool, str]:
    """(usable, human reason) for the video backend on this machine."""
    if sys.platform == "darwin":
        exe = _mlxgen_path()
        if exe.exists():
            return True, f"mlx-gen at {exe}"
        return False, (
            f"mlxgen not found at {exe} — run ./scripts/setup.sh (or: "
            "python3 -m venv .venv-video && .venv-video/bin/pip install -U mlx-gen)"
        )
    import importlib.util as _iu
    if _iu.find_spec("torch") is None or _iu.find_spec("diffusers") is None:
        return False, (
            "torch/diffusers not installed — run ./scripts/install_diffuser.sh"
        )
    return True, "diffusers WanPipeline (CUDA)"


def try_load_video_generator() -> Any:
    """Return a VideoGenerator when a backend is usable, else None.
    Cheap — no model load happens here (the subprocess loads weights
    per batch)."""
    usable, _reason = video_backend_status()
    if not usable:
        return None
    return VideoGenerator()


# ---------------------------------------------------------------------------
# Batch generation (mirrors generate_sounds)
# ---------------------------------------------------------------------------


def generate_videos(
    specs: list[dict],
    session_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    video_generator: Any = None,
    asset_paths: dict[str, Path] | None = None,
    model_id: str | None = None,
) -> dict[str, Path]:
    """Generate one MP4 per spec and return {name: absolute_path}.

    `specs` come from `parse_videos_block`. Each is
    {name, prompt, seconds, image}. The `image` field names a session
    ASSET (looked up in `asset_paths`) used as the image-to-video first
    frame; unknown / missing image names degrade to text-to-video
    rather than failing the clip.

    Cache strategy: each (model, prompt, seconds, image signature)
    hashes to a key under `cache_dir`; cache hits hard-link (or copy)
    to a stable per-session path inside `session_dir` so the HTML
    file's `<video src>` references stay relative and predictable.

    `video_generator` is dependency-injected for tests. When None we
    attempt `try_load_video_generator()`; if THAT also returns None the
    function returns {} silently — the agent should log + proceed.
    """
    if not specs:
        return {}
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_root = Path(session_dir).parent / "_video_cache"
    else:
        cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if video_generator is None:
        video_generator = try_load_video_generator()
    if video_generator is None:
        return {}
    mid = model_id or default_video_model_id()

    out: dict[str, Path] = {}
    video_stats: list[dict[str, Any]] = []
    for spec in specs:
        import time
        t0 = time.time()
        name = _safe_filename(spec["name"])
        prompt = spec["prompt"]
        seconds = float(spec.get("seconds", _DEFAULT_SECONDS))
        image_name = spec.get("image")
        # Resolve the seed image: a session asset name (preferred) or a
        # path that exists on disk. Unknown name → degrade to T2V.
        image_path: Path | None = None
        if image_name:
            cand = (asset_paths or {}).get(str(image_name))
            if cand is None:
                # also accept the filesystem-safe variant of the name
                cand = (asset_paths or {}).get(_safe_filename(str(image_name)))
            if cand is not None and Path(cand).exists():
                image_path = Path(cand)
            elif Path(str(image_name)).exists():
                image_path = Path(str(image_name))
        if image_path is not None:
            try:
                st = image_path.stat()
                image_sig = f"{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                image_sig = "missing"
        else:
            image_sig = ""
        key = _cache_key(mid, prompt, seconds, image_sig)
        cache_path = cache_root / f"{name}__{key[:6]}.mp4"
        target_path = session_dir / f"{name}.mp4"
        stat: dict[str, Any] = {
            "name": name,
            "prompt": prompt[:140],
            "seconds": seconds,
            "image": str(image_name or ""),
            "i2v": image_path is not None,
            "cache_hit": False,
            "gen_seconds": 0.0,
        }
        if cache_path.exists():
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            stat["cache_hit"] = True
            stat["gen_seconds"] = round(time.time() - t0, 3)
            video_stats.append(stat)
            try:
                video_generator.last_stats = list(video_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue
        gen_path = _safe_generate(
            video_generator, prompt, target_path,
            seconds=seconds, image_path=image_path,
        )
        if gen_path is None:
            real_err = getattr(video_generator, "_last_error", None)
            stat["error"] = (
                f"video backend failed: {real_err}" if real_err else
                "video backend returned None (no exception captured)"
            )
            stat["gen_seconds"] = round(time.time() - t0, 3)
            video_stats.append(stat)
            try:
                video_generator.last_stats = list(video_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
            continue
        try:
            import shutil
            # The generator wrote straight to target_path; copy into the
            # cache so a re-run (or another session) gets it for free.
            shutil.copy2(target_path, cache_path)
            out[name] = target_path.resolve()
            stat["bytes"] = target_path.stat().st_size
        except Exception as e:
            stat["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            stat["gen_seconds"] = round(time.time() - t0, 3)
            video_stats.append(stat)
            try:
                video_generator.last_stats = list(video_stats)  # type: ignore[attr-defined]
            except Exception:
                pass
    try:
        video_generator.last_stats = video_stats  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def _safe_generate(
    gen: Any,
    prompt: str,
    out_path: Path,
    *,
    seconds: float,
    image_path: Path | None,
) -> str | None:
    """Wrap VideoGenerator.generate() so one failure doesn't poison the
    batch. Stamps `gen._last_error` with the real traceback."""
    try:
        return gen.generate(
            prompt, out_path, seconds=seconds, image_path=image_path,
        )
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


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink first (free); fall back to copy. Same as assets.py."""
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


def _filter_existing_videos(video_paths: dict[str, Path]) -> dict[str, Path]:
    """Drop entries whose MP4 isn't on disk so the page never gets
    file:// paths to nonexistent video."""
    kept: dict[str, Path] = {}
    for name, path in video_paths.items():
        try:
            if Path(path).exists():
                kept[name] = path
        except Exception:
            pass
    return kept


def render_video_paths_block(
    video_paths: dict[str, Path],
    session_html_path: Path | str,
) -> str:
    """Build the injection block listing generated cutscene paths.

    Paths are resolved relative to the directory of the HTML file. The
    loader pattern is the one proven in dragons-lair-deluxe.html: ONE
    absolutely-positioned <video> overlay covering the canvas, played
    by name at phase changes, skippable on any key, and the game must
    NEVER stall when a clip is missing or fails to start.
    """
    if not video_paths:
        return ""
    video_paths = _filter_existing_videos(video_paths)
    if not video_paths:
        return ""
    html_dir = Path(session_html_path).resolve().parent
    lines = [
        "================ GENERATED CUTSCENE VIDEOS ================",
        "Wan2.2 generated these MP4 cutscene clips and saved them next",
        "to your HTML file. Play them as full-screen overlays at the",
        "matching moments (intro on start, death clip on life lost,",
        "victory on win, ...). They have NO audio track.",
        "",
        "ULTRA IMPORTANT — pattern you MUST follow:",
        "",
        "  <!-- in <body>, positioned over the canvas: -->",
        "  <video id=\"cut\" muted playsinline preload=\"auto\"",
        "         style=\"position:absolute;inset:0;width:100%;height:100%;",
        "                object-fit:cover;display:none\"></video>",
        "",
        "  // play a cutscene by name; onDone continues the game.",
        "  const CUTS = {",
    ]
    for name, path in video_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"    {name}: './{rel}',")
    lines += [
        "  };",
        "  const vid = document.getElementById('cut');",
        "  let cutDone = null;",
        "  function playCut(name, onDone) {",
        "    const src = CUTS[name];",
        "    if (!src) { onDone(); return; }      // unknown -> skip",
        "    cutDone = onDone;",
        "    vid.onended = endCut; vid.onerror = endCut;",
        "    vid.src = src; vid.style.display = 'block';",
        "    vid.play().catch(endCut);            // missing/blocked -> skip",
        "  }",
        "  function endCut() {",
        "    if (!cutDone) return;",
        "    vid.style.display = 'none'; vid.pause();",
        "    const f = cutDone; cutDone = null; f();",
        "  }",
        "  // any key skips a running cutscene:",
        "  window.addEventListener('keydown', () => { if (cutDone) endCut(); });",
        "",
        "HARD RULES: the game must NEVER stall waiting on a video —",
        "every failure path (missing file, autoplay blocked, decode",
        "error) must call the same onDone continuation. Keep gameplay",
        "rendering on the canvas; the <video> overlay is ONLY for",
        "cutscene moments.",
        "",
        "Available cutscenes — name → relative path:",
    ]
    for name, path in video_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        lines.append(f"  - {name}: ./{rel}")
    lines.append(
        "==========================================================="
    )
    return "\n".join(lines)
