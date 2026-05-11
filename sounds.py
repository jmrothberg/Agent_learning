"""Per-session sound generation pipeline (Stable Audio Open, no server).

Sibling of assets.py — same lazy-load + cache + per-session-dir pattern,
but for short audio clips instead of sprite PNGs. The model can declare
a `<sounds>` block in Phase A:

    <sounds>
    [
      {"name": "laser",     "prompt": "short retro arcade laser shot, 8-bit synth blip",     "duration": 0.4},
      {"name": "explosion", "prompt": "short pixelated explosion, 8-bit boom",                "duration": 0.8},
      {"name": "music",     "prompt": "loopable 8-bit chiptune background, 90 bpm, upbeat",   "duration": 12.0, "loop": true}
    ]
    </sounds>

If a torch device + `diffusers.StableAudioPipeline` are reachable, this
module:

  1. Parses the JSON list out of the planning reply.
  2. Lazy-loads `StableAudioGenerator(model_id="stable-audio-open-small")`
     (free until the first call — the import + pipeline init only happen
     if the model actually requested sounds).
  3. Generates each missing OGG (cache hit by sha256 of (model, prompt,
     duration_s) so re-runs are free).
  4. Saves OGGs into `games/<slug>_<ts>_sounds/<name>.ogg` next to the
     working HTML file. The first-build prompt is later prepended with
     `render_sound_paths_block(...)` so the model knows the paths and
     the recommended `new Audio(...)` loader pattern.

Fully optional. When no `<sounds>` tag is emitted, OR no GPU /
diffusers is reachable, this module is a no-op and the agent proceeds
without audio exactly as before.

Why Stable Audio Open and not AudioCraft / AudioGen / MusicGen:
  - Same `diffusers` library the project already uses for Z-Image-Turbo
    sprites. One model, one runtime, one mental model.
  - Single model handles BOTH short SFX and looping music (up to 47s).
  - Native 44.1 kHz stereo output → directly usable as <audio> source.
  - Cross-platform: works on CUDA (Linux) and MPS (macOS) via the same
    diffusers device path used by assets.py.

Output format: OGG Vorbis. Smaller than WAV (~5x for ambient sounds);
browser-supported on every modern desktop browser; lossless-enough for
game SFX. Encoded via the `soundfile` package (libsndfile bindings,
cross-platform wheel on PyPI).

License note: Stable Audio Open ships under the Stability AI Community
License. Non-commercial use is freely permitted; commercial use may
require a paid Stability tier — match assets.py's posture and document
this in the user-facing README, not at runtime.
"""

from __future__ import annotations

import hashlib
import json
import os as _os
import re
import sys
from pathlib import Path
from typing import Any


# Per-sound default duration in seconds. Most game SFX (laser, jump,
# coin pickup) want < 1s; ambient pads / loops want 8-16s. 1.0 is the
# sensible middle default — overridden per-sound by the model.
_DEFAULT_DURATION_S = 1.0

# Hard caps so a chatty plan can't burn 5 minutes of GPU time. The
# typical session asks for 4-8 sounds totaling under 30s of audio.
_MAX_SOUNDS_PER_TURN = 8
_MAX_DURATION_S = 12.0
_MIN_DURATION_S = 0.2

# Where Stable Audio weights live on disk. Cross-platform — works on
# Linux (Models_Diffusers convention) and macOS (Diffusion_Models
# convention). The audio directory is a per-platform sibling of the
# image directory so users can manage them independently.
#
# Search order — first existing directory wins:
#   1. $AUDIO_MODELS_DIR env var (preferred override; audio-specific)
#   2. $DIFFUSION_MODELS_DIR/audio    (so the existing image override
#                                      naturally extends to audio)
#   3. Platform default bases (mirror of assets.py — see
#      _default_model_search_dirs): ~/.Diffusion_Models (hidden) before
#      ~/Diffusion_Models, etc.
#   4. HuggingFace fallback: `stabilityai/stable-audio-open-1.0` is
#      downloaded to ~/.cache/huggingface/hub/ on first run if no local
#      path matches (gated — HF login required).


def _default_model_search_dirs() -> list[str]:
    """Build the search list at import time. Mirrors assets.py — hidden
    ~/.Diffusion_Models / ~/.Models_Diffusers before visible siblings;
    Mac-first vs Linux-first ordering matches sprites."""
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
# `stable-audio-open-1.0` is the diffusers-compatible Stability audio
# model (~5 GB). The newer `stable-audio-open-small` exists on HF but
# ships with the `stable_audio_tools` layout, not the diffusers
# `model_index.json` shape — `StableAudioPipeline.from_pretrained` 404s
# on it. Stick with 1.0.
#
# IMPORTANT — gated model: download requires accepting Stability's
# license on the HF web page once, then `huggingface-cli login` (or
# `HF_TOKEN=<token>`) so diffusers can authenticate. Without this,
# from_pretrained returns a 401/403 on first call.
_HF_FALLBACK_MODEL_ID = "stabilityai/stable-audio-open-1.0"

_SOUNDS_RE = re.compile(
    r"<sounds>\s*(.*?)\s*</sounds>", re.DOTALL | re.IGNORECASE,
)
# Truncated case — matches the same tolerance pattern in assets.py.
# When a stream cuts off mid-block (long planning turn exhausting tokens),
# we still try to recover a partial list.
_SOUNDS_OPEN_RE = re.compile(
    r"<sounds>\s*(\[.*?)$", re.DOTALL | re.IGNORECASE,
)


def _extract_sounds_body(reply: str) -> str | None:
    """Pull the body of a <sounds>...</sounds> block, tolerating a
    missing closing tag. Returns None when nothing usable was found.

    Reasoning prose stripped first — see assets._strip_thinking. Same
    failure mode applies here: a reasoning model that mentions
    `<sounds>` in its CoT prose would otherwise corrupt the parse.
    """
    from assets import _strip_thinking
    reply = _strip_thinking(reply)
    m = _SOUNDS_RE.search(reply)
    if m:
        return m.group(1)
    m = _SOUNDS_OPEN_RE.search(reply)
    if m:
        return m.group(1)
    return None


def _try_repair_truncated_json_list(text: str) -> list[Any]:
    """Best-effort recovery of a JSON list whose stream was cut off.

    Same shape as assets.py's helper — drops the incomplete trailing
    entry and synthesizes a closing `]`. Returns [] when recovery fails.
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


def _parse_duration(raw: Any) -> float:
    """Accept int, float, or numeric string; clamp to [_MIN, _MAX]."""
    try:
        d = float(raw)
    except Exception:
        return _DEFAULT_DURATION_S
    if d != d:  # NaN
        return _DEFAULT_DURATION_S
    return max(_MIN_DURATION_S, min(_MAX_DURATION_S, d))


def parse_sounds_block(reply: str) -> list[dict]:
    """Extract the JSON list inside <sounds>...</sounds>.

    Tolerant of fenced ```json wrappers AND truncated streams.

    Returns [] if no <sounds> opener is present or recovery fails.
    Each returned dict has keys: name (str), prompt (str), duration (float),
    loop (bool). Specs missing name OR prompt are dropped.
    """
    if not reply:
        return []
    body = _extract_sounds_body(reply)
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
    # Dedupe by (normalized prompt, duration, loop). Mirrors the same
    # fix in assets.parse_assets_block — protects against the model
    # spamming numbered variants of the same audio template.
    seen_keys: set[tuple[str, float, bool]] = set()
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"sound_{i + 1}").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not name or not prompt:
            continue
        duration = _parse_duration(item.get("duration", _DEFAULT_DURATION_S))
        loop = bool(item.get("loop", False))
        norm_prompt = " ".join(prompt.lower().split())
        key = (norm_prompt, round(duration, 2), loop)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "name": name, "prompt": prompt,
            "duration": duration, "loop": loop,
        })
        if len(out) >= _MAX_SOUNDS_PER_TURN:
            break
    return out


def _cache_key(model_id: str, prompt: str, duration_s: float) -> str:
    """sha256 of (model_id, normalized prompt, duration) → 32-hex.

    Whitespace and case in the prompt are normalized so trivial
    formatting differences don't bust the cache.
    """
    norm_prompt = " ".join(prompt.strip().lower().split())
    norm = f"{model_id}|{norm_prompt}|{duration_s:.2f}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_filename(name: str) -> str:
    """Clean a sound name for filesystem use. Caps at 48 chars."""
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned[:48] or "sound"


# ---------------------------------------------------------------------------
# Stable Audio Open loader (mirrors ZImageTurboGenerator in shape)
# ---------------------------------------------------------------------------


def _resolve_stable_audio_path() -> str:
    """Find Stable Audio Open weights on disk, or return the HF model ID
    so diffusers downloads on first run. Search order:
      1. $AUDIO_MODELS_DIR (preferred override)
      2. $DIFFUSION_MODELS_DIR/audio (extends image dir naturally)
      3. _MODEL_SEARCH_DIRS (the user's standard layout)
      4. HuggingFace hub fallback ID
    """
    candidates: list[str] = []
    audio_dir = (_os.environ.get("AUDIO_MODELS_DIR") or "").strip()
    if audio_dir:
        candidates.extend([
            _os.path.join(audio_dir, "stable-audio-open-1.0"),
            _os.path.join(audio_dir, "stable-audio-open"),
        ])
    diff_dir = (_os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    if diff_dir:
        candidates.extend([
            _os.path.join(diff_dir, "audio", "stable-audio-open-1.0"),
            _os.path.join(diff_dir, "stable-audio-open-1.0"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            _os.path.join(base, "stable-audio-open-1.0"),
            _os.path.join(base, "audio", "stable-audio-open-1.0"),
        ])
    for c in candidates:
        if _os.path.isdir(c):
            return c
    return _HF_FALLBACK_MODEL_ID


class StableAudioGenerator:
    """In-process Stable Audio Open wrapper. No server, no subprocess.

    Usage:
        gen = StableAudioGenerator()
        path = gen.generate("short retro arcade laser shot", duration_s=0.4)

    The pipeline is loaded lazily on the first `.generate()` call so
    importing this module is cheap. After the first call, the model
    stays resident in GPU/MPS memory for the rest of the Python
    process — subsequent calls cost only the inference time
    (~3-8s per second of generated audio on Apple Silicon, faster
    on CUDA).
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _resolve_stable_audio_path()
        self._pipeline: Any = None
        self._device: str | None = None
        self._sample_rate: int = 44100   # set after pipeline load
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
            from diffusers import StableAudioPipeline
        except Exception as e:
            self._last_error = (
                f"import failed: {type(e).__name__}: {e!s}. "
                "Run `pip install -r requirements-diffuser.txt` (which "
                "now includes stable-audio deps) in the Agent_learning "
                "venv."
            )
            return False

        # Workaround for an INFINITE recursion bug in torchsde, triggered
        # by Stable Audio Open's default cosine-DPM-SDE scheduler. The
        # bug: `_Interval._split(midway)` bisects the interval to place
        # a child at `midway`. When float-precision drift makes `midway`
        # land ~1e-5 OUTSIDE [_start, _end] (e.g. midway=500.00006 for
        # an interval ending at 500.0), the bisection halves the
        # right-side child forever — the target is unreachable, so each
        # level recurses into the same shape. RecursionError fires
        # regardless of `sys.setrecursionlimit`.
        #
        # Fix: clamp `midway` to the interval bounds before bisecting.
        # When midway lands at or past a boundary (within epsilon), we
        # `_split_exact` at the boundary itself — a single non-recursive
        # call that produces a degenerate child but terminates cleanly.
        # Audio output is unaffected because BrownianInterval.__call__
        # already clamps `tb` to `_end` at line ~610 of brownian_interval.py;
        # the scheduler can't actually use values past the boundary anyway.
        # Idempotent — only patches once per process.
        try:
            import torchsde._brownian.brownian_interval as _bi
            _Interval = _bi._Interval
            if not getattr(_Interval, "_agent_clamp_patched", False):
                _orig_split = _Interval._split
                def _clamped_split(self, midway):
                    eps = abs(self._end - self._start) * 1e-9 + 1e-12
                    if midway >= self._end - eps:
                        return self._split_exact(self._end)
                    if midway <= self._start + eps:
                        return self._split_exact(self._start)
                    return _orig_split(self, midway)
                _Interval._split = _clamped_split
                _Interval._agent_clamp_patched = True
        except Exception:
            # If torchsde isn't importable yet, the scheduler hasn't been
            # constructed either and the bug can't fire. We'll try again
            # after pipeline load if needed (covered by the bumped
            # recursion limit below as defense in depth).
            pass

        # Defense in depth — if the patch above somehow doesn't take
        # effect (e.g. future torchsde refactor renames _split), the
        # higher recursion limit at least keeps small clips from
        # crashing. Process-global, set once.
        import sys as _sys
        if _sys.getrecursionlimit() < 10_000:
            _sys.setrecursionlimit(10_000)

        # Device selection mirrors assets.py: CUDA preferred (Linux dev
        # boxes); MPS fallback (Apple Silicon). CPU intentionally
        # excluded — Stable Audio Open inference on CPU is impractical.
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
            # Stable Audio Open's diffusion-transformer happens to be
            # numerically stable in fp16 on MPS unlike Z-Image-Turbo
            # (which NaNs out). If a future MPS regression appears,
            # downgrade to float32 here — same recovery path as the
            # image pipeline.
            dtype = torch.float16
        else:
            self._last_error = (
                "no CUDA and no MPS device available — torch sees "
                "neither. Stable Audio Open on CPU is not supported "
                "(would be minutes per second of audio)."
            )
            return False

        try:
            self._pipeline = StableAudioPipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
            )
            # Workaround for a torchsde float-precision bug. Stable Audio
            # Open's CosineDPMSolverMultistepScheduler uses torchsde's
            # BrownianInterval for SDE noise during the FINAL timestep.
            # On that step, the requested time lands ~6e-5 outside
            # [t0, t1] (e.g. tb=500.00006 vs t1=500), torchsde bisects
            # the interval to converge — but the target is unreachable,
            # so it recurses forever and dies with RecursionError.
            #
            # The scheduler exposes `euler_at_final` for exactly this
            # situation: when True, the last step uses an Euler update
            # (no SDE noise, no torchsde call) instead of the cosine-
            # DPM-solver SDE step. Audio quality is essentially
            # unchanged — the final step's job is just denoising into
            # the zero-sigma fixed point — but the bug goes away.
            try:
                self._pipeline.scheduler = type(self._pipeline.scheduler).from_config(
                    self._pipeline.scheduler.config,
                    euler_at_final=True,
                )
            except Exception:
                pass
            self._pipeline.to(device)
            self._device = device
            # Pipeline exposes the model's native sample rate via its
            # vae. Cache it so generate() can hand it to soundfile.
            try:
                self._sample_rate = int(
                    getattr(self._pipeline.vae, "sampling_rate", 44100)
                )
            except Exception:
                self._sample_rate = 44100
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

    def generate(self, prompt: str, duration_s: float = 1.0) -> str | None:
        """Run inference and save an OGG to a temp file. Returns the
        absolute path, or None on failure (caller skips that sound).
        On None, `self._last_error` carries the real exception or a
        descriptive reason.
        """
        self._last_error = None
        if not self._lazy_init():
            return None
        try:
            import tempfile
            import torch
            try:
                import soundfile as sf
            except Exception as e:
                self._last_error = (
                    f"soundfile import failed: {type(e).__name__}: {e!s}. "
                    "Run `pip install soundfile` (libsndfile wheel — "
                    "cross-platform, no system deps) in the venv."
                )
                return None

            gen = torch.Generator(self._device or "cpu").manual_seed(42)
            # Stable Audio Open's `audio_end_in_s` parameter governs
            # output length. Steps default to 50; for short SFX 25 is
            # plenty and roughly halves wall time.
            steps = 25 if duration_s <= 2.0 else 50
            result = self._pipeline(
                prompt=prompt,
                num_inference_steps=steps,
                audio_end_in_s=float(duration_s),
                generator=gen,
            )
            audios = getattr(result, "audios", None)
            if audios is None or len(audios) == 0:
                self._last_error = (
                    "pipeline returned no audios (empty .audios). "
                    "Likely an internal safety filter, OR a diffusers "
                    "API drift where the result attribute name "
                    "changed. Inspect the result object with "
                    f"type={type(result).__name__}, "
                    f"keys={list(getattr(result, '__dict__', {}).keys())}."
                )
                return None
            # audios[0] is shape (channels, samples) torch.Tensor on the
            # pipeline's device. soundfile wants (samples, channels)
            # numpy for stereo output, or 1D for mono.
            wave = audios[0].to(dtype=torch.float32).cpu().numpy()
            if wave.ndim == 2:
                # (channels, samples) → (samples, channels) for soundfile.
                wave = wave.T
            f = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
            f.close()
            sf.write(f.name, wave, self._sample_rate, format="OGG", subtype="VORBIS")
            return f.name
        except Exception as e:
            import traceback as _tb
            self._last_error = (
                f"{type(e).__name__}: {e!s} | "
                f"trace: {_tb.format_exc().splitlines()[-3:]}"
            )
            return None


# Module-level cache for a preloaded generator. Same fork-ordering
# rationale as assets.py: huggingface_hub / safetensors call
# subprocess.Popen during from_pretrained; doing it BEFORE Playwright
# opens its IPC pipes prevents `_posixsubprocess.fork_exec: bad
# value(s) in fds_to_keep`.
_PRELOADED: Any = None


def preload() -> Any:
    """Eagerly construct + load the Stable Audio Open pipeline RIGHT NOW.

    Call from your program's main entry, BEFORE any subprocess-spawning
    library opens FDs. Idempotent — subsequent calls return the cached
    instance.
    """
    global _PRELOADED
    if _PRELOADED is not None:
        return _PRELOADED
    gen = _construct_generator()
    if gen is None:
        return None
    gen._lazy_init()
    _PRELOADED = gen
    return gen


def _construct_generator() -> Any:
    """Internal: check imports and construct a wrapper. Returns None if
    torch + diffusers + soundfile aren't all available."""
    import importlib.util as _iu
    if (
        _iu.find_spec("torch") is None
        or _iu.find_spec("diffusers") is None
        or _iu.find_spec("soundfile") is None
    ):
        return None
    try:
        return StableAudioGenerator()
    except Exception:
        return None


def try_load_audio_generator(
    model_id: str = "stable-audio-open-1.0",  # kept for API stability
) -> Any:
    """Return the audio generator. If `preload()` ran earlier, reuses
    that already-loaded pipeline. Otherwise constructs a fresh wrapper
    that lazy-loads on first .generate().

    Returns None when torch+diffusers+soundfile aren't all installed.
    """
    if _PRELOADED is not None:
        return _PRELOADED
    return _construct_generator()


def generate_sounds(
    specs: list[dict],
    session_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    audio_generator: Any = None,
    model_id: str = "stable-audio-open-1.0",
) -> dict[str, Path]:
    """Generate one OGG per spec and return {name: absolute_path}.

    `specs` come from `parse_sounds_block`. Each is
    {name, prompt, duration, loop}. Returns a dict mapping name →
    absolute path of the saved OGG.

    Cache strategy: each (model_id, prompt, duration) hashes to a key
    under `cache_dir`; cache hits hardlink (or copy as fallback) to a
    stable per-session path inside `session_dir` so the HTML file's
    `<audio src>` references stay relative and predictable.

    `audio_generator` is dependency-injected for tests. When None we
    attempt `try_load_audio_generator()`; if THAT also returns None
    the function returns {} silently — the agent should log + proceed.

    Failures generating an individual sound are caught and logged via
    the returned dict's missing keys; we never abort the batch.
    """
    if not specs:
        return {}
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_root = Path(session_dir).parent / "_sound_cache"
    else:
        cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if audio_generator is None:
        audio_generator = try_load_audio_generator(model_id)
    if audio_generator is None:
        return {}

    out: dict[str, Path] = {}
    sound_stats: list[dict[str, Any]] = []
    for spec in specs:
        import time
        t0 = time.time()
        name = _safe_filename(spec["name"])
        prompt = spec["prompt"]
        duration = float(spec.get("duration", _DEFAULT_DURATION_S))
        loop = bool(spec.get("loop", False))
        key = _cache_key(model_id, prompt, duration)
        # Human-readable cache filename — same pattern as assets.py.
        # `<name>__<hash6>.ogg` keeps cache hits deterministic while
        # making the cache dir scannable.
        cache_path = cache_root / f"{name}__{key[:6]}.ogg"
        target_path = session_dir / f"{name}.ogg"
        stat: dict[str, Any] = {
            "name": name,
            "prompt": prompt[:140],
            "duration_s": duration,
            "loop": loop,
            "cache_hit": False,
            "gen_seconds": 0.0,
        }
        if cache_path.exists():
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            stat["cache_hit"] = True
            stat["gen_seconds"] = round(time.time() - t0, 3)
            sound_stats.append(stat)
            continue
        gen_path = _safe_generate(audio_generator, prompt, duration)
        if gen_path is None:
            real_err = getattr(audio_generator, "_last_error", None)
            stat["error"] = (
                f"audio diffuser failed: {real_err}" if real_err else
                "audio diffuser returned None (no exception captured — "
                "check generator's _last_error attribute)"
            )
            stat["gen_seconds"] = round(time.time() - t0, 3)
            sound_stats.append(stat)
            continue
        try:
            # Move temp OGG into cache, then link/copy into session dir.
            import shutil
            shutil.copy2(gen_path, cache_path)
            _link_or_copy(cache_path, target_path)
            out[name] = target_path.resolve()
            stat["bytes"] = target_path.stat().st_size
        except Exception as e:
            stat["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            stat["gen_seconds"] = round(time.time() - t0, 3)
            sound_stats.append(stat)
            # Best-effort cleanup of the temp file the pipeline wrote.
            try:
                _os.unlink(gen_path)
            except Exception:
                pass
    try:
        audio_generator.last_stats = sound_stats  # type: ignore[attr-defined]
    except Exception:
        pass
    return out


def _safe_generate(gen: Any, prompt: str, duration_s: float) -> str | None:
    """Wrap StableAudioGenerator.generate() so a single failure doesn't
    poison the whole batch. Stamps `gen._last_error` with the real
    traceback so the caller can surface it via per-sound stats.
    """
    try:
        return gen.generate(prompt, duration_s)
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


def _filter_existing_sounds(
    sound_paths: dict[str, Path],
) -> dict[str, Path]:
    """Drop entries whose OGG isn't on disk so the page never gets
    file:// paths to nonexistent audio."""
    kept: dict[str, Path] = {}
    dropped: list[str] = []
    for name, path in sound_paths.items():
        try:
            if Path(path).exists():
                kept[name] = path
            else:
                dropped.append(name)
        except Exception:
            dropped.append(name)
    if dropped:
        print(
            f"[sounds] dropped {len(dropped)} missing OGG path(s) "
            f"before injection: {', '.join(dropped[:5])}"
            + ("…" if len(dropped) > 5 else ""),
            flush=True,
        )
    return kept


def render_sound_paths_block(
    sound_paths: dict[str, Path],
    session_html_path: Path | str,
    *,
    looping_names: set[str] | None = None,
) -> str:
    """Build the injection block listing generated sound paths.

    Paths are resolved relative to the directory of the HTML file so
    the model can `new Audio('./<name>.ogg')` directly. Empty input →
    empty string (caller should not inject).

    `looping_names` is the set of sound names that were declared with
    `loop: true` in the original <sounds> block; the loader pattern
    sets `Audio.loop = true` for those so background music doesn't
    require manual restart.

    Phrasing mirrors render_asset_paths_block: aggressive about
    actually using the generated assets, because mid-tier models
    default to silent games when their training distribution didn't
    emphasize <audio> elements.
    """
    if not sound_paths:
        return ""
    sound_paths = _filter_existing_sounds(sound_paths)
    if not sound_paths:
        return ""
    looping = set(looping_names or [])
    html_dir = Path(session_html_path).resolve().parent
    lines = [
        "================ GENERATED SOUNDS ================",
        "Stable Audio Open generated these OGG files and saved them",
        "next to your HTML file. YOU MUST USE THEM via `new Audio(...)`",
        "for every event the user will hear (firing, hits, pickups,",
        "background music). A silent game when sound files were",
        "generated IS A REGRESSION on this turn.",
        "",
        "ULTRA IMPORTANT — pattern you MUST follow:",
        "",
        "  // 1. Build a sound-loader (do this ONCE at startup):",
        "  const SOUNDS = {};",
        "  function loadSounds() {",
        "    const entries = [",
    ]
    for name, path in sound_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        loop_flag = "true" if name in looping else "false"
        lines.append(f"      ['{name}', './{rel}', {loop_flag}],")
    lines += [
        "    ];",
        "    for (const [name, src, loop] of entries) {",
        "      const a = new Audio(src);",
        "      a.loop = loop;",
        "      SOUNDS[name] = a;",
        "    }",
        "  }",
        "  loadSounds();",
        "",
        "  // 2. To play an SFX (one-shot, can overlap with itself):",
        "  function play(name) {",
        "    const base = SOUNDS[name]; if (!base) return;",
        "    // cloneNode lets the same SFX overlap (rapid-fire laser);",
        "    // for music use the original (`SOUNDS.music.play()`).",
        "    if (base.loop) { base.play(); return; }",
        "    const inst = base.cloneNode(); inst.play();",
        "  }",
        "",
        "  // 3. Browsers require a user gesture before audio plays.",
        "  //    Start music on first keydown / pointerdown:",
        "  let _audioStarted = false;",
        "  function _startAudio() {",
        "    if (_audioStarted) return; _audioStarted = true;",
        "    if (SOUNDS.music) SOUNDS.music.play().catch(()=>{});",
        "  }",
        "  window.addEventListener('keydown',     _startAudio, {once:true});",
        "  window.addEventListener('pointerdown', _startAudio, {once:true});",
        "",
        "Available sounds — name → relative path:",
    ]
    for name, path in sound_paths.items():
        try:
            rel = Path(path).resolve().relative_to(html_dir)
        except ValueError:
            rel = path
        suffix = " (looping background)" if name in looping else ""
        lines.append(f"  - {name}: ./{rel}{suffix}")
    lines.append("")
    lines.append(
        "If the goal mentions sound, music, audio, or hit / fire / "
        "explode events and you ship a silent game when SOUNDS were "
        "generated, you have FAILED THIS TURN. Wire `play('name')` "
        "into the relevant event handlers."
    )
    lines.append(
        "=================================================="
    )
    return "\n".join(lines)
