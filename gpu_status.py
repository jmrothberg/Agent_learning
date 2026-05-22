"""Read-only GPU placement helpers for the TUI status panel.

Parses `nvidia-smi` (when available) to map PIDs to GPU indices and
formats labels for LLM backends and in-process diffusers. Best-effort:
never raises; returns empty structures when nvidia-smi is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


_SNAPSHOT_TTL_S = 2.5
# Per-process VRAM at or above this → treat GPU as LLM-resident (diffusers avoid).
_LLM_PROC_VRAM_MIB = 8000
_LLM_OLLAMA_SLICE_MIB = 2000
_cached_snapshot: tuple[float, "GpuSnapshot | None"] | None = None


@dataclass
class GpuProcess:
    gpu_index: int
    pid: int
    process_name: str
    memory_mib: int | None = None


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None


@dataclass
class GpuSnapshot:
    gpus: list[GpuInfo] = field(default_factory=list)
    processes: list[GpuProcess] = field(default_factory=list)
    # uuid -> gpu index
    _uuid_to_index: dict[str, int] = field(default_factory=dict, repr=False)


def _parse_mib(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(" MiB", "").replace("MiB", "").strip()
    try:
        return int(float(s))
    except ValueError:
        return None


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if r.returncode != 0:
        return ""
    return r.stdout or ""


def snapshot_gpus(*, force: bool = False) -> GpuSnapshot | None:
    """Return a cached GPU/process map, refreshed at most every ~2.5s."""
    global _cached_snapshot
    now = time.monotonic()
    if (
        not force
        and _cached_snapshot is not None
        and (now - _cached_snapshot[0]) < _SNAPSHOT_TTL_S
    ):
        return _cached_snapshot[1]

    out = GpuSnapshot()
    gpu_csv = _run_nvidia_smi([
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not gpu_csv.strip():
        _cached_snapshot = (now, None)
        return None

    for line in gpu_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        used = _parse_mib(parts[2]) if len(parts) > 2 else None
        total = _parse_mib(parts[3]) if len(parts) > 3 else None
        out.gpus.append(GpuInfo(index=idx, name=name, memory_used_mib=used, memory_total_mib=total))

    uuid_csv = _run_nvidia_smi([
        "--query-gpu=index,uuid",
        "--format=csv,noheader",
    ])
    for line in (uuid_csv or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            out._uuid_to_index[parts[1]] = int(parts[0])
        except ValueError:
            continue

    proc_csv = _run_nvidia_smi([
        "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader",
    ])
    for line in (proc_csv or "").strip().splitlines():
        # process_name may contain commas — split from the right.
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0].strip())
        except ValueError:
            continue
        mem = _parse_mib(parts[-1])
        gpu_uuid = parts[-2].strip()
        proc_name = ",".join(parts[1:-2]).strip()
        gpu_index = out._uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue
        out.processes.append(GpuProcess(
            gpu_index=gpu_index,
            pid=pid,
            process_name=proc_name,
            memory_mib=mem,
        ))

    _cached_snapshot = (now, out)
    return out


def cuda_device_label(device: str | None, device_index: int | None) -> str:
    """Human label for a torch device, including physical GPU when remapped."""
    if not device:
        return "not loaded"
    if device == "mps":
        return "Apple MPS"
    if device != "cuda":
        return device
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if device_index is None:
        return "cuda"
    if visible:
        parts = [p.strip() for p in visible.split(",") if p.strip()]
        if device_index < len(parts):
            return f"GPU {parts[device_index]} (cuda:{device_index})"
        return f"cuda:{device_index} (CUDA_VISIBLE_DEVICES={visible})"
    return f"GPU {device_index}"


def pids_on_gpus(snapshot: GpuSnapshot | None, *, pid: int) -> list[int]:
    """Sorted GPU indices where this PID has compute contexts."""
    if snapshot is None:
        return []
    return sorted({
        p.gpu_index for p in snapshot.processes if p.pid == pid
    })


def gpus_for_process_substring(
    snapshot: GpuSnapshot | None, needle: str,
) -> dict[int, list[GpuProcess]]:
    """Group matching processes by GPU index (case-insensitive substring)."""
    out: dict[int, list[GpuProcess]] = {}
    if snapshot is None:
        return out
    needle_l = needle.lower()
    for proc in snapshot.processes:
        if needle_l not in proc.process_name.lower():
            continue
        out.setdefault(proc.gpu_index, []).append(proc)
    return out


def _process_is_llm_heavy(proc: GpuProcess) -> bool:
    """True when this compute row looks like a loaded LLM, not desktop fluff."""
    name = proc.process_name.lower()
    mem = proc.memory_mib or 0
    if any(n in name for n in ("ollama", "llama", "runner")):
        return mem >= _LLM_OLLAMA_SLICE_MIB
    if "python" in name:
        return mem >= _LLM_PROC_VRAM_MIB
    return False


def gpu_indices_with_llm_vram(snapshot: GpuSnapshot | None) -> list[int]:
    """GPU indices that already host a large LLM compute context."""
    if snapshot is None:
        return []
    return sorted({
        p.gpu_index for p in snapshot.processes if _process_is_llm_heavy(p)
    })


def format_gpu_indices_label(
    indices: list[int],
    snapshot: GpuSnapshot | None,
    *,
    pending: bool = False,
    vram_gib: float | None = None,
) -> str:
    """Human placement label — never bare ``GPU ?``."""
    if indices:
        parts: list[str] = []
        for gi in indices:
            mem = format_gpu_memory(gi, snapshot)
            parts.append(f"GPU {gi}" + (f" [{mem}]" if mem else ""))
        return ", ".join(parts)
    if pending:
        return "pending (loads on first request)"
    if vram_gib is not None and vram_gib > 0:
        return f"VRAM ~{vram_gib:.1f} GB (GPU from nvidia-smi pending)"
    if snapshot is not None:
        return "VRAM allocated (GPU index pending)"
    return "pending (nvidia-smi unavailable)"


def format_model_gpu_placement(
    indices: list[int],
    snapshot: GpuSnapshot | None,
    *,
    vram_gib: float | None = None,
    not_loaded: bool = False,
) -> str:
    """Compact GPU line for a Model slot — never bare ``GPU ?``."""
    if not_loaded:
        return "not loaded"
    if not indices:
        if vram_gib is not None and vram_gib > 0:
            return f"VRAM ~{vram_gib:.1f} GB (GPU index pending)"
        return "not loaded"
    if len(indices) == 1:
        gi = indices[0]
        mem = format_gpu_memory(gi, snapshot)
        line = f"GPU {gi}"
        if mem:
            line += f" [{mem}]"
        elif vram_gib is not None and vram_gib > 0:
            line += f" (~{vram_gib:.1f} GB)"
        return line
    line = "GPU " + " + ".join(str(i) for i in indices)
    if vram_gib is not None and vram_gib > 0:
        line += f" (~{vram_gib:.1f} GB)"
    return line


def format_vram_footer(snapshot: GpuSnapshot | None) -> str:
    """Footer VRAM bar: ``0:20/48 · 1:28/48 · … GB``."""
    if snapshot is None or not snapshot.gpus:
        return ""
    parts: list[str] = []
    for g in sorted(snapshot.gpus, key=lambda x: x.index):
        if g.memory_used_mib is None or g.memory_total_mib is None:
            parts.append(str(g.index))
            continue
        parts.append(
            f"{g.index}:{g.memory_used_mib / 1024:.0f}/{g.memory_total_mib / 1024:.0f}"
        )
    if not parts:
        return ""
    return " · ".join(parts) + " GB"


def ollama_tensor_split_gpu_indices(snapshot: GpuSnapshot | None) -> list[int] | None:
    """GPU indices when one Ollama PID spans 2+ cards; else None."""
    if snapshot is None:
        return None
    by_pid: dict[int, list[int]] = {}
    for p in snapshot.processes:
        if "ollama" not in p.process_name.lower():
            continue
        if (p.memory_mib or 0) < _LLM_OLLAMA_SLICE_MIB:
            continue
        by_pid.setdefault(p.pid, []).append(p.gpu_index)
    for indices in by_pid.values():
        uniq = sorted(set(indices))
        if len(uniq) >= 2:
            return uniq
    return None


def ollama_is_tensor_split(snapshot: GpuSnapshot | None) -> bool:
    return ollama_tensor_split_gpu_indices(snapshot) is not None


def prefer_single_gpu_workstation(snapshot: GpuSnapshot | None) -> bool:
    """True when at least one GPU has ~48 GB VRAM (workstation class).

    False on small-GPU boxes (e.g. two 8–16 GB cards) where tensor split
    may be required — we do not auto-unload there.
    """
    if snapshot is None or not snapshot.gpus:
        return False
    totals = [g.memory_total_mib or 0 for g in snapshot.gpus]
    return max(totals) >= 40000


def ollama_chat_load_options(snapshot: GpuSnapshot | None = None) -> dict[str, Any]:
    """Extra Ollama ``options`` to bias toward full GPU offload on big boxes."""
    snap = snapshot if snapshot is not None else snapshot_gpus()
    if not prefer_single_gpu_workstation(snap):
        return {}
    # num_gpu = layer count on GPU (Ollama API), not physical GPU index.
    # High value asks Ollama to keep weights on GPU when VRAM allows.
    return {"num_gpu": 999}


def ollama_split_tip_short(snapshot: GpuSnapshot | None) -> str | None:
    """One-line split hint for the panel (no pid jargon)."""
    if ollama_multi_daemon_setup():
        return None
    indices = ollama_tensor_split_gpu_indices(snapshot)
    if not indices:
        return None
    gpus = "+".join(str(i) for i in indices)
    if prefer_single_gpu_workstation(snapshot):
        return (
            f"tip · still split on GPU {gpus} — "
            "run /unload all if auto-reload did not clear it"
        )
    return f"tip · split across GPU {gpus} (normal on small GPUs)"


def chat_process_gpu_vram(
    snapshot: GpuSnapshot | None,
    pid: int,
    *,
    min_mib: int = 2000,
) -> list[tuple[int, int]]:
    """(gpu_index, memory_mib) for this PID from nvidia-smi compute-apps."""
    if snapshot is None:
        return []
    out: list[tuple[int, int]] = []
    for p in snapshot.processes:
        if p.pid != pid:
            continue
        if "python" not in p.process_name.lower():
            continue
        mem = p.memory_mib or 0
        if mem >= min_mib:
            out.append((p.gpu_index, mem))
    return sorted(out, key=lambda x: x[0])


def format_diffuser_line(kind: str, generator: Any | None) -> str:
    """One diffuser row: kind · GPU · loaded, or kind · not loaded."""
    if generator is None:
        return f"{kind} · not loaded"
    return f"{kind} · {diffuser_placement(generator)} · loaded"


def format_four_gpu_summary(snapshot: GpuSnapshot | None) -> str:
    """One-line VRAM summary for all cards, e.g. '0 [20/49] · 1 [28/49]'."""
    if snapshot is None or not snapshot.gpus:
        return ""
    parts: list[str] = []
    for g in sorted(snapshot.gpus, key=lambda x: x.index):
        mem = format_gpu_memory(g.index, snapshot)
        parts.append(f"{g.index} [{mem}]" if mem else str(g.index))
    return " · ".join(parts)


def _normalize_ollama_base(raw: str) -> str:
    s = raw.strip().rstrip("/")
    if not s.startswith("http"):
        s = "http://" + s
    return s


def _normalize_hostport(raw: str) -> str:
    s = (raw or "").strip().rstrip("/")
    if s.startswith("http://"):
        s = s[len("http://"):]
    elif s.startswith("https://"):
        s = s[len("https://"):]
    if s.startswith("localhost:"):
        s = "127.0.0.1:" + s.split(":", 1)[1]
    return s


def _proc_environ(pid: int) -> dict[str, str]:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read()
    except OSError:
        return ""
    return data.replace(b"\0", b" ").decode(errors="replace").strip()


def ollama_endpoint_gpu_index(endpoint: str) -> int | None:
    """Physical GPU pinned to an `ollama serve` endpoint via CUDA_VISIBLE_DEVICES."""
    want = _normalize_hostport(endpoint or "")
    if not want:
        return None
    try:
        r = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    for raw_pid in (r.stdout or "").split():
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        if "ollama" not in _proc_cmdline(pid).lower():
            continue
        env = _proc_environ(pid)
        host = _normalize_hostport(env.get("OLLAMA_HOST") or "")
        if host != want:
            continue
        visible = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
        # Auto-pinned daemons use one visible physical GPU. If someone
        # configured a list, this endpoint is not a single-GPU pin.
        if "," in visible or not visible:
            return None
        try:
            return int(visible)
        except ValueError:
            return None
    return None


def ollama_all_api_bases() -> list[str]:
    """All Ollama HTTP bases: OLLAMA_HOST (+ HOST2/HOST3) and loopback fallbacks."""
    out: list[str] = []
    seen: set[str] = set()
    for key in ("OLLAMA_HOST", "OLLAMA_HOST2", "OLLAMA_HOST3"):
        raw = (os.environ.get(key) or "").strip()
        if not raw and key == "OLLAMA_HOST":
            raw = "127.0.0.1:11434"
        if not raw:
            continue
        s = _normalize_ollama_base(raw)
        if s not in seen:
            seen.add(s)
            out.append(s)
    for fallback in ("127.0.0.1:11434", "localhost:11434"):
        s = _normalize_ollama_base(fallback)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["http://127.0.0.1:11434"]


def _ollama_api_bases() -> list[str]:
    """Primary + loopback probes (backward-compatible alias)."""
    return ollama_all_api_bases()


def ollama_multi_daemon_setup() -> bool:
    """True when model2/model3 use separate Ollama ports (intentional multi-GPU)."""
    return bool(
        (os.environ.get("OLLAMA_HOST2") or "").strip()
        or (os.environ.get("OLLAMA_HOST3") or "").strip()
    )


def ollama_ps_at_endpoint(base: str) -> list[dict[str, Any]]:
    """Loaded models on one Ollama daemon (/api/ps)."""
    endpoint = _normalize_ollama_base(base)
    try:
        req = urllib.request.Request(endpoint.rstrip("/") + "/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ):
        return []
    if not isinstance(data, dict):
        return []
    out: list[dict[str, Any]] = []
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        size = int(m.get("size") or 0)
        vram = int(m.get("size_vram") or 0)
        out.append({
            "name": name,
            "endpoint": endpoint,
            "size_bytes": size,
            "size_vram_bytes": vram,
            "vram_gib": round(vram / (1024 ** 3), 1) if vram else None,
        })
    return out


def gpu_indices_for_ollama_loaded_model(
    snapshot: GpuSnapshot | None,
    *,
    vram_bytes: int | None = None,
    vram_gib: float | None = None,
) -> list[int]:
    """Best GPU index for one loaded Ollama model (single card, not global split)."""
    if snapshot is None:
        return []
    target_mib = 0
    if vram_bytes and vram_bytes > 0:
        target_mib = int(vram_bytes / (1024 * 1024))
    elif vram_gib and vram_gib > 0:
        target_mib = int(vram_gib * 1024)
    by_pid: dict[int, dict[int, int]] = {}
    for p in snapshot.processes:
        if "ollama" not in p.process_name.lower():
            continue
        mem = p.memory_mib or 0
        if mem < _LLM_OLLAMA_SLICE_MIB:
            continue
        by_pid.setdefault(p.pid, {})
        by_pid[p.pid][p.gpu_index] = by_pid[p.pid].get(p.gpu_index, 0) + mem
    if not by_pid:
        return []
    best_gpu: int | None = None
    best_score = -1.0
    for per_gpu in by_pid.values():
        total = sum(per_gpu.values())
        primary = max(per_gpu, key=lambda g: per_gpu[g])
        if target_mib > 0:
            # Match this model's /api/ps VRAM to the runner PID footprint.
            score = 1.0 / (1.0 + abs(total - target_mib))
        else:
            score = float(total)
        if score > best_score:
            best_score = score
            best_gpu = primary
    return [best_gpu] if best_gpu is not None else []


def ollama_loaded_models() -> list[dict[str, Any]]:
    """Models currently resident in Ollama (/api/ps), merged across all daemons."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for base in ollama_all_api_bases():
        for row in ollama_ps_at_endpoint(base):
            key = (row["endpoint"], row["name"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def infer_ollama_gpu_indices(
    snapshot: GpuSnapshot | None,
    *,
    min_vram_mib: int = _LLM_OLLAMA_SLICE_MIB,
) -> list[int]:
    """GPUs where Ollama (or similar) has substantial compute allocated."""
    if snapshot is None:
        return []
    by_gpu = gpus_for_process_substring(snapshot, "ollama")
    out: list[int] = []
    for idx, procs in sorted(by_gpu.items()):
        if any((p.memory_mib or 0) >= min_vram_mib for p in procs):
            out.append(idx)
    if out:
        return out
    # Fallback: any GPU with large non-desktop VRAM use (heuristic).
    for g in snapshot.gpus:
        if g.memory_used_mib is None or g.memory_used_mib < 8000:
            continue
        out.append(g.index)
    return sorted(set(out))


def ollama_tensor_split_warning(snapshot: GpuSnapshot | None) -> str | None:
    """Warn when one Ollama PID spans multiple GPUs (tensor split)."""
    if snapshot is None:
        return None
    by_pid: dict[int, list[int]] = {}
    for p in snapshot.processes:
        if "ollama" not in p.process_name.lower():
            continue
        if (p.memory_mib or 0) < _LLM_OLLAMA_SLICE_MIB:
            continue
        by_pid.setdefault(p.pid, []).append(p.gpu_index)
    for pid, indices in by_pid.items():
        uniq = sorted(set(indices))
        if len(uniq) >= 2:
            return (
                f"Ollama pid {pid} split across GPU "
                + ", ".join(str(i) for i in uniq)
                + " — single GPU is faster for 27B Q8"
            )
    return None


def large_python_gpu_indices(
    snapshot: GpuSnapshot | None,
    *,
    exclude_pid: int | None = None,
) -> list[int]:
    """GPUs with a large in-process python compute context (e.g. MLX / chat)."""
    if snapshot is None:
        return []
    return sorted({
        p.gpu_index for p in snapshot.processes
        if "python" in p.process_name.lower()
        and (p.memory_mib or 0) >= _LLM_PROC_VRAM_MIB
        and (exclude_pid is None or p.pid != exclude_pid)
    })


def _gpu_free_mib(snap: GpuSnapshot, index: int) -> int | None:
    for g in snap.gpus:
        if g.index != index:
            continue
        if g.memory_total_mib is None or g.memory_used_mib is None:
            return None
        return g.memory_total_mib - g.memory_used_mib
    return None


def _is_four_gpu_linux_nvidia_workstation(snap: GpuSnapshot | None) -> bool:
    """True on the auto-pinned 4×48 GB Linux box (diffusers → GPU 0)."""
    if not sys.platform.startswith("linux"):
        return False
    if snap is None or not snap.gpus:
        return False
    gpus = sorted(snap.gpus, key=lambda g: g.index)
    if len(gpus) != 4:
        return False
    if not all("nvidia" in (g.name or "").lower() for g in gpus):
        return False
    if not all((g.memory_total_mib or 0) >= 40000 for g in gpus):
        return False
    return [g.index for g in gpus] == [0, 1, 2, 3]


# Z-Image-Turbo needs ~14 GB VRAM; leave headroom for Stable-Audio on GPU 0.
_MIN_DIFFUSER_FREE_MIB = 12000
# Ollama slot daemons pin LLMs to these physical GPUs (see backend autopin).
_OLLAMA_SLOT_GPUS = (1, 2, 3)


def pick_diffuser_cuda_index(
    snapshot: GpuSnapshot | None = None,
    *,
    reuse_cuda_index: int | None = None,
) -> int | None:
    """Pick a CUDA device for Z-Image / Stable-Audio.

    On the 4×48 GB workstation, **prefer GPU 0** so GPUs 1–3 stay for the
    three Ollama slots. ``DIFFUSER_CUDA_DEVICE=N`` overrides. When
    ``reuse_cuda_index`` is set (e.g. image pipeline already on GPU 0),
    reuse that card if it still has room.
    """
    raw = (os.environ.get("DIFFUSER_CUDA_DEVICE") or "").strip()
    if raw.isdigit():
        return int(raw)

    snap = snapshot if snapshot is not None else snapshot_gpus()
    if snap is None or not snap.gpus:
        return None
    llm_gpus = set(gpu_indices_with_llm_vram(snap))
    on_workstation = _is_four_gpu_linux_nvidia_workstation(snap)

    def _ok(index: int) -> bool:
        # On the 4×48 GB workstation GPU 0 is the diffuser slot by
        # construction — once Z-Image-Turbo loads, our own Python process
        # makes the card look "LLM-heavy" to the generic detector. Skip
        # the veto for index 0 there so Stable-Audio colocates with
        # Z-Image instead of stealing GPU 1 from the coder slot.
        if index in llm_gpus and not (on_workstation and index == 0):
            return False
        free = _gpu_free_mib(snap, index)
        return free is not None and free >= _MIN_DIFFUSER_FREE_MIB

    if reuse_cuda_index is not None and _ok(reuse_cuda_index):
        return reuse_cuda_index

    if on_workstation and _ok(0):
        return 0

    reserved = set(_OLLAMA_SLOT_GPUS) if on_workstation else set()

    def _best(candidates: list[GpuInfo]) -> int | None:
        best_idx: int | None = None
        best_free = -1
        for g in candidates:
            if g.index in llm_gpus or g.index in reserved:
                continue
            free = _gpu_free_mib(snap, g.index)
            if free is None or free < _MIN_DIFFUSER_FREE_MIB:
                continue
            if free > best_free:
                best_free = free
                best_idx = g.index
        return best_idx

    non_llm = [g for g in snap.gpus if g.index not in llm_gpus]
    pick = _best(non_llm)
    if pick is not None:
        return pick
    return pick_least_loaded_cuda_index(snap)


def pick_least_loaded_cuda_index(snapshot: GpuSnapshot | None = None) -> int | None:
    """Pick the quietest GPU for diffusers, avoiding LLM-heavy cards first.

    Prefer :func:`pick_diffuser_cuda_index` for pipeline loads; this remains
    the global fallback when no card meets the diffuser free-VRAM floor.
    """
    snap = snapshot if snapshot is not None else snapshot_gpus()
    if snap is None or not snap.gpus:
        return None
    llm_gpus = set(gpu_indices_with_llm_vram(snap))

    def _best(candidates: list[GpuInfo]) -> int | None:
        best_idx: int | None = None
        best_free = -1
        for g in candidates:
            if g.memory_total_mib is None or g.memory_used_mib is None:
                continue
            free = g.memory_total_mib - g.memory_used_mib
            if free > best_free:
                best_free = free
                best_idx = g.index
        return best_idx

    non_llm = [g for g in snap.gpus if g.index not in llm_gpus]
    pick = _best(non_llm)
    if pick is not None:
        return pick
    return _best(list(snap.gpus))


def activate_cuda_device(index: int) -> None:
    """Set the current torch CUDA device before loading a pipeline."""
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    n = torch.cuda.device_count()
    if 0 <= index < n:
        torch.cuda.set_device(index)


def format_gpu_memory(gpu_index: int, snapshot: GpuSnapshot | None) -> str:
    """e.g. '28.1/49.1 GB' for a GPU row."""
    if snapshot is None:
        return ""
    for g in snapshot.gpus:
        if g.index != gpu_index:
            continue
        if g.memory_used_mib is None or g.memory_total_mib is None:
            return ""
        return (
            f"{g.memory_used_mib / 1024:.1f}/"
            f"{g.memory_total_mib / 1024:.1f} GB"
        )
    return ""


def diffuser_kind(generator: Any) -> str:
    """Short label for which pipeline class is generating."""
    cls = type(generator).__name__
    if cls == "ZImageTurboGenerator":
        return "Z-Image-Turbo"
    if cls == "Img2ImgGenerator":
        return "SD-Turbo img2img"
    if cls == "StableAudioGenerator":
        return "Stable-Audio"
    return cls


def diffuser_placement(generator: Any) -> str:
    """Device label for a loaded diffuser instance."""
    device = getattr(generator, "_device", None)
    idx = getattr(generator, "_cuda_device_index", None)
    if device == "cuda" and idx is None:
        try:
            import torch
            if torch.cuda.is_available():
                idx = int(torch.cuda.current_device())
        except Exception:
            pass
    return cuda_device_label(device, idx)
