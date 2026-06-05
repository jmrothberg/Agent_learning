"""system_tests.py — visible-browser system tests for the multi-GPU agent.

Validates plumbing (3 Ollama slots, traces, opening-book checks, GPU hygiene)
and runs a slower Pac-Man benchmark for quality. Artifacts land under
``games/system-tests/run_<timestamp>/`` with a compact ``SYSTEM_SUMMARY.md``.

Usage:
    python system_tests.py run --suite smoke
    python system_tests.py run --suite pacman --max-iters 3   # prompts before slow run
    python system_tests.py run --suite full --three-model     # smoke first, then prompts
    python system_tests.py run --suite pacman --yes           # no prompt (CI)
    python system_tests.py show run_20260521_120000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import GameAgent
from tools import LiveBrowser

import backend as backend_mod


def _timeouts_for_model(model: str) -> tuple[float, float, int]:
    from chat import resolve_session_timeouts
    stall_s, overall_s = resolve_session_timeouts(model)
    return stall_s, overall_s, 8192


REPO_ROOT = Path(__file__).resolve().parent
# Run OUTPUT (per-run artifacts) lives under games/ — that dir is for
# generated local games and is gitignored. Correct home for outputs.
SYSTEM_TESTS_ROOT = REPO_ROOT / "games" / "system-tests"
# The default battery is curated project INPUT useful on every machine, so it
# is committed under memory/ alongside the other opening-library data — NOT
# under games/ (which is gitignored user output). Moved 2026-06-02 after a
# fresh checkout had no battery and `system_tests.py run` errored.
# A user-local override at games/system-tests/battery.jsonl still wins if present.
DEFAULT_BATTERY = REPO_ROOT / "memory" / "system_battery.jsonl"
_LEGACY_BATTERY = SYSTEM_TESTS_ROOT / "battery.jsonl"
if not DEFAULT_BATTERY.exists() and _LEGACY_BATTERY.exists():
    # Back-compat: honor an existing local battery in the old location.
    DEFAULT_BATTERY = _LEGACY_BATTERY
DEFAULT_MODEL = os.environ.get("SYSTEM_TEST_MODEL", "qwen3.6:27b")

SUITE_TESTS = {
    "smoke": {"smoke-plumbing", "smoke-move", "smoke-board-select", "smoke-asset-animation"},
    "pacman": {"pacman-hard"},
    "full": {"smoke-plumbing", "smoke-move", "smoke-board-select", "smoke-asset-animation", "pacman-hard"},
}

# Slow acceptance benchmarks — prompt before run (smoke suite is fast).
SLOW_TEST_SLUGS = frozenset({"pacman-hard"})
SLOW_TEST_ESTIMATE = "~15-30 minutes"

THREE_MODEL_PORTS = (11434, 11435, 11436)


def _env_assume_yes() -> bool:
    return (os.environ.get("SYSTEM_TEST_ASSUME_YES") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def prompt_yes_no(prompt: str, *, default_no: bool = True) -> bool:
    """Ask on stdin; default No on empty/EOF so fast runs are not blocked."""
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return not default_no
    return ans in ("y", "yes")


def partition_slow_tests(
    tests: list["SystemBatteryTest"],
) -> tuple[list["SystemBatteryTest"], list["SystemBatteryTest"]]:
    fast = [t for t in tests if t.slug not in SLOW_TEST_SLUGS]
    slow = [t for t in tests if t.slug in SLOW_TEST_SLUGS]
    return fast, slow


def confirm_slow_tests_before_run(
    tests: list["SystemBatteryTest"],
    *,
    suite: str,
    assume_yes: bool,
) -> tuple[list["SystemBatteryTest"], list[str]]:
    """Gate slow tests. Pacman-only suite prompts once up front; others per slug."""
    fast, slow = partition_slow_tests(tests)
    if not slow or assume_yes:
        return tests, []

    notes: list[str] = []
    # Entire suite is the slow benchmark — ask before GPU hygiene / browser.
    if suite == "pacman" and not fast:
        print()
        print(
            f"The pacman suite runs `{slow[0].slug}` — a full Pac-Man build "
            f"({SLOW_TEST_ESTIMATE}). Smoke tests are much faster."
        )
        if not prompt_yes_no("Run the Pac-Man benchmark now? [y/N] "):
            notes.append(f"{slow[0].slug} skipped (user declined)")
            print("Skipped pacman-hard. Use --yes to run without prompting.")
            return [], notes
        return tests, notes

    # Mixed/full suite: run smoke first; pacman is confirmed in the run loop.
    return tests, notes


def confirm_slow_test_run(slug: str, *, assume_yes: bool) -> bool:
    """Per-test prompt when a slow slug appears mid-battery (e.g. --suite full)."""
    if slug not in SLOW_TEST_SLUGS or assume_yes:
        return True
    print()
    print(
        f"Next: `{slug}` — Pac-Man quality benchmark ({SLOW_TEST_ESTIMATE}). "
        "Earlier smoke tests in this run are usually done in a few minutes."
    )
    return prompt_yes_no(f"Run `{slug}` now? [y/N] ")


# ---------------------------------------------------------------------------
# Battery (extends tune JSONL with suite + pass_criteria)
# ---------------------------------------------------------------------------


@dataclass
class SystemBatteryTest:
    slug: str
    goal: str
    notes: str = ""
    suite: str = "smoke"
    max_iters: int | None = None
    pass_criteria: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "SystemBatteryTest":
        criteria = obj.get("pass_criteria") or {}
        if not isinstance(criteria, dict):
            criteria = {}
        return cls(
            slug=str(obj["slug"]).strip(),
            goal=str(obj["goal"]).strip(),
            notes=str(obj.get("notes", "")).strip(),
            suite=str(obj.get("suite", "smoke")).strip() or "smoke",
            max_iters=obj.get("max_iters"),
            pass_criteria={k: bool(v) for k, v in criteria.items()},
        )


def load_system_battery(path: Path) -> list[SystemBatteryTest]:
    out: list[SystemBatteryTest] = []
    if not path.exists():
        raise SystemExit(f"battery file not found: {path}")
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
            out.append(SystemBatteryTest.from_obj(obj))
        except Exception as e:
            raise SystemExit(f"battery {path}:{ln} parse error: {e}")
    if not out:
        raise SystemExit(f"battery {path} is empty")
    return out


def filter_tests(
    tests: list[SystemBatteryTest],
    *,
    suite: str,
    slugs: str | None,
) -> list[SystemBatteryTest]:
    wanted_suite = SUITE_TESTS.get(suite)
    if wanted_suite is None:
        raise SystemExit(f"unknown suite {suite!r}; choose smoke|pacman|full")
    out = [t for t in tests if t.slug in wanted_suite]
    if slugs:
        pick = {s.strip() for s in slugs.split(",") if s.strip()}
        out = [t for t in out if t.slug in pick]
    if not out:
        raise SystemExit(f"no tests matched suite={suite!r} slugs={slugs!r}")
    return out


# ---------------------------------------------------------------------------
# GPU hygiene + 3-model infrastructure
# ---------------------------------------------------------------------------


@dataclass
class GpuHygieneReport:
    nvidia_smi_before: str = ""
    nvidia_smi_after: str = ""
    ollama_ps_before: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    ollama_ps_after: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unload_results: list[str] = field(default_factory=list)
    placement_mode: str = ""
    placement_message: str = ""
    tensor_split_before: bool = False
    tensor_split_after: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _capture_nvidia_smi() -> str:
    try:
        import gpu_status as gs
        snap = gs.snapshot_gpus(force=True)
        if snap is None:
            return "(nvidia-smi unavailable)"
        lines = ["# nvidia-smi snapshot"]
        for g in sorted(snap.gpus, key=lambda x: x.index):
            used = g.memory_used_mib or 0
            total = g.memory_total_mib or 0
            lines.append(f"GPU{g.index}: {g.name}  {used}/{total} MiB")
        for p in snap.processes[:40]:
            if (p.memory_mib or 0) >= 500:
                lines.append(
                    f"  pid={p.pid} gpu={p.gpu_index} "
                    f"{p.process_name} {p.memory_mib}MiB"
                )
        return "\n".join(lines)
    except Exception as e:
        return f"(nvidia-smi error: {e})"


def _ollama_bases_for_hygiene(three_model: bool) -> list[str]:
    if three_model:
        return [f"http://127.0.0.1:{p}" for p in THREE_MODEL_PORTS]
    return [backend_mod.ollama_endpoint_url(1)]


def _snapshot_ollama_ps(bases: list[str]) -> dict[str, list[dict[str, Any]]]:
    import gpu_status as gs
    out: dict[str, list[dict[str, Any]]] = {}
    for base in bases:
        out[base] = gs.ollama_ps_at_endpoint(base)
    return out


def run_gpu_hygiene(
    run_dir: Path,
    *,
    three_model: bool,
    do_unload: bool,
) -> GpuHygieneReport:
    """Pre-run GPU hygiene: capture state, optionally unload stale models."""
    report = GpuHygieneReport()
    bases = _ollama_bases_for_hygiene(three_model)

    try:
        import gpu_status as gs
        snap = gs.snapshot_gpus(force=True)
        report.tensor_split_before = gs.ollama_is_tensor_split(snap)
    except Exception:
        pass

    report.nvidia_smi_before = _capture_nvidia_smi()
    report.ollama_ps_before = _snapshot_ollama_ps(bases)
    (run_dir / "gpu_nvidia_smi_before.txt").write_text(
        report.nvidia_smi_before, encoding="utf-8"
    )
    (run_dir / "ollama_ps_before.json").write_text(
        json.dumps(report.ollama_ps_before, indent=2), encoding="utf-8"
    )

    if do_unload:
        for base in bases:
            loaded = backend_mod._ollama_running_models(base)
            if not loaded:
                continue
            results = backend_mod.unload_all_ollama_models(base)
            for name, ok, msg in results:
                report.unload_results.append(f"{base}: {name} -> {ok} ({msg})")

    if three_model:
        placement = backend_mod.ensure_ollama_slot_daemons_for_chat(enabled=True)
        report.placement_mode = placement.mode
        report.placement_message = placement.message
        if placement.mode == "fallback":
            report.errors.append(
                f"3-model autopin fell back to single daemon: {placement.message}"
            )
        ok, errs, _ = validate_three_model_endpoints()
        report.errors.extend(errs)

    report.ollama_ps_after = _snapshot_ollama_ps(bases)
    report.nvidia_smi_after = _capture_nvidia_smi()
    (run_dir / "gpu_nvidia_smi_after.txt").write_text(
        report.nvidia_smi_after, encoding="utf-8"
    )
    (run_dir / "ollama_ps_after.json").write_text(
        json.dumps(report.ollama_ps_after, indent=2), encoding="utf-8"
    )

    try:
        import gpu_status as gs
        snap = gs.snapshot_gpus(force=True)
        report.tensor_split_after = gs.ollama_is_tensor_split(snap)
        if report.tensor_split_after and three_model:
            report.errors.append(
                "Ollama still tensor-split across GPUs after hygiene "
                "(stale placement — unload manually or stop extra ollama serve)"
            )
    except Exception:
        pass

    if report.tensor_split_before and three_model:
        report.errors.append(
            "Pre-run tensor-split detected on Ollama before unload"
        )

    return report


def validate_three_model_endpoints() -> tuple[bool, list[str], list[str]]:
    """Require three distinct loopback Ollama endpoints when 3-model mode is on."""
    eps = [
        backend_mod.ollama_endpoint_url(1).rstrip("/"),
        backend_mod.ollama_endpoint_url(2).rstrip("/"),
        backend_mod.ollama_endpoint_url(3).rstrip("/"),
    ]
    errors: list[str] = []
    if len(set(eps)) < 3:
        errors.append(
            f"3-model mode collapsed to {len(set(eps))} endpoint(s): {eps}"
        )
    for ep in eps:
        if not backend_mod._endpoint_ready(ep):
            errors.append(f"Ollama endpoint not reachable: {ep}")
    ports = []
    for ep in eps:
        m = re.search(r":(\d+)$", ep)
        if m:
            ports.append(int(m.group(1)))
    if ports and len(set(ports)) < 3:
        errors.append(f"expected ports 11434/11435/11436, got {ports}")
    return (not errors, errors, eps)


def build_three_model_backends(model: str) -> tuple[Any, Any | None, str | None, Any | None, str | None]:
    """Return (backend1, backend2, role2, backend3, role3) for GameAgent."""
    ep1 = backend_mod.ollama_endpoint_url(1)
    ep2 = backend_mod.ollama_endpoint_url(2)
    ep3 = backend_mod.ollama_endpoint_url(3)
    info1 = backend_mod.BackendInfo(
        name="ollama", model=model, source="system_tests slot1",
        endpoint=ep1,
    )
    info2 = backend_mod.BackendInfo(
        name="ollama", model=model, source="system_tests slot2",
        endpoint=ep2,
    )
    info3 = backend_mod.BackendInfo(
        name="ollama", model=model, source="system_tests slot3",
        endpoint=ep3,
    )
    b1 = backend_mod.make_backend(info1)
    b2 = backend_mod.make_backend(info2)
    b3 = backend_mod.make_backend(info3)
    # model2=architect (playtests), model3=critic (asset/animation audits)
    return b1, b2, "architect", b3, "critic"


# ---------------------------------------------------------------------------
# Trace + report analysis
# ---------------------------------------------------------------------------


@dataclass
class TraceAnalysis:
    kinds: set[str] = field(default_factory=set)
    stream_done_count: int = 0
    stream_durations: list[float] = field(default_factory=list)
    opening_book_retrieved: int = 0
    opening_book_sidecar: int = 0
    architect_notes: int = 0
    test_events: int = 0
    session_start: bool = False
    line_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinds": sorted(self.kinds),
            "stream_done_count": self.stream_done_count,
            "stream_durations": self.stream_durations,
            "opening_book_retrieved": self.opening_book_retrieved,
            "opening_book_sidecar": self.opening_book_sidecar,
            "architect_notes": self.architect_notes,
            "test_events": self.test_events,
            "session_start": self.session_start,
            "line_count": self.line_count,
        }


def parse_trace(trace_path: Path | None) -> TraceAnalysis:
    out = TraceAnalysis()
    if trace_path is None or not trace_path.exists():
        out.errors.append("trace missing")
        return out
    try:
        for raw in trace_path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s:
                continue
            out.line_count += 1
            try:
                ev = json.loads(s)
            except json.JSONDecodeError:
                continue
            kind = ev.get("kind") or ev.get("event")
            if not kind:
                continue
            out.kinds.add(str(kind))
            if kind == "session_start":
                out.session_start = True
            elif kind == "stream_done":
                out.stream_done_count += 1
                if ev.get("duration_s") is not None:
                    out.stream_durations.append(float(ev["duration_s"]))
            elif kind == "opening_book_retrieved":
                out.opening_book_retrieved += 1
            elif kind in ("opening_book_sidecar_proposal", "opening_book_sidecars_failed"):
                out.opening_book_sidecar += 1
            elif kind == "architect_note":
                out.architect_notes += 1
            elif kind in ("test", "event") and (
                ev.get("event") == "test" or kind == "test"
            ):
                out.test_events += 1
            # agent wraps some events as {"event": "test", ...}
            if ev.get("event") == "test":
                out.test_events += 1
    except Exception as e:
        out.errors.append(f"trace read error: {e}")
    return out


def score_observability(
    trace: TraceAnalysis,
    test_dir: Path,
    gpu_report: GpuHygieneReport | None,
) -> tuple[int, list[str]]:
    """0–100 observability score + human notes."""
    score = 0
    notes: list[str] = []
    if trace.session_start:
        score += 15
    else:
        notes.append("trace missing session_start")
    if "stream_start" in trace.kinds:
        score += 15
    else:
        notes.append("trace missing stream_start")
    if trace.test_events > 0:
        score += 20
    else:
        notes.append("trace missing browser test events")
    if trace.line_count >= 20:
        score += 15
    else:
        notes.append(f"trace thin ({trace.line_count} lines)")
    pngs = list(test_dir.rglob("*.png"))
    if pngs:
        score += 20
    else:
        notes.append("no PNG screenshots in test dir")
    if gpu_report and gpu_report.nvidia_smi_before:
        score += 15
    else:
        notes.append("no GPU snapshot in run dir")
    return min(100, score), notes


def evaluate_pass_criteria(
    criteria: dict[str, bool],
    *,
    last_report: dict[str, Any] | None,
    trace: TraceAnalysis,
    test_dir: Path,
    three_model: bool,
) -> tuple[bool, dict[str, bool]]:
    """Evaluate per-test pass_criteria; return (all_ok, per_key results)."""
    results: dict[str, bool] = {}
    report = last_report or {}
    input_test = report.get("input_test") or {}
    probes = report.get("probes") or []
    ob_checks = report.get("opening_book_checks") or []

    def _set(key: str, val: bool) -> None:
        results[key] = val

    for key, required in criteria.items():
        if not required:
            continue
        ok = True
        if key == "input_responsive":
            ok = bool(input_test.get("ran") and input_test.get("any_change") is True)
        elif key == "opening_book_trace":
            ok = trace.opening_book_retrieved > 0
        elif key == "opening_book_checks":
            ok = len(ob_checks) > 0
        elif key == "sidecar_trace_when_three_model":
            if three_model:
                ok = (
                    trace.opening_book_sidecar > 0
                    or trace.architect_notes > 0
                    or trace.stream_done_count >= 2
                )
            else:
                ok = True
        elif key == "no_page_errors":
            ok = not (report.get("page_errors") or report.get("errors"))
        elif key == "canvas_not_frozen":
            frozen = report.get("frozen")
            responsive = bool(
                input_test.get("ran") and input_test.get("any_change") is True
            )
            ok = responsive or frozen is not True
        elif key == "screenshots_present":
            ok = bool(report.get("screenshot") or report.get("screenshot_before"))
            if not ok:
                ok = bool(list(test_dir.rglob("*.png")))
        elif key == "restart_probe_or_reset":
            names = {str(p.get("name", "")).lower() for p in probes}
            ok = any("restart" in n or "reset" in n for n in names) or bool(
                report.get("ok")
            )
        else:
            ok = True
        _set(key, ok)

    all_ok = all(results.values()) if results else True
    return all_ok, results


# ---------------------------------------------------------------------------
# Per-test result + run
# ---------------------------------------------------------------------------


@dataclass
class SystemTestResult:
    slug: str
    goal: str
    model: str
    suite: str
    max_iters: int
    passed: bool = False
    infrastructure_ok: bool = True
    harness_ok: bool = False
    criteria_results: dict[str, bool] = field(default_factory=dict)
    observability_score: int = 0
    observability_notes: list[str] = field(default_factory=list)
    iters_used: int = 0
    duration_s: float = 0.0
    final_errors: list[str] = field(default_factory=list)
    final_soft_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trace_path: str | None = None
    trace_analysis: dict[str, Any] = field(default_factory=dict)
    best_html_path: str | None = None
    final_html_path: str | None = None
    opening_book_checks: int = 0

    def short_status(self) -> str:
        if not self.infrastructure_ok:
            return "INFRA FAIL"
        if self.passed:
            return "PASS"
        if self.harness_ok:
            return "FAIL (criteria)"
        return "FAIL"


async def _run_one_system_test(
    test: SystemBatteryTest,
    *,
    model: str,
    max_iters: int,
    run_dir: Path,
    browser: LiveBrowser,
    three_model: bool,
    memory_root: str,
    infrastructure_ok: bool,
) -> SystemTestResult:
    test_dir = run_dir / test.slug
    test_dir.mkdir(parents=True, exist_ok=True)
    out_path = test_dir / f"{test.slug}.html"

    stall_s, overall_s, num_ctx = _timeouts_for_model(model)

    agent_kwargs: dict[str, Any] = dict(
        model=model,
        out_path=out_path,
        browser=browser,
        max_iters=max_iters,
        best_of_n=1,
        num_ctx=num_ctx,
        stall_seconds=stall_s,
        overall_seconds=overall_s,
        prompt_version="v1",
        skeleton_mode="default",
        memory_root=memory_root,
        use_architect_split=three_model,
    )

    if three_model and infrastructure_ok:
        b1, b2, r2, b3, r3 = build_three_model_backends(model)
        agent_kwargs["backend"] = b1
        agent_kwargs["backend2"] = b2
        agent_kwargs["model2_role"] = r2
        agent_kwargs["backend3"] = b3
        agent_kwargs["model3_role"] = r3
    else:
        info = backend_mod.detect_backend("ollama")
        if model:
            info = backend_mod.BackendInfo(
                name="ollama", model=model, source="system_tests",
                endpoint=info.endpoint,
            )
        agent_kwargs["backend"] = backend_mod.make_backend(info)

    agent = GameAgent(**agent_kwargs)
    result = SystemTestResult(
        slug=test.slug,
        goal=test.goal,
        model=model,
        suite=test.suite,
        max_iters=max_iters,
        infrastructure_ok=infrastructure_ok,
        trace_path=str(agent.trace_path),
    )

    if not infrastructure_ok:
        result.notes.append("skipped agent run: infrastructure pre-check failed")
        (test_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )
        return result

    t0 = time.monotonic()
    last_report: dict[str, Any] | None = None

    try:
        async for ev in agent.run(test.goal):
            if ev.kind == "phase" and ev.text.startswith("iteration"):
                try:
                    result.iters_used = int(ev.text.split()[1].split("/")[0])
                except Exception:
                    pass
            elif ev.kind == "test":
                last_report = ev.data
                result.harness_ok = bool(ev.data.get("ok", False))
            elif ev.kind == "code":
                if ev.text and Path(ev.text).exists():
                    result.final_html_path = ev.text
            elif ev.kind == "error":
                result.notes.append(f"agent error: {ev.text[:200]}")
                break
    except Exception as e:
        result.notes.append(f"exception: {type(e).__name__}: {e}")

    result.duration_s = round(time.monotonic() - t0, 1)

    if last_report:
        result.final_errors = list(last_report.get("errors") or [])[:5]
        result.final_soft_warnings = list(last_report.get("soft_warnings") or [])[:5]
        ob = last_report.get("opening_book_checks") or []
        result.opening_book_checks = len(ob)

    if agent.best_path.exists():
        result.best_html_path = str(agent.best_path)
        result.harness_ok = True

    trace = parse_trace(Path(result.trace_path) if result.trace_path else None)
    result.trace_analysis = trace.to_dict()
    obs_score, obs_notes = score_observability(
        trace, test_dir, None,
    )
    result.observability_score = obs_score
    result.observability_notes = obs_notes

    criteria_ok, criteria_results = evaluate_pass_criteria(
        test.pass_criteria,
        last_report=last_report,
        trace=trace,
        test_dir=test_dir,
        three_model=three_model,
    )
    result.criteria_results = criteria_results
    result.passed = (
        infrastructure_ok
        and (result.harness_ok or bool(result.best_html_path))
        and criteria_ok
    )

    (test_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result


def write_system_summary(
    run_dir: Path,
    *,
    model: str,
    suite: str,
    three_model: bool,
    started_at: str,
    duration_s: float,
    gpu_report: GpuHygieneReport,
    results: list[SystemTestResult],
    infrastructure_ok: bool,
) -> str:
    n_pass = sum(1 for r in results if r.passed)
    lines = [
        f"# System test run — {run_dir.name}",
        "",
        f"- Started: `{started_at}`",
        f"- Model: `{model}`",
        f"- Suite: `{suite}`",
        f"- Three-model: `{three_model}`",
        f"- Infrastructure pre-check: `{'OK' if infrastructure_ok else 'FAIL'}`",
        f"- Duration: `{duration_s:.1f}s`",
        f"- Pass rate: **{n_pass}/{len(results)}**",
        "",
        "## GPU hygiene",
        "",
        f"- Placement: `{gpu_report.placement_mode}` — {gpu_report.placement_message}",
        f"- Tensor-split before: `{gpu_report.tensor_split_before}`",
        f"- Tensor-split after: `{gpu_report.tensor_split_after}`",
        "",
    ]
    if gpu_report.errors:
        lines.append("**Infrastructure errors:**")
        for e in gpu_report.errors:
            lines.append(f"- {e}")
        lines.append("")
    if gpu_report.unload_results:
        lines.append(f"- Unloaded: {len(gpu_report.unload_results)} model(s)")
        for u in gpu_report.unload_results[:8]:
            lines.append(f"  - {u}")
        lines.append("")

    lines.extend([
        "## Per-test results",
        "",
        "| Test | Status | Harness | Obs | Opening-book | Criteria |",
        "|------|--------|---------|-----|----------------|----------|",
    ])
    for r in results:
        crit = ", ".join(
            f"{k}:{'✓' if v else '✗'}" for k, v in sorted(r.criteria_results.items())
        ) or "—"
        lines.append(
            f"| `{r.slug}` | {r.short_status()} | "
            f"{'ok' if r.harness_ok else 'fail'} | {r.observability_score}/100 | "
            f"{r.opening_book_checks} | {crit} |"
        )

    lines.extend([
        "",
        "## Observability",
        "",
        "Traces should include `session_start`, `stream_start`, browser `test` events, "
        "and `opening_book_retrieved` on smoke runs. PNG screenshots should appear under "
        "each test dir. GPU snapshots: "
        f"`gpu_nvidia_smi_before.txt`, `gpu_nvidia_smi_after.txt`, "
        f"`ollama_ps_before.json`, `ollama_ps_after.json`.",
        "",
    ])
    for r in results:
        if r.observability_notes:
            lines.append(f"- **{r.slug}**: " + "; ".join(r.observability_notes[:4]))

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Manifest: `{run_dir / 'manifest.json'}`",
        "",
        "**Cleanup (optional, manual):** remove the run directory above when you "
        "no longer need traces/screenshots. This runner does not delete logs.",
        "",
    ])
    text = "\n".join(lines)
    (run_dir / "SYSTEM_SUMMARY.md").write_text(text, encoding="utf-8")
    return text


def _write_manifest(
    run_dir: Path,
    *,
    model: str,
    suite: str,
    three_model: bool,
    started_at: str,
    duration_s: float,
    infrastructure_ok: bool,
    gpu_report: GpuHygieneReport,
    results: list[SystemTestResult],
) -> None:
    obj = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "model": model,
        "suite": suite,
        "three_model": three_model,
        "infrastructure_ok": infrastructure_ok,
        "duration_s": round(duration_s, 1),
        "pass_count": sum(1 for r in results if r.passed),
        "test_count": len(results),
        "gpu_hygiene": {
            "placement_mode": gpu_report.placement_mode,
            "errors": gpu_report.errors,
            "tensor_split_before": gpu_report.tensor_split_before,
            "tensor_split_after": gpu_report.tensor_split_after,
        },
        "results": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _seed_memory(memory_root: Path) -> None:
    memory_root.mkdir(parents=True, exist_ok=True)
    shared_playbook = REPO_ROOT / "memory" / "playbook.jsonl"
    if shared_playbook.exists():
        dest = memory_root / "playbook.jsonl"
        if not dest.exists():
            try:
                dest.write_text(shared_playbook.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
    # Opening-book root seeds live under games/game-memory/ — use that tree.
    src_mem = REPO_ROOT / "games" / "game-memory"
    for name in (
        "playtests.jsonl", "asset_audits.jsonl", "animation_audits.jsonl",
        "implementation_outlines.jsonl",
    ):
        sp = src_mem / name
        if sp.exists():
            dp = memory_root / name
            if not dp.exists():
                try:
                    dp.write_text(sp.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass


async def cmd_run(args) -> int:
    battery_path = Path(args.battery) if args.battery else DEFAULT_BATTERY
    tests = filter_tests(
        load_system_battery(battery_path),
        suite=args.suite,
        slugs=args.tests,
    )

    assume_yes = bool(args.yes) or _env_assume_yes()
    tests, skip_notes = confirm_slow_tests_before_run(
        tests, suite=args.suite, assume_yes=assume_yes,
    )
    if not tests:
        for note in skip_notes:
            print(note)
        return 0

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = SYSTEM_TESTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    memory_root = Path(args.memory_root).resolve() if args.memory_root else run_dir / "_memory"
    _seed_memory(memory_root)

    three_model = bool(args.three_model)
    headless = bool(args.headless)

    print(
        f"== system test · suite={args.suite} · model={args.model} · "
        f"three_model={three_model} · tests={len(tests)}"
    )
    print(f"== artifacts: {run_dir}")

    gpu_report = run_gpu_hygiene(
        run_dir, three_model=three_model, do_unload=not args.no_unload,
    )
    infra_ok = gpu_report.ok
    if three_model:
        ok_eps, ep_errors, _ = validate_three_model_endpoints()
        infra_ok = infra_ok and ok_eps
        gpu_report.errors.extend(ep_errors)

    if not infra_ok:
        print("== INFRASTRUCTURE FAIL (see SYSTEM_SUMMARY.md after run)", file=sys.stderr)
        for e in gpu_report.errors:
            print(f"   {e}", file=sys.stderr)

    started_at = datetime.now().isoformat(timespec="seconds")
    browser = LiveBrowser(
        viewport=(800, 600), run_seconds=3.0, headless=headless,
    )
    try:
        await browser.start()
    except Exception as e:
        print(f"could not launch Chromium: {e}", file=sys.stderr)
        return 2

    results: list[SystemTestResult] = []
    t0 = time.monotonic()
    rc = 0 if infra_ok else 2

    try:
        for i, test in enumerate(tests, 1):
            if not confirm_slow_test_run(test.slug, assume_yes=assume_yes):
                print(f"[{i:>2}/{len(tests)}] {test.slug:<24}  SKIPPED (user declined)")
                r = SystemTestResult(
                    slug=test.slug, goal=test.goal, model=args.model,
                    suite=test.suite,
                    max_iters=test.max_iters or args.max_iters,
                    notes=["skipped: user declined slow benchmark"],
                )
                (run_dir / test.slug).mkdir(parents=True, exist_ok=True)
                (run_dir / test.slug / "result.json").write_text(
                    json.dumps(asdict(r), indent=2), encoding="utf-8"
                )
                results.append(r)
                continue

            max_iters = test.max_iters if test.max_iters is not None else args.max_iters
            print(f"[{i:>2}/{len(tests)}] {test.slug:<24}  ", end="", flush=True)
            if not infra_ok and not args.force:
                r = SystemTestResult(
                    slug=test.slug, goal=test.goal, model=args.model,
                    suite=test.suite, max_iters=max_iters,
                    infrastructure_ok=False,
                    notes=["infrastructure pre-check failed"],
                )
                (run_dir / test.slug).mkdir(parents=True, exist_ok=True)
                (run_dir / test.slug / "result.json").write_text(
                    json.dumps(asdict(r), indent=2), encoding="utf-8"
                )
            else:
                try:
                    r = await _run_one_system_test(
                        test,
                        model=args.model,
                        max_iters=max_iters,
                        run_dir=run_dir,
                        browser=browser,
                        three_model=three_model,
                        memory_root=str(memory_root),
                        infrastructure_ok=infra_ok,
                    )
                except KeyboardInterrupt:
                    print("INTERRUPTED")
                    rc = 130
                    break
                except Exception as e:
                    print(f"CRASH: {type(e).__name__}: {e}")
                    r = SystemTestResult(
                        slug=test.slug, goal=test.goal, model=args.model,
                        suite=test.suite, max_iters=max_iters,
                        notes=[f"runner exception: {type(e).__name__}: {e}"],
                    )
            results.append(r)
            print(
                f"{r.short_status():<16}  obs={r.observability_score}/100  "
                f"{r.duration_s:>6.1f}s"
            )
            _write_manifest(
                run_dir, model=args.model, suite=args.suite,
                three_model=three_model, started_at=started_at,
                duration_s=time.monotonic() - t0, infrastructure_ok=infra_ok,
                gpu_report=gpu_report, results=results,
            )
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    duration_total = time.monotonic() - t0
    write_system_summary(
        run_dir, model=args.model, suite=args.suite,
        three_model=three_model, started_at=started_at,
        duration_s=duration_total, gpu_report=gpu_report,
        results=results, infrastructure_ok=infra_ok,
    )
    _write_manifest(
        run_dir, model=args.model, suite=args.suite,
        three_model=three_model, started_at=started_at,
        duration_s=duration_total, infrastructure_ok=infra_ok,
        gpu_report=gpu_report, results=results,
    )

    n_pass = sum(1 for r in results if r.passed)
    print()
    print(f"== {n_pass}/{len(results)} passing  ({duration_total:.1f}s total)")
    print(f"== summary: {run_dir / 'SYSTEM_SUMMARY.md'}")
    print()
    print(
        "Cleanup (optional): when finished reviewing, you may remove:\n"
        f"  {run_dir}\n"
        "This runner keeps all traces and screenshots for inspection."
    )

    if not infra_ok or n_pass < len(results):
        return 2 if rc == 0 else rc
    return rc


def _resolve_run(run_id: str) -> Path | None:
    p = Path(run_id)
    if p.exists() and p.is_dir():
        return p
    direct = SYSTEM_TESTS_ROOT / run_id
    if direct.exists():
        return direct
    if not SYSTEM_TESTS_ROOT.exists():
        return None
    matches = [d for d in SYSTEM_TESTS_ROOT.glob("run_*") if d.name.startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    return None


def cmd_show(args) -> int:
    run_dir = _resolve_run(args.run_id)
    if run_dir is None:
        print(f"no run matching {args.run_id!r}", file=sys.stderr)
        return 2
    summary = run_dir / "SYSTEM_SUMMARY.md"
    if not summary.exists():
        print(f"no SYSTEM_SUMMARY.md in {run_dir}", file=sys.stderr)
        return 2
    print(summary.read_text(encoding="utf-8"))
    return 0


def cmd_list(_args) -> int:
    if not SYSTEM_TESTS_ROOT.exists():
        print("no system test runs yet")
        return 0
    runs = sorted([p for p in SYSTEM_TESTS_ROOT.glob("run_*") if p.is_dir()])
    if not runs:
        print("no system test runs yet")
        return 0
    print(f"{'run_id':<32}  pass  manifest")
    for r in runs:
        mpath = r / "manifest.json"
        if not mpath.exists():
            print(f"{r.name:<32}  —")
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            print(f"{r.name:<32}  {m['pass_count']}/{m['test_count']}  suite={m.get('suite','?')}")
        except Exception:
            print(f"{r.name:<32}  (bad manifest)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Visible system tests for the coding agent.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a system-test suite")
    pr.add_argument("--suite", default="smoke", choices=["smoke", "pacman", "full"])
    pr.add_argument("--model", default=DEFAULT_MODEL)
    pr.add_argument("--max-iters", type=int, default=2,
                    help="default iter cap when test has no max_iters")
    pr.add_argument("--tests", default=None, help="comma-separated slug subset")
    pr.add_argument("--battery", default=None, help=f"battery JSONL (default {DEFAULT_BATTERY})")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--three-model", action="store_true",
                    help="use 3 Ollama slots (11434/11435/11436) with architect+critic sidecars")
    pr.add_argument("--headless", action="store_true",
                    help="hide Chromium (default: visible browser)")
    pr.add_argument("--no-unload", action="store_true",
                    help="skip pre-run Ollama unload on all slot endpoints")
    pr.add_argument("--force", action="store_true",
                    help="run tests even when infrastructure pre-check failed")
    pr.add_argument("--memory-root", default=None)
    pr.add_argument("--keep-artifacts", action="store_true", default=True,
                    help="always keep artifacts (default); do not delete logs")
    pr.add_argument("-y", "--yes", action="store_true",
                    help="run slow benchmarks without prompting (also SYSTEM_TEST_ASSUME_YES=1)")

    ps = sub.add_parser("show", help="print SYSTEM_SUMMARY.md for a run")
    ps.add_argument("run_id")

    sub.add_parser("list", help="list past system-test runs")

    args = p.parse_args()
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "list":
        return cmd_list(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
