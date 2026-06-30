# ARCHIVED — one-shot migration already applied; see AGENTS.md §1b. Do not re-run.
#!/usr/bin/env python3
"""One-shot apply agent.py modular split. Run from repo root after mixin files exist."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent.py"


def method_ranges(lines: list[str]) -> list[tuple[int, str]]:
    ga = next(i for i, l in enumerate(lines) if l.startswith("class GameAgent"))
    out: list[tuple[int, str]] = []
    for i in range(ga, len(lines)):
        m = re.match(r"    (async )?def (\w+)\(", lines[i])
        if m:
            out.append((i, m.group(2)))
    return out


def method_range(method_starts: list[tuple[int, str]], name: str, n_lines: int) -> tuple[int, int]:
    for idx, (start, n) in enumerate(method_starts):
        if n == name:
            end = method_starts[idx + 1][0] if idx + 1 < len(method_starts) else n_lines
            return start, end
    raise KeyError(name)


def delete_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> None:
    for start, end in sorted(ranges, key=lambda x: x[0], reverse=True):
        del lines[start:end]


def module_names(path: Path) -> list[str]:
    mod = ast.parse(path.read_text())
    names: list[str] = []
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
    return sorted(set(names))


def apply_phase1(lines: list[str]) -> None:
    start = next(i for i, l in enumerate(lines) if l.startswith("_PROJECT_CONFIG_FILES"))
    end = next(
        i for i, l in enumerate(lines)
        if l.startswith("# Centipede trace 20260512")
    )
    chunk = lines[start:end]
    names = module_names(ROOT / "agent_helpers.py")
    imp = (
        "# Pure helpers — seed media, HTML parsing, compaction constants (agent_helpers.py).\n"
        "from agent_helpers import (\n"
        + "".join(f"    {n},\n" for n in names)
        + ")\n\n"
    )
    del lines[start:end]
    lines[start:start] = [imp]


def apply_phase2(lines: list[str]) -> None:
    ga = next(i for i, l in enumerate(lines) if l.startswith("class GameAgent"))
    ms = method_ranges(lines)
    feedback_methods = [
        "_record_classifier_overrule", "_queue_internal_feedback",
        "_apply_initial_goal_scoping", "_derive_allowed_forbidden_tags",
        "_clear_scoped_constraints", "_previous_iter_clean_for_scope_guard",
        "_configure_scoped_constraints", "_scoped_reply_violation",
        "_scoped_retry_instruction", "_apply_scoped_check_to_report",
        "_consumed_feedback_summary", "_has_active_blocker", "_route_forces_fix_mode",
        "_should_defer_feedback_for_blocker", "_feedback_shingles", "_feedback_keywords",
        "_deferral_signature", "_count_recent_deferrals", "_detect_repeated_feedback",
        "_maybe_clear_asset_reprompt_via_code", "_is_post_clean_instruction",
        "_compact_post_clean_context", "_feedback_route_cache_key",
        "_route_user_feedback_llm", "_parse_feedback_route_json",
        "_precompute_feedback_route", "_flush_user_injections",
    ]
    fb_start = next(
        i for i, l in enumerate(lines)
        if l.startswith("# Centipede trace 20260512")
    )
    fb_end = next(
        i for i, l in enumerate(lines)
        if l.startswith("def _repetition_loop_abort_message")
    )
    delete_ranges(lines, [(fb_start, fb_end)] + [method_range(ms, n, len(lines)) for n in feedback_methods])
    # remove orphan classifier threshold if present
    for i, l in enumerate(lines):
        if l.strip() == "_CLASSIFIER_AUTO_DISABLE_THRESHOLD = 2":
            del lines[i]
            break
    fb_names = [n for n in module_names(ROOT / "agent_feedback.py") if n != "FeedbackRoutingMixin"]
    imp = (
        "from agent_feedback import FeedbackRoutingMixin\n"
        "from agent_feedback import (\n"
        + "".join(f"    {n},\n" for n in fb_names)
        + ")\n"
    )
    for i, l in enumerate(lines):
        if l.strip() == "from agent_memory import MemoryRetrievalMixin":
            lines.insert(i + 1, "from agent_prompts import PromptBuildingMixin")
            lines.insert(i + 2, "from agent_compaction import CompactionMixin")
            lines.insert(
                i + 3,
                "from agent_stream import StreamMaterializeMixin, _repetition_loop_abort_message",
            )
            lines.insert(i + 4, imp.replace("from agent_feedback", "from agent_feedback", 1))
            break
    for i, l in enumerate(lines):
        if l.startswith("class GameAgent("):
            lines[i] = "class GameAgent(FeedbackRoutingMixin, ProbeHandlingMixin, MemoryRetrievalMixin):"
            break


def apply_phase3(lines: list[str]) -> None:
    ms = method_ranges(lines)
    prompt_methods = [
        "_prompt_orders_full_rewrite", "_seed_structural_tokens", "_report_digest_lines",
        "_wrap_report_block", "_build_structured_summary", "_build_visual_playtest_prompt",
        "_compact_warnings_for_prompt", "_format_report_for_model", "_identifiers",
        "_identifier_occurrence_slice", "_signature_focus_identifiers", "_diagnose_is_shotgun",
        "_diagnose_mentions_subsystem", "_patches_touch_subsystem_idents", "_seed_html_for_prompt",
        "_focused_slice", "_partial_patch_recovery_block", "_repeat_error_fastpath_block",
        "_build_fix_prompt",
    ]
    delete_ranges(lines, [method_range(ms, n, len(lines)) for n in prompt_methods])
    for i, l in enumerate(lines):
        if l.strip() == "from agent_feedback import FeedbackRoutingMixin":
            lines.insert(i + 1, "from agent_prompts import PromptBuildingMixin")
            break
    for i, l in enumerate(lines):
        if l.startswith("class GameAgent("):
            lines[i] = "class GameAgent(PromptBuildingMixin, FeedbackRoutingMixin, ProbeHandlingMixin, MemoryRetrievalMixin):"
            break


def apply_phase4(lines: list[str]) -> None:
    ms = method_ranges(lines)
    compaction_methods = ["_summarize_content", "_maybe_reset_continuation_context", "_prune_messages"]
    delete_ranges(lines, [method_range(ms, n, len(lines)) for n in compaction_methods])
    for i, l in enumerate(lines):
        if l.strip() == "from agent_prompts import PromptBuildingMixin":
            lines.insert(i + 1, "from agent_compaction import CompactionMixin")
            break
    for i, l in enumerate(lines):
        if l.startswith("class GameAgent("):
            lines[i] = (
                "class GameAgent(PromptBuildingMixin, CompactionMixin, "
                "FeedbackRoutingMixin, ProbeHandlingMixin, MemoryRetrievalMixin):"
            )
            break


def apply_phase5(lines: list[str]) -> None:
    ms = method_ranges(lines)
    stream_methods = [
        "_stub_rejected_reply", "_should_skip_format_doctor", "_no_usable_code_fallback",
        "_run_format_doctor", "_stream", "_materialize", "_truncation_diagnosis",
        "_extract_html", "_extract_html_inner", "_extract_question", "_extract_diagnose",
        "_hash_warning", "_advance_warning_persistence", "_clean_actionable_vision_note",
    ]
    mod_start = next(i for i, l in enumerate(lines) if l.startswith("def _repetition_loop_abort_message"))
    mod_end = next(i for i, l in enumerate(lines) if l.startswith("@dataclass") and i > mod_start)
    delete_ranges(
        lines,
        [(mod_start, mod_end)] + [method_range(ms, n, len(lines)) for n in stream_methods],
    )
    for i, l in enumerate(lines):
        if l.strip() == "from agent_compaction import CompactionMixin":
            lines.insert(
                i + 1,
                "from agent_stream import StreamMaterializeMixin, _repetition_loop_abort_message\n",
            )
            break
    for i, l in enumerate(lines):
        if l.startswith("class GameAgent("):
            lines[i] = (
                "class GameAgent(PromptBuildingMixin, CompactionMixin, StreamMaterializeMixin, "
                "FeedbackRoutingMixin, ProbeHandlingMixin, MemoryRetrievalMixin):"
            )
            break


def apply_phase6(lines: list[str]) -> None:
    run_start = next(i for i, l in enumerate(lines) if re.match(r"    async def run\(", lines[i]))
    run_end = next(i for i in range(run_start + 1, len(lines)) if re.match(r"    async def run_with_restarts\(", lines[i]))
    phase_b = next(
        i for i in range(run_start, run_end)
        if "# ---- PHASE B: build/iterate" in lines[i]
        and i + 1 < len(lines)
        and "awaiting_confirm" in lines[i + 1]
    )
    exit_turn = next(i for i in range(run_start, run_end) if "# ---- Item 5: exit-decision turn" in lines[i])
    body_start = next(
        i for i in range(run_start, run_end)
        if lines[i].strip() == "if not continuation:"
    )
    part1 = lines[body_start:phase_b]
    part2 = lines[phase_b:exit_turn]
    part3 = lines[exit_turn:run_end]

    def patch_returns(block: list[str]) -> list[str]:
        out: list[str] = []
        for l in block:
            if re.match(r"^        return$", l):
                out.append("        self._run_session_complete = True")
            out.append(l)
        return out

    part1 = patch_returns(part1)
    part2 = patch_returns(part2)
    part2 = [l.replace("awaiting_confirm = False", "self._awaiting_confirm = False") for l in part2]
    part2 = [re.sub(r"\bawaiting_confirm\b", "self._awaiting_confirm", l) for l in part2]
    part3 = [re.sub(r"\bawaiting_confirm\b", "self._awaiting_confirm", l) for l in part3]

    doc_lines = lines[run_start:body_start]

    new_run = doc_lines + [
        "        self._run_session_complete = False",
        "        async for ev in self._run_phase_a_and_first_build(",
        "            goal,",
        "            continuation=continuation,",
        "            plan_only=plan_only,",
        "            patch_only=patch_only,",
        "        ):",
        "            yield ev",
        "        if self._run_session_complete:",
        "            return",
        "        async for ev in self._run_build_iterate_loop(continuation=continuation):",
        "            yield ev",
        "        if self._run_session_complete:",
        "            return",
        "        async for ev in self._run_exit_and_finalize():",
        "            yield ev",
        "",
        "    async def _run_phase_a_and_first_build(",
        "        self,",
        "        goal: str,",
        "        *,",
        "        continuation: bool,",
        "        plan_only: bool,",
        "        patch_only: bool,",
        "    ) -> AsyncIterator[AgentEvent]:",
        '        """Phase A planning, optional assets, first-build message assembly."""',
    ] + part1 + [
        "",
        "    async def _run_build_iterate_loop(",
        "        self,",
        "        *,",
        "        continuation: bool,",
        "    ) -> AsyncIterator[AgentEvent]:",
        '        """Phase B iteration loop and cap-reached bonus turn."""',
    ] + part2 + [
        "",
        "    async def _run_exit_and_finalize(self) -> AsyncIterator[AgentEvent]:",
        '        """Exit-decision turn, final test, session outcome."""',
    ] + part3

    del lines[run_start:run_end]
    lines[run_start:run_start] = new_run


def add_imports(lines: list[str]) -> None:
    if "from agent_probes import ProbeHandlingMixin" in "".join(lines[:130]):
        return
    for i, l in enumerate(lines):
        if l.strip() == "from agent_memory import MemoryRetrievalMixin":
            lines.insert(i + 1, "from agent_memory import MemoryRetrievalMixin")
            break


def add_toc(lines: list[str]) -> None:
    needle = "        continuation `goal` arg as feedback, leaving the original intact.\n        \"\"\"\n        if not continuation:"
    toc = (
        "        continuation `goal` arg as feedback, leaving the original intact.\n        \"\"\"\n"
        "        # ---- run() TOC (see AGENTS.md §1b) --------------------------------\n"
        "        # Session init + continuation baseline sanitize\n"
        "        # ---- PHASE A: planning ------------------------------------------\n"
        "        # ---- seed file OR memory skeleton for the first build ----------\n"
        "        # ---- PHASE B: build/iterate -------------------------------------\n"
        "        if not continuation:"
    )
    text = "\n".join(lines)
    if "run() TOC" in text:
        return
    text = text.replace(
        "        continuation `goal` arg as feedback, leaving the original intact.\n        \"\"\"\n        if not continuation:",
        toc,
        1,
    )
    lines[:] = text.splitlines()


def main() -> None:
    lines = AGENT.read_text().splitlines()
    add_toc(lines)
    apply_phase1(lines)
    apply_phase2(lines)
    apply_phase3(lines)
    apply_phase4(lines)
    apply_phase5(lines)
    apply_phase6(lines)
    AGENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {AGENT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
