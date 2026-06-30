# ARCHIVED — one-shot migration already applied; see AGENTS.md §1b. Do not re-run.
#!/usr/bin/env python3
"""Extract agent_gates, agent_critic, agent_assets mixins from agent.py."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "agent.py"

GATES_METHODS = [
    "_apply_undrawn_art_intent_gate",
    "_classify_failure",
    "_synthetic_report_no_browser",
    "_apply_dead_animation_check_to_report",
    "_apply_still_frame_frozen_downgrade",
    "_apply_player_stuck_downgrade",
]

CRITIC_METHODS = [
    "_critic_note_fingerprint",
    "_queue_visual_critic_coaching",
    "_detect_vlm",
    "_maybe_inject_visual_playtest_auto_probes",
    "_maybe_inject_media_probes",
    "_maybe_inject_asset_miss_probe",
    "_animation_expected",
    "_augment_recipe_for_animation",
    "_action_frame_question_indices",
    "_strip_action_frame_questions",
    "_critic_abstained",
    "_parse_visual_playtest_response",
    "_format_visual_playtest_critique",
    "_sanitize_ask_reply",
    "_ask_html_excerpt",
    "run_ask_turn",
    "run_visual_critic",
    "_run_opening_book_sidecars",
    "_critic_runs_on_independent_slot",
    "_drain_pending_critic_task",
    "_spawn_visual_critic",
    "_autonomous_playtest_disabled",
    "_evaluate_behavior_playtest_check",
    "_run_autonomous_playtest",
    "_run_structured_local_vlm_critique",
    "_run_vision_judge",
]

ASSETS_METHODS = [
    "_scan_html_for_asset_refs",
    "_check_asset_alignment",
    "_early_rehydrate_seed_media",
    "_render_seed_media_contract",
    "_scan_html_for_sound_refs",
    "_check_sound_alignment",
    "_maybe_prewarm_diffusers_during_phase_a",
    "_mark_unused_media_as_stale_for_continuation",
    "_filter_media_specs_to_allowed",
    "_maybe_generate_assets_and_sounds",
]

ASSET_CLASS_ATTR_START = "    # Pattern catches:"
ASSET_CLASS_ATTR_END = "    _SEED_TOKEN_STOPWORDS = frozenset({"


def method_ranges(lines: list[str]) -> list[tuple[int, str]]:
    ga = next(i for i, l in enumerate(lines) if l.startswith("class GameAgent"))
    out: list[tuple[int, str]] = []
    for i in range(ga, len(lines)):
        m = re.match(r"    (async )?def (\w+)\(", lines[i])
        if m:
            out.append((i, m.group(2)))
    return out


def extract_methods(lines: list[str], names: list[str]) -> list[str]:
    ms = method_ranges(lines)
    chunks: list[str] = []
    for name in names:
        for idx, (start, n) in enumerate(ms):
            if n != name:
                continue
            end = ms[idx + 1][0] if idx + 1 < len(ms) else len(lines)
            chunks.append("".join(l + "\n" for l in lines[start:end]))
            break
        else:
            raise KeyError(f"missing method {name}")
    return chunks


def extract_asset_class_attrs(lines: list[str]) -> str:
    start = next(i for i, l in enumerate(lines) if l.startswith(ASSET_CLASS_ATTR_START))
    end = next(i for i, l in enumerate(lines) if l.startswith(ASSET_CLASS_ATTR_END))
    return "".join(l + "\n" for l in lines[start:end])


def delete_method_ranges(lines: list[str], names: list[str]) -> None:
    ms = method_ranges(lines)
    ranges: list[tuple[int, int]] = []
    for name in names:
        for idx, (start, n) in enumerate(ms):
            if n == name:
                end = ms[idx + 1][0] if idx + 1 < len(ms) else len(lines)
                ranges.append((start, end))
                break
    for start, end in sorted(ranges, key=lambda x: x[0], reverse=True):
        del lines[start:end]


def delete_asset_block(lines: list[str]) -> None:
    start = next(i for i, l in enumerate(lines) if l.startswith("    # In the donkey-kong trace 20260513_122154"))
    end = next(i for i, l in enumerate(lines) if l.startswith("    def _is_local_backend"))
    del lines[start:end]


def write_mixin(path: Path, class_name: str, doc: str, body: str, imports: str) -> None:
    path.write_text(
        f'"""{doc}\n\nMoved VERBATIM from `GameAgent` (no behavior change).\n"""\n\n'
        f"from __future__ import annotations\n\n"
        f"{imports}\n\n"
        f"class {class_name}:\n\n"
        f'    """{doc}"""\n\n'
        f"{body}",
        encoding="utf-8",
    )


def main() -> None:
    lines = AGENT.read_text(encoding="utf-8").splitlines(keepends=True)
    if "GateProcessingMixin" in "".join(lines[:250]):
        print("Phase 7 already applied")
        return

    asset_attrs = extract_asset_class_attrs(lines)
    gates_body = "".join(extract_methods(lines, GATES_METHODS))
    critic_body = "".join(extract_methods(lines, CRITIC_METHODS))
    assets_body = asset_attrs + "".join(extract_methods(lines, ASSETS_METHODS))

    write_mixin(
        ROOT / "agent_gates.py",
        "GateProcessingMixin",
        "Report post-processing gates for GameAgent.",
        gates_body,
        "from typing import Any\n",
    )

    write_mixin(
        ROOT / "agent_critic.py",
        "CriticMixin",
        "VLM / visual playtest / autonomous critic for GameAgent.",
        critic_body,
        (
            "from collections.abc import AsyncIterator\n"
            "from typing import Any\n\n"
            "from agent import AgentEvent\n"
        ),
    )

    write_mixin(
        ROOT / "agent_assets.py",
        "AssetGenerationMixin",
        "Mid-session asset/sound generation and alignment scans.",
        assets_body,
        (
            "from collections.abc import AsyncIterator\n"
            "from typing import Any\n\n"
            "from agent import AgentEvent\n"
        ),
    )

    delete_method_ranges(lines, GATES_METHODS)
    delete_method_ranges(lines, CRITIC_METHODS)
    delete_method_ranges(lines, ASSETS_METHODS)
    delete_asset_block(lines)

    text = "".join(lines)
    mro_old = (
        "class GameAgent(\n"
        "    PromptBuildingMixin,\n"
        "    CompactionMixin,\n"
        "    StreamMaterializeMixin,\n"
        "    FeedbackRoutingMixin,\n"
        "    ProbeHandlingMixin,\n"
        "    MemoryRetrievalMixin,\n"
        "):"
    )
    mro_new = (
        "class GameAgent(\n"
        "    PromptBuildingMixin,\n"
        "    CompactionMixin,\n"
        "    StreamMaterializeMixin,\n"
        "    AssetGenerationMixin,\n"
        "    CriticMixin,\n"
        "    GateProcessingMixin,\n"
        "    FeedbackRoutingMixin,\n"
        "    ProbeHandlingMixin,\n"
        "    MemoryRetrievalMixin,\n"
        "):"
    )
    text = text.replace(mro_old, mro_new)

    imp_anchor = "from agent_stream import StreamMaterializeMixin, _repetition_loop_abort_message\n"
    imp_new = (
        imp_anchor
        + "from agent_gates import GateProcessingMixin\n"
        + "from agent_critic import CriticMixin\n"
        + "from agent_assets import AssetGenerationMixin\n"
    )
    text = text.replace(imp_anchor, imp_new)

    mod_anchor = "            agent_stream,\n"
    mod_new = (
        mod_anchor
        + "            __import__(\"agent_gates\"),\n"
        + "            __import__(\"agent_critic\"),\n"
        + "            __import__(\"agent_assets\"),\n"
    )
    text = text.replace(mod_anchor, mod_new, 1)

    AGENT.write_text(text, encoding="utf-8")
    print("Phase 7 applied")


if __name__ == "__main__":
    main()
