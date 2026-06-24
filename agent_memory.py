"""Opening-book / component / lean-budget memory retrieval extracted from agent.py.

`MemoryRetrievalMixin` holds the first-build/plan-turn memory assembly helpers
that `GameAgent` used to carry inline:

  - `_retrieve_opening_book_block` — retrieve the matched implementation outline
    (+ universal fallback), playtests, asset/animation audits, and the optional
    VLM checklist, rendered into the `<opening-book>` block.
  - `_retrieve_components_block` — retrieve mechanics-level JS snippets, pinning
    the engine skeleton (game loop + buffered input) to the protected FRONT for
    open-domain goals so they survive the lean budget.
  - `_apply_lean_memory_budget` — cap the COMBINED size of the three first-build
    memory blocks (opening-book > components > playbook) in lean mode.
  - `_detect_open_domain_build` — genre-free novelty check from the retrieved
    outline id/score, used to decide component count + pinning + protection.

These were moved VERBATIM out of `GameAgent` (no behavior change). `GameAgent`
inherits this mixin, so every `self.*` attribute reference (`self._memory`,
`self._criteria`, `self._session_assets`) and helper (`self._trace`,
`self._lean_prompt_active`) resolves unchanged through normal MRO lookup.
"""
from __future__ import annotations

from memory import (
    OpeningBookHit,
    render_components_block,
    render_opening_book_block,
    render_vlm_checklist_section,
    VLM_CHECKLIST_SKIP_IDS,
)


class MemoryRetrievalMixin:
    """First-build / plan-turn memory retrieval for GameAgent (see docstring)."""

    # Combined char ceiling for the three first-build memory blocks in lean
    # mode. opening-book outline (~1.7K) + components (~2.2K) fit; the
    # lower-priority playbook is dropped when it would push past this. Keeps
    # a local model from reading 8KB of overlapping "past lessons" before
    # it writes a line.
    _LEAN_MEMORY_COMBINED_BUDGET = 4500

    # Below this retrieved-outline score a fresh goal is treated as
    # OPEN-DOMAIN (no strong opening-library match). There is no goal->
    # prompt_library fuzzy matcher and a genre/title list is forbidden, so
    # the outline score is the genre-free proxy. Open-domain first builds
    # get one extra component (k=4) and have the game-loop+input snippets
    # PROTECTED from the lean budget — battlezone 20260622 dropped ALL
    # components+playbook on its first build and had to invent the loop.
    _OPEN_DOMAIN_OUTLINE_FLOOR = 0.5

    def _retrieve_opening_book_block(
        self,
        goal: str,
        *,
        stage: str = "plan",
        char_budget: int | None = None,
        deep: bool | None = None,
        extra_tokens: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve compact root/live opening-book recipes with hard caps.

        `extra_tokens` are appended to the modality tokens used for
        retrieval — used by the seed path to pass structural tokens pulled
        from the working file (DOM ids, function names) so the goal alone
        ("add a button") doesn't mis-rank the outline.
        """
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(goal)
                    + _detect_board_intent(goal)
                    + _detect_dom_intent(goal)
                )
            except Exception:
                mod_toks = []
            if extra_tokens:
                mod_toks = mod_toks + list(extra_tokens)
            outline = self._memory.retrieve_implementation_outline(goal, mod_toks)
            # Universal fallback: a novel/open-domain goal may match NO outline
            # (Jaccard empty and no recipe route). Rather than plan with no
            # state/order/traps contract at all, force the genre-free
            # controllable-canvas-game outline so every build still inherits a
            # state-on-window / input / dt-cap / draw-order / restart skeleton.
            outline_fallback = False
            if outline is None:
                outline = self._memory._outline_item_by_id(
                    "outline-controllable-canvas-game"
                )
                outline_fallback = outline is not None
            playtests = self._memory.retrieve_playtests(
                goal, mod_toks, k=3 if stage == "plan" else 1,
            )
            asset_audits = self._memory.retrieve_asset_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            animation_audits = self._memory.retrieve_animation_audits(
                goal, mod_toks, k=2 if stage == "plan" else 1,
            )
            vlm_checklist: str | None = None
            if stage == "plan":
                try:
                    vp_recipe, vp_diag = self._memory.find_visual_playtest_for(
                        goal=goal or "",
                        plan_text=self._criteria or "",
                        asset_names=list(self._session_assets.keys()),
                    )
                except Exception:
                    vp_recipe, vp_diag = None, {}
                if (
                    vp_recipe is not None
                    and vp_recipe.id not in VLM_CHECKLIST_SKIP_IDS
                ):
                    vlm_checklist = render_vlm_checklist_section(vp_recipe)
                    if vlm_checklist:
                        self._trace({
                            "kind": "vlm_checklist_injected",
                            "recipe_id": vp_recipe.id,
                            "top_candidates": (vp_diag or {}).get("top_candidates"),
                        })
            # Plan stage deep-renders the ONE matched outline's recipe
            # (state/order/traps/tuning/probes) under a 3600-char cap —
            # ~+600 tokens in the smallest prompt of the session. Code
            # stage stays shallow at 1400 so iterate prompts never grow.
            block = render_opening_book_block(
                outline, playtests, asset_audits, animation_audits,
                char_budget=char_budget if char_budget is not None else (3600 if stage == "plan" else 1400),
                deep=deep if deep is not None else (stage == "plan"),
                vlm_checklist=vlm_checklist,
            )

            def _row(kind: str, hit: OpeningBookHit) -> dict:
                return {
                    "kind": kind,
                    "id": hit.item.id,
                    "tier": hit.item.source_tier,
                    "score": round(hit.score, 4),
                    "recipe": hit.item.recipe,
                }

            hits: list[dict] = []
            if outline:
                hits.append(_row("outline", outline))
            hits.extend(_row("playtest", h) for h in playtests)
            hits.extend(_row("asset_audit", h) for h in asset_audits)
            hits.extend(_row("animation_audit", h) for h in animation_audits)
            if hits:
                self._trace({
                    "kind": "opening_book_retrieved",
                    "stage": stage,
                    "hits": hits,
                    "modality_tokens": mod_toks,
                    "outline_fallback": outline_fallback,
                })
            return block, hits
        except Exception as e:
            self._trace({"kind": "opening_book_error", "stage": stage, "err": str(e)})
            return "", []

    def _retrieve_components_block(
        self,
        query: str,
        *,
        stage: str = "plan",
        k: int = 3,
        ensure_ids: list[str] | None = None,
    ) -> str:
        """Retrieve component-library snippets as a <components> block.

        Capability-round item 1: tested, mechanics-level JS the model
        pastes and ADAPTS (memory/components.jsonl). `query` is the goal
        at first build; at fix turns it is the blocker text so a snippet
        is only injected when it matches the actual failure. Returns ""
        when nothing matches (safe degradation).
        """
        try:
            mod_toks: list[str] = []
            try:
                from memory import (
                    _detect_3d_intent, _detect_board_intent, _detect_dom_intent,
                )
                mod_toks = (
                    _detect_3d_intent(query)
                    + _detect_board_intent(query)
                    + _detect_dom_intent(query)
                )
            except Exception:
                mod_toks = []
            hits = self._memory.retrieve_components(query, mod_toks, k=k)
            # Universal-fallback pinning: guarantee the engine-skeleton snippets
            # (game loop, input) are present AND FIRST for open-domain goals so
            # they survive the lean budget. Jaccard can either miss a pinned id
            # entirely OR retrieve it but rank it below the char cap (the latter
            # silently dropped input-manager-buffered for novel goals); both
            # cases are handled by moving every pinned hit to the protected
            # front and fetching any that were not retrieved at all.
            if ensure_ids:
                by_id = {h.item.id: h for h in hits}
                pinned: list[OpeningBookHit] = []
                for cid in ensure_ids:
                    if cid in by_id:
                        pinned.append(by_id.pop(cid))
                    else:
                        pinned.extend(self._memory.components_by_ids([cid]))
                if pinned:
                    rest = [h for h in hits if h.item.id in by_id]
                    hits = pinned + rest
            block = render_components_block(
                hits, char_budget=2200 if stage == "plan" else 1400,
            )
            if hits:
                self._trace({
                    "kind": "components_injected",
                    "stage": stage,
                    "ids": [h.item.id for h in hits],
                    "scores": [round(h.score, 4) for h in hits],
                    "chars": len(block or ""),
                    "rendered": bool(block),
                })
            return block or ""
        except Exception as e:
            self._trace({"kind": "components_error", "stage": stage, "err": str(e)})
            return ""

    def _apply_lean_memory_budget(
        self, opening_block: str, components_block: str, playbook_block: str,
        *, protect_components: bool = False,
    ) -> tuple[str, str, str]:
        """In lean mode, cap the COMBINED size of the first-build memory
        blocks. Priority: opening-book outline > components > playbook —
        whole blocks are dropped (never mangled) lowest-priority first once
        the budget is exceeded. No-op outside lean mode.

        `protect_components` (seed continuations) keeps the components block
        even when opening already filled the budget — on a seed iter 1 the
        components are the copy-paste-correct snippets (help overlay, media
        loader) the weak model most needs, so the playbook is trimmed first
        instead. (2026-06-21 seed trace dropped `help-overlay-modal` here.)"""
        if not self._lean_prompt_active():
            return opening_block, components_block, playbook_block
        budget = self._LEAN_MEMORY_COMBINED_BUDGET
        used = len(opening_block or "")  # opening (priority 1) always kept
        blocked = False

        def _fit(block: str) -> str:
            nonlocal used, blocked
            if not block:
                return ""
            if blocked or used + len(block) > budget:
                blocked = True
                return ""
            used += len(block)
            return block

        if protect_components and components_block:
            # Keep components unconditionally (count it against the budget so
            # playbook still yields), then fit playbook in what remains.
            cb = components_block
            used += len(components_block)
            pb = _fit(playbook_block)
        else:
            cb = _fit(components_block)
            pb = _fit(playbook_block)
        if (cb != components_block) or (pb != playbook_block):
            self._trace({
                "kind": "lean_memory_budget_applied",
                "budget": budget,
                "kept_opening_chars": len(opening_block or ""),
                "kept_components_chars": len(cb),
                "kept_playbook_chars": len(pb),
                "dropped_components": bool(components_block) and not cb,
                "dropped_playbook": bool(playbook_block) and not pb,
            })
        return opening_block, cb, pb

    def _detect_open_domain_build(
        self, opening_hits: list[dict]
    ) -> tuple[bool, list[dict]]:
        """Genre-free open-domain detection for a fresh first build.

        A fresh goal is "novel" when it matches NO outline, falls back to the
        universal controllable-canvas-game outline, or only weakly matches a
        specific one. We can't proxy via prompt_library (no goal->library
        matcher; genre/title lists are forbidden), and the recipe-routed
        outline score is usually 1.0, so we key on the outline ID + score
        together. Returns (open_domain_build, outline_rows). Extracted
        verbatim from run() so it can be unit-tested in isolation.
        """
        outline_rows = [h for h in opening_hits if h.get("kind") == "outline"]
        if not outline_rows:
            return True, outline_rows
        o = outline_rows[0]
        open_domain = (
            o.get("id") == "outline-controllable-canvas-game"
            or o.get("score", 1.0) < self._OPEN_DOMAIN_OUTLINE_FLOOR
        )
        return open_domain, outline_rows
