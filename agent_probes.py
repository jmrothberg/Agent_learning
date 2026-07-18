"""Probe-result handlers extracted from agent.py for readability/debuggability.

`ProbeHandlingMixin` holds the methods that post-process a browser test
`report` dict's model-authored acceptance probes:

  - `_apply_impossible_probe_downgrade_to_report` — demote structurally
    impossible self-probes (read state the code never assigns) to non-gating
    advisories so a behaviorally-correct build is not gated forever.
  - `_probe_shape_key` / `_refresh_probe_error_fields` / `_remove_probe_warnings`
    — small helpers used by the quarantine pipeline.
  - `_handle_probe_eval_errors` — trace / soften / quarantine probes that fail
    at eval time, the all-probes-quarantined ship gate, and the C1
    healthy-harness softening.

These were moved VERBATIM out of `GameAgent` (no behavior change). `GameAgent`
inherits this mixin, so every `self.*` attribute reference (all set in
`GameAgent.__init__`: `self._probe_lint_findings`, `self._probe_eval_error_streak`,
`self._probe_eval_error_shape_streak`, `self._probe_names_ever_passed`,
`self._pending_probe_quarantine_notices`, `self._probes`,
`self._all_probes_quarantined_gate_used`) and helper (`self._trace`) resolves
unchanged through normal MRO lookup.
"""
from __future__ import annotations

from typing import Any


class ProbeHandlingMixin:
    """Probe-result post-processing for GameAgent (see module docstring)."""

    def _apply_impossible_probe_downgrade_to_report(self, report: dict[str, Any]) -> None:
        """Demote STRUCTURALLY-IMPOSSIBLE self-probes to non-gating advisories.

        The model authors its own acceptance probes in Phase A. Sometimes a
        probe reads state the game's code NEVER produces (a DOM `<img>` in a
        pure-canvas game, `window.gameState` / `window.game.reset` that were
        never assigned). Such a probe evaluates FALSY every iteration no matter
        how good or broken the build is — it carries zero behavioral signal,
        yet `tools.py` appends a gating `PROBE FAILED` soft_warning for it, so
        `report["ok"]` can never become True. That is the unwinnable-loop shape
        (see `_apply_player_stuck_downgrade` / `_apply_dead_animation_check_to_report`):
        a behaviorally-correct build stays ok=False forever, `_save_best()`
        never runs, and the loop reads the permanent failure as "stuck" and
        burns best-of-N retries instead of honoring the user's next feedback.

        Holochess trace 20260623_204052: the model's `all_piece_images_loaded`
        (queried DOM `<img>` in a canvas game), `loader_has_all_pieces`
        (`window.gameState||window.state`) and `pieces_render_after_reset`
        (`window.game.reset`) read state the game never assigns. `probe_lint`
        ALREADY detected all three every iter (`unassigned_property_read` /
        `probe_bait_flag` findings in `self._probe_lint_findings`) — but the
        finding was advisory-only: it nagged the model, it never removed the
        probe from the gate. This routes those already-detected impossible
        probes into the non-gating `warnings` channel and recomputes `ok`.

        Genre-free: the trigger is purely "the probe references properties the
        code never assigns", a signal the harness already computes. Real
        failures still hard-gate untouched — JS `errors`, `page_errors`,
        `console_errors`, and any probe that reads REAL game state. Eval-error
        probes are left to `_handle_probe_eval_errors` (run just before this).
        """
        findings = self._probe_lint_findings or []
        flagged = {
            str(f.get("name"))
            for f in findings
            if f.get("kind") in ("unassigned_property_read", "probe_bait_flag")
            and f.get("name")
        }
        if not flagged:
            return
        # Only demote probes that are FAILING FALSY this iter (skip eval_error;
        # those are quarantined separately) and whose name probe_lint flagged.
        demote = {
            str(p.get("name"))
            for p in (report.get("probes") or [])
            if not p.get("ok")
            and p.get("kind") != "eval_error"
            and str(p.get("name")) in flagged
        }
        if not demote:
            return
        # Strip the gating `PROBE FAILED [name]` soft_warnings for these probes.
        self._remove_probe_warnings(report, demote)
        names = ", ".join(f"`{n}`" for n in sorted(demote))
        warns = list(report.get("warnings") or [])
        warns.append(
            "IMPOSSIBLE PROBE (advisory — does not block shipping): these "
            f"self-probes read state the game's code never assigns — {names}. "
            "They evaluate falsy regardless of game behavior, so they cannot "
            "test anything real and must NOT gate shipping. If you still want "
            "the check, rewrite the probe to read state the running game "
            "actually exposes; otherwise drop it. Real errors and probes that "
            "read real game state still gate."
        )
        report["warnings"] = warns
        # ok recompute mirrors tools.py: (no errors) and (no soft_warnings).
        report["ok"] = not (report.get("errors") or []) and not report["soft_warnings"]
        self._trace({
            "kind": "impossible_probe_downgraded",
            "names": sorted(demote),
        })

    @staticmethod
    def _probe_shape_key(p: dict[str, Any]) -> str:
        name = str(p.get("name") or "probe").strip().lower()
        expr = str(p.get("expr") or "").strip().lower()
        err_class = str(p.get("error_class") or "eval_error").strip().lower()
        return f"{name}:{err_class}:{expr[:80]}"

    @staticmethod
    def _refresh_probe_error_fields(report: dict[str, Any]) -> None:
        probes = list(report.get("probes") or [])
        report["probe_errors"] = [
            f"{p.get('name','?')}: {p.get('err','')[:160]}"
            for p in probes
            if not p.get("ok") and p.get("err")
        ]
        report["probe_eval_errors"] = [
            {
                "name": p.get("name", "probe"),
                "expr_preview": (p.get("expr") or "")[:120],
                "error_class": p.get("error_class") or "eval_error",
                "err": (p.get("err") or "")[:200],
            }
            for p in probes
            if not p.get("ok") and p.get("kind") == "eval_error"
        ]

    @staticmethod
    def _remove_probe_warnings(report: dict[str, Any], names: set[str]) -> None:
        if not names:
            return
        kept: list[str] = []
        for w in list(report.get("soft_warnings") or []):
            drop = False
            for name in names:
                if (
                    f"PROBE FAILED [{name}]" in w
                    or f"PROBE BROKEN [{name}]" in w
                ):
                    drop = True
                    break
            if not drop:
                kept.append(w)
        report["soft_warnings"] = kept

    # Max iters the all-probes-quarantined gate blocks a clean ship before it
    # downgrades to advisory (so a model that cannot emit a parseable probe
    # still ships on the harness's own input/canvas/error checks, not loops).
    _ALL_PROBES_QUARANTINED_GATE_CAP = 2

    # Max iters the PARTIAL-quarantine gate (serial10 chess game 5) blocks a
    # clean ship when SOME probes survive but a behavioral probe was
    # syntax-quarantined on a recipe-matched game. Bounded so a model that
    # cannot author a parseable replacement does not loop forever.
    # Max iters the partial-quarantine gate blocks a clean ship after a
    # syntax-quarantined probe on a recipe-matched game. Cap=1 (was 2): with
    # tune batches at max_iters=3, two holds left only one iter for real
    # gameplay fixes (run_16 Holochess: 6/6 probes green on iter 2 but still
    # ok=False solely from this gate; local prompt_tokens ~30k). One repair
    # chance is enough; then ship on surviving probes + harness checks.
    _PARTIAL_QUARANTINE_GATE_CAP = 1

    # Name of the single harness-authored self-heal probe (C2). Distinct so it
    # is easy to recognise, reconcile, and never confuse with a model probe.
    _OUTLINE_STATE_PROBE_NAME = "auto_outline_state_present"

    @staticmethod
    def _first_state_field(state_contract: str) -> str | None:
        """Parse the first FIELD name from an outline `state:` contract.

        Contracts look like 'state={player:{x,y,vx,vy},entities[],score,over}
        on window' — we want the first property INSIDE the object ('player'),
        not the variable name ('state'). Falls back to the first identifier in
        a brace-free contract ('score; lives; over' -> 'score'). Returns None
        when there is nothing parseable."""
        import re as _re
        if not state_contract:
            return None
        # Prefer the body of the first {...} object so we read a real field.
        m = _re.search(r"\{(.*)", state_contract, _re.S)
        inner = m.group(1) if m else state_contract
        fm = _re.search(r"[A-Za-z_$][A-Za-z0-9_$]*", inner)
        return fm.group(0) if fm else None

    def _maybe_inject_outline_state_probe(self, iteration: int) -> str | None:
        """Self-heal (C2): when ALL model probes were quarantined, author ONE
        probe from the matched outline's `state:` contract so the build keeps
        at least one behavioral self-check instead of shipping with zero. The
        probe reads a field the outline says the game SHOULD expose
        (`window.state.<field>`). It is advisory (never gates — see
        `_reconcile_harness_authored_probes`) and is dropped automatically once
        the model emits its own probe. Returns the probe name, or None when
        there is no outline state contract to draw from."""
        if self._probes is None:
            self._probes = []
        if any(p.get("name") == self._OUTLINE_STATE_PROBE_NAME for p in self._probes):
            return None
        recipes = getattr(self, "_active_opening_book_recipes", None) or []
        state_str = ""
        for row in recipes:
            if isinstance(row, dict) and row.get("kind") == "outline":
                state_str = str((row.get("recipe") or {}).get("state") or "")
                break
        field = self._first_state_field(state_str)
        if not field:
            return None
        expr = (
            "(()=>{const s=window.state||window.game||window.gameState||{};"
            f"return ('{field}' in s);}})()"
        )
        self._probes.append({
            "name": self._OUTLINE_STATE_PROBE_NAME,
            "expr": expr,
            "harness_authored": True,
            "advisory": True,
        })
        self._trace({
            "kind": "outline_state_probe_injected",
            "iteration": iteration,
            "field": field,
        })
        return self._OUTLINE_STATE_PROBE_NAME

    def _reconcile_harness_authored_probes(self, report: dict[str, Any]) -> None:
        """Keep the C2 self-heal probe advisory and temporary.

        - Once the model supplies its OWN probe again, drop the harness probe
          from both `self._probes` and the report ("until the model replaces
          it").
        - While only the harness probe is present, never let it gate: a falsy
          result is moved out of the gating `soft_warnings` channel into the
          advisory `warnings` channel. A passing result is left as-is so the
          build still has one real green behavioral check.
        """
        probes = list(report.get("probes") or [])
        harness = [p for p in probes if p.get("harness_authored")]
        if not harness:
            return
        names = {str(p.get("name") or "") for p in harness}
        model_probes = [p for p in probes if not p.get("harness_authored")]
        if model_probes:
            # Model authored real probes again — remove the placeholder.
            self._probes = [
                p for p in (self._probes or []) if not p.get("harness_authored")
            ]
            report["probes"] = model_probes
            self._remove_probe_warnings(report, names)
            self._refresh_probe_error_fields(report)
            return
        # Only the harness probe is present: never gate on it.
        failing = [p for p in harness if not p.get("ok")]
        if failing:
            self._remove_probe_warnings(report, names)
            warns = list(report.get("warnings") or [])
            warns.append(
                "Harness-authored state-contract probe is failing (advisory): "
                "the game does not yet expose the field the matched outline "
                "lists in its state contract. Re-emit <probes> reading the real "
                "window.state fields your code assigns."
            )
            report["warnings"] = warns
            self._refresh_probe_error_fields(report)

    def _handle_probe_eval_errors(self, report: dict[str, Any], iteration: int) -> None:
        """Trace, soften, and quarantine probes that fail at eval time.

        Falsy probes are still game failures. Eval-error probes are broken
        probe expressions; after two consecutive eval errors for the same
        name, drop the probe from future iterations so it stops drowning the
        model in stale harness errors.
        """
        probes = list(report.get("probes") or [])
        # C2 reconcile: keep any harness-authored self-heal probe advisory, and
        # drop it the moment the model re-emits its own probe (may mutate
        # report["probes"], so re-read below).
        self._reconcile_harness_authored_probes(report)
        probes = list(report.get("probes") or [])
        if not probes:
            return

        failing = [p for p in probes if not p.get("ok")]
        eval_failing = [
            p for p in failing
            if p.get("kind") == "eval_error"
        ]

        for p in probes:
            name = str(p.get("name") or "probe")
            if p.get("ok"):
                self._probe_names_ever_passed.add(name)
                self._probe_eval_error_streak[name] = 0
                self._probe_eval_error_shape_streak[self._probe_shape_key(p)] = 0
                continue
            if p.get("kind") == "eval_error":
                shape = self._probe_shape_key(p)
                self._probe_eval_error_streak[name] = (
                    self._probe_eval_error_streak.get(name, 0) + 1
                )
                self._probe_eval_error_shape_streak[shape] = (
                    self._probe_eval_error_shape_streak.get(shape, 0) + 1
                )
                self._trace({
                    "kind": "probe_eval_error",
                    "iteration": iteration,
                    "name": name,
                    "expr_preview": (p.get("expr") or "")[:120],
                    "error_class": p.get("error_class") or "eval_error",
                    "streak": self._probe_eval_error_streak[name],
                    "shape_streak": self._probe_eval_error_shape_streak[shape],
                })
            else:
                # A real falsy result breaks the eval-error streak.
                self._probe_eval_error_streak[name] = 0

        # Two-strike rule for general eval errors, ONE-STRIKE for
        # SyntaxError. Syntax errors mean the probe expression itself
        # is malformed JavaScript — re-evaluating it next iter cannot
        # produce a different result; the only way it would "pass" is
        # if the model re-emits a corrected probe. Holding the broken
        # probe in the active set for a second iter just lets the
        # syntax error mask other harness signals (the 2026-05-23
        # chess trace's iter 2/3 chased a `new Promise(r=>{...})`
        # parser error for two iters while the actual
        # ASSETS_LOADED_BUT_UNDRAWN problem was hidden behind it).
        # Model-agnostic: any model that emits unparseable JS triggers
        # the fast-path; legitimate runtime-eval errors still get the
        # tolerant 2-strike treatment.
        def _is_syntax_error(p: dict) -> bool:
            err_class = (p.get("error_class") or "").lower()
            err_text = (p.get("err") or "").lower()
            return (
                "syntaxerror" in err_class
                or "syntaxerror" in err_text
                or "unexpected token" in err_text
                or "unexpected identifier" in err_text
                or "unterminated" in err_text
            )
        quarantine_names: set[str] = set()
        for p in eval_failing:
            name = str(p.get("name") or "probe")
            streak = self._probe_eval_error_streak.get(name, 0)
            if streak >= 2 or _is_syntax_error(p):
                quarantine_names.add(name)
        if quarantine_names:
            for p in eval_failing:
                name = str(p.get("name") or "probe")
                if name not in quarantine_names:
                    continue
                is_syntax = _is_syntax_error(p)
                self._trace({
                    "kind": "probe_quarantined",
                    "iteration": iteration,
                    "name": name,
                    "expr_preview": (p.get("expr") or "")[:120],
                    "eval_error_class": p.get("error_class") or "eval_error",
                    "streak": self._probe_eval_error_streak.get(name, 0),
                    "fast_path": "syntax_error" if is_syntax else None,
                })
                if is_syntax:
                    notice = (
                        f"PROBE {name} QUARANTINED IMMEDIATELY: the "
                        "expression is unparseable JavaScript (syntax "
                        "error). Re-emit <probes>...</probes> with a "
                        "corrected expression — re-running malformed "
                        "JS cannot produce a different result and "
                        "would mask other harness signals."
                    )
                else:
                    notice = (
                        f"PROBE {name} QUARANTINED: had eval errors twice; "
                        "re-emit <probes>...</probes> to replace, or leave it "
                        "dropped if it was not testing real behavior."
                    )
                if notice not in self._pending_probe_quarantine_notices:
                    self._pending_probe_quarantine_notices.append(notice)

            self._probes = [
                p for p in self._probes
                if str(p.get("name") or "probe") not in quarantine_names
            ]
            report["probes"] = [
                p for p in probes
                if str(p.get("name") or "probe") not in quarantine_names
            ]
            self._remove_probe_warnings(report, quarantine_names)
            warnings = list(report.get("warnings") or [])
            warnings.append(
                "Probe quarantine: dropped "
                f"{len(quarantine_names)} probe(s) after repeated eval-time "
                "errors; re-emit <probes> to replace if needed."
            )
            report["warnings"] = warnings
            self._refresh_probe_error_fields(report)

        # Guard: if quarantine emptied the ENTIRE model-authored probe set this
        # iter, the build would otherwise ship "clean" with zero behavioral
        # self-checks (battlezone 20260622 iter 4: 7/7 syntax-quarantined ->
        # probes_total:0, ok:true). Gate the clean ship and ask for one valid
        # probe, but only for a bounded number of iters so a model that cannot
        # author a parseable probe does not loop forever — after the cap we
        # ship on the harness's own input/canvas/error checks (advisory only).
        if quarantine_names and not report.get("probes"):
            if (
                self._all_probes_quarantined_gate_used
                < self._ALL_PROBES_QUARANTINED_GATE_CAP
            ):
                self._all_probes_quarantined_gate_used += 1
                sw = list(report.get("soft_warnings") or [])
                sw.append(
                    "ALL acceptance probes were quarantined as malformed/broken "
                    "— the build has zero behavioral self-checks. Re-emit "
                    "<probes>...</probes> with at least one valid expression "
                    "that reads real game state (e.g. window.state fields your "
                    "code actually assigns) before this can ship clean."
                )
                report["soft_warnings"] = sw
                self._trace({
                    "kind": "all_probes_quarantined_gate",
                    "iteration": iteration,
                    "used": self._all_probes_quarantined_gate_used,
                    "cap": self._ALL_PROBES_QUARANTINED_GATE_CAP,
                })
                # C2: author one advisory probe from the matched outline's
                # state contract so the NEXT iteration has at least one real
                # behavioral self-check instead of zero.
                self._maybe_inject_outline_state_probe(iteration)
            else:
                warns = list(report.get("warnings") or [])
                warns.append(
                    "All acceptance probes remain quarantined after repeated "
                    "attempts; shipping on harness behavioral checks "
                    "(input/canvas/errors). Advisory only — no model probe "
                    "verified this build."
                )
                report["warnings"] = warns
                self._trace({
                    "kind": "all_probes_quarantined_advisory",
                    "iteration": iteration,
                })

        # Partial-quarantine gate (serial10 chess game 5): even when SOME
        # probes survive, if a model-authored probe was quarantined for a
        # SYNTAX error on a recipe-matched game, a behavioral self-check was
        # silently lost. The surviving probes (canvas_present, state_exposed,
        # …) can report a clean 5/6 pass that enables <done/>/<confirm_done/>
        # while the move-commit gate is dead. Hold the clean ship for a
        # BOUNDED number of iters so the model must re-emit a valid
        # replacement probe. Scoped to recipe-matched games (a known genre
        # with behavioral expectations) so novel/unmatched games are
        # untouched; strong models rarely emit syntax errors so this seldom
        # triggers for them = no penalty.
        if quarantine_names and report.get("probes"):
            syntax_quarantined = any(
                _is_syntax_error(p)
                for p in eval_failing
                if str(p.get("name") or "probe") in quarantine_names
            )
            recipe_matched = bool(
                getattr(self, "_active_visual_playtest_recipe_id", None)
            )
            if (
                syntax_quarantined
                and recipe_matched
                and self._partial_quarantine_gate_used
                < self._PARTIAL_QUARANTINE_GATE_CAP
            ):
                self._partial_quarantine_gate_used += 1
                sw = list(report.get("soft_warnings") or [])
                sw.append(
                    "A behavioral acceptance probe was quarantined as "
                    "malformed (syntax error) on a recipe-matched game, so a "
                    "core mechanic is no longer verified. Re-emit "
                    "<probes>...</probes> with a corrected expression that "
                    "reads real window.state fields before this can ship clean."
                )
                report["soft_warnings"] = sw
                self._trace({
                    "kind": "partial_quarantine_gate",
                    "iteration": iteration,
                    "used": self._partial_quarantine_gate_used,
                    "cap": self._PARTIAL_QUARANTINE_GATE_CAP,
                    "quarantined": sorted(quarantine_names),
                })

        # C1: before a probe is quarantined, avoid blocking on eval-error-only
        # probes when the rest of the harness says the game is healthy.
        remaining = list(report.get("probes") or [])
        remaining_failing = [p for p in remaining if not p.get("ok")]
        if remaining_failing and all(
            p.get("kind") == "eval_error" for p in remaining_failing
        ):
            input_test = report.get("input_test") or {}
            canvas = report.get("canvas") or {}
            healthy_page = not (report.get("page_errors") or report.get("console_errors"))
            healthy_harness = (
                input_test.get("ran") is True
                and input_test.get("any_change") is True
                and canvas.get("blank") is False
                and healthy_page
            )
            proven_probe_bug = all(
                (
                    str(p.get("name") or "probe") in self._probe_names_ever_passed
                    or self._probe_eval_error_shape_streak.get(self._probe_shape_key(p), 0) >= 2
                )
                for p in remaining_failing
            )
            if healthy_harness and proven_probe_bug:
                softened = {str(p.get("name") or "probe") for p in remaining_failing}
                for p in remaining_failing:
                    p["ok"] = True
                    p["downgraded"] = (
                        "probe eval error softened: input/canvas/page checks "
                        "were healthy and this probe shape repeatedly failed "
                        "before evaluating"
                    )
                self._remove_probe_warnings(report, softened)
                warnings = list(report.get("warnings") or [])
                warnings.append(
                    "Probe eval errors softened: all remaining probe failures "
                    "were eval-time errors while input/canvas/page checks were healthy."
                )
                report["warnings"] = warnings
                self._refresh_probe_error_fields(report)

        report["ok"] = (
            len(report.get("errors") or []) == 0
            and len(report.get("soft_warnings") or []) == 0
        )
