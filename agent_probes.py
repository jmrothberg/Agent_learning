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

import json
import re
from typing import Any

from agent_helpers import _PROBES_RE, _strip_thinking


class ProbeHandlingMixin:
    """Probe-result post-processing for GameAgent (see module docstring)."""

    # Properties whose presence/size is established at init, NOT at
    # runtime — comparisons against them aren't dynamic even when they
    # use `>` / `<`. `state.barrels.length > 0` is "we spawned a
    # barrel", which the game does on iter 1; `c.toDataURL().length >
    # 200` is "the canvas has rendered SOMETHING", true the moment the
    # HUD draws. Excluding these closes the DK trace 20260514_104131
    # false-negative (5/5 'structural-with-floor' probes scored as
    # 40% dynamic, nudge stayed silent on a game where gameplay never
    # advanced).
    _STRUCTURAL_NUMERIC_PROPS = (
        "length", "size", "width", "height", "bytelength",
        "innerwidth", "innerheight", "clientwidth", "clientheight",
        "offsetwidth", "offsetheight",
    )
    _CMP_RE = re.compile(r"(\S+?)\s*([<>]=?)\s*(-?\d+)")
    _CMP_REV_RE = re.compile(r"(-?\d+)\s*([<>]=?)\s*(\S+)")
    # Probes that explicitly try to capture a time-delta — IIFE or
    # ternary patterns that store a `t0` / `before` snapshot and
    # compare later. Genuine dynamic checks; only fire when paired with
    # an explicit time signal (await/setTimeout/etc.) so an `x !== 0`
    # at init doesn't false-positive.

    @staticmethod
    def _is_dynamic_probe(expr: str) -> bool:
        """True if `expr` verifies behavior over time, not just existence.

        Heuristic (no LLM call — this is a regex problem, not a
        reasoning problem; weak models can't reliably self-critique
        their own probe list per TACL 2024 / arXiv 2411.17501).

        Dynamic signals (each is sufficient):
          - awaits or returns a Promise (`await`, `.then(`, `Promise.`)
          - references a timer / clock (`setTimeout`, `setInterval`,
            `requestAnimationFrame`, `performance.now`, `Date.now`)
          - reads canvas state via getImageData (the resulting pixel
            array is genuine runtime state, not a `.length` check)
          - compares against a NON-ZERO numeric threshold AND the LHS
            isn't a known structural property (.length / .size /
            .width / .height etc.)

        Structural-only (returns False):
          - `!!window.state`, `typeof X === 'object'`
          - `X.length > 0`, `X.length > 200`, `width > 0` (existence
            with a floor; true at init for any rendered game)
          - `state.player.x > 0` — at init most games have positive
            starting coordinates; "non-zero" doesn't mean "moved"
          - bare delta-marker identifiers (`prev`, `t0`, `lastFire`)
            — too many false positives like `lastFireTime > 0` which
            just means "fired at init"

        DK trace 20260514_104131 pin: the five probes (`canvas_exists`,
        `player_state`, `barrels_active`, `score_visible`,
        `game_not_blank`) ALL classify as structural under this rule,
        so the nudge fires.
        """
        if not expr:
            return False
        e = expr.strip()
        low = e.lower()
        # 1. Awaits / promises — only way a probe can actually wait.
        if "await " in e or ".then(" in e or "promise." in low:
            return True
        # 2. Timers / clocks — the probe is taking a time-delta.
        for tok in (
            "settimeout", "setinterval", "requestanimationframe",
            "performance.now", "date.now",
        ):
            if tok in low:
                return True
        # 3. Canvas pixel read — getImageData returns runtime state.
        # Exclude `getImageData(...).data.length` (still a presence
        # check on the returned typed array).
        if "getimagedata" in low and ".data.length" not in low:
            return True
        # 4. Numeric comparison against a non-trivial threshold (|N| >=
        # 1) where the LHS isn't a structural-presence property.
        def _is_structural_lhs(lhs: str) -> bool:
            low_lhs = lhs.lower()
            for prop in ProbeHandlingMixin._STRUCTURAL_NUMERIC_PROPS:
                if "." + prop in low_lhs:
                    return True
            return False

        for m in ProbeHandlingMixin._CMP_RE.finditer(e):
            lhs = m.group(1)
            threshold = int(m.group(3))
            if threshold == 0:
                continue
            if _is_structural_lhs(lhs):
                continue
            return True
        for m in ProbeHandlingMixin._CMP_REV_RE.finditer(e):
            rhs = m.group(3)
            threshold = int(m.group(1))
            if threshold == 0:
                continue
            if _is_structural_lhs(rhs):
                continue
            return True
        return False

    @staticmethod
    def _classify_probes_dynamic(probes: list[dict]) -> dict:
        """Bulk-classify probes into structural vs dynamic.

        Returns {"dynamic": [names], "structural": [names], "ratio": float}
        where ratio is dynamic_count / total (0.0 when probes is empty).
        """
        if not probes:
            return {"dynamic": [], "structural": [], "ratio": 0.0}
        dyn: list[str] = []
        struct: list[str] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if ProbeHandlingMixin._is_dynamic_probe(expr):
                dyn.append(name)
            else:
                struct.append(name)
        total = len(dyn) + len(struct)
        return {
            "dynamic": dyn,
            "structural": struct,
            "ratio": (len(dyn) / total) if total else 0.0,
        }

    # Pattern 1: probe binds a local `const x0 = …` (or `let`/`var`), then
    # immediately returns a literal `true`/`false` without ever reading
    # x0 again. The temp binding is wasted; the probe is tautological.
    # Donkey-kong trace 20260516_142445 iter 1 `mario_moves`:
    #   `(()=>{const x0=s.player.x; setTimeout(()=>{}, 100); return true;})()`
    # Pattern 2: probe asserts `typeof obj.NAME === 'undefined'` (or the
    # short form `obj.NAME === undefined`) and returns false on that
    # path, but `obj.NAME` is never assigned anywhere in the game body.
    # That check is permanently true → probe always returns false →
    # zero signal. Donkey-kong trace 20260516_124628 iter 1 `barrels_move`:
    #   `if (typeof b.x0 === 'undefined') return false; …`  (b.x0 is
    #   never assigned).
    # The undefined-property check needs the on-disk HTML, so it runs
    # later — at materialize time, not at probe-parse time. The
    # tautological-temp check is purely structural and runs immediately.
    # Match: `const NAME = …;` followed (eventually) by `return <literal>`.
    # The `.*?` is lazy so we don't span over an outer return that
    # belongs to a different function body. We accept newlines (DOTALL)
    # and braces inside (e.g. an inner `setTimeout(()=>{})`) because
    # the "is the temp dead?" check below uses `expr.count(temp_name)`
    # to confirm the binding is actually unused.
    _PROBE_TAUTOLOGY_RE = re.compile(
        r"(?:const|let|var)\s+(\w+)\s*=\s*[^;]+;"
        r".*?return\s+(?:true|false|0|1)\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    # MK trace 20260517_220025 fix. Iter 2 probe:
    #   (()=>{try{const s=window.state; if(!s)return false;
    #          return s.cpuPunchFlipped===true;}catch(e){return false;}})()
    # was bait — the SAME iter's patch added `cpuPunchFlipped: true`
    # to state.reset(), so the probe trivially passed without
    # verifying the actual draw transform. Detect this shape so the
    # next user-turn fix prompt surfaces the bait. The probe-side
    # pattern matches any `<ident>.<prop> === true|false` shape (the
    # probe in the MK trace used `s.X` where `s` was a local alias
    # for `window.state`). False-positive risk is bounded by the
    # second-pass check that `<prop>` was also assigned to a literal
    # boolean by the same iter's patch.
    _PROBE_BAIT_PROP_RE = re.compile(
        r"\b\w+\.(\w+)\s*===\s*(?:true|false)\b",
        re.IGNORECASE,
    )
    # Inline constant assignment matchers — both object-literal
    # (`cpuPunchFlipped: true`) and statement (`state.X = true`,
    # `window.X = true`) forms. Conservative on purpose: we only
    # match LITERAL true/false to flag the exact bait shape.
    _PROBE_BAIT_OBJLIT_RE = re.compile(
        r"\b(\w+)\s*:\s*(?:true|false)\b",
        re.IGNORECASE,
    )
    _PROBE_BAIT_ASSIGN_RE = re.compile(
        r"(?:state|window|self|globalThis)\.(\w+)\s*=\s*(?:true|false)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _probes_baited_by_patches(
        probes: list[dict], applied_replaces: list[str],
    ) -> list[dict]:
        """Detect probes whose only assertion is `state.X === true|false`
        when the SAME iter's applied patch also writes that exact flag
        as a literal constant. Returns the same `{name, kind, message}`
        shape as `_lint_probes`. MK trace 20260517_220025 iter 2 is
        the case study: a probe checked `state.cpuPunchFlipped===true`
        while the patch added `cpuPunchFlipped: true` to reset() — the
        probe passed without actually testing the rendered flip.

        Conservative: requires LITERAL true/false on both sides, and
        only flags when the same property name appears in BOTH the
        probe expr and a REPLACE constant assignment. False positives
        would noisily lint legitimate flag checks; false negatives
        keep the existing behavior.
        """
        if not probes or not applied_replaces:
            return []
        # Build the set of property names assigned to a literal
        # boolean by the applied patches.
        assigned_to_bool: set[str] = set()
        for replace_text in applied_replaces:
            if not isinstance(replace_text, str) or not replace_text:
                continue
            for m in ProbeHandlingMixin._PROBE_BAIT_OBJLIT_RE.finditer(replace_text):
                assigned_to_bool.add(m.group(1))
            for m in ProbeHandlingMixin._PROBE_BAIT_ASSIGN_RE.finditer(replace_text):
                assigned_to_bool.add(m.group(1))
        if not assigned_to_bool:
            return []
        findings: list[dict] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            for m in ProbeHandlingMixin._PROBE_BAIT_PROP_RE.finditer(expr):
                prop = m.group(1)
                if prop in assigned_to_bool:
                    findings.append({
                        "name": name,
                        "kind": "probe_bait_flag",
                        "message": (
                            f"probe `{name}` checks "
                            f"`{prop} === true|false` but the same "
                            f"iter's patch sets `{prop}` to a literal "
                            f"boolean — the probe passes without "
                            f"verifying the actual behavior. Replace "
                            f"the flag check with a draw-path / DOM / "
                            f"canvas-state assertion that fails when "
                            f"the visual change is missing."
                        ),
                    })
                    break
        return findings

    @staticmethod
    def _lint_probes(probes: list[dict]) -> list[dict]:
        """Return a list of `{name, kind, message}` lint findings for
        probes that pass JSON parse but are structurally tautological.

        Catches the donkey-kong 20260516_142445 `mario_moves` shape —
        the probe binds a `const x0` then returns `true` without ever
        comparing against x0. The setTimeout callback is empty, so the
        probe provides no behavioral signal.
        """
        findings: list[dict] = []
        for p in probes:
            name = str(p.get("name", "?"))
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            m = ProbeHandlingMixin._PROBE_TAUTOLOGY_RE.search(expr)
            if m:
                temp_name = m.group(1)
                # Confirm the temp is referenced ONLY in its declaration:
                # split on the temp name; if there's only one occurrence
                # left after stripping the decl prefix, the temp is dead.
                if expr.count(temp_name) <= 1:
                    findings.append({
                        "name": name,
                        "kind": "tautological_constant_return",
                        "message": (
                            f"probe `{name}` binds `{temp_name}` but "
                            f"never reads it again before returning a "
                            f"constant — the probe always returns the "
                            f"same value regardless of game behavior."
                        ),
                    })
            if re.search(r"\bstate\.scene\s*>=\s*1\b", expr):
                findings.append({
                    "name": name,
                    "kind": "fragile_initial_scene_index",
                    "message": (
                        f"probe `{name}` requires `state.scene >= 1`; "
                        "most scene arrays are zero-indexed and start at "
                        "scene 0, so this probe can force a bad state hack. "
                        "Prefer `typeof state.scene === 'number'` and test "
                        "progression after dispatching input."
                    ),
                })
            # Initial-state-equality trap (trace 20260613_213711, Opus 4.8):
            # a probe asserting `state.<counter> === 0` samples AFTER the
            # harness input smoke test has already fired keys — in a QTE /
            # reaction game those keys legitimately advance the counter, so
            # the probe reads e.g. room 2 and fails. The probe is frozen once
            # accepted, so the model can only fix it by contorting gameplay
            # (it added a `started` gate that swallowed real input for 4
            # iters). Catch it at Phase A — before the probe freezes — unless
            # the expr resets first so it samples a fresh initial state.
            counter_eq0 = re.search(
                r"\bstate\.(room|scene|level|stage)\s*===?\s*0\b", expr
            )
            if counter_eq0 and "reset(" not in expr:
                counter = counter_eq0.group(1)
                findings.append({
                    "name": name,
                    "kind": "fragile_initial_state_equality",
                    "message": (
                        f"probe `{name}` asserts `state.{counter} === 0`, "
                        "but the harness input smoke test fires control keys "
                        "BEFORE probes run — in a timed/QTE game those keys "
                        "advance the counter, so this probe fails on a working "
                        "game and cannot be changed once accepted. Either call "
                        "`window.game.reset()` at the START of the probe so it "
                        f"samples a fresh state, or assert `typeof state.{counter} "
                        "=== 'number'` plus a progression check after dispatching "
                        "input."
                    ),
                })
            # run_16 bullet-hell: `const b0=state.bullets.length; await 1500ms;
            # return length>b0` fails at steady-state (94 live bullets, cull =
            # spawn). Require length>=N or a totalSpawned/fired counter instead.
            if (
                "setTimeout" in expr
                and ".length" in expr
                and re.search(
                    r"(?:const|let|var)\s+(\w+)\s*=\s*[^;]*\.length\b",
                    expr,
                )
                and re.search(
                    r"\.length\s*>\s*\w+|\w+\s*<\s*[^;]*\.length",
                    expr,
                )
            ):
                findings.append({
                    "name": name,
                    "kind": "fragile_length_growth_probe",
                    "message": (
                        f"probe `{name}` captures an array `.length`, waits, then "
                        "requires growth. Culled arrays (bullets/particles) often "
                        "stay flat at steady-state even when spawning correctly. "
                        "Prefer `state.bullets.length >= 8` or assert a "
                        "`totalSpawned`/`fired` counter increases."
                    ),
                })
        return findings

    @staticmethod
    def _lint_probe_syntax(probes: list[dict]) -> list[dict]:
        """Catch unparseable probe expr at Phase A (run_vlm10 OutRun).

        Model-authored async probes with a missing `)` burn iter 1 before
        Chromium ever runs — syntax-check here and re-prompt the plan once.

        run_13: also catch bracket imbalance before the node check (Solitaire /
        SimCity truncated probes) — works even when node is unavailable.
        """
        import subprocess

        from tools import _bracket_imbalance

        findings: list[dict] = []
        for p in probes:
            name = str(p.get("name") or "?")
            expr = str(p.get("expr") or "").strip()
            if not expr:
                continue
            # Fast path: unbalanced (), [], {} — same micro-probe helper.
            imb = _bracket_imbalance(expr)
            bad_br = {k: v for k, v in imb.items() if v != 0}
            if bad_br:
                findings.append({
                    "name": name,
                    "kind": "syntax_error",
                    "message": (
                        f"probe `{name}` has unbalanced brackets "
                        f"{bad_br} — re-emit a complete balanced expr"
                    ),
                })
                continue
            js = (
                "try { new Function(" + json.dumps(expr) + "); } "
                "catch (e) { console.error(String(e.message||e)); "
                "process.exit(1); }"
            )
            try:
                r = subprocess.run(
                    ["node", "-e", js],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
            except FileNotFoundError:
                return findings
            except Exception as e:
                findings.append({
                    "name": name,
                    "kind": "syntax_check_error",
                    "message": f"probe `{name}` syntax check failed: {e}",
                })
                continue
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "syntax error").strip()[:200]
                findings.append({
                    "name": name,
                    "kind": "syntax_error",
                    "message": (
                        f"probe `{name}` has a JavaScript syntax error: "
                        f"{err}"
                    ),
                })
        return findings

    @staticmethod
    def _probes_referencing_unassigned_props(
        probes: list[dict], html: str,
    ) -> list[dict]:
        """For each probe, find `obj.PROP` accesses where PROP is never
        assigned anywhere in `html`. Returns the same `{name, kind,
        message}` shape as `_lint_probes`.

        Catches the donkey-kong 20260516_124628 `barrels_move` shape:
        the probe gates on `b.x0` but the game code never sets `b.x0`,
        so the probe trivially returns false. The check is permissive —
        only properties that look like real game-state names (≥ 2 chars,
        not a reserved word, not a built-in property) are checked, to
        avoid flagging legitimate property reads on DOM / built-in
        objects.
        """
        if not probes or not html:
            return []
        # Cheap negation: skip if the HTML body is too small to contain
        # any real game state. The check below assumes a parseable script.
        if "<script" not in html.lower():
            return []
        findings: list[dict] = []
        # Properties to never flag: DOM, canvas, common built-ins. If a
        # probe accesses `c.width` and the HTML never says `c.width = X`,
        # that's fine — `width` is a DOM property of the canvas element.
        _IGNORE = {
            "length", "size", "width", "height", "x", "y", "left", "top",
            "right", "bottom", "data", "value", "textContent",
            "innerText", "innerHTML", "style", "className", "id", "name",
            "parent", "child", "next", "prev", "node", "type", "kind",
        }
        # Extract candidate property *accesses* (not method *calls*).
        # Negative lookahead `(?!\s*\()` skips `obj.method(args)` —
        # methods aren't assignable state, so flagging them as
        # "unassigned" produces false positives (e.g. `obj.toString`,
        # `document.querySelector`). Method-call hallucinations are
        # caught by the separate API-allowlist micro-probe in tools.py.
        prop_re = re.compile(
            r"\b\w+\.([A-Za-z_$][\w$]*)\b(?!\s*\()"
        )
        # Build the assignment set ONCE per HTML — assignments look like
        # `obj.prop =`, `obj.prop +=`, `obj.prop:` (object literal),
        # `prop:` inside object literals at all depths.
        # Conservatively, scan for both patterns.
        assigned: set[str] = set()
        for m in re.finditer(
            r"\.([A-Za-z_$][\w$]*)\s*(?:=|\+=|-=|\*=|/=)(?!=)",
            html,
        ):
            assigned.add(m.group(1))
        # Object-literal entries: `name:` at line start or after `,` /
        # `{` / `(`. This is a loose match (catches some non-assignments
        # like ternary labels) which is FINE — false negatives (skipping
        # a real warning) are cheaper than false positives.
        for m in re.finditer(
            r"(?:[,{(\n]\s*|^\s*)([A-Za-z_$][\w$]*)\s*:",
            html,
        ):
            assigned.add(m.group(1))
        for p in probes:
            name = str(p.get("name", "?"))
            # Recipe-injected auto_* probes are alias-tolerant by design
            # (e.g. s.floors||s.ground, return true when absent) — the
            # model cannot fix them; skip to avoid misleading fix coaching.
            if name.startswith("auto_"):
                continue
            expr = str(p.get("expr", ""))
            if not expr:
                continue
            missing: set[str] = set()
            for m in prop_re.finditer(expr):
                prop = m.group(1)
                if prop in _IGNORE or prop.startswith("_"):
                    continue
                # The probe accesses obj.prop; if `prop` is never
                # assigned anywhere in the file, flag it.
                if prop not in assigned:
                    missing.add(prop)
            if missing:
                sample = ", ".join(f"`{p}`" for p in sorted(missing)[:3])
                findings.append({
                    "name": name,
                    "kind": "unassigned_property_read",
                    "message": (
                        f"probe `{name}` reads {sample} but the game "
                        f"code never assigns those properties — the "
                        f"probe will trivially fail (or short-circuit "
                        f"return) regardless of game behavior. Prefer "
                        f"fixing the probe to read real runtime state; "
                        f"do not add a constant pass-flag assignment "
                        f"unless it is genuinely updated by gameplay."
                    ),
                })
        return findings

    @staticmethod
    def _extract_probes(reply: str) -> list[dict]:
        """Pull a JSON list-of-{name,expr} out of <probes>...</probes>.

        Tolerant: accepts either a JSON list of objects, or a list of
        plain strings (treated as `expr`, with name auto-assigned). Bad
        JSON returns []; the agent shouldn't ever crash on a probe parse
        failure since universal probes still cover the basics.
        """
        reply = _strip_thinking(reply)
        m = _PROBES_RE.search(reply)
        if not m:
            return []
        body = m.group(1).strip()
        # Strip a fenced ```json``` if present.
        body = re.sub(r"^```(?:json|JSON)?\s*\n", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
        try:
            obj = json.loads(body)
        except Exception:
            return []
        out: list[dict] = []
        if isinstance(obj, list):
            for i, item in enumerate(obj, 1):
                if isinstance(item, dict) and item.get("expr"):
                    out.append({
                        "name": str(item.get("name") or f"probe_{i}")[:60],
                        "expr": str(item["expr"])[:600],
                    })
                elif isinstance(item, str):
                    out.append({"name": f"probe_{i}", "expr": item[:600]})
        return out[:8]   # cap so a chatty model can't bloat the verifier

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
