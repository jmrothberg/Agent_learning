"""Corrective, idempotent pass: fold the QTE "visual telegraph" concept into
the EXISTING outline-cutscene-qte recipe fields (the deep-render budget caps
order<=7, tuning<=3, probes<=3, traps<=140 chars, section<=2600). Adds the
concept by editing existing lines rather than appending new ones. Safe to
re-run; only touches outline-cutscene-qte.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "memory/implementation_outlines.jsonl"
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

for r in recs:
    if r.get("id") != "outline-cutscene-qte":
        continue
    rec = r["recipe"]

    # content: keep terse telegraph clause (replace any earlier long version)
    c = r["content"]
    long_marker = "TELEGRAPH the timing visually: during cueMs"
    if long_marker in c:
        i = c.index(" TELEGRAPH the timing visually")
        j = c.index("only timing signal.", i) + len("only timing signal.")
        c = c[:i] + c[j:]
    terse = (
        " Telegraph timing visually: windup during cueMs, then move the threat "
        "along hazardPath to the threatened part by hitMs (pointOnPath) so timing "
        "reads from the picture; the word only says WHICH key."
    )
    if "Telegraph timing visually" not in c:
        anchor = "rect backgrounds are fallback only."
        c = c.replace(anchor, anchor + terse)
    r["content"] = c

    # order: drop the appended telegraph step, fold it into frame cycling (keep <=7)
    rec["order"] = [
        o for o in rec["order"]
        if not o.startswith("telegraph: windup during cueMs")
    ]
    rec["order"] = [
        ("cycle generated frames ~7 FPS; move threat along hazardPath to the "
         "threatened part by hitMs (pointOnPath) so it visibly closes in")
        if o == "cycle generated frames at ~7 FPS" else o
        for o in rec["order"]
    ]

    # traps: replace the over-long telegraph trap with a <=140-char version
    rec["traps"] = [
        t for t in rec["traps"]
        if not t.startswith("no visible threat motion")
    ]
    tele_trap = ("threat sits static until the keypress: move it along hazardPath "
                 "toward the threatened part across cue->hit so timing reads from the picture")
    assert len(tele_trap) <= 140, len(tele_trap)
    if tele_trap not in rec["traps"]:
        rec["traps"].append(tele_trap)

    # tuning: drop appended line, fold windup timing into existing lines (keep <=3)
    rec["tuning"] = [
        u for u in rec["tuning"]
        if not u.startswith("cue/windup 300-700 ms before the window; threat reaches")
    ]
    rec["tuning"] = [
        "frame ~140 ms (7 FPS); cue/windup 300-700 ms before the window"
        if u == "frame ~140 ms (7 FPS)" else u
        for u in rec["tuning"]
    ]
    rec["tuning"] = [
        "qte window 800-1800 ms shrinking; 3 lives; threat reaches threatened part at hitMs"
        if u == "qte window 800-1800 ms shrinking; 3 lives" else u
        for u in rec["tuning"]
    ]

    # probes: drop appended line, fold visible-approach check into the draw probe (keep <=3)
    rec["probes"] = [
        pr for pr in rec["probes"]
        if not pr.startswith("hazard/threat draw position changes between cue and hit")
    ]
    rec["probes"] = [
        ("at least one generated bg/sprite is drawn with drawImage; threat draw "
         "position changes between cue and hit (visible approach)")
        if pr == "at least one generated bg/sprite is drawn with drawImage" else pr
        for pr in rec["probes"]
    ]

p.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
    encoding="utf-8",
)
print("done")
