"""research.py — open-domain Wikipedia reference fetcher.

Why this exists: a 30B local model has thin world knowledge. When the
user asks for "Missile Command" it cheerfully builds Space Invaders with
the labels swapped, because its memory of the game is a few token
fragments — not a description of the actual mechanics. The trace in
games/traces/game-of-misile-command-good-gr_20260505_133453.* is the
canonical example.

Fix: before planning, look the goal up on Wikipedia. If we find a page
the user genuinely named, prepend a <reference> block to the planning
turn so the model's <plan> + <criteria> are grounded in the real
mechanics, not the model's prior.

Open-domain by design — no game list. We let Wikipedia's `opensearch`
endpoint pick the best title for the user's free-form goal, then
ACCEPT that title only if its words appear (case-insensitive, with
single-edit fuzzy match for typos like "misile") inside the goal. So:

  - "missile command, good graphics" → matches "Missile Command" → ✓
  - "make a calculator"              → matches "Calculator"      → ✓
  - "bunnies vs robots"              → matches "Bunny" or "Robot"
                                       at single-token level only,
                                       which our threshold rejects   → None

That filter catches the noise (random Wikipedia article on "Bunny"
isn't useful for a game) without ever hardcoding "is this a known
game" — which is exactly what feedback_no_genre_list.md forbids.

Public API:

    research.fetch(goal: str) -> str | None
        Returns a complete <reference>...</reference> block ready to
        prepend to the planning user-turn, or None if no good match.
"""

from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Optional


# Live-debug 2026-05-19 (build-a-complete-playable-2d-r trace): every
# research.fetch call returned None on representative goals because
# Python 3.12's framework build ships an empty CA bundle path
# (/Library/Frameworks/Python.framework/Versions/3.12/etc/openssl/cert.pem
# is missing on the user's machine), so every HTTPS handshake to
# en.wikipedia.org failed with CERTIFICATE_VERIFY_FAILED. _http_json
# silently swallowed the SSLError, returned None, and the caller saw
# zero opensearch hits across "asteroids", "pacman", "tetris",
# "donkey kong", "space invaders", etc. Use certifi's bundled CA
# certs when available — same approach the requests/httpx libraries
# take. Falls back to the platform default if certifi isn't installed
# (the agent's venv installs it transitively via diffusers).
def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL_CONTEXT = _build_ssl_context()

_TIMEOUT = 8.0
_MAX_REFERENCE_CHARS = 1800
# Wikipedia silently returns empty results (HTTP 200, but `[]` titles)
# when anonymous requests come in too fast. Live-debug 2026-05-19: 0.6s
# spacing tripped the rate limiter after ~5 calls. Empirical floor that
# survives a 26-query test burst: 1.5s. Costs at most a few extra
# seconds on a single-game lookup; sessions only run research once.
# If you see "MISS" on titles that you know exist (Joust, Defender,
# Mr. Do!, etc.), bump this to 2.0s — Wikimedia's anonymous quota
# fluctuates by region and time of day.
_REQUEST_SPACING_S = 1.5
_LAST_REQUEST_AT: list[float] = [0.0]
# A real UA is required by Wikimedia; a generic one gets 403'd.
_USER_AGENT = "Agent-Learning/1.0 (local-llm coding harness)"
_WP_API = "https://en.wikipedia.org/w/api.php"
_WP_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"

_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Words we strip from the FRONT and BACK of each goal segment before
# handing it to Wikipedia's opensearch. Why: opensearch is title-prefix
# biased — "game of misile command" returns nothing, but "misile command"
# returns "Missile Command". We only strip from the edges so a real game
# title like "Space Invaders" or "Legend of Zelda" stays intact in the
# middle. Conservative list — when in doubt, leave a word in.
_FILLER_WORDS = frozenset({
    "a", "an", "the",
    "of", "for", "with", "without", "and", "or", "to",
    "in", "on", "at", "by", "from", "about",
    "make", "makes", "build", "create", "code", "write", "do", "draw",
    "game", "games", "version", "clone", "remake",
    "like", "as", "kind", "sort", "type",
    "original", "classic", "old", "new", "retro",
    "arcade", "console",
    "please", "just", "i", "me", "my",
    "can", "could", "you", "we",
    "good", "bad", "great", "nice", "cool", "fun", "best",
    "plain", "simple", "basic", "fancy",
    "hello", "hi",
    # Decorative qualifiers that aren't part of a title.
    "graphics", "sound", "sounds", "audio", "music",
    "animation", "animations", "animated",
    "color", "colors", "colour", "colours", "colored", "coloured",
    "fast", "slow",
    "mode", "style",
})

_PUNCT_SPLIT_RE = re.compile(r"[,.;:!?\n]+")
_GAMEISH_DESC_WORDS = (
    "video game", "arcade game", "computer game", "puzzle game",
    "platform game", "shoot 'em up", "shooter", "platformer",
)


def _norm(text: str) -> str:
    """Lowercase, replace non-alnum with spaces, collapse runs."""
    return " ".join(_ALNUM_RE.sub(" ", (text or "").lower()).split())


def _http_json(url: str, *, timeout: float = _TIMEOUT):
    """Best-effort GET → parsed JSON. Returns None on any failure.

    Wikimedia rate-limits anonymous requests; we throttle to one request
    per ~250 ms (still imperceptible end-to-end). Set a real UA so we
    don't get blanket-403'd.
    """
    delay = _REQUEST_SPACING_S - (time.monotonic() - _LAST_REQUEST_AT[0])
    if delay > 0:
        time.sleep(delay)
    _LAST_REQUEST_AT[0] = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(
            req, timeout=timeout, context=_SSL_CONTEXT,
        ) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _search_queries(goal: str) -> list[str]:
    """Produce a deduped list of opensearch queries to try, in priority
    order. Wikipedia's opensearch matches against page titles starting
    at the beginning, so we have to feed it queries that LOOK like
    titles — leading filler words ("game of", "make a") kill matches.

    Strategy:
      - Try the whole goal (covers the easy "asteroids" case).
      - Split on punctuation; for each segment, strip leading and
        trailing filler words, then submit the residue.
      - Also submit prefix slices of length 2-3 from each cleaned
        segment, in case the title is just the start of a longer
        descriptive phrase.
    """
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        if not q or q.lower() in seen:
            return
        seen.add(q.lower())
        queries.append(q)

    add(goal)
    for segment in _PUNCT_SPLIT_RE.split(goal):
        toks = [w for w in segment.split() if w]
        while toks and toks[0].lower().strip("'\"`-") in _FILLER_WORDS:
            toks.pop(0)
        while toks and toks[-1].lower().strip("'\"`-") in _FILLER_WORDS:
            toks.pop()
        if not toks:
            continue
        add(" ".join(toks))
        if len(toks) > 3:
            add(" ".join(toks[:3]))
        if len(toks) > 2:
            add(" ".join(toks[:2]))
    return queries


def _opensearch(query: str, *, timeout: float = _TIMEOUT) -> list[str]:
    """Fuzzy-match titles for `query`. Returns up to 5 page titles.

    Wikipedia's opensearch handles typos and partial matches, so
    `misile command` → ["Missile Command", ...] without us doing
    anything clever. We rely on this aggressively.
    """
    qs = urllib.parse.urlencode({
        "action": "opensearch",
        "search": query,
        "limit": "5",
        "namespace": "0",
        "format": "json",
        "redirects": "resolve",
    })
    data = _http_json(f"{_WP_API}?{qs}", timeout=timeout)
    if not isinstance(data, list) or len(data) < 2:
        return []
    titles = data[1] if isinstance(data[1], list) else []
    return [t for t in titles if isinstance(t, str) and t]


def _summary(title: str, *, timeout: float = _TIMEOUT) -> dict | None:
    """REST summary endpoint — gives us extract + description.

    Follows Wikipedia redirects automatically (e.g. "Pacman" → "Pac-Man").
    """
    url = _WP_REST + urllib.parse.quote(title, safe="")
    data = _http_json(url, timeout=timeout)
    return data if isinstance(data, dict) else None


# ---- gameplay-section fetching -------------------------------------------

# Section headings we want to surface verbatim — these are where Wikipedia
# describes how a game actually works, beyond the lead paragraph. We try
# them in order; first hit wins.
_GAMEPLAY_SECTIONS = (
    "gameplay",
    "game play",
    "mechanics",
    "rules",
    "how to play",
    "objective",
)


def _gameplay_section(title: str, *, timeout: float = _TIMEOUT) -> str:
    """Return Wikipedia's 'Gameplay' (or similar) section as cleaned text.

    Two API hops: list sections → fetch the matching section's wikitext →
    strip wikitext markup. Returns "" if no gameplay-class section exists
    (common for non-game pages — "Calculator" has no "Gameplay" section,
    so we just return the lead summary).
    """
    qs = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "sections",
        "format": "json",
        "redirects": "1",
    })
    data = _http_json(f"{_WP_API}?{qs}", timeout=timeout)
    if not isinstance(data, dict):
        return ""
    sections = ((data.get("parse") or {}).get("sections")) or []
    target_idx: str | None = None
    for s in sections:
        line = (s.get("line") or "").strip().lower()
        if line in _GAMEPLAY_SECTIONS:
            target_idx = str(s.get("index") or "")
            break
    if not target_idx:
        return ""

    qs2 = urllib.parse.urlencode({
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "section": target_idx,
        "format": "json",
        "redirects": "1",
    })
    d2 = _http_json(f"{_WP_API}?{qs2}", timeout=timeout)
    if not isinstance(d2, dict):
        return ""
    wt = ((d2.get("parse") or {}).get("wikitext") or {}).get("*", "")
    if not isinstance(wt, str):
        return ""
    return _strip_wikitext(wt)


# Wikitext is a soup of references, templates, links, files, and HTML
# comments. We don't need a full parser — just enough cleanup to make the
# text readable as prose for the LLM.
_REF_RE = re.compile(r"<ref[^>]*?>.*?</ref>|<ref[^/]*?/>", re.DOTALL | re.IGNORECASE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_FILE_RE = re.compile(r"\[\[(?:File|Image):[^\[\]]*?(?:\[\[[^\[\]]*\]\][^\[\]]*?)*\]\]", re.IGNORECASE)
_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_BOLD_RE = re.compile(r"'''(.*?)'''", re.DOTALL)
_ITALIC_RE = re.compile(r"''(.*?)''", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)


def _strip_wikitext(wt: str) -> str:
    s = _REF_RE.sub("", wt)
    s = _HTML_COMMENT_RE.sub("", s)
    # Templates can nest; loop until stable (bounded so a pathological
    # page can't hang us).
    for _ in range(5):
        new = _TEMPLATE_RE.sub("", s)
        if new == s:
            break
        s = new
    s = _FILE_RE.sub("", s)
    # [[Target|Display]] → Display ; [[Target]] → Target
    s = _LINK_RE.sub(lambda m: m.group(2) or m.group(1), s)
    s = _BOLD_RE.sub(r"\1", s)
    s = _ITALIC_RE.sub(r"\1", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _HEADING_RE.sub(r"\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ---- title-vs-goal fuzzy gate -------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein. Used only on short tokens (≤ ~20 chars), so
    O(n*m) is fine and we avoid pulling in a third-party dep.
    """
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _close_token(title_tok: str, goal_tok: str) -> bool:
    """Are these two tokens 'the same word, modulo a typo'?"""
    if title_tok == goal_tok:
        return True
    if abs(len(title_tok) - len(goal_tok)) > 2:
        return False
    threshold = 1 if max(len(title_tok), len(goal_tok)) < 6 else 2
    return _edit_distance(title_tok, goal_tok) <= threshold


def _title_in_goal(title: str, goal: str) -> bool:
    """Filter: does the goal genuinely name this Wikipedia title?

    Strips disambiguators ("Foo (video game)" → "Foo"), then tokenizes
    both sides and requires ≥ 60% of title tokens (length ≥ 3) to have
    a near-match in goal tokens. Single-token titles must match
    exactly-or-typo.

    Without this filter, opensearch happily returns "Bunny" for any
    prompt mentioning bunnies, and we'd inject a useless reference.
    """
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    t = _norm(bare)
    g = _norm(goal)
    if not t or not g:
        return False
    # Plain substring (handles "asteroids" in "asteroids game").
    if t in g:
        return True
    # Squashed substring (handles "Pac-Man" / title "pac man" matching
    # goal "pacman" — users routinely concatenate hyphenated names).
    t_squash = t.replace(" ", "")
    g_squash = g.replace(" ", "")
    if len(t_squash) >= 5 and t_squash in g_squash:
        return True
    title_tokens = [tok for tok in t.split() if len(tok) >= 3]
    if not title_tokens:
        return False
    goal_tokens = g.split()
    matched = 0
    for tt in title_tokens:
        for gt in goal_tokens:
            if _close_token(tt, gt):
                matched += 1
                break
    return matched / len(title_tokens) >= 0.6


# ---- public entry point --------------------------------------------------

def fetch(
    goal: str,
    *,
    timeout: float = _TIMEOUT,
    max_chars: int = _MAX_REFERENCE_CHARS,
) -> Optional[str]:
    """Fetch a Wikipedia reference for `goal`. Returns a <reference>
    block ready to prepend to the planning turn, or None.

    Strategy:
      1. opensearch(goal) → up to 5 candidate titles.
      2. Also try opensearch(goal + " video game") and "+ arcade game"
         to bias toward gaming pages when the bare query is ambiguous.
      3. Filter candidates with _title_in_goal — open-domain check;
         no hardcoded list of games.
      4. For the first survivor, fetch summary + Gameplay section.
      5. Render as a compact <reference> block, capped at max_chars.

    Failure modes — all silent, all return None:
      - network down / Wikipedia 5xx
      - no candidate matches the goal
      - summary endpoint missing extract
    The agent treats None as "no reference, just plan from priors".
    """
    if not goal or not goal.strip():
        return None

    # Iteration strategy designed to MINIMIZE Wikipedia calls (it
    # aggressively rate-limits anonymous bursts):
    #
    #   For each query variant we generate (in priority order — the goal
    #   itself first, then leading/trailing-filler-stripped segments):
    #     try opensearch with no suffix; if any candidate passes the
    #     goal-name filter AND its summary looks like a game, RETURN.
    #     Otherwise stash any non-game match as a fallback and try
    #     adding " video game" / " arcade game" suffixes.
    #
    # In the common case (a named game) we stop after 1–2 opensearch
    # calls plus 1 summary fetch. In the worst case we exhaust all
    # variants and either return the best non-gameish fallback or None.
    base_queries = _search_queries(goal)
    seen_titles: set[str] = set()
    fallback: tuple[str, dict] | None = None  # best non-gameish match seen

    def _is_gameish(title: str, s: dict) -> bool:
        title_l = title.lower()
        desc_l = (s.get("description") or "").lower()
        extract_l = (s.get("extract") or "").lower()[:300]
        if "(video game)" in title_l or "(arcade" in title_l:
            return True
        if any(w in desc_l for w in _GAMEISH_DESC_WORDS):
            return True
        if any(w in extract_l for w in _GAMEISH_DESC_WORDS):
            return True
        return False

    chosen: tuple[str, dict] | None = None
    for base in base_queries:
        if chosen is not None:
            break
        for suffix in ("", " video game", " arcade game"):
            q = (base + suffix).strip()
            for title in _opensearch(q, timeout=timeout):
                key = title.lower()
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                if not _title_in_goal(title, goal):
                    continue
                s = _summary(title, timeout=timeout)
                if not (s and isinstance(s.get("extract"), str) and s["extract"].strip()):
                    continue
                if _is_gameish(title, s):
                    chosen = (title, s)
                    break
                if fallback is None:
                    fallback = (title, s)
            if chosen is not None:
                break

    if chosen is None:
        chosen = fallback
    if chosen is None:
        return None

    title, summary_obj = chosen
    extract = (summary_obj.get("extract") or "").strip()
    description = (summary_obj.get("description") or "").strip()
    canonical_title = (summary_obj.get("title") or title).strip()
    page_url = (
        ((summary_obj.get("content_urls") or {}).get("desktop") or {}).get("page")
        or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
    )

    gameplay = _gameplay_section(title, timeout=timeout)

    parts: list[str] = [f"TITLE: {canonical_title}"]
    if description:
        parts.append(f"DESCRIPTION: {description}")
    parts.append(f"SOURCE: {page_url}")
    if extract:
        parts.append(f"SUMMARY:\n{extract}")
    if gameplay:
        parts.append(f"GAMEPLAY (from Wikipedia):\n{gameplay}")
    body = "\n\n".join(parts)

    if len(body) > max_chars:
        # Truncate at a sentence/paragraph boundary so we don't cut
        # mid-word.
        cut = body[:max_chars]
        # Prefer the last paragraph break, fall back to last newline.
        for sep in ("\n\n", "\n", ". "):
            i = cut.rfind(sep)
            if i > max_chars * 0.6:
                cut = cut[:i]
                break
        body = cut.rstrip() + "\n[truncated]"

    return f"<reference source=\"wikipedia\">\n{body}\n</reference>"


# ---- CLI for quick manual testing ----------------------------------------

def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python research.py <goal text...>", flush=True)
        return 2
    goal = " ".join(argv[1:])
    block = fetch(goal)
    if block is None:
        print("[no reference matched]", flush=True)
        return 1
    print(block, flush=True)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
