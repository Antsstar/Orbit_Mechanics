#!/usr/bin/env python3
"""
Export the flashcard decks to a tab-separated file Anki can import.

    python docs/flashcards/export.py                 # write orbital_engine_cards.tsv
    python docs/flashcards/export.py --check         # validate only, no output
    python docs/flashcards/export.py -o cards.tsv    # choose the output path

**Importing into Anki.** File -> Import, choose the .tsv, set the note type to Basic, field
separator to Tab, and tick "Allow HTML in fields". Map column 1 to Front, 2 to Back, 3 to Tags.
Anki keys on the first field, so re-importing after we add or reword cards *updates* the existing
note rather than duplicating it -- provided the question text is unchanged. Reword a question and
you get a new card; that is usually what you want, since a reworded question is a different prompt.

**Source format.** Each card is a level-3 heading (the question), the answer beneath it, and a
trailing blockquote of metadata:

    ### Why do `parent_indices` and `body_sys_map` disagree for the Moon?

    Because they answer different questions...

    > src: docs/architecture.md - Why the two graphs
    > sym: Simulation.calc_global
    > tags: architecture, graphs

`src` points at the prose that justifies the answer, so a card can be traced rather than trusted.
`sym` is optional and names a code symbol the card depends on; `--check` verifies every named symbol
still exists in `src/`, which is what stops the deck rotting silently as the engine changes.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DECK_DIR = Path(__file__).parent
REPO_ROOT = DECK_DIR.parent.parent
# Cards legitimately cite test helpers (invariants.py) as well as engine symbols, so both trees
# are scanned. Anything outside them is prose and cannot be checked mechanically.
CODE_DIRS = [REPO_ROOT / "src", REPO_ROOT / "tests"]

CARD_RE = re.compile(r"^### (?P<q>.+?)$", re.MULTILINE)
META_RE = re.compile(r"^> (?P<key>src|sym|tags): (?P<val>.+?)$", re.MULTILINE)


@dataclass
class Card:
    question: str
    answer: str
    deck: str
    source: str = ""
    symbols: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    origin: str = ""


def _to_html(markdown: str) -> str:
    """
    Minimal markdown -> HTML for Anki. Deliberately small: inline code, bold, italic, line breaks.

    Escaping runs first so that answer text containing < or & survives; the inline patterns are then
    applied to the escaped string, which is safe because none of them can match escaped entities.
    """
    text = html.escape(markdown.strip())
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    return text.replace("\n\n", "<br><br>").replace("\n", " ")


def parse_deck(path: Path) -> List[Card]:
    raw = path.read_text(encoding="utf-8")
    deck = re.search(r"^# (.+?)$", raw, re.MULTILINE)
    deck_name = deck.group(1).strip() if deck else path.stem

    cards: List[Card] = []
    matches = list(CARD_RE.finditer(raw))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[m.end():end]

        meta = {mm.group("key"): mm.group("val").strip() for mm in META_RE.finditer(body)}
        answer = META_RE.sub("", body).strip()

        cards.append(Card(
            question=m.group("q").strip(),
            answer=answer,
            deck=deck_name,
            source=meta.get("src", ""),
            symbols=[s.strip() for s in meta.get("sym", "").split(",") if s.strip()],
            tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
            origin=f"{path.name}:{raw[:m.start()].count(chr(10)) + 1}",
        ))
    return cards


def load_all() -> List[Card]:
    cards: List[Card] = []
    for path in sorted(DECK_DIR.glob("[0-9]*.md")):
        cards.extend(parse_deck(path))
    return cards


def check(cards: List[Card]) -> List[str]:
    """
    Validate the deck. Returns a list of problems; empty means clean.

    The symbol check is the one that matters. A card explaining why `calc_global` reads
    `body_sys_map` is wrong the moment that function is renamed, and nothing else in the repo would
    notice -- prose does not fail a test suite.
    """
    problems: List[str] = []
    seen: dict[str, str] = {}

    src_blob = "\n".join(
        f.read_text(encoding="utf-8")
        for d in CODE_DIRS if d.exists()
        for f in d.rglob("*.py")
    )

    for c in cards:
        where = f"{c.origin} :: {c.question[:60]}"
        if not c.answer:
            problems.append(f"{where} -- empty answer")
        if not c.source:
            problems.append(f"{where} -- no `src:` line; a card should be traceable")
        if not c.tags:
            problems.append(f"{where} -- no `tags:` line")
        if c.question in seen:
            problems.append(f"{where} -- duplicate question, also at {seen[c.question]}")
        seen[c.question] = c.origin

        for sym in c.symbols:
            leaf = sym.split(".")[-1]
            if leaf and leaf not in src_blob:
                problems.append(f"{where} -- cites `{sym}`, not found in src/ or tests/ any more")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(REPO_ROOT / "orbital_engine_cards.tsv"))
    ap.add_argument("--check", action="store_true", help="validate only; write nothing")
    args = ap.parse_args()

    cards = load_all()
    if not cards:
        print("no cards found", file=sys.stderr)
        return 1

    problems = check(cards)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)

    by_deck: dict[str, int] = {}
    for c in cards:
        by_deck[c.deck] = by_deck.get(c.deck, 0) + 1
    for deck, n in by_deck.items():
        print(f"  {n:>3}  {deck}")
    print(f"  {len(cards):>3}  TOTAL")

    if problems:
        print(f"\n{len(problems)} problem(s)", file=sys.stderr)
        return 1
    if args.check:
        print("\ndeck is clean")
        return 0

    lines = []
    for c in cards:
        tags = " ".join(t.replace(" ", "-") for t in c.tags)
        back = _to_html(c.answer)
        if c.source:
            back += f'<br><br><span style="opacity:.55;font-size:.85em">{html.escape(c.source)}</span>'
        lines.append("\t".join([_to_html(c.question), back, tags]))

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {len(cards)} cards -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
