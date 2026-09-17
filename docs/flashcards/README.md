# Flashcards

Spaced-repetition cards covering OrbitalEngine's architecture, the orbital mechanics it implements,
and the reasoning behind decisions that are not recoverable from the code.

## Why these exist

The obvious reason is self-assessment. The less obvious one matters more:

**A card is a specification of what must be understood.** If a design decision cannot be phrased as a
question with a one-breath answer, either the decision is not well understood or it is not well
documented — and both are worth knowing. Writing the cards is therefore a documentation-completeness
check that runs on the *reasoning*, in the same way the test suite runs on the code.

Every card carries a `src:` pointer to the prose that justifies it, so an answer can be traced rather
than trusted. A decision that has no source to point at is a decision that was never written down.

## The decks

| Deck | Cards | Covers |
|---|---:|---|
| `01-architecture.md` | 16 | Design decisions and their reasoning — the "why is it like that" |
| `02-arena-invariants.md` | 12 | The memory arena and the invariants that produce silent wrongness when violated |
| `03-orbital-theory.md` | 14 | The domain itself — answerable without reference to this codebase |
| `04-validation.md` | 15 | Why the test suite is shaped this way; verification vs comparison |
| `05-performance.md` | 12 | What was measured, what it meant, and the reasoning errors it exposed |
| `06-forces-and-propagation.md` | 14 | Force-model composition, RSW, J2, Cowell, and what "fails without the fix" has to mean |
| `07-frontier.md` | 12 | J2 truth, secular J2 and mean seeding, the sweep and frontier plot, time-reversibility, verifying which model ran |

## Using them

```bash
python docs/flashcards/export.py            # -> orbital_engine_cards.tsv
python docs/flashcards/export.py --check    # validate only
```

In Anki: **File → Import**, note type **Basic**, field separator **Tab**, tick **Allow HTML in
fields**. Map column 1 → Front, 2 → Back, 3 → Tags.

Anki keys on the first field, so re-importing after cards are added or reworded **updates** existing
notes rather than duplicating them — provided the question text is unchanged. Rewording a question
produces a new card, which is usually right: a reworded question is a different prompt, and your
scheduling history for the old one no longer means what it did.

## Adding a card

Append to the relevant deck file:

```markdown
### The question, as you would actually ask it

The answer. One breath. Markdown is fine — inline code, bold, italic survive the export.

> src: docs/architecture.md - Section name
> sym: some_function, SomeClass
> tags: architecture, gotcha
```

`src` is required and should point at prose that justifies the answer. `sym` is optional and names
code symbols the card depends on.

## The rot check

`--check` verifies that every symbol named in a `sym:` line still exists somewhere in `src/` or
`tests/`. This is the part that keeps the deck honest as the engine changes.

A card explaining why `calc_global` reads `body_sys_map` becomes wrong the moment that function is
renamed, and **nothing else in the repository would notice** — prose does not fail a test suite. The
check closes that gap for the mechanically checkable part of a card. It caught a wrong citation on
its first run.

It does not, and cannot, verify that an *answer* is still true. That remains a human judgement, which
is the argument for reviewing the deck at the end of each phase rather than only when something
breaks.

## Writing guidance

- **Ask what you would actually ask.** "What is the arena?" tests nothing. "Why does `local_states`
  reference `body_sys_map` rather than `parent_indices`?" tests whether the two-graph split is
  understood.
- **One idea per card.** A card that needs three paragraphs is two or three cards.
- **Prefer the decision over the fact.** "COE column 0 is `p`" is a lookup. "Why `p` and not `a`?"
  is the card — it carries the parabolic-representability reasoning with it.
- **Include the numbers where a number is the point.** The lunar divergence card is only useful if it
  carries the estimate *and* the measurement, because the relationship between them is the lesson.
