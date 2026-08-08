# Model and effort delegation

Notes for choosing a model and effort level when delegating work on this project. This is guidance
for the *human* at delegation time — an agent cannot act on it, because its model and effort are
already fixed by the time it reads anything. The decisions themselves live in
`.claude/agents/*.md` frontmatter, where they take effect.

Last checked against the docs: 2026-08-08.

---

## The one rule worth memorising

From Anthropic's guidance on choosing a model and effort level:

- Raise **effort** when the model didn't *try* hard enough — skipped a file, didn't run the tests,
  didn't double-check its work.
- Raise **model** when it had all the context, clearly tried, and still got it wrong.

Effort is a thoroughness dial. Model is a capability dial. Reaching for the wrong one is the most
common way to spend money without improving the result.

---

## Defaults

| Model | Levels | Default |
|---|---|---|
| Fable 5, Opus 5, Sonnet 5, Opus 4.8 | low, medium, high, xhigh, max | `high` |
| Opus 4.7 | low, medium, high, xhigh, max | `xhigh` |
| Opus 4.6, Sonnet 4.6 | low, medium, high, max | `high` |
| Haiku 4.5 | *none — no effort setting at all* | n/a |

Two things that quietly invalidate most advice found online:

- `high` is *exactly* equivalent to omitting the parameter.
- **Effort scales are calibrated per model.** The same level name does not represent the same
  underlying value across models, so "use xhigh" is meaningless without naming the model.

---

## Two claims worth correcting

**"xhigh is the sweet spot for Opus."** True for Opus 4.7 and 4.8 — it was near-verbatim official
guidance, and 4.7 defaults to `xhigh`. **Not** true for Opus 5, whose documentation says start at
`high`, use `low`/`medium` liberally as the primary cost control, and explicitly warns against
carrying effort settings over from an earlier model without re-sweeping.

Practitioner reports converge with that: Opus 5 tends to over-scope and over-engineer at high and
xhigh. Anthropic's own Opus 5 prompting guide devotes sections to task-scope control and
over-verification. **Turning effort down is a real move on this model** in a way it was not on 4.7.

**"Fable is only worth choosing over Opus at xhigh/max."** No support found, and the official
numbers point the other way — at *max* effort Opus 5 lands within 0.5% of Fable 5's peak on
CursorBench at half the cost per task. Max is where Opus closes the gap, not where Fable pulls away.
Fable's own guidance recommends `medium`/`low` for routine work and notes its lower-effort settings
often exceed `xhigh` on prior models.

**Fable's edge is task shape, not task difficulty:** long-horizon autonomy, multi-day goal-directed
runs, ambiguous root-cause investigation, and dispatching parallel subagents.

---

## How that maps onto this project

| Agent | Model | Effort | Why |
|---|---|---|---|
| `physics-kernel` | opus | xhigh | Bounded and well-specified, but capability-sensitive. The tight per-feature contract guards against the scope creep Opus 5 otherwise shows at this level. |
| `numerics-debug` | fable | high | Ambiguous root-cause investigation is Fable's documented strength, and "plausible but wrong" is this project's defining failure mode. |
| `validation-harness` | sonnet | high | Well-precedented work landing against a harness that catches errors mechanically. |

The orchestrating session is a separate decision. Orchestration is mostly sequencing and integration
— the architecture plan already carries the decomposition — so it does not need peak capability.
On Opus 5, `medium` is a reasonable orchestration default with escalation when a task earns it.

**Spend metered credits on task shape, not on difficulty.** A bounded novel kernel with a citation
and a validation case is well-specified; Opus handles it for less. Reserve Fable for the arena
refactor and for "the trajectory is plausibly wrong and I don't know why".

---

## What is not established

No rigorous published comparison exists for model × effort × task-type. The orchestrator patterns
circulating in blog posts ("Opus plans, Sonnet implements, Sonnet reviews") have no measurement
behind them. Anthropic's own answer is to run an effort sweep on your own evals.

Treat the table above as a starting point to be measured, not a result.

Be wary of secondary sources on this topic — a large share of search results recycle the docs with
verifiable errors (omitting `xhigh` entirely, or claiming it ranks above `max`). Prefer the primary
docs below.

---

## Sources

- [Effort — Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Model configuration — Claude Code Docs](https://code.claude.com/docs/en/model-config)
- [Choosing a Claude model and effort level in Claude Code](https://claude.com/blog/claude-model-and-effort-level-in-claude-code)
- [Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)
- [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)
- [CodeRabbit: Opus 5 effort A/B for code review](https://www.coderabbit.ai/blog/opus-5-model-review)
