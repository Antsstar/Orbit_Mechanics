# Model and effort delegation

Notes for choosing a model and effort level when delegating work on this project, and for invoking
agents so the choice actually takes effect. This is guidance for the *human or orchestrator* at
delegation time — an agent cannot act on it, because its model and effort are already fixed by the
time it reads anything. The decisions themselves live in `.claude/agents/*.md` frontmatter.

**Last checked against the primary docs: 2026-09-16.** Model guidance goes stale within weeks. Before
relying on anything here, re-check the sources at the bottom; see the engineering log for what
happened the last time this was trusted without checking.

---

## The one rule worth memorising

- Raise **effort** when the model didn't *try* hard enough — skipped a file, didn't run the tests.
- Raise **model** when it had all the context, clearly tried, and still got it wrong.

Effort is a thoroughness dial. Model is a capability dial. Reaching for the wrong one is the most
common way to spend money without improving the result.

---

## Current models

| Model | ID | Input / Output per MTok | Cache read | Default effort |
|---|---|---|---|---|
| Fable 5.1 | `claude-fable-5-1` | $10 / $50 | **$0.25** (2.5% of input) | `high` |
| Opus 5 | `claude-opus-5` | $5 / $25 | $0.50 (10%) | `high` |
| Sonnet 5 | `claude-sonnet-5` | $2 / $10 | $0.20 (10%) | `high` |
| Haiku 4.5 | `claude-haiku-4-5` | $1 / $5 | $0.10 (10%) | *no effort setting* |

All three current large models accept `low`, `medium`, `high`, `xhigh`, `max`. Effort scales are
calibrated per model, so "use xhigh" means nothing without naming the model.

### What the cache-read price means for this project

Long agentic sessions are dominated by cache reads: every turn re-reads the growing conversation. On
Fable 5.1 those cost $0.25/MTok — **half** of Opus 5's $0.50. So on a long, heavily cached coding
session, Fable 5.1's premium over Opus 5 sits almost entirely in *output* tokens and uncached input,
not in the headline 2× input price. For long-horizon agentic work the real cost gap is much smaller
than the sticker suggests. For short, uncached, output-heavy tasks it is the full 2×.

---

## Invoking agents in Claude Code so the choice takes effect

Verified against the Claude Code subagent docs:

1. **Pin full model IDs in frontmatter** (`model: claude-fable-5-1`), not aliases. An alias resolves to
   whatever the client currently maps it to; an ID does not move.
2. **Do not pass a per-invocation `model` when spawning a pinned agent.** Resolution order is:
   per-invocation `model` → frontmatter `model` → `CLAUDE_CODE_SUBAGENT_MODEL` → main conversation.
   The per-invocation parameter only accepts aliases, so passing `model: "fable"` silently overrides a
   precise pin with an alias.
3. **Set `effort` in frontmatter.** It overrides the session effort for that agent only.
4. **New agent files may not register until the session restarts.** If a `subagent_type` is reported
   as not found, restart rather than falling back to `general-purpose` with an inlined prompt — the
   fallback loses the frontmatter's model and effort pins.
5. **Verify the served model from the transcript, never from the agent's report.** Every "Fable" agent
   in September 2026 reported `claude-fable-5-1` while every one of its API responses came from
   `claude-sonnet-5`. The authoritative record is the `"model"` field of the assistant messages in
   `~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`:
   `grep -o '"model":"[^"]*"' agent-<id>.jsonl | sort | uniq -c`. Check it after the agent's first few
   turns, not after it finishes. See the engineering log entry "Every Fable agent actually ran on
   Sonnet 5".
6. **Parallel agents in worktrees must set `PYTHONPATH`.** The package is an editable install of the
   main repository, so a worktree's bare `pytest` imports the *main* repo's code and reports green on
   changes it never exercised. Run `PYTHONPATH="$(pwd)/src" <env>/python.exe -m pytest` from the
   worktree root.

---

## Per-model prompting deltas that matter here

### Fable 5.1

From *Prompting Claude Fable 5.1*. Existing Fable 5 prompts work, but:

- **Start at `high`, not `xhigh`.** At `xhigh`/`max` it can draft a long deliverable in its thinking and
  then write it out again. Move up only where a quality gain is measured. At `medium` it roughly
  matches Fable 5 at lower cost; at `low` it is often competitive with Opus and Sonnet on cost per task.
- **May issue one tool call per turn** in coding loops where Fable 5 batched. Fix: *"First privately
  list what you need next; then request every item that doesn't depend on another's result in this
  one response."*
- **More likely to rewrite whole files for small changes.** Fix: *"The number of tokens used to edit
  files is best minimized, all else being equal. Therefore, when it will not affect the end result, try
  to surgically edit a file rather than rewrite the entire thing."* This matters for `simulator.py`.
- **Writes fewer progress updates** during long tool runs. Remove any "hold findings for the final
  response" lines; ask for a closing recap that stands on its own.
- **Can stop early** — describing the next step instead of doing it, or asking permission for work
  already requested. The guide's autonomous-operation block fixes this; its opening sentence ("You are
  operating autonomously. The user is not watching in real time…") carries most of the effect and
  should be kept as written.
- **Adds unrequested fixes and extra test files** on open-ended features. The guide's "keep changes and
  tests to what the task asks for" block reduces this with no measured loss of task success.
- **Denser prose.** *"Please remove all mannered prose."* works.

### Opus 5

From *Prompting Claude Opus 5*:

- **Start at `high`; step up to `xhigh` for demanding coding and agentic work.** `low` and `medium` are
  strong and are the primary cost control. Re-sweep rather than carrying effort over from older models.
- **Remove self-verification instructions** ("include a final verification step", "use a subagent to
  verify", "double-check your answer"). Opus 5 already verifies, and these cause over-verification with
  no quality gain.
- **Expands scope** on narrow tasks — use the guide's scope-discipline snippet.
- **Delegates to subagents readily**, which multiplies cost on small tasks. Say when delegation is
  warranted.
- **Longer responses and longer written files** than prior models; effort does not shorten visible
  output, so prompt for length explicitly.

---

## How that maps onto this project

| Agent | Model | Effort | Why |
|---|---|---|---|
| `force-model-architect` | Fable 5.1 | high | The architectural core of the force-model phase. Long-horizon, novel, load-bearing for everything after it — Fable's documented strength. |
| `numerics-debug` | Fable 5.1 | high | Ambiguous root-cause investigation, and "plausible but wrong" is this project's defining failure mode. |
| `physics-kernel` | Opus 5 | high | Bounded and well-specified, from a citation. `xhigh` only for a genuinely demanding model. |
| `kernel-twin` | Opus 5 | high | Fixed four-part job held honest by an equivalence test. |
| `validation-harness` | Sonnet 5 | high | Established patterns landing against a harness that catches errors mechanically. |

**Spend Fable on task shape, not difficulty.** A bounded kernel with a citation and a validation case is
well specified, and Opus handles it for half the price. Reserve Fable for long-horizon design and for
"the trajectory is plausibly wrong and I don't know why".

---

## What is not established

No rigorous published comparison exists for model × effort × task type. Anthropic's own answer is to
run an effort sweep on your own evals. Treat the table above as a starting point to be measured.

Be wary of secondary sources: a large share of search results recycle the docs with verifiable errors.
Prefer the primary docs below — and prefer the model IDs in the orchestrating session's own system
prompt over any cached copy of a model table.

---

## Sources

- [Models overview](https://platform.claude.com/docs/en/models/overview)
- [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview)
- [What's new in Claude Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1)
- [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1)
- [Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Claude Code subagents](https://code.claude.com/docs/en/sub-agents)
