# Engineering log

A running record of problems hit on this project and how they were resolved — environment quirks,
traps in the codebase, and mistakes made while working on it (including by AI assistants).

**Purpose.** Anyone picking this project up, human or agent, starts without the context of how these
were found. Checking here first is cheaper than rediscovering them. Entries are written as
*symptom → cause → fix → how to avoid*, so they are searchable by the symptom you actually have.

Add to this when something costs you more than a few minutes. A near-miss is worth recording too.

---

## Environment

### Base conda has neither pytest nor mypy

**Symptom.** `python -m pytest` → `No module named pytest`, despite the suite working elsewhere.

**Cause.** The project environment is `orbital_env`; base conda is a different interpreter.

**Fix.** Always invoke the environment explicitly:

```
C:/Users/antss/miniconda3/envs/orbital_env/python.exe -m pytest -q
C:/Users/antss/miniconda3/envs/orbital_env/python.exe -m mypy src/ --strict
```

---

### `gh` and the SSH host alias

The remote is `git@github.com-pers:Antsstar/Orbit_Mechanics.git`, using an alias defined in
`~/.ssh/config` for two-account separation. This *was expected* to break `gh`'s repo detection.

**It does not.** `gh run list` resolves the alias correctly with no `-R` flag needed. Recorded
because the opposite was assumed, and the assumption would have added a pointless flag everywhere.

Note `gh auth switch` is **global**, not per-repo. After working here, switch back for professional
repos:

```
gh auth switch --user <your-other-account>
```

---

### SSH passphrase handling

Load the key into the agent yourself; never pass a passphrase through a chat transcript, a file, or
an environment variable.

```
ssh-add ~/.ssh/git_personal_ed2
```

`SSH_AUTH_SOCK` is inherited by tool shells, so once the agent holds the key, pushes work without the
secret ever being exposed. `ssh-add -l` shows fingerprints only and is safe to run.

`AddKeysToAgent yes` in `~/.ssh/config` normally auto-prompts on first use, but a non-interactive
tool shell has no TTY, so a push will hang or fail rather than prompting. Pre-load the key.

---

## Traps found in the codebase

### A test that asserted nothing for months

**Symptom.** None. `test_rv_coe_conversions_vectorized_fuzzing` passed on every run since it was
written, and `frames.py` showed 80% coverage.

**Cause.**

```python
mask = (np.linalg.norm(h, axis=-1) < 1e-3)   # selects near-radial orbits
assert np.all(np.abs(r_out[mask] - r_in[mask]) < 1e-7)
```

The generated orbits have `|h|` in the thousands, so the mask selected **0 of 100** every time.
Indexing with an all-false mask yields an empty array, and `np.all()` on an empty array is `True`.

**Fix.** Invert the mask to test valid orbits, and guard that the filter kept something:

```python
assert testable.sum() > 90, f"expected ~100 usable orbits, got {testable.sum()}"
```

**How to avoid.** Any filter-then-assert needs a non-emptiness guard. More generally: **coverage
measures execution, not verification.** This test executed every line it was supposed to and checked
none of the results — coverage was 70% before the fix and 70% after. Mutation testing (`mutmut`,
`cosmic-ray`) is the tool that catches this class; a vacuous test survives every mutation.

---

### In-memory SQLite gives each connection its own database

**Symptom.** Latent — not yet triggered. Would appear as "no such table" from a session that
looked correctly connected.

**Cause.** Every new connection to `sqlite:///:memory:` creates a *fresh, empty* database. The
fixture worked only because there was exactly one session per test.

**Fix.** Pin the engine to a single connection:

```python
create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
```

Added `db_session_factory` for tests needing several genuinely independent databases.

---

### `.gitignore` was ignoring itself

**Symptom.** An edit to `.gitignore` never appeared in `git status`.

**Cause.** Line 1 of the file was `.gitignore`, so it excluded itself and had never been tracked. A
fresh clone got no ignore rules at all.

**Fix.** Removed the self-ignore, committed the file.

**How to avoid.** `git ls-files <path>` returns nothing for an untracked file, and
`git check-ignore -v <path>` names the rule and line that excludes it.

---

### CI gated only `main`, so branch work was never checked

**Symptom.** Four `mypy --strict` errors sat undetected on `feature/vop-propagator`.

**Cause.** `on.push.branches: [main, master]`.

**Fix.** `branches: ['**']` for push; `pull_request` still targets main.

**Note.** GitHub uses the workflow file *at the pushed ref*, so a branch whose commits predate this
change still won't trigger CI until it is rebased onto main.

---

### `python -m orbital_engine.simulator` destroys tracked data

Its `__main__` block calls `seed_test_universe()`, which drops and rewrites the git-tracked
`src/orbital_engine/data/planets.db`. Never run it to "check things work". Build an in-memory session
the way `tests/conftest.py` does.

---

## Mistakes made while working, and their corrections

Recorded honestly, because the correction is the reusable part.

### Trusting stale shell output over git

**What happened.** An `ls` reported that `CLAUDE.md` and `docs/historical/` did not exist, so work
began to create them. They already existed — the output was stale.

**How it surfaced.** `git mv coe_from_rv.py docs/historical/coe_from_rv.py` failed with
`fatal: bad source`, because the rename was already staged in the index.

**Correction.** Stopped and ran `git status --short`, `git ls-files`, and `ls -R docs` before
touching anything further, then verified content rather than recreating it.

**Lesson.** When a git command fails in a way that contradicts your model of the filesystem,
**the git command is right.** Prefer `git status` over `ls` for repository state; it reads the index
rather than a possibly-cached directory listing.

---

### Guessing dependency versions instead of querying

**What happened.** Bumping deprecated GitHub Actions, the versions were written from memory as
`checkout@v5` and `setup-python@v6`.

**Correction.** Queried the authoritative source before committing:

```
gh api repos/actions/checkout/releases/latest --jq .tag_name       # v7.0.1
gh api repos/actions/setup-python/releases/latest --jq .tag_name   # v7.0.0
```

Both were **v7**. The guessed versions may not even have cleared the Node 20 deprecation.

**Lesson.** Version numbers are exactly the sort of fact that feels known and isn't. One API call
beats a plausible recollection.

---

### Asserting an expectation instead of deriving it

**What happened.** A new test asserted that the Earth–Moon barycenter carries Earth's *seeded*
ellipse (`e = 0.0167086`). It failed: the measured value was `0.0166223`.

**Diagnosis.** The engine was right; the expectation was wrong. The EMB sits ~4770 km from Earth,
but the dominant effect is velocity: Earth circles the EMB monthly at ~0.0127 km/s against a
heliocentric 29.8 km/s, so `Δv/v ≈ 4.3e-4`. Since `p = h²/mu`, this predicts
`Δp/p ≈ 2·Δv/v ≈ 8.5e-4`. Measured `8.73e-4`.

**Correction.** The test now asserts the *predicted* bound with a factor of two of headroom, plus a
lower bound so a collapsed barycenter cannot pass trivially.

**Lesson.** The temptation on a failing numerical test is to widen the tolerance until it passes.
That converts a test into a snapshot and destroys its value. Derive the expected magnitude first; if
the derivation matches the observation, the tolerance follows from the derivation. If it doesn't,
you have found a real bug. **A failing test may be the test's fault — establish which before
changing either side.**

---

### Background agents can stop without completing

**What happened.** A subagent was launched and reported back as `stopped` with no result, having been
interrupted.

**Correction.** Relaunched it. Did not fabricate or assume its findings.

**Lesson.** A background agent's completion is not guaranteed. Never report results that have not
actually arrived, and check status before relying on delegated work.

---

## Conventions that emerged

- **Tolerances are budgets, not observations.** Set them from an analytic argument, roughly an order
  or two above expected floating-point noise, with a comment justifying the number. Never loosen one
  to make a failing test pass without recording why.
- **Compare vectors, not magnitudes**, where direction carries meaning. Conserving `|h|` while
  rotating the orbital plane is wrong, and a magnitude check will not notice.
- **Seed randomness explicitly** with `np.random.default_rng(seed)`. `np.random.seed()` mutates
  global state shared between tests, making results depend on execution order, and a clock-derived
  seed makes a failure impossible to reproduce.
- **Parametrise rather than loop.** A failure should name the case that broke, and the remaining
  cases should still run.
