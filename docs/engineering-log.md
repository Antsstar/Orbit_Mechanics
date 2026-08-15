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

### A diverging solver returned NaN and reported success

**Symptom.** None from the solver. `Anomalies.mean_to_eccentric` returned `nan` without raising, and
`ConvergenceError` never fired no matter how badly the iteration diverged.

**Cause.** The convergence test was written as *keep what has demonstrably diverged*:

```python
active = active[np.abs(delta) > tol]      # NaN > tol is False  ->  dropped as "converged"
```

A diverging hyperbolic iterate overflows `sinh` to `inf`, then evaluates `inf - inf = nan`. Every
comparison against NaN is `False`, so the element was removed from the active set as though it had
converged, and the loop exited cleanly with NaN in the output.

**Fix.** Invert the test to *keep what has not demonstrably converged*:

```python
active = active[~(np.abs(delta) <= tol)]  # NaN <= tol is False -> ~False -> stays active
```

The element then survives to the `max_ite` check and raises. Same cost, opposite failure mode.

**How to avoid.** With NaN in play, `keep = ~converged` and `keep = diverged` are **not** the same
predicate, and the difference is exactly whether a numerical failure is loud or silent. Prefer the
negated form in any iterative solver.

This one is worth generalising. NaN defeats assertion-based testing entirely: `nan < tol`,
`nan > tol` and `nan == nan` are all `False`, so a NaN sails through every conserved-quantity check
in `tests/validation/invariants.py` — those assert that a drift is *small*, and NaN is never
measured as large. An explicit `np.all(np.isfinite(...))` is the only thing that catches it, which
is why `test_solver_never_returns_nan_while_reporting_success` exists as its own test rather than
being folded into a tolerance assertion.

**How it was found.** A test written to assert that divergence *raises*. The prediction was right
and the stated mechanism was wrong, which is the useful kind of failing test: it disproved the
mechanism, not just the outcome.

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

### Coverage under numba reports ~15% for a module that is ~88% covered

**Symptom.** `kernels.py` showed 15% coverage despite being exercised by every propagation test in
the suite.

**Cause.** `coverage.py` traces Python bytecode. A function compiled by `@njit` executes as machine
code, so its body is never traced and every line inside it reads as unhit.

**Fix.** Measure coverage with numba disabled, which runs the same kernels as interpreted Python:

```
PYTHONPATH=<dir containing a numba.py that raises ImportError> pytest --cov=orbital_engine
```

`kernels.py` then reports 88% and the project total moves 70% -> 79%.

**How to avoid.** Treat the numba-enabled coverage figure as invalid for any compiled module, not as
a gap to fill. The reverse mistake is worse: chasing that 15% by writing tests for code already
covered wastes effort and adds nothing. Note the interpreted run is also ~4x slower (22s against 5s),
which is another reason it belongs in a separate invocation rather than the default one.

---

### The `np.any()` guard is an anti-optimisation on small arrays

**Symptom.** 37 `np.any()` calls per step, accounting for 21% of the entire step budget - 18855
reduction calls across 500 steps of a *five body* simulation.

**Cause.** The pattern `if np.any(mask): x[mask] = ...` guards a masked operation so no work happens
when the mask is empty. On a large array that is a real saving. On a five-element array the guard
costs ~5 us of Python and NumPy dispatch to avoid ~1.5 us of work, and the masked operation on an
empty selection would have been nearly free anyway.

Worse, two of the guards inside `mean_to_eccentric`'s Newton loop were **loop-invariant** - the same
`np.any(mask_ell)` and `np.any(mask4)` recomputed on every iteration, over quantities that could not
change.

**Fix.** Split the population once, up front, and iterate over contiguous subarrays. Where a loop
needs an active set, hold it as an **integer index array** rather than a boolean mask: the
termination test becomes `active.size`, a Python attribute lookup, instead of `np.any(active)`, a
NumPy reduction.

**How to avoid.** Guards are worth their cost in proportion to the work they skip. Below roughly a
thousand elements, dispatch dominates and the guard is a net loss.

---

### Optimisation order was wrong until it was measured

**Symptom.** None - the plan simply had the phases in the wrong order, and would have delivered a
fraction of the gain for most of the effort.

**Cause.** The plan sequenced arena compaction and tier-slicing first, with compiled kernels last as
polish. Profiling showed `calc_global` - the thing tier-slicing targets - was **26 us of a 630 us
step**, about 4%. The remaining 96% was Python-level NumPy dispatch inside the propagator, which only
compilation removes.

The tell was the scaling measurement: step cost was **flat** from arena capacity 64 to 10000, and 3
bodies cost 542 us against 602 bodies at 3583 us. Fitting that gives ~490 us of fixed per-step
overhead and ~5 us marginal per body. A cost that does not move with problem size is not a data
problem.

**Fix.** Reordered to compile first. Final result 556 -> 3.0 us/step, of which the arena work
contributed a small part and compilation almost all.

**How to avoid.** Re-profile after *every* change, not once at the start. The ordering flipped twice:
after the propagator was compiled, `calc_global` went from 4% to **87%** of the step and genuinely
became the next target; after that, `_record_state` at 600 bodies turned out to cost 1875 us, ten
times the physics step it was recording. None of those three were predictable from the code alone,
and each was invisible until the one before it was fixed.

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

### `mypy --strict` passed locally and failed on Python 3.10 in CI

**Symptom.** Clean locally (3.11), three `[assignment]` errors on the 3.10 CI job:

```
Incompatible types in assignment (expression has type
  "ndarray[tuple[int, ...], dtype[signedinteger[_64Bit]]]", variable has type
  "ndarray[tuple[int],      dtype[signedinteger[_64Bit]]]")
```

**Cause.** An attribute initialised as `self._sib_idx = np.empty(0, dtype=np.int64)` has its type
*inferred* from that first assignment. Some numpy versions type `np.empty(0, ...)` with the narrow
shape `tuple[int]` (definitely 1-D), while `np.flatnonzero(...).astype(...)` returns the general
`tuple[int, ...]`. Reassigning the second to the first is then an error. Which numpy resolves per
interpreter differs, so 3.11 and 3.12 passed and 3.10 did not.

**Fix.** Annotate the attribute with the general type instead of letting it be inferred:

```python
self._sib_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
```

**How to avoid.** Annotate any array attribute whose first assignment is a placeholder — `np.empty`,
`np.zeros(0)`, or similar. The placeholder is the *least* representative value it will ever hold, so
inferring from it is close to guaranteed to be too narrow.

More generally: a green local `mypy` on one interpreter does not predict the matrix. This is the
second time the CI matrix has caught something invisible locally, and it is the argument for gating
every branch rather than only main.

---

### PowerShell here-strings break on embedded double quotes

**Symptom.** `git commit -m @'...'@` failed with `error: pathspec 'Python' did not match any file(s)`.
The commit message had been split into separate git arguments.

**Cause.** The message contained a double-quoted phrase. A single-quoted here-string should treat it
literally, but when the command is wrapped and re-parsed by the tooling layer the quotes terminate
the string early, and the remainder of the message reaches git as pathspecs.

**Fix.** Write the message to a file and use `git commit -F <file>`. Robust regardless of message
content — quotes, backticks, `$`, apostrophes.

**How to avoid.** Use `-F` for any commit message longer than one line. The failure is loud here
because git rejected it, but the same class of quoting bug can silently truncate a message instead.

---

### PowerShell `Select-Object -First N` reports a false failure

**Symptom.** `python benchmarks/bench_step.py 2>&1 | Select-Object -First 8` exited with code 255 and
looked like a crash. The same command with `-Last 20` completed cleanly.

**Cause.** `-First` stops the pipeline as soon as it has N objects, closing the pipe under the still
running process. The non-zero exit is the broken pipe, not the script.

**How to avoid.** Use `-Last N` to trim long output, or redirect to a file and read that. Do not
conclude a failure from an exit code when the command was piped into `-First`.

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
