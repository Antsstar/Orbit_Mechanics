# OrbitalEngine::Validation

Why the test suite is shaped the way it is. The failure mode this project guards against is not a
crash — it is a trajectory that integrates cleanly, plots plausibly, and is wrong by a few percent.

---

### Name the four validation layers and what each one catches that the others cannot.

| Layer | Question | Catches |
|---|---|---|
| Unit | Do transformations invert? | Algebra errors, singularity handling |
| Invariant | Is energy / momentum / centre of mass conserved? | Sign errors, mass-ratio errors |
| Equivalence | Do the two implementations agree? | Optimisation bugs |
| Reference | Does the trajectory match independent integration? | Everything the first three can share |

The reference layer exists because the first three can all be satisfied by an engine that is
**self-consistently wrong**.

> src: docs/architecture.md - Validation layers
> tags: validation, philosophy

---

### Distinguish verification from comparison, and say why conflating them is dangerous.

**Verification:** the engine's model is mathematically *exact* for this case (two-body, massive or
massless secondary). The two methods must agree to integration tolerance, and disagreement is an
engine bug.

**Comparison:** the model is an *approximation* (Sun-Earth-Moon omits the solar term in the lunar
orbit). The two *must* diverge, and the size of the divergence is the result being measured.

Reading a comparison divergence as a bug leads to "fixing" a correct engine. Reading a verification
divergence as a result hides a real one.

> src: docs/architecture.md - Verification is not comparison
> tags: validation, philosophy, vandv

---

### Why does `reference.py` deliberately share no code with the engine?

So that an error cannot appear on **both sides** of a comparison.

It integrates Newtonian point-mass acceleration in inertial Cartesian coordinates with DOP853 at
`rtol=1e-13`: no classical elements, no Kepler solver, no hierarchy, no barycentres. A sign error in
`frames.py` would corrupt a round-trip test and the engine equally — it cannot corrupt this.

That independence is precisely the weakness of round-trip testing, which is why both exist.

> src: docs/architecture.md; src/orbital_engine/reference.py
> sym: reference_for, nbody_acceleration
> tags: validation, reference

---

### What is a negative control in this suite, and why does every comparison need one?

A test that **deliberately perturbs the engine and asserts the comparison detects it** — a 1% wrong
`mu`, or a 1e-9 nudge to one element.

Without it, a tolerance looser than the effect it is meant to catch passes regardless of correctness,
and nothing reveals that. This repository had exactly that failure for months, so both the
equivalence and reference suites now carry one.

> src: README.md - Negative controls
> tags: validation, philosophy

---

### A fuzzing test here passed for months while asserting nothing. What happened?

It filtered to *degenerate* orbits instead of valid ones:

```python
mask = np.linalg.norm(h, axis=-1) < 1e-3   # selects near-radial orbits
assert np.all(np.abs(r_out[mask] - r_in[mask]) < 1e-7)
```

Generated orbits had `|h|` in the thousands, so the mask selected **0 of 100** every time — and
`np.all()` on an empty array is `True`.

Two lessons: every filter-then-assert needs a non-emptiness guard, and **coverage measures execution,
not verification** (it read 70% before the fix and 70% after).

> src: docs/engineering-log.md - A test that asserted nothing for months
> tags: validation, gotcha, coverage

---

### Why does NaN defeat the entire validation suite, and what is the only defence?

Every comparison against NaN is `False` — `nan < tol`, `nan > tol`, and `nan == nan` alike.

Conserved-quantity tests assert that a drift is *small*, and NaN is never measured as large, so it
passes every one of them silently. A NaN position propagates into the barycentre and contaminates
every body downstream.

The only defence is an explicit `np.all(np.isfinite(...))`, which is why that exists as its own test
rather than folded into a tolerance assertion.

> src: docs/engineering-log.md - A diverging solver returned NaN
> tags: validation, nan, gotcha

---

### A solver returned NaN while reporting success. What was the one-character-class fix?

The convergence test kept whatever had *demonstrably diverged*:

```python
active = active[np.abs(delta) > tol]        # NaN > tol is False -> dropped as converged
```

A diverging hyperbolic iterate overflows `sinh` to `inf`, then `inf - inf = nan`, so the element was
silently removed from the active set and the loop exited cleanly with NaN in hand.

Inverting it to keep whatever has *not demonstrably converged* fixes it:

```python
active = active[~(np.abs(delta) <= tol)]    # NaN <= tol is False -> ~False -> stays active
```

With NaN in play, `keep = ~converged` and `keep = diverged` are **not** the same predicate.

> src: docs/engineering-log.md - A diverging solver returned NaN
> sym: _iterate_kepler
> tags: validation, nan, numerics

---

### Why is angular momentum compared as a vector rather than a magnitude?

Because a propagator that **preserved `|h|` while rotating the orbital plane** would pass a magnitude
check and still be completely wrong.

Direction carries physical meaning here — the orbital plane's orientation is a conserved quantity of
the two-body problem, not an incidental detail.

> src: README.md; tests/validation/invariants.py
> sym: specific_angular_momentum
> tags: validation, invariants

---

### How is a tolerance chosen in this suite?

**Derived, then given headroom.** From an analytic argument or a floating-point-limit argument,
typically one to two orders above expected numerical noise, recorded in a named constant with a
comment justifying the number.

Never loosen one to make a failing test pass without first establishing whether the *test* or the
*engine* is wrong. Widening a tolerance until it passes converts a test into a snapshot and destroys
its value.

> src: docs/engineering-log.md - Conventions that emerged
> tags: validation, tolerances, philosophy

---

### The compiled and NumPy `calc_global` paths are asserted *bit-identical*. Why not a tolerance?

Because they perform the **same additions in the same topological order**, and floating-point
addition is deterministic. Only a different order or a different operand set can perturb the result.

Asserting exact equality is therefore strictly more sensitive than any tolerance, and it costs
nothing.

> src: tests/validation/test_kernel_equivalence.py
> sym: calc_global_states
> tags: validation, equivalence, kernels

---

### Centre-of-mass drift measures 1e-19 km. What does that actually prove?

**The implementation, not the physics.** The reflex kick maintains the centre of mass
*algebraically* — it is constructed to cancel — so the result is exact to rounding by construction.

It is a sharp test of mass aggregation and of the kick's sign and normalisation. It says nothing
about whether the trajectory is right, which is what the reference layer is for.

> src: tests/validation/test_barycentric_dynamics.py
> tags: validation, invariants, interpretation

---

### Two-body agreement with DOP853 is 7.5e-5 km over ten days. Whose error is that?

**The integrator's.** The analytic propagator solves Kepler's equation to machine precision, so it is
the more accurate party; DOP853 at `rtol=1e-13` accumulates roughly 1e-9 relative over tens of orbits.

The residual is therefore a bound on the *reference*, not on the engine — which is why the tolerance
is set from the integrator's expected accumulation rather than the propagator's.

> src: tests/validation/test_reference_agreement.py
> tags: validation, reference, interpretation

---

### What does the *massive*-secondary two-body case validate that the massless one cannot?

**The reflex kick's mass ratio.**

Both bodies now orbit their common barycentre, and the motion is still exactly Keplerian — but only
if the kick is right. A wrong mass ratio displaces the two bodies in opposite directions while
leaving the centre of mass fixed, so every barycentric invariant still passes.

Only comparison against an independent trajectory catches it.

> src: tests/validation/test_reference_agreement.py
> tags: validation, reference, barycentric

---

### Why are the scaling tests written as ratios rather than microsecond thresholds?

Because a ratio **cancels machine speed, CI runner variance, and interpreter version**, so the bound
can be tight in the property while staying loose in the timing.

A 1024x arena-capacity increase is allowed at most 5x in step time. Per-step work that scaled with
capacity would overshoot by roughly two orders of magnitude; ordinary noise is nowhere near it.

They carry the `perf` marker so they can be deselected with `-m "not perf"` on a contended machine.

> src: tests/validation/test_scaling_invariants.py
> tags: validation, performance, ci

---

### Why must coverage be measured with Numba disabled?

Because `coverage.py` traces Python **bytecode**, and an `@njit` function executes as machine code —
so every line inside it reads as unhit.

The compiled run reports ~15% for `kernels.py`, a module that is ~88% covered. The project total is
79%, not the 70% the compiled run shows.

Chasing that phantom gap by writing tests for already-covered code is the trap.

> src: docs/engineering-log.md - Coverage under numba
> tags: validation, coverage, tooling
