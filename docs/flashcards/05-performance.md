# OrbitalEngine::Performance

What was measured, what it meant, and the reasoning errors the measurements exposed. The recurring
lesson is that the bottleneck was never where the plan assumed.

---

### Phase 1 was planned in the wrong order. What did profiling reveal?

The plan scheduled memory-layout work first — compaction, tier-slicing, `bincount` — with Numba last
as polish. Two measurements inverted it:

- Step cost was **flat** from arena capacity 64 to 10,000.
- Three bodies cost 542 us; 602 bodies cost 3583 us.

Fitting those gives **~490 us of fixed per-step overhead** against ~5 us marginal per body. A cost
that does not move with problem size is **not a data-layout problem** — it was Python-level NumPy
dispatch. Tier-slicing targeted `calc_global`, which was 4% of the step.

> src: docs/engineering-log.md - Optimisation order was wrong until measured
> tags: performance, profiling, methodology

---

### The optimisation ordering flipped twice more *during* phase 1. How?

Each fix exposed the next bottleneck, and none were predictable from the code:

1. Propagator compiled → **`calc_global` went from 4% to 87%** of the step.
2. `calc_global` compiled → **`_record_state` at 600 bodies cost 1875 us**, ten times the physics step
   it was recording.

The lesson is to re-profile after *every* change, not once at the start. A bottleneck is only visible
once the one in front of it is gone.

> src: docs/engineering-log.md - Optimisation order was wrong until measured
> tags: performance, profiling, methodology

---

### Why is `if np.any(mask):` an anti-optimisation on arena-sized arrays?

Guards are worth their cost in proportion to the work they skip. On a five-element array, the guard
costs ~5 us of Python and NumPy dispatch to avoid ~1.5 us of work — and the masked operation on an
empty selection would have been nearly free anyway.

Profiling found **37 `np.any()` calls per step**, 21% of the entire step budget, on a *five body*
simulation. Two of them sat inside the Newton loop and were **loop-invariant**.

Below roughly a thousand elements, dispatch dominates and the guard is a net loss.

> src: docs/engineering-log.md - The np.any() guard
> tags: performance, numpy, gotcha

---

### What replaces a boolean mask when a loop needs an active set, and why?

An **integer index array**. The termination test becomes `active.size` — a Python attribute lookup —
rather than `np.any(active)`, which is a NumPy reduction costing microseconds of dispatch.

It also lets the compiled kernel index directly without a mask-to-index conversion each step.

> src: CLAUDE.md - Conventions
> sym: _sib_idx, _head_idx
> tags: performance, numpy

---

### Why are `kernels.py` loops scalar when the rest of the codebase is vectorised?

Because the bottleneck was **per-call dispatch, not per-element arithmetic**, and no amount of
vectorising removes a per-call cost.

Under `@njit` a scalar loop compiles to tight machine code with no temporaries; the vectorised form
would still allocate an intermediate array per operation. The style inversion is deliberate and
confined to that one module.

> src: CLAUDE.md - The two-implementation rule
> sym: kepler_propagate
> tags: performance, kernels

---

### Give the phase-1 step-cost progression for Sun-Earth-Moon.

| Stage | us/step |
|---|---|
| Baseline | 556 |
| Kepler solver rewrite | 474 |
| Compiled propagator | 28.8 |
| Compiled global states | 8.3 |
| Columnar history | **3.0** |

**185x overall.** Marginal cost per body fell from 1.26 us to 0.196 us; 2400 bodies now step in
470 us.

> src: README.md - Performance
> tags: performance, results

---

### Why did history recording cost ten times the physics step, and what fixed it?

The recorder built **one sixteen-key dictionary per body per step**. At 600 satellites that was
1875 us against a 120 us physics step — the simulation spent 94% of its time describing itself.

The fix was columnar: three array slices per step, with the long-format DataFrame assembled lazily in
the `history` property, so a run that never inspects its history never pays for the expansion.
1875 us -> 62 us, with the DataFrame contract unchanged.

> src: README.md; src/orbital_engine/simulator.py
> sym: _record_state, clear_history
> tags: performance, results

---

### Why does the benchmark harness use minimum-of-batches rather than a mean?

Because **timing noise on a desktop OS is one-sided**: scheduler preemption, interrupts and frequency
scaling can only make a run slower, never faster.

The mean therefore estimates "typical machine load during the run", which is not the quantity of
interest. The minimum batch estimates the cost of the work itself. The spread between minimum and
median is reported as a noise ratio so a contended measurement is visible rather than averaged in.

> src: src/orbital_engine/benchmark.py
> sym: measure, noise_ratio
> tags: performance, methodology

---

### Why does the benchmark disable the garbage collector and run warmup calls?

**Warmup** matters because a JIT-compiled kernel would otherwise charge its entire compilation cost
to the first batch, inflating it by orders of magnitude.

**GC disabled** because a collection triggered by unrelated allocation could land inside a timed
batch and be attributed to the code under test. It is restored afterwards.

> src: src/orbital_engine/benchmark.py
> sym: measure
> tags: performance, methodology

---

### What makes step cost independent of `max_capacity`, and how is it protected?

Scratch is **allocated once and zeroed per active slot**, never per capacity.

The original propagator allocated four `(max_capacity, 3)` arrays every step, so five bodies cost
2093 us in a 65536-slot arena against 479 us in a 64-slot one — identical physics, 4.4x the cost. The
compiled path is flat at ~3.0 us across that entire range.

`test_scaling_invariants.py` protects it as a ratio: a 1024x capacity increase may cost at most 5x in
time.

> src: tests/validation/test_scaling_invariants.py
> sym: _kick, _accum
> tags: performance, scaling, arena

---

### Why did the solver rewrite alone save 82 us/step before any compilation?

It split elliptic from hyperbolic **once, up front**, and iterated over contiguous subarrays.

The previous version carried both branches through a single loop and re-derived the split on every
iteration by indexing a global-length mask with the shrinking active set — which was also a latent
correctness bug in the successive-substitution branch.

`np.any()` calls per step fell from 37.7 to 22.

> src: docs/engineering-log.md
> sym: _iterate_kepler
> tags: performance, numerics

---

### Numba is an optional dependency. What guarantees the fallback path still works?

A **dedicated CI job** that installs without the `[perf]` extra and asserts `NUMBA_AVAILABLE` is
`False` before running the full suite. The compiled job asserts the converse.

Without those assertions, a silent fallback would turn one job into a duplicate of the other and
nobody would notice the compiled path had stopped being tested.

> src: .github/workflows/ci.yml
> sym: NUMBA_AVAILABLE
> tags: performance, ci, kernels
