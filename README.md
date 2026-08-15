# OrbitalEngine
[![Orbital Engine CI](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml/badge.svg)](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml)

A hierarchical orbital mechanics engine built on Data-Oriented Design, developed as an educational
platform for comparing orbital models against one another.

**Direction.** Most astrodynamics libraries treat *running a simulation* as the primary operation.
This one is being built the other way round: the intended main loop is a **sweep** — one scenario run
under many model configurations, diffed against a common reference — so that the effect of a
modelling assumption on accuracy, stability and runtime is a measurable output rather than a
footnote. Model configurations are therefore designed to be enumerable data, not code paths.

---

## Current capabilities

- **Barycentric DOD arena.** All state lives in flat, pre-allocated NumPy arrays indexed by integer
  slot; no object owns per-body state. The same vectorised pipeline handles isolated two-body
  point-mass motion and nested N-body barycentric systems, selected by how the scenario is
  described rather than by branching in the physics.
- **Decoupled physics and kinematics graphs.** `parent_indices` records what a body's orbital
  elements are measured against; `body_sys_map` records what its Cartesian state is measured
  against. The two deliberately diverge — the Moon's elements are relative to Earth while its
  position is relative to the Earth–Moon barycenter.
- **Topological stratification.** A vectorised breadth-first sort groups bodies into dependency
  tiers so each tier resolves in one array operation, and sibling bodies within a system are
  processed together as barycentric calculations require.
- **State-space transformations.** Bidirectional Cartesian ↔ classical orbital elements, vectorised
  over `(N, 6)` inputs, with analytic fallbacks for the circular, equatorial, polar and parabolic
  singularities and a success mask for states that admit no valid orbit.
- **Full anomaly stack.** True ↔ eccentric ↔ mean for elliptic, parabolic and hyperbolic regimes,
  with Newton–Raphson and successive-substitution solvers, plus Kepler's and Barker's equations.
- **Reflex-kick barycenters.** A head body wobbles about its own system's centre of mass rather than
  sitting fixed, which keeps angular momentum well defined when a system's members are only
  partially loaded.

Column 0 of the element array is the **semi-latus rectum** `p`, not the semi-major axis, so parabolic
trajectories stay representable throughout.

- **Compiled kernels.** The Keplerian step and global-state accumulation are compiled to machine code
  with Numba, held elementwise equivalent to a readable NumPy reference implementation that is
  retained as the definition of the physics.

### Not yet wired

A propagator registry exists but is not read: `Simulation.step()` selects Keplerian propagation
unconditionally, choosing only between the compiled and NumPy implementations of it. Perturbation
models, numerical integrators and the sweep harness are in progress. See `CLAUDE.md` for the current
state of play, including known-broken code.

---

## Performance

Optimisation was driven by measurement rather than by plan, and the plan was wrong. Profiling showed
step cost was **flat** from arena capacity 64 to 10 000, and that three bodies cost 542 µs against
602 bodies at 3583 µs — roughly 490 µs of fixed per-step overhead against ~5 µs marginal per body.
A cost that does not move with problem size is not a data-layout problem, so the memory-layout work
that was scheduled first would have addressed about 4% of the step. Compilation addressed the rest.

Sun–Earth–Moon, microseconds per step:

| | µs/step | |
|---|---|---|
| Baseline | 556 | |
| Kepler solver rewrite | 474 | removed per-iteration dispatch |
| Compiled propagator | 28.8 | |
| Compiled global states | 8.3 | previous bottleneck at 87% |
| Columnar history | **3.0** | **185× total** |

Marginal cost fell from 1.26 to 0.196 µs per body; 2400 bodies now step in 470 µs. Step cost is
independent of arena capacity — five bodies cost the same in a 64-slot arena as in a 65 536-slot one,
which the original allocated-per-step scratch arrays violated by two orders of magnitude.

Numba is an optional extra. Without it the engine falls back to the NumPy propagator and remains
correct, and a dedicated CI job gates that path.

---

## Verification

| | |
|---|---|
| Tests | 214 passing |
| Type checking | `mypy --strict`, clean across 15 modules |
| CI | Python 3.10 / 3.11 / 3.12 with compiled kernels, plus a NumPy-fallback job |
| Coverage | 79% overall |

> Coverage must be measured with Numba disabled. `coverage.py` traces bytecode, so a `@njit` function
> reads as entirely unhit — the compiled run reports 15% for a module that is 88% covered.

The suite is split along the **Verification and Validation** distinction used in computational
science — *are we solving the equations right* versus *are we solving the right equations*:

```text
tests/
├── unit/         # Isolated transformations. Analytic geometries and round-trip properties.
├── integration/  # ORM polymorphism, simulation construction, history recording contract.
└── validation/   # Physical correctness against closed-form, conserved-quantity and
                  # numerically-integrated references, plus kernel equivalence.
```

**Validation tests assert a stated expected error magnitude, not merely that nothing crashed.** This
is deliberate: the failure mode in orbital mechanics is rarely an exception. A sign error or a
transposed term produces a trajectory that integrates cleanly, plots plausibly, and is wrong by a few
percent. Some examples of what the suite pins down:

- **Conserved quantities.** Specific energy and angular momentum over a full orbit; angular momentum
  is compared as a *vector*, since a propagator that preserved `|h|` while rotating the orbital plane
  would pass a magnitude check and still be wrong.
- **Step-size independence.** An analytic propagator solves Kepler's equation in closed form, so its
  closure error must not scale with `dt`. Measured flat from 1e-11 to 7e-10 km across 10/100/1000
  steps per period. This also acts as a tripwire if the propagator is ever silently replaced by a
  numerical one.
- **Barycentric mass moments.** `mu_Sun·r_Sun + mu_EMB·r_EMB = 0` is a definition rather than an
  approximation, so it holds to floating-point noise and any error in the mass aggregation or the
  reflex kick surfaces immediately as a drifting centre of mass.
- **Predicted discrepancies.** The Earth–Moon barycenter does *not* inherit Earth's seeded ellipse;
  it differs by `Δp/p ≈ 2·Δv/v ≈ 8.5e-4`, derived from Earth's monthly circulation about the
  barycenter. The test asserts that prediction, with a lower bound so a collapsed barycenter cannot
  satisfy it trivially.
- **Reproducible randomness.** Property tests seed an explicit `np.random.default_rng`, so a failure
  can be replayed exactly and no global state leaks between tests.
- **Independent reference trajectories.** `reference.py` integrates Newtonian N-body motion with
  `scipy`'s DOP853 at `rtol=1e-13`, sharing no code with the engine — no elements, no Kepler solver,
  no hierarchy — so an error in `frames.py` cannot appear on both sides of a comparison. Where the
  engine's model is exact, the two agree to 7.5e-5 km over ten days.
- **Negative controls.** Equivalence and reference tests each include a test that *deliberately
  perturbs* the engine and asserts the comparison detects it. A tolerance looser than the effect it
  is meant to catch passes regardless of correctness, which is how this project's one genuinely
  vacuous test survived for months.

### Verification is not comparison

Where the engine's model is an approximation, it *must* disagree with the reference, and the
disagreement is the measurement rather than a defect. Hierarchical two-body motion neglects the solar
term in the lunar orbit entirely, so over 30 days the Moon diverges by 3.3e4 km. That figure is
asserted against a derived band, not a recorded snapshot: solar tidal acceleration on the Moon is
`2·μ_Sun·r_EM/d³ ≈ 3.05e-8 km/s²`, which acting coherently over 30 days would displace it by
`½at² ≈ 1.0e5 km`, and partial coherence over a synodic month puts the true value comfortably below.

Quantifying that error is the point of the engine, so a test asserting agreement there would be
asserting that the engine is wrong.

---

## Directory layout

```text
Orbit_Mechanics/
├── .claude/agents/      # Subagent definitions with model and effort pinned per task type
├── .github/workflows/   # Multi-version CI, plus a NumPy-fallback job
├── benchmarks/
│   └── bench_step.py    # Step-cost instrument: propagator comparison, scaling, breakdown
├── docs/
│   ├── engineering-log.md   # Problems hit and how they were resolved. Check before debugging
│   ├── historical/      # Superseded design documents, retained for provenance only
│   └── model-delegation.md
├── notebooks/           # Derivations and visualisation
├── src/orbital_engine/
│   ├── benchmark.py     # Timing primitive (min-of-batches)
│   ├── body.py          # BodyHandle dataclass (arena pointer, not state)
│   ├── constants.py     # Physical and unit constants
│   ├── custom_types.py  # Type aliases and column-index enums
│   ├── database.py      # Polymorphic SQLAlchemy 2.0 ORM
│   ├── exceptions.py    # Domain-specific error hierarchy
│   ├── frames.py        # Coordinate and state-space transformations
│   ├── kernels.py       # Compiled scalar kernels (numba); optional
│   ├── propagators.py   # NumPy reference propagators
│   ├── reference.py     # DOP853 N-body truth trajectories (scipy); optional
│   ├── registry.py      # Model and propagator registration
│   ├── scenarios.py     # Reusable scenario builders, shared by tests and benchmarks
│   ├── simulator.py     # DOD memory arena and simulation control
│   ├── utilities.py     # Anomalies, Kepler, Barker, rotations, perturbations
│   └── data/planets.db
├── tests/
├── CLAUDE.md            # Engine ground truth: invariants, conventions, known-broken code
└── pyproject.toml
```

---

## Installation

```bash
conda create -n orbital_env python=3.11 -y
conda activate orbital_env
pip install -e ".[dev,test]"
```

Optional extras: `[perf]` pulls in Numba for the compiled kernels, `[reference]` pulls in SciPy for
the DOP853 truth generator. Neither is required — the engine falls back to pure NumPy without them.

Editable mode links the source into the environment so changes propagate to the notebooks
immediately.

```bash
pytest                    # test suite
mypy src/ --strict        # type checking
pytest --cov=orbital_engine --cov-report=term-missing
```

> **Note.** Do not run `python -m orbital_engine.simulator`. Its `__main__` block calls
> `seed_test_universe()`, which drops and rewrites the git-tracked database. Build an in-memory
> SQLite session the way `tests/conftest.py` does instead.

---

## Usage

<details>
<summary><b>State vector to classical orbital elements</b></summary>

```python
import numpy as np
from orbital_engine.frames import ReferenceFrames

mu = 398600.4418
r = np.array([7000.0, 0.0, 0.0])
v = np.array([0.0, np.sqrt(mu / 7000.0), 0.0])

coe, success = ReferenceFrames.rv_to_coe(r, v, mu)
# coe -> [p, e, i, RAAN, arg_pe, theta]; success flags states with no valid orbit
```
</details>

<details>
<summary><b>Classical orbital elements back to a state vector</b></summary>

```python
r_out, v_out, valid = ReferenceFrames.coe_to_rv(coe, mu)
```

Both directions are vectorised: pass `(N, 3)` or `(N, 6)` arrays to convert many bodies at once.
Coordinate singularities resolve through analytic fallbacks rather than raising.
</details>

---

## Roadmap

Ordered so that each stage makes the next one safe rather than merely possible.

1. ~~**Validation infrastructure**~~ — invariant harnesses, kernel equivalence, and an independent
   DOP853 reference generator. **Done.** This is what makes later physics safe to accept quickly.
2. ~~**Compiled kernels**~~ — **Done**, 185× on the hierarchical step, with step cost now independent
   of arena capacity. Reordered ahead of memory-layout work on the strength of profiling.
3. **Force-model interface and event system** — bitmask composition so enabling a perturbation is a
   value rather than a code path, plus exact-time event handling for impulsive manoeuvres.
4. **Benchmark harness** — scenario sweeps across model configurations, reporting error against
   reference versus wall time.
5. **Model library** — geopotential harmonics, atmospheric drag across fidelity tiers, solar
   radiation pressure with shadow geometry, third-body perturbations, continuous and impulsive
   thrust, Cowell and Encke formulations, symplectic integrators, and an SGP4 bridge.
6. **Constellation and inter-satellite link modelling** — the eventual application this engine is
   being shaped for.

Established external implementations are wrapped rather than reimplemented — SGP4, atmospheric
density models, planetary ephemerides and IAU frame/time transformations all have well-tested
libraries, and the interesting question here is how models *compare*, not whether they can be
retyped.
