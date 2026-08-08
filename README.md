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

### Not yet wired

Propagators are written as stateless kernels and a registry exists, but `Simulation.step()` currently
calls the Keplerian propagator directly rather than dispatching through it. Perturbation models,
numerical integrators and the sweep harness are in progress. See `CLAUDE.md` for the current state of
play, including known-broken code.

---

## Verification

| | |
|---|---|
| Tests | 120 passing |
| Type checking | `mypy --strict`, clean across 11 modules |
| CI | GitHub Actions on Python 3.10 / 3.11 / 3.12, every branch |
| Coverage | 72% overall |

The suite is split along the **Verification and Validation** distinction used in computational
science — *are we solving the equations right* versus *are we solving the right equations*:

```text
tests/
├── unit/         # Isolated transformations. Analytic geometries and round-trip properties.
├── integration/  # ORM polymorphism and simulation construction from the database.
└── validation/   # Physical correctness against closed-form and conserved-quantity references.
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

---

## Directory layout

```text
Orbit_Mechanics/
├── .claude/agents/      # Subagent definitions with model and effort pinned per task type
├── .github/workflows/   # Multi-version CI
├── docs/
│   ├── historical/      # Superseded design documents, retained for provenance only
│   └── model-delegation.md
├── notebooks/           # Derivations and visualisation
├── src/orbital_engine/
│   ├── body.py          # BodyHandle dataclass (arena pointer, not state)
│   ├── constants.py     # Physical and unit constants
│   ├── custom_types.py  # Type aliases and column-index enums
│   ├── database.py      # Polymorphic SQLAlchemy 2.0 ORM
│   ├── exceptions.py    # Domain-specific error hierarchy
│   ├── frames.py        # Coordinate and state-space transformations
│   ├── propagators.py   # Stateless propagation kernels
│   ├── registry.py      # Model and propagator registration
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

1. **Validation infrastructure** — reference-data ingestion, invariant and convergence-order
   harnesses. In progress; this is what makes later physics safe to accept quickly.
2. **Arena compaction and compiled kernels** — dense active-slot packing so hot-loop operations are
   contiguous views rather than gathers, then Numba on the kernels.
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
