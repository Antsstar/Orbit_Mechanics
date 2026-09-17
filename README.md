# OrbitalEngine
[![Orbital Engine CI](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml/badge.svg)](https://github.com/Antsstar/Orbit_Mechanics/actions/workflows/ci.yml)

A hierarchical orbital mechanics engine for measuring what a modelling assumption costs. One scenario
runs under many model configurations (analytic Kepler, secular J2, numerically integrated Cowell with
composable force models), and each is scored against an independent high-accuracy reference for both
position error and wall time.

Most astrodynamics libraries treat *running a simulation* as the primary operation. Here the main loop
is a **sweep**: model configurations are plain data, so the effect of an assumption on accuracy and
runtime is a measured output rather than a footnote.

Python 3.10+, NumPy and SQLAlchemy 2.0. Numba (compiled kernels) and SciPy (reference trajectories)
are optional.

---

## The model-fidelity frontier

`orbital_engine.sweep` runs one scenario under several model configurations and scores each against a
common DOP853 + J2 reference, computed once and reused. `benchmarks/frontier_plot.py` runs it on a
12-satellite, 550 km / 53° constellation over 24 hours:

![Model-fidelity frontier](docs/figures/frontier.png)

Median position error after 24 hours, and the wall time to reach that state:

- **Kepler: 608 km in 26 µs.** No J2 at all.
- **Kepler + secular J2, seeded from osculating elements: 431 km in about 40 µs.** The secular drift removes
  the cross-track error. The along-track error remains, because mean motion is taken from the
  *osculating* semi-major axis, whose short-period J2 term acts as a fixed rate bias.
- **The same propagator seeded with a first-order *mean* semi-major axis: 4.0 km in about 40 µs**
  (`propagators.mean_seeded_p`, after Kozai 1959 / Brouwer 1959). The remaining few km are the
  short-period oscillation an averaged theory cannot represent.
- **Cowell + `point_mass_gravity` + `j2`: 231 km at a 160 s step (5.5 ms) down to 0.0004 km at 10 s
  (85 ms).** Wall time rises about 16× across that range, as fourth-order Runge-Kutta predicts: each
  halving of the step doubles the cost and cuts the error about 16-fold.

So the mean-seeded analytic tier sits on the frontier down to a few km. Doing better needs numerical
integration, at roughly 200–3,000× the cost: 0.27 km for 20 ms, 0.0004 km for 85 ms. Timings vary
by about 20% from run to run (the two secular tiers do identical work and read 38 and 50 µs on the
plotted run); errors do not.

Two caveats:

- **The analytic tiers reach the horizon in one step; Cowell has to step.** Kepler and secular J2 are
  closed-form in time, so their error does not depend on step size: 60 s steps and a single 24-hour
  step gave identical errors. The plot measures the cost of the horizon state, not of an ephemeris
  sampled at a fixed cadence, which would charge the analytic tiers per output point.
- **Single-satellite figures depend on starting phase.** Against the same truth, one secular-J2
  satellite's error after 10 orbits ranges from 0.2 km to 574 km depending on its initial argument of
  latitude (see `docs/architecture.md`). That is why the sweep reports statistics over all bodies.

Every tier runs compiled. Cowell uses `kernels.cowell_rk4_step`, a fused twin of RK4 +
`point_mass_gravity` + `j2`, held to the NumPy path at 1e-12 relative. The re-base that places a
Cowell or secular-J2 body on its parent's end-of-step position is compiled too
(`kernels.rebase_relative_states`, bit-identical to the NumPy block). Measured on the 12-satellite
scenario, one step costs about 6 µs for Cowell and 7 µs for secular J2 against 5 to 6 µs for Kepler;
with the NumPy re-base those were 17 and 24 µs.

---

## What is in the engine

**Architecture**

- **Data-oriented arena.** All state lives in flat, pre-allocated NumPy arrays indexed by integer
  slot. No object owns per-body state, and step cost is independent of arena capacity.
- **Two parent graphs, deliberately different.** `parent_indices` records what a body's orbital
  elements are measured against; `body_sys_map` records what its Cartesian state is measured against.
  The Moon's elements are relative to Earth while its position is relative to the Earth–Moon
  barycentre.
- **Reflex-kick barycentres.** A system's head body moves about the system's centre of mass. Siblings
  couple through that shared motion while every body's anomaly still advances analytically, so the
  analytic path steps backwards to rounding level (tested).
- **Two implementations of every hot path.** A readable NumPy reference defines the physics. A
  compiled Numba twin is held equivalent to it by tests.

**Models, selected per body as data**

- **Propagators** (`Simulation.set_propagator`): Keplerian; secular J2 (first-order drift of Ω, ω and
  M, with optional mean seeding); and Cowell (fourth-order Runge-Kutta, integrated relative to the
  body's parent so an accelerating parent is handled correctly).
- **Force models** (`Simulation.enable_force_model`): `point_mass_gravity` and `j2`, composed through a
  per-body bitmask. A model can register a configuration-time check: `j2` refuses bodies whose parent
  is a barycentre.
- **Sweep** (`orbital_engine.sweep`): `ModelConfig` lists run against one scenario. It reports median,
  RMS and max error over bodies, plus minimum-of-batches wall time.
- **Independent truth** (`reference.py`): Newtonian N-body integration with DOP853, optionally with J2,
  sharing no code with the engine.

**Orbital toolbox**

- Cartesian ↔ classical elements, vectorised, with analytic fallbacks for circular, equatorial, polar
  and parabolic cases. Column 0 is the semi-latus rectum `p`, so parabolic orbits stay representable.
- The RSW (radial, along-track, cross-track) frame.
- True ↔ eccentric ↔ mean anomaly for elliptic, parabolic and hyperbolic orbits, plus Kepler's and
  Barker's equations.

---

## Verification

| | |
|---|---|
| Tests | 345 passing |
| Type checking | `mypy --strict`, clean across 20 source files |
| CI | Python 3.10 / 3.11 / 3.12 with compiled kernels, plus a job without Numba |
| Coverage | 86%, measured with Numba disabled |

> Measure coverage with Numba disabled. `coverage.py` traces bytecode, so a `@njit` function reads as
> entirely unhit, and the compiled run reports a well-covered kernel module as mostly untested.

The suite follows the verification and validation split used in computational science: *are the
equations solved right* versus *are they the right equations*.

```text
tests/
├── unit/         # Isolated transformations: analytic geometries and round-trip properties
├── integration/  # ORM polymorphism, simulation construction, history recording
└── validation/   # Physical correctness against closed forms, conserved quantities and
                  # independent numerical references, plus kernel equivalence
```

**Validation tests assert an expected error magnitude derived in advance, not just that nothing
crashed.** In orbital mechanics a sign error rarely raises. It produces a trajectory that integrates
cleanly, plots plausibly and is wrong by a few percent. Some of what the suite pins down:

- **Conserved quantities.** Specific energy, and angular momentum as a *vector*, since a propagator
  that preserved `|h|` while rotating the orbital plane would pass a magnitude check.
- **Convergence order.** Cowell's error must fall about 16-fold per halving of the step, both against
  Keplerian motion and against the J2 reference. It converges to 9.2e-8 km at 2048 steps per orbit.
  A missing physical term does not shrink with the step size, so this catches bugs a fixed tolerance
  would miss. An early Cowell version that ignored the parent's own acceleration, found in review,
  fails it.
- **Independent derivations.** The reference J2 field is derived in spherical coordinates, while the
  engine uses the Cartesian form. They agree to 1.2e-15. Where the engine's model is exact, engine and
  reference agree to 7.5e-5 km over ten days.
- **Time-reversibility.** Stepping forward then backward returns the analytic arena to rounding level
  (3.4e-11 relative after 1000 steps each way). An `abs(dt)` bug planted in the compiled kernel
  passed every other test and was caught only here.
- **Barycentric mass moments.** `mu_Sun·r_Sun + mu_EMB·r_EMB = 0` is a definition, so it holds to
  floating-point noise, and any error in mass aggregation or the reflex kick shows as a drifting
  centre of mass.
- **Negative controls and mutation checks.** Equivalence and reference suites each include a test that
  deliberately breaks the engine and asserts the comparison notices. New physics is also checked by
  mutating the real source, such as a flipped sign or a wrong coefficient, and confirming the suite
  fails.

### Verification is not comparison

Where the engine's model is an approximation, it *must* disagree with the reference, and the
disagreement is the measurement. Hierarchical two-body motion neglects the solar term in the lunar
orbit, so over 30 days the Moon diverges by 3.3e4 km. That figure is asserted against a derived band:
solar tidal acceleration on the Moon is `2·μ_Sun·r_EM/d³ ≈ 3.05e-8 km/s²`, which acting coherently for
30 days would displace it by `½at² ≈ 1.0e5 km`. Partial coherence over a synodic month puts the true
value below that.

---

## Performance

Optimisation followed measurement, and the original plan turned out to be wrong. Step cost was
**flat** from arena capacity 64 to 10,000, and three bodies cost 542 µs against 602 bodies at 3583 µs.
That is roughly 490 µs of fixed per-step overhead against ~5 µs marginal per body. A cost that does not
move with problem size is not a data-layout problem, so the memory-layout work scheduled first would
have addressed about 4% of the step. Compilation addressed the rest.

Sun–Earth–Moon, microseconds per step:

| | µs/step | |
|---|---|---|
| Baseline | 556 | |
| Kepler solver rewrite | 474 | removed per-iteration dispatch |
| Compiled propagator | 28.8 | |
| Compiled global states | 8.3 | previous bottleneck at 87% |
| Columnar history | **3.0** | **185× total** |

Marginal cost fell from 1.26 to 0.196 µs per body; 2400 bodies step in 470 µs. Five bodies cost the
same in a 64-slot arena as in a 65,536-slot one. The original allocated-per-step scratch arrays
violated that by two orders of magnitude.

---

## How this project is developed

- **`docs/architecture.md`** records why the engine is shaped this way, including decisions that look
  accidental and are load-bearing.
- **`docs/engineering-log.md`** records problems hit and how they were resolved, including mistakes
  made during development and review.
- **`docs/flashcards/`** holds 96 spaced-repetition cards, exportable to Anki, on the architecture, the
  orbital mechanics and the reasoning behind each decision. A `--check` mode fails when a card cites
  a code symbol that no longer exists.
- **AI-assisted, review-gated.** Much of the implementation was written with Claude Code agents. Every
  change is re-verified independently before merge, with separate probes, mutation checks and derived
  error bounds, and the corrections that review produced are recorded in the engineering log.

---

## Directory layout

```text
Orbit_Mechanics/
├── .claude/agents/          # Agent definitions used during development
├── .github/workflows/       # Multi-version CI, plus a job without Numba
├── benchmarks/
│   ├── bench_step.py        # Step-cost instrument: propagator comparison, scaling, breakdown
│   └── frontier_plot.py     # Runs the sweep and writes docs/figures/frontier.png
├── docs/
│   ├── architecture.md      # Why the engine is shaped this way
│   ├── engineering-log.md   # Problems hit and how they were resolved
│   ├── figures/
│   ├── flashcards/          # Anki decks and exporter
│   └── historical/          # Superseded design documents, kept for provenance only
├── notebooks/               # Derivations and visualisation
├── src/orbital_engine/
│   ├── simulator.py         # Data-oriented arena and simulation control
│   ├── propagators.py       # NumPy reference propagators: Keplerian, secular J2, Cowell adapter
│   ├── kernels.py           # Compiled Numba twins (optional)
│   ├── integrators.py       # RK4, integrating relative to each body's parent
│   ├── forces.py            # Force-model composition and the kernel contract
│   ├── gravity.py           # point_mass_gravity
│   ├── geopotential.py      # j2
│   ├── registry.py          # Force-model and propagator registration
│   ├── sweep.py             # Model configurations as data; error and timing statistics
│   ├── reference.py         # Independent DOP853 truth, optionally with J2 (SciPy, optional)
│   ├── scenarios.py         # Scenario builders shared by tests and benchmarks
│   ├── frames.py            # Coordinate and state-space transformations, RSW
│   ├── utilities.py         # Anomalies, Kepler, Barker, rotations
│   ├── benchmark.py         # Minimum-of-batches timing
│   ├── database.py          # Polymorphic SQLAlchemy 2.0 ORM
│   ├── body.py, constants.py, custom_types.py, exceptions.py
│   └── data/planets.db
├── tests/
├── CLAUDE.md                # Engine ground truth: invariants, conventions, known-broken code
└── pyproject.toml
```

---

## Installation and running

```bash
conda create -n orbital_env python=3.11 -y
conda activate orbital_env
pip install -e ".[dev,test]"
```

`[test]` includes Numba and SciPy so the full suite runs; `[dev]` adds matplotlib for the plot. For a
minimal install, `[perf]` adds Numba alone and `[reference]` adds SciPy alone. The engine itself runs
on NumPy without either.

```bash
pytest                               # test suite
mypy src/ --strict                   # type checking
python benchmarks/frontier_plot.py   # regenerate the frontier figure (about a minute)
python benchmarks/bench_step.py      # step-cost benchmarks
```

> **Note.** Do not run `python -m orbital_engine.simulator`. Its `__main__` block rewrites the
> git-tracked database. Build an in-memory SQLite session as below, or as `tests/conftest.py` does.

---

## Usage

<details open>
<summary><b>Sweep three model configurations against the J2 truth</b></summary>

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ
from orbital_engine.sweep import ForceModelSpec, ModelConfig, run_sweep


def build_scenario():
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return scenarios.earth_constellation(sessionmaker(bind=engine)(), n_sats=12, n_planes=1)


j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
day = 86400.0
configs = [
    ModelConfig("kepler", PropagatorType.KEPLERIAN, dt=day),
    ModelConfig("secular J2, mean-seeded", PropagatorType.SECULAR_J2, dt=day,
                propagator_coefficients=j2, mean_seed=True),
    ModelConfig("cowell + J2, 40 s", PropagatorType.COWELL, dt=40.0,
                force_models=(ForceModelSpec("point_mass_gravity"), ForceModelSpec("j2", j2))),
]

for r in run_sweep(build_scenario, configs, horizon_s=day, oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)}):
    print(f"{r.config_name:24s} median error {r.error.median_km:9.3f} km   {r.wall_time_us:9.1f} us")
```

```text
kepler                   median error   607.611 km        21.7 us
secular J2, mean-seeded  median error     4.045 km        35.8 us
cowell + J2, 40 s        median error     0.266 km     35338.5 us
```
</details>

<details>
<summary><b>State vector to classical orbital elements and back</b></summary>

```python
import numpy as np
from orbital_engine.frames import ReferenceFrames

mu = 398600.4418
r = np.array([7000.0, 0.0, 0.0])
v = np.array([0.0, np.sqrt(mu / 7000.0), 0.0])

coe, success = ReferenceFrames.rv_to_coe(r, v, mu)
# coe -> [p, e, i, RAAN, arg_pe, theta]; success flags states with no valid orbit
r_out, v_out, valid = ReferenceFrames.coe_to_rv(coe, mu)
```

Both directions are vectorised: pass `(N, 3)` or `(N, 6)` arrays to convert many bodies at once.
Coordinate singularities resolve through analytic fallbacks rather than raising.
</details>

---

## Known limitations

- **Parent-relative forces only.** Cowell bodies feel only forces relative to their own parent, so
  third-body and full N-body forces are not modelled yet. Cowell bodies must be massless, and heads
  and barycentres cannot use Cowell.
- **J2 assumes a fixed spin axis.** The parent's spin axis is taken as the frame's z-axis. That is
  exact for the Earth-centred scenarios and 23.4° off in the ecliptic Sun–Earth–Moon scenario.
- **Secular J2 is first order.** Mean seeding corrects only the semi-major axis.
- **Some citations are unverified.** Several textbook equation numbers (Curtis, Vallado,
  Kozai/Brouwer) were written from memory. They are marked unverified in the source, beside the
  self-contained derivations that the tests check.
- **One known edge case.** When *every* input state is degenerate, `rv_to_coe` returns an array of the
  wrong shape. See `CLAUDE.md`.

---

## Roadmap

Ordered so that each stage makes the next one safe rather than merely possible.

1. ~~**Validation infrastructure**~~: invariant harnesses, kernel equivalence and an independent DOP853
   reference. **Done.**
2. ~~**Compiled kernels**~~: 185× on the hierarchical step, with step cost independent of arena
   capacity. **Done.**
3. **Force-model interface and events.** Bitmask composition and per-body propagator selection are
   **done**. Exact-time event handling for impulsive manoeuvres is not started.
4. ~~**Benchmark harness**~~: the sweep and the frontier plot. **Done.**
5. **Model library.** J2 (force model, secular propagator and reference) and Cowell RK4 are **done**.
   Planned: higher geopotential harmonics, atmospheric drag across fidelity tiers, solar radiation
   pressure with shadow geometry, third-body perturbations, thrust, Encke, symplectic integrators and
   an SGP4 bridge.
6. **Constellation and inter-satellite link modelling**, the application this engine is being shaped
   for.

Established external implementations are wrapped rather than reimplemented. SGP4, atmospheric density
models, planetary ephemerides and IAU frame and time transformations all have well-tested libraries,
and the interesting question here is how models *compare*, not whether they can be retyped.
