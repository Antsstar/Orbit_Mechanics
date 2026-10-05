# OrbitalEngine

Hierarchical orbital mechanics engine built on Data-Oriented Design. Python 3.10+, NumPy, SQLAlchemy 2.0.

**Direction:** this is becoming a *model-fidelity comparison engine* — one scenario run under many model
configurations, diffed against a common reference. The benchmark sweep is the primary use case, not a
script layered on top. Design choices should favour making model configurations enumerable data.

---

## Environment

Base conda has neither pytest nor mypy. Always use the project env explicitly:

```
C:/Users/antss/miniconda3/envs/orbital_env/python.exe
```

| Task | Command |
|---|---|
| Tests | `<env>/python.exe -m pytest -q` |
| Types | `<env>/python.exe -m mypy src/ --strict` |
| Coverage | `<env>/python.exe -m pytest -q --cov=orbital_engine --cov-report=term-missing` |
| Benchmarks | `<env>/python.exe benchmarks/bench_step.py` |
| Skip timing tests | `<env>/python.exe -m pytest -q -m "not perf"` |

CI gates on `mypy --strict` across Python 3.10 / 3.11 / 3.12, plus a second job that installs
*without* numba to gate the fallback path. Both must stay clean.

**Check CI after every push — a local green run is not enough.** The runners resolve their own
numpy, so `mypy --strict` can fail there while passing here (local numpy 2.4.6, mypy 2.1.0):

```
gh run list --limit 3                    # newest runs, with status
gh run view <run-id> --log-failed        # only the failing steps
```

The recurring case is `redundant-cast`: `cast(ArrayFloat, np.linalg.norm(...))` is needed where the
stub returns `Any` and redundant where it does not. Prefer an **annotated assignment**
(`out: ArrayFloat = np.linalg.norm(...); return out`), which satisfies both.

The other recurring case is an array built **without an explicit dtype**: older stubs type
`np.linspace(a, b, n)` as `floating[Any]` and a fancy-indexed integer array as `signedinteger[Any]`,
which then fails against an `NDArray[np.float64]` / `NDArray[np.int64]` target on the Python 3.10 job
only. Pass `dtype=np.float64` (or wrap in `np.asarray(..., dtype=...)`) and annotate the target. Two
agent merges in a row (`isl.py`, `msis_diurnal.py`) failed CI on exactly this after passing locally.

**Coverage is under-reported for `kernels.py`.** `coverage.py` traces bytecode, and `@njit` functions
run as machine code, so the compiled run shows ~15% for a module that is ~88% covered. Measure it
with numba disabled — see `docs/engineering-log.md`. Do not write tests to chase that phantom gap.

Optional extras: `[perf]` = numba, `[reference]` = scipy, `[sgp4]` = sgp4, `[msis]` = pymsis. None is needed to run the engine.

**Never run `python -m orbital_engine.simulator`.** Its `__main__` block calls `seed_test_universe()`,
which drops and rewrites the git-tracked `src/orbital_engine/data/planets.db`. To exercise the engine,
build an in-memory SQLite session the way `tests/conftest.py` does.

**Working in a git worktree?** The package is an editable install of the *main* checkout, so a bare
`pytest` in a worktree imports the main repo's code and passes on changes it never exercised. Run
`PYTHONPATH="$(pwd)/src" <env>/python.exe -m pytest -q` from the worktree root.

---

## Arena layout — `Simulation.__init__`

All state lives in flat pre-allocated arrays indexed by an integer slot. Nothing owns per-body state.

| Array | Shape | Meaning |
|---|---|---|
| `mu_array` | `(C,)` | GM, km³/s². For a barycenter this holds the *summed* system mass after `_recalculate_all_barycenters` |
| `local_states` | `(C,6)` | `[x y z vx vy vz]` relative to the body's **system bubble** |
| `global_states` | `(C,6)` | Same, relative to the simulation root |
| `coe_states` | `(C,6)` | `[p e i Ω ω θ]` — see `COEIndex` |
| `parent_indices` | `(C,)` int32 | Keplerian graph: what the COE is measured against |
| `body_sys_map` | `(C,)` int32 | Kinematic graph: what `local_states` is measured against |
| `sys_head_map` | `(C,)` int32 | O(1) lookup to the head of each local system bubble |
| `active_mask` / `is_system` / `is_head` | `(C,)` bool | Slot filters |
| `force_model_mask` | `(C,)` uint64 | One bit per registered force model, OR'd per body. Enabled physics is data |
| `accel_accum` | `(C,3)` | Shared acceleration accumulator, km/s², arena-owned scratch |
| `force_model_params` | `dict[name → (C,k)]` | Per-model, per-body coefficients, allocated on first resolve |

Units throughout: **km, km/s, radians, seconds**, `mu` in km³/s².

### Invariants that are not obvious from reading the code

- `local_states[i]` is relative to `global_states[body_sys_map[i]]` — **not** to `parent_indices[i]`.
  `Simulation.calc_global` depends on this. The two graphs diverge deliberately.
- Heads carry **deliberately zeroed COEs** (`_rehydrate_coes`). A head's motion is the reflex kick
  about its barycenter, not an orbit. So when Earth heads the Earth-Moon system its eccentricity reads
  0.0 by design, and the heliocentric ellipse lives on the barycenter's row instead.
- Root nodes self-reference: `parent_indices[i] == i`.
- COE column 0 is the **semi-latus rectum `p`**, not semi-major axis — chosen so parabolic orbits stay
  representable. Always index via `COEIndex` (`custom_types.py`), never bare integers.
- Slots come off a free list (`free_indices`) and are never returned. There is no despawn path.
- **`Simulation.accelerations(t, state)` returns `accel_accum` itself, not a copy.** An integrator that
  holds more than one stage result (every RK method) must copy each before requesting the next, or all
  stages alias one buffer and the integration is silently wrong.
- **Call `resolve_force_models()` after any change to `force_model_mask` or `active_mask`.**
  `enable_force_model` does it for you. It also clears `accel_accum`: `compose_accelerations` only
  writes rows in the current dispatch set, so without that clear a disabled model's last acceleration
  would persist in its body's row. It also rebuilds the Cowell dispatch plan (`_refresh_cowell_plan`),
  which decides whether the fused compiled kernel applies; `set_propagator` rebuilds it too.
- **A Cowell body integrates its state relative to its `parent_indices` parent, never its absolute
  global state.** `step()` saves that relative state and re-bases it onto the parent's *end-of-step*
  position after `calc_global()`. Integrating the absolute state with parent-relative forces silently
  assumes the parent is fixed. `two_body` cannot catch that, because its primary never moves; use
  `sun_earth_moon(moon_mu=0.0)`, whose Earth accelerates toward the Sun.
- **`step()` splits at a scheduled manoeuvre epoch *and* at an event crossing** (`events.py`), and
  `_advance` is the unsplit step both go through. A step with no manoeuvre queued and no event
  registered short-circuits to `_advance` and is bit-identical to the pre-event engine; so is a step
  with events registered but no crossing, by construction, because the detection is two pure reads
  around the *same* single `_advance` call and the speculative advance is kept rather than repeated.
- **`step()` burns propellant after propagating, never during.** `thrust.deplete_mass` runs on the
  cached `_thrust_idx` at the end of `step()`, so every RK4 stage of one step reads the mass that step
  began with. Anything that changes `force_model_mask` must go through `resolve_force_models()`, which
  is what rebuilds `_thrust_idx` — writing the bit directly leaves the burn silently not happening.
- **A Cowell body's `coe_states` row is stale**, left at whatever it held before reassignment. Read
  its `(r, v)` from `global_states`.
- Change propagators only through `set_propagator`, which validates and calls
  `_refresh_active_indices`. Writing `propagator_type` directly leaves the dispatch index arrays stale.
- **A `SECULAR_J2` body's `(j2, r_eq)` coefficients live in `force_model_params["j2"]`** — the same
  array `enable_force_model("j2", ...)` populates for Cowell — but `set_propagator` never sets the
  `"j2"` mask bit, so the body is not added to `accelerations()`'s dispatch set. The three secular
  rates (`[dRAAN/dt, dARGPE/dt, dM/dt]`) are derived once at `set_propagator` time and cached in
  `_secular_j2_rates`, not recomputed per step: `p`, `e`, `i` are constant under first-order secular
  J2 theory. A `SECULAR_J2` body's propagated position is only relative to `parent_indices` until
  `step()` re-bases it onto the parent's end-of-step `global_states`, exactly like a Cowell body.

---

## Existing tools — reuse, do not rewrite

One line per module: what it is, plus the trap that has bitten. **The full rows - coefficients,
validation figures, headline numbers, design reasoning - are in `docs/module-reference.md`.** Read a
module's row there before changing that module or writing a test against it.

| Module | What it does — and the trap |
|---|---|
| `frames.ReferenceFrames` | `rv_to_coe` / `coe_to_rv` (vectorised, degenerate fallbacks, success mask); body-fixed, RaDec, long/lat; RSW: `RSW_basis` (rows R, S, W), `cart_to_RSW`, `RSW_to_cart`. Undefined RSW frames return exactly zero; its rectilinear test is relative, `rv_to_coe`'s is absolute |
| `utilities` | `Transformations` (batched `Rx`..`Rzxz`, spherical), `Anomalies` (true/eccentric/mean incl. hyperbolic and parabolic; `"S.S"` diverges on hyperbolic, use `"N-R"`), `Kepler` / `Barker` |
| `iod.py` | `lambert` (bisection, zero-rev), `gibbs`, `herrick_gibbs`, `gauss` (+ improvement). Two-body, boundary only. **Gibbs ignores time**: on a perturbed arc prefer Lambert or Herrick-Gibbs |
| `simulator._topological_sort` | Vectorised BFS tiering over any parent-index array |
| `database.py` | Polymorphic ORM (`CelestialBodyORM` / `VesselORM` / `VirtualBodyORM`, `SystemORM`) |
| `kernels.py` | The compiled twins — see the two-implementation rule |
| `scenarios.py` | Every scenario builder. **Build scenarios here, never inline in a test.** `sun_earth_moon(moon_mu=0.0)` is the accelerating-parent case `two_body` cannot catch; heliocentric arenas are round-off limited at ~1e-5 km, so convergence tests use Earth-rooted ones (`eclipsed_satellite`). Twin-control builders: `powered_vessel`, `hohmann_pair`, `zonal_twins`. Also `earth_constellation`, `ground_station_pass`, `station_keeping_satellites`, `sun_synchronous_satellites`, `geostationary_satellites`, `coplanar_satellites`, `tle_satellites`, `artemis2`, `artemis3_rendezvous` |
| `reference.py` | Independent DOP853 truth. J2 / zonal / tesseral are passed **explicitly**, never read from `force_model_params`, and omitting them is bit-identical. Frontier truth uses `rtol=TRUTH_RTOL, atol=TRUTH_ATOL`. `energy_drift` is mu-weighted, so it certifies nothing about massless bodies |
| `forces.py` | `ForceKernel` contract (additive `+=`, stateless, allocation-free) and `AccelerationProvider`. Read its docstring before writing a force model |
| `registry.py` | `@register_force_model` assigns mask bits in registration order — **sweep configs persist model names, never mask integers**. `validate_bodies` / `validate_coefficients` hooks run before any bit is set |
| `gravity.py` | `"point_mass_gravity"`: parent-only central term with summed mu. Fused |
| `geopotential.py` | `"j2"`. Pair with `EARTH_R_EQ` (6378.137), **never** `scenarios.EARTH_RADIUS` (6371, mean). Spin axis = frame +z. Refuses barycentre parents. Fused |
| `zonal.py` | `"zonal"`: J3..J6, **additive to `"j2"`**, never a replacement. EGM96 values from memory, unverified. Fused |
| `tesseral.py` | `"tesseral"`: orders m >= 1 to degree 4, additive to j2 + zonal. **Reads `t`** (`theta = theta0 + omega t`, absolute sim time); refuses `omega == 0` on a live row. Fused |
| `drag.py` / `atmosphere.py` | `"drag"`, density law chosen per body by the `density_model` coefficient (0 exponential = the all-zero default, 1 Vallado table, 2 averaged NRLMSIS, 3 diurnal NRLMSIS). `pymsis` runs only at configuration time (`msis_bridge.py`, `msis_diurnal.py`), never in a step. Laws 0-2 fused; a law-3 row sends Cowell to NumPy. Table transcribed from memory |
| `srp.py` | `"srp"`, cannonball, anti-sunward; `source` and `p_srp` mandatory. **Prefer the conical shadow**: the cylindrical terminator costs RK4 its order unless `sim.add_event(events.shadow_event(sim))`. Column 7 is engine-owned. Not fused |
| `thirdbody.py` | `"third_body"`: one arena perturber, direct minus indirect; **first order** (perturber frozen within the step). Not fused |
| `ephemeris.py` | `"ephemeris_third_body"`: tabulated perturbers (Hermite) at every RK4 stage time — fourth order. Tables are centred on the parent; out-of-range queries raise. Not fused |
| `thrust.py` | `"thrust"`, Cowell-only, RSW direction (norm throttles). **Mass is state**: re-seed `mass_kg` to re-run. Not fused |
| `manoeuvres.py` | Impulsive Δv in RSW under every propagator; `schedule_delta_v` splits the step at the epoch. Keplerian/secular-J2 re-derive elements (and rates). Not a registry entry |
| `events.py` | Event-driven step splitting (Illinois false position over trial propagations) with `Event.latch`, plus `Event.action` / `max_fires` (`apsis_event`, `node_event`, `burn`, `enable_model`). Blind to an even number of crossings in one step. An action must not add or clear events |
| `integrators.py` | `RK4Integrator`: copies each stage's accelerations (the provider returns a shared buffer). Not allocation-free |
| `propagators.py` | `SecularJ2Propagator` (`PropagatorType.SECULAR_J2`, coefficients mandatory); `mean_seed=True` corrects `p` only |
| `geometry.py` | Elevation / azimuth / range rate (**positive = opening**), access windows, `line_of_sight`, `segment_clearance`. Spherical, radians, azimuth from north. **`theta0` is referenced to `epoch_s`** (default absolute sim time), unlike `viz.ground_track` |
| `access.py` / `isl.py` | Error in contact windows (ground / inter-satellite) and the exported contact datasets. **`sample_dt_s` must be an integer multiple of every config's `dt`.** `h_graze_km` is required. ISL dataset edges are biased at 60 s; sample at 15 s for export |
| `sweep.py` | `run_sweep`: configurations as data against one truth; `access=`, `isl=`, `station_keeping=` + `delta_v_baseline=` (or per config: `ModelConfig.station_keeping`), `zonal=`, `tesseral=`, `external=` (e.g. SGP4 tiers) |
| `stationkeeping.py` | Dead-band altitude controller keyed on mean altitude; `observe()` after every step. The Δv sweep metric needs a named baseline config |
| `viz.py` | Plot data, no matplotlib. **`sample_states` advances the simulation.** Body-fixed rotation takes `-theta` |
| `sgp4_bridge.py` | SGP4 wrapped, never reimplemented; TEME taken as inertial. A TLE's mean elements **never** reach `coe_to_rv`. SGP4 is an `ExternalTier`, not a `PropagatorType` |
| `horizons_bridge.py` / `artemis2.py` | JPL Horizons at the boundary (network only in `fetch`); Artemis II data in `data/artemis2/` (TDB seconds from `EPOCH_TDB`). ICRF +z is 0.147 deg from the true pole |
| `artemis2_replay.py` | Artemis II under four tiers vs NASA's navigation data; `export_dashboard` writes `demo/artemis2/data/` |
| `artemis3.py` | Prospective Artemis III LEO rendezvous: planners of rising fidelity, plans flown in the truth tier. Every orbit/vehicle number beyond ~430 km / 33 deg is an assumption stated in its docstring |
| `benchmark.py` | `measure(fn)`: min-of-batches timing |
| `Simulation` model API | `enable_force_model(name, bodies, **coefficients)`, `set_propagator(bodies, PropagatorType.COWELL \| SECULAR_J2, ...)`, `resolve_force_models()`, `accelerations(t, state=None)`. Cowell is refused on heads, barycentres, roots, inactive slots and **any body with `mu != 0`** |

**Fused compiled Cowell set:** `point_mass_gravity`, `j2`, `drag` (laws 0-2), `zonal`, `tesseral`. Any
other bit on any Cowell body, or a Cowell body parenting another, sends the whole Cowell set down the
NumPy path.

---

## Known-broken and in-flight

### On `main` — live

- `rv_to_coe`: the no-valid-orbit early return has shape `(N,3)` where callers expect `(N,6)`.
  Only reachable when *every* input state is degenerate, which is why it has not bitten yet.

`main` is `mypy --strict` clean and all four CI jobs are green. Do not go looking for the items below
on this branch — they are not here.

### On `feature/vop-propagator` only — not on `main`

That branch carries the WIP Gauss variational work and is the source of the 4 `mypy --strict` errors.
Rebase it onto `main` before resuming (its workflow file predates the all-branches CI trigger, so CI
does not currently run on it):

- `Perturbations.GVP_COE` has four equation errors: `1.0 + e + cos θ` where it should be `e * cos θ`;
  a missing `*` combined with the `ȧ` form where the correct one is `ṗ = (2 p r / h) · a_S`; and a
  `+` that should be `*`.
- Its `cart_to_RSW` / `RSW_to_cart` are broken **and superseded**: correct, tested versions now live on
  `main` with a different signature (see the tools table). When rebasing, drop the branch's versions
  rather than resolving conflicts in their favour.

Fixed in phase 1 (`Anomalies.mean_to_eccentric`, rewritten): the `"S.S"` global-length mask indexing
bug, and a silent-NaN path where a diverging iterate returned NaN *reporting success* — `abs(nan) > tol`
is `False`, so the element was dropped from the active set as converged. Both are now covered by
`tests/unit/test_anomalies.py`. `"S.S"` on hyperbolic orbits is formally divergent and raises by
design; use `"N-R"` there.

### Unwired scaffolding — do not build on without discussing first

Both halves of `registry.py` are now wired. `step()` dispatches Cowell bodies through
`_PROPAGATOR_REGISTRY` and reads `propagator_type`, which `set_propagator` writes. Only `KEPLERIAN`,
`COWELL` and `SECULAR_J2` drive dispatch; any other `PropagatorType` member is unimplemented. `BodyHandle`
(`body.py`) is still never instantiated and `sim.bodies` is always empty.

Cowell's compiled twin is **fused**: `kernels.cowell_rk4_step` hard-codes RK4 with `point_mass_gravity`,
`j2`, `drag` (all three density laws), `zonal` and `tesseral` (per-body flags; `t` is an argument), because numba cannot dispatch over the Python kernel list
`forces.compose_accelerations` walks. A Cowell body carrying any other model, or parented by another
Cowell body, sends the whole Cowell set down the NumPy `RK4Integrator` path (`_cowell_fused_ok`). The
composition layer itself, and every other force model, remain NumPy only; a new force model that
wants to be timed fairly against the analytic tiers needs its term added to `kernels._cowell_accel`
and the plan's accepted-bit set. `test_constant_accel` and `test_radial_bias` are deliberately not
in that set.

Two test-fixture kernels, `test_constant_accel` and `test_radial_bias`, are registered from
`forces.py` itself, so they occupy two of the 64 mask bits and appear in `all_force_models()` for
every user. They are fixtures, not physics — do not sweep over them.

---

## Conventions

- Vectorised NumPy over Python loops for anything touching the arena — **except inside `kernels.py`**,
  see below.
- Boolean masks and integer-array indexing **copy**; assignment targets do not. Prefer basic slicing
  in hot paths, and verify with `np.shares_memory` when it matters.
- No stateful third-party objects inside a step. Dependencies may own data at the boundary
  (ingest, kernel load, coefficient extraction) and never inside `step()`.
- Signatures use the aliases in `custom_types.py`; array internals stay `NDArray[np.float64]`.
- Prefer `np.bincount` / `np.add.reduceat` over `np.add.at`, which is unbuffered and slow.
- Avoid `if np.any(mask):` as a guard around masked work on arena-sized arrays. Below ~1000 elements
  the guard costs more than the work it skips. Where a loop needs an active set, hold it as an
  **integer index array** so the test is `active.size`, not a NumPy reduction.

### The two-implementation rule

Hot paths exist twice: a readable NumPy version and a compiled scalar version.

| | Reference | Compiled |
|---|---|---|
| Propagation | `propagators.KeplerianPropagator` | `kernels.kepler_propagate` |
| Secular-J2 propagation | `propagators.SecularJ2Propagator` | `kernels.secular_j2_propagate` |
| Cowell step (RK4 + `point_mass_gravity` + `j2` + `drag` + `zonal` + `tesseral`) | `integrators.RK4Integrator` over `forces.compose_accelerations` | `kernels.cowell_rk4_step` (fused; other masks fall back to the reference) |
| Global states | `Simulation.calc_global` (else branch) | `kernels.calc_global_states` |
| Re-base of Cowell / secular-J2 rows after `calc_global` | `Simulation._rebase` (else branch) | `kernels.rebase_relative_states` (bit-identical; NumPy path if a body's parent is in its own re-base set) |

`Simulation.use_compiled_kernel` selects between them; it defaults to `NUMBA_AVAILABLE`, because
without numba the kernels run as *interpreted* Python and are slower than the NumPy path.

Rules when touching either side:

1. **Change both, or neither.** They are held equivalent by
   `tests/validation/test_kernel_equivalence.py` — elementwise to 1e-12 relative for propagation,
   and *bit-identical* for `calc_global`.
2. **The reference is the definition.** It is optimised for being obviously correct. Do not
   micro-optimise it; that is what the compiled path is for.
3. **Scalar loops belong only in `kernels.py`.** The inversion of the house style is deliberate and
   confined there.
4. **`fastmath` stays off.** It licenses reassociation and assumes no NaN — and the Kepler solver
   detects divergence *by* testing for non-finite values.
5. Kernels allocate nothing. Scratch (`_kick`, `_accum`) is arena-owned and zeroed per active slot,
   never per capacity. That is what keeps step cost independent of `max_capacity`
   (`tests/validation/test_scaling_invariants.py`).

### Verification vs comparison — do not conflate them

`reference.py` produces DOP853 truth trajectories. What a disagreement with it *means* depends
entirely on the case:

- **Verification** — the model is exact (two-body, with or without a massive secondary). The two must
  agree to ~1e-8 relative. Disagreement is an engine bug.
- **Comparison** — the model is an approximation (Sun-Earth-Moon neglects the solar term in the lunar
  orbit). The two *must* diverge; the divergence is the result. Currently 3.3e4 km over 30 days,
  bounded above by the coherent-forcing estimate 0.5·a·t² ≈ 1.0e5 km.

Treating a comparison divergence as a bug leads to "fixing" a correct engine.

### Per-feature contract for any new physics model

1. **Citation** — reference and equation numbers
2. **Kernel** — stateless, allocation-free, array-in / array-out
3. **Registration** — its flag, so the model is immediately sweepable
4. **Validation case** — published test vector or analytic result, wired into the suite
5. **Expected error magnitude** — what "correct" looks like numerically

Item 5 is the guard that matters. The bugs listed above do not raise and do not crash; they produce
orbits that look entirely plausible. A stated expected magnitude catches them; code review does not.

### Do not reimplement

SGP4/SDP4 (use `sgp4` — it has an array API), atmospheric density (`pymsis` - wrapped in `msis_bridge.py`), planetary ephemerides
(`jplephem`), IAU frames and time scales (`pyerfa`). Wrap them at the boundary.

**Never convert *externally defined* mean elements ↔ osculating elements.** A TLE's mean elements are
defined by SGP4's own force model; feeding them to `coe_to_rv` is wrong by kilometres. Cartesian
`(r, v)` is the only safe interchange format between representations from an external source such as
SGP4/TLE. This does not extend to a propagator applying a first-order short-period correction that
belongs to *its own* theory to its *own* seed — `propagators.mean_seeded_p` (`SecularJ2Propagator`'s
`mean_seed` option) is the one case in the engine, and it is scoped narrowly: it corrects only the
semi-latus rectum (and so the cached mean motion), never `e` or `i`, and never reads an externally
supplied mean element.

---

## Why the engine is shaped this way

`docs/architecture.md` holds the reasoning this file only summarises — the module map, why the two
graphs diverge, why the reflex kick uses a real head rather than a virtual one (time-invariance), and
what is deliberately not built. Read it before proposing a structural change; several decisions that
look accidental are load-bearing.

## Check before debugging

`docs/engineering-log.md` records problems already hit on this project and how they were resolved —
environment quirks, traps in the code, and mistakes made while working on it. Check it when
something behaves unexpectedly; it is cheaper than rediscovering. Add to it when a problem costs you
more than a few minutes.

## Do not trust

`docs/historical/` is superseded material kept for provenance only. It describes modules that were
never built and an API that no longer exists. See `docs/historical/README.md` for the specifics.
