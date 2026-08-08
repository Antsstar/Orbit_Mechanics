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

CI gates on `mypy --strict` across Python 3.10 / 3.11 / 3.12. It must stay clean.

**Never run `python -m orbital_engine.simulator`.** Its `__main__` block calls `seed_test_universe()`,
which drops and rewrites the git-tracked `src/orbital_engine/data/planets.db`. To exercise the engine,
build an in-memory SQLite session the way `tests/conftest.py` does.

---

## Arena layout — `simulator.py:40-61`

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

Units throughout: **km, km/s, radians, seconds**, `mu` in km³/s².

### Invariants that are not obvious from reading the code

- `local_states[i]` is relative to `global_states[body_sys_map[i]]` — **not** to `parent_indices[i]`.
  `calc_global` (`simulator.py:445-450`) depends on this. The two graphs diverge deliberately.
- Heads carry **deliberately zeroed COEs** (`simulator.py:380`). A head's motion is the reflex kick
  about its barycenter, not an orbit. So when Earth heads the Earth-Moon system its eccentricity reads
  0.0 by design, and the heliocentric ellipse lives on the barycenter's row instead.
- Root nodes self-reference: `parent_indices[i] == i`.
- COE column 0 is the **semi-latus rectum `p`**, not semi-major axis — chosen so parabolic orbits stay
  representable. Always index via `COEIndex` (`custom_types.py:17`), never bare integers.
- Slots come off a free list (`simulator.py:47`) and are never returned. There is no despawn path.

---

## Existing tools — reuse, do not rewrite

| Module | What it already does |
|---|---|
| `frames.ReferenceFrames` | `rv_to_coe` (`frames.py:44`) and `coe_to_rv` (`frames.py:170`) — vectorised over `(N,3)`/`(N,6)`, handle circular/equatorial/polar/parabolic fallbacks, return a success mask. Also body-fixed, RaDec and long/lat transforms |
| `utilities.Transformations` | `Rx` `Ry` `Rz` `Rxyz` `Rzyx` `Rzxz` (batched `(N,3,3)` tensors), `cart_to_sphe` / `sphe_to_cart` (`utilities.py:15`) |
| `utilities.Anomalies` | Full true ↔ eccentric ↔ mean stack including hyperbolic and parabolic, Newton-Raphson and successive-substitution solvers (`utilities.py:135`) |
| `utilities.Kepler` / `utilities.Barker` | Time ↔ mean anomaly for elliptic/hyperbolic and parabolic cases (`utilities.py:441`, `:482`) |
| `simulator._topological_sort` | Vectorised BFS tier stratification — generic over any parent-index array (`simulator.py:219`) |
| `database.py` | Polymorphic ORM: `BaseBodyORM` / `CelestialBodyORM` / `VesselORM` / `VirtualBodyORM` plus `SystemORM`. `VesselORM` already carries `dry_mass`, `fuel_mass`, `drag_area` |

---

## Known-broken and in-flight

Do not treat these as intentional, and do not silently work around them.

- `Perturbations.GVP_COE` (`utilities.py:520`) is WIP with four errors: line 544 has `1.0 + e + cos θ`
  where it should be `e * cos θ`; line 552 is missing a `*` **and** uses the `ȧ` form (correct is
  `ṗ = (2 p r / h) · a_S`); line 577's second `+` should be `*`.
- `cart_to_RSW` (`frames.py:292`) is missing `@staticmethod` and its shape math adds an `int` to a
  `tuple`. `RSW_to_cart` (`frames.py:322`) is a stub typed as returning an array.
- The above are the 4 current `mypy --strict` errors.
- `Anomalies.mean_to_eccentric` (`utilities.py:332`): the `"S.S"` solver branch indexes with a
  global-length mask against active-length arrays. The `"N-R"` branch was fixed; this one was not.
- `rv_to_coe` (`frames.py:84`): the no-valid-orbit early return has shape `(N,3)` where callers expect `(N,6)`.

### Unwired scaffolding — do not build on without discussing first

`_PROPAGATOR_REGISTRY` (`registry.py`) is written and never read; `step()` (`simulator.py:459`)
hardcodes `KeplerianPropagator`; `propagator_type` and `g_env` are allocated and never read;
`BodyHandle` (`body.py`) is never instantiated and `sim.bodies` is always empty.

---

## Conventions

- Vectorised NumPy over Python loops for anything touching the arena.
- Boolean masks and integer-array indexing **copy**; assignment targets do not. Prefer basic slicing
  in hot paths, and verify with `np.shares_memory` when it matters.
- No stateful third-party objects inside a step. Dependencies may own data at the boundary
  (ingest, kernel load, coefficient extraction) and never inside `step()`.
- Signatures use the aliases in `custom_types.py`; array internals stay `NDArray[np.float64]`.
- Prefer `np.bincount` / `np.add.reduceat` over `np.add.at`, which is unbuffered and slow.

### Per-feature contract for any new physics model

1. **Citation** — reference and equation numbers
2. **Kernel** — stateless, allocation-free, array-in / array-out
3. **Registration** — its flag, so the model is immediately sweepable
4. **Validation case** — published test vector or analytic result, wired into the suite
5. **Expected error magnitude** — what "correct" looks like numerically

Item 5 is the guard that matters. The bugs listed above do not raise and do not crash; they produce
orbits that look entirely plausible. A stated expected magnitude catches them; code review does not.

### Do not reimplement

SGP4/SDP4 (use `sgp4` — it has an array API), atmospheric density (`pymsis`), planetary ephemerides
(`jplephem`), IAU frames and time scales (`pyerfa`). Wrap them at the boundary.

**Never convert mean elements ↔ osculating elements.** A TLE's mean elements are defined by SGP4's own
force model; feeding them to `coe_to_rv` is wrong by kilometres. Cartesian `(r, v)` is the only safe
interchange format between representations.

---

## Check before debugging

`docs/engineering-log.md` records problems already hit on this project and how they were resolved —
environment quirks, traps in the code, and mistakes made while working on it. Check it when
something behaves unexpectedly; it is cheaper than rediscovering. Add to it when a problem costs you
more than a few minutes.

## Do not trust

`docs/historical/` is superseded material kept for provenance only. It describes modules that were
never built and an API that no longer exists. See `docs/historical/README.md` for the specifics.
