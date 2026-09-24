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

| Module | What it already does |
|---|---|
| `frames.ReferenceFrames` | `rv_to_coe` and `coe_to_rv` — vectorised over `(N,3)`/`(N,6)`, handle circular/equatorial/polar/parabolic fallbacks, return a success mask. Also body-fixed, RaDec and long/lat transforms |
| `frames.ReferenceFrames` RSW | `RSW_basis(r, v)` → `(N,3,3)` with rows R, S, W; `cart_to_RSW(r, v, vec)` and `RSW_to_cart(r, v, vec_rsw)`. A `(3,)` direction such as prograde `[0,1,0]` broadcasts over N states. Success mask for rectilinear, zero or non-finite reference states — those rows come back exactly zero. The rectilinear test is *relative* (`|r×v| > tol·|r||v|`), unlike `rv_to_coe`'s absolute `|h|`, so the two masks can disagree near the boundary |
| `utilities.Transformations` | `Rx` `Ry` `Rz` `Rxyz` `Rzyx` `Rzxz` (batched `(N,3,3)` tensors), `cart_to_sphe` / `sphe_to_cart` |
| `utilities.Anomalies` | Full true ↔ eccentric ↔ mean stack including hyperbolic and parabolic, Newton-Raphson and successive-substitution solvers |
| `utilities.Kepler` / `utilities.Barker` | Time ↔ mean anomaly for elliptic/hyperbolic and parabolic cases |
| `simulator._topological_sort` | Vectorised BFS tier stratification — generic over any parent-index array |
| `database.py` | Polymorphic ORM: `BaseBodyORM` / `CelestialBodyORM` / `VesselORM` / `VirtualBodyORM` plus `SystemORM`. `VesselORM` already carries `dry_mass`, `fuel_mass`, `drag_area` |
| `kernels.py` | Compiled scalar kernels: `kepler_propagate`, `secular_j2_propagate`, `cowell_rk4_step`, `calc_global_states`, plus reusable `coe_to_rv_scalar` / `solve_kepler_scalar` / `advance_true_anomaly` |
| `scenarios.py` | `two_body`, `sun_earth_moon(moon_mu=…)`, `earth_constellation(n_sats=…)`, `powered_vessel(n_vessels=…, n_powered=…)`, `hohmann_pair()` (Earth plus four co-located massless vessels on one circular orbit — `KEPLER-SAT`/`COWELL-SAT` manoeuvre, `*-TWIN` are the controls; deliberately inclined and rotated so an RSW-vs-inertial frame error cannot hide) — shared by tests and benchmarks. `moon_mu=0.0` gives a massless Moon with a genuinely accelerating parent. `powered_vessel` returns Earth plus N co-located massless Cowell vessels, the first `n_powered` carrying `"thrust"` and the rest acting as **thrust-free twins** in the same arena (that is how a burn is separated from the gravity turn, and how integrator drift is measured). `eclipsed_satellite(n_sats=, altitude_km=, inclination_deg=)` seeds **Earth at the arena root** plus a massless luminous marker (`LIGHT_SOURCE_NAME`) 1 AU out on +x and satellites that pass through Earth's umbra once per orbit - it exists because a Cowell body in a *heliocentric* arena loses nine digits to cancellation and is round-off limited at **~1e-5 km** over 1.5 LEO orbits, so RK4's fourth order is **not observable** in `sun_earth_moon` at all, shadow or no shadow. `ground_station_pass(n_sats=, altitude_km=, inclination_deg=)` seeds Earth plus Keplerian vessels at true anomaly 180 deg for `geometry.py`'s closed-form pass, with companion constants `STATION_LATITUDE_DEG` / `STATION_LONGITUDE_DEG` / `STATION_ALTITUDE_KM` (the station is not a body — it lives in `geometry.py`'s arguments); its equatorial default is the only geometry with an exact answer, and its `inclination_deg` exists because that same symmetry hides a transposed SEZ axis. `zonal_twins(arg_pe_deg=(...), copies=2, semi_major_axis_km=, eccentricity=, inclination_deg=)` seeds `copies` co-located Cowell + pm + `j2` satellites per perigee (`zonal_twin_name(orbit, copy)`), copy 0 the J2 control - the differential idiom `zonal.py`'s orbit-dynamics tests need. **Build scenarios from here, never inline in a test** |
| `reference.py` | `reference_for(sim, times, oblateness={"Earth": (j2, r_eq)})` → DOP853 N-body truth trajectory, optionally with J2 on named bodies (reaction included, spin axis = frame +z). Independent of all engine code: oblateness is passed **explicitly**, never read from `force_model_params`, and the field is written from a different derivation than `geopotential.py`. Omitting it is bit-identical to the point-mass truth. `zonal={"Earth": (r_eq, {3: j3, 4: j4, 5: j5, 6: j6})}` adds J3..J6 the same way (explicit per-degree monomials, a different derivation from `zonal.py`'s recursion; degree 2 refused - J2 is `oblateness`'s) and is bit-identical when omitted; `run_sweep(zonal=)` forwards it. **For frontier-plot truth pass `rtol=TRUTH_RTOL, atol=TRUTH_ATOL` (1e-13, 1e-12)**: the default `atol=1e-9` governs LEO velocity components and costs ~4x in truth error (3.5e-9 against 8.6e-10 km per orbit). `energy_drift` is mu-weighted, so it certifies nothing about massless satellites |
| `benchmark.py` | `measure(fn)` → min-of-batches timing with noise ratio |
| `forces.py` | Force-model composition. `ForceKernel` is the contract a force model implements — additive (`+=`, never `=`), stateless, allocation-free, called once per model per evaluation. `AccelerationProvider` is what an integrator consumes: `(t, state) -> (C,3)`, with `state` explicit so RK sub-stages never touch the arena. Read the module docstring before writing a force model |
| `registry.py` | `@register_force_model(name, param_names=…, citation=…)` assigns the next uint64 mask bit. Optional `validate_bodies=` hook (`BodyValidator`), run by `enable_force_model` before any bit is set, for conditions a kernel cannot see. `get_force_model` / `all_force_models` / `mask_for`. Bits follow registration order within a process, so **sweep configs persist model names, never raw mask integers** |
| `geopotential.py` | Force model `"j2"` (`J2_MODEL`): the J2 perturbation only, relative to each body's `parent_indices` parent — not the central `-mu r/r^3` term. Coefficients `(j2, r_eq)` live on the *perturbed* body's row. Earth values `EARTH_J2` and `EARTH_R_EQ` = 6378.137 km (equatorial) — do **not** pair J2 with `scenarios.EARTH_RADIUS` (6371 km, mean), which is 0.2% off. Assumes the parent's spin axis is the frame's +z. Registered on `import orbital_engine`. The kernel cannot see a barycentre parent (it has no `is_system`), so `"j2"` registers `barycentre_parented` as its `validate_bodies` hook: `enable_force_model("j2", ...)` raises `ValueError` for such bodies before setting any bit. Writing `force_model_mask` directly bypasses it. Compiled twin: fused into `kernels.cowell_rk4_step` with RK4 and `point_mass_gravity` |
| `zonal.py` | Force model `"zonal"` (`ZONAL_MODEL`): the J3..J6 zonal perturbation of the `parent_indices` parent, **additive to `"j2"`, never a replacement** - a J2..J6 config enables both, `"j2"` keeps its fused twin and `SECULAR_J2` keeps reading `force_model_params["j2"]`. Coefficients `(r_eq, j3, j4, j5, j6)` (unnormalised `J_n`) on the perturbed body's row; spin axis = frame +z. Kernel: Legendre recursions, `a_n = (mu/r^2) J_n (R/r)^n [P'_{n+1}(s) r_hat - P'_n(s) z_hat]`; a zero `J_n` adds exactly nothing. EGM96 `EARTH_J3`..`EARTH_J6` (`EARTH_ZONALS` is the keyword set) from `EGM96_CBAR_N0` via `J_n = -sqrt(2n+1) Cbar_n0`, **from memory, unverified** - the same table's `Cbar_20` reproduces `EARTH_J2` within its printed digits (`EARTH_J2` is the truncation, 5.5e-14 low). Refuses barycentre parents, non-finite coefficients and `r_eq <= 0` on any row whose effective `J_n` is non-zero. **Not in the fused plan**: Cowell + zonal runs NumPy (130x the fused `j2` tier at 15 s), so its timings are not comparable until the `kernel-twin` follow-up. Validated in `tests/validation/test_zonal.py`: against `reference.zonal_field` (<= 8.9e-14 of scale), a finite-differenced potential, bitwise parity, fourth-order convergence on zonal truth, and differentially against J2 twins - J3 `de/dt` +0.42 %, J4 `dRAAN/dt` +0.97 % against a 2 % budget. Headline (`benchmarks/zonal_sweep.py`, 12 sats, 550 km / 53 deg, 24 h, J2..J6 truth): omitting J3..J6 costs **1.26 km** median (RK4 at 15 s: 2.7e-3 km; at 60 s the 1.88 km truncation is larger) but only **0.07 s mean / 0.59 s max** rise shift, no pass lost - under the 1 s pad |
| `drag.py` | Force model `"drag"`: `-(1/2) rho B |v_rel| v_rel`, with `v_rel = v - w x r` relative to `parent_indices` and the atmosphere co-rotating about the frame's +z (the same assumption as `j2`). Per-body coefficients are `(ballistic_coeff m^2/kg, rho0 kg/m^3, h0 km, scale_height km, r_ref km, omega rad/s, density_model, f107, f107a, ap)` - the last three read **only** on `DENSITY_MODEL_MSIS` rows, so a legacy 7-column array still works, and the one unit conversion (`rho*B` from 1/m to 1/km, x1e3) sits in the kernel. `EARTH_OMEGA` is provided. `scale_height <= 0` means a silent zero contribution **under the single-band law only**. Refuses barycentre parents via `validate_bodies`; `validate_coefficients` refuses an unknown `density_model`, MSIS without **all three** indices in the same call (so `pymsis` never downloads them), solar indices on a non-MSIS row, and out-of-range indices - then evaluates the MSIS profile, before any bit is set. Not in the fused compiled Cowell twin, so a Cowell body with drag sends the **whole Cowell set** down the NumPy path. Validated against orbit-averaged decay, energy balance and the co-rotation factor (`tests/validation/test_drag.py`) |
| `atmosphere.py` | The density law `"drag"` uses, selected per body by its `density_model` coefficient - **not** a second force model and **not** a mask bit, so it is one more key in a `sweep.ForceModelSpec`'s `coefficients`. `DENSITY_MODEL_EXPONENTIAL` = 0.0 is the single band and is what an **unwritten (all-zero) row already selects**, so every pre-existing configuration is unchanged; `DENSITY_MODEL_LAYERED` = 1.0 is Vallado 4e Table 8-4, 28 piecewise-exponential bands from 0 to 1000+ km, evaluated by one vectorised `np.searchsorted`. Under the layered law `rho0`, `h0` and `scale_height` are **ignored**. The table is **transcribed from memory and unverified against the text**; its internal check is continuity, which 27 of 28 boundaries pass inside 1e-4 - the 0-25 km boundary closes to 1.4e-3 against a 7.7e-4 rounding bound and is asserted as a **named anomaly**, so treat `H = 7.249 km` as the table's least trustworthy entry. Extrapolates the terminal bands at both ends; a masked row returns exactly 0.0 without evaluating `exp`. No compiled twin (drag has none to fuse into). The headline comparison: over 3 days from 355 km at `B = 0.4 m^2/kg` the table predicts **27.1 km more decay** than a single band matched at 355 km with `H = 60 km`, 30 % more, each within 1.9e-4 of its own orbit-averaged mean ODE (`tests/validation/test_atmosphere.py`). **`DENSITY_MODEL_MSIS` = 2.0 is NRLMSIS 2.0 via `msis_bridge.py`** (`[msis]` extra, `pymsis` imported lazily, `version=2.0` pinned): `pymsis` is **never called in a step** - `drag`'s `validate_coefficients` evaluates, once per `(f107, f107a, ap)`, the global mean over latitude (8 Gauss-Legendre), local time (8) and day of year (24) at 601 nodes 0-1000 km (1.3 s), **memoised as a pure function of the triple**; the kernel evaluates it log-linearly through the shared `piecewise_exponential_density` (+29 us/call) and **raises `LookupError`** on an unevaluated triple rather than evaluating lazily. Presets `SOLAR_ACTIVITY_LOW/_MODERATE/_HIGH` (ECSS 65/65/0, 140/140/15, 250/250/45, from memory), `msis_coefficients(...)` for a `ForceModelSpec`. **The average discards** the diurnal bulge (2.30x at 400 km), the season (1.63x) and index time variation - a model choice. Legacy rows **bit-identical** (frozen pre-MSIS kernel; 2000-step run). Table vs moderate MSIS, 150-1000 km: **0.79 (800 km) to 1.28 (160 km)**, a plausibility check, not verification. **Headline** (`benchmarks/msis_sweep.py`, 6 days, station-keeping scenario, baseline MSIS moderate 3.047 m/s/day): solar low **-71.9 %**, high **+134.4 %**, table +12.8 %, single band -3.1 %, each within 6e-4 of its pre-run prediction - **solar activity is 5-10x the atmosphere-model choice**, and the static laws straddle moderate MSIS. Validated in `tests/validation/test_msis.py` / `_wiring.py` / `_delta_v.py` |
| `srp.py` | Force model `"srp"` (`SRP_MODEL`): cannonball solar radiation pressure, `a = -nu * cr * p_srp * (A/m) * (AU/d)^2 * u_hat` with `u_hat` pointing body->source, so the push is **anti-sunward**. The light source is a float slot in column `source` and is **mandatory**, named in sweeps as `ForceModelSpec(..., body_coefficients={"source": "Sun"})` exactly like `third_body`'s perturber; `p_srp` is mandatory too (`SOLAR_PRESSURE_1AU` = 4.5598e-6 N/m^2 = 1367/c; the IAU 2015 TSI 1361 gives 0.44 % less). Coefficients `(cr, area_mass m^2/kg, p_srp N/m^2, source, r_occ km, r_source km, shadow_model)`; the one unit conversion is `1e-3`. **The occulter is the Keplerian parent**; `r_occ <= 0` disables the shadow (the heliocentric case). `shadow_model` selects the geometry the way `drag`'s `density_model` selects a density law: `SHADOW_MODEL_CYLINDRICAL` = 0.0 (the all-zero default, `nu` exactly 0 or 1) or `SHADOW_MODEL_CONICAL` = 1.0 (apparent-disc overlap, umbra/penumbra/annular, continuous). **The cylinder's terminator is a genuine discontinuity and costs RK4 its order** - measured step-halving ratio 1.96 where a smooth problem gives 16, and 7x the error of no shadow at all; conical is indistinguishable from no shadow, which is the reason to prefer it. Exports `umbra_clearance` (the cylindrical terminator as a *signed scalar*, negative inside the umbra), `shadow_clearance` (the `(sim, bodies)` event adapter), `cylindrical_shadow_bodies` and `latch_shadow_branch`; column 7 `shadow_latch` is **engine-owned state** written and released inside one step by `events.py` and refused as a user coefficient. The cylinder's order loss is now **fixable**: `sim.add_event(events.shadow_event(sim))`. `validate_coefficients` rejects a missing/non-slot/self/parent/barycentre/inactive source, `cr` outside [0,2], negative areas or radii, an unknown `shadow_model`, a conical shadow with no `r_source`, and a barycentre-parented body asking for a shadow. The frozen-source lag is negligible here (5e-6 km/day at dt=60 s). Foreign to the fused plan, so Cowell runs on NumPy. No compiled twin. **SRP overtakes drag at 631 km** under `atmosphere.py`'s table at `C_r/C_d = 1.3/2.2` (A/m cancels exactly); the usual 800 km figure assumes ~8x that density. Validated in `tests/validation/test_srp.py` against the closed form, an independent disc-overlap quadrature, and the orbit-averaged Gauss rate `<de/dt> = -(3/2)(h_hat x f)/(n a)` (predicted 2.674e-5, measured 2.678e-5, 0.76 %) |
| `gravity.py` | Force model `"point_mass_gravity"`: the central term `-(mu_body + mu_parent) r / r^3` relative to `parent_indices`, the same summed mu the Keplerian path uses. Parent only, so a two-body term and **not** N-body or third-body. Registered on `import orbital_engine`. Name constant `POINT_MASS_MODEL`. Compiled twin: fused into `kernels.cowell_rk4_step` |
| `thirdbody.py` | Force model `"third_body"` (`THIRD_BODY_MODEL`): one named perturber's point-mass pull relative to `parent_indices`, **direct minus indirect** `mu_s[(r_s-r)/\|r_s-r\|^3 - r_s/\|r_s\|^3]`. The perturber is a float slot in `force_model_params["third_body"]` column `perturber`, and it is **mandatory**. In sweeps give it by name: `ForceModelSpec(..., body_coefficients={"perturber": "Sun"})`, resolved by `apply_config`. It is validated through the registry's `validate_coefficients` hook, which rejects self, parent, barycentre, inactive and massless perturbers. One perturber per body. **The perturber is frozen at start of step within RK4, so convergence is first order, not fourth:** the Moon under the Sun is 2.64 km off after 30 days at dt=3600 s, matching `(h/2)·dr/dτ` to 2.6 % (`tests/validation/test_third_body.py`). The bit is foreign to the fused plan, so Cowell runs on NumPy. No compiled twin |
| `thrust.py` | Force model `"thrust"` (`THRUST_MODEL`): `a = (T/m) d_RSW * 1e-3`, the direction law a per-body **RSW** vector (`(0,1,0)` prograde, `(1,0,0)` radial-out) rotated through `frames.RSW_to_cart`. Coefficients `(thrust_n N, isp_s s, mass_kg kg, dry_mass_kg kg, dir_r, dir_s, dir_w)`; `STANDARD_GRAVITY` = 9.80665. The direction is **not normalised** — its norm throttles, `(0,0,0)` coasts. **Mass is state**: column 2 is mutated once per step by `Simulation.step` -> `thrust.deplete_mass` (`m <- max(m - T/(Isp g0) dt, m_dry)`), so re-running a scenario needs `mass_kg` re-seeded. All four RK4 stages see the step's starting mass, making the *mass* first order: `Delta v` under-delivers by `(dt/2)(a_end - a_start)`, measured -8.84e-5 of `Delta v` against a derived -8.79e-5. `validate_bodies` refuses `mu != 0`; `validate_coefficients` range-checks thrust/Isp/mass. Empty tanks, zero thrust and an undefined RSW frame give **exactly** 0.0. Foreign to the fused plan, so Cowell runs on NumPy. No compiled twin. Validated in `tests/validation/test_thrust.py` against a closed form, the rocket equation via a thrust-free twin, `da/dt = 2f/n` orbit raising and exact burnout |
| `manoeuvres.py` | Impulsive **Delta-v**, usable under *every* propagator - the complement of `thrust.py`, which is Cowell-only. Given in **RSW** (`(0, dv, 0)` prograde) of the body's state relative to `parent_indices`, rotated by `frames.RSW_to_cart`. `Simulation.apply_delta_v(bodies, dv_rsw)` applies now; `schedule_delta_v(bodies, dv_rsw, epoch_s)` queues it and **`step()` splits the step at that epoch**, so timing is exact rather than quantised to `dt` (`_advance` is the unsplit step; `pending_manoeuvres` / `clear_manoeuvres` are the queue). Per propagator: Cowell changes the Cartesian state only (its `coe_states` stays stale by design); Keplerian **also re-derives `coe_states`** - the elements are that propagator's state of record, and an impulse that skipped them would be erased by the next step; secular-J2 does that **and rebuilds the cached `_secular_j2_rates`**, which depend on the `p`/`e`/`i` the impulse just changed. Refuses heads, barycentres, roots, inactive slots, `mu != 0` and an impulse opening a secular-J2 orbit, all **before mutating anything** - the kernel computes the transaction and commits only the valid rows. Splitting is exact for analytic bodies (measured 0.0 km on a non-manoeuvring Keplerian twin) but **not** for a Cowell one, which sees one local truncation error (`r(nh)^5/120`, measured 4.7e-4 km over 2.3 orbits). No mass coupling to `thrust.py`'s propellant. Not a registry entry - an impulse is neither an acceleration nor a propagator; see its docstring. Validated in `tests/validation/test_manoeuvres.py` |
| `events.py` | **Event-driven step splitting**: `step()` cuts itself wherever a continuous scalar function of the arena state changes sign - the generalisation of `manoeuvres.py`'s scheduled-epoch split to an epoch that must be *found*. `Event(name, function, bodies, direction, tol_s, latch)` is plain frozen data; `function(sim, bodies) -> (k,)` must be a **pure read**. Configured through `Simulation.add_event` / `clear_events` / `registered_events`, with `event_splits` / `event_evaluations` / `event_nudges` / `event_epochs` as output. Not a registry entry - see its docstring. **Detection is per body, the split is per arena** (one clock; a non-crossing body pays exactly what manoeuvre splitting costs it). The crossing is bracketed by Illinois-modified false position over **trial propagations** (snapshot, `_advance(tau)`, evaluate, restore), terminating on bracket *width*; `DEFAULT_EVENT_TOL_S` = 1e-6 s. **Splitting alone is not enough**: RK4's stage 4 samples `O(h^3 |da/dt|)` off the trajectory, so a sub-step ending at the discontinuity still puts one stage (weight 1/6) on the far side - hence `Event.latch`, which pins the model to one branch over a crossing-free sub-interval, and hence three advances per crossing (`tau_lo`, a micro-step across the bracket, then strictly past it). The postcondition tests `<= 0` because `umbra_clearance` rounds to exactly `0.0` near the surface and `-0.0 < 0.0` is **false**. `shadow_event(sim)` wraps `srp.umbra_clearance`; conical bodies are excluded (nothing to split at). Measured on `scenarios.eclipsed_satellite`: ratios **17.28 16.66 16.23 17.83** split against **16.99 6.40 2.57 1.46** unsplit, 5.19e-9 km at h=1.25 s against a no-shadow control's 5.37e-9 and an unsplit 9.18e-7 (**177x**). Cost 6-12 trial propagations per crossing, **zero** for a step with no crossing, which is **bit-identical** to today. Blind spot: an *even* number of crossings of one body inside one interval fires nothing. Validated in `tests/validation/test_events.py` |
| `integrators.py` | `Integrator` protocol and `RK4Integrator(max_capacity)`: `step(provider, t, state, dt, indices, primaries)` advances `state[indices]` relative to `state[primaries]`, copying each stage's accelerations (the provider returns a shared buffer). Named stage buffers are allocated once in `__init__`, but fancy indexing still allocates temporaries every stage, so it is not allocation-free. The frozen-primary formulation is exact only for forces that depend on position relative to the parent, which is every registered model except `third_body` (first order, see its row). Compiled twin `kernels.cowell_rk4_step`, fused with `point_mass_gravity` and `j2`; `step()` uses it when `use_compiled_kernel` is set **and** `Simulation._cowell_fused_ok` (rebuilt by `_refresh_cowell_plan`: every Cowell body's mask is a subset of those two models and no Cowell body parents another), otherwise this NumPy path runs |
| `propagators.py` | `SecularJ2Propagator` (`PropagatorType.SECULAR_J2`): analytic Keplerian propagation plus first-order secular drift of RAAN, argument of periapsis and mean anomaly under J2 — Vallado 4e Eq. 9-41 is the likely reference, **unverified against the text**. `p`, `e`, `i` held constant, matching the theory; `secular_j2_rates(...)` derives the three rates once, cached rather than recomputed per step. Compiled twin `kernels.secular_j2_propagate`. Configured only through `Simulation.set_propagator(bodies, PropagatorType.SECULAR_J2, j2=..., r_eq=...)` — coefficients are mandatory, unlike `enable_force_model("j2", ...)`'s silent-no-op convention. Optional `mean_seed=True` (default `False`, bit-identical to the prior behaviour) replaces the seeded `p` with a first-order mean value via `propagators.mean_seeded_p` — Kozai 1959 / Brouwer 1959, **unverified against the text**; corrects only `p`, never `e` or `i`. See `docs/architecture.md`'s Cowell section for the shared restriction reasoning and its own section for what this propagator adds |
| `geometry.py` | Observation geometry, pure array functions with no `Simulation`: `elevation_azimuth(positions, times, velocities_km_s=None, latitude_rad=, longitude_rad=, altitude_km=, omega=, body_radius_km=, theta0=, epoch_s=)` → `Topocentric(elevation_rad, azimuth_rad, range_km, range_rate_km_s)` shaped `(n_times, n_stations, n_bodies)`. **`range_rate_km_s` is `None` unless velocities are passed** (central-body-relative *inertial*, i.e. `viz.sample_states`' columns 3:6), and its sign is **positive = opening (receding)** — stated because an inverted convention inverts every downstream Doppler and passes every magnitude check. It carries the transport term `omega x r_station`: dropping that replaces the station-relative rate `n - omega` by `n`, **0.4646 km/s of 6.5213 km/s (7.1 %)** at the horizon of a 550 km equatorial pass, and 0.4646 km/s is exactly `R omega`. Doppler, carrier frequency and link budget are **out of scope by design** — radio properties, not orbit properties. `AccessWindow` also carries `peak_time_s`, the *sampled* time `peak_elevation_rad` came from (`nan` by default for hand-built records); `access_windows(times, elevation_rad, mask_angle_rad=)` → `list[AccessWindow]`; `line_of_sight(r1, r2, body_radius_km=)`. **Spherical/geocentric** station latitude, east longitude, altitude above `body_radius_km` — there is no ellipsoid in this engine. Azimuth is clockwise **from north**, everything in radians (`viz.ground_track` reports degrees). **`theta0` is referenced to `epoch_s`, default 0.0 = absolute sim time, deliberately *unlike* `viz.ground_track`'s `times_s[0]`** — a grid starting at `t=28 s` otherwise moves every rise/set by 2.0 s, silently. Window edges are **linearly interpolated**, `O(h^2)`, and the error is a *bias*: the elevation curve is **convex at the horizon**, so rises read early, sets late and durations long by `Omega cot(lambda_0)/2 · ab` (0.23 s at `h=30 s` on a 784 s pass, ratio 4.03 under halving). `peak_elevation_rad` is **sampled, not refined** (`O(h)` for an overhead pass — a corner, not a smooth max). `line_of_sight` clamps the segment parameter to `[0,1]`, which is what keeps a station-to-satellite link from reading as occluded. Not a registry entry — no acceleration, no state. Validated in `tests/validation/test_geometry.py` against horizon geometry on `scenarios.ground_station_pass` |
| `access.py` | Model error in **contact windows** instead of km, for `sweep.py`. `AccessSpec(stations=[GroundStation(...)], central_body=, omega=, body_radius_km=, mask_angle_rad=, sample_dt_s=)` -> `run_sweep(..., access=spec)` fills `SweepResult.access` with `AccessMetrics`: signed rise/set shift and duration error (`ShiftStats`: mean, median, mean-abs, max-abs), total contact time and error, and **passes gained or lost**. Windows are matched to truth's by **time overlap only**, greedily largest-first; a window with no overlapping counterpart is an orphan (lost or gained) and never gets a fictional shift - there is no match tolerance to tune. Orphans and clipped edges count toward total contact time but **not** toward the shift statistics. Sign: positive = model **late**. **`sample_dt_s` must be an integer multiple of every config's `dt`** and `access_metrics_for` raises otherwise - `viz.sample_states` sub-divides to *at most* `max_dt`, so a 10 s grid asked of a `dt=60 s` tier would silently propagate it at 10 s; `DEFAULT_SAMPLE_DT_S` = 60.0 for that reason. Truth and model share one grid so `geometry.py`'s convex-horizon edge bias is common-mode and cancels to `C*Delta*h` (2.0e-2 s at 60 s, 7 % of the finest tier's signal). **The contact dataset** — what a downstream network/ISL consumer exports — is `contact_windows(positions, velocities, times, spec)` / `contacts_from_simulation(sim, bodies, times, spec, max_dt=)` → `ContactWindow(window, rise, peak, set)` with each of the three a `ContactSample(time_s, range_km, range_rate_km_s)`; `window` is `windows_from_positions`' `AccessWindow` **unchanged** (asserted by value) and both paths go through one `_topocentric` call. Rise/set are interpolated times so both series are `np.interp`'d onto them (`|f''|h^2/8`; range rate 8.5e-5 km/s at `h=10 s`, measured 7.0e-5), while the peak is a grid time that inherits `peak_elevation_rad`'s `O(h)` offset — its range rate is **not** zero but bounded by `rho'' h/2` (0.42 km/s at `h=10 s`, measured 0.070). No Doppler and no range-rate metric: the dataset is an export, not a second `AccessMetrics` field. Adds **one** `reference_for` call per sweep, not per config; `ErrorStats` is bit-identical with `access=` on. Not a registry entry. 24 h, 12 sats, 3 stations, 5 deg mask, 189 passes: Kepler 31.7 s mean shift / 5 lost / 4 gained, secular-J2 osc 30.8 s / 1 / 2, secular-J2 mean-seeded 1.60 s / 1 / 0, **Cowell+j2 at 60 s 0.063 s / 0 / 0** - the first tier under 1 s (`docs/figures/access_windows.png`). Validated in `tests/validation/test_access.py` |
| `stationkeeping.py` | The atmosphere choice in **Delta-v**: a dead-band altitude controller. `StationKeepingSpec(lower_km, upper_km, r_ref_km=EARTH_R_EQ, impulses=2)` is plain data - when the mean altitude falls to `lower_km`, raise it to `upper_km` by a Hohmann pair between mean radii (second burn via `schedule_delta_v`) or one prograde burn. `StationKeeper(sim, bodies, spec, dt).observe()` after **every** step (raises if a step is skipped); `run_station_keeping(...)` drives it and returns osculating/mean altitude series, the `Burn` log and `summarise_burns` (total Delta-v, raises, mean interval). **Keys on the one-period time average of osculating altitude**, never osculating altitude (J2 swings it 12 km peak to peak at 300 km / 51.6 deg): exact-length trapezoid window, extrapolated `T/2` along a two-window slope (a trailing mean reads `\|mdot\| T/2` = 0.18 km high); blind ~2.5 orbits after each raise. **Checked between steps, not an `events.Event`** - an event function is a pure read of instantaneous state, `Event` has no action hook (that belongs upstream), and the timing buys 2-4 m. Not a registry entry. Build its scenario with `scenarios.station_keeping_satellites(n_sats=, altitude_km=, inclination_deg=)`: co-located Cowell + pm + J2 satellites, altitude above `EARTH_R_EQ`, **mean 6.8 km below an osculating-circular seed** at 300 km / 51.6 deg. Validated (`tests/validation/test_stationkeeping.py`) against `(n/2) da` of the orbit-averaged decay with density convexity, RK4 drift and the Hohmann transfer phase derived: **+4.0e-4 / +2.3e-4 / +1.04e-3** against a 3e-3 budget. **RK4's Kepler energy constant is `(n h)^6/36` per step, not 1/72**; and at `dt = 60 s` a drag satellite's RK4 decay exceeds a drag-free twin's by 3e-3 of the drag rate, so **a drag-free twin does not calibrate drift under drag** - the test runs at 30 s. Headline (`docs/figures/station_keeping.png`, 10 days): table 3.436 m/s/day, single band matched at 355 km **14.1 % less**, the same band matched at the station **0.14 %**. **Also a sweep metric**: `run_sweep(..., station_keeping=spec, delta_v_baseline="<config>")` -> `SweepResult.delta_v` (`DeltaVMetrics`: per-body total, raises, steady m/s/day excluding the first raise, medians, signed `rate_error_rel`/`total_error_rel`, negative = under-budgets) - one extra propagation per config at **its own `dt`**, scored against a **named baseline config** (required, never inferred; truth has no drag), `ErrorStats`/`access` bit-identical with it on. A drag-free tier scores exactly -1.0, reported not refused; `dt` above `window/MIN_SAMPLES_PER_ORBIT` (~1355 s at 292 km) is **refused by config name** before truth, never re-stepped; `ExternalTier` gets `None`. Through the sweep on 18 h: single band at 355 km -0.1453 vs derived -0.1416 (`tests/validation/test_sweep_delta_v.py`); over 5 days (~11 raise cycles) it converges to the 10-day study, -0.14137 against -0.1412 |
| `viz.py` | Plot-*data* preparation, no matplotlib: `sample_states(sim, bodies, times, relative_to=, max_dt=)` steps the sim over a time grid (**it advances the simulation**) and returns `(n_times, n_bodies, 6)`; `max_dt` is the propagation step, so `max_dt == config.dt` reproduces a `sweep` run step for step. `ground_track(positions, times, omega=, body_radius_km=, theta0=)` → lat/lon/alt in degrees (central-body-relative input; longitude wraps at +/-180, so a line plot must break there). `altitude_series`, `position_error`, `error_curve` — the last reduces over bodies like `sweep.ErrorStats`, and its endpoint equals `run_sweep`'s median. **The body-fixed rotation takes `-theta`**: `Transformations.Rz` is an *active* vector rotation, so the frame transform is its inverse; getting it backwards drifts the track east and raises nothing. Consumed by `benchmarks/figures.py` |
| `sgp4_bridge.py` | **SGP4 wrapped, never reimplemented** (`sgp4` extra, imported lazily; the engine imports without it). `TLE(name, line1, line2)` -> `sgp4_states(tles, epoch, times_s)` evaluates a `SatrecArray` and returns **Cartesian TEME** `(n_times, n_sats, 6)` km, km/s plus SGP4's error codes; a flagged row is **NaN-filled here** because SGP4 returns a *finite* state with code 6 (decayed). Times are **seconds from a chosen epoch** (`JulianDate` = split `(jd, fr)`, whole days on `jd` - putting a 3.5-yr offset on the fraction cost 1.6e-7 km); `sgp4_tsince` (minutes from the TLE's own epoch) exists only for the published cases. A TLE's mean elements **never** reach `coe_to_rv`: `seed_elements` runs `rv_to_coe` on SGP4's own `(r, v)`, and `scenarios.tle_satellites(session, tles=None, epoch=None)` (Earth at the root, one massless vessel per TLE named `TLE.name`, default `ISS_TLE`) stores those osculating elements under `MU_EARTH`. **TEME is taken as the engine's inertial frame** - it ignores the equinox/pole drift, bounded by `TEME_DRIFT_RAD_PER_DAY` (0.31 arcsec/day, ~11 m/day at LEO); unacceptable for months-long horizons or any comparison against GCRF data (use `pyerfa`). **SGP4 is a sweep tier of a second kind**: `sgp4_tier(tles, epoch, dt=)` -> `sweep.ExternalTier(name, dt, bodies, central_body, positions)`, scored by `run_sweep(..., external=[...])` / `sweep.score_external` against the same truth, access grid included - not a `PropagatorType` (`PropagatorType.SGP4` stays unimplemented), because a `Satrec` is a stateful object and must not live in `step()`. Validated in `tests/validation/test_sgp4_bridge.py` - **the first test against published numbers**: Vallado et al. 2006 `SGP4-VER.TLE` / `tcppver.out`, read from the installed `sgp4` package, all 31 in-tolerance cases (1788 components) agree to the file's print rounding (**mean |dr| 2.515e-9 km against a derived 2.5e-9**), the 3.5-yr SL-12 case to 1.17e-7 km (the package's own tolerance 2e-7; its native path gives the same), and all 7 published error outcomes. Cowell+J2 seeded from SGP4's Cartesian state lands on the predicted local expansion (3.70e-3 km at 300 s; SGP4's `v` differs from its `dr/dt` by 1.06e-5 km/s, an O(J2^2) property of the theory). Mean elements into `coe_to_rv` instead: **7.64 km** off at epoch. One day against J2 truth (ISS): SGP4 0.98 km = +1.20 WGS-72 `mu` + 0.95 drag - 1.18 J3/J4/theory |
| `Simulation` model API | `enable_force_model(name, bodies, **coefficients)` and `set_propagator(bodies, PropagatorType.COWELL \| PropagatorType.SECULAR_J2, **coefficients)` are the sweep-configuration surface; `resolve_force_models()`; `accelerations(t, state=None)`. `set_propagator` raises `ValueError` for Cowell on heads, barycentres, kinematic roots, inactive slots and **any body with `mu != 0`**, since a Cowell body bypasses the barycentric accumulation that carries its mass into the reflex kick; `SECULAR_J2` carries the same restrictions plus a non-barycentre Keplerian parent, `0 <= e < 1`, and mandatory `j2`/`r_eq` coefficients |

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

Cowell's compiled twin is **fused**: `kernels.cowell_rk4_step` hard-codes RK4 with `point_mass_gravity`
and `j2` (per-body flags), because numba cannot dispatch over the Python kernel list
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
| Cowell step (RK4 + `point_mass_gravity` + `j2`) | `integrators.RK4Integrator` over `forces.compose_accelerations` | `kernels.cowell_rk4_step` (fused; other masks fall back to the reference) |
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
