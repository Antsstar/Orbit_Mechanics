# Architecture

What each module is for, why the unusual decisions were made, and what is deliberately not built yet.

`CLAUDE.md` is the operational summary — invariants, conventions, known-broken code — and is what an
agent reads at session start. This document is the *reasoning* behind it: longer, less prescriptive,
and aimed at someone trying to understand why the engine is shaped this way rather than how to work
in it. `docs/engineering-log.md` is the third leg: problems already hit, so they are not rediscovered.

---

## Module map

Roles marked **unchanged** have kept their original purpose since the project began.

| Module | Role | |
|---|---|---|
| `simulator.py` | Overall orchestrator. Owns the memory arena, builds the universe from the database, drives `step()` | unchanged |
| `database.py` | Catalogue of every body a scenario could ever want. A simulation loads a named subset | unchanged |
| `frames.py` | Coordinate and state-space transformations, now including the RSW frame. Core toolbox | unchanged |
| `utilities.py` | Anomalies, Kepler, Barker, rotations, perturbations. Core toolbox | unchanged |
| `custom_types.py` | Type aliases and column-index enums. Readability and debuggability | unchanged |
| `constants.py` | Physical and unit constants | unchanged |
| `exceptions.py` | Domain error hierarchy — readability, plus flagging and controlling unique situations | unchanged |
| `registry.py` | Catalogue of physics models and propagators, so the simulation can query what is available | **wired**: force models by mask bit, propagators by `PropagatorType` |
| `propagators.py` | State advancement — **now specifically the readable *reference* implementation**. Also holds `SecularJ2Propagator` (`PropagatorType.SECULAR_J2`): analytic Kepler plus first-order secular J2 drift of RAAN, argument of periapsis and mean anomaly, `p`/`e`/`i` held constant, coefficients reusing `geopotential`'s `(j2, r_eq)` row | role narrowed, **grown**: a third propagator alongside Keplerian and Cowell |
| `body.py` | `BodyHandle`, a UI-facing pointer into the arena | unchanged, never instantiated |
| `kernels.py` | Compiled scalar twins of the hot paths | **new** |
| `scenarios.py` | Declarative universe builders, shared by tests and benchmarks | **new** |
| `reference.py` | Independent DOP853 N-body truth trajectories, optionally with explicitly passed J2 and J3..J6 | **new** |
| `benchmark.py` | Timing primitive (minimum-of-batches) | **new** |
| `forces.py` | Force-model composition: enabled physics as a per-body bitmask, additive stateless kernels, and the acceleration contract integrators consume | **new** |
| `gravity.py` | `point_mass_gravity`: the central two-body term relative to the gravitational parent | **new** |
| `geopotential.py` | `j2`: the J2 zonal perturbation only, composable with `point_mass_gravity` | **new** |
| `thirdbody.py` | `third_body`: one named perturber's point-mass pull, direct minus indirect, relative to the parent | **new** |
| `integrators.py` | Fixed-step RK4, integrating a body's state relative to its parent | **new** |
| `drag.py` | `drag`: atmospheric drag in a co-rotating atmosphere, composable with `point_mass_gravity` and `j2`. The density law is a per-body coefficient, not a hard-coded formula | **new** |
| `atmosphere.py` | The density laws `drag` chooses between: one exponential band, Vallado Table 8-4's 28-band piecewise-exponential profile, or an NRLMSIS 2.0 profile - all but the first through one piecewise-exponential evaluator | **new** |
| `msis_bridge.py` | NRLMSIS 2.0 wrapped at the boundary: `pymsis` averaged into an altitude profile at configuration time, memoised per solar-activity triple, read by `drag`'s kernel | **new** |
| `srp.py` | `srp`: cannonball solar radiation pressure from a named light source, with a cylindrical or conical shadow. The shadow geometry is a per-body coefficient, the same way `drag` selects its density law | **new** |
| `viz.py` | Plot-*data* preparation: trajectory sampling over a time grid, ground tracks via `frames`' body-fixed transforms, altitude series, and error curves against a `reference.py` truth. No matplotlib import, so the library stays installable without it — `benchmarks/figures.py` is the consumer that draws | **new** |
| `thrust.py` | `thrust`: continuous rocket thrust along a per-body RSW direction law, with propellant depletion. The first model whose coefficients are state | **new** |
| `manoeuvres.py` | Impulsive Delta-v in RSW, applied now or scheduled at an epoch `step()` splits for. The first physics that is neither a force model nor a propagator, and the first that every propagator can use | **new** |
| `geometry.py` | Observation geometry: ground-station look angles, interpolated access windows, and the spherical line-of-sight test. Pure functions of position arrays, like `viz.py`'s transforms — the primitive layer under a future access-based error metric | **new** |
| `access.py` | The sweep-level access metric built on `geometry.py`: model contact windows matched to truth's by time overlap, then differenced into rise/set shift, duration error, total contact error and **passes gained or lost**. Consumed by `sweep.run_sweep(access=...)`, which gains one optional field and changes nothing else | **new** |
| `isl.py` | Inter-satellite link visibility: pair windows cut on `geometry.segment_clearance`, the contact dataset's satellite-to-satellite half, and `run_sweep(isl=...)` - scored by `access.py`'s own matcher and statistics | **new** |
| `events.py` | Event-driven step splitting: a step cut where a continuous function of the arena state changes sign, located by a bracketed root find over trial propagations. The second thing `step()` splits for, after a scheduled manoeuvre — and the first that has to *find* its own epoch | **new** |
| `zonal.py` | `zonal`: the J3..J6 zonal harmonics of the parent, additive to `j2`, with a matching truth option in `reference.py` written from a different derivation | **new** |

Nothing was removed. No module lost a responsibility. The only deletion was `register_model` /
`get_model` in `registry.py`, which nothing had ever called, replaced by the force-model registry.

---

## The one structural change: hot paths exist twice

```
                     ┌── propagators.py ── NumPy, vectorised, readable   ← the DEFINITION
 Simulation.step() ──┤                                                      held equal (1e-12 rel)
                     └── kernels.py     ── compiled scalar loops, fast   ← an OPTIMISATION of it
```

`propagators.py` did not change purpose so much as narrow it. It is still "methods for evaluating the
next state of a body", but it is now the *authoritative, readable statement* of what that means, and
is deliberately **not** optimised. `kernels.py` is the twin that actually runs.

`Simulation.use_compiled_kernel` selects between them. It defaults to whether numba is importable,
because without numba the kernels execute as interpreted Python and are *slower* than the NumPy path
they replace — so "compiled if available" cannot simply be spelled `True`.

### Why this shape rather than just optimising in place

Profiling showed the engine was bound by Python-level NumPy dispatch, not arithmetic: step cost was
flat from arena capacity 64 to 10 000, and 3 bodies cost 542 µs against 602 bodies at 3583 µs — about
490 µs of fixed per-step overhead. Removing that means removing *calls*, which means compiling, which
means scalar loops. Scalar loops are much harder to read than the vectorised form.

Keeping both sides resolves the conflict instead of trading one for the other. The reference stays
legible enough to check by eye against a textbook; the kernel stays fast; and an equivalence test
makes "they disagree" a build failure rather than a subtle physics bug.

**The standing rule is: change both, or neither.** See `CLAUDE.md` for the full list.

### What the equivalence test does and does not prove

It proves the two implementations agree. It says nothing about whether either is *correct* — that is
what the validation suite and `reference.py` are for. The two are complementary and neither
substitutes for the other.

---

## Why the two graphs

Every body carries two parent pointers, and they deliberately disagree.

| Array | Answers | Used by |
|---|---|---|
| `parent_indices` | *What are this body's orbital elements measured against?* | COE rehydration, propagation |
| `body_sys_map` | *What is this body's Cartesian state measured against?* | `calc_global`, barycentre aggregation |

The Moon is the canonical case: its elements are relative to **Earth**, while its position is relative
to the **Earth–Moon barycentre**. Collapsing these into one pointer would force a choice between
correct elements and correct kinematics.

The separation is also what makes execution order derivable. A topological sort over `body_sys_map`
gives an order in which every body's reference frame is resolved before the body itself, which is
what `calc_global` walks. Determining that order without the dual map is the hard part, and is why
neither map can simply be left alone.

### Both maps are mutated during build — intentionally

Two places rewrite `parent_indices` after it is read from the database:

- `_resolve_circular` — binary systems declare each other as parent. One is elected head by mass, and
  the pair is repointed at their shared bubble.
- `_unfold_database_to_global`, step 8 — siblings are repointed at the head of their system.

The trigger for the second is a body whose own system is not loaded *and* whose parent is not the head
of the parent system. This is what lets a user load an arbitrary subset of the catalogue without the
hierarchy becoming unresolvable.

The consequence to be aware of: **`sim.parent_indices` is not what the database declared**, so the
arena cannot be reconstructed from its own state alone. The intended eventual resolution is temporary
system functionality — synthesising a transient system for a partially loaded hierarchy — which would
let the declared graph survive intact. Until then this is documented behaviour, not an accident.

---

## The barycentric model, and why it is shaped this way

This is the least obvious decision in the engine and the one most likely to be "corrected" by someone
who does not know why it was made.

### The choice that was available

**Option A — virtual head.** Treat each system as a pure two-body problem about a *fixed* barycentre,
with a synthetic head body always positioned opposite the sibling.

**Option B — real head with reflex motion.** Compute the head's reflex displacement first, from the
mass-weighted contributions of *all* siblings, then place each sibling relative to the real head.

**Option B was chosen.**

### What that buys

Siblings couple to each other *indirectly*, through their shared tug on the head. Saturn's orbital
elements about the Sun are unaffected by Jupiter — but Saturn's **global position** is, because
Jupiter displaces the Sun and Saturn is placed relative to the real Sun. The coupling is real, and it
is analytic.

The visible signature is that a barycentre traces a *band* rather than a line: a stable cyclic region
whose width depends on the other bodies in the system, instead of a single clean ellipse.

### The property this preserves

The model stays **purely Keplerian and time-invariant**. Every body's anomaly advances analytically
from its own elements, with no accumulated numerical state, so the engine can be stepped *backwards*
and reconstruct prior states to solver precision. There is no integrator error to un-integrate.

That is a genuinely strong property and it is the reason for the design. It is also the reason the
reflex kick is applied as a *derived* quantity each step rather than integrated.

### What it is not

It is **not** N-body. Sibling–sibling forces are absent; only the shared displacement of the head is
transmitted. For two bodies the model is exactly Keplerian and matches numerical integration to
7.9e-5 km over ten days. For three or more siblings in one bubble it is an approximation whose error
against true N-body is currently **unmeasured**.

That gap is not a defect — quantifying exactly this kind of modelling error is the purpose of the
project. It is the first headline comparison the engine should produce once the sweep harness exists.

### Time-reversibility, tested

`tests/validation/test_time_reversibility.py` steps the arena forward N times and back N times. The
arena must return to its initial global state within a bound derived from the Kepler solver's
stopping tolerance plus a rounding allowance. It runs on both implementations, for `two_body`,
`sun_earth_moon`, the constellation, and the constellation on secular J2.

Measured relative return errors after 1000 steps each way:

| Case | Relative error |
|---|---:|
| two_body | 3.4e-11 |
| sun_earth_moon | 3.7e-11 |
| constellation | 4.4e-13 |
| Cowell control (RK4) | 2.2e-4 |

The analytic cases sit at the rounding floor. Cowell's RK4 is not time-symmetric, and its return error
is the test's negative control.

It is the only test that steps backwards. A planted `abs(dt)` in the compiled elliptic anomaly advance
passed all 319 other tests and failed exactly the three compiled cases here. `test_orbit_closes_after_
integer_periods` checks a different property, return after a whole period.

---

## `mu_array` holds summed mass on barycentre rows

A barycentre's row holds the total mass of its system, not zero. This is not overloading: the entry
consistently means *"the gravitational parameter of the entity at this slot"*, and a barycentre's
entity is the subsystem it represents. The summed value is load-bearing — it is what the barycentre
uses to compute its own Keplerian orbit about its parent, and what the reflex kick divides by.

The one place this needs care is any code treating slots as independent physical bodies. `reference.py`
filters `~is_system` before integrating, because barycentres are virtual aggregates and including them
alongside their members would double-count. That filter is required by what a barycentre *is*, not by
how its mass is stored.

---

## Cowell propagation: frame, restriction, and what it does not do

`integrators.py` (RK4) and `gravity.py` (`point_mass_gravity`) add the engine's second propagator:
direct numerical integration of the Cartesian equations of motion, selectable per body through
`Simulation.set_propagator` and dispatched by `step()` alongside the unchanged analytic Keplerian
path. The design problem is that the arena is hierarchical and Cowell is not — a Cartesian integrator
has no notion of a system bubble or a reflex kick — so the frame it integrates in has to be chosen
deliberately rather than inherited from whichever array happens to be lying around.

**Frame: stored in the simulation-root inertial frame (`global_states`); *integrated* relative to
`parent_indices`.** These are two different questions and an earlier version of this section — and of
the code — conflated them, which was a real correctness bug (see below). Every registered force kernel
(`forces.ForceKernel`) indexes its `state` argument by absolute arena slot, exactly like
`global_states` — a kernel computing a relative vector to a body's parent (`point_mass_gravity`,
`geopotential.j2_kernel`) reads both rows from the *same* array, so `provider` still needs a common,
absolute-frame array to call into; `global_states` is the only array with that property, since
`local_states` is bubble-relative and a different bubble for every body. But *what RK4 integrates* is
not that absolute position — it is the state relative to `parent_indices[body]`,
`(r_body - r_parent, v_body - v_parent)`. Every current force kernel's output depends only on that
difference, never on the parent's absolute position, so the relative state obeys a self-contained ODE
that does not care how the parent itself is moving. `integrators.RK4Integrator` reconstructs an
absolute candidate row (`state[primaries] + candidate_relative_state`) at each sub-stage purely so
`provider` has a common frame to read, discards it once the acceleration is read back, and
`Simulation.step` re-bases the integrator's relative result onto the parent's freshly
Keplerian-propagated position once `calc_global()` has produced it (see below). `global_states` is
still where the result lives; it is simply not the variable the numerical method advances.

**The bug this replaced.** The first version of this propagator integrated the body's *absolute*
`global_states` row directly, using only the parent-relative point-mass acceleration. That silently
assumed the parent never moves: nothing in the formulation ever subtracted the parent's own velocity or
acceleration. It passed every test built at the time because they all used `scenarios.two_body`, whose
primary is fixed by construction — review caught it by reasoning about `scenarios.sun_earth_moon`
instead, where the Moon's parent (Earth) genuinely accelerates toward the Sun. With the Moon on Cowell
and only `point_mass_gravity` enabled, the bug produced an Earth-Moon separation of ~1.9e7 km after 30
days against a true ~4e5 km — unbound, not merely inaccurate — matching the analytic estimate
`0.5 · (mu_Sun/AU²) · t²`. See `docs/engineering-log.md` for the full account and
`tests/validation/test_cowell_propagator.py::test_cowell_matches_keplerian_when_the_parent_accelerates`
for the regression guard.

**Convention: two-body mass sum via `parent_indices`, matching the Keplerian propagator exactly.**
`point_mass_gravity` computes `mu = mu_array[body] + mu_array[parent_indices[body]]`, the same sum
`Simulation._rehydrate_coes` and `KeplerianPropagator` already use. A Cowell body running only this
model is integrating the *identical* equation of motion the Keplerian propagator solves analytically,
so the two agree to RK4's own truncation error rather than to some looser "close enough" tolerance —
that is what makes the fourth-order convergence test in `tests/validation/test_cowell_propagator.py`
meaningful rather than a coincidence of similar-but-different physics.

**Restriction: Cowell may only be assigned to an active body that is not a head, not a system
barycenter, and not its own kinematic bubble** (`body_sys_map[i] != i`), enforced by
`Simulation.set_propagator`. Two independent reasons, not one:

- A head's or barycenter's motion is the reflex kick `kepler_propagate` / `_recalculate_all_barycenters`
  compute from *every* sibling's mass and position, every step. Replacing it with an independently
  integrated Cartesian state would not just be wrong for that one body — it would desynchronise the
  reflex kick for *every other sibling in the bubble*, since the kick is derived from the head's row.
- A kinematic root (`body_sys_map[i] == i`) is what `calc_global` treats as the coordinate origin: tier
  0 is unconditionally zeroed every step (`global_states[tier_0] = 0.0`). A root Cowell body's
  integrated motion would be silently discarded by that line rather than raise, producing a body that
  looks like it never moves — the failure mode this project treats as worse than a crash.

**Cowell also requires `mu_array[body] == 0.0`, enforced by the same call.** A body's mass reaches the
rest of the arena only through `kepler_propagate`'s barycentric accumulation (every active,
non-Cowell sibling's mass-weighted position summed onto its head), which a Cowell body never passes
through — `_kepler_sib_idx` excludes it. A massive Cowell body would therefore silently stop
contributing to its own head's reflex kick, corrupting every other sibling in the bubble exactly like
the head/barycenter case above, just through mass rather than position. This is enforced rather than
left as a documented-but-permitted gap; a massless body (the common case for a satellite, or a test
secondary with `mu_secondary=0.0`) is unaffected, since it never contributed to a reflex kick anyway.

**How the result reaches `local_states` and `global_states`.** Cowell integration runs *before* the
Keplerian propagation and `calc_global()` in `step()`. Before it runs, `Simulation.step` snapshots each
Cowell body's parent's *start-of-step* global state (`parent_state_at_start`); the integrator's result
is therefore expressed relative to that snapshot, not a true absolute position, and is saved.
`calc_global()` then runs as normal — it still walks *every* active slot in topological order,
including Cowell ones, briefly overwriting them with a stale, pre-step value derived from last step's
`local_states`, which is immediately discarded — and once it has propagated the parent (a Keplerian
body) to its true end-of-step position, `step()` recovers the pure relative state
(`cowell_result - parent_state_at_start`) and adds the parent's *fresh* position to it, before writing
the result into `global_states`. `local_states[i]` is then rebuilt as
`global_states[i] - global_states[body_sys_map[i]]`, using `body_sys_map` — the kinematic bubble
reference, generally a *different* slot from `parent_indices` (the Moon is the canonical case: its
`body_sys_map` is the Earth-Moon barycentre, its `parent_indices` is Earth). This keeps the documented
invariant — `local_states[i]` relative to `global_states[body_sys_map[i]]` — true for a Cowell body
exactly as it is for a Keplerian one, so nothing downstream (`_record_state`, `history`, a future spawn
path) needs to know which propagator produced a given row.

**What a Cowell body's orbital elements mean: nothing.** `coe_states` for a Cowell body is left
untouched by `step()` — stale at whatever it held before the body was reassigned, the same convention
already used for a head's deliberately zeroed COE. Recomputing osculating elements from the fresh
`(r, v)` each step would be legitimate (it is exactly what `_rehydrate_coes` already does at build
time) but is not needed by anything Cowell currently does and was left out rather than half-built; see
the follow-up note in the module docstrings.

**What this does not do.** Cowell bodies feel gravity from their own parent only, never from every
other active body (a two-body term, not N-body) — a different, separately registered model would be
needed for a genuine third-body or N-body force. That is the one place a real approximation remains:
because every *current* force kernel depends only on a body's position relative to its own parent, the
relative-state formulation above carries no error at all from the parent's motion during a step, no
matter how fast or non-uniformly it accelerates — but a hypothetical future force depending on some
*other* body's absolute position (a solar tidal term on the Moon, say) would see that other body frozen
at its start-of-step position across all four RK4 sub-stages, since this integrator only ever
re-evaluates `indices` and reads (never advances) `primaries` between stages. `third_body` is now
such a model, and this freeze is a live limitation. It is quantified in its own section below.
Both restrictions above (mass and kinematic role) are also documented restrictions of *this* phase, not
accidents.

**The compiled twin is fused, not composed.** `kernels.cowell_rk4_step` is the second-implementation
half of this propagator: one scalar loop per body doing all four RK4 stages with `point_mass_gravity`,
`j2` and `zonal` (J3..J6) inlined (`kernels._cowell_accel`, each term's arithmetic in the same order
as its NumPy kernel, and the terms added in registration order, as `compose_accelerations` adds
them), selected by `use_compiled_kernel` exactly like the Keplerian and secular twins. It cannot
call `forces.compose_accelerations` - numba does not dispatch over a list of Python callables - so
instead of a general compiled composition layer it hard-codes the one combination the fidelity sweep
runs, with per-body flags so a mixed arena (some satellites with J2, some with J2..J6, some without)
still qualifies. `Simulation._refresh_cowell_plan` decides at configuration time, from the mask alone,
whether every Cowell body's models are a subset of those three and no Cowell body is parented by another Cowell body
(the kernel reads a parent's row as fixed across the stages, where `RK4Integrator` would see its stage
candidates); if not, the whole Cowell set runs the NumPy path. The plan is rebuilt by both
`_refresh_active_indices` and `resolve_force_models`, so it tracks `set_propagator` and
`enable_force_model` without a per-step reduction over the mask. Both paths write the parent-relative
result into `_cowell_rel` - the kernel by subtracting the parent from the row it has just rounded onto
the absolute grid, the same arithmetic `step()` applies to the NumPy result - and share one re-base
(`Simulation._rebase`, whose own compiled twin `kernels.rebase_relative_states` is held bit-identical,
like `calc_global_states`, since it is one addition and one subtraction per component);
for a heliocentric Moon that rounding is an ulp of 1.5e8 km, 8e-14 of the lunar distance per step,
close enough to the 1e-12 equivalence bound that returning the relative result directly would have
been a mistake. `tests/validation/test_kernel_equivalence.py` holds the pair at 1e-12 relative on
direct calls and through `step()` (measured 1e-14 after 50 steps; single steps bit-identical), with a
negative control and a spy test that the plan really selects the kernel and really falls back.

---

## Secular-J2 propagation: the third tier, and why it is a propagator, not a force model

`propagators.SecularJ2Propagator` (`PropagatorType.SECULAR_J2`) advances RAAN, argument of periapsis
and mean anomaly by their first-order secular J2 rates, on top of the ordinary analytic anomaly advance
`KeplerianPropagator` already performs. It sits between the two existing tiers on cost and accuracy:
essentially free like Keplerian propagation (no integrator, no force evaluation), but with most of
Keplerian's dominant nodal-regression error removed — the reason a model-fidelity sweep wants it as a
distinct, named configuration rather than treating "Kepler" and "Kepler + J2" as the same thing with a
coefficient toggled.

**Why a propagator and not a force model.** Every other physics addition so far (`point_mass_gravity`,
`j2`) is a `forces.ForceKernel`: an acceleration, composed additively, consumed by an integrator that
does not know what produced it. Secular-J2 drift is not an acceleration a Cowell body could integrate —
it is a *closed-form correction to the analytic solution itself*, expressed as three additional element
rates. Forcing it through the force-model contract would mean either inventing a fictitious acceleration
that reproduces these rates under RK4 (needless numerical error for a case that has an exact solution)
or teaching `KeplerianPropagator` to consult the force-model mask (coupling two axes `registry.py`
deliberately keeps separate — see its module docstring). Registering it as a third `PropagatorType`
instead keeps propagator *selection* and force *composition* on the same two independent axes they
already were.

**Why it reuses `geopotential`'s `(j2, r_eq)` coefficient row rather than owning a separate one.** The
frontier plot this propagator exists for compares three tiers on the *same* scenario, and the third tier
(Cowell + `point_mass_gravity` + `j2`) already carries Earth's J2 coefficients in
`force_model_params["j2"]`. Storing this propagator's coefficients in the same array, under the same
`("j2", "r_eq")` column layout, means one coefficient can be shared by both the analytic and numerical
J2 tiers in a sweep, so a coefficient-sensitivity sweep cannot silently compare them at different values
by only updating one of the two call sites. The cost is that this never goes through
`enable_force_model`, since that call also sets the mask bit that adds a body to `accelerations()`'s
dispatch set — irrelevant and wrong here, since a `SECULAR_J2` body never calls `accelerations()` at
all. `Simulation.set_propagator`'s `SECULAR_J2` branch writes the same array directly instead.

**Restrictions mirror Cowell's, for the analogous reasons**, not coincidentally: both propagators
replace a body's entry in `_kepler_sib_idx`, the set `kepler_propagate`'s reflex-kick accumulation
reads, so both need active/non-head/non-barycentre/non-bubble-root and `mu == 0` for the identical
reason (see the Cowell section above). Two further restrictions are specific to this propagator: the
Keplerian parent must not itself be a barycentre (`geopotential.barycentre_parented` — a barycentre's
J2 is meaningless, the same guard `"j2"` documents for Cowell), and the orbit must be closed
(`0 <= e < 1` — the rates derive from mean motion `n = sqrt(mu/a^3)`, undefined for an open orbit).

**How position is recovered.** Unlike Cowell, there is no integrator and so no `parent_state_at_start`
snapshot — `propagate` only updates `coe_states` and writes the resulting state, relative to
`parent_indices`, into scratch (`Simulation._secular_j2_rel`). `step()` adds that onto the parent's
*end-of-step* global position once `calc_global()` has produced it, and rebuilds `local_states` against
`body_sys_map` — the same re-basing pattern Cowell uses, for the same reason the two parent graphs
diverge (see above), just without anything to subtract first.

**What "mean elements" actually costs here.** `p`, `e`, `i` are held at whatever the body's *osculating*
elements were when `set_propagator` was called, because `CLAUDE.md` forbids converting osculating
elements to mean ones. Measuring this against Cowell + `point_mass_gravity` + `j2` (a numerically exact
comparison, to RK4's own truncation error) at 550 km / 53 deg showed two distinct effects: a bounded,
orbit-period oscillation of a few km (the short-period terms proper averaging would remove), and a
*linearly growing* along-track drift — 57.4 km after 1 orbit, 172.1 km after 3, 573.6 km after 10,
linear to better than 1% — because the mean motion cached from the seeded osculating `p` carries a
small, fixed fractional bias relative to the true mean `p`, and that bias accumulates rather than
averaging out. The second effect dominates within a handful of orbits and was not the first guess (a
bounded ~6 km oscillation looked like the obvious answer until it was actually measured against
Cowell) — see `tests/validation/test_secular_j2_propagator.py` and
`propagators.SecularJ2Propagator`'s docstring for the full account. This growing error, not a fixed
accuracy figure, is what makes "Kepler + secular J2" a genuinely distinct point on a fidelity/cost
frontier rather than a free upgrade to plain Keplerian propagation.

**The drift depends on starting phase, and vanishes at u₀ = 45° (measured in review).** The figures above
are for one satellite starting at argument of latitude u₀ = 0, which turns out to be the worst case.
Against the independent J2 truth (`reference_for(..., oblateness=...)`), after 10 orbits at 550 km / 53°:

| u₀ | 0°, 90°, 180°, 270° | 30°, 60°, 120°, … | 45°, 135°, 225°, 315° |
|---|---:|---:|---:|
| Kepler + secular J2 | 574 km | 287 km | **0.2 km** |
| Kepler | 523–689 km | 276–424 km | 207–214 km |

The secular error is proportional to |cos 2u₀|. That is the signature of the short-period term in the
*osculating* semi-major axis at epoch (∝ sin²i cos 2u for a near-circular orbit). Where that term is
zero, osculating a equals mean a, and the propagator is essentially exact. Every other part of it,
including the secular rates, was already right. Secular J2 removes the cross-track error at every
phase. Along-track it can do worse than Kepler for an individual satellite: at 700 km / 98°, u₀ = 0,
it was 862 km against Kepler's 320 km, because Kepler's missing rates partly cancel the same bias.
Over 12 evenly spaced phases it is better on median and RMS: 287/406 km against 421/457 km at 53°, and
432/610 km against 652/825 km at 98°.

Two consequences. A sweep must report error **statistics over initial phase**, not one satellite,
or the tier ranking depends on which slot was picked. And seeding this propagator with a *mean*
semi-major axis removes nearly all of its error. The user decided to allow that, and `CLAUDE.md`'s rule
now forbids converting only *externally defined* mean elements (SGP4/TLE). It is implemented as
`set_propagator(..., mean_seed=True)`, which calls `propagators.mean_seeded_p`:
a_mean = a_osc · (1 − (3/2) J2 (R/p)² sin²i cos 2u₀).
After 10 orbits at 550 km / 53°, the error at |cos 2u₀| = 1 falls from 574 km to 6.1–6.2 km, and at
u₀ = 45° it is 0.16 km. The remainder is the bounded short-period oscillation, which an averaged
theory cannot represent. The osculating seed stays the default, so the two can be swept side by side.

---

## Atmospheric drag: a velocity-dependent force model

`drag.py` registers `"drag"`, the first model whose acceleration depends on **velocity**. It fits the
existing Cowell formulation without change, because `RK4Integrator` writes `state[primaries] +
candidate` into the velocity columns at every stage as well as the position columns. So
`state[i,3:] - state[P,3:]` is the stage's parent-relative velocity. Drag depends only on
parent-relative position and velocity, and the relative-state argument in the Cowell section holds
for it unchanged.

**Units live in one place.** The coefficients use their conventional units (`B` in m^2/kg, `rho0` in
kg/m^3, lengths in km, `omega` in rad/s). The kernel applies one factor of 1e3 to `rho * B` to turn
1/m into 1/km. A slip there gives a smooth decay that is wrong by 1000x, and a dropped 1/2 gives one
that is wrong by 2x. Neither raises an error, so the validation pins the magnitude.
`tests/validation/test_drag.py` checks a closed-form acceleration recomputed in SI and orbit-averaged
decay `da/dt = -rho B sqrt(mu a)`, integrated as an ODE so the density rise over the run is resolved.
It also checks an energy balance against `integral(a_drag . v dt)` and the co-rotation factor
`(1 -/+ omega a/v)^2` on prograde and retrograde equatorial orbits. Measured errors are 4e-5 against
a derived 2e-4 budget. Real-file mutants of the co-rotation sign, the 1/2 and the unit conversion each
fail these checks.

**Coefficients sit on the body's row**, as `j2`'s do, although five of the six describe the parent's
atmosphere. The reasons are the same: one setter, and per-body sweeps in a single arena.
`VesselORM.drag_area` and `dry_mass` could supply `B` at ingest, but a drag coefficient `C_d` is not
stored, so `B` is an explicit coefficient for now.

**What it does not do.** Spherical altitude `|r| - r_ref`, no diurnal bulge, and solar and
geomagnetic activity only as constant indices under the MSIS law (see "NRLMSIS 2.0" below). The fused compiled Cowell twin does not include drag, so enabling it on any Cowell body
sends the whole Cowell set down the NumPy path.

---

## The atmosphere: making a model *choice* into data

`atmosphere.py` exists because this repo is a model-fidelity comparison engine, and "which atmosphere"
is exactly the kind of axis it should be able to sweep. Drag previously had one density law welded into
its kernel. Now it has two, and the choice is a number on a body's row.

**Why a coefficient and not a second registered force model.** `"drag_layered"` as its own registered
model would have been the obvious move and is the wrong one. It would consume a second of the 64 mask
bits for physics that is the *same* force; it would let a body carry both laws at once, silently
double-counting drag with no validation able to see it; and it would need `_refresh_cowell_plan`'s
foreign-bit set updated, coupling a density table to the compiled dispatch plan. A `density_model`
column costs no bit, is mutually exclusive by construction, and rides through
`sweep.ForceModelSpec(...).coefficients` with no new machinery at all:

    ForceModelSpec("drag", {"ballistic_coeff": B, "r_ref": EARTH_R_EQ, "omega": EARTH_OMEGA,
                            "density_model": DENSITY_MODEL_LAYERED})

**Why `0.0` is the old law.** `force_model_params` rows are zero-filled on allocation, so a
configuration written before the layered law existed selects the single exponential without being
touched, and `test_drag.py` still passes unchanged. That is a deliberate convention, not a
coincidence: any future selector column added to any model should be numbered so that zero means "what
this model did before".

**Why the kernel evaluates both laws and masks, rather than branching.** A `Simulation` can hold bodies
on different laws at once, so a branch would have to be per body. Two masked NumPy expressions summed
(the masks partition the rows, so exactly one term is non-zero each) keeps the kernel branch-free and
avoids `CLAUDE.md`'s warning against `if np.any(mask):` guards on arena-sized arrays. The band lookup
is one `np.searchsorted` over the whole set.

**How much the table can be trusted.** It was transcribed from memory, like every other citation in
this repo that is marked unverified, so it carries an internal check instead of an appeal to
authority: a piecewise-exponential *fit* is continuous, and 27 interior boundaries close to better
than 1e-4 - three independently remembered numbers per band cannot satisfy 27 constraints by accident.
One boundary does not: 0-25 km closes to 1.4e-3 against a 7.7e-4 rounding bound, and it is asserted as
a named anomaly rather than folded into the tolerance. Nothing orbits at 25 km, and the check that
matters for orbits is that every boundary from 30 km up is inside 1e-4.

**What the choice costs, which is the point.** Two identical satellites, 3 days from 355 km at
`B = 0.4 m^2/kg`, one on the table and one on a single band matched to the table at 355 km with
`H = 60 km`: the table predicts **116.25 km** of decay against the single band's **89.20 km**, a
difference of **27.06 km**, or 30 % more decay. The sign is derivable before it is measured - every
table scale height between 355 and 239 km is under 60 km, so the table is denser all the way down -
and the size is bracketed by the closed form `Delta a = H ln(1 - k t / H)`, which gives -89.7 km at
`H = 60` and -110.2 km at the starting band's `H = 53.3`. Each satellite also matches its *own*
orbit-averaged mean ODE to 1.9e-4 against a derived 8e-4 budget, so the comparison is between two
validated models rather than between a model and a bug. This is a **comparison**, in the sense of the
Validation layers section: the divergence is the result.

Above the match the relationship reverses and then reverses again - the table is thinner at 480 km
(its `H` there is still under 60) and 2.9x denser by 830 km (its `H` is 125 km by then). A single band
cannot reproduce that shape at any choice of `H`, which is the whole argument for the table. The first
draft of the test asserted a monotone divergence in both directions and was wrong.

**Still not modelled** by these two laws: solar and geomagnetic activity, the diurnal bulge, winds,
seasonal and latitudinal variation. Real density at 400 km swings by more than an order of magnitude
over a solar cycle, which is larger than the gap between the two laws. The next section adds activity
through `pymsis`, as a third value of the same selector. No compiled twin: drag is not in the fused
plan, so a compiled density law would have nothing to plug into.

---

## NRLMSIS 2.0: solar activity as a sweep axis, evaluated at the boundary

`msis_bridge.py` wraps `pymsis` (NRLMSIS 2.0; Emmert et al. 2021, citation from memory) as a third
density law, `DENSITY_MODEL_MSIS = 2.0`, with three new `"drag"` coefficients `f107`, `f107a`, `ap`
(columns 7-9). A solar-activity sweep is therefore coefficients alone -
`ForceModelSpec("drag", {..., "density_model": 2.0, "f107": 140, "f107a": 140, "ap": 15})`, or
`msis_coefficients(SOLAR_ACTIVITY_MODERATE)` - with no new mask bit, for the reasons the previous
section gives for the table.

**Boundary, not step.** `pymsis` holds process-global Fortran state and costs ~1.3 us per point; a
step must never call it (`CLAUDE.md`). So `drag`'s `validate_coefficients` hook evaluates, **once per
distinct `(f107, f107a, ap)`**, the MSIS mass density averaged over latitude (8-point Gauss-Legendre
in `sin lat`), local time (8 longitudes at 00:00 UT) and day of year (24 whole days) at 601 altitude
nodes, 0-1000 km (1 km to 200 km, 2 km above) - 1.3 s. The kernel reads that profile through the same
piecewise-exponential evaluator the table uses, each band's scale height chosen to close exactly on
the next node, i.e. log-linear interpolation: +29 us per kernel call against the table's cost. The
profile is **memoised as a pure function of the triple** - the only module state, and it cannot leak
between simulations because nothing else enters it. A kernel that meets an unevaluated triple (a row
switched to MSIS by direct assignment) raises `LookupError` instead of evaluating lazily, because a
lazy evaluation would be `pymsis` inside a step.

**No network, ever.** `pymsis` downloads CelesTrak's space-weather file whenever an index is `None`.
Here the indices are configuration data, never looked up by date: selecting MSIS without all three in
the same call is refused, solar indices on a non-MSIS row are refused (they would be silently
ignored), and a test configures a fresh profile with sockets patched to raise.

**Bit-identical for everything that existed.** The masks for the first two laws are the same booleans
as before and the MSIS term adds exactly `+0.0`; the kernel reads columns 7-9 only on MSIS rows, so
even a legacy 7-column parameter array works. Asserted against a frozen copy of the pre-MSIS kernel on
random mixed arenas, and checked on a 2000-step Cowell + J2 + drag run against the pre-change package
(bitwise equal).

**What the average discards is a model choice**, and its size is measured from MSIS (moderate
activity): the diurnal bulge, **2.30x** day/night at 400 km on the equator (1.68x at 292 km); the
semi-annual season, **1.63x** in the global mean at 400 km; latitude, ~4 % in zonal means and
+0.19 % for a 51.6 deg orbit's sampling at 292 km; and any time variation of the indices. An orbit
whose plane sweeps all local times sees the global mean on average (about two months for a
non-sun-synchronous LEO); a dawn-dusk sun-synchronous orbit never does. The index choice moves the
400 km mean by **24x** between ECSS low (65/65/0) and high (250/250/45) activity - the reason the law
exists.

**The from-memory table, cross-checked** (a plausibility check against an independent model, *not*
verification against Vallado's text): table over MSIS at ECSS moderate activity (140/140/15), every
10 km from 150 to 1000 km, lies between **0.793 (800 km) and 1.278 (160 km)**, inside a derived
factor-1.5 envelope (the table's unstated reference activity at 0.011 per sfu, plus 15-30 %
model-to-model spread). The table is 11-28 % denser than moderate MSIS below 540 km and up to 21 %
thinner above; above 700 km it carries a 2-4 % saw-tooth per band with minima at the base altitudes,
the size of a constant-`H` fit's own error. No band stands out beyond that - the one local feature, a
4 % rise to 350 km, sits on the only thermospheric band whose `H` falls (53.63 to 53.30 km), and
continuity closes both of its boundaries to 1e-4, so it is the fit, not a transcription slip.

**The headline, in Delta-v** (`benchmarks/msis_sweep.py`, 6 days, the station-keeping scenario of
the previous sections, baseline MSIS moderate). Estimated before running from the cycle model with
each law's density and local scale height:

| tier | density / MSIS mod at 292 km | predicted | **measured** | raises | m/s/day |
|---|---|---|---|---|---|
| MSIS moderate (140/140/15) | 1 | baseline | baseline | 13 | 3.047 |
| MSIS low (65/65/0) | 0.280 | -0.7196 | **-0.7193** | 4 | 0.855 |
| MSIS high (250/250/45) | 2.341 | +1.3443 | **+1.3439** | 29 | 7.142 |
| layered table | 1.128 | +0.1288 | **+0.1282** | 14 | 3.438 |
| single band at 355 km, H = 60 km | 0.969 | -0.0311 | **-0.0311** | 12 | 2.952 |

**The solar-activity assumption swings the station-keeping budget by -72 % / +134 % (0.86 to 7.14
m/s/day, a factor 8.3); the atmosphere-model choice by 16 % (2.95 to 3.44).** Solar activity is five
to ten times the larger question. And the two static laws *straddle* moderate MSIS: the station-keeping
section's "single band under-budgets by 14 %" was measured against a table that itself sits 13 %
above moderate MSIS at 292 km - against MSIS the single band is 3 % low. The fast version
(`tests/validation/test_msis_delta_v.py`, 20 h, without the 40.7 h-cycle low tier) measures +1.3432,
+0.1311 and -0.0333 against the same predictions, inside a 1e-2 budget on `(1 + error)`.

**Validation** (`tests/validation/test_msis.py`, `test_msis_wiring.py`, `test_msis_delta_v.py`): the
profile against a direct `pymsis` average built independently, within the log-linear bound
`|f''| dh^2/8` with `f''` from MSIS itself - and, above 100 km, *equal* to the leading-order term to
0.95-0.999; the averaged mass density equal to MSIS's own species sum to 5.4e-8 (column and units);
sea level 1.213 against USSA-76's 1.225; high/low 24x at 400 km; decay against the orbit-averaged
mean ODE on the profile (moderate 3.2e-4, low 5.9e-4, each against a derived budget). Six source
mutations each fail 9-19 tests; dropping the version pin fails none, because in `pymsis` 0.13.0 the
2.1 mass density is bitwise 2.0's - an equivalent mutant, not a gap.

**Not built.** Density as a function of the satellite's own local time and latitude (the diurnal
bulge) - that needs the Sun's direction and an epoch inside the kernel, and a table in more than one
dimension; time-varying indices (storms, 27-day rotation); geodetic altitude; MSIS in `reference.py`'s
truth (truth has no drag, and the Delta-v metric compares configurations, not truth). No compiled twin,
as for the other laws.

---

## Third-body perturbation: the first force that is not parent-relative

`thirdbody.py` registers `third_body`: the point-mass pull of one named perturber, relative to the
body's parent, `mu_s [(r_s - r)/|r_s - r|^3 - r_s/|r_s|^3]`. The second (indirect) term is the
perturber's pull on the parent, which the parent-relative frame inherits. For the Moon each term is
about 5.9e-6 km/s^2 while their difference, the tide, is about 3e-8.

**Naming the perturber.** The kernel contract has only float `params` rows, so the perturber is stored
as a float slot index, which is exact. Slots are not stable across builds, though, so sweeps never
carry them. `sweep.ForceModelSpec.body_coefficients={"perturber": "Sun"}` is resolved to a slot by
`apply_config`. Validation needs the coefficient's value, which `BodyValidator` never sees, so the
registry gained a second, additive hook, `validate_coefficients`. It rejects a missing or non-slot
perturber, and one that is the body itself, its parent, a barycentre, inactive, or massless. A
perturber must be massive, so it is always Keplerian, never Cowell.

**What freezing the perturber costs.** `RK4Integrator` holds every row outside the Cowell set at its
start-of-step value, so the tide is evaluated with the Sun `h/2` late (RK4's effective evaluation time
is mid-step). That error is first order in the step size, not fourth. With the Moon massless,
Sun-Earth is an exact two-body pair, which makes the error predictable as a vector:
`err = (h/2) dr/dtau + O(h^2)`. Here `dr/dtau` is the truth's sensitivity to delaying the Sun, taken from
`reference.py` alone. Measured over 30 days: 2.64 km at h = 3600 s and 1.34 km at 1800 s, within
2.6 % and 1.3 % of that prediction. Without `third_body` the error is 2.5e4 km. A test-only oracle
that supplies the Sun at each stage's true time brings back fourth order (ratios 17.9, 17.0, 16.5,
down to 2.1e-4 km at 2700 s). So the freeze is the only residual, and it dominates RK4's own
error below a step of about a day. Removing it means advancing perturbers per stage, which is not
built. A LEO satellite perturbed by the Moon sees a perturber turning 13 times faster, and its
coefficient is unmeasured.

**Composition.** Its bit is foreign to `_refresh_cowell_plan`, so any Cowell set that includes a
`third_body` body runs on the NumPy path. There is no compiled twin.

---

## Solar radiation pressure: the first model with a discontinuity

`srp.py` registers `srp`: the cannonball form `a = -nu C_r P_srp (A/m) (AU/d)^2 u_hat`, where `u_hat`
points from the body **to** the light source, so the acceleration is anti-sunward, and `nu` in
`[0, 1]` is the shadow factor. With `drag.py` already in place it completes the comparison an analyst
actually wants: which perturbation dominates at which altitude. Measured against `atmosphere.py`'s
Vallado Table 8-4 at `C_r = 1.3`, `C_d = 2.2`, the two are equal at **631 km** - the area-to-mass
ratio cancels exactly from the balance, so the crossover is a property of `C_r/C_d` and the
atmosphere alone. The 800 km figure usually quoted assumes an atmosphere about eight times denser
(elevated solar activity); the crossover moves roughly 140 km per decade of density, so both numbers
are the same physics under different atmospheres.

**Naming the Sun** reuses `thirdbody.py`'s answer wholesale: a float slot in `force_model_params`,
given by body *name* in a sweep (`body_coefficients={"source": "Sun"}`) and resolved by
`apply_config`, validated through the registry's `validate_coefficients` hook. Both `source` and
`p_srp` are mandatory - an unwritten row would name slot 0 at zero pressure and produce exactly no
force, which is the plausible-looking silent failure the per-feature contract's item 5 exists to
catch. The pressure is a coefficient rather than a constant because the literature genuinely forks:
`S = 1367 W/m^2` gives the usual `4.56e-6 N/m^2`, the IAU 2015 nominal `1361` gives 0.44 % less.

**Two shadow models, selected per body**, exactly as `atmosphere.py` is selected: `shadow_model = 0.0`
is a cylindrical umbra (the all-zero default), `1.0` is the conical umbra/penumbra computed as the
occulted fraction of the source's apparent disc. The occulter is the body's Keplerian parent, and
`r_occ <= 0` means no occulter at all - the right answer for a heliocentric body, and the same
silent-no-op convention `drag.py` uses for `scale_height <= 0`.

**The cylinder is discontinuous, and that is the model.** `nu` steps between 0 and 1 across a surface
of zero thickness, so the acceleration jumps by the full `C_r P_srp (A/m) (AU/d)^2`. This is the
first force model here whose right-hand side is not continuous, and it breaks an assumption RK4
makes: on the step that straddles the terminator the four stages disagree about which side the body
is on, the switch is mistimed by up to `h`, and the resulting velocity error is about `Delta_a h / 2`
per crossing with essentially random sign. That is **first order in `h`**, against RK4's own fourth.
It is measured rather than argued: over 1.5 orbits with 4 crossings, successive step-halving
differences fall 4.2x then 1.96x - the 1.96 is the first-order signature, where a smooth problem
would give 16. The same run without a shadow gives `2.27e-5 km` against the cylinder's `1.55e-4`, so
the discontinuity is 7x the rest of the error budget, and the conical model's `2.25e-5` is
indistinguishable from no shadow at all. **That, not the penumbra's extra fidelity, is the reason to
prefer conical.** The real fix is event detection - stepping exactly to the terminator - and it is
not built.

**The frozen source**, the approximation `thirdbody.py` derives in full, applies here too and is
negligible. The fastest-varying part of the geometry is the body's own position, and `RK4Integrator`
stages that correctly; what the freeze misses is the Sun turning at `1.99e-7 rad/s` as seen from the
parent, worth about `5e-6 km` per day at `h = 60 s` - four orders below the eclipse crossings.
Measured end to end, changing `dt` from 40 s to 10 s moves the orbit-scale result by `1.1e-4`
relative, against a model residual of 0.76 %.

**What it is validated against.** A closed form recomputed in SI, with the direction checked as a
*vector projection* onto the body-Sun line rather than a magnitude, because a magnitude cannot see a
sign error. The conical shadow against a one-dimensional numerical quadrature of the disc overlap
that shares no code with the two-arccos closed form. And at orbit scale, the orbit-averaged Gauss
rate for the eccentricity *vector* under a constant inertial force, derived from
`de/dt = [f x h + r (v.f) - f (v.r)]/mu` as `<de/dt> = -(3/2)(h_hat x f)/(n a)` - so the eccentricity
vector grows **perpendicular** to the force. Over 20 orbits that predicts `|de| = 2.674e-5`; measured
`2.678e-5`, a vector mismatch of 0.76 % against a budgeted 1-2 %, with `cos = 0.999972`.

**Composition.** Its bit is foreign to `_refresh_cowell_plan`, so a Cowell set containing an `srp`
body runs on the NumPy path. There is no compiled twin. What it does not model: attitude and flat
panels, Earth albedo and infrared re-radiation (the largest omission, 10-30 % of direct SRP in LEO),
thermal re-radiation, Poynting-Robertson, solar limb darkening, and atmospheric refraction into the
umbra. Only one occulter per body, and it must be the Keplerian parent, so a lunar eclipse of an
Earth satellite is not modelled.

---

## Thrust: the first model whose coefficients are state

`thrust.py` registers `thrust`: `a = (T/m)(d_R R + d_S S + d_W W)`, with `m_dot = T/(Isp g0)` and the
direction law given per body as an RSW vector. `(0, 1, 0)` is a prograde burn. The rotation is
`frames.ReferenceFrames.RSW_to_cart`, which exists for exactly this; nothing is re-derived here.

**Where the mass lives, and why it is not an arena array.** A burning vessel's mass falls, so somebody
has to own a mutable per-body kilogram figure. It sits in column 2 of
`force_model_params["thrust"]`, and `Simulation.step` advances it once per step through
`thrust.deplete_mass`, clamped at the dry mass. The reason is the `ForceKernel` contract: a kernel's
only per-body data channel is `params`. Mass held in an arena array would have to be copied into
`params` before every evaluation, or the shared signature would have to grow a column no other model
needs — drag already folds `1/m` into its ballistic coefficient, and `mu_array` is *gravitational*
mass, which a massless thrusting body does not have. `validate_bodies` refuses `mu != 0` precisely so
that the two never need reconciling; the day a massive thrusting body is wanted is the day mass gets
promoted to the arena, and the promotion is mechanical. `VesselORM.dry_mass`/`fuel_mass` seed the
column at build (`scenarios.powered_vessel`) and are never read again inside a step.

The price is that this one coefficient array is not idempotent across runs: re-running a scenario
means re-seeding `mass_kg`, exactly as it means re-seeding an initial state.

**Frozen mass makes the scheme first order in the mass, fourth order in position.** All four RK4
stages see the step's starting mass, so the integrated `Delta v` is a *left* Riemann sum of `T/m(t)`,
which rises through a burn — the burn is therefore flown slightly too heavy and under-delivers by
`(dt/2)(a_end - a_start)`. It is the same trade `third_body` makes with its frozen perturber, and it
is a prediction, not slack: measured -8.84e-5 of `Delta v` against a derived -8.79e-5, and halving
the step halves it. The direction law carries no such penalty, because the RSW basis is rebuilt from
each stage's own candidate state.

**How the rocket equation is tested at all.** A powered orbit's speed is not `v0 + Delta v`; the orbit
turns. `scenarios.powered_vessel` therefore builds co-located, identical, massless twins in one arena,
one thrusting, and the burn is differenced between them. Their difference obeys
`d(delta_v)/dt = a_T + G delta_r`, so the contamination is the gravity gradient across their
separation, `~(n tau)^2`. Placing the burn on a 400 000 km orbit drives that to 1e-6, and the twin
difference then reproduces the discrete delta-v to 4.9e-7. Orbit raising is checked separately
against the energy-derived `da/dt = 2 f/n` (7.1e-6 relative over five orbits, of which 6.8e-6 is the
integrator drift the thrust-free twin measures), with a radial burn as the axis control: it must
raise `a` only at `O(e)`, and transposing R and S in the kernel is what that test exists to catch.

**Composition.** The bit is foreign to `_refresh_cowell_plan`, so any Cowell set containing a
thrusting body runs on the NumPy path. There is deliberately no compiled twin.

**What it does not do.** No attitude or slew rate, no throttle- or pressure-dependent `Isp`, and no
coupling into drag's ballistic coefficient. Burn duration is quantised to `dt`: fuel that would run
out mid-step is spent over the whole step, so no propellant is invented but the final step's impulse
can exceed the physical one by up to `T dt`.

---

## Impulsive manoeuvres: physics on a third axis

`manoeuvres.py` adds an instantaneous Delta-v: position continuous, velocity discontinuous, elements
re-derived. It is the complement of `thrust.py` — and the first piece of physics here that **every**
propagator can use, which is the point of it. A Delta-v on a Keplerian body is not an integration
problem at all; it is `rv_to_coe` of the new `(r, v)`. That makes the analytic tiers usable for
mission work, which they were not before: a Hohmann transfer on the Keplerian propagator now costs
two element re-derivations and reaches the target radius to `3e-6 km`.

**Why it is not in `registry.py`.** Both existing axes are wrong for it. A force model is an
acceleration `forces.compose_accelerations` sums and an integrator consumes — an impulse smeared over
a step *is* `thrust.py`, which already exists, and running an impulse through that axis would deny it
to the analytic tiers for no gain. A `PropagatorType` is a way of advancing state through time, which
an impulse is not. So a third axis would be needed ("scheduled events"), and one event type does not
justify it: manoeuvres are a plain list on the `Simulation`, sorted by epoch, configured through
`schedule_delta_v` exactly as propagator assignment is configured through `set_propagator`. The day
`sweep.py` wants to enumerate mission profiles as configuration data, that list is what it would
serialise, and that is the point to add the axis rather than before.

**Per propagator, which is the whole difficulty.** The arena keeps a body's state in up to three
places and which of them is *authoritative* depends on the propagator. A Cowell body's truth is
`global_states`, so an impulse is one vector addition and its `coe_states` row stays stale as
documented. A Keplerian body's truth is `coe_states` — `KeplerianPropagator` rewrites `local_states`
from the elements every single step — so an impulse that changed only the Cartesian state would be
silently erased by the next `step()`, with no error and a perfectly plausible orbit. A secular-J2
body carries a third thing: `_secular_j2_rates`, cached at `set_propagator` time because `p`, `e` and
`i` are constant under that theory. An impulse changes all three, so the cache is rebuilt here. That
is the failure this feature's sharpest test exists for: a stale rate leaves the position, the
elements and the energy all correct and only the nodal drift wrong, which nothing else would notice.
The closed form is exact — a prograde kick `dv = d·v_c` on a circular orbit scales the nodal rate by
`(1 − 2d − d²)^{3/2} (1 + d)^{−4} = 1 − 7d + 22d²`, matched to 2.2e-13.

**Step splitting, and its honest cost.** `step()` cuts itself at a manoeuvre's epoch, so a burn lands
where it was asked for rather than at the next multiple of `dt`; quantising the Hohmann
circularisation to a 60 s grid would place it 21 s off apoapsis and leave `e ~ 1e-4` instead of
`4e-10`. The split is for the whole arena, so the question is what it costs a body that is not
manoeuvring. For an analytic body, nothing: propagating `h1` then `h2` is the same closed-form advance
as `h1 + h2` (measured difference exactly 0.0). For a **Cowell** body it cannot be nothing and is not
claimed to be — RK4's stages sit at nodes fixed by the step size, so one split perturbs that step by
its own local truncation error, `r(nh)^5/120`, measured at `4.7e-4 km` over the following 2.3 orbits
against a derived bound of `1.5e-3 km`, and 2.4e-8 of the manoeuvring vessel's own displacement.

**Restrictions mirror `set_propagator`'s**, and for the arena's reasons rather than physics': heads
(their motion is a reflex kick recomputed every step, so an impulse on one is overwritten), barycentres,
roots, inactive slots, and `mu != 0` — a massive body's impulse would have to reach its barycentre's
own element row, a different slot this call does not touch. The kernel computes the entire transaction
before committing any of it, so a refusal leaves the arena bit-identical, including the bodies of the
call that were valid.

**What it does not do.** No propellant coupling: an impulse does not draw down `thrust.py`'s mass,
because the rocket equation needs an `Isp` belonging to the stage rather than to the impulse, plus a
policy for an impulse the tanks cannot deliver — guessing either was worse than leaving it out. RSW is
the only input frame, and there is no finite-burn correction; `thrust.py` is the model for when burn
duration matters.

---

## Event-driven step splitting: finding the epoch instead of being told it

`manoeuvres.py` established that `step()` can cut itself in two. It knows *where* to cut because a
scheduled impulse carries its epoch. `events.py` is the same cut with the epoch defined implicitly —
as the root of a continuous scalar function of the arena state — and so has to find it during the
step. That is the whole of the new machinery; everything downstream (`_advance` runs, the clock is
pinned to `t + dt`) is the manoeuvre path unchanged.

**The problem is real and was already measured.** `srp.py`'s cylindrical shadow makes the
acceleration discontinuous at the terminator. `test_srp.py` records what that costs RK4: step-halving
error ratios of 4.23 then 1.96 where a smooth problem gives 16, and seven times the error of running
with no shadow at all. The conical shadow makes the symptom go away by smoothing `nu`, which is a fine
reason to prefer it and not a fix. The fix is to stop the step at the discontinuity so that no step
ever integrates across it — which is what a Runge-Kutta method's order derivation assumes in the first
place.

**Detected per body, split per arena.** The event function returns one value per body and each body's
sign change is tracked separately, so a constellation registers its satellites' crossings
independently. But there is one clock: `_advance` moves the whole arena, because `calc_global` and the
Cowell/secular-J2 re-base assume every row is at the same instant. Per-body step sizes would be a
different engine. So when one body crosses and another does not, *everything* takes two sub-steps, and
what that costs the passengers is exactly what manoeuvre splitting costs them — nothing for an
analytic body (measured: a Keplerian body 1 AU out moves less than 1e-4 km under eight splits), and
one extra local truncation error `r(nh)^5/120` for a Cowell one.

**Locating the crossing.** The engine has no dense output, so there is no way to ask for the state at
`t + tau` without advancing to it: the root find works by snapshot, `_advance(tau)`, evaluate,
restore. Several bodies may cross in one interval and the split has to land on the *earliest*, so the
vector is reduced to `H(tau) = max_j -sign(g_j(0)) g_j(tau)`, which is negative until any body has
crossed and non-negative after — one scalar root find for any number of bodies. The method is false
position with the Illinois modification, terminating on **bracket width** rather than on `|H|`,
because the quantity that must be small is a time.

**Splitting alone does not restore the order, and that is the part worth reading.** Two things got in
the way, both found by measuring rather than by reading:

- *RK4's stages are not on the trajectory.* Stage 4 samples `r + h v(k3)`, off the solution by
  `O(h^3 |da/dt|)` — 5.6e-4 km at `h = 5 s` in LEO. A sub-step ending exactly at the terminator
  evaluates that stage on whichever side of the surface an off-trajectory point falls, and a stage on
  the wrong side carries weight 1/6 of a full `Delta_a h`. Measured 3.38e-6 km against a derived
  `n Delta_a (h/6) T_rem` = 3.26e-6 km: still first order, and at `h = 1.25 s` *worse* than not
  splitting. The remedy is a **branch latch** — for a sub-interval known to contain no crossing, the
  model is pinned to the branch it starts on, so the right-hand side really is smooth over that
  interval at all four stages. `srp.py` carries it as an engine-owned `shadow_latch` column, written
  and released inside one step, with the same precedent as `thrust.py`'s mutated mass column.
- *The event function quantises.* `umbra_clearance` is `sqrt(...) - r_occ`, which rounds to exactly
  `0.0` within an ulp of the surface — and a converging root find lands there routinely. `H` is then
  `-0.0`, and `-0.0 < 0.0` is false, so a strict "are we past it?" test declares the crossing resolved
  while the body sits on a surface whose own membership test is strict and reads *lit*. The
  postcondition is checked with `<= 0` and nudged by one tolerance until the sign has genuinely
  flipped. This is precisely the class of bug this repo's item-5 rule exists for: it raised nothing,
  crashed nothing, and cost a factor of 1000 in accuracy at the finest step.

**What it delivers.** On `scenarios.eclipsed_satellite`, error against a fine reference at steps
20/10/5/2.5/1.25 s:

| | ratios | error at `h = 1.25 s` |
|---|---|---|
| no shadow (control) | 17.31 16.68 16.30 17.18 | 5.37e-9 km |
| cylinder, no split | 16.99 6.40 2.57 1.46 | 9.18e-7 km |
| cylinder, split | 17.28 16.66 16.23 17.83 | 5.19e-9 km |

Fourth order, fully recovered, at the no-shadow control's own error. What is left is the crossing-time
tolerance: `err / (Delta_a tol_s T_sum)` measured at 2.0–2.7 across three decades, which at the
default `tol_s = 1e-6 s` is 7.4e-12 km — orders below RK4's own truncation, by design.

**Why a new scenario.** None of the above is observable in `sun_earth_moon`. A Cowell body whose
`global_states` row is heliocentric (`~1.5e8 km`) loses nine digits to cancellation against its
parent, flooring the trajectory at `~1e-5 km` over 1.5 LEO orbits — the same size as RK4's truncation
at `h = 10 s`. Fourth-order convergence cannot be measured there with or without a shadow.
`eclipsed_satellite` puts Earth at the arena root and a massless luminous marker 1 AU out; the floor
drops by six orders and the ladder reads 16. That is a fact about the *arena*, not about events, and
it is worth knowing before anyone else tries to measure an integrator on a heliocentric scenario.

**Cost.** A step with no crossing costs two evaluations of the event function and the snapshot copies
— **no extra propagation**, which is what keeps it bit-identical to today. A crossing costs 6 to 12
trial propagations plus three advances instead of one; over 1.5 orbits at `dt = 10 s` that is 23 trial
propagations against 880 steps, under 3 % of the run.

**What is not built.** No interior sampling, so a body crossing an even number of times inside one
interval is invisible — the standard endpoint-detection blind spot, asserted as a test rather than
left in prose, and defended by the fact that `dt` is seconds while an eclipse is thousands. No
tangential events. No adaptive integrator: an adaptive method would want the event search folded into
its own step-size controller (reject the trial step, retry ending at the crossing) rather than layered
above it, and it would want the latch to survive a rejected step. `events.py` is deliberately written
against the `Integrator` protocol's *caller*, not against RK4, so that change is confined to
`_advance_with_events`.

---

## Observation geometry: reporting in the units of a decision

`geometry.py` answers "when can this station talk to that satellite", which is the question a
constellation or network study actually asks. The engine's existing output is a position error in
kilometres, and no one procures on kilometres; they procure on contact minutes per day, on gap
length, on whether a marginal pass clears the mask. This module is the primitive layer that a later
sweep-level metric needs, and it is deliberately *only* that layer.

**Three functions, no state.** `elevation_azimuth` (look angles from a station fixed to the rotating
central body), `access_windows` (rise/set intervals above a mask angle), `line_of_sight` (two points
past a sphere). Each is a pure function of arrays, in the same spirit as `viz.py`'s transforms and
for the same reason its docstring gives: a function that takes no `Simulation` can be validated
against closed-form geometry with no propagation in the loop, so a failure is unambiguously the
transform's. Nothing is registered in `registry.py` — there is no acceleration to compose and no
state to advance, the same argument `manoeuvres.py` makes for itself one axis over.

**Spherical, and said so.** Latitude is geocentric, altitude is above a sphere of `body_radius_km`.
The engine carries no reference ellipsoid: its J2 is a gravity coefficient with no flattening
attached, and `viz.GroundTrack` already reports spherical latitude. Adopting a geodetic station
without an ellipsoidal gravity field would be internally inconsistent in a more expensive way than
being uniformly spherical is.

**The rotation epoch is absolute, unlike `viz.ground_track`'s.** `theta(t) = theta0 + omega (t -
epoch_s)` with `epoch_s = 0.0` by default, where `ground_track` references `theta0` to `times_s[0]`.
That divergence is deliberate and was bought with a bug: a ground track is a shape, and rotating the
whole figure leaves it valid, but a *pass time* referenced to the first sample moves when the grid
does. Sampling one 550 km pass on a grid starting at `t = 28 s` instead of `t = 0` displaced every
rise and set by 2.0 s — ten times the interpolation error the edges are refined to, and completely
silent. See `docs/engineering-log.md`.

**Interpolated edges, and why that is the whole point.** Snapping a crossing to the nearest sample
quantises the reported duration by up to one sample interval — on a 60 s sweep grid, 7 % of an 800 s
LEO pass, and a *biased* 7 %, not noise. Linear inverse interpolation on the elevation series makes
the edge `O(h^2)`, with the error term written out in `access_windows`' docstring. The sign matters
and the tests assert it: the elevation curve is **convex at the horizon** (it leaves at rate `n -
omega` and reaches far more than that by mid-pass), so rises come out early, sets late, and durations
are systematically long — the opposite of the intuitive "concave, so short" reading. At 550 km the
coefficient is `Omega cot(lambda_0)/2 = 1.21e-3` per second, giving 0.23 s of edge error at `h = 30 s`
against a 784 s window, and a measured convergence ratio of 4.03 under halving.

**What is validated, and against what.** The coplanar circular pass has exact answers throughout:
horizon half-angle `acos(R/r)`, mask half-angle `acos((R/r) cos e) - e`, horizon range
`sqrt(r^2 - R^2)`, duration `2 lambda / (n - omega)`. `scenarios.ground_station_pass` seeds exactly
that geometry, and `tests/validation/test_geometry.py` measures 90 deg overhead to 8e-15 rad, 0 deg at
the horizon to 7e-16 rad, and durations matching the closed form plus the derived interpolation bias
to within 0.5 % of that bias. The inclined case is checked against the general
station–satellite–centre triangle with the station's inertial position written out independently,
because the equatorial case has an identically zero SEZ *south* component and cannot discriminate a
transposed south/east axis — the same symmetry trap `hohmann_pair` documents.

**Range rate, and where the boundary sits.** `elevation_azimuth` optionally takes velocities and
returns `d|rho|/dt`, **positive = opening (receding)**. It extends that function rather than
standing beside it because it is the derivative of the `rho` that function already builds: a second
entry point would have to repeat the station construction and the `-theta` rotation, and repeating
that rotation is precisely how this module goes quietly wrong. The content of the calculation is the
transport term — the station is fixed in the *rotating* frame, so its inertial velocity is
`omega x r_station` and `rho_dot = v_body - omega x r_station`. Drop it and the coplanar closed form
silently replaces the station-relative rate `n - omega` with `n`: 0.4646 km/s of 6.521 km/s at the
horizon of a 550 km equatorial pass, 7.1 %, and 0.4646 km/s is exactly `R omega`, the station's own
ground speed, because at the horizon the line of sight is tangent to the sphere at the station.
`tests/validation/test_geometry.py` asserts that difference as a number, checks the analytic rate
against a central difference of the range series (which shares no algebra with it, and converges at
the claimed second order — ratios 4.000), and pins the sign as two signs rather than a magnitude,
since an inverted convention passes every magnitude check and inverts every downstream Doppler.

The boundary is drawn *at* range rate: Doppler shift, carrier frequency and link margin are
properties of a radio, not of an orbit, and no part of this engine computes them.

**Deliberately not built.** No refraction or
terrain horizon, no multi-station scheduling or conjunction search, and no sweep-level aggregation *in this module*:
contact totals and pass-count statistics are the metric layer that consumes this one, and they
now live in `access.py` (next section) rather than here, so `geometry.py` stays free of
`Simulation`. `AccessWindow.peak_elevation_rad` is the one field that is *sampled* rather than refined,
`O(h)` for an overhead pass because the elevation has a corner at the zenith, and it is labelled as a
lower bound rather than quietly parabola-fitted.

---

## Access metrics: the sweep's second currency

`sweep.py` reports position error in kilometres. No decision is denominated in kilometres. A
constellation or ground-station study decides on *contact*: when a pass opens, how long it lasts,
whether a marginal pass exists at all. A model wrong by 500 km that moves every window by two seconds
changes nothing; one wrong by 5 km that deletes a pass changes the plan. `access.py` is the layer that
converts a sweep into those units, and `geometry.py` is the primitive layer under it.

**Why a module of its own rather than more of `sweep.py`.** Three reasons, none stylistic. `sweep.py`
imports `reference`, `benchmark` and `simulator`; the access metric additionally needs `viz`
(sampling) and `geometry` (look angles), so folding it in would make the harness depend on the
presentation stack. The interesting half of the feature — *matching* model windows to truth's — is
pure interval arithmetic over `geometry.AccessWindow` records, and keeping it where no `Simulation`
is in scope is what lets it be tested on hand-built window lists whose right answer is written down.
And `run_sweep`'s existing contract is then unchanged *by construction* rather than by claim: it gains
one optional keyword and one optional field, and the position-error path is untouched. The test that
matters asserts the three `ErrorStats` fields are **bit-identical** with and without `access=`.

Like `geometry.py` and `manoeuvres.py`, nothing here is registered in `registry.py`. There is no
acceleration to compose and no state to advance; the registry's two dispatch mechanisms have no third
slot for "a metric", and an `AccessSpec` is an argument to `run_sweep` in the same position
`oblateness` already occupies.

### Matching is the design decision

Before anything can be differenced, each model window must be paired with the truth window it *is*.
The rule is the strictest defensible one: **two windows may be paired only if they overlap in time**;
among overlapping candidates, take the largest overlap first, remove both, repeat. Anything left over
is an orphan — an unmatched truth window is a **lost pass**, an unmatched model window a **gained**
one — and orphans contribute to the counts and to total contact time but never to a shift statistic.

The alternative, nearest-midpoint pairing, needs a tolerance, and a tolerance here is a dial that
converts "this model lost a pass" into "this model was late". That is exactly the distinction the
metric exists to preserve. If a model's window does not overlap the real one at all, an operator who
pointed an antenna at the prediction would have received nothing for the whole of the real pass and
nothing for the whole of the predicted one: two failures, not one late pass, and a reported "900 s
shift" would be a fiction. The price is that a badly phased tier reports `lost == gained == n` rather
than one enormous shift, which is the honest summary of a model whose passes no longer correspond to
reality.

Greedy is unambiguous here because windows within one list are disjoint and ordered; ties break by
earliest truth window then earliest model window, so the result does not depend on input order, which
a test asserts by reversing both lists.

### The sampling grid is derived, not chosen

`geometry.access_windows` interpolates its edges linearly. That error is an `O(h^2)` **bias**, not
scatter — rises early, sets late, by `C a b` with `C = Omega cot(lambda_0)/2 = 1.21e-3` per second at
550 km. At `h = 30 s` that is 0.27 s, which is the *same order* as the quantity being measured: the
best tier in this engine is 1.88 km out after 24 h, and `(1.88/6921)/Omega = 0.265 s`.

The bias is dealt with by sampling truth and model on **one shared grid**, so it is common-mode.
Writing it as `beta(a) = C a (h - a)`, the model's crossing sits at `a + Delta` and the residual is
`C Delta (h - 2a - Delta)`, bounded by `C Delta h`. The bound survives the case where the two
crossings land in different brackets, because `beta` vanishes at both ends of a bracket — worth
checking rather than assuming, and the first draft of the derivation got it wrong in the pessimistic
direction. At `h = 60 s` and `Delta = 0.27 s` the residual is 0.020 s, 7 % of the signal.

**The grid also has a constraint from the other direction, and it is a trap.** `viz.sample_states`
divides a sample interval into sub-steps of *at most* `max_dt`. Ask for a 10 s access grid from a
`dt = 60 s` configuration and it will take one 10 s step per sample — propagating a sixfold finer
model than the one the sweep timed and scored a position error for, silently and entirely plausibly.
`sweep.access_metrics_for` raises unless `config.dt` divides the sample spacing, and
`DEFAULT_SAMPLE_DT_S` is 60 s for that reason rather than the 10 s the bias argument alone would
prefer.

### What it measures, on the tiers that exist

12 satellites at 550 km / 53 deg, three stations, 5 deg mask, 24 h, 189 true passes
(`docs/figures/access_windows.png`). Mean and max |rise shift|, then passes lost / gained:

| Tier | mean | max | lost | gained |
|---|---|---|---|---|
| Kepler | 31.65 s | 154.21 s | 5 | 4 |
| Secular J2, osculating-seeded | 30.82 s | 115.82 s | 1 | 2 |
| Secular J2, mean-seeded | 1.60 s | 8.10 s | 1 | 0 |
| Cowell + j2, `dt = 60 s` | **0.063 s** | **0.223 s** | 0 | 0 |

Against a 1 s threshold — roughly the acquisition pad a real schedule already carries — Cowell + J2 is
the first tier that clears it and the only one that neither loses nor invents a pass. Two results the
kilometre metric cannot produce. The **discrete** one: Kepler does not merely mistime passes, it
deletes five that happen and predicts four that do not, and those nine are scheduling decisions rather
than error bars. The **ranking** one: mean-seeded secular J2 is 100x better than Kepler in kilometres
(4.04 km against 607.61 km) but only 20x better in mean window shift, and it still drops a marginal
pass — an averaged theory reproduces along-track position far better than it reproduces the elevation
profile near the horizon, which is where a marginal pass lives.

Cowell's 0.223 s is the check on the measurement itself: its 1.88 km position error corresponds to a
0.265 s pass shift at `Omega = 1.024e-3 rad/s`, so what is being reported is the along-track share of
a known error and not the sampling grid.

### The contact dataset

`AccessMetrics` says how wrong a model's *schedule* is. The export to a downstream network or ISL
study needs the contact's *contents* as well, which for a link budget is range and range rate.
`contact_windows` / `contacts_from_simulation` return `ContactWindow`: the `AccessWindow` unchanged,
plus a `ContactSample(time_s, range_km, range_rate_km_s)` at rise, peak elevation and set. Those
three instants are chosen because the window already names them — no new grid, no re-propagation —
and because they bracket what a link budget asks: the edges are the worst-case range and the
*extremes* of range rate (`rho'' = 0` at the horizon exactly, for a coplanar circular pass), the
peak is closest approach, where the range rate passes through zero.

Rise and set are interpolated times, so both series are linearly interpolated onto them, `O(h^2)`;
the peak is a grid time and needs no interpolation but inherits `peak_elevation_rad`'s `O(h)`
sampling offset, so its range rate is bounded by `rho'' h / 2` rather than being zero — 0.42 km/s at
`h = 10 s` on the reference pass, measured 0.070. Nothing about the metrics changed: the windows
`contact_windows` returns are asserted equal to `windows_from_positions`' windows, and both go
through one `_topocentric` call so the two paths cannot drift apart.

### Deliberately not built

No scheduling or conflict resolution, no gap statistics (maximum outage is the obvious next
reduction and needs a use case first), no Doppler shift or link budget (range rate is the kinematic
quantity; the rest belongs to a radio), no range-rate *error* statistic in `AccessMetrics` — the
dataset is an export, not a second metric — and no matching across
stations or bodies — `station_index` and `body_index` partition the problem, and a pass one station
saw is not a pass another saw however well the intervals line up. `peak_elevation_rad` is not
differenced: `geometry.py` samples it rather than refining it, so it is a lower bound, and it appears
here only where the *tests* choose a mask angle with it.

## Inter-satellite links: the contact dataset's second half

`access.py` exports ground contacts. The downstream network repository (routing, handover, link
budgets) also needs **satellite-to-satellite** contacts, and the boundary is `README.md`'s: visibility,
windows, range and range rate are properties of orbits and live here; anything that needs an antenna or
a graph lives there. `isl.py` is that half. It adds no physics, no registry entry and nothing to
`step()`.

**Why a segment clearance, and why continuous.** Two satellites see each other when the straight
*segment* between them clears the body by a grazing altitude `h_graze_km` - the atmosphere and
refraction margin a crosslink keeps (~100 km is common), a required `IslSpec` field so no spec can
silently assume a bare-surface link. `geometry.line_of_sight` already had the geometry, including the
`[0, 1]` clamp that keeps two radially stacked satellites from reading as occulted (the infinite line
through them passes through the centre; the segment does not). But a boolean on a grid can only snap a
window edge to a sample, a quantisation of up to one sample interval. `geometry.segment_clearance` is
the same test as a signed distance in km - `min |r1 + tau (r2 - r1)| - (R + h_graze)` over the clamped
`tau` - so edges are inverse-interpolated exactly as elevation is. It is continuous with a continuous
first derivative across the clamp switch, which is all second-order interpolation needs. An optional
`max_range_km` is folded in as `min(clearance, max_range - range)`, still one continuous margin.

**Why reuse access matching rather than write a second one.** The metric question is identical -
pair each model window with the truth window it *is*, then difference the edges - and `access.py`'s
answer to it (overlap only, greedy largest-first, orphans counted as lost or gained and never given a
fictional shift, clipped edges excluded from shifts but not from contact time) is the part of that
module most worth not re-deriving. So `IslWindow` adapts to `AccessWindow` (`station_index <- a`,
`body_index <- b`, `peak_elevation_rad = nan`, which the matcher never reads) and `access.compare_windows`
does the rest; `SweepResult.isl` *is* an `AccessMetrics`, grouped per pair. The dataset types are thin
siblings rather than re-uses: an ISL has no station and its peak is a closest approach, not an
elevation, and a range stored in a field named `peak_elevation_rad` is the kind of plausible mis-read
this engine's docs spend most of their effort preventing. `access.ContactSample` *is* re-used unchanged
(range, range rate **positive = opening**).

**The edge bias has the opposite sign.** `geometry.access_windows`' linear interpolation carries the
bias `t_hat - t* = -(f''/(2 f')) a b`. (Its docstring had the sign of that expression flipped relative
to its own - correct - conclusion that convex elevation reads rises early; fixed alongside this work.)
Elevation is convex at the horizon; the ISL clearance is **concave** at its crossing - a maximum at
conjunction, falling away on both sides - so ISL rises read **late**, sets **early**, windows **short**.
On `scenarios.coplanar_satellites` at 550 / 1200 km, `h_graze = 100 km`, every rise and set is
closed-form (tangent angle `psi* = acos(rho/r1) + acos(rho/r2)` = 52.047 deg, window 13 152.4 s every
45 485.9 s) and `|f''/(2 f')| = 8.58e-5 /s`: 0.077 s worst case at `h = 60 s`. Each measured edge equals
`-(f''/(2f')) a b` for its own bracket offsets within the `O(h^3)` remainder, and a phase-locked edge
halves by 4.00. Truth and model share one grid, so the bias cancels to `C Delta h` exactly as for ground
access.

**Wiring.** `run_sweep(..., isl=IslSpec(...))` is `access=`'s twin: the same divisibility rule (factored
into `sweep._sample_grid`, one message for both), one dense truth per sweep - **the access truth itself**
when both are on and their grids coincide, so turning on ISL beside access costs no truth integration
at all - and one extra propagation per configuration. `ErrorStats` and `access` are bit-identical with it
on (tested), and `score_external` scores an `ExternalTier` (SGP4) on ISL windows the same way it scores
access, refusing a spec centred on a different body from the tier's positions. When the ISL grid differs
from access's, the ISL truth is a separate `reference_for` call; a test asserts every truth call in a
sweep receives identical model keywords, so a new truth option threaded into one call and not the other
fails loudly instead of scoring ISL against a different model.

### What it measures, on the tiers that exist

12 satellites, 550 km / 53 deg, **3 planes of 4** (not the access figure's single plane of 12, in which
every pair keeps a fixed phase and a link is always up or never), 24 h, truth J2..J6, `h_graze = 100 km`,
all 66 pairs, 60 s grid; the ground columns are the same sweep's three stations at a 5 deg mask
(`python benchmarks/isl_sweep.py`, ~14 s). In-plane neighbours are 90 deg apart, beyond the 41.55 deg the
grazing sphere allows, so only the 48 cross-plane pairs link: 366 windows, 251-981 s long.

| Tier | km (median) | ISL mean / max \|rise\| | ISL mean / max \|set\| | ISL lost / gained | ISL contact error | ground mean / max \|rise\| | ground lost / gained |
|---|---|---|---|---|---|---|---|
| Kepler | 906.4 | 20.61 / 41.56 s | 32.93 / 75.55 s | 0 / 0 | +4310 s | 55.75 / 163.66 s | 7 / 6 |
| Secular J2, mean-seeded | 6.33 | 0.323 / 0.767 s | 0.304 / 0.575 s | 0 / 0 | +92.4 s | 2.069 / 12.54 s | 1 / 0 |
| Cowell + j2, 60 s | 2.01 | 0.125 / 0.347 s | 0.124 / 0.338 s | 0 / 0 | +1.4 s | 0.107 / 0.730 s | 0 / 0 |
| Cowell + j2 + zonal, 60 s | 1.877 | 0.082 / 0.226 s | 0.097 / 0.261 s | 0 / 0 | -5.1 s | 0.082 / 0.255 s | 0 / 0 |

The estimate written before the run (`benchmarks/isl_sweep.py`'s docstring) rested on three facts: an ISL
cannot see a common rotation of the constellation; an along-track error common to both ends is a pure
time translation of the pair geometry, `-delta_s / v` = 0.132 s per km; radial error enters at
`0.467 (dr_a + dr_b) / |c'|`, with `|c'|` estimated at 1.3-2.2 km/s (measured afterwards at the truth's
edges: 1.65-1.71 km/s). Against it:

- **Cowell + j2 + zonal, 60 s** - RK4's common phase lag and nothing else: predicted 0.094 s mean,
  0.25 s max, every window early. Measured 0.082 / 0.226 s rise, signed mean -0.081 s (equal to the
  mean-abs: every window moves the same way). The duration error, -0.018 s, was predicted as ~0 and is
  not: the pre-run decomposition also showed a *common radial* error of -0.023 km at 24 h, which closes
  the link margin at both ends by `2 x 0.467 x 0.023 / 1.69` ~ 0.013 s - the right size, and a term the
  estimate listed in its table but did not carry through.
- **Cowell + j2, 60 s** adds the missing J3..J6: predicted ~0.15 / 0.4 s, measured 0.125 / 0.347 s.
- **Secular J2, mean-seeded** - predicted ~1.5-2 s mean, measured **0.32 s**: a 5x over-estimate, and
  the reason is instructive. The estimate scaled each satellite's day-mean radial short-period error
  (3.3 km) as if the two ends were independent; at the link edges their radial errors are strongly
  *anti*-correlated (J2's short-period radius goes as `cos 2u`, and a cross-plane pair crossing the
  grazing sphere sits at arguments of latitude that nearly cancel it), and the measured pair sum there is
  **0.93 km**, not ~4.7. The linearised shift `-delta_c / c'` evaluated at the truth's edges reproduces
  the measurement (0.314 s against 0.313 s, the mean of rise and set).
- **Kepler** - predicted ~30 s mean / ~110 s max, about half its ground error or less; measured 20.6 /
  41.6 s rise, 32.9 / 75.6 s set, ground 55.7 s: **0.37x** the ground error at the rise, 0.59x at the
  set. The magnitude and the ratio held; the mechanism in the estimate (half the pairs cancelling
  alternating-sign seed errors) did not - every Kepler ISL window is *late* (signed mean = mean-abs), so
  at the edges the along-track errors act as a common lag. The linearised `-delta_c / c'` again reproduces
  it (26.7 s against 26.8 s, the mean of rise and set). The nodal regression Kepler omits (4.5 deg in a
  day) moves no ISL window at all - a tested invariance - though it is only part of Kepler's km error
  (rotating the Kepler solution by it takes the median from 906 to 861 km; the rest is seed-dependent
  along-track drift).

**Does a tier that is fine for ground contacts stay fine for ISLs?** Yes, and at this horizon the
converse holds too: **the ISL windows are no harder than the ground ones for any tier**, and markedly
easier for the analytic ones. The two Cowell tiers score within 20 % of their ground figures (same mean,
smaller worst case), because their error is almost entirely a common along-track lag, which shifts a
ground pass and a link window by the same `delta_s / v`. The analytic tiers do much better on links -
secular J2 6.4x, Kepler 2.7x - because the ISL is blind to the common rotation and partly blind to
errors correlated between the two ends, and **no tier loses or gains a single ISL window** where Kepler
loses 7 ground passes and invents 6. The ISL windows are robust in count because none of them is
marginal: losing one takes a shift comparable to its length, and the shortest here is 251 s, whereas a
ground pass whose peak elevation sits just above the mask disappears under a small cross-track error. So for a 24 h link schedule, mean-seeded secular J2 (sub-second, nothing lost) is already
adequate where it is not for ground passes (2 s mean, 12.5 s worst, one pass lost); for ground
contacts Cowell + J2 remains the first tier under the 1 s pad.

**How this feeds the downstream repository.** `isl_contacts_from_simulation(sim, bodies, times, spec)`
(or `isl_contacts` on sampled states) is the export: per pair, `IslContact(window, rise, peak, set)`
with range and range rate at each, the same `ContactSample` the ground dataset carries, so a consumer
builds one contact graph from both. Everything a link budget needs beyond that (antennas, pointing,
Doppler as a carrier offset, data rate) is a property of hardware and stays downstream.

**Deliberately not built.** No link budget, no pointing or antenna constraints, no Sun-exclusion angle
for optical links (a geometric constraint that would fit here later, as a second `min` term in the
margin, once a consumer needs it), no ellipsoid or atmosphere model beyond the constant grazing
altitude, and no inter-pair matching - a pair's windows are matched only to the same pair's.

## Station-keeping: the atmosphere model in the units of propellant

`access.py` put model error into contact windows. `stationkeeping.py` puts the **atmosphere** choice
into the unit an operator signs off on: the Delta-v it takes to hold a satellite in an altitude band.
It adds no physics — drag, density and impulse are `drag.py`, `atmosphere.py` and `manoeuvres.py` —
only the decision of when to burn and how much, and like `manoeuvres.py` and `access.py` it is not a
registry entry: a controller is neither an acceleration nor a propagator.

**The policy is data.** `StationKeepingSpec(lower_km, upper_km, r_ref_km, impulses)`: when the mean
altitude falls to `lower_km`, raise it to `upper_km`, as a two-impulse Hohmann transfer between mean
radii (circular after) or one prograde burn. `StationKeeper.observe()` runs after every step;
`run_station_keeping` drives it. The burn log (`Burn`) and `summarise_burns` give total Delta-v, number
of raises and mean interval per body.

**What it keys on is the design decision.** Under J2 a 300 km, 51.6 deg orbit's osculating altitude
swings 12 km peak to peak about its mean; a controller keyed to it fires once per orbit at the J2 dip
while the orbit is kilometres inside the band. The quantity is the **one-period time average of
osculating altitude**, which annihilates every short-period term because each is periodic in the
argument of latitude. Two corrections make it good to metres rather than tens of metres: the window
is the trapezoidal integral over *exactly* one Keplerian period (a rectangle of `round(T/dt)` samples
leaks the oscillation), and it is extrapolated `T/2` forward along the slope of two consecutive
windows (a trailing mean reads `|mdot| T/2` = 0.18 km high, 7 % of the band). No Brouwer/Kozai mean
element: that would be a theory-internal osculating-to-mean conversion `CLAUDE.md` scopes narrowly,
and a time average needs no theory. The price is that the controller is blind for ~2.5 orbits after
each raise while it re-establishes the average.

**Between steps, not an `events.Event`.** An event function must be a pure read of the instantaneous
arena, and a mean is a function of history; `events.Event` has no action to fire a burn from; and
the timing precision would be worthless — a smooth threshold crossed at `|mdot| dt` = 2–4 m per step
moves nothing, and cancels out of the Delta-v rate. **What belongs upstream** if event-triggered
manoeuvres are ever wanted: an `Event.action(sim, bodies)` hook run between the split sub-steps.

**Validation** (`tests/validation/test_stationkeeping.py`) predicts the Delta-v *rate* without the
controller's bookkeeping: `dv = (n/2) da` from Gauss's equation, and the cycle time from the
orbit-averaged decay of an inclined circular orbit in a co-rotating atmosphere (matching an average
of `drag_kernel` itself to 2e-15). Three further terms were each derived, and one of them only after
measurement disagreed:

| term | size | how known |
|---|---|---|
| density convexity `1 + <delta^2>/(2H^2)` | +3.7e-3 table, +2.2e-3 single band | `<rho(h_osc)>/rho(mean)` measured 3.72e-3 against 3.76e-3 |
| RK4 truncation `(n h)^6 / 36` per step | 30 m/day at 60 s, 0.94 at 30 s | 1/36 from a standalone scalar RK4, engine within 5 % |
| Hohmann transfer phase `tau (U - L)/(2H)` | +2.0e-3 table, +1.3e-3 single band | **found**: the layered residual was +2.48e-3 against a controller-free +4.7e-4 |

Measured against the full prediction: **+4.0e-4 (table), +2.3e-4 (single band), +1.04e-3
(single-impulse)** against `RATE_REL_TOL = 3e-3`, each equal to 1e-4 to the same satellite's
controller-free decay residual, so the controller's accounting adds nothing measurable. The band is
held and every burn fires within 10 m of the lower bound on an **independent** mean (a
secular-plus-harmonics least-squares fit), intervals exceed the fastest possible band crossing, and
the drag-free control spends exactly zero.

**Two findings about integrator error.** The RK4 energy constant on a circular Kepler orbit is
**1/36, not the harmonic oscillator's 1/72** that `test_atmosphere.py`'s budget quoted until 9976353 corrected it (harmless
there — it is a 1e-5 budget item). And at `dt = 60 s` a *drag-carrying* satellite's RK4 decay exceeds
the drag-free twin's by 3.2e-3 (table) / 1.7e-3 (single band) of the drag rate, falling as `h^4`:
**a drag-free twin does not calibrate integrator drift under drag.** The test runs at 30 s for that
reason; a Delta-v budget read off a coarse-step Cowell run inherits the bias.

**The headline** (`docs/figures/station_keeping.png`, 10 days, B = 0.05 m^2/kg, band [291, 293.5]
km): the table costs 3.436 m/s/day, 24 raises every 10.1 h, 34.78 m/s in total. The single band
matched at 355 km with H = 60 km — `test_atmosphere.py`'s "one number for LEO" — costs 2.951 m/s/day
(20 raises, 28.98 m/s): **14.1 % under-budgeted**, 0.49 m/s/day, because at 292 km the table's scale
height is 45.5 km. The *same* single band anchored at the station instead differs by **0.14 %**. So
what this choice costs is where the single band is anchored, not how many bands there are.

**Negative controls**, each a one-line mutation of `stationkeeping.py`, restored with `git checkout
--`: trigger on osculating altitude, **16/23 fail** (including the drag-free control burning); burn
retrograde, **14/23 fail** (chattering at exactly the 13 560 s re-observation lockout); drop the lag
correction, **8/23 fail** (burns 0.18 km late, band breached); rectangle window instead of the exact
trapezoid, **2/23 fail** — caught, but thinly: at 30 s the 0.3-sample mismatch leaks only ~10 m.

**Not built.** Eccentricity, inclination and phasing control; propellant mass (the report is Delta-v);
a finite-burn model. Per-config controllers - see the next section.

---

## Delta-v as a sweep metric: the sweep's third currency

`run_sweep(..., station_keeping=StationKeepingSpec(...), delta_v_baseline="<config name>")` fills
`SweepResult.delta_v` with `stationkeeping.DeltaVMetrics`: per-body total Delta-v, raises and steady
m/s/day (first raise excluded, as in the study), their medians over bodies, and the signed relative
budget error against the baseline, negative = under-budgets. It is wired exactly like `access`: one
extra propagation per engine configuration (`sweep.station_keeping_for`, a fresh build through
`apply_config` and then `run_station_keeping` - the study's own function), never touching the error or
timing runs, so `ErrorStats` and `AccessMetrics` are bit-identical with it on or off (tested).

**Why a baseline config and not truth.** Truth (`reference.py`) has no drag, so it has no Delta-v:
every drag tier's "error against truth" would be +100 %, which says nothing. The comparison that is
meaningful is the one the study made - this atmosphere against that one - so the reference is a
*configuration* of the same sweep, named explicitly and never inferred ("the Cowell one", "the last
one"): which model is the yardstick is the premise of the comparison, and a default would choose it
silently. A tier with no drag (Keplerian, secular J2, Cowell without drag) predicts no raise and
scores exactly **-1.0**; that is the finding, not an error, and it is reported rather than refused.

**Why a coarse `dt` is refused rather than adapted.** The controller's mean is a trapezoid over one
period of samples, so it needs `MIN_SAMPLES_PER_ORBIT` = 4 steps per window (below 3 the J2 `2u` term
aliases into the mean). `check_station_keeping_dt` refuses a config below that, by name, before truth
is integrated - the same stance as `access_metrics_for`'s divisibility check. Substituting a finer step
would score a model nobody configured. An analytic tier's position error does not depend on `dt`, so a
caller sweeping one for Delta-v simply gives it a controller-compatible `dt`.

**Measured through the sweep** (`tests/validation/test_sweep_delta_v.py`, 18 h, one cycle): the single
band matched at 355 km reads **-0.1453** against a derived -0.1416 (tolerance 1e-2, the single-cycle
endpoint term), at the band centre -0.0030 against -0.0019, the baseline 3.451 m/s/day against the
study's 3.436; and the same numbers to 1e-12 as `run_station_keeping` called directly. Over a short
horizon the *total* Delta-v is nearly blind (-1e-4: both tiers made two raises) - the steady rate is
the headline.

**Not built: sweeping controllers.** The spec is per sweep, so every configuration faces the same
policy. Comparing *policies* (band width, one impulse against two, a different trigger) is the natural
next step and would put an optional `StationKeepingSpec` on `ModelConfig`. It is deliberately not done
here: a controller is a decision policy, not a model of the physics, and mixing the two in one ranking
would confound "which atmosphere is right" with "which operator is thrifty". `ExternalTier` (SGP4)
always gets `delta_v = None` - a pure function of time cannot take a burn.

`benchmarks/figures.py`'s `figure_station_keeping` still calls `run_station_keeping` directly: it
plots the altitude series, which a `SweepResult` does not carry, and it runs all four satellites in one
arena, where each burn's scheduled second impulse splits the step for every Cowell body - so its
numbers would not be bit-identical to four single-config arenas (the split is inexact for Cowell,
`manoeuvres.py`), and routing it through the sweep would move the published figure.

---

## SGP4: an external tier, not a propagator

`sgp4_bridge.py` puts the operational propagator on the frontier plot without reimplementing it and
without letting TLE mean elements into the engine.

**Why a sweep tier and not a `PropagatorType`.** A propagator is something `step()` advances. SGP4 is
evaluated by `sgp4.api.Satrec` / `SatrecArray`, stateful C++ objects, and `CLAUDE.md` forbids stateful
third-party objects inside a step. It also has nothing the arena could hold as state: its state of
record is the TLE (mean elements plus `B*`), which the engine must never store as elements. And it
does not need stepping - it is closed-form in time. So it is scored the way it is actually used: as a
trajectory evaluated at the times asked for. `sweep.ExternalTier` is plain data (bodies by name, a
central body, a pure `positions(times_s)`), and `run_sweep(..., external=[...])` scores it through
`score_external` against the *same* truth, with the same statistics, the same minimum-of-batches
timing (one state per `dt` to the horizon) and the same access grid. An engine configuration's path is
untouched. `PropagatorType.SGP4` exists in `custom_types.py` and stays unimplemented deliberately.

**Every tier starts from SGP4's Cartesian state.** `scenarios.tle_satellites` seeds each vessel from
`sgp4_bridge.seed_elements`: SGP4's `(r, v)` at the scenario epoch, converted to the engine's
*osculating* elements by `rv_to_coe` - a change of coordinates on one state, reproduced by the build
to 9.1e-13 km. The TLE's elements are never read. Doing that instead (mean elements into
`coe_to_rv`) misses SGP4's own epoch state by **7.64 km** for the ISS, against a first-order
short-period scale `J2 R^2 / a` = 6.48 km; `test_sgp4_bridge.py` keeps it as a named anti-pattern.

**Frame.** SGP4 outputs TEME of date. The engine's Earth-centred scenarios need one fixed inertial
frame with +z on the spin axis, and the bridge identifies the two at the scenario epoch without a
rotation. What that ignores is precession plus nutation of TEME-of-date away from TEME-of-epoch,
bounded (IAU 1980 rates, from memory, unverified) at `TEME_DRIFT_RAD_PER_DAY` = 0.31 arcsec/day, about
11 m/day at LEO radius: two orders below SGP4's disagreement with a J2 truth over a day, and far below
SGP4's own error against real orbits. It stops being acceptable over months (precession alone is
0.33 km per 100 days at LEO), for any comparison with externally referenced data (GCRF ephemerides,
GPS truth - the TEME -> GCRF rotation belongs to `pyerfa`), and for station geometry below ~10 m,
where polar motion also enters. The Earth-rotation phase for stations is `Satrec.gsto`
(`greenwich_angle`).

**Published verification.** Vallado, Crawford, Hujsak & Kelso, *Revisiting Spacetrack Report #3*
(AIAA 2006-6753), publishes 33 verification TLEs and their C++ output. The `sgp4` package ships both
(`SGP4-VER.TLE`, `tcppver.out`), and the test reads them from there. Through the bridge, all 31
cases inside print precision agree to the file's rounding: max 5.00e-9 km, and a **mean |dr| of
2.515e-9 km against the 2.5e-9 km that uniform rounding predicts**. The mean is asserted to within
six standard errors, so a systematic error of 1e-10 km would fail. The last case, SL-12 at 3.5 years
past epoch, differs by 1.17e-7 km. That comes from the package, not the bridge: `Satrec.sgp4_tsince`
gives the same value. Before the bridge put whole days on the Julian date, this case measured
1.59e-7 km. All seven published error outcomes are reproduced, and SGP4 returns a *finite* state
with code 6, so the bridge NaN-fills every flagged row.

**Interchange, and what SGP4 disagrees with.** Cowell + J2 seeded from SGP4's state differs from
SGP4 by the difference between their local expansions. The first term is linear: SGP4 is first order
in J2, so its velocity equals the derivative of its position only to O(J2^2 v). That was predicted at
9e-6 km/s and measured at 1.06e-5 km/s, and it is not drag. The second term is quadratic: J3/J4 plus
O(J2^2) in the acceleration, measured at 4.5e-8 km/s^2. The combined difference is 3.70e-3 km at
300 s. The engine lands on the prediction with a cubic remainder. Over one day on the ISS against
the J2 truth at 60 s:

| Tier | Error after 1 day |
|---|---:|
| Kepler | 646 km |
| Secular J2 (osculating seed) | 778 km |
| Secular J2 (mean seed) | 2.95 km |
| Cowell + J2, `dt = 60 s` | 2.22 km (0.079 at 30 s, 0.0031 at 15 s) |
| **SGP4** | **0.98 km**, 0.10 km after one orbit |

This is one satellite, so the sweep's statistics over bodies reduce to one value. The SGP4 row is a
*comparison*, not a verification. Re-running the truth with SGP4's constants and zero `B*` splits
its 0.966 km along-track error into three parts. +1.20 km comes from WGS-72 `mu` against `MU_EARTH`:
seeding the same `(r, v)` under a different `mu` changes the mean motion by `2 dmu/mu`, which
derives to 1.19 km. +0.95 km is drag; the TLE's `ndot/2` predicts 0.75 km, and the drag term is
quadratic in time, 4.0x from 12 h to 24 h. -1.18 km is J3/J4 and SGP4's theory, which was not
derived. At the ISS's 410 km altitude, the 60 s Cowell tier's *truncation* error (2.22 km) is larger
than SGP4's *model* difference, so at that step the integrator, not the force model, sets the error.

**Limitations.** No TEME -> GCRF rotation, no UT1/polar motion, and no ingest from files or the
network (TLEs are strings). All satellites share one scenario epoch, and a TLE whose epoch is far
from it is evaluated at a large `tsince`, which is legitimate but degrades SGP4. SGP4's mean-element
state is not re-fitted from engine output, which would need a differential corrector. The external
tier has no `ModelConfig` and cannot carry force models: it is whatever the external model is.

---

## Higher zonals: J3..J6 as a second model, not a wider first one

`zonal.py` registers `"zonal"`: the J3..J6 perturbation of the Keplerian parent, per-body coefficients
`(r_eq, j3, j4, j5, j6)`, spin axis = frame +z like `j2`. It exists to turn "does my task need more
than J2?" into a sweep result.

**Why additive to `j2` rather than a J2..Jn replacement.** A general-degree model that included n = 2
would have had to either duplicate J2 - two models that can both be enabled on a body, each adding
J2, which is a double count that raises nothing - or replace `"j2"`, which is the one force model
with a fused compiled Cowell twin (`kernels.cowell_rk4_step`) and whose `force_model_params["j2"]` row
`SECULAR_J2` reads its coefficients from. Degrees 3..6 as a separate bit touch neither: a J2..J6
configuration is `"j2"` plus `"zonal"`, the J2-only configurations are unchanged bit for bit, and the
sweep's model ladder gets one more rung instead of a rewritten one. The truth mirrors the split:
`reference_for(..., oblateness=..., zonal=...)`, with degree 2 refused on the zonal side so it has
exactly one door.

**Why the truth needed a second derivation.** The truth is the only thing that can catch a kernel
error that looks plausible, and it can only do that if it does not share the error. The kernel
evaluates `a_n = (mu/r^2) J_n (R/r)^n [P_{n+1}'(s) r_hat - P_n'(s) z_hat]` from Bonnet's recursion and
the derivative recursion. `reference.zonal_field` never recurses and never projects: each `P_n` is
typed out as its explicit polynomial, split into Cartesian monomials `c z^k r^-m`, and each monomial is
differentiated in x, y, z directly. The two agree to 1e-14..9e-14 of each degree's natural scale over
80 points from LEO to GEO, poles and equator included, and a third evaluation (a finite-differenced
potential built on `numpy.polynomial.legendre`) agrees with the kernel to 2e-11. Mutations show the
split has teeth: a wrong Bonnet coefficient, an odd-degree sign flip, or a wrong Legendre index in the
kernel each fail the cross-check and 4-5 other tests; a wrong degree-5 table entry in the truth fails
the cross-check and the convergence test; a truth field that is not the gradient of its potential
fails the truth's energy check. The field cross-check and the engine-converges-to-truth test each
catch all five; parity catches only the index error, which is what it is for.

**Orbit dynamics, measured differentially.** J2's short-period eccentricity signal (2.0e-3 peak to peak
here) is as large as three days of J3's drift, and J2^2 is as large as J4, so each is measured as the
difference between co-located twins that differ only in the degree (`scenarios.zonal_twins`). J3's
long-period rate, derived by averaging as `de/dt = -(3/2) n J3 (R/p)^3 (1 - e^2) sin i (1 - (5/4)
sin^2 i) cos w` (the `(1 - e^2)` is absent from the commonly quoted form; at e = 0.02 it is 4e-4):
predicted +/-2.4288e-4 over three days, measured +/-2.4390e-4 (+0.42 %, sign following `cos w`), of
which +0.39 % is the osculating-versus-mean semi-major axis of the seed. J4's secular node rate
`(15/16) n J4 (R/p)^4 (1 + 3e^2/2) cos i (4 - 7 sin^2 i)`: predicted -5.898e-4 rad, measured -5.956e-4
(+0.97 %, of which +0.50 % is mean `a`; the rest is J2 x J4 cross terms at the estimated size).

**The headline.** 12 satellites, 550 km / 53 deg, 24 h, three stations, 5 deg mask, 189 passes, truth
with J2..J6 (`python benchmarks/zonal_sweep.py`, ~11 s):

| Tier | median km | max km | mean \|rise\| | max \|rise\| | lost / gained | total contact |
|---|---|---|---|---|---|---|
| Cowell + j2, 60 s | 3.109 | 4.721 | 0.101 s | 0.730 s | 0 / 0 | -12.0 s |
| Cowell + j2 + zonal, 60 s | 1.877 | 1.893 | 0.063 s | 0.223 s | 0 / 0 | -2.2 s |
| Cowell + j2, 15 s | 1.265 | 2.851 | 0.070 s | 0.588 s | 0 / 0 | -9.9 s |
| Cowell + j2 + zonal, 15 s | 0.0027 | 0.0027 | 0.0003 s | 0.0004 s | 0 / 0 | 0.0 s |
| Secular J2, mean-seeded | 5.262 | 8.044 | 1.608 s | 8.026 s | 1 / 0 | +159.6 s |

The estimate written before the run (in `benchmarks/zonal_sweep.py`'s docstring) was ~1-2 km of
along-track from omitting J3..J6 - J4's secular drift of the mean argument of latitude, -1.3 km common
to every satellite, plus ~1 km per satellite from the J3/J4 short-period offset of each seed's mean
`a` - and ~0.2 s of window shift. Measured: 1.26 km median (the 15 s pair, where RK4 truncation is
2.7e-3 km and the difference is model alone), 2.85 km worst, and 0.07 s mean / 0.59 s worst rise
shift. So:

- **In kilometres, J3..J6 matter** - omitting them costs 470x the 15 s error - but they do **not**
  dominate Cowell's 60 s truncation (1.88 km); the two add nearly in phase to 3.11 km. Below about a
  50 s step, the missing zonals are the larger error.
- **In windows, they do not move a 24 h schedule past the 1 s pad**: 0.59 s at worst, no pass gained
  or lost, total contact 9.9 s short over 189 passes. A task that is decided in contact windows at this
  horizon does not need more than J2; one decided in kilometres, or over several days (the J4 term
  alone is secular), does. The Cowell + j2 + zonal 60 s tier reproduces exactly the Cowell + j2 60 s
  figures measured against J2-only truth (1.88 km, 0.063 / 0.223 s), as it should: the zonal model
  removes the model error and leaves the truncation.

**Cost.** The zonal term is fused into the compiled Cowell step (`kernels._cowell_accel`, per-body
`has_zonal` flag, the same two Legendre recursions as `zonal.zonal_kernel` in the same order), so a
Cowell + j2 + zonal configuration stays on the compiled path and its wall time is comparable with the
other fused tiers. Held equivalent to the NumPy path in `tests/validation/test_kernel_equivalence.py`:
the term alone to 4.1e-14 of its scale (bound 1e-12; a relative-1e-9 change to J3 reads 3.6e-9), and
the Cowell state to 1.6e-14 elementwise at 50 steps and 1.4e-13 by norm over 5.2 orbits.

Estimated before measuring, by counting: the zonal block is ~100 flops per evaluation against ~46 for
point mass + J2, but its cost is a serial chain of six dependent divisions (the `/ n` of Bonnet's
recursion, which cannot become a reciprocal multiply without changing the rounding the equivalence
test holds), ~150 cycles of latency - so ~2-3x the fused `j2` kernel's per-body cost, and ~1.3x its
whole-step cost at 12 satellites, where fixed per-step overhead dominates. Measured
(`benchmarks/bench_step.py`, Cowell section): marginal kernel cost 0.206 us per body-step against
0.113 for `j2` (1.8x; the chain overlaps better than the latency count assumed), and whole-step 7.9 us
against 6.8 us at 12 satellites (1.16x), 18.4 against 12.8 at 60 (1.44x). In the sweep above, 24 h:

| Tier | wall, fused | NumPy path (before the twin) |
|---|---|---|
| Cowell + j2, 60 s | 0.010 s | - |
| Cowell + j2 + zonal, 60 s | 0.012 s | 1.25 s |
| Cowell + j2, 15 s | 0.040-0.044 s | - |
| Cowell + j2 + zonal, 15 s | 0.047-0.048 s | 4.9-5.4 s |

J3..J6 now cost **1.10-1.17x** the `j2`-only tier at the same step, down from ~120x (130x as first
recorded) - the remaining difference is physics, not implementation. A zonal body that also carries
a model outside the fused set (`drag`, `srp`, `third_body`, `thrust`) still sends the whole Cowell set
down the NumPy path.

---

## Tesseral harmonics: the first force that reads the clock

`tesseral.py` registers `"tesseral"`: orders m >= 1 of degrees 2..4 - (2,1), (2,2), (3,1)..(3,3),
(4,1)..(4,4) - relative to the Keplerian parent, unnormalised `C_nm`/`S_nm` on the perturbed body's row
after `(r_eq, omega, theta0)`. `EARTH_TESSERALS` (EGM96, from memory; its normalisation checked against
an independently remembered JGM-3 unnormalised table to <= 1.5e-3) and `EARTH_J22`. J2..J6 cannot move a
geostationary satellite in longitude; this is the field that does.

**Additive, for the zonal model's reasons.** A 4x4 field is `"j2"` + `"zonal"` (J3, J4) + `"tesseral"`.
An m = 0 term here would either double-count J2 or replace the fused Cowell kernel's one fast path; the
truth mirrors the split (`reference_for(..., tesseral=...)` refuses m = 0 - it has its own two doors).

**Time.** The field is fixed in the body, so in inertial space it turns at `omega`: the prime meridian
sits at `theta = theta0 + omega * t`, **t = absolute simulation time** - `geometry.elevation_azimuth`'s
convention with its default `epoch_s = 0`, not `viz.ground_track`'s. That makes it the first kernel whose
output depends on the `t` it is handed, and so the first to test a contract that had never been
exercised: that `RK4Integrator` hands its four stages `t, t + h/2, t + h/2, t + h`, that
`Simulation._advance` hands the integrator `self.t`, and that a step cut by a manoeuvre or an event
starts its second half at the cut's time. Reading the code said all three hold; the tests prove it.
Cowell + pm + tesseral on an exaggerated, fast-turning field (EGM96 x 1000 at 10x Earth rate) converges
on tesseral truth at ratios 17.1 and 16.6; the same run with every stage handed the step's start time
converges at 1.995 and sits 4.2e6x further off. A zero-Delta-v manoeuvre cutting 64 steps at 40 % moves
the result 3.2e-6 km (a restart at the step's start time would be ~1e-3). Mutating stage 4 to
`t + h/2` is caught by five tesseral tests - and by **nothing else in the suite**, since no other kernel
reads `t`. The truth reads the clock too: `reference_for` shifts `theta0` by `omega * sim.t`, so a truth
started mid-run is in phase.

**The truth's derivation.** The kernel is Cunningham's V/W recursion in body-fixed Cartesian
coordinates with explicit rotation matrices. `reference.tesseral_field` has neither: it writes
`cos(phi)^m e^{i m lambda} = w^m / r^m` with `w = (x + i y) e^{-i theta}` - the body-fixed longitude
by complex phase - times the typed polynomial `d^m P_n / ds^m`, and differentiates monomial by monomial
in inertial coordinates. They agree to <= 1.5e-12 of each pair's scale over 80 points and 5 angles; a
Legendre-series finite difference agrees to <= 8.3e-10. A rotating non-axisymmetric body does work, so
the truth's `energy_drift` reports the Jacobi-type integral `E - omega L_z` (the field depends on `t`
only through a rigid rotation about z, so `dE/dt = omega dL_z/dt`): held to 1.4e-13 with a massive
rotating primary, while `E` alone drifts 2.3e-2.

**GEO: the decision-relevant result.** Satellites seeded at rest in the rotating frame
(`scenarios.geostationary_satellites`), 6 sidereal days, drift acceleration from a quadratic fit to
whole-day means of longitude minus a pm + J2 control's (RK4's Kepler energy drift, `(3/2) n (nh)^6 /
(36 h)` - predicted 3.5041e-17 rad/s^2, measured 3.5059e-17). Gauss gives `lambda_ddot = -3 a_S / a`,
so for J22 alone `lambda_ddot = +18 n^2 (R/a)^2 J22 sin 2(lambda - lambda22)` with `lambda22 = (1/2)
atan2(S22, C22)` = -14.93 deg the long axis - **plus**, not the minus sign often quoted, which with this
`lambda22` would make the long axis stable. Engine against the first-order pendulum passed through the
same estimator: <= 8.2e-5 of `K = 3.976e-15 rad/s^2 = 1.70e-3 deg/day^2` over 14 longitudes, and the
right sign either side of all four equilibria. Holding a slot costs `a |lambda_ddot| / 3`
(`benchmarks/tesseral_sweep.py`, 24 slots, measured within 1.3e-4 K of the closed forms):

| | stable slots | unstable | worst slot | worst Delta-v |
|---|---|---|---|---|
| J22 alone | 75.07 E, 104.93 W | 14.93 W, 165.07 E | lambda22 +- 45 deg (30 E, 120 E, 60 W, 150 W) | 1.764 m/s/yr (all four equal) |
| 4x4 EGM96 | 74.94 E, 105.09 W | 11.52 W, 161.90 E | 117.4 E | 2.066 m/s/yr (others 1.87, 1.71, 1.48) |

The 4x4 stable points are verified dynamically (zero crossing between satellites at +-1 deg, within
2e-3 deg of the closed-form roots) and *compared* with the published 75.1 E / 105.3 W: -0.16 and +0.21
deg, inside a 0.4 deg allowance for degree >= 5 (Kaula: up to ~0.1 deg), the published rounding and
the field used. J33 does most of the reshaping - its equatorial term is 14 % of J22's - which is why
the four worst slots stop being equal and the worst one costs 17 % more than J22 alone predicts. The
literature's "roughly 1.7-2 m/s per year" is the J22 figure and the 4x4 worst case. Not modelled here,
and the larger part of a real GEO budget: lunisolar inclination drift (north-south station keeping,
typically quoted at ~45-50 m/s/yr, an order above east-west), and SRP's eccentricity drift. The J22 run cannot show a full libration (816 days at 10 deg amplitude) - the tests show the first
quarter of it: acceleration toward the stable points and away from the unstable ones.

**LEO: yes, it moves contact windows.** The zonal sweep's setup (12 satellites, 550 km / 53 deg, 24 h,
three stations, 189 passes) against a truth with J2 + J3..J6 + 4x4 tesseral, both tiers Cowell + j2 +
zonal at 15 s (`benchmarks/tesseral_sweep.py --leo`, 93 s):

| Tier | median km | rms km | max km | mean \|rise\| | max \|rise\| | lost / gained |
|---|---|---|---|---|---|---|
| without `"tesseral"` | 3.819 | 5.707 | 9.046 | 0.288 s | 1.150 s | 0 / 0 |
| with `"tesseral"` | 0.0027 | 0.0027 | 0.0027 | 0.000 s | 0.000 s | 0 / 0 |

Omitting the 4x4 tesserals costs **3x what omitting J3..J6 does** (3.8 km against 1.26 km), and the
worst rise shift passes the 1 s pad; no pass is gained or lost. The estimate written first was ~1 km:
right mechanism, magnitude 3x low. The shared osculating seed sits at a different *mean* semi-major
axis under each model (J22's and J31's short-period terms in `a`, which are the same size at LEO). The
first-orbit mean-`a` offset, up to 0.068 km, predicts each satellite's 24 h along-track error as
`(3/2) n t da` with correlation 0.9992 and 0.44 km rms residual - the bounded daily term. The tesseral
tier reproduces the zonal tier's RK4 truncation (2.7e-3 km) exactly: the model error is gone and only
the step remains.

**Cost.** No compiled twin: `kernels._cowell_accel` has no `t`, so a Cowell body with `"tesseral"` sends
the whole Cowell set down the NumPy path (`_cowell_fused_ok`). The LEO tesseral tier took 21.4 s
against 0.066 s fused (~320x); the 49-satellite GEO scan runs in ~3 s at a 598 s step. It wants a
`kernel-twin` job before it is timed against the analytic tiers.

---

## Validation layers

Four distinct kinds of check, each catching what the others cannot.

| Layer | Question | Catches |
|---|---|---|
| Unit | Do transformations invert? | Algebra errors, singularity handling |
| Invariant | Is energy / momentum / centre of mass conserved? | Sign errors, mass-ratio errors |
| Equivalence | Do the two implementations agree? | Optimisation bugs |
| Reference | Does the trajectory match independent integration? | Everything the above can share |

The reference layer exists because the first three can all be satisfied by an engine that is
self-consistently wrong. `reference.py` shares no code with the engine — no elements, no Kepler solver,
no hierarchy — so an error in `frames.py` cannot appear on both sides of the comparison.

### J2 in the truth, and what keeps it independent

The frontier plot judges models that include J2, so the truth has to include it too. Three rules keep
that from quietly coupling the truth to the engine:

- **Configuration is passed, not read.** `reference_for(..., oblateness={"Earth": (j2, r_eq)})` takes the
  coefficients as an argument. Reading `force_model_mask` / `force_model_params` would copy any bug in
  how the engine stores its configuration into the truth that is supposed to catch it.
- **The formula has a different derivation.** `geopotential.j2_kernel` is Curtis' Cartesian gradient;
  `reference.j2_field` takes the gradient in spherical coordinates and projects it onto `r_hat` and the
  spin axis. A transcription error in one form does not produce the same error in the other. The two
  agree pointwise to 1.2e-15 relative.
- **The physics is complete for what it models.** The oblate body feels the reaction, so momentum is
  conserved, and the field acts on every other integrated body. The engine applies `j2` only where it
  is enabled, relative to the parent. Where those differ, the case is a comparison.

`tests/validation/test_reference_j2.py` holds the truth's own invariants, the secular nodal rate, and
the verification that Cowell + `point_mass_gravity` + `j2` converges to it at fourth order down to a
predicted ~1e-9 km floor. It also has a negative control, a doubled spin-axis term, which is caught at
136 km. For truth runs use `TRUTH_RTOL` / `TRUTH_ATOL` (1e-13, 1e-12). With the default `atol=1e-9`
the absolute tolerance, not `rtol`, limits LEO velocities.

### Verification is not comparison

What a disagreement with `reference.py` *means* depends on the case:

- **Verification** — the model is exact for this case (two-body, massive or massless secondary). The
  two must agree. Disagreement is an engine bug.
- **Comparison** — the model is an approximation (Sun–Earth–Moon omits the solar term in the lunar
  orbit). The two *must* diverge, and the divergence is the measurement.

Reading a comparison divergence as a bug leads to "fixing" a correct engine.

### Negative controls

Equivalence and reference suites each contain a test that deliberately perturbs the engine and asserts
the comparison notices. A tolerance looser than the effect it is meant to catch passes regardless of
correctness — which is how this repository's one genuinely vacuous test survived for months. See the
engineering log.

---

## Build sequence

`Simulation.__init__` → `_build_universe`:

1. Query bodies and systems; pair each body with its system, orphaning what cannot be paired
2. Allocate slots from the free list; populate mass and elements
3. `_resolve_circular` — elect heads for binary systems *(runs, but accuracy is untested)*
4. `_build_sys_head_map` — O(1) slot → system head lookup
5. `_topological_sort(parent_indices)` → `_unfold_database_to_global` — seed global states from elements
6. `_topological_sort(body_sys_map)` — re-sort for barycentric ordering
7. `_recalculate_all_barycenters` — bottom-up mass aggregation and barycentre placement
8. `_zero_roots` — translate so the root sits at the origin
9. `_rehydrate_coes` — recompute local states and elements from final global positions
10. `_refresh_active_indices` — cache sibling/head index arrays and the flattened topological order

Step 10 is the hook a future spawn/despawn path must call. After build, only `set_propagator` calls it.

Per `step()`: split the step at any scheduled manoeuvre epoch (`_advance` per sub-interval, impulse
between them) → then, per sub-interval: Cowell integrate (relative to each parent's start-of-step state) → secular-J2 advance
(RAAN/argument of periapsis/mean anomaly, writing a parent-relative state to scratch) → Keplerian
propagate → `calc_global` → re-base Cowell bodies onto their parents' end-of-step states → re-base
secular-J2 bodies the same way (both through `_rebase`, compiled by default) → advance `t` →
optionally record.

---

## Deliberately not built

- **A compiled force-composition layer.** Cowell's compiled twin is fused for `point_mass_gravity`,
  `j2` and `zonal` only (see the Cowell section); `forces.compose_accelerations` and any other model
  stay NumPy, and a Cowell body carrying one falls back to `RK4Integrator`. A general compiled dispatcher would
  need force models to be registered as compiled callables, which no model yet asks for.
- **Massive Cowell bodies, N-body forces, and perturbers advanced per stage.** Massive Cowell bodies
  are rejected and N-body forces are unregistered; see the Cowell section above for why each needs
  more than a new kernel. `third_body` exists, but its perturber is frozen within a step (see its
  section).
- **Slot compaction and handle indirection.** Dropped from phase 1 once measurement showed they
  addressed ~4% of the step. They become worth doing when there is a despawn path to compact *for*.
- **Spawn / despawn.** Slots come off a free list and are never returned. Loading bodies mid-run — a
  founding goal of the database design — is not yet implemented.
- **`BodyHandle`.** Written, never instantiated; `sim.bodies` is always empty.
- **Accuracy testing of `_resolve_circular`.** The binary-system path was written and exercised as a
  robustness check — *does this run* — not a correctness one. Alpha Centauri is in the seed database
  for that reason. Whether the elected head and repointed parents are numerically *right* is untested.

---

## Known-broken

Tracked in `CLAUDE.md` so it is visible at session start. Summarised here for completeness:
on `main`, `rv_to_coe`'s no-valid-orbit early return shape. On `feature/vop-propagator` only,
`Perturbations.GVP_COE` (four equation errors) and that branch's superseded `cart_to_RSW` /
`RSW_to_cart`; those are the branch's `mypy --strict` failures. Correct RSW transforms are on `main`.
