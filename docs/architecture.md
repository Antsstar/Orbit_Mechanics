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
| `reference.py` | Independent DOP853 N-body truth trajectories, optionally with explicitly passed J2 | **new** |
| `benchmark.py` | Timing primitive (minimum-of-batches) | **new** |
| `forces.py` | Force-model composition: enabled physics as a per-body bitmask, additive stateless kernels, and the acceleration contract integrators consume | **new** |
| `gravity.py` | `point_mass_gravity`: the central two-body term relative to the gravitational parent | **new** |
| `geopotential.py` | `j2`: the J2 zonal perturbation only, composable with `point_mass_gravity` | **new** |
| `thirdbody.py` | `third_body`: one named perturber's point-mass pull, direct minus indirect, relative to the parent | **new** |
| `integrators.py` | Fixed-step RK4, integrating a body's state relative to its parent | **new** |
| `drag.py` | `drag`: atmospheric drag in a co-rotating atmosphere, composable with `point_mass_gravity` and `j2`. The density law is a per-body coefficient, not a hard-coded formula | **new** |
| `atmosphere.py` | The density laws `drag` chooses between: one exponential band, or Vallado Table 8-4's 28-band piecewise-exponential profile | **new** |
| `srp.py` | `srp`: cannonball solar radiation pressure from a named light source, with a cylindrical or conical shadow. The shadow geometry is a per-body coefficient, the same way `drag` selects its density law | **new** |
| `viz.py` | Plot-*data* preparation: trajectory sampling over a time grid, ground tracks via `frames`' body-fixed transforms, altitude series, and error curves against a `reference.py` truth. No matplotlib import, so the library stays installable without it — `benchmarks/figures.py` is the consumer that draws | **new** |
| `thrust.py` | `thrust`: continuous rocket thrust along a per-body RSW direction law, with propellant depletion. The first model whose coefficients are state | **new** |
| `manoeuvres.py` | Impulsive Delta-v in RSW, applied now or scheduled at an epoch `step()` splits for. The first physics that is neither a force model nor a propagator, and the first that every propagator can use | **new** |
| `geometry.py` | Observation geometry: ground-station look angles, interpolated access windows, and the spherical line-of-sight test. Pure functions of position arrays, like `viz.py`'s transforms — the primitive layer under a future access-based error metric | **new** |

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
half of this propagator: one scalar loop per body doing all four RK4 stages with `point_mass_gravity`
and `j2` inlined (`kernels._cowell_accel`, each term's arithmetic in the same order as its NumPy
kernel), selected by `use_compiled_kernel` exactly like the Keplerian and secular twins. It cannot
call `forces.compose_accelerations` - numba does not dispatch over a list of Python callables - so
instead of a general compiled composition layer it hard-codes the one combination the fidelity sweep
runs, with per-body flags so a mixed arena (some satellites with J2, some without) still qualifies.
`Simulation._refresh_cowell_plan` decides at configuration time, from the mask alone, whether every
Cowell body's models are a subset of those two and no Cowell body is parented by another Cowell body
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

**What it does not do.** Spherical altitude `|r| - r_ref`, no solar or geomagnetic activity, and no
diurnal bulge. The fused compiled Cowell twin does not include drag, so enabling it on any Cowell body
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

**Still not modelled**, by either law: solar and geomagnetic activity, the diurnal bulge, winds,
seasonal and latitudinal variation. Real density at 400 km swings by more than an order of magnitude
over a solar cycle, which is larger than the gap between the two laws offered here. `pymsis`
(NRLMSISE-00) is the right dependency for that and `CLAUDE.md` forbids reimplementing it. No compiled
twin: drag is not in the fused plan, so a compiled density law would have nothing to plug into.

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

**Deliberately not built.** No range rate (so no Doppler, and no link budget), no refraction or
terrain horizon, no multi-station scheduling or conjunction search, and no sweep-level aggregation:
contact minutes per day, maximum gap and pass-count statistics are the metric layer that consumes
this one, and building them before there is a sweep to attach them to would be guessing at the
reduction. `AccessWindow.peak_elevation_rad` is the one field that is *sampled* rather than refined,
`O(h)` for an overhead pass because the elevation has a corner at the zenith, and it is labelled as a
lower bound rather than quietly parabola-fitted.

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

- **A compiled force-composition layer.** Cowell's compiled twin is fused for `point_mass_gravity` +
  `j2` only (see the Cowell section); `forces.compose_accelerations` and any other model stay NumPy,
  and a Cowell body carrying one falls back to `RK4Integrator`. A general compiled dispatcher would
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
