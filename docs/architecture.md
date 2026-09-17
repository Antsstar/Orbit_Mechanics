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
| `integrators.py` | Fixed-step RK4, integrating a body's state relative to its parent | **new** |

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

### Open validation

Time-reversibility is a falsifiable claim and is **not currently tested**. `test_orbit_closes_after_
integer_periods` checks a different property — return after a whole period, not backward recovery. A
forward-then-backward test would directly exercise the design rationale above and is cheap to write.

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
re-evaluates `indices` and reads (never advances) `primaries` between stages. No such model is
registered today, so this is a documented constraint on what CAN be added next, not a live limitation.
Both restrictions above (mass and kinematic role) are also documented restrictions of *this* phase, not
accidents.

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

Per `step()`: Cowell integrate (relative to each parent's start-of-step state) → secular-J2 advance
(RAAN/argument of periapsis/mean anomaly, writing a parent-relative state to scratch) → Keplerian
propagate → `calc_global` → re-base Cowell bodies onto their parents' end-of-step states → re-base
secular-J2 bodies the same way → advance `t` → optionally record.

---

## Deliberately not built

- **Compiled twins for Cowell and the force models.** Per-body propagator selection now exists
  (`set_propagator`), but `RK4Integrator`, `point_mass_gravity` and `j2` are NumPy only. The
  compiled/reference switch still applies to the Keplerian path alone.
- **Massive Cowell bodies, and N-body or third-body forces.** Rejected and unregistered respectively;
  see the Cowell section above for why each needs more than a new kernel.
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
