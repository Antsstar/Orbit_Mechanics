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
| `registry.py` | Catalogue of physics models and propagators, so the simulation can query what is available | force-model half **wired**; propagator half still unwired |
| `propagators.py` | State advancement — **now specifically the readable *reference* implementation** | role narrowed |
| `body.py` | `BodyHandle`, a UI-facing pointer into the arena | unchanged, never instantiated |
| `kernels.py` | Compiled scalar twins of the hot paths | **new** |
| `scenarios.py` | Declarative universe builders, shared by tests and benchmarks | **new** |
| `reference.py` | Independent DOP853 N-body truth trajectories | **new** |
| `benchmark.py` | Timing primitive (minimum-of-batches) | **new** |
| `forces.py` | Force-model composition: enabled physics as a per-body bitmask, additive stateless kernels, and the acceleration contract integrators consume | **new** |

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

**Frame: the simulation-root inertial frame, i.e. `global_states` itself.** Every registered force
kernel (`forces.ForceKernel`) indexes its `state` argument by absolute arena slot, exactly like
`global_states` — a kernel computing a relative vector to a body's parent (`point_mass_gravity`, and
any future third-body or J2 term) reads both rows from the *same* array, so that array has to be one
common frame every active body's position is already expressed in. `global_states` is the only array
with that property; `local_states` is bubble-relative and a different bubble for every body. Choosing
the global frame is also what makes `Simulation.accelerations` — already a `(t, state) -> (C,3)`
function over the whole arena — usable by Cowell with no change: RK4 calls it with candidate rows of
`global_states` and nothing else needs to know a Cowell body is being integrated at all.

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

**How the result reaches `local_states` and `global_states`.** Cowell integration runs *before* the
Keplerian propagation and `calc_global()` in `step()`, using the arena state as committed at the start
of the step, and its result is saved. `calc_global()` then runs as normal — it still walks *every*
active slot in topological order, including Cowell ones, briefly overwriting them with a stale,
pre-step value derived from last step's `local_states` — after which the saved Cowell result is
written back into `global_states` and `local_states[i]` is rebuilt as
`global_states[i] - global_states[body_sys_map[i]]`, using the head's *freshly Keplerian-propagated*
position. This keeps the documented invariant — `local_states[i]` relative to
`global_states[body_sys_map[i]]` — true for a Cowell body exactly as it is for a Keplerian one, so
nothing downstream (`_record_state`, `history`, a future spawn path) needs to know which propagator
produced a given row.

**What a Cowell body's orbital elements mean: nothing.** `coe_states` for a Cowell body is left
untouched by `step()` — stale at whatever it held before the body was reassigned, the same convention
already used for a head's deliberately zeroed COE. Recomputing osculating elements from the fresh
`(r, v)` each step would be legitimate (it is exactly what `_rehydrate_coes` already does at build
time) but is not needed by anything Cowell currently does and was left out rather than half-built; see
the follow-up note in the module docstrings.

**What this does not do.** Force evaluation during one Cowell step treats every *other* body's
position as fixed at its start-of-step value across all four RK4 sub-stages — see `integrators.py`'s
module docstring and the `docs/engineering-log.md` entry on it. Exact for a fixed or non-existent
primary (every validation scenario here), an approximation if a Cowell body's own parent is itself
moving substantially within one macro-step. Also out of scope by design: Cowell bodies feel gravity
from their own parent only, never from every other active body (a two-body term, not N-body), and a
Cowell body's nonzero mass does not feed back into its head's reflex kick — only bodies dispatched
through the Keplerian kernel contribute to that sum. Both are documented restrictions of *this* phase,
not accidents.

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

Step 10 is the hook a future spawn/despawn path must call. Nothing calls it after build today.

Per `step()`: propagate → `calc_global` → advance `t` → optionally record.

---

## Deliberately not built

- **Propagator dispatch.** `registry.py` is written and never read. `step()` selects Keplerian
  unconditionally; the compiled/reference choice is an *implementation* switch, not a model switch.
  Per-body propagator selection belongs with the force-model interface, so it arrives as part of the
  sweep configuration rather than as a second ad-hoc flag.
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
