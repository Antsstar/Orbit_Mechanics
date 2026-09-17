# OrbitalEngine::Forces

The force-model layer, the RSW frame, J2 and Cowell. These are phase 2's decisions, and the two bugs
that looked correct until someone reasoned about a case the tests did not cover.

---

### Why is a body's enabled physics a `uint64` bitmask instead of a list of model objects?

Two reasons. A mask is **data**: a sweep can enumerate, store and vary it without editing code. And
dispatch cost scales with the number of **models**, not the number of bodies:
`resolve_force_models` builds one index set per model, and `compose_accelerations` calls each kernel
**once** on its whole set. Tens of calls per evaluation, not thousands.

> src: src/orbital_engine/forces.py - module docstring
> sym: force_model_mask, resolve_force_models, compose_accelerations
> tags: forces, architecture, sweep

---

### Why must a force kernel write `out[indices] += a`, never `out[indices] = a`?

`out` is the **shared accumulator** `accel_accum`. Several models write to overlapping bodies in the
same evaluation. Assigning would erase every model that ran before, and the result would still look
like a sensible acceleration, just missing physics.

> src: src/orbital_engine/forces.py - ForceKernel docstring
> sym: ForceKernel, accel_accum
> tags: forces, contract, gotcha

---

### Disabling a force model left its acceleration behind. Why, and what is the fix?

`compose_accelerations` zeroes and writes only the rows in the **current dispatch set**. Once a body's
last model is disabled, the body leaves that set, so nothing touches its row again and the last
acceleration stays there indefinitely.

Fix: `Simulation.resolve_force_models` clears the **whole** `accel_accum` on every re-resolve. This is
cheap because it runs on configuration changes, not every step.

> src: docs/engineering-log.md - Disabling a force model left its acceleration behind
> sym: resolve_force_models, test_disabling_a_model_leaves_no_stale_acceleration
> tags: forces, bug, gotcha

---

### `accelerations()` returns `accel_accum` itself. What goes wrong in RK4 if a stage isn't copied?

Every stage "result" is the **same buffer**. The intermediate states are still built from the right
values, but by the final weighted sum a1 through a4 all hold stage 4's acceleration. The integrator
still runs and still produces a plausible orbit, but it is no longer fourth order.

`RK4Integrator` copies each stage out immediately. `_AliasedRK4Integrator` in the tests reproduces the
bug as a negative control, and the convergence-order test catches it.

> src: CLAUDE.md - Invariants that are not obvious from reading the code
> sym: RK4Integrator, _AliasedRK4Integrator
> tags: forces, integrators, aliasing, gotcha

---

### Why do sweep configs store force-model *names* and never raw mask integers?

`register_force_model` assigns bits **in registration order within a process**. Change the import
order, or register a new model earlier, and bit 3 means a different model. A stored integer would
silently enable the wrong physics.

> src: CLAUDE.md - Existing tools (registry.py row)
> sym: register_force_model, mask_for
> tags: forces, registry, sweep

---

### How is the RSW frame built, and why is S not simply v/|v|?

- **R** = r/|r|
- **W** = h/|h|, with h = r × v
- **S** = W × R

S lies along v only when the radial velocity is zero: everywhere on a circular orbit, but only at
periapsis and apoapsis otherwise. Elsewhere v = v_r·R + v_t·S, and S is the in-plane direction
perpendicular to R. `RSW_basis` returns a validity mask. A near-rectilinear state
fails the **relative** test `|r×v| > tol·|r||v|`, which is scale-invariant, unlike `rv_to_coe`'s absolute
`|h|` test.

> src: CLAUDE.md - Existing tools (frames RSW row)
> sym: RSW_basis, RSW_RECTILINEAR_TOL, cart_to_RSW, RSW_to_cart
> tags: frames, rsw, theory

---

### What is the J2 acceleration, and why must R be 6378.137 km and not 6371 km?

a = −(3/2) J2 μ R² / r⁴ · [ (x/r)(1 − 5z²/r²), (y/r)(1 − 5z²/r²), (z/r)(3 − 5z²/r²) ]

J2 is a coefficient **normalised to a reference radius**, and for Earth's published J2 that radius is the
equatorial one (WGS-84, 6378.137 km). Pairing it with the mean radius scales the whole term by
(6371/6378.137)², which is 0.22% too weak, silently.

> src: CLAUDE.md - Existing tools (geopotential.py row)
> sym: j2_kernel, EARTH_R_EQ, EARTH_J2
> tags: j2, theory, gotcha

---

### Why does the `j2` model add only the perturbation, not the central −μr/r³ term?

Models **compose additively**. `point_mass_gravity` supplies the central term, and `j2` adds on top of
it. If each model carried the central term, enabling both would double-count it. Each model is one
physical effect, so a sweep can switch that effect on and off alone.

> src: CLAUDE.md - Existing tools (geopotential.py row)
> sym: j2_kernel, point_mass_gravity_kernel
> tags: j2, forces, architecture

---

### What two physical assumptions does the J2 kernel make that it cannot check itself?

1. **The parent's spin axis is the frame's +z.** That holds for the Earth-centred constellation, but
   is 23.4° off in the ecliptic-framed `sun_earth_moon`.
2. **The parent is a real body.** The kernel has no `is_system`, so on its own it would return a
   finite, meaningless value for a barycentre parent. `"j2"` registers `barycentre_parented` as a
   `validate_bodies` hook, so `enable_force_model` refuses such bodies before setting any bit.

> src: CLAUDE.md - Existing tools (geopotential.py row)
> sym: barycentre_parented, _reject_barycentre_parents
> tags: j2, assumptions, gotcha

---

### Cowell's first version drifted the Moon 1.9e7 km in 30 days. What was wrong, and why did no test see it?

It integrated the Moon's **absolute** state using the **Earth-relative** acceleration. That silently
assumes Earth does not accelerate. Earth accelerates toward the Sun at μ☉/AU² ≈ **5.9e-6 km/s²**, about
twice the Moon's pull toward Earth (2.7e-6). The estimate ½at² over 30 days ≈ 2e7 km matched the
measurement.

Every Cowell test used `two_body`, whose primary **never moves by construction**, so the missing term
was exactly zero in every tested case.

> src: docs/engineering-log.md - Cowell silently assumed the parent never moves
> sym: test_cowell_matches_keplerian_when_the_parent_accelerates
> tags: cowell, bug, frames, validation

---

### How does `step()` place a Cowell body now, and what future force would break it?

1. Snapshot the parent's start-of-step state.
2. Integrate the state **relative to the parent**, with the parent frozen across RK stages. This is
   exact, because every current force depends only on position relative to the parent.
3. After `calc_global()`, re-base the relative result onto the parent's **end-of-step** state.

It breaks for a force that depends on some **other** body's absolute position, such as a solar tidal
term on the Moon. That body would be frozen at its start-of-step position across all four stages.

> src: docs/architecture.md - Cowell propagation: frame, restriction, and what it does not do
> sym: RK4Integrator, set_propagator
> tags: cowell, integrators, frames

---

### Why does `set_propagator` refuse Cowell for a body with `mu != 0`?

A body's mass reaches the rest of the arena **only** through the Keplerian kernel's barycentric
accumulation, the mass-weighted sum that drives the head's reflex kick. Cowell bodies are excluded from
that pass (`_kepler_sib_idx`). A massive Cowell body would silently vanish from its head's reflex kick
and move every sibling. Rejecting it costs nothing, because a satellite is massless anyway.

> src: docs/architecture.md - Cowell propagation: frame, restriction, and what it does not do
> sym: set_propagator, _kepler_sib_idx
> tags: cowell, barycentre, restriction

---

### Why does the Cowell regression test check convergence *ratios* rather than an error threshold?

RK4 error scales as dt⁴, so halving dt should divide the error by **~16**. A missing physical term
does not shrink with dt at all. On the real pre-fix commit the errors went 1.07e5 → 9.10e4 km, ratios
**1.02–1.09**. A magnitude threshold catches that only if someone happened to pick one tight enough.
A ratio catches it by construction.

> src: docs/engineering-log.md - A subagent's "fails on the old code" figure came from a reconstruction
> sym: test_cowell_matches_keplerian_when_the_parent_accelerates, test_cowell_matches_keplerian_at_fourth_order
> tags: cowell, validation, convergence

---

### "The test fails without the fix." What makes that claim actual evidence?

Two conditions, each learned from a failure here:

1. **It fails for the intended reason.** Read the assertion message, not the exit code. A
   `git stash push <file>` once removed a whole uncommitted layer, so the test failed with an
   attribute error.
2. **It fails against the real old code.** Use the actual commit plus the minimum needed on top. A
   reconstruction of old code measures the reconstruction: one reported 5.12e6 km where the real commit
   gave ~1e5 km.

Then check the quoted number against a back-of-envelope estimate.

> src: docs/engineering-log.md - A regression test "failed without the fix" for the wrong reason
> tags: validation, methodology, process
