# OrbitalEngine::Architecture

Design decisions and the reasoning behind them. These are the cards that matter most for explaining
the project to someone else — each answers a "why is it like that?" whose answer is not recoverable
by reading the code.

---

### What is OrbitalEngine's central design inversion, and what follows from it?

Most astrodynamics libraries treat *running a simulation* as the main loop, with comparison as a
script layered on top. Here the **sweep** is the main loop: one scenario, N model configurations,
diffed against a common reference.

Three consequences follow, and getting any of them wrong breaks the thesis. Scenarios must be
**declarative data**, because you cannot sweep over a function call. Enabling a model must be **a
value, not a code path**, or configurations are not enumerable. Results must be structured and
reproducible, because the deliverable is a *differenced quantity*, not a trajectory.

> src: README.md - Direction; docs/architecture.md
> tags: architecture, thesis

---

### Why does every body carry two parent pointers that disagree?

They answer different questions. `parent_indices` records what a body's **orbital elements** are
measured against; `body_sys_map` records what its **Cartesian state** is measured against.

The Moon is the canonical case: its elements are relative to Earth, but its position is relative to
the Earth-Moon barycentre. One pointer would force a choice between correct elements and correct
kinematics.

> src: docs/architecture.md - Why the two graphs
> sym: parent_indices, body_sys_map
> tags: architecture, graphs

---

### Besides representation, what does the two-graph split buy?

**Derivable execution order.** A topological sort over `body_sys_map` yields an order in which every
body's reference frame is resolved before the body itself, which is exactly what `calc_global` walks.
Working that order out without the dual map is the hard part, and is why neither map can simply be
left alone.

> src: docs/architecture.md - Why the two graphs
> sym: calc_global, _topological_sort
> tags: architecture, graphs

---

### Two designs were available for barycentric motion. Which was chosen?

**Option A, virtual head:** treat each system as a pure two-body problem about a *fixed* barycentre,
with a synthetic body always positioned opposite the sibling.

**Option B, real head with reflex motion:** compute the head's reflex displacement first from the
mass-weighted contributions of *all* siblings, then place each sibling relative to the real head.

**Option B was chosen.**

> src: docs/architecture.md - The barycentric model
> tags: architecture, barycentric, design-choice

---

### Under a purely Keplerian model with no N-body term, how does Saturn feel Jupiter?

**Indirectly, through their shared tug on the Sun.** Saturn's orbital *elements* about the Sun are
unaffected by Jupiter, but its **global position** is, because Jupiter displaces the Sun and Saturn
is placed relative to the *real* Sun rather than a fixed point.

The coupling is real and it is analytic. This is the payoff of choosing a real head over a virtual
one.

> src: docs/architecture.md - What that buys
> tags: architecture, barycentric

---

### What is the visible signature of the reflex-kick design?

A barycentre traces a **band** rather than a line: a stable cyclic region whose width depends on the
other bodies in the system, instead of a single clean ellipse.

> src: docs/architecture.md - What that buys
> tags: architecture, barycentric

---

### What property does the reflex-kick design preserve, and why is it the whole point?

The model stays **purely Keplerian and time-invariant**. Every body's anomaly advances analytically
from its own elements with no accumulated numerical state, so the engine can be stepped *backwards*
and reconstruct prior states to solver precision. There is no integrator error to un-integrate.

That is why the reflex kick is applied as a **derived** quantity each step rather than integrated.

> src: docs/architecture.md - The property this preserves
> tags: architecture, barycentric, time-invariance

---

### What is the barycentric model explicitly *not*?

**It is not N-body.** Sibling-to-sibling forces are absent; only the shared displacement of the head
is transmitted.

For two bodies the model is exactly Keplerian and matches numerical integration to 7.9e-5 km over ten
days. For three or more siblings in one bubble it is an approximation whose error against true N-body
is currently **unmeasured** — and quantifying exactly that kind of error is the purpose of the
project, so this is a scheduled measurement rather than a defect.

> src: docs/architecture.md - What it is not
> tags: architecture, barycentric, limitations

---

### Why does `mu_array` hold summed system mass on a barycentre's row rather than zero?

It is not overloading. The entry consistently means *the gravitational parameter of the entity at
this slot*, and a barycentre's entity is the subsystem it represents.

The summed value is load-bearing: it is what the barycentre uses to compute its own Keplerian orbit
about its parent, and what the reflex kick divides by.

> src: docs/architecture.md - mu_array holds summed mass
> sym: mu_array, _recalculate_all_barycenters
> tags: architecture, arena

---

### Why must `reference.py` filter out `is_system` rows before integrating?

Because barycentres are **virtual aggregates**. Integrating them alongside their own members would
double-count every body they represent.

The filter is required by what a barycentre *is*, not by how its mass happens to be stored.

> src: docs/architecture.md - mu_array holds summed mass
> sym: reference_for, is_system
> tags: architecture, validation

---

### Both parent maps are mutated during build. Why, and what is the consequence?

Two places rewrite `parent_indices` after it is read from the database: `_resolve_circular`, because
binary systems declare each other as parent so one must be elected head by mass; and the dynamic
reparenting step in `_unfold_database_to_global`, where siblings repoint at their system head.

The trigger is a body whose own system is not loaded **and** whose parent is not the head of the
parent system, which is what lets a user load an arbitrary subset of the catalogue.

**Consequence:** `sim.parent_indices` is not what the database declared, so the arena cannot be
reconstructed from its own state alone.

> src: docs/architecture.md - Both maps are mutated during build
> sym: _resolve_circular, _unfold_database_to_global
> tags: architecture, build, gotcha

---

### What is the two-implementation rule?

Hot paths exist **twice**: a readable vectorised NumPy version and a compiled scalar version.
Propagation is `propagators.KeplerianPropagator` against `kernels.kepler_propagate`; global states
are `Simulation.calc_global` against `kernels.calc_global_states`.

The rule is **change both, or neither**. The reference is the *definition* and is optimised for being
obviously correct, so it must not be micro-optimised. An equivalence test makes disagreement a build
failure rather than a subtle physics bug.

> src: CLAUDE.md - The two-implementation rule
> sym: kepler_propagate, calc_global_states
> tags: architecture, kernels

---

### Why keep a slow reference implementation at all, rather than just the fast one?

Because removing the dispatch overhead means removing *calls*, which means compiling, which means
scalar loops, and scalar loops are much harder to read than the vectorised form.

Keeping both resolves the conflict instead of trading one away: the reference stays legible enough to
check against a textbook, the kernel stays fast, and the equivalence test keeps them honest. It also
gives debugging a free bisection axis — if the two disagree the bug is in the kernel; if they agree
and are both wrong, it is in the physics.

> src: docs/architecture.md - Why this shape
> tags: architecture, kernels, debugging

---

### Why is Numba's `fastmath` deliberately left off?

It licenses reassociation of floating-point operations and assumes no NaN or infinity. Both are
depended upon here: the Kepler solver **detects divergence by testing for non-finite values**, and
reassociation would break the 1e-12 equivalence bound against the reference.

Speed bought by weakening the arithmetic is not worth having in a numerical engine.

> src: CLAUDE.md - The two-implementation rule
> sym: NUMBA_AVAILABLE
> tags: architecture, kernels, numerics

---

### Why does `use_compiled_kernel` default to `NUMBA_AVAILABLE` rather than `True`?

Because without Numba the kernels still **run**, as interpreted Python, and are then *slower* than
the vectorised NumPy path they replace. "Compiled if available" therefore cannot be spelled `True`.

It also means the kernel is never an untested code path: a dedicated CI job installs without Numba
and exercises the interpreted branch.

> src: CLAUDE.md - The two-implementation rule
> sym: use_compiled_kernel, NUMBA_AVAILABLE
> tags: architecture, kernels, ci

---

### What did `registry.py`, `BodyHandle` and `propagator_type` have in common as of phase 1?

All were **written and never read** — unwired scaffolding. `_PROPAGATOR_REGISTRY` was never queried,
`BodyHandle` was never instantiated so `sim.bodies` is always empty, and `propagator_type` was
allocated but never consulted.

`step()` selected Keplerian propagation unconditionally; the compiled-versus-reference choice is an
*implementation* switch, not a model switch. Wiring the registry was deliberately deferred to the
force-model phase so it arrives as part of the sweep configuration rather than as a second ad-hoc
flag.

> src: CLAUDE.md - Unwired scaffolding
> sym: BodyHandle
> tags: architecture, scaffolding
