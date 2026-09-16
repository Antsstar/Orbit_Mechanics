# OrbitalEngine::Arena

The memory arena and the invariants that are not obvious from reading the code. These are the ones
that produce plausible-but-wrong results when violated, rather than an exception.

---

### What does column 0 of `coe_states` hold, and why not the semi-major axis?

The **semi-latus rectum** `p`, related to angular momentum by `p = h^2 / mu`.

Semi-major axis `a` is undefined for a parabolic trajectory (`e = 1` sends `a` to infinity), whereas
`p` stays finite and meaningful across elliptic, parabolic and hyperbolic regimes. Choosing `p` keeps
parabolic orbits representable throughout.

Always index via `COEIndex`, never a bare integer.

> src: CLAUDE.md - Invariants that are not obvious
> sym: COEIndex, coe_states
> tags: arena, coe, representation

---

### Why does a head body carry deliberately zeroed classical elements?

Because a head's motion is a **reflex kick about its barycentre, not an orbit**. Elements would be
meaningless.

The consequence surprises people: when Earth heads the Earth-Moon system its eccentricity reads
exactly 0.0, and the heliocentric ellipse lives on the *barycentre's* row instead.

> src: CLAUDE.md - Invariants that are not obvious
> sym: is_head, _rehydrate_coes
> tags: arena, coe, gotcha

---

### The Earth-Moon barycentre does not inherit Earth's seeded ellipse. By how much, and why?

By about `dp/p ~ 8.5e-4`, and the dominant cause is **velocity, not position**.

Earth circles the barycentre monthly at roughly 0.0127 km/s against a heliocentric 29.8 km/s, so
`dv/v ~ 4.3e-4`. Since `p = h^2/mu` and `h ~ r*v`, the element shift is `dp/p ~ 2 * dv/v ~ 8.5e-4`.
The ~4770 km positional offset contributes a much smaller `3.2e-5`.

Measured: `8.73e-4`. The test asserts the *prediction*, with a lower bound so a collapsed barycentre
cannot pass trivially.

> src: docs/engineering-log.md - Asserting an expectation instead of deriving it
> tags: arena, coe, validation, derivation

---

### What is `local_states[i]` measured relative to?

`global_states[body_sys_map[i]]` — the body's **system bubble**, *not* `parent_indices[i]`.

Conflating the two is a standing trap, and it is what makes `calc_global` correct. A body whose
element parent differs from its kinematic parent (any barycentric sibling) would get a smooth,
plausible, wrong trajectory.

> src: CLAUDE.md - Invariants that are not obvious
> sym: local_states, body_sys_map
> tags: arena, graphs, gotcha

---

### What distinguishes `is_system` from `is_head`?

`is_system` marks a slot as a **barycentre** — a virtual aggregate with no physical body, whose
`mu_array` row holds summed system mass.

`is_head` marks the **primary attractor within a bubble** — a real body that the siblings orbit, and
which wobbles reflexively in response to them.

A body can be a head without being a system; a barycentre is a system and never a head.

> src: CLAUDE.md - Arena layout
> sym: is_system, is_head
> tags: arena, masks

---

### How does a root node identify itself?

**Self-reference:** `parent_indices[i] == i`. The same convention applies in `body_sys_map` for a slot
that heads no bubble.

This is what makes the topological sort's first tier fall out naturally — roots are the fixed points
of the parent map.

> src: CLAUDE.md - Invariants that are not obvious
> sym: parent_indices
> tags: arena, graphs

---

### Slots come off a free list and are never returned. What follows from that?

There is **no despawn path**. Loading bodies mid-run, a founding goal of the database design, is not
yet implemented, and slot compaction has nothing to compact.

It also means index stability is free: a slot index, once handed out, is valid forever, so nothing
needs handle indirection yet.

> src: CLAUDE.md - Invariants; docs/architecture.md - Deliberately not built
> sym: free_indices
> tags: arena, lifecycle, limitations

---

### Why does the arena cache `_sib_idx` and `_head_idx` instead of recomputing masks each step?

Because they change only when the **active set** changes, not every step, and the compiled kernel
reads integer index arrays rather than boolean masks.

Integer indices also make the loop-termination test `active.size` — a Python attribute lookup —
instead of `np.any(active)`, which is a NumPy reduction costing microseconds of dispatch.

Anything that mutates `active_mask` or `is_head` must call `_refresh_active_indices`, or the kernel
propagates a stale body set rather than failing loudly.

> src: CLAUDE.md - Conventions
> sym: _refresh_active_indices, _sib_idx, _head_idx
> tags: arena, performance, gotcha

---

### Why is arena scratch zeroed per *active slot* rather than per *capacity*?

Because zeroing per capacity makes step cost scale with `max_capacity` instead of with the number of
bodies, which turns generous arena sizing into a performance trap.

The original propagator allocated four `(max_capacity, 3)` scratch arrays every step: five bodies
cost 2093 us in a 65536-slot arena against 479 us in a 64-slot one, for identical physics. The
compiled path is flat at ~3.0 us across that entire range.

`tests/validation/test_scaling_invariants.py` asserts this as a ratio, not an absolute threshold.

> src: CLAUDE.md - The two-implementation rule
> sym: _kick, _accum
> tags: arena, performance, scaling

---

### In NumPy, which indexing forms copy and which give a view?

**Basic slicing gives a view.** **Boolean masks and integer-array (fancy) indexing copy.**

Assignment targets are the exception — `arr[mask] = x` writes in place because `__setitem__` is
called rather than producing a new array.

In hot paths, prefer basic slicing, and verify with `np.shares_memory` when it matters.

> src: CLAUDE.md - Conventions
> tags: arena, numpy, performance

---

### Why prefer `np.bincount` or `np.add.reduceat` over `np.add.at`?

`np.add.at` is **unbuffered** — it exists to handle repeated indices correctly, and pays for that by
giving up vectorisation, making it dramatically slower than the buffered alternatives.

`bincount` and `reduceat` achieve the same scatter-add via sorted or counted paths that stay
vectorised.

> src: CLAUDE.md - Conventions
> tags: arena, numpy, performance

---

### What are the engine's units, everywhere, without exception?

**Kilometres, kilometres per second, radians, seconds**, with `mu` in km^3/s^2.

This matters most at dependency boundaries: nearly every atmosphere model and physical-constant
library is SI (metres), so a wrapper that forgets to convert produces a result wrong by exactly 10^3
or 10^9 — large enough to be obvious in a plot, small enough to look like a modelling difference.

> src: CLAUDE.md - Arena layout
> tags: arena, units, gotcha
