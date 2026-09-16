# OrbitalEngine::Theory

The orbital mechanics the engine implements. Cards here should be answerable without reference to
this codebase — they are the domain, not the implementation.

---

### Define the semi-latus rectum and give its relation to angular momentum.

`p` is the conic's half-width at the focus: the radius when true anomaly is 90 degrees.

    p = h^2 / mu
    r = p / (1 + e cos(theta))

It relates to the semi-major axis by `p = a(1 - e^2)` for an ellipse, but unlike `a` it stays finite
and meaningful for every conic including the parabola.

> src: docs/architecture.md; Vallado ch. 1
> tags: theory, coe

---

### State Kepler's equation and explain why it must be solved iteratively.

For an ellipse:

    M = E - e sin(E)

`M` (mean anomaly) advances **linearly in time**; `E` (eccentric anomaly) is the geometric quantity
needed to find position. Going forward is trivial; going backward — `M` to `E` — is transcendental,
mixing `E` linearly and inside a sine, so no closed-form inverse exists.

Hence a root-find, typically Newton-Raphson.

> src: Vallado, Fundamentals of Astrodynamics, Algorithm 2
> sym: solve_kepler_scalar, mean_to_eccentric
> tags: theory, kepler, numerics

---

### Give the hyperbolic analogue of Kepler's equation.

    M = e sinh(H) - H

with `H` the hyperbolic anomaly. Note the sign arrangement flips relative to the elliptic form
`M = E - e sin(E)`.

> src: Vallado ch. 2
> sym: solve_kepler_scalar
> tags: theory, kepler, hyperbolic

---

### Why is a parabolic orbit a special case, and what handles it?

At `e = 1` the semi-major axis is undefined (`a` diverges), so Kepler's equation cannot be formed at
all. **Barker's equation** covers this case, in terms of `p` rather than `a`:

    M_p = tan(theta/2) + (1/3) tan^3(theta/2)

It is solvable in closed form via Cardano's cubic solution — no iteration needed, which makes the
parabolic branch the *easiest* of the three rather than the hardest.

> src: Vallado, Eq. 2-13
> sym: Barker, true_to_mean_parabolic
> tags: theory, parabolic

---

### Why does successive substitution diverge for hyperbolic orbits?

The iteration rearranges to `H <- e sinh(H) - M`, whose derivative is `e cosh(H)`.

Since `e > 1` and `cosh(H) >= 1`, that derivative **exceeds 1 everywhere**, so the map is expansive
and the fixed-point iteration is formally divergent — it cannot converge for any non-trivial input.

The engine preserves the rearrangement rather than silently substituting a different one, and raises
`ConvergenceError` by design. Use Newton-Raphson on the hyperbolic branch.

> src: CLAUDE.md - Known-broken and in-flight; utilities.Anomalies
> sym: _iterate_kepler, ConvergenceError
> tags: theory, numerics, hyperbolic

---

### Why does Newton-Raphson on Kepler's equation need eccentricity-banded seeds?

Because a poor initial guess can step the iterate out of the basin of convergence, especially as
`e` approaches 1 where the function flattens near periapsis.

Vallado's Algorithm 2 bands them: `E = M` below `e = 0.55`, a cube-root guess to `e = 0.95`, and
`E = pi` above that — seeding at `M` there would diverge.

> src: Vallado, Algorithm 2
> sym: _solve_elliptic
> tags: theory, kepler, numerics

---

### State the vis-viva equation and say what it is useful for.

    v^2 = mu (2/r - 1/a)

It links speed to radius and semi-major axis using only conserved quantities, so it recovers `a` from
a single state vector without going near the element conversion code.

That independence is why the test suite uses it: recovering `a` via vis-viva and checking Kepler's
third law validates the propagator *without* trusting `frames.py`.

> src: tests/validation/test_keplerian_propagator.py
> tags: theory, energy, validation

---

### State Kepler's third law in the form the engine uses.

    T = 2 pi sqrt(a^3 / mu)

with `mu = G(m1 + m2)` — the **sum** of both masses, not just the primary. The restricted (massless
secondary) case is the limit where `m2` vanishes.

The engine's two-body mass sum convention (`mu_parent + mu_child`) follows directly from this.

> src: tests/validation/test_keplerian_propagator.py
> tags: theory, kepler

---

### State the barycentre condition and explain why it is exact rather than approximate.

For a two-body system with the barycentre at the origin:

    mu_1 * r_1 + mu_2 * r_2 = 0

This is the **definition** of the centre of mass, not a physical approximation, so it holds to
floating-point noise regardless of how long the simulation runs.

That makes it a sharp test: any error in mass aggregation or in the reflex kick shows up immediately
as a drifting centre of mass, and essentially nothing else does.

> src: tests/validation/test_barycentric_dynamics.py
> tags: theory, barycentric, validation

---

### How large is the Sun's wobble about the Earth-Moon system, and how do you estimate it?

Roughly **450 km** — well inside the Sun itself.

From the barycentre condition, the displacement is the mass fraction times the separation:

    r_Sun = [mu_EMB / (mu_Sun + mu_EMB)] * |r_Sun - r_EMB|

With `mu_EMB / mu_Sun ~ 3e-6` and a separation of 1 AU, that is ~450 km.

> src: tests/validation/test_barycentric_dynamics.py
> tags: theory, barycentric, estimation

---

### Estimate the solar perturbation on the lunar orbit, and say why the estimate matters.

Solar **tidal** acceleration on the Moon relative to Earth:

    a ~ 2 mu_Sun r_EM / d^3
      = 2 (1.327e11)(3.844e5) / (1.496e8)^3
      ~ 3.05e-8 km/s^2

Acting coherently for 30 days that would displace the Moon by `0.5 a t^2 ~ 1.0e5 km`. It does not act
coherently — the perturbation reverses over a synodic month — so the true figure sits well below that
ceiling. Measured against DOP853: **3.3e4 km**, a factor of 3 under.

The estimate matters because it turns a test from a recorded snapshot into an asserted *prediction*.

> src: tests/validation/test_reference_agreement.py
> tags: theory, perturbations, estimation, validation

---

### Why can mean elements never be converted directly to osculating elements?

Because **mean elements are defined by the force model that produced them.** A TLE's mean elements
are whatever SGP4's own averaging says they are; they are not a filtered version of some universal
truth.

Feeding them to `coe_to_rv` produces a position wrong by kilometres, silently — no exception, just a
plausible trajectory in the wrong place.

**Cartesian `(r, v)` is the only safe interchange format** between element representations.

> src: CLAUDE.md - Do not reimplement
> tags: theory, elements, gotcha

---

### What are the coordinate singularities in the classical elements, and when does each bite?

**Circular (`e -> 0`):** periapsis is undefined, so the argument of periapsis has no meaning.
**Equatorial (`i -> 0`):** the ascending node is undefined, so RAAN has no meaning.
**Both:** only the true longitude remains well defined.

These are singularities of the *representation*, not of the physics — the trajectory is perfectly
well behaved. `rv_to_coe` resolves them with analytic fallbacks and returns a success mask rather
than raising.

> src: CLAUDE.md - Existing tools
> sym: rv_to_coe
> tags: theory, coe, singularities

---

### What is the RSW frame and why does the force-model phase need it?

A **body-centred rotating** frame: R along the radius vector, S in-track (perpendicular to R, in the
orbital plane, positive along motion), W along the orbital angular momentum.

It is the natural frame for expressing perturbations and thrust, because direction laws are almost
always stated relative to the orbit rather than to inertial space — "burn prograde" is an S-axis
statement.

Gauss's variational equations take their accelerations in RSW, which is why those transforms are
phase 2 scope rather than leftover cleanup.

> src: CLAUDE.md - Known-broken and in-flight
> tags: theory, frames, phase-2
