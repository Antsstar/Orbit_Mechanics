# OrbitalEngine::Frontier

The J2 truth, the secular-J2 tier and its seeding, the sweep and frontier plot, time-reversibility,
and two process lessons from building them.

---

### Why does `reference_for` take J2 as an `oblateness=` argument instead of reading the engine's force-model configuration?

The truth exists to catch engine bugs. If it read `force_model_mask` / `force_model_params`, a bug in
how the engine *stores* its configuration, such as a wrong row or coefficient order, would be copied
into the truth and cancel out of the comparison. Passing the coefficients explicitly keeps the two
sides independent.

> src: docs/architecture.md - J2 in the truth, and what keeps it independent
> sym: reference_for, test_oblateness_is_not_read_from_the_engine_configuration
> tags: validation, independence, j2

---

### The engine's J2 uses Curtis' Cartesian form. Why is the truth's J2 field derived in spherical coordinates?

Independent derivations make independent mistakes. A transcription error, such as a wrong
coefficient in the z-term, is unlikely to appear identically in both. The truth takes the gradient
in spherical coordinates, radial part plus spin-axis part:
a = (3μJ2R²/r⁴)[((5s² − 1)/2) r̂ − s ẑ], with s = z/r. Expanded, it matches Curtis component by
component, and the two agree pointwise to 1.2e-15.

> src: docs/architecture.md - J2 in the truth, and what keeps it independent
> sym: j2_field, test_field_agrees_with_the_engine_kernel_written_independently
> tags: j2, theory, independence

---

### Why must frontier-plot truth use `atol=1e-12` when the default is `atol=1e-9` with `rtol=1e-13`?

scipy bounds each component's error by `atol + rtol·|y|`. For a LEO position, rtol·|r| ≈ 7e-10 km,
comparable to atol. For a velocity, rtol·|v| ≈ 7.6e-13 km/s, so `atol=1e-9` is 1300 times larger and
is what actually limits accuracy. Tightening it cut the truth error from 3.5e-9 to 8.6e-10 km per orbit.

> src: docs/engineering-log.md - `reference.py`'s default `atol` limits LEO velocities, contrary to its comment
> sym: TRUTH_ATOL, TRUTH_RTOL
> tags: numerics, integrators, gotcha

---

### Why is secular J2 a *propagator* and not a force model?

A force model returns an **acceleration** for an integrator to consume. Secular J2 is **averaged
element drift** (dΩ/dt, dω/dt, dM/dt): rates on orbital elements that an analytic propagator
advances. The averaging has already removed the part that an acceleration would describe, so there
is nothing to integrate.

> src: docs/architecture.md - Secular-J2 propagation: the third tier, and why it is a propagator, not a force model
> sym: SecularJ2Propagator, secular_j2_rates
> tags: architecture, j2, propagators

---

### Seeded from osculating elements, secular J2 removes the cross-track error but not the along-track. Why, and why does the error scale with |cos 2u₀|?

The rates for Ω and ω are right, which is why cross-track error disappears. Mean motion, though, is
computed from the **osculating** semi-major axis. That includes a short-period J2 term,
∝ sin²i·cos 2u, at the starting argument of latitude u₀. The result is a fixed rate bias, which
accumulates along-track every orbit. At u₀ = 45° the term vanishes and the error after 10 orbits was
0.2 km. At u₀ = 0° it was 574 km.

> src: docs/architecture.md - Secular-J2 propagation: the third tier, and why it is a propagator, not a force model
> sym: SecularJ2Propagator
> tags: j2, theory, measurement

---

### What does `mean_seed=True` do, and what error is left afterwards?

It removes the short-period term from the seed:
a_mean = a_osc·(1 − (3/2)J2(R/p)²sin²i·cos 2u₀).
It corrects only `p`, never `e` or `i`. The error left is the **bounded** short-period oscillation an
averaged theory cannot represent: 6.1–6.2 km at |cos 2u₀| = 1 after 10 orbits (from 574 km), 0.16 km at
45°, and 4.0 km median over the 24-hour frontier run.

> src: docs/architecture.md - Secular-J2 propagation: the third tier, and why it is a propagator, not a force model
> sym: mean_seeded_p, test_mean_seeding_collapses_the_phase_dependent_error_against_j2_truth
> tags: j2, theory, seeding

---

### CLAUDE.md forbids mean ↔ osculating conversion. Why is `mean_seeded_p` allowed?

The rule targets **externally defined** mean elements. A TLE's mean elements belong to SGP4's own
force model, so feeding them to another theory is wrong by kilometres. `mean_seeded_p` applies a
first-order correction from **this propagator's own theory** to **its own seed**, so the two are
consistent. The user decided to narrow the rule on those grounds.

> src: CLAUDE.md - Do not reimplement
> sym: mean_seeded_p
> tags: conventions, j2, decision

---

### Why does the sweep report median, RMS and max error over all bodies instead of one satellite?

Model error depends on each body's initial conditions. A single secular-J2 satellite's error ranged
from 0.2 km to 574 km depending only on its starting phase. At 98°, one satellite on secular J2 did
*worse* than Kepler, while the constellation median did better. Report one satellite and the tier
ranking depends on which slot was picked.

> src: docs/architecture.md - Secular-J2 propagation: the third tier, and why it is a propagator, not a force model
> sym: run_sweep, ErrorStats
> tags: sweep, methodology, measurement

---

### On the frontier plot, Cowell's wall time rises 16× while its error falls ~5.7 orders of magnitude. Is that consistent with RK4?

Yes. Each halving of the step doubles the cost and divides RK4's error by ~16. Four halvings give 16×
the time and 16⁴ ≈ 6.6e4× less error. The coarsest point, at 160 s, is pre-asymptotic, which accounts
for the rest.

> src: README.md - Model-fidelity frontier
> sym: run_sweep, cowell_rk4_step
> tags: frontier, integrators, performance

---

### Why do the analytic tiers on the frontier plot take a single step to the 24-hour horizon, and what went wrong before they did?

Kepler and secular J2 are **closed-form in time**: their horizon error is the same whether they get
there in one step or in 1440. The first plot stepped them every 60 s, which charged them for steps
they don't need. That made Cowell at a 160 s step look about as cheap as Kepler. With one step the
errors were identical (608, 431, 4.0 km) and the cost was 21–37 µs. Mean-seeded secular J2 then
reaches 4 km for 37 µs, and Cowell needs about 1,000–4,000× the cost to do better.

The question a plot's timing answers has to be stated. This plot measures the cost of the horizon
state, not the cost of an ephemeris sampled at a fixed cadence.

> src: README.md - Model-fidelity frontier
> sym: run_sweep
> tags: frontier, methodology, measurement

---

### Why are the analytic propagators time-reversible when Cowell isn't, and what is the only test that steps backwards?

Analytic bodies advance their anomaly from their own elements, and the reflex kick is derived rather
than integrated, so there is no integrator history to undo. Stepping forward N times and back N times
returns to rounding level: 3.4e-11 to 3.7e-11 relative, or 4.4e-13 on the constellation. RK4 is not
time-symmetric, and Cowell returned 2.2e-4 relative.

`test_time_reversibility.py` is the only test that uses negative `dt`. A planted `abs(dt)` in the
compiled anomaly advance passed all 319 other tests.

> src: docs/architecture.md - Time-reversibility, tested
> sym: test_keplerian_arena_returns_to_its_initial_state, test_negative_control_cowell_is_not_time_reversible
> tags: validation, architecture, reversibility

---

### A negative control's revised estimate landed within 15% of the measurement. Why wasn't that evidence it was right?

It used 6πf/n² for along-track drift under a constant radial force, where Clohessy-Wiltshire gives
4πf/n² for trajectories starting from the same state. It had been revised *after* the 136 km
measurement was known, and matched only because of the wrong factor. A direct numerical breakdown
showed 76 km from the radial term, 29 km cross-track, and the rest from orbit-varying terms.
Closeness after the fact is not independent confirmation.

> src: docs/engineering-log.md - A negative control's predicted magnitude left out a term
> sym: test_negative_control_doubled_spin_axis_term_fails_verification
> tags: methodology, validation, process

---

### Five "Fable 5.1" agents reported `claude-fable-5-1` as their model. How was it found that they ran on Sonnet 5, and what is the rule now?

The user noticed their Fable credits hadn't moved. Each subagent transcript,
`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`, records the serving model on
every API response. All 721 responses said `claude-sonnet-5`. A Fable *main* session was served by
Fable, so only subagents were rerouted.

The rule: **a model's statement about its own identity is not evidence.** Check the transcript a few
turns after spawning.

> src: docs/engineering-log.md - Every Fable agent actually ran on Sonnet 5
> tags: process, delegation, verification
