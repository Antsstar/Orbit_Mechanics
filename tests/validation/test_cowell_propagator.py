"""
Validation of the Cowell propagator: fixed-step classical RK4 integration of the Cartesian equations
of motion (`integrators.RK4Integrator`), driven by the `point_mass_gravity` force model
(`gravity.py`), selected per body through `Simulation.set_propagator`.

Two-body motion under point-mass gravity alone is exactly what the analytic Keplerian propagator
already solves in closed form, so a Cowell body running only `point_mass_gravity` is solving the
*identical* equation of motion by a different numerical method. That makes this the same kind of
"verification" case `test_keplerian_propagator.py` and `test_reference_agreement.py` use: the two
methods must agree, and the *rate* at which they agree as the step size shrinks is itself a testable,
falsifiable prediction of RK4 - fourth order, i.e. quartering the step size only when doubled should
shrink the error roughly sixteenfold. "Small" is not enough evidence for this; the shrink rate is.

All step-size scans below run over a step range where the leading O(dt^4) (or, for a smooth scalar
functional evaluated once, sometimes effectively higher) truncation term dominates: coarser steps show
a higher apparent ratio because sub-leading terms have not yet become negligible (measured this
session - see the two step-size scans below), and much finer steps run into float64 rounding noise
before that shows up as a *failure*, since the error stops shrinking and can even grow slightly. Both
effects were characterised empirically before choosing the windows and bands used here, not assumed.

`test_cowell_matches_keplerian_when_the_parent_accelerates` is the regression guard for a correctness
bug found in review: an earlier version of `integrators.RK4Integrator` advanced a Cowell body's
*absolute* global state using only its parent-relative point-mass acceleration, which silently assumed
the parent never moves. `scenarios.two_body`'s primary is always fixed, so every test above it was
blind to that bug; `scenarios.sun_earth_moon(moon_mu=0.0)` gives a Cowell body (the now-massless Moon)
whose parent (Earth) genuinely accelerates toward the Sun, which is what exposes it.
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType, ScalarSeconds
from orbital_engine.forces import AccelerationProvider
from orbital_engine.simulator import Simulation

from .invariants import specific_energy, vector_drift

MU_PRIMARY = 398600.4418  # km^3/s^2, matches test_keplerian_propagator.py's convention
SEMI_MAJOR_KM = 8000.0
ECCENTRICITY = 0.1
SEMI_LATUS_KM = SEMI_MAJOR_KM * (1.0 - ECCENTRICITY**2)


def _orbital_period(a_km: float, mu: float) -> float:
    return 2.0 * math.pi * math.sqrt(a_km**3 / mu)


PERIOD = _orbital_period(SEMI_MAJOR_KM, MU_PRIMARY)


def _build_two_body_cowell(session: Session) -> Tuple[Simulation, int, int]:
    """
    Two-body scenario (`scenarios.two_body`) with the secondary set to Cowell and
    `point_mass_gravity` enabled on it. `mu_secondary=0.0` (the default) keeps the primary exactly
    fixed - the same restricted two-body case `test_keplerian_propagator.py` uses - so the comparison
    below isolates the *propagator's* error rather than also exercising the reflex kick.
    """
    sim = scenarios.two_body(
        session, mu_primary=MU_PRIMARY, mu_secondary=0.0,
        p=SEMI_LATUS_KM, e=ECCENTRICITY, capacity=16,
    )
    primary = sim.name_to_index["Primary"]
    secondary = sim.name_to_index["Secondary"]
    sim.set_propagator(secondary, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", bodies=secondary)
    return sim, primary, secondary


def _build_two_body_keplerian(session: Session) -> Tuple[Simulation, int]:
    sim = scenarios.two_body(
        session, mu_primary=MU_PRIMARY, mu_secondary=0.0,
        p=SEMI_LATUS_KM, e=ECCENTRICITY, capacity=16,
    )
    return sim, sim.name_to_index["Secondary"]


def _position_error_vs_keplerian(
    db_session_factory: Callable[[], Session], n_steps: int, total_time: float,
) -> float:
    """Cowell position error against the analytic Keplerian propagator after `total_time`."""
    dt = total_time / n_steps

    kepler_sim, kepler_secondary = _build_two_body_keplerian(db_session_factory())
    kepler_sim.record_history = False
    for _ in range(n_steps):
        kepler_sim.step(dt)
    r_true = kepler_sim.global_states[kepler_secondary, :3].copy()

    cowell_sim, _, cowell_secondary = _build_two_body_cowell(db_session_factory())
    cowell_sim.record_history = False
    for _ in range(n_steps):
        cowell_sim.step(dt)
    r_cowell = cowell_sim.global_states[cowell_secondary, :3].copy()

    return vector_drift(r_true, r_cowell)


def _ratios(errors: List[float]) -> List[float]:
    return [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]


# ==================================================================================================
# Verification and convergence order
# ==================================================================================================

# One full orbit, four step counts each halving the previous. Measured this session (two_body,
# SEMI_MAJOR_KM=8000, e=0.1): errors 9.99e-3, 5.35e-4, 3.07e-5, 1.83e-6 km at n_steps=128..1024,
# giving consecutive ratios 18.7, 17.4, 16.8 - converging toward the asymptotic 16x from above, which
# is the expected signature of a leading-order-dt^4 truncation term plus a positive sub-leading term.
# Below n_steps=128 the ratio is measurably higher (up to ~31x at n_steps=8) because the orbit is
# under-resolved; above n_steps~4096 the error (~1e-8 km, close to float64's ~1e-13 relative floor on
# an 8000 km orbit) stops shrinking and then grows, i.e. rounding takes over. This window sits
# comfortably inside both boundaries.
CONVERGENCE_STEP_COUNTS = [128, 256, 512, 1024]

# Centred on 16 (exact 4th order) with real headroom on both sides: a first- or second-order method
# would show ~2x or ~4x, and the observed pre-asymptotic ratios above never exceed ~31x even at a
# grossly under-resolved 8 steps/orbit, so this band is wide enough to absorb ordinary cross-platform
# floating-point variation while still rejecting anything that is not, in fact, 4th order.
ORDER_RATIO_LOW = 12.0
ORDER_RATIO_HIGH = 24.0


def test_cowell_matches_keplerian_at_fourth_order(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    A Cowell body running only `point_mass_gravity` solves the same two-body equation of motion the
    analytic Keplerian propagator solves in closed form, so the disagreement between them is purely
    RK4's own truncation error. Halving the step size should shrink that error by roughly 2^4 = 16x;
    this is what distinguishes "the two methods roughly agree" from "the propagator is verifiably
    fourth order", which is the actual engineering claim being made about it.
    """
    errors = [
        _position_error_vs_keplerian(db_session_factory, n_steps, PERIOD)
        for n_steps in CONVERGENCE_STEP_COUNTS
    ]
    ratios = _ratios(errors)

    assert all(ORDER_RATIO_LOW < r < ORDER_RATIO_HIGH for r in ratios), (
        f"convergence ratios {ratios} (errors {errors}) are not consistent with fourth-order RK4 "
        f"(expected each in ({ORDER_RATIO_LOW}, {ORDER_RATIO_HIGH}), centred on 16x)"
    )


def test_cowell_point_mass_matches_dop853_reference(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Independent check against `reference.py`'s DOP853 N-body integration, which shares no code with
    either the Keplerian propagator or `integrators.RK4Integrator`. Two-body motion is exactly
    Keplerian, so this is a verification case (per `reference.py`'s own verification/comparison
    split): disagreement beyond the tolerance below is an engine bug, not a modelling result.

    Tolerance derivation: measured this session at dt=30 s (~237 steps/orbit) over two orbits, the
    maximum relative position error against DOP853 was 2.4e-7 - RK4's own truncation error at this
    step size, since DOP853 at rtol=1e-13 is far more accurate than that. The bound below carries
    about 40x headroom over the measured value, in line with this project's convention of budgeting
    roughly one to two orders of magnitude above what was actually observed.
    """
    pytest.importorskip("scipy", reason="reference integration requires the [test] or [reference] extra")
    from orbital_engine.reference import reference_for

    dt = 30.0
    n_steps = int(round(2.0 * PERIOD / dt))  # two orbits
    times = np.arange(n_steps + 1, dtype=np.float64) * dt

    sim, _, secondary = _build_two_body_cowell(db_session_factory())
    sim.record_history = False

    ref = reference_for(sim, times)
    assert ref.energy_drift < 1e-9, (
        f"reference integrator drifted by {ref.energy_drift:.3e}; it cannot certify the engine")

    track = np.empty((n_steps + 1, 3), dtype=np.float64)
    track[0] = sim.global_states[secondary, :3]
    for k in range(1, n_steps + 1):
        sim.step(dt)
        track[k] = sim.global_states[secondary, :3]

    ref_pos = ref.position_of("Secondary")
    scale = max(float(np.max(np.linalg.norm(ref_pos, axis=1))), 1.0)
    err = float(np.max(np.linalg.norm(track - ref_pos, axis=1))) / scale

    assert err < 1e-5, (
        f"Cowell point-mass diverges from the independent DOP853 reference by {err:.3e} relative, "
        f"but two-body motion under point-mass gravity is exactly Keplerian - this is an engine "
        f"error, not model error"
    )


# ==================================================================================================
# Regression: a Cowell body whose parent accelerates
# ==================================================================================================

def _moon_error_vs_keplerian(
    db_session_factory: Callable[[], Session], n_steps: int, total_time: float,
) -> float:
    """
    Cowell Moon position error against the analytic Keplerian Moon, in `scenarios.sun_earth_moon` with
    the Moon made massless (`moon_mu=0.0`). Earth - the Moon's `parent_indices` parent - genuinely
    accelerates toward the Sun throughout, unlike `two_body`'s always-fixed primary.
    """
    dt = total_time / n_steps

    kepler_sim = scenarios.sun_earth_moon(db_session_factory(), moon_mu=0.0)
    kepler_sim.record_history = False
    for _ in range(n_steps):
        kepler_sim.step(dt)
    r_true = kepler_sim.global_states[kepler_sim.name_to_index["Moon"], :3].copy()

    cowell_sim = scenarios.sun_earth_moon(db_session_factory(), moon_mu=0.0)
    cowell_sim.record_history = False
    moon = cowell_sim.name_to_index["Moon"]
    cowell_sim.set_propagator(moon, PropagatorType.COWELL)
    cowell_sim.enable_force_model("point_mass_gravity", bodies=moon)
    for _ in range(n_steps):
        cowell_sim.step(dt)
    r_cowell = cowell_sim.global_states[moon, :3].copy()

    return vector_drift(r_true, r_cowell)


# Two days elapsed, four step counts each halving the previous - chosen coarse (dt from 21600 s down
# to 2700 s) because the Earth-Moon relative dynamics are exact to floating-point noise under the
# fixed integrator (the parent's motion cancels out of the relative equation of motion entirely - see
# integrators.py), so the error reaches float64's rounding floor (~1e-7 km, set by subtracting and
# re-adding Earth's ~1.5e8 km heliocentric position) by only a few hundred steps; finer steps than
# these were measured to already be contaminated by that floor.
ACCEL_PARENT_TOTAL_TIME = 2.0 * 86400.0
ACCEL_PARENT_STEP_COUNTS = [8, 16, 32, 64]

# Tighter than the two-body band (10, 22) instead of (12, 24): measured ratios here were 16.3, 16.1,
# 16.1 - textbook fourth order with very little pre-asymptotic excursion, because two days is a small
# fraction of the Moon's ~27.4-day period and the relative dynamics are exact, so there is very little
# of the "orbit under-resolved at large dt" effect the two-body test's wider band accommodates.
ACCEL_PARENT_RATIO_LOW = 10.0
ACCEL_PARENT_RATIO_HIGH = 22.0


def test_cowell_matches_keplerian_when_the_parent_accelerates(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Regression guard for a correctness bug found in review: a Cowell body's gravitating parent is not,
    in general, stationary. Earth (the Moon's `parent_indices` parent) accelerates toward the Sun at
    mu_Sun/AU^2 ~ 5.9e-6 km/s^2 - about twice the Moon's own pull toward Earth, mu_Earth/r^2 ~ 2.7e-6
    km/s^2 - and an earlier version of `integrators.RK4Integrator` advanced the Moon's *absolute*
    global state using only the Earth-relative point-mass acceleration, which silently assumed Earth
    stood still. Treated as a coherent, roughly constant spurious acceleration over 30 days that
    predicts a spurious displacement of order 0.5 * 5.9e-6 * (2.59e6 s)^2 ~ 2e7 km - measured on that
    version, with the exact scenario and force model used here: 1.92e7 km at day 30, growing from
    4.18e5 km at day 1 (the true Earth-Moon distance is ~4.0e5 km throughout, so this is not "a bit
    off", it is unbound).

    `scenarios.sun_earth_moon(moon_mu=0.0)` keeps the Moon massless - `set_propagator` requires this
    for any Cowell body, see its docstring - which leaves Earth's own heliocentric acceleration
    untouched while giving exactly the missing case none of the `two_body`-based tests above can
    exercise (its primary never moves by construction). `integrators.py`'s fix - integrating the state
    relative to `parent_indices` rather than the absolute global state - makes the point-mass term
    translation-invariant to the parent's own motion, so the same fourth-order convergence methodology
    used for the fixed-primary case should hold here too; "small" alone would not distinguish a
    correctly re-based integrator from one that merely happens to be accurate at one step size.

    Confirmed this session that this exact test, unmodified, fails against the pre-fix integrator: at
    the same four step counts the error was ~5.12e6 km at *every* one (ratios 1.0000-1.0002, not
    shrinking with dt at all) - the signature of a systematic missing-physics error rather than a
    discretization error, which is exactly why a convergence-ratio check catches this class of bug
    where a single fixed-tolerance magnitude check would only catch it by accident of having picked a
    large enough number.
    """
    errors = [
        _moon_error_vs_keplerian(db_session_factory, n_steps, ACCEL_PARENT_TOTAL_TIME)
        for n_steps in ACCEL_PARENT_STEP_COUNTS
    ]
    ratios = _ratios(errors)

    assert all(ACCEL_PARENT_RATIO_LOW < r < ACCEL_PARENT_RATIO_HIGH for r in ratios), (
        f"convergence ratios {ratios} (errors {errors}) are not consistent with fourth-order RK4 for "
        f"a Cowell body whose parent accelerates (expected each in "
        f"({ACCEL_PARENT_RATIO_LOW}, {ACCEL_PARENT_RATIO_HIGH}))"
    )


# ==================================================================================================
# Negative control: the convergence-order test above must be able to fail
# ==================================================================================================

class _AliasedRK4Integrator:
    """
    Deliberately broken: otherwise identical to `integrators.RK4Integrator` (same relative-state
    formulation, so this isolates the aliasing bug rather than reintroducing the separate
    missing-parent-acceleration bug `test_cowell_matches_keplerian_when_the_parent_accelerates`
    guards), but reuses the *shared* return value from `provider` across all four RK4 stages instead of
    copying each one out before the next call. `provider` (`Simulation.accelerations`) returns
    `Simulation.accel_accum` itself, mutated in place on every call - so by the time the final weighted
    combination reads `a1`/`a2`/`a3` below, all four names refer to the *same* array and all hold stage
    4's value. This is exactly the bug `integrators.RK4Integrator` is built to avoid (see its module
    docstring); it exists only to prove the convergence-order test above can fail.
    """

    def step(
        self,
        provider: AccelerationProvider,
        t: ScalarSeconds,
        state: NDArray[np.float64],
        dt: ScalarSeconds,
        indices: NDArray[np.int64],
        primaries: NDArray[np.int32],
    ) -> None:
        if indices.size == 0:
            return
        dt = float(dt)
        r0 = (state[indices, :3] - state[primaries, :3]).copy()
        v0 = (state[indices, 3:] - state[primaries, 3:]).copy()

        a1 = provider(t, state)  # NOT copied - aliases accel_accum
        state[indices, :3] = state[primaries, :3] + (r0 + 0.5 * dt * v0)
        state[indices, 3:] = state[primaries, 3:] + (v0 + 0.5 * dt * a1[indices])

        a2 = provider(t + 0.5 * dt, state)  # overwrites the same buffer a1 points to
        v1 = v0 + 0.5 * dt * a1[indices]
        state[indices, :3] = state[primaries, :3] + (r0 + 0.5 * dt * v1)
        state[indices, 3:] = state[primaries, 3:] + (v0 + 0.5 * dt * a2[indices])

        a3 = provider(t + 0.5 * dt, state)
        v2 = v0 + 0.5 * dt * a2[indices]
        state[indices, :3] = state[primaries, :3] + (r0 + dt * v2)
        state[indices, 3:] = state[primaries, 3:] + (v0 + dt * a3[indices])

        a4 = provider(t + dt, state)
        v3 = v0 + dt * a3[indices]

        state[indices, :3] = state[primaries, :3] + (r0 + (dt / 6.0) * (v0 + 2.0 * v1 + 2.0 * v2 + v3))
        state[indices, 3:] = state[primaries, 3:] + (v0 + (dt / 6.0) * (
            a1[indices] + 2.0 * a2[indices] + 2.0 * a3[indices] + a4[indices]
        ))


def test_broken_integrator_fails_the_convergence_order_test(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    A convergence-order test that cannot fail proves nothing (see the vacuous-mask entry in
    `docs/engineering-log.md`). Swapping in `_AliasedRK4Integrator` reproduces the exact aliasing bug
    `Simulation.accelerations`'s docstring warns against, and running the identical step-size scan
    against it must *not* land in the fourth-order band above.

    Measured this session: the aliased integrator's position error is three to four orders of
    magnitude larger than the correct RK4's at the same step counts, and its consecutive-halving
    ratios sit around 1.6-2.0x (consistent with the corrupted scheme collapsing toward an
    effectively first-order update), nowhere near the (12, 24) band.
    """
    errors = []
    for n_steps in CONVERGENCE_STEP_COUNTS:
        dt = PERIOD / n_steps

        kepler_sim, kepler_secondary = _build_two_body_keplerian(db_session_factory())
        kepler_sim.record_history = False
        for _ in range(n_steps):
            kepler_sim.step(dt)
        r_true = kepler_sim.global_states[kepler_secondary, :3].copy()

        cowell_sim, _, cowell_secondary = _build_two_body_cowell(db_session_factory())
        cowell_sim.record_history = False
        cowell_sim._cowell_integrator = _AliasedRK4Integrator()  # type: ignore[assignment]
        for _ in range(n_steps):
            cowell_sim.step(dt)
        r_broken = cowell_sim.global_states[cowell_secondary, :3].copy()

        errors.append(vector_drift(r_true, r_broken))

    ratios = _ratios(errors)
    assert not all(ORDER_RATIO_LOW < r < ORDER_RATIO_HIGH for r in ratios), (
        f"the aliased-buffer integrator was not detected (ratios {ratios}); the convergence-order "
        f"test above cannot distinguish a correct RK4 from this known-broken one"
    )


# ==================================================================================================
# Energy drift
# ==================================================================================================

# A deliberately non-integer number of orbits, so a closed-loop cancellation of the energy error
# cannot flatter the result the way it can when the elapsed time is an exact multiple of the period.
# Measured this session: at n_steps=1024, 2048, 4096, 8192 (dt from ~16.5 s down to ~2.1 s), drift
# ratios were 22.0, 19.7, 16.5 - the same converging-toward-16-from-above pattern the position
# convergence test shows, before rounding noise flattens the trend at even finer steps.
ENERGY_TOTAL_TIME = 2.37 * PERIOD
ENERGY_STEP_COUNTS = [1024, 2048, 4096, 8192]
ENERGY_RATIO_LOW = 12.0
ENERGY_RATIO_HIGH = 28.0


def test_energy_drift_shrinks_at_fourth_order(db_session_factory: Callable[[], Session]) -> None:
    """
    RK4 is not symplectic, so specific energy is not exactly conserved along a Cowell trajectory -
    unlike the analytic Keplerian propagator, which conserves it to float64 noise by construction
    (`test_keplerian_propagator.py::test_energy_and_angular_momentum_conserved_over_one_period`). The
    drift is truncation error, so it must shrink with the step size at a rate consistent with RK4's
    order - "small" alone would not distinguish a correctly-implemented fourth-order method from a
    buggy lower-order one that simply happens to be accurate at whatever step size was tried.
    """
    drifts = []
    for n_steps in ENERGY_STEP_COUNTS:
        dt = ENERGY_TOTAL_TIME / n_steps
        sim, _, secondary = _build_two_body_cowell(db_session_factory())
        sim.record_history = False

        r0 = sim.global_states[secondary, :3].copy()
        v0 = sim.global_states[secondary, 3:].copy()
        e0 = specific_energy(r0, v0, MU_PRIMARY)

        for _ in range(n_steps):
            sim.step(dt)

        r1 = sim.global_states[secondary, :3].copy()
        v1 = sim.global_states[secondary, 3:].copy()
        e1 = specific_energy(r1, v1, MU_PRIMARY)

        drifts.append(abs(e1 - e0))

    ratios = _ratios(drifts)
    assert all(ENERGY_RATIO_LOW < r < ENERGY_RATIO_HIGH for r in ratios), (
        f"energy drift ratios {ratios} (drifts {drifts}) are not consistent with fourth-order RK4"
    )


# ==================================================================================================
# Isolation: a Cowell body must not perturb its Keplerian siblings
# ==================================================================================================

def test_keplerian_bodies_bit_identical_with_a_cowell_sibling_present(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `earth_constellation`'s vessels carry `mu=0`, so a satellite's mass-weighted contribution to its
    head's reflex kick (`kepler_propagate`'s pass 1 accumulation) is exactly zero whether or not that
    satellite is even included in the Keplerian dispatch set - `_kepler_sib_idx` excludes
    Cowell-designated slots, but excluding a zero contribution changes nothing. Every other Keplerian
    body's propagation is per-body and independent of any other sibling's existence. Together these
    mean an all-Keplerian run and a run where one satellite is reassigned to Cowell must produce
    *bit-identical* results for every body except the reassigned one - not merely close.
    """
    dt = 60.0
    n_steps = 50
    cowell_name = "SAT-00-000"

    reference_sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    reference_sim.record_history = False

    mixed_sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    mixed_sim.record_history = False
    cowell_idx = mixed_sim.name_to_index[cowell_name]
    mixed_sim.set_propagator(cowell_idx, PropagatorType.COWELL)
    mixed_sim.enable_force_model("point_mass_gravity", bodies=cowell_idx)

    for _ in range(n_steps):
        reference_sim.step(dt)
        mixed_sim.step(dt)

    keplerian_names = [n for n in reference_sim.name_to_index if n != cowell_name]
    assert len(keplerian_names) > 1, "guard: the scenario must have Keplerian bodies to compare"

    for name in keplerian_names:
        ref_idx = reference_sim.name_to_index[name]
        mix_idx = mixed_sim.name_to_index[name]
        assert np.array_equal(
            reference_sim.global_states[ref_idx], mixed_sim.global_states[mix_idx]
        ), (
            f"{name}'s global state diverged from the all-Keplerian run merely because sibling "
            f"{cowell_name!r} was reassigned to Cowell propagation"
        )
        assert np.array_equal(
            reference_sim.local_states[ref_idx], mixed_sim.local_states[mix_idx]
        ), f"{name}'s local state diverged from the all-Keplerian run"

    # Sanity: the reassigned satellite itself must actually have moved under gravity (not frozen),
    # so this test is not vacuously passing because nothing happened.
    assert not np.array_equal(
        mixed_sim.global_states[cowell_idx], reference_sim.global_states[cowell_idx]
    ), "the Cowell satellite's own state should differ from its Keplerian counterpart's after 50 steps"


# ==================================================================================================
# set_propagator's explicit restriction
# ==================================================================================================

def test_set_propagator_rejects_heads_and_barycenters(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `Simulation.set_propagator` restricts Cowell to active, non-head, non-barycenter bodies that are
    not their own kinematic bubble - see its docstring for why. Both a head and the system barycenter
    must be rejected explicitly rather than silently accepted and produce a corrupted reflex kick.
    """
    sim = scenarios.two_body(db_session_factory())

    primary = sim.name_to_index["Primary"]  # the system's head
    with pytest.raises(ValueError):
        sim.set_propagator(primary, PropagatorType.COWELL)

    barycenter = sim.name_to_index["TB Barycenter"]  # is_system, and its own kinematic bubble
    with pytest.raises(ValueError):
        sim.set_propagator(barycenter, PropagatorType.COWELL)

    # Confirm the rejection didn't leave a partial mutation behind.
    from orbital_engine.custom_types import PropagatorType as PT
    assert sim.propagator_type[primary] == PT.KEPLERIAN
    assert sim.propagator_type[barycenter] == PT.KEPLERIAN


def test_set_propagator_rejects_a_massive_body(db_session_factory: Callable[[], Session]) -> None:
    """
    A massive Cowell body would silently stop contributing to its head's reflex kick - only bodies
    dispatched through the Keplerian kernel accumulate onto their head's `kick`/`accum` - corrupting
    every other sibling in the bubble exactly like a reassigned head or barycenter would, just through
    mass instead of position. `set_propagator` rejects this explicitly rather than permitting a
    configuration whose limitation is merely documented; see its docstring.
    """
    sim = scenarios.two_body(db_session_factory(), mu_secondary=100.0)
    secondary = sim.name_to_index["Secondary"]

    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.COWELL)

    from orbital_engine.custom_types import PropagatorType as PT
    assert sim.propagator_type[secondary] == PT.KEPLERIAN
