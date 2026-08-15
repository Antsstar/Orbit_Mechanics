"""
Engine trajectories against an independent DOP853 N-body integration.

`orbital_engine.reference` shares no code with the propagator: it integrates Newtonian point-mass
acceleration in inertial Cartesian coordinates, with no classical elements, no Kepler solver and no
hierarchy. That independence is the point - an error in `frames.py` or the anomaly stack cannot
appear on both sides of these comparisons, which is exactly the weakness of a round-trip test.

The file is split along a distinction that matters more than any tolerance in it:

* **Verification** - where the engine's model is mathematically exact, the two methods must agree,
  and disagreement is an engine bug. Two-body motion is exactly Keplerian whether or not the
  secondary carries mass, so both variants belong here.

* **Comparison** - where the engine's model is an approximation, the two *will* diverge, and the
  size of the divergence is the result being measured. Hierarchical two-body motion plus a reflex
  kick neglects the Sun's perturbation of the lunar orbit entirely, so a Sun-Earth-Moon system must
  diverge, and a test asserting agreement there would be asserting that the engine is wrong.

Reading a comparison failure as a bug leads to "fixing" a correct engine; reading a verification
failure as a result hides a real one.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.reference import reference_for
from orbital_engine.simulator import Simulation

pytest.importorskip("scipy", reason="reference integration requires the [test] or [reference] extra")


# Where Keplerian motion is exact, the residual is the *integrator's* accumulated error, not the
# engine's - the analytic propagator is the more accurate of the two parties. DOP853 at rtol=1e-13
# accumulates roughly 1e-9 relative over tens of orbits, and the engine's Kepler solve converges to
# machine precision, so 1e-7 is that expectation with two orders of headroom.
EXACT_AGREEMENT_REL_TOL = 1e-7

# The integrator must conserve energy far better than the agreement it is being used to certify,
# otherwise it cannot certify it.
REFERENCE_ENERGY_DRIFT_TOL = 1e-11


def _engine_track(sim: Simulation, names: list[str], dt: float, n_steps: int) -> dict[str, np.ndarray]:
    """Step the engine, sampling global position at the same instants as the reference."""
    sim.record_history = False
    track = {n: np.empty((n_steps + 1, 3), dtype=np.float64) for n in names}
    for n in names:
        track[n][0] = sim.global_states[sim.name_to_index[n], :3]

    for k in range(1, n_steps + 1):
        sim.step(dt)
        for n in names:
            track[n][k] = sim.global_states[sim.name_to_index[n], :3]
    return track


def _max_relative_error(engine: np.ndarray, ref: np.ndarray) -> float:
    """Scaled by the largest reference radius, so a fixed error is judged against the orbit size."""
    scale = max(float(np.max(np.linalg.norm(ref, axis=1))), 1.0)
    return float(np.max(np.linalg.norm(engine - ref, axis=1)) / scale)


# ==================================================================================================
# Verification: cases where the engine's model is exact
# ==================================================================================================

@pytest.mark.parametrize(
    "label,mu_secondary",
    [("massless secondary", 0.0), ("massive secondary", 3.986004418e5 * 0.3)],
)
def test_two_body_matches_numerical_integration(
    label: str, mu_secondary: float, db_session_factory: Callable[[], Session],
) -> None:
    """
    Two-body motion is exactly Keplerian, so analytic propagation and numerical integration must
    agree to the integrator's own accuracy.

    The massive-secondary variant is the more interesting of the two: it is still exactly Keplerian,
    but only if the reflex kick is right. Both bodies now orbit their common barycenter, and an
    incorrect mass ratio in the kick would displace them in opposite directions while leaving the
    centre of mass fixed - invisible to the barycentric invariants in `test_barycentric_dynamics`,
    and immediately visible here.
    """
    dt, n_steps = 120.0, 1440              # 2 days, ~16 orbits of an 11000 km semi-latus rectum
    times = np.arange(n_steps + 1, dtype=np.float64) * dt

    sim = scenarios.two_body(db_session_factory(), mu_secondary=mu_secondary, p=11000.0, e=0.2)
    ref = reference_for(sim, times)

    assert ref.energy_drift < REFERENCE_ENERGY_DRIFT_TOL, (
        f"reference integrator drifted by {ref.energy_drift:.3e}; it cannot certify the engine")

    track = _engine_track(sim, ref.names, dt, n_steps)

    for name in ref.names:
        err = _max_relative_error(track[name], ref.position_of(name))
        assert err < EXACT_AGREEMENT_REL_TOL, (
            f"{label}: {name} diverges from the reference by {err:.3e} relative, but two-body "
            f"motion is exactly Keplerian - this is an engine error, not model error")


def test_reference_would_detect_a_wrong_trajectory(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Verifies the comparison above can fail.

    A reference comparison whose tolerance exceeds the error it should catch passes regardless.
    Propagating the engine with a 1% wrong `mu` produces a trajectory that still conserves energy,
    still closes, and is still a perfectly good ellipse - just the wrong one. It must be caught.
    """
    dt, n_steps = 120.0, 720
    times = np.arange(n_steps + 1, dtype=np.float64) * dt

    sim = scenarios.two_body(db_session_factory(), p=11000.0, e=0.2)
    ref = reference_for(sim, times)

    sim.mu_array[sim.name_to_index["Primary"]] *= 1.01
    track = _engine_track(sim, ref.names, dt, n_steps)

    err = _max_relative_error(track["Secondary"], ref.position_of("Secondary"))
    assert err > EXACT_AGREEMENT_REL_TOL, (
        f"a 1% mu error produced only {err:.3e} relative divergence; the tolerance is too loose "
        f"to certify anything")


# ==================================================================================================
# Comparison: cases where the engine's model is an approximation
# ==================================================================================================

def test_hierarchical_keplerian_model_error_is_the_expected_size(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Quantifies the modelling error of hierarchical two-body motion on a Sun-Earth-Moon system.

    The engine propagates the Moon about Earth under mu = mu_Earth + mu_Moon and the Earth-Moon
    barycenter about the Sun, with no solar term in the lunar orbit at all. Real three-body motion
    has one, so the Moon *must* drift. This test asserts that drift lands in a physically derived
    band - it is a measurement with bounds, not a regression snapshot.

    **Upper bound.** Solar tidal acceleration on the Moon relative to Earth is

        a ~ 2 mu_Sun r_EM / d^3
          = 2 (1.327e11)(3.844e5) / (1.496e8)^3
          ~ 3.05e-8 km/s^2

    If that acted coherently for the whole 30 days it would displace the Moon by
    0.5 a t^2 ~ 1.0e5 km. It does not - the perturbation reverses over a synodic month - so the true
    figure must be comfortably below that ceiling, and the ceiling is the bound asserted.

    **Lower bound.** The drift must exceed 1000 km. Below that the two methods would effectively
    agree, which would mean either the reference is not independent or the solar perturbation is not
    reaching the Moon - both of which make every other comparison in this file meaningless.

    Measured at time of writing: Moon 3.30e4 km, Earth 4.07e2 km, Sun 3.9e-6 km.
    """
    dt, n_steps = 3600.0, 720              # 30 days
    times = np.arange(n_steps + 1, dtype=np.float64) * dt

    sim = scenarios.sun_earth_moon(db_session_factory())
    ref = reference_for(sim, times)

    assert ref.energy_drift < REFERENCE_ENERGY_DRIFT_TOL

    track = _engine_track(sim, ref.names, dt, n_steps)

    def drift_km(name: str) -> float:
        return float(np.max(np.linalg.norm(track[name] - ref.position_of(name), axis=1)))

    moon, earth, sun = drift_km("Moon"), drift_km("Earth"), drift_km("Sun")

    assert 1.0e3 < moon < 1.02e5, (
        f"lunar model error {moon:.3e} km is outside the derived band [1e3, 1.02e5] km")

    # The hierarchy of error must match the hierarchy of neglected physics: the Moon's orbit is the
    # one missing a term, the Earth-Moon barycenter's heliocentric orbit is very nearly right, and
    # the Sun is barely displaced at all.
    assert earth < moon, "Earth drifted more than the Moon; the neglected term is in the lunar orbit"
    assert sun < earth, "the Sun drifted more than the Earth"


def test_model_error_grows_with_integration_time(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Model error must accumulate. A neglected force produces a secular drift, so the discrepancy at
    30 days has to exceed the discrepancy at 5.

    This distinguishes genuine model error from a fixed offset. A constant discrepancy - equal at
    both durations - would indicate a frame or epoch mismatch in the comparison rather than missing
    physics, and would invalidate the test above rather than confirming it.
    """
    dt = 3600.0
    sim = scenarios.sun_earth_moon(db_session_factory())

    long_steps = 720
    times = np.arange(long_steps + 1, dtype=np.float64) * dt
    ref = reference_for(sim, times)
    track = _engine_track(sim, ref.names, dt, long_steps)

    err = np.linalg.norm(track["Moon"] - ref.position_of("Moon"), axis=1)

    early = float(np.max(err[: 5 * 24]))       # first 5 days
    late = float(np.max(err))                  # full 30 days

    assert late > 3.0 * early, (
        f"lunar model error barely grew ({early:.3e} km at 5 days, {late:.3e} km at 30); "
        f"a near-constant offset suggests a frame mismatch rather than neglected physics")
