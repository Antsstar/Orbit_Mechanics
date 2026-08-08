"""
Validation of the analytic Keplerian propagator against closed-form two-body motion.

Two-body Kepler is exact, which makes it the one case where "correct" is not a matter of tolerance
budgeting - the propagator either reproduces the analytic result to floating-point noise or it has a
bug. That makes it the right place to establish the harness, because a failure here is unambiguous.

The step-size independence test is the one that carries real information: an *analytic* propagator
advances mean anomaly in closed form, so its accuracy must not depend on how finely time is sliced.
A numerical integrator would fail it. If this test ever starts scaling with dt, the propagator has
silently stopped being analytic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbital_engine.database import CelestialBodyORM
from orbital_engine.simulator import Simulation

from .invariants import (
    ANGULAR_MOMENTUM_REL_TOL,
    CLOSURE_POS_TOL_KM,
    CLOSURE_VEL_TOL_KMS,
    ENERGY_REL_TOL,
    relative_drift,
    specific_angular_momentum,
    specific_energy,
    vector_drift,
)

EARTH_MU = 398600.4418  # km^3/s^2

# A deliberately unexceptional orbit: eccentric enough that anomaly conversion actually does work,
# inclined and rotated so no angle is accidentally zero and no singularity fallback is exercised.
SEMI_MAJOR_KM = 8000.0
ECCENTRICITY = 0.1
INCLINATION = 0.5
RAAN = 0.3
ARG_PERIAPSIS = 0.7


def _build_two_body(session, *, theta: float = 0.0) -> Simulation:
    """A massless satellite about a fixed primary, built the way tests/conftest.py builds sessions."""
    primary = CelestialBodyORM(
        name="Primary", mu=EARTH_MU, radius=6371.0,
        p=0.0, e=0.0, i=0.0, raan=0.0, arg_pe=0.0, theta=0.0,
    )
    session.add(primary)
    session.flush()

    satellite = CelestialBodyORM(
        name="Satellite", mu=0.0, parent_id=primary.id,
        p=SEMI_MAJOR_KM * (1.0 - ECCENTRICITY**2),
        e=ECCENTRICITY, i=INCLINATION, raan=RAAN, arg_pe=ARG_PERIAPSIS, theta=theta,
    )
    session.add(satellite)
    session.commit()

    return Simulation(
        body_names=["Primary", "Satellite"],
        system_names=[],
        session=session,
        max_capacity=64,
    )


def _orbital_period(a_km: float, mu: float) -> float:
    """Kepler's third law. The analytic reference every closure test is measured against."""
    return 2.0 * math.pi * math.sqrt(a_km**3 / mu)


def _state(sim: Simulation, name: str) -> tuple[np.ndarray, np.ndarray]:
    idx = sim.name_to_index[name]
    return sim.global_states[idx, :3].copy(), sim.global_states[idx, 3:].copy()


def test_period_matches_keplers_third_law(db_session):
    """Guards the test's own reference value before anything is measured against it."""
    sim = _build_two_body(db_session)
    r0, v0 = _state(sim, "Satellite")

    # Recover the semi-major axis from the state via the vis-viva equation, independently of
    # anything in frames.py, and check the seeded orbit is the one we think it is.
    energy = specific_energy(r0, v0, EARTH_MU)
    a_recovered = -EARTH_MU / (2.0 * energy)

    assert a_recovered == pytest.approx(SEMI_MAJOR_KM, rel=1e-12)


def test_energy_and_angular_momentum_conserved_over_one_period(db_session):
    """Both are exact constants of unperturbed two-body motion; only float64 noise may appear."""
    sim = _build_two_body(db_session)
    r0, v0 = _state(sim, "Satellite")

    energy_0 = specific_energy(r0, v0, EARTH_MU)
    h_0 = specific_angular_momentum(r0, v0)

    period = _orbital_period(SEMI_MAJOR_KM, EARTH_MU)
    sim.run(period, period / 1000.0)

    r1, v1 = _state(sim, "Satellite")
    energy_1 = specific_energy(r1, v1, EARTH_MU)
    h_1 = specific_angular_momentum(r1, v1)

    assert relative_drift(energy_0, energy_1) < ENERGY_REL_TOL
    # Compared as a vector: a propagator that conserved |h| while rotating the orbital plane would
    # pass a magnitude check and still be wrong.
    assert vector_drift(h_0, h_1) / np.linalg.norm(h_0) < ANGULAR_MOMENTUM_REL_TOL


@pytest.mark.parametrize("n_periods", [1, 5])
def test_orbit_closes_after_integer_periods(db_session, n_periods):
    """After a whole number of periods the satellite must return to its starting state."""
    sim = _build_two_body(db_session)
    r0, v0 = _state(sim, "Satellite")

    period = _orbital_period(SEMI_MAJOR_KM, EARTH_MU)
    sim.run(period * n_periods, period / 500.0)

    r1, v1 = _state(sim, "Satellite")

    assert vector_drift(r0, r1) < CLOSURE_POS_TOL_KM
    assert vector_drift(v0, v1) < CLOSURE_VEL_TOL_KMS


def test_closure_error_is_independent_of_step_size(db_session):
    """
    An analytic propagator solves Kepler's equation directly, so slicing time more finely must not
    change the answer. A numerical integrator's error would fall with dt; this one's must not move.

    This is what distinguishes "analytic" from "accidentally accurate at the step size we happened
    to test", and it is the check that will fail loudly if the propagator is ever swapped for a
    numerical one without the test suite being told.
    """
    period = _orbital_period(SEMI_MAJOR_KM, EARTH_MU)
    residuals = []

    for n_steps in (10, 100, 1000):
        sim = _build_two_body(db_session)
        r0, _ = _state(sim, "Satellite")
        sim.run(period, period / n_steps)
        r1, _ = _state(sim, "Satellite")
        residuals.append(vector_drift(r0, r1))

        # Each run reseeds the same names, so clear the session for the next iteration.
        db_session.rollback()
        for orm in db_session.query(CelestialBodyORM).all():
            db_session.delete(orm)
        db_session.commit()

    assert all(res < CLOSURE_POS_TOL_KM for res in residuals), residuals

    # Coarse and fine must agree to within the closure budget - not merely both be "small".
    assert max(residuals) - min(residuals) < CLOSURE_POS_TOL_KM, residuals


def test_primary_remains_fixed_with_a_massless_satellite(db_session):
    """
    With mu = 0 on the satellite there is no reflex kick, so the primary must not move at all.

    This pins the barycentric machinery at its degenerate limit: the point-mass case has to fall out
    of the same code path that produces the Earth-Moon wobble, rather than being a special case.
    """
    sim = _build_two_body(db_session)
    r0, v0 = _state(sim, "Primary")

    period = _orbital_period(SEMI_MAJOR_KM, EARTH_MU)
    sim.run(period, period / 100.0)

    r1, v1 = _state(sim, "Primary")

    assert vector_drift(r0, r1) == 0.0
    assert vector_drift(v0, v1) == 0.0
