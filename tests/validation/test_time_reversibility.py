"""
Time-reversibility of the analytic propagators: step forward N times, then backward N times, and the
arena must return to its initial global state.

`docs/architecture.md` ("The property this preserves") gives this as the reason for the barycentric
design: every analytic body's anomaly advances from its own elements, and the reflex kick is derived
rather than integrated, so there is no accumulated integrator error to undo. This checks that claim.
`test_orbit_closes_after_integer_periods` checks something different, return after whole periods.

**Derived bound, stated before measuring.** Each step solves Kepler's equation by Newton iteration,
stopping once the last update |dE| <= tol (`kernels._SOLVER_TOL` = 1e-5). Newton converges
quadratically: after an update of size d, the remaining error is at most |f''/(2 f')| d^2 <=
e / (2 (1 - e)) * tol^2. Add a rounding allowance of 1000 * eps per step for the chain of
anomaly conversions, and after 2N solves the relative position error is bounded by

    2N * (e_max / (2 (1 - e_max)) * tol^2 + 1000 * eps),

where e_max is the largest eccentricity among the propagating bodies. This is a worst case. Newton
usually lands far below its stopping tolerance, and measured errors are near the rounding floor:
at N = 1000 they were 3.4e-11 (two_body), 3.7e-11 (sun_earth_moon) and 4.4e-13 (constellation), on
both implementations.

**Negative control.** Cowell + RK4 is not time-symmetric: stepping back with -dt does not undo a
forward RK4 step. The same check on Cowell bodies must exceed the analytic bound by a wide margin.
Measured at N = 1000: relative 2.2e-4.
"""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.custom_types import COEIndex, PropagatorType
from orbital_engine.kernels import _SOLVER_TOL
from orbital_engine.simulator import Simulation

N_STEPS = 500
ROUNDING_PER_STEP = 1000.0 * sys.float_info.epsilon
EARTH_J2 = 1.0826266835e-3
EARTH_R_EQ = 6378.137

SCENARIOS = {
    "two_body": (lambda s: scenarios.two_body(s), 30.0),
    "sun_earth_moon": (lambda s: scenarios.sun_earth_moon(s), 3600.0),
    "earth_constellation": (lambda s: scenarios.earth_constellation(s, n_sats=12, n_planes=3), 60.0),
}


def _relative_bound(sim: Simulation) -> float:
    propagating = sim.active_mask & ~sim.is_head
    e_max = float(np.max(sim.coe_states[propagating, COEIndex.E]))
    per_solve = e_max / (2.0 * (1.0 - e_max)) * _SOLVER_TOL ** 2
    return 2 * N_STEPS * (per_solve + ROUNDING_PER_STEP)


def _return_error(sim: Simulation, dt: float) -> float:
    """Max over physical bodies of |r_return - r_initial| / |r_initial|, bodies at the origin excluded."""
    physical = np.flatnonzero(sim.active_mask & ~sim.is_system)
    r0 = sim.global_states[physical, :3].copy()
    for _ in range(N_STEPS):
        sim.step(dt)
    for _ in range(N_STEPS):
        sim.step(-dt)
    assert sim.t == 0.0
    radius = np.linalg.norm(r0, axis=1)
    moved = radius > 0.0
    drift = np.linalg.norm(sim.global_states[physical, :3] - r0, axis=1)
    return float(np.max(drift[moved] / radius[moved]))


def _satellites(sim: Simulation) -> np.ndarray:
    return np.array(sorted(i for name, i in sim.name_to_index.items() if name.startswith("SAT")), dtype=np.int64)


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "reference"])
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_keplerian_arena_returns_to_its_initial_state(
    db_session_factory: Callable[[], Session], name: str, compiled: bool,
) -> None:
    build, dt = SCENARIOS[name]
    sim = build(db_session_factory())
    sim.use_compiled_kernel = compiled
    bound = _relative_bound(sim)
    error = _return_error(sim, dt)
    assert error < bound, f"{name}: relative return error {error:.3e} exceeds derived bound {bound:.3e}"


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "reference"])
def test_secular_j2_bodies_return_to_their_initial_state(
    db_session_factory: Callable[[], Session], compiled: bool,
) -> None:
    build, dt = SCENARIOS["earth_constellation"]
    sim = build(db_session_factory())
    sim.use_compiled_kernel = compiled
    sim.set_propagator(_satellites(sim), PropagatorType.SECULAR_J2, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    bound = _relative_bound(sim)
    error = _return_error(sim, dt)
    assert error < bound, f"relative return error {error:.3e} exceeds derived bound {bound:.3e}"


def test_negative_control_cowell_is_not_time_reversible(db_session_factory: Callable[[], Session]) -> None:
    """The check must be able to fail: RK4 bodies miss the analytic bound by orders of magnitude."""
    build, dt = SCENARIOS["earth_constellation"]
    sim = build(db_session_factory())
    sats = _satellites(sim)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", sats.tolist())
    bound = _relative_bound(sim)
    error = _return_error(sim, dt)
    assert error > 100.0 * bound, f"Cowell return error {error:.3e} is not clearly above the bound {bound:.3e}"
