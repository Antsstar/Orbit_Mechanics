"""
Body-fixed thrust driven by a tracked attitude (`AttitudeTracker.steer_thrust`).

Scenario: `scenarios.powered_vessel` with two powered vessels, 1 N at Isp 300 s on 300 kg, circular at
550 km, burning for one orbit (5,740 s, Delta a ~ 35 km). `THRUSTER-00` burns along RSW S, the ideal
direction; `THRUSTER-01` carries a thruster on body +z and an `AttitudeTracker` on the `velocity` law.

Expected, derived before measuring:
- **Holding the law reproduces the ideal burn.** Velocity and S differ by the spiral's flight-path angle.
  From a circular start the radial velocity is `v_eq (1 - cos n t)`, `v_eq = 2 a_T / n`, so the angle
  peaks at `2 v_eq / v = 1.6e-3` rad and averages 8e-4. *Measured* 1.6e-3 peak and 8.1e-4 mean.
  Delta a agrees to 3e-9 relative. The position differs by 0.043 km after one orbit. The estimate
  was ~1e-2 km and is low by 4x: the radial forcing oscillates at the orbital frequency, so it is
  resonant. Bound 0.1 km.
- **A slew costs exactly the thrust it points away.** Started 90 deg off under a torque limit, the
  orbit-raising deficit is `sum(c_k / m_k) / sum(1 / m_k)` (Gauss: `da/dt = 2 a_S / n` on a near-circular
  orbit), with `c_k` the S component the simulation wrote at step `k`. *Measured* a 3.19% deficit,
  predicted within 0.36% of itself at every step size; bound 1%.
- **The direction lags by half a step, so the error is first order in `dt`.** The attitude is advanced
  after the orbits, so a step thrusts along its start-of-step attitude. While the slew turns `c` from 0
  to 1, the left sum undercounts `int c dt` by `(dt / 2)(c_end - c_start)`, a ratio-to-ideal deficit of
  `dt / (2 T_burn) = 8.71e-4` at `dt = 10 s`. *Measured* (Richardson over 10 / 5 / 2.5 s) 8.72e-4,
  with successive differences in the ratio 2.0.

Found while writing this: under the `velocity` law, `attitude_matrix` took the body's +x from the
velocity, which *is* the boresight. Every target therefore came from the fallback axis, which switches
when the velocity's x component crosses 0.9, so the roll jumped twice an orbit and the controller
chasing it wobbled the boresight by 0.018 rad. Now the `velocity` law takes its roll from the
direction to the reference, and the `inertial` law holds a fixed roll.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

from orbital_engine import scenarios
from orbital_engine.attitude import Attitude, body_dv_to_rsw
from orbital_engine.attitude_dynamics import (AttitudeTracker, PointingController, RigidBody, matrix_from_quat,
                                              quat_mul)
from orbital_engine.simulator import Simulation
from orbital_engine.thrust import THRUST_MODEL

LAW = Attitude("velocity", reference="Earth")
BODY = RigidBody((50.0,) * 3)
IDEAL, TRACKED = "THRUSTER-00", "THRUSTER-01"
T_BURN = 5740.0


def _session():  # type: ignore[no-untyped-def]
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _rel(sim: Simulation, name: str) -> Tuple[np.ndarray, np.ndarray]:
    k, e = sim.name_to_index[name], sim.name_to_index["Earth"]
    return sim.global_states[k, :3] - sim.global_states[e, :3], sim.global_states[k, 3:] - sim.global_states[e, 3:]


def _sma(sim: Simulation, name: str) -> float:
    r, v = _rel(sim, name)
    return float(1.0 / (2.0 / np.linalg.norm(r) - v @ v / scenarios.MU_EARTH))


def _burn(dt: float, slew: bool) -> Dict[str, float]:
    sim = scenarios.powered_vessel(_session(), n_vessels=2, n_powered=2, thrust_n=1.0)
    ideal_q = AttitudeTracker(sim, [TRACKED], LAW, BODY, None).q
    if slew:
        off = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])  # 90 deg about body x
        tracker = AttitudeTracker(sim, [TRACKED], LAW, BODY, PointingController(0.05, 1.5, 0.002),
                                  q0=quat_mul(ideal_q, off))
    else:                                    # on the law, turning with it at the orbit rate
        r, v = _rel(sim, TRACKED)
        omega = np.cross(r, v) / (r @ r)
        tracker = AttitudeTracker(sim, [TRACKED], LAW, BODY, PointingController(1.0, 3.0),
                                  omega0=(matrix_from_quat(ideal_q)[0].T @ omega)[None])
    sim.attach_attitude_tracker(tracker)
    tracker.steer_thrust(sim)
    a0 = _sma(sim, IDEAL)
    params = sim.force_model_params[THRUST_MODEL]
    k = sim.name_to_index[TRACKED]
    num = den = 0.0
    angles: List[float] = []
    for _ in range(int(round(T_BURN / dt))):
        mass = float(params[k, 2])
        sim.step(dt)
        d = params[k, 4:7]                                        # written at the start of this step
        num += d[1] / mass
        den += 1.0 / mass
        angles.append(math.acos(min(1.0, float(d[1]))))
    gap = sim.global_states[sim.name_to_index[IDEAL], :3] - sim.global_states[k, :3]
    return {"ratio": (_sma(sim, TRACKED) - a0) / (_sma(sim, IDEAL) - a0), "predicted": num / den,
            "gap_km": float(np.linalg.norm(gap)), "peak_rad": max(angles), "mean_rad": float(np.mean(angles))}


@pytest.fixture(scope="module")
def slews() -> Dict[float, Dict[str, float]]:
    return {dt: _burn(dt, slew=True) for dt in (10.0, 5.0, 2.5)}


def test_holding_the_law_reproduces_the_ideal_burn() -> None:
    run = _burn(10.0, slew=False)
    assert run["peak_rad"] < 2e-3 and 5e-4 < run["mean_rad"] < 1.1e-3      # the flight-path angle
    assert abs(run["ratio"] - 1.0) < 1e-6
    assert run["gap_km"] < 0.1


def test_a_slew_costs_the_thrust_it_points_away(slews: Dict[float, Dict[str, float]]) -> None:
    for run in slews.values():
        deficit = 1.0 - run["ratio"]
        assert 0.01 < deficit < 0.1
        assert abs(run["ratio"] - run["predicted"]) < 0.01 * deficit


def test_the_direction_lag_is_first_order(slews: Dict[float, Dict[str, float]]) -> None:
    r10, r5, r25 = (slews[dt]["ratio"] for dt in (10.0, 5.0, 2.5))
    assert 1.8 < (r5 - r10) / (r25 - r5) < 2.2
    lag = 2.0 * (r5 - r10)                                      # Richardson: r(0) - r(10)
    assert abs(lag / (10.0 / (2.0 * T_BURN)) - 1.0) < 0.1


def test_the_throttle_is_kept() -> None:
    sim = scenarios.powered_vessel(_session(), n_vessels=2, n_powered=2, thrust_n=1.0)
    tracker = AttitudeTracker(sim, [IDEAL, TRACKED], LAW, BODY, PointingController(1.0, 3.0))
    sim.attach_attitude_tracker(tracker)
    params = sim.force_model_params[THRUST_MODEL]
    params[sim.name_to_index[IDEAL], 4:7] = 0.0                       # a coast
    params[sim.name_to_index[TRACKED], 4:7] = (0.0, 0.5, 0.0)         # half throttle
    tracker.steer_thrust(sim)
    sim.step(10.0)
    assert np.all(params[sim.name_to_index[IDEAL], 4:7] == 0.0)
    assert abs(np.linalg.norm(params[sim.name_to_index[TRACKED], 4:7]) - 0.5) < 1e-12


def test_body_impulse_matches_the_ideal_law_when_on_it() -> None:
    sim = scenarios.powered_vessel(_session(), n_vessels=1, n_powered=1)
    tracker = AttitudeTracker(sim, [IDEAL], LAW, BODY, None)
    r, v = _rel(sim, IDEAL)
    for dv in ((1e-3, 0.0, 0.0), (0.0, 2e-3, 0.0), (0.0, 0.0, 3e-3)):
        expected = body_dv_to_rsw(LAW, r, v, np.asarray(dv), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        assert np.allclose(tracker.body_dv_rsw(sim, dv)[0], expected, atol=1e-15)
    assert np.allclose(tracker.body_dv_rsw(sim, (0.0, 0.0, 1.0))[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_refusals() -> None:
    sim = scenarios.powered_vessel(_session(), n_vessels=2, n_powered=1)
    bare = AttitudeTracker(sim, ["THRUSTER-01"], LAW, BODY, None)
    sim.attach_attitude_tracker(bare)
    with pytest.raises(ValueError, match="do not carry"):
        bare.steer_thrust(sim)
    loose = AttitudeTracker(sim, [IDEAL], LAW, BODY, None)
    with pytest.raises(ValueError, match="attach the tracker"):
        loose.steer_thrust(sim)
    sim.attach_attitude_tracker(loose)
    with pytest.raises(ValueError, match="non-zero"):
        loose.steer_thrust(sim, (0.0, 0.0, 0.0))
