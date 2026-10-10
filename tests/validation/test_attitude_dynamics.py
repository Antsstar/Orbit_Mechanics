"""
Rigid-body attitude under a pointing controller (`attitude_dynamics.py`).

Expected, derived before measuring:
- **Torque-free, the invariants hold to RK4's truncation.** With no controller, the inertial angular
  momentum `R(q) I omega` and the rotational energy are conserved; what RK4 loses is fourth order in the
  sub-step, so halving it divides the drift by ~16. *Measured* 2.0e-7 -> 1.3e-8 relative (x16.0) at 0.2 ->
  0.1 s sub-steps over 1,000 s, an asymmetric body tumbling at 0.3 rad/s.
- **Axisymmetric free precession has a closed form.** With `I1 = I2 = I` and `I3`, `omega3` is constant
  and the transverse rate turns in the body at `lambda = (I3 - I) / I * omega3`: Euler's equations give
  `omega1_dot = -lambda omega2`, `omega2_dot = lambda omega1`, so after `T` its phase is `+lambda T` (with
  the spin when `I3 > I`; the first version of this docstring had the sign backwards).
- **A small-angle step is a second-order system.** Each axis has `omega_n = sqrt(k_p / 2I)` and
  `zeta = k_d / (2 sqrt(k_p I / 2))`, because the quaternion error's vector part is half the angle. At
  `zeta = 0.5` the overshoot is `exp(-pi zeta / sqrt(1 - zeta^2)) = 16.30 %`. *Measured* 16.297 % (16.29 % before the inertial law held its roll, when its target turned slowly).
- **A rotating target is tracked without steady error.** Nadir pointing turns at the orbital rate, and
  with the target's rate fed forward the loop is type 2: after the transient the error falls to the
  slerp-interpolation level within a step. Without feed-forward it settles at `2 k_d n / k_p` (6.6e-3
  rad for k_p = 1, k_d = 3 at 550 km), the factor 2 because the quaternion error is half the angle.
- **A torque limit makes a slew take time.** From inertial pointing to nadir 90 deg away, with
  `tau_max`, the fastest possible (bang-bang, eigen-axis) slew takes `2 sqrt(theta I / tau_max)`. The
  PD-plus-saturation slew must take at least that, and not wildly more with a well-damped gain.
  *Measured* 325 s to within 1 deg, against 177 s (1.8x).
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.attitude import Attitude
from orbital_engine.attitude_dynamics import (
    AttitudeTracker, PointingController, RigidBody, quat_conj, quat_mul,
)
from orbital_engine.simulator import Simulation


def _sim(session: Session) -> Tuple[Simulation, str]:
    sim = scenarios.earth_constellation(session, n_sats=1, n_planes=1, altitude_km=550.0, inclination_deg=53.0)
    sim.record_history = False
    return sim, "SAT-00-000"


INERTIAL = Attitude("inertial", vector=(0.0, 0.0, 1.0))


def _rotation_x(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]])


def _tumble_drift(session: Session, substep: float) -> Tuple[float, float]:
    sim, name = _sim(session)
    tr = AttitudeTracker(sim, [name], INERTIAL, RigidBody((10.0, 20.0, 25.0)), None,
                         omega0=np.array([[0.05, 0.3, 0.02]]), max_substep_s=substep)
    sim.attach_attitude_tracker(tr)
    h0 = tr.angular_momentum_inertial().copy()
    e0 = 0.5 * float(np.sum(tr.omega ** 2 * tr.inertia))
    for _ in range(100):
        sim.step(10.0)
    e1 = 0.5 * float(np.sum(tr.omega ** 2 * tr.inertia))
    return float(np.linalg.norm(tr.angular_momentum_inertial() - h0) / np.linalg.norm(h0)), abs(e1 / e0 - 1.0)


def test_torque_free_invariants(db_session_factory: Callable[[], Session]) -> None:
    coarse = _tumble_drift(db_session_factory(), 0.2)
    fine = _tumble_drift(db_session_factory(), 0.1)
    assert fine[0] < 1e-6 and fine[1] < 1e-6
    assert 10.0 < coarse[0] / fine[0] < 25.0                        # fourth order


def test_axisymmetric_precession_closed_form(db_session: Session) -> None:
    sim, name = _sim(db_session)
    i, i3, w3, wp = 10.0, 16.0, 0.4, 0.05
    tr = AttitudeTracker(sim, [name], INERTIAL, RigidBody((i, i, i3)), None,
                         omega0=np.array([[wp, 0.0, w3]]), max_substep_s=0.05)
    sim.attach_attitude_tracker(tr)
    for _ in range(30):
        sim.step(10.0)
    lam = (i3 - i) / i * w3
    expected = wp * np.array([math.cos(lam * 300.0), math.sin(lam * 300.0), 0.0]) + np.array([0, 0, w3])
    assert float(np.linalg.norm(tr.omega[0] - expected)) < 1e-9


def test_small_angle_step_overshoot(db_session: Session) -> None:
    sim, name = _sim(db_session)
    inertia, kp, zeta = 10.0, 0.2, 0.5
    kd = 2.0 * zeta * math.sqrt(kp * inertia / 2.0)
    # The inertial law's target is not the identity (its roll is a fixed axis made perpendicular to the
    # boresight). The error is measured against the tracker's own target, as the controller sees it.
    start = AttitudeTracker(sim, [name], INERTIAL, RigidBody((inertia,) * 3), None).q
    tr = AttitudeTracker(sim, [name], INERTIAL, RigidBody((inertia,) * 3), PointingController(kp, kd),
                         q0=quat_mul(start, _rotation_x(0.01)), max_substep_s=0.1)
    sim.attach_attitude_tracker(tr)
    angle = []
    for _ in range(600):
        sim.step(1.0)
        q_e = quat_mul(quat_conj(tr._target), tr.q)
        angle.append(2.0 * math.atan2(q_e[0, 1], q_e[0, 0]))
    overshoot = -min(angle) / 0.01
    assert abs(overshoot - math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))) < 0.01
    assert abs(angle[-1]) < 1e-6


@pytest.mark.parametrize("feed_forward", [True, False])
def test_nadir_tracking(db_session: Session, feed_forward: bool) -> None:
    """With feed-forward the error settles to ~0; without, to `2 k_d n / k_p` (module docstring)."""
    sim, name = _sim(db_session)
    kp, kd = 1.0, 3.0
    nadir = Attitude("nadir", reference="Earth")
    tr = AttitudeTracker(sim, [name], nadir, RigidBody((10.0, 10.0, 10.0)),
                         PointingController(kp, kd, feed_forward=feed_forward), max_substep_s=0.5)
    sim.attach_attitude_tracker(tr)
    for _ in range(120):
        sim.step(10.0)
    n = math.sqrt(scenarios.MU_EARTH / (scenarios.EARTH_RADIUS + 550.0) ** 3)
    err = float(tr.pointing_error_rad()[0])
    if feed_forward:
        assert err < 1e-5
    else:
        assert abs(err / (2.0 * kd * n / kp) - 1.0) < 0.02


def test_a_torque_limited_slew_takes_at_least_the_bang_bang_time(db_session: Session) -> None:
    sim, name = _sim(db_session)
    inertia, tau = 50.0, 0.01
    nadir = Attitude("nadir", reference="Earth")
    tr0 = AttitudeTracker(sim, [name], nadir, RigidBody((inertia,) * 3), None)
    # Start 90 deg away from the nadir target, about the body x axis, at rest.
    tr = AttitudeTracker(sim, [name], nadir, RigidBody((inertia,) * 3),
                         PointingController(kp=0.05, kd=1.5, max_torque_nm=tau),
                         q0=quat_mul(tr0.q, _rotation_x(math.pi / 2)), max_substep_s=0.5)
    sim.attach_attitude_tracker(tr)
    bang_bang = 2.0 * math.sqrt((math.pi / 2) * inertia / tau)       # 177 s
    t, settled = 0.0, None
    for _ in range(400):
        sim.step(5.0)
        t += 5.0
        if settled is None and tr.pointing_error_rad()[0] < math.radians(1.0):
            settled = t
    assert settled is not None and bang_bang <= settled < 4.0 * bang_bang
