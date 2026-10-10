"""
Vessel-relative frames and Clohessy-Wiltshire targeting (`relative.py`).

Expected, derived before measuring:
- **The frame round-trips exactly** (an orthogonal rotation and its transpose, the same omega x rho).
- **The CW matrix is the linearisation of two-body relative motion**, so against the exact propagation
  (`iod.kepler_universal` of both vessels) its error is second order in the separation: x10 the
  separation, x100 the error. *Measured* 3.6e-4 km at 1 km and 3.6e-2 km at 10 km over a quarter orbit
  at 7,000 km (~2.6 rho^2 / r). The matrix was written from memory; this is the check of it.
- **A CW rendezvous flown in the engine** (`scenarios.artemis3_rendezvous`, Keplerian, chaser below and
  behind the target at 6,800 km, 0.4-orbit transfer, impulses applied with `apply_delta_v` in the
  chaser's RSW axes) misses by the linearisation error, again ~ rho^2: *measured* 1.0e-2 / 0.26 / 6.4 km
  at 4.1 / 20.6 / 103 km initial separation (x25 per x5). A Lambert plan for the same transfer is exact
  two-body: 1e-10 km at every separation (a verification of the frame conversions and the impulses).
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import iod, relative, scenarios

MU = scenarios.MU_EARTH


def _circular(r0: float) -> Tuple[np.ndarray, np.ndarray, float]:
    return np.array([r0, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / r0), 0.0]), math.sqrt(MU / r0 ** 3)


def test_the_frame_round_trips_and_a_coorbital_chaser_is_at_rest() -> None:
    rt, vt, n = _circular(7000.0)
    rho, rho_dot = np.array([0.3, -1.0, 0.2]), np.array([1e-4, -5e-5, 2e-5])
    rc, vc = relative.lvlh_to_inertial(rt, vt, rho, rho_dot)
    back = relative.lvlh_state(rt, vt, rc, vc)
    assert np.allclose(back[0], rho, atol=1e-12) and np.allclose(back[1], rho_dot, atol=1e-15)
    # Same circular orbit, 1 deg behind: at rest in the rotating frame.
    phase = math.radians(-1.0)
    rc = 7000.0 * np.array([math.cos(phase), math.sin(phase), 0.0])
    vc = math.sqrt(MU / 7000.0) * np.array([-math.sin(phase), math.cos(phase), 0.0])
    _, rest = relative.lvlh_state(rt, vt, rc, vc)
    assert float(np.linalg.norm(rest)) < 1e-12


def test_cw_is_second_order_accurate() -> None:
    rt, vt, n = _circular(7000.0)
    quarter = 0.25 * 2 * math.pi / n
    err = {}
    for sep in (1.0, 10.0):
        rho0, rd0 = np.array([0.3, -1.0, 0.2]) * sep, np.array([1e-4, -5e-5, 0.0]) * sep
        rc, vc = relative.lvlh_to_inertial(rt, vt, rho0, rd0)
        rt1, vt1 = iod.kepler_universal(rt, vt, quarter, MU)
        rc1, vc1 = iod.kepler_universal(rc, vc, quarter, MU)
        exact = relative.lvlh_state(rt1, vt1, rc1, vc1)[0]
        err[sep] = float(np.linalg.norm((relative.cw_stm(n, quarter) @ np.concatenate([rho0, rd0]))[:3] - exact))
    assert 80.0 < err[10.0] / err[1.0] < 120.0 and err[1.0] < 1e-3


def test_cw_rendezvous_refuses_a_whole_orbit() -> None:
    _, _, n = _circular(7000.0)
    with pytest.raises(ValueError, match="singular"):
        relative.cw_rendezvous(np.array([0.0, -1.0, 0.0]), np.zeros(3), n, 2 * math.pi / n)


RT = 6800.0
N = math.sqrt(MU / RT ** 3)
TOF = 0.4 * 2 * math.pi / N
STEPS = 60


def _fly(session: Session, scale_km: float, planner: str) -> Tuple[float, float]:
    sim = scenarios.artemis3_rendezvous(session, chaser_radius_km=RT - 0.5 * scale_km, target_radius_km=RT,
                                        inclination_deg=51.6, target_lead_deg=math.degrees(2.0 * scale_km / RT))
    sim.record_history = False
    c, t, e = (sim.name_to_index[k] for k in (scenarios.ORION_NAME, scenarios.LANDER_NAME, "Earth"))

    def state(k: int) -> Tuple[np.ndarray, np.ndarray]:
        x = sim.global_states[k] - sim.global_states[e]
        return x[:3], x[3:]
    (rt, vt), (rc, vc) = state(t), state(c)
    rho0, rd0 = relative.lvlh_state(rt, vt, rc, vc)
    if planner == "cw":
        dv1, _ = relative.cw_rendezvous(rho0, rd0, N, TOF)
        sim.apply_delta_v([c], relative.lvlh_dv_to_rsw(rt, vt, rc, vc, dv1))
    else:
        rtf, _ = iod.kepler_universal(rt, vt, TOF, MU)
        v1, _ = iod.lambert(rc, rtf, TOF, MU)
        sim.apply_delta_v([c], relative.inertial_dv_to_rsw(rc, vc, v1 - vc))
    for _ in range(STEPS):
        sim.step(TOF / STEPS)
    (rt, vt), (rc, vc) = state(t), state(c)
    return float(np.linalg.norm(rho0)), float(np.linalg.norm(relative.lvlh_state(rt, vt, rc, vc)[0]))


def test_cw_rendezvous_misses_by_the_linearisation_and_lambert_does_not(
        db_session_factory: Callable[[], Session]) -> None:
    misses = {}
    for scale in (2.0, 10.0):
        sep, misses[scale] = _fly(db_session_factory(), scale, "cw")
        _, exact = _fly(db_session_factory(), scale, "lambert")
        assert exact < 1e-8
    assert 20.0 < misses[10.0] / misses[2.0] < 30.0               # x5 separation, x25 miss
    assert misses[2.0] < 0.05
