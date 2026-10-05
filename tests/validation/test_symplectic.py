"""
Symplectic Cowell integrators (`integrators.LeapfrogIntegrator`, `Yoshida4Integrator`) against RK4,
on `scenarios.two_body` (e = 0.2, p = 11,000 km, period 12,207 s), Cowell + point mass only, against
the exact conic (`iod.kepler_universal`).

Expected, derived before measuring:
- **Order.** Leapfrog is second order, Yoshida's composition fourth: halving h divides the one-orbit
  error by 4 and 16. (RK4 is fourth order too; at these steps its ratio is still above its asymptotic
  16.) *Measured:* 4.02 / 4.00 and 16.00 / 16.00; RK4 21.4 / 19.1. The Yoshida weights were cited from
  memory: this is the test that verifies them.
- **Energy.** A symplectic method conserves a nearby "shadow" energy, so its energy error oscillates
  with the orbit and does **not** grow; RK4's drifts linearly. The phase-independent measure is the
  maximum of |dE/E| over whole orbits: orbits 96-100 against 1-5 should give ~1 for leapfrog and
  Yoshida and ~100/5 = 20 for RK4. *Measured:* 1.00, 1.00, 20.05. (Sampling once per *true* period
  instead shows the symplectic error growing as t^2: the numerical orbit drifts in phase, and the
  oscillating error is sampled at a moving phase - an artefact, which is why the window maximum is used.)
- **Position, and where the methods cross.** RK4's energy drift makes its along-track error grow as
  t^2; a symplectic method's phase error grows as t. At the same four evaluations per step (h = P/64)
  RK4 starts ahead and Yoshida overtakes it after some tens of orbits. *Measured:* RK4 0.67 km vs
  Yoshida 11.3 km after one orbit; 3,190 km vs 1,130 km after 100. Crossover near 30 orbits.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import iod, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec, ModelConfig, apply_config

MU = scenarios.MU_EARTH
P_KM, E = 11000.0, 0.2
PERIOD = 2.0 * math.pi * math.sqrt((P_KM / (1.0 - E * E)) ** 3 / MU)


def _build(session: Session, integrator: str) -> Tuple[Simulation, int]:
    sim = scenarios.two_body(session, p=P_KM, e=E)
    k = sim.name_to_index["Secondary"]
    sim.record_history = False
    sim.set_propagator(np.array([k], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [k])
    sim.set_cowell_integrator(integrator)
    return sim, k


def _rel(sim: Simulation, k: int) -> np.ndarray:
    out: np.ndarray = sim.global_states[k] - sim.global_states[sim.parent_indices[k]]
    return out


def _energy(x: np.ndarray) -> float:
    return 0.5 * float(x[3:] @ x[3:]) - MU / float(np.linalg.norm(x[:3]))


def _error_after(session_factory: Callable[[], Session], integrator: str, steps_per_orbit: int, orbits: int) -> float:
    sim, k = _build(session_factory(), integrator)
    x0 = _rel(sim, k).copy()
    for _ in range(steps_per_orbit * orbits):
        sim.step(PERIOD / steps_per_orbit)
    r_true, _ = iod.kepler_universal(x0[:3], x0[3:], orbits * PERIOD, MU)
    return float(np.linalg.norm(_rel(sim, k)[:3] - r_true))


@pytest.mark.parametrize("integrator, ratio", [("leapfrog", 4.0), ("yoshida4", 16.0)])
def test_convergence_order(db_session_factory: Callable[[], Session], integrator: str, ratio: float) -> None:
    coarse = _error_after(db_session_factory, integrator, 64, 1)
    fine = _error_after(db_session_factory, integrator, 128, 1)
    assert abs(coarse / fine / ratio - 1.0) < 0.05


def test_symplectic_energy_is_bounded_and_rk4_drifts(db_session_factory: Callable[[], Session]) -> None:
    ratios = {}
    for name in ("rk4", "leapfrog", "yoshida4"):
        sim, k = _build(db_session_factory(), name)
        e0 = _energy(_rel(sim, k))
        worst = []
        for _ in range(100):
            m = 0.0
            for _ in range(64):
                sim.step(PERIOD / 64)
                m = max(m, abs(_energy(_rel(sim, k)) / e0 - 1.0))
            worst.append(m)
        ratios[name] = max(worst[95:]) / max(worst[:5])
    assert abs(ratios["leapfrog"] - 1.0) < 0.05 and abs(ratios["yoshida4"] - 1.0) < 0.05
    assert 15.0 < ratios["rk4"] < 25.0


def test_yoshida_overtakes_rk4_at_long_horizons(db_session_factory: Callable[[], Session]) -> None:
    early = _error_after(db_session_factory, "rk4", 64, 1) / _error_after(db_session_factory, "yoshida4", 64, 1)
    late = _error_after(db_session_factory, "rk4", 64, 100) / _error_after(db_session_factory, "yoshida4", 64, 100)
    assert early < 0.2 and late > 2.0


def test_selection_is_data_and_stays_on_the_fused_path(db_session: Session) -> None:
    sim, k = _build(db_session, "rk4")
    assert sim._cowell_fused_ok
    sim.set_cowell_integrator("yoshida4")
    assert sim.cowell_integrator == "yoshida4" and sim._cowell_fused_ok   # each integrator has its own twin
    with pytest.raises(ValueError, match="unknown integrator"):
        sim.set_cowell_integrator("euler")
    cfg = ModelConfig("y", PropagatorType.COWELL, 60.0, force_models=(ForceModelSpec("point_mass_gravity"),),
                      integrator="leapfrog")
    apply_config(sim, cfg)
    assert sim.cowell_integrator == "leapfrog"
    with pytest.raises(ValueError, match="only applies to"):
        apply_config(sim, ModelConfig("k", PropagatorType.KEPLERIAN, 60.0, integrator="leapfrog"))
