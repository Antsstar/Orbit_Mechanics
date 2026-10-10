"""
Adaptive Cowell sub-stepping (`Simulation.set_cowell_tolerance`, `ModelConfig.cowell_tolerance_km`):
inside each arena step the Cowell set takes as many RK4 sub-steps as a step-doubling error estimate
needs (`Simulation._cowell_adaptive`).

Expected, derived before measuring:
- **Off by default, bit for bit.** `None` is the fixed step every other test exercises.
- **The error follows the tolerance, and the steps go where the error is.** On a two-body e = 0.9 orbit
  (semi-latus rectum 11,000 km, so periapsis at 5,789 km and apoapsis at 110,000 km), RK4's
  local error is concentrated at periapsis, so sub-steps should bunch there (max per arena step far
  above the median) and the end error should fall roughly in proportion to the tolerance.
- **An event's micro-step must not collapse the step.** The crossing machinery advances ~1e-9 s across
  a bracket; the first version let that set the next sub-step suggestion and asked the following step
  for ~1e12 sub-steps (it hung). Regression: the regime flyby at 3,600 s with a tolerance completes in
  a bounded number of sub-steps. *Measured* 1,941 at 0.01 km.
- **The frontier.** The staged regime flyby (`test_regimes.py`) at a 3,600 s arena step and 1e-5 km:
  3.6e-3 km in 4,200 sub-steps, where the fixed step needs 30 s (57,600 steps) for 1.9e-2 km.
"""
from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import iod, scenarios
from orbital_engine.custom_types import PropagatorType as PT
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.hierarchy import EncounterPolicy
from orbital_engine.reference import reference_for
from orbital_engine.regimes import Regime, RegimeSwitch
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec as F, ModelConfig, apply_config

DAY = 86400.0
MU = scenarios.MU_EARTH
P_KM, E = 11000.0, 0.9
PERIOD = 2.0 * math.pi * math.sqrt((P_KM / (1.0 - E * E)) ** 3 / MU)


def _eccentric(session: Session, compiled: bool = True) -> Tuple[Simulation, int]:
    sim = scenarios.two_body(session, p=P_KM, e=E)
    sim.use_compiled_kernel = compiled
    sim.record_history = False
    k = sim.name_to_index["Secondary"]
    sim.set_propagator(np.array([k], dtype=np.int64), PT.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [k])
    return sim, k


def _rel(sim: Simulation, k: int) -> np.ndarray:
    out: np.ndarray = sim.global_states[k] - sim.global_states[sim.parent_indices[k]]
    return out


def _orbit_error(sim: Simulation, k: int, steps: int) -> Tuple[float, list]:
    x0 = _rel(sim, k).copy()
    per_step = []
    for _ in range(steps):
        c = sim.cowell_substeps_taken
        sim.step(PERIOD / steps)
        per_step.append(sim.cowell_substeps_taken - c)
    r, _ = iod.kepler_universal(x0[:3], x0[3:], PERIOD, MU)
    return float(np.linalg.norm(_rel(sim, k)[:3] - r)), per_step


def test_off_by_default_and_validated(db_session_factory: Callable[[], Session]) -> None:
    a, ka = _eccentric(db_session_factory())
    b, kb = _eccentric(db_session_factory())
    assert a.cowell_tolerance_km is None
    b.set_cowell_tolerance(None)
    for _ in range(20):
        a.step(600.0)
        b.step(600.0)
    assert np.array_equal(a.global_states, b.global_states) and b.cowell_substeps_taken == 0
    with pytest.raises(ValueError, match="positive"):
        b.set_cowell_tolerance(0.0)


def test_error_follows_the_tolerance_and_steps_bunch_at_periapsis(
        db_session_factory: Callable[[], Session]) -> None:
    errors = {}
    for tol in (1e-2, 1e-4):
        sim, k = _eccentric(db_session_factory())
        sim.set_cowell_tolerance(tol)
        errors[tol], per_step = _orbit_error(sim, k, 16)
        assert max(per_step) >= 8 * float(np.median(per_step))
    fixed, k = _eccentric(db_session_factory())
    fixed_err, _ = _orbit_error(fixed, k, 16)
    assert errors[1e-4] < errors[1e-2] / 10.0
    assert errors[1e-2] < fixed_err / 100.0


def test_compiled_and_numpy_adaptive_paths_agree(db_session_factory: Callable[[], Session]) -> None:
    out = []
    for compiled in (True, False):
        sim, k = _eccentric(db_session_factory(), compiled)
        sim.set_cowell_tolerance(1e-3)
        _orbit_error(sim, k, 16)
        out.append((_rel(sim, k).copy(), sim.cowell_substeps_taken))
    assert out[0][1] == out[1][1]
    assert float(np.max(np.abs(out[0][0][:3] - out[1][0][:3]))) < 1e-6


P, EARTH, S = scenarios.FLYBY_CRAFT, scenarios.FLYBY_PLANET, "Sun"


def _staged(centre: str, perturber: str) -> Regime:
    return Regime(centre, PT.COWELL, (F("point_mass_gravity"),
                                      F("third_body", {"staged": 1.0}, body_coefficients={"perturber": perturber})))


SWITCH = RegimeSwitch(P, EARTH, EncounterPolicy(0.76, 0.8, unit="hill"), _staged(EARTH, S), _staged(S, EARTH))


def _flyby_error(session: Session, tol: float) -> Tuple[float, int]:
    sim = scenarios.planet_flyby(session, t_ca_s=10 * DAY)
    sim.record_history = False
    truth = reference_for(sim, np.array([0.0, 20 * DAY]))
    apply_config(sim, ModelConfig("adaptive", PT.KEPLERIAN, 3600.0, bodies=[P], regimes=(SWITCH,),
                                  cowell_tolerance_km=tol))
    for _ in range(int(20 * DAY / 3600.0)):
        sim.step(3600.0)
    err = float(np.linalg.norm(sim.global_states[sim.name_to_index[P], :3] - truth.position_of(P)[-1]))
    return err, sim.cowell_substeps_taken


def test_an_event_micro_step_does_not_collapse_the_step(db_session: Session) -> None:
    err, substeps = _flyby_error(db_session, 1e-2)
    assert substeps < 4000
    assert err < 20.0


def test_adaptive_beats_the_finest_fixed_step(db_session: Session) -> None:
    err, substeps = _flyby_error(db_session, 1e-5)
    assert err < 1.86e-2                       # the fixed 30 s step's error (test_regimes / the frontier)
    assert substeps < 57600 / 5                # its 57,600 steps


def test_a_cowell_parent_is_refused(db_session: Session) -> None:
    sim = scenarios.sun_earth_moon(db_session, moon_mu=0.0, leo_satellite=True)
    sim.record_history = False
    moon, sat = sim.name_to_index["Moon"], sim.name_to_index["LEO-SAT"]
    sim.set_propagator(np.array([moon], dtype=np.int64), PT.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [moon])
    with pytest.raises(ValueError, match="massless"):
        sim.reparent(sat, moon)                          # a massless parent defines no orbit
    sim.parent_indices[sat] = moon                       # so the case is built by hand: only the refusal is checked
    sim.set_propagator(np.array([sat], dtype=np.int64), PT.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [sat])
    sim.set_cowell_tolerance(1e-3)
    with pytest.raises(ValueError, match="non-Cowell"):
        sim.step(60.0)
