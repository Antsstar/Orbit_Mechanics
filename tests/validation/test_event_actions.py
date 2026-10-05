"""
Event actions (`events.Event.action`, `max_fires`; `Simulation._fire_events`): something happens at a
located crossing, not only a split.

Expected magnitudes, stated before measuring:

- **A Hohmann second burn placed by an apoapsis event** against the same burn scheduled at the
  analytic transfer time (`test_manoeuvres.py`'s LEO-to-GEO case): the event fires late by at most
  `(1 + nudges) tol_s` ~ 1e-5 s, so the two runs part by ~`DV2 dt` = 1.47 km/s x 1e-5 s = 1.5e-5 km at
  the burn, growing a few-fold along-track over the next day: bound 1e-3 km, for the Keplerian and
  the Cowell vessel alike (for Cowell the two runs also split RK4 at slightly different instants, a
  change of `r (n h)^5 / 120` ~ 1e-15 km at GEO). The fire time is within 1e-5 s of `T_TRANSFER`.
  *Measured:* the Keplerian runs agree to 9.8e-6 km, its fire 6.8e-6 s **before** `T_TRANSFER` (I had
  assumed after: element re-derivation at the departure burn moves the numerical apoapsis). The Cowell
  bound was wrong in kind: its vessel reaches apoapsis 0.027 s early (60 s RK4 error over the
  transfer), so the event and the schedule burn 0.04 km apart - and the event's burn, at the
  integrated orbit's own apoapsis, circularises 23x better (e 6.2e-8 vs 1.4e-6). That is asserted.
- **Ascending node under Cowell + J2**: each fire leaves the body strictly past the equator, `0 <= z
  <= |v_z| (1 + MAX_CROSSING_NUDGES) tol_s` = 4 km/s x 9e-6 s ~ 4e-5 km.
- **`max_fires=1`**: one fire and one split over three orbits, then the body is no longer detected.
- **A do-nothing action** is bit-identical to no action: the fire path only reads the arena.
- **`enable_model`**: the model's mask bit is set from the fire on, and the trajectory up to the fire
  is the action-free one, bit for bit.
"""
from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
from sqlalchemy.orm import Session

from orbital_engine import events, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation

MU = scenarios.MU_EARTH
R1, R2 = 7000.0, 42164.0
A_T = 0.5 * (R1 + R2)
DV1 = math.sqrt(MU / R1) * (math.sqrt(2.0 * R2 / (R1 + R2)) - 1.0)
DV2 = math.sqrt(MU / R2) * (1.0 - math.sqrt(2.0 * R1 / (R1 + R2)))
T_TRANSFER = math.pi * math.sqrt(A_T ** 3 / MU)
DT = 60.0


def _fly(sim: Simulation, until: float, dt: float = DT) -> None:
    for _ in range(int(math.ceil(until / dt))):
        sim.step(dt)


def test_hohmann_circularisation_placed_by_an_apoapsis_event(db_session_factory: Callable[[], Session]) -> None:
    horizon = T_TRANSFER + 86400.0
    scheduled = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    by_event = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sats = [scheduled.name_to_index["KEPLER-SAT"], scheduled.name_to_index["COWELL-SAT"]]
    scheduled.schedule_delta_v(sats, (0.0, DV1, 0.0), 0.0)
    scheduled.schedule_delta_v(sats, (0.0, DV2, 0.0), T_TRANSFER)
    by_event.apply_delta_v(sats, (0.0, DV1, 0.0))
    by_event.add_event(events.apsis_event(np.array(sats), action=events.burn(np.array([0.0, DV2, 0.0])),
                                          max_fires=1))
    _fly(scheduled, horizon)
    _fly(by_event, horizon)
    kep, cow = sats
    fired_at = {f[1][0]: f[2] for f in by_event.event_fires}
    assert len(by_event.event_fires) == 2 and set(fired_at) == {kep, cow}
    # Keplerian: the event lands on the analytic apoapsis (measured 6.8e-6 s early: the departure burn
    # re-derives the elements, whose rounding moves the numerical apoapsis by microseconds), and the
    # two runs agree (measured 9.8e-6 km).
    assert abs(fired_at[kep] - T_TRANSFER) < 1e-5
    assert float(np.linalg.norm(by_event.global_states[kep, :3] - scheduled.global_states[kep, :3])) < 1e-3
    # Cowell at 60 s reaches apoapsis 0.027 s early (its RK4 error over the transfer). The event burns
    # there, at the integrated orbit's own apoapsis, so it circularises better than the analytic
    # schedule does: e 6.2e-8 against 1.4e-6, measured.
    assert 0.0 < T_TRANSFER - fired_at[cow] < 1.0
    assert _eccentricity(by_event, cow) < 0.1 * _eccentricity(scheduled, cow)


def _eccentricity(sim: Simulation, k: int) -> float:
    rel = sim.global_states[k] - sim.global_states[sim.parent_indices[k]]
    h = np.cross(rel[:3], rel[3:])
    e = np.cross(rel[3:], h) / MU - rel[:3] / np.linalg.norm(rel[:3])
    return float(np.linalg.norm(e))


def _inclined_cowell(session: Session, j2: bool = True) -> tuple[Simulation, int]:
    sim = scenarios.coplanar_satellites(session, altitudes_km=[500.0], phases_deg=[30.0], inclination_deg=33.0)
    k = sim.name_to_index[scenarios.coplanar_satellite_name(0)]
    sim.set_propagator(np.array([k], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [k])
    if j2:
        sim.enable_force_model(J2_MODEL, [k], j2=EARTH_J2, r_eq=EARTH_R_EQ)
    return sim, k


def _period(sim: Simulation, k: int) -> float:
    r = float(np.linalg.norm(sim.global_states[k, :3]))
    return 2.0 * math.pi * math.sqrt(r ** 3 / MU)


def test_ascending_node_fires_just_past_the_equator(db_session: Session) -> None:
    sim, k = _inclined_cowell(db_session)
    seen: List[tuple[float, float]] = []

    def record(s: Simulation, fired: np.ndarray, t: float) -> None:
        rel = s.global_states[fired[0]] - s.global_states[s.parent_indices[fired[0]]]
        seen.append((float(rel[2]), float(rel[5])))

    sim.add_event(events.node_event(np.array([k]), action=record))
    _fly(sim, 3.0 * _period(sim, k), dt=30.0)
    assert len(seen) == 3
    bound = (1 + events.MAX_CROSSING_NUDGES) * events.DEFAULT_EVENT_TOL_S
    for z, vz in seen:
        assert vz > 0.0 and 0.0 <= z <= vz * bound


def test_max_fires_retires_the_body(db_session: Session) -> None:
    sim, k = _inclined_cowell(db_session)
    sim.add_event(events.node_event(np.array([k]), max_fires=1))
    _fly(sim, 3.0 * _period(sim, k), dt=30.0)
    assert len(sim.event_fires) == 1 and sim.event_splits == 1


def test_a_do_nothing_action_is_bit_identical(db_session_factory: Callable[[], Session]) -> None:
    plain, k = _inclined_cowell(db_session_factory())
    acting, _ = _inclined_cowell(db_session_factory())
    plain.add_event(events.node_event(np.array([k])))
    acting.add_event(events.node_event(np.array([k]), action=lambda s, f, t: None))
    for sim in (plain, acting):
        _fly(sim, 2.0 * _period(sim, k), dt=30.0)
    assert np.array_equal(plain.global_states, acting.global_states)
    assert plain.event_epochs == acting.event_epochs and len(acting.event_fires) == 2


def test_enable_model_action_switches_physics_at_the_fire(db_session_factory: Callable[[], Session]) -> None:
    """Point mass only until the first ascending node, then J2 on: the mask bit appears at the fire,
    and before the fire the run is the J2-free one, bit for bit."""
    free, k = _inclined_cowell(db_session_factory(), j2=False)
    switched, _ = _inclined_cowell(db_session_factory(), j2=False)
    free.add_event(events.node_event(np.array([k])))
    switched.add_event(events.node_event(np.array([k]), max_fires=1,
                                         action=events.enable_model(J2_MODEL, j2=EARTH_J2, r_eq=EARTH_R_EQ)))
    period = _period(free, k)
    _fly(free, 0.5 * period, dt=30.0)
    _fly(switched, 0.5 * period, dt=30.0)
    assert np.array_equal(free.global_states, switched.global_states) and not switched.event_fires
    _fly(free, 1.0 * period, dt=30.0)
    _fly(switched, 1.0 * period, dt=30.0)
    assert len(switched.event_fires) == 1 and J2_MODEL in switched.force_model_params
    assert float(np.linalg.norm(free.global_states[k, :3] - switched.global_states[k, :3])) > 1e-3
