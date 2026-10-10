"""
Reparenting a massless body (`hierarchy.reparent`, `Simulation.reparent`) and patched conics as events
(`hierarchy.patch_events`, `Simulation.watch_patch`, `ModelConfig.patches`).

Expected, derived before measuring:
- **A frame change moves nothing**, and a body handed to a parent and back returns its elements to
  round-off. Built `sun_earth_moon(leo_satellite=True)` puts LEO-SAT about Earth inside the Earth-Moon
  bubble; handing it to the barycentre and back must reproduce the untouched build's track.
- **A barycentre parent is the monopole.** Handed to the Earth-Moon barycentre, the satellite follows
  the conic about it with the system's summed `mu`: the Keplerian propagator *is* that conic, so this
  is exact (*measured* 1.1e-7 km after 600 s). Measured against truth, the monopole's error is the
  system's quadrupole; that is the next switch, not tested here.
- **Patched conics on a strong flyby** (`scenarios.planet_flyby`: Earth, 10,000 km periapsis, v_inf
  3 km/s, a 109 deg turn, periapsis at day 30 of 60). Never handed over, the probe misses Earth's
  deflection entirely: ~v_inf x 30 d x (turn) ~ 1e7 km. *Measured* 1.38e7 km. Handed over inside
  0.76 Hill radii (symmetric 1.05 hysteresis) it is 2.0e5 km off, 70x better but still large: a 109
  deg bend amplifies any arrival error, so patched conics are a design tool here. The best radius
  over 0.01-10x the planet's mass (periapsis scaled with mass so the hyperbola keeps its shape) is
  0.71-0.86 Hill radii and scales as s^0.36, between Hill (0.33) and Laplace (0.40): unlike the
  asteroid pair, this test does not separate them (`benchmarks/encounter_sweep.py`).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import iod, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.hierarchy import EncounterPolicy, PatchSpec, sphere_radius_km
from orbital_engine.reference import reference_for
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ModelConfig, run_sweep

DAY = 86400.0


def _sem(session: Session, compiled: bool = True) -> Tuple[Simulation, int, int, int]:
    sim = scenarios.sun_earth_moon(session, leo_satellite=True)
    sim.use_compiled_kernel = compiled
    sim.record_history = False
    return sim, sim.name_to_index["LEO-SAT"], sim.name_to_index["Earth"], sim.name_to_index["EMB"]


@pytest.mark.parametrize("compiled", [True, False])
def test_there_and_back_is_the_built_hierarchy(db_session_factory: Callable[[], Session], compiled: bool) -> None:
    ref, *_ = _sem(db_session_factory(), compiled)
    sim, sat, earth, emb = _sem(db_session_factory(), compiled)
    g0, c0 = sim.global_states.copy(), sim.coe_states.copy()
    assert sim.reparent(sat, emb) == emb
    assert np.array_equal(sim.global_states, g0)
    assert sim.parent_indices[sat] == emb and sim.body_sys_map[sat] == emb
    assert sim.reparent(sat, earth) == emb                 # Earth heads the bubble: back inside it
    assert float(np.max(np.abs(sim.coe_states[sat] - c0[sat]) / np.maximum(1.0, np.abs(c0[sat])))) < 1e-12
    for _ in range(100):
        ref.step(60.0)
        sim.step(60.0)
    assert float(np.max(np.abs(sim.global_states[:, :3] - ref.global_states[:, :3]))) < 1e-6
    assert [c.kind for c in sim.hierarchy_changes] == ["reparent", "reparent"]


def test_a_barycentre_parent_is_the_monopole(db_session: Session) -> None:
    sim, sat, earth, emb = _sem(db_session)
    sim.reparent(sat, emb)
    x0 = sim.global_states[sat] - sim.global_states[emb]
    for _ in range(10):
        sim.step(60.0)
    r, _ = iod.kepler_universal(x0[:3], x0[3:], 600.0, float(sim.mu_array[emb]))
    assert float(np.linalg.norm(sim.global_states[sat, :3] - sim.global_states[emb, :3] - r)) < 1e-6


def test_a_cowell_body_follows_its_new_parent(db_session: Session) -> None:
    sim, sat, earth, emb = _sem(db_session)
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [sat])
    sim.reparent(sat, emb)
    x0 = sim.global_states[sat] - sim.global_states[emb]
    for _ in range(60):
        sim.step(10.0)
    r, _ = iod.kepler_universal(x0[:3], x0[3:], 600.0, float(sim.mu_array[emb]))
    # RK4 at 10 s on a 7,000 km orbit: ~1e-6 km over 600 s.
    assert float(np.linalg.norm(sim.global_states[sat, :3] - sim.global_states[emb, :3] - r)) < 1e-4


def test_refusals(db_session: Session) -> None:
    sim, sat, earth, emb = _sem(db_session)
    moon, sun = sim.name_to_index["Moon"], sim.name_to_index["Sun"]
    with pytest.raises(ValueError, match="has mass"):
        sim.reparent(moon, sun)
    with pytest.raises(ValueError, match="system or a head"):
        sim.reparent(earth, sun)
    with pytest.raises(ValueError, match="own parent"):
        sim.reparent(sat, sat)
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, [sat])
    sim.enable_force_model(J2_MODEL, [sat], j2=EARTH_J2, r_eq=EARTH_R_EQ)
    with pytest.raises(ValueError, match="beyond point_mass_gravity"):
        sim.reparent(sat, emb)
    assert sim.parent_indices[sat] == earth                # refused before anything changed


# --- Patched conics ------------------------------------------------------------------------------------

DT = 600.0
N = int(60 * DAY / DT)


def _flyby(session: Session) -> Tuple[Simulation, int, int]:
    sim = scenarios.planet_flyby(session)
    sim.record_history = False
    return sim, sim.name_to_index[scenarios.FLYBY_CRAFT], sim.name_to_index[scenarios.FLYBY_PLANET]


def test_the_flyby_is_built_by_time_reversal(db_session: Session) -> None:
    sim, probe, earth = _flyby(db_session)
    times = np.arange(0.0, 60 * DAY + 1.0, 3600.0)
    truth = reference_for(sim, times)
    d = np.linalg.norm(truth.position_of(scenarios.FLYBY_CRAFT) - truth.position_of(scenarios.FLYBY_PLANET), axis=1)
    assert abs(float(d.min()) - 10000.0) < 1.0 and abs(times[int(np.argmin(d))] - 30 * DAY) < 1.0


def test_patched_conics_against_truth(db_session_factory: Callable[[], Session]) -> None:
    sim, probe, earth = _flyby(db_session_factory())
    truth = reference_for(sim, np.array([0.0, 60 * DAY]))
    errors = {}
    for k in (None, 0.76):
        sim, probe, earth = _flyby(db_session_factory())
        if k is not None:
            sim.watch_patch(probe, earth, EncounterPolicy(k, 1.05 * k, unit="hill"))
        for _ in range(N):
            sim.step(DT)
        errors[k] = float(np.linalg.norm(sim.global_states[probe, :3] - truth.position_of(scenarios.FLYBY_CRAFT)[-1]))
        if k is not None:
            changes = sim.hierarchy_changes
            assert [(c.kind, c.bodies[1]) for c in changes] == [("reparent", earth), ("reparent", sim.name_to_index["Sun"])]
            assert changes[0].t < 30 * DAY < changes[1].t
    assert 1e7 < errors[None] < 2e7
    assert errors[0.76] < errors[None] / 30.0


def test_the_handover_lands_on_the_radius(db_session_factory: Callable[[], Session]) -> None:
    policy = EncounterPolicy(0.5, 0.6, unit="laplace")
    sim, probe, earth = _flyby(db_session_factory())
    sim.watch_patch(probe, earth, policy)
    for _ in range(N):
        sim.step(DT)
    t_in = sim.hierarchy_changes[0].t
    twin, tp, te = _flyby(db_session_factory())
    n = int(t_in // DT)
    for _ in range(n):
        twin.step(DT)
    twin.step(t_in - n * DT)
    d = float(np.linalg.norm(twin.global_states[tp, :3] - twin.global_states[te, :3]))
    assert abs(d - 0.5 * sphere_radius_km(twin, te, "laplace")) < 1e-2


def test_a_patch_is_a_sweep_configuration(db_session_factory: Callable[[], Session]) -> None:
    names = [scenarios.FLYBY_CRAFT]
    policy = EncounterPolicy(0.76, 0.8, unit="hill")
    results = run_sweep(
        lambda: scenarios.planet_flyby(db_session_factory()),
        [ModelConfig("never", PropagatorType.KEPLERIAN, DT, bodies=names),
         ModelConfig("patched", PropagatorType.KEPLERIAN, DT, bodies=names,
                     patches=(PatchSpec(scenarios.FLYBY_CRAFT, scenarios.FLYBY_PLANET, policy),))],
        60 * DAY, timing_batches=1, timing_warmup=0)
    assert results[1].error.max_km < results[0].error.max_km / 30.0
    with pytest.raises(ValueError, match="heads a system"):
        sim, probe, earth = _flyby(db_session_factory())
        sim.watch_patch(probe, earth, policy, target="system")


def test_patching_into_a_system_barycentre(db_session: Session) -> None:
    """`target="system"`: inside the radius the body orbits the Earth-Moon barycentre (the monopole)."""
    sim, sat, earth, emb = _sem(db_session)
    sun = sim.name_to_index["Sun"]
    sim.reparent(sat, sun)
    sim.watch_patch(sat, earth, EncounterPolicy(0.5, 0.6, unit="hill"), target="system")
    assert sim.parent_indices[sat] == emb                  # starts deep inside: handed over at once
