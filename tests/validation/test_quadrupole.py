"""
The system quadrupole (`quadrupole.py`, `"system_quadrupole"`) and the fidelity ladder it completes:
monopole -> averaged quadrupole -> instantaneous quadrupole -> resolved members.

Scenario: `scenarios.binary_probe`, an isolated Earth-Moon system (d ~ 3.84e5 km) with a massless probe
on a 30 deg inclined orbit; truth is DOP853 N-body of Earth, Moon and probe.

Expected, derived before measuring:
- **The field.** The kernel equals the exact two-point field minus the monopole, up to the octupole:
  a residual relative to the quadrupole of order `(d / r)` times the mass asymmetry `(m_E - m_M) / M =
  0.976`. *Measured* 0.257 at d/r = 0.196 and 0.516 at 0.392: linear in d/r, coefficient ~1.3.
- **The average is the average.** The averaged kernel equals the live kernel averaged over one inner
  orbit at a fixed probe position. *Measured* 3.6e-15 relative.
- **The ladder against truth** (r = 2e6 km, adaptive steps so truncation is negligible):
  resolved (Earth + Moon as a staged third body) is exact: 3.5e-7 km at 30 d (a verification).
  Instantaneous quadrupole: its residual is the octupole, ~(1.3 d/r) of the quadrupole, so ~4x better
  than the monopole. *Measured* 13 vs 54 km at 30 d, 117 vs 5,750 km at 360 d.
  Averaged quadrupole: it keeps the secular effect and drops the periodic ones, but it is started from
  the **osculating** state, not the mean one. That leaves a short-period velocity offset that drifts
  linearly (like any averaged theory seeded with osculating elements; compare `CLAUDE.md` on mean
  elements). The offset dominates at first; the secular term it captures grows as t^2 and wins later.
  *Measured* worse than the monopole at 30 d (76 vs 54 km), better from ~90 d (393 vs 835 km), 6.9x
  better at 360 d (835 vs 5,750 km).
- **Mean seeding removes that offset** (`quadrupole.mean_seed`). The periodic quadrupole has amplitude
  ~`a_Q` at twice the Moon's mean motion, so the osculating seed carries ~`a_Q / (2n)` = 2.5e-5 km/s and
  ~`a_Q / (2n)^2` = 5 km. *Measured* 2.2e-5 km/s and 5.0 km. Seeded, the averaged rung should then
  track the instantaneous one to about that 5 km periodic amplitude, because both omit the same
  octupole. *Measured* within 10 km through 90 d (unseeded: 317 km). Against truth it falls from 76
  to 18 km at 30 d, now below the monopole's 54, and 74 km at 90 d, against the instantaneous rung's 82.
  Not explained: beyond ~120 d the two rungs part again (34 km apart at 120 d, against 505 unseeded),
  and the seeded averaged rung is the closer to truth at 180 and 360 d (103 and 34 km, against 190 and
  117). That is not claimed as the averaged model being better; it is an open residual.
  First claim wrong: the offset was first integrated over a window *starting* now, and 0.8x of it fitted
  better than 1x by 120 d. The probe moves ~0.5 rad of its orbit in one lunar month, so a one-sided
  window biases the mean; a centred one put the best fit back at 1x.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType as PT
from orbital_engine.quadrupole import QUADRUPOLE_MODEL, mean_seed, mean_seed_offset, quadrupole_kernel
from orbital_engine.reference import reference_for
from orbital_engine.simulator import Simulation

DAY = 86400.0
DT = 3600.0


def _slots(sim: Simulation) -> Tuple[int, int, int, int]:
    return tuple(sim.name_to_index[n] for n in ("Earth", "Moon", "EMB", scenarios.BINARY_PROBE))  # type: ignore[return-value]


def _kernel(sim: Simulation, x: np.ndarray, mode: float, t: float = 0.0) -> np.ndarray:
    earth, moon, emb, probe = _slots(sim)
    state = sim.global_states.copy()
    state[probe, :3] = x
    parents = sim.parent_indices.copy()
    parents[probe] = emb
    params = np.zeros((sim.max_capacity, 4))
    params[probe] = [mode, earth, moon, 0.0]
    out = np.zeros((sim.max_capacity, 3))
    quadrupole_kernel(np.array([probe]), t, state, sim.mu_array, parents, params, out)
    result: np.ndarray = out[probe]
    return result


def test_the_field_is_the_quadrupole(db_session: Session) -> None:
    sim = scenarios.binary_probe(db_session)
    earth, moon, emb, probe = _slots(sim)
    g, mu = sim.global_states, sim.mu_array
    d = float(np.linalg.norm(g[moon, :3] - g[earth, :3]))
    rng = np.random.default_rng(3)
    worst = {}
    for r in (2.0e6, 1.0e6):
        res = []
        for _ in range(20):
            u = rng.normal(size=3)
            x = g[emb, :3] + r * u / np.linalg.norm(u)
            exact = sum(-mu[k] * (x - g[k, :3]) / np.linalg.norm(x - g[k, :3]) ** 3 for k in (earth, moon))
            beyond_monopole = exact + mu[emb] * (x - g[emb, :3]) / r ** 3
            res.append(np.linalg.norm(_kernel(sim, x, 1.0) - beyond_monopole) / np.linalg.norm(beyond_monopole))
        worst[d / r] = max(res)
    (small, w_small), (large, w_large) = sorted(worst.items())
    assert w_small < 0.3 and 1.7 < w_large / w_small < 2.3        # the octupole: linear in d/r


def test_the_averaged_field_is_the_time_average(db_session: Session) -> None:
    sim = scenarios.binary_probe(db_session)
    earth, moon, emb, probe = _slots(sim)
    g, mu = sim.global_states, sim.mu_array
    d0, v0 = g[moon, :3] - g[earth, :3], g[moon, 3:] - g[earth, 3:]
    m = mu[earth] + mu[moon]
    a = 1.0 / (2.0 / np.linalg.norm(d0) - v0 @ v0 / m)
    period = 2.0 * np.pi * np.sqrt(a ** 3 / m)
    x = g[emb, :3] + np.array([7e5, 4e5, 5e5])
    live = np.mean([_kernel(sim, x, 1.0, t) for t in np.linspace(0.0, period, 4001)[:-1]], axis=0)
    assert float(np.linalg.norm(live - _kernel(sim, x, 0.0)) / np.linalg.norm(live)) < 1e-12


def test_refusals(db_session: Session) -> None:
    sim = scenarios.binary_probe(db_session)
    earth, moon, emb, probe = _slots(sim)
    sim.set_propagator(np.array([probe]), PT.COWELL)
    with pytest.raises(ValueError, match="not a barycentre"):
        sim.enable_force_model(QUADRUPOLE_MODEL, [probe], primary=float(earth), secondary=float(moon))
    sim.reparent(probe, emb)
    with pytest.raises(ValueError, match="engine-owned"):
        sim.enable_force_model(QUADRUPOLE_MODEL, [probe], primary=float(earth), secondary=float(moon), t0=1.0)
    with pytest.raises(ValueError, match="mode"):
        sim.enable_force_model(QUADRUPOLE_MODEL, [probe], primary=float(earth), secondary=float(moon), mode=2.0)
    with pytest.raises(ValueError, match="needs 'secondary'"):
        sim.enable_force_model(QUADRUPOLE_MODEL, [probe], primary=float(earth))
    with pytest.raises(ValueError, match="not exactly"):
        sim.enable_force_model(QUADRUPOLE_MODEL, [probe], primary=float(earth), secondary=float(emb))


def _ladder(session_factory: Callable[[], Session], rung: str, horizons_d: Tuple[float, ...],
            positions: bool = False) -> Dict[float, float]:
    sim = scenarios.binary_probe(session_factory(), probe_radius_km=2.0e6)
    sim.record_history = False
    earth, moon, emb, probe = _slots(sim)
    truth = reference_for(sim, np.array([0.0] + [h * DAY for h in horizons_d]))
    seeded = rung == "averaged-seeded"
    rung = "averaged" if seeded else rung
    sim.set_propagator(np.array([probe]), PT.COWELL)
    if rung == "resolved":
        sim.enable_force_model("point_mass_gravity", [probe])
        sim.enable_force_model("third_body", [probe], perturber=float(moon), staged=1.0)
    else:
        sim.reparent(probe, emb)
        sim.enable_force_model("point_mass_gravity", [probe])
        if rung != "monopole":
            sim.enable_force_model(QUADRUPOLE_MODEL, [probe], mode=0.0 if rung == "averaged" else 1.0,
                                   primary=float(earth), secondary=float(moon))
    if seeded:
        mean_seed(sim, [probe])
    sim.set_cowell_tolerance(1e-5)
    out = {}
    for k, h in enumerate(horizons_d):
        while sim.t < h * DAY - 1.0:
            sim.step(DT)
        out[h] = sim.global_states[probe, :3].copy() if positions else float(
            np.linalg.norm(sim.global_states[probe, :3] - truth.position_of(scenarios.BINARY_PROBE)[k + 1]))
    return out


def test_the_ladder_against_truth(db_session_factory: Callable[[], Session]) -> None:
    horizons = (30.0, 180.0)
    err = {rung: _ladder(db_session_factory, rung, horizons)
           for rung in ("monopole", "averaged", "instantaneous", "resolved")}
    assert err["resolved"][30.0] < 1e-5                                   # verification: exact physics
    for h in horizons:
        assert err["instantaneous"][h] < err["monopole"][h] / 3.0
    assert err["averaged"][30.0] > err["monopole"][30.0]                  # osculating start: the offset first
    assert err["averaged"][180.0] < err["monopole"][180.0] / 3.0          # then the secular term wins


def test_the_mean_seed_offset_is_the_periodic_amplitude(db_session: Session) -> None:
    sim = scenarios.binary_probe(db_session, probe_radius_km=2.0e6)
    earth, moon, emb, probe = _slots(sim)
    sim.set_propagator(np.array([probe]), PT.COWELL)
    sim.reparent(probe, emb)
    sim.enable_force_model(QUADRUPOLE_MODEL, [probe], mode=0.0, primary=float(earth), secondary=float(moon))
    g, mu = sim.global_states, sim.mu_array
    d = float(np.linalg.norm(g[moon, :3] - g[earth, :3]))
    a_q = 3.0 * mu[earth] * mu[moon] / mu[emb] * d * d / 2.0e6 ** 4
    two_n = 2.0 * math.sqrt(mu[emb] / d ** 3)
    off = mean_seed_offset(sim, probe)
    assert 0.5 < np.linalg.norm(off[3:]) / (a_q / two_n) < 2.0
    assert 0.5 < np.linalg.norm(off[:3]) / (a_q / two_n ** 2) < 2.0
    before = g[probe].copy()
    removed = mean_seed(sim, [probe])
    assert np.array_equal(g[probe], before - removed[0])


def test_the_seeded_average_tracks_the_instantaneous_rung(db_session_factory: Callable[[], Session]) -> None:
    horizons = (30.0, 60.0, 90.0)
    live = _ladder(db_session_factory, "instantaneous", horizons, positions=True)
    seeded = _ladder(db_session_factory, "averaged-seeded", horizons, positions=True)
    raw = _ladder(db_session_factory, "averaged", horizons, positions=True)
    for h in horizons:
        assert np.linalg.norm(seeded[h] - live[h]) < 15.0                 # the periodic amplitude, ~5 km
    assert np.linalg.norm(raw[90.0] - live[90.0]) > 200.0                 # the osculating offset, drifting


def test_mean_seeding_beats_the_monopole_from_the_start(db_session_factory: Callable[[], Session]) -> None:
    seeded = _ladder(db_session_factory, "averaged-seeded", (30.0,))
    monopole = _ladder(db_session_factory, "monopole", (30.0,))
    assert seeded[30.0] < monopole[30.0] / 2.0


def test_mean_seeding_refusals(db_session: Session) -> None:
    sim = scenarios.binary_probe(db_session, probe_radius_km=2.0e6)
    earth, moon, emb, probe = _slots(sim)
    sim.set_propagator(np.array([probe]), PT.COWELL)
    sim.reparent(probe, emb)
    with pytest.raises(ValueError, match="mode = 0"):
        mean_seed(sim, [probe])                                            # no quadrupole at all
    sim.enable_force_model(QUADRUPOLE_MODEL, [probe], mode=1.0, primary=float(earth), secondary=float(moon))
    with pytest.raises(ValueError, match="mode = 0"):
        mean_seed(sim, [probe])                                            # the instantaneous rung
