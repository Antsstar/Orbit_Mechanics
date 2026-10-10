"""
Walker constellations (`scenarios.walker_elements`, `scenarios.walker_constellation`).

Expected, derived before measuring:
- **The definition.** `T/P` satellites per plane `2 pi P / T` apart in argument of latitude; adjacent
  planes `2 pi / P` apart in RAAN (delta) or `pi / P` (star), and phased `2 pi F / T` apart.
- **F = 0 delta is the existing builder.** `earth_constellation` lays satellites out the same way with
  no inter-plane phasing, so for T divisible by P the two arenas are bit-identical.
- **A Walker pattern is a group orbit, in space and time together.** Not at one instant: a satellite
  over the equator and one at its highest latitude have different neighbourhoods. The symmetry is a
  rotation by one plane spacing combined with a time shift, so **every satellite sees exactly what
  satellite (0, 0) sees when (0, 0) reaches the same argument of latitude**. Under two-body motion
  with identical circular orbits this is exact. It is checked to 1e-9 relative by propagating a twin
  until (0, 0) reaches each satellite's argument of latitude. A negative control with an inter-plane
  offset that is not a Walker phasing (0.1 rad) breaks it, so the check can fail. (The first version
  of this test asserted the instant-wise property and failed on every pattern: that claim was wrong.)
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.simulator import Simulation


def test_the_definition() -> None:
    raan, u, plane = scenarios.walker_elements(24, 3, 1)                 # Galileo-like 24/3/1
    assert raan.shape == u.shape == plane.shape == (24,)
    assert np.allclose(np.unique(raan), [0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    first = [u[plane == p][0] for p in range(3)]
    assert np.allclose(np.diff(first), 2 * math.pi * 1 / 24)
    assert np.allclose(np.diff(u[plane == 0]), 2 * math.pi / 8)
    star_raan, _, _ = scenarios.walker_elements(66, 6, 2, pattern="star")
    assert np.isclose(np.max(star_raan), math.pi * 5 / 6)
    for bad in ((25, 3, 1), (24, 3, 3), (24, 0, 0)):
        with pytest.raises(ValueError):
            scenarios.walker_elements(*bad)
    with pytest.raises(ValueError, match="pattern"):
        scenarios.walker_elements(24, 3, 1, pattern="rosette")


def test_f0_delta_is_earth_constellation(db_session_factory: Callable[[], Session]) -> None:
    a = scenarios.earth_constellation(db_session_factory(), n_sats=24, n_planes=4)
    b = scenarios.walker_constellation(db_session_factory(), total=24, planes=4, phasing=0,
                                       inclination_deg=53.0, altitude_km=550.0)
    assert a.name_to_index == b.name_to_index
    assert np.array_equal(a.global_states, b.global_states)


def _sats(sim: Simulation) -> list:
    return [k for n, k in sim.name_to_index.items() if n.startswith("SAT-")]


def _view(sim: Simulation, j: int) -> np.ndarray:
    """The sorted distances from satellite slot `j` to every satellite."""
    x = sim.global_states[_sats(sim), :3]
    out: np.ndarray = np.sort(np.linalg.norm(x - sim.global_states[j, :3], axis=1))
    return out


def _symmetry_error(build: Callable[[], Simulation]) -> float:
    """Worst relative mismatch between each satellite's view now and satellite (0, 0)'s view when it
    reaches that satellite's argument of latitude."""
    sim = build()
    sats = _sats(sim)
    first = sats[0]
    radius = float(np.linalg.norm(sim.global_states[first, :3]))
    n = math.sqrt(scenarios.MU_EARTH / radius ** 3)
    u = sim.coe_states[sats, 5]
    worst = 0.0
    for j, k in enumerate(sats):
        twin = build()
        twin.record_history = False
        tau = float(np.mod(u[j] - u[0], 2 * math.pi)) / n
        if tau > 0.0:
            twin.step(tau)
        mine, theirs = _view(sim, k), _view(twin, first)            # same build, so same slots
        worst = max(worst, float(np.max(np.abs(mine - theirs)) / np.max(mine)))
    return worst


@pytest.mark.parametrize("total, planes, phasing", [(24, 3, 1), (24, 6, 4), (12, 4, 0), (30, 5, 3)])
def test_every_satellite_sees_what_the_first_sees_at_its_latitude(
        db_session_factory: Callable[[], Session], total: int, planes: int, phasing: int) -> None:
    def build() -> Simulation:
        return scenarios.walker_constellation(db_session_factory(), total=total, planes=planes,
                                              phasing=phasing, inclination_deg=56.0, altitude_km=1200.0)
    assert _symmetry_error(build) < 1e-9


def test_a_non_walker_phasing_breaks_the_symmetry(
        db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch) -> None:
    real = scenarios.walker_elements

    def skewed(total: int, planes: int, phasing: int, pattern: str = "delta"):  # type: ignore[no-untyped-def]
        raan, u, plane = real(total, planes, phasing, pattern)
        return raan, np.mod(u + 0.1 * plane, 2 * math.pi), plane

    monkeypatch.setattr(scenarios, "walker_elements", skewed)

    def build() -> Simulation:
        return scenarios.walker_constellation(db_session_factory(), total=24, planes=3, phasing=1,
                                              inclination_deg=56.0, altitude_km=1200.0)
    assert _symmetry_error(build) > 1e-3


def test_a_star_pattern_spans_half_a_circle(db_session: Session) -> None:
    sim = scenarios.walker_constellation(db_session, total=66, planes=6, phasing=2, inclination_deg=86.4,
                                         altitude_km=780.0, pattern="star")
    r = np.linalg.norm(sim.global_states[_sats(sim), :3], axis=1)
    assert np.ptp(r) / r.mean() < 1e-12
    raan = np.mod(sim.coe_states[_sats(sim), 3], 2 * math.pi)
    raan[np.isclose(raan, 2 * math.pi)] = 0.0                       # rv_to_coe may return 2 pi for 0
    assert np.allclose(np.unique(np.round(raan, 9)), np.arange(6) * math.pi / 6)
