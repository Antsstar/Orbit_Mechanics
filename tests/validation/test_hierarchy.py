"""
Temporary systems (`hierarchy.py`): `form_system` / `dissolve_system` at runtime.

Expected, derived before measuring:
- **A frame change moves nothing.** Forming or dissolving keeps every global state; the new
  barycentre is the members' mass-weighted mean, so it is exact to round-off (an ulp of 1.5e8 km is
  ~3e-8 km). Every row the operation does not touch is bit-identical: its float work is local.
  *Measured:* members moved 0.0; barycentre 7.5e-9 km; untouched rows bit-identical.
- **A zero-length step after a restructure moves nothing** either. It re-derives every local state
  from the new elements and the reflex kicks, so it checks that the rehydrated elements and the
  kick agree, at both levels (the head's offset from the new barycentre; the outer head's kick, where
  `M_S r_S = m_a r_a + m_b r_b`). The yardstick is the arena's own floor: a zero step on an
  *unrestructured* `sun_earth_moon` already moves Earth by 2.7e-4 km at build and 4.7e-6 km after
  five days (the elements-to-state round trip of a near-equatorial orbit at 1.5e8 km). Bound 1e-12
  of the position scale, 1.5e-4 km. *Measured:* 5.1e-6 km after forming, 3.7e-5 km after dissolving
  (Earth's z, where its heliocentric inclination is 4e-5 rad).
- **Dissolve then re-form returns the built hierarchy.** Starting from `sun_earth_moon`, dissolving the
  Earth-Moon system and forming it again reuses the freed slot and returns every element to round-off
  (*measured* 1.5e-14 relative). Thirty days later it agrees with the untouched build to round-off
  growth: *measured* 1.3e-7 km compiled, 1.4e-7 km NumPy, at 1.5e8 km.
- **Which model is better is a comparison, not a verification.** Against DOP853 N-body truth over 30
  days, the formed Earth-Moon system misses by the solar perturbation of the lunar orbit the
  hierarchy neglects (3.3e4 km, `docs/architecture.md`), well under the 1e5 km coherent-forcing bound.
  Dissolved, the Moon and Earth each fly their own heliocentric conic, and the Moon keeps the ~1 km/s
  it had relative to Earth: ~1 km/s x 2.6e6 s ~ 1e6 km. *Measured:* 3.3e4 km and 2.0e6 km.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.reference import reference_for
from orbital_engine.simulator import Simulation

DAY = 86400.0


def _sem(session: Session, compiled: bool = True, **kw: object) -> Simulation:
    sim = scenarios.sun_earth_moon(session, **kw)  # type: ignore[arg-type]
    sim.use_compiled_kernel = compiled
    return sim


def _slots(sim: Simulation, *names: str) -> list[int]:
    return [sim.name_to_index[n] for n in names]


@pytest.mark.parametrize("compiled", [True, False])
def test_dissolve_then_form_returns_the_built_hierarchy(
        db_session_factory: Callable[[], Session], compiled: bool) -> None:
    ref = _sem(db_session_factory(), compiled)
    sim = _sem(db_session_factory(), compiled)
    earth, moon, emb, sun, ssb = _slots(sim, "Earth", "Moon", "EMB", "Sun", "SSB")
    g0, c0, l0 = sim.global_states.copy(), sim.coe_states.copy(), sim.local_states.copy()

    head, sib = sim.dissolve_system(emb)
    assert (head, sib) == (earth, moon)
    assert emb in sim.free_indices and "EMB" not in sim.name_to_index and not sim.active_mask[emb]
    assert np.array_equal(sim.global_states[[earth, moon]], g0[[earth, moon]])
    assert np.array_equal(sim.global_states[[sun, ssb]], g0[[sun, ssb]])
    assert sim.parent_indices[moon] == sun and sim.body_sys_map[moon] == ssb and not sim.is_head[earth]

    s = sim.form_system(earth, moon, name="EMB")
    assert s == emb                                          # the freed slot comes back first
    assert sim.is_head[earth] and sim.parent_indices[moon] == earth and sim.body_sys_map[moon] == emb
    rows = [earth, moon, emb]
    scale = np.maximum(1.0, np.abs(c0[rows]))
    assert float(np.max(np.abs(sim.coe_states[rows] - c0[rows]) / scale)) < 1e-13
    assert float(np.max(np.abs(sim.global_states - g0))) < 1e-7
    assert np.array_equal(sim.coe_states[[sun, ssb]], c0[[sun, ssb]])
    assert np.array_equal(sim.local_states[[sun, ssb]], l0[[sun, ssb]])

    for _ in range(30):
        ref.step(DAY)
        sim.step(DAY)
    d = np.linalg.norm(sim.global_states[rows, :3] - ref.global_states[rows, :3], axis=1)
    assert float(np.max(d)) < 1e-5


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("operation", ["dissolve", "form"])
def test_a_zero_step_after_a_restructure_moves_nothing(
        db_session_factory: Callable[[], Session], compiled: bool, operation: str) -> None:
    sim = _sem(db_session_factory(), compiled)
    earth, moon, emb = _slots(sim, "Earth", "Moon", "EMB")
    if operation == "form":
        sim.dissolve_system(emb)
        sim.step(5 * DAY)                                    # the pair drifts apart heliocentrically
        sim.form_system(earth, moon)
    else:
        sim.step(5 * DAY)
        sim.dissolve_system(emb)
    g = sim.global_states.copy()
    sim.step(0.0)
    assert float(np.max(np.abs(sim.global_states - g))) < 1e-12 * 1.5e8


def test_forming_touches_only_the_pair(db_session_factory: Callable[[], Session]) -> None:
    sim = _sem(db_session_factory())
    earth, moon, emb, sun, ssb = _slots(sim, "Earth", "Moon", "EMB", "Sun", "SSB")
    sim.dissolve_system(emb)
    sim.step(DAY)
    g, c, l = sim.global_states.copy(), sim.coe_states.copy(), sim.local_states.copy()
    s = sim.form_system(moon, earth)                         # order does not pick the head; mass does
    assert sim.is_head[earth] and not sim.is_head[moon] and sim.name_to_index["Earth+Moon"] == s
    assert np.array_equal(sim.global_states[[earth, moon]], g[[earth, moon]])
    other = [sun, ssb]
    for a, b in ((sim.global_states, g), (sim.coe_states, c), (sim.local_states, l)):
        assert np.array_equal(a[other], b[other])
    m = sim.mu_array
    expected = (m[earth] * g[earth] + m[moon] * g[moon]) / (m[earth] + m[moon])
    assert np.array_equal(sim.global_states[s], expected)
    assert m[s] == m[earth] + m[moon] and sim.is_system[s] and sim.parent_indices[s] == sun


def test_history_follows_the_restructure(db_session_factory: Callable[[], Session]) -> None:
    sim = _sem(db_session_factory())
    emb = sim.name_to_index["EMB"]
    sim.step(DAY)
    sim.dissolve_system(emb)
    sim.step(DAY)
    counts = sim.history.groupby("body").size().to_dict()
    assert counts == {"EMB": 1, "Earth": 2, "Moon": 2, "Sun": 2, "SSB": 2}


def test_formed_pair_against_nbody_truth(db_session_factory: Callable[[], Session]) -> None:
    """The comparison in the module docstring: forming the pair is what keeps the Moon with Earth."""
    times = np.arange(31, dtype=np.float64) * DAY
    errors = {}
    for formed in (True, False):
        sim = _sem(db_session_factory())
        sim.record_history = False
        if not formed:
            sim.dissolve_system(sim.name_to_index["EMB"])
        truth = reference_for(sim, times)
        moon = sim.name_to_index["Moon"]
        for _ in range(30):
            sim.step(DAY)
        errors[formed] = float(np.linalg.norm(sim.global_states[moon, :3] - truth.position_of("Moon")[-1]))
    assert 1e4 < errors[True] < 1e5
    assert errors[False] > 1e6


def test_refusals(db_session_factory: Callable[[], Session]) -> None:
    sim = _sem(db_session_factory(), leo_satellite=True)
    earth, moon, emb, sun, sat = _slots(sim, "Earth", "Moon", "EMB", "Sun", "LEO-SAT")
    with pytest.raises(ValueError, match="3 members"):
        sim.dissolve_system(emb)                             # Earth, Moon and the satellite
    with pytest.raises(ValueError, match="root"):
        sim.dissolve_system(sim.name_to_index["SSB"])
    with pytest.raises(ValueError, match="not an active system"):
        sim.dissolve_system(earth)
    with pytest.raises(ValueError, match="system or a head"):
        sim.form_system(earth, moon)
    with pytest.raises(ValueError, match="distinct"):
        sim.form_system(moon, moon)
    with pytest.raises(ValueError, match="system or a head"):
        sim.form_system(emb, sun)

    sim = _sem(db_session_factory(), moon_mu=0.0, leo_satellite=True)
    moon, sat = _slots(sim, "Moon", "LEO-SAT")
    with pytest.raises(ValueError, match="both members are massless"):
        sim.form_system(moon, sat)
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    with pytest.raises(ValueError, match="not Keplerian"):
        sim.form_system(moon, sat)

    sim = _sem(db_session_factory())
    earth, moon, emb = _slots(sim, "Earth", "Moon", "EMB")
    sim.dissolve_system(emb)
    with pytest.raises(ValueError, match="already exists"):
        sim.form_system(earth, moon, name="Sun")
    while sim.free_indices:
        sim.free_indices.pop()
    with pytest.raises(ValueError, match="arena is full"):
        sim.form_system(earth, moon)


def test_a_dependant_blocks_both(db_session_factory: Callable[[], Session]) -> None:
    """Only the refusals are checked, so the satellite is rewired by hand (its state is not re-derived)."""
    sim = _sem(db_session_factory(), leo_satellite=True)
    earth, moon, emb, sun, ssb, sat = _slots(sim, "Earth", "Moon", "EMB", "Sun", "SSB", "LEO-SAT")
    sim.body_sys_map[sat] = ssb              # out of the Earth-Moon bubble, still parented to Earth
    with pytest.raises(ValueError, match="depend on 'EMB'"):
        sim.dissolve_system(emb)
    sim.parent_indices[sat] = sun
    sim.dissolve_system(emb)
    sim.parent_indices[sat] = moon
    with pytest.raises(ValueError, match="depend on a member"):
        sim.form_system(earth, moon)
