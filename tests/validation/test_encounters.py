"""
Encounter events (`hierarchy.encounter_events`, `Simulation.watch_encounter`): a pair is formed when its
separation falls through `form_km` and dissolved when it rises through `dissolve_km`.

Scenario: `scenarios.asteroid_encounter` defaults, Ceres- and Vesta-like masses (mu 62.6 and 17.3
km^3/s^2) at 2.77 AU, 2,000 km nominal miss at 1 km/s, closest approach at day 5; truth is DOP853 N-body
(`reference_for`), 10 days, 1 h steps. Truth's closest approach is 1,950 km at day 5.00.

Expected, derived before measuring:
- **Unpaired, the mutual deflection is missing.** Impulse approximation: the relative velocity turns
  by `2 mu / (b v)` = 2 x 79.9 / (2000 x 1) = 0.080 km/s, shared by mass, so B (the lighter) loses
  `mu_A / M` of it, 0.062 km/s, and A 0.017 km/s. Over the 4.3e5 s after closest approach that is
  ~2.7e4 km for B and ~7.4e3 km for A. *Measured:* 2.71e4 and 7.47e3 km.
- **Paired between the radii, the error is what remains outside them**: the mutual pull neglected
  beyond `form_km` (`dv ~ mu / (r v)`: 8e-4 km/s at 1e5 km, ~2e2 km over the remaining days), plus the
  solar tide neglected inside it. *Measured* at 1e5 / 2e5 km: 270 km for B, 100x better than unpaired.
  The error keeps falling with the radius in this scenario (5e3 km: 3.4e3; 2e5 km: 118; paired from
  the start: 80), because the two terms balance near the Hill scale `R (m / M)^(1/3)` ~ 3e5 km, not the
  Laplace sphere of influence (7.7e4 km). That is phase 3's sweep; here only the mechanism is tested.
- **The switches land on the radii.** The form fires within `(1 + nudges) tol_s` of the crossing, so
  the separation there is `form_km` to ~1e-6 km at 1 km/s. The pre-formation trajectory is analytic,
  so an unpaired twin advanced to the same epoch reproduces it to round-off. Bound 1e-3 km.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.hierarchy import EncounterPolicy, encounter_events, paired_system, separation_km
from orbital_engine.reference import reference_for
from orbital_engine.simulator import Simulation

DAY = 86400.0
DT = 3600.0
N_STEPS = 240
POLICY = EncounterPolicy(form_km=1.0e5, dissolve_km=2.0e5)


def _build(session: Session, compiled: bool = True) -> Tuple[Simulation, int, int]:
    sim = scenarios.asteroid_encounter(session)
    sim.use_compiled_kernel = compiled
    sim.record_history = False
    return sim, sim.name_to_index[scenarios.ASTEROID_A], sim.name_to_index[scenarios.ASTEROID_B]


def _run(sim: Simulation, steps: int = N_STEPS) -> None:
    for _ in range(steps):
        sim.step(DT)


def test_policy_needs_hysteresis() -> None:
    with pytest.raises(ValueError, match="hysteresis"):
        EncounterPolicy(form_km=1e5, dissolve_km=1e5)
    with pytest.raises(ValueError, match="hysteresis"):
        EncounterPolicy(form_km=-1.0, dissolve_km=1e5)
    with pytest.raises(ValueError, match="tol_s"):
        EncounterPolicy(form_km=1e5, dissolve_km=2e5, tol_s=0.0)


@pytest.mark.parametrize("compiled", [True, False])
def test_forms_once_and_dissolves_once_around_closest_approach(
        db_session: Session, compiled: bool) -> None:
    sim, a, b = _build(db_session, compiled)
    sim.watch_encounter(a, b, POLICY)
    _run(sim)
    kinds = [(c.kind, c.bodies) for c in sim.hierarchy_changes]
    assert kinds == [("form", (a, b)), ("dissolve", (a, b))]
    t_form, t_dissolve = (c.t for c in sim.hierarchy_changes)
    assert t_form < 5.0 * DAY < t_dissolve
    assert paired_system(sim, a, b) is None and sim.body_sys_map[a] == sim.body_sys_map[b]


def test_the_switches_land_on_the_radii(db_session_factory: Callable[[], Session]) -> None:
    sim, a, b = _build(db_session_factory())
    sim.watch_encounter(a, b, POLICY)
    _run(sim)
    for change, radius in zip(sim.hierarchy_changes, (POLICY.form_km, POLICY.dissolve_km)):
        twin, ta, tb = _build(db_session_factory())
        if change.kind == "dissolve":
            twin.watch_encounter(ta, tb, POLICY)   # the pair must be formed on the way, as in `sim`
        n = int(change.t // DT)
        _run(twin, n)
        twin.step(change.t - n * DT)
        assert abs(separation_km(twin, ta, tb) - radius) < 1e-3


def test_hysteresis_survives_a_grazing_pair(db_session: Session) -> None:
    """Formed just outside closest approach: the separation dips below the form radius, rises back
    through it (no event) and only the dissolve radius ends the pairing. Before pairing the engine flies
    the unpaired conics, whose closest approach is the nominal 2,000 km (the focused 1,950 km is the
    truth's), so the form radius sits just above 2,000 km."""
    sim, a, b = _build(db_session)
    sim.watch_encounter(a, b, EncounterPolicy(form_km=2010.0, dissolve_km=3000.0))
    _run(sim)
    assert [c.kind for c in sim.hierarchy_changes] == ["form", "dissolve"]
    assert sim.hierarchy_changes[1].t - sim.hierarchy_changes[0].t < 0.1 * DAY


def test_pairing_is_the_mutual_deflection(db_session_factory: Callable[[], Session]) -> None:
    """The comparison in the module docstring."""
    times = np.arange(N_STEPS + 1, dtype=np.float64) * DT
    errors = {}
    for policy in (None, POLICY):
        sim, a, b = _build(db_session_factory())
        truth = reference_for(sim, times)
        if policy is not None:
            sim.watch_encounter(a, b, policy)
        _run(sim)
        errors[policy is not None] = [
            float(np.linalg.norm(sim.global_states[k, :3] - truth.position_of(name)[-1]))
            for k, name in ((a, scenarios.ASTEROID_A), (b, scenarios.ASTEROID_B))]
    assert 5e3 < errors[False][0] < 1e4 and 2e4 < errors[False][1] < 3.5e4
    assert errors[True][1] < errors[False][1] / 50.0 and errors[True][0] < errors[False][0] / 50.0


def test_a_pair_starting_inside_forms_at_once(db_session: Session) -> None:
    sim, a, b = _build(db_session)
    assert separation_km(sim, a, b) < 5e5
    sim.watch_encounter(a, b, EncounterPolicy(form_km=5e5, dissolve_km=1e6))
    assert [(c.kind, c.t) for c in sim.hierarchy_changes] == [("form", 0.0)]
    assert paired_system(sim, a, b) is not None


def test_an_ineligible_pair_is_skipped_and_logged(db_session: Session) -> None:
    sim, a, b = _build(db_session)
    sim.watch_encounter(a, b, POLICY)
    sim.free_indices.clear()                       # no slot for the barycentre
    _run(sim)
    skips = [c for c in sim.hierarchy_changes if c.kind == "skip"]
    assert len(skips) == 1 and "arena is full" in skips[0].note and skips[0].system == -1
    assert len(sim.hierarchy_changes) == 1         # never formed, so nothing to dissolve
    assert paired_system(sim, a, b) is None


def test_events_are_reusable_data(db_session_factory: Callable[[], Session]) -> None:
    """The actions read the pairing from the arena, so one pair of events serves two simulations."""
    s1, a, b = _build(db_session_factory())
    s2, a2, b2 = _build(db_session_factory())
    assert (a, b) == (a2, b2)
    form, dissolve = encounter_events(a, b, POLICY)
    for sim in (s1, s2):
        sim.add_event(form)
        sim.add_event(dissolve)
        _run(sim)
        assert [c.kind for c in sim.hierarchy_changes] == ["form", "dissolve"]
    assert np.array_equal(s1.global_states, s2.global_states)


def test_compiled_and_numpy_paths_agree(db_session_factory: Callable[[], Session]) -> None:
    out = []
    for compiled in (True, False):
        sim, a, b = _build(db_session_factory(), compiled)
        sim.watch_encounter(a, b, POLICY)
        _run(sim)
        out.append(sim.global_states[[a, b], :3].copy())
    assert float(np.max(np.abs(out[0] - out[1]))) < 1e-4
