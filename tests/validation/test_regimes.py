"""
Regime switching (`regimes.py`): a massless body's centre, propagator and force models switched by
event, and each combination a sweep configuration.

Scenario: `scenarios.planet_flyby` with periapsis at day 10 of 20 (10,000 km, v_inf 3 km/s, 109 deg
turn); the switch at 0.76 / 0.80 Earth Hill radii; truth is DOP853 N-body.

Expected, derived before measuring:
- **Applying a regime moves nothing**, and a Cowell -> Kepler switch must re-derive the elements: a
  Cowell body's are stale by design, so without that the next Keplerian step would jump to wherever the
  stale elements put it. Bound: a zero step after the switch moves the probe < 1e-6 km relative to its
  heliocentric distance (the elements round trip).
- **Which configuration wins, and why** (*measured*, dt 600 s; at 120 s in brackets):
  Kepler never handed over 4.3e6 km; patched Kepler 2.0e5 km (phase 4); Kepler far / Cowell (Earth +
  Sun third body) near 1.1e5 km (1.0e5), limited by the Earth pull neglected on the far side, which the
  flyby amplifies; Cowell about the Sun throughout, with Earth as a third body, 1.1e6 km (1.6e5): Earth
  is a stiff perturber that `third_body` freezes within each step; Cowell on both sides with the centre
  switched 1.8e4 km (3.3e3). The switch of *centre* is what matters: the same physics about one centre
  is 47-60x worse at every step from 30 s to 600 s. Both Cowell configurations converge at *first*
  order (error halves with the step), set by `third_body`'s perturber frozen within each step, not by
  RK4; a fourth-order perturber (`ephemeris_third_body`) is the next lever.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType as PT
from orbital_engine.hierarchy import EncounterPolicy
from orbital_engine.regimes import Regime, RegimeSwitch, apply_regime
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec as F, ModelConfig, run_sweep

DAY = 86400.0
P, E, S = scenarios.FLYBY_CRAFT, scenarios.FLYBY_PLANET, "Sun"
POLICY = EncounterPolicy(0.76, 0.80, unit="hill")
KEP_SUN, KEP_EARTH = Regime(S), Regime(E)
COW_SUN = Regime(S, PT.COWELL, (F("point_mass_gravity"), F("third_body", body_coefficients={"perturber": E})))
COW_EARTH = Regime(E, PT.COWELL, (F("point_mass_gravity"), F("third_body", body_coefficients={"perturber": S})))


def _flyby(session: Session) -> Tuple[Simulation, int, int, int]:
    sim = scenarios.planet_flyby(session, t_ca_s=10 * DAY)
    sim.record_history = False
    return sim, sim.name_to_index[P], sim.name_to_index[E], sim.name_to_index[S]


def test_a_regime_is_validated() -> None:
    with pytest.raises(ValueError, match="ignores force models"):
        Regime(S, PT.KEPLERIAN, (F("point_mass_gravity"),))
    with pytest.raises(ValueError, match="KEPLERIAN or COWELL"):
        Regime(S, PT.SECULAR_J2)


def test_applying_a_regime_moves_nothing_and_rewires_everything(db_session: Session) -> None:
    sim, probe, earth, sun = _flyby(db_session)
    g = sim.global_states.copy()
    apply_regime(sim, probe, COW_EARTH)
    assert np.array_equal(sim.global_states, g)
    assert sim.parent_indices[probe] == earth and sim.propagator_type[probe] == np.uint8(PT.COWELL)
    assert sim.force_model_params["third_body"][probe, 0] == float(sun)
    apply_regime(sim, probe, KEP_SUN)                      # clears the third body before reparenting
    assert sim.parent_indices[probe] == sun and int(sim.force_model_mask[probe]) == 0
    with pytest.raises(KeyError, match="Mars"):
        apply_regime(sim, probe, Regime("Mars"))


@pytest.mark.parametrize("rehydrate", [True, False])
def test_cowell_to_kepler_rederives_the_elements(db_session: Session, monkeypatch: pytest.MonkeyPatch,
                                                  rehydrate: bool) -> None:
    """The stale-elements trap, with a negative control: skip the re-derivation and the next Keplerian
    step teleports the probe to where its pre-Cowell elements say it is."""
    sim, probe, earth, sun = _flyby(db_session)
    apply_regime(sim, probe, COW_SUN)
    for _ in range(24):
        sim.step(3600.0)                                   # a day of Cowell: the elements go stale
    if not rehydrate:
        monkeypatch.setattr(sim, "_rehydrate_coes", lambda rows=None: None)
    apply_regime(sim, probe, KEP_SUN)
    g = sim.global_states[probe, :3].copy()
    sim.step(0.0)
    moved = float(np.linalg.norm(sim.global_states[probe, :3] - g)) / float(np.linalg.norm(g))
    assert (moved < 1e-12) if rehydrate else (moved > 1e-6)


def test_the_switch_runs_both_ways_by_event(db_session: Session) -> None:
    sim, probe, earth, sun = _flyby(db_session)
    sim.watch_regimes(RegimeSwitch(P, E, POLICY, COW_EARTH, KEP_SUN))
    for _ in range(int(20 * DAY / 600.0)):
        sim.step(600.0)
    changes = [(c.kind, c.bodies[1], c.note) for c in sim.hierarchy_changes if c.kind == "regime"]
    assert changes == [("regime", sun, "KEPLERIAN about Sun"), ("regime", earth, "COWELL about Earth"),
                       ("regime", sun, "KEPLERIAN about Sun")]
    times = [c.t for c in sim.hierarchy_changes if c.kind == "regime"]
    assert times[0] == 0.0 and times[1] < 10 * DAY < times[2]
    assert sim.parent_indices[probe] == sun and int(sim.force_model_mask[probe]) == 0


def _configs(dt: float) -> List[ModelConfig]:
    def cfg(name: str, inside: Regime, outside: Regime, policy: EncounterPolicy = POLICY) -> ModelConfig:
        return ModelConfig(name, PT.KEPLERIAN, dt, bodies=[P],
                           regimes=(RegimeSwitch(P, E, policy, inside, outside),))
    return [ModelConfig("never", PT.KEPLERIAN, dt, bodies=[P]),
            cfg("patched", KEP_EARTH, KEP_SUN),
            cfg("kepler far", COW_EARTH, KEP_SUN),
            cfg("switched", COW_EARTH, COW_SUN),
            cfg("sun only", COW_SUN, COW_SUN)]


def test_switching_the_centre_is_what_wins(db_session_factory: Callable[[], Session]) -> None:
    res = {r.config_name: r.error.max_km for r in run_sweep(
        lambda: scenarios.planet_flyby(db_session_factory(), t_ca_s=10 * DAY), _configs(600.0),
        20 * DAY, timing_batches=1, timing_warmup=0)}
    assert 3e6 < res["never"] < 6e6
    assert 1.5e5 < res["patched"] < 2.5e5
    assert res["switched"] < res["kepler far"] / 3.0 < res["patched"] / 3.0
    assert res["switched"] < res["sun only"] / 30.0
