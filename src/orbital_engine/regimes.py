"""
Regime switching: a massless body's whole model - parent, propagator and force models - chosen by
where it is, and switched by event.

**Why.** Phase 4 (`hierarchy.patch_events`) hands a probe between primaries but keeps it Keplerian on
both sides, so each side neglects the other primary entirely. On a strong flyby that costs ~2e5 km:
the arrival geometry is amplified by the bend. The fix is to change the *model*, not only the frame,
at the boundary: Kepler about the Sun far from the planet, where nothing else matters much; Cowell
about the planet with the Sun as a third body near it, where the perturbation is strong and the
integration is well conditioned because the dominant pull is the central term. This is the
central-body switch flight-dynamics tools make at a sphere of influence. Here it is data: a
`Regime` per side, a radius policy between them, and each combination a sweep configuration whose
cost and error are measured against N-body truth.

**What a regime is.** `Regime(parent, propagator, force_models)`: the body the regime is centred on,
`KEPLERIAN` or `COWELL`, and the force models (`sweep.ForceModelSpec`, coefficients by value or by
body name). `RegimeSwitch(body, planet, policy, inside, outside)` applies `inside` while the body is
within `policy.form_km` of `planet` (in km or the planet's live Hill or Laplace radius, as for
`hierarchy.EncounterPolicy`) and `outside` once it is beyond `policy.dissolve_km`.

**Applying a regime** (`apply_regime`) is, in order: clear the body's force models (their
coefficients belong to the old centre - J2 of the wrong planet, a perturber that is now the parent);
`hierarchy.reparent` to the new centre (a frame change; nothing moves); set the propagator; enable the
new force models; re-derive the body's elements, because a Cowell body's elements are stale by design
and a Keplerian one propagates from them. Each application is logged as a `"regime"` entry in
`Simulation.hierarchy_changes`.

**What it does not do.** There is one clock: a regime cannot change the step, so the far regime runs
at the near regime's step. What a cheap far regime saves is cost per step, not steps; adaptive or
per-regime stepping is the next lever. `set_cowell_integrator` is arena-wide, so a regime cannot pick
its own integrator either. Massive bodies cannot be Cowell (`Simulation.set_propagator`), so regimes
are for massless bodies, as patched conics are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from . import events
from .custom_types import PropagatorType
from .hierarchy import EncounterPolicy, HierarchyChange, reparent, _patch_targets

if TYPE_CHECKING:
    from .simulator import Simulation
    from .sweep import ForceModelSpec

#: The propagators a regime can use: the two a massless body can be moved between at runtime.
REGIME_PROPAGATORS = (PropagatorType.KEPLERIAN, PropagatorType.COWELL)


@dataclass(frozen=True)
class Regime:
    """One side of a switch: what the body orbits, how it is propagated, and what forces act on it.
    `force_models` only apply to `COWELL` (a Keplerian body ignores accelerations), so a Keplerian
    regime with force models is refused rather than silently ignoring them."""

    parent: str
    propagator: PropagatorType = PropagatorType.KEPLERIAN
    force_models: Sequence["ForceModelSpec"] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.propagator not in REGIME_PROPAGATORS:
            raise ValueError(f"Regime propagator {self.propagator!r}: use KEPLERIAN or COWELL.")
        if self.propagator == PropagatorType.KEPLERIAN and self.force_models:
            raise ValueError("a Keplerian regime ignores force models; give them to a COWELL regime.")


@dataclass(frozen=True)
class RegimeSwitch:
    """`body` (by name) is in `inside` within `policy.form_km` of `planet` and in `outside` beyond
    `policy.dissolve_km`: what `sweep.ModelConfig.regimes` holds."""

    body: str
    planet: str
    policy: EncounterPolicy
    inside: Regime
    outside: Regime


def _clear_force_models(sim: "Simulation", body: int) -> None:
    sim.force_model_mask[body] = np.uint64(0)
    for params in sim.force_model_params.values():
        params[body] = 0.0
    sim.resolve_force_models()


def apply_regime(sim: "Simulation", body: int, regime: Regime) -> None:
    """Put `body` in `regime` now. See the module docstring for the order and why."""
    body = int(body)
    if regime.parent not in sim.name_to_index:
        raise KeyError(f"regime parent {regime.parent!r} is not in this simulation")
    parent = sim.name_to_index[regime.parent]
    _clear_force_models(sim, body)
    if int(sim.parent_indices[body]) != parent:
        reparent(sim, body, parent)
    rows = np.array([body], dtype=np.int64)
    sim.set_propagator(rows, regime.propagator)
    for fm in regime.force_models:
        coefficients = dict(fm.coefficients)
        for key, name in fm.body_coefficients.items():
            if name not in sim.name_to_index:
                raise KeyError(f"regime force model {fm.name!r}: {key}={name!r} is not in this simulation")
            coefficients[key] = float(sim.name_to_index[name])
        sim.enable_force_model(fm.name, [body], **coefficients)
    sim._rehydrate_coes(rows=rows)
    names = {v: k for k, v in sim.name_to_index.items()}
    sim._hierarchy_log.append(HierarchyChange(
        float(sim.t), "regime", names[body], (body, parent), int(sim.body_sys_map[body]),
        f"{regime.propagator.name} about {regime.parent}"))


def regime_events(sim: "Simulation", switch: RegimeSwitch) -> Tuple[events.Event, events.Event]:
    """The two events that run `switch`: inward through `form_km` applies `inside`, outward through
    `dissolve_km` applies `outside`. Stateless like the other hierarchy events: each action checks
    whether the body is already in its regime (by parent and propagator) before applying it."""
    body, planet = sim.name_to_index[switch.body], sim.name_to_index[switch.planet]
    entity, _, _ = _patch_targets(sim, planet, "body")
    policy = switch.policy
    bodies = np.array([body], dtype=np.int64)

    def dist(s: "Simulation") -> float:
        return float(np.linalg.norm(s.global_states[body, :3] - s.global_states[planet, :3]))

    def g_in(s: "Simulation", _b: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([dist(s) - policy.entity_radii_km(s, entity)[0]], dtype=np.float64)

    def g_out(s: "Simulation", _b: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([dist(s) - policy.entity_radii_km(s, entity)[1]], dtype=np.float64)

    def to(regime: Regime) -> events.EventAction:
        def act(s: "Simulation", fired: NDArray[np.int64], t: float) -> None:
            if not _in_regime(s, body, regime):
                apply_regime(s, body, regime)
        return act

    return (events.Event(name="regime: inside", function=g_in, bodies=bodies, direction=-1,
                         tol_s=policy.tol_s, action=to(switch.inside)),
            events.Event(name="regime: outside", function=g_out, bodies=bodies, direction=1,
                         tol_s=policy.tol_s, action=to(switch.outside)))


def _in_regime(sim: "Simulation", body: int, regime: Regime) -> bool:
    return (int(sim.parent_indices[body]) == sim.name_to_index[regime.parent]
            and sim.propagator_type[body] == np.uint8(regime.propagator))


def watch_regimes(sim: "Simulation", switch: RegimeSwitch) -> Tuple[events.Event, events.Event]:
    """
    Register `regime_events(sim, switch)` and put the body in the regime its current position calls
    for: `inside` if it starts within `form_km`, else `outside`. Applying one at the start, rather
    than leaving whatever the body was built with, is what makes the configuration's model fully
    defined by its regimes.
    """
    inside_event, outside_event = regime_events(sim, switch)
    sim.add_event(inside_event)
    sim.add_event(outside_event)
    body, planet = sim.name_to_index[switch.body], sim.name_to_index[switch.planet]
    entity, _, _ = _patch_targets(sim, planet, "body")
    d = float(np.linalg.norm(sim.global_states[body, :3] - sim.global_states[planet, :3]))
    apply_regime(sim, body, switch.inside if d < switch.policy.entity_radii_km(sim, entity)[0]
                 else switch.outside)
    return inside_event, outside_event
