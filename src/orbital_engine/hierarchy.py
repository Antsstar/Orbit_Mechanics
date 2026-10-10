"""
Temporary systems: forming and dissolving a two-body bubble at runtime.

**What a temporary system is.** The hierarchy built from the database already holds bubbles of the
Earth-Moon shape: a barycentre slot (`is_system`, carrying the members' summed `mu`), a head (the
heavier member, zeroed elements, placed each step by the reflex kick) and a sibling orbiting the
head. That representation is exact two-body motion for a pair, and the barycentre itself is an
ordinary sibling of the outer bubble on a Keplerian orbit about the outer head. A temporary system
is that same shape created while the simulation runs - two asteroids passing close enough that
their mutual attraction matters more than the Sun's differential pull across them - and removed
when they separate. Inside the pair the mutual gravity is exact; what the model neglects is the
outer primary's tide across the pair, which is the error a formation policy trades.

The mechanism is not new physics. N-body codes have long replaced a close pair with its centre of
mass in the outer integration and treated the relative motion separately (Aarseth's KS
regularisation, *Gravitational N-Body Simulations*, 2003, ch. 5), and patched conics switch a
craft's primary at the sphere of influence. What is particular here is that both levels stay
analytic (Kepler inside, Kepler outside, joined by the reflex kick), and that *when* to form and
dissolve is a policy measured against independent N-body truth (`reference.py`).

**A frame change, not physics.** Forming or dissolving moves no body: every member's global state is
kept and only the frame it is expressed in changes. The float work is therefore *local* - the new
barycentre's state from its members, then `Simulation._rehydrate_coes(rows=...)` on the changed
rows - and every other row of the arena is left bit-identical. Re-running the build's
`_recalculate_all_barycenters` / `_rehydrate_coes` over the whole arena would instead perturb every
unrelated body at round-off. The integer structure (`sys_head_map`, the topological tiers, the
dispatch caches) is re-derived in full, which is exact.

Continuity holds at both levels. Inside, the head's reflex kick `-m_b r / (m_a + m_b)` is exactly
its offset from the new barycentre. Outside, the outer head's kick sums `m_i r_i` over its siblings,
and `M_S r_S = m_a r_a + m_b r_b`, so replacing the pair by its barycentre leaves the outer kick
unchanged.

**Restrictions (phase 1), each refused rather than approximated:**

- Exactly two members, siblings of the same bubble with the same parent; neither a head nor a
  system. Three or more siblings in one bubble is the reflex model's unmeasured approximation
  (siblings do not attract each other directly), so it is not created implicitly.
- Both members Keplerian. A massive body cannot be Cowell anyway (`set_propagator`).
- Nothing else depends on either member (no body parented to it or in its bubble), and nothing
  depends on a dissolved system except its two members.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from . import events
from .custom_types import PropagatorType

if TYPE_CHECKING:
    from .simulator import Simulation


@dataclass(frozen=True)
class HierarchyChange:
    """One entry of `Simulation.hierarchy_changes`: what happened, when, to which bodies.

    `kind` is `"form"`, `"dissolve"` or `"skip"` (an encounter that crossed its formation radius but
    whose pair was not eligible; `note` says why). `system` is the barycentre slot, `-1` for a skip."""

    t: float
    kind: str
    name: str
    bodies: Tuple[int, int]
    system: int
    note: str = ""


def _dependants(sim: "Simulation", slots: NDArray[np.int64], allowed: NDArray[np.int64]) -> NDArray[np.int64]:
    """Active slots outside `allowed` whose parent or bubble is one of `slots`."""
    active = np.flatnonzero(sim.active_mask)
    hit = np.isin(sim.parent_indices[active], slots) | np.isin(sim.body_sys_map[active], slots)
    out: NDArray[np.int64] = active[hit & ~np.isin(active, allowed)].astype(np.int64)
    return out


def _restructure(sim: "Simulation") -> None:
    """Re-derive the integer structure after a change to the maps or masks. Exact: no float work."""
    sim._build_sys_head_map()
    sim._topological_sort(sim.body_sys_map)
    sim._refresh_active_indices()
    sim.resolve_force_models()


def _head_first(sim: "Simulation", a: int, b: int) -> Tuple[int, int]:
    """The heavier body first; a tie goes to the lower slot, as `_resolve_circular` elects."""
    if sim.mu_array[a] > sim.mu_array[b] or (sim.mu_array[a] == sim.mu_array[b] and a < b):
        return a, b
    return b, a


def form_refusal(sim: "Simulation", a: int, b: int, name: Optional[str] = None) -> Optional[str]:
    """Why `form_system(a, b, name=name)` would refuse, or `None` if it would not. A pure read."""
    a, b = int(a), int(b)
    if a == b:
        return "form_system needs two distinct bodies."
    names = {v: k for k, v in sim.name_to_index.items()}
    for k in (a, b):
        if not (0 <= k < sim.max_capacity) or not sim.active_mask[k]:
            return f"slot {k} is not an active body."
        if sim.is_system[k] or sim.is_head[k]:
            return f"{names[k]!r} is a system or a head; only plain siblings can pair."
        if sim.propagator_type[k] != np.uint8(PropagatorType.KEPLERIAN):
            return f"{names[k]!r} is not Keplerian; a temporary system's members must be."
    if sim.body_sys_map[a] != sim.body_sys_map[b] or sim.parent_indices[a] != sim.parent_indices[b]:
        return f"{names[a]!r} and {names[b]!r} are not siblings of one bubble with one parent."
    if sim.mu_array[a] + sim.mu_array[b] <= 0.0:
        return "a system needs mass: both members are massless."
    pair = np.array([a, b], dtype=np.int64)
    deps = _dependants(sim, pair, pair)
    if deps.size:
        return f"{[names[int(k)] for k in deps]} depend on a member; not supported yet."
    if not sim.free_indices:
        return f"the arena is full (capacity {sim.max_capacity})."
    head, sib = _head_first(sim, a, b)
    if (name if name is not None else f"{names[head]}+{names[sib]}") in sim.name_to_index:
        return f"a body named {name if name is not None else names[head] + '+' + names[sib]!r} already exists."
    return None


def form_system(sim: "Simulation", a: int, b: int, *, name: Optional[str] = None) -> int:
    """
    Make `a` and `b` a two-body system inside their shared bubble and return its barycentre's slot.

    The heavier member becomes the head (ties go to the lower slot, as `_resolve_circular` elects);
    the other orbits it. The barycentre takes the pair's place in the outer bubble, on the orbit
    about the shared parent that the pair's centre of mass is actually on. `name` defaults to
    `"<a>+<b>"`, heavier first. Raises `ValueError` with `form_refusal`'s reason, before changing
    anything, if the pair is not eligible.
    """
    a, b = int(a), int(b)
    refusal = form_refusal(sim, a, b, name)
    if refusal is not None:
        raise ValueError(refusal)
    names = {v: k for k, v in sim.name_to_index.items()}
    head, sib = _head_first(sim, a, b)
    name = name if name is not None else f"{names[head]}+{names[sib]}"

    outer, parent = int(sim.body_sys_map[a]), int(sim.parent_indices[a])
    s = sim.free_indices.pop()
    sim.name_to_index[name] = s
    mu = sim.mu_array[head] + sim.mu_array[sib]
    sim.active_mask[s] = True
    sim.is_system[s] = True
    sim.is_head[s] = False
    sim.mu_array[s] = mu
    sim.propagator_type[s] = np.uint8(PropagatorType.KEPLERIAN)
    sim.parent_indices[s] = parent
    sim.body_sys_map[s] = outer
    sim.global_states[s] = (sim.mu_array[head] * sim.global_states[head]
                            + sim.mu_array[sib] * sim.global_states[sib]) / mu

    sim.is_head[head] = True
    sim.body_sys_map[head] = s
    sim.body_sys_map[sib] = s
    sim.parent_indices[sib] = head          # the head keeps the outer parent, as a built head does

    _restructure(sim)
    sim._rehydrate_coes(rows=np.array([s, head, sib], dtype=np.int64))
    sim._hierarchy_log.append(HierarchyChange(float(sim.t), "form", name, (head, sib), s))
    return s


def dissolve_system(sim: "Simulation", system: int) -> Tuple[int, int]:
    """
    Return a two-member system's members to the outer bubble and free its barycentre slot.

    The inverse of `form_system`, and it accepts a system built from the database as well (the
    Earth-Moon system, for instance). Each member gets the barycentre's parent and bubble, and its
    own Keplerian orbit about that parent from its current state. Returns `(head, sibling)`.
    """
    s = int(system)
    names = {v: k for k, v in sim.name_to_index.items()}
    if not (0 <= s < sim.max_capacity) or not sim.active_mask[s] or not sim.is_system[s]:
        raise ValueError(f"slot {s} is not an active system.")
    if sim.body_sys_map[s] == s:
        raise ValueError(f"{names[s]!r} is a root; there is no outer bubble to return its members to.")
    members = np.flatnonzero(sim.active_mask & (sim.body_sys_map == s)).astype(np.int64)
    members = members[members != s]
    if members.size != 2:
        raise ValueError(f"{names[s]!r} has {members.size} members; only a pair can be dissolved.")
    if bool(sim.is_system[members].any()):
        raise ValueError(f"{names[s]!r} has a nested system as a member; not supported yet.")
    deps = _dependants(sim, np.append(members, s), members)
    if deps.size:
        raise ValueError(f"{[names[int(k)] for k in deps]} depend on {names[s]!r}; not supported yet.")
    head = int(members[sim.is_head[members]][0]) if bool(sim.is_head[members].any()) else int(members[0])
    sib = int(members[members != head][0])

    outer, parent = int(sim.body_sys_map[s]), int(sim.parent_indices[s])
    for k in (head, sib):
        sim.is_head[k] = False
        sim.body_sys_map[k] = outer
        sim.parent_indices[k] = parent

    del sim.name_to_index[names[s]]
    _release_slot(sim, s)
    _restructure(sim)
    sim._rehydrate_coes(rows=np.array([head, sib], dtype=np.int64))
    sim._hierarchy_log.append(HierarchyChange(float(sim.t), "dissolve", names[s], (head, sib), s))
    return head, sib


def _release_slot(sim: "Simulation", s: int) -> None:
    """Clear every per-slot row and return the slot to the free list, so a later spawn starts clean."""
    sim.active_mask[s] = False
    sim.is_system[s] = False
    sim.is_head[s] = False
    sim.mu_array[s] = 0.0
    sim.global_states[s] = 0.0
    sim.local_states[s] = 0.0
    sim.coe_states[s] = 0.0
    sim.parent_indices[s] = -1
    sim.body_sys_map[s] = -1
    sim.sys_head_map[s] = s
    sim.propagator_type[s] = np.uint8(PropagatorType.KEPLERIAN)
    sim.force_model_mask[s] = np.uint64(0)
    for params in sim.force_model_params.values():
        params[s] = 0.0
    sim._secular_j2_rates[s] = 0.0
    sim.accel_accum[s] = 0.0
    sim.free_indices.append(s)


# --- Encounters: forming and dissolving a pair by event ----------------------------------------------

#: Units an `EncounterPolicy`'s radii can be given in.
ENCOUNTER_UNITS = ("km", "hill", "laplace")


@dataclass(frozen=True)
class EncounterPolicy:
    """
    When a pair is formed and dissolved: inside `form_km` of each other it is paired, and it stays
    paired until the separation exceeds `dissolve_km`. The gap between the two is hysteresis, and it
    must be positive: with one radius, a pair grazing it would form and dissolve on every crossing,
    each restructure costing ~1 ms and carrying the switching error. Plain data, so a policy can be a
    sweep configuration. `tol_s` is the event tolerance the crossings are located to.

    `unit="hill"` reads both radii as multiples of the pair's Hill radius (`hill_radius_km`),
    evaluated **at every event evaluation**, so the threshold follows the pair as its distance from
    the outer primary changes. That makes a policy dimensionless and transferable between scenarios;
    whether one multiple is right across masses is what `benchmarks/encounter_sweep.py` measures.
    `unit="laplace"` does the same with the Laplace sphere of influence `R (m / M)^(2/5)`, the
    textbook patched-conic boundary, so the two can be compared as configurations. The field names
    keep `_km` for the default unit.

    The same policy drives patched conics (`patch_events`): there `form_km` is where a massless body
    is handed to the planet and `dissolve_km` where it is handed back, and the radius is the planet's.
    """

    form_km: float
    dissolve_km: float
    tol_s: float = events.DEFAULT_EVENT_TOL_S
    unit: str = "km"

    def __post_init__(self) -> None:
        if not (0.0 < self.form_km < self.dissolve_km):
            raise ValueError(
                f"EncounterPolicy needs 0 < form_km < dissolve_km (got {self.form_km}, "
                f"{self.dissolve_km}): the gap is the hysteresis that stops a grazing pair flickering.")
        if not self.tol_s > 0.0:
            raise ValueError(f"EncounterPolicy tol_s={self.tol_s!r} must be positive.")
        if self.unit not in ENCOUNTER_UNITS:
            raise ValueError(f"EncounterPolicy unit={self.unit!r}: use one of {ENCOUNTER_UNITS}.")

    def radii_km(self, sim: "Simulation", a: int, b: int) -> Tuple[float, float]:
        """`(form, dissolve)` in km for the pair now: the radii themselves, or times its Hill or
        Laplace radius."""
        scale = 1.0 if self.unit == "km" else _pair_radius_km(sim, a, b, self.unit)
        return self.form_km * scale, self.dissolve_km * scale

    def entity_radii_km(self, sim: "Simulation", entity: int) -> Tuple[float, float]:
        """`(form, dissolve)` in km about one body or system (`entity`): the radii, or times the
        entity's Hill or Laplace radius about its own parent (`sphere_radius_km`)."""
        scale = 1.0 if self.unit == "km" else sphere_radius_km(sim, entity, self.unit)
        return self.form_km * scale, self.dissolve_km * scale


def separation_km(sim: "Simulation", a: int, b: int) -> float:
    """Distance between two bodies' global positions, km."""
    return float(np.linalg.norm(sim.global_states[a, :3] - sim.global_states[b, :3]))


def _scaled_radius(r: float, m: float, big_m: float, unit: str) -> float:
    if unit == "hill":
        return r * float(np.cbrt(m / (3.0 * big_m)))
    return r * float(np.exp(0.4 * np.log(m / big_m)))       # laplace: R (m / M)^(2/5)


def sphere_radius_km(sim: "Simulation", entity: int, unit: str = "hill") -> float:
    """
    The Hill (`R (m / 3M)^(1/3)`) or Laplace (`R (m / M)^(2/5)`) radius of a body or system about its
    own Keplerian parent, now: `m` the entity's `mu` (summed, for a system), `M` the parent's, `R`
    their current separation. A head is measured as its whole system (Earth as the Earth-Moon
    barycentre), since that is what orbits the outer parent.
    """
    e = int(sim.body_sys_map[entity]) if sim.is_head[entity] else int(entity)
    parent = int(sim.parent_indices[e])
    if parent == e:
        raise ValueError("a root has no parent to measure a sphere of influence against.")
    r = float(np.linalg.norm(sim.global_states[e, :3] - sim.global_states[parent, :3]))
    return _scaled_radius(r, float(sim.mu_array[e]), float(sim.mu_array[parent]), unit)


def _pair_radius_km(sim: "Simulation", a: int, b: int, unit: str) -> float:
    if unit == "hill":
        return hill_radius_km(sim, a, b)
    m = float(sim.mu_array[a] + sim.mu_array[b])
    s = int(sim.body_sys_map[a])
    paired = s == int(sim.body_sys_map[b]) and bool(sim.is_head[a] or sim.is_head[b])
    parent = int(sim.parent_indices[s]) if paired else int(sim.parent_indices[a])
    cm = (sim.mu_array[a] * sim.global_states[a, :3] + sim.mu_array[b] * sim.global_states[b, :3]) / m
    r = float(np.linalg.norm(cm - sim.global_states[parent, :3]))
    return _scaled_radius(r, m, float(sim.mu_array[parent]), unit)


def hill_radius_km(sim: "Simulation", a: int, b: int) -> float:
    """
    The Hill radius of the pair `(a, b)` about their shared parent, now: `R (m / 3 M)^(1/3)`, with
    `m` the pair's summed `mu`, `M` the parent's own `mu` and `R` the distance from the pair's centre
    of mass to the parent. It is the separation at which the pair's mutual attraction equals the
    parent's tidal pull across it (Hill's problem; Murray & Dermott, *Solar System Dynamics*, 1999;
    the form is quoted from memory, not checked against the text). Read from global states, so it is the same whether the pair is formed or not. The
    current distance is used, not a semi-major axis, so the radius breathes with an eccentric orbit.
    """
    m = float(sim.mu_array[a] + sim.mu_array[b])
    # Paired, one member heads the pair's bubble and the outer parent is the barycentre's; unpaired,
    # the two are plain siblings sharing it. O(1): this runs at every event evaluation.
    s = int(sim.body_sys_map[a])
    paired = s == int(sim.body_sys_map[b]) and bool(sim.is_head[a] or sim.is_head[b])
    parent = int(sim.parent_indices[s]) if paired else int(sim.parent_indices[a])
    cm = (sim.mu_array[a] * sim.global_states[a, :3] + sim.mu_array[b] * sim.global_states[b, :3]) / m
    r = float(np.linalg.norm(cm - sim.global_states[parent, :3]))
    return r * float(np.cbrt(m / (3.0 * float(sim.mu_array[parent]))))


def paired_system(sim: "Simulation", a: int, b: int) -> Optional[int]:
    """The system slot if `a` and `b` are exactly the two members of one system, else `None`."""
    s = int(sim.body_sys_map[a])
    if s != int(sim.body_sys_map[b]) or s in (a, b) or not sim.is_system[s]:
        return None
    members = np.flatnonzero(sim.active_mask & (sim.body_sys_map == s))
    members = members[members != s]
    return s if members.size == 2 else None


def encounter_events(a: int, b: int, policy: EncounterPolicy) -> Tuple[events.Event, events.Event]:
    """
    The two events that run `policy` for the pair `(a, b)`: separation falling through `form_km`
    forms the pair, separation rising through `dissolve_km` dissolves it.

    Both are detected on `a`'s slot (an event is per body; the partner is part of the function), and
    both actions read the pairing state from the arena rather than holding it, so the events are
    plain data and reusable across simulations, like every other `Event`. An inward crossing of
    `dissolve_km`, or an outward one of `form_km`, fires nothing: that is the hysteresis. A pair that
    is not eligible when it crosses inward (a member already paired elsewhere, say) is logged as a
    `"skip"` in `Simulation.hierarchy_changes` and left unpaired, never approximated.

    **The step must be shorter than the encounter.** Events find a sign change between the start and
    end of a step; a pair that enters and leaves `form_km` within one step crosses twice and fires
    nothing (`events.py`'s even-crossing blind spot). Keep `dt` below `form_km / v_rel`.
    """
    a, b = int(a), int(b)
    bodies = np.array([a], dtype=np.int64)

    def inside_form(sim: "Simulation", _bodies: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([separation_km(sim, a, b) - policy.radii_km(sim, a, b)[0]], dtype=np.float64)

    def inside_dissolve(sim: "Simulation", _bodies: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([separation_km(sim, a, b) - policy.radii_km(sim, a, b)[1]], dtype=np.float64)

    def form(sim: "Simulation", fired: NDArray[np.int64], t: float) -> None:
        _form_if_eligible(sim, a, b)

    def dissolve(sim: "Simulation", fired: NDArray[np.int64], t: float) -> None:
        s = paired_system(sim, a, b)
        if s is not None:
            dissolve_system(sim, s)

    return (events.Event(name="encounter: form", function=inside_form, bodies=bodies, direction=-1,
                         tol_s=policy.tol_s, action=form),
            events.Event(name="encounter: dissolve", function=inside_dissolve, bodies=bodies, direction=1,
                         tol_s=policy.tol_s, action=dissolve))


def _form_if_eligible(sim: "Simulation", a: int, b: int) -> None:
    if paired_system(sim, a, b) is not None:
        return
    refusal = form_refusal(sim, a, b)
    if refusal is None:
        form_system(sim, a, b)
    else:
        names = {v: k for k, v in sim.name_to_index.items()}
        sim._hierarchy_log.append(HierarchyChange(
            float(sim.t), "skip", f"{names[a]}+{names[b]}", (a, b), -1, refusal))


@dataclass(frozen=True)
class EncounterSpec:
    """A pair, by body name, and the policy that pairs it: what `sweep.ModelConfig.encounters` holds,
    so a formation policy is a configuration like any other and can be swept."""

    a: str
    b: str
    policy: EncounterPolicy


def watch_encounter(sim: "Simulation", a: int, b: int, policy: EncounterPolicy) -> Tuple[events.Event, events.Event]:
    """
    Register `encounter_events(a, b, policy)` on `sim`, and form the pair now if it is already inside
    `form_km` (an event only sees a crossing, so a pair that starts inside would otherwise never
    form). Between the two radii it starts unpaired. Returns the two events.
    """
    form_event, dissolve_event = encounter_events(a, b, policy)
    sim.add_event(form_event)
    sim.add_event(dissolve_event)
    if separation_km(sim, a, b) < policy.radii_km(sim, a, b)[0]:
        _form_if_eligible(sim, a, b)
    return form_event, dissolve_event


# --- Reparenting a massless body: patched conics -------------------------------------------------------

def _bubble_for_parent(sim: "Simulation", parent: int) -> int:
    """The kinematic bubble a body orbiting `parent` belongs to: the system itself when the parent is
    a barycentre (the body sees the system as one point, its summed `mu`); the head's system when the
    parent heads one (the body orbits it inside that bubble, like a satellite of Earth inside the
    Earth-Moon system); otherwise the parent alone, a point-mass bubble."""
    if sim.is_system[parent]:
        return parent
    if sim.is_head[parent]:
        return int(sim.body_sys_map[parent])
    return parent


def reparent_refusal(sim: "Simulation", body: int, parent: int) -> Optional[str]:
    """Why `reparent(body, parent)` would refuse, or `None` if it would not. A pure read."""
    body, parent = int(body), int(parent)
    names = {v: k for k, v in sim.name_to_index.items()}
    for k in (body, parent):
        if not (0 <= k < sim.max_capacity) or not sim.active_mask[k]:
            return f"slot {k} is not active."
    if body == parent:
        return "a body cannot be its own parent."
    if sim.is_system[body] or sim.is_head[body]:
        return f"{names[body]!r} is a system or a head; only a plain body can be reparented."
    if sim.mu_array[body] != 0.0:
        return (f"{names[body]!r} has mass; reparenting a massive body changes reflex kicks and summed "
                f"masses, which is form_system's job, not a patched conic's.")
    kind = sim.propagator_type[body]
    if kind not in (np.uint8(PropagatorType.KEPLERIAN), np.uint8(PropagatorType.COWELL)):
        return (f"{names[body]!r} is {PropagatorType(int(kind)).name}; only Keplerian and Cowell bodies "
                f"can be reparented (secular J2's rates belong to its parent).")
    from .gravity import POINT_MASS_MODEL
    from .registry import get_force_model
    pm_bit = np.uint64(1) << np.uint64(get_force_model(POINT_MASS_MODEL).bit)
    if int(sim.force_model_mask[body] & ~pm_bit) != 0:
        return (f"{names[body]!r} carries force models beyond point_mass_gravity; their coefficients "
                f"(J2, an atmosphere) belong to its current parent and would be applied to the new one.")
    one = np.array([body], dtype=np.int64)
    deps = _dependants(sim, one, one)
    if deps.size:
        return f"{[names[int(k)] for k in deps]} depend on {names[body]!r}; not supported yet."
    return None


def reparent(sim: "Simulation", body: int, parent: int) -> int:
    """
    Make massless `body` orbit `parent` from now on: a patched-conic switch, and a change of frame
    only (its global state is kept; its local state and elements are re-derived against the new
    parent). Returns the body's new kinematic bubble (`_bubble_for_parent`).

    `parent` may be a barycentre: the body then orbits the system as one point mass, with the
    system's summed `mu`. That is the monopole approximation of the system; its error is the system's
    quadrupole, of relative size `(d / r)^2` for an inner separation `d`, so that switch has its own
    radius, not the Hill radius. Refusals (`reparent_refusal`) are checked before anything changes.
    """
    refusal = reparent_refusal(sim, body, parent)
    if refusal is not None:
        raise ValueError(refusal)
    body, parent = int(body), int(parent)
    bubble = _bubble_for_parent(sim, parent)
    sim.parent_indices[body] = parent
    sim.body_sys_map[body] = bubble
    _restructure(sim)
    sim._rehydrate_coes(rows=np.array([body], dtype=np.int64))
    names = {v: k for k, v in sim.name_to_index.items()}
    sim._hierarchy_log.append(HierarchyChange(float(sim.t), "reparent", names[body], (body, parent), bubble))
    return bubble


@dataclass(frozen=True)
class PatchSpec:
    """A massless body, a planet (body or system) it may be handed to, and the radii: what
    `sweep.ModelConfig.patches` holds. `target` is what the body orbits once inside: the planet itself
    (`"body"`) or, when the planet heads a system, that system's barycentre (`"system"`)."""

    body: str
    planet: str
    policy: EncounterPolicy
    target: str = "body"


def _patch_targets(sim: "Simulation", planet: int, target: str) -> Tuple[int, int, int]:
    """`(entity, inside, outside)`: what the radius is measured about, what the body is handed to
    inside, and what it is handed back to outside."""
    entity = int(sim.body_sys_map[planet]) if sim.is_head[planet] else int(planet)
    if target == "system":
        if not sim.is_head[planet]:
            raise ValueError("target='system' needs a planet that heads a system.")
        inside = entity
    elif target == "body":
        inside = int(planet)
    else:
        raise ValueError(f"target={target!r}: use 'body' or 'system'.")
    return entity, inside, int(sim.parent_indices[entity])


def patch_events(sim: "Simulation", body: int, planet: int, policy: EncounterPolicy,
                 target: str = "body") -> Tuple[events.Event, events.Event]:
    """
    Patched conics as events: `body`'s distance from `planet` falling through `form_km` hands it to
    the planet (or its system, `target="system"`), rising through `dissolve_km` hands it back to the
    parent the planet orbits. Radii in km, or in the planet's Hill or Laplace radius about its own
    parent, evaluated live (`EncounterPolicy.entity_radii_km`). Stateless like `encounter_events`: the
    actions read the body's current parent. Needs `sim` only to resolve the planet's system and
    outer parent, which a restructure elsewhere does not change. An ineligible handover is logged as a
    `"skip"`. Keep `dt` below `form_km / v_inf`, as for encounters.
    """
    body, planet = int(body), int(planet)
    entity, inside, outside = _patch_targets(sim, planet, target)
    bodies = np.array([body], dtype=np.int64)

    def dist(s: "Simulation") -> float:
        return float(np.linalg.norm(s.global_states[body, :3] - s.global_states[planet, :3]))

    def g_in(s: "Simulation", _b: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([dist(s) - policy.entity_radii_km(s, entity)[0]], dtype=np.float64)

    def g_out(s: "Simulation", _b: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.array([dist(s) - policy.entity_radii_km(s, entity)[1]], dtype=np.float64)

    def hand(s: "Simulation", to: int) -> None:
        if int(s.parent_indices[body]) == to:
            return
        refusal = reparent_refusal(s, body, to)
        if refusal is None:
            reparent(s, body, to)
        else:
            names = {v: k for k, v in s.name_to_index.items()}
            s._hierarchy_log.append(HierarchyChange(
                float(s.t), "skip", names[body], (body, to), -1, refusal))

    def enter(s: "Simulation", fired: NDArray[np.int64], t: float) -> None:
        hand(s, inside)

    def leave(s: "Simulation", fired: NDArray[np.int64], t: float) -> None:
        hand(s, outside)

    return (events.Event(name="patch: enter", function=g_in, bodies=bodies, direction=-1,
                         tol_s=policy.tol_s, action=enter),
            events.Event(name="patch: leave", function=g_out, bodies=bodies, direction=1,
                         tol_s=policy.tol_s, action=leave))


def watch_patch(sim: "Simulation", body: int, planet: int, policy: EncounterPolicy,
                target: str = "body") -> Tuple[events.Event, events.Event]:
    """Register `patch_events` on `sim`, and hand the body over now if it already starts inside
    `form_km`. Returns the two events."""
    enter_event, leave_event = patch_events(sim, body, planet, policy, target)
    sim.add_event(enter_event)
    sim.add_event(leave_event)
    entity, inside, _ = _patch_targets(sim, int(planet), target)
    d = float(np.linalg.norm(sim.global_states[int(body), :3] - sim.global_states[int(planet), :3]))
    if d < policy.entity_radii_km(sim, entity)[0] and int(sim.parent_indices[int(body)]) != inside:
        reparent(sim, body, inside)
    return enter_event, leave_event
