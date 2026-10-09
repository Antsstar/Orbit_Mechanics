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

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .custom_types import PropagatorType

if TYPE_CHECKING:
    from .simulator import Simulation


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


def form_system(sim: "Simulation", a: int, b: int, *, name: Optional[str] = None) -> int:
    """
    Make `a` and `b` a two-body system inside their shared bubble and return its barycentre's slot.

    The heavier member becomes the head (ties go to the lower slot, as `_resolve_circular` elects);
    the other orbits it. The barycentre takes the pair's place in the outer bubble, on the orbit
    about the shared parent that the pair's centre of mass is actually on. `name` defaults to
    `"<a>+<b>"`, heavier first.
    """
    a, b = int(a), int(b)
    if a == b:
        raise ValueError("form_system needs two distinct bodies.")
    names = {v: k for k, v in sim.name_to_index.items()}
    pair = np.array([a, b], dtype=np.int64)
    for k in (a, b):
        if not (0 <= k < sim.max_capacity) or not sim.active_mask[k]:
            raise ValueError(f"slot {k} is not an active body.")
        if sim.is_system[k] or sim.is_head[k]:
            raise ValueError(f"{names[k]!r} is a system or a head; only plain siblings can pair.")
        if sim.propagator_type[k] != np.uint8(PropagatorType.KEPLERIAN):
            raise ValueError(f"{names[k]!r} is not Keplerian; a temporary system's members must be.")
    if sim.body_sys_map[a] != sim.body_sys_map[b] or sim.parent_indices[a] != sim.parent_indices[b]:
        raise ValueError(f"{names[a]!r} and {names[b]!r} are not siblings of one bubble with one parent.")
    if sim.mu_array[a] + sim.mu_array[b] <= 0.0:
        raise ValueError("a system needs mass: both members are massless.")
    deps = _dependants(sim, pair, pair)
    if deps.size:
        raise ValueError(f"{[names[int(k)] for k in deps]} depend on a member; not supported yet.")
    if not sim.free_indices:
        raise ValueError(f"the arena is full (capacity {sim.max_capacity}).")

    head, sib = (a, b) if (sim.mu_array[a] > sim.mu_array[b] or
                           (sim.mu_array[a] == sim.mu_array[b] and a < b)) else (b, a)
    name = name if name is not None else f"{names[head]}+{names[sib]}"
    if name in sim.name_to_index:
        raise ValueError(f"a body named {name!r} already exists.")

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
