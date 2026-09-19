"""
Impulsive manoeuvres: an instantaneous Delta-v applied to a body, under **any** propagator.

This is the complement of `thrust.py`. Continuous thrust is an acceleration, so it is a
`forces.ForceKernel` and only a Cowell body can feel it. An impulse is not an acceleration at all -
it is a discontinuity in the state - so it needs no integrator, and the analytic tiers can fly a
mission profile as cheaply as they fly a coast. On a Keplerian body a Delta-v is nothing more than a
re-derivation of the orbital elements from the new `(r, v)`.

Model
-----
For body `i` with Keplerian parent `P = parent_indices[i]`, with `r = global[i,:3] - global[P,:3]`
and `v = global[i,3:] - global[P,3:]` the parent-relative state at the instant of the burn:

    dv_cart = dv_R R(r, v) + dv_S S(r, v) + dv_W W(r, v)        (frames.RSW_to_cart)
    r  ->  r                        position is continuous
    v  ->  v + dv_cart              velocity is discontinuous
    coe -> rv_to_coe(r, v + dv_cart, mu_i + mu_P)               analytic propagators only

The RSW basis is the *pre-burn* one (`frames.ReferenceFrames.RSW_basis`; R radial-out, W along
`r x v`, S = W x R along-track), so `(0, dv, 0)` is a prograde, horizontal burn - the same direction
convention, and the same rotation code, `thrust.py` uses for its continuous direction law. **S is the
along-track axis, not the velocity direction**: the two coincide only at zero flight-path angle
(circular orbits, apsides), which is exactly where a Hohmann burn is placed.

The frame is **parent-relative**, never absolute. A satellite of Earth gets its RSW frame from its
motion about Earth, not about the simulation root; using `global_states` directly would give a
heliocentric frame and a Delta-v pointing somewhere else entirely
(`tests/validation/test_manoeuvres.py::test_impulse_is_parent_relative_with_a_moving_parent`).

Per propagator
--------------
| propagator | what changes |
|---|---|
| `KEPLERIAN` | `global_states`/`local_states` velocity, **and** `coe_states`, which is this propagator's state of record - `KeplerianPropagator` rewrites `local_states` from the elements every step, so an impulse that did not re-derive the elements would be erased on the next step |
| `SECULAR_J2` | as Keplerian, **plus** the cached secular rates: `Simulation._secular_j2_rates` is derived once at `set_propagator` time from `p`, `e`, `i`, all three of which an impulse changes. `Simulation.apply_delta_v` recomputes them through `propagators.secular_j2_rates`; leaving them stale drifts the node at the old orbit's rate and raises nothing |
| `COWELL` | the Cartesian state, and nothing else. Its `coe_states` row is already documented stale (`CLAUDE.md`), so this deliberately does not touch it |

Citations
---------
- **The impulsive model itself** - a burn short compared with the orbital period, idealised as a
  velocity discontinuity at a fixed position: Vallado, *Fundamentals of Astrodynamics and
  Applications*, 4th ed., Sec. 6.1; Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed.,
  Sec. 6.1. There is no equation to get wrong here: `v -> v + dv` is the definition.
- **Hohmann transfer** (the validation case, not used by this module): Vallado 4e Sec. 6.3,
  Curtis 3e Sec. 6.2. Section numbers from memory and unverified, as elsewhere in this repo - but the
  closed form is re-derived from vis-viva in `tests/validation/test_manoeuvres.py`'s docstring, which
  is the version that should be checked, and it needs no text.
- **RSW frame**: Vallado 4e Sec. 3.3, implemented and cited in `frames.py`. Nothing is re-derived
  here; this module calls `ReferenceFrames.RSW_to_cart`.
- **Element re-derivation**: `frames.ReferenceFrames.rv_to_coe`, which carries its own citation and
  its own circular/equatorial fallbacks. An impulse routinely produces exactly the degenerate cases
  those fallbacks exist for (a prograde kick on a circular orbit makes `e` jump from 0), so nothing
  here special-cases them.

Why this is not a registered force model
----------------------------------------
`registry.py` has two dispatch axes: force models (a bit in `force_model_mask`, consumed by
`forces.compose_accelerations` as an acceleration) and propagators (`PropagatorType`, consumed by
`step()`). An impulse is neither. It is not an acceleration - giving it one would mean smearing it
over a step, which is precisely the continuous-thrust model that already exists - and it is not a way
of advancing state through time. There is no third axis ("scheduled events") in `registry.py`, and
inventing a parallel registry for one event type would be worse than the alternative: manoeuvres are
held as a plain sorted list on the `Simulation`, configured through `Simulation.schedule_delta_v` /
`apply_delta_v`, the same way propagator assignment is configured through `set_propagator`. If a
sweep ever needs to enumerate mission profiles as data, that queue is the thing `sweep.py` would
serialise, and *then* an events axis in the registry is worth adding. See the report in
`docs/architecture.md`'s manoeuvre section.

Restrictions
------------
`Simulation.apply_delta_v` refuses, with `ValueError` and before mutating anything, exactly what
`set_propagator` refuses for Cowell, for the same arena reasons:

- **inactive slots** - not part of the simulation;
- **system heads** - a head's motion is the reflex kick of its whole bubble, computed every step from
  its siblings' elements; a velocity written onto it is overwritten by the next `step()`, silently;
- **barycentres** (`is_system`) - a mass-weighted mean, not an object that can carry a thruster;
- **kinematic or Keplerian roots** (`body_sys_map[i] == i` or `parent_indices[i] == i`) - there is no
  parent to define an RSW frame or an element set against, and `calc_global` zeroes tier-0 rows
  anyway;
- **massive bodies** (`mu != 0`) - an impulse on a massive body changes its system barycentre's
  momentum, and the barycentre's own element row (its orbit about *its* parent) is a separate slot
  this call does not touch. The result would conserve neither momentum nor the barycentre's
  definition while looking entirely plausible. Massless is also what `set_propagator` already requires
  of every Cowell and secular-J2 body, and what `thrust.py` requires of a burning one, so in practice
  nothing that could carry an engine is excluded.

Limitations
-----------
- **No mass coupling.** An impulse does not consume `force_model_params["thrust"]`'s propellant. The
  rocket equation `m1 = m0 exp(-dv / (Isp g0))` would need an `Isp` that is a property of the *stage*,
  not of the impulse, plus a policy for an impulse larger than the tanks can deliver. Left out
  deliberately rather than guessed at; a vessel that both impulses and burns continuously therefore
  keeps its thrust mass across the impulse.
- **One frame.** Delta-v is given in RSW only. An inertial or a body-fixed direction law would be a
  second entry point, not a flag; `frames.py` already has the transforms if one is ever wanted.
- **No finite-burn correction.** A real burn of duration `tau` loses roughly `(1/24) g_r n^2 tau^2`
  worth of Delta-v to gravity and finite-arc losses. This model is the `tau -> 0` limit, and
  `thrust.py` is what to use when `tau` matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, COEIndex, PropagatorType, ScalarSeconds
from .frames import ReferenceFrames

__all__ = ["Manoeuvre", "apply_delta_v"]


@dataclass(frozen=True)
class Manoeuvre:
    """
    One scheduled impulse: `dv_rsw` applied to `bodies` at simulation time `epoch_s` seconds.

    Plain data, frozen, with no reference to a `Simulation` - so a mission profile is a list of these
    and nothing else, and two simulations differing only in their profile differ only in this list.
    `dv_rsw` is `(3,)` (one Delta-v applied to every body of the entry) or `(len(bodies), 3)` (one per
    body), in km/s, RSW components.

    `epoch_s` is seconds on the `Simulation.t` clock, not a `datetime`: every other time in the engine
    is, and `Simulation.current_epoch` converts when a wall-clock epoch is wanted.
    """

    epoch_s: float
    bodies: NDArray[np.int64]
    dv_rsw: ArrayFloat
    label: str = field(default="", compare=False)


# Sub-step shorter than this is treated as "already at the manoeuvre epoch" by `Simulation.step`, and
# no propagation is performed for it. One nanosecond of simulated time: far below any step size the
# engine is used at (seconds to hours) and far above the rounding of `t + dt` accumulated over a run,
# so the guard never fires spuriously and never hides a real sub-step. Its only job is to stop a
# zero-or-denormal-length Kepler advance being attempted for a manoeuvre placed exactly on a step
# boundary - the common case, since `t=0` is.
MIN_SUBSTEP_S: Final[float] = 1.0e-9


def apply_delta_v(
    indices: NDArray[np.int64],
    dv_rsw: ArrayFloat,
    global_states: NDArray[np.float64],
    local_states: NDArray[np.float64],
    coe_states: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    propagator_type: NDArray[np.uint8],
) -> NDArray[np.bool_]:
    """
    Apply an RSW Delta-v to `indices`, in place, and re-derive the elements of the analytic ones.

    Stateless and array-in/array-out: everything it reads and writes is passed in, and it holds no
    reference to a `Simulation`. `Simulation.apply_delta_v` is the validating wrapper.

    Parameters
    ----------
    indices : (k,) int64
        Arena slots, **duplicate-free** - the writes below are fancy-indexed, so a repeated slot would
        be a many-to-one scatter and one of its Delta-vs would vanish silently. Order is free and is
        preserved, so row `k` of a per-body `dv_rsw` belongs to `indices[k]`.
    dv_rsw : (3,) or (k, 3)
        Delta-v in RSW components, km/s. A single `(3,)` applies to every slot.
    global_states, local_states, coe_states : (C, 6)
        Mutated in place, for the valid rows only.
    mu_array, parent_indices, propagator_type : (C,)
        Read only. `mu` for the element re-derivation is the two-body sum
        `mu_array[i] + mu_array[parent]`, matching `KeplerianPropagator` and `_rehydrate_coes`.

    Returns
    -------
    valid : (k,) bool
        True where the impulse was applied. False where the pre-burn RSW frame is undefined
        (rectilinear, zero or non-finite parent-relative state - `RSW_basis` returns exactly-zero rows
        there, never NaN), where `rv_to_coe` could not classify the post-burn orbit of an analytic
        body, or where the post-burn orbit of a `SECULAR_J2` body is not closed (`e >= 1`), which its
        cached rates - built on mean motion `n = sqrt(mu/a^3)` - have no meaning for.
        **A false row is left completely untouched**: the whole transaction is computed first and
        committed only for the valid rows, so a partly-degenerate call cannot leave a body with a new
        velocity and stale elements, and `Simulation.apply_delta_v` can raise on a false row knowing
        the arena is exactly as it found it.

    Notes
    -----
    `local_states` takes the same Cartesian Delta-v as `global_states`, with no re-derivation: it is
    measured against `body_sys_map[i]`, whose own state this impulse does not change (the caller
    guarantees a massless body, so the bubble barycentre does not move, and a head - which does move
    reflexively - is refused outright).

    Not allocation-free. The temporaries are `(k, 3)` and `(k, 6)` for the *manoeuvring* set, not for
    `max_capacity`, and this runs on a scheduled event rather than every step; readability wins here
    the same way it does in `propagators.py`. There is no compiled twin and no case for one.
    """
    parents = parent_indices[indices]
    r_rel = global_states[indices, :3] - global_states[parents, :3]
    v_rel = global_states[indices, 3:] - global_states[parents, 3:]

    # Pre-burn frame: position is continuous across an impulse, so R, S, W are the same before and
    # after; the velocity is not, which is why this is computed before anything is written.
    dv_cart, frame_ok = ReferenceFrames.RSW_to_cart(r_rel, v_rel, np.atleast_2d(dv_rsw))

    analytic = propagator_type[indices] != np.uint8(PropagatorType.COWELL)
    new_coe, coe_ok = ReferenceFrames.rv_to_coe(
        r_rel, v_rel + dv_cart, mu_array[indices] + mu_array[parents],
    )

    # A secular-J2 body's cached rates derive from n = sqrt(mu/a^3), so an impulse that opens its
    # orbit has no representation in that theory - the same 0 <= e < 1 restriction `set_propagator`
    # applies at configuration time, applied again at the one other point that can break it. Checked
    # here, before the commit, so the refusal leaves the body on its old (valid) orbit rather than on
    # a hyperbola with NaN rates.
    #
    # `new_coe`'s eccentricity column is read unconditionally: on `rv_to_coe`'s degenerate
    # no-valid-orbit early return the array is `(k, 3)` rather than `(k, 6)` (a known shape bug, see
    # `CLAUDE.md`), but column 1 exists either way, and every row is already false through `coe_ok`.
    is_secular = propagator_type[indices] == np.uint8(PropagatorType.SECULAR_J2)
    e_new = new_coe[:, COEIndex.E]
    closed_ok = ~is_secular | ((e_new < 1.0) & np.isfinite(e_new))

    valid: NDArray[np.bool_] = frame_ok & (coe_ok | ~analytic) & closed_ok

    # Commit. Integer index arrays, so the guards below are `.size` tests on the manoeuvring set and
    # never a reduction over an arena-sized array.
    rows = indices[valid]
    if rows.size > 0:
        global_states[rows, 3:] += dv_cart[valid]
        local_states[rows, 3:] += dv_cart[valid]

    coe_rows = indices[valid & analytic]
    if coe_rows.size > 0:
        coe_states[coe_rows] = new_coe[valid & analytic]

    return valid


def due_before(manoeuvres: list[Manoeuvre], t_end: ScalarSeconds) -> int:
    """
    How many entries of a list sorted by `epoch_s` fall at or before `t_end`.

    A linear scan from the front: a mission profile is tens of entries, and `Simulation.step` needs
    the count once per step only when the queue is non-empty. Bisection would be the same answer with
    a comparison key and no measurable gain.
    """
    n = 0
    for m in manoeuvres:
        if m.epoch_s > t_end:
            break
        n += 1
    return n
