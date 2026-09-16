"""
The force-model composition layer.

**What this module is for.** The engine has exactly one model today: hierarchical Keplerian
propagation. This module is what makes a second one possible, and a tenth one cheap: it defines the
two contracts a force model and an integrator hold each other to, and the composition function that
sits between them. Nothing here is itself a force model - see `_constant_accel_kernel` and
`_radial_bias_kernel` at the bottom, which exist only to prove the contract is implementable and are
explicitly not physics.

**The two contracts.**

1. `ForceKernel` - what a force model implements. Stateless, allocation-free, array-in/array-out; see
   its docstring for the exact signature.
2. `AccelerationProvider` - what an integrator consumes. A plain `(t, state) -> acceleration`
   callable that knows nothing about models, masks, or how many of either exist. `Simulation.accelerations`
   implements it directly - a bound method already satisfies the protocol, nothing needs wrapping.

**Why composition is not itself a hot path.** `CLAUDE.md`'s two-implementation rule (readable NumPy
reference plus a compiled scalar twin, held equivalent by test) applies to code whose cost scales with
*body* count - that is what made `kepler_propagate` and `calc_global_states` worth compiling; the
profiling note in `docs/engineering-log.md` is specifically about per-call overhead multiplying by
thousands of bodies. `resolve_force_models` and `compose_accelerations` below scale with *model*
count instead - tens, per the design brief - so the Python-level loop in `compose_accelerations` never
grows past tens of iterations regardless of how many bodies are active. That is a different cost shape
than the one compilation was invented to fix, so this layer stays a single, readable implementation.

**This does not mean force kernels are exempt.** A production kernel like J2 *is* a hot path in the
traditional sense - it runs every step (or every integrator sub-stage) over every body that has it
enabled, which can be thousands. The two-implementation rule applies to *that* kernel, the same way it
applies to `kepler_propagate`: a readable NumPy reference, a compiled scalar twin in `kernels.py`
following the existing `@njit` pattern, and an equivalence test holding them together. That decision
and its implementation belong to whoever writes the kernel, not to this module - `ForceKernel` is
deliberately agnostic to whether a given implementation is the NumPy reference or the compiled twin;
both satisfy the same signature.

**On `np.add.at` vs. plain `+=`.** `indices` is always a sorted, duplicate-free array of arena slots
(built by `np.flatnonzero`), and it identifies both which rows of `out` a kernel may touch and which
rows of `params` describe it - never a many-to-one target. Plain fancy-indexed `out[indices] += ...`
is therefore exactly right and `np.add.at` is not needed here. (Contrast the reflex kick in
`kernels.kepler_propagate`, which *does* need it: several sibling bodies scatter onto the *same* head
slot, a genuine many-to-one accumulation.) A kernel that ever needs to scatter onto a target that can
repeat - a third-body perturbation accumulating onto a shared primary, say - would need `np.add.at`
for that step, for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .custom_types import ForceModelMask, ScalarSeconds
from .registry import ForceModel, all_force_models, register_force_model

__all__ = [
    "ForceKernel", "AccelerationProvider", "ResolvedForceModel",
    "resolve_force_models", "compose_accelerations",
]


# ==================================================================================================
# Contract 1: what a force model implements
# ==================================================================================================

class ForceKernel(Protocol):
    """
    The contract a force model implements. A kernel is a plain function - not a method, not a class
    instance - matching this call signature and registered against a name with
    `registry.register_force_model`.

    **Stateless.** No closures over mutable state, no attribute writes anywhere (there is no `self`).
    Called with the same arguments, a kernel must produce the same result, every time.

    **Allocates nothing.** Every array the kernel touches - including any per-body coefficients it
    needs, such as a J2 kernel's oblateness coefficient and reference radius, or a drag kernel's Cd
    and reference area - arrives as an argument. A kernel must not call `np.zeros`, `np.empty`, or
    anything else that allocates, inside its body. `params` is the caller-owned, arena-sized array
    that holds whatever per-body coefficients this model was registered with (see
    `registry.ForceModel.param_names`); a parameterless model still receives one, shaped `(C, 0)`.

    **Called once per step per *model*, never once per body.** `indices` is already the resolved set
    of arena slots this model applies to for this call - the composition layer did that filtering.
    The kernel's job is to do the arithmetic for that whole set as one vectorised NumPy expression (or,
    for a compiled twin, a scalar loop over `indices` only - see the module docstring on when that
    twin is warranted), not to be called once per element of `indices`.

    **Adds, never sets.** `out` is a shared accumulator: several models may write to overlapping
    bodies in the same call, and a later model's contribution must not erase an earlier one's. Always
    `out[indices] += contribution`, never `out[indices] = contribution`.

    **Must be a true no-op on an empty `indices`.** The composition layer already skips a model with
    no active bodies rather than calling it - see `resolve_force_models` - but a kernel should not
    *rely* on that as its only correctness guarantee; called directly (as the unit tests do) with
    `indices = np.array([], dtype=np.int64)`, it must touch nothing and return cleanly. Ordinary NumPy
    fancy indexing already gives this for free.

    Parameters
    ----------
    indices : NDArray[np.int64], shape (k,)
        Sorted, duplicate-free arena slots this model is active for on this call - the intersection
        of `Simulation.active_mask` and "has this model's bit set" in `Simulation.force_model_mask`.
        The kernel may read `state`, `mu_array`, and `parent_indices` at *other* rows too (a
        third-body kernel needs the perturbing body's position, not just the perturbed body's), but
        may only write `out` at these rows.
    t : ScalarSeconds
        Evaluation time, seconds, forwarded unchanged from whatever called
        `AccelerationProvider`. For a time-invariant model such as J2 this is simply unused.
    state : NDArray[np.float64], shape (C, 6)
        `[x y z vx vy vz]` per arena slot, same indexing and units (km, km/s) as
        `Simulation.global_states` - but **not necessarily that array**. An integrator evaluating an
        intermediate sub-stage passes its own candidate state here; see `AccelerationProvider`. Treat
        this as read-only.
    mu_array : NDArray[np.float64], shape (C,)
        Gravitational parameters, km^3/s^2, arena-indexed exactly like `Simulation.mu_array`. Read-only.
    parent_indices : NDArray[np.int32], shape (C,)
        The Keplerian parent graph, exactly `Simulation.parent_indices` - what each body's elements
        (and, for a model like J2, its perturbing geometry) are measured against. Root slots
        self-reference (`parent_indices[i] == i`), which a kernel using it for a relative vector must
        treat as "no primary" (the resulting relative vector is zero, not a divide-by-zero). Read-only.
    params : NDArray[np.float64], shape (C, k)
        This model's own per-body coefficients, `k = len(ForceModel.param_names)`, arena-sized and
        owned by the composition layer's caller (`Simulation.force_model_params[name]`). Read-only from
        the kernel's side; `Simulation.enable_force_model` is how a caller populates it.
    out : NDArray[np.float64], shape (C, 3)
        The shared acceleration accumulator, km/s^2, arena-indexed like `state`'s position columns.
        Add this model's contribution at `indices` and nowhere else; do not zero it (the composition
        layer does that once, before the first kernel runs).

    Returns
    -------
    None. All output is the in-place mutation of `out`.
    """

    def __call__(
        self,
        indices: NDArray[np.int64],
        t: ScalarSeconds,
        state: NDArray[np.float64],
        mu_array: NDArray[np.float64],
        parent_indices: NDArray[np.int32],
        params: NDArray[np.float64],
        out: NDArray[np.float64],
    ) -> None:
        ...


# ==================================================================================================
# Contract 2: what an integrator consumes
# ==================================================================================================

class AccelerationProvider(Protocol):
    """
    The contract an integrator consumes. `Simulation.accelerations` implements this directly - a
    bound method already satisfies the protocol, so an integrator holds a reference to
    `sim.accelerations` and calls it; nothing needs constructing or wrapping.

    A provider is a pure function of `(t, state)` for a fixed model configuration - it does not read
    "the current step" from anywhere, and in particular does not assume `state` is
    `Simulation.global_states`. That is what lets an RK-family integrator use it unchanged: such a
    method evaluates its derivative function at several intermediate stages per step, none of which is
    a state the arena should (yet) hold, and each call simply passes whatever candidate state that
    stage needs. This is the same shape as `reference.py`'s own `rhs(t, y)` closure, which
    `scipy.integrate.solve_ivp` calls the same way.

    The point of this contract is exactly that an integrator written against it needs to know nothing
    about how many force models are registered, which ones are enabled on which bodies, or how their
    contributions were combined - only that calling it returns the total.
    """

    def __call__(
        self, t: ScalarSeconds, state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        t : evaluation time, seconds.
        state : (C, 6) `[x y z vx vy vz]`, arena-indexed like `Simulation.global_states`.

        Returns (C, 3) acceleration, km/s^2, indexed identically to `state`'s body axis.

        The returned array is caller-owned scratch (`Simulation.accel_accum`), reused on every call -
        copy it before requesting the next stage if more than one result must stay alive at once.

        A body with no force model enabled reads back exactly `0.0`, not "small". That holds because
        the owner of the buffer clears it whenever the configuration is re-resolved (see
        `Simulation.resolve_force_models`) and `compose_accelerations` never writes rows outside the
        current dispatch set - both halves are needed. Without the clear, disabling a body's last
        model would leave that model's final acceleration in its row indefinitely.
        """
        ...


# ==================================================================================================
# The composition layer itself
# ==================================================================================================

@dataclass(frozen=True)
class ResolvedForceModel:
    """
    One force model, resolved against one arena's mask state: which slots it applies to right now,
    and the per-body parameter array it reads them from. Built by `resolve_force_models`, consumed by
    `compose_accelerations`. Not constructed directly outside this module.
    """
    name: str
    kernel: ForceKernel
    indices: NDArray[np.int64]
    params: NDArray[np.float64]


def resolve_force_models(
    active_mask: NDArray[np.bool_],
    force_model_mask: ForceModelMask,
    max_capacity: int,
    params_by_name: Dict[str, NDArray[np.float64]],
) -> Tuple[List[ResolvedForceModel], NDArray[np.int64]]:
    """
    Resolve every registered force model against one arena's current mask state.

    This is the boundary where a per-body integer mask turns into the small set of vectorised calls
    `compose_accelerations` will actually make - the only place in the composition layer that costs
    O(`max_capacity`) x O(registered models). Call it once, after the arena is built, and again
    whenever `force_model_mask` or `active_mask` changes (`Simulation.enable_force_model` does this
    automatically) - never once per step. `Simulation._build_universe` calls it once at the end of
    build, matching the existing `_refresh_active_indices` convention for exactly the same reason.

    Returns `(resolved, dispatch_idx)`:

    - `resolved` holds one `ResolvedForceModel` per registered model that has **at least one** active
      body with its bit set, in registration (bit) order. A model with zero active bodies is dropped
      entirely rather than resolved with an empty index array, so `len(resolved)` - and therefore the
      cost of `compose_accelerations` - tracks *models in use*, not *models registered*.
    - `dispatch_idx` is the sorted union of every resolved model's `indices`: every active body that
      has at least one force-model bit set. `compose_accelerations` zeros exactly these rows before
      dispatching. A body outside `dispatch_idx` is *never touched*, by zeroing or by any kernel - a
      mask of `0` is therefore a genuine no-op, not a defensively-skipped one: there is no code path
      that could write to that body's row.

    `params_by_name` is mutated in place: any resolved model without an entry yet is given a freshly
    zeroed `(max_capacity, n_params)` array under its name. Existing entries are reused untouched, so
    coefficients set by an earlier call (e.g. by `Simulation.enable_force_model` for a different body)
    survive a later resolve.
    """
    resolved: List[ResolvedForceModel] = []

    for model in all_force_models():
        bit_value = np.uint64(1) << np.uint64(model.bit)
        idx = np.flatnonzero(
            active_mask & ((force_model_mask & bit_value) != np.uint64(0))
        ).astype(np.int64)
        if idx.size == 0:
            continue

        params = params_by_name.get(model.name)
        if params is None:
            params = np.zeros((max_capacity, model.n_params), dtype=np.float64)
            params_by_name[model.name] = params

        resolved.append(ResolvedForceModel(name=model.name, kernel=model.kernel, indices=idx, params=params))

    if resolved:
        dispatch_idx = np.unique(np.concatenate([rm.indices for rm in resolved]))
    else:
        dispatch_idx = np.empty(0, dtype=np.int64)

    return resolved, dispatch_idx


def compose_accelerations(
    resolved: Sequence[ResolvedForceModel],
    dispatch_idx: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    out: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    One accumulation pass: zero `out` at `dispatch_idx`, dispatch every resolved model's kernel in
    turn, and return `out`.

    This is the entire per-step composition layer, and its only Python-level cost is one call per
    element of `resolved` - bounded by registered model count (tens), never by body count, by
    construction of `resolved` in `resolve_force_models`. The O(active bodies) arithmetic happens
    inside each kernel's own vectorised (or compiled) body, not in this loop.

    `out` is caller-owned scratch (`Simulation.accel_accum`); this function allocates nothing, and
    neither may any conforming `ForceKernel`. The returned array is `out` itself, not a copy.
    """
    out[dispatch_idx] = 0.0
    for rm in resolved:
        rm.kernel(rm.indices, t, state, mu_array, parent_indices, rm.params, out)
    return out


# ==================================================================================================
# Demonstration kernels - NOT physics.
#
# These exist to prove the contract above is implementable end to end: per-body parameters, additive
# composition of independent models, and (for the second one) reading relative geometry the way a
# real perturbation would. Neither corresponds to a real force. J2, drag, SRP, and any other physical
# model are explicitly out of scope here - see the module docstring - and should be registered from
# their own module, the way these are registered from this one.
# ==================================================================================================

@register_force_model(
    "test_constant_accel", param_names=("ax", "ay", "az"),
    citation="none - composition-layer test fixture, not a physical model",
)
def _constant_accel_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Adds a fixed, per-body acceleration vector taken straight from `params`. Ignores `t`, `state`,
    `mu_array`, and `parent_indices` entirely - proves a kernel that only needs `params` and `out`."""
    out[indices] += params[indices]


@register_force_model(
    "test_radial_bias", param_names=("k",),
    citation="none - composition-layer test fixture, not a physical model",
)
def _radial_bias_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Adds `k * unit(r_body - r_parent)` for each body: a constant-magnitude push directly away from
    its Keplerian parent, `k` read per-body from `params`. The magnitude law is deliberately not an
    inverse-square (or any other physical) law, so this cannot be mistaken for gravity or J2 - it
    exists only to prove a kernel can read `state` and `parent_indices` for relative geometry, the way
    a real perturbation (J2 not least) needs to.

    A root body (`parent_indices[i] == i`) has a zero relative vector and contributes nothing, handled
    by the same `safe` mask that guards the general divide-by-zero.
    """
    primaries = parent_indices[indices]
    rel = state[indices, :3] - state[primaries, :3]
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))

    unit = np.zeros_like(rel)
    safe = r > 0.0
    unit[safe] = rel[safe] / r[safe, None]

    out[indices] += params[indices, 0:1] * unit
