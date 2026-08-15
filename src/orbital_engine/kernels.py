"""
Compiled propagation kernels.

**Why this module exists.** Benchmarking showed the engine spends its time in Python-level NumPy
dispatch rather than in arithmetic: a 5-body step cost 556 us, and a 602-body step cost 3583 us, so
roughly 490 us was fixed overhead independent of workload. Vectorising harder cannot remove a cost
that is per-*call* rather than per-element. Compiling the whole step into one call can.

The kernels below are therefore written as **scalar loops over bodies**, which is the opposite of the
`utilities`/`frames` house style. That inversion is deliberate and confined to this module: under
`@njit` a scalar loop compiles to tight machine code with no temporaries, whereas the vectorised form
would still allocate an intermediate array per operation.

**Numba is optional.** When it is absent, `njit` degrades to an identity decorator and every function
here still runs as ordinary Python - slower than the NumPy path, but correct and fully testable. That
matters: it means the kernel is never an untested code path, and `NUMBA_AVAILABLE` selects an
implementation rather than gating a feature.

**Correctness contract.** These kernels must agree elementwise with the NumPy implementation in
`propagators.KeplerianPropagator`, which remains the reference. That equivalence is asserted in
`tests/validation/test_kernel_equivalence.py` rather than assumed. The reference is the readable
definition of the physics; this is an optimisation of it, and an optimisation that disagrees with its
reference is simply wrong.

References
----------
Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed.
  - Alg. 2 (Kepler's equation, Newton-Raphson with banded seeding)
  - Eq. 2-103 / Alg. 10 (COE -> r,v via the 3-1-3 perifocal-to-inertial rotation)
  - Eq. 2-13 (Barker's equation for the parabolic case)
"""
from __future__ import annotations

import math
from typing import Any, Callable, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NUMBA_AVAILABLE", "kepler_propagate", "calc_global_states",
    "coe_to_rv_scalar", "solve_kepler_scalar",
]

_F = TypeVar("_F", bound=Callable[..., Any])

try:  # pragma: no cover - exercised by whichever branch the environment provides
    from numba import njit as _numba_njit

    NUMBA_AVAILABLE = True

    def njit(func: _F) -> _F:
        """
        Compile in nopython mode, cached to disk so only the first process pays compilation.

        `fastmath` is deliberately off. It would licence the compiler to reassociate floating-point
        operations and to assume no NaN or infinity, both of which this code depends on: the Kepler
        solver detects divergence *by* testing for non-finite values, and the equivalence tests
        assert agreement with the NumPy reference at 1e-12 relative, which reassociation would break.
        Speed bought by weakening the arithmetic is not worth having in a numerical engine.

        The cast is needed because numba types the decorated result as `Dispatcher[_F]` rather than
        `_F`; the call signature is preserved, so this is accurate for callers.
        """
        return cast(_F, _numba_njit(cache=True, fastmath=False)(func))

except ImportError:  # pragma: no cover - exercised by whichever branch the environment provides
    NUMBA_AVAILABLE = False

    def njit(func: _F) -> _F:
        """Identity decorator. Without numba the kernels run as plain Python: slow, still correct."""
        return func


# `_step_anomalies` classifies parabolic orbits with `np.isclose(e, 1.0, atol=1e-9)`. np.isclose
# also applies its *default* rtol of 1e-5, so the effective band is |e - 1| <= 1e-9 + 1e-5, not the
# 1e-9 the call site suggests. Replicated exactly here so the kernel and the reference agree; see
# the note in CLAUDE.md, as narrowing it is a physics change rather than a refactor.
PARABOLIC_BAND = 1e-9 + 1e-5

# Matches `coe_to_rv`, which treats a non-positive semi-latus rectum as "no valid orbit" and leaves
# the state vector zeroed rather than raising.
MIN_SEMI_LATUS_RECTUM = 1e-12

# Matches `Kepler.t_to_M`, which returns zero mean-anomaly advance for a degenerate semi-major axis.
MIN_SEMI_MAJOR_AXIS = 1e-9

_SOLVER_TOL = 1e-5
_SOLVER_MAX_ITE = 1000


# ==================================================================================================
# Scalar anomaly stack
# ==================================================================================================

@njit
def solve_kepler_scalar(M: float, e: float, tol: float, max_ite: int) -> float:
    """
    Mean anomaly to eccentric/hyperbolic anomaly for one body.

    Mirrors `Anomalies.mean_to_eccentric` including its seed bands and its convergence test. The
    test is `not (abs(delta) <= tol)` rather than `abs(delta) > tol` for the same reason as the
    reference: a diverging iterate produces NaN, NaN compares False against everything, and the
    first form would exit reporting success with NaN in hand. NaN is returned here rather than
    raised - the caller checks - because raising from nopython code costs the compiler its
    optimisations.
    """
    if e > 1.0:
        E = math.asinh(M / e)
    elif e <= 0.55:
        E = M
    elif e <= 0.95:
        A = 6.0 * M
        E = math.copysign(abs(A) ** (1.0 / 3.0), A)
    else:
        E = math.pi

    for _ in range(max_ite):
        if e > 1.0:
            f = e * math.sinh(E) - E - M
            f_prime = 1.0 - e * math.cosh(E)
        else:
            f = E - e * math.sin(E) - M
            f_prime = e * math.cos(E) - 1.0

        if f_prime == 0.0:
            return math.nan

        delta = f / f_prime
        E += delta

        if abs(delta) <= tol:
            return E
        if not math.isfinite(E):
            return math.nan

    return math.nan


@njit
def true_to_mean_scalar(theta: float, e: float) -> float:
    """True anomaly to mean anomaly. Elliptic and hyperbolic branches only; caller filters parabolic."""
    half = math.tan(theta / 2.0)
    if e > 1.0:
        H = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * half)
        return e * math.sinh(H) - H
    E = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * half)
    return E - e * math.sin(E)


@njit
def mean_to_true_scalar(M: float, e: float, tol: float, max_ite: int) -> float:
    """Mean anomaly to true anomaly, via the eccentric/hyperbolic anomaly."""
    E = solve_kepler_scalar(M, e, tol, max_ite)
    if not math.isfinite(E):
        return math.nan

    if e > 1.0:
        return 2.0 * math.atan(math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(E / 2.0))
    y = math.sqrt(1.0 + e) * math.sin(E / 2.0)
    x = math.sqrt(1.0 - e) * math.cos(E / 2.0)
    return 2.0 * math.atan2(y, x)


@njit
def advance_true_anomaly(theta: float, p: float, e: float, mu: float, dt: float) -> float:
    """
    Advance true anomaly by `dt` under two-body motion. Mirrors `KeplerianPropagator._step_anomalies`.

    Elliptic mean anomaly is wrapped to [0, 2pi) but hyperbolic and parabolic are not, since those
    are not periodic and wrapping would be meaningless.
    """
    if abs(e - 1.0) <= PARABOLIC_BAND:
        # Barker's equation. Parabolic "mean anomaly" is dimensionless.
        delta_M = 2.0 * math.sqrt(mu / (p * p * p)) * dt
        half = math.tan(theta / 2.0)
        M_new = (half + half * half * half / 3.0) + delta_M

        A = 1.5 * M_new
        B_arg = A + math.sqrt(A * A + 1.0)
        B = math.copysign(abs(B_arg) ** (1.0 / 3.0), B_arg)
        return 2.0 * math.atan(B - 1.0 / B)

    a = p / (1.0 - e * e)
    if e >= 1.0:
        a = abs(a)

    delta_M = 0.0
    if abs(a) > MIN_SEMI_MAJOR_AXIS:
        delta_M = math.sqrt(mu / (a * a * a)) * dt

    M_new = true_to_mean_scalar(theta, e) + delta_M
    if e < 1.0:
        M_new = M_new % (2.0 * math.pi)

    return mean_to_true_scalar(M_new, e, _SOLVER_TOL, _SOLVER_MAX_ITE)


# ==================================================================================================
# Scalar state-vector construction
# ==================================================================================================

@njit
def coe_to_rv_scalar(
    p: float, e: float, inc: float, raan: float, arg_pe: float, theta: float, mu: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Classical elements to inertial position and velocity for one body.

    The perifocal vectors have zero z-component, so only the first two columns of the 3-1-3
    rotation Rz(raan) @ Rx(inc) @ Rz(arg_pe) are ever needed. Forming those six entries directly
    avoids building and multiplying a 3x3 matrix per body, which is the bulk of what the vectorised
    `coe_to_rv` spends its time on.
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    r_mag = p / (1.0 + e * cos_t)
    r_x = r_mag * cos_t
    r_y = r_mag * sin_t

    mu_h = math.sqrt(mu / p)
    v_x = -mu_h * sin_t
    v_y = mu_h * (e + cos_t)

    cO = math.cos(raan)
    sO = math.sin(raan)
    ci = math.cos(inc)
    si = math.sin(inc)
    cw = math.cos(arg_pe)
    sw = math.sin(arg_pe)

    m00 = cO * cw - sO * ci * sw
    m01 = -cO * sw - sO * ci * cw
    m10 = sO * cw + cO * ci * sw
    m11 = -sO * sw + cO * ci * cw
    m20 = si * sw
    m21 = si * cw

    return (
        m00 * r_x + m01 * r_y,
        m10 * r_x + m11 * r_y,
        m20 * r_x + m21 * r_y,
        m00 * v_x + m01 * v_y,
        m10 * v_x + m11 * v_y,
        m20 * v_x + m21 * v_y,
    )


# ==================================================================================================
# The step kernel
# ==================================================================================================

@njit
def kepler_propagate(
    dt: float,
    coe_states: NDArray[np.float64],
    local_states: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    body_sys_map: NDArray[np.int32],
    sys_head_map: NDArray[np.int32],
    is_system: NDArray[np.bool_],
    sib_idx: NDArray[np.int64],
    head_idx: NDArray[np.int64],
    kick: NDArray[np.float64],
    accum: NDArray[np.float64],
) -> None:
    """
    One Keplerian step over the whole arena, in place.

    `sib_idx` and `head_idx` are the active non-head and active head slots respectively; the caller
    owns them because they change only when the active set changes, not every step. `kick` and
    `accum` are caller-owned scratch of shape (n_slots, 6); this kernel allocates nothing.

    Three passes, matching the reference:

    1. Advance each sibling's anomaly, rebuild its state vector relative to its *parent*, and
       accumulate the mass-weighted sum onto its head if it is a barycentric sibling.
    2. Convert each head's accumulated sum into its reflex kick, dividing by the total system mass
       held on the barycenter's `mu_array` row.
    3. Add the head's kick to its barycentric siblings, and write it as the head's own state.

    A head's local state is *set* to the kick rather than accumulated, so `kick` must be zeroed for
    every active head each step - a head whose siblings all failed validity must end at rest, not
    retain last step's value.
    """
    n_sibs = sib_idx.shape[0]
    n_heads = head_idx.shape[0]

    # Zero only the slots this step will touch, rather than the whole arena. Cost then scales with
    # the active set instead of with max_capacity.
    for k in range(n_heads):
        h = head_idx[k]
        for c in range(6):
            kick[h, c] = 0.0
            accum[h, c] = 0.0

    # --- Pass 1: advance siblings, accumulate mass moments onto heads -----------------------------
    for k in range(n_sibs):
        s = sib_idx[k]
        par = parent_indices[s]
        mu_total = mu_array[s] + mu_array[par]

        p = coe_states[s, 0]
        e = coe_states[s, 1]

        theta = advance_true_anomaly(coe_states[s, 5], p, e, mu_total, dt)
        coe_states[s, 5] = theta

        # Mirrors coe_to_rv's validity mask: a degenerate p or a NaN anomaly yields no state vector.
        if p <= MIN_SEMI_LATUS_RECTUM or math.isnan(theta):
            continue

        rx, ry, rz, vx, vy, vz = coe_to_rv_scalar(
            p, e, coe_states[s, 2], coe_states[s, 3], coe_states[s, 4], theta, mu_total)

        local_states[s, 0] = rx
        local_states[s, 1] = ry
        local_states[s, 2] = rz
        local_states[s, 3] = vx
        local_states[s, 4] = vy
        local_states[s, 5] = vz

        # Barycentric sibling: lives in a real barycenter's bubble *and* orbits that bubble's head.
        sys_slot = body_sys_map[s]
        if sys_slot < 0 or not is_system[sys_slot]:
            continue
        if par != sys_head_map[s]:
            continue

        m = mu_array[s]
        accum[par, 0] += rx * m
        accum[par, 1] += ry * m
        accum[par, 2] += rz * m
        accum[par, 3] += vx * m
        accum[par, 4] += vy * m
        accum[par, 5] += vz * m

    # --- Pass 2: heads' reflex kicks --------------------------------------------------------------
    for k in range(n_heads):
        h = head_idx[k]
        bc = body_sys_map[h]
        if bc < 0:
            continue
        total_mass = mu_array[bc]
        if total_mass <= 0.0:
            continue
        for c in range(6):
            kick[h, c] = -accum[h, c] / total_mass

    # --- Pass 3: apply kicks ----------------------------------------------------------------------
    for k in range(n_sibs):
        s = sib_idx[k]
        par = parent_indices[s]

        p = coe_states[s, 0]
        if p <= MIN_SEMI_LATUS_RECTUM or math.isnan(coe_states[s, 5]):
            continue
        sys_slot = body_sys_map[s]
        if sys_slot < 0 or not is_system[sys_slot]:
            continue
        if par != sys_head_map[s]:
            continue

        for c in range(6):
            local_states[s, c] += kick[par, c]

    for k in range(n_heads):
        h = head_idx[k]
        for c in range(6):
            local_states[h, c] = kick[h, c]


@njit
def calc_global_states(
    topo_order: NDArray[np.int64],
    n_roots: int,
    body_sys_map: NDArray[np.int32],
    local_states: NDArray[np.float64],
    global_states: NDArray[np.float64],
) -> None:
    """
    Accumulate local states into global states, in place.

    `topo_order` is every active slot in topological order of the *kinematic* graph
    (`body_sys_map`), with the `n_roots` self-referencing roots first. Because the order is
    topological, a single forward pass guarantees each slot's system bubble is already resolved by
    the time the slot is read - so the tier structure the reference implementation loops over is not
    needed here at all, only the flattening of it.

    That is the whole optimisation. The reference issues roughly four fancy-indexed NumPy operations
    per tier, each a gather and a scatter over non-contiguous slots; at three tiers and five bodies
    that is ~24 us of dispatch to perform thirty additions.

    Note this reads `body_sys_map`, not `parent_indices`. The two graphs deliberately diverge:
    `local_states[i]` is relative to `global_states[body_sys_map[i]]`, while `parent_indices[i]` is
    what the orbital elements are measured against. Using the latter here would produce a plausible
    trajectory that is wrong for any body whose bubble is not its element parent - which is exactly
    the Moon.
    """
    for k in range(n_roots):
        s = topo_order[k]
        for c in range(6):
            global_states[s, c] = 0.0

    for k in range(n_roots, topo_order.shape[0]):
        s = topo_order[k]
        p = body_sys_map[s]
        for c in range(6):
            global_states[s, c] = global_states[p, c] + local_states[s, c]
