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
    "coe_to_rv_scalar", "solve_kepler_scalar", "secular_j2_propagate", "cowell_rk4_step",
    "rebase_relative_states", "DRAG_LAW_EXPONENTIAL", "DRAG_LAW_LAYERED", "DRAG_LAW_MSIS",
    "LAYERED_TABLE_ROW", "TABLE_N_NODES_COL", "TABLE_F107_COL",
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
def secular_j2_propagate(
    dt: float,
    coe_states: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    rates: NDArray[np.float64],
    indices: NDArray[np.int64],
    rel_out: NDArray[np.float64],
) -> None:
    """
    One first-order secular-J2 step over `indices`, in place on `coe_states`, writing the resulting
    parent-relative state vector into caller-owned `rel_out` (shape `(n_slots, 6)`) - compiled twin of
    `propagators.SecularJ2Propagator.propagate`, held equivalent by
    `tests/validation/test_kernel_equivalence.py` at 1e-12 relative.

    `rates[s]` (`[dRAAN/dt, dARGPE/dt, dM/dt]`, cached once at `Simulation.set_propagator` time by
    `propagators.secular_j2_rates`) advances mean anomaly and the two secular angles linearly in `dt`;
    `p`, `e`, `i` are untouched, since first-order secular J2 theory holds them constant. `rel_out` is
    NOT `local_states` - see `SecularJ2Propagator`'s docstring for why the caller must re-base it onto
    the parent's end-of-step global position rather than writing it directly.

    An invalid orbit (`p` degenerate, or a non-finite anomaly) leaves `coe_states[s, 3:5]` updated
    (matching `kepler_propagate`'s own "elements always advance" convention) but `rel_out[s]` untouched,
    mirroring `kepler_propagate`'s identical convention for a failed `coe_to_rv`.
    """
    n = indices.shape[0]
    for k in range(n):
        s = indices[k]
        par = parent_indices[s]
        mu = mu_array[s] + mu_array[par]

        p = coe_states[s, 0]
        e = coe_states[s, 1]
        inc = coe_states[s, 2]
        theta = coe_states[s, 5]

        M_old = true_to_mean_scalar(theta, e)
        M_new = (M_old + rates[s, 2] * dt) % (2.0 * math.pi)
        theta_new = mean_to_true_scalar(M_new, e, _SOLVER_TOL, _SOLVER_MAX_ITE)

        raan_new = coe_states[s, 3] + rates[s, 0] * dt
        argpe_new = coe_states[s, 4] + rates[s, 1] * dt

        coe_states[s, 3] = raan_new
        coe_states[s, 4] = argpe_new
        coe_states[s, 5] = theta_new

        if p <= MIN_SEMI_LATUS_RECTUM or math.isnan(theta_new):
            continue

        rx, ry, rz, vx, vy, vz = coe_to_rv_scalar(p, e, inc, raan_new, argpe_new, theta_new, mu)
        rel_out[s, 0] = rx
        rel_out[s, 1] = ry
        rel_out[s, 2] = rz
        rel_out[s, 3] = vx
        rel_out[s, 4] = vy
        rel_out[s, 5] = vz


# ==================================================================================================
# Cowell: RK4 with point-mass gravity, J2, drag and J3..J6, fused
# ==================================================================================================

# Highest zonal degree `zonal.zonal_kernel` evaluates (`zonal.ZONAL_DEGREES[-1]`). Restated rather than
# imported so this module stays free of force-model imports; `zonal_params` columns are
# `(r_eq, j3, j4, j5, j6)`, `J_n` in column `n - 2`, exactly `zonal.ZONAL_PARAM_NAMES`.
_ZONAL_MAX_DEGREE = 6

# `drag.DRAG_PARAM_NAMES` column layout and `drag._PER_M_TO_PER_KM`, restated for the same reason:
# `(ballistic_coeff, rho0, h0, scale_height, r_ref, omega, density_model, f107, f107a, ap)`.
_DRAG_B_COL = 0
_DRAG_RHO0_COL = 1
_DRAG_H0_COL = 2
_DRAG_SCALE_HEIGHT_COL = 3
_DRAG_R_REF_COL = 4
_DRAG_OMEGA_COL = 5
_DRAG_DENSITY_MODEL_COL = 6
_DRAG_F107_COL = 7
_DRAG_PER_M_TO_PER_KM = 1.0e3

# Integer codes for the three density laws, decoded from the float `density_model` selector by
# `_drag_law` with `drag.drag_kernel`'s own half-way thresholds.
DRAG_LAW_EXPONENTIAL = 0
DRAG_LAW_LAYERED = 1
DRAG_LAW_MSIS = 2
# Row of the stacked density tables that holds Vallado's layered table (`drag.density_tables`).
LAYERED_TABLE_ROW = 0
# Columns of `drag.DensityTables.meta`: each table row's node count (as a float), then the MSIS
# `(f107, f107a, ap)` it was built from (NaN on the layered row, which no MSIS body may match).
TABLE_N_NODES_COL = 0
TABLE_F107_COL = 1


@njit
def _drag_law(selector: float) -> int:
    """
    `drag.drag_kernel`'s dispatch on the `density_model` float, verbatim: `>= 0.5` leaves the single
    exponential, `>= 1.5` is MSIS, in between is the layered table. Written `not (selector >= 0.5)` so
    a NaN selector means the exponential law, as it does in the reference (`~(NaN >= 0.5)` is True).
    """
    if not (selector >= 0.5):
        return DRAG_LAW_EXPONENTIAL
    if selector >= 1.5:
        return DRAG_LAW_MSIS
    return DRAG_LAW_LAYERED


@njit
def _piecewise_density(h: float, tables: NDArray[np.float64], table: int, n_nodes: int) -> float:
    """
    `atmosphere.piecewise_exponential_density` for one altitude on row `table` of the stacked tables
    (`tables[0]` base altitudes, `[1]` base densities, `[2]` scale heights), whose first `n_nodes`
    entries are that table's bands: `rho_k exp(-(h - h_k) / H_k)`, kg/m^3.

    The band is found by a scalar binary search that reproduces `np.searchsorted(..., side="right")
    - 1`: `lo` ends as the count of base altitudes `<= h`, so an `h` exactly on a base altitude selects
    the band *starting* there. Clipping to `[0, n_nodes - 1]` extrapolates the bottom band below the
    first base altitude and the top band above the last, as the reference's `np.clip` does (`lo` never
    exceeds `n_nodes`, so only the lower clip can bind). A NaN `h` lands in band 0 here and in the top
    band there; both then produce a NaN density, so the band is immaterial.
    """
    lo = 0
    hi = n_nodes
    while lo < hi:
        mid = (lo + hi) // 2
        if tables[0, table, mid] <= h:
            lo = mid + 1
        else:
            hi = mid
    band = lo - 1
    if band < 0:
        band = 0
    exponent = (h - tables[0, table, band]) / tables[2, table, band]
    density: float = tables[1, table, band] * math.exp(-exponent)
    return density


@njit
def _gravity_accel(
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    mu_total: float, mu_par: float,
    has_point_mass: bool, has_j2: bool, j2: float, r_eq: float,
) -> tuple[float, float, float]:
    """
    `point_mass_gravity` then `j2` at candidate position `(cx, cy, cz)` with the parent at `(px, py,
    pz)`, summed into a running total starting at 0.0 - the first two terms of `_cowell_accel`. Scalars
    only, so the compiler inlines it into the stage loop.
    """
    ax = 0.0
    ay = 0.0
    az = 0.0

    if has_point_mass:
        # point_mass_gravity: rel points from the body toward its primary; mu is the two-body sum.
        rx = px - cx
        ry = py - cy
        rz = pz - cz
        r2 = rx * rx + ry * ry + rz * rz
        r = math.sqrt(r2)
        if r > 0.0:
            k = mu_total / (r2 * r)
            ax += k * rx
            ay += k * ry
            az += k * rz

    if has_j2:
        # j2: rel points from the primary to the body; mu is the parent's alone.
        x = cx - px
        y = cy - py
        z = cz - pz
        r2 = x * x + y * y + z * z
        if r2 > 0.0:
            inv_r2 = 1.0 / r2
            k = 1.5 * j2 * mu_par * r_eq * r_eq * inv_r2 * inv_r2 * math.sqrt(inv_r2)
            five_s2 = 5.0 * z * z * inv_r2
            ax += k * x * (five_s2 - 1.0)
            ay += k * y * (five_s2 - 1.0)
            az += k * z * (five_s2 - 3.0)

    return ax, ay, az


@njit
def _drag_term(
    ax: float, ay: float, az: float,
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    pvx: float, pvy: float, pvz: float,
    cvx: float, cvy: float, cvz: float,
    drag_law: int, drag_b: float, drag_rho0: float, drag_h0: float,
    drag_scale_height: float, drag_r_ref: float, drag_omega: float, drag_table: int,
    tables: NDArray[np.float64], n_nodes: int,
) -> tuple[float, float, float]:
    """
    `(ax, ay, az)` minus the drag acceleration at candidate state `(c, cv)` with the parent at `(p, pv)`
    - `drag.drag_kernel` one row at a time, its `out -= k v_rel` applied to the running sum. The drag
    coefficients are scalars the caller read once per body per step (nothing writes them inside a
    step); `drag_table` / `n_nodes` are the stacked-table row of the body's density law and its length,
    unread under the single exponential.

    **Velocity.** This is the only fused term that reads it. `cowell_rk4_step` passes each RK4 stage's
    own candidate velocity, rebuilt as `pv + v_k` exactly as `RK4Integrator` writes `state[indices, 3:]`
    before each provider call, and `cv - pv` is taken back off it here as `drag_kernel` does.
    """
    # rel points from the primary to the body. v_rel = v - w x r with w = omega z_hat, i.e.
    # (vx + omega y, vy - omega x, vz).
    x = cx - px
    y = cy - py
    z = cz - pz
    vx = cvx - pvx
    vy = cvy - pvy
    vz = cvz - pvz
    r2 = x * x + y * y + z * z
    vrx = vx + drag_omega * y
    vry = vy - drag_omega * x
    vrz = vz
    speed = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)
    altitude = math.sqrt(r2) - drag_r_ref

    # The reference sums three masked terms, exactly one of which can be non-zero per row:
    # `rho0 * factor` (factor 0.0 off the exponential law, or without separation or scale height),
    # then `+ layered` and `+ msis` (each exactly +0.0 off its own law or without separation). The
    # same sum is formed here, so even an exponential row's `+ 0.0 + 0.0` is reproduced.
    separated = r2 > 0.0
    exp_factor = 0.0
    if separated and drag_law == DRAG_LAW_EXPONENTIAL and drag_scale_height > 0.0:
        exp_factor = math.exp(-((altitude - drag_h0) / drag_scale_height))
    layered = 0.0
    msis = 0.0
    if separated and drag_law == DRAG_LAW_LAYERED:
        layered = _piecewise_density(altitude, tables, drag_table, n_nodes)
    elif separated and drag_law == DRAG_LAW_MSIS:
        msis = _piecewise_density(altitude, tables, drag_table, n_nodes)
    density = drag_rho0 * exp_factor + layered + msis                                # kg/m^3

    # 0.5 * rho [kg/m^3] * B [m^2/kg] * 1e3 -> 1/km; times |v_rel| [km/s] times v_rel [km/s].
    k = 0.5 * density * drag_b * _DRAG_PER_M_TO_PER_KM * speed
    ax -= k * vrx
    ay -= k * vry
    az -= k * vrz
    return ax, ay, az


@njit
def _zonal_term(
    ax: float, ay: float, az: float,
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    mu_par: float, zonal_params: NDArray[np.float64], row: int,
) -> tuple[float, float, float]:
    """
    `(ax, ay, az)` plus the J3..J6 acceleration: equation (Z) of zonal.py, the same two Legendre
    recursions in the same order as `zonal.zonal_kernel`, one scalar at a time. rel points from the
    primary to the body; mu is the parent's alone. The reference's zero-separation row adds an exact
    0.0 through inv_r = 0; here the guard skips it, which adds nothing either.
    """
    x = cx - px
    y = cy - py
    z = cz - pz
    r2 = x * x + y * y + z * z
    if r2 > 0.0:
        inv_r2 = 1.0 / r2
        inv_r = math.sqrt(inv_r2)
        s = z * inv_r
        rho = zonal_params[row, 0] * inv_r                           # R / r

        p_prev = 1.0              # P_0
        p_curr = s                # P_1
        dp_curr = 1.0             # P_1'
        rho_n = rho               # (R/r)^1
        radial = 0.0
        axial = 0.0
        for n in range(2, _ZONAL_MAX_DEGREE + 2):
            p_next = ((2 * n - 1) * s * p_curr - (n - 1) * p_prev) / n      # P_n
            dp_next = n * p_curr + s * dp_curr                              # P_n'
            # dp_next is P_n': the radial term of degree n - 1 and the axial term of degree n.
            # J_m sits in column m - 2.
            if n >= 4:
                radial += zonal_params[row, n - 3] * rho_n * dp_next
            rho_n = rho_n * rho                                             # (R/r)^n
            if 3 <= n <= _ZONAL_MAX_DEGREE:
                axial += zonal_params[row, n - 2] * rho_n * dp_next
            p_prev = p_curr
            p_curr = p_next
            dp_curr = dp_next

        k = mu_par * inv_r2                                                 # mu / r^2
        ax += k * radial * x * inv_r
        ay += k * radial * y * inv_r
        az += k * (radial * s - axial)
    return ax, ay, az


@njit
def _cowell_accel(
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    pvx: float, pvy: float, pvz: float,
    cvx: float, cvy: float, cvz: float,
    mu_total: float, mu_par: float,
    has_point_mass: bool, has_j2: bool, j2: float, r_eq: float,
    has_drag: bool, drag_law: int, drag_b: float, drag_rho0: float, drag_h0: float,
    drag_scale_height: float, drag_r_ref: float, drag_omega: float, drag_table: int,
    tables: NDArray[np.float64], n_nodes: int,
    has_zonal: bool, zonal_params: NDArray[np.float64], row: int,
) -> tuple[float, float, float]:
    """
    Acceleration on one body at candidate state `(cx, cy, cz, cvx, cvy, cvz)` with its parent at
    `(px, py, pz, pvx, pvy, pvz)`: the sum `forces.compose_accelerations` would build from
    `gravity.point_mass_gravity_kernel`, `geopotential.j2_kernel`, `drag.drag_kernel` and
    `zonal.zonal_kernel`, with each term's arithmetic written in the same order as its NumPy reference
    so the two agree to rounding. A disabled term contributes exactly `0.0`, as an absent model does.

    **Composition order matters with several terms.** IEEE addition is commutative but not associative,
    so `(pm + j2) + zonal` and `pm + (j2 + zonal)` may differ in the last bit. The terms are added here
    in registration order - `point_mass_gravity`, `j2`, `drag`, `zonal` (bits 0, 1, 2, 6) - which is
    the order `forces.resolve_force_models` walks `registry.all_force_models()` and therefore the order
    `compose_accelerations` accumulates them into `out`. Were the registration order ever different,
    the two would differ by an ulp of the sum per evaluation, still far inside the 1e-12 bound.

    **This is the composition `cowell_rk4_step` performs**, stage by stage, from the same three pieces
    in the same order - it calls them itself rather than this function only so that the density tables
    and zonal row are never passed through a per-stage call for a body that does not use them (measured:
    doing so tripled the per-body cost of the pm and j2 tiers). The field-level equivalence tests call
    this function; `test_cowell_*` hold the step to the same reference.

    `zonal_params[row]` is read only behind `has_zonal`, and the drag table only behind `has_drag`, so
    one-row dummies are safe when no body has those bits - the same convention as `j2_params` in
    `cowell_rk4_step`.
    """
    ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, has_point_mass, has_j2, j2, r_eq)
    if has_drag:
        ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                drag_law, drag_b, drag_rho0, drag_h0, drag_scale_height, drag_r_ref,
                                drag_omega, drag_table, tables, n_nodes)
    if has_zonal:
        ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, row)
    return ax, ay, az


@njit
def cowell_rk4_step(
    dt: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    indices: NDArray[np.int64],
    has_point_mass: NDArray[np.bool_],
    has_j2: NDArray[np.bool_],
    j2_params: NDArray[np.float64],
    has_zonal: NDArray[np.bool_],
    zonal_params: NDArray[np.float64],
    has_drag: NDArray[np.bool_],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_tables: NDArray[np.float64],
    density_meta: NDArray[np.float64],
    rel_out: NDArray[np.float64],
) -> int:
    """
    One classical RK4 step of every body in `indices`, integrated relative to its `parent_indices`
    parent under any subset of `point_mass_gravity`, `j2`, `drag` and `zonal` - the compiled twin of
    `integrators.RK4Integrator.step` driving `Simulation.accelerations` with exactly those models, fused
    with the subtraction `Simulation.step` performs afterwards. Held equivalent to that pair at 1e-12
    relative by `tests/validation/test_kernel_equivalence.py`.

    **Why fused rather than dispatched.** The NumPy path composes an arbitrary list of Python force
    kernels; numba cannot call into that list. This kernel therefore hard-codes the one combination
    the fidelity sweep runs (`_cowell_accel`'s composition), and `Simulation._refresh_cowell_plan`
    selects it only when every Cowell body's mask is a subset of `{point_mass_gravity, j2, drag,
    zonal}` - any other model falls back to the NumPy path, so the fallback is data
    (`Simulation._cowell_fused_ok`), not a branch on a model name inside the step.

    **Drag's density tables** are data the plan stacks at configuration time (`drag.density_tables`):
    `density_tables` is `(3, rows, width)` - base altitudes, base densities, scale heights - with row
    `LAYERED_TABLE_ROW` Vallado's table and each further row one memoised NRLMSIS profile, padded to one
    width; `density_meta[row]` is the row's real node count and the `(f107, f107a, ap)` it was built
    from. `drag_table_of[s]` is body `s`'s MSIS row (-1 where it has none). The kernel reads the
    `density_model` selector and every other drag coefficient *live*, as `drag.drag_kernel` does, so
    the only thing the plan can hold stale is the MSIS row choice.

    **Return value.** Before touching any state, every drag body whose live selector says MSIS is
    checked against its plan row's triple. On a mismatch - a profile never planned, or indices written
    straight into `force_model_params` after the plan was built - the kernel returns that slot and
    writes nothing, and `Simulation.step` re-plans and retries (or falls back to the reference, which
    raises `LookupError` on an unevaluated triple exactly as before). Otherwise it returns -1. The check
    is three float comparisons per MSIS body per step.

    **What it writes.** `state[s]` is left holding `state[parent] + relative_result`, exactly as
    `RK4Integrator.step` leaves it, and `rel_out[s]` holds that row minus `state[parent]` - the same
    arithmetic `Simulation.step` applies to the NumPy result before re-basing it onto the parent's
    end-of-step position. The subtraction is done here, on the value already rounded to the absolute
    grid, rather than by returning the relative result directly: for a Moon at 1.5e8 km heliocentric
    the two differ by an ulp of the absolute position, 8e-14 of the lunar distance per step, which is
    too close to the equivalence bound to leave to chance. `state[parent]` is read and never written,
    so a Cowell body whose parent is itself in `indices` is excluded by the caller's plan.

    Per-body flags rather than one global set so a mixed arena - some satellites with J2, some
    without, some with J3..J6 or drag - stays on the compiled path. `j2_params` is
    `force_model_params["j2"]` when any body has the J2 bit, and a one-row dummy otherwise; a row is
    only ever read behind its body's `has_j2`. `zonal_params` / `has_zonal` and `drag_params` /
    `has_drag` follow the same convention for `force_model_params["zonal"]` and `["drag"]`. `t` is not
    a parameter: no fused model depends on time (drag's atmosphere is steady in the frame co-rotating
    with the parent). Scalar stage values live in registers, so unlike `RK4Integrator` this needs no
    stage scratch and allocates nothing.
    """
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0
    n = indices.shape[0]

    # The staleness check, before anything is written - see "Return value".
    for k in range(n):
        s = indices[k]
        if has_drag[s] and _drag_law(drag_params[s, _DRAG_DENSITY_MODEL_COL]) == DRAG_LAW_MSIS:
            row = drag_table_of[s]
            if row < 0:
                return int(s)
            for c in range(3):
                if not (drag_params[s, _DRAG_F107_COL + c] == density_meta[row, TABLE_F107_COL + c]):
                    return int(s)

    for k in range(n):
        s = indices[k]
        par = parent_indices[s]
        pm = has_point_mass[s]
        jj = has_j2[s]
        zz = has_zonal[s]
        dd = has_drag[s]
        mu_par = mu_array[par]
        mu_total = mu_array[s] + mu_par
        j2 = 0.0
        r_eq = 0.0
        if jj:
            j2 = j2_params[s, 0]
            r_eq = j2_params[s, 1]
        law = DRAG_LAW_EXPONENTIAL
        b = 0.0
        rho0 = 0.0
        h0 = 0.0
        scale_height = 0.0
        r_ref = 0.0
        omega = 0.0
        table = LAYERED_TABLE_ROW
        if dd:
            b = drag_params[s, _DRAG_B_COL]
            rho0 = drag_params[s, _DRAG_RHO0_COL]
            h0 = drag_params[s, _DRAG_H0_COL]
            scale_height = drag_params[s, _DRAG_SCALE_HEIGHT_COL]
            r_ref = drag_params[s, _DRAG_R_REF_COL]
            omega = drag_params[s, _DRAG_OMEGA_COL]
            law = _drag_law(drag_params[s, _DRAG_DENSITY_MODEL_COL])
            if law == DRAG_LAW_MSIS:
                table = drag_table_of[s]
        n_nodes = int(density_meta[table, TABLE_N_NODES_COL])

        px = state[par, 0]
        py = state[par, 1]
        pz = state[par, 2]
        pvx = state[par, 3]
        pvy = state[par, 4]
        pvz = state[par, 5]

        # Initial relative state y0 = state[s] - state[par]; stage 1 is evaluated on the committed row,
        # velocity included (drag reads it).
        cx = state[s, 0]
        cy = state[s, 1]
        cz = state[s, 2]
        cvx = state[s, 3]
        cvy = state[s, 4]
        cvz = state[s, 5]
        r0x = cx - px
        r0y = cy - py
        r0z = cz - pz
        v0x = cvx - pvx
        v0y = cvy - pvy
        v0z = cvz - pvz

        # Each stage below is `_cowell_accel`'s composition - gravity, then drag, then zonal, in
        # registration order - written out so that only a body with drag or zonal passes their tables.
        a1x, a1y, a1z = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            a1x, a1y, a1z = _drag_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, pvx, pvy, pvz,
                                       cvx, cvy, cvz, law, b, rho0, h0, scale_height, r_ref, omega,
                                       table, density_tables, n_nodes)
        if zz:
            a1x, a1y, a1z = _zonal_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        v1x = v0x + half_dt * a1x
        v1y = v0y + half_dt * a1y
        v1z = v0z + half_dt * a1z

        # Stage 2 at t + dt/2: candidate row rebuilt as parent + (r0 + dt/2 v0, v1), as the reference
        # does - the velocity too, since drag reads it (`RK4Integrator` writes `state[primaries, 3:] +
        # v1` before this call, and each later stage's own `v`, never `v0`).
        cx = px + (r0x + half_dt * v0x)
        cy = py + (r0y + half_dt * v0y)
        cz = pz + (r0z + half_dt * v0z)
        cvx = pvx + v1x
        cvy = pvy + v1y
        cvz = pvz + v1z
        a2x, a2y, a2z = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            a2x, a2y, a2z = _drag_term(a2x, a2y, a2z, px, py, pz, cx, cy, cz, pvx, pvy, pvz,
                                       cvx, cvy, cvz, law, b, rho0, h0, scale_height, r_ref, omega,
                                       table, density_tables, n_nodes)
        if zz:
            a2x, a2y, a2z = _zonal_term(a2x, a2y, a2z, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        v2x = v0x + half_dt * a2x
        v2y = v0y + half_dt * a2y
        v2z = v0z + half_dt * a2z

        # Stage 3 at t + dt/2.
        cx = px + (r0x + half_dt * v1x)
        cy = py + (r0y + half_dt * v1y)
        cz = pz + (r0z + half_dt * v1z)
        cvx = pvx + v2x
        cvy = pvy + v2y
        cvz = pvz + v2z
        a3x, a3y, a3z = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            a3x, a3y, a3z = _drag_term(a3x, a3y, a3z, px, py, pz, cx, cy, cz, pvx, pvy, pvz,
                                       cvx, cvy, cvz, law, b, rho0, h0, scale_height, r_ref, omega,
                                       table, density_tables, n_nodes)
        if zz:
            a3x, a3y, a3z = _zonal_term(a3x, a3y, a3z, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        v3x = v0x + dt * a3x
        v3y = v0y + dt * a3y
        v3z = v0z + dt * a3z

        # Stage 4 at t + dt.
        cx = px + (r0x + dt * v2x)
        cy = py + (r0y + dt * v2y)
        cz = pz + (r0z + dt * v2z)
        cvx = pvx + v3x
        cvy = pvy + v3y
        cvz = pvz + v3z
        a4x, a4y, a4z = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            a4x, a4y, a4z = _drag_term(a4x, a4y, a4z, px, py, pz, cx, cy, cz, pvx, pvy, pvz,
                                       cvx, cvy, cvz, law, b, rho0, h0, scale_height, r_ref, omega,
                                       table, density_tables, n_nodes)
        if zz:
            a4x, a4y, a4z = _zonal_term(a4x, a4y, a4z, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)

        # Weighted combination on the relative state, then back onto the parent's start-of-step row.
        rnx = r0x + sixth_dt * (v0x + 2.0 * v1x + 2.0 * v2x + v3x)
        rny = r0y + sixth_dt * (v0y + 2.0 * v1y + 2.0 * v2y + v3y)
        rnz = r0z + sixth_dt * (v0z + 2.0 * v1z + 2.0 * v2z + v3z)
        vnx = v0x + sixth_dt * (a1x + 2.0 * a2x + 2.0 * a3x + a4x)
        vny = v0y + sixth_dt * (a1y + 2.0 * a2y + 2.0 * a3y + a4y)
        vnz = v0z + sixth_dt * (a1z + 2.0 * a2z + 2.0 * a3z + a4z)

        gx = px + rnx
        gy = py + rny
        gz = pz + rnz
        gvx = pvx + vnx
        gvy = pvy + vny
        gvz = pvz + vnz
        state[s, 0] = gx
        state[s, 1] = gy
        state[s, 2] = gz
        state[s, 3] = gvx
        state[s, 4] = gvy
        state[s, 5] = gvz
        rel_out[s, 0] = gx - px
        rel_out[s, 1] = gy - py
        rel_out[s, 2] = gz - pz
        rel_out[s, 3] = gvx - pvx
        rel_out[s, 4] = gvy - pvy
        rel_out[s, 5] = gvz - pvz

    return -1


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


@njit
def rebase_relative_states(
    indices: NDArray[np.int64],
    parent_indices: NDArray[np.int32],
    body_sys_map: NDArray[np.int32],
    rel: NDArray[np.float64],
    global_states: NDArray[np.float64],
    local_states: NDArray[np.float64],
) -> None:
    """
    Place each slot in `indices` at its `parent_indices` parent's current global state plus `rel[s]`,
    then rebuild its `local_states` row against its `body_sys_map` bubble, in place - the compiled twin
    of the NumPy block in `Simulation._rebase`, which `step()` runs for Cowell and secular-J2 bodies
    after `calc_global()` has moved their parents. Held **bit-identical** to that block by
    `tests/validation/test_kernel_equivalence.py`, the same standard as `calc_global_states`: one
    addition and one subtraction per component, in the same order on the same operands, so nothing
    is left for a tolerance to absorb.

    Why this exists: the NumPy block is eight small fancy-indexed operations, about 12 us of dispatch
    per step, paid by every tier that re-bases and not by the Keplerian tier - measured to dominate
    the Cowell and secular-J2 tiers' step cost once their propagation ran compiled.

    The NumPy block gathers every parent row *before* it assigns any, so a parent that is itself in
    `indices` is read at its pre-re-base value. This loop reads it live, which is why
    `Simulation._refresh_active_indices` falls back to the NumPy path when any body's parent lies in
    the same re-base set (`_rebase_compiled_ok`) rather than letting the two diverge there. A bubble
    is a system slot, which neither propagator may be assigned to, so the subtraction has no such
    hazard.
    """
    n = indices.shape[0]
    for k in range(n):
        s = indices[k]
        p = parent_indices[s]
        for c in range(6):
            global_states[s, c] = global_states[p, c] + rel[s, c]
    for k in range(n):
        s = indices[k]
        b = body_sys_map[s]
        for c in range(6):
            local_states[s, c] = global_states[s, c] - global_states[b, c]
