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
    "LAYERED_TABLE_ROW", "TABLE_N_NODES_COL", "TABLE_F107_COL", "TESSERAL_VW_SIZE",
    "COWELL_POINT_MASS", "COWELL_J2", "COWELL_ZONAL", "COWELL_DRAG", "COWELL_TESSERAL", "COWELL_THIRD_BODY",
    "cowell_leapfrog_step", "cowell_yoshida4_step", "cowell_encke_step",
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
# Cowell: RK4 with point-mass gravity, J2, drag, J3..J6 and the 4x4 tesseral field, fused
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


# `tesseral.TESSERAL_PARAM_NAMES` column layout, restated like the zonal and drag layouts above:
# `(r_eq, omega, theta0, c21, s21, c22, s22, ..., c44, s44)`, pair `j` of `tesseral.TESSERAL_PAIRS` -
# (2,1) (2,2) (3,1) (3,2) (3,3) (4,1) (4,2) (4,3) (4,4) - with C in column 3 + 2j and S in 4 + 2j. The
# pairs are generated below as `for n in 2..4: for m in 1..n`, which is that same order.
_TESS_R_EQ_COL = 0
_TESS_OMEGA_COL = 1
_TESS_THETA0_COL = 2
_TESS_FIRST_CS_COL = 3
_TESS_MAX_DEGREE = 4
# V/W are needed to degree and order `_TESS_MAX_DEGREE + 1`; the caller-owned scratch is
# `(2, TESSERAL_VW_SIZE, TESSERAL_VW_SIZE)`, V in `[0]` and W in `[1]`, indexed `[n, m]`.
TESSERAL_VW_SIZE = _TESS_MAX_DEGREE + 2


@njit
def _tesseral_angle(tesseral_params: NDArray[np.float64], row: int, t: float) -> tuple[float, float]:
    """
    `(cos theta, sin theta)` of the body-fixed prime meridian at time `t`, `theta = theta0 + omega * t`
    - `tesseral.tesseral_kernel`'s own expression, `t` the absolute simulation time of the stage being
    evaluated. Split from `_tesseral_term` so `cowell_rk4_step` evaluates it once for RK4's two
    mid-step stages, which share the time `t + dt/2` and so the same angle bit for bit.
    """
    theta = tesseral_params[row, _TESS_THETA0_COL] + tesseral_params[row, _TESS_OMEGA_COL] * t
    return math.cos(theta), math.sin(theta)


@njit
def _tesseral_term(
    ax: float, ay: float, az: float,
    px: float, py: float, pz: float,
    cx: float, cy: float, cz: float,
    mu_par: float, tesseral_params: NDArray[np.float64], row: int,
    cos_t: float, sin_t: float, vw: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    `(ax, ay, az)` plus the tesseral/sectoral acceleration of degrees 2..4, orders 1..n: equation (T) of
    tesseral.py, `tesseral.tesseral_kernel` one scalar at a time - the rotation into the body frame by
    `-theta`, the Cunningham V/W recursion to degree and order 5 in the same order with the same
    integer factors, the nine pairs accumulated in `TESSERAL_PAIRS` order from 0.0, and the rotation back
    by `+theta`. `(cos_t, sin_t)` is `_tesseral_angle` at this stage's own time.

    `vw` is caller-owned `(2, 6, 6)` scratch the recursion fills (V then W); every entry it reads was
    written earlier in the same call, so nothing needs clearing. Zero separation is not branched around:
    `inv_r2 = 0` there, as in the reference's `np.divide(..., where=r2 > 0)`, which makes every V/W and
    therefore the term exactly (signed) zero; likewise `r_eq <= 0` gives `mu / R^2 = 0`.
    """
    x = cx - px
    y = cy - py
    z = cz - pz
    r_eq = tesseral_params[row, _TESS_R_EQ_COL]

    # Inertial -> body-fixed: Rz(-theta).
    xb = cos_t * x + sin_t * y
    yb = -sin_t * x + cos_t * y
    zb = z

    r2 = xb * xb + yb * yb + zb * zb
    inv_r2 = 0.0
    if r2 > 0.0:
        inv_r2 = 1.0 / r2
    f = r_eq * inv_r2                                   # R / r^2
    rr = r_eq * f                                       # R^2 / r^2

    top = _TESS_MAX_DEGREE + 1
    vw[0, 0, 0] = r_eq * math.sqrt(inv_r2)              # R / r
    vw[1, 0, 0] = 0.0
    for m in range(top + 1):
        if m > 0:
            vp = vw[0, m - 1, m - 1]
            wp = vw[1, m - 1, m - 1]
            vw[0, m, m] = (2 * m - 1) * (xb * f * vp - yb * f * wp)
            vw[1, m, m] = (2 * m - 1) * (xb * f * wp + yb * f * vp)
        if m + 1 <= top:
            vw[0, m + 1, m] = (2 * m + 1) * zb * f * vw[0, m, m]
            vw[1, m + 1, m] = (2 * m + 1) * zb * f * vw[1, m, m]
        for n in range(m + 2, top + 1):
            vw[0, n, m] = ((2 * n - 1) * zb * f * vw[0, n - 1, m]
                           - (n + m - 1) * rr * vw[0, n - 2, m]) / (n - m)
            vw[1, n, m] = ((2 * n - 1) * zb * f * vw[1, n - 1, m]
                           - (n + m - 1) * rr * vw[1, n - 2, m]) / (n - m)

    tx = 0.0
    ty = 0.0
    tz = 0.0
    col = _TESS_FIRST_CS_COL
    for n in range(2, _TESS_MAX_DEGREE + 1):
        for m in range(1, n + 1):
            c = tesseral_params[row, col]
            s = tesseral_params[row, col + 1]
            col += 2
            fac = (n - m + 2) * (n - m + 1)
            vu = vw[0, n + 1, m + 1]
            wu = vw[1, n + 1, m + 1]
            vd = vw[0, n + 1, m - 1]
            wd = vw[1, n + 1, m - 1]
            tx += 0.5 * ((-c * vu - s * wu) + fac * (c * vd + s * wd))
            ty += 0.5 * ((-c * wu + s * vu) + fac * (-c * wd + s * vd))
            tz += (n - m + 1) * (-c * vw[0, n + 1, m] - s * vw[1, n + 1, m])

    inv_r_eq2 = 0.0
    if r_eq > 0.0:
        inv_r_eq2 = 1.0 / (r_eq * r_eq)
    k = mu_par * inv_r_eq2                              # mu / R^2
    # Body-fixed -> inertial: Rz(+theta).
    ax += k * (cos_t * tx - sin_t * ty)
    ay += k * (sin_t * tx + cos_t * ty)
    az += k * tz
    return ax, ay, az


# Columns of `force_model_params["third_body"]` (`thirdbody.THIRD_BODY_PARAM_NAMES`); `kernels` cannot
# import `thirdbody` (it sits above), and a test asserts these equal its own constants.
_THIRD_PERTURBER_COL = 0
_THIRD_STAGED_COL = 1
_THIRD_T0_COL = 2


@njit
def _third_body_rel(
    state: NDArray[np.float64], mu_array: NDArray[np.float64], third_params: NDArray[np.float64],
    row: int, px: float, py: float, pz: float, pvx: float, pvy: float, pvz: float,
    mu_par: float, t: float,
) -> tuple[float, float, float, float]:
    """
    `(mu_s, r_s)` for body `row` at stage time `t`: the perturber's gravitational parameter and its
    position relative to the parent (at `(px, py, pz)`, velocity `(pvx, pvy, pvz)`) - the first half of
    `thirdbody.third_body_kernel`.

    **Why reading `state` here is the reference's behaviour.** A perturber is massive, so it is never a
    Cowell body (Cowell refuses `mu != 0`), and a Cowell body's parent is never Cowell either (the
    plan's `parent_is_cowell` test). Neither row is written by any kernel in this file, so they hold
    their start-of-step values throughout, exactly the frozen rows the NumPy path hands
    `third_body_kernel` through `state` at every stage.

    `staged == 0`: the start-of-step relative position as is. `staged == 1`: carried along the two-body
    conic (`mu_s + mu_par`) by `elapsed = t - t0` with `t0` the engine-owned column 2 - **not** this
    kernel's own step-start argument, which under adaptive sub-stepping is a later sub-step's start
    while `t0` stays the arena step's. As the reference does, no advance happens at `elapsed == 0.0`.
    """
    s = int(third_params[row, _THIRD_PERTURBER_COL])
    mu_s = mu_array[s]
    rsx = state[s, 0] - px
    rsy = state[s, 1] - py
    rsz = state[s, 2] - pz
    if third_params[row, _THIRD_STAGED_COL] == 1.0:
        elapsed = t - third_params[row, _THIRD_T0_COL]
        if elapsed != 0.0:
            rsx, rsy, rsz, _vx, _vy, _vz = _kepler_advance_scalar(
                rsx, rsy, rsz, state[s, 3] - pvx, state[s, 4] - pvy, state[s, 5] - pvz,
                elapsed, mu_s + mu_par)
    return mu_s, rsx, rsy, rsz


@njit
def _third_body_term(
    ax: float, ay: float, az: float,
    px: float, py: float, pz: float, cx: float, cy: float, cz: float,
    mu_s: float, rsx: float, rsy: float, rsz: float,
) -> tuple[float, float, float]:
    """Add `mu_s [ (r_s - r)/|r_s - r|^3 - r_s/|r_s|^3 ]` to `(ax, ay, az)`, `r` the body relative to its
    parent: the second half of `thirdbody.third_body_kernel`, in its operation order. A zero separation
    contributes exactly 0.0 to its term, as the reference's `where=` does."""
    dx = rsx - (cx - px)
    dy = rsy - (cy - py)
    dz = rsz - (cz - pz)
    d2 = dx * dx + dy * dy + dz * dz
    s2 = rsx * rsx + rsy * rsy + rsz * rsz
    inv_d3 = 0.0
    inv_s3 = 0.0
    if d2 > 0.0:
        inv_d3 = 1.0 / (d2 * math.sqrt(d2))
    if s2 > 0.0:
        inv_s3 = 1.0 / (s2 * math.sqrt(s2))
    return (ax + mu_s * (dx * inv_d3 - rsx * inv_s3),
            ay + mu_s * (dy * inv_d3 - rsy * inv_s3),
            az + mu_s * (dz * inv_d3 - rsz * inv_s3))


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
    has_tesseral: bool, tesseral_params: NDArray[np.float64], t: float, vw: NDArray[np.float64],
    has_third: bool, third_params: NDArray[np.float64], state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Acceleration on one body at candidate state `(cx, cy, cz, cvx, cvy, cvz)` and time `t` with its
    parent at `(px, py, pz, pvx, pvy, pvz)`: the sum `forces.compose_accelerations` would build from
    `gravity.point_mass_gravity_kernel`, `geopotential.j2_kernel`, `drag.drag_kernel`,
    `zonal.zonal_kernel` and `tesseral.tesseral_kernel`, with each term's arithmetic written in the same
    order as its NumPy reference so the two agree to rounding. A disabled term contributes exactly
    `0.0`, as an absent model does.

    **Composition order matters with several terms.** IEEE addition is commutative but not associative,
    so `(pm + j2) + zonal` and `pm + (j2 + zonal)` may differ in the last bit. The terms are added here
    in registration order - `point_mass_gravity`, `j2`, `drag`, `zonal`, `tesseral` (bits 0, 1, 2, 6,
    7) - which is the order `forces.resolve_force_models` walks `registry.all_force_models()` and
    therefore the order `compose_accelerations` accumulates them into `out`. Were the registration
    order ever different, the two would differ by an ulp of the sum per evaluation, still far inside
    the 1e-12 bound.

    **Time.** Only the tesseral term reads `t` (the field turns with its body). `cowell_rk4_step` hands
    each stage its own time, `t`, `t + dt/2`, `t + dt/2`, `t + dt`, as `RK4Integrator` hands the
    provider.

    **This is the composition `cowell_rk4_step` performs**, stage by stage, from the same pieces in the
    same order - it calls them itself rather than this function only so that the density tables, zonal
    row and tesseral row are never passed through a per-stage call for a body that does not use them
    (measured: doing so tripled the per-body cost of the pm and j2 tiers). The field-level equivalence
    tests call this function; `test_cowell_*` hold the step to the same reference.

    `zonal_params[row]` is read only behind `has_zonal`, `tesseral_params[row]` and `vw` only behind
    `has_tesseral`, and the drag table only behind `has_drag`, so one-row dummies are safe when no body
    has those bits - the same convention as `j2_params` in `cowell_rk4_step`.
    """
    ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, has_point_mass, has_j2, j2, r_eq)
    if has_drag:
        ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                drag_law, drag_b, drag_rho0, drag_h0, drag_scale_height, drag_r_ref,
                                drag_omega, drag_table, tables, n_nodes)
    if has_zonal:
        ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, row)
    if has_tesseral:
        cos_t, sin_t = _tesseral_angle(tesseral_params, row, t)
        ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, tesseral_params, row,
                                    cos_t, sin_t, vw)
    if has_third:
        mu_s, rsx, rsy, rsz = _third_body_rel(state, mu_array, third_params, row, px, py, pz,
                                              pvx, pvy, pvz, mu_par, t)
        ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_s, rsx, rsy, rsz)
    return ax, ay, az


# Per-body model flags of `cowell_rk4_step`, one bit each in a single int64 array. Packed rather than
# five bool arrays because numba's per-call dispatch cost scales with the argument count (~0.1 us per
# array, measured), and it is paid on every tier whether or not the body uses the term. Compile-time
# constants under `@njit`; a bit is only ever tested with `& BIT`, never compared, so any other bits a
# caller sets are ignored.
COWELL_POINT_MASS = 1
COWELL_J2 = 2
COWELL_ZONAL = 4
COWELL_DRAG = 8
COWELL_TESSERAL = 16
COWELL_THIRD_BODY = 32


@njit
def cowell_rk4_step(
    dt: float,
    t: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    indices: NDArray[np.int64],
    flags: NDArray[np.int64],
    j2_params: NDArray[np.float64],
    zonal_params: NDArray[np.float64],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_tables: NDArray[np.float64],
    density_meta: NDArray[np.float64],
    tesseral_params: NDArray[np.float64],
    tesseral_vw: NDArray[np.float64],
    third_params: NDArray[np.float64],
    rel_out: NDArray[np.float64],
) -> int:
    """
    One classical RK4 step of every body in `indices` from absolute simulation time `t`, integrated
    relative to its `parent_indices` parent under any subset of `point_mass_gravity`, `j2`, `drag`,
    `zonal` and `tesseral` - the compiled twin of
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

    **Argument layout.** `flags[s]` is body `s`'s enabled terms as an OR of `COWELL_POINT_MASS`,
    `COWELL_J2`, `COWELL_ZONAL`, `COWELL_DRAG`, `COWELL_TESSERAL` (decoded with `&` once per body); it
    replaced five separate bool arrays, which changed only the argument count, never the arithmetic -
    results are bit-identical to the five-array signature. The coefficient arrays stay separate
    because they are the live `force_model_params` arrays the kernel reads without a copy; stacking
    them would need a per-step re-sync or a stale-coefficient hazard.

    Per-body flags rather than one global set so a mixed arena - some satellites with J2, some
    without, some with J3..J6 or drag - stays on the compiled path. `j2_params` is
    `force_model_params["j2"]` when any body has the J2 bit, and a one-row dummy otherwise; a row is
    only ever read behind its body's `has_j2`. `zonal_params` / `has_zonal` and `drag_params` /
    `has_drag` follow the same convention for `force_model_params["zonal"]` and `["drag"]`, and
    `tesseral_params` / `has_tesseral` for `["tesseral"]`. Scalar stage values live in registers, so
    unlike `RK4Integrator` this needs no stage scratch; the one array scratch is `tesseral_vw`, the
    caller-owned `(2, TESSERAL_VW_SIZE, TESSERAL_VW_SIZE)` V/W table the tesseral recursion fills per
    evaluation (arena-owned, `Simulation._cowell_tesseral_vw`), so the kernel allocates nothing.

    **Third body.** `third_params` is `force_model_params["third_body"]` (a one-row dummy when no
    body has the bit); a row is read only behind `COWELL_THIRD_BODY`. The perturber and parent rows of
    `state` are frozen at their start-of-step values across the stages (see `_third_body_rel`); a
    `staged == 1` row advances the perturber along its conic by `stage time - third_params[s, 2]`.
    It is added last, after the tesseral term, as `third_body` is registered after it.

    **Time.** `t` is the absolute simulation time at the start of this step - `Simulation.t` at the
    call, which after a manoeuvre or event split is the sub-step's own start, exactly the `t` the
    reference path hands `RK4Integrator.step`. Stage `k` is evaluated at `RK4Integrator`'s own
    expressions: `t`, `t + 0.5 * dt`, `t + 0.5 * dt`, `t + dt` (`half_dt` is `0.5 * dt`, so the
    mid-step time is the same double). Only the tesseral term reads it (drag's atmosphere is steady in
    the frame co-rotating with the parent); its rotation angle is computed once per distinct stage time,
    since stages 2 and 3 share one.
    """
    half_dt = 0.5 * dt
    sixth_dt = dt / 6.0
    n = indices.shape[0]
    # The four stage times, as `RK4Integrator.step` forms them.
    t1 = t
    t2 = t + half_dt
    t4 = t + dt

    # The staleness check, before anything is written - see "Return value".
    for k in range(n):
        s = indices[k]
        if (flags[s] & COWELL_DRAG) != 0 and _drag_law(drag_params[s, _DRAG_DENSITY_MODEL_COL]) == DRAG_LAW_MSIS:
            row = drag_table_of[s]
            if row < 0:
                return int(s)
            for c in range(3):
                if not (drag_params[s, _DRAG_F107_COL + c] == density_meta[row, TABLE_F107_COL + c]):
                    return int(s)

    for k in range(n):
        s = indices[k]
        par = parent_indices[s]
        fl = flags[s]
        pm = (fl & COWELL_POINT_MASS) != 0
        jj = (fl & COWELL_J2) != 0
        zz = (fl & COWELL_ZONAL) != 0
        dd = (fl & COWELL_DRAG) != 0
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
        # The tesseral field's orientation at the three distinct stage times, read once per body.
        tt = (fl & COWELL_TESSERAL) != 0
        th = (fl & COWELL_THIRD_BODY) != 0
        cos1 = 1.0
        sin1 = 0.0
        cos2 = 1.0
        sin2 = 0.0
        cos4 = 1.0
        sin4 = 0.0
        if tt:
            cos1, sin1 = _tesseral_angle(tesseral_params, s, t1)
            cos2, sin2 = _tesseral_angle(tesseral_params, s, t2)
            cos4, sin4 = _tesseral_angle(tesseral_params, s, t4)

        px = state[par, 0]
        py = state[par, 1]
        pz = state[par, 2]
        pvx = state[par, 3]
        pvy = state[par, 4]
        pvz = state[par, 5]

        # The perturber relative to the parent at the three distinct stage times, read once per body
        # (stages 2 and 3 share one). Frozen rows return the same start-of-step value at all three.
        mus = 0.0
        q1x = 0.0
        q1y = 0.0
        q1z = 0.0
        q2x = 0.0
        q2y = 0.0
        q2z = 0.0
        q4x = 0.0
        q4y = 0.0
        q4z = 0.0
        if th:
            mus, q1x, q1y, q1z = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                                 pvx, pvy, pvz, mu_par, t1)
            mus, q2x, q2y, q2z = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                                 pvx, pvy, pvz, mu_par, t2)
            mus, q4x, q4y, q4z = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                                 pvx, pvy, pvz, mu_par, t4)

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

        # Each stage below is `_cowell_accel`'s composition - gravity, then drag, then zonal, then
        # tesseral, in registration order - written out so that only a body with drag, zonal or
        # tesseral passes their tables. Stage k's tesseral angle is that of its own time (above).
        a1x, a1y, a1z = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            a1x, a1y, a1z = _drag_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, pvx, pvy, pvz,
                                       cvx, cvy, cvz, law, b, rho0, h0, scale_height, r_ref, omega,
                                       table, density_tables, n_nodes)
        if zz:
            a1x, a1y, a1z = _zonal_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        if tt:
            a1x, a1y, a1z = _tesseral_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, mu_par,
                                           tesseral_params, s, cos1, sin1, tesseral_vw)
        if th:
            a1x, a1y, a1z = _third_body_term(a1x, a1y, a1z, px, py, pz, cx, cy, cz, mus, q1x, q1y, q1z)
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
        if tt:
            a2x, a2y, a2z = _tesseral_term(a2x, a2y, a2z, px, py, pz, cx, cy, cz, mu_par,
                                           tesseral_params, s, cos2, sin2, tesseral_vw)
        if th:
            a2x, a2y, a2z = _third_body_term(a2x, a2y, a2z, px, py, pz, cx, cy, cz, mus, q2x, q2y, q2z)
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
        if tt:
            a3x, a3y, a3z = _tesseral_term(a3x, a3y, a3z, px, py, pz, cx, cy, cz, mu_par,
                                           tesseral_params, s, cos2, sin2, tesseral_vw)
        if th:
            a3x, a3y, a3z = _third_body_term(a3x, a3y, a3z, px, py, pz, cx, cy, cz, mus, q2x, q2y, q2z)
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
        if tt:
            a4x, a4y, a4z = _tesseral_term(a4x, a4y, a4z, px, py, pz, cx, cy, cz, mu_par,
                                           tesseral_params, s, cos4, sin4, tesseral_vw)
        if th:
            a4x, a4y, a4z = _third_body_term(a4x, a4y, a4z, px, py, pz, cx, cy, cz, mus, q4x, q4y, q4z)

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


# ==================================================================================================
# Compiled twins of the non-RK4 Cowell integrators: leapfrog, Yoshida 4, Encke
# ==================================================================================================
#
# Each is one kernel with `cowell_rk4_step`'s argument layout and return contract, and the same
# fused force composition (`_gravity_accel`, then `_drag_term`, `_zonal_term`, `_tesseral_term` behind
# their per-body flags, written in the step itself - see the engineering log on why a per-stage call
# through `_cowell_accel` costs the no-drag tiers). The per-body decoding block is repeated in each
# rather than factored into a helper for the same reason: a non-inlined call per body would cost as
# much as the whole pm+j2 body-step.

# Yoshida's triple-jump weights, the same expressions as `integrators._YOSHIDA_W1/_W0` (a test asserts
# they are bit-identical - `kernels` cannot import `integrators`, which sits above `forces`).
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


@njit
def _msis_stale_slot(
    indices: NDArray[np.int64],
    flags: NDArray[np.int64],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_meta: NDArray[np.float64],
) -> int:
    """The staleness check of `cowell_rk4_step`, verbatim: the first MSIS drag body whose live indices
    do not match its plan row (its slot), or -1. Run before any state is written."""
    for k in range(indices.shape[0]):
        s = indices[k]
        if (flags[s] & COWELL_DRAG) != 0 and _drag_law(drag_params[s, _DRAG_DENSITY_MODEL_COL]) == DRAG_LAW_MSIS:
            row = drag_table_of[s]
            if row < 0:
                return int(s)
            for c in range(3):
                if not (drag_params[s, _DRAG_F107_COL + c] == density_meta[row, TABLE_F107_COL + c]):
                    return int(s)
    return -1


@njit
def cowell_leapfrog_step(
    dt: float,
    t: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    indices: NDArray[np.int64],
    flags: NDArray[np.int64],
    j2_params: NDArray[np.float64],
    zonal_params: NDArray[np.float64],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_tables: NDArray[np.float64],
    density_meta: NDArray[np.float64],
    tesseral_params: NDArray[np.float64],
    tesseral_vw: NDArray[np.float64],
    third_params: NDArray[np.float64],
    rel_out: NDArray[np.float64],
) -> int:
    """
    One Stormer-Verlet (kick-drift-kick) step of every body in `indices`: the compiled twin of
    `integrators.LeapfrogIntegrator.step` driving `Simulation.accelerations`, fused with the
    subtraction `Simulation.step` applies afterwards. Argument layout, return value and state/`rel_out`
    contract are `cowell_rk4_step`'s. Two force evaluations per step: at the committed row at `t`
    (velocity included), and at `(parent + r', parent_v + v_half)` at `t + dt`, which the reference
    accumulates as `tau = t; tau += 1.0 * dt`. Scalars live in registers; the only scratch is the
    caller-owned tesseral V/W table.
    """
    stale = _msis_stale_slot(indices, flags, drag_params, drag_table_of, density_meta)
    if stale >= 0:
        return stale
    for k in range(indices.shape[0]):
        s = indices[k]
        par = parent_indices[s]
        fl = flags[s]
        pm = (fl & COWELL_POINT_MASS) != 0
        jj = (fl & COWELL_J2) != 0
        zz = (fl & COWELL_ZONAL) != 0
        dd = (fl & COWELL_DRAG) != 0
        tt = (fl & COWELL_TESSERAL) != 0
        th = (fl & COWELL_THIRD_BODY) != 0
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
        cx = state[s, 0]
        cy = state[s, 1]
        cz = state[s, 2]
        cvx = state[s, 3]
        cvy = state[s, 4]
        cvz = state[s, 5]
        rx = cx - px
        ry = cy - py
        rz = cz - pz
        vx = cvx - pvx
        vy = cvy - pvy
        vz = cvz - pvz

        # a(t, committed row): the first evaluation.
        ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                    law, b, rho0, h0, scale_height, r_ref, omega, table, density_tables, n_nodes)
        if zz:
            ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        if tt:
            cos_t, sin_t = _tesseral_angle(tesseral_params, s, t)
            ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par,
                                        tesseral_params, s, cos_t, sin_t, tesseral_vw)
        if th:
            mus, qx, qy, qz = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                              pvx, pvy, pvz, mu_par, t)
            ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mus, qx, qy, qz)

        # The one substep, weight 1.0.
        tau = t
        h = 1.0 * dt
        hh = 0.5 * h
        vhx = vx + hh * ax
        vhy = vy + hh * ay
        vhz = vz + hh * az
        rx = rx + h * vhx
        ry = ry + h * vhy
        rz = rz + h * vhz
        tau += h
        cx = px + rx
        cy = py + ry
        cz = pz + rz
        cvx = pvx + vhx
        cvy = pvy + vhy
        cvz = pvz + vhz
        ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                    law, b, rho0, h0, scale_height, r_ref, omega, table, density_tables, n_nodes)
        if zz:
            ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        if tt:
            cos_t, sin_t = _tesseral_angle(tesseral_params, s, tau)
            ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par,
                                        tesseral_params, s, cos_t, sin_t, tesseral_vw)
        if th:
            mus, qx, qy, qz = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                              pvx, pvy, pvz, mu_par, tau)
            ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mus, qx, qy, qz)
        vx = vhx + hh * ax
        vy = vhy + hh * ay
        vz = vhz + hh * az

        gx = px + rx
        gy = py + ry
        gz = pz + rz
        gvx = pvx + vx
        gvy = pvy + vy
        gvz = pvz + vz
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
def cowell_yoshida4_step(
    dt: float,
    t: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    indices: NDArray[np.int64],
    flags: NDArray[np.int64],
    j2_params: NDArray[np.float64],
    zonal_params: NDArray[np.float64],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_tables: NDArray[np.float64],
    density_meta: NDArray[np.float64],
    tesseral_params: NDArray[np.float64],
    tesseral_vw: NDArray[np.float64],
    third_params: NDArray[np.float64],
    rel_out: NDArray[np.float64],
) -> int:
    """
    One Yoshida fourth-order step (three kick-drift-kick substeps, weights `w1, w0, w1`) of every body
    in `indices`: the compiled twin of `integrators.Yoshida4Integrator.step`. Layout and contract are
    `cowell_rk4_step`'s. The acceleration ending one substep starts the next (first-same-as-last), so
    the step costs four evaluations, the first at the committed row. Substep times accumulate as the
    reference does (`tau = t; tau += h` per substep, `h = w * dt`).
    """
    stale = _msis_stale_slot(indices, flags, drag_params, drag_table_of, density_meta)
    if stale >= 0:
        return stale
    for k in range(indices.shape[0]):
        s = indices[k]
        par = parent_indices[s]
        fl = flags[s]
        pm = (fl & COWELL_POINT_MASS) != 0
        jj = (fl & COWELL_J2) != 0
        zz = (fl & COWELL_ZONAL) != 0
        dd = (fl & COWELL_DRAG) != 0
        tt = (fl & COWELL_TESSERAL) != 0
        th = (fl & COWELL_THIRD_BODY) != 0
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
        cx = state[s, 0]
        cy = state[s, 1]
        cz = state[s, 2]
        cvx = state[s, 3]
        cvy = state[s, 4]
        cvz = state[s, 5]
        rx = cx - px
        ry = cy - py
        rz = cz - pz
        vx = cvx - pvx
        vy = cvy - pvy
        vz = cvz - pvz

        ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
        if dd:
            ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                    law, b, rho0, h0, scale_height, r_ref, omega, table, density_tables, n_nodes)
        if zz:
            ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
        if tt:
            cos_t, sin_t = _tesseral_angle(tesseral_params, s, t)
            ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par,
                                        tesseral_params, s, cos_t, sin_t, tesseral_vw)
        if th:
            mus, qx, qy, qz = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                              pvx, pvy, pvz, mu_par, t)
            ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mus, qx, qy, qz)

        tau = t
        for sub in range(3):
            w = _YOSHIDA_W1
            if sub == 1:
                w = _YOSHIDA_W0
            h = w * dt
            hh = 0.5 * h
            vhx = vx + hh * ax
            vhy = vy + hh * ay
            vhz = vz + hh * az
            rx = rx + h * vhx
            ry = ry + h * vhy
            rz = rz + h * vhz
            tau += h
            cx = px + rx
            cy = py + ry
            cz = pz + rz
            cvx = pvx + vhx
            cvy = pvy + vhy
            cvz = pvz + vhz
            ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
            if dd:
                ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                        law, b, rho0, h0, scale_height, r_ref, omega, table, density_tables, n_nodes)
            if zz:
                ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
            if tt:
                cos_t, sin_t = _tesseral_angle(tesseral_params, s, tau)
                ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par,
                                            tesseral_params, s, cos_t, sin_t, tesseral_vw)
            if th:
                mus, qx, qy, qz = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                                  pvx, pvy, pvz, mu_par, tau)
                ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mus, qx, qy, qz)
            vx = vhx + hh * ax
            vy = vhy + hh * ay
            vz = vhz + hh * az

        gx = px + rx
        gy = py + ry
        gz = pz + rz
        gvx = pvx + vx
        gvy = pvy + vy
        gvz = pvz + vz
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
def _stumpff_scalar(z: float) -> tuple[float, float]:
    """Stumpff `C(z), S(z)` - `integrators._stumpff` for one value: the same series below `|z| = 1e-3`
    (so no cancellation), the same trigonometric / hyperbolic closed forms outside it."""
    if abs(z) < 1e-3:
        c = 0.5 - z / 24.0 + z * z / 720.0 - z * z * z / 40320.0
        s = 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z * z * z / 362880.0
    elif z > 0.0:
        sp = math.sqrt(z)
        c = (1.0 - math.cos(sp)) / z
        s = (sp - math.sin(sp)) / (sp * sp * sp)
    else:
        sn = math.sqrt(-z)
        c = (math.cosh(sn) - 1.0) / (-z)
        s = (math.sinh(sn) - sn) / (sn * sn * sn)
    return c, s


@njit
def _battin_f_scalar(q: float) -> float:
    """`integrators.battin_f` for one value: `(1 + q)^(3/2) - 1` without cancellation."""
    return q * (3.0 + 3.0 * q + q * q) / (1.0 + math.pow(1.0 + q, 1.5))


@njit
def _kepler_advance_scalar(
    r0x: float, r0y: float, r0z: float, v0x: float, v0y: float, v0z: float, dt: float, mu: float,
) -> tuple[float, float, float, float, float, float]:
    """
    `integrators.kepler_advance` for one body: the universal-variable Kepler equation by Newton to
    1e-12 in the universal anomaly (Curtis Alg. 3.3; bracketed and safeguarded, with Vallado's starting
    guess on a hyperbola: see the reference), f and g with their derivatives. Both stop each body at its own convergence (the reference freezes a
    converged body), so they take the same iterates. The `chi**3` of the
    reference is `chi * chi * chi` here (within an ulp of libm's `pow`). Raises `RuntimeError` after 100
    iterations, as the reference does.
    """
    rn = math.sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
    vr = (r0x * v0x + r0y * v0y + r0z * v0z) / rn
    alpha = 2.0 / rn - (v0x * v0x + v0y * v0y + v0z * v0z) / mu
    sq = math.sqrt(mu)
    a0 = rn * vr / sq
    b0 = 1.0 - alpha * rn
    hx = r0y * v0z - r0z * v0y
    hy = r0z * v0x - r0x * v0z
    hz = r0x * v0y - r0y * v0x
    p = (hx * hx + hy * hy + hz * hz) / mu
    r_p = p / (1.0 + math.sqrt(max(0.0, 1.0 - p * alpha)))
    bound = 1.01 * sq * abs(dt) / r_p if r_p > 0.0 else math.inf
    if dt < 0.0:
        lo, hi = -bound, 0.0
    else:
        lo, hi = 0.0, bound
    if alpha != 0.0:
        chi = sq * abs(alpha) * dt
    else:
        chi = sq * dt / rn
    if alpha < 0.0 and dt != 0.0:   # Vallado's hyperbolic guess, kept only where the reference keeps it
        sg = 1.0 if dt > 0.0 else -1.0
        den = rn * vr + sg * math.sqrt(-mu / alpha) * b0
        if den != 0.0:
            arg = -2.0 * mu * alpha * dt / den
            if arg > 0.0:
                chi_h = sg * math.sqrt(-1.0 / alpha) * math.log(arg)
                if math.isfinite(chi_h) and chi_h * dt > 0.0:
                    chi = chi_h
    chi = min(max(chi, lo), hi)
    dx_old = hi - lo
    dx = dx_old
    converged = False
    bisected = False
    for _ in range(100):
        z = alpha * chi * chi
        c, s = _stumpff_scalar(z)
        f_val = a0 * chi * chi * c + b0 * (chi * chi * chi) * s + rn * chi - sq * dt
        f_der = a0 * chi * (1.0 - z * s) + b0 * chi * chi * c + rn
        finite = math.isfinite(f_val)
        if (f_val < 0.0) if finite else (chi < 0.0):   # overflow: chi is too far out
            lo = chi
        if (f_val > 0.0) if finite else (chi > 0.0):
            hi = chi
        new = 0.5 * (lo + hi)
        bisected = True
        if finite:
            newton = chi - f_val / f_der
            if lo < newton < hi and abs(2.0 * f_val) <= abs(dx_old * f_der):
                new = newton
                bisected = False
        step = new - chi
        chi = new
        dx_old = dx
        dx = abs(step)
        if abs(step) <= 1e-12 * max(1.0, abs(chi)):
            converged = True
            break
    if not converged:
        raise RuntimeError("kepler_advance: universal Kepler equation did not converge in 100 iterations")
    if bisected:   # the reference's polish: a bisection stops at the bracket's width, not round-off
        z = alpha * chi * chi
        c, s = _stumpff_scalar(z)
        f_val = a0 * chi * chi * c + b0 * (chi * chi * chi) * s + rn * chi - sq * dt
        f_der = a0 * chi * (1.0 - z * s) + b0 * chi * chi * c + rn
        chi = chi - f_val / f_der
    z = alpha * chi * chi
    c, s = _stumpff_scalar(z)
    f = 1.0 - chi * chi / rn * c
    g = dt - (chi * chi * chi) * s / sq
    rx = f * r0x + g * v0x
    ry = f * r0y + g * v0y
    rz = f * r0z + g * v0z
    rr = math.sqrt(rx * rx + ry * ry + rz * rz)
    f_dot = sq / (rr * rn) * (z * chi * s - chi)
    g_dot = 1.0 - chi * chi / rr * c
    return (rx, ry, rz,
            f_dot * r0x + g_dot * v0x, f_dot * r0y + g_dot * v0y, f_dot * r0z + g_dot * v0z)


@njit
def cowell_encke_step(
    dt: float,
    t: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    indices: NDArray[np.int64],
    flags: NDArray[np.int64],
    j2_params: NDArray[np.float64],
    zonal_params: NDArray[np.float64],
    drag_params: NDArray[np.float64],
    drag_table_of: NDArray[np.int64],
    density_tables: NDArray[np.float64],
    density_meta: NDArray[np.float64],
    tesseral_params: NDArray[np.float64],
    tesseral_vw: NDArray[np.float64],
    third_params: NDArray[np.float64],
    rel_out: NDArray[np.float64],
) -> int:
    """
    One Encke step of every body in `indices`: the compiled twin of `integrators.EnckeIntegrator.step`.
    Layout and contract are `cowell_rk4_step`'s. The summed `mu = mu_array[s] + mu_array[parent]` is
    read here (it is the `mu_total` the point-mass term uses). The reference conic is advanced to
    `dt/2` and `dt` by `_kepler_advance_scalar`; the deviation `dy = (dr, dv)` is then integrated by RK4
    under `-(mu / rho^3)(f(q) r + dr) + a_p`, `a_p` the fused total acceleration plus `mu r / r^3`.
    The four stages are one loop body so the fused force composition appears once; each stage's `dy`
    is `0`, `(h/2) k1`, `(h/2) k2`, `h k3` and the weighted sum accumulates in the reference's order
    `((k1 + 2 k2) + 2 k3) + k4`. **Every body must carry the point-mass bit** - the caller checks
    (`Simulation.step`); a body without it would have its central term subtracted from nothing.
    A Kepler solve that fails to converge raises `RuntimeError` mid-loop, leaving the bodies before it
    already advanced; the reference raises before writing any state of the failing call.
    """
    stale = _msis_stale_slot(indices, flags, drag_params, drag_table_of, density_meta)
    if stale >= 0:
        return stale
    h = dt
    half = 0.5 * h
    for k in range(indices.shape[0]):
        s = indices[k]
        par = parent_indices[s]
        fl = flags[s]
        pm = (fl & COWELL_POINT_MASS) != 0
        jj = (fl & COWELL_J2) != 0
        zz = (fl & COWELL_ZONAL) != 0
        dd = (fl & COWELL_DRAG) != 0
        tt = (fl & COWELL_TESSERAL) != 0
        th = (fl & COWELL_THIRD_BODY) != 0
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
        r0x = state[s, 0] - px
        r0y = state[s, 1] - py
        r0z = state[s, 2] - pz
        v0x = state[s, 3] - pvx
        v0y = state[s, 4] - pvy
        v0z = state[s, 5] - pvz

        mrx, mry, mrz, mvx, mvy, mvz = _kepler_advance_scalar(r0x, r0y, r0z, v0x, v0y, v0z, half, mu_total)
        erx, ery, erz, evx, evy, evz = _kepler_advance_scalar(r0x, r0y, r0z, v0x, v0y, v0z, h, mu_total)

        # k0..k5 hold the previous stage's rate (dv, then dd), s0..s5 the weighted sum so far.
        k0 = 0.0
        k1 = 0.0
        k2 = 0.0
        k3 = 0.0
        k4 = 0.0
        k5 = 0.0
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        s4 = 0.0
        s5 = 0.0
        fac = 0.0
        for stage in range(4):
            # Stage inputs: time, reference point on the conic, and the deviation dy.
            if stage == 0:
                tau = t
                rrx = r0x
                rry = r0y
                rrz = r0z
                vrx = v0x
                vry = v0y
                vrz = v0z
                d0 = 0.0
                d1 = 0.0
                d2 = 0.0
                d3 = 0.0
                d4 = 0.0
                d5 = 0.0
            else:
                if stage == 3:
                    tau = t + h
                    rrx = erx
                    rry = ery
                    rrz = erz
                    vrx = evx
                    vry = evy
                    vrz = evz
                    fac = h
                else:
                    tau = t + half
                    rrx = mrx
                    rry = mry
                    rrz = mrz
                    vrx = mvx
                    vry = mvy
                    vrz = mvz
                    fac = 0.5 * h
                d0 = fac * k0
                d1 = fac * k1
                d2 = fac * k2
                d3 = fac * k3
                d4 = fac * k4
                d5 = fac * k5

            rx = rrx + d0
            ry = rry + d1
            rz = rrz + d2
            cx = px + rx
            cy = py + ry
            cz = pz + rz
            cvx = pvx + vrx + d3
            cvy = pvy + vry + d4
            cvz = pvz + vrz + d5
            ax, ay, az = _gravity_accel(px, py, pz, cx, cy, cz, mu_total, mu_par, pm, jj, j2, r_eq)
            if dd:
                ax, ay, az = _drag_term(ax, ay, az, px, py, pz, cx, cy, cz, pvx, pvy, pvz, cvx, cvy, cvz,
                                        law, b, rho0, h0, scale_height, r_ref, omega, table, density_tables, n_nodes)
            if zz:
                ax, ay, az = _zonal_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par, zonal_params, s)
            if tt:
                cos_t, sin_t = _tesseral_angle(tesseral_params, s, tau)
                ax, ay, az = _tesseral_term(ax, ay, az, px, py, pz, cx, cy, cz, mu_par,
                                            tesseral_params, s, cos_t, sin_t, tesseral_vw)
            if th:
                mus, qx, qy, qz = _third_body_rel(state, mu_array, third_params, s, px, py, pz,
                                                  pvx, pvy, pvz, mu_par, tau)
                ax, ay, az = _third_body_term(ax, ay, az, px, py, pz, cx, cy, cz, mus, qx, qy, qz)
            r2 = rx * rx + ry * ry + rz * rz
            rn = math.sqrt(r2)
            kc = mu_total / (r2 * rn)
            apx = ax + kc * rx
            apy = ay + kc * ry
            apz = az + kc * rz
            rho = math.sqrt(rrx * rrx + rry * rry + rrz * rrz)
            q = (d0 * (d0 - 2.0 * rx) + d1 * (d1 - 2.0 * ry) + d2 * (d2 - 2.0 * rz)) / r2
            coef = -(mu_total / (rho * rho * rho))
            bf = _battin_f_scalar(q)
            k0 = d3
            k1 = d4
            k2 = d5
            k3 = coef * (bf * rx + d0) + apx
            k4 = coef * (bf * ry + d1) + apy
            k5 = coef * (bf * rz + d2) + apz
            if stage == 0:
                s0 = k0
                s1 = k1
                s2 = k2
                s3 = k3
                s4 = k4
                s5 = k5
            elif stage == 3:
                s0 = s0 + k0
                s1 = s1 + k1
                s2 = s2 + k2
                s3 = s3 + k3
                s4 = s4 + k4
                s5 = s5 + k5
            else:
                s0 = s0 + 2.0 * k0
                s1 = s1 + 2.0 * k1
                s2 = s2 + 2.0 * k2
                s3 = s3 + 2.0 * k3
                s4 = s4 + 2.0 * k4
                s5 = s5 + 2.0 * k5

        sixth = h / 6.0
        gx = px + erx + sixth * s0
        gy = py + ery + sixth * s1
        gz = pz + erz + sixth * s2
        gvx = pvx + evx + sixth * s3
        gvy = pvy + evy + sixth * s4
        gvz = pvz + evz + sixth * s5
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
