"""
Tesseral and sectoral geopotential harmonics (order m >= 1, degrees 2..4), registered as `"tesseral"`.

**Additive to `"j2"` and `"zonal"`, never a replacement.** Those two carry every m = 0 term; this model
carries only the nine longitude-dependent pairs `(n, m)` = (2,1), (2,2), (3,1), (3,2), (3,3), (4,1),
(4,2), (4,3), (4,4). A 4x4 field is `"j2"` + `"zonal"` (J3, J4) + `"tesseral"`. The reasoning is the
zonal model's (`docs/architecture.md`): a model that also carried m = 0 would either double-count J2
or replace the one fast path the engine has.

**The engine's first force that depends on `t`.** The field is fixed in the rotating body, so in the
inertial frame it turns at `omega`. The kernel reads the evaluation time it is handed, and that is only
correct if every RK4 stage is handed its own time (`t`, `t + h/2`, `t + h/2`, `t + h`) and every
step - including a sub-step that starts mid-step after a manoeuvre or event split - starts from the
right clock. `integrators.RK4Integrator` and `Simulation._advance` do; freezing the rotation at the
step's start drops RK4 to first order, which `tests/validation/test_tesseral.py` demonstrates with a
convergence test and a frozen-time negative control.

Like `"j2"` this contributes **only the perturbation**, relative to each body's `parent_indices`
parent, with the parent's spin axis the frame's +z.

Rotation convention
-------------------
The body-fixed prime meridian's angle from the inertial +x axis is

    theta(t) = theta0 + omega * t,          t = absolute simulation time (`Simulation.t`), seconds

which is `geometry.elevation_azimuth`'s convention **with its default `epoch_s = 0.0`** (and *not*
`viz.ground_track`'s, which references `theta0` to its first sample time). East longitude in the body
frame is inertial longitude minus `theta`: a body-fixed vector is `Rz(-theta)` of the inertial one, and
the body-fixed acceleration is rotated back by `Rz(+theta)`. Getting that backwards makes the field
turn at `-omega`, which raises nothing; the co-rotating invariance test is what catches it.

Physics
-------
The gravitational potential (geodesy sign, `a = +grad U`) of one tesseral term, unnormalised
coefficients, `P_nm` the associated Legendre function **without** the Condon-Shortley phase:

    U_nm = (mu / r) (R / r)^n P_nm(sin phi) [C_nm cos(m lambda) + S_nm sin(m lambda)]

with `phi` geocentric latitude and `lambda` body-fixed east longitude. The zonal part is
`C_n0 = -J_n`; that is `"j2"` / `"zonal"`'s job. Following Cunningham (1970), as presented by
Montenbruck & Gill, write

    V_nm = (R/r)^(n+1) P_nm(sin phi) cos(m lambda),    W_nm = (R/r)^(n+1) P_nm(sin phi) sin(m lambda)

so `U = (mu / R) sum (C_nm V_nm + S_nm W_nm)`. They satisfy, in body-fixed Cartesian `(x, y, z)`,

    V_00 = R / r,  W_00 = 0
    V_mm = (2m - 1) [ (x R / r^2) V_{m-1,m-1} - (y R / r^2) W_{m-1,m-1} ]
    W_mm = (2m - 1) [ (x R / r^2) W_{m-1,m-1} + (y R / r^2) V_{m-1,m-1} ]
    V_nm = [ (2n - 1) (z R / r^2) V_{n-1,m} - (n + m - 1) (R^2 / r^2) V_{n-2,m} ] / (n - m)     (n > m)

(W likewise, with `V_{m-1,m} = 0`), and the gradient of one term is, for m >= 1,

    a_x = (mu / R^2) (1/2) [ -C V_{n+1,m+1} - S W_{n+1,m+1}
                             + (n-m+2)(n-m+1) ( C V_{n+1,m-1} + S W_{n+1,m-1}) ]
    a_y = (mu / R^2) (1/2) [ -C W_{n+1,m+1} + S V_{n+1,m+1}
                             + (n-m+2)(n-m+1) (-C W_{n+1,m-1} + S V_{n+1,m-1}) ]
    a_z = (mu / R^2) (n - m + 1) [ -C V_{n+1,m} - S W_{n+1,m} ]                              (T)

which is what the kernel evaluates (degree 5, order 5 of V/W are needed for the degree-4 terms). It
is regular everywhere off the origin, poles included - there is no `1/cos(phi)`.

Citation
--------
Montenbruck, O. & Gill, E., *Satellite Orbits* (2000), Section 3.2.4-3.2.5: the V/W recursions and the
Cartesian acceleration (their Eqs. 3.29-3.33 as best remembered); Cunningham, L. E. (1970), *Celestial
Mechanics* 2, 207-216. **Section and equation numbers are from memory and unverified.** What checks
the kernel is `tests/validation/test_tesseral.py`: against `reference.tesseral_field` (explicit
`d^m P_n/ds^m` polynomials times complex powers `((x + i y) e^{-i theta})^m`, differentiated
monomial by monomial - no recursion, no rotation matrix), against a finite-differenced potential
built on a third evaluation (`numpy.polynomial.legendre` derivatives times `cos(phi)^m`), and against
a closed-form equatorial value.

Coefficients
------------
**Unnormalised**, like `zonal.py`'s `J_n`. `EGM96_CS_BAR` holds EGM96's fully normalised
`(Cbar_nm, Sbar_nm)` (Lemoine et al. 1998, NASA/TP-1998-206861), **quoted from memory and unverified
against the document**; `unnormalise(n, m, value)` applies

    C_nm = N_nm Cbar_nm,    N_nm = sqrt( (2 - delta_m0) (2n + 1) (n - m)! / (n + m)! )

the factor `zonal.py` uses at m = 0 (`J_n = -sqrt(2n + 1) Cbar_n0`). What guards the table: the
unnormalised results agree with the JGM-3 unnormalised values tabulated by Montenbruck & Gill
(Table 3.2, also from memory) to within the ~1e-3 by which two modern fields differ in these terms -
a normalisation error is a factor of sqrt(2) or more - and `J22 = 1.8155e-6`, `lambda22 = -14.93 deg`
match the widely quoted figures. `EARTH_TESSERALS` is the full 4x4 keyword set, `EARTH_J22` the (2,2)
pair alone; pair them with `geopotential.EARTH_R_EQ` and `drag.EARTH_OMEGA`, never
`scenarios.EARTH_RADIUS`.

Expected magnitudes
-------------------
Per term, the natural scale is `mu |C_nm, S_nm| R^n / r^(n+2)`, and `|a_nm|` reaches up to ~9x it for
(2,2) and ~500x for (4,4) (`P_nm` itself does). At LEO (r = 6921 km) the (2,2) scale is 1.3e-8 km/s^2
(J2's field is 1.1e-5..2.3e-5); at GEO (r = 42165 km) its equatorial tangential component,
`-(mu / a^2)(R / a)^2 6 J22 sin 2(lambda - lambda22)`, peaks at 5.6e-11 km/s^2 and is what drives the
longitude drift: `lambda_ddot = -3 a_S / a = +18 n^2 (R/a)^2 J22 sin 2(lambda - lambda22)`, at most
3.98e-15 rad/s^2 = 1.70e-3 deg/day^2 and 1.76 m/s per year of east-west station keeping.

Design decisions
----------------
**Coefficients on the perturbed body's row**, `param_names = ("r_eq", "omega", "theta0", "c21",
"s21", ..., "c44", "s44")`, for the reasons `geopotential.py` gives. `r_eq` is stored again rather
than read from `"j2"`'s row, since each model reads only its own `params`.

**Zero means absent, exactly.** Every term is a coefficient times a finite `V`/`W`, so an all-zero row
adds exact zeros. Selecting "J22 only" is `**EARTH_J22` with the other pairs left at zero.

**Refusals.** `validate_bodies`: a barycentre parent (no shape). `validate_coefficients`: a non-finite
coefficient; `r_eq <= 0` on a row whose effective coefficients are non-zero (the terms scale as
`R^n`); and **`omega == 0`** on such a row. A non-rotating tesseral body is physically possible but
the unwritten column reads 0.0, so "forgot to pass `omega`" would silently freeze the Earth's field
in inertial space - wrong by one full turn a day, and raising nothing.

**Not in the fused compiled Cowell plan.** A Cowell body with `"tesseral"` sends the whole Cowell set
down the NumPy `RK4Integrator` path (`Simulation._cowell_fused_ok` rejects the bit as foreign). A
compiled twin would need `t` threaded into `kernels._cowell_accel`, which does not take it today.
"""
from __future__ import annotations

import math
from typing import Final, Mapping, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .geopotential import barycentre_parented
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "TESSERAL_MODEL", "TESSERAL_PARAM_NAMES", "TESSERAL_PAIRS", "EGM96_CS_BAR",
    "normalisation_factor", "unnormalise", "EARTH_TESSERALS", "EARTH_J22",
    "j22_amplitude_and_longitude", "tesseral_kernel",
]

TESSERAL_MODEL: Final[str] = "tesseral"

# (n, m) in column order. Every m >= 1 of degrees 2..4.
TESSERAL_PAIRS: Final[Tuple[Tuple[int, int], ...]] = (
    (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (4, 4),
)
_R_EQ_COL: Final[int] = 0
_OMEGA_COL: Final[int] = 1
_THETA0_COL: Final[int] = 2
_FIRST_CS_COL: Final[int] = 3        # C of pair j in column 3 + 2j, S in 4 + 2j
_MAX_DEGREE: Final[int] = 4

TESSERAL_PARAM_NAMES: Final[Tuple[str, ...]] = ("r_eq", "omega", "theta0") + tuple(
    f"{cs}{n}{m}" for n, m in TESSERAL_PAIRS for cs in ("c", "s")
)

# EGM96 fully normalised (Cbar_nm, Sbar_nm), Lemoine et al. 1998. From memory, unverified - see the
# module docstring for what checks them.
EGM96_CS_BAR: Final[Mapping[Tuple[int, int], Tuple[float, float]]] = {
    (2, 1): (-0.186987635955e-9, 0.119528012031e-8),
    (2, 2): (0.243914352398e-5, -0.140016683654e-5),
    (3, 1): (0.203046201047e-5, 0.248513158716e-6),
    (3, 2): (0.904787894809e-6, -0.619025944205e-6),
    (3, 3): (0.721321757121e-6, 0.141435626958e-5),
    (4, 1): (-0.536157389388e-6, -0.473567346518e-6),
    (4, 2): (0.350501623962e-6, 0.662480026275e-6),
    (4, 3): (0.990856766672e-6, -0.200956723567e-6),
    (4, 4): (-0.188560802735e-6, 0.308803882149e-6),
}


def normalisation_factor(n: int, m: int) -> float:
    """`N_nm = sqrt((2 - delta_m0)(2n + 1)(n - m)!/(n + m)!)`: unnormalised = N_nm * normalised."""
    if not 0 <= m <= n:
        raise ValueError(f"need 0 <= m <= n, got n={n}, m={m}")
    k = 1.0 if m == 0 else 2.0
    return math.sqrt(k * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m))


def unnormalise(n: int, m: int, value: float) -> float:
    """Unnormalised `C_nm` (or `S_nm`) from a fully normalised coefficient."""
    return normalisation_factor(n, m) * value


# The keyword sets a configuration passes, e.g.
# `sim.enable_force_model("tesseral", sats, r_eq=EARTH_R_EQ, omega=EARTH_OMEGA, **EARTH_TESSERALS)`.
EARTH_TESSERALS: Final[Mapping[str, float]] = {
    f"{cs}{n}{m}": unnormalise(n, m, EGM96_CS_BAR[(n, m)][k])
    for n, m in TESSERAL_PAIRS for k, cs in enumerate(("c", "s"))
}
EARTH_J22: Final[Mapping[str, float]] = {"c22": EARTH_TESSERALS["c22"], "s22": EARTH_TESSERALS["s22"]}


def j22_amplitude_and_longitude(c22: float, s22: float) -> Tuple[float, float]:
    """
    `(J22, lambda22)` with `C22 = J22 cos(2 lambda22)`, `S22 = J22 sin(2 lambda22)` (unnormalised), so
    the (2,2) potential is `(mu/r)(R/r)^2 3 J22 cos^2(phi) cos 2(lambda - lambda22)`. `lambda22` is
    the longitude of the equator's **long axis** (the potential maximum); the GEO stable points are
    `lambda22 + 90 deg` and `lambda22 + 270 deg`. Earth (EGM96): 1.8155e-6, -14.93 deg.
    """
    return math.hypot(c22, s22), 0.5 * math.atan2(s22, c22)


def _reject_barycentre_parents(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook: refuse bodies whose Keplerian parent is a barycentre."""
    offending = barycentre_parented(sim.is_system, sim.parent_indices, bodies)
    if offending.size > 0:
        raise ValueError(
            f"force model '{TESSERAL_MODEL}' is meaningless for bodies whose parent is a barycentre (a "
            f"barycentre has no shape to expand); slot(s) {offending.tolist()} qualify. See "
            f"tesseral.py's module docstring."
        )


def _validate_tesseral_coefficients(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """
    `validate_coefficients` hook: every passed coefficient finite; and on every body whose *effective*
    row (passed values over stored ones) carries a non-zero C/S, `r_eq > 0` and `omega != 0`. See the
    module docstring for why each must raise rather than run.
    """
    for key, value in coefficients.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"force model '{TESSERAL_MODEL}': {key}={value!r} is not finite")

    stored = sim.force_model_params.get(TESSERAL_MODEL)

    def effective(col: int) -> NDArray[np.float64]:
        name = TESSERAL_PARAM_NAMES[col]
        if name in coefficients:
            return np.full(bodies.shape, float(coefficients[name]))
        if stored is None:
            return np.zeros(bodies.shape)
        column: NDArray[np.float64] = stored[bodies, col]
        return column

    has_field = np.zeros(bodies.shape, dtype=np.bool_)
    for col in range(_FIRST_CS_COL, len(TESSERAL_PARAM_NAMES)):
        has_field |= effective(col) != 0.0
    bad_r = bodies[has_field & ~(effective(_R_EQ_COL) > 0.0)]
    if bad_r.size > 0:
        raise ValueError(
            f"force model '{TESSERAL_MODEL}': slot(s) {bad_r.tolist()} carry a non-zero C_nm/S_nm with "
            f"r_eq <= 0; the terms scale as r_eq^n. Pass r_eq (e.g. geopotential.EARTH_R_EQ)."
        )
    bad_w = bodies[has_field & (effective(_OMEGA_COL) == 0.0)]
    if bad_w.size > 0:
        raise ValueError(
            f"force model '{TESSERAL_MODEL}': slot(s) {bad_w.tolist()} carry a non-zero C_nm/S_nm with "
            f"omega = 0, which freezes the body's field in inertial space. Pass omega (e.g. "
            f"drag.EARTH_OMEGA)."
        )


@register_force_model(
    TESSERAL_MODEL,
    param_names=TESSERAL_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    validate_coefficients=_validate_tesseral_coefficients,
    citation=(
        "Montenbruck & Gill, Satellite Orbits (2000), Sec. 3.2.4-3.2.5, Eqs. 3.29-3.33 (Cunningham "
        "1970 V/W recursions and Cartesian acceleration; numbers from memory, unverified); orders "
        "m >= 1 of degrees 2..4, unnormalised C_nm/S_nm, field rotating at omega about +z"
    ),
)
def tesseral_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add the tesseral/sectoral acceleration of each body's Keplerian parent to `out[indices]`, km/s^2.

    For body `i` with parent `P`, `rel = state[i,:3] - state[P,:3]`, `mu = mu_array[P]`: rotate `rel`
    into the body frame by `-theta`, `theta = theta0 + omega * t`, evaluate equation (T) of the module
    docstring summed over the nine `(n, m)` pairs, and rotate the result back by `+theta`.

    **`t` is used**, and must be each RK stage's own time. A root body (zero separation) or an all-zero
    row contributes exactly `0.0`, with no division evaluated.
    """
    primaries = parent_indices[indices]
    rel = state[indices, :3] - state[primaries, :3]
    r_eq = params[indices, _R_EQ_COL]

    theta = params[indices, _THETA0_COL] + params[indices, _OMEGA_COL] * float(t)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Inertial -> body-fixed: Rz(-theta).
    xb = cos_t * rel[:, 0] + sin_t * rel[:, 1]
    yb = -sin_t * rel[:, 0] + cos_t * rel[:, 1]
    zb = rel[:, 2]

    r2 = xb * xb + yb * yb + zb * zb
    inv_r2 = np.divide(1.0, r2, out=np.zeros_like(r2), where=r2 > 0.0)
    f = r_eq * inv_r2                                   # R / r^2
    rr = r_eq * f                                       # R^2 / r^2

    # V_nm, W_nm for n <= 5, m <= n: the degree-4 accelerations read degree 5.
    top = _MAX_DEGREE + 1
    v: dict[Tuple[int, int], NDArray[np.float64]] = {}
    w: dict[Tuple[int, int], NDArray[np.float64]] = {}
    v[(0, 0)] = r_eq * np.sqrt(inv_r2)                  # R / r
    w[(0, 0)] = np.zeros_like(r2)
    for m in range(top + 1):
        if m > 0:
            vp, wp = v[(m - 1, m - 1)], w[(m - 1, m - 1)]
            v[(m, m)] = (2 * m - 1) * (xb * f * vp - yb * f * wp)
            w[(m, m)] = (2 * m - 1) * (xb * f * wp + yb * f * vp)
        if m + 1 <= top:
            v[(m + 1, m)] = (2 * m + 1) * zb * f * v[(m, m)]
            w[(m + 1, m)] = (2 * m + 1) * zb * f * w[(m, m)]
        for n in range(m + 2, top + 1):
            v[(n, m)] = ((2 * n - 1) * zb * f * v[(n - 1, m)] - (n + m - 1) * rr * v[(n - 2, m)]) / (n - m)
            w[(n, m)] = ((2 * n - 1) * zb * f * w[(n - 1, m)] - (n + m - 1) * rr * w[(n - 2, m)]) / (n - m)

    ax = np.zeros_like(r2)
    ay = np.zeros_like(r2)
    az = np.zeros_like(r2)
    for j, (n, m) in enumerate(TESSERAL_PAIRS):
        c = params[indices, _FIRST_CS_COL + 2 * j]
        s = params[indices, _FIRST_CS_COL + 2 * j + 1]
        fac = (n - m + 2) * (n - m + 1)
        vu, wu = v[(n + 1, m + 1)], w[(n + 1, m + 1)]
        vd, wd = v[(n + 1, m - 1)], w[(n + 1, m - 1)]
        ax += 0.5 * ((-c * vu - s * wu) + fac * (c * vd + s * wd))
        ay += 0.5 * ((-c * wu + s * vu) + fac * (-c * wd + s * vd))
        az += (n - m + 1) * (-c * v[(n + 1, m)] - s * w[(n + 1, m)])

    inv_r_eq2 = np.divide(1.0, r_eq * r_eq, out=np.zeros_like(r_eq), where=r_eq > 0.0)
    k = mu_array[primaries] * inv_r_eq2                 # mu / R^2
    # Body-fixed -> inertial: Rz(+theta).
    out[indices, 0] += k * (cos_t * ax - sin_t * ay)
    out[indices, 1] += k * (sin_t * ax + cos_t * ay)
    out[indices, 2] += k * az
