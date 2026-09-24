"""
Higher zonal harmonics J3..J6 of the parent body, registered as the force model `"zonal"`.

**Additive to `"j2"`, never a replacement.** Degrees 3 to 6 only. J2 stays where it is: `"j2"` keeps
its fused compiled Cowell twin, and `PropagatorType.SECULAR_J2` keeps reading its coefficients from
`force_model_params["j2"]`. A configuration that wants J2..J6 enables both models. The reasoning is in
`docs/architecture.md`'s zonal section; in short, a J2..Jn model would have forced either a second
copy of J2 (and a new way to double-count it) or a rewrite of the one fast path the engine has.

Like `"j2"` this contributes **only the perturbation**, relative to each body's `parent_indices`
parent, with the parent's spin (symmetry) axis taken to be the frame's +z. Same frame caveat as
`geopotential.py`'s module docstring: correct in an Earth-equatorial frame, a tilted Earth in an
ecliptic one.

Physics
-------
The zonal part of the geopotential, as potential energy per unit mass (physicist's sign, `a = -grad
Phi`, the same convention as `geopotential.j2_potential`), with `s = z / r = sin(latitude)`:

    Phi_n(r) = + mu J_n R^n P_n(s) / r^(n+1)

With `grad r^-(n+1) = -(n+1) r^-(n+2) r_hat` and `grad s = (z_hat - s r_hat) / r`,

    a_n = -grad Phi_n = (mu J_n R^n / r^(n+2)) [ ((n+1) P_n(s) + s P_n'(s)) r_hat  -  P_n'(s) z_hat ]

and the Legendre identity `P_{n+1}'(s) = (n+1) P_n(s) + s P_n'(s)` collapses the radial bracket:

    a_n = (mu / r^2) J_n (R/r)^n [ P_{n+1}'(s) r_hat  -  P_n'(s) z_hat ]                        (Z)

That is what the kernel evaluates, for every degree at once, from two three-term recursions
(Bonnet's, and the same identity shifted down one degree):

    n P_n = (2n - 1) s P_{n-1} - (n - 1) P_{n-2},      P_0 = 1, P_1 = s
    P_n'  = n P_{n-1} + s P_{n-1}',                    P_0' = 0, P_1' = 1

Check at n = 2: `P_3' = (15 s^2 - 3)/2` and `P_2' = 3 s` give `(3 mu J2 R^2 / r^4)[(5 s^2 - 1)/2 r_hat -
s z_hat]`, which is `reference.py`'s equation (*) and expands to Curtis' Cartesian J2 form - so (Z) is
the general-degree statement of the J2 kernel the engine already trusts. It is not evaluated for
n = 2 here; that is `"j2"`'s job.

Parity: `P_n(-s) = (-1)^n P_n(s)`, so under `z -> -z` an odd degree's x, y components flip sign and
its z component does not, and an even degree's the reverse. Every operation in the recursion is
sign-symmetric in IEEE arithmetic, so the kernel reproduces this **bit for bit**, per degree.

Citation
--------
Montenbruck, O. & Gill, E., *Satellite Orbits* (2000), Section 3.2: the zonal expansion
`U = (GM/r) [1 - sum_n J_n (R/r)^n P_n(sin phi)]` (their Eq. 3.28 for the expansion) and the Legendre
recursions (their Eq. 3.30 family). Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
Section 8.7, derives the general geopotential acceleration through spherical partials. **Section and
equation numbers are from memory and unverified.** The six-line derivation above is self-contained
and is what should be checked; `tests/validation/test_zonal.py` checks the kernel against an
independently written field (`reference.zonal_field`, monomial-by-monomial Cartesian differentiation
of the explicit `P_n`) and against a finite-differenced potential.

Earth coefficients
------------------
`EARTH_J3`..`EARTH_J6` are EGM96 unnormalised zonals, `J_n = -sqrt(2n + 1) * Cbar_n0` (the fully
normalised `Cbar_n0` of Lemoine et al. 1998, NASA/TP-1998-206861, kept in `EGM96_CBAR_N0`). The
normalised values are **quoted from memory and unverified against that document**. Two things guard
them: the same table's `Cbar_20` must reproduce `geopotential.EARTH_J2` to its printed digits (it
does, to 5.5e-14 absolute - `EARTH_J2` is that value *truncated*, not rounded, at its eleventh digit),
and the resulting `J3 = -2.5327e-6`, `J4 = -1.6196e-6`, `J5 = -2.2730e-7`, `J6 = +5.4068e-7` match the
widely quoted magnitudes and signs. They are a matched set with `geopotential.EARTH_R_EQ` =
6378.137 km and `scenarios.MU_EARTH`, like `EARTH_J2`; never pair them with `scenarios.EARTH_RADIUS`.

Expected magnitudes
-------------------
At 550 km (r = 6921 km), per degree over all latitudes: J3 2.5e-8..6.6e-8 km/s^2, J4
1.7e-8..4.9e-8, J5 2.4e-9..7.5e-9, J6 5.7e-9..1.9e-8 - between 1e-3 and 3e-3 of J2's 1.1e-5..2.3e-5.
None of them vanishes anywhere on the sphere (the zeros of `P_{n+1}'` and `P_n'` interlace), so a
per-degree relative error is well defined at every latitude.

Design decisions
----------------
**Coefficients on the perturbed body's row**, `param_names = ("r_eq", "j3", "j4", "j5", "j6")`, for
exactly the reasons `geopotential.py` gives for `"j2"`. `r_eq` is stored again rather than read from
`force_model_params["j2"]`: a kernel reads only its own `params`, and the two models must be
independently enableable.

**Zero means absent, exactly.** Each degree's term is `J_n * (R/r)^n * P'`, so a zero `J_n` adds an
exact `0.0` to the accumulators and an all-zero row adds an exact `0.0` to `out`. Enabling `"zonal"`
without coefficients is therefore the same silent no-op as `"j2"` without coefficients - pass them.
Selecting "J3 only" is `j3=EARTH_J3` with the rest left at zero, which is what the differential
validation cases do.

**Refusals, and why.** `validate_bodies` is `geopotential.barycentre_parented` (a barycentre has no
shape; the kernel cannot see `is_system`). `validate_coefficients` refuses a non-finite coefficient
(one NaN poisons the whole acceleration row and raises nothing) and `r_eq <= 0` on a body whose
effective row - passed values over already-stored ones - carries any non-zero `J_n`: with `r_eq = 0`
every term is multiplied by `0^n` and the model silently contributes nothing, with `r_eq < 0` the odd
degrees silently flip sign.

**In the fused compiled Cowell plan.** `kernels.cowell_rk4_step` fuses this term with
`point_mass_gravity` and `j2` (`kernels._cowell_accel`, per-body `has_zonal` flag, the same recursions
in the same order), so a Cowell body whose models are a subset of those three stays on the compiled
path and its wall time is comparable with the other fused tiers. This function is the reference: the
twin is held to it in `tests/validation/test_kernel_equivalence.py`, and a change here must be made
there too. A zonal body that also carries any other model still sends the whole Cowell set down the
NumPy `RK4Integrator` path.
"""
from __future__ import annotations

import math
from typing import Final, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .geopotential import barycentre_parented
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "ZONAL_MODEL", "ZONAL_PARAM_NAMES", "ZONAL_DEGREES", "EGM96_CBAR_N0",
    "EARTH_J3", "EARTH_J4", "EARTH_J5", "EARTH_J6", "EARTH_ZONALS",
    "zonal_kernel",
]

ZONAL_MODEL: Final[str] = "zonal"
ZONAL_PARAM_NAMES: Final = ("r_eq", "j3", "j4", "j5", "j6")
ZONAL_DEGREES: Final = (3, 4, 5, 6)
_R_EQ_COL: Final[int] = 0
_FIRST_J_COL: Final[int] = 1          # column of J3; J_n sits in column n - 2

# EGM96 fully normalised zonal coefficients Cbar_n0 (Lemoine et al. 1998). From memory, unverified -
# see the module docstring for what checks them. Degree 2 is here only so the J2 cross-check can be
# made from the same table; `"zonal"` never uses it.
EGM96_CBAR_N0: Final[Mapping[int, float]] = {
    2: -0.484165371736e-3,
    3: 0.957254173792e-6,
    4: 0.539873863789e-6,
    5: 0.685323475630e-7,
    6: -0.149957994714e-6,
}

EARTH_J3: Final[float] = -math.sqrt(7.0) * EGM96_CBAR_N0[3]     # -2.5326564853e-6
EARTH_J4: Final[float] = -math.sqrt(9.0) * EGM96_CBAR_N0[4]     # -1.6196215914e-6
EARTH_J5: Final[float] = -math.sqrt(11.0) * EGM96_CBAR_N0[5]    # -2.2729608287e-7
EARTH_J6: Final[float] = -math.sqrt(13.0) * EGM96_CBAR_N0[6]    # +5.4068123911e-7

# The keyword set a full J3..J6 Earth configuration passes, e.g.
# `sim.enable_force_model("zonal", sats, r_eq=EARTH_R_EQ, **EARTH_ZONALS)`.
EARTH_ZONALS: Final[Mapping[str, float]] = {
    "j3": EARTH_J3, "j4": EARTH_J4, "j5": EARTH_J5, "j6": EARTH_J6,
}


def _reject_barycentre_parents(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook for `"zonal"`: refuse bodies whose Keplerian parent is a barycentre."""
    offending = barycentre_parented(sim.is_system, sim.parent_indices, bodies)
    if offending.size > 0:
        raise ValueError(
            f"force model '{ZONAL_MODEL}' is meaningless for bodies whose parent is a barycentre (a "
            f"barycentre has no shape to expand); slot(s) {offending.tolist()} qualify. See "
            f"zonal.py's module docstring."
        )


def _validate_zonal_coefficients(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """
    `validate_coefficients` hook for `"zonal"`: every passed coefficient finite, and `r_eq > 0` on any
    body whose *effective* row carries a non-zero `J_n`. Effective means the value being passed if it
    is, otherwise what the row already holds - so enabling `j4` on a body whose `r_eq` was written by
    an earlier call is accepted, and passing `r_eq=0.0` over stored non-zero `J_n` is refused. See the
    module docstring for why each case must raise rather than run.
    """
    for key, value in coefficients.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"force model '{ZONAL_MODEL}': {key}={value!r} is not finite")

    stored = sim.force_model_params.get(ZONAL_MODEL)

    def effective(col: int) -> NDArray[np.float64]:
        name = ZONAL_PARAM_NAMES[col]
        if name in coefficients:
            return np.full(bodies.shape, float(coefficients[name]))
        if stored is None:
            return np.zeros(bodies.shape)
        column: NDArray[np.float64] = stored[bodies, col]
        return column

    has_zonal = np.zeros(bodies.shape, dtype=np.bool_)
    for col in range(_FIRST_J_COL, len(ZONAL_PARAM_NAMES)):
        has_zonal |= effective(col) != 0.0
    offending = bodies[has_zonal & ~(effective(_R_EQ_COL) > 0.0)]
    if offending.size > 0:
        raise ValueError(
            f"force model '{ZONAL_MODEL}': slot(s) {offending.tolist()} carry a non-zero J_n with "
            f"r_eq <= 0; the zonal terms scale as r_eq^n, so r_eq = 0 silently disables them and "
            f"r_eq < 0 silently flips the odd degrees. Pass r_eq (e.g. geopotential.EARTH_R_EQ)."
        )


@register_force_model(
    ZONAL_MODEL,
    param_names=ZONAL_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    validate_coefficients=_validate_zonal_coefficients,
    citation=(
        "Montenbruck & Gill, Satellite Orbits (2000), Sec. 3.2, Eq. 3.28 (zonal expansion) and the "
        "Legendre recursions (numbers from memory, unverified); derived in zonal.py as -grad of "
        "Phi_n = mu J_n R^n P_n(z/r) / r^(n+1), n = 3..6"
    ),
)
def zonal_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add the J3..J6 perturbing acceleration of each body's Keplerian parent to `out[indices]`, km/s^2.

    For body `i` with parent `P = parent_indices[i]`, `rel = state[i,:3] - state[P,:3]`,
    `mu = mu_array[P]`, `R = params[i,0]`, `J_n = params[i, n-2]`, equation (Z) of the module
    docstring summed over n = 3..6:

        a = (mu / r^2) sum_n J_n (R/r)^n [ P_{n+1}'(s) r_hat - P_n'(s) z_hat ],     s = z / r

    A root body (zero separation) contributes exactly `0.0`, with no division evaluated, as in
    `geopotential.j2_kernel`. `t` is unused.
    """
    primaries = parent_indices[indices]
    rel = state[indices, :3] - state[primaries, :3]
    r2 = np.einsum("ij,ij->i", rel, rel)

    # inv_r = 0 at zero separation zeroes s, R/r and mu/r^2 below, so the row adds exactly 0.0.
    inv_r2 = np.divide(1.0, r2, out=np.zeros_like(r2), where=r2 > 0.0)
    inv_r = np.sqrt(inv_r2)
    s = rel[:, 2] * inv_r
    rho = params[indices, _R_EQ_COL] * inv_r                          # R / r

    # Legendre values and derivatives, degree 0 and 1, then climb. At each n the loop holds
    # P_{n-1}, P_{n-2}, P_{n-1}' and needs P_n' (axial) and P_{n+1}' (radial) for degree n.
    p_prev = np.ones_like(s)          # P_0
    p_curr = s.copy()                 # P_1
    dp_curr = np.ones_like(s)         # P_1'
    rho_n = rho.copy()                # (R/r)^1
    radial = np.zeros_like(s)
    axial = np.zeros_like(s)
    for n in range(2, ZONAL_DEGREES[-1] + 2):
        p_next = ((2 * n - 1) * s * p_curr - (n - 1) * p_prev) / n       # P_n
        dp_next = n * p_curr + s * dp_curr                               # P_n'
        # dp_next is P_n'. It is the radial term of degree n - 1 and the axial term of degree n.
        if n - 1 in ZONAL_DEGREES:
            radial += params[indices, _FIRST_J_COL + (n - 1) - 3] * rho_n * dp_next
        rho_n = rho_n * rho                                              # (R/r)^n
        if n in ZONAL_DEGREES:
            axial += params[indices, _FIRST_J_COL + n - 3] * rho_n * dp_next
        p_prev, p_curr, dp_curr = p_curr, p_next, dp_next

    k = mu_array[primaries] * inv_r2                                     # mu / r^2
    out[indices, 0] += k * radial * rel[:, 0] * inv_r
    out[indices, 1] += k * radial * rel[:, 1] * inv_r
    out[indices, 2] += k * (radial * s - axial)
