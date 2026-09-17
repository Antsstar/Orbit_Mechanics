"""
Zonal-harmonic geopotential perturbations, registered as force models.

Currently one model: `"j2"`, the second-degree zonal (oblateness) term. It contributes **only the
perturbation** - never the central `-mu r / r^3` term - so it composes additively with the Cowell
propagator's point-mass force model instead of double-counting it.

Reference implementation only. `j2_kernel` runs over every body it is enabled on, every integrator
sub-stage, so under `CLAUDE.md`'s two-implementation rule it wants a compiled scalar twin in
`kernels.py`, held equivalent by test. That twin is a separate follow-up and is deliberately not here.

Citation
--------
Curtis, H. D., *Orbital Mechanics for Engineering Students*, 3rd ed., Chapter 10 ("Introduction to
Orbital Perturbations"), the J2 perturbing acceleration

    p = (3/2) J2 mu R^2 / r^4 * [ (x/r)(5 z^2/r^2 - 1) i
                                 + (y/r)(5 z^2/r^2 - 1) j
                                 + (z/r)(5 z^2/r^2 - 3) k ]

which is Eq. (10.30) in the 3rd edition **as recalled from memory, not checked against the text**.
Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., gives the same expression in its
Chapter 8/9 treatment of the geopotential; no equation number is claimed for it here for the same
reason. Neither number should be treated as verified until someone checks a copy.

What *is* checkable without trusting either book is the derivation, which is short. The J2 term of
the geopotential, written as a potential energy per unit mass (physicist's sign, `a = -grad Phi`), is

    Phi_J2(r) = (mu J2 R^2 / r^3) * P2(z/r),      P2(s) = (3 s^2 - 1) / 2
              = (mu J2 R^2 / 2) * (3 z^2 r^-5 - r^-3)

Differentiating, with d(r^-n)/dx = -n x r^-(n+2):

    dPhi/dx = (mu J2 R^2 / 2) * (-15 z^2 x r^-7 + 3 x r^-5)
    dPhi/dz = (mu J2 R^2 / 2) * ( 6 z r^-5 - 15 z^3 r^-7 + 3 z r^-5)

so `a = -grad Phi` is exactly the Curtis form above. `tests/validation/test_geopotential.py` checks the kernel
against a central finite difference of `Phi_J2` - the independent route - so the citation's equation
number is corroboration, not the only guard.

Sign sanity: on the equator (z = 0) the perturbation points *inward* (extra equatorial bulge mass
strengthens gravity), magnitude (3/2) J2 mu R^2/r^4; over a pole it points *outward*, magnitude
3 J2 mu R^2/r^4.

Earth coefficients
------------------
`EARTH_R_EQ` = 6378.137 km is the WGS-84 defining semi-major axis (NIMA TR8350.2). `EARTH_J2` is the
EGM96 unnormalised zonal, `J2 = -sqrt(5) * Cbar20` with `Cbar20 = -0.484165371736e-3`
(Lemoine et al. 1998, NASA/TP-1998-206861). Both values are **quoted from memory, not checked against
those documents**. They are a matched set with `scenarios.MU_EARTH = 3.986004418e5` km^3/s^2 (the
WGS-84/EGM96 GM): a J2 value is only meaningful together with the reference radius it was normalised
against, so do not mix `EARTH_J2` with `scenarios.EARTH_RADIUS` (6371 km, a *mean* radius) - that
alone would be a 0.2% error in the perturbation, exactly the plausible-looking kind.

Design decisions
----------------
**Coefficients live on the perturbed body's row** (`param_names = ("j2", "r_eq")`), even though they
describe the parent. Storing them on the parent's row instead would remove the duplication, but would
break two things this layer states as contract: `forces.py` says `indices` identifies "which rows of
`params` describe it", and `Simulation.enable_force_model` - the only sweep surface - writes
coefficients to exactly the rows whose bit it sets. Parent-row storage would therefore need either a
second setter (a parallel configuration mechanism) or would force the parent's own bit on, silently
enabling "J2 from the parent's parent" on it. The duplication costs 16 bytes per body. It also buys
something a comparison engine wants: bodies in one arena can carry *different* J2 values, so a
coefficient sensitivity sweep over massless satellites runs vectorised in a single simulation.
The price is that coefficients do not follow a change of parent - a body whose `parent_indices` entry
is re-pointed (e.g. a future sphere-of-influence handover) must have its row rewritten too.

**Frame.** The expression assumes the parent's spin (symmetry) axis is the +z axis of the frame
`state` is expressed in. The engine's arena frame is whatever the scenario's elements were seeded in;
for an Earth satellite that is only correct in an Earth-equatorial frame (e.g. GCRF/J2000 equatorial,
neglecting precession-nutation of the pole). In an ecliptic-seeded scenario such as
`scenarios.sun_earth_moon` the z axis is the ecliptic pole, 23.4 degrees off Earth's spin axis, and
this kernel then models an Earth tilted into the ecliptic. No rotation is applied here.

**Degenerate parents.**

- *Root body* (`parent_indices[i] == i`): the relative position is exactly zero. The kernel masks
  `r^2 > 0` and contributes exactly `0.0` - no division is evaluated there, so no NaN and no warning.
  The same holds for any body exactly coincident with its parent.
- *Zero coefficients*: a body whose bit is set but whose `j2` or `r_eq` was never written contributes
  exactly `0.0`. This is also a silent failure mode - enabling `"j2"` without coefficients does
  nothing - so always pass them to `enable_force_model`.
- *Barycentre parent* (`is_system[parent]`): its `mu_array` row holds summed system mass, and J2 of a
  barycentre is meaningless. **The kernel cannot detect this case**: `ForceKernel`'s signature does
  not carry `is_system`, and neither `mu_array` nor `parent_indices` distinguishes a barycentre row.
  On such a body it returns a finite (never NaN, for non-zero separation) but physically meaningless
  value computed from the barycentre's row. The guard is at configuration time instead:
  `barycentre_parented(is_system, parent_indices, bodies)` returns the offending slots, and `"j2"`
  registers it as its `validate_bodies` hook, so `Simulation.enable_force_model("j2", ...)` raises
  `ValueError` for such bodies before setting any mask bit. Only a caller that writes
  `force_model_mask` directly bypasses the guard.
  None of the shipped scenarios parent a body to a barycentre; barycentres parent to bodies.
"""
from __future__ import annotations

from typing import Final, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarKilometers, ScalarSeconds
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "J2_MODEL", "J2_PARAM_NAMES", "EARTH_J2", "EARTH_R_EQ",
    "j2_kernel", "j2_potential", "barycentre_parented",
]

J2_MODEL: Final[str] = "j2"
J2_PARAM_NAMES: Final = ("j2", "r_eq")
_J2_COL: Final[int] = 0
_R_EQ_COL: Final[int] = 1

# Matched with scenarios.MU_EARTH. Provenance (from memory, unverified) in the module docstring.
EARTH_J2: Final[float] = 1.0826266835e-3              # dimensionless, EGM96: -sqrt(5) * Cbar20
EARTH_R_EQ: Final[ScalarKilometers] = 6378.137         # km, WGS-84 semi-major axis


def _reject_barycentre_parents(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook for `"j2"`: refuse bodies whose Keplerian parent is a barycentre."""
    offending = barycentre_parented(sim.is_system, sim.parent_indices, bodies)
    if offending.size > 0:
        raise ValueError(
            f"force model '{J2_MODEL}' is meaningless for bodies whose parent is a barycentre (its "
            f"mu row holds summed system mass); slot(s) {offending.tolist()} qualify. See "
            f"geopotential.py's module docstring."
        )


@register_force_model(
    J2_MODEL,
    param_names=J2_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    citation=(
        "Curtis, Orbital Mechanics for Engineering Students, 3rd ed., Eq. 10.30 (equation number from "
        "memory, unverified); derived in geopotential.py as -grad of Phi_J2 = (mu J2 R^2/r^3) P2(z/r)"
    ),
)
def j2_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add the J2 perturbing acceleration of each body's Keplerian parent to `out[indices]`, km/s^2.

    For body `i` with parent `P = parent_indices[i]`, `(x, y, z) = state[i,:3] - state[P,:3]`,
    `mu = mu_array[P]`, `J2 = params[i,0]`, `R = params[i,1]`:

        a = (3/2) J2 mu R^2 / r^5 * [ x (5 z^2/r^2 - 1),  y (5 z^2/r^2 - 1),  z (5 z^2/r^2 - 3) ]

    See the module docstring for the citation, derivation, the frame assumption (parent spin axis is
    the frame's +z), and the degenerate-parent behaviour. The central point-mass term is *not*
    included. `t` is unused: the field is time-invariant in the frame it assumes.
    """
    primaries = parent_indices[indices]
    rel = state[indices, :3] - state[primaries, :3]
    r2 = np.einsum("ij,ij->i", rel, rel)

    # Exactly-zero separation (root bodies self-parent) contributes exactly zero: the division is
    # never evaluated there, and inv_r2 = 0 then zeroes every term below.
    inv_r2 = np.divide(1.0, r2, out=np.zeros_like(r2), where=r2 > 0.0)

    r_eq = params[indices, _R_EQ_COL]
    k = 1.5 * params[indices, _J2_COL] * mu_array[primaries] * r_eq * r_eq \
        * inv_r2 * inv_r2 * np.sqrt(inv_r2)                              # (3/2) J2 mu R^2 / r^5
    five_s2 = 5.0 * rel[:, 2] * rel[:, 2] * inv_r2                        # 5 z^2 / r^2

    out[indices, 0] += k * rel[:, 0] * (five_s2 - 1.0)
    out[indices, 1] += k * rel[:, 1] * (five_s2 - 1.0)
    out[indices, 2] += k * rel[:, 2] * (five_s2 - 3.0)


def j2_potential(
    rel: NDArray[np.float64], mu: float, j2: float, r_eq: ScalarKilometers,
) -> NDArray[np.float64]:
    """
    The J2 disturbing potential energy per unit mass, km^2/s^2, at relative positions `rel` `(N,3)`:
    `Phi_J2 = (mu J2 R^2 / 2) (3 z^2 / r^5 - 1 / r^3)`, with the sign convention `a = -grad Phi`.

    Not used by the kernel, and deliberately written from the potential rather than from the
    acceleration, so that differentiating it numerically is an independent check of `j2_kernel`.
    Useful beyond tests as the J2 contribution to specific orbital energy.
    """
    r: NDArray[np.float64] = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    z: NDArray[np.float64] = rel[:, 2]
    phi: NDArray[np.float64] = 0.5 * mu * j2 * r_eq * r_eq * (3.0 * z * z / r ** 5 - 1.0 / r ** 3)
    return phi


def barycentre_parented(
    is_system: NDArray[np.bool_],
    parent_indices: NDArray[np.int32],
    bodies: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Configuration-time guard: the subset of `bodies` whose Keplerian parent is a barycentre, for
    which J2 is meaningless and `j2_kernel` cannot tell (see the module docstring). Call before
    `enable_force_model("j2", ...)`, never inside a step.
    """
    return bodies[is_system[parent_indices[bodies]]]
