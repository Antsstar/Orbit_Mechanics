"""
Atmospheric drag, registered as the force model `"drag"`.

The drag acceleration on a body moving through an atmosphere that co-rotates rigidly with its
Keplerian parent, using a single-exponential density profile. It is the dominant non-gravitational
perturbation in LEO, and it composes additively with `point_mass_gravity` and `j2` on Cowell bodies.

Reference implementation only. Like `j2_kernel`, this runs on every enabled body at every RK4 stage,
so under `CLAUDE.md`'s two-implementation rule it wants a compiled twin in `kernels.py`. That twin is
**deliberately not written**. `Simulation._refresh_cowell_plan` treats any mask bit other than
`point_mass_gravity` and `j2` as foreign, so a Cowell body carrying `"drag"` sends the whole Cowell set
down the NumPy `RK4Integrator` path (`tests/validation/test_drag.py` asserts this).

Model
-----
For body `i` with parent `P = parent_indices[i]`, let `r = state[i,:3] - state[P,:3]` and
`v = state[i,3:] - state[P,3:]`:

    v_rel = v - w x r,        w = (0, 0, omega)          (atmosphere co-rotates about the frame's +z)
    h     = |r| - r_ref
    rho   = rho0 * exp(-(h - h0) / H)
    a     = -(1/2) rho B |v_rel| v_rel,    B = C_d A / m

Written out, `w x r = (-omega y, omega x, 0)`, so `v_rel = (vx + omega y, vy - omega x, vz)`.

Citations
---------
- Drag acceleration and the relative velocity `v - w x r`: Vallado, *Fundamentals of Astrodynamics
  and Applications*, 4th ed., Sec. 8.6.2, Eq. 8-28 (drag) and Eq. 8-29 (relative velocity). Montenbruck
  & Gill, *Satellite Orbits* (2000), Sec. 3.5, Eq. 3.97 and 3.98, give the same pair. Curtis, *Orbital
  Mechanics for Engineering Students*, 3rd ed., Sec. 10.2, gives the drag form. **All equation numbers
  are from memory and unverified.** The section numbers are more likely right than the equation
  numbers.
- Exponential density: Vallado 4e, Sec. 8.6.2, Eq. 8-33 and Table 8-4 (piecewise `h0`, `rho0`, `H`).
  **Unverified.** This module implements one band of that table; the per-body coefficients are one
  row of it.

The physics checks that do not rely on the citations are in `tests/validation/test_drag.py`. They
are a closed-form acceleration recomputed in SI units, orbit-averaged decay
`da/dt = -rho B sqrt(mu a) (1 - omega a / v)^2` for a circular equatorial orbit, and an energy
balance `dE/dt = a_drag . v`. See the test module docstring for the derivations.

Units: the single conversion
----------------------------
Every coefficient follows its usual convention, and the kernel works in engine units:

| param           | column | unit    |
|---|---|---|
| `ballistic_coeff` | 0 | m^2/kg, `B = C_d A / m` (no factor 1/2 folded in) |
| `rho0`          | 1 | kg/m^3 |
| `h0`            | 2 | km     |
| `scale_height`  | 3 | km     |
| `r_ref`         | 4 | km     |
| `omega`         | 5 | rad/s, signed, about the frame's +z |

`rho [kg/m^3] * B [m^2/kg]` has units of 1/m, and the engine needs 1/km. **The only unit conversion is
`_PER_M_TO_PER_KM = 1e3`**, applied once to `rho * B`. Then `(1/km) * (km/s)^2 = km/s^2`. Scale height
and `h0` are in km, so the exponent `(h - h0)/H` is dimensionless without any conversion.

Design decisions
----------------
**The coefficients sit on the perturbed body's row**, for the same reasons `geopotential.py` gives
for `(j2, r_eq)`. `B` is a property of the body. The atmosphere constants (`rho0`, `h0`, `H`, `r_ref`,
`omega`) belong to the parent, but storing them on the body keeps `enable_force_model` as the single
setter, and a sweep can vary them per body in one arena.

**B could come from `VesselORM`**, which carries `drag_area` (m^2), `dry_mass` and `fuel_mass` (kg).
`B = C_d * drag_area / (dry_mass + fuel_mass)` needs a `C_d` the ORM does not store. For now it stays
an explicit `enable_force_model` coefficient, and deriving it at ingest is a follow-up. Mass is held
constant here.

**Frame.** The same spin-axis assumption as `geopotential.py`: the parent spins about the +z axis of
the frame `state` is written in. In an ecliptic-seeded scenario that tilts the atmosphere.

**`h` is geocentric radius minus `r_ref`**, a spherical altitude, not a geodetic one. With
`r_ref = EARTH_R_EQ` it overestimates polar altitude by up to 21 km, which at `H = 60 km` is about a
factor of 1.4 in density. Whether that matters is up to the caller. Vallado's table is written in
terms of ellipsoidal height.

**Degenerate rows contribute exactly `0.0`**, with no division, overflow or warning:

- *`scale_height <= 0`*, which includes a body whose bit is set with coefficients never written. As
  with `"j2"`, that is a silent no-op, so always pass the coefficients.
- *zero separation*, which includes a root body that parents itself.

**Barycentre parents are refused at configuration time.** The `validate_bodies` hook reuses
`geopotential.barycentre_parented`. A barycentre has no surface and no atmosphere, and the kernel
cannot see `is_system`.

**Velocity during RK4 stages.** `integrators.RK4Integrator` writes `state[primaries] + candidate`
into both position and velocity columns at every stage, and never writes `state[primaries]`. So
`state[i,3:] - state[P,3:]` is the stage's candidate velocity relative to the parent. Drag depends
only on relative position and relative velocity, so the relative-state formulation is exact for it,
as it is for gravity and J2.

Limitations
-----------
- There is a single exponential band. Real density falls by about 10 over 100 km in LEO, and `H` itself
  grows with altitude (about 8 km near the ground, 60 km or more above 500 km). One band is accurate
  over a few scale heights around `h0`. A decaying orbit that crosses several bands needs a
  piecewise table (Vallado Table 8-4) or NRLMSISE via `pymsis`. Both are out of scope.
- The model ignores solar and geomagnetic activity, day/night bulge and winds. Those effects are what
  `pymsis` would add, and they matter more than the table choice does for real prediction.
- Nothing stops the orbit when it reaches `h < 0`. The density grows exponentially, and the RK4 step
  eventually fails.
"""
from __future__ import annotations

from typing import Final, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .geopotential import barycentre_parented
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "DRAG_MODEL", "DRAG_PARAM_NAMES", "EARTH_OMEGA", "drag_kernel",
]

DRAG_MODEL: Final[str] = "drag"
DRAG_PARAM_NAMES: Final = ("ballistic_coeff", "rho0", "h0", "scale_height", "r_ref", "omega")
_B_COL: Final[int] = 0
_RHO0_COL: Final[int] = 1
_H0_COL: Final[int] = 2
_SCALE_HEIGHT_COL: Final[int] = 3
_R_REF_COL: Final[int] = 4
_OMEGA_COL: Final[int] = 5

# rho [kg/m^3] * B [m^2/kg] is 1/m; the engine wants 1/km. This is the model's only unit conversion.
_PER_M_TO_PER_KM: Final[float] = 1.0e3

# Nominal mean Earth rotation rate, rad/s (WGS-84 / IERS conventional value 7.292115e-5, quoted from
# memory). The sidereal rate; the difference from the true, variable rate is ~1e-8 relative.
EARTH_OMEGA: Final[float] = 7.292115e-5


def _reject_barycentre_parents(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook for `"drag"`: refuse bodies whose Keplerian parent is a barycentre."""
    offending = barycentre_parented(sim.is_system, sim.parent_indices, bodies)
    if offending.size > 0:
        raise ValueError(
            f"force model '{DRAG_MODEL}' is meaningless for bodies whose parent is a barycentre (a "
            f"barycentre has no atmosphere to co-rotate); slot(s) {offending.tolist()} qualify. See "
            f"drag.py's module docstring."
        )


@register_force_model(
    DRAG_MODEL,
    param_names=DRAG_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    citation=(
        "Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Sec. 8.6.2, Eq. 8-28/8-29 "
        "(drag, v_rel = v - w x r) and Eq. 8-33 (exponential density); Montenbruck & Gill, Satellite "
        "Orbits, Eq. 3.97/3.98. Equation numbers from memory, unverified."
    ),
)
def drag_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add `-(1/2) rho(h) B |v_rel| v_rel` to `out[indices]`, km/s^2, with `v_rel = v - w x r` measured
    against each body's Keplerian parent and `w = omega z_hat`.

    Coefficient units and the single conversion (`_PER_M_TO_PER_KM`) are in the module docstring.
    `t` and `mu_array` are unused: the atmosphere is steady in the frame co-rotating with the parent.
    """
    primaries = parent_indices[indices]
    rel = state[indices] - state[primaries]                     # (k, 6): [r | v] relative to parent
    r2 = np.einsum("ij,ij->i", rel[:, :3], rel[:, :3])

    coeff = params[indices]
    omega = coeff[:, _OMEGA_COL]
    scale_height = coeff[:, _SCALE_HEIGHT_COL]

    # v_rel = v - w x r, with w x r = (-omega y, omega x, 0).
    vrx = rel[:, 3] + omega * rel[:, 1]
    vry = rel[:, 4] - omega * rel[:, 0]
    vrz = rel[:, 5]
    speed = np.sqrt(vrx * vrx + vry * vry + vrz * vrz)

    altitude = np.sqrt(r2) - coeff[:, _R_REF_COL]

    # Rows with no scale height or no separation contribute exactly zero. The exponential is not
    # evaluated there, so an unconfigured row (H = 0) or a root body (r = 0, which would put h far
    # below h0) can neither divide by zero nor overflow.
    valid = (scale_height > 0.0) & (r2 > 0.0)
    exponent = np.divide(altitude - coeff[:, _H0_COL], scale_height,
                         out=np.zeros_like(r2), where=valid)
    density = np.exp(-exponent, out=np.zeros_like(r2), where=valid) * coeff[:, _RHO0_COL]  # kg/m^3

    # 0.5 * rho [kg/m^3] * B [m^2/kg] * 1e3 -> 1/km; times |v_rel| [km/s] times v_rel [km/s].
    k = 0.5 * density * coeff[:, _B_COL] * _PER_M_TO_PER_KM * speed

    out[indices, 0] -= k * vrx
    out[indices, 1] -= k * vry
    out[indices, 2] -= k * vrz
