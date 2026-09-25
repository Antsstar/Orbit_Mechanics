"""
Atmospheric drag, registered as the force model `"drag"`.

The drag acceleration on a body moving through an atmosphere that co-rotates rigidly with its
Keplerian parent, using a single-exponential density profile. It is the dominant non-gravitational
perturbation in LEO, and it composes additively with `point_mass_gravity` and `j2` on Cowell bodies.

Reference implementation only. Like `j2_kernel`, this runs on every enabled body at every RK4 stage,
so under `CLAUDE.md`'s two-implementation rule it wants a compiled twin in `kernels.py`. That twin is
**deliberately not written**. `Simulation._refresh_cowell_plan` treats any mask bit other than
`point_mass_gravity`, `j2` and `zonal` as foreign, so a Cowell body carrying `"drag"` sends the whole Cowell set
down the NumPy `RK4Integrator` path (`tests/validation/test_drag.py` asserts this).

Model
-----
For body `i` with parent `P = parent_indices[i]`, let `r = state[i,:3] - state[P,:3]` and
`v = state[i,3:] - state[P,3:]`:

    v_rel = v - w x r,        w = (0, 0, omega)          (atmosphere co-rotates about the frame's +z)
    h     = |r| - r_ref
    rho   = rho(h)                                       (see "Choosing a density law" below)
    a     = -(1/2) rho B |v_rel| v_rel,    B = C_d A / m

Written out, `w x r = (-omega y, omega x, 0)`, so `v_rel = (vx + omega y, vy - omega x, vz)`.

Choosing a density law
----------------------
`rho(h)` is a **per-body model choice**, held in the `density_model` coefficient and implemented in
`atmosphere.py`:

| `density_model` | law | which coefficients it reads |
|---|---|---|
| `0.0` (`DENSITY_MODEL_EXPONENTIAL`, the default) | `rho0 exp(-(h - h0) / H)` | `rho0`, `h0`, `scale_height` |
| `1.0` (`DENSITY_MODEL_LAYERED`) | piecewise exponential, 28 bands, Vallado Table 8-4 | none of those three |
| `2.0` (`DENSITY_MODEL_MSIS`) | NRLMSIS 2.0 global-mean profile via `pymsis` (`msis_bridge.py`) | `f107`, `f107a`, `ap` |

An unwritten coefficient row is all zeros, so **0.0 is the single-exponential law and every existing
configuration keeps its previous behaviour bit for bit**. Under the layered and MSIS laws `rho0`, `h0`
and `scale_height` are ignored entirely - including `scale_height <= 0`, which is a silent no-op only
for the single-band law, since the other laws have no per-body scale height that could be missing.

**MSIS is evaluated at configuration time, never in the kernel.** `validate_coefficients` refuses
`density_model = 2.0` unless `f107`, `f107a` and `ap` are all given in the same call (so `pymsis` is
never left to download them), refuses solar indices on a row whose law would ignore them, and then
evaluates the profile for each requested triple (`msis_bridge.msis_profile`, memoised). The kernel
only reads that memo. A solar-activity sweep is therefore coefficients alone:

    ForceModelSpec("drag", {"ballistic_coeff": B, "r_ref": EARTH_R_EQ, "omega": EARTH_OMEGA,
                            "density_model": DENSITY_MODEL_MSIS, "f107": 150.0, "f107a": 150.0,
                            "ap": 15.0})

The choice is a float coefficient rather than a second registered model, so it is one more key in a
`sweep.ForceModelSpec`'s `coefficients` mapping and costs no mask bit:

    ForceModelSpec("drag", {"ballistic_coeff": B, "r_ref": EARTH_R_EQ, "omega": EARTH_OMEGA,
                            "density_model": DENSITY_MODEL_LAYERED})

`validate_coefficients` rejects any value that is not exactly one of the three selectors, so a typo in
a sweep config raises at configuration time rather than silently rounding to the nearer law.

Citations
---------
- Drag acceleration and the relative velocity `v - w x r`: Vallado, *Fundamentals of Astrodynamics
  and Applications*, 4th ed., Sec. 8.6.2, Eq. 8-28 (drag) and Eq. 8-29 (relative velocity). Montenbruck
  & Gill, *Satellite Orbits* (2000), Sec. 3.5, Eq. 3.97 and 3.98, give the same pair. Curtis, *Orbital
  Mechanics for Engineering Students*, 3rd ed., Sec. 10.2, gives the drag form. **All equation numbers
  are from memory and unverified.** The section numbers are more likely right than the equation
  numbers.
- Density: Vallado 4e, Sec. 8.6.2, Eq. 8-33 (exponential) and Table 8-4 (the piecewise `h0`, `rho0`,
  `H` fit). **Unverified.** Under `DENSITY_MODEL_EXPONENTIAL` the per-body coefficients are one row of
  that table, supplied by the caller; under `DENSITY_MODEL_LAYERED` the whole table is used. See
  `atmosphere.py` for the table, its citation, and how far it can be trusted.

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
| `density_model` | 6 | selector, dimensionless: 0.0 single exponential, 1.0 layered, 2.0 MSIS |
| `f107`          | 7 | sfu, previous-day F10.7 - read only under MSIS |
| `f107a`         | 8 | sfu, 81-day mean F10.7 - read only under MSIS |
| `ap`            | 9 | daily Ap, 0..400 - read only under MSIS |

Columns 7-9 are read only on MSIS rows, so a legacy 7-column parameter array still evaluates the
first two laws unchanged.

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
- Under `DENSITY_MODEL_EXPONENTIAL` there is a single band. Real density falls by a factor of roughly
  5 to 12 per 100 km in LEO, and `H` itself grows with altitude (about 8 km near the ground, 60 km or
  more above 500 km). One band is accurate over a few scale heights around `h0`; a decaying orbit that
  crosses several bands is not. `DENSITY_MODEL_LAYERED` is the answer to that, and
  `tests/validation/test_atmosphere.py` measures what the choice costs in predicted decay.
- The two static laws model no solar or geomagnetic activity. `DENSITY_MODEL_MSIS` adds activity as
  constant indices, but it too averages away the day/night bulge, seasons and latitude, and none of
  the three has winds - see `msis_bridge.py` for the sizes of what the average discards.
- Nothing stops the orbit when it reaches `h < 0`. The density grows exponentially, and the RK4 step
  eventually fails.
"""
from __future__ import annotations

from typing import Final, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .atmosphere import (
    DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS, DENSITY_MODELS,
    exponential_density, layered_density,
)
from .msis_bridge import MSIS_SOLAR_COEFFICIENTS, check_solar_activity, msis_density, msis_profile
from .custom_types import ScalarSeconds
from .geopotential import barycentre_parented
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "DRAG_MODEL", "DRAG_PARAM_NAMES", "EARTH_OMEGA", "drag_kernel",
    "DENSITY_MODEL_EXPONENTIAL", "DENSITY_MODEL_LAYERED", "DENSITY_MODEL_MSIS",
]

DRAG_MODEL: Final[str] = "drag"
DRAG_PARAM_NAMES: Final = (
    "ballistic_coeff", "rho0", "h0", "scale_height", "r_ref", "omega", "density_model",
    "f107", "f107a", "ap",
)
_B_COL: Final[int] = 0
_RHO0_COL: Final[int] = 1
_H0_COL: Final[int] = 2
_SCALE_HEIGHT_COL: Final[int] = 3
_R_REF_COL: Final[int] = 4
_OMEGA_COL: Final[int] = 5
_DENSITY_MODEL_COL: Final[int] = 6
_SOLAR_COLS: Final = slice(7, 10)          # f107, f107a, ap - MSIS_SOLAR_COEFFICIENTS, in order

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


def _validate_density_coefficients(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """
    `validate_coefficients` hook for `"drag"`, run before any bit or coefficient is written:

    1. `density_model` must be exactly one of `atmosphere.DENSITY_MODELS`. The kernel dispatches on
       half-way thresholds, so `3.0` would otherwise pick MSIS silently - a config typo that changes
       the physics and raises nothing, precisely the failure mode `CLAUDE.md`'s item 5 exists to
       catch. Omitting it is still legal and keeps the row's current law (0.0 on a fresh row).
    2. Asking for MSIS requires `f107`, `f107a` and `ap` **in the same call**. `pymsis` would fetch
       missing indices from the network by date; here they are configuration data, never looked up.
       (`ap = 0` is a legitimate value, so an unwritten column cannot be told from a deliberate zero -
       hence "in the same call" rather than "non-zero".) Later calls may update single indices on a
       row that is already MSIS.
    3. Solar indices on a row whose law is not MSIS are refused: they would be silently ignored, and
       a sweep over them would measure nothing.
    4. For every distinct `(f107, f107a, ap)` the call leaves on an MSIS row, the indices are
       range-checked and the NRLMSIS 2.0 profile is **evaluated now** (`msis_profile`, memoised) -
       this is the configuration-time boundary; the kernel only reads the memo.
    """
    requested = coefficients.get("density_model")
    if requested is not None and float(requested) not in DENSITY_MODELS:
        raise ValueError(
            f"force model '{DRAG_MODEL}': density_model={requested!r} is not a known density law; "
            f"use atmosphere.DENSITY_MODEL_EXPONENTIAL ({DENSITY_MODEL_EXPONENTIAL}), "
            f"DENSITY_MODEL_LAYERED ({DENSITY_MODEL_LAYERED}) or DENSITY_MODEL_MSIS "
            f"({DENSITY_MODEL_MSIS})."
        )

    existing = sim.force_model_params.get(DRAG_MODEL)
    if requested is not None:
        law = np.full(bodies.size, float(requested))
    elif existing is not None:
        law = existing[bodies, _DENSITY_MODEL_COL]
    else:
        law = np.zeros(bodies.size)
    on_msis = law == DENSITY_MODEL_MSIS

    given = [key for key in MSIS_SOLAR_COEFFICIENTS if key in coefficients]
    if given and not np.all(on_msis):
        raise ValueError(
            f"force model '{DRAG_MODEL}': {given} are read only under density_model="
            f"DENSITY_MODEL_MSIS ({DENSITY_MODEL_MSIS}); slot(s) "
            f"{bodies[~on_msis].tolist()} would silently ignore them."
        )
    if requested is not None and float(requested) == DENSITY_MODEL_MSIS and len(given) < 3:
        missing = [key for key in MSIS_SOLAR_COEFFICIENTS if key not in coefficients]
        raise ValueError(
            f"force model '{DRAG_MODEL}': density_model=DENSITY_MODEL_MSIS needs f107, f107a and ap "
            f"given explicitly (missing {missing}). Solar activity is configuration data here and is "
            f"never looked up or downloaded - see msis_bridge.SOLAR_ACTIVITY_* for presets."
        )
    if not np.any(on_msis):
        return

    activity = (existing[bodies][:, _SOLAR_COLS].copy() if existing is not None
                else np.zeros((bodies.size, 3)))
    for column, key in enumerate(MSIS_SOLAR_COEFFICIENTS):
        if key in coefficients:
            activity[:, column] = float(coefficients[key])
    for f107, f107a, ap in np.unique(activity[on_msis], axis=0):
        check_solar_activity(float(f107), float(f107a), float(ap))
        msis_profile(float(f107), float(f107a), float(ap))


@register_force_model(
    DRAG_MODEL,
    param_names=DRAG_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    validate_coefficients=_validate_density_coefficients,
    citation=(
        "Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Sec. 8.6.2, Eq. 8-28/8-29 "
        "(drag, v_rel = v - w x r), Eq. 8-33 (exponential density) and Table 8-4 (the piecewise "
        "table in atmosphere.py); Montenbruck & Gill, Satellite Orbits, Eq. 3.97/3.98; NRLMSIS 2.0 "
        "(Emmert et al. 2021, Earth and Space Science 8, e2020EA001321) via pymsis, averaged at "
        "configuration time (msis_bridge.py). Equation and table numbers from memory, unverified."
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

    `rho(h)` is whichever law the row's `density_model` coefficient selects - `atmosphere.py`'s
    single exponential (0.0, the default), its 28-band piecewise table (1.0), or the NRLMSIS 2.0
    profile memoised at configuration time for the row's `(f107, f107a, ap)` (2.0). Each law is
    evaluated on its own masked rows and the three are summed, which keeps the kernel branch-free over
    a mixed arena where different bodies have been configured with different atmospheres. The MSIS
    term never calls `pymsis`; it raises `LookupError` if a row's profile was never evaluated.

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

    # Which density law each row uses. The three masks partition the rows, so exactly one of the
    # terms below is non-zero per row and the sum is a branchless select - no `np.where` over a
    # freshly allocated pair, and no `if np.any(...)` guard (`CLAUDE.md`, Conventions). For a row on
    # the first two laws the masks are the same booleans they were before MSIS existed and the MSIS
    # term is exactly +0.0, so those rows are bit-identical to the two-law kernel.
    #
    # Rows with no separation contribute exactly zero under either law, and a row with no scale
    # height contributes zero under the single-band law only (an unconfigured row, H = 0). Neither
    # exponential is evaluated on an excluded row, so a root body (r = 0, altitude = -r_ref, which
    # would overflow `exp`) can neither divide by zero nor overflow.
    separated = r2 > 0.0
    selector = coeff[:, _DENSITY_MODEL_COL]
    beyond_exponential = selector >= 0.5          # the two-law kernel's `use_layered`, verbatim
    use_exponential = ~beyond_exponential         # so even a NaN selector keeps its old meaning
    use_msis = selector >= 1.5
    use_layered = beyond_exponential & ~use_msis
    density = (                                                                       # kg/m^3
        exponential_density(altitude, coeff[:, _RHO0_COL], coeff[:, _H0_COL], scale_height,
                            separated & use_exponential & (scale_height > 0.0))
        + layered_density(altitude, separated & use_layered)
        + msis_density(altitude, coeff[:, _SOLAR_COLS], separated & use_msis)
    )

    # 0.5 * rho [kg/m^3] * B [m^2/kg] * 1e3 -> 1/km; times |v_rel| [km/s] times v_rel [km/s].
    k = 0.5 * density * coeff[:, _B_COL] * _PER_M_TO_PER_KM * speed

    out[indices, 0] -= k * vrx
    out[indices, 1] -= k * vry
    out[indices, 2] -= k * vrz
