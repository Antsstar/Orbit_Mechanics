"""
Atmospheric density laws — the *density* half of `drag.py`, split out so the choice of law is a
swept model dimension rather than a hard-coded formula.

`drag.py` owns the drag acceleration `-(1/2) rho B |v_rel| v_rel`. This module owns `rho(h)`, and
offers three laws:

| law | selector | what it is |
|---|---|---|
| single exponential | `DENSITY_MODEL_EXPONENTIAL` (0.0) | `rho0 exp(-(h - h0) / H)`, one band, the original |
| piecewise exponential | `DENSITY_MODEL_LAYERED` (1.0) | 28 bands, 0 to 1000+ km, from the table below |
| NRLMSIS 2.0 mean profile | `DENSITY_MODEL_MSIS` (2.0) | `pymsis` averaged at configuration time, see `msis_bridge.py` |

The third law is evaluated by the same `piecewise_exponential_density` as the table - log-linear
between altitude nodes - with the nodes, densities and scale heights read from a profile
`msis_bridge.msis_profile(f107, f107a, ap)` computed **once, at configuration time**, from `pymsis`.
Nothing in this module imports `pymsis`, and nothing in a step calls it; see `msis_bridge.py`.

The selector is the `density_model` column of `force_model_params["drag"]`, so it is a per-body float
and a `sweep.ForceModelSpec` can set it like any other coefficient. **0.0 is the default an
unwritten row already holds**, which is what keeps every pre-existing configuration on the single
exponential unchanged.

The piecewise table
-------------------
Within band `k`, spanning `[h_k, h_{k+1})`:

    rho(h) = rho_k * exp(-(h - h_k) / H_k)

which is the same functional form as the single-band law with `(rho0, h0, H)` read from the band
containing `h` instead of from the body's own row. Above the last base altitude the top band is
extrapolated, and below 0 km the bottom band is extrapolated downwards — the model has no floor.

Citation
--------
Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Sec. 8.6.2, **Table 8-4**
("Exponential Atmospheric Model"), a piecewise-exponential fit to the US Standard Atmosphere 1976
below 86 km and to CIRA-72 above it. Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed.,
Table 10.1 (also 28 bands) and Wertz, *Space Mission Analysis and Design*, carry the same fit.
**The section, table and equation numbers are from memory and unverified against the text**, in the
same way `drag.py` and `propagators.py` mark theirs.

How much to trust the numbers
-----------------------------
The table below was transcribed from memory, so it needs an internal check rather than an appeal to
authority. It has one: a piecewise-exponential *fit* is continuous, so for every interior boundary

    rho_k * exp(-(h_{k+1} - h_k) / H_k)  ==  rho_{k+1}

to the table's own 4-significant-figure rounding. Three independently remembered numbers per band
cannot satisfy 27 such constraints by accident, and
`tests/validation/test_atmosphere.py::test_density_is_continuous_across_every_band_boundary`
asserts it at 5e-4 relative — derived in that module's docstring from the rounding alone. Treat that
test, not this docstring, as the statement of confidence.

The individual bands most worth a second look before quoting a number from here are 130-150 km
(`H` = 12.636 and 16.149 km), which are the two the continuity check constrains most weakly because
their band width is smaller than their scale height. Everything from 150 km up — the range that
actually governs orbit decay — is constrained to better than 1e-4 by continuity.

Not modelled
------------
By the two static laws: solar and geomagnetic activity, the diurnal bulge, winds, and any seasonal or
latitudinal variation. The table is a single static mean profile. Real thermospheric density at
400 km varies by more than an order of magnitude over a solar cycle (NRLMSIS 2.0's global mean moves
by 24x between ECSS low and high activity, `msis_bridge.py`), which is larger than the difference
between the two static laws. `DENSITY_MODEL_MSIS` is the answer to the solar-activity half; the
diurnal bulge, seasons and latitude are still averaged out, deliberately - see `msis_bridge.py` for
what that discards and how large it is.

Units
-----
Altitude in km, density in kg/m^3, scale height in km — the conventional units of the table, and the
ones `drag.py`'s single unit conversion (`rho * B`, 1/m to 1/km) already assumes.
"""
from __future__ import annotations

from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, ArrayKilometers

__all__ = [
    "DENSITY_MODEL_EXPONENTIAL", "DENSITY_MODEL_LAYERED", "DENSITY_MODEL_MSIS", "DENSITY_MODELS",
    "BASE_ALTITUDE_KM", "BASE_DENSITY_KG_M3", "SCALE_HEIGHT_KM",
    "layered_density", "exponential_density", "piecewise_exponential_density",
]

# Selector values for the `density_model` coefficient of the `"drag"` force model. They are floats
# because `Simulation.enable_force_model` takes `**coefficients: float` and `force_model_params` is a
# float array; the kernel dispatches on thresholds half-way between them (`>= 0.5`, `>= 1.5`), so any
# value rounds to the nearer law rather than silently selecting none, and `drag.py`'s
# `validate_coefficients` refuses anything that is not exactly one of them.
DENSITY_MODEL_EXPONENTIAL: Final[float] = 0.0
DENSITY_MODEL_LAYERED: Final[float] = 1.0
#: NRLMSIS 2.0 via `pymsis`, averaged into an altitude profile at configuration time
#: (`msis_bridge.py`). Needs the `f107`, `f107a` and `ap` coefficients on the same row.
DENSITY_MODEL_MSIS: Final[float] = 2.0
DENSITY_MODELS: Final = (DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, DENSITY_MODEL_MSIS)

# Vallado 4e Table 8-4. Column 1 is the base ellipsoidal altitude of the band in km, column 2 the
# nominal density there in kg/m^3, column 3 the band's scale height in km. Read the "How much to
# trust the numbers" section above before quoting any single row.
_TABLE: Final = (
    #   h0 km    rho0 kg/m^3    H km
    (     0.0,   1.225,           7.249),
    (    25.0,   3.899e-2,        6.349),
    (    30.0,   1.774e-2,        6.682),
    (    40.0,   3.972e-3,        7.554),
    (    50.0,   1.057e-3,        8.382),
    (    60.0,   3.206e-4,        7.714),
    (    70.0,   8.770e-5,        6.549),
    (    80.0,   1.905e-5,        5.799),
    (    90.0,   3.396e-6,        5.382),
    (   100.0,   5.297e-7,        5.877),
    (   110.0,   9.661e-8,        7.263),
    (   120.0,   2.438e-8,        9.473),
    (   130.0,   8.484e-9,       12.636),
    (   140.0,   3.845e-9,       16.149),
    (   150.0,   2.070e-9,       22.523),
    (   180.0,   5.464e-10,      29.740),
    (   200.0,   2.789e-10,      37.105),
    (   250.0,   7.248e-11,      45.546),
    (   300.0,   2.418e-11,      53.628),
    (   350.0,   9.518e-12,      53.298),
    (   400.0,   3.725e-12,      58.515),
    (   450.0,   1.585e-12,      60.828),
    (   500.0,   6.967e-13,      63.822),
    (   600.0,   1.454e-13,      71.835),
    (   700.0,   3.614e-14,      88.667),
    (   800.0,   1.170e-14,     124.64),
    (   900.0,   5.245e-15,     181.05),
    (  1000.0,   3.019e-15,     268.00),
)

_TABLE_ARRAY: Final[ArrayFloat] = np.array(_TABLE, dtype=np.float64)

#: Band base altitudes, km, strictly increasing — the `np.searchsorted` key.
BASE_ALTITUDE_KM: Final[ArrayFloat] = np.ascontiguousarray(_TABLE_ARRAY[:, 0])
#: Density at each band's base altitude, kg/m^3.
BASE_DENSITY_KG_M3: Final[ArrayFloat] = np.ascontiguousarray(_TABLE_ARRAY[:, 1])
#: Each band's scale height, km. Every entry is strictly positive, so the kernel never guards it.
SCALE_HEIGHT_KM: Final[ArrayFloat] = np.ascontiguousarray(_TABLE_ARRAY[:, 2])

def layered_density(
    altitude_km: ArrayKilometers, valid: Optional[NDArray[np.bool_]] = None,
) -> ArrayFloat:
    """
    Piecewise-exponential density, kg/m^3, for each altitude above the reference surface.

    Vectorised over the whole input with a single `np.searchsorted` band lookup — no Python loop over
    bodies, and no branch per band. `side="right"` minus one maps `h` to the band whose base altitude
    is the largest not exceeding `h`; clipping that index to `[0, n-1]` extrapolates the bottom band
    downwards (including to negative altitude) and the top band upwards, which is what makes the law
    total rather than raising on a re-entering or very high body.

    `valid`, if given, is a boolean mask the same shape as `altitude_km`: rows where it is `False`
    come back **exactly** `0.0` and the exponential is never evaluated there. That is how `drag.py`
    keeps a degenerate row (a body at zero separation from its parent, whose altitude reads
    `-r_ref`) from overflowing `exp`, under `np.errstate(all="raise")`.
    """
    return piecewise_exponential_density(
        altitude_km, BASE_ALTITUDE_KM, BASE_DENSITY_KG_M3, SCALE_HEIGHT_KM, valid)


def piecewise_exponential_density(
    altitude_km: ArrayKilometers,
    base_altitude_km: ArrayKilometers,
    base_density: ArrayFloat,
    scale_height_km: ArrayKilometers,
    valid: Optional[NDArray[np.bool_]] = None,
) -> ArrayFloat:
    """
    `rho_k exp(-(h - h_k) / H_k)` in the band `k` whose base `h_k` is the largest not exceeding `h`,
    kg/m^3 - the evaluator behind both `layered_density` (Vallado's 28 bands) and the NRLMSIS 2.0
    profile (`msis_bridge.py`, several hundred bands, `H_k` chosen so each band closes exactly on the
    next node, which makes this log-linear interpolation of `ln rho`).

    `base_altitude_km` must be strictly increasing and every `scale_height_km` positive; the tables
    are validated where they are built, not here. The terminal bands extrapolate at both ends, and
    `valid` behaves exactly as described in `layered_density`.
    """
    h = np.asarray(altitude_km, dtype=np.float64)
    band = np.clip(np.searchsorted(base_altitude_km, h, side="right") - 1, 0, base_altitude_km.size - 1)
    exponent = (h - base_altitude_km[band]) / scale_height_km[band]
    if valid is None:
        unmasked: ArrayFloat = base_density[band] * np.exp(-exponent)
        return unmasked
    factor = np.exp(-exponent, out=np.zeros_like(exponent), where=valid)
    out: ArrayFloat = base_density[band] * factor
    return out


def exponential_density(
    altitude_km: ArrayKilometers,
    rho0: ArrayFloat,
    h0: ArrayKilometers,
    scale_height_km: ArrayKilometers,
    valid: Optional[NDArray[np.bool_]] = None,
) -> ArrayFloat:
    """
    Single-band exponential density `rho0 exp(-(h - h0) / H)`, kg/m^3 — the law `drag.py` has always
    used, factored out unchanged so the two laws can be compared side by side and so the kernel reads
    as a choice between two named functions.

    `valid` behaves as in `layered_density`, and here it must **also** exclude rows with
    `scale_height_km <= 0`: that is `drag.py`'s silent-no-op convention for a body whose drag bit was
    set without coefficients, and the division is skipped, not merely masked afterwards.
    """
    h = np.asarray(altitude_km, dtype=np.float64)
    if valid is None:
        unmasked: ArrayFloat = rho0 * np.exp(-(h - h0) / scale_height_km)
        return unmasked
    exponent = np.divide(h - h0, scale_height_km, out=np.zeros_like(h), where=valid)
    factor = np.exp(-exponent, out=np.zeros_like(h), where=valid)
    out: ArrayFloat = rho0 * factor
    return out
