"""
NRLMSIS 2.0 at the boundary: `pymsis` evaluated **once, at configuration time**, into an altitude
profile that `drag.py`'s kernel reads as `atmosphere.DENSITY_MODEL_MSIS` (2.0).

**Reference.** The model is not implemented here - `CLAUDE.md`'s "Do not reimplement" rule. The
`pymsis` package (G. Lucas et al., SWxTREC; `pip install orbital_engine[msis]`, >= 0.10) wraps the
NRL Fortran code defined in:

- Emmert, J. T., Drob, D. P., Picone, J. M., et al., *NRLMSIS 2.0: A whole-atmosphere empirical
  model of temperature and neutral species densities*, Earth and Space Science 8, e2020EA001321
  (2021). **Citation from memory, unverified against the text.**

`pymsis` defaults to MSIS **2.1**; this module pins `version=2.0` (`MSIS_VERSION`) because 2.0 is the
model asked for and the published one above. The output column is `pymsis.Variable.MASS_DENSITY`
(index 0, **kg/m^3** - the unit `drag.py`'s single `1e3` conversion already assumes; the other
columns are number densities in m^-3 and temperature in K). Altitude goes in as **km**.
`tests/validation/test_msis.py` pins both without trusting this paragraph: the averaged mass density
must equal the averaged sum of `n_s m_s` over MSIS's own species columns and masses, which fails by
orders of magnitude for any other column and by 1e3 for a units slip.

The boundary, and why it is here
--------------------------------
`CLAUDE.md`: "No stateful third-party objects inside a step." `pymsis` holds process-global Fortran
state behind a lock and costs ~1.3 us per point - a 1536-point average per RK4 stage per body would be
unaffordable, and a call inside a step is forbidden regardless. So MSIS is evaluated here, once per
distinct solar-activity triple `(f107, f107a, ap)`, into a `MsisProfile`: density at 601 altitude
nodes, 0 to 1000 km (1 km spacing below 200 km, 2 km above). The kernel evaluates that profile with
the piecewise-exponential machinery the Vallado table already uses (`atmosphere.
piecewise_exponential_density`), with each band's scale height chosen so the band closes exactly on
the next node - i.e. **log-linear interpolation of `ln rho` between nodes**. Above 1000 km the top band
is extrapolated (as the table's is); below 0 km the bottom one.

*Interpolation error*, `|d^2 ln rho / dh^2| dh^2 / 8` at a band's midpoint, with the curvature taken
from MSIS itself (measured over solar min / moderate / max): below 150 km the curvature reaches
7.9e-3 /km^2, so **<= 1.0e-3** at 1 km spacing; 150-200 km, 9.3e-4 -> **1.2e-4**; 200-250 km, 2.0e-4
at 2 km -> **1.0e-4**; above 250 km **<= 5e-5**. Negligible against every other term in the model.

*Memoisation.* `msis_profile(f107, f107a, ap)` is a pure function of its three floats (MSIS is
deterministic, and no date, file or network enters - see below), memoised in a module dict. That is
the only module-level state, and it cannot leak between simulations because a profile depends on
nothing but its key. `drag.py`'s `validate_coefficients` hook calls `msis_profile` for every triple a
configuration asks for, **before** any mask bit is set; the kernel then reads the memo through
`cached_msis_profile`, which **raises** rather than evaluating on a miss (e.g. a `force_model_params`
row switched to MSIS by direct assignment, bypassing `enable_force_model`). A miss is a configuration
error, and computing it lazily would put `pymsis` inside the step.

*No network, ever.* `pymsis.calculate` downloads CelesTrak's space-weather file whenever any of
`f107s`, `f107as`, `aps` is `None`. This module always passes all three, and `drag.py` refuses
`density_model=DENSITY_MODEL_MSIS` unless `f107`, `f107a` and `ap` are given in the same call. So
solar activity is **configuration data** - a `sweep.ForceModelSpec` coefficient, hence a sweep axis -
and never looked up by date. `test_msis.py` computes a fresh profile with sockets and the download
function patched to raise.

What the profile averages over, and what that discards
------------------------------------------------------
The profile is the **global, all-local-time, all-season mean** of MSIS mass density at each altitude,
for constant `(f107, f107a, ap)`:

- *latitude*: 8-point Gauss-Legendre in `sin(lat)` (area weighting). Converged to 6e-6 against 32
  points.
- *local solar time*: 8 longitudes at 00:00 UT, i.e. local times 01:30, 04:30, ..., 22:30. At fixed UT
  longitude and local time are the same coordinate, so this also averages MSIS's longitude terms.
  Exact: 8 equally spaced samples integrate harmonics up to 7, and MSIS's tides stop at 3.
- *day of year*: 24 integer days (MSIS takes an integer day of year), year 2001. Converged to 5e-5
  against 48.

Arithmetic mean of `rho`, not of `ln rho`: drag is linear in density, so the mean force over the
sampled conditions is the mean density's. Altitude is MSIS's geodetic altitude, which the kernel reads
as its spherical `|r| - r_ref` - the same identification `drag.py` already makes for the table.

**Discarded, deliberately**, and measured from MSIS itself (moderate activity, `test_msis.py`):

- *the diurnal bulge*: at 400 km on the equator at equinox, density across local time spans a
  factor **2.31** (1.68 at 292 km). An orbit whose node is fixed in local time (sun-synchronous) sees
  one side of it all the time; the profile gives it the mean.
- *seasons*: the global mean swings by a factor **1.62** over the year at 400 km (1.42 at 292 km),
  the semi-annual variation. A profile is one number per altitude for the whole year.
- *latitude*: zonal means within +-4 % of the global mean. A 51.6 deg orbit's latitude sampling
  differs from area weighting by **+0.2 %**.
- *time variation of the indices*: `(f107, f107a, ap)` are constants for the run - no storms, no 27-day
  rotation, no solar-cycle trend. That is exactly what makes solar activity a sweep axis instead of a
  date lookup.

The first two are factors of 1.4 to 2.3 locally, but a satellite in a non-sun-synchronous orbit
averages over local time and season within days to weeks, which is what a global mean represents. The
index choice moves the mean by a factor of **24** at 400 km between ECSS low and high activity - far
larger than anything the average throws away, and the reason this law exists.

Solar-activity presets
----------------------
`SOLAR_ACTIVITY_LOW`, `_MODERATE` and `_HIGH` are the ECSS-E-ST-10-04C / ISO 14222 long-term levels as
recalled, **from memory and unverified**: low `F10.7 = F10.7a = 65`, `Ap = 0`; moderate 140, 140, 15;
high 250, 250, 45. Each is a plain mapping to splat into a `ForceModelSpec`'s coefficients, e.g.
`{"density_model": DENSITY_MODEL_MSIS, **SOLAR_ACTIVITY_MODERATE}` - or `msis_coefficients(...)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Final, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .atmosphere import DENSITY_MODEL_MSIS, piecewise_exponential_density
from .custom_types import ArrayFloat, ArrayKilometers

__all__ = [
    "MSIS_VERSION", "MSIS_ALTITUDE_GRID_KM", "MSIS_SOLAR_COEFFICIENTS", "MAX_AP",
    "N_DAYS", "N_LOCAL_TIMES", "N_LATITUDES", "QUADRATURE_YEAR",
    "SOLAR_ACTIVITY_LOW", "SOLAR_ACTIVITY_MODERATE", "SOLAR_ACTIVITY_HIGH",
    "MsisProfile", "msis_profile", "cached_msis_profile", "msis_mean_density", "msis_density",
    "msis_coefficients", "check_solar_activity", "quadrature_nodes",
]

#: The NRLMSIS version passed to `pymsis.calculate`. pymsis's own default is 2.1.
MSIS_VERSION: Final[float] = 2.0

#: Altitude nodes of every profile, km: 1 km spacing to 200 km, 2 km to 1000 km (601 nodes). The
#: spacing is sized by the interpolation bound in the module docstring.
MSIS_ALTITUDE_GRID_KM: Final[ArrayFloat] = np.concatenate(
    [np.arange(0.0, 200.0, 1.0), np.arange(200.0, 1000.0 + 1.0, 2.0)])
MSIS_ALTITUDE_GRID_KM.flags.writeable = False

#: The `"drag"` coefficient names that carry the solar-activity indices, in column order.
MSIS_SOLAR_COEFFICIENTS: Final = ("f107", "f107a", "ap")

#: The Ap index's scale ends at 400.
MAX_AP: Final[float] = 400.0

# Quadrature of the average - see "What the profile averages over".
N_DAYS: Final[int] = 24
N_LOCAL_TIMES: Final[int] = 8
N_LATITUDES: Final[int] = 8
QUADRATURE_YEAR: Final[int] = 2001

SOLAR_ACTIVITY_LOW: Final[Mapping[str, float]] = MappingProxyType(
    {"f107": 65.0, "f107a": 65.0, "ap": 0.0})
SOLAR_ACTIVITY_MODERATE: Final[Mapping[str, float]] = MappingProxyType(
    {"f107": 140.0, "f107a": 140.0, "ap": 15.0})
SOLAR_ACTIVITY_HIGH: Final[Mapping[str, float]] = MappingProxyType(
    {"f107": 250.0, "f107a": 250.0, "ap": 45.0})

_ActivityKey = Tuple[float, float, float]


@dataclass(frozen=True)
class MsisProfile:
    """
    One solar-activity triple's mean density profile - read-only arrays, all shaped like
    `MSIS_ALTITUDE_GRID_KM`. `scale_height_km[k]` closes band `k` exactly on node `k + 1`
    (`dh / ln(rho_k / rho_{k+1})`); the last entry repeats the one before it, so extrapolation above
    the top node continues the top band's slope.
    """
    f107: float
    f107a: float
    ap: float
    altitude_km: ArrayFloat
    density_kg_m3: ArrayFloat
    scale_height_km: ArrayFloat


def check_solar_activity(f107: float, f107a: float, ap: float) -> None:
    """Raise `ValueError` unless the indices are finite, the fluxes positive and `0 <= ap <= 400`."""
    values = (f107, f107a, ap)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"MSIS solar activity must be finite, got f107={f107}, f107a={f107a}, ap={ap}")
    if f107 <= 0.0 or f107a <= 0.0:
        raise ValueError(
            f"MSIS solar flux must be positive (sfu), got f107={f107}, f107a={f107a}; an unwritten "
            f"coefficient reads 0.0, so this usually means the indices were never set")
    if not 0.0 <= ap <= MAX_AP:
        raise ValueError(f"MSIS ap must lie in [0, {MAX_AP}], got ap={ap}")


def msis_coefficients(activity: Mapping[str, float]) -> Dict[str, float]:
    """`{"density_model": DENSITY_MODEL_MSIS, "f107": ..., "f107a": ..., "ap": ...}` for one of the
    presets (or any mapping with those three keys) - the drag coefficients a sweep config needs."""
    return {"density_model": DENSITY_MODEL_MSIS,
            **{key: float(activity[key]) for key in MSIS_SOLAR_COEFFICIENTS}}


def _pymsis() -> Any:
    try:
        import pymsis  # imported lazily: pymsis is an optional extra
    except ImportError as exc:
        raise ImportError(
            "density_model=DENSITY_MODEL_MSIS needs the optional 'pymsis' dependency: "
            "pip install 'orbital_engine[msis]'") from exc
    return pymsis


def quadrature_nodes() -> Tuple[NDArray[np.datetime64], ArrayFloat, ArrayFloat, ArrayFloat]:
    """
    `(dates, longitudes_deg, latitudes_deg, latitude_weights)` of the average: `N_DAYS` integer days
    of `QUADRATURE_YEAR` at 00:00 UT (the midpoints of 24 equal parts of the year, floored),
    `N_LOCAL_TIMES` equally spaced longitudes, and `N_LATITUDES` Gauss-Legendre nodes in `sin(lat)`
    with weights summing to 1. The day and longitude weights are uniform.
    """
    offsets = np.floor((np.arange(N_DAYS) + 0.5) * 365.0 / N_DAYS).astype(np.int64)
    dates = np.datetime64(f"{QUADRATURE_YEAR}-01-01T00:00:00") + offsets.astype("timedelta64[D]")
    longitudes: ArrayFloat = (
        (np.arange(N_LOCAL_TIMES) + 0.5) * (360.0 / N_LOCAL_TIMES)).astype(np.float64)
    x, w = np.polynomial.legendre.leggauss(N_LATITUDES)
    latitudes: ArrayFloat = np.degrees(np.arcsin(x))
    weights: ArrayFloat = 0.5 * w
    return dates, longitudes, latitudes, weights


def msis_mean_density(
    altitude_km: ArrayKilometers, f107: float, f107a: float, ap: float,
) -> ArrayFloat:
    """
    **Direct** NRLMSIS 2.0 evaluation of the profile's average (latitude, local time, day of year; see
    the module docstring) at arbitrary altitudes, kg/m^3. This is the boundary call: it imports and
    runs `pymsis`, always with all three indices given, so it never touches the network. It is what
    `msis_profile` evaluates at the grid nodes; use it directly only outside a step.
    """
    check_solar_activity(f107, f107a, ap)
    pymsis = _pymsis()
    dates, longitudes, latitudes, weights = quadrature_nodes()
    alts = np.atleast_1d(np.asarray(altitude_km, dtype=np.float64))
    n = dates.size
    raw = pymsis.calculate(
        dates, longitudes, latitudes, alts,
        np.full(n, f107), np.full(n, f107a), np.full((n, 7), ap),
        version=MSIS_VERSION,
    )
    grid = np.asarray(raw)
    if grid.shape != (n, longitudes.size, latitudes.size, alts.size, 11):
        raise RuntimeError(f"unexpected pymsis output shape {grid.shape}")
    rho = grid[..., int(pymsis.Variable.MASS_DENSITY)].astype(np.float64)   # kg/m^3, float32 in
    mean: ArrayFloat = np.einsum("dolh,l->h", rho, weights) / (n * longitudes.size)
    return mean


_PROFILES: Dict[_ActivityKey, MsisProfile] = {}


def _readonly(a: ArrayFloat) -> ArrayFloat:
    out: ArrayFloat = np.ascontiguousarray(a, dtype=np.float64)
    out.flags.writeable = False
    return out


def _build_profile(f107: float, f107a: float, ap: float) -> MsisProfile:
    """Evaluate MSIS on the grid and derive each band's closing scale height. Uncached."""
    rho = msis_mean_density(MSIS_ALTITUDE_GRID_KM, f107, f107a, ap)
    if not (np.all(np.isfinite(rho)) and np.all(rho > 0.0) and np.all(np.diff(rho) < 0.0)):
        raise ValueError(
            f"MSIS mean density for f107={f107}, f107a={f107a}, ap={ap} is not finite, positive and "
            f"strictly decreasing with altitude; it cannot be a piecewise-exponential profile")
    h = np.empty_like(rho)
    h[:-1] = np.diff(MSIS_ALTITUDE_GRID_KM) / np.log(rho[:-1] / rho[1:])
    h[-1] = h[-2]
    return MsisProfile(f107=f107, f107a=f107a, ap=ap, altitude_km=MSIS_ALTITUDE_GRID_KM,
                       density_kg_m3=_readonly(rho), scale_height_km=_readonly(h))


def msis_profile(f107: float, f107a: float, ap: float) -> MsisProfile:
    """
    The mean density profile for one solar-activity triple, memoised. **Configuration time only** -
    on a cache miss this runs `pymsis` (~1 s). `drag.py`'s `validate_coefficients` calls it for every
    triple `enable_force_model` is asked for, so a kernel never meets an unevaluated one.
    """
    key = (float(f107), float(f107a), float(ap))
    profile = _PROFILES.get(key)
    if profile is None:
        profile = _build_profile(*key)
        _PROFILES[key] = profile
    return profile


def cached_msis_profile(f107: float, f107a: float, ap: float) -> MsisProfile:
    """
    The step-time lookup: the memoised profile, or `LookupError` if it was never evaluated. Never
    calls `pymsis` - see "Memoisation" in the module docstring for why a miss raises.
    """
    try:
        return _PROFILES[(float(f107), float(f107a), float(ap))]
    except KeyError:
        raise LookupError(
            f"no NRLMSIS 2.0 profile was evaluated for f107={f107}, f107a={f107a}, ap={ap}. MSIS is "
            f"evaluated at configuration time only: enable 'drag' with density_model="
            f"DENSITY_MODEL_MSIS through Simulation.enable_force_model / sweep.apply_config, or call "
            f"msis_bridge.msis_profile(...) before stepping.") from None


def msis_density(
    altitude_km: ArrayKilometers,
    activity: NDArray[np.float64],
    valid: Optional[NDArray[np.bool_]] = None,
) -> ArrayFloat:
    """
    Step-time density under `DENSITY_MODEL_MSIS`, kg/m^3: each row's altitude evaluated on the
    memoised profile of its own `(f107, f107a, ap)` row of `activity` (shape `(n, 3)`). Rows where
    `valid` is `False` come back exactly `0.0` and are never looked up, so a non-MSIS row's zero
    indices cannot raise. Rows are grouped by distinct triple - one group in any ordinary
    configuration - and each group is one `piecewise_exponential_density` call.
    """
    h = np.asarray(altitude_km, dtype=np.float64)
    out: ArrayFloat = np.zeros_like(h)
    rows = np.flatnonzero(valid) if valid is not None else np.arange(h.size)
    if rows.size == 0:
        return out
    rows_activity = activity[rows]
    first = rows_activity[0]
    if np.all(rows_activity == first):
        # The ordinary case - every MSIS row shares one triple - skips `np.unique(axis=0)`, which
        # costs ~100 us per call and would double the kernel's cost for nothing.
        groups = [(first, rows)]
    else:
        keys, inverse = np.unique(rows_activity, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        groups = [(keys[g], rows[inverse == g]) for g in range(keys.shape[0])]
    for key, sel in groups:
        profile = cached_msis_profile(float(key[0]), float(key[1]), float(key[2]))
        out[sel] = piecewise_exponential_density(
            h[sel], profile.altitude_km, profile.density_kg_m3, profile.scale_height_km)
    return out
