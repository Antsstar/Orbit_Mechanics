"""
NRLMSIS 2.0 **with its diurnal bulge**: `pymsis` evaluated once, at configuration time, into a table
of density over **altitude x latitude x local solar time** for one day, which `drag.py`'s kernel reads
as `atmosphere.DENSITY_MODEL_MSIS_DIURNAL` (3.0) - the time-resolved companion of `msis_bridge.py`'s
global-mean profile (`DENSITY_MODEL_MSIS`, 2.0).

**Why it exists.** `msis_bridge.py`'s profile averages MSIS over latitude, local time and season, and
records what that throws away: a 2.30x day/night bulge at 400 km and a 1.63x semi-annual swing. An
orbit whose plane precesses through every local time sees the mean over a couple of months; a
sun-synchronous orbit's plane is *locked* to one local time and never does. Whether that matters to a
Delta-v budget is a question for numbers, which this law makes answerable - see `docs/architecture.md`,
"NRLMSIS 2.0 with the diurnal bulge", and `benchmarks/msis_diurnal_sweep.py`.

**Reference.** The model is NRLMSIS 2.0 via `pymsis`, exactly as in `msis_bridge.py` (Emmert et al.
2021, *Earth and Space Science* 8, e2020EA001321; citation from memory), `version=2.0` pinned, mass
density column, altitude in km. Nothing here reimplements it. Local solar time needs the Sun, from
`solar_ephemeris.py` (the Astronomical Almanac's low-precision series).

What the table is
-----------------
For one solar-activity triple `(f107, f107a, ap)` and one UT calendar **day** `D`, the table holds

    ln rho(h, phi, s) = ln [ (1/6) sum_{UT in 0,4,..,20 h} rho_MSIS(D + UT, lon = 15 (s - UT), phi, h) ]

on `MSIS_ALTITUDE_GRID_KM` (601 nodes, 0-1000 km, the averaged profile's own grid) x latitude
`phi` = -90..90 deg every 5 deg (37) x mean local solar time `s` = 0..23.5 h every 0.5 h (48, periodic).
MSIS forms its local time as `UT + lon / 15`, so at fixed `s` the UT samples walk the longitude round
the globe: the table is the **zonal mean at fixed local time**, not MSIS at one arbitrary longitude.

*Why average over UT.* MSIS keeps a genuine longitude/UT dependence at fixed local time (geomagnetic
coordinates, non-migrating tides). Measured at 2024-03-20, moderate activity: one UT's slice differs
from the zonal mean by **1.3 % rms, 4.1 % max** at 400 km (2.7 % / 8.8 % at 292 km, 3.4 % / 13 % at
500 km), while the *global* mean moves by at most 0.16 % with UT. A satellite passes over every
longitude within a day, so the orbit sees the zonal mean plus a fluctuation of that rms that averages
down orbit by orbit; tabulating one UT would instead freeze one arbitrary longitude to each local time
for the whole run. Six UT samples integrate the longitude/UT harmonics exactly: against 24 samples the
table agrees to **3e-6** (the float32 floor; four samples leave 2.7e-4).

*Cost.* 6 x 601 x 37 x 48 = 6.4 million `pymsis` points, **~9 s** per distinct `(f107, f107a, ap, day)`
- against 1.3 s for the averaged profile - memoised, 8.5 MB of float64.

What is frozen, and what is not
-------------------------------
- **The season is frozen at the epoch's day.** MSIS takes an integer day of year (`pymsis` truncates
  the date); the table is built for the UT day containing `epoch_days` and used for the whole run. The
  Sun's declination, the semi-annual variation and the annual asymmetry therefore stay at that day.
  Their drift is a stated error, measured from MSIS itself in `tests/validation/test_msis_diurnal.py`:
  **at the 2024 March equinox the global mean at 400 km moves by +3.9 % over 10 days** (the semi-annual
  maximum is in April), so a 10-day run averages ~2 % low of a season-following model; near a
  semi-annual extremum it is well under 1 %.
- **Local time is not frozen.** It is computed at every kernel call from the satellite's own position
  and the kernel's `t`: `s = 12 h + (alpha_sat - L(t)) / 15`, with `L` the Sun's mean longitude at
  `epoch_days + t / 86400` (`solar_ephemeris.local_solar_time_hours`) - so it follows both the
  satellite round its orbit and the Sun's ~1 deg/day motion. RK4 hands each stage its own time
  (`tests/validation/test_tesseral.py` proves that contract), and this law reads it.
- **Why `L` and not the Sun's right ascension.** MSIS's own local time is `UT + lon / 15` - mean solar
  time. `solar_ephemeris.py` shows `12 h + (alpha_sat - L) / 15` is that same quantity to 0.24 s; the
  apparent-sun version differs by the equation of time (-14.2 to +16.4 min), which at the steepest point
  of the bulge is ~3 % of density. This is a deliberate deviation from "LST = 12 h + (alpha_sat -
  alpha_sun)", made because the table is indexed by MSIS's local time and a faithful wrap must look it
  up with the same definition.
- **Latitude is the geocentric declination** `asin(z / |r|)`; MSIS takes geodetic latitude. The two
  differ by `f sin 2 phi`, at most 0.19 deg at 45 deg, against a largest latitude gradient of
  `d ln rho / d phi` ~ 1.3 %/deg in the table (400 km): <= **0.25 %** locally, and odd in latitude over
  an orbit. **Altitude** is `drag.py`'s spherical `|r| - r_ref`, common to every density law (see
  `drag.py`), not MSIS's geodetic height.
- **No Earth rotation angle is needed.** Local solar time is inertial geometry, and the zonal mean has
  no longitude left in it.

Interpolation, and its error
----------------------------
Trilinear in `ln rho`: linear in altitude between nodes (the averaged profile's log-linear scheme,
`|f_hh| dh^2 / 8` <= 5e-5 above 250 km, <= 1e-3 below 150 km; terminal bands extrapolated at both
ends exactly as the profile does), linear in latitude between 5 deg nodes (never extrapolated: the
nodes include both poles), and **linear and periodic in local time** between 30 min nodes (23:30 wraps
to 00:00). Per axis the error is `|f''| delta^2 / 8` at a cell midpoint; with `f''` from MSIS (moderate
activity, equinox) the local-time term is <= 2.0e-3 at 400 km (3.2e-3 at 500, 3.8e-3 at 700) and the
latitude term <= 1.4e-3 at 400 km (2.4e-3 at 500, 3.0e-3 at 700). A bias of that size has the sign of
`-f''`, which over a full cycle of local time integrates to zero, so orbit-averaged densities carry
much less than the local bound. `test_msis_diurnal.py` asserts the bound against direct evaluations.

The boundary
------------
Identical to `msis_bridge.py`'s, for the same reasons. `msis_diurnal_table(f107, f107a, ap,
epoch_days)` is a pure function of its four floats - through the UT day, so two epochs on one day share
one table - memoised in module dicts; `drag.py`'s `validate_coefficients` calls it for every key a
configuration asks for, before any bit is set. The kernel reads the memo through
`cached_msis_diurnal_table`, which **raises `LookupError`** on a miss rather than calling `pymsis` in a
step. All three indices and the epoch are mandatory in the same call, so `pymsis` never downloads
anything.

Not modelled: winds, time variation of the indices within the run, the season after the epoch day,
geodetic altitude, and the UT/longitude structure the zonal mean removes (quantified above). No
compiled twin - a body on this law sends the Cowell set down the NumPy path
(`Simulation._refresh_cowell_plan`); the fused twin reads density laws 0-2 only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Final, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .atmosphere import DENSITY_MODEL_MSIS_DIURNAL
from .custom_types import ArrayFloat, ArrayKilometers, ScalarSeconds
from .msis_bridge import (
    MSIS_ALTITUDE_GRID_KM, MSIS_SOLAR_COEFFICIENTS, MSIS_VERSION, _pymsis, check_solar_activity,
)
from .solar_ephemeris import J2000_UT, SECONDS_PER_DAY, local_solar_time_hours

__all__ = [
    "DIURNAL_LATITUDE_STEP_DEG", "DIURNAL_LATITUDE_GRID_DEG", "DIURNAL_LST_STEP_HOURS",
    "DIURNAL_LST_GRID_HOURS", "DIURNAL_UT_HOURS", "EPOCH_COEFFICIENT", "MIN_EPOCH_DAYS",
    "MAX_EPOCH_DAYS", "MsisDiurnalTable", "check_epoch", "table_date", "msis_zonal_mean_density",
    "msis_diurnal_table", "cached_msis_diurnal_table", "msis_diurnal_density",
    "msis_diurnal_coefficients",
]

#: Latitude nodes, degrees - both poles included, so latitude is never extrapolated.
DIURNAL_LATITUDE_STEP_DEG: Final[float] = 5.0
DIURNAL_LATITUDE_GRID_DEG: Final[ArrayFloat] = np.linspace(-90.0, 90.0, 37)
DIURNAL_LATITUDE_GRID_DEG.flags.writeable = False
#: Mean local solar time nodes, hours; periodic (the node after 23.5 h is 0 h).
DIURNAL_LST_STEP_HOURS: Final[float] = 0.5
DIURNAL_LST_GRID_HOURS: Final[ArrayFloat] = np.arange(48, dtype=np.float64) * 0.5
DIURNAL_LST_GRID_HOURS.flags.writeable = False
#: UT samples of the zonal mean, hours of the table's day.
DIURNAL_UT_HOURS: Final[ArrayFloat] = np.arange(6, dtype=np.float64) * 4.0
DIURNAL_UT_HOURS.flags.writeable = False

#: The `"drag"` coefficient that carries the epoch: UT days from `solar_ephemeris.J2000_UT`
#: (2000-01-01T12:00 UT) at simulation time `t = 0`.
EPOCH_COEFFICIENT: Final[str] = "epoch_days"
#: The solar series' stated validity, 1950-2050, as `epoch_days`. Outside it the epoch is refused.
MIN_EPOCH_DAYS: Final[float] = -18262.5
MAX_EPOCH_DAYS: Final[float] = 18262.5

# Altitude nodes per `pymsis` call: bounds the float32 output buffer (11 columns) to ~12 MB.
_ALTITUDE_CHUNK: Final[int] = 160

_DiurnalKey = Tuple[float, float, float, str]          # (f107, f107a, ap, "YYYY-MM-DD")
_EpochKey = Tuple[float, float, float, float]          # (f107, f107a, ap, epoch_days)


@dataclass(frozen=True)
class MsisDiurnalTable:
    """
    One `(f107, f107a, ap, day)` table - read-only arrays. `ln_density` is `(altitude, latitude, local
    time)` on `altitude_km` x `latitude_deg` x `lst_hours`, natural log of kg/m^3. `date` is the UT day
    (ISO `YYYY-MM-DD`) it was built for.
    """
    f107: float
    f107a: float
    ap: float
    date: str
    altitude_km: ArrayFloat
    latitude_deg: ArrayFloat
    lst_hours: ArrayFloat
    ln_density: ArrayFloat

    def density_at(
        self, altitude_km: ArrayKilometers, latitude_deg: ArrayFloat, lst_hours: ArrayFloat,
    ) -> ArrayFloat:
        """
        Trilinear interpolation of `ln rho` (module docstring), kg/m^3, for matching 1-D arrays.
        Altitude extrapolates the terminal bands; latitude must lie in `[-90, 90]`; local time is
        reduced modulo 24 h.
        """
        h = np.asarray(altitude_km, dtype=np.float64)
        alt = self.altitude_km
        k = np.clip(np.searchsorted(alt, h, side="right") - 1, 0, alt.size - 2)
        wh = (h - alt[k]) / (alt[k + 1] - alt[k])

        x = (np.asarray(latitude_deg, dtype=np.float64) - self.latitude_deg[0]) / DIURNAL_LATITUDE_STEP_DEG
        j = np.clip(np.floor(x).astype(np.int64), 0, self.latitude_deg.size - 2)
        wl = x - j

        n_lst = self.lst_hours.size
        s = np.mod(np.asarray(lst_hours, dtype=np.float64), 24.0) / DIURNAL_LST_STEP_HOURS
        s_floor = np.floor(s)
        i0 = s_floor.astype(np.int64) % n_lst
        i1 = (i0 + 1) % n_lst
        ws = s - s_floor

        f = self.ln_density

        def plane(kk: NDArray[np.int64]) -> ArrayFloat:
            lo = (1.0 - ws) * f[kk, j, i0] + ws * f[kk, j, i1]
            hi = (1.0 - ws) * f[kk, j + 1, i0] + ws * f[kk, j + 1, i1]
            out: ArrayFloat = (1.0 - wl) * lo + wl * hi
            return out

        ln_rho = (1.0 - wh) * plane(k) + wh * plane(k + 1)
        rho: ArrayFloat = np.exp(ln_rho)
        return rho


def check_epoch(epoch_days: float) -> None:
    """Raise `ValueError` unless `epoch_days` is finite and inside the solar series' 1950-2050."""
    if not math.isfinite(epoch_days):
        raise ValueError(f"MSIS diurnal epoch_days must be finite, got {epoch_days}")
    if not MIN_EPOCH_DAYS <= epoch_days <= MAX_EPOCH_DAYS:
        raise ValueError(
            f"MSIS diurnal epoch_days={epoch_days} lies outside 1950-2050 ([{MIN_EPOCH_DAYS}, "
            f"{MAX_EPOCH_DAYS}] days from J2000), where the low-precision solar series is stated to "
            f"hold 0.01 deg")


def table_date(epoch_days: float) -> np.datetime64:
    """The UT calendar day (`datetime64[D]`) containing `epoch_days` - the day the table is built for."""
    instant = J2000_UT + np.timedelta64(int(round(epoch_days * SECONDS_PER_DAY * 1e9)), "ns")
    day: np.datetime64 = instant.astype("datetime64[D]")
    return day


def msis_diurnal_coefficients(activity: Mapping[str, float], epoch_days: float) -> Dict[str, float]:
    """`{"density_model": DENSITY_MODEL_MSIS_DIURNAL, "f107": ..., "f107a": ..., "ap": ...,
    "epoch_days": ...}` - the drag coefficients a sweep config needs, from a preset such as
    `msis_bridge.SOLAR_ACTIVITY_MODERATE` and an epoch (`solar_ephemeris.epoch_days(...)`)."""
    return {"density_model": DENSITY_MODEL_MSIS_DIURNAL,
            **{key: float(activity[key]) for key in MSIS_SOLAR_COEFFICIENTS},
            EPOCH_COEFFICIENT: float(epoch_days)}


def msis_zonal_mean_density(
    altitude_km: ArrayKilometers,
    latitude_deg: ArrayFloat,
    lst_hours: ArrayFloat,
    f107: float, f107a: float, ap: float,
    date: np.datetime64,
) -> ArrayFloat:
    """
    **Direct** NRLMSIS 2.0 zonal mean at fixed local time (module docstring) on the grid
    `altitude_km x latitude_deg x lst_hours` for UT day `date`, kg/m^3, shaped
    `(n_alt, n_lat, n_lst)`: the mean over `DIURNAL_UT_HOURS` of `pymsis` at `lon = 15 (lst - UT)`.
    This is the boundary call - it imports and runs `pymsis`, always with all three indices, so it
    never touches the network. `msis_diurnal_table` evaluates it on the table's grid.
    """
    check_solar_activity(f107, f107a, ap)
    pymsis = _pymsis()
    alts = np.atleast_1d(np.asarray(altitude_km, dtype=np.float64))
    lats = np.atleast_1d(np.asarray(latitude_deg, dtype=np.float64))
    lsts = np.atleast_1d(np.asarray(lst_hours, dtype=np.float64))
    day = np.asarray(date, dtype="datetime64[D]")
    column = int(pymsis.Variable.MASS_DENSITY)
    total = np.zeros((alts.size, lats.size, lsts.size))
    for ut in DIURNAL_UT_HOURS:
        when = day + np.timedelta64(int(round(ut * 3600.0)), "s")
        lons = np.mod(15.0 * (lsts - ut), 360.0)
        for start in range(0, alts.size, _ALTITUDE_CHUNK):
            chunk = alts[start:start + _ALTITUDE_CHUNK]
            raw = np.asarray(pymsis.calculate(
                when, lons, lats, chunk, [f107], [f107a], [[ap] * 7], version=MSIS_VERSION))
            # A (1, 1, 1, 1)-point call comes back in fly-through shape (1, 11); every other call in
            # grid shape (1, n_lst, n_lat, n_alt, 11). The reshape is valid for both.
            grid = raw.reshape(1, lsts.size, lats.size, chunk.size, 11)
            rho = grid[0, ..., column].astype(np.float64)                  # (lst, lat, alt), float32 in
            total[start:start + chunk.size] += np.transpose(rho, (2, 1, 0))
    mean: ArrayFloat = total / DIURNAL_UT_HOURS.size
    return mean


_TABLES: Dict[_DiurnalKey, MsisDiurnalTable] = {}
_BY_EPOCH: Dict[_EpochKey, MsisDiurnalTable] = {}


def _readonly(a: ArrayFloat) -> ArrayFloat:
    out: ArrayFloat = np.ascontiguousarray(a, dtype=np.float64)
    out.flags.writeable = False
    return out


def _build_table(f107: float, f107a: float, ap: float, date: np.datetime64) -> MsisDiurnalTable:
    """Evaluate MSIS on the table's grid for one day. Uncached."""
    rho = msis_zonal_mean_density(MSIS_ALTITUDE_GRID_KM, DIURNAL_LATITUDE_GRID_DEG,
                                  DIURNAL_LST_GRID_HOURS, f107, f107a, ap, date)
    if not (np.all(np.isfinite(rho)) and np.all(rho > 0.0) and np.all(np.diff(rho, axis=0) < 0.0)):
        raise ValueError(
            f"MSIS zonal-mean density for f107={f107}, f107a={f107a}, ap={ap} on {date} is not finite, "
            f"positive and strictly decreasing with altitude everywhere; it cannot be interpolated in "
            f"ln rho as a piecewise-exponential profile")
    return MsisDiurnalTable(
        f107=f107, f107a=f107a, ap=ap, date=str(np.asarray(date, dtype="datetime64[D]")),
        altitude_km=MSIS_ALTITUDE_GRID_KM, latitude_deg=DIURNAL_LATITUDE_GRID_DEG,
        lst_hours=DIURNAL_LST_GRID_HOURS, ln_density=_readonly(np.log(rho)),
    )


def _epoch_key(f107: float, f107a: float, ap: float, epoch_days: float) -> _EpochKey:
    return (float(f107), float(f107a), float(ap), float(epoch_days))


def msis_diurnal_table(f107: float, f107a: float, ap: float, epoch_days: float) -> MsisDiurnalTable:
    """
    The table for one solar-activity triple and epoch, memoised - **configuration time only**: on a
    miss this runs `pymsis` (~9 s). A pure function of its four floats; it depends on `epoch_days`
    only through `table_date`, and epochs on one UT day share one table object.
    """
    key = _epoch_key(f107, f107a, ap, epoch_days)
    table = _BY_EPOCH.get(key)
    if table is not None:
        return table
    check_solar_activity(*key[:3])
    check_epoch(key[3])
    date = table_date(key[3])
    day_key: _DiurnalKey = (key[0], key[1], key[2], str(date))
    table = _TABLES.get(day_key)
    if table is None:
        table = _build_table(key[0], key[1], key[2], date)
        _TABLES[day_key] = table
    _BY_EPOCH[key] = table
    return table


def cached_msis_diurnal_table(f107: float, f107a: float, ap: float, epoch_days: float) -> MsisDiurnalTable:
    """
    The step-time lookup: the memoised table, or `LookupError` if it was never evaluated. Never calls
    `pymsis` - a miss is a configuration error, and evaluating lazily would put `pymsis` in a step.
    """
    try:
        return _BY_EPOCH[_epoch_key(f107, f107a, ap, epoch_days)]
    except KeyError:
        raise LookupError(
            f"no NRLMSIS 2.0 diurnal table was evaluated for f107={f107}, f107a={f107a}, ap={ap}, "
            f"epoch_days={epoch_days}. MSIS is evaluated at configuration time only: enable 'drag' "
            f"with density_model=DENSITY_MODEL_MSIS_DIURNAL through Simulation.enable_force_model / "
            f"sweep.apply_config, or call msis_diurnal.msis_diurnal_table(...) before stepping."
        ) from None


def msis_diurnal_density(
    altitude_km: ArrayKilometers,
    position_km: NDArray[np.float64],
    t: ScalarSeconds,
    coefficients: NDArray[np.float64],
    valid: Optional[NDArray[np.bool_]] = None,
) -> ArrayFloat:
    """
    Step-time density under `DENSITY_MODEL_MSIS_DIURNAL`, kg/m^3. Row `n` is evaluated on the memoised
    table of its own `(f107, f107a, ap, epoch_days)` row of `coefficients` (`(n, 4)`), at altitude
    `altitude_km[n]`, geocentric latitude `atan2(z, sqrt(x^2 + y^2))` of `position_km[n]` (`(n, 3)`,
    relative to the Earth's centre, equator-of-date frame with `+z` the spin axis) and mean local solar
    time `12 h + (alpha - L) / 15` with `L` the Sun's mean longitude at `epoch_days + t / 86400`.

    Rows where `valid` is `False` come back exactly `0.0` and are never looked up. Rows are grouped by
    distinct key (one group in any ordinary configuration); `pymsis` is never called.
    """
    h = np.asarray(altitude_km, dtype=np.float64)
    out: ArrayFloat = np.zeros_like(h)
    rows = np.flatnonzero(valid) if valid is not None else np.arange(h.size)
    if rows.size == 0:
        return out
    keys = coefficients[rows]
    first = keys[0]
    groups: List[Tuple[NDArray[np.float64], NDArray[np.int64]]]
    if np.all(keys == first):
        groups = [(first, rows)]
    else:
        distinct, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        groups = [(distinct[g], rows[inverse == g]) for g in range(distinct.shape[0])]
    for key, sel in groups:
        table = cached_msis_diurnal_table(float(key[0]), float(key[1]), float(key[2]), float(key[3]))
        r = position_km[sel]
        latitude = np.degrees(np.arctan2(r[:, 2], np.hypot(r[:, 0], r[:, 1])))
        lst = local_solar_time_hours(r, float(key[3]) + float(t) / SECONDS_PER_DAY)
        out[sel] = table.density_at(h[sel], latitude, lst)
    return out
