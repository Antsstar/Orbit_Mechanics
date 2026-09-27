"""
A low-precision analytic solar ephemeris: the Sun's direction and mean longitude as a **pure function
of time**, for the one thing in the engine that needs to know where the Sun is without a body in the
arena - the local solar time of `atmosphere.DENSITY_MODEL_MSIS_DIURNAL` (`msis_diurnal.py`).

This is not a planetary ephemeris and does not replace one (`CLAUDE.md`: planetary ephemerides are
`jplephem`'s). It is the closed-form series every astrodynamics text carries for "where is the Sun,
to a hundredth of a degree", which is all local solar time needs: 0.01 deg of solar right ascension
is 2.4 s of local time.

Reference
---------
The Astronomical Almanac, Section C, "Low-precision formulas for the Sun" (`n` days from J2000.0):

    L = 280.460 deg + 0.9856474 deg n          mean longitude, corrected for aberration
    g = 357.528 deg + 0.9856003 deg n          mean anomaly
    lambda = L + 1.915 deg sin g + 0.020 deg sin 2g      ecliptic longitude (beta = 0)
    R = 1.00014 - 0.01671 cos g - 0.00014 cos 2g         distance, AU
    eps = 23.439 deg - 0.0000004 deg n         obliquity of the ecliptic
    alpha = atan2(cos eps sin lambda, cos lambda),   delta = asin(sin eps sin lambda)
    equation of time = L - alpha               (x 4 min/deg)

stated as good to **0.01 deg** in `alpha` and `delta` between 1950 and 2050. Vallado, *Fundamentals of
Astrodynamics and Applications*, 4th ed., Sec. 5.1, Algorithm 29 ("SUN") is the same series written in
Julian centuries (`36000.771 T` = `0.9856474 n`). **Both citations, and the coefficients, are from
memory and unverified against the texts.** `tests/validation/test_msis_diurnal.py` checks the series
against facts that do not depend on either text: the declination through zero at the remembered
2024 equinox instants and at `+-eps` at the solstices, the ecliptic longitude through 0/90/180/270
deg at those instants, a year-average right-ascension rate of `360 deg / 365.2422 d`, the equation
of time's published extremes and the perihelion/aphelion distances.

Time scale
----------
`days` is **UT days from 2000-01-01T12:00 UT** (`J2000_UT`) - the J2000.0 instant read on the UT scale,
the same `n` the series takes. The series is defined on TT; the 64-69 s of `Delta T` over 2000-2025
moves the Sun by <= 0.0008 deg, a tenth of the series' own error, so the engine does not carry a
second time scale for it (IAU time scales are `pyerfa`'s, `CLAUDE.md`). `epoch_days(...)` converts a
`numpy.datetime64` (UTC; leap seconds ignored, likewise sub-arcsecond).

Frame
-----
Right ascension and declination are referred to the **equator and equinox of date** (the series has
no precession split out; it is the apparent-ish Sun of the date). The engine's inertial frame is
taken to be that equator, with `+z` the spin axis - the assumption `geopotential.py`, `drag.py` and
`sgp4_bridge.py` (TEME) already make. Local solar time is a *difference* of two right ascensions, so
only a mismatch between the arena's frame and the equinox of date enters it: an arena written in the
J2000 equator carries the accumulated precession, 50.3 arcsec/yr - 0.34 deg, i.e. 80 s of local time,
at a 2024 epoch. That is 4 % of the diurnal table's 30 min node spacing, and is not corrected.

Mean versus apparent solar time
-------------------------------
`local_solar_time_hours` returns **mean** local solar time, `12 h + (alpha_sat - L) / 15`, with `L`
the Sun's mean longitude - *not* `12 h + (alpha_sat - alpha_sun) / 15`, the apparent solar time. The
reason is the model it feeds: NRLMSIS takes UT and longitude and forms its local time as
`UT + lon / 15`, which is mean solar time at that longitude. The two agree exactly once one uses
`GMST = L + 15 deg (UT - 12 h)`: IAU 1982 `GMST = 280.46061837 + 360.98564736629 n` against
`L = 280.460 + 0.9856474 n` differ by `360 n + 0.0006 deg - 3.4e-8 deg n`, i.e. by `15 UT - 180 deg`
to 0.001 deg (0.24 s) over this century. The apparent time differs from that by the equation of time,
-14.2 to +16.4 min, which would move every density lookup by up to 4 deg of local time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Union

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat

__all__ = [
    "J2000_UT", "SECONDS_PER_DAY", "SunPosition", "epoch_days", "solar_position",
    "sun_mean_longitude_deg", "equation_of_time_minutes", "local_solar_time_hours",
]

#: The instant `days = 0` refers to: 2000-01-01T12:00 **UT** (see "Time scale").
J2000_UT: Final = np.datetime64("2000-01-01T12:00:00", "ns")
SECONDS_PER_DAY: Final[float] = 86400.0

# The Astronomical Almanac's low-precision series, degrees and degrees per day (from memory).
_L0_DEG: Final[float] = 280.460
_L_RATE_DEG_PER_DAY: Final[float] = 0.9856474
_G0_DEG: Final[float] = 357.528
_G_RATE_DEG_PER_DAY: Final[float] = 0.9856003
_C1_DEG: Final[float] = 1.915
_C2_DEG: Final[float] = 0.020
_EPS0_DEG: Final[float] = 23.439
_EPS_RATE_DEG_PER_DAY: Final[float] = -4.0e-7
_R0_AU: Final[float] = 1.00014
_R1_AU: Final[float] = -0.01671
_R2_AU: Final[float] = -0.00014

_Days = Union[float, ArrayFloat]


@dataclass(frozen=True)
class SunPosition:
    """
    Geocentric Sun at one or more instants, radians and AU, each shaped like the `days` input:
    right ascension in `[0, 2 pi)`, declination, **mean** longitude `L` in `[0, 2 pi)` (the right
    ascension of the fictitious mean sun, to the precision here), ecliptic longitude in `[0, 2 pi)`,
    distance in AU, and obliquity.
    """
    right_ascension_rad: ArrayFloat
    declination_rad: ArrayFloat
    mean_longitude_rad: ArrayFloat
    ecliptic_longitude_rad: ArrayFloat
    distance_au: ArrayFloat
    obliquity_rad: ArrayFloat


def epoch_days(when: Union[np.datetime64, str]) -> float:
    """UT days from `J2000_UT` to `when` (a `numpy.datetime64` or ISO string, read as UTC)."""
    instant = np.asarray(when, dtype="datetime64[ns]")
    nanoseconds = (instant - J2000_UT).astype(np.int64)
    return float(nanoseconds) / (SECONDS_PER_DAY * 1e9)


def sun_mean_longitude_deg(days: _Days) -> ArrayFloat:
    """The Sun's mean longitude `L`, degrees in `[0, 360)` - the only solar quantity the diurnal
    density law reads at step time (see "Mean versus apparent solar time")."""
    n = np.asarray(days, dtype=np.float64)
    out: ArrayFloat = np.mod(_L0_DEG + _L_RATE_DEG_PER_DAY * n, 360.0)
    return out


def solar_position(days: _Days) -> SunPosition:
    """The full low-precision series (module docstring) at `days`, vectorised."""
    n = np.asarray(days, dtype=np.float64)
    mean_lon = np.radians(sun_mean_longitude_deg(n))
    g = np.radians(np.mod(_G0_DEG + _G_RATE_DEG_PER_DAY * n, 360.0))
    lam = mean_lon + np.radians(_C1_DEG) * np.sin(g) + np.radians(_C2_DEG) * np.sin(2.0 * g)
    eps = np.radians(_EPS0_DEG + _EPS_RATE_DEG_PER_DAY * n)
    ra = np.mod(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)), 2.0 * math.pi)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    dist = _R0_AU + _R1_AU * np.cos(g) + _R2_AU * np.cos(2.0 * g)
    return SunPosition(
        right_ascension_rad=ra, declination_rad=dec, mean_longitude_rad=mean_lon,
        ecliptic_longitude_rad=np.mod(lam, 2.0 * math.pi), distance_au=dist, obliquity_rad=eps,
    )


def equation_of_time_minutes(days: _Days) -> ArrayFloat:
    """Apparent minus mean solar time, minutes: `4 (L - alpha)`, `L - alpha` reduced to +-180 deg."""
    sun = solar_position(days)
    diff = np.degrees(sun.mean_longitude_rad - sun.right_ascension_rad)
    out: ArrayFloat = 4.0 * (np.mod(diff + 180.0, 360.0) - 180.0)
    return out


def local_solar_time_hours(position_km: NDArray[np.float64], days: _Days) -> ArrayFloat:
    """
    **Mean** local solar time at the sub-satellite point, hours in `[0, 24)`:
    `12 h + (alpha_sat - L) / 15 deg`, `alpha_sat = atan2(y, x)` of `position_km` (`(n, 3)`, relative
    to the Earth's centre in the equator-of-date frame) and `L` the Sun's mean longitude at `days`
    (scalar, or one per row). This is NRLMSIS's own `UT + lon / 15` - see the module docstring for why
    it is not the apparent time. Independent of `|r|` and of the Earth's rotation angle.
    """
    r = np.asarray(position_km, dtype=np.float64)
    alpha = np.degrees(np.arctan2(r[..., 1], r[..., 0]))
    out: ArrayFloat = np.mod(12.0 + (alpha - sun_mean_longitude_deg(days)) / 15.0, 24.0)
    return out
