"""
Validation of `solar_ephemeris.py`, the Astronomical Almanac's low-precision Sun (coefficients from
memory, unverified against the text). Nothing here needs `pymsis`.

The series is stated to hold **0.01 deg** in right ascension and declination, 1950-2050. It is checked
against facts that do not come from either cited text:

1. **The 2024 equinoxes and solstices**, instants as remembered (UTC): March 20 03:06, June 20 20:51,
   September 22 12:44, December 21 09:21. By definition the Sun's apparent ecliptic longitude is 0, 90,
   180, 270 deg then. The instants are quoted to the minute, and the longitude moves 0.95-1.02 deg/day,
   i.e. <= 0.0004 deg per half-minute of rounding; so the series' own 0.01 deg is the budget:
   `LONGITUDE_TOL_DEG = 0.01`. **Measured 0.0038, 0.0018, 0.0029, 0.0042 deg.** A wrong day count (the
   J2000 instant at 00:00 rather than 12:00) is 0.49 deg; a sign slip in the equation of centre 3.8 deg.
   The declination follows: through zero at the equinoxes (moving 0.40 deg/day there, so 0.01 deg in
   longitude is 0.004 deg of declination - asserted `< 0.01`), and `+-eps` at the solstices, with
   `eps(2024)` = 23.4355 deg from the series' own obliquity (IAU value 23.4362, a 7e-4 deg difference
   below the series' precision; asserted within 0.01 of 23.436).
2. **Right ascension advances 360 deg per tropical year** (365.2422 d), a mean 0.98565 deg/day: after
   one tropical year it must return to within the series' 0.01 deg (**measured 1.1e-4 deg**); the rate
   swings between ~0.90 and ~1.11 deg/day (the equation of time's two causes), **measured 0.896 and
   1.110**, asserted inside [0.88, 0.92] and [1.09, 1.13].
3. **The equation of time's extremes**, as published in every almanac: about -14.2 min near
   February 11-12 and +16.4 min near November 3-4. At the extremes the curve is flat, so the 0.01 deg
   precision is the tolerance on the value (0.04 min); the published values are themselves rounded to
   0.1 min, so `+-0.1 min` is asserted. **Measured -14.20 min on day 42.0 (Feb 12) and +16.45 on day
   306.8 (Nov 3).**
4. **Distance**: perihelion 0.98329 AU in the first days of January, aphelion 1.01671 AU in the first
   days of July. **Measured 0.98329 (Jan 3) and 1.01671 (Jul 4-5)**, asserted to 1e-4 AU.
5. **Mean local solar time is NRLMSIS's `UT + lon / 15`**: with the IAU 1982 GMST (written out here as
   an independent oracle, Aoki et al. 1982 as recalled), `12 h + (alpha - L) / 15` equals
   `UT + (alpha - GMST) / 15` to 0.001 deg - `(GMST - L) - (15 UT - 180)` is 0.0006 deg plus 3.4e-8 deg
   per day, plus 3.9e-4 T^2 - at most 0.00062 deg, **0.149 s**, over 2000-2050. Asserted under 0.16 s
   on random instants 2000-2050 (**measured 0.146 s**). The apparent-Sun version misses it
   by the equation of time, which the same test shows (a mutation guard).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbital_engine import solar_ephemeris as se

LONGITUDE_TOL_DEG = 0.01
EVENTS_2024 = [("2024-03-20T03:06", 0.0), ("2024-06-20T20:51", 90.0),
               ("2024-09-22T12:44", 180.0), ("2024-12-21T09:21", 270.0)]


def gmst_iau1982_deg(days: np.ndarray) -> np.ndarray:
    """Independent oracle: IAU 1982 GMST (degrees) at UT days from 2000-01-01T12:00."""
    t = days / 36525.0
    return np.mod(280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t ** 3 / 38710000.0,
                  360.0)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def test_epoch_days_counts_from_noon_ut_on_2000_01_01() -> None:
    assert se.epoch_days("2000-01-01T12:00") == 0.0
    assert se.epoch_days("2000-01-01T00:00") == -0.5
    assert se.epoch_days(np.datetime64("2024-03-20T00:00")) == 8844.5


@pytest.mark.parametrize("when, longitude", EVENTS_2024, ids=["mar", "jun", "sep", "dec"])
def test_equinoxes_and_solstices_of_2024(when: str, longitude: float) -> None:
    sun = se.solar_position(se.epoch_days(when))
    lam = math.degrees(float(sun.ecliptic_longitude_rad))
    dec = math.degrees(float(sun.declination_rad))
    assert abs(_wrap(lam - longitude)) < LONGITUDE_TOL_DEG, lam
    if longitude in (0.0, 180.0):
        assert abs(dec) < 0.01, dec
    else:
        assert abs(abs(dec) - 23.436) < 0.01 and math.copysign(1.0, dec) == (1.0 if longitude == 90.0 else -1.0)
    assert math.degrees(float(sun.obliquity_rad)) == pytest.approx(23.436, abs=0.01)


def test_right_ascension_advances_360_deg_per_tropical_year() -> None:
    start = se.epoch_days("2024-01-01T00:00")
    a0 = se.solar_position(start).right_ascension_rad
    a1 = se.solar_position(start + 365.2422).right_ascension_rad
    assert abs(_wrap(math.degrees(float(a1 - a0)))) < LONGITUDE_TOL_DEG
    days = start + np.arange(0.0, 366.0, 0.25)
    rate = np.degrees(np.diff(np.unwrap(se.solar_position(days).right_ascension_rad))) / 0.25
    assert 0.88 < rate.min() < 0.92 and 1.09 < rate.max() < 1.13, (rate.min(), rate.max())
    assert float(np.mean(rate)) == pytest.approx(360.0 / 365.2422, rel=2e-3)


def test_equation_of_time_extremes() -> None:
    start = se.epoch_days("2024-01-01T00:00")
    days = start + np.arange(0.0, 366.0, 0.05)
    eot = se.equation_of_time_minutes(days)
    assert eot.min() == pytest.approx(-14.2, abs=0.1)
    assert eot.max() == pytest.approx(16.4, abs=0.1)
    assert 40.0 < days[eot.argmin()] - start < 44.0           # Feb 10-14 (day 0 = Jan 1)
    assert 305.0 < days[eot.argmax()] - start < 309.0         # Nov 1-5


def test_perihelion_and_aphelion() -> None:
    start = se.epoch_days("2024-01-01T00:00")
    days = start + np.arange(0.0, 366.0, 0.25)
    dist = se.solar_position(days).distance_au
    assert dist.min() == pytest.approx(0.98329, abs=1e-4) and days[dist.argmin()] - start < 6.0
    assert dist.max() == pytest.approx(1.01671, abs=1e-4) and 181.0 < days[dist.argmax()] - start < 190.0


def test_mean_local_solar_time_is_ut_plus_longitude_over_15() -> None:
    rng = np.random.default_rng(11)
    days = rng.uniform(0.0, 18262.0, 200)
    alpha = rng.uniform(0.0, 2.0 * math.pi, 200)
    lat = rng.uniform(-1.4, 1.4, 200)
    r = 7000.0 * np.stack([np.cos(lat) * np.cos(alpha), np.cos(lat) * np.sin(alpha), np.sin(lat)], 1)
    lst = se.local_solar_time_hours(r, days)
    ut_hours = np.mod(days + 0.5, 1.0) * 24.0
    oracle = np.mod(ut_hours + (np.degrees(alpha) - gmst_iau1982_deg(days)) / 15.0, 24.0)
    diff_s = (np.mod(lst - oracle + 12.0, 24.0) - 12.0) * 3600.0
    assert np.max(np.abs(diff_s)) < 0.16, np.max(np.abs(diff_s))
    # The apparent-Sun definition misses MSIS's local time by the equation of time - up to 16 min.
    apparent = np.mod(12.0 + np.degrees(alpha - se.solar_position(days).right_ascension_rad) / 15.0, 24.0)
    miss_min = np.abs(np.mod(apparent - oracle + 12.0, 24.0) - 12.0) * 60.0
    assert miss_min.max() > 10.0
