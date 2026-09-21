"""
SGP4 at the boundary: two-line element sets in, **Cartesian TEME states** out.

**Reference.** SGP4/SDP4 is not implemented here - `CLAUDE.md`'s "Do not reimplement" rule. The
`sgp4` package (Brandon Rhodes, `sgp4` >= 2.x, `pip install orbital_engine[sgp4]`) wraps Vallado's
C++ implementation, which is the one defined in:

- Hoots, F. R. and Roehrich, R. L., *Spacetrack Report No. 3: Models for Propagation of NORAD
  Element Sets*, 1980 - the original theory.
- Vallado, D. A., Crawford, P., Hujsak, R. and Kelso, T. S., *Revisiting Spacetrack Report #3*,
  AIAA 2006-6753 - the reconciled code, its `SGP4-VER.TLE` verification cases and their expected
  output `tcppver.out`. **Both files ship inside the installed `sgp4` package** (next to its
  `__init__.py`), and `tests/validation/test_sgp4_bridge.py` reproduces them through this module -
  the first test in this repository against published numbers rather than a derivation. The
  gravity model is WGS-72 (`sgp4.api.WGS72`, the package default and the one `tcppver.out` was
  generated with), and the operation mode is AFSPC-improved (`'i'`), also the default.

**What this module guarantees.** A TLE's elements are *mean* elements in SGP4's own theory - Kozai
mean motion, Brouwer-style mean `e`, `i`, angles, and a drag term `B*` that is a fitting parameter,
not a ballistic coefficient. They are **never** handed to `frames.coe_to_rv` or stored in any engine
element column (`CLAUDE.md`: "Never convert externally defined mean elements <-> osculating
elements"). The only thing that crosses into the engine is SGP4's own output: position and velocity,
km and km/s. `seed_elements` then turns that *Cartesian* state into the engine's *osculating*
elements with `frames.rv_to_coe`, which is an exact change of coordinates on one state - not a
mean-to-osculating conversion - and is what `scenarios.tle_satellites` stores.

**Stateful objects stay out of the step.** `sgp4.api.Satrec` / `SatrecArray` are stateful C++
objects. They are built here, at ingest, and evaluated here; nothing in `Simulation.step` ever holds
or calls one. That is why SGP4 is a sweep tier of kind `sweep.ExternalTier` (a trajectory the sweep
compares against truth) and not a `PropagatorType` driven by `step()` - see `docs/architecture.md`,
"SGP4: an external tier, not a propagator". `PropagatorType.SGP4` exists in `custom_types.py` and
remains unimplemented on purpose.

**Frame: TEME, treated as the engine's inertial frame.** SGP4 outputs True Equator, Mean Equinox
(TEME) coordinates *of date* (Vallado et al. 2006, Sec. "Coordinate frames"). The engine's
Earth-centred scenarios use one fixed inertial frame whose +z is the spin axis (`geopotential.py`,
`drag.py` and `reference.py` all assume that). This module identifies the two at the scenario epoch
and does not rotate anything. What that ignores is how far TEME-of-date turns away from
TEME-of-epoch over the horizon, which is precession plus nutation of the pole and equinox:

- general precession in longitude, 50.3 arcsec/yr = 0.138 arcsec/day;
- the fastest nutation rates: 18.6-yr term 17.2 arcsec amplitude -> 0.016 arcsec/day, semi-annual
  1.32 arcsec -> 0.045 arcsec/day, fortnightly 0.23 arcsec -> 0.11 arcsec/day (IAU 1980 series,
  amplitudes from memory, **unverified against the text**; pyerfa is not a dependency here).

Their sum bounds the rotation at `TEME_DRIFT_RAD_PER_DAY` = 0.31 arcsec/day = 1.5e-6 rad/day,
i.e. **about 11 m per day at LEO radius** (6800 km) - three orders below SGP4's own ~1 km-scale
error against a precise orbit, and two below what it disagrees with a J2 truth by over a day (see
the architecture section). It stops being acceptable when (a) the horizon reaches months (precession
alone is 0.33 km at LEO per 100 days, and 4 km at GEO), (b) SGP4 output is compared with an
externally referenced trajectory (GCRF / J2000 ephemerides, ground truth from GPS) - then the full
TEME -> GCRF rotation is needed, and it belongs to `pyerfa`, not here - or (c) ground stations are
placed with sub-10 m accuracy, where polar motion (~10 m) also enters. Earth rotation for station
geometry is `Satrec.gsto` (GMST at the TLE epoch, IAU-82), exposed as `greenwich_angle`.

**Units.** SGP4 works in minutes since TLE epoch; this module's public API is **seconds since a
chosen epoch** (the engine's unit) and converts once, in `_julian_offsets`. `sgp4_tsince` exists only
for the published verification cases, whose times are in minutes since each TLE's own epoch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, ArraySeconds, ScalarGravitationalParameter
from .frames import ReferenceFrames

if TYPE_CHECKING:
    from .sweep import ExternalTier

__all__ = [
    "TLE", "ISS_TLE", "SGP4States", "JulianDate",
    "SECONDS_PER_DAY", "MINUTES_PER_DAY", "TEME_DRIFT_RAD_PER_DAY",
    "tle_epoch", "greenwich_angle", "sgp4_states", "sgp4_tsince", "teme_seed", "seed_elements",
    "sgp4_tier",
]

SECONDS_PER_DAY = 86400.0
MINUTES_PER_DAY = 1440.0

# Bound on the rotation of TEME-of-date away from TEME-of-epoch, rad/day - the sum of the rates in
# the module docstring (0.138 + 0.016 + 0.045 + 0.11 arcsec/day, rounded up to 0.31).
TEME_DRIFT_RAD_PER_DAY = 0.31 / 206264.806

# A Julian date split as (whole-ish part, fraction), the way `sgp4` carries it (`jdsatepoch`,
# `jdsatepochF`) so that sub-millisecond resolution survives in float64.
JulianDate = Tuple[float, float]


@dataclass(frozen=True)
class TLE:
    """One two-line element set. `name` is what the scenario calls the body; it is not parsed."""
    name: str
    line1: str
    line2: str


# The ISS element set used in the `sgp4` package's own README (epoch 2019-12-09). Both line checksums
# are verified in tests/validation/test_sgp4_bridge.py, so a transcription slip here would fail.
ISS_TLE = TLE(
    name="ISS (ZARYA)",
    line1="1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991",
    line2="2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482",
)


@dataclass(frozen=True)
class SGP4States:
    """
    SGP4 output on a time grid.

    `states` is `(n_times, n_sats, 6)` `[x y z vx vy vz]` in km and km/s, TEME - the same layout as
    `viz.sample_states`, so it can be differenced against an engine run directly. `errors` is
    `(n_times, n_sats)` with SGP4's own error codes (`sgp4.api.SGP4_ERRORS`, 0 = success). A row with
    a non-zero code is **set to NaN here** - SGP4 does that itself for most codes but returns a finite
    state with code 6 (decayed), and a flagged state must not reach a comparison as a plausible
    number. SGP4's errors are per evaluation, not sticky: Vallado's case 33333 is flagged at 25 and
    30 min and valid again at 50.
    """
    states: ArrayFloat
    errors: NDArray[np.uint8]


def _satrec(tle: TLE) -> Any:
    from sgp4.api import Satrec  # imported lazily: sgp4 is an optional extra

    return Satrec.twoline2rv(tle.line1, tle.line2)


def _satrec_array(tles: Sequence[TLE]) -> Any:
    from sgp4.api import SatrecArray  # imported lazily: sgp4 is an optional extra

    if len(tles) == 0:
        raise ValueError("need at least one TLE")
    return SatrecArray([_satrec(t) for t in tles])


def tle_epoch(tle: TLE) -> JulianDate:
    """The TLE's epoch as a split Julian date (UTC), exactly as `sgp4` parsed it."""
    sat = _satrec(tle)
    return float(sat.jdsatepoch), float(sat.jdsatepochF)


def greenwich_angle(tle: TLE) -> float:
    """Greenwich mean sidereal angle at the TLE's epoch, rad (`Satrec.gsto`, IAU-82) - the angle
    between TEME's x axis and the prime meridian, for `geometry.py`'s `theta0`."""
    return float(_satrec(tle).gsto)


def _julian_offsets(epoch: JulianDate, times_s: ArraySeconds) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    `epoch + times_s` as the split Julian date `sgp4` takes. The seconds -> days conversion lives
    here and nowhere else.

    Whole days go onto `jd` and only the remainder onto the fraction. Putting the whole offset on the
    fraction costs resolution in proportion to the horizon: at `tsince` = 1.84e6 min (Vallado's
    last SL-12 case, 3.5 yr) the fraction is ~1280 d, its half-ulp is 1.1e-13 d = 9.8e-9 s, and at
    the case's 10.3 km/s that is 1e-7 km - measured 1.6e-7 km against `tcppver.out` before this
    split, 5e-9 km (the file's print rounding) after it.
    """
    t = np.asarray(times_s, dtype=np.float64)
    whole = np.floor(t / SECONDS_PER_DAY)
    jd: ArrayFloat = np.asarray(epoch[0] + whole, dtype=np.float64)
    fr: ArrayFloat = np.asarray(epoch[1] + (t - whole * SECONDS_PER_DAY) / SECONDS_PER_DAY, dtype=np.float64)
    return jd, fr


def _evaluate(sats: Any, jd: ArrayFloat, fr: ArrayFloat) -> SGP4States:
    e, r, v = sats.sgp4(jd, fr)  # (n_sats, n_times), (n_sats, n_times, 3) x2
    states: ArrayFloat = np.concatenate(
        [np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64)], axis=2,
    ).transpose(1, 0, 2).copy()
    errors: NDArray[np.uint8] = np.asarray(e, dtype=np.uint8).T.copy()
    # SGP4 itself returns NaN for most error codes but a *finite* state for code 6 ("decayed") -
    # measured on Vallado's cases 28872 and 29141. A flagged state is never passed on as a number.
    states[errors != 0] = np.nan
    return SGP4States(states=states, errors=errors)


def sgp4_states(tles: Sequence[TLE], epoch: JulianDate, times_s: ArraySeconds) -> SGP4States:
    """
    Every TLE evaluated at `epoch + times_s` (seconds), in one `SatrecArray` call.

    All satellites share one clock - the scenario's - so a satellite whose own TLE epoch differs from
    `epoch` is simply evaluated at a non-zero `tsince`, which is how SGP4 is meant to be used.
    """
    jd, fr = _julian_offsets(epoch, times_s)
    return _evaluate(_satrec_array(tles), jd, fr)


def sgp4_tsince(tle: TLE, tsince_min: ArrayFloat) -> SGP4States:
    """
    One TLE at `tsince_min` **minutes** from its *own* epoch - the convention of Vallado's
    `tcppver.out`, and so the verification path. Goes through the same `SatrecArray` evaluation as
    `sgp4_states`, so the published cases exercise the code the sweep runs.
    """
    epoch = tle_epoch(tle)
    seconds: ArrayFloat = np.asarray(
        np.asarray(tsince_min, dtype=np.float64) * (SECONDS_PER_DAY / MINUTES_PER_DAY), dtype=np.float64,
    )
    jd, fr = _julian_offsets(epoch, seconds)
    return _evaluate(_satrec_array([tle]), jd, fr)


def teme_seed(tles: Sequence[TLE], epoch: JulianDate) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    SGP4's Cartesian state for every TLE at `epoch`: `(r (n,3) km, v (n,3) km/s)`, TEME.

    This is the interchange format - the one quantity both SGP4 and the engine define identically.
    Raises `ValueError` naming the satellite if SGP4 reports an error at the epoch, rather than
    seeding the engine with NaN.
    """
    out = sgp4_states(tles, epoch, np.zeros(1, dtype=np.float64))
    bad = np.flatnonzero(out.errors[0] != 0)
    if bad.size:
        names = [tles[int(k)].name for k in bad]
        raise ValueError(f"SGP4 failed at the scenario epoch for {names} (codes {out.errors[0, bad]})")
    r: ArrayFloat = out.states[0, :, :3].copy()
    v: ArrayFloat = out.states[0, :, 3:].copy()
    return r, v


def seed_elements(
    tles: Sequence[TLE], epoch: JulianDate, mu: ScalarGravitationalParameter,
) -> ArrayFloat:
    """
    The engine's **osculating** `[p e i RAAN argp theta]` for each TLE, from SGP4's Cartesian state
    at `epoch` under the engine's own `mu` - `frames.rv_to_coe` on `teme_seed`'s output.

    No TLE element is read. The elements returned describe the osculating Kepler orbit through SGP4's
    `(r, v)`; `coe_to_rv` of them reproduces that `(r, v)` to round-off, which is the property
    `scenarios.tle_satellites` relies on and the tests assert.
    """
    r, v = teme_seed(tles, epoch)
    coe, ok = ReferenceFrames.rv_to_coe(r, v, mu)
    if not np.all(ok):
        raise ValueError(f"rv_to_coe failed for {[t.name for t, good in zip(tles, ok) if not good]}")
    out: ArrayFloat = np.asarray(coe, dtype=np.float64).reshape(len(tles), 6)
    return out


def sgp4_tier(
    tles: Sequence[TLE],
    epoch: JulianDate,
    *,
    dt: float,
    central_body: str = "Earth",
    name: str = "SGP4",
) -> "ExternalTier":
    """
    SGP4 as a `sweep.ExternalTier`: the bodies are the TLEs' names (as `scenarios.tle_satellites`
    seeds them), positions are TEME relative to `central_body`, the clock starts at `epoch`.

    The `SatrecArray` is built here, once, and captured - ingest, not step. `dt` is the cadence at
    which `run_sweep` asks for states when timing the tier, so its cost is comparable to an engine
    tier that delivers one state per step.
    """
    from .sweep import ExternalTier

    sats = _satrec_array(tles)

    def positions(times_s: ArraySeconds) -> ArrayFloat:
        jd, fr = _julian_offsets(epoch, times_s)
        result = _evaluate(sats, jd, fr)
        if np.any(result.errors != 0):
            raise ValueError(
                f"tier '{name}': SGP4 reported errors {sorted(set(result.errors.ravel().tolist()) - {0})} "
                f"inside the requested span - a decayed or diverged element set cannot be scored."
            )
        out: ArrayFloat = result.states[..., :3]
        return out

    names: List[str] = [t.name for t in tles]
    return ExternalTier(name=name, dt=dt, bodies=names, central_body=central_body, positions=positions)
