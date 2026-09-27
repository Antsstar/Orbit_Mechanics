"""
The Artemis II replay data set: NASA/JSC's Orion trajectory and DE441 Moon and Sun from JPL Horizons,
the mission's event list, and the burns identified **from the trajectory itself**.

This module reads only the committed files under `data/artemis2/` (via `horizons_bridge.load_cached`)
and never the network. It is analysis at the boundary - nothing here runs inside `Simulation.step`.
`scenarios.artemis2` seeds an engine run from the same data.

The data (see `data/artemis2/README.md` for the provenance record)
------------------------------------------------------------------
- Orion: Horizons object -1024, "Post-launch Orion II trajectory data from NASA/JSC navigation
  (concatenated)" - fourteen OEM/OD files joined end to end, listed in the object-data header. Earth
  centred, ICRF, **1 min** from 2026-04-02 02:00 to 2026-04-10 23:54 TDB (the file covers
  01:58:32.3 to 23:54:22.9 TDB).
- Moon (301) and Sun (10): DE441, Earth centred, ICRF, **10 min**, covering Orion's span. At 10 min
  a cubic Hermite interpolant of the Moon's geocentric position is good to ~6e-9 km (fourth-order
  error `h^4 |r''''| / 384` with `|r''''| ~ n^4 r`, `n` the lunar mean motion), so nothing is lost
  by not storing them at Orion's cadence; `tests/validation/test_artemis2.py` checks it by
  decimation (1.0e-7 km derived at 20 min, 9.1e-8 measured).
- Why 1 min for Orion, and not finer: Horizons' interpolant of the JSC data **rings** across a burn
  (below), so finer sampling adds no information about a burn - the Delta-v is an integral between
  clean samples, which 1 min resolves (TLI: 388.6 m/s against NASA's 388.3). At the flyby the 1-min
  minimum distance is within `r'' (h/2)^2 / 2` = 0.071 km of the true one before refinement, and
  the Hermite refinement uses Orion's own velocities. The whole data set is ~0.8 MB.

`EPOCH_TDB` = 2026-04-02T02:00:00 TDB is `t_s = 0` for all three tables.

Burn identification
-------------------
Each 1-min interval of Orion's trajectory is re-predicted from its own start state by a local coast
model - Earth point mass + J2 (EGM96, about ICRF +z unless a pole is given), Moon and Sun point
masses from the tables (direct minus indirect), RK4 at `DETECTION_SUBSTEPS` sub-steps - and the
velocity residual at the interval's end is the unexplained Delta-v. Intervals above
`DETECTION_THRESHOLD_KM_S` are merged into clusters (gaps of up to `DETECTION_MERGE_GAP` quiet
intervals are bridged, because the data **ring** around a burn - see below), padded by
`DETECTION_PAD` clean samples, and each cluster's impulsive equivalent is found by coasting forward
from the clean state before it and backward from the clean state after it: the epoch is where the two
coasts pass closest (`closure_km`), and the Delta-v is the velocity jump between them there. A true
impulse gives zero closure, and so does a straight, constant-thrust finite burn to first order: the
two coasts then meet at its midpoint (their separation is `|dv| |t - t_mid|`). What is left is the
gravity gradient, steering and mass loss over the burn (TLI: 3.0 km, i.e. `closure_s` = 7.9 s) - or a
**position** discontinuity in the data, which no impulse closes. `closure_s = closure_km / |dv|`
separates the two: burns measure 0.0-109 s, discontinuities >= 1265 s; `IMPULSIVE_CLOSURE_S` = 300 s.
`delivered_dv` is the vector sum of the per-interval residuals - the burn's integral of thrust
acceleration, what a mission report calls its Delta-v. The two differ by the finite-burn loss.

**The data ring.** Horizons interpolates the JSC ephemeris between its nodes; across a burn that the
source does not break into segments, the interpolant oscillates with amplitudes of several m/s^2
for minutes either side (measured at 5 s around TLI: +-6 m/s^2 from 23:45:40 TDB, while the
engine itself burns at ~1.07 m/s^2 from 23:50:40 to 23:56:30). The oscillation integrates to
almost nothing between clean samples, which is why clusters are bridged and scored end to end, never
interval by interval.

**File joins.** Where one of the fourteen files ends and the next begins the two OD solutions need
not agree; the join appears as a residual like a burn. Every detection records whether a join (from
the header's table, to its printed minute) falls inside its cluster.

Time and frame
--------------
All epochs here are **TDB**; `horizons_bridge.utc_to_tdb` converts the event list's UTC
(TT - UTC = 69.184 s, TDB - TT = +1.66 ms at the replay). The frame is ICRF. Its +z is **not** the
Earth's spin axis: `earth_orientation()` recovers the true pole (0.1469 deg from +z, towards RA
0.8 deg) and the prime meridian from Horizons' own Earth-orientation model (three Earth-fixed sites as
ICRF vectors, `earth_sites.npz`), `mean_pole_of_date()` the mean pole (0.1462 deg, IAU 1976's
`theta_A` to 0.26"), and `pole_misalignment_cost()` states what using +z costs: 0.70 km over the
15 h coast after TLI, metres elsewhere. `gmst_iau1982_rad` less `precession_in_ra_rad` gives the
engine's `theta0` (ICRF right ascension of the prime meridian) to 1.1" of Horizons' value.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import horizons_bridge as hb
from .custom_types import ArrayFloat
from .geopotential import EARTH_J2, EARTH_R_EQ

__all__ = [
    "EPOCH_TDB", "LAUNCH_UTC", "FRAME_CHECK_BODIES", "DATA_DIR",
    "MU_EARTH_DE440", "MU_MOON_DE440", "MU_SUN_DE440", "MOON_MEAN_RADIUS_KM", "EARTH_MEAN_RADIUS_KM",
    "LUNAR_SOI_KM", "PUBLISHED", "PublishedFigure",
    "DETECTION_THRESHOLD_KM_S", "DETECTION_SUBSTEPS", "DETECTION_MERGE_GAP", "DETECTION_PAD",
    "EVENT_MATCH_TOLERANCE_S", "EVENT_MATCH_WINDOW_S", "IMPULSE_STEP_S", "IMPULSIVE_CLOSURE_S",
    "DEFAULT_REPLAY_EPOCH_TDB", "EARTH_SITE_COORDS", "ListedBurn", "listed_burns",
    "MissionEvent", "Burn", "Extremum", "EarthOrientation", "ArcCost", "TrajectoryFile",
    "load", "tdb_seconds", "tdb_instant", "hermite_position",
    "parse_major_events", "load_events", "write_events_csv", "trajectory_files",
    "CoastModel", "coast", "interval_residuals", "detect_burns", "write_burns_csv", "match_events",
    "closest_lunar_approach", "maximum_earth_distance",
    "earth_orientation", "mean_pole_of_date", "precession_pole_tilt_rad", "precession_in_ra_rad",
    "gmst_iau1982_rad", "greenwich_apparent_sidereal_hours", "pole_misalignment_cost",
]

DATA_DIR: Final[Path] = hb.DATA_DIR / "artemis2"

#: `t_s = 0` for every committed table, a TDB instant.
EPOCH_TDB: Final[str] = "2026-04-02T02:00:00"
#: Where `scenarios.artemis2` seeds Orion by default: the first clean coast after TLI (TDB).
DEFAULT_REPLAY_EPOCH_TDB: Final[str] = "2026-04-03T01:00:00"
#: Launch, UTC, as the object-data header and NASA state it (22:35:12 UTC, 2026-04-01).
LAUNCH_UTC: Final[str] = "2026-04-01T22:35:12"
#: Horizons IDs whose ICRF and ITRF93 positions make up `frame_check.npz`.
FRAME_CHECK_BODIES: Final[Tuple[str, ...]] = ("301", "10", "599")
#: Horizons geodetic `SITE_COORD`s (E-lon deg, lat deg, alt km) of `earth_sites.npz`: the ITRF93 x
#: axis, y axis and north pole on the WGS-84 ellipsoid.
EARTH_SITE_COORDS: Final[Tuple[str, ...]] = ("0,0,0", "90,0,0", "0,90,0")

# DE440 gravitational parameters, km^3/s^2 (Park et al. 2021, AJ 161:105, Table 8 - from memory,
# unverified against the text). The coast model's result is insensitive to them at this level:
# a 1e-8 relative change in mu moves a 60 s prediction by < 1e-10 km/s.
MU_EARTH_DE440: Final[float] = 398600.435507
MU_MOON_DE440: Final[float] = 4902.800118
MU_SUN_DE440: Final[float] = 132712440041.279419

#: IAU mean lunar radius, km - the radius Horizons' own header uses for the Moon (R_eq = 1737.4).
MOON_MEAN_RADIUS_KM: Final[float] = 1737.4
#: IUGG mean Earth radius, km (the engine's `scenarios.EARTH_RADIUS`).
EARTH_MEAN_RADIUS_KM: Final[float] = 6371.0
#: The object-data header's lunar "SOI" (Hill-sphere definition), km from the Moon's centre.
LUNAR_SOI_KM: Final[float] = 62800.0

FT_S: Final[float] = 0.3048  # m per ft (exact)


@dataclass(frozen=True)
class PublishedFigure:
    """A number NASA (or Horizons) published, with the exact source it came from."""
    value: float
    unit: str
    what: str
    source: str


#: Every published number this module compares against. Nothing here is recalled; each was read
#: from the cited page on 2026-09-27. The Horizons header entries are in `orion_object_data.txt`.
PUBLISHED: Final[Dict[str, PublishedFigure]] = {
    "closest_approach_altitude_km": PublishedFigure(
        6545.0, "km", "closest approach, above the lunar surface (4,067 mi), ~7 p.m. EDT 2026-04-06",
        "https://www.nasa.gov/blogs/missions/2026/04/06/artemis-ii-flight-day-6-lunar-flyby-updates"),
    "closest_approach_center_km": PublishedFigure(
        8282.0, "km", "closest approach to Moon centre, 2026-04-06 23:01 UTC",
        "Horizons -1024 object data, MAJOR EVENTS (orion_object_data.txt)"),
    "max_distance_nasa_mi": PublishedFigure(
        252756.0, "mi", "maximum distance from Earth (406,771 km), 7:02 p.m. EDT 2026-04-06",
        "https://www.nasa.gov/blogs/missions/2026/04/06/artemis-ii-flight-day-6-lunar-flyby-updates"),
    "max_distance_center_km": PublishedFigure(
        413146.2, "km", "maximum distance from Earth centre, 2026-04-06 23:05 UTC",
        "Horizons -1024 object data, MAJOR EVENTS (orion_object_data.txt)"),
    "tli_dv_ft_s": PublishedFigure(
        1274.0, "ft/s", "TLI burn, 5 min 49 s (planned) / 5 min 50 s (flown) from 7:49 p.m. EDT 2026-04-02",
        "https://www.nasa.gov/blogs/missions/2026/04/02/artemis-ii-flight-update-perigee-raise-burn-complete/"),
    "prb_duration_s": PublishedFigure(
        43.0, "s", "perigee raise burn on the service-module main engine, 2026-04-02 (no Delta-v stated)",
        "https://www.nasa.gov/blogs/missions/2026/04/02/artemis-ii-flight-update-perigee-raise-burn-complete/"),
    "otc_duration_s": PublishedFigure(
        17.5, "s", "outbound correction burn from 11:03 p.m. EDT 2026-04-05 (no Delta-v stated)",
        "https://www.nasa.gov/blogs/missions/2026/04/05/artemis-ii-flight-day-5-correction-burn-complete/"),
    "rtc1_dv_ft_s": PublishedFigure(
        1.6, "ft/s", "first return correction burn, 15 s from 8:03 p.m. EDT 2026-04-07",
        "https://www.nasa.gov/blogs/missions/2026/04/07/artemis-ii-flight-day-7-first-return-correction-burn-complete/"),
    "rtc2_dv_ft_s": PublishedFigure(
        5.3, "ft/s", "second return correction burn, 9 s from 10:53 p.m. EDT 2026-04-09",
        "https://www.nasa.gov/blogs/missions/2026/04/10/artemis-ii-flight-day-9-second-return-correction-burn-complete/"),
    "rtc3_dv_ft_s": PublishedFigure(
        4.2, "ft/s", "third return correction burn, 8 s from 2:53 p.m. EDT 2026-04-10",
        "https://www.nasa.gov/blogs/missions/2026/04/10/artemis-ii-flight-day-10-crew-completes-final-burn-before-splashdown/"),
}

# Burn detection parameters. The threshold sits well above the coast model's measured floor on
# quiet intervals and well below the smallest listed burn (0.49 m/s); see `detect_burns`.
DETECTION_THRESHOLD_KM_S: Final[float] = 3.0e-5
DETECTION_SUBSTEPS: Final[int] = 6
DETECTION_MERGE_GAP: Final[int] = 1
DETECTION_PAD: Final[int] = 2
#: A detected impulsive epoch matches a listed burn if it lies within this of the listed start plus
#: half the listed duration. The list's times are printed to the minute (+-30 s), the data are
#: sampled at 60 s, and planned and flown ignition times differ by tens of seconds (TLI: NASA gives
#: 7:49 p.m. EDT, the header 23:49 UTC, the data's thrust onset is ~23:49:30 UTC).
EVENT_MATCH_TOLERANCE_S: Final[float] = 180.0
#: How far from its listed time a burn is searched for (the list's times may be planning values).
EVENT_MATCH_WINDOW_S: Final[float] = 3600.0
#: RK4 step of the forward/backward coasts that locate an impulse; divides the 60 s sample spacing.
IMPULSE_STEP_S: Final[float] = 5.0
#: A detection whose closure time `closure_km / |dv|` is below this is a burn; above it, a position
#: discontinuity in the data (a file join, or an interpolation artefact). See `detect_burns`.
IMPULSIVE_CLOSURE_S: Final[float] = 300.0


# --------------------------------------------------------------------------------------------------
# Loading and time
# --------------------------------------------------------------------------------------------------

def load(name: str) -> hb.HorizonsVectors:
    """One committed table: `"orion"`, `"moon"` or `"sun"`."""
    return hb.load_cached(name, "artemis2")


def tdb_seconds(when_tdb: Union[str, np.datetime64]) -> float:
    """Seconds from `EPOCH_TDB` to the TDB instant `when_tdb`."""
    ns = (hb.as_ns(when_tdb) - hb.as_ns(EPOCH_TDB)).astype(np.int64)
    return float(ns) / 1e9


def tdb_instant(t_s: float) -> np.datetime64:
    """The TDB instant `t_s` seconds after `EPOCH_TDB` (to the nanosecond)."""
    return hb.as_ns(EPOCH_TDB) + np.timedelta64(int(round(t_s * 1e9)), "ns")


def _utc_of_tdb(t_s: float) -> np.datetime64:
    """UTC instant of TDB `t_s` (inverts `utc_to_tdb` by one fixed-point pass; the offset changes by
    < 1 ns over the difference)."""
    tdb = tdb_instant(t_s)
    guess = tdb - np.timedelta64(int(round(hb.tdb_minus_utc_s(tdb) * 1e9)), "ns")
    return tdb - np.timedelta64(int(round(hb.tdb_minus_utc_s(guess) * 1e9)), "ns")


def hermite_position(table: hb.HorizonsVectors, t_s: Union[float, ArrayFloat]) -> ArrayFloat:
    """
    Cubic Hermite interpolation of `table`'s position at `t_s` (seconds from its epoch), from the
    bracketing samples' positions **and velocities**. Shape `(n, 3)` for `(n,)` input, `(3,)` for a
    scalar. Raises `ValueError` outside the table - an ephemeris is never extrapolated here.
    """
    t = np.atleast_1d(np.asarray(t_s, dtype=np.float64))
    knots = table.t_s
    if np.any(t < knots[0] - 1e-6) or np.any(t > knots[-1] + 1e-6):
        raise ValueError(f"{table.target}: time outside the table [{knots[0]}, {knots[-1]}] s")
    k = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, knots.size - 2)
    h = (knots[k + 1] - knots[k])[:, None]
    s = (t - knots[k])[:, None] / h
    s2, s3 = s * s, s * s * s
    out: ArrayFloat = (
        (2.0 * s3 - 3.0 * s2 + 1.0) * table.position_km[k]
        + (s3 - 2.0 * s2 + s) * h * table.velocity_km_s[k]
        + (-2.0 * s3 + 3.0 * s2) * table.position_km[k + 1]
        + (s3 - s2) * h * table.velocity_km_s[k + 1]
    )
    if np.ndim(t_s) == 0:
        single: ArrayFloat = out[0]
        return single
    return out


# --------------------------------------------------------------------------------------------------
# The event list
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionEvent:
    """
    One line of the Horizons MAJOR EVENTS list. `met` is the list's own "days/HH:MM[:SS]" string,
    `utc` the list's UTC (to the minute unless it prints seconds), `tdb` that instant on TDB.
    `delta_v_m_s` and `duration_s` are what the line states (`nan` if nothing), `cancelled` marks
    "[CANCELLED]", `in_span` whether `tdb` lies inside Orion's committed table.
    """
    name: str
    met: str
    utc: np.datetime64
    tdb: np.datetime64
    delta_v_m_s: float
    duration_s: float
    cancelled: bool
    in_span: bool

    @property
    def is_burn(self) -> bool:
        """A propulsive event of Orion's own (not the ICPS's, not a separation)."""
        n = self.name.lower()
        return ("burn" in n or "maneuver" in n or n.startswith("start translunar")) and "icps" not in n \
            and "disposal" not in n


_EVENT_RE: Final = re.compile(
    r"^\s*(?:launch\s*\+?)?\s*(\d+)/(\d{2}):(\d{2})(?::(\d{2}))?\s+(\d{1,2})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s*-?\s*(.*)$"
)


def parse_major_events(text: str, *, year: int = 2026) -> List[MissionEvent]:
    """
    The MAJOR EVENTS section of Horizons' -1024 object data as `MissionEvent`s, in list order.

    Continuation lines (no MET column) are appended to the event above them. The month comes from
    the column header ("MET     Apr  UTC"). Delta-v is read from "delta-v= 388 m/s" / "delta-v 3 m/s"
    / "Total delta-v = 0.49 m/s", duration from "length 15 sec", "length 18 sec" or "(5m 55s)".
    """
    if "MAJOR EVENTS" not in text:
        raise ValueError("no MAJOR EVENTS section")
    section = text.split("MAJOR EVENTS", 1)[1].split("SPACECRAFT DETAILS", 1)[0]
    month_m = re.search(r"MET\s+([A-Z][a-z]{2})\s+UTC", section)
    if month_m is None:
        raise ValueError("MAJOR EVENTS has no 'MET <Mon> UTC' column header")
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(
        month_m.group(1)) + 1
    orion = load("orion")
    span = (orion.epoch + np.timedelta64(int(orion.t_s[0] * 1e9), "ns"),
            orion.epoch + np.timedelta64(int(orion.t_s[-1] * 1e9), "ns"))

    raw: List[Tuple[str, str, str]] = []  # (met, utc_iso, text)
    lines = section.splitlines()
    first = next(k for k, ln in enumerate(lines) if month_m.group(0) in ln) + 1
    for line in lines[first:]:
        m = _EVENT_RE.match(line)
        if m is not None:
            d, hh, mm, ss, day, uh, um, us, desc = m.groups()
            met = f"{d}/{hh}:{mm}" + (f":{ss}" if ss else "")
            utc = f"{year}-{month:02d}-{int(day):02d}T{uh}:{um}:{us or '00'}"
            raw.append((met, utc, desc.strip()))
        elif raw and line.strip() and not re.match(r"^\s*Day \d+", line):
            met, utc, desc = raw[-1]
            raw[-1] = (met, utc, f"{desc} {line.strip()}")

    events: List[MissionEvent] = []
    for met, utc, desc in raw:
        dv = re.search(r"delta-v\s*=?\s*([\d.]+)\s*m/s", desc, flags=re.IGNORECASE)
        dur = re.search(r"length\s+(\d+(?:\.\d+)?)\s*sec", desc, flags=re.IGNORECASE)
        dur_ms = re.search(r"\((\d+)m\s*(\d+)s\)", desc)
        duration = float(dur.group(1)) if dur else (
            60.0 * float(dur_ms.group(1)) + float(dur_ms.group(2)) if dur_ms else math.nan)
        when_utc = hb.as_ns(utc)
        when_tdb = hb.utc_to_tdb(when_utc)
        events.append(MissionEvent(
            name=re.sub(r"\s+", " ", desc), met=met, utc=when_utc, tdb=when_tdb,
            delta_v_m_s=float(dv.group(1)) if dv else math.nan, duration_s=duration,
            cancelled="[CANCELLED]" in desc.upper(), in_span=bool(span[0] <= when_tdb <= span[1]),
        ))
    return events


def load_events() -> List[MissionEvent]:
    """`parse_major_events` of the committed object-data header."""
    return parse_major_events((DATA_DIR / "orion_object_data.txt").read_text(encoding="utf-8"))


def _iso(when: np.datetime64, unit: Literal["s", "ms", "ns"] = "ms") -> str:
    return str(np.datetime_as_string(when, unit=unit))


def write_events_csv(events: Sequence[MissionEvent], path: Union[str, Path]) -> None:
    """`events.csv`: name, MET, UTC, TDB, stated delta-v and duration, flags."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["name", "met", "utc", "tdb", "delta_v_m_s", "duration_s", "cancelled", "in_span", "is_burn"])
        for e in events:
            w.writerow([e.name, e.met, _iso(e.utc, "s"), _iso(e.tdb), "" if math.isnan(e.delta_v_m_s) else e.delta_v_m_s,
                        "" if math.isnan(e.duration_s) else e.duration_s, int(e.cancelled), int(e.in_span),
                        int(e.is_burn)])


@dataclass(frozen=True)
class TrajectoryFile:
    """One of the concatenated source files in the object-data header, start/stop in TDB (to the
    minute, as printed)."""
    name: str
    start_tdb: np.datetime64
    stop_tdb: np.datetime64


def trajectory_files() -> List[TrajectoryFile]:
    """The header's "Trajectory name / Start (TDB) / Stop (TDB)" table."""
    text = (DATA_DIR / "orion_object_data.txt").read_text(encoding="utf-8")
    out: List[TrajectoryFile] = []
    for m in re.finditer(r"^\s*(\S+)\s+(\d{4}-[A-Z][a-z]{2}-\d{2} \d{2}:\d{2})\s+(\d{4}-[A-Z][a-z]{2}-\d{2} \d{2}:\d{2})\s*$",
                         text, flags=re.MULTILINE):
        start = hb.parse_calendar_tdb("A.D. " + m.group(2) + ":00")
        stop = hb.parse_calendar_tdb("A.D. " + m.group(3) + ":00")
        out.append(TrajectoryFile(m.group(1), start, stop))
    return out


# --------------------------------------------------------------------------------------------------
# The coast model
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CoastModel:
    """
    The local prediction model burns are measured against: Earth point mass + J2 about `pole`
    (unit vector, ICRF; default +z), Moon and Sun point masses from the committed tables, direct
    minus indirect. Accelerations in km/s^2 of an Earth-centred ICRF position `r` at `t_s`.
    """
    moon: hb.HorizonsVectors
    sun: hb.HorizonsVectors
    pole: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    j2: float = EARTH_J2
    r_eq: float = EARTH_R_EQ
    mu_earth: float = MU_EARTH_DE440
    mu_moon: float = MU_MOON_DE440
    mu_sun: float = MU_SUN_DE440

    def acceleration(self, t_s: ArrayFloat, r: ArrayFloat) -> ArrayFloat:
        rn = np.linalg.norm(r, axis=1)[:, None]
        acc: ArrayFloat = -self.mu_earth * r / rn**3
        if self.j2 != 0.0:
            p = np.asarray(self.pole, dtype=np.float64)
            rp = (r @ p)[:, None]
            k = -1.5 * self.j2 * self.mu_earth * self.r_eq**2 / rn**5
            acc = acc + k * ((1.0 - 5.0 * rp**2 / rn**2) * r + 2.0 * rp * p)
        for table, mu in ((self.moon, self.mu_moon), (self.sun, self.mu_sun)):
            rb = hermite_position(table, t_s)
            d = rb - r
            acc = acc + mu * (d / np.linalg.norm(d, axis=1)[:, None] ** 3
                              - rb / np.linalg.norm(rb, axis=1)[:, None] ** 3)
        return acc


def coast(model: CoastModel, t0: ArrayFloat, state0: ArrayFloat, t1: ArrayFloat, n_steps: int) -> ArrayFloat:
    """
    RK4 coast of each row of `state0` `(n, 6)` from `t0` to `t1` (both `(n,)`, seconds from
    `EPOCH_TDB`; `t1 < t0` integrates backward) in `n_steps` equal steps. Returns `(n, 6)`.
    """
    t = np.asarray(t0, dtype=np.float64).copy()
    h = (np.asarray(t1, dtype=np.float64) - t) / n_steps
    y = np.asarray(state0, dtype=np.float64).copy()
    hc = h[:, None]

    def f(tt: ArrayFloat, yy: ArrayFloat) -> ArrayFloat:
        out: ArrayFloat = np.concatenate([yy[:, 3:], model.acceleration(tt, yy[:, :3])], axis=1)
        return out

    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * hc * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * hc * k2)
        k4 = f(t + h, y + hc * k3)
        y = y + hc / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
    return y


def _default_model(pole: Optional[Sequence[float]] = None) -> CoastModel:
    return CoastModel(moon=load("moon"), sun=load("sun"),
                      pole=tuple(float(x) for x in pole) if pole is not None else (0.0, 0.0, 1.0))  # type: ignore[arg-type]


def interval_residuals(
    orion: Optional[hb.HorizonsVectors] = None,
    model: Optional[CoastModel] = None,
    *,
    substeps: int = DETECTION_SUBSTEPS,
) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    `(dr, dv)`, each `(n-1, 3)`: observed minus predicted state at the end of every sample interval,
    the prediction a coast from the observed state at the interval's start. Vectorised over all
    intervals at once.
    """
    o = orion if orion is not None else load("orion")
    m = model if model is not None else _default_model()
    pred = coast(m, o.t_s[:-1], o.state[:-1], o.t_s[1:], substeps)
    obs = o.state[1:]
    dr: ArrayFloat = obs[:, :3] - pred[:, :3]
    dv: ArrayFloat = obs[:, 3:] - pred[:, 3:]
    return dr, dv


# --------------------------------------------------------------------------------------------------
# Burns
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Burn:
    """
    One detected velocity discontinuity. `start_tdb` / `end_tdb` bound the padded cluster (clean
    samples), `epoch_tdb` / `epoch_utc` are the impulsive-equivalent epoch, `delta_v_km_s` (ICRF) its
    velocity jump, `rsw_m_s` that jump in the RSW frame of the pre-burn coast state relative to
    `central` ("Earth", or "Moon" inside `LUNAR_SOI_KM`), `closure_km` how far apart the forward and
    backward coasts pass at the epoch, `delivered_m_s` the magnitude of the summed per-interval
    residuals, `peak_interval_m_s` the largest single-interval residual. `file_join` names the
    trajectory-file boundary inside the cluster, if any; `matched_event` the listed event it was
    matched to (`match_events`).
    """
    start_tdb: np.datetime64
    end_tdb: np.datetime64
    epoch_tdb: np.datetime64
    epoch_utc: np.datetime64
    t_s: float
    delta_v_km_s: ArrayFloat
    dv_m_s: float
    rsw_m_s: ArrayFloat
    central: str
    closure_km: float
    delivered_m_s: float
    peak_interval_m_s: float
    n_intervals: int
    file_join: str
    closure_s: float = math.nan
    kind: str = "burn"
    matched_event: str = ""
    match_offset_s: float = math.nan


def _rsw(r: ArrayFloat, v: ArrayFloat, vec: ArrayFloat) -> ArrayFloat:
    rhat = r / np.linalg.norm(r)
    w = np.cross(r, v)
    what = w / np.linalg.norm(w)
    shat = np.cross(what, rhat)
    out: ArrayFloat = np.array([vec @ rhat, vec @ shat, vec @ what], dtype=np.float64)
    return out


def _clusters(flag: NDArray[np.bool_], merge_gap: int) -> List[Tuple[int, int]]:
    """Runs of True in `flag` as inclusive `(first, last)`, merging runs separated by `<= merge_gap`."""
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    out: List[Tuple[int, int]] = []
    first = last = int(idx[0])
    for k in idx[1:]:
        if int(k) - last - 1 <= merge_gap:
            last = int(k)
        else:
            out.append((first, last))
            first = last = int(k)
    out.append((first, last))
    return out


def _impulse(model: CoastModel, orion: hb.HorizonsVectors, a: int, b: int,
             step_s: float = IMPULSE_STEP_S) -> Tuple[float, ArrayFloat, float, ArrayFloat]:
    """
    `(epoch t_s, dv (3,), closure km, pre-burn state (6,))` for the clean samples `a < b`: coast
    forward from `a` and backward from `b` together on a `step_s` grid, take the grid point where
    they pass closest, then refine inside the step on the linear relative motion `dr + dv tau`
    (exact to first order in `tau`; the grid spacing divides the 60 s samples).
    """
    ta, tb = float(orion.t_s[a]), float(orion.t_s[b])
    n = int(round((tb - ta) / step_s))
    grid = ta + step_s * np.arange(n + 1, dtype=np.float64)
    fwd = np.empty((n + 1, 6), dtype=np.float64)
    bwd = np.empty((n + 1, 6), dtype=np.float64)
    y = np.stack([orion.state[a], orion.state[b]])
    t = np.array([ta, tb], dtype=np.float64)
    step = np.array([step_s, -step_s], dtype=np.float64)
    fwd[0], bwd[n] = y[0], y[1]
    for k in range(1, n + 1):
        y = coast(model, t, y, t + step, 1)
        t = t + step
        fwd[k], bwd[n - k] = y[0], y[1]
    k = int(np.argmin(np.linalg.norm(fwd[:, :3] - bwd[:, :3], axis=1)))
    dr = bwd[k, :3] - fwd[k, :3]
    dv: ArrayFloat = bwd[k, 3:] - fwd[k, 3:]
    tau = float(np.clip(-(dr @ dv) / max(float(dv @ dv), 1e-300), -step_s, step_s))
    tau = float(np.clip(tau, ta - grid[k], tb - grid[k]))
    pre: ArrayFloat = fwd[k].copy()
    pre[:3] += pre[3:] * tau
    return float(grid[k] + tau), dv, float(np.linalg.norm(dr + dv * tau)), pre


def detect_burns(
    orion: Optional[hb.HorizonsVectors] = None,
    model: Optional[CoastModel] = None,
    *,
    threshold_km_s: float = DETECTION_THRESHOLD_KM_S,
    merge_gap: int = DETECTION_MERGE_GAP,
    pad: int = DETECTION_PAD,
) -> List[Burn]:
    """
    Every velocity discontinuity in Orion's trajectory above `threshold_km_s` per 1-min interval,
    as impulsive-equivalent `Burn`s in time order (see the module docstring for the method), each
    classified by its closure time (`kind`). Matching to the event list is `match_events`.
    """
    o = orion if orion is not None else load("orion")
    m = model if model is not None else _default_model()
    _, dv = interval_residuals(o, m)
    mag = np.linalg.norm(dv, axis=1)
    joins = trajectory_files()[1:]
    burns: List[Burn] = []
    for first, last in _clusters(mag > threshold_km_s, merge_gap):
        a = max(0, first - pad)
        b = min(o.t_s.size - 1, last + 1 + pad)
        t_star, jump, closure, pre = _impulse(m, o, a, b)
        moon_r = hermite_position(m.moon, t_star)
        near_moon = bool(np.linalg.norm(pre[:3] - moon_r) < LUNAR_SOI_KM)
        if near_moon:
            h = 1.0
            moon_v = (hermite_position(m.moon, t_star + h) - hermite_position(m.moon, t_star - h)) / (2.0 * h)
            rsw = _rsw(pre[:3] - moon_r, pre[3:] - moon_v, jump)
        else:
            rsw = _rsw(pre[:3], pre[3:], jump)
        start, end = tdb_instant(float(o.t_s[a])), tdb_instant(float(o.t_s[b]))
        join = [f.name for f in joins
                if start - np.timedelta64(60, "s") <= f.start_tdb <= end + np.timedelta64(60, "s")]
        dv_km_s = float(np.linalg.norm(jump))
        closure_s = closure / dv_km_s if dv_km_s > 0.0 else math.inf
        burns.append(Burn(
            start_tdb=start, end_tdb=end, epoch_tdb=tdb_instant(t_star), epoch_utc=_utc_of_tdb(t_star),
            t_s=t_star, delta_v_km_s=jump, dv_m_s=dv_km_s * 1e3,
            rsw_m_s=rsw * 1e3, central="Moon" if near_moon else "Earth", closure_km=closure,
            delivered_m_s=float(np.linalg.norm(dv[first:last + 1].sum(axis=0)) * 1e3),
            peak_interval_m_s=float(mag[first:last + 1].max() * 1e3), n_intervals=last - first + 1,
            file_join=";".join(join),
            closure_s=closure_s, kind="burn" if closure_s < IMPULSIVE_CLOSURE_S else "discontinuity",
        ))
    return burns


@dataclass(frozen=True)
class ListedBurn:
    """A flown burn from the event list, a "Start/Begin ... End ..." pair folded into one:
    `start_utc` / `start_tdb`, the stated `duration_s` and `delta_v_m_s` (from either line), `nan`
    where the list states nothing."""
    name: str
    start_utc: np.datetime64
    start_tdb: np.datetime64
    duration_s: float
    delta_v_m_s: float


def listed_burns(events: Optional[Sequence[MissionEvent]] = None) -> List[ListedBurn]:
    """The flown (not cancelled), in-span Orion burns of the event list, Start/End pairs merged."""
    evs = list(events if events is not None else load_events())
    out: List[ListedBurn] = []
    for k, e in enumerate(evs):
        if not e.is_burn or e.cancelled or not e.in_span or e.name.lower().startswith("end "):
            continue
        dur, dv = e.duration_s, e.delta_v_m_s
        nxt = evs[k + 1] if k + 1 < len(evs) else None
        if nxt is not None and nxt.name.lower().startswith("end "):
            dur = dur if not math.isnan(dur) else nxt.duration_s
            dv = dv if not math.isnan(dv) else nxt.delta_v_m_s
        out.append(ListedBurn(e.name, e.utc, e.tdb, dur, dv))
    return out


def match_events(burns: Sequence[Burn], events: Optional[Sequence[MissionEvent]] = None,
                 *, window_s: float = EVENT_MATCH_WINDOW_S) -> List[Burn]:
    """
    `burns` with `matched_event` / `match_offset_s` filled: each `listed_burns` entry is matched to
    the `kind == "burn"` detection whose impulsive epoch is nearest its listed start plus half its
    stated duration, if within `window_s`. The offset is detected minus listed. Whether it is inside
    `EVENT_MATCH_TOLERANCE_S` is the caller's question: the list's perigee raise time is a planning
    value, 38 min early (`docs/architecture.md`, "Artemis II replay data").
    """
    out = list(burns)
    for lb in listed_burns(events):
        half = 0.0 if math.isnan(lb.duration_s) else 0.5 * lb.duration_s
        target = lb.start_tdb + np.timedelta64(int(half * 1e9), "ns")
        cands = [k for k, b in enumerate(out) if b.kind == "burn"]
        if not cands:
            continue
        offsets = {k: float((out[k].epoch_tdb - target).astype(np.int64)) / 1e9 for k in cands}
        best = min(offsets, key=lambda j: abs(offsets[j]))
        if abs(offsets[best]) <= window_s:
            out[best] = Burn(**{**out[best].__dict__, "matched_event": lb.name, "match_offset_s": offsets[best]})
    return out


def write_burns_csv(burns: Sequence[Burn], path: Union[str, Path]) -> None:
    """`burns.csv`: one row per detection, matched to the event list."""
    matched = match_events(burns)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["epoch_utc", "epoch_tdb", "cluster_start_tdb", "cluster_end_tdb", "dv_m_s", "delivered_m_s",
                    "dv_r_m_s", "dv_s_m_s", "dv_w_m_s", "central", "kind", "closure_km", "closure_s", "n_intervals", "file_join",
                    "matched_event", "match_offset_s"])
        for b in matched:
            w.writerow([_iso(b.epoch_utc), _iso(b.epoch_tdb), _iso(b.start_tdb, "s"), _iso(b.end_tdb, "s"),
                        f"{b.dv_m_s:.4f}", f"{b.delivered_m_s:.4f}",
                        *(f"{x:.4f}" for x in b.rsw_m_s), b.central, b.kind, f"{b.closure_km:.4f}", f"{b.closure_s:.1f}", b.n_intervals,
                        b.file_join, b.matched_event, "" if math.isnan(b.match_offset_s) else f"{b.match_offset_s:.1f}"])


# --------------------------------------------------------------------------------------------------
# Flyby and apogee
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Extremum:
    """An extremum of a distance along Orion's trajectory: `sampled_km` at the best 1-min sample,
    `refined_km` after sub-sample refinement at `epoch_tdb` / `epoch_utc`."""
    epoch_tdb: np.datetime64
    epoch_utc: np.datetime64
    sampled_km: float
    refined_km: float


def _refine(orion: hb.HorizonsVectors, other: Optional[hb.HorizonsVectors], k: int, sign: float) -> Extremum:
    """Refine the extremum of `|r_orion - r_other|` near sample `k` on a 0.1 s Hermite grid over
    the two bracketing intervals (Orion interpolated from its own position and velocity)."""
    t = np.arange(orion.t_s[max(k - 1, 0)], orion.t_s[min(k + 1, orion.t_s.size - 1)] + 1e-9, 0.1, dtype=np.float64)
    ro = hermite_position(orion, t)
    rel = ro - hermite_position(other, t) if other is not None else ro
    d = np.linalg.norm(rel, axis=1)
    j = int(np.argmin(sign * d))
    sampled = orion.position_km[k] - (hermite_position(other, float(orion.t_s[k])) if other is not None else 0.0)
    return Extremum(epoch_tdb=tdb_instant(float(t[j])), epoch_utc=_utc_of_tdb(float(t[j])),
                    sampled_km=float(np.linalg.norm(sampled)), refined_km=float(d[j]))


def closest_lunar_approach(orion: Optional[hb.HorizonsVectors] = None,
                           moon: Optional[hb.HorizonsVectors] = None) -> Extremum:
    """Minimum Orion-Moon centre distance: best 1-min sample, then Hermite-refined."""
    o = orion if orion is not None else load("orion")
    mo = moon if moon is not None else load("moon")
    d = np.linalg.norm(o.position_km - hermite_position(mo, o.t_s), axis=1)
    return _refine(o, mo, int(np.argmin(d)), 1.0)


def maximum_earth_distance(orion: Optional[hb.HorizonsVectors] = None) -> Extremum:
    """Maximum Orion-Earth centre distance: best 1-min sample, then Hermite-refined."""
    o = orion if orion is not None else load("orion")
    return _refine(o, None, int(np.argmax(np.linalg.norm(o.position_km, axis=1))), -1.0)


# --------------------------------------------------------------------------------------------------
# Earth orientation: ICRF +z against the true pole
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EarthOrientation:
    """
    The ICRF -> ITRF93 rotation recovered from `earth_sites.npz` at each of its epochs (6 h from
    `EPOCH_TDB`): `jd_tdb` `(m,)`, `t_s` `(m,)` seconds from `EPOCH_TDB`, `rotation` `(m, 3, 3)` with
    `r_itrf = rotation @ r_icrf`, `pole` `(m, 3)` the ITRF93 z axis (the true rotation pole,
    precession + nutation + polar motion) in ICRF, `pole_tilt_rad` its angle from ICRF +z,
    `pole_ra_rad` its right ascension, `prime_meridian_ra_rad` the ICRF right ascension of the ITRF93
    x axis (the engine's `theta0` for a rotation about ICRF +z), and `orthogonality` the largest
    `|cos|` between the three recovered axes (a check on the data, ~1e-16).
    """
    jd_tdb: ArrayFloat
    t_s: ArrayFloat
    rotation: ArrayFloat
    pole: ArrayFloat
    pole_tilt_rad: ArrayFloat
    pole_ra_rad: ArrayFloat
    prime_meridian_ra_rad: ArrayFloat
    orthogonality: ArrayFloat


def earth_orientation() -> EarthOrientation:
    """
    The Earth's orientation in ICRF from Horizons' own Earth-orientation model (ITRF93, IERS EOP):
    the ICRF vectors of the geodetic sites (0 E, 0 N), (90 E, 0 N) and the north pole on the WGS-84
    ellipsoid are the ITRF93 x, y and z axes up to their lengths (a, a, b). Geometric, same instant,
    so a pure rotation; no memory-sourced constant enters.
    """
    with np.load(DATA_DIR / "earth_sites.npz", allow_pickle=False) as data:
        jd = np.asarray(data["jd_tdb"], dtype=np.float64)
        sites = np.asarray(data["site_icrf_km"], dtype=np.float64)  # (m, 3 sites, 3)
    axes = sites / np.linalg.norm(sites, axis=2, keepdims=True)
    rot: ArrayFloat = np.ascontiguousarray(axes)  # rows are the ITRF axes in ICRF: r_itrf = rot @ r_icrf
    ortho = np.max(np.abs(np.stack([
        np.sum(axes[:, 0] * axes[:, 1], axis=1), np.sum(axes[:, 0] * axes[:, 2], axis=1),
        np.sum(axes[:, 1] * axes[:, 2], axis=1)], axis=1)), axis=1)
    pole = axes[:, 2, :]
    t_s: ArrayFloat = np.asarray((jd - hb.julian_date(EPOCH_TDB)) * hb.SECONDS_PER_DAY, dtype=np.float64)
    return EarthOrientation(
        jd_tdb=jd, t_s=np.round(t_s, 3), rotation=rot, pole=pole,
        pole_tilt_rad=np.arccos(np.clip(pole[:, 2], -1.0, 1.0)),
        pole_ra_rad=np.mod(np.arctan2(pole[:, 1], pole[:, 0]), 2.0 * math.pi),
        prime_meridian_ra_rad=np.mod(np.arctan2(axes[:, 0, 1], axes[:, 0, 0]), 2.0 * math.pi),
        orthogonality=ortho,
    )


def mean_pole_of_date() -> Tuple[ArrayFloat, ArrayFloat]:
    """
    `(jd_tdb (m,), pole (m, 3))`: the Earth's **mean** pole of date in ICRF, from `frame_check.npz`
    (Moon, Sun and Jupiter in ICRF and in Horizons' Earth "BODY EQUATOR" frame, which is the mean
    equator and node of date - it does not rotate with the Earth, whatever its header's "ITRF93"
    suggests). Wahba's problem by SVD on the three unit vectors; the pole is the third row.
    """
    with np.load(DATA_DIR / "frame_check.npz", allow_pickle=False) as data:
        jd = np.asarray(data["jd_tdb"], dtype=np.float64)
        a = np.asarray(data["icrf_km"], dtype=np.float64)
        b = np.asarray(data["itrf93_km"], dtype=np.float64)
    ua = a / np.linalg.norm(a, axis=2, keepdims=True)
    ub = b / np.linalg.norm(b, axis=2, keepdims=True)
    pole = np.empty((jd.size, 3), dtype=np.float64)
    for k in range(jd.size):
        u, _, vt = np.linalg.svd(ub[k].T @ ua[k])
        d = np.sign(np.linalg.det(u @ vt))
        pole[k] = (u @ np.diag([1.0, 1.0, d]) @ vt)[2]
    return jd, pole


def _julian_centuries_tt(jd_tt: float) -> float:
    return (jd_tt - 2451545.0) / 36525.0


def precession_pole_tilt_rad(jd_tt: float) -> float:
    """
    Angle between the mean pole of date and the J2000 pole: the IAU 1976 precession angle
    `theta_A = 2004.3109" T - 0.42665" T^2 - 0.041833" T^3` (Lieske et al. 1977, A&A 58, 1; T Julian
    centuries of TT from J2000 - **from memory, unverified against the text**; checked against the
    mean pole recovered from Horizons, `mean_pole_of_date`).
    """
    t = _julian_centuries_tt(jd_tt)
    return math.radians((2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) / 3600.0)


def precession_in_ra_rad(jd_tt: float) -> float:
    """
    `zeta_A + z_A`, the IAU 1976 general precession in right ascension (Lieske et al. 1977:
    `zeta_A = 2306.2181" T + 0.30188" T^2 + 0.017998" T^3`, `z_A = 2306.2181" T + 1.09468" T^2 +
    0.018203" T^3`, **from memory**): to first order the equinox of date sits at ICRF right ascension
    `-(zeta_A + z_A)`, so an angle measured from the equinox of date (GMST) becomes one measured from
    ICRF +x by subtracting this.
    """
    t = _julian_centuries_tt(jd_tt)
    arcsec = 4612.4362 * t + 1.39656 * t * t + 0.036201 * t ** 3
    return math.radians(arcsec / 3600.0)


def gmst_iau1982_rad(jd_ut1: float) -> float:
    """
    Greenwich mean sidereal time, IAU 1982 (Aoki et al. 1982), in the degree form
    `solar_ephemeris.py` already documents: `280.46061837 + 360.98564736629 d + 0.000387933 T^2 -
    T^3 / 38710000` deg, `d` UT1 days from J2000 and `T = d / 36525` (**from memory**; checked
    against Horizons' apparent sidereal time in `time_check.txt`, which differs by the equation of the
    equinoxes, <= 1.2 s of time).
    """
    d = jd_ut1 - 2451545.0
    t = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t ** 3 / 38710000.0
    return math.radians(deg % 360.0)


def greenwich_apparent_sidereal_hours() -> Tuple[List[np.datetime64], ArrayFloat, ArrayFloat]:
    """`(utc instants, apparent sidereal time at Greenwich (h), TDB - UT (s))` from the committed
    Horizons observer table `time_check.txt`."""
    text = (DATA_DIR / "time_check.txt").read_text(encoding="utf-8")
    body = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    when: List[np.datetime64] = []
    last: List[float] = []
    dt: List[float] = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 5:
            continue
        stamp = cells[0] if cells[0].count(":") == 2 else cells[0] + ":00"  # "2026-Apr-02 00:00"
        when.append(hb.parse_calendar_tdb("A.D. " + stamp))
        last.append(float(cells[3]))
        dt.append(float(cells[4]))
    return when, np.asarray(last, dtype=np.float64), np.asarray(dt, dtype=np.float64)


@dataclass(frozen=True)
class ArcCost:
    """One coast arc between detected burns: start/end TDB, how far the ICRF-+z-J2 coast ends from
    the true-pole-J2 coast (`pole_km`), and how far the true-pole coast ends from NASA's state
    (`model_km`, the coast model's own miss - context for the first number)."""
    start_tdb: np.datetime64
    end_tdb: np.datetime64
    pole_km: float
    model_km: float


def pole_misalignment_cost(
    step_s: float = 20.0,
    burns: Optional[Sequence[Burn]] = None,
    arcs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[ArcCost]:
    """
    What the engine's "J2 about the frame's +z" assumption costs on this replay: every coast arc
    between detected burns (clean cluster ends, arcs of 10 min or more) - or the TDB `(start, stop)`
    pairs in `arcs` - is propagated twice from NASA's state, J2 about ICRF +z and J2 about the true
    pole from `earth_orientation` at the replay's midpoint (the pole moves 0.0001 deg over the
    replay), and the end positions compared. All arcs integrate together, at most `step_s` per step.
    """
    o = load("orion")
    eo = earth_orientation()
    pole = eo.pole[int(np.argmin(np.abs(eo.t_s - o.t_s[o.t_s.size // 2])))]
    z_model = _default_model()
    p_model = _default_model(pole)
    spans: List[Tuple[float, float]] = []
    if arcs is not None:
        spans = [(tdb_seconds(s0), tdb_seconds(s1)) for s0, s1 in arcs]
    else:
        bs = list(burns) if burns is not None else detect_burns(o, z_model)
        starts = [float(o.t_s[0])] + [tdb_seconds(b.end_tdb) for b in bs]
        stops = [tdb_seconds(b.start_tdb) for b in bs] + [float(o.t_s[-1])]
        spans = [(s0, s1) for s0, s1 in zip(starts, stops) if s1 - s0 >= 600.0]
    i0 = np.array([int(np.argmin(np.abs(o.t_s - s0))) for s0, _ in spans], dtype=np.int64)
    i1 = np.array([int(np.argmin(np.abs(o.t_s - s1))) for _, s1 in spans], dtype=np.int64)
    t0: ArrayFloat = np.asarray(o.t_s[i0], dtype=np.float64)
    t1: ArrayFloat = np.asarray(o.t_s[i1], dtype=np.float64)
    n = max(1, int(math.ceil(float(np.max(t1 - t0)) / step_s)))
    y0: ArrayFloat = np.asarray(o.state[i0], dtype=np.float64)
    yz = coast(z_model, t0, y0, t1, n)
    yp = coast(p_model, t0, y0, t1, n)
    return [ArcCost(start_tdb=tdb_instant(float(t0[k])), end_tdb=tdb_instant(float(t1[k])),
                    pole_km=float(np.linalg.norm(yz[k, :3] - yp[k, :3])),
                    model_km=float(np.linalg.norm(yp[k, :3] - o.position_km[i1[k]])))
            for k in range(len(spans))]
