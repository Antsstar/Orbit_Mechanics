"""
The Artemis II replay: Orion flown under four model tiers through the burns NASA flew, every tier scored
against NASA's navigation trajectory, and the result written in the dashboard's data contract
(`demo/artemis2/SCHEMA.md`). `scripts/build_artemis2_demo.py` is the thin driver.

Everything here reads the committed Horizons data set (`artemis2.py`, `data/artemis2/`); nothing touches
the network. The physics is the engine's: `scenarios.artemis2` seeds Orion (massless Cowell, point mass
Earth, optional J2), `ephemeris.py` adds the Moon and Sun from the DE441 tables at every RK4 stage time,
and `Simulation.schedule_delta_v` applies each burn at its exact epoch.

Tiers (`TIERS`)
---------------
| id | label | physics |
|---|---|---|
| `earth` | Earth only | Earth point mass |
| `earth_j2` | Earth + bulge | + Earth's J2 (EGM96, about ICRF +z) |
| `earth_moon` | Earth + Moon | + the Moon, DE441 via Horizons, direct minus indirect |
| `earth_moon_sun` | Earth + Moon + Sun | + the Sun the same way |

Truth is NASA's trajectory (`nasa`, Horizons -1024). Every tier starts from the **same NASA state** at
`SEED_TDB` = 2026-04-03T01:00 TDB (`artemis2.DEFAULT_REPLAY_EPOCH_TDB`), bit-identical to the data:
the TLI cluster ends at 00:01 TDB and the EPH_OEM file join at 00:06; by 01:00 the per-minute coast
residual is 0.03 mm/s, the coast model's floor, so the seed carries no burn or join transient.

The burns (`replay_burns`)
--------------------------
Every `kind == "burn"` row of `burns.csv` after the seed, applied as `schedule_delta_v` in **RSW about
Earth** of the tier's own pre-burn state, with the components `burns.csv` measured in RSW about Earth of
NASA's pre-burn coast state. The two frames coincide for a tier near NASA (a few km at 400,000 km is
1e-5 rad, 3e-5 m/s of a 3 m/s burn); for a tier thousands of km off they do not, and neither frame is
"right" - such a tier has already lost the mission.

**One exception, found in the data, not in NASA's event list.** On 2026-04-05 the navigation data
switch between two families of solutions that differ by ~1.8 m/s: the `od011v1` solution (to 02:45
TDB) and the early part of `Orion_OEM_20260406_1028` (04:35-14:48) fly one trajectory; the 1125 OEM
(02:45-04:35) and everything after 15:10 - which carries the flyby and the return - fly another. Coasted
across, the second is the first plus one impulse at ~01:25 TDB (end-to-end closure ~1 km, i.e.
impulsive), at the epoch where `od011v1` itself models a 0.164 m/s impulse (plus 0.120 m/s at 01:41).
`burns.csv` sees the switch only as three "discontinuities" (02:43, 04:19, 14:46 - the last with 86 km of
closure). Flown with only the two small modelled impulses, every tier would be ~1.7 m/s off the
flyby trajectory from 5 April; so by default (`reconstruct_family_switch=True`) the pair is replaced
by the impulse `artemis2`'s own forward/backward coast method finds between the clean samples
2026-04-05T01:20 and 15:20 TDB (`FAMILY_SWITCH`). Its size is inferred from NASA's data, not reported
by NASA; the report states both runs.

Two views
---------
- **Replay** (`position_error_km`): one flight per tier from the seed to entry interface or the end of
  the data, all burns applied. It answers "what would this model have predicted", and includes the
  truth's own artefacts.
- **Per arc** (`arc_error_km`): each tier re-seeded from NASA's clean state at the end of every
  detected cluster in `burns.csv` - burns **and** navigation-data discontinuities - and flown to the
  start of the next. Splitting at the discontinuities too is deliberate: an arc then lies inside one
  self-consistent navigation solution, so its error is the model's alone (the `artemis2` ingest used the
  same arcs for its coast-model misses, 0.5 km after TLI and 2.6 km through the flyby).

Definitions
-----------
- Altitude is above the WGS-84 ellipsoid (`a` 6378.137 km, `f` 1/298.257223563) at the geocentric
  latitude about Earth's true pole (`artemis2.earth_orientation`); at 122 km this is geodetic height to
  < 1 m. **Entry interface**: 400,000 ft = 121.92 km. The truth's data end at 23:54:00 TDB, 172 km up,
  ~30 s before it; the truth row carries the event list's 23:53 UTC.
- **Closest lunar approach**: minimum distance from the Moon's centre (DE441), reported as altitude above
  the IAU mean radius 1,737.4 km; refined on a 0.05 s cubic-Hermite grid of the flight's own positions
  and velocities. **Farthest from Earth**: from Earth's centre.
- **DSN contact**: elevation above `DSN_MASK_DEG` (10 deg) at Goldstone DSS-14, Madrid DSS-63 or
  Canberra DSS-43 **and** the Moon not blocking the line of sight. Coordinates: DSN 810-005 module 301,
  Table 5 (WGS-84 geodetic). The 10 deg mask is the DSN 70-m transmit elevation limit in the same
  module (10.4 deg DSS-14, 10.2 deg DSS-43/63), i.e. a two-way link; the mechanical limit is ~6 deg.
  Earth rotation: `earth_rotation()`, a rotation about the true pole fitted to Horizons' own Earth
  orientation (`earth_sites.npz`), which `geometry.elevation_azimuth` consumes as its +z.
- **Lunar blackout**: the Moon's sphere (1,737.4 km) blocks the segment from Orion to Earth's centre.
  NASA reported loss of signal 6:44 p.m. to 7:24 p.m. EDT, "about 40 minutes"
  (https://www.nasa.gov/blogs/missions/2026/04/06/artemis-ii-flight-day-6-lunar-flyby-updates).
- **Solar eclipse** (`solar_eclipses`): the Moon (mean radius) or Earth (equatorial radius, airless)
  covers part (`solar_eclipse_partial`) or all (`solar_eclipse`) of the Sun's apparent disc (radius
  695,700 km) seen from Orion; DE441 Sun and Moon. NASA's trajectory gives a total eclipse by the Moon
  of 54.4 min from 2026-04-07 00:34:27 UTC, ~1.5 h after closest approach, and one by Earth on 3 April
  from 00:10 UTC that the models, seeded at 00:58:51, join already in progress.

Pre-run estimates (written before the first full run; `tests/validation/test_artemis2_replay.py` and
the report hold the measurements against them)
-----------------------------------------------------------------------------------------------------
Open-loop integrals along NASA's trajectory, seed to closest approach (3.92 d), of each missing term,
`dr = int (T - t) a dt`, `dv = int a dt` (no dynamical feedback):

| missing term | dr at flyby | dv at flyby |
|---|---|---|
| Moon (tiers `earth`, `earth_j2`) | 15,800 km | 720 m/s |
| Sun (tier `earth_moon`) | 750 km | 5.4 m/s |
| J2 (tier `earth` vs `earth_j2`) | 40 km | 0.12 m/s |

- `earth` and `earth_j2` **miss the Moon altogether**: the flyby is a 53 deg hyperbolic turn
  (`v_inf` 0.85 km/s, `r_p` 8,282 km, impact parameter `b` 13,450 km). Without the Moon, Orion's straight
  line in the Moon's frame passes near `b`, so closest approach ~11,700 km altitude (vs 6,545), with an
  error at the flyby epoch of order `b - r_p` to the open-loop 15,800 km: **5,000-16,000 km**. Afterwards
  the missing 0.77 km/s gravity assist carries them off: two-body, the seed state is an ellipse of
  apogee 452,900 km and period 12.7 d, so at the end of the data they are near apogee, **~4e5 km** from
  NASA's Orion at Earth, and return to a 196 km perigee (TLI's) around 15 April - above 121.92 km, so
  **no entry**. J2 changes this by tens of km.
- `earth_moon` (no Sun): ~750 km off at the flyby (the Sun's tide, 2e-9 km/s^2 at the seed growing to
  2.3e-8 at lunar distance). The flyby then amplifies it: the turn angle moves 6.0e-5 rad per km of
  B-plane shift, so a few hundred km of shift is ~25 m/s after the flyby, **~1e4 km at the end**.
  Closest-approach altitude off by up to ~0.9 x the B-plane shift, **hundreds of km**, and minutes in
  time. Entry interface is probably missed.
- `earth_moon_sun`: per arc, the ingest's coast-model misses, **~0.5 km over the 14 h after the seed and
  ~2.6 km through the flyby arc** (no lunar harmonics: lunar J2 is 1e-9 km/s^2 at 8,282 km), plus
  J2 about ICRF +z instead of the true pole (0.70 km on the post-TLI coast), plus solar radiation
  pressure, not modelled (A/m ~0.002 m^2/kg: ~1e-11 km/s^2, ~3 km open loop over the 7.9 d). In the
  replay the truth's own inconsistencies dominate: the file joins carry 0.05-0.23 m/s each (~10-50 km
  over the following days) and the 5 April reconstruction is uncertain by ~0.06 m/s (~10 km by the
  flyby): **tens of km at the flyby, then amplified by the flyby's 6e-5 rad/km to ~1 m/s, i.e. hundreds
  of km by entry; entry interface within minutes of NASA's.**

RK4 at 60 s (sub-stepped to one hundredth of a radian of local turning, `MAX_TURN_PER_STEP`) is ~1e-4 km
per flight, far below every number above; `tests/validation/test_artemis2_replay.py` checks it by
halving.

The measurements against these estimates (Earth only 16,093 km at the flyby, Earth + Moon 1,045 km,
Earth + Moon + Sun 46 km and entry 1.8 min after NASA's; per arc 15 m median, 2.65 km through the flyby)
are tabulated in `docs/architecture.md`, "Artemis II replay: four tiers against NASA's navigation".
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from . import artemis2 as a2
from . import horizons_bridge as hb
from . import scenarios
from .custom_types import ArrayFloat
from .database import Base
from .ephemeris import EPHEMERIS_MODEL, EphemerisTable, ephemeris_coefficients
from .geometry import access_windows, elevation_azimuth, segment_clearance
from .simulator import Simulation
from .srp import SUN_RADIUS

__all__ = [
    "SEED_TDB", "EI_ALTITUDE_KM", "WGS84_A_KM", "WGS84_F", "DSN_MASK_DEG", "MAX_TURN_PER_STEP", "GRID_S",
    "FAMILY_SWITCH", "Tier", "TIERS", "ReplayBurn", "Station", "DSN_STATIONS", "Flight", "Extremum",
    "EarthRotation", "Window", "TierResult", "Replay",
    "new_session", "moon_table", "sun_table", "replay_burns", "build_simulation", "fly", "truth_flight",
    "altitude_km", "closest_lunar_approach", "max_earth_distance", "entry_interface", "return_perigee",
    "earth_rotation", "dsn_windows", "lunar_blackouts", "solar_eclipses", "arc_spans", "fly_arcs", "run_replay",
]

SEED_TDB: Final[str] = a2.DEFAULT_REPLAY_EPOCH_TDB
EI_ALTITUDE_KM: Final[float] = 400000.0 * 0.3048 / 1000.0  # 121.92 km, exact
WGS84_A_KM: Final[float] = 6378.137
WGS84_F: Final[float] = 1.0 / 298.257223563
DSN_MASK_DEG: Final[float] = 10.0
#: Sub-step each 60 s grid interval so that `n dt <= MAX_TURN_PER_STEP`, `n` the larger of Orion's
#: angular rate about Earth and about the Moon (0.01 rad: RK4 local error ~(n dt)^5 r / 120 ~ 1e-11 r).
MAX_TURN_PER_STEP: Final[float] = 0.01
#: Output grid, and the Horizons table's own spacing: the truth is compared at its samples, never
#: interpolated.
GRID_S: Final[float] = 60.0
#: Clean samples bracketing the 5 April navigation-solution switch (see the module docstring).
FAMILY_SWITCH: Final[Tuple[str, str]] = ("2026-04-05T01:20:00", "2026-04-05T15:20:00")
#: The two `od011v1` impulses the reconstruction replaces (TDB window).
_FAMILY_SWITCH_REPLACES: Final[Tuple[str, str]] = ("2026-04-05T01:00:00", "2026-04-05T02:00:00")


# --------------------------------------------------------------------------------------------------
# Configuration as data
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier:
    """One model tier: dashboard identity and the physics switches."""
    model_id: str
    label: str
    description: str
    colour: str
    colour_dark: str
    j2: bool
    moon: bool
    sun: bool


TIERS: Final[Tuple[Tier, ...]] = (
    Tier("earth", "Earth only",
         "Earth as a point mass and nothing else. Blind to the Moon, so it cannot fly a lunar flyby.",
         "#2a78d6", "#3987e5", False, False, False),
    Tier("earth_j2", "Earth + bulge",
         "Adds Earth's equatorial bulge (J2). It matters near Earth and hardly at all out by the Moon.",
         "#eb6834", "#d95926", True, False, False),
    Tier("earth_moon", "Earth + Moon",
         "Adds the Moon's gravity, with the Moon where JPL's ephemeris puts it. The first tier that can fly the flyby.",
         "#1baf7a", "#199e70", True, True, False),
    Tier("earth_moon_sun", "Earth + Moon + Sun",
         "Adds the Sun's pull, which differs slightly between Earth and Orion. The engine's best model here.",
         "#eda100", "#c98500", True, True, True),
)


@dataclass(frozen=True)
class ReplayBurn:
    """
    One impulse every tier receives: `t_s` on the tables' clock (TDB seconds from
    `artemis2.EPOCH_TDB`), `rsw_m_s` in RSW about Earth, `key` the dashboard event key, `name` the event
    list's name (or a description for an unlisted one), `reported_m_s` NASA's stated Delta-v (`nan` if
    none), `delivered_m_s` the data's integral, `listed_utc` the event list's time ("" if unlisted),
    `source` "burns.csv" or "reconstructed".
    """
    key: str
    name: str
    t_s: float
    rsw_m_s: Tuple[float, float, float]
    dv_m_s: float
    delivered_m_s: float
    reported_m_s: float
    listed_utc: str
    source: str

    @property
    def rsw_km_s(self) -> ArrayFloat:
        out: ArrayFloat = np.asarray(self.rsw_m_s, dtype=np.float64) * 1e-3
        return out


@dataclass(frozen=True)
class Station:
    """A DSN antenna, WGS-84 geodetic (DSN 810-005 module 301, Table 5)."""
    name: str
    antenna: str
    latitude_deg: float
    longitude_deg: float
    height_km: float


def _dms(d: int, m: int, s: float) -> float:
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


DSN_STATIONS: Final[Tuple[Station, ...]] = (
    Station("Goldstone", "DSS-14", _dms(35, 25, 33.24518), _dms(243, 6, 37.66967) - 360.0, 1.002114),
    Station("Madrid", "DSS-63", _dms(40, 25, 52.34908), _dms(355, 45, 7.16030) - 360.0, 0.865544),
    Station("Canberra", "DSS-43", _dms(-35, 24, 8.74388), _dms(148, 58, 52.55394), 0.689608),
)


# --------------------------------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------------------------------

def new_session() -> Session:
    """An isolated in-memory database (the pattern of `tests/conftest.py`)."""
    engine = create_engine("sqlite:///:memory:", echo=False, connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _table(name: str, offset_s: float) -> EphemerisTable:
    t = a2.load(name.lower())
    return EphemerisTable(name, t.t_s - offset_s, t.position_km, t.velocity_km_s, centre="Earth")


def moon_table(offset_s: float) -> EphemerisTable:
    """DE441 Moon on the arena clock of a sim seeded `offset_s` after `artemis2.EPOCH_TDB`."""
    return _table("Moon", offset_s)


def sun_table(offset_s: float) -> EphemerisTable:
    """DE441 Sun, likewise."""
    return _table("Sun", offset_s)


_BURN_NAMES: Final[Dict[str, Tuple[str, str]]] = {
    "Orion upper stage separation burn": ("upper_stage_separation_burn", "Orion upper stage separation burn"),
    "Perigee raise burn": ("perigee_raise_burn", "Perigee raise burn"),
    "Start Translunar Injection burn": ("translunar_injection", "Translunar injection"),
    "Begin trajectory correction burn #3": ("outbound_correction_burn_3", "Outbound trajectory correction burn 3 (OTC-3)"),
    "Return trajectory correction burn #1": ("return_correction_burn_1", "Return trajectory correction burn 1 (RTC-1)"),
    "Return trajectory correction burn #2": ("return_correction_burn_2", "Return trajectory correction burn 2 (RTC-2)"),
    "Return trajectory correction burn #3": ("return_correction_burn_3", "Return trajectory correction burn 3 (RTC-3)"),
    "Crew module raise burn": ("crew_module_raise_burn", "Crew module raise burn"),
}


def _listed(matched: str, events: Sequence[a2.MissionEvent]) -> Tuple[str, str, float, str]:
    """(key, name, reported Delta-v m/s, listed UTC) of a `burns.csv` `matched_event`."""
    for prefix, (key, name) in _BURN_NAMES.items():
        if matched.startswith(prefix):
            ev = next(e for e in events if e.name == matched)
            dv = ev.delta_v_m_s
            if math.isnan(dv):  # a "Start ..." line: the Delta-v is on its "End ..." line
                k = events.index(ev)
                nxt = events[k + 1] if k + 1 < len(events) else None
                if nxt is not None and nxt.name.lower().startswith("end "):
                    dv = nxt.delta_v_m_s
            return key, name, dv, str(np.datetime_as_string(ev.utc, unit="s"))
    raise KeyError(f"no dashboard key for listed burn {matched!r}")


def _read_burns_csv() -> List[Dict[str, str]]:
    with open(a2.DATA_DIR / "burns.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def family_switch_impulse() -> Tuple[float, ArrayFloat, float]:
    """
    `(t_s, rsw m/s (3,), closure km)`: the impulse joining NASA's trajectory before 2026-04-05T01:20
    TDB to the one after 15:20 (`FAMILY_SWITCH`), by `artemis2`'s forward/backward coast method at a
    20 s step (the arc is 14 h; `IMPULSE_STEP_S` = 5 s is for minute-long clusters).
    """
    orion = a2.load("orion")
    model = a2._default_model()
    a, b = orion.at_tdb(FAMILY_SWITCH[0]), orion.at_tdb(FAMILY_SWITCH[1])
    t_star, dv, closure, pre = a2._impulse(model, orion, a, b, step_s=20.0)
    rsw: ArrayFloat = a2._rsw(pre[:3], pre[3:], dv) * 1e3
    return t_star, rsw, closure


def replay_burns(seed_tdb: str = SEED_TDB, *, reconstruct_family_switch: bool = True,
                 include_before_seed: bool = False) -> List[ReplayBurn]:
    """
    The impulses the tiers receive, in time order: `burns.csv`'s `kind == "burn"` rows after
    `seed_tdb` (all of them with `include_before_seed`, for the truth's event list), with the 5 April
    pair replaced by `family_switch_impulse` when `reconstruct_family_switch`.
    """
    events = a2.load_events()
    t_seed = a2.tdb_seconds(seed_tdb)
    lo, hi = (a2.tdb_seconds(x) for x in _FAMILY_SWITCH_REPLACES)
    out: List[ReplayBurn] = []
    for row in _read_burns_csv():
        if row["kind"] != "burn":
            continue
        t = a2.tdb_seconds(row["epoch_tdb"])
        if t <= t_seed and not include_before_seed:
            continue
        rsw = (float(row["dv_r_m_s"]), float(row["dv_s_m_s"]), float(row["dv_w_m_s"]))
        if reconstruct_family_switch and lo <= t <= hi:
            continue
        if row["matched_event"]:
            key, name, reported, utc = _listed(row["matched_event"], events)
        else:
            key, name, reported, utc = "unlisted_impulse", "Unlisted impulse in the navigation data", math.nan, ""
        out.append(ReplayBurn(key, name, t, rsw, float(row["dv_m_s"]), float(row["delivered_m_s"]),
                              reported, utc, "burns.csv"))
    if reconstruct_family_switch and (include_before_seed or lo > t_seed):
        t, rsw_v, _ = family_switch_impulse()
        dv = float(np.linalg.norm(rsw_v))
        out.append(ReplayBurn("unlisted_velocity_change", "Unlisted velocity change (reconstructed from the data)",
                              t, (float(rsw_v[0]), float(rsw_v[1]), float(rsw_v[2])), dv, dv, math.nan, "",
                              "reconstructed"))
    return sorted(out, key=lambda b: b.t_s)


# --------------------------------------------------------------------------------------------------
# Flying a tier
# --------------------------------------------------------------------------------------------------

def build_simulation(tier: Tier, seed_tdb: str = SEED_TDB) -> Tuple[Simulation, int, int]:
    """`(sim, orion slot, earth slot)`: `scenarios.artemis2` at `seed_tdb` with the tier's physics."""
    sim = scenarios.artemis2(new_session(), epoch_tdb=seed_tdb, j2=tier.j2)
    sim.record_history = False
    i = sim.name_to_index[scenarios.ORION_NAME]
    offset = a2.tdb_seconds(seed_tdb)
    perturbers: List[Tuple[EphemerisTable, float]] = []
    if tier.moon:
        perturbers.append((moon_table(offset), a2.MU_MOON_DE440))
    if tier.sun:
        perturbers.append((sun_table(offset), a2.MU_SUN_DE440))
    if perturbers:
        sim.enable_force_model(EPHEMERIS_MODEL, np.array([i], dtype=np.int64),
                               **ephemeris_coefficients(perturbers, epoch_s=0.0))
    return sim, i, sim.name_to_index["Earth"]


@dataclass
class Flight:
    """A trajectory on the tables' clock: `t_s` `(n,)`, Earth-centred ICRF `state` `(n, 6)`, and how it
    ended (`"entry"`, `"end"`)."""
    model_id: str
    t_s: ArrayFloat
    state: ArrayFloat
    ended: str = "end"

    def table(self) -> EphemerisTable:
        """The flight as a Hermite-interpolable table (positions and velocities)."""
        return EphemerisTable(self.model_id, self.t_s, self.state[:, :3], self.state[:, 3:])


_POLE_CACHE: List[ArrayFloat] = []


def _pole() -> ArrayFloat:
    """Earth's true pole in ICRF, mean over the replay (it moves 1e-4 deg)."""
    if not _POLE_CACHE:
        p = a2.earth_orientation().pole.mean(axis=0)
        _POLE_CACHE.append(p / np.linalg.norm(p))
    return _POLE_CACHE[0]


def altitude_km(r_km: ArrayFloat) -> ArrayFloat:
    """Height above the WGS-84 ellipsoid (radius at the geocentric latitude about the true pole) of
    Earth-centred ICRF positions `(n, 3)`. Within 1 m of geodetic height at 122 km."""
    r = np.atleast_2d(np.asarray(r_km, dtype=np.float64))
    rn = np.linalg.norm(r, axis=1)
    s = (r @ _pole()) / rn
    e2 = WGS84_F * (2.0 - WGS84_F)
    r_ell = WGS84_A_KM * np.sqrt((1.0 - e2) / (1.0 - e2 * (1.0 - s * s)))
    out: ArrayFloat = rn - r_ell
    return out


def _turn_rate(t: float, x: ArrayFloat, moon: hb.HorizonsVectors) -> float:
    """The larger of Orion's angular rates about Earth and about the Moon, rad/s."""
    rate = float(np.linalg.norm(x[3:]) / np.linalg.norm(x[:3]))
    if moon.t_s[0] + 1.0 <= t <= moon.t_s[-1] - 1.0:
        rm = a2.hermite_position(moon, t)
        vm = (a2.hermite_position(moon, t + 1.0) - a2.hermite_position(moon, t - 1.0)) / 2.0
        rate = max(rate, float(np.linalg.norm(x[3:] - vm) / np.linalg.norm(x[:3] - rm)))
    return rate


def fly(tier: Tier, t_end_s: float, burns: Sequence[ReplayBurn] = (), *, seed_tdb: str = SEED_TDB,
        stop_at_entry: bool = True, grid_s: float = GRID_S, max_turn: float = MAX_TURN_PER_STEP) -> Flight:
    """
    Fly `tier` from NASA's state at `seed_tdb` to `t_end_s` (tables' clock), recording every `grid_s`,
    with each burn after the seed and before the end scheduled at its epoch. Each grid interval is cut
    into `ceil(grid_s n / max_turn)` equal steps. Stops at the first sample below entry interface when
    `stop_at_entry`. Tiers with the Moon or Sun must end inside the tables.
    """
    sim, i, e = build_simulation(tier, seed_tdb)
    t0 = a2.tdb_seconds(seed_tdb)
    for b in burns:
        if t0 < b.t_s < t_end_s:
            sim.schedule_delta_v(np.array([i], dtype=np.int64), b.rsw_km_s, epoch_s=b.t_s - t0, label=b.key)
    moon = a2.load("moon")
    n = int(math.floor((t_end_s - t0) / grid_s + 1e-9))
    times = np.empty(n + 1, dtype=np.float64)
    states = np.empty((n + 1, 6), dtype=np.float64)
    times[0] = t0
    states[0] = sim.global_states[i] - sim.global_states[e]
    ended = "end"
    last = n
    for k in range(1, n + 1):
        x = states[k - 1]
        m = max(1, int(math.ceil(grid_s * _turn_rate(float(times[k - 1]), x, moon) / max_turn)))
        for _ in range(m):
            sim.step(grid_s / m)
        times[k] = t0 + k * grid_s
        states[k] = sim.global_states[i] - sim.global_states[e]
        if stop_at_entry and float(altitude_km(states[k, :3])[0]) < EI_ALTITUDE_KM:
            ended, last = "entry", k
            break
    return Flight(tier.model_id, times[:last + 1].copy(), states[:last + 1].copy(), ended)


def truth_flight() -> Flight:
    """NASA's trajectory as a `Flight` (1-min Horizons samples)."""
    o = a2.load("orion")
    return Flight("nasa", o.t_s.copy(), o.state.copy(), "end")


# --------------------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Extremum:
    """`t_s` (tables' clock) and `value_km` of a refined extremum or crossing."""
    t_s: float
    value_km: float


def _refine(flight: Flight, k: int, distance: "DistanceFn", sign: float, step_s: float = 0.05) -> Extremum:
    lo = float(flight.t_s[max(k - 1, 0)])
    hi = float(flight.t_s[min(k + 1, flight.t_s.size - 1)])
    t = np.arange(lo, hi + 1e-9, step_s, dtype=np.float64)
    t = t[t <= flight.t_s[-1]]
    d = distance(t, flight.table().position(t))
    j = int(np.argmin(sign * d))
    return Extremum(float(t[j]), float(d[j]))


class DistanceFn:
    """Distance of positions `(n, 3)` at `t_s` from a reference (Earth's centre, or the Moon's)."""

    def __init__(self, moon: bool) -> None:
        self._moon = a2.load("moon") if moon else None

    def __call__(self, t: ArrayFloat, r: ArrayFloat) -> ArrayFloat:
        ref = a2.hermite_position(self._moon, t) if self._moon is not None else 0.0
        out: ArrayFloat = np.linalg.norm(r - ref, axis=1)
        return out


def _in_moon_span(flight: Flight) -> NDArray[np.bool_]:
    moon = a2.load("moon")
    ok: NDArray[np.bool_] = (flight.t_s >= moon.t_s[0]) & (flight.t_s <= moon.t_s[-1])
    return ok


def closest_lunar_approach(flight: Flight) -> Extremum:
    """Minimum distance from the Moon's centre, km (subtract 1,737.4 for altitude)."""
    f = DistanceFn(True)
    idx = np.flatnonzero(_in_moon_span(flight))
    d = f(flight.t_s[idx], flight.state[idx, :3])
    return _refine(flight, int(idx[int(np.argmin(d))]), f, 1.0)


def max_earth_distance(flight: Flight, after_s: float = -math.inf) -> Extremum:
    """Maximum distance from Earth's centre after `after_s`, km."""
    idx = np.flatnonzero(flight.t_s >= after_s)
    r = np.linalg.norm(flight.state[idx, :3], axis=1)
    return _refine(flight, int(idx[int(np.argmax(r))]), DistanceFn(False), -1.0)


def entry_interface(flight: Flight) -> Optional[Extremum]:
    """First crossing of `EI_ALTITUDE_KM`, bisected on the Hermite interpolant; `None` if none."""
    h = altitude_km(flight.state[:, :3])
    below = np.flatnonzero(h < EI_ALTITUDE_KM)
    if below.size == 0 or below[0] == 0:
        return None
    k = int(below[0])
    tab = flight.table()
    lo, hi = float(flight.t_s[k - 1]), float(flight.t_s[k])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(altitude_km(tab.position(np.array([mid])))[0]) < EI_ALTITUDE_KM:
            hi = mid
        else:
            lo = mid
    return Extremum(0.5 * (lo + hi), EI_ALTITUDE_KM)


def return_perigee(flight: Flight, after_s: float) -> Optional[Extremum]:
    """The first interior minimum of distance from Earth's centre after `after_s`; `None` if the
    flight ends still falling."""
    idx = np.flatnonzero(flight.t_s > after_s)
    r = np.linalg.norm(flight.state[idx, :3], axis=1)
    inner = np.flatnonzero((r[1:-1] <= r[:-2]) & (r[1:-1] < r[2:])) + 1
    if inner.size == 0:
        return None
    return _refine(flight, int(idx[int(inner[0])]), DistanceFn(False), 1.0)


# --------------------------------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EarthRotation:
    """
    Earth's rotation as `geometry.elevation_azimuth` consumes it: a fixed rotation `frame` `(3, 3)`
    (rows: the axes of a frame whose +z is the true pole, in ICRF) and, in that frame, the prime
    meridian's angle `theta0 + omega t` (tables' clock). `rms_rad` / `max_rad` are the linear fit's
    residuals against Horizons' orientation; `station_error_km` the largest station misplacement it
    implies at the 6-h samples.
    """
    frame: ArrayFloat
    theta0: float
    omega: float
    rms_rad: float
    max_rad: float
    station_error_km: float


def _station_fixed() -> ArrayFloat:
    """ITRF positions of `DSN_STATIONS` on WGS-84, km `(3, 3)`."""
    out = np.empty((len(DSN_STATIONS), 3), dtype=np.float64)
    e2 = WGS84_F * (2.0 - WGS84_F)
    for k, s in enumerate(DSN_STATIONS):
        lat, lon = math.radians(s.latitude_deg), math.radians(s.longitude_deg)
        n = WGS84_A_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        out[k] = [(n + s.height_km) * math.cos(lat) * math.cos(lon), (n + s.height_km) * math.cos(lat) * math.sin(lon),
                  (n * (1.0 - e2) + s.height_km) * math.sin(lat)]
    return out


def earth_rotation() -> EarthRotation:
    """Fit `EarthRotation` to `artemis2.earth_orientation()` (ICRF -> ITRF93 every 6 h)."""
    eo = a2.earth_orientation()
    z = _pole()
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    w: ArrayFloat = np.stack([x, y, z])
    itrf_x = eo.rotation[:, 0, :] @ w.T  # ITRF x axis in the pole frame
    ang = np.unwrap(np.arctan2(itrf_x[:, 1], itrf_x[:, 0]))
    coef = np.polyfit(eo.t_s, ang, 1)
    resid = ang - np.polyval(coef, eo.t_s)
    fixed = _station_fixed()
    worst = 0.0
    for k in range(eo.t_s.size):
        th = float(np.polyval(coef, eo.t_s[k]))
        c, s = math.cos(th), math.sin(th)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        model = (w.T @ rz @ fixed.T).T
        truth = (eo.rotation[k].T @ fixed.T).T
        worst = max(worst, float(np.max(np.linalg.norm(model - truth, axis=1))))
    return EarthRotation(w, float(coef[1]), float(coef[0]), float(np.sqrt(np.mean(resid ** 2))),
                         float(np.max(np.abs(resid))), worst)


@dataclass(frozen=True)
class Window:
    """A visibility interval on the tables' clock. `station` is "" for a lunar blackout and the occulting
    body for a solar eclipse."""
    kind: str
    station: str
    start_s: float
    end_s: float


def _edges(t: ArrayFloat, f: ArrayFloat) -> List[Tuple[float, float]]:
    """Intervals where `f > 0`, edges linearly interpolated (clipped at the grid ends)."""
    on = f > 0.0
    out: List[Tuple[float, float]] = []
    start: Optional[float] = float(t[0]) if on[0] else None
    for k in range(1, t.size):
        if on[k] and not on[k - 1]:
            start = float(t[k - 1] + (t[k] - t[k - 1]) * f[k - 1] / (f[k - 1] - f[k]))
        elif on[k - 1] and not on[k]:
            end = float(t[k - 1] + (t[k] - t[k - 1]) * f[k - 1] / (f[k - 1] - f[k]))
            out.append((start if start is not None else float(t[0]), end))
            start = None
    if start is not None:
        out.append((start, float(t[-1])))
    return out


def _moon_clearance(t: ArrayFloat, r_orion: ArrayFloat, r_other: ArrayFloat) -> ArrayFloat:
    """Signed clearance above the Moon's sphere of the segment Orion -> `r_other` (Earth-centred)."""
    rm = a2.hermite_position(a2.load("moon"), t)
    out: ArrayFloat = segment_clearance(r_orion - rm, r_other - rm, body_radius_km=a2.MOON_MEAN_RADIUS_KM)
    return out


def lunar_blackouts(flight: Flight, *, dense_s: float = 1.0) -> List[Window]:
    """Intervals when the Moon blocks Orion -> Earth's centre, edges from a `dense_s` Hermite grid
    around each sampled crossing."""
    idx = np.flatnonzero(_in_moon_span(flight))
    t = flight.t_s[idx]
    c = _moon_clearance(t, flight.state[idx, :3], np.zeros((idx.size, 3)))
    tab = flight.table()
    out: List[Window] = []
    for a, b in _edges(t, -c):
        edges = []
        for x in (a, b):
            tt = np.arange(x - GRID_S, x + GRID_S, dense_s, dtype=np.float64)
            tt = tt[(tt >= t[0]) & (tt <= t[-1])]
            cc = -_moon_clearance(tt, tab.position(tt), np.zeros((tt.size, 3)))
            sub = _edges(tt, cc)
            edges.append(sub[0][0] if x == a and sub else sub[-1][1] if sub else x)
        out.append(Window("lunar_blackout", "", edges[0], edges[1]))
    return out


def _eclipse_margins(t: ArrayFloat, r_orion: ArrayFloat, occulter: str) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    Angular margins (rad) of `occulter`'s disc over the Sun's, seen from Orion: `alpha + beta - sep`
    (positive while any of the Sun is covered) and `beta - alpha - sep` (positive while all of it is),
    `alpha`, `beta` the apparent radii of Sun and occulter and `sep` their centres' separation -
    Montenbruck & Gill Sec. 3.4.2, the geometry `srp.shadow_factor` integrates. Margins, not the lit
    fraction, so the edges interpolate linearly. Earth is a sphere of the WGS-84 equatorial radius.
    """
    to_sun = a2.hermite_position(a2.load("sun"), t) - r_orion
    if occulter == "Moon":
        to_occ = a2.hermite_position(a2.load("moon"), t) - r_orion
        radius = a2.MOON_MEAN_RADIUS_KM
    else:
        to_occ = -r_orion
        radius = WGS84_A_KM
    ds = np.linalg.norm(to_sun, axis=1)
    do = np.linalg.norm(to_occ, axis=1)
    sep = np.arctan2(np.linalg.norm(np.cross(to_sun, to_occ), axis=1), np.einsum("ij,ij->i", to_sun, to_occ))
    alpha = np.arcsin(SUN_RADIUS / ds)
    beta = np.arcsin(np.minimum(radius / do, 1.0))
    partial: ArrayFloat = alpha + beta - sep
    total: ArrayFloat = beta - alpha - sep
    return partial, total


def solar_eclipses(flight: Flight, *, dense_s: float = 1.0) -> List[Window]:
    """
    Intervals when the Moon or Earth hides the Sun from Orion: `solar_eclipse_partial` (any of the
    Sun's disc covered, so it contains the total phase) and `solar_eclipse` (all of it), `station` the
    occulter. Edges from a `dense_s` Hermite grid around each sampled crossing, as `lunar_blackouts`.
    No atmosphere: Earth's refraction and its ~50 km of absorbing air are left out.
    """
    idx = np.flatnonzero(_in_moon_span(flight))
    t = flight.t_s[idx]
    tab = flight.table()
    out: List[Window] = []
    for occ in ("Earth", "Moon"):
        for j, kind in ((0, "solar_eclipse_partial"), (1, "solar_eclipse")):
            f = _eclipse_margins(t, flight.state[idx, :3], occ)[j]
            for a, b in _edges(t, f):
                edges = []
                for x in (a, b):
                    tt = np.arange(x - GRID_S, x + GRID_S, dense_s, dtype=np.float64)
                    tt = tt[(tt >= t[0]) & (tt <= t[-1])]
                    sub = _edges(tt, _eclipse_margins(tt, tab.position(tt), occ)[j])
                    edges.append(sub[0][0] if x == a and sub else sub[-1][1] if sub else x)
                out.append(Window(kind, occ, edges[0], edges[1]))
    out.sort(key=lambda w: w.start_s)
    return out


def dsn_windows(flight: Flight, rotation: EarthRotation, *, mask_deg: float = DSN_MASK_DEG) -> List[Window]:
    """
    `dsn_contact` windows per station: elevation above `mask_deg` **and** the station's line of sight
    clear of the Moon. Elevation from `geometry.elevation_azimuth` in the true-pole frame, with WGS-84
    **geodetic** latitude as its latitude: that makes its local vertical the geodetic one the mask is
    defined against, at the price of a station moved <= 21 km, which subtends <= 0.05 deg from 25,000 km
    and 0.003 deg from the Moon.
    """
    w = rotation.frame
    pos = flight.state[:, :3] @ w.T
    lat = np.radians([s.latitude_deg for s in DSN_STATIONS])
    lon = np.radians([s.longitude_deg for s in DSN_STATIONS])
    r_st = np.linalg.norm(_station_fixed(), axis=1)
    topo = elevation_azimuth(pos, flight.t_s, latitude_rad=lat, longitude_rad=lon,
                             altitude_km=r_st - WGS84_A_KM, omega=rotation.omega, body_radius_km=WGS84_A_KM,
                             theta0=rotation.theta0, epoch_s=0.0)
    el = topo.elevation_rad[:, :, 0]
    in_moon = _in_moon_span(flight)
    out: List[Window] = []
    th = rotation.theta0 + rotation.omega * flight.t_s
    fixed = _station_fixed()
    for k, st in enumerate(DSN_STATIONS):
        f = el[:, k] - math.radians(mask_deg)
        # Station in ICRF: frame^T Rz(theta) r_fixed.
        c, s = np.cos(th), np.sin(th)
        rx = c * fixed[k, 0] - s * fixed[k, 1]
        ry = s * fixed[k, 0] + c * fixed[k, 1]
        st_icrf = np.stack([rx, ry, np.full_like(rx, fixed[k, 2])], axis=1) @ w
        clear = np.full(flight.t_s.size, 1.0e6)  # outside the Moon table: nothing to occult
        idx = np.flatnonzero(in_moon)
        clear[idx] = _moon_clearance(flight.t_s[idx], flight.state[idx, :3], st_icrf[idx])
        # Contact needs both; each margin is interpolated in its own units, then intersected.
        for a, b in _intersect(_edges(flight.t_s, f), _edges(flight.t_s, clear)):
            out.append(Window("dsn_contact", st.name, a, b))
    return out


def contact_gaps(windows: Sequence[Window], t0: float, t1: float) -> List[Tuple[float, float]]:
    """Intervals inside `[t0, t1]` when **no** station has a `dsn_contact` window."""
    spans = sorted((w.start_s, w.end_s) for w in windows if w.kind == "dsn_contact")
    gaps: List[Tuple[float, float]] = []
    cursor = t0
    for a, b in spans:
        if a > cursor:
            gaps.append((cursor, min(a, t1)))
        cursor = max(cursor, b)
        if cursor >= t1:
            break
    if cursor < t1:
        gaps.append((cursor, t1))
    return [(a, b) for a, b in gaps if b > a]


def _intersect(p: Sequence[Tuple[float, float]], q: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Intersection of two sorted lists of disjoint intervals."""
    out: List[Tuple[float, float]] = []
    i = j = 0
    while i < len(p) and j < len(q):
        a, b = max(p[i][0], q[j][0]), min(p[i][1], q[j][1])
        if a < b:
            out.append((a, b))
        if p[i][1] < q[j][1]:
            i += 1
        else:
            j += 1
    return out


# --------------------------------------------------------------------------------------------------
# Arcs
# --------------------------------------------------------------------------------------------------

def arc_spans(seed_tdb: str = SEED_TDB, *, min_s: float = 600.0) -> List[Tuple[str, float]]:
    """`(start TDB, end t_s)` of every coast arc between the clusters of `burns.csv` (burns and
    discontinuities) after `seed_tdb`: start at a cluster's clean end, stop at the next one's clean
    start (or the data end). Arcs shorter than `min_s` are dropped."""
    rows = sorted(_read_burns_csv(), key=lambda r: r["cluster_start_tdb"])
    t_seed = a2.tdb_seconds(seed_tdb)
    end = float(a2.load("orion").t_s[-1])
    starts: List[str] = [seed_tdb]
    stops: List[float] = []
    for r in rows:
        a, b = a2.tdb_seconds(r["cluster_start_tdb"]), a2.tdb_seconds(r["cluster_end_tdb"])
        if b <= t_seed:
            continue
        stops.append(a)
        starts.append(r["cluster_end_tdb"])
    stops.append(end)
    return [(s0, s1) for s0, s1 in zip(starts, stops) if s1 - a2.tdb_seconds(s0) >= min_s]


def fly_arcs(tier: Tier, spans: Sequence[Tuple[str, float]]) -> List[Flight]:
    """One re-seeded coast per arc (no burns inside an arc, by construction)."""
    return [fly(tier, s1, (), seed_tdb=s0) for s0, s1 in spans]


# --------------------------------------------------------------------------------------------------
# The whole replay
# --------------------------------------------------------------------------------------------------

@dataclass
class TierResult:
    """Everything the export and the summary need about one tier."""
    tier: Tier
    flight: Flight
    arcs: List[Flight]
    closest: Extremum
    farthest: Extremum
    entry: Optional[Extremum]
    perigee: Optional[Extremum]
    windows: List[Window]
    extended: Optional[Flight] = None


@dataclass
class Replay:
    """The truth, the tiers, the burns and the Earth rotation used."""
    truth: Flight
    tiers: List[TierResult]
    burns: List[ReplayBurn]
    rotation: EarthRotation
    truth_closest: Extremum
    truth_farthest: Extremum
    truth_windows: List[Window]
    notes: List[str] = field(default_factory=list)


def _end_for(tier: Tier) -> float:
    """Tiers with an ephemeris end inside the Moon/Sun tables; the others 6 h past the data end."""
    moon = a2.load("moon")
    data_end = float(a2.load("orion").t_s[-1])
    if tier.moon or tier.sun:
        return float(moon.t_s[-1]) - GRID_S
    return data_end + 6.0 * 3600.0


def run_replay(tiers: Sequence[Tier] = TIERS, *, reconstruct_family_switch: bool = True,
               arcs: bool = True, extend_days: float = 16.0) -> Replay:
    """
    Fly every tier (the replay view and, with `arcs`, the per-arc view), find its events and windows.
    A tier that neither enters nor turns round before its end is flown on, unexported, to at most
    `extend_days` after the seed to find its return perigee (Earth-only tiers only: the others would
    leave the tables).
    """
    truth = truth_flight()
    rot = earth_rotation()
    burns = replay_burns(reconstruct_family_switch=reconstruct_family_switch)
    spans = arc_spans() if arcs else []
    t_ca_truth = closest_lunar_approach(truth)
    results: List[TierResult] = []
    for tier in tiers:
        flight = fly(tier, _end_for(tier), burns)
        closest = closest_lunar_approach(flight)
        farthest = max_earth_distance(flight)
        entry = entry_interface(flight)
        perigee = None if entry is not None else return_perigee(flight, farthest.t_s)
        extended: Optional[Flight] = None
        if entry is None and perigee is None and not (tier.moon or tier.sun):
            extended = fly(tier, a2.tdb_seconds(SEED_TDB) + extend_days * 86400.0, burns, grid_s=GRID_S)
            entry = entry_interface(extended)
            if entry is None:
                perigee = return_perigee(extended, max_earth_distance(extended).t_s)
        wins = dsn_windows(flight, rot) + lunar_blackouts(flight) + solar_eclipses(flight)
        results.append(TierResult(tier, flight, fly_arcs(tier, spans) if arcs else [], closest, farthest,
                                  entry, perigee, wins, extended))
    return Replay(truth, results, burns, rot, t_ca_truth, max_earth_distance(truth),
                  dsn_windows(truth, rot) + lunar_blackouts(truth) + solar_eclipses(truth))


# --------------------------------------------------------------------------------------------------
# Export: the dashboard's data contract (demo/artemis2/SCHEMA.md)
# --------------------------------------------------------------------------------------------------

#: Dashboard `t_s = 0`: launch, 2026-04-01T22:35:12 UTC, on TDB.
LAUNCH_TDB: Final[np.datetime64] = hb.utc_to_tdb(a2.LAUNCH_UTC)
#: Tables' clock minus the dashboard's: add this to a table time to get the dashboard's `t_s`.
EXPORT_OFFSET_S: Final[float] = -a2.tdb_seconds(LAUNCH_TDB)
#: Discontinuities in NASA's data marked as `milestone` events when their closure exceeds this.
MARK_JUMP_KM: Final[float] = 5.0
TRUTH_ID: Final[str] = "nasa"


def _utc_t(utc: str) -> float:
    """Dashboard `t_s` of a UTC instant."""
    return a2.tdb_seconds(hb.utc_to_tdb(utc)) + EXPORT_OFFSET_S


def _near(flight: Flight) -> NDArray[np.bool_]:
    """Within 60,000 km of Earth or 40,000 km of the Moon (the contract's dense-sampling rule)."""
    near: NDArray[np.bool_] = np.linalg.norm(flight.state[:, :3], axis=1) < 60000.0
    idx = np.flatnonzero(_in_moon_span(flight))
    dm = DistanceFn(True)(flight.t_s[idx], flight.state[idx, :3])
    near[idx] |= dm < 40000.0
    return near


def _on(t: ArrayFloat, step: float) -> NDArray[np.bool_]:
    """Whole multiples of `step` seconds (the tables' clock is on whole minutes)."""
    r = np.mod(t, step)
    out: NDArray[np.bool_] = (r < 1e-6) | (step - r < 1e-6)
    return out


def _select(flight: Flight, fine: float, coarse: float,
            extra: Sequence[Tuple[float, float]] = ()) -> NDArray[np.int64]:
    """Sample indices: every `fine` s where `_near`, every `coarse` s elsewhere, every sample inside
    the `extra` intervals, and both ends."""
    t = flight.t_s
    keep = (_near(flight) & _on(t, fine)) | _on(t, coarse)
    for a, b in extra:
        keep |= (t >= a) & (t <= b)
    keep[0] = True
    keep[-1] = True
    out: NDArray[np.int64] = np.flatnonzero(keep).astype(np.int64)
    return out


def _discontinuities(after_s: float = -math.inf) -> List[Dict[str, str]]:
    return [r for r in _read_burns_csv()
            if r["kind"] == "discontinuity" and a2.tdb_seconds(r["epoch_tdb"]) > after_s]


def _truth_index(truth: Flight, t: ArrayFloat) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Indices of `truth` samples at exactly the times `t` (the shared 1-min grid), and which exist."""
    k = np.clip(np.searchsorted(truth.t_s, t), 0, truth.t_s.size - 1).astype(np.int64)
    ok: NDArray[np.bool_] = np.abs(truth.t_s[k] - t) < 1e-6
    return k, ok


def position_error(flight: Flight, truth: Flight) -> Tuple[ArrayFloat, ArrayFloat]:
    """`(t_s, |r - r_truth|)` at every sample both have (never interpolated)."""
    k, ok = _truth_index(truth, flight.t_s)
    err: ArrayFloat = np.linalg.norm(flight.state[ok, :3] - truth.state[k[ok], :3], axis=1)
    t: ArrayFloat = flight.t_s[ok]
    return t, err


def _fmt(x: float, nd: int) -> str:
    s = f"{x:.{nd}f}"
    return "0" if float(s) == 0.0 else s


def burn_value(b: ReplayBurn) -> float:
    """What the dashboard shows for a burn: NASA's reported Delta-v, else the data's delivered one."""
    return b.reported_m_s if not math.isnan(b.reported_m_s) else b.delivered_m_s


def _burn_note(b: ReplayBurn) -> str:
    utc = str(np.datetime_as_string(a2._utc_of_tdb(b.t_s), unit="s"))
    if b.source == "reconstructed":
        return (f"Not in NASA's event list: the navigation data after 5 April imply a {b.dv_m_s:.2f} m/s "
                f"velocity change at about {utc} UTC (reconstructed by OrbitalEngine; every model receives it)")
    if not b.listed_utc:
        return (f"Not in NASA's event list: a {b.delivered_m_s:.3f} m/s impulse in the navigation data at "
                f"{utc} UTC (detected by OrbitalEngine)")
    rep = f"NASA reported {b.reported_m_s:g} m/s" if not math.isnan(b.reported_m_s) else "NASA stated no delta-v"
    return (f"{b.name}: {rep}; the data show {b.delivered_m_s:.3f} m/s delivered, impulsive equivalent at "
            f"{utc} UTC (listed start {b.listed_utc} UTC)")


_Event = Tuple[float, str, str, str, str, str, str]


def _events(replay: Replay, all_burns: Sequence[ReplayBurn]) -> List[_Event]:
    x0 = EXPORT_OFFSET_S
    t_seed = a2.tdb_seconds(SEED_TDB)
    ev: List[_Event] = []

    def add(t: float, mid: str, key: str, kind: str, value: Optional[float], unit: str, note: str,
            nd: int = 1) -> None:
        ev.append((t, mid, key, kind, "" if value is None else _fmt(value, nd), unit if value is not None else "", note))

    add(_utc_t(a2.LAUNCH_UTC), TRUTH_ID, "launch", "milestone", None, "",
        "Liftoff from Kennedy Space Center LC-39B, 22:35:12 UTC (NASA)")
    add(_utc_t("2026-04-02T01:59:30"), TRUTH_ID, "orion_icps_separation", "milestone", None, "",
        "Orion separates from the upper stage (event list, 01:59:30 UTC); NASA's trajectory data start here")
    for b in all_burns:
        t = _utc_t(b.listed_utc) if b.listed_utc else b.t_s + x0
        add(t, TRUTH_ID, b.key, "burn", burn_value(b), "m/s", _burn_note(b), 2)
    add(t_seed + x0, TRUTH_ID, "model_seed", "milestone", None, "",
        "Every model starts from NASA's position and velocity here (2026-04-03 00:58:51 UTC), an hour after "
        "the translunar injection burn")
    for key, utc, note in [
        ("enters_lunar_sphere_of_influence", "2026-04-06T05:38:44",
         "Enters the Moon's sphere of influence, 62,800 km from its centre (event list)"),
        ("exits_lunar_sphere_of_influence", "2026-04-07T16:23:00", "Leaves the Moon's sphere of influence (event list)"),
        ("crew_service_module_separation", "2026-04-10T23:33:00",
         "The crew module separates from the European-built service module (event list)"),
        ("splashdown", "2026-04-11T00:07:00", "Splashdown in the Pacific off Baja California (event list)"),
    ]:
        add(_utc_t(utc), TRUTH_ID, key, "milestone", None, "", note)
    add(_utc_t("2026-04-06T23:01:00"), TRUTH_ID, "closest_lunar_approach", "apsis", 6545.0, "km",
        "Reported by NASA: 4,067 mi (6,545 km) above the lunar surface; time 23:01 UTC from NASA's event list "
        "(8,282 km from the Moon's centre). NASA's own trajectory gives "
        f"{replay.truth_closest.value_km - a2.MOON_MEAN_RADIUS_KM:,.1f} km above the 1,737.4 km mean radius")
    add(_utc_t("2026-04-06T23:05:00"), TRUTH_ID, "max_earth_distance", "apsis", 413146.2, "km",
        "Reported in NASA's event list: 413,146.2 km from Earth's CENTRE at 23:05 UTC, the reference every "
        "model here uses. NASA's public record, 252,756 mi = 406,771 km, is measured from Earth's SURFACE")
    add(_utc_t("2026-04-10T23:53:00"), TRUTH_ID, "entry_interface", "milestone", EI_ALTITUDE_KM, "km",
        "NASA's event list: entry interface (122 km) at 23:53 UTC. NASA's trajectory data end at 23:52:51 UTC, "
        "172 km up")
    for r in _discontinuities(t_seed):
        if float(r["closure_km"]) > MARK_JUMP_KM:
            where = f" where file {r['file_join']} takes over" if r["file_join"] else " inside one navigation file"
            add(a2.tdb_seconds(r["cluster_start_tdb"]) + x0, TRUTH_ID, "navigation_data_jump", "milestone",
                float(r["closure_km"]), "km",
                f"Artefact in NASA's data, not a spacecraft event: the trajectory jumps (closure "
                f"{float(r['closure_km']):.0f} km) between {r['cluster_start_tdb'][5:16]} and "
                f"{r['cluster_end_tdb'][5:16]} TDB{where}. Every model's error steps here")
    for res in replay.tiers:
        mid = res.tier.model_id
        add(res.closest.t_s + x0, mid, "closest_lunar_approach", "apsis",
            res.closest.value_km - a2.MOON_MEAN_RADIUS_KM, "km",
            f"Predicted closest approach: {res.closest.value_km:,.1f} km from the Moon's centre; the value is the "
            "altitude above the 1,737.4 km mean radius")
        add(res.farthest.t_s + x0, mid, "max_earth_distance", "apsis", res.farthest.value_km, "km",
            "Predicted farthest distance from Earth's centre (within the flown span)")
        if res.entry is not None:
            add(res.entry.t_s + x0, mid, "entry_interface", "milestone", EI_ALTITUDE_KM, "km",
                "Predicted entry interface: 121.92 km (400,000 ft) above the WGS-84 ellipsoid")
        elif res.perigee is not None:
            add(res.perigee.t_s + x0, mid, "return_perigee", "apsis", res.perigee.value_km - WGS84_A_KM, "km",
                "Closest return to Earth, which misses the atmosphere (altitude above the 6,378.137 km equatorial "
                "radius)" + (" - found by flying on past the end of the data" if res.extended is not None else ""))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def export_dashboard(replay: Replay, out_dir: str, *, generated_utc: str, engine_commit: str,
                     engine_version: Optional[str], notes: Sequence[str] = ()) -> Dict[str, int]:
    """
    Write `meta.json`, `models.csv`, `trajectory.csv`, `metrics.csv`, `events.csv`, `windows.csv`
    to `out_dir` per `demo/artemis2/SCHEMA.md`, `t_s` counted from launch. Returns each file's size.
    """
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)

    def path(name: str) -> str:
        return os.path.join(out_dir, name)

    x0 = EXPORT_OFFSET_S
    truth = replay.truth
    jumps = [(a2.tdb_seconds(r["cluster_start_tdb"]) - 120.0, a2.tdb_seconds(r["cluster_end_tdb"]) + 120.0)
             for r in _discontinuities()]

    with open(path("models.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model_id", "label", "description", "colour", "colour_dark", "is_truth"])
        w.writerow([TRUTH_ID, "NASA navigation",
                    "Orion's trajectory from NASA's navigation team, via JPL Horizons. The reference every model is "
                    "scored against.", "#17212e", "#f2f5f8", 1])
        for res in replay.tiers:
            tr = res.tier
            w.writerow([tr.model_id, tr.label, tr.description, tr.colour, tr.colour_dark, 0])

    moon = a2.load("moon")
    t_max = max(float(res.flight.t_s[-1]) for res in replay.tiers)
    with open(path("trajectory.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["t_s", "model_id", "body", "x_km", "y_km", "z_km"])
        for f in [truth] + [res.flight for res in replay.tiers]:
            for k in _select(f, 60.0, 300.0):
                p = f.state[k]
                w.writerow([_fmt(float(f.t_s[k]) + x0, 3), f.model_id, "orion", _fmt(p[0], 3), _fmt(p[1], 3),
                            _fmt(p[2], 3)])
        tm = np.arange(0.0, min(t_max, float(moon.t_s[-1])) + 1e-9, 600.0, dtype=np.float64)
        for tq, pq in zip(tm, a2.hermite_position(moon, tm)):
            w.writerow([_fmt(float(tq) + x0, 3), TRUTH_ID, "moon", _fmt(pq[0], 3), _fmt(pq[1], 3), _fmt(pq[2], 3)])

    all_burns = replay_burns(include_before_seed=True,
                             reconstruct_family_switch=any(b.source == "reconstructed" for b in replay.burns))
    with open(path("metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["t_s", "model_id", "metric", "value", "unit"])

        def rows(f: Flight, name: str, unit: str, idx: NDArray[np.int64], vals: ArrayFloat, nd: int) -> None:
            for k, v in zip(idx, vals):
                if math.isfinite(float(v)):
                    w.writerow([_fmt(float(f.t_s[k]) + x0, 3), f.model_id, name, _fmt(float(v), nd), unit])

        flights: List[Tuple[Flight, List[Flight]]] = [(truth, [])]
        flights += [(res.flight, res.arcs) for res in replay.tiers]
        for f, arcs in flights:
            sel = _select(f, 120.0, 600.0, jumps)
            if f.model_id != TRUTH_ID:
                k, ok = _truth_index(truth, f.t_s[sel])
                err = np.full(sel.size, np.nan)
                err[ok] = np.linalg.norm(f.state[sel[ok], :3] - truth.state[k[ok], :3], axis=1)
                rows(f, "position_error_km", "km", sel, err, 4)
            rows(f, "earth_range_km", "km", sel, np.linalg.norm(f.state[sel, :3], axis=1), 1)
            inm = sel[_in_moon_span(f)[sel]]
            rows(f, "moon_range_km", "km", inm, DistanceFn(True)(f.t_s[inm], f.state[inm, :3]), 1)
            rows(f, "speed_km_s", "km/s", sel, np.linalg.norm(f.state[sel, 3:], axis=1), 5)
            if f.model_id == TRUTH_ID:
                dv = np.array([sum(burn_value(b) for b in all_burns if b.t_s <= t + 1e-6) for t in f.t_s[sel]],
                              dtype=np.float64)
                rows(f, "delta_v_m_s", "m/s", sel, dv, 2)
            for a in arcs:
                s2 = _select(a, 120.0, 600.0, jumps)
                k, ok = _truth_index(truth, a.t_s[s2])
                ae = np.full(s2.size, np.nan)
                ae[ok] = np.linalg.norm(a.state[s2[ok], :3] - truth.state[k[ok], :3], axis=1)
                rows(Flight(f.model_id, a.t_s, a.state), "arc_error_km", "km", s2, ae, 4)

    with open(path("events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["t_s", "model_id", "event", "kind", "value", "unit", "note"])
        for row in _events(replay, all_burns):
            w.writerow([_fmt(row[0], 3), *row[1:]])

    with open(path("windows.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model_id", "kind", "station", "start_s", "end_s"])
        groups: List[Tuple[str, List[Window]]] = [(TRUTH_ID, replay.truth_windows)]
        groups += [(res.tier.model_id, res.windows) for res in replay.tiers]
        for mid, wins in groups:
            for win in wins:
                w.writerow([mid, win.kind, win.station, _fmt(win.start_s + x0, 1), _fmt(win.end_s + x0, 1)])

    meta = {
        "synthetic": False,
        "title": "Artemis II lunar flyby",
        "epoch_utc": "2026-04-01T22:35:12.000Z",
        "epoch_tdb": str(np.datetime_as_string(LAUNCH_TDB, unit="ms")) + " TDB",
        "epoch_label": "Launch (T+0)",
        "frame": "Earth-centred ICRF (J2000 axes), km",
        "source": ("JPL Horizons API 1.2, retrieved 2026-09-27T07:04:29Z: Orion -1024 {source: Artemis_II_merged}, "
                   "object data revised Apr 20, 2026; Moon 301 and Sun 10 {source: DE441}. Models: OrbitalEngine "
                   "artemis2_replay (Cowell, RK4 at 60 s, sub-stepped to 0.01 rad of turning)"),
        "truth_source": "NASA/JSC Orion navigation (14 concatenated OEM/OD files) via JPL Horizons",
        "generated_utc": generated_utc,
        "engine_commit": engine_commit,
        "engine_version": engine_version,
        "notes": list(notes),
    }
    with open(path("meta.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, indent=2) + "\n")
    names = ["meta.json", "models.csv", "trajectory.csv", "metrics.csv", "events.csv", "windows.csv"]
    return {n: os.path.getsize(path(n)) for n in names}
