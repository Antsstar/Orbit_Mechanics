"""
`artemis2.py` and `scenarios.artemis2` against the committed Artemis II data set - offline.

Published numbers compared here come from `artemis2.PUBLISHED` (each with its source). What agreement
to expect, and why:

- **Closest lunar approach.** NASA: 6,545 km above the surface (4,067 mi); Horizons' own event list:
  8,282 km from the Moon's centre. The data's minimum is found on the 1-min grid and refined by
  Hermite interpolation. The sampled minimum can exceed the true one by at most
  `r'' (h/2)^2 / 2` with `r'' = (v_rel^2 - mu/r)/r` at periselene: 0.071 km for `h = 60 s`
  (`v_rel` = 1.381 km/s at 8282 km). Both published figures are rounded to 1 km (+-0.5), and the
  altitude depends on the lunar radius used (1737.4 mean vs 1738.1 equatorial: 0.7 km). Budget 1.3 km
  on the altitude, 0.6 km on the centre distance; measured 0.46 km and 0.06 km.
- **Maximum Earth distance.** Horizons' list: 413,146.2 km from the centre at 23:05 UTC; the data give
  413,144.9 km at 23:04:45 UTC - **1.3 km lower, a named anomaly** (the list was written before the
  final trajectory files; the data are what they are). NASA's 252,756 mi = 406,771.1 km "from Earth"
  is consistent with the centre distance less an Earth radius of 6,373.8 km, inside the ellipsoid's
  6,356.8-6,378.1 km - i.e. it is a surface distance.
- **Burns.** Each flown burn in the event list is found within `EVENT_MATCH_TOLERANCE_S` (180 s) of its
  listed start plus half its listed duration - **except the perigee raise burn**, listed at 11:30 UTC
  (a planning time) and found at 12:08:14 UTC, 38 min later: a named anomaly, pinned at 2294 +- 60 s.
  Delta-v agrees with the list to max(0.05 m/s, 3 %) - except the crew-module raise burn (listed
  3.0 m/s, detected 2.71 m/s: its cluster contains the CM/SM separation and is cut by the end of the
  data), a named anomaly. TLI: delivered 388.6 m/s against 388 m/s (list) and 1,274 ft/s = 388.3 m/s
  (NASA), budget 1.0 m/s; the impulsive equivalent is 0.8 % lower (finite-burn loss).
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import artemis2 as a2, horizons_bridge as hb, scenarios
from orbital_engine.frames import ReferenceFrames

FT = 0.3048


@pytest.fixture(scope="module")
def burns() -> List[a2.Burn]:
    return a2.match_events(a2.detect_burns())


@pytest.fixture(scope="module")
def by_event(burns: List[a2.Burn]) -> Dict[str, a2.Burn]:
    return {b.matched_event.split(" (")[0].split(" Total")[0]: b for b in burns if b.matched_event}


# --- event list ---------------------------------------------------------------------------------------

def test_event_list_parses_with_met_consistent_with_utc() -> None:
    events = a2.load_events()
    assert len(events) == 31
    assert sum(e.cancelled for e in events) == 3
    launch = np.datetime64(a2.LAUNCH_UTC, "ns")
    for e in events:
        d, hms = e.met.split("/")
        parts = [int(x) for x in hms.split(":")]
        met_s = int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) > 2 else 0)
        # A MET given to the minute and a UTC given to the minute disagree by < 60 s (launch at :12 s).
        gap = abs(float((e.utc - launch).astype(np.int64)) / 1e9 - met_s)
        assert gap < 60.0, (e.name, gap)
        shift = float((e.tdb - e.utc).astype(np.int64)) / 1e9
        assert 69.1845 < shift < 69.1860
    tli_end = next(e for e in events if e.name.startswith("End TLI"))
    assert tli_end.delta_v_m_s == 388.0


def test_listed_burns_fold_start_end_pairs() -> None:
    listed = {lb.name.split(" (")[0].split(" Total")[0]: lb for lb in a2.listed_burns()}
    assert set(listed) == {
        "Orion upper stage separation burn", "Perigee raise burn", "Start Translunar Injection burn",
        "Begin trajectory correction burn #3", "Return trajectory correction burn #1",
        "Return trajectory correction burn #2", "Return trajectory correction burn #3", "Crew module raise burn",
    }
    tli = listed["Start Translunar Injection burn"]
    assert (tli.duration_s, tli.delta_v_m_s) == (355.0, 388.0)
    otc = listed["Begin trajectory correction burn #3"]
    assert (otc.duration_s, otc.delta_v_m_s) == (18.0, 3.0)


def test_trajectory_files_are_contiguous() -> None:
    files = a2.trajectory_files()
    assert len(files) == 14
    for f0, f1 in zip(files[:-1], files[1:]):
        assert f0.stop_tdb == f1.start_tdb
    assert files[-1].stop_tdb == np.datetime64("2026-04-10T23:51", "ns")


def test_committed_derived_tables_are_current(tmp_path: Path, burns: List[a2.Burn]) -> None:
    a2.write_events_csv(a2.load_events(), tmp_path / "events.csv")
    assert (tmp_path / "events.csv").read_text() == (a2.DATA_DIR / "events.csv").read_text()
    a2.write_burns_csv(burns, tmp_path / "burns.csv")
    assert (tmp_path / "burns.csv").read_text() == (a2.DATA_DIR / "burns.csv").read_text()


# --- the Moon and Sun tables ------------------------------------------------------------------------

def test_moon_distance_and_velocity_are_the_moons() -> None:
    moon = a2.load("moon")
    r = np.linalg.norm(moon.position_km, axis=1)
    assert 356_000.0 < r.min() and r.max() < 407_000.0   # the Moon's orbital range (perigee/apogee)
    assert 393_000.0 < r.min() < 394_000.0 and 404_900.0 < r.max() < 405_100.0   # measured 393,564 / 404,970
    # The tabulated velocity is the derivative of the tabulated position: central difference over
    # 2 x 600 s has error (h^2/6)|r'''| ~ (600^2/6) n^3 r = 2.8e-7 km/s.
    fd = (moon.position_km[2:] - moon.position_km[:-2]) / 1200.0
    assert np.max(np.linalg.norm(fd - moon.velocity_km_s[1:-1], axis=1)) < 1e-6


def test_hermite_interpolation_by_decimation() -> None:
    """Interpolating from every other sample (h = 20 min) onto the omitted ones: truncation
    `h^4 n^4 r / 384` = 1.0e-7 km for the Moon (measured 9.1e-8); the Sun is at the tables'
    print floor (16 significant digits of 1.5e8 km: 1e-7 km; measured 1.2e-7). Bound 5e-7 km."""
    for name in ("moon", "sun"):
        t = a2.load(name)
        half = dataclasses.replace(t, jd_tdb=t.jd_tdb[::2], t_s=t.t_s[::2],
                                   position_km=t.position_km[::2], velocity_km_s=t.velocity_km_s[::2])
        odd = t.t_s[1::2]
        odd = odd[odd <= half.t_s[-1]]
        err = np.linalg.norm(a2.hermite_position(half, odd) - t.position_km[1::2][:odd.size], axis=1)
        assert err.max() < 5e-7, name
    with pytest.raises(ValueError):
        a2.hermite_position(a2.load("moon"), -1.0)


# --- flyby and apogee ---------------------------------------------------------------------------------

def test_closest_lunar_approach_matches_published() -> None:
    ca = a2.closest_lunar_approach()
    assert 0.0 <= ca.sampled_km - ca.refined_km <= 0.072
    assert abs(ca.refined_km - a2.PUBLISHED["closest_approach_center_km"].value) <= 0.6
    altitude = ca.refined_km - a2.MOON_MEAN_RADIUS_KM
    assert abs(altitude - a2.PUBLISHED["closest_approach_altitude_km"].value) <= 1.3
    # Horizons' list: 23:01 UTC (to the minute); NASA: "about" 7 p.m. EDT = 23:00 UTC.
    when = float((ca.epoch_utc - np.datetime64("2026-04-06T23:01", "ns")).astype(np.int64)) / 1e9
    assert abs(when) <= 60.0


def test_maximum_earth_distance_matches_published() -> None:
    mx = a2.maximum_earth_distance()
    header = a2.PUBLISHED["max_distance_center_km"].value
    assert -1.5 < mx.refined_km - header < -1.0      # named anomaly: the list is 1.3 km high
    surface = a2.PUBLISHED["max_distance_nasa_mi"].value * 1.609344
    implied_radius = mx.refined_km - surface
    assert 6356.752 <= implied_radius <= 6378.137    # NASA's figure is a distance from the surface
    when = float((mx.epoch_utc - np.datetime64("2026-04-06T23:05", "ns")).astype(np.int64)) / 1e9
    assert abs(when) <= 60.0


# --- burns --------------------------------------------------------------------------------------------

def test_coast_model_floor_is_far_below_the_threshold(burns: List[a2.Burn]) -> None:
    """Outside the detected clusters (95.5 % of the intervals) the per-minute residual is the coast
    model's floor: median 1.1e-9 km/s, 99th percentile 1.0e-6 (lunar J2 near the flyby, the JPL OD
    files' extra ~1.7e-9 km/s^2), maximum 2.4e-5 - ringing tails just under the 3e-5 threshold."""
    orion = a2.load("orion")
    _, dv = a2.interval_residuals(orion)
    mag = np.linalg.norm(dv, axis=1)
    inside = np.zeros(mag.size, dtype=bool)
    for b in burns:
        inside[orion.at_tdb(b.start_tdb):orion.at_tdb(b.end_tdb)] = True
    quiet = mag[~inside]
    assert quiet.size > 0.95 * mag.size
    assert np.median(quiet) < 1e-8
    assert np.percentile(quiet, 99) < a2.DETECTION_THRESHOLD_KM_S / 10.0


def test_every_listed_burn_is_detected(by_event: Dict[str, a2.Burn]) -> None:
    listed = {lb.name.split(" (")[0].split(" Total")[0]: lb for lb in a2.listed_burns()}
    assert set(by_event) == set(listed)
    for name, b in by_event.items():
        assert b.kind == "burn"
        if name == "Perigee raise burn":
            assert abs(b.match_offset_s - 2294.0) < 60.0   # named anomaly: listed at a planning time
        else:
            assert abs(b.match_offset_s) <= a2.EVENT_MATCH_TOLERANCE_S, (name, b.match_offset_s)
        stated = listed[name].delta_v_m_s
        if math.isnan(stated) or name == "Start Translunar Injection burn" or name == "Crew module raise burn":
            continue
        assert abs(b.dv_m_s - stated) <= max(0.05, 0.03 * stated), (name, b.dv_m_s, stated)


def test_tli_against_nasa(by_event: Dict[str, a2.Burn]) -> None:
    tli = by_event["Start Translunar Injection burn"]
    nasa = a2.PUBLISHED["tli_dv_ft_s"].value * FT
    assert abs(tli.delivered_m_s - nasa) < 1.0 and abs(tli.delivered_m_s - 388.0) < 1.0
    loss = 1.0 - tli.dv_m_s / tli.delivered_m_s
    assert 0.005 < loss < 0.015                        # finite-burn loss, measured 0.80 %
    assert tli.rsw_m_s[1] > 350.0                      # overwhelmingly along-track


def test_return_corrections_against_nasa(by_event: Dict[str, a2.Burn]) -> None:
    for name, key in (("Return trajectory correction burn #1", "rtc1_dv_ft_s"),
                      ("Return trajectory correction burn #2", "rtc2_dv_ft_s"),
                      ("Return trajectory correction burn #3", "rtc3_dv_ft_s")):
        nasa = a2.PUBLISHED[key].value * FT
        assert abs(by_event[name].dv_m_s - nasa) <= max(0.02, 0.03 * nasa), name


def test_crew_module_raise_burn_is_a_named_anomaly(by_event: Dict[str, a2.Burn]) -> None:
    cm = by_event["Crew module raise burn"]
    assert 2.5 < cm.dv_m_s < 2.9                       # listed 3.0 m/s
    assert cm.end_tdb == a2.tdb_instant(float(a2.load("orion").t_s[-1]))   # cut by the end of the data


def test_perigee_raise_burn_raises_perigee_to_the_published_orbit(by_event: Dict[str, a2.Burn]) -> None:
    """Across the burn the osculating perigee rises from ~110 to ~191 km (above 6378.137 km);
    Wikipedia's high Earth orbit is 192 x 70,174 km (radius convention unstated: +-7 km)."""
    o = a2.load("orion")
    b = by_event["Perigee raise burn"]
    before = o.at_tdb(b.start_tdb)
    after = o.at_tdb(b.end_tdb)
    coe, ok = ReferenceFrames.rv_to_coe(o.position_km[[before, after]], o.velocity_km_s[[before, after]],
                                        a2.MU_EARTH_DE440)
    assert np.all(ok)
    p, e = coe[:, 0], coe[:, 1]
    rp_alt = p / (1.0 + e) - 6378.137
    assert 100.0 < rp_alt[0] < 120.0
    assert abs(rp_alt[1] - 192.0) < 8.0
    assert 6.3 < b.dv_m_s < 6.6


def test_unlisted_detections(burns: List[a2.Burn]) -> None:
    """Four impulsive detections the list does not name: the proximity-operations manoeuvres at the
    start of the data (1.10 and 1.99 m/s) and two small impulses (0.164, 0.120 m/s) at whole UTC
    minutes inside the JPL OD file of 2026-04-04/05. Everything else unmatched is a discontinuity."""
    unmatched = [b for b in burns if b.kind == "burn" and not b.matched_event]
    assert [round(b.dv_m_s, 2) for b in unmatched] == [1.10, 1.99, 0.16, 0.12]
    for b in unmatched[2:]:
        seconds = float((b.epoch_utc - b.epoch_utc.astype("datetime64[m]")).astype(np.int64)) / 1e9
        assert min(seconds, 60.0 - seconds) < 0.2


def test_every_file_join_is_flagged(burns: List[a2.Burn]) -> None:
    for f in a2.trajectory_files()[1:]:
        hits = [b for b in burns if b.start_tdb - np.timedelta64(60, "s") <= f.start_tdb <= b.end_tdb + np.timedelta64(60, "s")]
        assert hits, f.name
    joins = [b for b in burns if b.file_join and b.kind == "discontinuity"]
    assert max(b.closure_km for b in joins) > 10.0     # the largest joins move Orion by ~12 km


# --- frame and time -----------------------------------------------------------------------------------

def test_earth_orientation_from_horizons() -> None:
    eo = a2.earth_orientation()
    assert np.max(eo.orthogonality) < 1e-12
    tilt_deg = np.degrees(eo.pole_tilt_rad)
    assert np.all((0.1466 < tilt_deg) & (tilt_deg < 0.1472))           # true pole: 0.1469 deg
    jd, mean_pole = a2.mean_pole_of_date()
    for k in (0, jd.size - 1):
        mean_tilt = math.acos(mean_pole[k, 2])
        assert abs(mean_tilt - a2.precession_pole_tilt_rad(jd[k])) < math.radians(0.5 / 3600)  # measured 0.26"
    # True minus mean pole is nutation (+ polar motion): <= ~10 arcsec.
    k = int(np.argmin(np.abs(jd - eo.jd_tdb[0])))
    assert math.degrees(math.acos(float(np.clip(eo.pole[0] @ mean_pole[k], -1, 1)))) * 3600 < 12.0
    # Rotation rate between 6 h samples: the sidereal rate 7.292115e-5 rad/s.
    d = np.unwrap(eo.prime_meridian_ra_rad)
    rate = np.diff(d) / np.diff(eo.t_s)
    assert np.max(np.abs(rate - 7.292115e-5)) < 5e-10


def test_theta0_at_the_epoch_from_gmst() -> None:
    """The engine's `theta0` at `EPOCH_TDB` (01:58:50.8 UTC) is the ICRF right ascension of the prime
    meridian: Horizons gives 219.8118 deg; IAU 1982 GMST (UT1 ~ UTC) less the precession in right
    ascension gives it to 1.1" measured (budget 3": the first-order transfer neglects the pole's
    offset, and |DUT1| < 0.9 s adds up to 13" - taken as ~0 here)."""
    eo = a2.earth_orientation()
    utc = a2._utc_of_tdb(0.0)
    gmst = a2.gmst_iau1982_rad(hb.julian_date(utc))
    est = (gmst - a2.precession_in_ra_rad(hb.julian_date(a2.EPOCH_TDB))) % (2 * math.pi)
    assert abs(math.degrees(est - eo.prime_meridian_ra_rad[0])) * 3600 < 3.0
    assert abs(math.degrees(eo.prime_meridian_ra_rad[0]) - 219.8118) < 1e-3
    # Horizons' apparent sidereal time minus our GMST is the equation of the equinoxes (<= 1.2 s).
    when, gast_h, _ = a2.greenwich_apparent_sidereal_hours()
    for w, g in zip(when, gast_h):
        eqeq_s = ((g - math.degrees(a2.gmst_iau1982_rad(hb.julian_date(w))) / 15.0 + 12.0) % 24.0 - 12.0) * 3600
        assert abs(eqeq_s) < 1.2


def test_pole_misalignment_cost() -> None:
    """J2 about ICRF +z instead of the true pole (0.147 deg away): 0.70 km over the 15 h coast after
    TLI (the only near-Earth coast with a fast perigee departure), metres on the pre-TLI arcs."""
    post_tli, pre = a2.pole_misalignment_cost(
        step_s=60.0, arcs=[("2026-04-03T00:06", "2026-04-03T15:18"), ("2026-04-02T03:31", "2026-04-02T09:47")])
    assert 0.5 < post_tli.pole_km < 0.9
    assert pre.pole_km < 0.01
    assert post_tli.model_km < 1.0          # the coast model itself: 0.5 km over the same 15 h


# --- scenario -----------------------------------------------------------------------------------------

def test_scenario_seeds_exactly_the_nasa_state(db_session: Session) -> None:
    sim = scenarios.artemis2(db_session)
    orion = a2.load("orion")
    k = orion.at_tdb(a2.DEFAULT_REPLAY_EPOCH_TDB)
    i = sim.name_to_index[scenarios.ORION_NAME]
    assert np.array_equal(sim.global_states[i], orion.state[k])
    assert sim.start_epoch.isoformat() == a2.DEFAULT_REPLAY_EPOCH_TDB
    # The stored elements are the osculating ones through that state.
    r, v = ReferenceFrames.coe_to_rv(sim.coe_states[i:i + 1], scenarios.MU_EARTH)[:2]
    assert np.allclose(np.asarray(r).reshape(3), orion.position_km[k], rtol=0, atol=1e-6)


def test_scenario_one_hour_miss_is_the_missing_third_bodies(db_session: Session) -> None:
    """pm + J2 only, from the default epoch (r ~ 29,000 km): the Moon's and Sun's tidal accelerations
    (~6e-9 and ~2.4e-9 km/s^2) displace Orion by ~0.5 a t^2 = 0.04 + 0.016 km in an hour; measured
    0.051 km."""
    sim = scenarios.artemis2(db_session)
    orion = a2.load("orion")
    for _ in range(60):
        sim.step(60.0)
    k = orion.at_tdb("2026-04-03T02:00:00")
    miss = float(np.linalg.norm(sim.global_states[sim.name_to_index[scenarios.ORION_NAME], :3] - orion.position_km[k]))
    assert 0.02 < miss < 0.1


def test_scenario_refuses_an_off_grid_epoch(db_session: Session) -> None:
    with pytest.raises(KeyError):
        scenarios.artemis2(db_session, epoch_tdb="2026-04-03T01:00:30")


def test_default_epoch_is_clean(burns: List[a2.Burn]) -> None:
    t = np.datetime64(a2.DEFAULT_REPLAY_EPOCH_TDB, "ns")
    assert not any(b.start_tdb <= t <= b.end_tdb for b in burns)
