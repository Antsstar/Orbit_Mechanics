"""
The Artemis II replay (`orbital_engine.artemis2_replay`) on short arcs; the full run is
`scripts/build_artemis2_demo.py`.

Expected magnitudes, each derived before measuring (the module docstring has the whole budget):

- **Seed**: NASA's state, bit for bit (`scenarios.artemis2` writes it; nothing converts it).
- **Burns**: exactly `burns.csv`'s `kind == "burn"` rows after the seed; the 5 April pair replaced by one
  reconstructed impulse of ~1.8 m/s whose end-to-end closure is ~1 km (an impulse: a data jump closes
  to tens of km or more).
- **A zero Delta-v**: on a step boundary, bit-identical (`v + 0 = v`, and nothing splits); off a
  boundary, the split moves RK4's nodes, a local truncation change `~ r (n h)^5 / 120` = 1e-8 km here.
- **Engine vs `artemis2.coast`** (the ingest's independent RK4 over the same tables, same state): the
  only model difference is Earth's GM (`scenarios.MU_EARTH` 398600.4418 against DE440's 398600.4355,
  1.6e-8 relative), worth `~1.6e-8 r (1 + 3 n t / 2)` ~ 1e-3 km over the 14 h first arc; bound 0.01 km.
- **The first arc** (seed -> 04-03 15:18 TDB, 14.3 h) under Earth + Moon + Sun against NASA: the ingest's
  coast from 00:06 missed by 0.5 km (true pole) with a further 0.70 km from the ICRF-z pole, both
  dominated by the first hour near perigee, which this arc starts after; bound 0.1 km.
- **The flyby arc** (04-06 03:07 -> 17:34 TDB) against NASA: the ingest's 2.6 km (no lunar harmonics).
  Without the Sun, the Sun's tide over 14.5 h, `~0.5 a t^2` with `a` ~2e-8 km/s^2: ~27 km.
- **Halving the step** on that arc: RK4 at `n h` <= 0.01 is ~1e-4 km; bound 1e-3 km.
- **Earth rotation**: the fitted rotation reproduces Horizons' station vectors to tens of metres, and
  `theta0` the ingest's 219.8118 deg.
- **Lunar blackout** (NASA's trajectory): NASA reported LOS 22:44 to AOS 23:24 UTC, "about 40 minutes";
  the geometric occultation of Earth's centre must last 40 +- 1 min and sit within 3 min of it (the
  station on Earth's disc and NASA's minute rounding).
- **Solar eclipse** (NASA's trajectory): at mid-eclipse Orion is 11,650 km from the Moon, whose disc
  (8.57 deg radius) is 32 times the Sun's (0.266 deg) in radius, so it covers the Sun for
  `2 (beta - alpha) / w`, `w` Orion's angular rate about the shadow axis: 54.4 min from 2026-04-07
  00:34:27 UTC, computed on 2026-10-04 by an independent scratch script (same DE441 tables, plain
  `arccos`); hold it to 0.2 min and 5 s. The penumbral phase adds `4 alpha / w`, ~4 min. Inside the
  windows `srp.shadow_factor` (separate code for the same disc geometry) must read exactly 0 (total)
  and below 1 (partial), and outside them 1 and above 0.
- **The free trajectory**: OTC-3 (3.017 m/s, 19.95 h before closest approach) displaces Orion at the
  Moon by about `|dv| t` = 217 km (182 km from its radial and normal parts alone), lunar focusing aside:
  hold the gap between flights with and without it to 150-300 km. The conic helpers are checked against
  `utilities.Anomalies` and the closed-form `tan(gamma) = e sin(theta) / (1 + e cos(theta))`, and the
  extrapolation of NASA's last sample against the event list's entry interface (23:53 UTC, given to the
  minute, so within 60 s).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from orbital_engine import artemis2 as a2
from orbital_engine import artemis2_replay as R
from orbital_engine import horizons_bridge as hb

BEST = R.TIERS[-1]
MOON_ONLY = R.TIERS[2]


def _err_end(flight: R.Flight) -> float:
    t, e = R.position_error(flight, R.truth_flight())
    assert t[-1] == flight.t_s[-1]
    return float(e[-1])


# --- seed and burns ----------------------------------------------------------------------------------

@pytest.mark.parametrize("tier", R.TIERS, ids=lambda t: t.model_id)
def test_seed_is_nasas_state_bit_for_bit(tier: R.Tier) -> None:
    sim, i, e = R.build_simulation(tier)
    orion = a2.load("orion")
    k = orion.at_tdb(R.SEED_TDB)
    assert np.array_equal(sim.global_states[i] - sim.global_states[e], orion.state[k])
    f = R.fly(tier, orion.t_s[k] + 60.0)
    assert np.array_equal(f.state[0], orion.state[k])


def _csv_burns_after_seed() -> List[Dict[str, str]]:
    with open(a2.DATA_DIR / "burns.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    t0 = a2.tdb_seconds(R.SEED_TDB)
    return [r for r in rows if r["kind"] == "burn" and a2.tdb_seconds(r["epoch_tdb"]) > t0]


def test_burns_are_burns_csv_exactly() -> None:
    rows = _csv_burns_after_seed()
    burns = R.replay_burns(reconstruct_family_switch=False)
    assert len(burns) == len(rows) == 7
    for b, r in zip(burns, rows):
        assert b.t_s == a2.tdb_seconds(r["epoch_tdb"])
        assert b.rsw_m_s == (float(r["dv_r_m_s"]), float(r["dv_s_m_s"]), float(r["dv_w_m_s"]))
        assert b.dv_m_s == float(r["dv_m_s"])
    keys = [b.key for b in burns]
    assert keys == ["unlisted_impulse", "unlisted_impulse", "outbound_correction_burn_3", "return_correction_burn_1",
                    "return_correction_burn_2", "return_correction_burn_3", "crew_module_raise_burn"]
    assert [b.reported_m_s for b in burns[2:]] == [3.0, 0.49, 1.62, 1.28, 3.0]


def test_family_switch_is_an_impulse_and_replaces_the_pair() -> None:
    t, rsw, closure = R.family_switch_impulse()
    dv = float(np.linalg.norm(rsw))
    assert 1.7 < dv < 2.0
    assert closure < 2.0  # an impulse; the 14:46 discontinuity alone closes to 86 km
    assert abs(t - a2.tdb_seconds("2026-04-05T01:25:00")) < 600.0
    burns = R.replay_burns()
    assert len(burns) == 6
    assert sum(b.source == "reconstructed" for b in burns) == 1
    assert not any(b.key == "unlisted_impulse" for b in burns)


# --- zero Delta-v ------------------------------------------------------------------------------------

def _zero(t_s: float) -> R.ReplayBurn:
    return R.ReplayBurn("zero", "zero", t_s, (0.0, 0.0, 0.0), 0.0, 0.0, math.nan, "", "test")


@pytest.mark.parametrize("tier", [R.TIERS[0], BEST], ids=lambda t: t.model_id)
def test_zero_delta_v_on_a_step_boundary_is_bit_identical(tier: R.Tier) -> None:
    t0 = a2.tdb_seconds(R.SEED_TDB)
    plain = R.fly(tier, t0 + 3600.0)
    zero = R.fly(tier, t0 + 3600.0, [_zero(t0 + 1800.0)])
    assert np.array_equal(plain.state, zero.state)


def test_zero_delta_v_off_a_boundary_costs_only_the_split() -> None:
    t0 = a2.tdb_seconds(R.SEED_TDB)
    plain = R.fly(BEST, t0 + 3600.0)
    zero = R.fly(BEST, t0 + 3600.0, [_zero(t0 + 1817.3)])
    d = float(np.max(np.linalg.norm(plain.state[:, :3] - zero.state[:, :3], axis=1)))
    assert 0.0 < d < 1e-6


# --- against independent code and against NASA --------------------------------------------------------

def test_first_arc_matches_the_ingest_coast_and_nasa() -> None:
    orion = a2.load("orion")
    k0, k1 = orion.at_tdb(R.SEED_TDB), orion.at_tdb("2026-04-03T15:18:00")
    flight = R.fly(BEST, float(orion.t_s[k1]))
    n = int(round((orion.t_s[k1] - orion.t_s[k0]) / 20.0))
    coast = a2.coast(a2._default_model(), orion.t_s[k0:k0 + 1], orion.state[k0:k0 + 1], orion.t_s[k1:k1 + 1], n)
    assert float(np.linalg.norm(flight.state[-1, :3] - coast[0, :3])) < 0.01
    assert _err_end(flight) < 0.1


def test_flyby_arc_matches_the_ingest_miss_and_needs_the_sun() -> None:
    t1 = a2.tdb_seconds("2026-04-06T17:34:00")
    best = R.fly(BEST, t1, seed_tdb="2026-04-06T03:07:00")
    no_sun = R.fly(MOON_ONLY, t1, seed_tdb="2026-04-06T03:07:00")
    assert 2.0 < _err_end(best) < 3.2
    assert 10.0 < _err_end(no_sun) < 60.0
    half = R.fly(BEST, t1, seed_tdb="2026-04-06T03:07:00", grid_s=30.0, max_turn=R.MAX_TURN_PER_STEP / 2)
    assert float(np.linalg.norm(half.state[-1, :3] - best.state[-1, :3])) < 1e-3


def test_otc3_arc_through_the_burn() -> None:
    """From the clean state before OTC-3 across it to the flyby-arc end: the flyby arc's 2.6 km plus the
    3 h pre-burn arc (0.06 km) and the burn's own error (~0.01 m/s x 14 h = 0.5 km): bound 4 km. A burn
    applied in the wrong frame (3 m/s misdirected) is ~100 km."""
    t1 = a2.tdb_seconds("2026-04-06T17:34:00")
    burns = [b for b in R.replay_burns() if b.key == "outbound_correction_burn_3"]
    flight = R.fly(BEST, t1, burns, seed_tdb="2026-04-06T00:03:00")
    assert _err_end(flight) < 4.0


def test_closest_approach_of_nasas_trajectory() -> None:
    ca = R.closest_lunar_approach(R.truth_flight())
    assert abs(ca.value_km - 8281.94) < 0.05
    utc = a2._utc_of_tdb(ca.t_s)
    assert abs(float((utc - np.datetime64("2026-04-06T23:01:00")).astype("timedelta64[ms]").astype(np.int64)) / 1e3) < 90.0


# --- the free trajectory --------------------------------------------------------------------------------

def test_without_corrections_drops_exactly_the_four_corrections() -> None:
    burns = R.replay_burns()
    keys = R.correction_keys(burns)
    assert keys == ["outbound_correction_burn_3", "return_correction_burn_1", "return_correction_burn_2",
                    "return_correction_burn_3"]
    free = R.without_corrections(burns)
    assert [b.key for b in free] == ["unlisted_velocity_change", "crew_module_raise_burn"]
    assert [b.key for b in R.without_corrections(burns, keys[:1])] == [
        "unlisted_velocity_change", "outbound_correction_burn_3", "crew_module_raise_burn"]


def test_conic_helpers_against_closed_forms() -> None:
    from orbital_engine import scenarios
    from orbital_engine.utilities import Anomalies

    mu, rp, ra = scenarios.MU_EARTH, 6400.0, 400000.0
    a, e = 0.5 * (rp + ra), (ra - rp) / (ra + rp)
    va = math.sqrt(mu * (2.0 / ra - 1.0 / a))
    apogee = np.array([-ra, 0.0, 0.0, 0.0, -va, 0.0])
    assert abs(R.vacuum_perigee_km(apogee) - (rp - R.WGS84_A_KM)) < 1e-6
    r = 6500.0
    out = R.conic_to_radius(apogee, r)
    assert out is not None
    theta = -math.acos((a * (1.0 - e * e) / r - 1.0) / e)          # inbound: negative true anomaly
    m = float(Anomalies.eccentric_to_mean(Anomalies.true_to_eccentric(theta, e), e)) % (2.0 * math.pi)
    n = math.sqrt(mu / a ** 3)
    assert abs(out[0] - (m - math.pi) / n) < 1e-6
    gamma = math.degrees(math.atan(e * math.sin(theta) / (1.0 + e * math.cos(theta))))
    assert abs(out[1] - gamma) < 1e-9 and out[1] < 0.0
    assert R.conic_to_radius(apogee, rp - 1.0) is None


def test_nasas_entry_from_its_last_sample() -> None:
    aim = R.entry_aim(R.truth_flight())
    assert aim.extrapolated and aim.entry_t_s is not None and aim.entry_fpa_deg is not None
    assert abs(aim.entry_t_s - a2.tdb_seconds(hb.utc_to_tdb("2026-04-10T23:53:00"))) < 60.0
    assert aim.entry_fpa_deg < 0.0 and aim.vacuum_perigee_km < R.EI_ALTITUDE_KM


def test_otc3_moves_the_flyby_by_its_lead_times_delta_v() -> None:
    burns = [b for b in R.replay_burns() if b.key == "outbound_correction_burn_3"]
    t_ca = R.closest_lunar_approach(R.truth_flight()).t_s
    seed = "2026-04-06T02:00:00"
    with_burn = R.fly(BEST, t_ca + 600.0, burns, seed_tdb=seed)
    without = R.fly(BEST, t_ca + 600.0, seed_tdb=seed)
    k = int(np.argmin(np.abs(with_burn.t_s - t_ca)))
    assert 150.0 < float(np.linalg.norm(with_burn.state[k, :3] - without.state[k, :3])) < 300.0


# --- visibility ---------------------------------------------------------------------------------------

def test_earth_rotation_reproduces_horizons() -> None:
    rot = R.earth_rotation()
    assert rot.station_error_km < 0.05
    assert abs(math.degrees(rot.theta0) % 360.0 - 219.8118) < 1e-3
    assert abs(rot.omega - 7.292115e-5) < 1e-10


def test_lunar_blackout_matches_nasas_loss_of_signal() -> None:
    truth = R.truth_flight()
    (w,) = R.lunar_blackouts(truth)
    assert abs((w.end_s - w.start_s) / 60.0 - 40.0) < 1.0
    los = a2.tdb_seconds(hb.utc_to_tdb("2026-04-06T22:44:00"))
    assert abs(w.start_s - los) < 180.0


def test_solar_eclipse_by_the_moon_on_nasas_trajectory() -> None:
    wins = [w for w in R.solar_eclipses(R.truth_flight()) if w.station == "Moon"]
    (tot,) = [w for w in wins if w.kind == "solar_eclipse"]
    (par,) = [w for w in wins if w.kind == "solar_eclipse_partial"]
    assert abs((tot.end_s - tot.start_s) / 60.0 - 54.43) < 0.2
    assert abs(tot.start_s - a2.tdb_seconds(hb.utc_to_tdb("2026-04-07T00:34:27"))) < 5.0
    assert par.start_s < tot.start_s and tot.end_s < par.end_s
    assert 3.0 < (par.end_s - par.start_s) / 60.0 - 54.43 < 5.0


def test_solar_eclipse_edges_agree_with_srp_shadow_factor() -> None:
    from orbital_engine.srp import SUN_RADIUS, shadow_factor

    truth = R.truth_flight()
    wins = R.solar_eclipses(truth)
    tab = truth.table()
    sun, moon = a2.load("sun"), a2.load("moon")
    for w in wins:
        for t, inside in ((w.start_s + 2.0, True), (w.end_s - 2.0, True), (w.start_s - 2.0, False),
                          (w.end_s + 2.0, False)):
            if not truth.t_s[0] < t < truth.t_s[-1]:
                continue
            tt = np.array([t])
            r = tab.position(tt)
            occ = a2.hermite_position(moon, tt) - r if w.station == "Moon" else -r
            rad = a2.MOON_MEAN_RADIUS_KM if w.station == "Moon" else R.WGS84_A_KM
            nu = float(shadow_factor(a2.hermite_position(sun, tt) - r, occ, np.array([SUN_RADIUS]),
                                     np.array([rad]), np.array([True]))[0])
            if w.kind == "solar_eclipse":
                assert (nu == 0.0) == inside
            else:
                assert (nu < 1.0) == inside
    assert {(w.station, w.kind) for w in wins} == {(o, k) for o in ("Earth", "Moon")
                                                    for k in ("solar_eclipse", "solar_eclipse_partial")}


def test_dsn_covers_orion_outside_the_blackout() -> None:
    """At the 6 deg mechanical limit the three sites hand over with at most half-hour gaps mid-mission
    (Orion sits at -25..-30 deg declination, so the northern sites see it low); at 10 deg each site
    recurs once a sidereal-ish day and the daily gaps grow to about an hour."""
    truth = R.truth_flight()
    rot = R.earth_rotation()
    t0, t1 = a2.tdb_seconds("2026-04-03T00:00:00"), a2.tdb_seconds("2026-04-09T00:00:00")
    black = R.lunar_blackouts(truth)[0]
    for mask, limit in ((6.0, 30.0), (10.0, 90.0)):
        wins = R.dsn_windows(truth, rot, mask_deg=mask)
        for a, b in R.contact_gaps(wins, t0, t1):
            if b <= black.start_s or a >= black.end_s:
                assert (b - a) / 60.0 < limit
    wins = R.dsn_windows(truth, rot)
    for st in R.DSN_STATIONS:
        rises = np.array([w.start_s for w in wins if w.station == st.name and t0 < w.start_s < t1])
        spacing = np.diff(rises) / 3600.0
        long = spacing[spacing > 18.0]  # the blackout splits one Canberra pass in two
        assert rises.size >= 5 and np.all(np.abs(long - 24.0) < 1.5)


# --- export -------------------------------------------------------------------------------------------

def test_export_follows_the_contract(tmp_path: Path) -> None:
    t0 = a2.tdb_seconds(R.SEED_TDB)
    truth = R.truth_flight()
    rot = R.earth_rotation()
    tiers = []
    for tier in (R.TIERS[0], BEST):
        f = R.fly(tier, t0 + 7200.0)
        tiers.append(R.TierResult(tier, f, [R.fly(tier, t0 + 3600.0)], R.closest_lunar_approach(f),
                                  R.max_earth_distance(f), None, None, R.dsn_windows(f, rot)))
    rep = R.Replay(truth, tiers, R.replay_burns(reconstruct_family_switch=False), rot,
                   R.closest_lunar_approach(truth), R.max_earth_distance(truth), R.dsn_windows(truth, rot))
    sizes = R.export_dashboard(rep, str(tmp_path), generated_utc="2026-01-01T00:00:00Z", engine_commit="test",
                               engine_version=None, notes=["n"])
    assert set(sizes) == {"meta.json", "models.csv", "trajectory.csv", "metrics.csv", "events.csv", "windows.csv"}
    import json
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["synthetic"] is False and meta["epoch_utc"] == "2026-04-01T22:35:12.000Z"
    models = list(csv.DictReader(open(tmp_path / "models.csv", encoding="utf-8")))
    assert [m["is_truth"] for m in models].count("1") == 1
    ids = {m["model_id"] for m in models}
    numeric = {"trajectory.csv": ("t_s", "x_km", "y_km", "z_km"), "metrics.csv": ("t_s", "value"),
               "events.csv": ("t_s",), "windows.csv": ("start_s", "end_s")}
    for name, cols in numeric.items():
        for row in csv.DictReader(open(tmp_path / name, encoding="utf-8")):
            assert row["model_id"] in ids
            assert all(math.isfinite(float(row[c])) for c in cols)
    last: Dict[str, float] = {}
    for row in csv.DictReader(open(tmp_path / "trajectory.csv", encoding="utf-8")):
        key = row["model_id"] + row["body"]
        assert float(row["t_s"]) > last.get(key, -math.inf)
        last[key] = float(row["t_s"])
    kinds = {row["kind"] for row in csv.DictReader(open(tmp_path / "events.csv", encoding="utf-8"))}
    assert kinds <= {"burn", "apsis", "milestone"}
    metrics = {row["metric"] for row in csv.DictReader(open(tmp_path / "metrics.csv", encoding="utf-8"))}
    assert metrics == {"position_error_km", "arc_error_km", "earth_range_km", "moon_range_km", "speed_km_s",
                       "delta_v_m_s"}
    # t_s counts from launch: the seed is 1 d 2 h 23 min 39.8 s after it.
    seed = next(r for r in csv.DictReader(open(tmp_path / "events.csv", encoding="utf-8")) if r["event"] == "model_seed")
    assert abs(float(seed["t_s"]) - (a2.tdb_seconds(R.SEED_TDB) + R.EXPORT_OFFSET_S)) < 1e-3
    assert abs(float(seed["t_s"]) - 95018.8) < 0.1
