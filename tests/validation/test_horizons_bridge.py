"""
`horizons_bridge.py`: the parser, the store and the time conversions, against the committed Artemis II
data set - offline. Nothing here touches the network; `test_no_network_*` asserts that.

Expected magnitudes
-------------------
- Parsing is exact: the committed raw Horizons responses (CSV and text layout) reproduce the first
  eleven rows of `orion.npz` **bit for bit**, and `format_vectors_csv` -> `parse_vectors` round-trips
  the whole table bit for bit (17 significant digits).
- The printed Julian date and the calendar column agree to Horizons' 9 decimals plus one float64 ulp
  (<= 1e-9 d).
- `TDB - UTC`: 69.184 s + the 1.66 ms periodic term. Against Horizons' own `TDB - UT` column
  (`time_check.txt`, 1e-6 s printed) the two-term series agrees to **2.3e-5 s** measured - the terms it
  drops; bound 5e-5 s.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orbital_engine import artemis2, horizons_bridge as hb

DATA = hb.DATA_DIR / "artemis2"


@pytest.fixture(scope="module")
def orion() -> hb.HorizonsVectors:
    return hb.load_cached("orion")


# --- parsing ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["raw_orion_sample_csv.txt", "raw_orion_sample_text.txt"])
def test_raw_horizons_bytes_parse_to_the_committed_rows(orion: hb.HorizonsVectors, raw: str) -> None:
    v = hb.parse_vectors((DATA / raw).read_text(encoding="utf-8"), epoch_tdb=orion.epoch_tdb)
    n = v.t_s.size
    assert n == 11
    assert v.target == "Artemis II (spacecraft) (-1024)"
    assert v.center == "Earth (399)"
    assert (v.frame, v.time_scale) == ("ICRF", "TDB")
    for field in ("t_s", "jd_tdb", "position_km", "velocity_km_s"):
        assert np.array_equal(getattr(v, field), getattr(orion, field)[:n]), field


def test_csv_round_trip_is_bit_identical(orion: hb.HorizonsVectors) -> None:
    for table in (orion, hb.load_cached("moon")):
        back = hb.parse_vectors(hb.format_vectors_csv(table), epoch_tdb=table.epoch_tdb)
        for field in ("t_s", "jd_tdb", "position_km", "velocity_km_s"):
            assert np.array_equal(getattr(back, field), getattr(table, field)), field
        assert (back.target, back.center, back.frame, back.header) == (table.target, table.center, table.frame,
                                                                        table.header)


def test_npz_round_trip(orion: hb.HorizonsVectors, tmp_path: Path) -> None:
    hb.save_npz(orion, tmp_path / "x.npz")
    back = hb.load_npz(tmp_path / "x.npz")
    assert back.epoch_tdb == orion.epoch_tdb and back.header == orion.header
    assert np.array_equal(back.state, orion.state) and np.array_equal(back.t_s, orion.t_s)


def test_horizons_error_text_is_refused() -> None:
    err = "API VERSION: 1.2\nAPI SOURCE: NASA/JPL Horizons API\n\nNo ephemeris for target \"x\" after A.D. 2026\n"
    with pytest.raises(hb.HorizonsError, match="No ephemeris"):
        hb.parse_vectors(err)
    good = (DATA / "raw_orion_sample_csv.txt").read_text(encoding="utf-8")
    with pytest.raises(hb.HorizonsError, match="KM-S"):
        hb.parse_vectors(good.replace("Output units    : KM-S", "Output units    : AU-D"))


def test_build_query_is_the_documented_query() -> None:
    q = hb.build_query("-1024", start="2026-04-02 02:00", stop="2026-04-10 23:54", step="1m")
    assert q["COMMAND"] == "'-1024'" and q["CENTER"] == "'500@399'"
    assert (q["REF_SYSTEM"], q["REF_PLANE"], q["OUT_UNITS"], q["VEC_TABLE"]) == ("'ICRF'", "'FRAME'", "'KM-S'", "'2'")


# --- spans and sampling -------------------------------------------------------------------------------

def test_orion_grid_is_whole_minutes_inside_the_file_span(orion: hb.HorizonsVectors) -> None:
    assert orion.epoch_tdb.startswith(artemis2.EPOCH_TDB)
    assert orion.t_s.size == 12835
    assert np.all(np.diff(orion.t_s) == 60.0)
    assert orion.t_s[0] == 0.0
    assert artemis2.tdb_instant(float(orion.t_s[-1])) == np.datetime64("2026-04-10T23:54:00", "ns")
    # Horizons' stated coverage of -1024: 2026-04-02 01:58:32.3050 to 2026-04-10 23:54:22.8576 TDB.
    assert artemis2.tdb_seconds("2026-04-02T01:58:32.305") < orion.t_s[0]
    assert orion.t_s[-1] < artemis2.tdb_seconds("2026-04-10T23:54:22.8576")
    jd_from_t = hb.julian_date(artemis2.EPOCH_TDB) + orion.t_s / 86400.0
    # 9 printed decimals (+-5e-10 d) plus one float64 ulp at JD 2.46e6 (4.7e-10 d); measured 9.3e-10.
    assert np.max(np.abs(orion.jd_tdb - jd_from_t)) < 1.0e-9


def test_moon_and_sun_cover_orion_at_ten_minutes(orion: hb.HorizonsVectors) -> None:
    for name, target in (("moon", "Moon (301)"), ("sun", "Sun (10)")):
        t = hb.load_cached(name)
        assert t.target == target and t.center == "Earth (399)" and t.frame == "ICRF"
        assert np.all(np.diff(t.t_s) == 600.0)
        assert t.t_s[0] <= orion.t_s[0] and t.t_s[-1] >= orion.t_s[-1]
        assert "DE441" in t.header


# --- time scales -------------------------------------------------------------------------------------

def test_tdb_minus_utc_against_horizons_own_column() -> None:
    when, _, tdb_ut = artemis2.greenwich_apparent_sidereal_hours()
    ours = np.array([hb.tdb_minus_utc_s(w) for w in when])
    assert np.max(np.abs(ours - tdb_ut)) < 5e-5
    assert abs(hb.tdb_minus_utc_s("2026-04-02T02:00:00") - hb.TT_MINUS_UTC_S) < 1.7e-3


def test_utc_to_tdb_is_a_69_186_s_shift() -> None:
    tdb = hb.utc_to_tdb("2026-04-06T23:00:00")
    shift = float((tdb - np.datetime64("2026-04-06T23:00:00", "ns")).astype(np.int64)) / 1e9
    assert 69.1845 < shift < 69.1860
    with pytest.raises(ValueError):
        hb.tdb_minus_utc_s("2016-06-01T00:00:00")


# --- no network ----------------------------------------------------------------------------------------

_BLOCKED = """
import socket, sys
def _refuse(*a, **k):
    raise AssertionError("network access attempted")
socket.socket.connect = _refuse  # type: ignore[method-assign]
socket.socket.connect_ex = _refuse  # type: ignore[method-assign]
socket.create_connection = _refuse  # type: ignore[assignment]
sys.path.insert(0, {src!r})
import orbital_engine
from orbital_engine import horizons_bridge as hb, artemis2, scenarios
assert orbital_engine.__file__.startswith({src!r}), orbital_engine.__file__
assert "urllib.request" not in sys.modules, "urllib.request imported at import time"
for name in ("orion", "moon", "sun"):
    hb.load_cached(name)
artemis2.load_events(); artemis2.trajectory_files(); artemis2.earth_orientation(); artemis2.mean_pole_of_date()
artemis2.closest_lunar_approach()
assert "urllib.request" not in sys.modules
print("offline-ok")
"""


def test_no_network_at_import_or_load() -> None:
    src = str(Path(artemis2.__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", _BLOCKED.format(src=src)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "offline-ok" in out.stdout


def test_fetch_is_the_only_network_entry_point() -> None:
    text = Path(hb.__file__).read_text(encoding="utf-8")
    assert text.count("import urllib") == 2  # `urllib.parse` and `urllib.request`, both inside `fetch`
    body = text.split("def fetch(", 1)[1].split("\ndef ", 1)[0]
    assert "import urllib.request" in body and "import urllib.parse" in body
