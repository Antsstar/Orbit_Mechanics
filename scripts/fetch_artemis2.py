"""
Download the Artemis II replay data set from JPL Horizons into `src/orbital_engine/data/artemis2/`.

This is the **only** place in the repository that talks to the network (through
`horizons_bridge.fetch`). The engine and the tests read the committed copies. Run from the repository
root with the project environment:

    C:/Users/antss/miniconda3/envs/orbital_env/python.exe scripts/fetch_artemis2.py

`--derive-only` skips the download and only regenerates the derived tables (`events.csv`,
`burns.csv`) from the files already on disk - that path is offline.

What it writes, and why each grid is what it is, is in the data folder's `README.md`. In short:

- `orion.npz`  Horizons -1024, Earth-centred ICRF, **1 min** over the whole trajectory file span.
- `moon.npz`, `sun.npz`  301 and 10, Earth-centred ICRF, **10 min** over a span that covers Orion's.
- `orion_object_data.txt`  the raw object-data response (MAJOR EVENTS, trajectory file list).
- `raw_orion_sample_csv.txt`, `raw_orion_sample_text.txt`  two raw responses for the first ten
  minutes, CSV and text layout, kept so the parser is tested on Horizons' own bytes.
- `frame_check.npz`  Moon, Sun and Jupiter positions in ICRF **and** in Horizons' Earth
  "BODY EQUATOR" frame (the Earth's *mean equator and node of date* - not rotating, despite the
  header's "ITRF93", which names the pole model) at 12 h spacing: the rotation between them is the
  mean pole of date, checked against IAU 1976 precession (`artemis2.mean_pole_of_date`).
- `earth_sites.npz`  three Earth-fixed geodetic sites (0 E 0 N, 90 E 0 N, the north pole; ITRF93)
  as ICRF vectors every 6 h from the epoch: the full ICRF -> ITRF93 rotation - true pole and prime
  meridian (`artemis2.earth_orientation`).
- `time_check.txt`  a raw observer table at Greenwich: apparent sidereal time and `TDB - UT`.
- `events.csv`, `burns.csv`  derived (`artemis2.parse_major_events`, `artemis2.detect_burns`).
- `retrieval.json`  every URL queried and the UTC time of retrieval.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbital_engine import artemis2, horizons_bridge as hb  # noqa: E402

OUT = hb.DATA_DIR / "artemis2"

# Horizons reports the -1024 file as covering 2026-04-02 01:58:32.3050 to 2026-04-10 23:54:22.8576
# TDB (its own "No ephemeris ... prior to / after" messages). The grids sit on whole minutes inside it.
ORION_START, ORION_STOP, ORION_STEP = "2026-04-02 02:00", "2026-04-10 23:54", "1m"
BODY_START, BODY_STOP, BODY_STEP = "2026-04-02 02:00", "2026-04-11 00:00", "10m"
FRAME_START, FRAME_STOP, FRAME_STEP = "2026-04-02 00:00", "2026-04-11 00:00", "12h"
SITES_START, SITES_STOP, SITES_STEP = "2026-04-02 02:00", "2026-04-11 02:00", "6h"
SAMPLE_STOP = "2026-04-02 02:10"


def _get(params: Dict[str, str], log: List[Dict[str, str]], what: str) -> str:
    url, text = hb.fetch(params)
    log.append({"what": what, "url": url})
    print(f"  {what}: {len(text):,} bytes")
    return text


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log: List[Dict[str, str]] = []
    retrieved = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    text = _get(hb.build_query("-1024", ephemeris=False, obj_data=True), log, "object data")
    (OUT / "orion_object_data.txt").write_bytes((text).encode("utf-8"))

    epoch = artemis2.EPOCH_TDB
    for name, command, start, stop, step in [
        ("orion", "-1024", ORION_START, ORION_STOP, ORION_STEP),
        ("moon", "301", BODY_START, BODY_STOP, BODY_STEP),
        ("sun", "10", BODY_START, BODY_STOP, BODY_STEP),
    ]:
        raw = _get(hb.build_query(command, start=start, stop=stop, step=step), log, name)
        vec = hb.parse_vectors(raw, epoch_tdb=epoch)
        hb.save_npz(vec, OUT / f"{name}.npz")

    csv = _get(hb.build_query("-1024", start=ORION_START, stop=SAMPLE_STOP, step="1m"), log, "sample csv")
    (OUT / "raw_orion_sample_csv.txt").write_bytes((csv).encode("utf-8"))
    txt = _get(hb.build_query("-1024", start=ORION_START, stop=SAMPLE_STOP, step="1m",
                              extra={"CSV_FORMAT": "'NO'"}), log, "sample text")
    (OUT / "raw_orion_sample_text.txt").write_bytes((txt).encode("utf-8"))

    # Frame check: the same three bodies in ICRF and in the Earth mean-equator-of-date frame.
    icrf: List[np.ndarray] = []
    itrf: List[np.ndarray] = []
    jd = None
    for command in artemis2.FRAME_CHECK_BODIES:
        a = hb.parse_vectors(_get(hb.build_query(command, start=FRAME_START, stop=FRAME_STOP, step=FRAME_STEP),
                                  log, f"frame {command} ICRF"), epoch_tdb=epoch)
        b = hb.parse_vectors(_get(hb.build_query(command, start=FRAME_START, stop=FRAME_STOP, step=FRAME_STEP,
                                                 extra={"REF_PLANE": "'BODY EQUATOR'"}),
                                  log, f"frame {command} body equator"), epoch_tdb=epoch)
        icrf.append(a.position_km)
        itrf.append(b.position_km)
        jd = a.jd_tdb
    np.savez_compressed(
        OUT / "frame_check.npz", jd_tdb=np.asarray(jd), bodies=np.asarray(artemis2.FRAME_CHECK_BODIES),
        icrf_km=np.stack(icrf, axis=1), itrf93_km=np.stack(itrf, axis=1),
    )
    download_earth_sites(log)

    obs = {
        "format": "text", "COMMAND": "'10'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'000@399'",
        "START_TIME": "'2026-04-02 00:00 UT'", "STOP_TIME": "'2026-04-11 00:00'", "STEP_SIZE": "'6h'",
        "QUANTITIES": "'7,30'", "CSV_FORMAT": "'YES'", "ANG_FORMAT": "'DEG'", "EXTRA_PREC": "'YES'",
    }
    (OUT / "time_check.txt").write_bytes((_get(obs, log, "time check")).encode("utf-8"))

    _write_json({"retrieved_utc": retrieved, "queries": log})


def download_earth_sites(log: List[Dict[str, str]]) -> None:
    """`earth_sites.npz`: Earth's centre seen from three Earth-fixed geodetic sites, negated."""
    sites: List[np.ndarray] = []
    jd = None
    for coord in artemis2.EARTH_SITE_COORDS:
        v = hb.parse_vectors(_get(hb.build_query("399", center="coord@399", start=SITES_START, stop=SITES_STOP,
                                                 step=SITES_STEP,
                                                 extra={"COORD_TYPE": "'GEODETIC'", "SITE_COORD": f"'{coord}'"}),
                                  log, f"earth site {coord}"), epoch_tdb=artemis2.EPOCH_TDB)
        sites.append(-v.position_km)
        jd = v.jd_tdb
    np.savez_compressed(OUT / "earth_sites.npz", jd_tdb=np.asarray(jd), coords=np.asarray(artemis2.EARTH_SITE_COORDS),
                        site_icrf_km=np.stack(sites, axis=1))


def derive() -> None:
    events = artemis2.parse_major_events((OUT / "orion_object_data.txt").read_text(encoding="utf-8"))
    artemis2.write_events_csv(events, OUT / "events.csv")
    burns = artemis2.detect_burns()
    artemis2.write_burns_csv(burns, OUT / "burns.csv")
    print(f"  events.csv: {len(events)} rows; burns.csv: {len(burns)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--derive-only", action="store_true", help="skip the download (offline)")
    parser.add_argument("--earth-sites-only", action="store_true",
                        help="fetch only earth_sites.npz, appending its queries to retrieval.json")
    args = parser.parse_args()
    if args.earth_sites_only:
        log: List[Dict[str, str]] = []
        download_earth_sites(log)
        rec = json.loads((OUT / "retrieval.json").read_text(encoding="utf-8"))
        rec.setdefault("later", []).append({
            "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "queries": log})
        _write_json(rec)
        return
    if not args.derive_only:
        print("downloading from JPL Horizons ...")
        download()
    print("deriving tables ...")
    derive()


def _write_json(rec: Dict[str, object]) -> None:
    (OUT / "retrieval.json").write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
