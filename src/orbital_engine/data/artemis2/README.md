# Artemis II replay data

Real trajectories for replaying Artemis II through the engine's model tiers. Everything here was
downloaded **once** from JPL Horizons by `scripts/fetch_artemis2.py` and is read offline by
`orbital_engine.horizons_bridge.load_cached` / `orbital_engine.artemis2`. Nothing in the engine or
the test suite touches the network.

## Provenance

| | |
|---|---|
| Source | NASA/JPL Horizons, API `https://ssd.jpl.nasa.gov/api/horizons.api` (API version 1.2) |
| Retrieved | 2026-09-27T07:04:29Z (UTC); `earth_sites.npz` 2026-09-27 ~11:23Z. Every URL queried is in `retrieval.json` |
| Orion | Horizons object **-1024**, "Artemis II (spacecraft)", `{source: Artemis_II_merged}`; object data "Revised: Apr 20, 2026". "Post-launch Orion II trajectory data from NASA/JSC navigation (concatenated)": 14 source files (OEMs from NASA/JSC and two JPL OD solutions), listed with their TDB spans in `orion_object_data.txt` |
| Moon, Sun | Horizons 301 and 10, `{source: DE441}` |
| Earth orientation | Horizons' EOP file `eop.260925.p261222` (data-based to 2026-09-25, so the whole mission is on measured EOP), pole/equator model ITRF93 |
| Query (Orion) | `COMMAND='-1024' MAKE_EPHEM='YES' EPHEM_TYPE='VECTORS' CENTER='500@399' START_TIME='2026-04-02 02:00' STOP_TIME='2026-04-10 23:54' STEP_SIZE='1m' VEC_TABLE='2' REF_PLANE='FRAME' REF_SYSTEM='ICRF' OUT_UNITS='KM-S' CSV_FORMAT='YES' VEC_LABELS='NO'` (built by `horizons_bridge.build_query`) |

## Units, frame, time scale

- **km, km/s**; geometric states (no light time, no aberration).
- **Earth-centred ICRF** (`500@399`, `REF_PLANE='FRAME'`). ICRF +z is **not** the Earth's spin axis in
  2026: the true pole is 0.1469 deg away (mean pole 0.1462 deg, IAU 1976 precession to 0.26").
- **TDB.** Every table's `t_s` is seconds from **2026-04-02T02:00:00 TDB** (`artemis2.EPOCH_TDB`),
  exact to 0.1 ms from the calendar column; `jd_tdb` is Horizons' printed Julian date.
  `TDB - UTC` = 69.184 s + (TDB - TT) = **69.1856 s** during the mission (Horizons' own `TDB-UT`
  column, `time_check.txt`: 69.185633-69.185635 s).

## Files

| File | What | Size |
|---|---|---|
| `orion.npz` | -1024, **1 min**, 2026-04-02 02:00 -> 2026-04-10 23:54 TDB, 12,835 states. The file itself covers 01:58:32.305 -> 23:54:22.858 TDB (Horizons' "No ephemeris prior to / after" limits) | 625 kB |
| `moon.npz`, `sun.npz` | 301 / 10, **10 min**, 02:00 Apr 2 -> 00:00 Apr 11 TDB, 1,285 states each | 67 / 63 kB |
| `orion_object_data.txt` | raw object-data response: MAJOR EVENTS list and trajectory-file table | 8 kB |
| `raw_orion_sample_csv.txt`, `raw_orion_sample_text.txt` | raw responses for 02:00-02:10 TDB in both layouts - the parser's test input | 7 kB each |
| `frame_check.npz` | Moon, Sun, Jupiter at 12 h in ICRF and in Horizons' Earth "BODY EQUATOR" frame (mean equator and node of date - it does **not** rotate; its header's "ITRF93" names the pole model) | 4 kB |
| `earth_sites.npz` | Earth-fixed geodetic sites (0E 0N), (90E 0N), (north pole) as ICRF vectors every 6 h from the epoch - the full ICRF -> ITRF93 rotation | 3 kB |
| `time_check.txt` | raw observer table at Greenwich, 6 h: apparent sidereal time, `TDB - UT` | 9 kB |
| `events.csv` | the MAJOR EVENTS list parsed (`artemis2.parse_major_events`): name, MET, UTC, TDB, stated delta-v and duration, cancelled / in-span / burn flags | derived |
| `burns.csv` | discontinuities detected in the Orion data (`artemis2.detect_burns`, matched by `match_events`) | derived |
| `retrieval.json` | every query URL and the retrieval time | |

## Why these steps

- **Orion at 1 min.** Horizons interpolates the JSC ephemeris; across a burn the source does not
  segment, the interpolant **rings** (+-6 m/s^2 for minutes around TLI, measured at 5 s), so a finer
  grid adds no information about a burn. A burn's Delta-v is an integral between clean samples, which
  1 min resolves (TLI 388.6 m/s delivered against NASA's 388.3 m/s). At the flyby a 1-min minimum is
  within 0.071 km of the true one before Hermite refinement.
- **Moon and Sun at 10 min.** Cubic Hermite on position and velocity: truncation `h^4 n^4 r / 384` =
  6e-9 km for the Moon (1.0e-7 at 20 min, 9.1e-8 measured by decimation); the Sun sits at the table's
  own print floor (~1e-7 km).

## What is in the data that the event list does not say

- **File joins.** The 13 joins between source files are position discontinuities of up to ~12 km
  (04-06 17:35 and 04-09 09:24 TDB); every one is flagged.
- **Unlisted discontinuities** (closure times >= 1265 s, i.e. not impulses): e.g. 2026-04-05 14:46-15:16
  TDB, 86 km of position closure with per-minute interpolant swings to 1.9 km/s. Treat the data inside
  `burns.csv`'s `discontinuity` clusters as unreliable at the km level.
- **Unlisted impulses**: 1.10 and 1.99 m/s during proximity operations at the start of the data, and
  0.164 / 0.120 m/s at 01:24:00 / 01:40:00 UTC on 2026-04-05 (on whole UTC minutes, inside a JPL OD
  file - modelled impulses of that solution, not crew burns as far as the list says).
- The list's perigee raise burn time (11:30 UTC) is a planning value; the data put it at 12:08:14 UTC.
