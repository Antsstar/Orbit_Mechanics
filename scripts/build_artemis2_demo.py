"""
Build the Artemis II dashboard's real data set: fly every model tier through NASA's burns, score it
against NASA's navigation trajectory, and write `demo/artemis2/data/` per `demo/artemis2/SCHEMA.md`.

    python scripts/build_artemis2_demo.py                 # ~3 min: replay, arcs, windows, export
    python scripts/build_artemis2_demo.py --convergence   # + the best tier at half the step (~1 min)

All the physics and every definition live in `orbital_engine.artemis2_replay` (its docstring carries
the pre-run estimates this summary is printed next to). This script only runs it, prints the summary
table and writes the files. Offline: it reads the committed Horizons data set only.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from orbital_engine import artemis2 as a2  # noqa: E402
from orbital_engine import horizons_bridge as hb  # noqa: E402
from orbital_engine import artemis2_replay as R  # noqa: E402

#: The estimates written into `artemis2_replay`'s docstring before the first full run.
ESTIMATES = {
    "earth": "CA err 5e3-1.6e4 km, CA alt ~11,700 km, end ~4e5 km, no entry (perigee ~196 km, ~15 Apr)",
    "earth_j2": "as Earth only, to tens of km",
    "earth_moon": "CA err ~750 km, CA alt off by hundreds of km / minutes, end ~1e4 km, entry probably missed",
    "earth_moon_sun": "arcs 0.5 / 2.6 km; replay tens of km at CA, hundreds by entry, entry within minutes",
}


def _utc(t_s: float) -> str:
    return str(np.datetime_as_string(a2._utc_of_tdb(t_s), unit="s")).replace("T", " ")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("orbital_engine")
    except Exception:  # noqa: BLE001 - not installed: the pyproject version is the fallback
        for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def _err_at(flight: R.Flight, truth: R.Flight, t: float) -> float:
    ts, err = R.position_error(flight, truth)
    return float(err[int(np.argmin(np.abs(ts - t)))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "demo" / "artemis2" / "data"))
    ap.add_argument("--convergence", action="store_true", help="also fly the best tier at a 30 s grid")
    args = ap.parse_args()

    rep = R.run_replay()
    truth = rep.truth
    t_ca_nasa = a2.tdb_seconds(hb.utc_to_tdb("2026-04-06T23:01:00"))
    t_ei_nasa = a2.tdb_seconds(hb.utc_to_tdb("2026-04-10T23:53:00"))
    switch = next(b for b in rep.burns if b.source == "reconstructed")
    black_t = [w for w in rep.truth_windows if w.kind == "lunar_blackout"]

    print("\nArtemis II replay - seed", R.SEED_TDB, "TDB; burns applied:")
    for b in rep.burns:
        print(f"  {_utc(b.t_s)} UTC  {b.key:28s} {b.dv_m_s:7.3f} m/s  RSW {np.round(b.rsw_m_s, 3)}  ({b.source})")
    print(f"\nNASA (truth): closest approach {rep.truth_closest.value_km:,.2f} km from the Moon's centre = "
          f"{rep.truth_closest.value_km - a2.MOON_MEAN_RADIUS_KM:,.1f} km altitude at {_utc(rep.truth_closest.t_s)} UTC "
          f"(reported 6,545 km, 23:01 UTC); farthest {rep.truth_farthest.value_km:,.1f} km from the centre at "
          f"{_utc(rep.truth_farthest.t_s)} UTC (reported 413,146.2 km centre / 406,771 km surface)")
    for w in black_t:
        print(f"  lunar blackout {_utc(w.start_s)} - {_utc(w.end_s)} UTC, {(w.end_s - w.start_s) / 60:.1f} min "
              "(NASA: LOS 22:44, AOS 23:24 UTC, 'about 40 minutes')")
    gaps = R.contact_gaps(rep.truth_windows, float(truth.t_s[0]), float(truth.t_s[-1]))
    print(f"  DSN gaps at {R.DSN_MASK_DEG:.0f} deg (no station in contact): "
          + ", ".join(f"{_utc(a)[5:16]} {(b - a) / 60:.0f} min" for a, b in gaps))
    print(f"  Earth rotation fit: theta0 {math.degrees(rep.rotation.theta0) % 360:.4f} deg at EPOCH_TDB, "
          f"omega {rep.rotation.omega:.10e} rad/s, residual rms {rep.rotation.rms_rad:.1e} rad, station error "
          f"<= {rep.rotation.station_error_km * 1e3:.0f} m")

    hdr = ("tier", "final err km", "err@CA km", "CA alt km (d NASA)", "CA time (d NASA)", "max dist km (d)",
           "entry / return", "DSN G/M/C", "blackout min")
    print("\n" + " | ".join(hdr))
    for r in rep.tiers:
        ts, err = R.position_error(r.flight, truth)
        alt = r.closest.value_km - a2.MOON_MEAN_RADIUS_KM
        if r.entry is not None:
            back = f"EI {_utc(r.entry.t_s)[5:19]} ({(r.entry.t_s - t_ei_nasa) / 60:+.1f} min)"
        elif r.perigee is not None:
            back = f"perigee {r.perigee.value_km - R.WGS84_A_KM:,.0f} km {_utc(r.perigee.t_s)[5:16]}"
        else:
            back = "no entry before the tables end"
        dsn = "/".join(str(sum(1 for w in r.windows if w.station == s.name)) for s in R.DSN_STATIONS)
        bl = [w for w in r.windows if w.kind == "lunar_blackout"]
        print(" | ".join([
            r.tier.model_id, f"{err[-1]:,.2f}", f"{_err_at(r.flight, truth, rep.truth_closest.t_s):,.2f}",
            f"{alt:,.1f} ({alt - 6545.0:+,.1f})", f"{_utc(r.closest.t_s)[11:19]} ({(r.closest.t_s - t_ca_nasa) / 60:+.1f} min)",
            f"{r.farthest.value_km:,.0f} ({r.farthest.value_km - 413146.2:+,.0f})", back, dsn,
            ", ".join(f"{(w.end_s - w.start_s) / 60:.1f} from {_utc(w.start_s)[5:16]}" for w in bl) or "none"]))
        print("   estimate:", ESTIMATES[r.tier.model_id])
        arc_end = []
        for a in r.arcs:
            _, e = R.position_error(a, truth)
            arc_end.append(float(e[-1]))
        if arc_end:
            print(f"   arcs: {len(arc_end)}, end error median {np.median(arc_end):.3f} km, max {max(arc_end):.3f} km")

    best = R.TIERS[-1]
    plain = R.fly(best, R._end_for(best), R.replay_burns(reconstruct_family_switch=False))
    ent = R.entry_interface(plain)
    print(f"\nWithout the reconstructed 5 April velocity change ({best.label}): error at CA "
          f"{_err_at(plain, truth, rep.truth_closest.t_s):,.1f} km, at the data end "
          f"{R.position_error(plain, truth)[1][-1]:,.0f} km, entry {'none' if ent is None else _utc(ent.t_s)}")
    notes: List[str] = [
        "Every model starts from NASA's exact position and velocity at 2026-04-03 00:58:51 UTC (01:00 TDB), an hour "
        "after translunar injection, and receives the same impulsive burns NASA flew afterwards, each at its measured "
        "epoch and in its measured local orbital (RSW) frame. The burns were detected in NASA's trajectory itself.",
        "One velocity change is not in NASA's event list: on 5 April NASA's navigation data switch between two "
        f"solutions that differ by {switch.dv_m_s:.2f} m/s. Everything from the flyby on lies on the second, "
        "which is the first plus one impulse at about 01:24 UTC, so every model receives that impulse (reconstructed "
        "by OrbitalEngine). Without it the best model is "
        f"{_err_at(plain, truth, rep.truth_closest.t_s):,.0f} km off at the flyby instead of "
        f"{_err_at(rep.tiers[-1].flight, truth, rep.truth_closest.t_s):,.0f} km.",
        "NASA's trajectory is 14 navigation files joined end to end. Where one hands over to the next, the trajectory "
        "jumps (up to ~12 km, and ~90 km inside one file on 5 April); these are marked 'Navigation data jump' and every "
        "model's error steps there. They are artefacts of the data, not of any model, and are not smoothed.",
        "arc_error_km: each model restarted from NASA's state after every burn and every data jump, and flown to the "
        "next - the model's own error over each coast, free of the data's jumps.",
        "Deep Space Network contact: elevation above 10 deg at Goldstone DSS-14, Madrid DSS-63 and Canberra DSS-43 "
        "(WGS-84 coordinates from DSN document 810-005, module 301) and the Moon not in the way. 10 deg is the DSN 70-m "
        "transmit limit (10.2-10.4 deg); with the ~6 deg mechanical limit most of the daily gaps close. Earth rotation "
        "from JPL Horizons' own Earth orientation (true pole, fitted rotation).",
        "Lunar blackout: the Moon blocks the line from Orion to Earth's centre (geometry, no signal margins). NASA "
        "reported loss of signal from 22:44 to 23:24 UTC, about 40 minutes (NASA Artemis II blog, flight day 6).",
        "Closest approach is altitude above the Moon's mean radius (1,737.4 km). Farthest distance is from Earth's "
        "centre for every row, NASA's included (event list, 413,146.2 km); NASA's public 252,756 mi (406,771 km) is from "
        "the surface. Entry interface: 121.92 km (400,000 ft) above the WGS-84 ellipsoid.",
        "Models with the Moon and Sun are flown until JPL's tables end (00:00 TDB 11 April); the Earth-only models "
        "to 6 h past the data, and further, unshown, to find where they come back to Earth.",
        "Times are UTC; the trajectories are on TDB, 69.186 s ahead. Earth's J2 is taken about the ICRF z axis, "
        "0.147 deg from the true pole (<= 0.7 km on the first day, metres after).",
    ]
    if args.convergence:
        half = R.fly(best, R._end_for(best), rep.burns, grid_s=30.0, max_turn=R.MAX_TURN_PER_STEP / 2.0)
        full = rep.tiers[-1].flight
        k = np.searchsorted(half.t_s, full.t_s)
        ok = (k < half.t_s.size)
        ok[ok] &= np.abs(half.t_s[k[ok]] - full.t_s[ok]) < 1e-6
        d = np.linalg.norm(half.state[k[ok], :3] - full.state[ok, :3], axis=1)
        j = int(np.argmin(np.abs(full.t_s[ok] - rep.truth_closest.t_s)))
        print(f"\nConvergence ({best.label}, 60 s vs 30 s grid): {d[j]:.2e} km at CA, max {d.max():.2e} km, "
              f"at the end {d[-1]:.2e} km")

    sizes = R.export_dashboard(rep, args.out, generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               engine_commit=_git_commit(), engine_version=_version(), notes=notes)
    print("\nWrote", args.out)
    for name, n in sizes.items():
        print(f"  {name:16s} {n / 1e6:6.3f} MB")
    print(f"  total            {sum(sizes.values()) / 1e6:6.3f} MB")


if __name__ == "__main__":
    main()
