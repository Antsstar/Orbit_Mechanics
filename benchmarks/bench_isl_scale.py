"""
Scan cost of the scalable ISL contact scan, NumPy against the compiled per-pair geometry.

    <env>/python.exe benchmarks/bench_isl_scale.py [--sizes 1000 10000] [--profile]

A Walker 53 deg constellation at 550 km (`scenarios.walker_constellation`) is propagated for a few 30 s
samples into a `history.HistorySink`, then `isl_contact_table_from_recording` scans the recording once per
path and range limit; the figure is seconds per 30 s sample interval. `--profile` adds a cProfile of the
compiled scan, to show what dominates once the geometry is compiled. Timings are indicative: they move
with whatever else the machine is doing. Nothing here asserts.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import isl, kernels, scenarios  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.history import HistorySink  # noqa: E402
from orbital_engine.isl_scale import isl_contact_table_from_recording  # noqa: E402

SAMPLES = 4          # steps recorded after the initial snapshot -> SAMPLES scanned intervals
DT = 30.0


def record(n: int, directory: Path) -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    planes = max(1, int(round(n ** 0.5)))
    while n % planes:
        planes -= 1
    sim = scenarios.walker_constellation(sessionmaker(bind=engine)(), total=n, planes=planes, phasing=1,
                                         inclination_deg=53.0, altitude_km=550.0)
    sim.record_history = False
    with HistorySink(directory, sim, chunk_snapshots=SAMPLES) as sink:
        sim.attach_history_sink(sink)
        for _ in range(SAMPLES):
            sim.step(DT)


def scan(directory: Path, max_range: float, compiled: bool) -> float:
    spec = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=max_range)
    t0 = time.perf_counter()
    isl_contact_table_from_recording(directory, spec, compiled=compiled)
    return (time.perf_counter() - t0) / SAMPLES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    print(f"numba available: {kernels.NUMBA_AVAILABLE}")
    spec_warm = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=2000.0)
    for n in args.sizes:
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "run"
            record(n, rec)
            if kernels.NUMBA_AVAILABLE:                               # compile outside the timings
                isl_contact_table_from_recording(rec, spec_warm, compiled=True)
            for rng_km in (2000.0, 5000.0):
                numpy_s = min(scan(rec, rng_km, False) for _ in range(2))
                line = f"N={n:6d} range={rng_km:6.0f} km  numpy {numpy_s:8.4f} s/sample"
                if kernels.NUMBA_AVAILABLE:
                    comp_s = min(scan(rec, rng_km, True) for _ in range(2))
                    line += f"  compiled {comp_s:8.4f} s/sample  ({numpy_s / comp_s:4.1f}x)"
                print(line)
            if args.profile:
                spec = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=5000.0)
                prof = cProfile.Profile()
                prof.enable()
                isl_contact_table_from_recording(rec, spec, compiled=kernels.NUMBA_AVAILABLE)
                prof.disable()
                pstats.Stats(prof).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":
    main()
