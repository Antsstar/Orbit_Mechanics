"""
Sweeps along an axis (`orbital_engine.grid`): when a model stops working, and where one implementation
overtakes another.

    <env>/python.exe benchmarks/scaling_sweep.py

Writes `docs/figures/scaling.png` and prints both tables.

**Step size.** Cowell + J2 on six 550 km satellites over 6,400 s (every step divides it), against the
DOP853 + J2 truth, from 10 s to 1,280 s. `grid.stability_limit` reports the largest step that keeps
the median error under 1 km, 100 km and 6,000 km (about the orbit itself).

**Body count.** The same physics (Cowell + J2, 60 s) compiled (`ModelConfig.compiled=True`) and NumPy
(`False`), with Kepler as the analytic floor, on `earth_constellation` from 6 to 384 satellites over
6,000 s. `grid.crossover` reports where the two implementations' wall times cross, if they do.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import geopotential, grid, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.kernels import NUMBA_AVAILABLE
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec, ModelConfig

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "scaling.png"
J2 = {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ}
OBL = {"Earth": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)}
FM = (ForceModelSpec("point_mass_gravity"), ForceModelSpec("j2", J2))
DTS = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0]
NS = [6.0, 24.0, 96.0, 384.0]
THRESHOLDS_KM = [1.0, 100.0, 6000.0]


def session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build_n(n: float) -> Callable[[], Simulation]:
    return lambda: scenarios.earth_constellation(session(), n_sats=int(n), n_planes=6)


def main() -> int:
    if not NUMBA_AVAILABLE:
        print("numba is required: the body-count axis compares the compiled and NumPy implementations")
        return 1
    dt_rows = grid.run_grid("dt_s", DTS, lambda v: build_n(6.0),
                            lambda v: [ModelConfig("cowell+j2", PropagatorType.COWELL, v, force_models=FM)],
                            6400.0, oblateness=OBL, timing_batches=1, timing_warmup=0)
    print("Step size: Cowell + J2, 550 km, 6,400 s")
    for r in dt_rows:
        print(f"  dt {r.value:6.0f} s   median error {r.result.error.median_km:11.4e} km")
    limits = {thr: grid.stability_limit(dt_rows, "cowell+j2", thr) for thr in THRESHOLDS_KM}
    for thr, (ok, bad) in limits.items():
        print(f"  under {thr:g} km up to dt = {ok} s; fails from {bad} s")

    def configs(v: float) -> List[ModelConfig]:
        return [ModelConfig("Cowell + J2, compiled", PropagatorType.COWELL, 60.0, force_models=FM, compiled=True),
                ModelConfig("Cowell + J2, NumPy", PropagatorType.COWELL, 60.0, force_models=FM, compiled=False),
                ModelConfig("Kepler", PropagatorType.KEPLERIAN, 6000.0)]

    n_rows = grid.run_grid("n_bodies", NS, build_n, configs, 6000.0, oblateness=OBL, timing_batches=3,
                           timing_warmup=1)
    print("\nBody count: wall time per propagation (us) and per body (us)")
    for r in n_rows:
        print(f"  N {r.value:5.0f}  {r.config_name:22s} {r.result.wall_time_us:11.1f} {r.wall_time_per_body_us:9.2f}")
    cross = grid.crossover(n_rows, "Cowell + J2, compiled", "Cowell + J2, NumPy")
    print(f"  compiled vs NumPy crossover: {cross if cross is not None else 'none on this grid'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x, y = grid.series(dt_rows, "cowell+j2", "median_km")
    ax1.loglog(x, y, "o-", color="#1f77b4", label="Cowell + J2 (RK4)")
    for thr in THRESHOLDS_KM:
        ax1.axhline(thr, color="0.6", lw=0.8, ls="--")
        ok, bad = limits[thr]
        ax1.text(x[0], thr * 1.3, f"{thr:g} km: dt <= {ok:g} s" if ok else f"{thr:g} km", fontsize=8, color="0.3")
    ax1.set_xlabel("step size (s)")
    ax1.set_ylabel("median position error vs J2 truth (km)")
    ax1.set_title("When does a model stop working? (550 km, 6,400 s)")
    ax1.grid(True, which="both", alpha=0.3)
    for name, colour in (("Cowell + J2, compiled", "#1f77b4"), ("Cowell + J2, NumPy", "#ff7f0e"), ("Kepler", "#d62728")):
        x, y = grid.series(n_rows, name, "wall_time_per_body_us")
        ax2.loglog(x, y, "o-", color=colour, label=name)
    ax2.set_xlabel("satellites")
    ax2.set_ylabel("wall time per satellite per propagation (us)")
    ax2.set_title("Where does one implementation overtake another?")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    note = ("Crossover (compiled vs NumPy): " + (f"N = {cross:.0f}" if cross is not None else "none - compiled is cheaper at every N"))
    fig.text(0.01, 0.01, note, fontsize=8, family="monospace")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
