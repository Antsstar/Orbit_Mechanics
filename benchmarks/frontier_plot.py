"""
The fidelity/cost frontier plot: one scenario, four model-fidelity tiers, diffed against a common
DOP853 + J2 truth, via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/frontier_plot.py

Writes `docs/figures/frontier.png`. This script is the only place in the project that imports
matplotlib - `sweep.py` itself has no plotting dependency (see its module docstring).

**Scenario.** `scenarios.earth_constellation` at 550 km / 53 deg, one plane, 12 satellites evenly
phased around it (`theta` spread 0-330 deg in 30 deg steps) - enough points to see the phase
dependence `docs/architecture.md`'s secular-J2 section documents, without the truth generation cost of
a full multi-plane constellation.

**Tiers.**

- Kepler (`PropagatorType.KEPLERIAN`), compiled.
- Secular J2, osculating-seeded (`PropagatorType.SECULAR_J2`, `mean_seed=False`), compiled.
- Secular J2, mean-seeded (`mean_seed=True`), compiled.
- Cowell + `point_mass_gravity` + `j2`, NumPy only (no compiled twin - see `CLAUDE.md`'s "Unwired
  scaffolding" section), swept over several step sizes to trace a fidelity/cost curve.

**Axes.** Error (log) against wall time (log), both from `sweep.SweepResult` - median position error
over the 12 satellites, and minimum-of-batches wall time for the propagation alone (truth generation
and `Simulation` construction excluded, per `sweep.run_sweep`).

**The compiled-vs-NumPy caveat.** Kepler and secular-J2 run compiled (if numba is installed - both
tiers fall back to interpreted Python otherwise, which `Simulation.use_compiled_kernel`'s docstring
already flags as slower than the NumPy path it replaces). Cowell and its force models have no compiled
twin at all. The Cowell curve's wall time therefore mixes a real modelling-cost difference (RK4 at a
given step size against closed-form propagation) with an implementation penalty neither this plot nor
`sweep.py` can separate - printed here, and repeated on the figure itself and in `README.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import geopotential, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.kernels import NUMBA_AVAILABLE
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec, ModelConfig, SweepResult, run_sweep

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "figures" / "frontier.png"

ALTITUDE_KM = 550.0
INCLINATION_DEG = 53.0
N_SATS = 12
HORIZON_S = 86400.0  # 1 day

KEPLER_DT_S = 60.0
COWELL_DTS_S = [160.0, 80.0, 40.0, 20.0, 10.0]

# Reduced from `run_sweep`'s defaults (5 batches, 2 warmup calls) to keep this script's total runtime
# reasonable at the smallest Cowell step size (~8640 steps/config): still minimum-of-batches timing,
# just over fewer batches.
TIMING_BATCHES = 3
TIMING_WARMUP = 1

CAVEAT = (
    "Kepler and secular J2 run compiled (numba); Cowell + point_mass_gravity + j2 has no compiled\n"
    "twin and runs as NumPy only, so its wall time carries an implementation penalty alongside the\n"
    "genuine step-size/fidelity cost."
)


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build_scenario() -> Simulation:
    return scenarios.earth_constellation(
        fresh_session(), n_sats=N_SATS, n_planes=1,
        altitude_km=ALTITUDE_KM, inclination_deg=INCLINATION_DEG,
    )


def build_configs() -> List[ModelConfig]:
    j2_coeffs = {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ}

    configs: List[ModelConfig] = [
        ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=KEPLER_DT_S),
        ModelConfig(
            name="secular-j2 (osculating-seeded)", propagator=PropagatorType.SECULAR_J2,
            dt=KEPLER_DT_S, propagator_coefficients=j2_coeffs,
        ),
        ModelConfig(
            name="secular-j2 (mean-seeded)", propagator=PropagatorType.SECULAR_J2,
            dt=KEPLER_DT_S, mean_seed=True, propagator_coefficients=j2_coeffs,
        ),
    ]
    for dt in COWELL_DTS_S:
        configs.append(ModelConfig(
            name=f"cowell + point_mass_gravity + j2 (dt={dt:.0f}s)",
            propagator=PropagatorType.COWELL, dt=dt,
            force_models=(
                ForceModelSpec("point_mass_gravity"),
                ForceModelSpec("j2", j2_coeffs),
            ),
        ))
    return configs


def plot(results: List[SweepResult]) -> None:
    kepler = [r for r in results if r.config_name == "kepler"]
    secular = [r for r in results if r.config_name.startswith("secular-j2")]
    cowell = [r for r in results if r.config_name.startswith("cowell")]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for r in kepler:
        ax.scatter(r.wall_time_us, r.error.median_km, marker="s", s=70, color="tab:red", zorder=3,
                   label="Kepler")
    for r in secular:
        marker = "^" if "osculating" in r.config_name else "v"
        color = "tab:orange" if "osculating" in r.config_name else "tab:green"
        label = "Secular J2 (osculating-seeded)" if "osculating" in r.config_name else \
            "Secular J2 (mean-seeded)"
        ax.scatter(r.wall_time_us, r.error.median_km, marker=marker, s=70, color=color, zorder=3,
                   label=label)

    if cowell:
        cowell_sorted = sorted(cowell, key=lambda r: r.wall_time_us)
        xs = [r.wall_time_us for r in cowell_sorted]
        ys = [r.error.median_km for r in cowell_sorted]
        ax.plot(xs, ys, marker="o", color="tab:blue", zorder=2, label="Cowell + point_mass_gravity + j2")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time per propagation (us, log scale, min-of-batches)")
    ax.set_ylabel("median position error vs J2 truth at horizon (km, log scale)")
    ax.set_title(
        f"Model-fidelity frontier: earth_constellation, {ALTITUDE_KM:.0f} km / {INCLINATION_DEG:.0f} "
        f"deg, {N_SATS} sats, {HORIZON_S/3600:.0f} h horizon"
    )
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.text(0.01, 0.01, CAVEAT, fontsize=7.5, va="bottom", ha="left", family="monospace")
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"wrote {OUTPUT_PATH}")


def main() -> int:
    print(f"numba available: {NUMBA_AVAILABLE}")
    print(CAVEAT)

    configs = build_configs()
    results = run_sweep(
        build_scenario, configs, HORIZON_S,
        oblateness={"Earth": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)},
        timing_batches=TIMING_BATCHES, timing_warmup=TIMING_WARMUP,
    )

    print(f"\n{'config':<40}{'n_bodies':>9}{'median_km':>12}{'rms_km':>10}{'max_km':>10}{'us':>12}")
    for r in results:
        print(
            f"{r.config_name:<40}{r.n_bodies:>9}{r.error.median_km:>12.4f}{r.error.rms_km:>10.4f}"
            f"{r.error.max_km:>10.4f}{r.wall_time_us:>12.1f}"
        )

    plot(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
