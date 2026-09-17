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
- Cowell + `point_mass_gravity` + `j2`, compiled through the fused twin `kernels.cowell_rk4_step`,
  swept over several step sizes to trace a fidelity/cost curve.

**Axes.** Error (log) against wall time (log), both from `sweep.SweepResult` - median position error
over the 12 satellites, and minimum-of-batches wall time for the propagation alone (truth generation
and `Simulation` construction excluded, per `sweep.run_sweep`).

**The implementation caveat.** With numba installed every tier runs compiled: Kepler and secular J2
through their kernels, Cowell through `kernels.cowell_rk4_step`, which fuses RK4 with exactly the
`point_mass_gravity` + `j2` combination this script sweeps (any other model would fall back to the
NumPy `RK4Integrator` path, and `Simulation._cowell_fused_ok` says which applied). Without numba,
Kepler and secular J2 run as interpreted Python and Cowell as vectorised NumPy, so the horizontal axis
then measures implementation as much as model - printed here, on the figure itself and in
`README.md`. The re-base each Cowell and secular-J2 body needs after `calc_global()` also runs compiled
(`kernels.rebase_relative_states`, bit-identical to the NumPy block in `Simulation._rebase`); before
it did, that block cost 11 us per Cowell step and 17 us per secular-J2 step (12 satellites, measured
with `benchmark.measure`) and dominated both tiers: a step now costs ~6 us against ~5-6 us for
Kepler. What remains
is the ordinary Python of `step()`, paid by every tier alike; `README.md` carries the measured
per-tier timings.

**The analytic tiers take one step to the horizon (`KEPLER_DT_S = HORIZON_S`).** Kepler and secular J2
are closed-form in time, so their horizon error does not depend on step size: 60 s steps and a single
24 h step gave identical medians (607.6, 431.3 and 4.04 km). An earlier version stepped them every 60 s,
which charged them for 1440 steps they do not need. It made Cowell at a 160 s step look about as cheap
as Kepler. Measured with one step: Kepler 21-25 us, secular J2 37-38 us, and Cowell 8.6-8.8 ms at 160 s up to
143-148 ms at 10 s, across two runs. This plot measures the cost of reaching the horizon state. A plot of the cost of a
fixed-cadence ephemeris would be a different question, with different analytic-tier costs.
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

KEPLER_DT_S = HORIZON_S  # analytic tiers are closed-form: one step reaches the horizon
COWELL_DTS_S = [160.0, 80.0, 40.0, 20.0, 10.0]

# Reduced from `run_sweep`'s defaults (5 batches, 2 warmup calls) to keep this script's total runtime
# reasonable at the smallest Cowell step size (~8640 steps/config): still minimum-of-batches timing,
# just over fewer batches.
TIMING_BATCHES = 3
TIMING_WARMUP = 1

CAVEAT_COMPILED = (
    "All tiers run compiled (numba): Kepler and secular J2 through their kernels, Cowell through the\n"
    "fused kernels.cowell_rk4_step (RK4 + point_mass_gravity + j2). Wall time is the model, not the\n"
    "implementation: the per-step re-base of Cowell and secular-J2 rows is compiled too (bit-identical twin).\n"
    "Kepler and secular J2 are closed-form, so they reach the 24 h horizon in a single step; Cowell must step."
)
CAVEAT_INTERPRETED = (
    "numba is NOT installed: Kepler and secular J2 ran as interpreted Python and Cowell as vectorised\n"
    "NumPy, so wall time here measures implementation as much as model. Install [perf] for a fair\n"
    "timing comparison."
)
CAVEAT = CAVEAT_COMPILED if NUMBA_AVAILABLE else CAVEAT_INTERPRETED


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
