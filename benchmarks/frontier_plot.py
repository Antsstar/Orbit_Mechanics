"""
The fidelity/cost frontier plot: one scenario, four model-fidelity tiers, diffed against a common
DOP853 + J2 truth, via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/frontier_plot.py

Writes `docs/figures/frontier.png`, two panels side by side. This script is the only place in the
project that imports matplotlib - `sweep.py` itself has no plotting dependency (see its module
docstring). Needs the `sgp4` extra for panel B.

**Panel B: SGP4.** `scenarios.tle_satellites` with `sgp4_bridge.ISS_TLE` (the only TLE the bridge
ships, so one satellite and the "median" is that satellite's error), one-day horizon, the same
DOP853 + J2 truth, the same tiers (Cowell swept over panel A's steps plus 60, 30 and 15 s so the
panel reproduces the one-day table in `docs/architecture.md`), plus `sgp4_bridge.sgp4_tier` scored
as an `ExternalTier` through `run_sweep(external=)`. SGP4 is closed-form in time, so like the
analytic tiers it is timed as one evaluation at the horizon (`dt = HORIZON_S`). **SGP4's distance
from the truth is a model difference, not an accuracy error**: the truth is J2-only and SGP4 adds
J3/J4, drag and WGS-72 constants (+1.20 km mu, +0.95 km drag, -1.18 km J3/J4 and theory of its
0.966 km along-track gap; the figure says so). The constellation cannot host SGP4 - it has no
TLEs, and fitting one to engine states is forbidden.

**Scenario.** `scenarios.earth_constellation` at 550 km / 53 deg, one plane, 12 satellites evenly
phased around it (`theta` spread 0-330 deg in 30 deg steps) - enough points to see the phase
dependence `docs/architecture.md`'s secular-J2 section documents, without the truth generation cost of
a full multi-plane constellation.

**Tiers.**

- Kepler (`PropagatorType.KEPLERIAN`), compiled.
- Secular J2, osculating-seeded (`PropagatorType.SECULAR_J2`, `mean_seed=False`), compiled.
- Secular J2, mean-seeded (`mean_seed=True`), compiled.
- Cowell + `point_mass_gravity` + `j2`, compiled through the fused twin `kernels.cowell_rk4_step`
  (which also fuses `zonal`, not enabled here), swept over several step sizes to trace a
  fidelity/cost curve.

**Axes.** Error (log) against wall time (log), both from `sweep.SweepResult` - median position error
over the 12 satellites, and minimum-of-batches wall time for the propagation alone (truth generation
and `Simulation` construction excluded, per `sweep.run_sweep`).

**The implementation caveat.** With numba installed every tier runs compiled: Kepler and secular J2
through their kernels, Cowell through `kernels.cowell_rk4_step`, which fuses RK4 with
`point_mass_gravity`, `j2`, `drag` and `zonal` - this script sweeps the first two (any model outside
those four would fall back to the NumPy `RK4Integrator` path, and `Simulation._cowell_fused_ok` says which applied). Without numba,
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
as Kepler. Measured with one step, compiled re-base, idle machine, two runs: Kepler 20 us, secular J2
23-25 us, and Cowell 3.1-3.7 ms at 160 s up to 51-54 ms at 10 s. This plot measures the cost of reaching the horizon state. A plot of the cost of a
fixed-cadence ephemeris would be a different question, with different analytic-tier costs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import geopotential, scenarios, sgp4_bridge
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
# Panel B sweeps the same steps plus 60, 30 and 15 s, the steps `docs/architecture.md`'s one-day ISS
# table quotes, so the panel can be checked against that table digit for digit.
ISS_COWELL_DTS_S = sorted(set(COWELL_DTS_S) | {60.0, 30.0, 15.0}, reverse=True)
SGP4_NAME = "SGP4"

# Reduced from `run_sweep`'s defaults (5 batches, 2 warmup calls) to keep this script's total runtime
# reasonable at the smallest Cowell step size (~8640 steps/config): still minimum-of-batches timing,
# just over fewer batches.
TIMING_BATCHES = 3
TIMING_WARMUP = 1

CAVEAT_COMPILED = (
    "All tiers run compiled (numba): Kepler and secular J2 through their kernels, Cowell through the\n"
    "fused kernels.cowell_rk4_step (RK4 + point_mass_gravity + j2; it also fuses drag, zonal, tesseral). Wall time\n"
    "is the model, not the "
    "implementation: the per-step re-base of Cowell and secular-J2 rows is compiled too (bit-identical twin).\n"
    "Kepler and secular J2 are closed-form, so they reach the 24 h horizon in a single step; Cowell must step."
)
CAVEAT_INTERPRETED = (
    "numba is NOT installed: Kepler and secular J2 ran as interpreted Python and Cowell as vectorised\n"
    "NumPy, so wall time here measures implementation as much as model. Install [perf] for a fair\n"
    "timing comparison."
)
CAVEAT_SGP4 = (
    "Panel B: SGP4's distance from the truth is a model difference, not an accuracy error. The truth is\n"
    "J2-only; SGP4 carries J3/J4, drag (B*) and WGS-72 constants. Of its 0.966 km along-track gap:\n"
    "+1.20 km WGS-72 mu vs MU_EARTH, +0.95 km drag, -1.18 km J3/J4 and SGP4's theory (docs/architecture.md).\n"
    "SGP4 is closed-form in time, timed like the analytic tiers: one evaluation at the 24 h horizon."
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


def build_configs(cowell_dts_s: Sequence[float]) -> List[ModelConfig]:
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
    for dt in cowell_dts_s:
        configs.append(ModelConfig(
            name=f"cowell + point_mass_gravity + j2 (dt={dt:.0f}s)",
            propagator=PropagatorType.COWELL, dt=dt,
            force_models=(
                ForceModelSpec("point_mass_gravity"),
                ForceModelSpec("j2", j2_coeffs),
            ),
        ))
    return configs


def build_iss_scenario() -> Simulation:
    return scenarios.tle_satellites(fresh_session(), [sgp4_bridge.ISS_TLE])


def draw_panel(ax: Axes, results: List[SweepResult], title: str) -> None:
    """Draw one frontier panel. Colours and markers are fixed per tier, so the panels agree."""
    kepler = [r for r in results if r.config_name == "kepler"]
    secular = [r for r in results if r.config_name.startswith("secular-j2")]
    cowell = [r for r in results if r.config_name.startswith("cowell")]
    sgp4 = [r for r in results if r.config_name == SGP4_NAME]

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

    for r in sgp4:
        ax.scatter(r.wall_time_us, r.error.median_km, marker="D", s=80, color="tab:purple", zorder=4,
                   label="SGP4 (external tier)")
        ax.annotate(
            "SGP4: J3/J4, drag, WGS-72 mu\nthat the J2 truth lacks -\nmodel difference, not error",
            xy=(r.wall_time_us, r.error.median_km), xytext=(0.27, 0.40), textcoords="axes fraction",
            fontsize=7.5, color="tab:purple", ha="left", va="top",
            arrowprops={"arrowstyle": "->", "color": "tab:purple", "lw": 0.8},
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time per propagation (us, log scale, min-of-batches)")
    ax.set_ylabel("median position error vs J2 truth at horizon (km, log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)


def plot(results: List[SweepResult], iss_results: List[SweepResult]) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.0, 6.2))

    draw_panel(
        ax_a, results,
        f"A: earth_constellation, {ALTITUDE_KM:.0f} km / {INCLINATION_DEG:.0f} deg, {N_SATS} sats, "
        f"{HORIZON_S/3600:.0f} h (median)",
    )
    draw_panel(ax_b, iss_results, f"B: ISS from its TLE (tle_satellites), 1 sat, {HORIZON_S/3600:.0f} h")
    fig.suptitle("Model-fidelity frontier", fontsize=12)
    fig.text(0.01, 0.01, CAVEAT + "\n" + CAVEAT_SGP4, fontsize=7.5, va="bottom", ha="left",
             family="monospace")
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"wrote {OUTPUT_PATH}")


def print_table(results: List[SweepResult]) -> None:
    print(f"\n{'config':<40}{'n_bodies':>9}{'median_km':>12}{'rms_km':>10}{'max_km':>10}{'us':>12}")
    for r in results:
        print(
            f"{r.config_name:<40}{r.n_bodies:>9}{r.error.median_km:>12.4f}{r.error.rms_km:>10.4f}"
            f"{r.error.max_km:>10.4f}{r.wall_time_us:>12.1f}"
        )


def main() -> int:
    print(f"numba available: {NUMBA_AVAILABLE}")
    print(CAVEAT)
    oblateness = {"Earth": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)}

    print("\nPanel A: earth_constellation")
    results = run_sweep(
        build_scenario, build_configs(COWELL_DTS_S), HORIZON_S, oblateness=oblateness,
        timing_batches=TIMING_BATCHES, timing_warmup=TIMING_WARMUP,
    )
    print_table(results)

    print("\nPanel B: ISS, tle_satellites")
    tier = sgp4_bridge.sgp4_tier(
        [sgp4_bridge.ISS_TLE], sgp4_bridge.tle_epoch(sgp4_bridge.ISS_TLE), dt=HORIZON_S, name=SGP4_NAME,
    )
    iss_results = run_sweep(
        build_iss_scenario, build_configs(ISS_COWELL_DTS_S), HORIZON_S, oblateness=oblateness,
        external=[tier], timing_batches=TIMING_BATCHES, timing_warmup=TIMING_WARMUP,
    )
    print_table(iss_results)

    plot(results, iss_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
