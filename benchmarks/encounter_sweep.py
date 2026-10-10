"""
Temporary systems, phase 3: the formation radius as a sweep axis (`hierarchy.EncounterPolicy`,
`ModelConfig.encounters`).

    <env>/python.exe benchmarks/encounter_sweep.py

Writes `docs/figures/encounters.png` and prints the tables.

**Scenario.** `scenarios.asteroid_encounter` with a long approach: closest approach at day 60 of 120,
2,000 km nominal miss at 1 km/s, starting ~5e6 km apart (10-100 Hill radii). The two masses are the
Ceres/Vesta-like defaults times a scale `s` in {0.01, 0.1, 1, 10}. Truth is DOP853 N-body.

**Configurations.** Keplerian throughout, at 1,200 s, which keeps `dt v_rel` (1,200 km) below the
smallest formation radius. One configuration per formation radius `k` Hill radii (dissolving at `2k`,
evaluated dynamically), plus `unpaired`. The metric is the larger of the two bodies' errors at day
120, which is always the lighter body's.

**What it tests.** The phase 2 argument: the error accumulates as velocity over the encounter, so the
neglected mutual pull outside the radius (`dv ~ mu / (r v)`) balances the neglected solar tide inside
it (`dv ~ mu_sun r^2 / (R^3 v)`) at `r ~ R (m / M)^(1/3)`. That is the Hill scaling, so the best `k`
should not depend on the mass. Laplace's sphere of influence, `R (m / M)^(2/5)`, would move the best
radius as `s^0.4`, so the best `k` would drift as `s^0.067`, by 1.6x over these three decades. The
script fits the exponent of the best radius against `s`.

**Patched conics (phase 4), the third panel.** The same question for a massless probe handed to a
planet: `scenarios.planet_flyby` (Earth's mass at 1 AU, 10,000 km periapsis at v_inf 3 km/s, a 109 deg
turn, periapsis at day 30 of 60), with the planet's mass scaled by `s` and the periapsis by `s` too, so
the hyperbola keeps its shape. One configuration per hand-over radius `k` Hill radii, handed back at
`1.05 k` (a flyby crosses each radius once each way, so it cannot flicker; a symmetric boundary keeps
"the radius" unambiguous). Keplerian at 600 s; the metric is the probe's error at day 60.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import grid, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.hierarchy import EncounterPolicy, EncounterSpec, PatchSpec, hill_radius_km, sphere_radius_km
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ModelConfig

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "encounters.png"
DAY = 86400.0
T_CA = 60.0 * DAY
HORIZON = 2.0 * T_CA
DT = 1200.0
SCALES = [0.01, 0.1, 1.0, 10.0]
KS = [float(k) for k in np.geomspace(0.05, 10.0, 25)]
NAMES = [scenarios.ASTEROID_A, scenarios.ASTEROID_B]
# Ordinal blue ramp for the mass scales (validated: one hue, monotone lightness, light end 2.06:1);
# categorical slots 1-3 for measured / Hill / Laplace.
SCALE_COLOURS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MEASURED, HILL, LAPLACE = "#2a78d6", "#1baf7a", "#eb6834"
INK, MUTED = "#2b2b2a", "#8a897f"


def session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build_for(s: float) -> Callable[[], Simulation]:
    return lambda: scenarios.asteroid_encounter(
        session(), mu_a=scenarios.MU_CERES * s, mu_b=scenarios.MU_VESTA * s, t_ca_s=T_CA)


def configs_for(_s: float) -> List[ModelConfig]:
    out = [ModelConfig("unpaired", PropagatorType.KEPLERIAN, DT, bodies=NAMES)]
    for k in KS:
        policy = EncounterPolicy(form_km=k, dissolve_km=2.0 * k, unit="hill")
        out.append(ModelConfig(f"k={k:.4g}", PropagatorType.KEPLERIAN, DT, bodies=NAMES,
                               encounters=(EncounterSpec(*NAMES, policy),)))
    return out


FLYBY_DT = 600.0
FLYBY_HORIZON = 60.0 * DAY
FLYBY_KS = [float(k) for k in np.geomspace(0.3, 2.5, 21)]


def build_flyby(s: float) -> Callable[[], Simulation]:
    return lambda: scenarios.planet_flyby(session(), mu_planet=scenarios.MU_EARTH * s,
                                          periapsis_km=10000.0 * s)


def flyby_configs(_s: float) -> List[ModelConfig]:
    names = [scenarios.FLYBY_CRAFT]
    out = [ModelConfig("never", PropagatorType.KEPLERIAN, FLYBY_DT, bodies=names)]
    for k in FLYBY_KS:
        policy = EncounterPolicy(form_km=k, dissolve_km=1.05 * k, unit="hill")
        out.append(ModelConfig(f"k={k:.4g}", PropagatorType.KEPLERIAN, FLYBY_DT, bodies=names,
                               patches=(PatchSpec(scenarios.FLYBY_CRAFT, scenarios.FLYBY_PLANET, policy),)))
    return out


def fit_scaling(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Per prediction (slope held fixed): `(amplitude, rms log-misfit)`."""
    out = {}
    for name, p in (("Hill", 1.0 / 3.0), ("Laplace", 0.4)):
        amp = float(np.exp(np.mean(np.log(ys) - p * np.log(xs))))
        out[name] = (amp, float(np.sqrt(np.mean((np.log(ys) - np.log(amp * xs ** p)) ** 2))))
    return out


def best_k(ks: np.ndarray, errs: np.ndarray) -> float:
    """Minimum of a parabola through the lowest point and its neighbours, in log k / log error."""
    i = int(np.argmin(errs))
    if i == 0 or i == len(ks) - 1:
        return float(ks[i])
    x, y = np.log(ks[i - 1:i + 2]), np.log(errs[i - 1:i + 2])
    c = np.polyfit(x, y, 2)
    return float(np.exp(-c[1] / (2.0 * c[0]))) if c[0] > 0 else float(ks[i])


def main() -> int:
    rows = grid.run_grid("mass_scale", SCALES, build_for, configs_for, HORIZON,
                         timing_batches=1, timing_warmup=0)
    r_hill = {s: hill_radius_km(*(lambda sim: (sim, sim.name_to_index[NAMES[0]], sim.name_to_index[NAMES[1]]))(
        build_for(s)())) for s in SCALES}
    curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    unpaired: Dict[float, float] = {}
    print("Error of the lighter asteroid at day 120 (km), by formation radius k (Hill radii):")
    for s in SCALES:
        mine = [r for r in rows if r.value == s]
        unpaired[s] = next(r.result.error.max_km for r in mine if r.config_name == "unpaired")
        ks = np.array([float(r.config_name[2:]) for r in mine if r.config_name != "unpaired"])
        es = np.array([r.result.error.max_km for r in mine if r.config_name != "unpaired"])
        order = np.argsort(ks)
        curves[s] = (ks[order], es[order])
        line = "  ".join(f"{k:.3g}:{e:.3g}" for k, e in zip(*curves[s]))
        print(f"  s={s:<5g} r_H(t=0)={r_hill[s]:.3e} km  unpaired {unpaired[s]:.3e}\n    {line}")

    k_star = {s: best_k(*curves[s]) for s in SCALES}
    r_star = {s: k_star[s] * r_hill[s] for s in SCALES}
    slope, intercept = np.polyfit(np.log(SCALES), np.log([r_star[s] for s in SCALES]), 1)
    k_slope = np.polyfit(np.log(SCALES), np.log([k_star[s] for s in SCALES]), 1)[0]
    print("\nBest formation radius:")
    for s in SCALES:
        e_best = float(np.min(curves[s][1]))
        print(f"  s={s:<5g} k*={k_star[s]:.3f}  r*={r_star[s]:.3e} km  error {e_best:.3e} km "
              f"(unpaired {unpaired[s]:.3e}, {unpaired[s] / e_best:.0f}x)")
    print(f"  r* ~ s^{slope:.3f}  (Hill 1/3 = 0.333, Laplace 2/5 = 0.400); k* ~ s^{k_slope:.3f}")

    frows = grid.run_grid("planet_mass_scale", SCALES, build_flyby, flyby_configs, FLYBY_HORIZON,
                          timing_batches=1, timing_warmup=0)
    f_hill = {s: sphere_radius_km(b, b.name_to_index[scenarios.FLYBY_PLANET], "hill")
              for s, b in ((s, build_flyby(s)()) for s in SCALES)}
    f_star: Dict[float, float] = {}
    print("\nPatched conics: probe error at day 60 (km), by hand-over radius k (Hill radii):")
    for s in SCALES:
        mine = [r for r in frows if r.value == s]
        never = next(r.result.error.max_km for r in mine if r.config_name == "never")
        ks = np.array([float(r.config_name[2:]) for r in mine if r.config_name != "never"])
        es = np.array([r.result.error.max_km for r in mine if r.config_name != "never"])
        order = np.argsort(ks)
        kb = best_k(ks[order], es[order])
        f_star[s] = kb * f_hill[s]
        print(f"  s={s:<5g} r_H={f_hill[s]:.3e} km  never {never:.3e}  best k={kb:.3f} r*={f_star[s]:.3e} "
              f"km  error {float(np.min(es)):.3e} km ({never / float(np.min(es)):.0f}x)")
    f_slope = float(np.polyfit(np.log(SCALES), np.log([f_star[s] for s in SCALES]), 1)[0])
    f_fit = fit_scaling(np.array(SCALES), np.array([f_star[s] for s in SCALES]))
    print(f"  r* ~ s^{f_slope:.3f}; fixed-slope rms misfit Hill {f_fit['Hill'][1]:.3f}, "
          f"Laplace {f_fit['Laplace'][1]:.3f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.4))
    for s, colour in zip(SCALES, SCALE_COLOURS):
        ks, es = curves[s]
        ax1.loglog(ks, es, "-", color=colour, lw=2, label=f"masses x{s:g}")
        ax1.plot(ks, es, "o", color=colour, ms=4)
        ax1.axhline(unpaired[s], color=colour, lw=1, ls=":")
    ax1.axvline(1.0, color=MUTED, lw=1, ls="--")
    ax1.text(1.06, ax1.get_ylim()[1] * 0.4, "1 Hill radius", fontsize=8, color=MUTED)
    ax1.set_xlabel("formation radius k (Hill radii; dissolves at 2k)")
    ax1.set_ylabel("error of the lighter asteroid at day 120 vs N-body (km)")
    ax1.set_title("Error against formation radius (dotted: never paired)", fontsize=11, color=INK)
    ax1.grid(True, which="major", alpha=0.25)
    ax1.legend(fontsize=8, frameon=False, loc="lower left")

    xs = np.array(SCALES)
    ys = np.array([r_star[s] for s in SCALES])
    ax2.loglog(xs, ys, "o", color=MEASURED, ms=8, label="measured best radius", zorder=3)
    # Each prediction gets its own best-fitting amplitude with its slope held fixed, so the lines are
    # compared on shape alone; pinning both to one point would flatter whichever passes through it.
    for p, colour, style, label in ((1.0 / 3.0, HILL, "-", "Hill"), (0.4, LAPLACE, "--", "Laplace")):
        amp = float(np.exp(np.mean(np.log(ys) - p * np.log(xs))))
        rms = float(np.sqrt(np.mean((np.log(ys) - np.log(amp * xs ** p)) ** 2)))
        ax2.loglog(xs, amp * xs ** p, style, color=colour, lw=2,
                   label=f"{label} scaling, s^({'1/3' if p < 0.35 else '2/5'}): rms misfit {100 * rms:.0f}%")
        print(f"  {label}: rms log-misfit {rms:.3f}")
    ax2.set_xlabel("mass scale s (x Ceres/Vesta-like masses)")
    ax2.set_ylabel("best formation radius r* (km)")
    ax2.set_title(f"Best radius scales as s^{slope:.2f}: Hill, not Laplace", fontsize=11, color=INK)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")

    fys = np.array([f_star[s] for s in SCALES])
    ax3.loglog(xs, fys, "o", color=MEASURED, ms=8, label="measured best hand-over radius", zorder=3)
    for (name, (amp, rms)), colour, style in zip(f_fit.items(), (HILL, LAPLACE), ("-", "--")):
        ax3.loglog(xs, amp * xs ** (1.0 / 3.0 if name == "Hill" else 0.4), style, color=colour, lw=2,
                   label=f"{name} scaling: rms misfit {100 * rms:.0f}%")
    ax3.set_xlabel("planet mass scale s (x Earth; periapsis scaled with it)")
    ax3.set_ylabel("best hand-over radius r* (km)")
    ax3.set_title(f"Patched conics: s^{f_slope:.2f}, between the two", fontsize=11, color=INK)
    ax3.grid(True, which="major", alpha=0.25)
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
