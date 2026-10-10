"""
Temporary systems, phase 5: regime switching on the cost-error frontier (`regimes.py`).

    <env>/python.exe benchmarks/regime_frontier.py

Writes `docs/figures/regimes.png` and prints the table.

**Scenario.** `scenarios.planet_flyby`, periapsis at day 10 of 20: a probe passes Earth at 10,000 km
with v_inf 3 km/s, a 109 deg turn. Truth is DOP853 N-body (Sun, Earth, probe).

**Configurations**, each a `ModelConfig` whose probe model is set by a `RegimeSwitch` at 0.76 / 0.80
Earth Hill radii (the best patched-conic radius, phase 4), at steps from 30 s to 1,200 s:

- `Kepler, never handed over` - heliocentric Kepler throughout.
- `patched conics` - Kepler about the Sun far, Kepler about Earth near (phase 4).
- `Kepler far, Cowell near` - near Earth: Cowell, Earth point mass + the Sun as third body.
- `Cowell, centre switched` - Cowell on both sides: about the Sun with Earth as third body far, about
  Earth with the Sun as third body near.
- `Cowell about the Sun` - the same physics, never switching centre.

Error is the probe's position error at day 20; cost is the wall time of the propagation
(`run_sweep`, one batch). The engine has one clock, so every configuration takes the same steps;
what a regime saves is cost per step.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType as PT
from orbital_engine.database import Base
from orbital_engine.hierarchy import EncounterPolicy
from orbital_engine.regimes import Regime, RegimeSwitch
from orbital_engine.sweep import ForceModelSpec as F, ModelConfig, run_sweep

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "regimes.png"
DAY = 86400.0
T_CA = 10.0 * DAY
HORIZON = 2.0 * T_CA
DTS = [1200.0, 600.0, 300.0, 120.0, 60.0, 30.0]
P, E, S = scenarios.FLYBY_CRAFT, scenarios.FLYBY_PLANET, "Sun"
POLICY = EncounterPolicy(0.76, 0.80, unit="hill")
KEP_SUN, KEP_EARTH = Regime(S), Regime(E)
COW_SUN = Regime(S, PT.COWELL, (F("point_mass_gravity"), F("third_body", body_coefficients={"perturber": E})))
COW_EARTH = Regime(E, PT.COWELL, (F("point_mass_gravity"), F("third_body", body_coefficients={"perturber": S})))
# Categorical slots 1-5 in fixed order (validated; three below 3:1 on the surface, so every series also
# has its own marker and a direct label).
SERIES = [
    ("Kepler, never handed over", None, "#2a78d6", "o"),
    ("patched conics", (KEP_EARTH, KEP_SUN), "#eb6834", "s"),
    ("Kepler far, Cowell near", (COW_EARTH, KEP_SUN), "#1baf7a", "^"),
    ("Cowell, centre switched", (COW_EARTH, COW_SUN), "#eda100", "D"),
    ("Cowell about the Sun", (COW_SUN, COW_SUN), "#e87ba4", "v"),
]
INK = "#2b2b2a"


def session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def configs(dt: float) -> List[ModelConfig]:
    out = []
    for name, pair, _, _ in SERIES:
        regimes = () if pair is None else (RegimeSwitch(P, E, POLICY, pair[0], pair[1]),)
        out.append(ModelConfig(name, PT.KEPLERIAN, dt, bodies=[P], regimes=regimes))
    return out


def main() -> int:
    points: Dict[str, List[Tuple[float, float, float]]] = {name: [] for name, *_ in SERIES}
    for dt in DTS:
        for r in run_sweep(lambda: scenarios.planet_flyby(session(), t_ca_s=T_CA), configs(dt), HORIZON,
                           timing_batches=1, timing_warmup=0):
            points[r.config_name].append((dt, r.wall_time_us / 1e6, r.error.max_km))
    print("Probe error at day 20 (km) and wall time (s), by step:")
    print("  " + " " * 28 + "".join(f"{dt:>17.0f} s" for dt in DTS))
    for name, *_ in SERIES:
        print(f"  {name:28s}" + "".join(f"  {e:9.3e} / {w:5.2f}" for _, w, e in points[name]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    for name, _, colour, marker in SERIES:
        dts = [p[0] for p in points[name]]
        walls = [p[1] for p in points[name]]
        errs = [p[2] for p in points[name]]
        ax1.loglog(walls, errs, "-", color=colour, lw=2, marker=marker, ms=6, label=name)
        ax1.annotate(name, (walls[-1], errs[-1]), textcoords="offset points", xytext=(6, -3),
                     fontsize=8, color=INK)
        ax2.loglog(dts, errs, "-", color=colour, lw=2, marker=marker, ms=6, label=name)
    ax1.set_xlabel("wall time of the 20-day propagation (s)")
    ax1.set_ylabel("probe error at day 20 vs N-body (km)")
    ax1.set_title("Cost against error: each point a step size (30-1,200 s)", fontsize=11, color=INK)
    ax1.grid(True, which="major", alpha=0.25)
    ax2.set_xlabel("step (s)")
    ax2.set_ylabel("probe error at day 20 vs N-body (km)")
    ax2.set_title("Both Cowells converge at first order (third_body's frozen perturber);\n"
                  "switching the centre is ~50x better at every step", fontsize=11, color=INK)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.invert_xaxis()
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
