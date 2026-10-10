"""
A system's field seen from outside, as a fidelity ladder (`quadrupole.py`):
monopole -> averaged quadrupole -> instantaneous quadrupole -> resolved members.

    <env>/python.exe benchmarks/multipole_ladder.py

Writes `docs/figures/multipole.png` and prints the tables.

**Scenario.** `scenarios.binary_probe`: an isolated Earth-Moon system (no Sun) and a massless probe on a
30 deg inclined orbit about it. Truth is DOP853 N-body of Earth, Moon and probe. Every rung is Cowell at
a 3,600 s step with adaptive sub-stepping at 1e-5 km, so integration error is far below every model
error shown (the resolved rung, the same physics as truth, lands at ~1e-7 km).

**Rungs.** `monopole`: the probe orbits the barycentre, `point_mass_gravity` with the summed `mu`.
`averaged quadrupole`: plus the orbit-averaged quadrupole (a ring). `instantaneous quadrupole`: plus the
live quadrupole, every periodic harmonic of the inner orbit included. `resolved`: the probe orbits Earth
with the Moon as a staged third body - Earth and Moon as two points, which is exact here.

**Panel A** is error against horizon at r = 2e6 km (d/r ~ 0.19). **Panel B** is error at 180 days against
d/r, r from 1e6 to 4e6 km.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import scenarios
from orbital_engine.custom_types import PropagatorType as PT
from orbital_engine.database import Base
from orbital_engine.quadrupole import QUADRUPOLE_MODEL
from orbital_engine.reference import reference_for

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "multipole.png"
DAY = 86400.0
DT = 3600.0
TOL_KM = 1e-5
HORIZONS_D = [15.0, 30.0, 60.0, 90.0, 180.0, 270.0, 360.0]
RADII_KM = [1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6]
# Categorical slots 1-4 in fixed order (validated with the 5- and 7-slot sets; aqua and yellow are below
# 3:1 on the surface, so every series also has its own marker and a direct label).
RUNGS = [("monopole", "#2a78d6", "o"), ("averaged quadrupole", "#eb6834", "s"),
         ("instantaneous quadrupole", "#1baf7a", "^"), ("resolved", "#eda100", "D")]
INK = "#2b2b2a"


def session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def run(radius_km: float, rung: str, horizons_d: List[float]) -> List[float]:
    sim = scenarios.binary_probe(session(), probe_radius_km=radius_km)
    sim.record_history = False
    earth, moon, emb, probe = (sim.name_to_index[n] for n in ("Earth", "Moon", "EMB", scenarios.BINARY_PROBE))
    truth = reference_for(sim, np.array([0.0] + [h * DAY for h in horizons_d]))
    sim.set_propagator(np.array([probe]), PT.COWELL)
    if rung == "resolved":
        sim.enable_force_model("point_mass_gravity", [probe])
        sim.enable_force_model("third_body", [probe], perturber=float(moon), staged=1.0)
    else:
        sim.reparent(probe, emb)
        sim.enable_force_model("point_mass_gravity", [probe])
        if rung != "monopole":
            sim.enable_force_model(QUADRUPOLE_MODEL, [probe], mode=0.0 if rung.startswith("averaged") else 1.0,
                                   primary=float(earth), secondary=float(moon))
    sim.set_cowell_tolerance(TOL_KM)
    out = []
    for k, h in enumerate(horizons_d):
        while sim.t < h * DAY - 1.0:
            sim.step(DT)
        out.append(float(np.linalg.norm(sim.global_states[probe, :3]
                                        - truth.position_of(scenarios.BINARY_PROBE)[k + 1])))
    return out


def main() -> int:
    d_km = 3.844e5
    by_horizon: Dict[str, List[float]] = {r: run(2.0e6, r, HORIZONS_D) for r, _, _ in RUNGS}
    print("Error (km) against horizon at r = 2e6 km:")
    print("  " + " " * 26 + "".join(f"{h:>10.0f} d" for h in HORIZONS_D))
    for name, _, _ in RUNGS:
        print(f"  {name:26s}" + "".join(f"{e:12.3e}" for e in by_horizon[name]))
    by_radius: Dict[str, List[float]] = {r: [run(rad, r, [180.0])[0] for rad in RADII_KM] for r, _, _ in RUNGS}
    print("\nError (km) at 180 days against probe distance:")
    print("  " + " " * 26 + "".join(f"{r:>12.1e}" for r in RADII_KM))
    for name, _, _ in RUNGS:
        print(f"  {name:26s}" + "".join(f"{e:12.3e}" for e in by_radius[name]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    xs = np.array([d_km / r for r in RADII_KM])
    for name, colour, marker in RUNGS:
        ax1.loglog(HORIZONS_D, by_horizon[name], "-", color=colour, lw=2, marker=marker, ms=6, label=name)
        ax1.annotate(name, (HORIZONS_D[-1], by_horizon[name][-1]), textcoords="offset points",
                     xytext=(6, -3), fontsize=8, color=INK)
        ax2.loglog(xs, by_radius[name], "-", color=colour, lw=2, marker=marker, ms=6, label=name)
    ax1.set_xlabel("horizon (days)")
    ax1.set_ylabel("probe error vs N-body (km)")
    ax1.set_title("Probe at 2e6 km from the Earth-Moon barycentre:\naveraged starts worse, then overtakes "
                  "the monopole", fontsize=11, color=INK)
    ax1.grid(True, which="major", alpha=0.25)
    ax1.set_xlim(HORIZONS_D[0] * 0.8, HORIZONS_D[-1] * 3.5)
    ax2.set_xlabel("inner separation / probe distance, d / r")
    ax2.set_ylabel("probe error at 180 days vs N-body (km)")
    ax2.set_title("A far-field ladder: it holds to d/r = 0.26 and breaks down at 0.38\n"
                  "(r = 2.6 d, near the limit of stable orbits around a binary)", fontsize=11, color=INK)
    ax2.grid(True, which="major", alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
