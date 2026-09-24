"""
Does a 550 km constellation need more than J2? The `"zonal"` headline, via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/zonal_sweep.py

Prints one line per tier: median / RMS / max position error at 24 h against a DOP853 truth carrying
J2..J6, then the access metrics (`access.format_metrics`). No figure, no file written. Takes a few
minutes: truth is integrated twice (horizon and the 60 s access grid) with the monomial zonal field.

**Scenario and stations** are the access figure's (`benchmarks/figures.py`, section 6): 12 satellites,
one plane, 550 km / 53 deg, 24 h; Kiruna, Wallops, Santiago, 5 deg mask, 60 s grid. Restated here so
this script does not import matplotlib.

**Tiers.** Cowell + point_mass_gravity + j2, with and without `"zonal"` (EGM96 J3..J6), each at 60 s
and 15 s, plus mean-seeded secular J2. The 15 s pair is not in the design brief; it is there because
RK4's own truncation at 60 s is ~1.9 km at this horizon (`docs/architecture.md`, access section),
the same size as the J3..J6 estimate below, so the 60 s pair alone cannot separate model from step.
Timing is one batch and is **not comparable**: a Cowell body carrying `"zonal"` runs the NumPy path,
the others the fused compiled kernel (see `zonal.py`).

**Estimate, written before the first run** (circular, i = 53 deg, a = 6921 km, n = 1.0965e-3 rad/s):

- J4's first-order secular rate of the mean argument of latitude, `d(M + w)/dt = -(3/8) n J4 (R/a)^4
  [10 F - cos^2 i ((35/2) sin^2 i - 10)]`, `F = (35/8) sin^4 i - 5 sin^2 i + 1` = -0.409, is
  -2.17e-9 rad/s: **-1.3 km along-track** at 24 h, the same for every satellite.
- Short-period J3/J4 terms in `a` at the (shared, osculating) seed shift each satellite's mean `a` by
  ~`2 J_n (R/a)^n x 0.3` ~ 1e-6 relative, a drift `(3/2) n t da/a` of **~1 km, sign varying with the
  seed's argument of latitude**.
- J3 drives the eccentricity vector toward the frozen value (~8.6e-4) at J2's apsidal rate (3 deg/day):
  ~4.6e-5 in a day, **0.3 km radial / 0.6 km along-track**, once per orbit.
- J4's node rate here is +3.4e-10 rad/s: 0.16 km cross-track. J5, J6: a few tenths of a km.

So omitting J3..J6 should cost **~1-2 km** (median) at 24 h - comparable to, not dominating, Cowell's
60 s truncation (1.9 km), and far above its 15 s truncation (~3e-3 km). In windows, 1.5 km along-track
is `1.5 / 6921 / 1.024e-3` = **~0.2 s** of rise/set shift - below the 1 s pad, with no pass gained or
lost. The measurement is recorded in `docs/architecture.md`'s zonal section.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import access, scenarios, sweep  # noqa: E402
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL  # noqa: E402
from orbital_engine.gravity import POINT_MASS_MODEL  # noqa: E402
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.zonal import EARTH_J3, EARTH_J4, EARTH_J5, EARTH_J6, EARTH_ZONALS, ZONAL_MODEL  # noqa: E402

HORIZON_S = 86400.0
STATIONS = [
    access.GroundStation("Kiruna", math.radians(67.86), math.radians(20.96), 0.40),
    access.GroundStation("Wallops", math.radians(37.94), math.radians(-75.46), 0.01),
    access.GroundStation("Santiago", math.radians(-33.15), math.radians(-70.67), 0.73),
]


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build_scenario() -> Simulation:
    sim = scenarios.earth_constellation(
        fresh_session(), n_sats=12, n_planes=1, altitude_km=550.0, inclination_deg=53.0
    )
    sim.record_history = False
    return sim


def build_configs() -> List[sweep.ModelConfig]:
    j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
    zonal = {"r_eq": EARTH_R_EQ, **EARTH_ZONALS}
    configs: List[sweep.ModelConfig] = []
    for dt in (60.0, 15.0):
        configs.append(sweep.ModelConfig(
            name=f"Cowell + j2 ({dt:.0f} s)", propagator=PropagatorType.COWELL, dt=dt,
            force_models=(sweep.ForceModelSpec(POINT_MASS_MODEL), sweep.ForceModelSpec(J2_MODEL, j2))))
        configs.append(sweep.ModelConfig(
            name=f"Cowell + j2 + zonal ({dt:.0f} s)", propagator=PropagatorType.COWELL, dt=dt,
            force_models=(sweep.ForceModelSpec(POINT_MASS_MODEL), sweep.ForceModelSpec(J2_MODEL, j2),
                          sweep.ForceModelSpec(ZONAL_MODEL, zonal))))
    configs.append(sweep.ModelConfig(
        name="Secular J2 (mean-seeded)", propagator=PropagatorType.SECULAR_J2, dt=60.0,
        mean_seed=True, propagator_coefficients=j2))
    return configs


def main() -> None:
    spec = access.AccessSpec(
        stations=STATIONS, central_body="Earth", omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS, mask_angle_rad=math.radians(5.0), sample_dt_s=60.0,
    )
    results = sweep.run_sweep(
        build_scenario, build_configs(), HORIZON_S,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
        zonal={"Earth": (EARTH_R_EQ, {3: EARTH_J3, 4: EARTH_J4, 5: EARTH_J5, 6: EARTH_J6})},
        timing_batches=1, timing_warmup=0, access=spec,
    )
    print("truth: DOP853, J2 + J3..J6 (EGM96), 12 sats at 550 km / 53 deg, 24 h")
    for r in results:
        print(f"{r.config_name:<34} median {r.error.median_km:9.4f} km  rms {r.error.rms_km:9.4f} km  "
              f"max {r.error.max_km:9.4f} km")
    for r in results:
        if r.access is not None:
            print("  " + access.format_metrics(r.config_name, r.access))


if __name__ == "__main__":
    main()
