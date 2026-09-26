"""
The `"tesseral"` headline: what the longitude-dependent field costs a geostationary slot, and whether
a LEO constellation's contact windows notice it.

Run with:

    <env>/python.exe benchmarks/tesseral_sweep.py          # GEO scan only, ~15 s
    <env>/python.exe benchmarks/tesseral_sweep.py --leo    # plus the LEO sweep, several minutes

No figure, no file written. Numbers are recorded in `docs/architecture.md`'s tesseral section.

**Part 1 - GEO east-west station keeping versus slot longitude.** 24 slots every 15 deg, each twice
(J22 alone and the full 4x4 EGM96 field), plus a pm + J2 control, all Cowell + pm + J2, seeded at rest
in the rotating frame (`scenarios.geostationary_satellites`), 6 sidereal days at 144 steps per day. The
drift acceleration is the curvature of a quadratic fitted to whole-sidereal-day means of body-fixed
longitude, minus the control's (RK4's energy drift, 8.8e-3 K at this step - `test_tesseral.py`).
Delta-v per year to hold the slot is `a |lambda_ddot| / 3` (Gauss: a mean tangential `a_S` changes the
drift rate by `-3 a_S / a`). Printed beside the closed forms `lambda_ddot = +18 omega^2 (R/a)^2 J22 sin
2(lambda - lambda22)` (J22) and `-3 a_S(lambda) / a` with the equatorial `P_nm(0)` sum (4x4).

*Estimate before the run* (derived in `tesseral.py`): J22 alone gives K = 3.976e-15 rad/s^2 =
1.70e-3 deg/day^2 at worst, **1.76 m/s/yr**, at lambda22 +- 45 deg (30 E, 120 E, 150 W, 60 W). J33's
equatorial term is ~14 % of J22's (15 x 3 x J33 (R/a)^3 against 3 x 2 x J22 (R/a)^2), so the 4x4
worst case should sit ~10-20 % higher, near 2 m/s/yr, and the four maxima should become unequal.

**Part 2 (--leo) - does omitting tesserals move LEO contact windows?** The `benchmarks/zonal_sweep.py`
setup (12 satellites, one plane, 550 km / 53 deg, 24 h; Kiruna, Wallops, Santiago, 5 deg mask, 60 s
grid) against a truth carrying J2 + J3..J6 + the 4x4 tesseral field, with Cowell + pm + j2 + zonal at
15 s with and without `"tesseral"`. The tesseral tier runs the NumPy Cowell path (no compiled twin),
so its wall time is not comparable with the fused tier's.

*Estimate before the run* (circular, a = 6921 km, n = 1.10e-3 rad/s): the osculating seed is shared,
and J22's short-period variation of `a` (the J2 analogue scaled by J22/J2 ~ 1.7e-3: ~2 a J22 (R/a)^2
~ 0.02 km) puts each satellite's mean `a` up to ~3e-6 off, a drift `(3/2) n t da/a` of **up to ~2-3 km
along-track in a day, sign and size varying with the seed's phase** - the same mechanism as J3/J4's
~1 km in the zonal sweep; the orbit-averaged J22 term adds a bounded daily along-track oscillation of
`6 n J22 (R/a)^2 F / (2 omega)` ~ 3e-5 rad = **~0.2 km**. So ~1 km median, ~0.13 s of rise/set shift
per km (7.6 km/s ground-projected), no pass gained or lost.

*Measured* (recorded in `docs/architecture.md`): GEO as estimated - 1.764 m/s/yr under J22, 2.066 at
117.4 E under 4x4. LEO **3.8 km median / 9.0 km max, 0.29 s mean / 1.15 s max rise shift**, no pass
lost - the mechanism above, but 3x the magnitude: the first-orbit mean-`a` offset reaches 0.068 km
(J31's short-period `a` term is as large as J22's at LEO, and the pairs add) and predicts each
satellite's along-track error with correlation 0.9992.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import access, scenarios, sweep, tesseral  # noqa: E402
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL  # noqa: E402
from orbital_engine.gravity import POINT_MASS_MODEL  # noqa: E402
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.tesseral import EARTH_TESSERALS, TESSERAL_MODEL, TESSERAL_PAIRS  # noqa: E402
from orbital_engine.zonal import EARTH_J3, EARTH_J4, EARTH_J5, EARTH_J6, EARTH_ZONALS, ZONAL_MODEL  # noqa: E402

MU = scenarios.MU_EARTH
R = EARTH_R_EQ
GEO_A = scenarios.geostationary_radius_km()
SIDEREAL_S = 2.0 * math.pi / EARTH_OMEGA
STEPS_PER_DAY = 144
DAYS = 6
YEAR_S = 365.25 * 86400.0
CS = {(n, m): (EARTH_TESSERALS[f"c{n}{m}"], EARTH_TESSERALS[f"s{n}{m}"]) for n, m in TESSERAL_PAIRS}
J22, LAMBDA22 = tesseral.j22_amplitude_and_longitude(*CS[(2, 2)])
K_J22 = 18.0 * EARTH_OMEGA ** 2 * (R / GEO_A) ** 2 * J22


def fresh_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def p_nm_equator(n: int, m: int) -> float:
    if (n - m) % 2:
        return 0.0

    def dfact(k: int) -> int:
        return 1 if k <= 0 else k * dfact(k - 2)
    return float((-1) ** ((n - m) // 2) * dfact(n + m - 1) / dfact(n - m))


def full_lddot(lam: float) -> float:
    east = sum((MU / GEO_A ** 2) * (R / GEO_A) ** n * p_nm_equator(n, m) * m
               * (-c * math.sin(m * lam) + s * math.cos(m * lam)) for (n, m), (c, s) in CS.items())
    return -3.0 * east / GEO_A


def dv_per_year(lddot: float) -> float:
    return GEO_A * abs(lddot) / 3.0 * YEAR_S * 1e3


def window_fit(times: NDArray[np.float64], lon: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_win = (times.size - 1) // STEPS_PER_DAY
    tm = np.array([times[i * STEPS_PER_DAY:(i + 1) * STEPS_PER_DAY].mean() for i in range(n_win)])
    lm = np.array([lon[i * STEPS_PER_DAY:(i + 1) * STEPS_PER_DAY].mean(axis=0) for i in range(n_win)])
    centre = 0.5 * (tm[0] + tm[-1])
    coeffs = np.polyfit(tm - centre, lm, 2)
    return 2.0 * coeffs[0], coeffs[2]


def geo_scan() -> None:
    slots = [float(x) for x in np.arange(-180.0, 180.0, 15.0)]
    lons = slots + slots + [0.0]
    sim = scenarios.geostationary_satellites(fresh_session(), longitudes_deg=lons)
    sim.record_history = False
    sats = np.array([sim.name_to_index[scenarios.geo_satellite_name(k)] for k in range(len(lons))])
    n = len(slots)
    sim.enable_force_model(TESSERAL_MODEL, sats[:n], r_eq=R, omega=EARTH_OMEGA, **tesseral.EARTH_J22)
    sim.enable_force_model(TESSERAL_MODEL, sats[n:2 * n], r_eq=R, omega=EARTH_OMEGA, **EARTH_TESSERALS)
    earth = sim.name_to_index["Earth"]
    n_steps = DAYS * STEPS_PER_DAY
    dt = SIDEREAL_S / STEPS_PER_DAY
    times = np.empty(n_steps + 1)
    lon = np.empty((n_steps + 1, len(lons)))
    start = time.perf_counter()
    for k in range(n_steps + 1):
        if k:
            sim.step(dt)
        rel = sim.global_states[sats, :3] - sim.global_states[earth, :3]
        times[k] = sim.t
        lon[k] = np.arctan2(rel[:, 1], rel[:, 0]) - EARTH_OMEGA * sim.t
    wall = time.perf_counter() - start
    ddot, lam_c = window_fit(times, np.unwrap(lon, axis=0))
    control = float(ddot[-1])
    ddot = ddot - control

    print(f"GEO scan: {len(lons)} Cowell satellites, {DAYS} sidereal days at dt = {dt:.1f} s, wall {wall:.1f} s "
          f"(NumPy path); control (RK4) lambda_ddot {control:.4e} rad/s^2, subtracted")
    print(f"J22 = {J22:.5e}, lambda22 = {math.degrees(LAMBDA22):.3f} deg, K = {K_J22:.4e} rad/s^2 "
          f"= {K_J22 * 86400 ** 2 * 180 / math.pi:.4e} deg/day^2, a = {GEO_A:.3f} km")
    print(f"{'slot':>7} | {'J22 closed':>10} {'J22 meas':>10} | {'4x4 closed':>10} {'4x4 meas':>10}   m/s/yr"
          f"  | 4x4 lambda_ddot deg/day^2")
    worst: Dict[str, Tuple[float, float]] = {"j22": (0.0, 0.0), "full": (0.0, 0.0)}
    residual = 0.0
    for k, slot in enumerate(slots):
        c22 = K_J22 * math.sin(2.0 * (float(lam_c[k]) - LAMBDA22))
        cfull = full_lddot(float(lam_c[n + k]))
        residual = max(residual, abs(ddot[k] - c22) / K_J22, abs(ddot[n + k] - cfull) / K_J22)
        m22, mfull = dv_per_year(ddot[k]), dv_per_year(ddot[n + k])
        worst["j22"] = max(worst["j22"], (m22, slot))
        worst["full"] = max(worst["full"], (mfull, slot))
        print(f"{slot:7.1f} | {dv_per_year(c22):10.4f} {m22:10.4f} | {dv_per_year(cfull):10.4f} {mfull:10.4f}"
              f"          | {ddot[n + k] * 86400 ** 2 * 180 / math.pi:+.4e}")
    print(f"worst measured slot: J22 {worst['j22'][0]:.4f} m/s/yr at {worst['j22'][1]:.0f} deg, "
          f"4x4 {worst['full'][0]:.4f} m/s/yr at {worst['full'][1]:.0f} deg; "
          f"max |measured - closed form| {residual:.1e} K")

    from scipy.optimize import brentq
    grid = np.radians(np.arange(-180.0, 180.0, 0.01))
    vals = np.array([full_lddot(float(x)) for x in grid])
    for i in np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:])):
        x = brentq(full_lddot, grid[i], grid[i + 1], xtol=1e-13)
        print(f"  4x4 equilibrium {math.degrees(x):9.3f} deg  {'stable' if vals[i] > 0 else 'unstable'}")
    j = int(np.argmax(np.abs(vals)))
    print(f"  4x4 closed-form worst slot {math.degrees(grid[j]):.2f} deg: {dv_per_year(vals[j]):.4f} m/s/yr "
          f"(J22 alone {dv_per_year(K_J22):.4f})")


def build_leo() -> Simulation:
    sim = scenarios.earth_constellation(fresh_session(), n_sats=12, n_planes=1, altitude_km=550.0, inclination_deg=53.0)
    sim.record_history = False
    return sim


def leo_sweep() -> None:
    stations = [
        access.GroundStation("Kiruna", math.radians(67.86), math.radians(20.96), 0.40),
        access.GroundStation("Wallops", math.radians(37.94), math.radians(-75.46), 0.01),
        access.GroundStation("Santiago", math.radians(-33.15), math.radians(-70.67), 0.73),
    ]
    spec = access.AccessSpec(stations=stations, central_body="Earth", omega=EARTH_OMEGA,
                             body_radius_km=scenarios.EARTH_RADIUS, mask_angle_rad=math.radians(5.0),
                             sample_dt_s=60.0)
    j2 = {"j2": EARTH_J2, "r_eq": R}
    zonal = {"r_eq": R, **EARTH_ZONALS}
    tess = {"r_eq": R, "omega": EARTH_OMEGA, "theta0": 0.0, **EARTH_TESSERALS}
    base = (sweep.ForceModelSpec(POINT_MASS_MODEL), sweep.ForceModelSpec(J2_MODEL, j2),
            sweep.ForceModelSpec(ZONAL_MODEL, zonal))
    configs: List[sweep.ModelConfig] = [
        sweep.ModelConfig("Cowell + j2 + zonal (15 s)", PropagatorType.COWELL, 15.0, force_models=base),
        sweep.ModelConfig("Cowell + j2 + zonal + tesseral (15 s)", PropagatorType.COWELL, 15.0,
                          force_models=base + (sweep.ForceModelSpec(TESSERAL_MODEL, tess),)),
    ]
    start = time.perf_counter()
    results = sweep.run_sweep(
        build_leo, configs, 86400.0,
        oblateness={"Earth": (EARTH_J2, R)},
        zonal={"Earth": (R, {3: EARTH_J3, 4: EARTH_J4, 5: EARTH_J5, 6: EARTH_J6})},
        tesseral={"Earth": (R, EARTH_OMEGA, 0.0, CS)},
        timing_batches=1, timing_warmup=0, access=spec,
    )
    print(f"\nLEO: truth DOP853 J2 + J3..J6 + 4x4 tesseral, 12 sats 550 km / 53 deg, 24 h "
          f"(sweep wall {time.perf_counter() - start:.0f} s)")
    for r in results:
        print(f"{r.config_name:<40} median {r.error.median_km:9.4f} km  rms {r.error.rms_km:9.4f} km  "
              f"max {r.error.max_km:9.4f} km  wall {r.wall_time_us * 1e-6:8.3f} s")
    for r in results:
        if r.access is not None:
            print("  " + access.format_metrics(r.config_name, r.access))


if __name__ == "__main__":
    geo_scan()
    if "--leo" in sys.argv[1:]:
        leo_sweep()
