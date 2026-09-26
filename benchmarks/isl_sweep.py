"""
Which propagation model keeps inter-satellite links right? The `isl=` headline, via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/isl_sweep.py

Prints one line per tier: position error at 24 h against a DOP853 truth carrying J2..J6, then the
**ground-station** access metrics and the **inter-satellite link** metrics for the same tier from the
same sweep (`access.format_metrics` for both). No figure, no file written. Takes ~15 s with numba.

**Scenario.** 12 satellites at 550 km / 53 deg in **3 planes of 4** (`scenarios.earth_constellation`,
`n_planes=3`: RAANs 0/120/240 deg, the same in-plane phasing 0/90/180/270 deg in every plane) - the
ground-track figure's constellation, not the access figure's single plane of 12: in one plane every
pair keeps a fixed phase, so a link is either always up or never and there would be no window to
time. Stations, mask and grid are the access figure's (Kiruna, Wallops, Santiago, 5 deg, 60 s), so
the ground and ISL columns come from one truth and one sweep. ISL: `h_graze_km = 100`, no range
limit, all 66 pairs. In-plane neighbours are 90 deg apart, beyond the 41.55 deg the 100 km grazing
sphere allows (`2 acos(6471/6921)`), so only the 48 cross-plane pairs ever link.

**Tiers.** Kepler, secular J2 (mean-seeded), Cowell + j2 at 60 s, Cowell + j2 + zonal at 60 s.

---

**Estimate, written before the first run of the ISL metric.** Inputs: each tier's error at 24 h
decomposed in the truth's RSW frame on this constellation (measured by position only, before any
window was computed):

| Tier | along-track, common / spread (end) | day-mean common / spread | radial spread | cross-track spread |
|---|---|---|---|---|
| Kepler | -75.7 / 853 km | 40.6 / 450 km | 11.7 km | 306 km |
| Secular J2, mean-seeded | 2.3 / 3.4 km | 0.80 / 6.5 km | 4.9 km (day-mean \|R\| 3.3) | 1.9 km |
| Cowell + j2, 60 s | 2.19 / 0.94 km | 0.88 / 0.44 km | 0.23 km | 0.18 km |
| Cowell + j2 + zonal, 60 s | 1.877 / 0.015 km | 0.71 / 0.006 km | 0.001 km | 0.001 km |

Three facts about the link geometry turn these into window shifts:

1. **A common rotation of the whole constellation about Earth's centre is invisible to an ISL** - the
   clearance depends on the two positions only and the body is a sphere. Kepler's missing nodal
   regression (the 306 km cross-track spread: one rotation about z, seen differently from each
   satellite) therefore costs the ISL metric *nothing*, while it moves every ground pass.
2. **An along-track error common to both ends is exactly a time translation** of the pair geometry
   (equal-radius circular orbits: the configuration depends only on the two arguments of latitude),
   so it shifts every window by `-delta_s / v`, `v = r n = 7.589 km/s`: **0.132 s per km**, the same
   sign on every window, and no duration change. For an approach symmetric between the two ends a
   *differential* error enters as the pair average, `-(delta_s_a + delta_s_b) / (2 v)`.
3. **Radial error** enters through `dc/dr_a = dc/dr_b = cos(phi*/2) / 2 = 0.467` (equal radii, at the
   critical central angle `phi* = 41.55 deg`), against a clearance rate
   `|c'| = (a/2) sin(phi*/2) |phi'| = 1227 km x |phi'|`, with `|phi'|` between `n` and ~`1.6 n` for
   these ~88 deg mutual inclinations: `|c'| ~ 1.3-2.2 km/s`, **~0.3 s per km** of radial pair sum.

Per tier, then:

- **Cowell + j2 + zonal, 60 s** is RK4's phase lag and nothing else - common along-track, spread
  0.015 km. Prediction: every ISL window moves **early** (the model is ahead) by the day-mean
  `0.71 / 7.589` = **0.094 s mean |shift|, signed mean ~ -0.094 s, max ~ 1.877 / 7.589 = 0.25 s**,
  duration error ~0, nothing lost or gained. The sharpest prediction here.
- **Cowell + j2, 60 s** adds the missing J3..J6: 0.44 km differential along-track and 0.12 km
  radial (day-mean) on top of the same lag: **~0.15 s mean, ~0.4 s max**, nothing lost or gained.
- **Secular J2 (mean-seeded)** misses J2's short-period terms: ~3.3 km radial per satellite ->
  `0.467 x 3.3 x sqrt(2) / 1.6` ~ 1.4 s, plus the 6.5 km along-track spread as a pair average
  ~0.6 s: **~1.5-2 s mean, ~5-8 s max** - about the ground figure (1.6 s on the single-plane
  constellation), with the odd marginal window lost or gained.
- **Kepler**: its 853 km along-track spread is the osculating-vs-mean seed offset, alternating with
  the seed's `cos 2u` - satellites 0/2 of each plane one way, 1/3 the other - so half the cross-plane
  pairs see the two ends' errors add and half see them cancel: **~0.5 x 450 / 7.589 ~ 30 s mean,
  ~110 s max**. On the ground every satellite carries its own full along-track error (~60 s mean)
  **plus** the nodal regression the ISL cannot see, so Kepler's ISL error should be **about half its
  ground error or less** - the one tier where ISL and ground disagree about the model. Windows lost
  and gained: likely, wherever a short window's shift exceeds its length.

Residual edge-interpolation bias after the shared grid cancels it: `C Delta h` with
`C = |f'' / (2 f')|` ~ 1e-3 /s here (the coplanar test's 8.6e-5 /s scaled by the ~11x faster
relative phase rate): ~0.015 s at `Delta = 0.25 s`, `h = 60 s` - below the finest tier's signal.

The measurement is recorded in `docs/architecture.md`'s ISL section.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import access, isl, scenarios, sweep  # noqa: E402
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL  # noqa: E402
from orbital_engine.gravity import POINT_MASS_MODEL  # noqa: E402
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.zonal import EARTH_J3, EARTH_J4, EARTH_J5, EARTH_J6, EARTH_ZONALS, ZONAL_MODEL  # noqa: E402

HORIZON_S = 86400.0
SAMPLE_DT_S = 60.0
H_GRAZE_KM = 100.0
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
        fresh_session(), n_sats=12, n_planes=3, altitude_km=550.0, inclination_deg=53.0
    )
    sim.record_history = False
    return sim


def build_configs() -> List[sweep.ModelConfig]:
    j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
    zonal = {"r_eq": EARTH_R_EQ, **EARTH_ZONALS}
    pm_j2 = (sweep.ForceModelSpec(POINT_MASS_MODEL), sweep.ForceModelSpec(J2_MODEL, j2))
    return [
        sweep.ModelConfig(name="Kepler", propagator=PropagatorType.KEPLERIAN, dt=60.0),
        sweep.ModelConfig(
            name="Secular J2 (mean-seeded)", propagator=PropagatorType.SECULAR_J2, dt=60.0,
            mean_seed=True, propagator_coefficients=j2),
        sweep.ModelConfig(
            name="Cowell + j2 (60 s)", propagator=PropagatorType.COWELL, dt=60.0, force_models=pm_j2),
        sweep.ModelConfig(
            name="Cowell + j2 + zonal (60 s)", propagator=PropagatorType.COWELL, dt=60.0,
            force_models=pm_j2 + (sweep.ForceModelSpec(ZONAL_MODEL, zonal),)),
    ]


def main() -> None:
    access_spec = access.AccessSpec(
        stations=STATIONS, central_body="Earth", omega=EARTH_OMEGA,
        body_radius_km=scenarios.EARTH_RADIUS, mask_angle_rad=math.radians(5.0),
        sample_dt_s=SAMPLE_DT_S,
    )
    isl_spec = isl.IslSpec(
        central_body="Earth", body_radius_km=scenarios.EARTH_RADIUS, h_graze_km=H_GRAZE_KM,
        sample_dt_s=SAMPLE_DT_S,
    )
    start = time.perf_counter()
    results = sweep.run_sweep(
        build_scenario, build_configs(), HORIZON_S,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
        zonal={"Earth": (EARTH_R_EQ, {3: EARTH_J3, 4: EARTH_J4, 5: EARTH_J5, 6: EARTH_J6})},
        timing_batches=3, timing_warmup=0, access=access_spec, isl=isl_spec,
    )
    elapsed = time.perf_counter() - start
    print("truth: DOP853, J2 + J3..J6 (EGM96), 12 sats at 550 km / 53 deg in 3 planes of 4, 24 h; "
          f"ISL h_graze = {H_GRAZE_KM:.0f} km, no range limit; sweep took {elapsed:.1f} s")
    for r in results:
        print(f"{r.config_name:<30} median {r.error.median_km:9.4f} km  max {r.error.max_km:9.4f} km")
    print("\nground stations (3 sites, 5 deg mask):")
    for r in results:
        if r.access is not None:
            print("  " + access.format_metrics(r.config_name, r.access))
    print("\ninter-satellite links (all pairs):")
    for r in results:
        if r.isl is not None:
            m = r.isl
            print("  " + access.format_metrics(r.config_name, m)
                  + f"  set {m.set.mean_s:+.3f}s (|.| {m.set.mean_abs_s:.3f}/{m.set.max_abs_s:.3f}s)")
    print("\nISL summary: tier | windows | mean |rise| | max |rise| | mean |set| | max |set| "
          "| lost / gained | contact error | ground mean |rise| (ratio ISL / ground)")
    for r in results:
        if r.isl is None or r.access is None:
            continue
        m, g = r.isl, r.access
        ratio = m.rise.mean_abs_s / g.rise.mean_abs_s if g.rise.mean_abs_s > 0 else math.nan
        print(f"  {r.config_name:<28} {m.n_truth_windows:>4} {m.rise.mean_abs_s:9.3f} "
              f"{m.rise.max_abs_s:9.3f} {m.set.mean_abs_s:9.3f} {m.set.max_abs_s:9.3f}   "
              f"{m.passes_lost:>2} / {m.passes_gained:<2} {m.total_contact_error_s:+10.2f} s   "
              f"{g.rise.mean_abs_s:8.3f} ({ratio:.2f})")


if __name__ == "__main__":
    main()
