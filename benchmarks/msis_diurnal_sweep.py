"""
Does the averaged NRLMSIS profile mislead the Delta-v budget of a sun-synchronous orbit? The diurnal
NRLMSIS headline, via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/msis_diurnal_sweep.py            # predictions, then the sweeps
    <env>/python.exe benchmarks/msis_diurnal_sweep.py --predict  # predictions only (~1 min)

Needs the `[msis]` extra. No figure, no file written.

**Scenario.** `scenarios.sun_synchronous_satellites`: a dawn-dusk (LTAN 18 h) and a noon-midnight
(LTAN 12 h) satellite, osculating-circular seed 297.0 km (mean ~292 km), 96.64 deg, Cowell + point mass
+ J2 so each plane precesses with the mean Sun. Station keeping exactly as `benchmarks/msis_sweep.py`:
band [291, 293.5] km, two-impulse raises, B = 0.05 m^2/kg, co-rotating atmosphere, dt = 30 s, 6 days -
so the numbers sit next to that sweep's solar-activity and atmosphere-model results at the same height.

**Tiers**, at ECSS moderate activity (140/140/15): the averaged profile (`DENSITY_MODEL_MSIS`, the
baseline) and the diurnal law (`DENSITY_MODEL_MSIS_DIURNAL`) with its table built for the epoch's day.
Two epochs, because the diurnal table also carries the **season** the average removes:
2024-03-20 (March equinox, near the semi-annual maximum) and 2024-07-01 (near the minimum).

**Prediction** (printed first, from the table alone): `tests/validation/test_msis_delta_v.py`'s cycle
model with each law's orbit-averaged, co-rotation-weighted density along the plane's own track, and
the decomposition `diurnal / averaged = (epoch-day global mean / annual mean) x (orbit / epoch-day
global mean)` - season times local time. Recorded before the sweeps ran:

    2024-03-20  season 1.143   dawn-dusk +0.0750 (LT factor 0.941)   noon-midnight +0.1516 (1.008)
    2024-07-01  season 0.811   dawn-dusk -0.2171 (LT factor 0.964)   noon-midnight -0.1548 (1.040)

The measurement is recorded in `docs/architecture.md`, "NRLMSIS 2.0 with the diurnal bulge".
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import scenarios, solar_ephemeris as se  # noqa: E402
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ  # noqa: E402
from orbital_engine.msis_bridge import (  # noqa: E402
    SOLAR_ACTIVITY_MODERATE, msis_coefficients, msis_profile,
)
from orbital_engine.msis_diurnal import (  # noqa: E402
    DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS, msis_diurnal_coefficients, msis_diurnal_table,
    msis_zonal_mean_density,
)
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.stationkeeping import StationKeepingSpec  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, SweepResult, run_sweep  # noqa: E402

MU = scenarios.MU_EARTH
B = 0.05
DT_S = 30.0
HORIZON_S = 6.0 * 86400.0
SEED_KM = 297.0
LOWER_KM, UPPER_KM = 291.0, 293.5
CENTRE_KM = 0.5 * (LOWER_KM + UPPER_KM)
SPEC = StationKeepingSpec(LOWER_KM, UPPER_KM)
LTANS = (18.0, 12.0)
NAMES = ("dawn-dusk", "noon-midnight")
EPOCHS = ("2024-03-20T00:00", "2024-07-01T00:00")
ACTIVITY = SOLAR_ACTIVITY_MODERATE
MOD = (ACTIVITY["f107"], ACTIVITY["f107a"], ACTIVITY["ap"])
VARIANCE_KM2 = 15.6          # test_msis_delta_v.py's <dh^2>; its effect on a ratio is ~1e-4


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def builder(epoch: float):  # type: ignore[no-untyped-def]
    def build() -> Simulation:
        return scenarios.sun_synchronous_satellites(_session(), epoch_days=epoch, ltan_hours=LTANS,
                                                    altitude_km=SEED_KM)
    return build


def configs(epoch: float) -> List[ModelConfig]:
    def drag(coefficients: Dict[str, float]) -> ForceModelSpec:
        return ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": B, "r_ref": EARTH_R_EQ,
                                           "omega": EARTH_OMEGA, **coefficients})
    return [
        ModelConfig("msis averaged", PropagatorType.COWELL, DT_S,
                    force_models=(drag(msis_coefficients(ACTIVITY)),)),
        ModelConfig("msis diurnal", PropagatorType.COWELL, DT_S,
                    force_models=(drag(msis_diurnal_coefficients(ACTIVITY, epoch)),)),
    ]


# ----------------------------------------------------------------------------------------------
# Prediction from the table alone
# ----------------------------------------------------------------------------------------------

def _track(a: float, node_deg: float, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inc = math.radians(scenarios.sun_synchronous_inclination_deg(EARTH_R_EQ + CENTRE_KM))
    node = math.radians(node_deg)
    cu, su = np.cos(u), np.sin(u)
    p = np.stack([cu, su * math.cos(inc), su * math.sin(inc)], axis=-1)
    q = np.stack([-su, cu * math.cos(inc), cu * math.sin(inc)], axis=-1)
    rot = np.array([[math.cos(node), -math.sin(node), 0.0], [math.sin(node), math.cos(node), 0.0],
                    [0.0, 0.0, 1.0]])
    return a * p @ rot.T, math.sqrt(MU / a) * q @ rot.T


def _hdot(law: str, epoch: float, h: np.ndarray, ltan: float) -> np.ndarray:
    table = msis_diurnal_table(*MOD, epoch)
    profile = msis_profile(*MOD)
    sun = float(se.sun_mean_longitude_deg(epoch))
    u = (np.arange(360) + 0.5) * 2.0 * math.pi / 360
    out = np.empty(h.size)
    for j, hj in enumerate(h):
        a = EARTH_R_EQ + hj
        r, v = _track(a, sun + 15.0 * (ltan - 12.0), u)
        v_rel = v - np.cross([0.0, 0.0, EARTH_OMEGA], r)
        w = np.linalg.norm(v_rel, axis=1) * np.einsum("ij,ij->i", v_rel, v)
        if law == "averaged":
            rho = np.full(u.size, np.exp(np.interp(hj, profile.altitude_km, np.log(profile.density_kg_m3))))
        else:
            lat = np.degrees(np.arctan2(r[:, 2], np.hypot(r[:, 0], r[:, 1])))
            rho = table.density_at(np.full(u.size, hj), lat, se.local_solar_time_hours(r, epoch))
        k = int(np.searchsorted(profile.altitude_km, hj, side="right") - 1)
        scale = profile.scale_height_km[k]
        out[j] = -(a * a / MU) * B * 1e3 * float(np.mean(rho * w)) * (1.0 + VARIANCE_KM2 / (2 * scale ** 2))
    return out


def t_cycle(law: str, epoch: float, ltan: float) -> float:
    h = np.linspace(LOWER_KM, UPPER_KM, 201)
    tau = math.pi * math.sqrt((EARTH_R_EQ + CENTRE_KM) ** 3 / MU)
    delta = abs(float(_hdot(law, epoch, np.array([CENTRE_KM]), ltan)[0])) * tau
    top = h <= UPPER_KM - delta
    y = 1.0 / np.abs(_hdot(law, epoch, h[top], ltan))
    return tau + float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(h[top])))


def season_factor(epoch: float, h: float = CENTRE_KM) -> float:
    """Epoch-day global mean over the annual-mean profile, at `h` (Simpson in latitude)."""
    lat = DIURNAL_LATITUDE_GRID_DEG
    s = np.ones(lat.size)
    s[1:-1:2], s[2:-1:2] = 4.0, 2.0
    w = np.cos(np.radians(lat)) * s
    w /= w.sum()
    z = msis_zonal_mean_density(np.array([h]), lat, DIURNAL_LST_GRID_HOURS, *MOD,
                                np.datetime64(se.J2000_UT + np.timedelta64(int(epoch * 86400e9), "ns"), "D"))
    profile = msis_profile(*MOD)
    return float(w @ z[0].mean(axis=1)) / float(np.exp(np.interp(h, profile.altitude_km,
                                                                  np.log(profile.density_kg_m3))))


def predictions() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for when in EPOCHS:
        epoch = se.epoch_days(when)
        row = {"season": season_factor(epoch)}
        for name, ltan in zip(NAMES, LTANS):
            row[name] = t_cycle("averaged", epoch, ltan) / t_cycle("diurnal", epoch, ltan) - 1.0
            density = float(np.mean(_hdot("diurnal", epoch, np.array([CENTRE_KM]), ltan))
                            / np.mean(_hdot("averaged", epoch, np.array([CENTRE_KM]), ltan)))
            row[name + " LT"] = density / row["season"]
        out[when] = row
    return out


def monthly_scan() -> None:
    """Table-free, cheap: the season factor and each plane's local-time factor through 2024."""
    print("month       season  dawn-dusk LT  noon-midnight LT   (292 km, orbit / epoch-day global mean)")
    profile = msis_profile(*MOD)
    rho_avg = float(np.exp(np.interp(CENTRE_KM, profile.altitude_km, np.log(profile.density_kg_m3))))
    lat, lst = DIURNAL_LATITUDE_GRID_DEG, DIURNAL_LST_GRID_HOURS
    for month in range(1, 13):
        when = f"2024-{month:02d}-15T00:00"
        epoch = se.epoch_days(when)
        z = np.log(msis_zonal_mean_density(np.array([CENTRE_KM]), lat, lst, *MOD,
                                           np.datetime64(when[:10]))[0])
        s = np.ones(lat.size)
        s[1:-1:2], s[2:-1:2] = 4.0, 2.0
        wlat = np.cos(np.radians(lat)) * s
        wlat /= wlat.sum()
        glob = float(wlat @ np.exp(z).mean(axis=1))
        sun = float(se.sun_mean_longitude_deg(epoch))
        u = (np.arange(360) + 0.5) * 2.0 * math.pi / 360
        factors = []
        for ltan in LTANS:
            r, v = _track(EARTH_R_EQ + CENTRE_KM, sun + 15.0 * (ltan - 12.0), u)
            la = np.degrees(np.arctan2(r[:, 2], np.hypot(r[:, 0], r[:, 1])))
            ls = se.local_solar_time_hours(r, epoch)
            j = np.clip(np.floor((la + 90.0) / 5.0).astype(int), 0, 35)
            wl = (la + 90.0) / 5.0 - j
            i0 = np.floor(ls / 0.5).astype(int) % 48
            ws = ls / 0.5 - np.floor(ls / 0.5)
            i1 = (i0 + 1) % 48
            f = ((1 - wl) * ((1 - ws) * z[j, i0] + ws * z[j, i1])
                 + wl * ((1 - ws) * z[j + 1, i0] + ws * z[j + 1, i1]))
            factors.append(float(np.mean(np.exp(f))) / glob)
        print(f"{when[:7]}    {glob / rho_avg:6.3f}  {factors[0]:12.3f}  {factors[1]:16.3f}")


def main() -> None:
    start = time.perf_counter()
    pred = predictions()
    print(f"Predictions ({time.perf_counter() - start:.0f} s): steady-rate error of the diurnal law "
          f"against the averaged profile")
    for when, row in pred.items():
        print(f"  {when[:10]}  season {row['season']:.3f}   "
              + "   ".join(f"{n} {row[n]:+.4f} (LT factor {row[n + ' LT']:.3f})" for n in NAMES))
    monthly_scan()
    if "--predict" in sys.argv:
        return
    for when in EPOCHS:
        epoch = se.epoch_days(when)
        t0 = time.perf_counter()
        results: List[SweepResult] = run_sweep(
            builder(epoch), configs(epoch), HORIZON_S, timing_batches=1, timing_warmup=0,
            oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)}, station_keeping=SPEC,
            delta_v_baseline="msis averaged")
        base, diurnal = (r.delta_v for r in results)
        assert base is not None and diurnal is not None
        print(f"\n{when[:10]}: 6 days, dt = {DT_S:.0f} s, band [{LOWER_KM}, {UPPER_KM}] km, B = {B} "
              f"({time.perf_counter() - t0:.0f} s)")
        print(f"{'plane':<15}{'avg m/s/day':>12}{'raises':>7}{'diurnal':>10}{'raises':>7}"
              f"{'measured':>10}{'predicted':>11}")
        for k, name in enumerate(NAMES):
            b, d = base.bodies[k], diurnal.bodies[k]
            measured = d.steady_rate_m_s_per_day / b.steady_rate_m_s_per_day - 1.0
            print(f"{name:<15}{b.steady_rate_m_s_per_day:>12.4f}{b.n_raises:>7}"
                  f"{d.steady_rate_m_s_per_day:>10.4f}{d.n_raises:>7}{measured:>+10.4f}"
                  f"{pred[when][name]:>+11.4f}")


if __name__ == "__main__":
    main()
