"""
The atmosphere-model choice against the solar-activity assumption, in Delta-v - the NRLMSIS headline,
via `orbital_engine.sweep`.

Run with:

    <env>/python.exe benchmarks/msis_sweep.py

Needs the `[msis]` extra (`pymsis`). Prints one line per tier: raises, steady-rate Delta-v (m/s/day,
first raise excluded), total Delta-v, and the signed steady-rate error against the baseline next to
the prediction. No figure, no file written. Takes a few minutes: five configurations, each propagated
twice (the sweep's error run and its station-keeping run) for 6 days at 30 s.

**Scenario** is `tests/validation/test_sweep_delta_v.py`'s: one satellite from
`scenarios.station_keeping_satellites` (298.85 km osculating seed, mean 292.0 km, 51.6 deg, Cowell +
point mass + J2), band [291, 293.5] km, two-impulse Hohmann raises, B = 0.05 m^2/kg, co-rotating
atmosphere. **Baseline: MSIS at moderate activity.**

**Tiers.** NRLMSIS 2.0 (`msis_bridge.py`) at the ECSS-E-ST-10-04C / ISO 14222 long-term levels as
recalled (from memory, unverified): low F10.7 = F10.7a = 65, Ap = 0; moderate 140/140/15; high
250/250/45. Plus Vallado's layered table and the single band matched to it at 355 km with H = 60 km.

**Prediction, written before the first run** (`tests/validation/test_msis_delta_v.py`'s cycle model -
orbit-averaged decay in a co-rotating atmosphere, density convexity with each law's own local scale
height, the Hohmann transfer phase): steady-rate error against MSIS moderate

    MSIS low      -0.7196     table (layered)   +0.1288
    MSIS high     +1.3443     single @355 km    -0.0311

The horizon is 6 days so that MSIS low (40.7 h per cycle, first raise after ~16 h) completes three
full cycles. The measurement is recorded in `docs/architecture.md`, "NRLMSIS 2.0".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from orbital_engine import scenarios  # noqa: E402
from orbital_engine.atmosphere import (  # noqa: E402
    DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, layered_density,
)
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.database import Base  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_R_EQ  # noqa: E402
from orbital_engine.msis_bridge import (  # noqa: E402
    SOLAR_ACTIVITY_HIGH, SOLAR_ACTIVITY_LOW, SOLAR_ACTIVITY_MODERATE, msis_coefficients,
)
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.stationkeeping import StationKeepingSpec  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, run_sweep  # noqa: E402

B = 0.05
DT_S = 30.0
HORIZON_S = 6.0 * 86400.0
SPEC = StationKeepingSpec(291.0, 293.5)
PREDICTED = {"msis low": -0.7196, "msis high": 1.3443, "layered": 0.1288, "single@355": -0.0311}


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build() -> Simulation:
    return scenarios.station_keeping_satellites(_session(), n_sats=1, altitude_km=298.85,
                                                inclination_deg=51.6)


def drag(**law: float) -> ForceModelSpec:
    return ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": B, "r_ref": EARTH_R_EQ,
                                       "omega": EARTH_OMEGA, **law})


def main() -> None:
    rho355 = float(layered_density(np.array([355.0]))[0])
    configs = [
        ModelConfig("msis moderate", PropagatorType.COWELL, DT_S,
                    force_models=(drag(**msis_coefficients(SOLAR_ACTIVITY_MODERATE)),)),
        ModelConfig("msis low", PropagatorType.COWELL, DT_S,
                    force_models=(drag(**msis_coefficients(SOLAR_ACTIVITY_LOW)),)),
        ModelConfig("msis high", PropagatorType.COWELL, DT_S,
                    force_models=(drag(**msis_coefficients(SOLAR_ACTIVITY_HIGH)),)),
        ModelConfig("layered", PropagatorType.COWELL, DT_S,
                    force_models=(drag(density_model=DENSITY_MODEL_LAYERED),)),
        ModelConfig("single@355", PropagatorType.COWELL, DT_S, force_models=(drag(
            density_model=DENSITY_MODEL_EXPONENTIAL, rho0=rho355, h0=355.0, scale_height=60.0),)),
    ]
    start = time.perf_counter()
    results = run_sweep(build, configs, HORIZON_S, timing_batches=1, timing_warmup=0,
                        station_keeping=SPEC, delta_v_baseline="msis moderate")
    print(f"6 days, dt = {DT_S:.0f} s, band [{SPEC.lower_km}, {SPEC.upper_km}] km, B = {B} m^2/kg, "
          f"baseline MSIS moderate ({time.perf_counter() - start:.0f} s)")
    print(f"{'tier':<15}{'raises':>7}{'m/s/day':>10}{'total m/s':>11}{'rate err':>10}{'predicted':>11}")
    for r in results:
        dv = r.delta_v
        assert dv is not None
        body = dv.bodies[0]
        predicted = PREDICTED.get(r.config_name)
        shown = "baseline" if predicted is None else f"{predicted:+.4f}"
        print(f"{r.config_name:<15}{body.n_raises:>7}{dv.median_steady_rate_m_s_per_day:>10.4f}"
              f"{dv.median_total_dv_m_s:>11.3f}{dv.rate_error_rel:>+10.4f}{shown:>11}")


if __name__ == "__main__":
    main()
