"""
The NRLMSIS headline in Delta-v: **how does the atmosphere-model choice compare with the
solar-activity assumption?** Through the sweep - `run_sweep(..., station_keeping=, delta_v_baseline=)`
- on `test_sweep_delta_v.py`'s scenario (one satellite, 298.85 km osculating seed, 51.6 deg, Cowell +
point mass + J2, band [291, 293.5] km, B = 0.05 m^2/kg, dt = 30 s), with **MSIS at moderate activity
as the baseline**.

The prediction, written before the first run
--------------------------------------------
Every tier shares the band, so the steady-rate ratio is the inverse ratio of cycle times, and a cycle
is `tau + integral dh / |hdot(h)|` with `hdot` the orbit-averaged decay of an inclined circular orbit in
a co-rotating atmosphere times the convexity factor `1 + <dh^2>/(2 H(h)^2)` - `test_sweep_delta_v.py`'s
model, restated below with the density law as a parameter; `H(h)` is each law's own local scale height
(the MSIS profile's band `H`). At the band the density ratios to MSIS moderate are low **0.280**, high
**2.341**, table **1.128**, single band at 355 km **0.969**, and the predicted steady-rate errors are:

    MSIS low (65/65/0)       -0.7196    (cycle 40.7 h)
    MSIS high (250/250/45)   +1.3443    (cycle 4.87 h)
    table (layered)          +0.1288    (cycle 10.11 h)
    single band at 355 km    -0.0311    (cycle 11.78 h)
    MSIS moderate            baseline   (cycle 11.41 h)

So the **solar-activity assumption moves the budget by -72 % / +134 %**, against **+13 % / -3 %** for
the two static laws - five to ten times the atmosphere-model choice. And the static laws straddle
moderate MSIS: the single band's "-14 %" of `test_sweep_delta_v.py` is measured against a table that
itself sits 13 % above moderate MSIS at 292 km.

*Tolerance.* As `test_sweep_delta_v.py`: over a short horizon a steady rate is one or a few cycles and
carries the single-cycle endpoint term, 4.5e-3 per rate, 9e-3 on a ratio in the worst case, so
`RATIO_TOL = 1e-2` - on `(1 + error)`, the ratio of the two rates, because MSIS high's error is +1.34
and a fixed absolute tolerance on it would be 2.3x looser than on the others.

This test runs 20 h - two raises for the slowest tier in it. MSIS low needs 40.7 h per cycle, so it is
**not** here; `benchmarks/msis_sweep.py` runs all five tiers over 6 days and the full numbers are in
`docs/architecture.md`.

**Measured** (20 h): MSIS high **+1.3432** (predicted +1.3443, 4 raises), table **+0.1311**
(+0.1288), single band **-0.0333** (-0.0311); on `(1 + error)` that is -4.5e-4, +2.0e-3 and -2.2e-3
against the 1e-2 budget. Baseline 3.051 m/s/day; the table's 3.4507 m/s/day is the same number
`test_sweep_delta_v.py` measures for it, as it must be - the tier is identical, only the baseline
moved.
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

pytest.importorskip("pymsis")

from orbital_engine import scenarios  # noqa: E402
from orbital_engine.atmosphere import (  # noqa: E402
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, SCALE_HEIGHT_KM,
    layered_density, piecewise_exponential_density,
)
from orbital_engine.custom_types import PropagatorType  # noqa: E402
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA  # noqa: E402
from orbital_engine.geopotential import EARTH_R_EQ  # noqa: E402
from orbital_engine.msis_bridge import (  # noqa: E402
    SOLAR_ACTIVITY_HIGH, SOLAR_ACTIVITY_LOW, SOLAR_ACTIVITY_MODERATE, msis_coefficients, msis_profile,
)
from orbital_engine.simulator import Simulation  # noqa: E402
from orbital_engine.stationkeeping import StationKeepingSpec  # noqa: E402
from orbital_engine.sweep import ForceModelSpec, ModelConfig, SweepResult, run_sweep  # noqa: E402

ArrF = NDArray[np.float64]
MU = scenarios.MU_EARTH
ALT_SEED_KM = 298.85
INC_DEG = 51.6
B = 0.05
LOWER_KM, UPPER_KM = 291.0, 293.5
CENTRE_KM = 0.5 * (LOWER_KM + UPPER_KM)
DT_S = 30.0
HORIZON_S = 20.0 * 3600.0
VARIANCE_KM2 = 15.6                  # the study's measured <dh^2>, test_stationkeeping.py
RATIO_TOL = 1e-2
SPEC = StationKeepingSpec(LOWER_KM, UPPER_KM)

ACTIVITY = {"msis low": SOLAR_ACTIVITY_LOW, "msis moderate": SOLAR_ACTIVITY_MODERATE,
            "msis high": SOLAR_ACTIVITY_HIGH}
PREDICTED = {"msis low": -0.7196, "msis high": 1.3443, "layered": 0.1288, "single@355": -0.0311}


def _drag(**law: float) -> ForceModelSpec:
    return ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": B, "r_ref": EARTH_R_EQ,
                                       "omega": EARTH_OMEGA, **law})


def _rho355() -> float:
    return float(layered_density(np.array([355.0]))[0])


def configs(names: tuple[str, ...]) -> list[ModelConfig]:
    table = {
        **{name: ModelConfig(name, PropagatorType.COWELL, DT_S,
                             force_models=(_drag(**msis_coefficients(act)),))
           for name, act in ACTIVITY.items()},
        "layered": ModelConfig("layered", PropagatorType.COWELL, DT_S,
                               force_models=(_drag(density_model=DENSITY_MODEL_LAYERED),)),
        "single@355": ModelConfig("single@355", PropagatorType.COWELL, DT_S, force_models=(_drag(
            density_model=DENSITY_MODEL_EXPONENTIAL, rho0=_rho355(), h0=355.0,
            scale_height=60.0),)),
    }
    return [table[n] for n in names]


def _fresh_session() -> Session:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build() -> Simulation:
    return scenarios.station_keeping_satellites(
        _fresh_session(), n_sats=1, altitude_km=ALT_SEED_KM, inclination_deg=INC_DEG)


# ==================================================================================================
# The independent prediction - test_sweep_delta_v.py's cycle model, density law as a parameter
# ==================================================================================================

def density_and_scale_height(model: str, h: ArrF) -> tuple[ArrF, ArrF]:
    if model in ACTIVITY:
        act = ACTIVITY[model]
        p = msis_profile(act["f107"], act["f107a"], act["ap"])
        band = np.searchsorted(p.altitude_km, h, side="right") - 1
        return (piecewise_exponential_density(h, p.altitude_km, p.density_kg_m3, p.scale_height_km),
                p.scale_height_km[band])
    if model == "layered":
        return layered_density(h), SCALE_HEIGHT_KM[np.searchsorted(BASE_ALTITUDE_KM, h, side="right") - 1]
    return _rho355() * np.exp(-(h - 355.0) / 60.0), np.full_like(h, 60.0)


def _hdot(model: str, h: ArrF) -> ArrF:
    a = EARTH_R_EQ + h
    v = np.sqrt(MU / a)
    cos_i, sin_i = math.cos(math.radians(INC_DEG)), math.sin(math.radians(INC_DEG))
    u = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    along = (v - EARTH_OMEGA * a * cos_i)[:, None]
    cross = (EARTH_OMEGA * a * sin_i)[:, None] * np.cos(u)[None, :]
    speed = np.mean(np.sqrt(along ** 2 + cross ** 2), axis=1)
    rho, scale = density_and_scale_height(model, h)
    kappa = 1.0 + VARIANCE_KM2 / (2.0 * scale ** 2)
    out: ArrF = -(a ** 2 / MU) * rho * (B * 1e3) * (v * v - v * EARTH_OMEGA * a * cos_i) * speed * kappa
    return out


def t_cycle(model: str) -> float:
    h = np.linspace(LOWER_KM, UPPER_KM, 2001)
    tau = math.pi * math.sqrt((EARTH_R_EQ + CENTRE_KM) ** 3 / MU)
    delta = abs(float(_hdot(model, np.array([CENTRE_KM]))[0])) * tau
    top = h <= UPPER_KM - delta
    y = 1.0 / np.abs(_hdot(model, h[top]))
    return tau + float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(h[top])))


def predicted_error(model: str) -> float:
    return t_cycle("msis moderate") / t_cycle(model) - 1.0


# ==================================================================================================

@pytest.fixture(scope="module")
def sweep() -> Dict[str, SweepResult]:
    names = ("msis moderate", "msis high", "layered", "single@355")
    results = run_sweep(build, configs(names), HORIZON_S, timing_batches=1, timing_warmup=0,
                        station_keeping=SPEC, delta_v_baseline="msis moderate")
    return {r.config_name: r for r in results}


def test_the_prediction_is_what_the_docstring_states() -> None:
    """The derivation on its own, so the numbers written before the first run stay pinned."""
    for model, value in PREDICTED.items():
        assert predicted_error(model) == pytest.approx(value, abs=5e-4), model
    assert t_cycle("msis low") / 3600.0 == pytest.approx(40.7, abs=0.1)


def test_every_tier_made_at_least_two_raises(sweep: Dict[str, SweepResult]) -> None:
    for name, result in sweep.items():
        dv = result.delta_v
        assert dv is not None and dv.bodies[0].n_raises >= 2, name
    base = sweep["msis moderate"].delta_v
    assert base is not None and base.rate_error_rel == 0.0


@pytest.mark.parametrize("name", ["msis high", "layered", "single@355"])
def test_steady_rate_error_matches_the_orbit_averaged_prediction(
    sweep: Dict[str, SweepResult], name: str,
) -> None:
    dv = sweep[name].delta_v
    assert dv is not None
    measured, predicted = dv.rate_error_rel, PREDICTED[name]
    assert (1.0 + measured) / (1.0 + predicted) - 1.0 == pytest.approx(0.0, abs=RATIO_TOL), (
        f"{name}: measured {measured:+.4f}, predicted {predicted:+.4f}")


def test_solar_activity_dominates_the_atmosphere_model_choice(sweep: Dict[str, SweepResult]) -> None:
    """The headline, as an inequality on measurements: the solar-activity swing (moderate to high)
    exceeds the table-versus-single-band swing by more than five times."""
    high, table, single = (sweep[n].delta_v for n in ("msis high", "layered", "single@355"))
    assert high is not None and table is not None and single is not None
    model_choice = abs(table.rate_error_rel - single.rate_error_rel)
    assert high.rate_error_rel > 5.0 * model_choice, (high.rate_error_rel, model_choice)
