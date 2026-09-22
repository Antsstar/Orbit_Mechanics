"""
Validation of the station-keeping Delta-v as a **sweep metric**: `run_sweep(..., station_keeping=spec,
delta_v_baseline=name)` -> `SweepResult.delta_v`.

The metric adds no physics and no controller logic: `sweep.station_keeping_for` builds a fresh
scenario, applies the config and calls `stationkeeping.run_station_keeping`, the function the
standalone study (`tests/validation/test_stationkeeping.py`, `benchmarks/figures.py`) calls. So the
validation has two halves.

1. **It is the study's computation.** Calling `run_station_keeping` directly on the same configuration
   in a fresh arena must give the same per-body totals, raise counts and relative errors - **bitwise**
   for the totals, and to rounding (1e-12) for the relative errors, which differ only in the m/s/day
   unit conversion applied before the division.

2. **The relative errors are the physics the study predicts**, derived before measuring.

Scenario: `scenarios.station_keeping_satellites(n_sats=1)`, the study's satellite (298.85 km osculating
seed, 51.6 deg, Cowell + point mass + J2, mean altitude 292.0 km), band `[291.0, 293.5]`, B = 0.05
m^2/kg, `dt = 30 s`, horizon **18 h** - just long enough for two raises of the slowest drag tier, which
is what a steady rate needs (`stationkeeping.steady_rate` drops the first raise).

The prediction
--------------
Every tier shares the band, so each cycle costs the same `dv_cycle = integral_L^U (n/2) dh` and the
steady-rate ratio is the inverse ratio of cycle times, `t_base / t_model`, with (as in the study)

    t_cycle = tau + integral_L^{U - delta} dh / |hdot(h)|,   delta = |hdot(mid)| tau,
    hdot    = hdot_drag(h) (1 + <delta_h^2> / (2 H(h)^2)),

`hdot_drag` the orbit-averaged decay of a circular orbit in a co-rotating atmosphere, `tau` half a
transfer orbit, and `<delta_h^2>` = 15.6 km^2 the study's measured altitude variance about the mean. RK4's
own decay (1.6e-4 of the rate at 30 s) is common to both tiers and cancels from the ratio below 1e-5.

    single band matched at 355 km, H = 60 km:     predicted -0.1416
    single band matched at the band centre:       predicted -0.0019

*Tolerance.* With two raises a steady rate is one cycle, `dv_2 / (t_2 - t_1)`. The study's endpoint
term - the difference between the altitude one raise achieved and the next was sized for, ~10 m of the
2.5 km band - was 1.5e-3 spread over three cycles, so **4.5e-3 per rate** over one, and 9e-3 on a ratio
of two in the worst case. The variance's ~10 % uncertainty moves the convexity terms apart by 1.6e-4;
the omitted density-velocity correlation and the drag-dependent RK4 residual differ between the tiers by
~1e-4 each. `RATIO_TOL = 1e-2`. The 355 km tier's signal is 14x that; the centre-anchored tier's is
below it, so for that tier the assertion is only that the sweep does not invent an error - its value is
pinned by half 1, not here.

Negative control
----------------
Each mutation was applied to the real source, this module run, and the file restored with
`git checkout --`.

- *Score against the wrong config* (`base = budgets[configs[0].name]` in `run_sweep`) - **4 of 10
  fail**: the direct-study equality, the baseline's exact zero, and both predicted relative errors.
- *Keep the start-up transient* (`steady_rate` sums every raise, not `[1:]`) - **1 fails**, the
  baseline's 3.436 m/s/day magnitude, which doubles. The relative errors do not move measurably: over
  18 h every drag tier's first raise is the same 1.45 m/s from the same seed, so it biases numerator
  and baseline alike. The magnitude assertion is the one that pins the definition.

Measured (18 h, one satellite): layered 3.4507 m/s/day; single band at 355 km **-0.1453** (predicted
-0.1416); at the band centre **-0.0030** (predicted -0.0019); drag-free -1.0 exactly. Total Delta-v
reads -1.0e-4 and -4.2e-4 for the same two tiers - both made two raises - which is why the steady
rate, not the total, is the headline on any horizon a few cycles long.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.access import AccessSpec, GroundStation
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, SCALE_HEIGHT_KM,
    layered_density,
)
from orbital_engine.custom_types import PropagatorType
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ
from orbital_engine.simulator import Simulation
from orbital_engine.stationkeeping import StationKeepingSpec, run_station_keeping
from orbital_engine.sweep import (
    ExternalTier, ForceModelSpec, ModelConfig, SweepResult, apply_config, run_sweep,
)

MU = scenarios.MU_EARTH
ALT_SEED_KM = 298.85
INC_DEG = 51.6
B = 0.05
LOWER_KM = 291.0
UPPER_KM = 293.5
DT_S = 30.0
HORIZON_S = 18.0 * 3600.0
MATCH_KM = 355.0
SINGLE_H_KM = 60.0
CENTRE_KM = 0.5 * (LOWER_KM + UPPER_KM)
VARIANCE_KM2 = 15.6                    # the study's measured <delta_h^2>, test_stationkeeping.py
RATIO_TOL = 1e-2

SPEC = StationKeepingSpec(LOWER_KM, UPPER_KM)
ArrF = NDArray[np.float64]


def _rho(h: float) -> float:
    return float(layered_density(np.array([h], dtype=np.float64))[0])


def _drag(**law: float) -> ForceModelSpec:
    return ForceModelSpec(DRAG_MODEL, {"ballistic_coeff": B, "r_ref": EARTH_R_EQ,
                                       "omega": EARTH_OMEGA, **law})


LAYERED = ModelConfig("layered", PropagatorType.COWELL, DT_S,
                      force_models=(_drag(density_model=DENSITY_MODEL_LAYERED),))
SINGLE_355 = ModelConfig("single@355", PropagatorType.COWELL, DT_S, force_models=(_drag(
    density_model=DENSITY_MODEL_EXPONENTIAL, rho0=_rho(MATCH_KM), h0=MATCH_KM,
    scale_height=SINGLE_H_KM),))
SINGLE_CENTRE = ModelConfig("single@centre", PropagatorType.COWELL, DT_S, force_models=(_drag(
    density_model=DENSITY_MODEL_EXPONENTIAL, rho0=_rho(CENTRE_KM), h0=CENTRE_KM,
    scale_height=SINGLE_H_KM),))
# The scenario already carries point mass + J2 on every satellite; these add nothing, so no drag.
NO_DRAG = ModelConfig("cowell, no drag", PropagatorType.COWELL, DT_S)
SECULAR = ModelConfig("secular J2", PropagatorType.SECULAR_J2, 60.0,
                      propagator_coefficients={"j2": EARTH_J2, "r_eq": EARTH_R_EQ})


def _fresh_session() -> Session:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _build() -> Simulation:
    return scenarios.station_keeping_satellites(
        _fresh_session(), n_sats=1, altitude_km=ALT_SEED_KM, inclination_deg=INC_DEG)


@pytest.fixture(scope="module")
def build() -> Callable[[], Simulation]:
    return _build


@pytest.fixture(scope="module")
def sweep(build: Callable[[], Simulation]) -> Dict[str, SweepResult]:
    results = run_sweep(
        build, [SINGLE_355, LAYERED, SINGLE_CENTRE, NO_DRAG, SECULAR], HORIZON_S,
        timing_batches=1, timing_warmup=0, station_keeping=SPEC, delta_v_baseline="layered",
    )
    return {r.config_name: r for r in results}


# ==================================================================================================
# The independent prediction
# ==================================================================================================

def _hdot(model: str, h: ArrF) -> ArrF:
    """Orbit-averaged da/dt (km/s) with the convexity factor - the study's form, restated."""
    a = EARTH_R_EQ + h
    v = np.sqrt(MU / a)
    cos_i, sin_i = math.cos(math.radians(INC_DEG)), math.sin(math.radians(INC_DEG))
    u = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    along = (v - EARTH_OMEGA * a * cos_i)[:, None]
    cross = (EARTH_OMEGA * a * sin_i)[:, None] * np.cos(u)[None, :]
    speed = np.mean(np.sqrt(along ** 2 + cross ** 2), axis=1)
    if model == "layered":
        rho = layered_density(h)
        scale = SCALE_HEIGHT_KM[np.searchsorted(BASE_ALTITUDE_KM, h, side="right") - 1]
    else:
        anchor = MATCH_KM if model == "single@355" else CENTRE_KM
        rho = _rho(anchor) * np.exp(-(h - anchor) / SINGLE_H_KM)
        scale = np.full_like(h, SINGLE_H_KM)
    kappa = 1.0 + VARIANCE_KM2 / (2.0 * scale ** 2)
    out: ArrF = -(a ** 2 / MU) * rho * (B * 1e3) * (v * v - v * EARTH_OMEGA * a * cos_i) * speed * kappa
    return out


def _t_cycle(model: str) -> float:
    h = np.linspace(LOWER_KM, UPPER_KM, 2001)
    a_t = EARTH_R_EQ + CENTRE_KM
    tau = math.pi * math.sqrt(a_t ** 3 / MU)
    delta = abs(float(_hdot(model, np.array([CENTRE_KM]))[0])) * tau
    top = h <= UPPER_KM - delta
    y = 1.0 / np.abs(_hdot(model, h[top]))
    return tau + float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(h[top])))


def _predicted_error(model: str) -> float:
    return _t_cycle("layered") / _t_cycle(model) - 1.0


# ==================================================================================================
# 1. It is the study's computation
# ==================================================================================================

def _direct_rate(build: Callable[[], Simulation], config: ModelConfig) -> tuple[float, float, int]:
    sim = build()
    sim.record_history = False
    idx = apply_config(sim, config)
    s = run_station_keeping(sim, idx, SPEC, HORIZON_S, config.dt).summary[0]
    return s.steady_rate_km_s_per_s, s.total_dv_km_s, s.n_burns


def test_sweep_reproduces_the_direct_study(
    sweep: Dict[str, SweepResult], build: Callable[[], Simulation],
) -> None:
    direct = {c.name: _direct_rate(build, c) for c in (LAYERED, SINGLE_355, SINGLE_CENTRE)}
    for name, (rate, total, n) in direct.items():
        dv = sweep[name].delta_v
        assert dv is not None
        assert dv.bodies[0].n_raises == n >= 2
        assert dv.bodies[0].total_dv_m_s == total * 1e3                # bitwise
        expected = rate / direct["layered"][0] - 1.0
        assert dv.rate_error_rel == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_baseline_scores_exactly_zero(sweep: Dict[str, SweepResult]) -> None:
    dv = sweep["layered"].delta_v
    assert dv is not None and dv.baseline == "layered"
    assert dv.rate_error_rel == 0.0 and dv.total_error_rel == 0.0
    # The study's headline magnitude, one cycle in: 3.436 m/s/day over 10 days. The single-cycle
    # endpoint term is 4.5e-3 (module docstring), convexity and the RK4 residual already inside it.
    assert dv.median_steady_rate_m_s_per_day == pytest.approx(3.436, rel=1e-2)


# ==================================================================================================
# 2. The relative errors are the predicted physics
# ==================================================================================================

@pytest.mark.parametrize("name", ["single@355", "single@centre"])
def test_relative_error_matches_the_orbit_averaged_prediction(
    sweep: Dict[str, SweepResult], name: str,
) -> None:
    dv = sweep[name].delta_v
    assert dv is not None
    predicted = _predicted_error(name)
    assert dv.rate_error_rel == pytest.approx(predicted, abs=RATIO_TOL), (dv.rate_error_rel, predicted)


def test_prediction_has_the_studys_size() -> None:
    """The derivation, checked on its own: -14.0 % and -0.14 %, the study's 10-day figures."""
    assert _predicted_error("single@355") == pytest.approx(-0.141, abs=2e-3)
    assert abs(_predicted_error("single@centre")) < 3e-3


@pytest.mark.parametrize("name", ["cowell, no drag", "secular J2"])
def test_a_drag_free_tier_under_budgets_by_exactly_100_percent(
    sweep: Dict[str, SweepResult], name: str,
) -> None:
    dv = sweep[name].delta_v
    assert dv is not None
    assert dv.bodies[0].n_raises == 0
    assert dv.median_total_dv_m_s == 0.0 and dv.median_steady_rate_m_s_per_day == 0.0
    assert dv.rate_error_rel == -1.0 and dv.total_error_rel == -1.0


# ==================================================================================================
# 3. Strictly additive, and refusals
# ==================================================================================================

def test_error_and_access_are_bit_identical_with_the_metric_on(build: Callable[[], Simulation]) -> None:
    """Two hours, so it is cheap. `wall_time_us` is a measurement and is never bit-identical between
    runs; it comes from `_time_propagation` by the same code with the metric on or off."""
    horizon = 7200.0
    access = AccessSpec(stations=[GroundStation("equator", 0.0, 0.0)], central_body="Earth",
                        omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ, sample_dt_s=60.0)
    stub = ExternalTier("stub", 60.0, ["SK-SAT-00"], "Earth",
                        lambda t: np.zeros((t.size, 1, 3), dtype=np.float64))
    common = dict(timing_batches=1, timing_warmup=0, access=access, external=[stub])
    configs = [LAYERED, NO_DRAG]
    off = run_sweep(build, configs, horizon, **common)  # type: ignore[arg-type]
    on = run_sweep(build, configs, horizon, station_keeping=SPEC, delta_v_baseline="layered",
                   **common)  # type: ignore[arg-type]
    for a, b in zip(off, on):
        assert a.config_name == b.config_name and a.n_bodies == b.n_bodies
        assert a.error == b.error
        assert a.access == b.access
        assert a.delta_v is None
    assert [r.delta_v is not None for r in on] == [True, True, False]   # the external tier: None


def test_refuses_a_missing_or_unknown_baseline(build: Callable[[], Simulation]) -> None:
    with pytest.raises(ValueError, match="delta_v_baseline"):
        run_sweep(build, [LAYERED], 3600.0, station_keeping=SPEC)
    with pytest.raises(ValueError, match="names no config"):
        run_sweep(build, [LAYERED], 3600.0, station_keeping=SPEC, delta_v_baseline="Layered")
    with pytest.raises(ValueError, match="without station_keeping"):
        run_sweep(build, [LAYERED], 3600.0, delta_v_baseline="layered")


def test_refuses_a_dt_the_controller_cannot_observe(build: Callable[[], Simulation]) -> None:
    """~5420 s window at 292 km; 1800 s is 3.0 samples, under `MIN_SAMPLES_PER_ORBIT` = 4. Raised
    before truth is integrated, naming the config; 1200 s (4.5 samples) is accepted by the check."""
    coarse = ModelConfig("coarse secular", PropagatorType.SECULAR_J2, 1800.0,
                         propagator_coefficients={"j2": EARTH_J2, "r_eq": EARTH_R_EQ})
    with pytest.raises(ValueError, match="coarse secular"):
        run_sweep(build, [LAYERED, coarse], 3600.0, station_keeping=SPEC, delta_v_baseline="layered")
    ok = ModelConfig("ok secular", PropagatorType.SECULAR_J2, 1200.0,
                     propagator_coefficients={"j2": EARTH_J2, "r_eq": EARTH_R_EQ})
    out: List[SweepResult] = run_sweep(build, [ok], 3600.0, timing_batches=1, timing_warmup=0,
                                       station_keeping=SPEC, delta_v_baseline="ok secular")
    assert out[0].delta_v is not None and out[0].delta_v.bodies[0].n_raises == 0
