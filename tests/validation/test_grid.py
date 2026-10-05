"""
Sweeps along an axis (`orbital_engine.grid`) and the implementation as configuration
(`ModelConfig.compiled`).

Expected, stated before measuring (the module docstring derives the physics):
- `crossover` interpolates `log(a/b)` linearly in `log(value)`, so for two power laws it is exact:
  `10 N` against `100 sqrt(N)` cross at N = 100 from grid points 10 and 1000.
- `stability_limit` stops at the first failure, counts a non-finite error as one, and ignores a curve
  that dips back under the threshold beyond it.
- Cowell + J2 at 550 km over 6,400 s: RK4's error ratio per step doubling ~2^4 = 16 at small steps;
  the 1 km limit between 80 and 160 s. *Measured:* 0.095 km at 80 s, 2.1 km at 160 s, ratios 17-22
  over 10-160 s. (The docstring's 15 km at 160 s was wrong by scaling the 24 h error linearly in time;
  along-track error grows faster. The limit's location was right.)
- `compiled=True` and `compiled=False` fly the same physics: errors agree to the twins' 1e-12
  relative standard (here 1e-9 on a km-scale error), and the flag reaches the simulation.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import geopotential, grid, scenarios, sweep
from orbital_engine.custom_types import PropagatorType
from orbital_engine.kernels import NUMBA_AVAILABLE
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ErrorStats, ForceModelSpec, ModelConfig, SweepResult

J2 = {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ}
OBL = {"Earth": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)}
FM = (ForceModelSpec("point_mass_gravity"), ForceModelSpec("j2", J2))


def _row(value: float, name: str, wall: float, err: float = 0.0) -> grid.GridRow:
    return grid.GridRow("x", value, name, SweepResult(name, ErrorStats(err, err, err), wall, 1))


def test_crossover_is_exact_for_power_laws() -> None:
    rows = [_row(n, "a", 10.0 * n) for n in (10.0, 1000.0)] + [_row(n, "b", 100.0 * math.sqrt(n)) for n in (10.0, 1000.0)]
    got = grid.crossover(rows, "a", "b")
    assert got is not None and abs(got - 100.0) < 1e-9
    assert grid.crossover([_row(n, "a", 1.0) for n in (1.0, 2.0)] + [_row(n, "b", 2.0) for n in (1.0, 2.0)],
                          "a", "b") is None


def test_stability_limit_stops_at_the_first_failure() -> None:
    rows = [_row(v, "c", 0.0, e) for v, e in ((1.0, 0.1), (2.0, 0.5), (4.0, float("nan")), (8.0, 0.2))]
    assert grid.stability_limit(rows, "c", 1.0) == (2.0, 4.0)
    assert grid.stability_limit(rows, "c", 0.05) == (None, 1.0)
    assert grid.stability_limit(rows[:2], "c", 1.0) == (2.0, None)


@pytest.fixture(scope="module")
def build() -> Callable[[], Simulation]:
    def make() -> Simulation:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from orbital_engine.database import Base
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        session: Session = sessionmaker(bind=engine)()
        return scenarios.earth_constellation(session, n_sats=6, n_planes=6)
    return make


def test_step_size_axis_finds_rk4s_limit(build: Callable[[], Simulation]) -> None:
    rows = grid.run_grid("dt_s", [10.0, 20.0, 40.0, 80.0, 160.0], lambda v: build,
                         lambda v: [ModelConfig("cowell+j2", PropagatorType.COWELL, v, force_models=FM)],
                         6400.0, oblateness=OBL, timing_batches=1, timing_warmup=0)
    assert grid.stability_limit(rows, "cowell+j2", 1.0) == (80.0, 160.0)
    _, err = grid.series(rows, "cowell+j2", "median_km")
    ratios = err[1:] / err[:-1]
    assert np.all((12.0 < ratios) & (ratios < 32.0))


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="needs numba for the compiled side")
def test_compiled_flag_selects_the_implementation(build: Callable[[], Simulation]) -> None:
    cfg = [ModelConfig("compiled", PropagatorType.COWELL, 60.0, force_models=FM, compiled=True),
           ModelConfig("numpy", PropagatorType.COWELL, 60.0, force_models=FM, compiled=False)]
    res = {r.config_name: r for r in sweep.run_sweep(build, cfg, 1800.0, oblateness=OBL,
                                                       timing_batches=1, timing_warmup=0)}
    a, b = res["compiled"].error.median_km, res["numpy"].error.median_km
    assert abs(a - b) <= 1e-9 * max(a, b)
    sim = build()
    sweep.apply_config(sim, cfg[1])
    assert sim.use_compiled_kernel is False
    sweep.apply_config(sim, cfg[0])
    assert sim.use_compiled_kernel is True


def test_compiled_without_numba_is_refused(build: Callable[[], Simulation], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "NUMBA_AVAILABLE", False)
    with pytest.raises(ValueError, match="needs numba"):
        sweep.apply_config(build(), ModelConfig("c", PropagatorType.COWELL, 60.0, force_models=FM, compiled=True))


def test_horizon_can_be_a_function_of_the_axis_value(build: Callable[[], Simulation]) -> None:
    """A horizon axis: each value gets its own horizon, and the error is the one at that horizon - a
    Keplerian config against the J2 truth drifts further the longer it runs."""
    rows = grid.run_grid("orbits", [1.0, 2.0], lambda v: build,
                         lambda v: [ModelConfig("kepler", PropagatorType.KEPLERIAN, 60.0)],
                         lambda v: v * 5760.0, oblateness=OBL, timing_batches=1, timing_warmup=0)
    _, err = grid.series(rows, "kepler", "median_km")
    assert err[1] > 1.5 * err[0]
