"""
Sweeps along an axis: the same comparison (`sweep.run_sweep`) repeated over a scenario or model
parameter - body count, step size, horizon - and the two questions asked of the result.

- **Where does one method overtake another?** `crossover`: the axis value at which config A's metric
  (wall time by default) crosses config B's, interpolated in log-log space between grid points.
  Along body count this is the scaling question: a compiled scalar kernel with a small fixed cost
  against a vectorised one with a large fixed cost and a small marginal one.
- **When does a model stop working?** `stability_limit`: the largest axis value (step size, horizon)
  at which a config's error stays finite and under a threshold, and the first one at which it does
  not. A non-finite error is a failure, not a missing point: a diverged integration is the answer.

Every grid point is a full `run_sweep` - a fresh truth and fresh builds - so nothing here adds physics
or scoring; `GridRow.result` is exactly what `run_sweep` returned at that value. Configurations are
matched across grid points **by name**, so `configs_for(value)` must name a config the same way at
every value (a step-size family keeps its name while its `dt` changes).

`ModelConfig.compiled` makes the implementation part of the configuration: the same physics, compiled
or NumPy, is then two configs, and the body-count axis says where (whether) one overtakes the other.

Expected magnitudes (`benchmarks/scaling_sweep.py` prints the measurements next to these)
-----------------------------------------------------------------------------------------
- **Step size, Cowell + J2, 550 km LEO, one-orbit horizon.** RK4's global error grows as `h^4`: the
  frontier's 24 h figures (230.8 km at 160 s, 7.6 km at 80 s) are ~15x smaller over one orbit, so ~15 km
  at 160 s and **a 1 km threshold near 80 s**. Fixed-step RK4 stops tracking the orbit at all once
  `n h` approaches 1 (n = 1.08e-3 rad/s): **error of the order of the orbit itself near 600-1000 s**.
- **Body count, per-body cost per step.** The fused compiled Cowell kernel costs ~5 us per step fixed
  plus ~0.2 us per body (`docs/architecture.md`); the NumPy path has a larger fixed cost (Python
  dispatch over the force kernels, four RK4 stages) and its own marginal cost. Predicted: the compiled
  kernel is cheaper at every N measured - **no crossover** - unless the NumPy marginal cost is below
  ~0.2 us per body-step, which vectorised four-stage RK4 with J2 is not expected to reach.

Measured (`benchmarks/scaling_sweep.py`, `docs/figures/scaling.png`)
--------------------------------------------------------------------
- **Step size.** 1.5e-5 km at 10 s up to 2.1 km at 160 s (ratios 17-22 per doubling, steepening to 30
  beyond), so the 1 km limit falls between 80 and 160 s as predicted. The 15 km at 160 s was wrong: it
  scaled the 24 h error linearly in time, and along-track error grows faster. 100 km holds to 320 s.
  Orbit-scale failure (6,000 km) falls between 640 and 1,280 s, inside the predicted 600-1,000 s.
- **Body count.** No crossover from 6 to 384 satellites: compiled ~6.5 us per step + 0.08 us per
  body-step, NumPy ~700 us per step + 2.5 us per body-step, so even the NumPy *marginal* cost is ~30x
  the compiled one. The ratio falls from 107x at N = 6 towards that asymptote (46x at 384). Both give
  identical errors at every N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .simulator import Simulation
from .sweep import ModelConfig, SweepResult, run_sweep

__all__ = ["GridRow", "run_grid", "series", "crossover", "stability_limit"]


@dataclass(frozen=True)
class GridRow:
    """One configuration at one axis value: `result` is `run_sweep`'s, unchanged."""
    axis: str
    value: float
    config_name: str
    result: SweepResult

    @property
    def wall_time_per_body_us(self) -> float:
        return self.result.wall_time_us / max(1, self.result.n_bodies)


def run_grid(
    axis: str,
    values: Sequence[float],
    build_for: Callable[[float], Callable[[], Simulation]],
    configs_for: Callable[[float], Sequence[ModelConfig]],
    horizon_s: float,
    **sweep_kwargs: Any,
) -> List[GridRow]:
    """
    `run_sweep(build_for(v), configs_for(v), horizon_s, **sweep_kwargs)` at every `v` in `values`,
    flattened into rows in (value, config) order. `axis` only labels the rows. Configs must keep their
    names across values (see the module docstring); a name missing at some value is simply absent from
    that value's rows.
    """
    rows: List[GridRow] = []
    for v in values:
        for r in run_sweep(build_for(float(v)), configs_for(float(v)), horizon_s, **sweep_kwargs):
            rows.append(GridRow(axis, float(v), r.config_name, r))
    return rows


def _metric(row: GridRow, metric: str) -> float:
    if metric == "wall_time_us":
        return row.result.wall_time_us
    if metric == "wall_time_per_body_us":
        return row.wall_time_per_body_us
    if metric == "median_km":
        return row.result.error.median_km
    if metric == "max_km":
        return row.result.error.max_km
    raise ValueError(f"unknown metric {metric!r}")


def series(rows: Sequence[GridRow], config_name: str,
           metric: str = "wall_time_us") -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`(axis values, metric values)` of one config, sorted by axis value."""
    pts = sorted((r.value, _metric(r, metric)) for r in rows if r.config_name == config_name)
    xs: NDArray[np.float64] = np.array([p[0] for p in pts], dtype=np.float64)
    ys: NDArray[np.float64] = np.array([p[1] for p in pts], dtype=np.float64)
    return xs, ys


def crossover(rows: Sequence[GridRow], a: str, b: str, metric: str = "wall_time_us") -> Optional[float]:
    """
    The first axis value at which config `a`'s `metric` crosses config `b`'s, by linear interpolation of
    `log(a / b)` against `log(value)` between the bracketing grid points; `None` if they never cross on
    the grid (or touch only at a single point without changing order). Both must be positive.
    """
    xa, ya = series(rows, a, metric)
    xb, yb = series(rows, b, metric)
    common = sorted(set(xa.tolist()) & set(xb.tolist()))
    if len(common) < 2:
        return None
    ra = dict(zip(xa.tolist(), ya.tolist()))
    rb = dict(zip(xb.tolist(), yb.tolist()))
    d = [math.log(ra[x] / rb[x]) for x in common]
    for k in range(1, len(common)):
        if (d[k - 1] < 0.0) != (d[k] < 0.0) and d[k - 1] != d[k]:
            lx0, lx1 = math.log(common[k - 1]), math.log(common[k])
            return math.exp(lx0 + (lx1 - lx0) * (0.0 - d[k - 1]) / (d[k] - d[k - 1]))
    return None


def stability_limit(
    rows: Sequence[GridRow], config_name: str, threshold_km: float, metric: str = "median_km",
) -> Tuple[Optional[float], Optional[float]]:
    """
    `(last good value, first bad value)` of one config, scanning the axis upward: a value is bad when
    its error `metric` is non-finite or above `threshold_km`. `(None, first)` if the smallest value is
    already bad; `(last, None)` if none is. The scan stops at the first bad value - an error curve
    that dips back under the threshold beyond it does not make the model usable there.
    """
    xs, ys = series(rows, config_name, metric)
    last: Optional[float] = None
    for x, y in zip(xs.tolist(), ys.tolist()):
        if not (math.isfinite(y) and y <= threshold_km):
            return last, float(x)
        last = float(x)
    return last, None
