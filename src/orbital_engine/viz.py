"""
Plot *data* preparation: a `Simulation` and a time grid in, plottable arrays out.

**No plotting dependency.** This module never imports `matplotlib`, exactly as `sweep.py` does not -
the library stays installable without it, and `benchmarks/figures.py` is the consumer that draws.
Everything here is ordinary NumPy and is covered by `tests/validation/test_viz.py`, which runs without
matplotlib installed.

**Why a sampler at all.** `Simulation.step` advances state in place and keeps no trajectory when
`record_history` is off (which every benchmark and sweep turns off, because the per-body dict append
dominates the step). A figure needs a *history*, so `sample_states` walks a requested time grid,
sub-stepping between samples at a caller-chosen `max_dt`, and copies the arena rows out at each
sample. Splitting that from the transforms below is what lets the transforms be tested against
analytic geometry with no simulation at all.

**Sub-stepping is the model, not a sampling detail.** `max_dt` is the propagation step size: passing
`max_dt=60.0` with samples 900 s apart runs fifteen 60 s steps between samples, which is exactly what
`sweep.run_sweep` does for a `ModelConfig(dt=60.0)` config. Leaving it `None` takes one step per
sample interval, which is what the closed-form tiers (Keplerian, secular J2) want - they reach any
horizon in a single step, and charging them for 1440 steps they do not need is the mistake
`benchmarks/frontier_plot.py`'s docstring records. `tests/validation/test_viz.py` ties the final
value of an error curve back to `sweep.run_sweep` for the same configuration, so the two paths are
held equal rather than merely believed equal.

**Frames.** `sample_states` returns rows of `global_states` (root-relative), optionally differenced
against one other slot. Ground tracks need the *central-body-relative* state, so pass
`relative_to=sim.name_to_index["Earth"]`; position error against `reference.ReferenceTrajectory`
needs the root-relative one, so pass nothing - `reference_for` seeds itself from `global_states` and
reports in the same frame (see `sweep._position_errors_km`, which differences exactly these two).

**The body-fixed rotation sign.** `frames.Transformations.Rz(theta)` is an *active* vector rotation:
`Rz(t) @ v` turns `v` counter-clockwise by `t`. Expressing an inertial vector in a frame whose axes
have themselves turned by `+theta` therefore takes `Rz(-theta)`, so `ground_track` calls
`ReferenceFrames.inertia_to_fixed(r, v, -theta)`. Getting this backwards does not raise and does not
look wrong - it produces a ground track that drifts *east* at the Earth's rotation rate instead of
west, which is why `test_viz.py` asserts the signed per-orbit drift and not its magnitude.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, ArraySeconds, ScalarSeconds
from .frames import ReferenceFrames
from .reference import ReferenceTrajectory
from .simulator import Simulation

__all__ = [
    "GroundTrack", "ErrorCurve",
    "sample_states", "ground_track", "altitude_series", "position_error", "error_curve",
]


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundTrack:
    """
    Geocentric sub-satellite track over time, in degrees, shape `(n_times, n_bodies)` throughout.

    `longitude_deg` is wrapped to `[-180, +180]` and therefore *discontinuous* at the antimeridian -
    a plotting caller that draws lines rather than points must break the series there (see
    `benchmarks/figures.py`). `altitude_km` is `|r| - body_radius_km`, a geocentric altitude above a
    spherical body, not a geodetic altitude above an ellipsoid: this engine's J2 is a gravity-field
    term only and carries no reference ellipsoid, so there is no oblate surface to measure against.
    """
    times_s: ArraySeconds
    latitude_deg: ArrayFloat
    longitude_deg: ArrayFloat
    altitude_km: ArrayFloat


@dataclass(frozen=True)
class ErrorCurve:
    """
    Position error against a truth trajectory over time, in km.

    `per_body_km` has shape `(n_times, n_bodies)`; `median_km`, `rms_km` and `max_km` are that
    reduced over bodies at each sample, shape `(n_times,)`. The statistics-over-bodies convention is
    `sweep.ErrorStats`'s, for the reason its docstring gives: a single satellite's error varies by
    two orders of magnitude with initial phase alone, so a curve of one body would rank tiers by
    which phase was sampled. `median_km[-1]` is the quantity `sweep.SweepResult.error.median_km`
    reports for the same configuration - `tests/validation/test_viz.py` asserts they agree.
    """
    times_s: ArraySeconds
    names: List[str]
    per_body_km: ArrayFloat
    median_km: ArrayFloat
    rms_km: ArrayFloat
    max_km: ArrayFloat


# --------------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------------

def sample_states(
    sim: Simulation,
    bodies: Sequence[int],
    times_s: ArraySeconds,
    *,
    relative_to: Optional[int] = None,
    max_dt: Optional[ScalarSeconds] = None,
) -> ArrayFloat:
    """
    Step `sim` across `times_s` and return `(n_times, n_bodies, 6)` states at each sample.

    **This advances the simulation**; it is not a read-only view. Pass a freshly built `Simulation`,
    the way `sweep.run_sweep` builds one per configuration.

    `times_s` must be non-decreasing and begin at or after `sim.t`; the first sample is taken after
    stepping to `times_s[0]`, so a grid starting at 0.0 on a fresh simulation records the initial
    condition without stepping. Between consecutive samples the gap is divided into equal sub-steps
    of at most `max_dt` (one step when `max_dt is None`), so a uniform grid with
    `max_dt == grid spacing` yields exactly one step per sample and reproduces a
    `sweep.ModelConfig(dt=spacing)` run step for step.

    `relative_to` differences every row against that slot's `global_states` row at the same instant,
    which is how a central-body-relative track is taken - `global_states` is root-relative, and for
    `scenarios.earth_constellation` the root *is* the Earth barycentre, not the Earth.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"times_s must be one-dimensional, got shape {times.shape}.")
    if times.size == 0:
        raise ValueError("times_s must contain at least one sample.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_s must be non-decreasing.")
    if times[0] < sim.t - 1e-9:
        raise ValueError(
            f"times_s starts at {times[0]} but the simulation is already at t={sim.t}; "
            f"sample_states only moves forward."
        )
    if max_dt is not None and max_dt <= 0.0:
        raise ValueError(f"max_dt must be positive, got {max_dt}.")

    slots = np.asarray(bodies, dtype=np.int64)
    out = np.empty((times.size, slots.size, 6), dtype=np.float64)

    t_cur = float(sim.t)
    for k in range(times.size):
        span = float(times[k]) - t_cur
        if span > 0.0:
            n_steps = 1 if max_dt is None else max(1, int(math.ceil(span / float(max_dt) - 1e-12)))
            h = span / n_steps
            for _ in range(n_steps):
                sim.step(h)
            t_cur = float(times[k])
        out[k] = sim.global_states[slots]
        if relative_to is not None:
            out[k] -= sim.global_states[relative_to]
    return out


# --------------------------------------------------------------------------------------------------
# Transforms - no `Simulation` involved, so they can be tested against analytic geometry
# --------------------------------------------------------------------------------------------------

def ground_track(
    positions_km: ArrayFloat,
    times_s: ArraySeconds,
    *,
    omega: float,
    body_radius_km: float,
    theta0: float = 0.0,
) -> GroundTrack:
    """
    Sub-satellite latitude, longitude and altitude for central-body-relative inertial positions.

    `positions_km` is `(n_times, n_bodies, 3)` (a `(n_times, 3)` single-body array is accepted and
    returned with one body column), measured from the central body's centre - i.e. `sample_states`
    with `relative_to` set to that body. `omega` is the body's rotation rate in rad/s about the
    frame's **+z axis**, the same assumption `geopotential.py` and `drag.py` already make, and
    `theta0` is its prime-meridian angle at `times_s[0]`.

    The rotation is `ReferenceFrames.inertia_to_fixed` at `-theta`; see the module docstring for why
    the sign is what it is. `fixed_to_longlat` then supplies the wrap to `[-pi, +pi]`.
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[:, np.newaxis, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_km must have shape (n_times, n_bodies, 3), got {pos.shape}.")
    times = np.asarray(times_s, dtype=np.float64)
    if times.shape != (pos.shape[0],):
        raise ValueError(
            f"times_s has shape {times.shape}, incompatible with {pos.shape[0]} position samples."
        )

    # One rotation angle per sample, broadcast across the body axis: the whole grid rotates together.
    theta = theta0 + omega * (times - times[0])
    theta_full = np.repeat(theta[:, np.newaxis], pos.shape[1], axis=1)

    # `inertia_to_fixed` is `Rz(theta) @ r`, an active rotation, so the frame transform takes -theta.
    # The trailing axis makes the batched `(N,M,3,3) @ (N,M,3,1)` matmul well posed; `Rz` builds a
    # `(N,M,3,3)` tensor from a `(N,M)` angle array.
    r_fixed, _ = ReferenceFrames.inertia_to_fixed(
        pos[..., np.newaxis], np.zeros_like(pos)[..., np.newaxis], -theta_full
    )
    longlat = ReferenceFrames.fixed_to_longlat(r_fixed[..., 0])

    return GroundTrack(
        times_s=times,
        latitude_deg=np.degrees(longlat[..., 2]),
        longitude_deg=np.degrees(longlat[..., 1]),
        altitude_km=longlat[..., 0] - body_radius_km,
    )


def altitude_series(positions_km: ArrayFloat, *, body_radius_km: float) -> ArrayFloat:
    """
    Geocentric altitude `|r| - body_radius_km`, shape `(n_times, n_bodies)`.

    `positions_km` is central-body-relative, as for `ground_track`. Spherical, not geodetic - see
    `GroundTrack`.
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[:, np.newaxis, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_km must have shape (n_times, n_bodies, 3), got {pos.shape}.")
    return cast(ArrayFloat, np.linalg.norm(pos, axis=2) - body_radius_km)


def position_error(
    positions_km: ArrayFloat, truth: ReferenceTrajectory, names: Sequence[str],
) -> ArrayFloat:
    """
    `|r_engine - r_truth|` per sample per body, shape `(n_times, n_bodies)`, in km.

    `positions_km` is **root-relative** (`sample_states` with no `relative_to`), matching the frame
    `reference.reference_for` reports in, and `names` gives the body order along its second axis.
    `truth` must have been sampled on the same time grid; the shapes are checked rather than
    interpolated, because silently resampling a truth trajectory would hide a grid mismatch as a
    plausible-looking error curve.
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_km must have shape (n_times, n_bodies, 3), got {pos.shape}.")
    if pos.shape[1] != len(names):
        raise ValueError(f"{pos.shape[1]} body columns but {len(names)} names.")
    if pos.shape[0] != truth.times.size:
        raise ValueError(
            f"{pos.shape[0]} samples but the truth trajectory has {truth.times.size}; "
            f"generate truth on the same time grid."
        )
    truth_pos = np.stack([truth.position_of(name) for name in names], axis=1)
    return cast(ArrayFloat, np.linalg.norm(pos - truth_pos, axis=2))


def error_curve(
    positions_km: ArrayFloat, truth: ReferenceTrajectory, names: Sequence[str],
) -> ErrorCurve:
    """Position error over time reduced over bodies - median, RMS and max at each sample.

    See `ErrorCurve` for why the reduction is over bodies and what `median_km[-1]` ties back to.
    """
    per_body = position_error(positions_km, truth, names)
    return ErrorCurve(
        times_s=np.asarray(truth.times, dtype=np.float64),
        names=list(names),
        per_body_km=per_body,
        median_km=cast(ArrayFloat, np.median(per_body, axis=1)),
        rms_km=cast(ArrayFloat, np.sqrt(np.mean(per_body ** 2, axis=1))),
        max_km=cast(ArrayFloat, np.max(per_body, axis=1)),
    )
