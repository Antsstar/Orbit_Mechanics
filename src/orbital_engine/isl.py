"""
Inter-satellite link (ISL) **visibility**: when two satellites can see each other past the central
body, the contact dataset that goes with it, and the sweep metric that scores a model's link windows
against truth's.

**Why this exists.** `access.py` exports ground-station contacts; a downstream network study (routing,
handover, link budgets - a separate repository that imports this engine) needs satellite-to-satellite
contacts as well, and this engine's thesis needs to say *which propagation model keeps them right*.
The boundary is `README.md`'s "Scope": **visibility, windows, range and range rate live here**; link
budgets, graphs and routing live downstream. Nothing here knows about a radio.

**Not a registry entry, no force model, no change to `step()`.** Like `geometry.py` and `access.py`,
there is no acceleration to compose and no state to advance.

## Visibility, stated as a geometric condition

Satellites `a` and `b` see each other when the straight **segment** between them clears the central
body by the grazing altitude `h_graze_km`:

    clearance(t) = min over tau in [0, 1] of |r_a + tau (r_b - r_a)|  -  (R + h_graze)  >  0

(`geometry.segment_clearance`, with `line_of_sight`'s `[0, 1]` clamp). `h_graze_km` is a required
field of `IslSpec` - the atmosphere/refraction margin a real crosslink keeps, commonly ~100 km - so a
spec can never silently assume a bare-surface link. `max_range_km`, when given, adds a second
geometric condition, `range < max_range_km`; it is a geometric limit, not a link budget.

The windowed signal is the **link margin**, a single continuous function in km:

    margin = clearance                                      (no range limit)
    margin = min(clearance, max_range_km - range)           (with one)

The minimum of two continuous functions is continuous, so every edge is found by the same linear
inverse interpolation `geometry.access_windows` applies to elevation - and the edge's `O(h^2)` bias
analysis carries over (below). Positions are **central-body-relative inertial**, and the body is a
sphere of `body_radius_km` - the same convention as `geometry.py`, with the same ellipsoid caveat.
Unlike a ground station, *nothing here rotates*: the clearance depends only on the two positions, so
an ISL is blind to a common rotation of the whole constellation about the body's centre (see the
headline in `docs/architecture.md` for why that matters).

## Pairs

Unordered pairs `(a, b)` with **`a < b` in the order of the body list the caller passes** - for a
sweep, the configuration's own body order, which is the same list for truth and model. `N` bodies
give `N (N - 1) / 2` pairs, enumerated lexicographically (`numpy.triu_indices(N, k=1)`), and windows
come back ordered by `(a, b)`, then time. Clearance, range and range rate are evaluated vectorised
over `(times, pairs)`, in chunks of pairs so memory stays bounded for a large constellation; the edge
extraction is vectorised over the same block. The only Python loop is the one that builds the output
records.

## Edge interpolation: the bias, redone for clearance

`geometry.access_windows` derives the linear-interpolation edge error as a **bias**,
`t_hat - t* = -(f'' / (2 f')) a b` with `a`, `b` the crossing's offsets into its bracket
(`a + b = h`), bounded by `|f''| h^2 / (8 |f'|)`. Elevation is *convex* at the horizon, so ground
rises read early and sets late. The ISL clearance is generally **concave** at its crossing - it is a
function of relative phase with a maximum at conjunction and it falls away on both sides - so the sign
is the **opposite**: an ISL window's rise reads *late*, its set *early*, and its duration *short*.
For the coplanar pair of `scenarios.coplanar_satellites` at 550 / 1200 km, `h_graze = 100 km`
(`tests/validation/test_isl.py`, derived there from the closed form): `f' = +/-0.2087 km/s`,
`f'' = -3.58e-5 km/s^2`, so `|f'' / (2 f')| = 8.58e-5 /s` and the worst-case bias is 0.077 s at
`h = 60 s`, quartering under halving.

As for ground access, truth and model are sampled on **one shared grid**, so this bias is common-mode
and cancels to `C Delta h` (`access.py`, "Sampling step"); the same `sample_dt_s` divisibility rule
applies, and for the same reason: `viz.sample_states` sub-steps a sample interval at *most* `max_dt`,
so an ISL grid finer than a configuration's `dt` would score a finer model than the one the sweep ran.

## The dataset and its types

`IslWindow` is a thin sibling of `geometry.AccessWindow` rather than a re-use of it: an access
window's identity is `(station_index, body_index)` and its peak is an *elevation*; an ISL window's
identity is an unordered pair `(body_a, body_b)` and its peak is a **closest approach** (the sampled
minimum range), and reading a range out of a field called `peak_elevation_rad` would be the kind of
plausible mis-read this engine's docs spend most of their effort preventing. `IslContact` is the
matching sibling of `access.ContactWindow`, and it **re-uses `access.ContactSample` unchanged** for
the rise / peak / set samples: `(time_s, range_km, range_rate_km_s)`, with range rate
`d|r_b - r_a|/dt` and **positive = opening (receding)**, negative = closing - the engine's one
convention, from `geometry.py`. There is no transport term: both ends are inertial.

The peak sample is the grid time of minimum range inside the window - sampled, not refined, so its
range rate is not zero but bounded by `rho'' h / 2`, exactly `ContactWindow.peak`'s caveat.

## The metric: `access.py`'s, not a copy of it

`compare_isl_windows` adapts each `IslWindow` to an `AccessWindow` (`IslWindow.to_access_window`:
`station_index <- body_a`, `body_index <- body_b`, `peak_elevation_rad = nan`) and hands both lists
to `access.compare_windows`. The result **is** an `access.AccessMetrics`: the same overlap-only
matching (now per pair, because the pair is the grouping key), the same signed rise/set shift and
duration statistics, the same lost/gained counts and total-contact error, the same clipped-edge
exclusions - by construction rather than by a second implementation that could drift. In an ISL
`AccessMetrics`, a `WindowMatch`'s `station_index` is pair member `a` and its `body_index` is `b`;
`peak_elevation_rad` is `nan` and is never read (the matcher never reads it).

References
----------
No new physics. The segment test is Vallado 4e Alg. 35 (`SIGHT`), cited in `geometry.py`; the tangent
angle `phi* = acos(rho / r1) + acos(rho / r2)` used in the tests is elementary plane geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .access import DEFAULT_SAMPLE_DT_S, AccessMetrics, ContactSample, compare_windows
from .custom_types import ArrayFloat, ArraySeconds, ScalarSeconds
from .geometry import AccessWindow, segment_clearance
from .reference import ReferenceTrajectory
from .simulator import Simulation
from .viz import sample_states

__all__ = [
    "IslSpec", "IslWindow", "IslContact", "IslSeries",
    "pair_indices", "link_series",
    "isl_windows", "isl_contacts",
    "isl_windows_from_truth", "isl_windows_from_simulation", "isl_contacts_from_simulation",
    "compare_isl_windows",
]

# Upper bound on the number of (time, pair) samples held in one block. Each block carries a handful
# of `(n_times, pairs, 3)` temporaries, so 1e6 keeps a block to a few tens of MB whatever the
# constellation size; a 12-satellite, 1441-sample sweep (66 pairs) is a single block.
_BLOCK_SAMPLES = 1_000_000


# --------------------------------------------------------------------------------------------------
# Configuration and records
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class IslSpec:
    """
    Everything that turns a set of trajectories into inter-satellite link windows.

    `central_body` names the occulting body; positions are taken relative to it.
    `body_radius_km` is its sphere (use the same radius the scenario was built with -
    `scenarios.EARTH_RADIUS` for every builder in `scenarios.py`). `h_graze_km` is the grazing
    altitude the segment must clear - **required**, never defaulted, because the choice between a
    bare-surface link and a ~100 km atmosphere margin is a modelling decision, not a detail.
    `max_range_km`, if given, closes the link beyond that range (geometric only).

    `sample_dt_s` has `access.AccessSpec`'s meaning and constraint: an integer multiple of every
    configuration's `dt` (`sweep.isl_metrics_for` refuses otherwise). `bodies` narrows the satellites
    to a named subset; by default a sweep uses each configuration's own target set, in its order,
    which is the order `a < b` refers to.
    """
    central_body: str
    body_radius_km: float
    h_graze_km: float
    max_range_km: Optional[float] = None
    sample_dt_s: float = DEFAULT_SAMPLE_DT_S
    bodies: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class IslWindow:
    """
    One contiguous interval during which satellites `body_a < body_b` could see each other.

    `rise_s` / `set_s` are **interpolated** on the link margin (module docstring: concave at the
    crossing, so rises read late and sets early by `O(h^2)`); `rise_clipped` / `set_clipped` mark an
    edge that is the grid endpoint rather than a crossing, exactly as in `geometry.AccessWindow`.
    `min_range_km` and `peak_time_s` are the **sampled** closest approach inside the window: a grid
    time and the range there, an upper bound on the true minimum.

    `body_a` / `body_b` index the body list the windows were computed from, not arena slots.
    """
    body_a: int
    body_b: int
    rise_s: float
    set_s: float
    duration_s: float
    min_range_km: float
    peak_time_s: float
    rise_clipped: bool
    set_clipped: bool

    def to_access_window(self) -> AccessWindow:
        """
        The `AccessWindow` the shared matcher consumes: `station_index <- body_a`,
        `body_index <- body_b`, `peak_elevation_rad = nan` (there is no elevation, and the matcher
        never reads it). Everything the metric differences - edges, duration, clip flags - is
        carried across unchanged.
        """
        return AccessWindow(
            station_index=self.body_a,
            body_index=self.body_b,
            rise_s=self.rise_s,
            set_s=self.set_s,
            duration_s=self.duration_s,
            peak_elevation_rad=math.nan,
            rise_clipped=self.rise_clipped,
            set_clipped=self.set_clipped,
            peak_time_s=self.peak_time_s,
        )


@dataclass(frozen=True)
class IslContact:
    """
    One ISL window plus range and range rate at rise, closest approach and set.

    The samples are `access.ContactSample`, unchanged: **range rate positive = opening**. Rise and
    set are interpolated times, so both series are linearly interpolated onto them (`|f''| h^2 / 8`);
    a clipped edge reads the endpoint sample. The peak is read at `window.peak_time_s`, a grid time.
    """
    window: IslWindow
    rise: ContactSample
    peak: ContactSample
    set: ContactSample

    @property
    def body_a(self) -> int:
        return self.window.body_a

    @property
    def body_b(self) -> int:
        return self.window.body_b

    @property
    def rise_s(self) -> float:
        return self.window.rise_s

    @property
    def set_s(self) -> float:
        return self.window.set_s

    @property
    def duration_s(self) -> float:
        return self.window.duration_s


@dataclass(frozen=True)
class IslSeries:
    """
    Per-sample link geometry for a set of pairs, each array `(n_times, n_pairs)`.

    `clearance_km` is `geometry.segment_clearance` against `R + h_graze`; `margin_km` is what the
    windows are cut on (equal to `clearance_km` without a range limit). `range_rate_km_s` is `None`
    unless velocities were given; **positive = opening**. `body_a` / `body_b` name each column.
    """
    times_s: ArraySeconds
    body_a: NDArray[np.int64]
    body_b: NDArray[np.int64]
    clearance_km: ArrayFloat
    margin_km: ArrayFloat
    range_km: ArrayFloat
    range_rate_km_s: Optional[ArrayFloat] = None


# --------------------------------------------------------------------------------------------------
# Geometry over (times, pairs)
# --------------------------------------------------------------------------------------------------

def pair_indices(n_bodies: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Every unordered pair of `n_bodies` bodies as two index arrays `(a, b)` with `a < b`, in
    lexicographic order: `(0,1), (0,2), ..., (0,N-1), (1,2), ...`. There are `N (N - 1) / 2`.
    """
    if n_bodies < 0:
        raise ValueError(f"n_bodies must be non-negative, got {n_bodies}.")
    a, b = np.triu_indices(n_bodies, k=1)
    return a.astype(np.int64), b.astype(np.int64)


def _validate(
    positions_km: ArrayFloat, times_s: ArraySeconds, velocities_km_s: Optional[ArrayFloat],
) -> Tuple[ArrayFloat, ArrayFloat, Optional[ArrayFloat]]:
    pos = np.asarray(positions_km, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_km must have shape (n_times, n_bodies, 3), got {pos.shape}.")
    times = np.asarray(times_s, dtype=np.float64)
    if times.shape != (pos.shape[0],):
        raise ValueError(
            f"times_s has shape {times.shape}, incompatible with {pos.shape[0]} position samples."
        )
    vel: Optional[ArrayFloat] = None
    if velocities_km_s is not None:
        vel = np.asarray(velocities_km_s, dtype=np.float64)
        if vel.shape != pos.shape:
            raise ValueError(
                f"velocities_km_s has shape {vel.shape}, incompatible with positions {pos.shape}."
            )
    return pos, times, vel


def _pair_block(
    pos: ArrayFloat,
    vel: Optional[ArrayFloat],
    ia: NDArray[np.int64],
    ib: NDArray[np.int64],
    spec: IslSpec,
) -> Tuple[ArrayFloat, ArrayFloat, ArrayFloat, Optional[ArrayFloat]]:
    """Clearance, margin, range and (optionally) range rate for one block of pairs, `(T, P)` each."""
    r_a = pos[:, ia, :]
    r_b = pos[:, ib, :]
    clearance = segment_clearance(
        r_a, r_b, body_radius_km=spec.body_radius_km, h_graze_km=spec.h_graze_km,
    )
    rel = r_b - r_a
    ranges: ArrayFloat = np.sqrt(np.sum(rel * rel, axis=-1))
    margin = clearance
    if spec.max_range_km is not None:
        margin = np.minimum(clearance, float(spec.max_range_km) - ranges)

    rates: Optional[ArrayFloat] = None
    if vel is not None:
        rel_v = vel[:, ib, :] - vel[:, ia, :]
        # d|rel|/dt = rel . rel_dot / |rel|; coincident satellites have an undefined rate and
        # report 0.0, `geometry.elevation_azimuth`'s convention (the numerator is 0 there too).
        rate: ArrayFloat = np.sum(rel * rel_v, axis=-1) / np.where(ranges > 0.0, ranges, 1.0)
        rates = rate
    return clearance, margin, ranges, rates


def _blocks(n_times: int, n_pairs: int) -> List[slice]:
    per_block = max(1, _BLOCK_SAMPLES // max(1, n_times))
    return [slice(s, min(s + per_block, n_pairs)) for s in range(0, n_pairs, per_block)]


def link_series(
    positions_km: ArrayFloat,
    times_s: ArraySeconds,
    spec: IslSpec,
    *,
    velocities_km_s: Optional[ArrayFloat] = None,
    pairs: Optional[Tuple[NDArray[np.int64], NDArray[np.int64]]] = None,
) -> IslSeries:
    """
    The full per-sample link geometry - clearance, margin, range and optionally range rate - for
    every pair (default) or the given `(a, b)` index arrays.

    `positions_km` (and `velocities_km_s`) are `(n_times, n_bodies, 3)`, **central-body-relative
    inertial**, e.g. `viz.sample_states(..., relative_to=central)`. This is the whole series in one
    array, for inspection and for tests; the window path (`isl_windows`) evaluates the same
    quantities block by block and never holds all pairs at once.
    """
    pos, times, vel = _validate(positions_km, times_s, velocities_km_s)
    ia, ib = pair_indices(pos.shape[1]) if pairs is None else (
        np.asarray(pairs[0], dtype=np.int64), np.asarray(pairs[1], dtype=np.int64)
    )
    clearance, margin, ranges, rates = _pair_block(pos, vel, ia, ib, spec)
    return IslSeries(
        times_s=times, body_a=ia, body_b=ib, clearance_km=clearance, margin_km=margin,
        range_km=ranges, range_rate_km_s=rates,
    )


# --------------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Edges:
    """The windows of one block, as flat arrays (one entry per window), pair-major then time."""
    pair: NDArray[np.int64]
    i0: NDArray[np.int64]
    i1: NDArray[np.int64]
    rise: ArrayFloat
    set: ArrayFloat
    rise_clipped: NDArray[np.bool_]
    set_clipped: NDArray[np.bool_]
    peak: NDArray[np.int64]


def _extract(times: ArrayFloat, margin: ArrayFloat, ranges: ArrayFloat) -> _Edges:
    """
    Every window of a `(T, P)` margin block, vectorised over both axes.

    Runs of `margin > 0` are found from the sign changes of a zero-padded indicator, so the rises
    and sets come out of `np.nonzero` already paired and in pair-major, time-ascending order. Edges
    are `geometry._crossing`'s linear inverse interpolation written over arrays, in the same
    floating-point order. The closest approach is the first sampled minimum of range in each run.
    """
    n_times, n_pairs = margin.shape
    indicator = np.zeros((n_pairs, n_times + 2), dtype=np.int8)
    indicator[:, 1:-1] = (margin > 0.0).T
    step = np.diff(indicator, axis=1)
    pair, i0 = np.nonzero(step == 1)            # first in-view sample of each run
    pair_end, j = np.nonzero(step == -1)        # first sample after each run
    i1 = j - 1
    if not np.array_equal(pair, pair_end):  # pragma: no cover - runs alternate by construction
        raise RuntimeError("ISL window extraction produced unpaired edges.")
    pair = pair.astype(np.int64)
    i0 = i0.astype(np.int64)
    i1 = i1.astype(np.int64)

    last = n_times - 1
    rise_clipped = i0 == 0
    set_clipped = i1 == last

    lo = np.where(rise_clipped, i0, i0 - 1)
    f0 = margin[lo, pair]
    f1 = margin[i0, pair]
    t0 = times[lo]
    t1 = times[i0]
    denom = np.where(rise_clipped, 1.0, f1 - f0)
    rise = np.where(rise_clipped, times[0], t0 + (t1 - t0) * (-f0) / denom)

    hi = np.where(set_clipped, i1, i1 + 1)
    g0 = margin[i1, pair]
    g1 = margin[hi, pair]
    s0 = times[i1]
    s1 = times[hi]
    denom = np.where(set_clipped, 1.0, g1 - g0)
    set_t = np.where(set_clipped, times[last], s0 + (s1 - s0) * (-g0) / denom)

    # Closest approach: flatten every run's range samples into one array of contiguous segments,
    # take each segment's minimum with `reduceat`, and keep the first sample that attains it.
    n_windows = pair.size
    peak = np.zeros(n_windows, dtype=np.int64)
    if n_windows:
        lengths = i1 - i0 + 1
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
        within = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(offsets, lengths)
        cols = np.repeat(i0, lengths) + within
        vals = ranges[cols, np.repeat(pair, lengths)]
        seg_min = np.minimum.reduceat(vals, offsets)
        hit = np.flatnonzero(vals == np.repeat(seg_min, lengths))
        seg_of_hit = np.repeat(np.arange(n_windows, dtype=np.int64), lengths)[hit]
        _, first = np.unique(seg_of_hit, return_index=True)
        peak = cols[hit[first]]

    return _Edges(
        pair=pair, i0=i0, i1=i1, rise=np.asarray(rise, dtype=np.float64),
        set=np.asarray(set_t, dtype=np.float64),
        rise_clipped=rise_clipped, set_clipped=set_clipped, peak=peak,
    )


def _edge_value(
    series: ArrayFloat, times: ArrayFloat, edges: _Edges, at_rise: bool,
) -> ArrayFloat:
    """One series linearly interpolated onto each window's rise (or set) instant; a clipped edge
    reads its endpoint sample, never an extrapolation (`np.interp`'s clamp, as in `access.py`)."""
    if at_rise:
        clipped, inside, t = edges.rise_clipped, edges.i0, edges.rise
        outside = np.where(clipped, inside, inside - 1)
    else:
        clipped, inside, t = edges.set_clipped, edges.i1, edges.set
        outside = np.where(clipped, inside, inside + 1)
    v_in = series[inside, edges.pair]
    v_out = series[outside, edges.pair]
    t_in = times[inside]
    t_out = times[outside]
    span = np.where(clipped, 1.0, t_out - t_in)
    w = np.where(clipped, 0.0, (t - t_in) / span)
    value: ArrayFloat = np.where(clipped, v_in, (1.0 - w) * v_in + w * v_out)
    return value


def _scan(
    positions_km: ArrayFloat,
    times_s: ArraySeconds,
    spec: IslSpec,
    velocities_km_s: Optional[ArrayFloat],
) -> Tuple[List[IslWindow], List[IslContact]]:
    pos, times, vel = _validate(positions_km, times_s, velocities_km_s)
    if times.size < 2:
        raise ValueError("ISL windows need at least two samples to bracket a crossing.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")

    ia_all, ib_all = pair_indices(pos.shape[1])
    windows: List[IslWindow] = []
    contacts: List[IslContact] = []
    for block in _blocks(times.size, ia_all.size):
        ia = ia_all[block]
        ib = ib_all[block]
        _, margin, ranges, rates = _pair_block(pos, vel, ia, ib, spec)
        edges = _extract(times, margin, ranges)
        if edges.pair.size == 0:
            continue

        body_a = ia[edges.pair].tolist()
        body_b = ib[edges.pair].tolist()
        peak_t = times[edges.peak]
        peak_r = ranges[edges.peak, edges.pair]
        block_windows = [
            IslWindow(
                body_a=a, body_b=b, rise_s=r, set_s=s, duration_s=s - r,
                min_range_km=pr, peak_time_s=pt, rise_clipped=rc, set_clipped=sc,
            )
            for a, b, r, s, pr, pt, rc, sc in zip(
                body_a, body_b, edges.rise.tolist(), edges.set.tolist(), peak_r.tolist(),
                peak_t.tolist(), edges.rise_clipped.tolist(), edges.set_clipped.tolist(),
            )
        ]
        windows.extend(block_windows)

        if rates is not None:
            rise_range = _edge_value(ranges, times, edges, True).tolist()
            rise_rate = _edge_value(rates, times, edges, True).tolist()
            set_range = _edge_value(ranges, times, edges, False).tolist()
            set_rate = _edge_value(rates, times, edges, False).tolist()
            peak_rate = rates[edges.peak, edges.pair].tolist()
            for k, w in enumerate(block_windows):
                contacts.append(IslContact(
                    window=w,
                    rise=ContactSample(w.rise_s, rise_range[k], rise_rate[k]),
                    peak=ContactSample(w.peak_time_s, w.min_range_km, peak_rate[k]),
                    set=ContactSample(w.set_s, set_range[k], set_rate[k]),
                ))
    return windows, contacts


def isl_windows(
    positions_km: ArrayFloat, times_s: ArraySeconds, spec: IslSpec,
) -> List[IslWindow]:
    """
    Link windows for every pair of `(n_times, n_bodies, 3)` **central-body-relative inertial**
    positions, ordered by `(body_a, body_b)` then time. Positions only - the metric path;
    `isl_contacts` is the velocity-carrying counterpart and returns the same windows.
    """
    windows, _ = _scan(positions_km, times_s, spec, None)
    return windows


def isl_contacts(
    positions_km: ArrayFloat,
    velocities_km_s: ArrayFloat,
    times_s: ArraySeconds,
    spec: IslSpec,
) -> List[IslContact]:
    """
    The ISL contact dataset: every window of `isl_windows` (identical records), each with range and
    range rate at rise, closest approach and set. Velocities are the same frame as positions
    (central-body-relative inertial, `viz.sample_states` columns `3:6`). **Positive range rate =
    opening.**
    """
    _, contacts = _scan(positions_km, times_s, spec, velocities_km_s)
    return contacts


# --------------------------------------------------------------------------------------------------
# Sources: truth and simulation
# --------------------------------------------------------------------------------------------------

def isl_windows_from_truth(
    truth: ReferenceTrajectory, body_names: Sequence[str], spec: IslSpec,
) -> List[IslWindow]:
    """
    Truth link windows from a `reference.ReferenceTrajectory` already sampled on the ISL grid - not
    resampled, for `access.windows_from_truth`'s reason: the edge bias only cancels if truth and
    model share the grid. `body_names` fixes the pair order.
    """
    central = truth.position_of(spec.central_body)
    stacked = np.stack([truth.position_of(name) - central for name in body_names], axis=1)
    return isl_windows(stacked, np.asarray(truth.times, dtype=np.float64), spec)


def isl_windows_from_simulation(
    sim: Simulation,
    bodies: Sequence[int],
    times_s: ArraySeconds,
    spec: IslSpec,
    *,
    max_dt: Optional[ScalarSeconds] = None,
) -> List[IslWindow]:
    """
    Propagate `sim` across `times_s` and return its link windows. **This advances the simulation**
    (`viz.sample_states`); `max_dt` is the propagation step. `bodies` are arena slots, and their
    order is the pair order.
    """
    central = sim.name_to_index[spec.central_body]
    states = sample_states(sim, bodies, times_s, relative_to=central, max_dt=max_dt)
    return isl_windows(np.ascontiguousarray(states[..., :3]), times_s, spec)


def isl_contacts_from_simulation(
    sim: Simulation,
    bodies: Sequence[int],
    times_s: ArraySeconds,
    spec: IslSpec,
    *,
    max_dt: Optional[ScalarSeconds] = None,
) -> List[IslContact]:
    """
    Propagate `sim` across `times_s` and return its ISL contact dataset - the export a downstream
    network study consumes. **This advances the simulation**, exactly as
    `isl_windows_from_simulation` does.
    """
    central = sim.name_to_index[spec.central_body]
    states = sample_states(sim, bodies, times_s, relative_to=central, max_dt=max_dt)
    return isl_contacts(
        np.ascontiguousarray(states[..., :3]), np.ascontiguousarray(states[..., 3:]), times_s, spec,
    )


# --------------------------------------------------------------------------------------------------
# The metric
# --------------------------------------------------------------------------------------------------

def compare_isl_windows(
    truth_windows: Sequence[IslWindow], model_windows: Sequence[IslWindow],
) -> AccessMetrics:
    """
    `access.compare_windows` on the adapted windows: overlap-only matching **per pair**, then the
    same statistics as ground access. See the module docstring for how to read the result's
    `WindowMatch.station_index` (`a`) and `body_index` (`b`).
    """
    return compare_windows(
        [w.to_access_window() for w in truth_windows],
        [w.to_access_window() for w in model_windows],
    )
