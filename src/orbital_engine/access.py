"""
Model-fidelity error measured in **contact windows** rather than kilometres.

**Why this exists.** `sweep.py` reports position error in km, and nobody's decision is in km. For a
constellation or ground-station study the decision is when a pass opens, how long it lasts, and
whether a marginal pass exists at all. A model wrong by 500 km that moves every window by two
seconds is fine; one wrong by 5 km that *deletes* a pass is not. This module turns a sweep's
trajectories into that second currency.

**Why a separate module and not more of `sweep.py`.** Three reasons, all structural rather than
stylistic.

1. `sweep.py` imports `reference`, `benchmark` and `simulator` and nothing else - it is about
   propagating configurations and timing them. Access metrics additionally need `viz` (sampling) and
   `geometry` (look angles, rise/set), which are the *consumer* layers. Folding them in would make
   the harness depend on the presentation stack.
2. The interesting half of this feature - **matching** a model's windows to truth's before
   differencing them - is pure interval arithmetic over `geometry.AccessWindow` records. Keeping it
   here, with no `Simulation` in sight, is what lets `match_windows` be tested on hand-built window
   lists where the right answer is written down rather than propagated (exactly the reason
   `geometry.py` itself holds no `Simulation`).
3. `sweep.run_sweep`'s contract is unchanged by construction: it gains one optional keyword and one
   optional field on `SweepResult`, and the position-error path is not touched. Had the window code
   lived inside it, "additive" would have been a claim rather than a fact.

**Not a registry entry.** There is no acceleration to compose and no state to advance, so nothing
here takes a `force_model_mask` bit - the precedent `geometry.py` and `manoeuvres.py` set. The
registry's two dispatch mechanisms (force models, propagators) have no third slot for "a metric",
and inventing a parallel one to make this sweepable would be the wrong move: an `AccessSpec` is an
argument to `run_sweep`, in the same position `oblateness` already occupies.

---

## The metrics, defined

Per `(station, body)` pair, `geometry.access_windows` gives a list of intervals for truth and a list
for the model. After matching (below), each metric is:

| Metric | Definition | Sign convention |
|---|---|---|
| rise shift | `model.rise_s - truth.rise_s` | **positive = model rises late** |
| set shift | `model.set_s - truth.set_s` | positive = model sets late |
| duration error | `model.duration_s - truth.duration_s` | positive = model's pass reads long |
| total contact | sum of `duration_s` over **every** window, matched or not | - |
| passes lost | truth windows with no model counterpart | - |
| passes gained | model windows with no truth counterpart | - |

Shifts are reported signed (mean and median) *and* absolute (mean-abs and max-abs), because the two
answer different questions: a systematic lead or lag is a clock-like offset a scheduler can absorb,
while a large mean-abs with a near-zero mean is scatter it cannot.

**Total contact time includes clipped windows** - a window still open at the horizon really was in
view for the part of it that fell inside the horizon, and that time is contact time. **Shift and
duration statistics exclude clipped edges**: the grid endpoint is not a rise, and differencing it
against truth's endpoint would contribute a guaranteed exact zero that dilutes every real
measurement. `AccessMetrics.n_clipped_edges` records how many were dropped.

## Matching: by overlap, and nothing else

Before anything can be differenced, each model window must be paired with the truth window it *is*.
The rule here is deliberately the strictest defensible one:

> Two windows may be paired only if they **overlap in time**. Among the pairings that overlap,
> take the largest overlap first, remove both windows, and repeat (greedy, descending overlap; ties
> broken by earliest truth window, then earliest model window). Anything left over is an orphan.

An unmatched truth window is a **lost pass**; an unmatched model window is a **gained** (spurious)
one. Both are reported as counts and neither contributes to the shift statistics, because neither
has anything to be differenced against.

**Why overlap and not nearest-midpoint.** A nearest-midpoint rule needs a tolerance, and a tolerance
on a window-matching metric is a dial that converts "this model lost a pass" into "this model was
late", which is precisely the distinction the metric exists to preserve. If a model's window does
not overlap the true one at all, then an operator who pointed an antenna at the model's prediction
would have received nothing for the whole of the real pass and nothing for the whole of the
predicted one: two failures, not one late pass. Reporting a 900 s "shift" there would be a fiction.
The cost of the strict rule is that a *very* badly phased tier reports `lost == gained == n` instead
of a huge shift - which is the honest summary of a model whose passes no longer correspond to
reality at all, and `passes_lost` is the field that says so.

**Why greedy is unambiguous here.** Within one list the windows are disjoint and ordered (a run of
samples above the mask ends before the next begins), so a model window can overlap at most a few
truth windows, and only when the shift is comparable to the gap between passes. Descending overlap
with a deterministic tie-break makes the outcome independent of list order.

## Sampling step: derive it before choosing it

`geometry.access_windows` interpolates its edges linearly, an `O(h^2)` **bias** (not scatter):
rises read early, sets late, by `C a b` with `C = Omega cot(lambda_0) / 2`, `a + b = h`. For a
550 km circular orbit over an equatorial station, `C = 1.206e-3 s^-1`, so the raw edge bias at
`h = 30 s` is up to `C h^2 / 4 = 0.27 s`.

That is the *same order* as the quantity being measured - the most accurate tier in this engine
(Cowell + J2 at `dt = 60 s`) is about 1.9 km out after 24 h, which at `Omega = 1.024e-3 rad/s` and
`r = 6921 km` is a window shift of `(1.9 / 6921) / Omega = 0.27 s`. So the bias must be dealt with,
not tolerated.

**It is dealt with by using one grid for both sides.** Truth windows and model windows are computed
on the *same* time samples, so the bias is common-mode and cancels in the difference. Writing the
bias as `beta(a) = C a (h - a)` for a crossing `a` into its bracket, the model's crossing sits at
`a + Delta` and the residual is

    beta(a + Delta) - beta(a) = C Delta (h - 2a - Delta),      |residual| <= C Delta h

and - this is the part worth checking rather than assuming - the bound survives the case where the
two crossings fall in *different* brackets. There `a ~ h` and `a + Delta - h ~ 0`, so **both** biases
are near zero (`beta` vanishes at both ends of a bracket); the difference is again `O(C Delta h)`.
There is no regime in which the cancellation fails:

    h = 60 s, Delta = 0.27 s  ->  residual <= 2.0e-2 s   (7.2 % of the signal)
    h = 30 s, Delta = 0.27 s  ->  residual <= 9.8e-3 s   (3.6 % of the signal)
    h = 10 s, Delta = 0.27 s  ->  residual <= 3.3e-3 s   (1.2 % of the signal)

`DEFAULT_SAMPLE_DT_S = 60.0` is the first line, and it is chosen there rather than lower because of
a constraint from the other direction: **the propagation step must divide the sample spacing.**
`viz.sample_states` splits each sample interval into sub-steps of *at most* `max_dt`, so asking for
a 10 s grid from a `dt = 60 s` configuration would silently propagate it at 10 s and report a
different model from the one the sweep timed and scored. `sweep.access_metrics_for` raises rather
than let that happen, so `sample_dt_s` must be an integer multiple of every configuration's `dt`,
and 60 s is the engine's habitual Cowell step. A finer grid is strictly better wherever the
configurations allow it.

A tier whose window shifts are below ~0.02 s cannot be resolved on the default grid, and no tier in
this engine is.

`peak_elevation_rad` is sampled rather than refined (`geometry.AccessWindow`), so it is a lower
bound and is **not** differenced here. It is used only to *choose* a mask angle in the tests.

## The contact dataset: range and range rate on each window

`AccessMetrics` answers "how wrong is this model's schedule". A downstream network or ISL study
needs the other half - the *contents* of each contact, which for a link budget means **range** and
**range rate**. `contact_windows` (and `contacts_from_simulation`, its propagating twin) return
`ContactWindow` records: the `geometry.AccessWindow` unchanged, plus a `ContactSample`
(`time_s`, `range_km`, `range_rate_km_s`) at **rise, peak elevation and set**.

**Why those three instants.** They are the three the window already names, so the dataset needs no
new time grid and no re-propagation, and together they bracket what a link budget asks: the two
edges are the worst-case range and the *extremes* of range rate (for a coplanar circular pass the
range rate is stationary at the horizon - `rho'' = 0` there exactly, since `r cos lambda_0 = R`
makes the centripetal and the `rho_dot^2 / rho` terms cancel), and the peak is the closest approach,
where the range rate passes through zero. A consumer that wants the whole series has it already:
`geometry.elevation_azimuth(..., velocities_km_s=...)` returns it sample by sample, and
`ContactWindow` is the per-contact reduction of exactly that.

**Sign convention is `geometry.py`'s and is not re-stated by re-deriving it: positive = opening
(receding).** Doppler shift, carrier frequency and link margin are not computed here or anywhere in
this engine - they are properties of a radio, not of an orbit.

**Accuracy of the three samples.** Rise and set are *interpolated* times, so the series are linearly
interpolated onto them: `O(h^2)` with coefficient `|f''| h^2 / 8`. On the 550 km reference pass
`rho'' = 0` at the horizon exactly, so the rise/set **range** carries almost no interpolation error
of its own and is dominated by the edge-time bias itself (`rho_dot * 0.030 s = 0.20 km` at
`h = 10 s`); the rise/set **range rate** carries `|rho_dot''| h^2 / 8 = 8.5e-5 km/s` at `h = 10 s`
and `3.1e-3 km/s` at `h = 60 s`. The peak sample needs no interpolation - `peak_time_s` is a grid
time - but it inherits `peak_elevation_rad`'s `O(h)` sampling offset, so its range rate is not zero
but bounded by `rho'' h / 2` (0.42 km/s at `h = 10 s` on the reference pass, measured 0.18). Read
the peak as "near closest approach", exactly as `peak_elevation_rad` reads as "at least this high".

References
----------
No physics is introduced by this module: `geometry.py` carries the Vallado citations for the
topocentric geometry, and the interpolation error order is derived in `geometry.access_windows`'s
own docstring. The one derivation of its own is the bias-cancellation bound above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .custom_types import ArrayFloat, ArraySeconds, ScalarFloat, ScalarSeconds
from .geometry import AccessWindow, Topocentric, access_windows, elevation_azimuth
from .reference import ReferenceTrajectory
from .simulator import Simulation
from .viz import sample_states

__all__ = [
    "GroundStation", "AccessSpec", "WindowMatch", "ShiftStats", "AccessMetrics",
    "ContactSample", "ContactWindow",
    "DEFAULT_SAMPLE_DT_S",
    "access_grid", "windows_from_positions", "windows_from_truth", "windows_from_simulation",
    "contact_windows", "contacts_from_simulation",
    "match_windows", "summarise_matches", "compare_windows",
]

# See "Sampling step" in the module docstring: the residual edge-interpolation bias is C*Delta*h,
# which at 60 s is 7 % of the smallest window shift any tier in this engine produces - and 60 s is
# the coarsest step the propagation-divides-the-grid constraint leaves room for.
DEFAULT_SAMPLE_DT_S = 60.0


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundStation:
    """
    One observing site, in `geometry.py`'s **spherical (geocentric)** convention: latitude, east
    longitude and altitude above a sphere of the spec's `body_radius_km`. There is no ellipsoid in
    this engine - see `geometry.py`'s "Coordinates" section.

    `name` is for labelling only; nothing keys off it.
    """
    name: str
    latitude_rad: float
    longitude_rad: float
    altitude_km: float = 0.0


@dataclass(frozen=True)
class AccessSpec:
    """
    Everything needed to turn a trajectory into contact windows: where the stations are, which body
    they are fixed to, how fast it turns, and how finely to sample.

    `central_body` is the name of the body the stations sit on; positions are taken relative to it
    (`viz.sample_states(relative_to=...)` for the engine, a difference of two `position_of` calls for
    truth), because `geometry.elevation_azimuth` wants central-body-relative inertial positions and
    `global_states` is root-relative.

    `omega`, `theta0` and `epoch_s` are `geometry.elevation_azimuth`'s rotation parameters unchanged;
    in particular `theta0` is referenced to `epoch_s` (default 0.0 = absolute simulation time), *not*
    to the first sample, so a sweep's windows do not move when the grid moves.

    `sample_dt_s` defaults to `DEFAULT_SAMPLE_DT_S`; the module docstring derives why. It must be an
    integer multiple of every configuration's `dt`, because `viz.sample_states` sub-divides a sample
    interval into steps of *at most* `max_dt` - a 10 s grid asked of a `dt = 60 s` configuration
    would quietly propagate it at 10 s and score a model the sweep never ran.
    `sweep.access_metrics_for` raises on that rather than reporting it.

    `bodies`, when given, names the subset of bodies to report access for; the default is whatever
    body set the configuration itself governed, so the access metrics and the position-error
    statistics describe the same satellites.
    """
    stations: Sequence[GroundStation]
    central_body: str
    omega: float
    body_radius_km: float
    mask_angle_rad: float = 0.0
    sample_dt_s: float = DEFAULT_SAMPLE_DT_S
    theta0: float = 0.0
    epoch_s: float = 0.0
    bodies: Optional[Sequence[str]] = None


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowMatch:
    """
    One truth window, one model window, or a pairing of the two.

    Exactly one of `truth` / `model` is `None` for an orphan: a `None` `model` is a **lost** pass
    (truth had it, the model does not), a `None` `truth` is a **gained** one. Both `None` never
    occurs.

    The three difference properties return `None` rather than a number whenever the difference is
    not meaningful - no counterpart, or an edge that `geometry.access_windows` clipped to the grid
    endpoint rather than interpolating. A caller that filters on `is not None` therefore gets exactly
    the population the statistics are taken over.
    """
    station_index: int
    body_index: int
    truth: Optional[AccessWindow]
    model: Optional[AccessWindow]

    @property
    def matched(self) -> bool:
        return self.truth is not None and self.model is not None

    @property
    def rise_shift_s(self) -> Optional[float]:
        """`model.rise_s - truth.rise_s`; positive means the model rises **late**."""
        if self.truth is None or self.model is None:
            return None
        if self.truth.rise_clipped or self.model.rise_clipped:
            return None
        return self.model.rise_s - self.truth.rise_s

    @property
    def set_shift_s(self) -> Optional[float]:
        """`model.set_s - truth.set_s`; positive means the model sets late."""
        if self.truth is None or self.model is None:
            return None
        if self.truth.set_clipped or self.model.set_clipped:
            return None
        return self.model.set_s - self.truth.set_s

    @property
    def duration_error_s(self) -> Optional[float]:
        """`model.duration_s - truth.duration_s`, only when **all four** edges were interpolated."""
        if self.truth is None or self.model is None:
            return None
        if (self.truth.rise_clipped or self.truth.set_clipped
                or self.model.rise_clipped or self.model.set_clipped):
            return None
        return self.model.duration_s - self.truth.duration_s


@dataclass(frozen=True)
class ContactSample:
    """
    One instant inside a contact, with the two kinematic quantities a link budget needs.

    `range_rate_km_s` is `geometry.py`'s convention - **positive = opening (receding)**, negative =
    closing - and includes the station's own motion (the `omega x r_station` transport term). No
    Doppler shift, carrier frequency or link margin is computed anywhere in this engine: those are
    properties of a radio, and the boundary is drawn here deliberately.
    """
    time_s: float
    range_km: float
    range_rate_km_s: float


@dataclass(frozen=True)
class ContactWindow:
    """
    One access window plus range and range rate at its rise, peak-elevation and set instants.

    `window` is the `geometry.AccessWindow` unchanged, so every caveat it carries still applies:
    rise/set are interpolated (`O(h^2)`, a bias), `peak_elevation_rad` and `peak_time_s` are
    *sampled*. The three `ContactSample`s inherit those caveats - see "The contact dataset" in the
    module docstring for what each is worth numerically.

    The delegating properties exist so a consumer can treat this as a window that happens to know
    its ranges, rather than having to reach through `.window` for the identity of the contact.
    """
    window: AccessWindow
    rise: ContactSample
    peak: ContactSample
    set: ContactSample

    @property
    def station_index(self) -> int:
        return self.window.station_index

    @property
    def body_index(self) -> int:
        return self.window.body_index

    @property
    def rise_s(self) -> float:
        return self.window.rise_s

    @property
    def set_s(self) -> float:
        return self.window.set_s

    @property
    def duration_s(self) -> float:
        return self.window.duration_s

    @property
    def peak_elevation_rad(self) -> float:
        return self.window.peak_elevation_rad


@dataclass(frozen=True)
class ShiftStats:
    """
    One differenced quantity summarised over every pair that contributed, in seconds.

    Signed `mean_s` / `median_s` alongside `mean_abs_s` / `max_abs_s` - see the module docstring on
    why both are reported. `n` is the population size; every field is 0.0 when `n == 0`, which is a
    statement that nothing was measurable, not that the error was zero (`n` is what distinguishes
    them).
    """
    n: int
    mean_s: float
    median_s: float
    mean_abs_s: float
    max_abs_s: float


@dataclass(frozen=True)
class AccessMetrics:
    """
    One configuration's contact-window error against truth, aggregated over every station and body.

    Counts first, because the discrete failure is the one no continuous metric shows:
    `passes_lost` / `passes_gained` are windows with no counterpart under the overlap matching rule
    (module docstring). `n_matched + passes_lost == n_truth_windows` and
    `n_matched + passes_gained == n_model_windows` always hold.

    `total_contact_*_s` sum every window's duration including clipped ones;
    `total_contact_error_s` is `model - truth`, the bottom-line number for a coverage study.

    `matches` retains the individual pairings, in time order within each `(station, body)` pair, so a
    figure or a report can show the distribution rather than only its summary.
    """
    n_truth_windows: int
    n_model_windows: int
    n_matched: int
    passes_lost: int
    passes_gained: int
    n_clipped_edges: int
    rise: ShiftStats
    set: ShiftStats
    duration: ShiftStats
    total_contact_truth_s: float
    total_contact_model_s: float
    total_contact_error_s: float
    matches: Tuple[WindowMatch, ...] = field(default=(), repr=False)


# --------------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------------

def access_grid(horizon_s: ScalarSeconds, sample_dt_s: ScalarSeconds) -> ArraySeconds:
    """
    The uniform sample grid `[0, horizon_s]` with spacing as close to `sample_dt_s` as an integer
    number of intervals allows.

    `np.linspace` rather than `np.arange`: the horizon must be hit *exactly*, so that a window still
    open at the end is flagged `set_clipped` at the intended time rather than at a
    floating-point-accumulated one, and so that truth and model share bit-identical sample times.
    """
    horizon = float(horizon_s)
    step = float(sample_dt_s)
    if horizon <= 0.0:
        raise ValueError(f"horizon_s must be positive, got {horizon}.")
    if step <= 0.0:
        raise ValueError(f"sample_dt_s must be positive, got {step}.")
    n_intervals = max(1, int(round(horizon / step)))
    grid: ArraySeconds = np.linspace(0.0, horizon, n_intervals + 1, dtype=np.float64)
    return grid


def _station_arrays(spec: AccessSpec) -> Tuple[ArrayFloat, ArrayFloat, ArrayFloat]:
    if len(spec.stations) == 0:
        raise ValueError("AccessSpec needs at least one station.")
    lat = np.array([s.latitude_rad for s in spec.stations], dtype=np.float64)
    lon = np.array([s.longitude_rad for s in spec.stations], dtype=np.float64)
    alt = np.array([s.altitude_km for s in spec.stations], dtype=np.float64)
    return lat, lon, alt


def _topocentric(
    positions_km: ArrayFloat,
    times_s: ArraySeconds,
    spec: AccessSpec,
    velocities_km_s: Optional[ArrayFloat] = None,
) -> Topocentric:
    """
    The single place `geometry.elevation_azimuth` is called from this module.

    Truth and model can then only ever differ in the trajectory handed in - never in the station
    geometry, the rotation phase or the mask angle - and the range-rate path cannot drift out of
    step with the look-angle path, since there is only one call.
    """
    lat, lon, alt = _station_arrays(spec)
    return elevation_azimuth(
        positions_km, times_s,
        velocities_km_s=velocities_km_s,
        latitude_rad=lat, longitude_rad=lon, altitude_km=alt,
        omega=spec.omega, body_radius_km=spec.body_radius_km,
        theta0=spec.theta0, epoch_s=spec.epoch_s,
    )


def windows_from_positions(
    positions_km: ArrayFloat, times_s: ArraySeconds, spec: AccessSpec,
) -> List[AccessWindow]:
    """
    Contact windows for `(n_times, n_bodies, 3)` **central-body-relative inertial** positions.

    Positions only, so no range rate: this is the path `AccessMetrics` uses, and truth trajectories
    from `reference.py` are differenced here by position. `contact_windows` is the velocity-carrying
    counterpart.
    """
    topo = _topocentric(positions_km, times_s, spec)
    return access_windows(times_s, topo.elevation_rad, mask_angle_rad=spec.mask_angle_rad)


def windows_from_truth(
    truth: ReferenceTrajectory, body_names: Sequence[str], spec: AccessSpec,
) -> List[AccessWindow]:
    """
    Truth windows from a `reference.ReferenceTrajectory` sampled on the access grid.

    `truth.times` must already be that grid - it is not resampled, for the same reason
    `viz.position_error` refuses to resample: a silent interpolation would hide a grid mismatch as a
    plausible-looking window shift, and the whole point of sharing one grid is that the edge bias
    cancels.
    """
    central = truth.position_of(spec.central_body)
    stacked = np.stack([truth.position_of(name) - central for name in body_names], axis=1)
    return windows_from_positions(stacked, np.asarray(truth.times, dtype=np.float64), spec)


def windows_from_simulation(
    sim: Simulation,
    bodies: Sequence[int],
    times_s: ArraySeconds,
    spec: AccessSpec,
    *,
    max_dt: Optional[ScalarSeconds] = None,
) -> List[AccessWindow]:
    """
    Propagate `sim` across `times_s` and return its contact windows.

    **This advances the simulation** - `viz.sample_states` does the stepping, so pass a freshly built
    one. `max_dt` is the propagation step: pass a configuration's own `dt` and the trajectory sampled
    here is step-for-step the one that configuration's error statistics were taken from.
    """
    central = sim.name_to_index[spec.central_body]
    states = sample_states(sim, bodies, times_s, relative_to=central, max_dt=max_dt)
    return windows_from_positions(states[..., :3], times_s, spec)


# --------------------------------------------------------------------------------------------------
# The contact dataset: windows carrying range and range rate
# --------------------------------------------------------------------------------------------------

def _sample_at(
    times: ArraySeconds, ranges: ArrayFloat, rates: ArrayFloat, t: float,
) -> ContactSample:
    """
    Both series read at one instant, linearly interpolated between the bracketing samples.

    `np.interp` is exact at a node, so a peak (a grid time) and a clipped edge (the first or last
    grid time) are read rather than interpolated, and it clamps outside the grid, so no edge can
    extrapolate a range that was never sampled.
    """
    return ContactSample(
        time_s=float(t),
        range_km=float(np.interp(t, times, ranges)),
        range_rate_km_s=float(np.interp(t, times, rates)),
    )


def contact_windows(
    positions_km: ArrayFloat,
    velocities_km_s: ArrayFloat,
    times_s: ArraySeconds,
    spec: AccessSpec,
) -> List[ContactWindow]:
    """
    Contact windows carrying range and range rate at rise, peak elevation and set.

    `positions_km` and `velocities_km_s` are both `(n_times, n_bodies, 3)`,
    **central-body-relative inertial** - columns `:3` and `3:` of
    `viz.sample_states(..., relative_to=central)`, which is why that function already returns full
    six-element states. The windows themselves are `windows_from_positions`' windows exactly: the
    same look angles, the same mask, the same interpolated edges.

    Rise and set fall *between* samples, so the range and range-rate series are linearly
    interpolated onto them (`np.interp`, which is exact at a node and clamps outside the grid - a
    clipped edge sits on the first or last sample and therefore reads that sample). The peak sample
    is read directly at `AccessWindow.peak_time_s`, a grid time. The module docstring gives what
    each of the three is worth numerically; the short version is that the rise/set range rate
    carries `|rho_dot''| h^2 / 8` and the peak range rate is not zero but bounded by `rho'' h / 2`.

    Sign: **positive range rate = opening (receding)**, `geometry.py`'s convention.
    """
    topo = _topocentric(positions_km, times_s, spec, velocities_km_s)
    rate = topo.range_rate_km_s
    if rate is None:  # pragma: no cover - `_topocentric` was given velocities, so this cannot fire
        raise ValueError("range rate was not computed; velocities_km_s must be provided.")

    times = np.asarray(times_s, dtype=np.float64)
    windows = access_windows(times_s, topo.elevation_rad, mask_angle_rad=spec.mask_angle_rad)

    contacts: List[ContactWindow] = []
    for w in windows:
        ranges = topo.range_km[:, w.station_index, w.body_index]
        rates = rate[:, w.station_index, w.body_index]
        contacts.append(ContactWindow(
            window=w,
            rise=_sample_at(times, ranges, rates, w.rise_s),
            peak=_sample_at(times, ranges, rates, w.peak_time_s),
            set=_sample_at(times, ranges, rates, w.set_s),
        ))
    return contacts


def contacts_from_simulation(
    sim: Simulation,
    bodies: Sequence[int],
    times_s: ArraySeconds,
    spec: AccessSpec,
    *,
    max_dt: Optional[ScalarSeconds] = None,
) -> List[ContactWindow]:
    """
    Propagate `sim` across `times_s` and return its contact dataset.

    **This advances the simulation**, exactly as `windows_from_simulation` does and for the same
    reason - `viz.sample_states` is what steps it. The only difference between the two is that this
    one keeps the velocity columns instead of discarding them.
    """
    central = sim.name_to_index[spec.central_body]
    states = sample_states(sim, bodies, times_s, relative_to=central, max_dt=max_dt)
    return contact_windows(
        np.ascontiguousarray(states[..., :3]), np.ascontiguousarray(states[..., 3:]), times_s, spec,
    )


# --------------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------------

def _group(windows: Sequence[AccessWindow]) -> Dict[Tuple[int, int], List[AccessWindow]]:
    grouped: Dict[Tuple[int, int], List[AccessWindow]] = {}
    for w in windows:
        grouped.setdefault((w.station_index, w.body_index), []).append(w)
    return grouped


def _match_one_pair(
    truth: Sequence[AccessWindow], model: Sequence[AccessWindow],
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Greedy maximum-overlap pairing of two window lists for a single `(station, body)` pair.

    Returns index pairs into the two lists, with `None` on the side that has no counterpart.
    Candidates are sorted by descending overlap, then ascending truth index, then ascending model
    index, so the result does not depend on the order the windows arrived in. See the module
    docstring for why overlap is the only admissible criterion.
    """
    candidates: List[Tuple[float, int, int]] = []
    for i, tw in enumerate(truth):
        for j, mw in enumerate(model):
            overlap = min(tw.set_s, mw.set_s) - max(tw.rise_s, mw.rise_s)
            if overlap > 0.0:
                candidates.append((-overlap, i, j))
    candidates.sort()

    partner_of_truth: Dict[int, int] = {}
    partner_of_model: Dict[int, int] = {}
    for _, i, j in candidates:
        if i in partner_of_truth or j in partner_of_model:
            continue
        partner_of_truth[i] = j
        partner_of_model[j] = i

    pairs: List[Tuple[Optional[int], Optional[int]]] = [
        (i, partner_of_truth.get(i)) for i in range(len(truth))
    ]
    pairs.extend((None, j) for j in range(len(model)) if j not in partner_of_model)
    # Time order, so a report reads chronologically: an orphaned model window sorts by its own rise.
    pairs.sort(key=lambda p: truth[p[0]].rise_s if p[0] is not None else model[p[1]].rise_s)  # type: ignore[index]
    return pairs


def match_windows(
    truth_windows: Sequence[AccessWindow], model_windows: Sequence[AccessWindow],
) -> List[WindowMatch]:
    """
    Pair each model window with the truth window it overlaps most, per `(station, body)`.

    Pure interval arithmetic - no `Simulation`, no propagation, no time grid. Ordered by station,
    then body, then time.
    """
    truth_groups = _group(truth_windows)
    model_groups = _group(model_windows)
    keys = sorted(set(truth_groups) | set(model_groups))

    matches: List[WindowMatch] = []
    for key in keys:
        station_index, body_index = key
        t_list = truth_groups.get(key, [])
        m_list = model_groups.get(key, [])
        for i, j in _match_one_pair(t_list, m_list):
            matches.append(WindowMatch(
                station_index=station_index,
                body_index=body_index,
                truth=None if i is None else t_list[i],
                model=None if j is None else m_list[j],
            ))
    return matches


# --------------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------------

def _shift_stats(values: Sequence[float]) -> ShiftStats:
    if len(values) == 0:
        return ShiftStats(n=0, mean_s=0.0, median_s=0.0, mean_abs_s=0.0, max_abs_s=0.0)
    arr = np.asarray(values, dtype=np.float64)
    return ShiftStats(
        n=int(arr.size),
        mean_s=float(np.mean(arr)),
        median_s=float(np.median(arr)),
        mean_abs_s=float(np.mean(np.abs(arr))),
        max_abs_s=float(np.max(np.abs(arr))),
    )


def _clipped_edges(window: Optional[AccessWindow]) -> int:
    if window is None:
        return 0
    return int(window.rise_clipped) + int(window.set_clipped)


def summarise_matches(
    matches: Sequence[WindowMatch],
    truth_windows: Sequence[AccessWindow],
    model_windows: Sequence[AccessWindow],
) -> AccessMetrics:
    """
    Reduce a matching to `AccessMetrics`.

    The two window lists are passed in as well as the matching, so total contact time is summed over
    the windows *as reported by `geometry.access_windows`* rather than reconstructed from the pairs -
    a matcher bug can then move windows between the matched and orphaned populations without also
    silently changing the total contact time, which is what makes the two families of numbers
    independent checks on each other.
    """
    rises = [m.rise_shift_s for m in matches]
    sets = [m.set_shift_s for m in matches]
    durations = [m.duration_error_s for m in matches]

    n_matched = sum(1 for m in matches if m.matched)
    truth_total = float(sum(w.duration_s for w in truth_windows))
    model_total = float(sum(w.duration_s for w in model_windows))

    return AccessMetrics(
        n_truth_windows=len(truth_windows),
        n_model_windows=len(model_windows),
        n_matched=n_matched,
        passes_lost=sum(1 for m in matches if m.truth is not None and m.model is None),
        passes_gained=sum(1 for m in matches if m.truth is None and m.model is not None),
        n_clipped_edges=sum(_clipped_edges(w) for w in truth_windows)
        + sum(_clipped_edges(w) for w in model_windows),
        rise=_shift_stats([v for v in rises if v is not None]),
        set=_shift_stats([v for v in sets if v is not None]),
        duration=_shift_stats([v for v in durations if v is not None]),
        total_contact_truth_s=truth_total,
        total_contact_model_s=model_total,
        total_contact_error_s=model_total - truth_total,
        matches=tuple(matches),
    )


def compare_windows(
    truth_windows: Sequence[AccessWindow], model_windows: Sequence[AccessWindow],
) -> AccessMetrics:
    """Match, then summarise - the whole comparison for two already-computed window lists."""
    return summarise_matches(
        match_windows(truth_windows, model_windows), truth_windows, model_windows
    )


# --------------------------------------------------------------------------------------------------
# Reporting helper
# --------------------------------------------------------------------------------------------------

def format_metrics(name: str, metrics: AccessMetrics) -> str:
    """One-line summary for a console sweep report; the figure script uses the fields directly."""
    return (
        f"{name:<34} windows {metrics.n_model_windows:>3}/{metrics.n_truth_windows:<3} "
        f"lost {metrics.passes_lost:>2} gained {metrics.passes_gained:>2}  "
        f"rise {metrics.rise.mean_s:>+9.3f}s (|.| mean {metrics.rise.mean_abs_s:>8.3f}s "
        f"max {metrics.rise.max_abs_s:>8.3f}s)  "
        f"dur {metrics.duration.mean_s:>+8.3f}s  "
        f"contact {metrics.total_contact_error_s:>+9.2f}s"
    )


def contact_fraction(metrics: AccessMetrics) -> float:
    """
    `total_contact_model_s / total_contact_truth_s`, or `math.nan` when truth had no contact at all.

    Reported as a ratio rather than only a difference because a coverage requirement is usually
    stated as a fraction of a duty cycle.
    """
    if metrics.total_contact_truth_s == 0.0:
        return math.nan
    return metrics.total_contact_model_s / metrics.total_contact_truth_s


def stations_from_mapping(mapping: Mapping[str, Tuple[float, float, float]]) -> List[GroundStation]:
    """
    Convenience: `{"Kiruna": (lat_rad, lon_rad, alt_km)}` -> a list of `GroundStation`.

    Ordered by the mapping's own iteration order, which is insertion order, so a station's index in
    `WindowMatch.station_index` is its position in the literal the caller wrote.
    """
    return [GroundStation(n, lat, lon, alt) for n, (lat, lon, alt) in mapping.items()]
