"""
Altitude station-keeping: hold a satellite's **mean** altitude inside a band with impulsive
prograde raises, and report what that costs in Delta-v.

The engine's purpose is to measure what a modelling assumption costs in the units of the decision it
affects. For drag that unit is **propellant**: how much Delta-v must be budgeted to keep a satellite on
station, and how much does the answer move when the atmosphere model changes? This module is the
controller that turns a decay into that number. It adds no physics - drag is `drag.py`, the density
law `atmosphere.py`, the impulse `manoeuvres.py` - only the decision of *when* to burn and *how much*.

The controller as data
----------------------
`StationKeepingSpec(lower_km, upper_km, r_ref_km, impulses)` is plain frozen data, like
`manoeuvres.Manoeuvre` and `events.Event`. The policy it encodes is the textbook dead-band one:

    when the mean altitude m falls to lower_km, raise it to upper_km.

`impulses=2` (the default) does the raise as two prograde burns half a transfer orbit apart - a
Hohmann transfer between the circular mean radii `a1 = r_ref + m` and `a2 = r_ref + upper_km` - which
leaves the orbit circular. `impulses=1` does it with one prograde burn, which raises apogee only and
leaves an eccentricity `e ~ (a2 - a1)/(2 a)`; it is offered because it is what a simple operator
does, and it costs the same Delta-v to first order. The Hohmann Delta-v, exactly
(Vallado 4e Sec. 6.3; Curtis 3e Sec. 6.2 - section numbers from memory, **unverified**, but the form
follows from vis-viva alone):

    dv1 = sqrt(mu/a1) (sqrt(2 a2/(a1 + a2)) - 1)
    dv2 = sqrt(mu/a2) (1 - sqrt(2 a1/(a1 + a2)))
    dv1 + dv2 = (n/2) (a2 - a1) [1 + O((a2 - a1)/a)],   n = sqrt(mu/a^3)

The single-impulse form is `sqrt(mu (2/a1 - 1/a2)) - sqrt(mu/a1)`, the same to first order.

`(n/2) da` is the whole economics of drag make-up. For a circular orbit Gauss's equation gives
`da/dt = 2 f_S / n` for an along-track specific force `f_S`, so undoing a loss `da` costs
`dv = (n/2) da` - and the Delta-v per unit time is the along-track drag acceleration itself.
`tests/validation/test_stationkeeping.py` checks the controller's total against exactly that, from
the orbit-averaged decay rate `drag.py` is already validated against, not from this module's
bookkeeping.

What the controller keys on, and why
------------------------------------
**Not the osculating altitude.** Under J2 a circular LEO orbit's osculating radius oscillates at twice
the orbital frequency with an amplitude of order `J2 R^2 / a` - measured **12 km peak to peak** at
300 km and 51.6 deg (288.3 to 300.0 km from a seeded 300.0 km) - and any eccentricity adds a
once-per-rev term `a e` on top. A controller that fires when *osculating* altitude touches the lower
bound fires once per orbit at the J2 minimum while the orbit is still kilometres inside the band, then
raises from that minimum and overshoots the upper bound by the same kilometres. The osculating
semi-major axis is no better: vis-viva `a` swings 287.9 to 300.0 km over the same orbit.

**Not a Brouwer/Kozai mean element either.** An osculating-to-mean conversion evaluated from the
instantaneous state would be the one quantity an `events.Event` could key on, but `CLAUDE.md` scopes
such conversions narrowly (`propagators.mean_seeded_p` is the only one, and it touches only `p`), and
a first-order theory leaves `O(J2^2)` residue of its own. A plain time average needs no theory.

**The one-period time average of the osculating altitude**, `m(t) = (1/T) integral_{t-T}^{t}
(|r| - r_ref) dt`, with `T` the Keplerian period at the band's midpoint radius. Every short-period
term - J2's `2u` terms and the eccentricity's `u` terms - is periodic in the argument of latitude with
a period dividing the orbital period, so a one-period average annihilates it. What survives is the
secular trend, which is what drag is and what station-keeping must answer. Radius rather than
semi-major axis because the band is an *altitude* band and density is a function of radius; under J2
the two means differ by a constant offset (0.79 km here) that neither the decay rate nor the
`(n/2) da` conversion notices at the `1e-4` level.

Two corrections make that estimator exact enough to hold a band a few km wide:

- **Exact window length.** `T / dt` is not an integer, and a rectangle over `round(T/dt)` samples
  leaks `A * |N dt - T| / T` of an amplitude-`A` oscillation - 30 m for `A = 6 km` at `dt = 60 s`. The
  window is instead the trapezoidal integral of the sampled series' piecewise-linear interpolant over
  exactly `T`, with the oldest interval taken fractionally. The trapezoidal rule over a whole period
  of a periodic function is spectrally accurate, so the residual is set by how far `T` is from the
  *true* short-period period (draconitic, `O(J2)` away from Keplerian) - measured in the drag-free
  control at **about 1 m rms** of fast residual. (That control's mean does drift, by 30 m/day at
  `dt = 60 s`: that is RK4's own energy loss, `(n h)^6 / 36` per step, not the estimator - it falls
  32-fold at 30 s, as `h^5` per unit time should.)
- **Lag.** A trailing average describes the altitude at the window's centre, `T/2` ago. Under a
  decay `mdot` it reads `|mdot| T/2` high - **0.18 km** at 300 km, B = 0.05 m^2/kg, which is 7 % of a
  2.5 km band and would let the true mean sag through the lower bound before every burn. The slope is
  estimated from two consecutive windows, `(m(t) - m(t - N dt)) / (N dt)`, and the estimate is
  extrapolated `T/2` forward. For a linear trend that is exact; the residual is `O(mddot T^2)`.

The price is **observability**. The estimate needs two full windows (`N + 2` and `N` samples) of
post-burn history, so after each raise the controller is blind for about `2 T` (plus the half
transfer orbit before a Hohmann second burn). It will not fire during that time, which is correct:
the orbit has just been put at the top of the band. `StationKeeper` raises nothing if the band is so
narrow that the orbit decays through it in that time - but `tests/validation/test_stationkeeping.py`
asserts the band is held, and the design condition is simply `upper - lower > 2.5 T |mdot|`.

Trigger: a check between steps, not an `events.Event`
-----------------------------------------------------
The trigger is evaluated once per step, after `Simulation.step`, by `StationKeeper.observe`. An
`events.Event` was considered and rejected, for three reasons, the first two of which are structural:

1. **An event function must be a pure read of the instantaneous arena** (`events.py`). The mean
   altitude is a function of the last two orbits of *history*, which a trial propagation inside a
   root find does not produce. The only instantaneous mean-altitude estimator is an osculating-to-mean
   conversion, rejected above.
2. **`events.Event` has no action.** It splits the step and records the crossing epoch; nothing in
   `events.py` or `Simulation` fires a callback at the crossing. Burning at an event would need an
   upstream hook - an `Event.action(sim, bodies)` run between the split sub-steps - which belongs in
   `events.py`/`simulator.py`, not here.
3. **The precision would buy nothing.** Event location exists for discontinuities that cost RK4 its
   order. The mean altitude is smooth and crosses the threshold at `|mdot| ~ 7e-5 km/s`, so detecting
   it up to one step late moves the trigger level by at most `|mdot| dt` = **4 m** at `dt = 60 s`,
   under 0.2 % of the band - and it cancels out of the Delta-v rate, because the burn raises from the
   level actually reached. The burn itself lands on a step boundary, where an impulse is exact.

The **second** Hohmann burn does have an exact epoch - half a transfer period after the first - and it
is placed with `Simulation.schedule_delta_v`, which splits the step there.

Limitations
-----------
- Altitude is geocentric radius minus `r_ref`, the same spherical altitude `drag.py` uses.
- Mean *altitude* only: nothing here controls eccentricity, inclination or phasing. A single-impulse
  raise leaves eccentricity that the two-impulse raise avoids.
- A two-impulse raise is sized at the trigger and takes half a transfer orbit, during which drag keeps
  acting at the mid-band density; the orbit ends `|mdot| tau` (~0.18 km here) short of `upper_km`.
  That is not corrected, deliberately - chasing it would need a third burn - and it shortens each
  cycle by `tau (U - L) / (2 H)`, 2e-3 of it at 300 km, which the validation's prediction carries.
- **Integrator truncation under drag is not calibrated by a drag-free twin.** At `dt = 60 s` a
  drag-carrying satellite's RK4 decay exceeds the drag-free control's by 3.2e-3 of the drag rate
  (table) or 1.7e-3 (single band), falling as `h^4`. A Delta-v budget read off a coarse-step Cowell
  run inherits it; `tests/validation/test_stationkeeping.py` runs at 30 s for that reason.
- The burn is impulsive (`manoeuvres.py`) and draws on no propellant model; the report is Delta-v, and
  converting it to mass needs an Isp this module has no business assuming.
- `observe` must be called after every step of one fixed `dt`, and raises otherwise.
- The window period is Keplerian at the band midpoint. A band hundreds of km wide, or a body whose
  draconitic period differs from Keplerian by much more than J2's `1e-3`, leaks more of the
  short-period oscillation into the estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .geopotential import EARTH_R_EQ

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "StationKeepingSpec", "Burn", "StationKeeper", "StationKeepingRun", "BurnSummary",
    "hohmann_raise_dv", "single_impulse_raise_dv", "summarise_burns", "run_station_keeping",
    "MIN_SAMPLES_PER_ORBIT", "window_period_s", "steady_rate", "BodyDeltaV", "DeltaVMetrics",
    "delta_v_metrics",
]

#: The fewest steps per window period `StationKeeper` accepts. The trapezoidal window integrates a
#: harmonic of `k` cycles per period to zero unless `k` is a multiple of the sample count, and J2's
#: short-period terms are `k = 1, 2`, so fewer than 3 samples alias the `2u` term straight into the
#: mean; 4 leaves one sample of margin. It is a floor for the estimator, not an accuracy statement:
#: a Cowell tier that coarse is far outside RK4's useful range long before the window fails.
MIN_SAMPLES_PER_ORBIT: Final = 4.0

#: Prograde, in `manoeuvres.py`'s RSW convention.
_PROGRADE: Final = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class StationKeepingSpec:
    """
    One station-keeping policy: hold the one-period mean altitude inside `[lower_km, upper_km]`.

    Plain frozen data. `r_ref_km` is the radius altitudes are measured from - `EARTH_R_EQ` by default,
    the same `r_ref` `drag.py` should be configured with, so "altitude" means the same thing to the
    atmosphere and to the controller. `impulses` is 2 (Hohmann-style raise, circular after) or 1
    (single prograde burn). See the module docstring for the policy and what it keys on.
    """

    lower_km: float
    upper_km: float
    r_ref_km: float = EARTH_R_EQ
    impulses: int = 2

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lower_km) and math.isfinite(self.upper_km)):
            raise ValueError("station-keeping band edges must be finite")
        if not self.upper_km > self.lower_km:
            raise ValueError(
                f"station-keeping band needs upper_km > lower_km, got [{self.lower_km}, "
                f"{self.upper_km}]")
        if not self.lower_km > -self.r_ref_km:
            raise ValueError("lower_km puts the band below the centre of the reference body")
        if self.impulses not in (1, 2):
            raise ValueError(f"impulses must be 1 or 2, got {self.impulses!r}")


@dataclass(frozen=True)
class Burn:
    """
    One raise, as the controller decided it: which slot, when, how much in total, and from where.

    `dv_km_s` is the **total** of the raise - both impulses of a two-impulse one. `epoch_s` is the
    first impulse; `second_epoch_s` is the second (`nan` for a single-impulse raise).
    `mean_altitude_km` is the controller's lag-corrected mean altitude at the trigger, and
    `target_km` the altitude it raised towards.
    """

    body: int
    epoch_s: float
    dv_km_s: float
    mean_altitude_km: float
    target_km: float
    second_epoch_s: float = field(default=float("nan"))


def hohmann_raise_dv(mu: float, a1: float, a2: float) -> tuple[float, float]:
    """The two prograde Delta-vs (km/s) of a Hohmann transfer between circular radii `a1 < a2` (km)."""
    at = 0.5 * (a1 + a2)
    dv1 = math.sqrt(mu / a1) * (math.sqrt(a2 / at) - 1.0)
    dv2 = math.sqrt(mu / a2) * (1.0 - math.sqrt(a1 / at))
    return dv1, dv2


def single_impulse_raise_dv(mu: float, a1: float, a2: float) -> float:
    """The prograde Delta-v (km/s) that takes a circular orbit of radius `a1` to semi-major axis `a2`."""
    return math.sqrt(mu * (2.0 / a1 - 1.0 / a2)) - math.sqrt(mu / a1)


def _window_weights(n_float: float) -> NDArray[np.float64]:
    """
    Weights over the newest `N + 2` samples (index 0 = newest) whose dot product with the samples is
    the mean of their piecewise-linear interpolant over exactly `n_float` sample intervals.

    `N = floor(n_float)` whole intervals are trapezoids; the remaining fraction `f` of the next
    interval back is integrated with the interpolant's value at the window edge,
    `s_N + f (s_{N+1} - s_N)`, so its area is `f (s_N + s_edge) / 2`. The weights sum to 1.
    """
    n = int(math.floor(n_float))
    f = n_float - n
    w = np.zeros(n + 2, dtype=np.float64)
    w[: n + 1] += 1.0
    w[0] -= 0.5
    w[n] -= 0.5
    # Partial interval [s_N, s_edge]: area f * (s_N + s_N + f (s_{N+1} - s_N)) / 2.
    w[n] += f * (1.0 - 0.5 * f)
    w[n + 1] += 0.5 * f * f
    w /= n_float
    return w


def window_period_s(spec: StationKeepingSpec, mu: float) -> float:
    """The controller's averaging window (s): the Keplerian period at the band's midpoint radius."""
    a_mid = spec.r_ref_km + 0.5 * (spec.lower_km + spec.upper_km)
    return 2.0 * math.pi * math.sqrt(a_mid ** 3 / mu)


class StationKeeper:
    """
    The dead-band controller, applied to `bodies` of one `Simulation` stepped at a fixed `dt`.

    Call `observe()` once after every `sim.step(dt)`, and once before the first step to take the
    initial sample (`run_station_keeping` does both). Each call samples every body's osculating
    altitude relative to its `parent_indices` parent, updates the lag-corrected one-period mean
    (`mean_altitude_km`, `nan` while a body has too little post-burn history), and fires a raise for
    every body whose mean has reached `spec.lower_km`: the first impulse now through
    `Simulation.apply_delta_v`, the second (if any) through `Simulation.schedule_delta_v`.

    State is the sample ring buffer and the burn log - held here, never in the arena, because the
    arena's state of record is what the physics needs and this is what the *controller* needs. The
    buffer is one `(2N + 2, k)` array shared by every body: all bodies are sampled on the same clock,
    and each body only differs in how many of the newest samples are post-burn (`_valid`).
    """

    def __init__(
        self,
        sim: "Simulation",
        bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
        spec: StationKeepingSpec,
        dt: float,
    ) -> None:
        idx = np.atleast_1d(np.asarray(bodies, dtype=np.int64))
        if idx.size == 0:
            raise ValueError("StationKeeper needs at least one body")
        if np.unique(idx).size != idx.size:
            raise ValueError(f"duplicate slots in {idx.tolist()}")
        if not dt > 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        parents = sim.parent_indices[idx]
        if bool(np.any(parents == idx)):
            raise ValueError("a root body has no parent to hold an altitude above")
        mu = sim.mu_array[idx] + sim.mu_array[parents]
        if not bool(np.all(mu > 0.0)):
            raise ValueError("every station-kept body needs a massive parent")

        self.sim = sim
        self.bodies: NDArray[np.int64] = idx
        self.spec = spec
        self.dt = float(dt)
        self._mu: NDArray[np.float64] = mu.astype(np.float64)

        # One window = the Keplerian period at the band's midpoint radius, in sample intervals.
        # Per body only through mu, which is the same for every satellite of one parent; take the
        # first body's and require the rest to agree, so the shared buffer means one thing.
        if not bool(np.all(mu == mu[0])):
            raise ValueError("all station-kept bodies must share one parent mass (one window length)")
        self.period_s = window_period_s(spec, float(mu[0]))
        self._n_float = self.period_s / self.dt
        if self._n_float < MIN_SAMPLES_PER_ORBIT:
            raise ValueError(
                f"dt={dt} s gives only {self._n_float:.2f} samples per orbit; the one-period mean "
                f"needs a step well below the orbital period ({self.period_s:.0f} s)")
        self._n_shift = int(math.floor(self._n_float))
        self._weights = _window_weights(self._n_float)
        self._need = self._weights.size + self._n_shift         # samples for both windows
        self._buf: NDArray[np.float64] = np.zeros((self._need, idx.size), dtype=np.float64)
        self._ptr = 0                                           # next write row
        self._valid: NDArray[np.int64] = np.zeros(idx.size, dtype=np.int64)
        self._holdoff: NDArray[np.float64] = np.full(idx.size, -np.inf, dtype=np.float64)
        self._last_t: Optional[float] = None

        self.mean_altitude_km: NDArray[np.float64] = np.full(idx.size, np.nan, dtype=np.float64)
        self.osculating_altitude_km: NDArray[np.float64] = np.full(
            idx.size, np.nan, dtype=np.float64)
        self.burns: List[Burn] = []

    # ----------------------------------------------------------------------------------------------

    def _altitude(self) -> NDArray[np.float64]:
        g = self.sim.global_states
        rel = g[self.bodies, :3] - g[self.sim.parent_indices[self.bodies], :3]
        out: NDArray[np.float64] = np.sqrt(np.einsum("ij,ij->i", rel, rel)) - self.spec.r_ref_km
        return out

    def _window_mean(self, age: int) -> NDArray[np.float64]:
        """Mean over the window whose newest sample is `age` samples old."""
        rows = (self._ptr - 1 - age - np.arange(self._weights.size)) % self._need
        out: NDArray[np.float64] = self._weights @ self._buf[rows]
        return out

    def observe(self) -> None:
        """Sample, update the mean, and fire any raise that is due. See the class docstring."""
        t = float(self.sim.t)
        if self._last_t is not None and abs((t - self._last_t) - self.dt) > 1e-6 * self.dt:
            raise ValueError(
                f"StationKeeper.observe must follow every step of dt={self.dt} s; the clock moved "
                f"{t - self._last_t} s since the last call")
        self._last_t = t

        h = self._altitude()
        self.osculating_altitude_km = h
        self._buf[self._ptr] = h
        self._ptr = (self._ptr + 1) % self._need
        # A sample counts towards a body's window only once every impulse of its last raise is in
        # the state: `schedule_delta_v` applies the second one inside the step that reaches its epoch,
        # so the sample at `t >= epoch` is already post-burn.
        accepted = t >= self._holdoff
        self._valid = np.where(accepted, self._valid + 1, 0)

        recent = self._window_mean(0)
        older = self._window_mean(self._n_shift)
        slope = (recent - older) / (self._n_shift * self.dt)
        estimate = recent + slope * (0.5 * self.period_s)       # centre of window -> now
        self.mean_altitude_km = np.where(self._valid >= self._need, estimate, np.nan)

        due = np.flatnonzero(self.mean_altitude_km <= self.spec.lower_km)
        if due.size > 0:
            self._fire(due, t)

    def _fire(self, due: NDArray[np.int64], t: float) -> None:
        spec = self.spec
        a2 = spec.r_ref_km + spec.upper_km
        for k in due.tolist():
            mu = float(self._mu[k])
            m = float(self.mean_altitude_km[k])
            a1 = spec.r_ref_km + m
            body = int(self.bodies[k])
            if spec.impulses == 2:
                dv1, dv2 = hohmann_raise_dv(mu, a1, a2)
                half_transfer = math.pi * math.sqrt((0.5 * (a1 + a2)) ** 3 / mu)
                epoch2 = t + half_transfer
                self.sim.apply_delta_v(body, dv1 * _PROGRADE)
                self.sim.schedule_delta_v(body, dv2 * _PROGRADE, epoch2, label="station-keeping")
                self._holdoff[k] = epoch2
                total = dv1 + dv2
            else:
                dv = single_impulse_raise_dv(mu, a1, a2)
                self.sim.apply_delta_v(body, dv * _PROGRADE)
                epoch2 = float("nan")
                self._holdoff[k] = t + 0.5 * self.dt        # the next sample is post-burn
                total = dv
            self._valid[k] = 0
            self.mean_altitude_km[k] = np.nan
            self.burns.append(Burn(
                body=body, epoch_s=t, dv_km_s=total, mean_altitude_km=m, target_km=spec.upper_km,
                second_epoch_s=epoch2,
            ))


@dataclass(frozen=True)
class BurnSummary:
    """Per-body totals: Delta-v (km/s), number of raises, the mean interval between them (s), and the
    steady Delta-v rate (km/s per s) - see `steady_rate`."""

    body: int
    total_dv_km_s: float
    n_burns: int
    mean_interval_s: float
    steady_rate_km_s_per_s: float = field(default=float("nan"))


def steady_rate(burns: Sequence[Burn]) -> float:
    """
    One body's steady Delta-v rate (km/s per s): `sum_{k=2..K} dv_k / (t_K - t_1)` over its raises in
    epoch order - the study's definition (`tests/validation/test_stationkeeping.py`, "Measured rate").
    The first raise is excluded because its timing and size depend on where the run *started* in the
    band, not on the atmosphere; each later raise repays exactly one dead-band cycle of decay.

    **Zero raises give exactly 0.0**: the model predicts no propellant over the horizon - a drag-free
    configuration's real answer. A drag model whose first raise falls beyond the horizon reads the
    same, which is why a horizon must span at least two cycles of the slowest-decaying tier. **One
    raise gives `nan`**: there is no cycle to measure.
    """
    if len(burns) == 0:
        return 0.0
    if len(burns) == 1:
        return float("nan")
    ordered = sorted(burns, key=lambda x: x.epoch_s)
    return float(sum(x.dv_km_s for x in ordered[1:]) / (ordered[-1].epoch_s - ordered[0].epoch_s))


def summarise_burns(burns: Sequence[Burn], bodies: Sequence[int]) -> List[BurnSummary]:
    """One `BurnSummary` per slot of `bodies`, in that order. A body with fewer than two raises has a
    `nan` mean interval - there is no interval to average."""
    out: List[BurnSummary] = []
    for b in bodies:
        mine = [x for x in burns if x.body == int(b)]
        epochs = np.array([x.epoch_s for x in mine], dtype=np.float64)
        interval = float(np.mean(np.diff(epochs))) if epochs.size >= 2 else float("nan")
        out.append(BurnSummary(
            body=int(b), total_dv_km_s=float(sum(x.dv_km_s for x in mine)), n_burns=len(mine),
            mean_interval_s=interval, steady_rate_km_s_per_s=steady_rate(mine),
        ))
    return out


# ==================================================================================================
# The sweep metric: Delta-v in decision units, relative to a named baseline configuration
# ==================================================================================================

@dataclass(frozen=True)
class BodyDeltaV:
    """
    One station-kept body's budget over a sweep horizon: total Delta-v (m/s, every raise including
    the first), number of raises, and the steady rate (m/s per day, `steady_rate`'s definition).
    """

    body: str
    total_dv_m_s: float
    n_raises: int
    steady_rate_m_s_per_day: float


@dataclass(frozen=True)
class DeltaVMetrics:
    """
    A configuration's station-keeping budget and its error against a **baseline configuration** of the
    same sweep (`sweep.run_sweep(..., delta_v_baseline=)`), never against truth - `reference.py`'s
    truth has no drag, so it has no Delta-v to compare with.

    `rate_error_rel = median_rate / baseline_median_rate - 1`, and `total_error_rel` likewise on total
    Delta-v; signed, **negative = under-budgets**. The steady rate is the headline: the total also
    carries the first raise, whose size depends on where in the band the run started. The baseline's
    own errors are exactly 0.0; a drag-free configuration's are exactly -1.0, a real result (it
    predicts no raise) and not an error. Both are `nan` when the baseline's median is zero or `nan`
    (fewer than two raises in the horizon): there is then nothing to be relative to. Medians are over
    bodies, like `sweep.ErrorStats`; a `nan` body rate propagates into the median, never skipped.
    """

    baseline: str
    bodies: Tuple[BodyDeltaV, ...]
    median_total_dv_m_s: float
    median_raises: float
    median_steady_rate_m_s_per_day: float
    rate_error_rel: float
    total_error_rel: float


def _relative(value: float, base: float) -> float:
    if not math.isfinite(base) or base == 0.0:
        return float("nan")
    return value / base - 1.0


def _medians(rows: Sequence[BodyDeltaV]) -> Tuple[float, float, float]:
    return (float(np.median([r.total_dv_m_s for r in rows])),
            float(np.median([r.n_raises for r in rows])),
            float(np.median([r.steady_rate_m_s_per_day for r in rows])))


def delta_v_metrics(
    bodies: Sequence[BodyDeltaV], baseline_bodies: Sequence[BodyDeltaV], baseline: str,
) -> DeltaVMetrics:
    """Reduce per-body budgets to medians over bodies and score them against the baseline's."""
    total, raises, rate = _medians(bodies)
    base_total, _, base_rate = _medians(baseline_bodies)
    return DeltaVMetrics(
        baseline=baseline, bodies=tuple(bodies), median_total_dv_m_s=total, median_raises=raises,
        median_steady_rate_m_s_per_day=rate, rate_error_rel=_relative(rate, base_rate),
        total_error_rel=_relative(total, base_total),
    )


@dataclass(frozen=True)
class StationKeepingRun:
    """
    A station-kept propagation: sample times `(n,)`, osculating and controller-mean altitude `(n, k)`
    in km (the mean is `nan` where the controller had too little post-burn history), the burn log,
    and the per-body summary, all for `bodies` in order.
    """

    times_s: NDArray[np.float64]
    bodies: NDArray[np.int64]
    osculating_altitude_km: NDArray[np.float64]
    mean_altitude_km: NDArray[np.float64]
    burns: List[Burn]
    summary: List[BurnSummary]


def run_station_keeping(
    sim: "Simulation",
    bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
    spec: StationKeepingSpec,
    horizon_s: float,
    dt: float,
) -> StationKeepingRun:
    """
    Step `sim` from its current time for `round(horizon_s / dt)` steps of `dt` under a
    `StationKeeper`, recording every step. **It advances the simulation**, like `viz.sample_states`.
    """
    keeper = StationKeeper(sim, bodies, spec, dt)
    n_steps = max(1, int(round(horizon_s / dt)))
    k = keeper.bodies.size
    times = np.empty(n_steps + 1, dtype=np.float64)
    osc = np.empty((n_steps + 1, k), dtype=np.float64)
    mean = np.empty((n_steps + 1, k), dtype=np.float64)

    keeper.observe()
    times[0] = float(sim.t)
    osc[0] = keeper.osculating_altitude_km
    mean[0] = keeper.mean_altitude_km
    for j in range(1, n_steps + 1):
        sim.step(dt)
        keeper.observe()
        times[j] = float(sim.t)
        osc[j] = keeper.osculating_altitude_km
        mean[j] = keeper.mean_altitude_km

    return StationKeepingRun(
        times_s=times, bodies=keeper.bodies, osculating_altitude_km=osc, mean_altitude_km=mean,
        burns=list(keeper.burns), summary=summarise_burns(keeper.burns, keeper.bodies.tolist()),
    )
