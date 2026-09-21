"""
Validation of `stationkeeping.py`: the dead-band altitude controller and the Delta-v it spends.

Scenario
--------
`scenarios.station_keeping_satellites(n_sats=4)`: four co-located Cowell satellites at 51.6 deg under
point-mass gravity + J2, `dt = 30 s`, 1.75 days. The 298.85 km osculating seed puts the one-period
mean altitude at 292.0 km (J2 holds it 6.8 km below an osculating-circular seed), and the osculating
altitude swings about 12 km peak to peak around it.

| slot | drag | controller |
|---|---|---|
| `SK-SAT-00` | layered (`atmosphere.py`'s table), B = 0.05 m^2/kg | two-impulse |
| `SK-SAT-01` | single band matched to the table at 355 km, H = 60 km (`test_atmosphere.py`'s) | two-impulse |
| `SK-SAT-02` | none - the **drag-free control** | two-impulse |
| `SK-SAT-03` | layered, as 00 | **single-impulse** |

Band `[291.0, 293.5]` km of mean altitude above `EARTH_R_EQ`; `omega = EARTH_OMEGA`.

1. The Delta-v rate, predicted independently
--------------------------------------------
The controller's bookkeeping is Hohmann transfers between mean radii. The prediction does not use it.
For a circular orbit Gauss's equation gives `da/dt = 2 f_S / n`, so undoing a loss costs
`dv = (n/2) da` to first order (the exact Hohmann sum differs by `O(da/a)`, 4e-4 at 2.5 km). With the
orbit-averaged decay `hdot(h)` of the mean altitude, one dead-band cycle - the mean from `U` down to
`L`, then one raise - costs and lasts

    dv_cycle = integral_L^U (n(h)/2) dh,        t_cycle = integral_L^U dh / |hdot(h)|

and the **predicted Delta-v rate** is `dv_cycle / t_cycle`, a property of the atmosphere alone.

`hdot` is the orbit average of `da/dt = (2 a^2/mu) v . f` over a circular orbit at inclination `i`,
with `f = -(1/2) rho B' |v_rel| v_rel` and `v_rel = v - w x r`. The atmosphere's velocity `w x r` has
an along-track component `w a cos i`, a cross-track one `w a sin i cos u` and no radial one, so

    hdot = -(a^2/mu) rho(h) B' (v^2 - v w a cos i) < sqrt((v - w a cos i)^2 + (w a sin i cos u)^2) >_u

which at `i = 0` is `drag.py`'s validated `-rho B' sqrt(mu a) (1 - w a/v)^2`. The `u` average is done
by quadrature; averaging `drag.drag_kernel` itself around the same orbit reproduces it to 2e-15. Two
terms are added, each derived:

- **Density convexity.** The satellite does not sit at its mean altitude; it oscillates about it by
  `delta` (J2's short-period terms and the eccentricity), and the density it samples averages to
  `rho(h) <exp(-delta/H)> = rho(h) (1 + <delta^2> / (2 H^2))`. `<delta^2>` is each satellite's own,
  measured from its osculating altitude about the independent mean below (not from the controller) -
  about 15.6 km^2, so **+3.7e-3** in the table's 250-300 km band (`H = 45.546 km`) and **+2.2e-3**
  for the single band (`H = 60 km`). Checked directly: `<rho(h_osc)> / rho(mean) - 1` over the
  recorded series is 3.72e-3 against the variance term's 3.76e-3. This is King-Hele's `I_0(ae/H)`
  factor to leading order.
- **RK4 truncation.** RK4 on a circular Kepler orbit loses `da/a = (n h)^6 / 36` per step. The
  constant is **1/36, not the harmonic oscillator's 1/72**: a standalone scalar RK4 on the unit circular
  orbit gives `da/a / h^6` = -0.02864, -0.02800, -0.02783, -0.02779 at h = 0.4, 0.2, 0.1, 0.05,
  converging on 1/36 = 0.02778 (Kepler's radial stiffness is not the oscillator's).
  `test_rk4_drift_is_the_derived_size` checks it on the engine: 30.2 m/day at `dt = 60 s`. At this
  test's 30 s it is 0.94 m/day, 1.6e-4 of the decay, and is added to `hdot`.

**Why `dt = 30 s` and not 60.** At 60 s the drag satellites decay faster than the prediction by
**3.2e-3 (layered) and 1.7e-3 (single)** even with both terms above, measured on the independent mean
with the controller out of the loop. At 30 s the excess falls to 4.7e-4 and 4.0e-4, a factor of about
16 once the ~3e-4 floor below is taken off, which is RK4's `h^4`. It is integrator truncation *that
depends on the drag*, so the drag-free control does not see it: the control's drift is 30 m/day at
60 s, the drag satellites' 49 and 39 m/day. It is not derived here, so it is removed by the step size
rather than budgeted.

*The transfer phase.* A two-impulse raise is not instantaneous: for half a transfer orbit
`tau = 2711 s` the satellite sits near mid-band, where the density is `exp((U - L)/(2H))` above its
value at `U`, and the second burn - sized at the trigger - leaves it `delta = |hdot(mid)| tau` short
of `U`. The cycle is therefore `tau + integral_L^{U - delta} dh/|hdot|`, shorter than the
instantaneous-raise cycle by `tau (U - L)/(2H)`: **74 s of 36 472 s (+2.0e-3 in rate) layered, 56 s
of 42 455 s (+1.3e-3) single band.** This term was *found*, not foreseen: the first version of this
test left it out, and the layered residual came out at +2.48e-3 against a controller-free decay
residual of only +4.7e-4. The +2.0e-3 gap, the matching +1.2e-3 for the single band, and the
single-impulse satellite (which has no transfer phase) showing no gap at all are what identified it. It
is now in the prediction for the two-impulse satellites.

*Measured rate.* `sum_{k=2..K} dv_k / (t_K - t_1)` over one body's raises: the first raise's timing
depends on the starting altitude, and each later raise repays one cycle of decay.

*Error budget, for `RATE_REL_TOL`.*

- *Endpoints.* The window runs from just after raise 1 to just before raise `K`, so the summed
  Delta-v differs from the decay by `U_1 - U_K`, the difference between two raises' achieved mean
  altitudes. Each raise misses its target by the controller's estimate error (~5 m) plus the response
  of the osculating orbit to an impulse sized from mean radii (`2 dv/v` relative to the `e ~ 9e-4`
  phase: 2e-3 of 2.5 km, 5 m). Over `K - 1 = 3` cycles, 7.5 km: **1.5e-3**.
- *Hohmann vs `(n/2) da`:* **4e-4**.
- *Density-velocity correlation.* Where the orbit is low it is also fast: `<rho v^3>` gains
  `3 <delta^2> / (a H)` = **1.5e-4**, the King-Hele `e I_1` term. Not included, so a positive bias.
- *Drag-dependent RK4 residual at 30 s:* 3.2e-3 / 16 = **2e-4**.
- *Trigger quantisation:* the burn fires up to `|hdot| dt = 2 m` below `L` and repays from the level
  reached, so it enters the time and the Delta-v alike and cancels to **< 1e-4**.

Sum 2.3e-3; `RATE_REL_TOL = 3e-3`. Under the table the convexity term alone exceeds that (3.7e-3, and
5.8e-3 for the single-impulse satellite), so there a prediction without it must fail, and that is
asserted - the tolerance is not wide enough to hide a derived effect. **Measured: +4.0e-4 layered,
+2.3e-4 single band, +1.04e-3 single-impulse** - each equal, to 1e-4, to the same satellite's
controller-free decay residual below (+4.7e-4, +4.0e-4, +1.03e-3), so the controller's accounting
adds nothing measurable and what is left is physics the prediction omits (the correlation term and the
RK4 residual, both positive, as observed). The endpoint term evidently came out far below its bound. The
physics half is also checked with the controller out of the loop: the independent mean's decay
between burns against `hdot` itself, `DECAY_REL_TOL = 1e-3` (correlation term 1.5e-4, RK4 residual
2e-4, and fit scatter of 7e-4 per window averaged over ~30 overlapping windows).

2. The band is held, and no burn fires inside it
-------------------------------------------------
Judged with an estimator **independent of the controller's**: a least-squares fit of
`c + s t + a1 cos(n t) + b1 sin(n t) + a2 cos(2 n t) + b2 sin(2 n t)` to 1.5 orbits of osculating
altitude, read at the window's end. It shares nothing with the controller's trapezoid-plus-lag
estimator but the samples. The fixed `n` misses the true short-period frequencies by `O(J2)`, so it
leaks `~A * 1e-3` = 6 m; with 2 m of trigger quantisation, `BAND_TOL_KM = 0.015`.

- *At every burn*, the independent mean just before it is within `BAND_TOL_KM` of `L`: no burn fires
  inside the band, and none late.
- *Throughout the run*, on impulse-free 1.5-orbit windows stepped by half an orbit, the independent
  mean stays inside `[L - tol, U + tol]`.

3. No chatter
-------------
The fastest the mean can cross the whole band is at the density of its bottom edge throughout,
`t_min = (U - L) / |hdot(L)|`. Every interval between one body's raises must exceed it (times
`1 - RATE_REL_TOL`). A controller keyed to the wrong quantity re-fires as soon as it has re-established
its estimate, about 2.5 orbits (3.8 h), a third of `t_min` here.

4. Drag-free control
--------------------
Zero burns and exactly zero Delta-v: its mean stays at 292.0 km.

Negative controls
-----------------
Each mutation was applied to the real `src/orbital_engine/stationkeeping.py`, this module run, and the
file restored with `git checkout --`; the results are in `docs/architecture.md`'s station-keeping
section.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, SCALE_HEIGHT_KM,
    layered_density,
)
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.simulator import Simulation
from orbital_engine.stationkeeping import (
    Burn, StationKeeper, StationKeepingSpec, _window_weights, hohmann_raise_dv,
    run_station_keeping, single_impulse_raise_dv, summarise_burns,
)

MU = scenarios.MU_EARTH
ALT_SEED_KM = 298.85
INC_DEG = 51.6
B = 0.05
LOWER_KM = 291.0
UPPER_KM = 293.5
DT_S = 30.0
HORIZON_S = 1.75 * 86400.0
SINGLE_MATCH_KM = 355.0
SINGLE_H_KM = 60.0

RATE_REL_TOL = 3e-3
DECAY_REL_TOL = 1e-3
BAND_TOL_KM = 0.015

ArrF = NDArray[np.float64]


# ==================================================================================================
# The run, shared by every test in the module
# ==================================================================================================

class _Run:
    def __init__(self, slots: List[int], times: ArrF, osc: ArrF, burns: List[Burn]) -> None:
        self.slots = slots
        self.times = times
        self.osc = osc
        self.burns = burns

    def burns_of(self, k: int) -> List[Burn]:
        return [b for b in self.burns if b.body == self.slots[k]]


def _fresh_session() -> Session:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def run() -> _Run:
    session = _fresh_session()
    try:
        sim = scenarios.station_keeping_satellites(
            session, n_sats=4, altitude_km=ALT_SEED_KM, inclination_deg=INC_DEG)
        sim.record_history = False
        slots = [sim.name_to_index[f"SK-SAT-{k:02d}"] for k in range(4)]
        common = dict(ballistic_coeff=B, r_ref=EARTH_R_EQ, omega=EARTH_OMEGA)
        sim.enable_force_model(DRAG_MODEL, [slots[0], slots[3]], density_model=DENSITY_MODEL_LAYERED,
                               **common)
        sim.enable_force_model(DRAG_MODEL, slots[1], density_model=DENSITY_MODEL_EXPONENTIAL,
                               rho0=_rho_match(), h0=SINGLE_MATCH_KM, scale_height=SINGLE_H_KM,
                               **common)

        # Two controllers on one arena, so the single-impulse satellite shares every step with the
        # rest; `run_station_keeping` drives one.
        two = StationKeeper(sim, slots[:3], StationKeepingSpec(LOWER_KM, UPPER_KM), DT_S)
        one = StationKeeper(sim, slots[3:], StationKeepingSpec(LOWER_KM, UPPER_KM, impulses=1), DT_S)
        n = int(round(HORIZON_S / DT_S))
        times = np.empty(n + 1, dtype=np.float64)
        osc = np.empty((n + 1, 4), dtype=np.float64)
        for j in range(n + 1):
            if j > 0:
                sim.step(DT_S)
            two.observe()
            one.observe()
            times[j] = sim.t
            osc[j] = np.concatenate([two.osculating_altitude_km, one.osculating_altitude_km])
        return _Run(slots, times, osc, two.burns + one.burns)
    finally:
        session.close()


# ==================================================================================================
# The independent predictor
# ==================================================================================================

def _trap(y: ArrF, x: ArrF) -> float:
    """Trapezoidal rule, written out rather than `np.trapezoid` (NumPy 2 only) / `np.trapz`."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _rho_match() -> float:
    return float(layered_density(np.array([SINGLE_MATCH_KM], dtype=np.float64))[0])


def _density(model: str, h: ArrF) -> ArrF:
    if model == "layered":
        return layered_density(h)
    out: ArrF = _rho_match() * np.exp(-(h - SINGLE_MATCH_KM) / SINGLE_H_KM)
    return out


def _scale_height(model: str, h: ArrF) -> ArrF:
    if model == "layered":
        band = np.searchsorted(BASE_ALTITUDE_KM, h, side="right") - 1
        out: ArrF = SCALE_HEIGHT_KM[band]
        return out
    return np.full_like(h, SINGLE_H_KM)


def _drag_hdot(model: str, h: ArrF) -> ArrF:
    """Orbit-averaged da/dt (km/s) of a circular orbit at altitude h - the module docstring's form."""
    a = EARTH_R_EQ + h
    v = np.sqrt(MU / a)
    cos_i, sin_i = math.cos(math.radians(INC_DEG)), math.sin(math.radians(INC_DEG))
    u = np.linspace(0.0, 2.0 * math.pi, 721)[:-1]
    along = (v - EARTH_OMEGA * a * cos_i)[:, None]
    cross = (EARTH_OMEGA * a * sin_i)[:, None] * np.cos(u)[None, :]
    speed = np.mean(np.sqrt(along ** 2 + cross ** 2), axis=1)
    out: ArrF = -(a ** 2 / MU) * _density(model, h) * (B * 1e3) * (v * v - v * EARTH_OMEGA * a * cos_i) * speed
    return out


def _rk4_hdot(h: ArrF, dt: float) -> ArrF:
    """RK4's truncation decay on a circular Kepler orbit, `(n h)^6 / 36` of `a` per step, km/s."""
    a = EARTH_R_EQ + h
    n = np.sqrt(MU / a ** 3)
    out: ArrF = -((n * dt) ** 6 / 36.0) * a / dt
    return out


def _hdot(model: str, h: ArrF, variance_km2: float, convexity: bool = True) -> ArrF:
    kappa = 1.0 + variance_km2 / (2.0 * _scale_height(model, h) ** 2) if convexity else 1.0
    out: ArrF = _drag_hdot(model, h) * kappa + _rk4_hdot(h, DT_S)
    return out


def _predicted_cycle(model: str, variance_km2: float, convexity: bool = True,
                     impulses: int = 2) -> Dict[str, float]:
    """
    Predicted Delta-v rate (km/s per s), cycle time (s) and cycle Delta-v (km/s).

    `impulses=2` adds the transfer phase (module docstring, "the transfer phase"): for half a transfer
    orbit `tau` the satellite decays at the mid-band rate, so the second burn leaves it `delta` below
    `U` and the circular part of the cycle is `integral_L^{U - delta}`.
    """
    h = np.linspace(LOWER_KM, UPPER_KM, 2001)
    hdot = _hdot(model, h, variance_km2, convexity)
    n = np.sqrt(MU / (EARTH_R_EQ + h) ** 3)
    dv_cycle = _trap(0.5 * n, h)
    if impulses == 2:
        a_t = EARTH_R_EQ + 0.5 * (LOWER_KM + UPPER_KM)
        tau = math.pi * math.sqrt(a_t ** 3 / MU)
        delta = abs(float(_hdot(model, np.array([a_t - EARTH_R_EQ]), variance_km2, convexity)[0])) * tau
        top = h <= UPPER_KM - delta
        t_cycle = tau + _trap(1.0 / np.abs(hdot[top]), h[top])
    else:
        t_cycle = _trap(1.0 / np.abs(hdot), h)
    return {"rate": dv_cycle / t_cycle, "t_cycle": t_cycle, "dv_cycle": dv_cycle,
            "hdot_lower": float(np.abs(hdot[0]))}


# ==================================================================================================
# The independent mean-altitude estimator
# ==================================================================================================

def _period() -> float:
    a_mid = EARTH_R_EQ + 0.5 * (LOWER_KM + UPPER_KM)
    return 2.0 * math.pi * math.sqrt(a_mid ** 3 / MU)


def _fit(times: ArrF, alt: ArrF, t_eval: float) -> tuple[float, float, float]:
    """
    Least-squares `c + s tau + harmonics(n t, 2 n t)` with `tau = t - t_eval`. Returns the secular
    value `c` at `t_eval`, the slope `s` (km/s), and the variance of the altitude about the secular
    line - `<delta^2>` for the convexity term.
    """
    n = 2.0 * math.pi / _period()
    tau = times - t_eval
    cols = [np.ones_like(tau), tau]
    for k in (1, 2):
        cols += [np.cos(k * n * times), np.sin(k * n * times)]
    design = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(design, alt, rcond=None)
    delta = alt - design[:, :2] @ coef[:2]
    return float(coef[0]), float(coef[1]), float(np.mean(delta * delta))


def _impulse_epochs(r: _Run, k: int) -> ArrF:
    ep: List[float] = []
    for b in r.burns_of(k):
        ep.append(b.epoch_s)
        if math.isfinite(b.second_epoch_s):
            ep.append(b.second_epoch_s)
    return np.array(sorted(ep), dtype=np.float64)


def _clean_windows(r: _Run, k: int, orbits: float) -> List[tuple[float, float, float]]:
    """`_fit` on every impulse-free window of `orbits` periods, ends stepped by half an orbit."""
    span = orbits * _period()
    impulses = _impulse_epochs(r, k)
    out = []
    for te in np.arange(span, r.times[-1], 0.5 * _period()):
        if np.any((impulses > te - span) & (impulses <= te)):
            continue
        sel = (r.times > te - span) & (r.times <= te)
        out.append(_fit(r.times[sel], r.osc[sel, k], float(te)))
    return out


def _variance(r: _Run, k: int) -> float:
    return float(np.mean([w[2] for w in _clean_windows(r, k, 1.5)]))


def _measured_rate(r: _Run, k: int) -> float:
    b = r.burns_of(k)
    return sum(x.dv_km_s for x in b[1:]) / (b[-1].epoch_s - b[0].epoch_s)


# ==================================================================================================
# The estimator and the burn arithmetic
# ==================================================================================================

def test_window_weights_average_a_linear_trend_and_reject_a_sinusoid() -> None:
    """
    The weights integrate the piecewise-linear interpolant over exactly `n_float` intervals, so on a
    straight line the window mean is the line's value at the window centre, to rounding. A sinusoid of
    the window's period (or half of it) is rejected up to the interpolant's error on the fractional end
    interval, bounded by `|f''| dt^2/8 * dt/T` = `(w dt)^2 / 8 * dt / T` for unit amplitude - derived,
    not fitted. A rectangle over `round(T/dt)` samples would leak `|N dt - T| / T`, 50x more here.
    """
    period = _period()
    n_float = period / DT_S
    w = _window_weights(n_float)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)

    t = -np.arange(w.size, dtype=np.float64) * DT_S          # newest first
    assert float(w @ (3.0 + 2e-4 * t)) == pytest.approx(3.0 - 2e-4 * 0.5 * period, abs=1e-12)

    for harmonic in (1, 2):
        omega = 2.0 * math.pi * harmonic / period
        bound = (omega * DT_S) ** 2 / 8.0 * DT_S / period
        worst = max(abs(float(w @ np.sin(omega * t + phi))) for phi in np.linspace(0, math.pi, 13))
        assert worst < bound, (harmonic, worst, bound)


def test_hohmann_raise_is_n_da_over_two_to_first_order() -> None:
    """`dv1 + dv2 = (n/2) da (1 + O(da/a))`, likewise one impulse; `da/a` = 3.7e-4, asserted at 1e-3.
    Vis-viva closes the single impulse exactly."""
    a1 = EARTH_R_EQ + LOWER_KM
    a2 = EARTH_R_EQ + UPPER_KM
    first_order = 0.5 * math.sqrt(MU / a1 ** 3) * (a2 - a1)
    assert sum(hohmann_raise_dv(MU, a1, a2)) == pytest.approx(first_order, rel=1e-3)
    assert single_impulse_raise_dv(MU, a1, a2) == pytest.approx(first_order, rel=1e-3)
    v = math.sqrt(MU / a1) + single_impulse_raise_dv(MU, a1, a2)
    assert 1.0 / (2.0 / a1 - v * v / MU) == pytest.approx(a2, rel=1e-13)


def test_rk4_drift_is_the_derived_size() -> None:
    """
    RK4's `(n h)^6 / 36` per step, on the engine: the drag-free satellite alone at `dt = 60 s`, where
    the drift is 30.2 m/day and the fit's scatter on the slope is under 1 %. Runs on the fused
    compiled path, so it is cheap. J2 changes the constant at `O(J2)`; asserted at 5 %.
    """
    session = _fresh_session()
    try:
        sim = scenarios.station_keeping_satellites(
            session, n_sats=1, altitude_km=ALT_SEED_KM, inclination_deg=INC_DEG)
        sim.record_history = False
        slot = sim.name_to_index["SK-SAT-00"]
        dt = 60.0
        # A band nothing reaches, so the controller only records.
        out = run_station_keeping(sim, slot, StationKeepingSpec(200.0, 400.0), 2.0 * 86400.0, dt)
    finally:
        session.close()
    assert out.burns == []
    fake = _Run([slot], out.times_s, out.osculating_altitude_km, [])
    windows = _clean_windows(fake, 0, 1.5)
    ends = np.arange(1.5 * _period(), out.times_s[-1], 0.5 * _period())
    slope = float(np.polyfit(ends, np.array([w[0] for w in windows]), 1)[0])
    mean_alt = float(np.mean([w[0] for w in windows]))
    derived = float(_rk4_hdot(np.array([mean_alt]), dt)[0])
    assert slope / derived == pytest.approx(1.0, abs=0.05), (slope * 86400e3, derived * 86400e3)


# ==================================================================================================
# The station-keeping run
# ==================================================================================================

def test_drag_free_control_never_burns(run: _Run) -> None:
    assert run.burns_of(2) == []
    summary = summarise_burns(run.burns, [run.slots[2]])[0]
    assert summary.n_burns == 0
    assert summary.total_dv_km_s == 0.0


@pytest.mark.parametrize("k, model", [(0, "layered"), (1, "single")])
def test_decay_matches_the_orbit_averaged_rate_between_burns(run: _Run, k: int, model: str) -> None:
    """The physics half of section 1, with the controller out of the loop: the independent mean's
    slope on impulse-free 3-orbit windows against `hdot` at the altitude of the window's *centre* -
    `_fit` reports the secular value at the window's end, 1.5 orbits (~0.55 km) lower, which would put
    the density ~1 % high."""
    orbits = 3.0
    windows = _clean_windows(run, k, orbits)
    assert len(windows) >= 10
    variance = _variance(run, k)
    half = 0.5 * orbits * _period()
    ratios = [s / float(_hdot(model, np.array([c - s * half]), variance)[0]) - 1.0
              for c, s, _ in windows]
    assert float(np.mean(ratios)) == pytest.approx(0.0, abs=DECAY_REL_TOL)


@pytest.mark.parametrize("k, model, impulses", [(0, "layered", 2), (1, "single", 2), (3, "layered", 1)])
def test_delta_v_rate_matches_the_orbit_averaged_decay(
    run: _Run, k: int, model: str, impulses: int,
) -> None:
    """Section 1: the headline validation, for both atmospheres and both raise styles."""
    burns = run.burns_of(k)
    assert len(burns) >= 4, "too few raises to measure a rate"
    measured = _measured_rate(run, k)
    variance = _variance(run, k)

    pred = _predicted_cycle(model, variance, impulses=impulses)
    assert measured / pred["rate"] - 1.0 == pytest.approx(0.0, abs=RATE_REL_TOL)

    # Where the derived convexity term exceeds the tolerance (the table's H = 45.5 km: 3.7e-3, and
    # 5.8e-3 for the single-impulse satellite's larger oscillation - not the single band's 2.2e-3),
    # leaving it out must fail, or the tolerance would be hiding a derived effect.
    convexity = variance / (2.0 * float(_scale_height(model, np.array([LOWER_KM]))[0]) ** 2)
    if convexity > RATE_REL_TOL:
        bare = _predicted_cycle(model, variance, convexity=False, impulses=impulses)
        assert abs(measured / bare["rate"] - 1.0) > RATE_REL_TOL

    # The same statement in time: the mean interval between raises is one predicted cycle.
    intervals = np.diff([b.epoch_s for b in burns])
    assert float(np.mean(intervals)) == pytest.approx(pred["t_cycle"], rel=RATE_REL_TOL + 0.5 * DT_S / pred["t_cycle"])


def test_single_band_under_budgets_by_its_predicted_ratio(run: _Run) -> None:
    """The headline comparison, predicted: matched at 355 km with H = 60 km, the single band is
    *thinner* than the table at 291-293.5 km (the table's H there is 45.5 km), so it must under-budget
    Delta-v, by the ratio of the two predicted rates."""
    predicted = (_predicted_cycle("single", _variance(run, 1))["rate"]
                 / _predicted_cycle("layered", _variance(run, 0))["rate"])
    assert predicted < 0.9
    assert _measured_rate(run, 1) / _measured_rate(run, 0) == pytest.approx(predicted, abs=2 * RATE_REL_TOL)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_every_burn_fires_at_the_lower_bound(run: _Run, k: int) -> None:
    """Section 2: the independent mean just before each burn is at `L`, within `BAND_TOL_KM`."""
    span = 1.5 * _period()
    burns = run.burns_of(k)
    assert burns
    for b in burns:
        sel = (run.times > b.epoch_s - span) & (run.times <= b.epoch_s)
        m = _fit(run.times[sel], run.osc[sel, k], b.epoch_s)[0]
        assert m == pytest.approx(LOWER_KM, abs=BAND_TOL_KM), (b, m)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_band_is_held_throughout(run: _Run, k: int) -> None:
    """Section 2: the independent mean stays inside the band on every impulse-free window."""
    means = [c for c, _, _ in _clean_windows(run, k, 1.5)]
    assert len(means) >= 10
    assert min(means) >= LOWER_KM - BAND_TOL_KM, min(means)
    assert max(means) <= UPPER_KM + BAND_TOL_KM, max(means)


@pytest.mark.parametrize("k, model", [(0, "layered"), (1, "single"), (3, "layered")])
def test_no_chatter(run: _Run, k: int, model: str) -> None:
    """Section 3: every interval exceeds the fastest possible crossing of the whole band. The Hohmann
    transfer phase shortens a cycle by `tau (U - L) / (2 H)` = 74 s against a 36 000 s cycle - that is
    what the `1 - RATE_REL_TOL` margin on `t_min` absorbs; a chattering controller misses by 60 %."""
    t_min = (UPPER_KM - LOWER_KM) / _predicted_cycle(model, _variance(run, k))["hdot_lower"]
    epochs = np.array([b.epoch_s for b in run.burns_of(k)], dtype=np.float64)
    assert epochs.size >= 2
    assert float(np.min(np.diff(epochs))) > t_min * (1.0 - RATE_REL_TOL), (np.diff(epochs), t_min)


# ==================================================================================================
# Configuration errors
# ==================================================================================================

def test_spec_rejects_a_bad_band() -> None:
    with pytest.raises(ValueError):
        StationKeepingSpec(300.0, 299.0)
    with pytest.raises(ValueError):
        StationKeepingSpec(290.0, 300.0, impulses=3)


def test_observe_refuses_a_skipped_step(db_session: Session) -> None:
    sim = scenarios.station_keeping_satellites(db_session, n_sats=1)
    keeper = StationKeeper(sim, [sim.name_to_index["SK-SAT-00"]], StationKeepingSpec(290.0, 295.0), 60.0)
    keeper.observe()
    sim.step(60.0)
    sim.step(60.0)
    with pytest.raises(ValueError):
        keeper.observe()


def test_keeper_refuses_a_root_body(db_session: Session) -> None:
    sim: Simulation = scenarios.station_keeping_satellites(db_session, n_sats=1)
    with pytest.raises(ValueError):
        StationKeeper(sim, [sim.name_to_index["Earth Barycenter"]], StationKeepingSpec(290.0, 295.0), 60.0)
