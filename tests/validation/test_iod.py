"""
Initial orbit determination (`orbital_engine.iod`): Lambert, Gibbs, Herrick-Gibbs, Gauss.

Truth is built **without** `iod`'s own universal-variable Kepler solver: states from
`frames.ReferenceFrames.coe_to_rv` at true anomalies, times from `utilities.Anomalies.true_to_mean`
and the mean motion (or the reverse, `mean_to_true` at tolerance 1e-14).

Expected magnitudes, stated before measuring:

- **Published cases.** Curtis Example 5.1 (Gibbs) and Example 5.2 (Lambert) are printed to five
  significant figures: agreement to 1e-4 relative. Their values here are recalled, not re-read from the
  book (one was recalled wrong and caught; see the Gibbs test). The exact-conic tests below do not depend
  on them.
- **Exact methods on exact data.** `lambert`, `gibbs` and the improved `gauss` are exact for a two-body
  conic, so they reproduce the truth to the solvers' tolerance and round-off: 1e-9 relative (Lambert's
  bisection runs to 1e-15 in z; the truth's Kepler solve to 1e-14).
- **Herrick-Gibbs** is a Taylor fit: truncation `~(n dt)^4` relative. At n dt = 0.066 (LEO, 60 s) that is
  ~2e-5; halving dt must divide the error by ~16 (fourth order). *Measured:* exactly 16.0 per halving,
  3.9e-7 at 60 s (the coefficient is ~1/50).
- **Unimproved Gauss** truncates f and g after their second terms. Estimated `(n tau)^4 ~ 1e-4`; *measured*
  2.0e-3. The estimate was wrong: the first omitted term of f is `mu tau^3 (r.v) / (2 r^5)`, third order,
  and `(n tau)^3` = 2.3e-3 matches. Bounded below 1e-2, and above 1e-7 so the improvement demonstrably
  does something (improved: 2.5e-15).
- **NASA's Orion, two-body methods on real data.** One hour after the seed (Orion ~30,000-60,000 km out),
  the largest unmodelled accelerations are Earth's J2 (~3e-8 km/s^2 at 30,000 km) and the Moon's tide
  (~5e-9). Over a one-hour arc these displace Orion by `a dt^2 / 2` ~ 0.2 km, which shows up in a
  velocity fitted across the arc as `~ a dt` ~ 0.1 m/s. Bound 0.5 m/s, for `gibbs` and for `lambert`.
  *Measured:* Lambert 0.04 / 0.05 m/s, as estimated. Gibbs 0.36 m/s, 3-4x the estimate, because Gibbs
  uses no times: the ~0.07 km of perturbation displacement is fitted by geometry alone across an arc of
  only 11 deg as seen from Earth (Orion is climbing nearly radially), which amplifies it. Its error grows
  with the span (0.18 / 0.36 / 0.67 m/s at 0.5 / 1 / 2 h spacing), the signature of the perturbation
  rather than of small-angle conditioning. Herrick-Gibbs on the same points is 0.055 m/s at 15 min
  spacing and 4.7 m/s at 1 h: its truncation, steep this close to Earth.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from orbital_engine import artemis2 as a2
from orbital_engine import iod
from orbital_engine.frames import ReferenceFrames
from orbital_engine.utilities import Anomalies

MU = 398600.0              # Curtis's value, for the published examples
MU_E = 398600.4418


def _state(p: float, e: float, i: float, raan: float, argp: float, theta: float, mu: float) -> np.ndarray:
    r, v, ok = ReferenceFrames.coe_to_rv(np.array([[p, e, i, raan, argp, theta]]), mu)
    assert ok.all()
    return np.concatenate([r[0], v[0]])


def _time_between(e: float, a: float, th1: float, th2: float, mu: float) -> float:
    """Seconds from true anomaly th1 to th2 (th2 > th1, within one revolution)."""
    m1 = float(Anomalies.true_to_mean(th1, e))
    m2 = float(Anomalies.true_to_mean(th2, e))
    n = math.sqrt(mu / abs(a) ** 3)
    dm = m2 - m1
    if e < 1.0:
        dm %= 2.0 * math.pi
    return dm / n


ORBITS = [  # p, e, i, raan, argp  (km, -, rad)
    (7000.0, 0.01, 0.9, 0.3, 0.2),
    (12000.0, 0.6, 2.6, 4.0, 1.0),          # retrograde
    (20000.0, 1.4, 0.4, 1.0, 5.0),          # hyperbolic
]


# --- published cases ----------------------------------------------------------------------------------

def test_curtis_example_5_2_lambert() -> None:
    v1, v2 = iod.lambert(np.array([5000.0, 10000.0, 2100.0]), np.array([-14600.0, 2500.0, 7000.0]), 3600.0, MU)
    np.testing.assert_allclose(v1, [-5.9925, 1.9254, 3.2456], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(v2, [-3.3125, -4.1966, -0.38529], rtol=1e-4, atol=1e-4)


def test_curtis_example_5_1_gibbs() -> None:
    v2 = iod.gibbs(np.array([-294.32, 4265.1, 5986.7]), np.array([-1365.5, 3637.6, 6346.8]),
                   np.array([-2940.3, 2473.7, 6555.8]), MU)
    # The z component (1.5990) was first written from memory as 3.1709; x and y and the exact-conic test
    # below settled it. Recalled values, not re-read from the book: see the module docstring.
    np.testing.assert_allclose(v2, [-6.2174, -4.0122, 1.5990], rtol=1e-4, atol=1e-4)


# --- exact methods on exact conics --------------------------------------------------------------------

@pytest.mark.parametrize("orbit", ORBITS, ids=["leo", "retrograde", "hyperbolic"])
@pytest.mark.parametrize("arc", [0.4, 2.0, 4.0], ids=["short", "long", "over_pi"])
def test_lambert_reproduces_the_conic(orbit: Tuple[float, ...], arc: float) -> None:
    p, e, i, raan, argp = orbit
    a = p / (1.0 - e * e)
    th1 = -0.3
    if e > 1.0:                                   # both points between the asymptotes, |theta| < th_inf
        th_inf = math.acos(-1.0 / e)
        th1, arc = -0.85 * th_inf, min(arc, 1.7 * th_inf)
    th2 = th1 + arc
    x1, x2 = _state(p, e, i, raan, argp, th1, MU_E), _state(p, e, i, raan, argp, th2, MU_E)
    tof = _time_between(e, a, th1, th2, MU_E)
    prograde = i < math.pi / 2
    if abs(math.sin(arc)) < 0.05:
        pytest.skip("transfer angle too close to 180 deg")
    v1, v2 = iod.lambert(x1[:3], x2[:3], tof, MU_E, prograde=prograde)
    np.testing.assert_allclose(v1, x1[3:], rtol=1e-9, atol=1e-9 * np.linalg.norm(x1[3:]))
    np.testing.assert_allclose(v2, x2[3:], rtol=1e-9, atol=1e-9 * np.linalg.norm(x2[3:]))


def test_lambert_refuses_a_degenerate_plane() -> None:
    with pytest.raises(ValueError):
        iod.lambert(np.array([7000.0, 0.0, 0.0]), np.array([-8000.0, 0.0, 0.0]), 3000.0, MU_E)


@pytest.mark.parametrize("orbit", ORBITS, ids=["leo", "retrograde", "hyperbolic"])
def test_gibbs_reproduces_the_conic(orbit: Tuple[float, ...]) -> None:
    p, e, i, raan, argp = orbit
    xs = [_state(p, e, i, raan, argp, th, MU_E) for th in (-0.5, 0.1, 0.7)]
    v2 = iod.gibbs(xs[0][:3], xs[1][:3], xs[2][:3], MU_E)
    np.testing.assert_allclose(v2, xs[1][3:], rtol=1e-9, atol=1e-9 * np.linalg.norm(xs[1][3:]))


def test_gibbs_refuses_non_coplanar_positions() -> None:
    xs = [_state(7000.0, 0.01, 0.9, 0.3, 0.2, th, MU_E) for th in (-0.5, 0.1, 0.7)]
    bent = xs[2][:3] + 200.0 * np.cross(xs[0][:3], xs[1][:3]) / np.linalg.norm(np.cross(xs[0][:3], xs[1][:3]))
    with pytest.raises(ValueError):
        iod.gibbs(xs[0][:3], xs[1][:3], bent, MU_E)


# --- truncated methods: order and magnitude -----------------------------------------------------------

def _leo_track(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Three states of a 7,000 km near-circular orbit at -dt, 0, +dt (independent Kepler solve)."""
    p, e, i, raan, argp = ORBITS[0]
    a = p / (1.0 - e * e)
    n = math.sqrt(MU_E / a ** 3)
    m0 = float(Anomalies.true_to_mean(0.2, e))
    times = np.array([-dt, 0.0, dt])
    states = []
    for t in times:
        th = float(Anomalies.mean_to_true(m0 + n * t, e, tol=1e-14))
        states.append(_state(p, e, i, raan, argp, th, MU_E))
    return times, np.array(states)


def test_herrick_gibbs_is_fourth_order() -> None:
    errs = []
    for dt in (60.0, 30.0):
        t, x = _leo_track(dt)
        v2 = iod.herrick_gibbs(x[0, :3], x[1, :3], x[2, :3], t[0], t[1], t[2], MU_E)
        errs.append(float(np.linalg.norm(v2 - x[1, 3:]) / np.linalg.norm(x[1, 3:])))
    assert errs[0] < 1e-4
    assert 12.0 < errs[0] / errs[1] < 20.0


def test_herrick_gibbs_unequal_spacing() -> None:
    """Equal spacing zeroes the middle coefficient (dt32 - dt21); -40 / 0 / +60 s exercises it. Still
    fourth order (measured 1.8e-7 at this spacing, 16x per halving). A wrong middle term costs
    `(dt32 - dt21) n / 12` ~ 2e-3 relative here, which these bounds catch."""
    p, e, i, raan, argp = ORBITS[0]
    a = p / (1.0 - e * e)
    n = math.sqrt(MU_E / a ** 3)
    m0 = float(Anomalies.true_to_mean(0.2, e))
    errs = []
    for scale in (1.0, 0.5):
        t = np.array([-40.0, 0.0, 60.0]) * scale
        x = np.array([_state(p, e, i, raan, argp, float(Anomalies.mean_to_true(m0 + n * tt, e, tol=1e-14)), MU_E)
                      for tt in t])
        v2 = iod.herrick_gibbs(x[0, :3], x[1, :3], x[2, :3], t[0], t[1], t[2], MU_E)
        errs.append(float(np.linalg.norm(v2 - x[1, 3:]) / np.linalg.norm(x[1, 3:])))
    assert errs[0] < 1e-5
    assert 12.0 < errs[0] / errs[1] < 20.0


def _site_and_los(dt: float) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray, np.ndarray]:
    """A ground site on a rotating sphere and its lines of sight to the LEO truth."""
    t, x = _leo_track(dt)
    lat, lon0, we, re = math.radians(40.0), math.radians(20.0), 7.292115e-5, 6378.137
    obs = np.array([[re * math.cos(lat) * math.cos(lon0 + we * tt), re * math.cos(lat) * math.sin(lon0 + we * tt),
                     re * math.sin(lat)] for tt in t])
    los = x[:, :3] - obs
    return (float(t[0]), float(t[1]), float(t[2])), obs, los / np.linalg.norm(los, axis=1)[:, None], x


def test_gauss_improved_is_exact_and_unimproved_is_close() -> None:
    t, obs, los, x = _site_and_los(120.0)
    r2, v2 = iod.gauss(t, obs, los, MU_E)
    np.testing.assert_allclose(r2, x[1, :3], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(v2, x[1, 3:], rtol=1e-8, atol=1e-8)
    r2u, v2u = iod.gauss(t, obs, los, MU_E, improve=False)
    rel = float(np.linalg.norm(r2u - x[1, :3]) / np.linalg.norm(x[1, :3]))
    assert 1e-7 < rel < 1e-2


def test_universal_kepler_matches_the_anomaly_solve() -> None:
    t, x = _leo_track(900.0)
    f, g, _ = iod.kepler_universal_fg(x[1, :3], x[1, 3:], 900.0, MU_E)
    np.testing.assert_allclose(f * x[1, :3] + g * x[1, 3:], x[2, :3], rtol=1e-10, atol=1e-7)
    r, v = iod.kepler_universal(x[1, :3], x[1, 3:], -900.0, MU_E)          # backwards, full state
    np.testing.assert_allclose(r, x[0, :3], rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(v, x[0, 3:], rtol=1e-10, atol=1e-10)


# --- NASA's Orion: two-body methods on real data ------------------------------------------------------

def _orion(offsets_h: Tuple[float, ...]) -> np.ndarray:
    orion = a2.load("orion")
    k0 = orion.at_tdb("2026-04-03T01:00:00")
    ks = [k0 + int(round(h * 60)) for h in offsets_h]
    return np.array([orion.state[k] for k in ks])


def test_gibbs_and_lambert_on_nasas_trajectory() -> None:
    x = _orion((1.0, 2.0, 3.0))
    v2 = iod.gibbs(x[0, :3], x[1, :3], x[2, :3], a2.MU_EARTH_DE440)
    assert float(np.linalg.norm(v2 - x[1, 3:])) * 1e3 < 0.5
    v1, v3 = iod.lambert(x[0, :3], x[2, :3], 7200.0, a2.MU_EARTH_DE440)
    assert float(np.linalg.norm(v1 - x[0, 3:])) * 1e3 < 0.5
    assert float(np.linalg.norm(v3 - x[2, 3:])) * 1e3 < 0.5
