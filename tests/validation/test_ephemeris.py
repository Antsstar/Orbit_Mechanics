"""
Validation of `ephemeris.py`: tabulated perturbers, cubic Hermite interpolation, and the force model
`"ephemeris_third_body"` evaluated at every RK4 stage's own time.

a. Interpolation. An analytic Keplerian Moon-like orbit is tabulated and interpolated between nodes.
   The midpoint error *attains* the Hermite bound `h^4 max|x''''| / 384` (it is the leading term, not
   slack), falls 16x per halving, and on a circular orbit is the predicted vector `(n h)^4/384 r`.
   The velocity attains `sqrt(3) h^3 max|x''''| / 216` and falls 8x.
b. Field. At node times the kernel equals the direct-minus-indirect tide computed by
   `reference.nbody_acceleration` (independent code) to rounding, and equals `third_body` for an
   arena perturber at the same place. Between nodes the error is the interpolation error pushed
   through the tidal gradient, predicted as a vector.
c. Order - the headline. The Moon under a tabulated Sun converges at **fourth order**, while
   `third_body` in the identical geometry is first order (its documented 2.64 km at 3600 s). Freezing
   the stage time reproduces `third_body` to 1e-7 of its error: the freeze is the whole difference.
d. Independent truth. A test-local DOP853 integration whose Sun comes from the analytic Kepler function,
   never from the table; it also agrees with `reference.py`'s N-body truth.
e. Cislunar flyby. A spacecraft swinging past a tabulated Moon at 5812 km: closest-approach time and
   distance converge at fourth order onto the DOP853 truth, the truth sits within 3 % of the
   patched-conic estimate, and with the Moon's `mu` set to zero the run is bit-identical to two-body
   Cowell.
f. Configuration: registration, keys, refusals, the automatic NumPy fallback, sweeps.

The analytic two-body function (`_Kepler`) lives here, not in the engine, so no engine code - not the
anomaly stack, not `frames.py` - sits on both sides of a comparison.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import ephemeris, reference, registry, scenarios, sweep
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.ephemeris import (
    EPHEMERIS_MODEL, EPHEMERIS_PARAM_NAMES, MAX_PERTURBERS, EphemerisTable, ephemeris_coefficients,
    ephemeris_key, ephemeris_third_body_kernel, hermite_position_bound, hermite_velocity_bound,
    register_ephemeris,
)
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.thirdbody import THIRD_BODY_MODEL, third_body_kernel

ArrF = NDArray[np.float64]
EPS = float(np.finfo(np.float64).eps)
DAY = 86400.0
MU_E, MU_M, MU_S = scenarios.MU_EARTH, scenarios.MU_MOON, scenarios.MU_SUN
TRUTH_TOL = dict(rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL)


# ==================================================================================================
# Test-local analytic two-body motion
# ==================================================================================================

class _Kepler:
    """
    Elliptic two-body motion from `(r0, v0)` at `t = 0`, vectorised over `t`: perifocal basis from the
    eccentricity and angular-momentum vectors, Newton on Kepler's equation. `snap` is `d^4 r/dt^4`
    from the closed forms `j = -mu [v/r^3 - 3 (r.v) r/r^5]` and
    `s = -mu [a/r^3 - 6 (r.v) v/r^5 - 3 (v.v + r.a) r/r^5 + 15 (r.v)^2 r/r^7]`.
    """

    def __init__(self, r0: ArrF, v0: ArrF, mu: float) -> None:
        r0 = np.asarray(r0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        self.mu = mu
        h = np.cross(r0, v0)
        rn = float(np.linalg.norm(r0))
        ev = np.cross(v0, h) / mu - r0 / rn
        self.e = float(np.linalg.norm(ev))
        self.a = 1.0 / (2.0 / rn - float(v0 @ v0) / mu)
        self.p_hat = ev / self.e if self.e > 1e-12 else r0 / rn
        w_hat = h / np.linalg.norm(h)
        self.q_hat = np.cross(w_hat, self.p_hat)
        self.n = math.sqrt(mu / self.a ** 3)
        b = math.sqrt(1.0 - self.e ** 2)
        e0 = math.atan2(float(r0 @ self.q_hat) / (self.a * b), float(r0 @ self.p_hat) / self.a + self.e)
        self.m0 = e0 - self.e * math.sin(e0)

    def state(self, t: Any) -> Tuple[ArrF, ArrF]:
        m = self.m0 + self.n * np.atleast_1d(np.asarray(t, dtype=np.float64))
        ecc_anom = np.array(m, dtype=np.float64, copy=True)
        for _ in range(60):
            step = (ecc_anom - self.e * np.sin(ecc_anom) - m) / (1.0 - self.e * np.cos(ecc_anom))
            ecc_anom -= step
            if np.all(np.abs(step) < 1e-15):
                break
        a, e, b = self.a, self.e, math.sqrt(1.0 - self.e ** 2)
        c, s = np.cos(ecc_anom), np.sin(ecc_anom)
        r = a * (c - e)[:, None] * self.p_hat + a * b * s[:, None] * self.q_hat
        k = math.sqrt(self.mu * a) / (a * (1.0 - e * c))
        v = k[:, None] * (-s[:, None] * self.p_hat + b * c[:, None] * self.q_hat)
        return r, v

    def snap(self, t: Any) -> ArrF:
        r, v = self.state(t)
        rho = np.linalg.norm(r, axis=1)[:, None]
        acc = -self.mu * r / rho ** 3
        rv = np.einsum("ij,ij->i", r, v)[:, None]
        vv = np.einsum("ij,ij->i", v, v)[:, None]
        ra = np.einsum("ij,ij->i", r, acc)[:, None]
        out: ArrF = -self.mu * (acc / rho ** 3 - 6.0 * rv * v / rho ** 5 - 3.0 * (vv + ra) * r / rho ** 5
                                + 15.0 * rv ** 2 * r / rho ** 7)
        return out


A_MOON = 384400.0


def _moon_like(ecc: float, incl_deg: float = 5.145) -> _Kepler:
    """A Moon-like orbit about Earth (mu_E + mu_M), starting at perigee, inclined `incl_deg`."""
    mu = MU_E + MU_M
    rp = A_MOON * (1.0 - ecc)
    vp = math.sqrt(mu * (1.0 + ecc) / rp)
    inc = math.radians(incl_deg)
    return _Kepler(np.array([rp, 0.0, 0.0]), np.array([0.0, vp * math.cos(inc), vp * math.sin(inc)]), mu)


def _tabulate(orbit: _Kepler, name: str, t_end: float, step: float, centre: str | None = None) -> EphemerisTable:
    t = np.arange(0.0, t_end + step, step, dtype=np.float64)
    r, v = orbit.state(t)
    return EphemerisTable(name, t, r, v, centre=centre)


def _interval_max_snap(orbit: _Kepler, t: ArrF, samples: int = 41) -> ArrF:
    """Per interval, per component, `max |x''''|` over `samples` points including both ends."""
    h = np.diff(t)
    grid = t[:-1, None] + h[:, None] * np.linspace(0.0, 1.0, samples, dtype=np.float64)[None, :]
    snap = np.abs(orbit.snap(grid.ravel())).reshape(t.size - 1, samples, 3)
    out: ArrF = snap.max(axis=1)
    return out


def _new_session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# ==================================================================================================
# a. Interpolation
# ==================================================================================================

def _good_table_args() -> Tuple[ArrF, ArrF, ArrF]:
    t = np.array([0.0, 10.0, 20.0, 35.0], dtype=np.float64)
    return t, np.arange(12, dtype=np.float64).reshape(4, 3), np.ones((4, 3), dtype=np.float64)


@pytest.mark.parametrize("mutate, match", [
    (lambda t, p, v: (t[:1], p[:1], v[:1]), "at least 2 nodes"),
    (lambda t, p, v: (t[::-1].copy(), p, v), "strictly increasing"),
    (lambda t, p, v: (np.array([0.0, 10.0, 10.0, 35.0]), p, v), "strictly increasing"),
    (lambda t, p, v: (t, p[:, :2], v), "shape"),
    (lambda t, p, v: (t, p, v[:3]), "shape"),
    (lambda t, p, v: (t.reshape(2, 2), p, v), "1-D"),
    (lambda t, p, v: (np.array([0.0, np.nan, 20.0, 35.0]), p, v), "finite"),
    (lambda t, p, v: (t, np.where(p == 4.0, np.inf, p), v), "finite"),
])
def test_table_refuses_malformed_data(mutate: Callable[..., Tuple[ArrF, ArrF, ArrF]], match: str) -> None:
    t, p, v = mutate(*_good_table_args())
    with pytest.raises(ValueError, match=match):
        EphemerisTable("bad", t, p, v)


def test_table_is_immutable_and_owns_a_copy() -> None:
    t, p, v = _good_table_args()
    table = EphemerisTable("x", t, p, v, centre="Earth")
    p[0, 0] = 1e9
    assert table.position_km[0, 0] == 0.0, "the table must copy its input"
    for arr in (table.t_s, table.position_km, table.velocity_km_s):
        assert not arr.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.name = "y"  # type: ignore[misc]
    with pytest.raises(ValueError, match="name"):
        EphemerisTable("", t, p, v)


def test_nodes_are_reproduced_exactly_and_the_ends_are_reachable() -> None:
    table = _tabulate(_moon_like(0.0549), "moon", 2.0 * DAY, 3600.0)
    assert np.array_equal(table.position(table.t_s), table.position_km)
    assert np.array_equal(table.velocity(table.t_s), table.velocity_km_s)
    assert np.array_equal(table.position(table.t_max)[0], table.position_km[-1])


@pytest.mark.parametrize("query", [-1e-6, 2.0 * DAY + 1e-3, float("nan"), float("inf")])
def test_out_of_range_and_non_finite_queries_raise(query: float) -> None:
    table = _tabulate(_moon_like(0.0549), "moon", 2.0 * DAY, 3600.0)
    with pytest.raises(ValueError, match="out of range"):
        table.position(np.array([DAY, query]))
    with pytest.raises(ValueError, match="out of range"):
        table.velocity(query)


# (E0) is the leading term at the midpoint, so over a table the worst midpoint error must reach the bound
# (ratio -> 1 from below) and never exceed it. The per-interval max |x''''| is sampled at 41 points, so
# it can under-read the true max by ~(h/40)^2 |x''''''|/|x''''| - below 1e-6 here; allowance 1e-4.
# Rounding: positions 4e5 km, ~5 eps |r| = 4e-10 km against >= 8.5e-6 km of error at 1 h.
# Halving: the next term is relative O((n h)^2) - 1.5e-3 at 4 h - so the ratio sits in 16 (1 +- 2e-3).
# Measured (e = 0.0549): 3.355e-3, 2.097e-4, 1.311e-5 km; ratio max 1.0000; halving 15.999, 16.000.
# e = 0.3: 2.454e-2, 1.535e-3, 9.592e-5 km, halving 15.992, 15.999.
BOUND_OVERSHOOT = 1e-4
BOUND_ATTAINED = 0.99
HALVING_16 = (15.95, 16.05)
HALVING_8 = (7.95, 8.05)
TABLE_STEPS = (4.0 * 3600.0, 2.0 * 3600.0, 3600.0)


@pytest.mark.parametrize("ecc", [0.0549, 0.3])
def test_position_error_attains_the_h4_bound_and_falls_16x_per_halving(ecc: float) -> None:
    orbit = _moon_like(ecc)
    period = 2.0 * math.pi / orbit.n
    worst = []
    for h in TABLE_STEPS:
        table = _tabulate(orbit, "moon", period, h)
        mid = 0.5 * (table.t_s[:-1] + table.t_s[1:])
        err = np.abs(orbit.state(mid)[0] - table.position(mid))
        bound = hermite_position_bound(np.diff(table.t_s)[:, None], _interval_max_snap(orbit, table.t_s))
        ratio = err / bound
        assert ratio.max() < 1.0 + BOUND_OVERSHOOT, ratio.max()
        assert ratio.max() > BOUND_ATTAINED, ratio.max()
        worst.append(float(np.linalg.norm(err, axis=1).max()))
    for coarse, fine in zip(worst, worst[1:]):
        assert HALVING_16[0] < coarse / fine < HALVING_16[1], worst


# (E1): the derivative's leading error is x'''' w'/24, maximal at s = (3 -+ sqrt 3)/6, so the maximum over
# 201 points per interval attains sqrt(3) h^3 max|x''''|/216 to (1/200)^2. Third order: 8x per halving.
# Measured (e = 0.0549): 7.17e-7, 8.97e-8, 1.12e-8 km/s; ratio max 1.0000.
def test_velocity_error_attains_the_h3_bound_and_falls_8x_per_halving() -> None:
    orbit = _moon_like(0.0549)
    period = 2.0 * math.pi / orbit.n
    worst = []
    for h in TABLE_STEPS:
        table = _tabulate(orbit, "moon", period, h)
        n_int = table.t_s.size - 1
        grid = table.t_s[:-1, None] + h * np.linspace(0.0, 1.0, 201, dtype=np.float64)[None, :]
        err = np.abs(orbit.state(grid.ravel())[1] - table.velocity(grid.ravel())).reshape(n_int, 201, 3).max(axis=1)
        ratio = err / hermite_velocity_bound(h, _interval_max_snap(orbit, table.t_s))
        assert BOUND_ATTAINED < ratio.max() < 1.0 + BOUND_OVERSHOOT, ratio.max()
        worst.append(float(err.max()))
    for coarse, fine in zip(worst, worst[1:]):
        assert HALVING_8[0] < coarse / fine < HALVING_8[1], worst


# A circular orbit has x'''' = n^4 x per component, and for a sinusoid the Hermite midpoint error is
# exactly 1 - cos X - (X/2) sin X = X^4/24 - X^6/360 with X = n h/2: the vector (n h)^4/384 * r_mid,
# radially *outward* (the interpolant cuts inside the circle), less a relative X^2/15 (2.5e-5 at 4 h).
# A Hermite without its velocity terms is linear interpolation, (n h)^2/8 r - 1e5 times larger at 1 h.
# Tolerance: 2 X^2/15 relative plus 20 eps |r| of rounding (1.7e-9 km; measured mismatch at 1 h 2.8e-5
# relative = 2.4e-10 km).
def test_circular_midpoint_error_is_the_predicted_radial_vector() -> None:
    orbit = _moon_like(0.0, incl_deg=23.4)
    period = 2.0 * math.pi / orbit.n
    for h in TABLE_STEPS:
        table = _tabulate(orbit, "moon", period, h)
        mid = 0.5 * (table.t_s[:-1] + table.t_s[1:])
        r_mid = orbit.state(mid)[0]
        err = r_mid - table.position(mid)
        pred = (orbit.n * h) ** 4 / 384.0 * r_mid
        x2 = (0.5 * orbit.n * h) ** 2
        tol = 2.0 * x2 / 15.0 * np.linalg.norm(pred, axis=1) + 20.0 * EPS * A_MOON
        assert np.all(np.linalg.norm(err - pred, axis=1) < tol), (h, np.linalg.norm(err - pred, axis=1).max())


# The estimate replaces max|x''''| by 6 x the third divided difference of the tabulated velocities. That
# samples v''' at an interior point up to ~1.5 h from any point of the interval, so for a near-circular
# orbit (|x^(5)| ~ n |x^(4)|) its relative error is at most ~1.5 n h: 5.7e-2 at 4 h, 1.4e-2 at 1 h.
# Measured: 8e-3 at 4 h, 2e-3 at 1 h.
def test_error_estimate_from_the_table_alone_tracks_the_true_bound() -> None:
    orbit = _moon_like(0.0549)
    period = 2.0 * math.pi / orbit.n
    for h in TABLE_STEPS:
        table = _tabulate(orbit, "moon", period, h)
        true_bound = np.linalg.norm(hermite_position_bound(h, _interval_max_snap(orbit, table.t_s)), axis=1)
        rel = np.abs(table.error_estimate_km() / true_bound - 1.0)
        assert rel.max() < 1.5 * orbit.n * h, (h, rel.max())
    short = EphemerisTable("short", np.array([0.0, 1.0, 2.0]), np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="4 nodes"):
        short.error_estimate_km()


# ==================================================================================================
# b. Field
# ==================================================================================================

PARENT_OFFSET = np.array([1.0e5, -2.0e5, 3.0e4])
GEO = 42164.0


def _bodies_around(r_s: ArrF) -> ArrF:
    """Body positions relative to the parent: near side, far side, perpendicular, and two generic
    points, all at GEO radius - where the lunar tide is ~7e-9 km/s^2 against ~3.3e-8 per term."""
    u = r_s / np.linalg.norm(r_s)
    perp = np.cross(u, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    generic = np.array([[0.3, -0.8, 0.52], [-0.6, 0.1, -0.79]])
    generic /= np.linalg.norm(generic, axis=1)[:, None]
    dirs = np.vstack([u, -u, perp, generic])
    out: ArrF = GEO * dirs
    return out


def _kernel_arena(rel: ArrF, params_rows: ArrF) -> Tuple[ArrF, ArrF, NDArray[np.int32], NDArray[np.int64], ArrF]:
    """Slot 0 the parent at PARENT_OFFSET, slots 1.. the bodies. Returns (state, mu, parents, idx, params)."""
    n = rel.shape[0]
    state = np.zeros((n + 1, 6), dtype=np.float64)
    state[:, :3] = PARENT_OFFSET
    state[1:, :3] += rel
    mu = np.zeros(n + 1, dtype=np.float64)
    parents = np.zeros(n + 1, dtype=np.int32)
    params = np.zeros((n + 1, len(EPHEMERIS_PARAM_NAMES)), dtype=np.float64)
    params[1:] = params_rows
    return state, mu, parents, np.arange(1, n + 1, dtype=np.int64), params


def _row(**coefficients: float) -> ArrF:
    row = np.zeros(len(EPHEMERIS_PARAM_NAMES), dtype=np.float64)
    for key, value in coefficients.items():
        row[EPHEMERIS_PARAM_NAMES.index(key)] = value
    return row


# The tide from `reference.nbody_acceleration` - direct Newtonian sums, shared with no engine code - with
# a *massless* parent so that `a_body - a_parent` is exactly the perturber's direct minus indirect pull.
# Each term is ~mu/d^2 and the tide ~2 mu r/d^3, a cancellation of d/(2r) ~ 4.6 at GEO, costing well
# under one digit on ~10 rounded operations, plus the parent offset's rounding (eps 6e5 km on 4e4 km):
# ~5e-15 relative. Budget 1e-13. A dropped indirect term is ~4x wrong; a sign slip ~8x.
FIELD_NODE_REL_TOL = 1e-13


def test_field_at_nodes_matches_the_nbody_tide() -> None:
    orbit = _moon_like(0.0549)
    table = _tabulate(orbit, "moon-field", 3.0 * DAY, 3600.0)
    key = register_ephemeris(table)
    for node in (0, 17, 40):
        t = float(table.t_s[node])
        r_s = table.position_km[node]
        rel = _bodies_around(r_s)
        state, mu, parents, idx, params = _kernel_arena(rel, _row(table_1=key, mu_1=MU_M))
        out = np.zeros((idx.size + 1, 3), dtype=np.float64)
        ephemeris_third_body_kernel(idx, t, state, mu, parents, params, out)

        positions = np.vstack([PARENT_OFFSET, PARENT_OFFSET + r_s, PARENT_OFFSET + rel])
        a = reference.nbody_acceleration(positions, np.array([0.0, MU_M] + [0.0] * idx.size))
        expected = a[2:] - a[0]
        err = np.linalg.norm(out[1:] - expected, axis=1) / np.linalg.norm(expected, axis=1)
        assert np.all(err < FIELD_NODE_REL_TOL), (node, err)
        assert np.all(out[0] == 0.0), "the kernel wrote the parent's row"
        # Magnitude band, in units of mu r/d^3 with x = r/d (0.11 here): the largest tide at fixed r is the
        # near side on the axis, exactly ((1 - x)^-2 - 1)/x (2.41 - *not* the first-order 2(1 + 1.5 x) =
        # 2.35, which only holds for x << 1); the smallest is about the perpendicular, 1 - O(x^2) (0.995).
        d = float(np.linalg.norm(r_s))
        x = GEO / d
        scale = MU_M * GEO / d ** 3
        mags = np.linalg.norm(out[1:], axis=1) / scale
        near = ((1.0 - x) ** -2 - 1.0) / x
        assert abs(mags[0] / near - 1.0) < 1e-13, (mags[0], near)
        assert np.all(mags > 0.98) and np.all(mags <= near * (1.0 + 1e-13)), mags


# Same geometry through `third_body` with the perturber as an arena row at the node position, relative to
# a parent at the origin so both kernels see bit-identical `r_s`. The two evaluate the same expression
# in the same order, so they agree to a few ulp; budget 1e-15 relative.
def test_field_equals_third_body_for_an_arena_perturber_at_the_same_place() -> None:
    orbit = _moon_like(0.0549)
    table = _tabulate(orbit, "moon-field", 3.0 * DAY, 3600.0)
    key = register_ephemeris(table)
    r_s = table.position_km[9]
    rel = _bodies_around(r_s)
    n = rel.shape[0]
    state = np.zeros((n + 2, 6), dtype=np.float64)
    state[1:n + 1, :3] = rel
    state[n + 1, :3] = r_s
    mu = np.zeros(n + 2, dtype=np.float64)
    mu[n + 1] = MU_M
    parents = np.zeros(n + 2, dtype=np.int32)
    idx = np.arange(1, n + 1, dtype=np.int64)
    eph_params = np.zeros((n + 2, len(EPHEMERIS_PARAM_NAMES)), dtype=np.float64)
    eph_params[idx] = _row(table_1=key, mu_1=MU_M)
    tb_params = np.zeros((n + 2, 1), dtype=np.float64)
    tb_params[idx, 0] = float(n + 1)
    out_eph = np.zeros((n + 2, 3), dtype=np.float64)
    out_tb = np.zeros((n + 2, 3), dtype=np.float64)
    ephemeris_third_body_kernel(idx, float(table.t_s[9]), state, mu, parents, eph_params, out_eph)
    third_body_kernel(idx, 0.0, state, mu, parents, tb_params, out_tb)
    rel_err = np.linalg.norm(out_eph - out_tb, axis=1)[idx] / np.linalg.norm(out_tb[idx], axis=1)
    assert np.all(rel_err < 1e-15), rel_err


# Between nodes the kernel sees r_s + dr_s, dr_s the interpolation error, so to first order
#   da = J dr_s,   J = mu [ (I - 3 d_hat d_hat^T)/|d|^3 - (I - 3 s_hat s_hat^T)/|r_s|^3 ]
# (gradients of the direct and indirect terms with respect to r_s; written here, not in the engine).
# A 4 h table puts |dr_s| ~ 3.4e-3 km at the midpoints and |J| ~ 1e-13 /s^2, so |da| ~ 3e-16 km/s^2.
# Second-order remainder ~ |dr_s|/d ~ 1e-8 relative; rounding ~20 eps mu/d^2 ~ 1.5e-22 km/s^2 (~5e-7
# relative). Tolerance 1e-5 relative + 1e-21. And |da| <= ||J||_2 |(E0) vector|, the bound pushed
# through the gradient.
def test_field_between_nodes_is_the_interpolation_error_through_the_tidal_gradient() -> None:
    orbit = _moon_like(0.0549)
    h = 4.0 * 3600.0
    table = _tabulate(orbit, "moon-coarse", 3.0 * DAY, h)
    key = register_ephemeris(table)
    snap_max = _interval_max_snap(orbit, table.t_s)
    for k in (2, 7, 11):
        t = 0.5 * float(table.t_s[k] + table.t_s[k + 1])
        r_true = orbit.state(t)[0][0]
        dr = table.position(t)[0] - r_true
        rel = _bodies_around(r_true)
        state, mu, parents, idx, params = _kernel_arena(rel, _row(table_1=key, mu_1=MU_M))
        interp = np.zeros((idx.size + 1, 3), dtype=np.float64)
        ephemeris_third_body_kernel(idx, t, state, mu, parents, params, interp)

        positions = np.vstack([PARENT_OFFSET, PARENT_OFFSET + r_true, PARENT_OFFSET + rel])
        a = reference.nbody_acceleration(positions, np.array([0.0, MU_M] + [0.0] * idx.size))
        exact = a[2:] - a[0]
        da = interp[1:] - exact

        bound_vec = hermite_position_bound(h, snap_max[k])
        for j, r in enumerate(rel):
            d = r_true - r
            dn, sn = np.linalg.norm(d), np.linalg.norm(r_true)
            jac = MU_M * ((np.eye(3) - 3.0 * np.outer(d, d) / dn ** 2) / dn ** 3
                          - (np.eye(3) - 3.0 * np.outer(r_true, r_true) / sn ** 2) / sn ** 3)
            pred = jac @ dr
            assert np.linalg.norm(da[j] - pred) < 1e-5 * np.linalg.norm(pred) + 1e-21, (k, j, da[j], pred)
            assert np.linalg.norm(da[j]) <= np.linalg.norm(jac, 2) * np.linalg.norm(bound_vec) * (1.0 + 1e-6)


def test_zero_mu_and_empty_slots_add_exactly_nothing() -> None:
    table = _tabulate(_moon_like(0.0549), "moon-zero", 1.0 * DAY, 3600.0)
    key = register_ephemeris(table)
    rel = _bodies_around(table.position_km[3])
    # An *unregistered* key behind mu = 0 proves the slot is never looked up.
    rows = np.vstack([_row(table_1=key, mu_1=0.0),
                      _row(table_2=123456789.0, mu_2=0.0),
                      _row(),
                      _row(epoch_s=1e12, table_3=key, mu_3=0.0),
                      _row(table_1=key, mu_1=0.0, table_2=key, mu_2=0.0)])
    state, mu, parents, idx, params = _kernel_arena(rel, rows)
    out = np.full((idx.size + 1, 3), 0.1234, dtype=np.float64)
    before = out.copy()
    ephemeris_third_body_kernel(idx, 3600.0, state, mu, parents, params, out)
    assert np.array_equal(out, before)
    ephemeris_third_body_kernel(np.empty(0, dtype=np.int64), 3600.0, state, mu, parents, params, out)
    assert np.array_equal(out, before)


def test_slots_are_additive_and_rows_are_grouped_by_table_and_epoch() -> None:
    """Two perturbers in slots 1 and 3 equal the sum of one-at-a-time calls, bit for bit (0 + a1 + a2
    both ways). Bodies with different tables and epochs in the *same* slot equal their separate calls."""
    moon = _tabulate(_moon_like(0.0549), "moon-add", 2.0 * DAY, 3600.0)
    sun_orbit = _Kepler(np.array([1.496e8, 0.0, 0.0]), np.array([0.0, 29.78, 0.0]), MU_S + MU_E)
    sun = _tabulate(sun_orbit, "sun-add", 2.0 * DAY, 3600.0)
    km, ks = register_ephemeris(moon), register_ephemeris(sun)
    rel = _bodies_around(moon.position_km[5])[:3]
    t = 5000.0

    def call(rows: ArrF) -> ArrF:
        state, mu, parents, idx, params = _kernel_arena(rel, rows)
        out = np.zeros((idx.size + 1, 3), dtype=np.float64)
        ephemeris_third_body_kernel(idx, t, state, mu, parents, params, out)
        return out

    both = call(np.tile(_row(table_1=km, mu_1=MU_M, table_3=ks, mu_3=MU_S), (3, 1)))
    only_m = call(np.tile(_row(table_1=km, mu_1=MU_M), (3, 1)))
    only_s = call(np.tile(_row(table_3=ks, mu_3=MU_S), (3, 1)))
    assert np.array_equal(both, only_m + only_s)

    mixed = call(np.vstack([_row(table_2=km, mu_2=MU_M, epoch_s=100.0),
                            _row(table_2=ks, mu_2=MU_S),
                            _row(table_2=km, mu_2=MU_M, epoch_s=-3000.0)]))
    for j, row in enumerate([_row(table_2=km, mu_2=MU_M, epoch_s=100.0), _row(table_2=ks, mu_2=MU_S),
                             _row(table_2=km, mu_2=MU_M, epoch_s=-3000.0)]):
        alone = call(np.tile(row, (3, 1)))
        assert np.array_equal(mixed[j + 1], alone[j + 1]), j


def test_kernel_raises_on_an_unknown_key_and_outside_the_table() -> None:
    table = _tabulate(_moon_like(0.0549), "moon-raise", 1.0 * DAY, 3600.0)
    key = register_ephemeris(table)
    rel = _bodies_around(table.position_km[0])[:1]
    state, mu, parents, idx, params = _kernel_arena(rel, _row(table_1=4242.0, mu_1=MU_M))
    out = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(LookupError, match="no ephemeris table"):
        ephemeris_third_body_kernel(idx, 0.0, state, mu, parents, params, out)
    params[1] = _row(table_1=key, mu_1=MU_M, epoch_s=1.0 * DAY)
    with pytest.raises(ValueError, match="out of range"):
        ephemeris_third_body_kernel(idx, 1.0, state, mu, parents, params, out)


# ==================================================================================================
# c/d. The headline: Moon under a tabulated Sun, against third_body in the same geometry
# ==================================================================================================
#
# Geometry: `scenarios.sun_earth_moon(moon_mu=0.0)`, the Moon on Cowell + point_mass_gravity, 30 days -
# `test_third_body.py`'s verification case. With the Moon massless, Sun-Earth is an isolated two-body
# pair, so the Sun relative to Earth is *exactly* `_Kepler(r, v, mu_S + mu_E)` from the arena's initial
# state. The table samples that function every hour (E0: 1.1e-7 km; its own estimate 1.0e-7), and
# the truth integrates the same function directly - the table never reaches the truth.
#
# Interpolation budget. The Moon feels a Sun-position error through the tide's gradient,
# ||J|| ~ 9 mu_S r/d^4 ~ 9e-16 /s^2 (direct and indirect gradients cancel to O(r/d)), so 1.1e-7 km gives
# 1e-22 km/s^2 and at most 0.5 * 1e-22 * (30 d)^2 = 3e-10 km, coherent. Negligible.
#
# RK4 budget. `test_third_body.py` scaled the two-body Cowell scan (1.0e-2 km per orbit at 128 steps per
# orbit on a = 8000 km) to the Moon: ~1.0 km at 21600 s, i.e. E(h) ~ 1.0 km (h / 21600 s)^4, and 7.7e-4 km
# at 3600 s. Its per-stage oracle - the Sun supplied at each stage's true time by a Keplerian twin -
# measured 1.04, 5.84e-2, 3.43e-3, 2.07e-4 km. This model *is* that oracle, built into the engine,
# so it must land on the same numbers. Band (0.3, 3) x the estimate; ratios (12, 24).
#
# third_body in the same geometry: first order, err = (h/2) dr/dtau + O(h^2) with dr/dtau the truth's
# sensitivity to lagging the Sun (here by shifting the analytic Sun's clock); |dr/dtau| = 1.504e-3 km/s,
# mismatch < 0.08 at 3600 s (thirdbody.py's derivation), ratio (1.8, 2.2).
#
# Freezing the ephemeris stage time hands every stage the step's start time, which is exactly
# third_body's frozen Sun: the two runs integrate the same numbers up to the table's 1e-7 km and
# rounding in the heliocentric arena (~1e-8 km per step, a ~1e-6 km random walk over 720 steps) -
# ~4e-7 of the 2.6 km error. Tolerance 1e-5 relative.
#
# Measured: ephemeris 1.0442, 5.8375e-2, 3.4304e-3, 2.1033e-4 km (ratios 17.9, 17.0, 16.3), 6.63e-4 km at
# 3600 s against third_body's 2.6379 km (3980x); third_body mismatch 2.64e-2 / 1.33e-2; frozen vs
# third_body 3.6e-7 km; local truth vs N-body truth 8.7e-7 km.

HORIZON = 30.0 * DAY
ORDER_STEPS = (21600.0, 10800.0, 5400.0, 2700.0)
RK4_AT_21600_KM = 1.0
RK4_BAND = (0.3, 3.0)
ORDER_BAND = (12.0, 24.0)
FIRST_ORDER_BAND = (1.8, 2.2)
LAG_MISMATCH_TOL = 0.08
FROZEN_VS_THIRD_BODY_REL = 1e-5
INTERPOLATION_BUDGET_KM = 3e-10


class _FrozenTimeIntegrator:
    """Negative control: the RK4 integrator with every stage handed the step's start time."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def step(self, provider: Callable[..., ArrF], t: float, state: ArrF, dt: float,
             indices: NDArray[np.int64], primaries: NDArray[np.int32]) -> None:
        self._inner.step(lambda _t, s: provider(t, s), t, state, dt, indices, primaries)  # type: ignore[attr-defined]


def _sun_about_earth(sim: Simulation) -> _Kepler:
    g = sim.global_states
    sun, earth = sim.name_to_index["Sun"], sim.name_to_index["Earth"]
    return _Kepler(g[sun, :3] - g[earth, :3], g[sun, 3:] - g[earth, 3:], MU_S + MU_E)


def _local_truth(y0: ArrF, t_end: float, mu_central: float, perturber: Callable[[float], ArrF],
                 mu_p: float, **kwargs: Any) -> Any:
    """DOP853 on r'' = -mu r/r^3 + mu_p [(r_p - r)/|r_p - r|^3 - r_p/|r_p|^3], r_p from `perturber(t)`."""
    from scipy.integrate import solve_ivp

    def rhs(t: float, y: ArrF) -> ArrF:
        r = y[:3]
        r_p = perturber(t)
        d = r_p - r
        a = -mu_central * r / np.linalg.norm(r) ** 3
        if mu_p != 0.0:
            a = a + mu_p * (d / np.linalg.norm(d) ** 3 - r_p / np.linalg.norm(r_p) ** 3)
        return np.concatenate([y[3:], a])

    return solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", **TRUTH_TOL, **kwargs)


def _moon_rel_earth(sim: Simulation) -> ArrF:
    g = sim.global_states
    out: ArrF = g[sim.name_to_index["Moon"], :3] - g[sim.name_to_index["Earth"], :3]
    return out


def _moon_run(dt: float, model: str, sun_table: EphemerisTable, frozen: bool = False) -> ArrF:
    sim = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    sim.record_history = False
    moon = sim.name_to_index["Moon"]
    sim.set_propagator(moon, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, moon)
    if model == EPHEMERIS_MODEL:
        sim.enable_force_model(EPHEMERIS_MODEL, moon, **ephemeris_coefficients([(sun_table, MU_S)]))
    else:
        sim.enable_force_model(THIRD_BODY_MODEL, moon, perturber=float(sim.name_to_index["Sun"]))
    # `third_body` has a fused twin now (`kernels._third_body_rel`); this comparison measures the NumPy
    # `RK4Integrator` path (and `_FrozenTimeIntegrator` wraps it), so pin that path explicitly.
    sim.use_compiled_kernel = False
    assert not (sim.use_compiled_kernel and sim._cowell_fused_ok), "guard: both models must run the NumPy path"
    if frozen:
        sim._cowell_integrator = _FrozenTimeIntegrator(sim._cowell_integrator)  # type: ignore[assignment]
    for _ in range(int(round(HORIZON / dt))):
        sim.step(dt)
    return _moon_rel_earth(sim)


@pytest.fixture(scope="module")
def sun_moon() -> Iterator[Dict[str, Any]]:
    pytest.importorskip("scipy", reason="truth integration requires the [test] or [reference] extra")
    base = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    sun = _sun_about_earth(base)
    g = base.global_states
    moon, earth = base.name_to_index["Moon"], base.name_to_index["Earth"]
    y0 = np.concatenate([g[moon, :3] - g[earth, :3], g[moon, 3:] - g[earth, 3:]])

    def truth(lag: float) -> ArrF:
        sol = _local_truth(y0, HORIZON, MU_E, lambda t: sun.state(t - lag)[0][0], MU_S, t_eval=[HORIZON])
        assert sol.success
        out: ArrF = sol.y[:3, -1].copy()
        return out

    table = _tabulate(sun, "Sun (analytic, 1 h)", HORIZON + 2.0 * 3600.0, 3600.0, centre="Earth")
    runs: Dict[Tuple[str, float, bool], ArrF] = {}

    def run(dt: float, model: str = EPHEMERIS_MODEL, frozen: bool = False) -> ArrF:
        key = (model, dt, frozen)
        if key not in runs:
            runs[key] = _moon_run(dt, model, table, frozen)
        return runs[key]

    lag = 3600.0
    yield {"base": base, "table": table, "truth": truth(0.0),
           "dr_dtau": (truth(lag) - truth(-lag)) / (2.0 * lag), "run": run}


def test_table_interpolation_is_negligible_for_the_sun(sun_moon: Dict[str, Any]) -> None:
    """(E0) at 1 h for the Sun seen from Earth: (n h)^4/384 * 1.5e8 km = 1.1e-7 km; the table's own
    estimate must say so without the analytic function."""
    est = float(sun_moon["table"].error_estimate_km().max())
    assert 0.5e-7 < est < 2.0e-7, est


def test_moon_under_a_tabulated_sun_is_fourth_order_against_independent_truth(
    sun_moon: Dict[str, Any],
) -> None:
    errors = [float(np.linalg.norm(sun_moon["run"](dt) - sun_moon["truth"])) for dt in ORDER_STEPS]
    ratios = [errors[k] / errors[k + 1] for k in range(len(errors) - 1)]
    assert RK4_BAND[0] * RK4_AT_21600_KM < errors[0] < RK4_BAND[1] * RK4_AT_21600_KM, errors
    assert all(ORDER_BAND[0] < q < ORDER_BAND[1] for q in ratios), (errors, ratios)

    # (d): the combined budget at 3600 s, RK4 plus interpolation.
    budget = RK4_AT_21600_KM * (3600.0 / 21600.0) ** 4 + INTERPOLATION_BUDGET_KM
    err_3600 = float(np.linalg.norm(sun_moon["run"](3600.0) - sun_moon["truth"]))
    assert RK4_BAND[0] * budget < err_3600 < RK4_BAND[1] * budget, (err_3600, budget)


def test_independent_truth_agrees_with_the_nbody_reference(sun_moon: Dict[str, Any]) -> None:
    """The test-local truth (analytic Sun) against `reference.py` (Sun integrated by DOP853). They
    differ by the N-body truth's own Sun-Earth error: 2.3e-3 km along-track after 30 days
    (`test_third_body.py`), reaching the Moon through ||J|| ~ 7.9e-14 /s^2 - at most
    J (2.3e-3 km) T^2/6 = 2e-4 km if every bit were coherent. Measured 8.7e-7 km."""
    ref = reference.reference_for(sun_moon["base"], np.array([0.0, HORIZON]), **TRUTH_TOL)
    nbody = ref.position_of("Moon")[-1] - ref.position_of("Earth")[-1]
    assert float(np.linalg.norm(nbody - sun_moon["truth"])) < 2e-4


def test_third_body_in_the_same_geometry_is_first_order(sun_moon: Dict[str, Any]) -> None:
    errs = {dt: sun_moon["run"](dt, THIRD_BODY_MODEL) - sun_moon["truth"] for dt in (3600.0, 1800.0)}
    norms = {dt: float(np.linalg.norm(e)) for dt, e in errs.items()}
    assert FIRST_ORDER_BAND[0] < norms[3600.0] / norms[1800.0] < FIRST_ORDER_BAND[1], norms
    pred = 0.5 * 3600.0 * sun_moon["dr_dtau"]
    assert float(np.linalg.norm(errs[3600.0] - pred) / np.linalg.norm(pred)) < LAG_MISMATCH_TOL, norms
    # The headline: same step, same geometry. Derived ratio 2.71 km / 7.7e-4 km ~ 3500.
    eph = float(np.linalg.norm(sun_moon["run"](3600.0) - sun_moon["truth"]))
    assert norms[3600.0] > 1000.0 * eph, (norms[3600.0], eph)


def test_frozen_stage_time_is_first_order_and_reproduces_third_body(sun_moon: Dict[str, Any]) -> None:
    frozen = {dt: sun_moon["run"](dt, EPHEMERIS_MODEL, True) - sun_moon["truth"] for dt in (3600.0, 1800.0)}
    norms = {dt: float(np.linalg.norm(e)) for dt, e in frozen.items()}
    assert FIRST_ORDER_BAND[0] < norms[3600.0] / norms[1800.0] < FIRST_ORDER_BAND[1], norms
    for dt, e in frozen.items():
        third = sun_moon["run"](dt, THIRD_BODY_MODEL) - sun_moon["truth"]
        rel = float(np.linalg.norm(e - third) / np.linalg.norm(third))
        assert rel < FROZEN_VS_THIRD_BODY_REL, (dt, rel)


# ==================================================================================================
# e. Cislunar flyby
# ==================================================================================================
#
# A spacecraft on an Earth orbit of perigee 6700 km and apogee 450000 km, inclined 28.5 deg, starts at
# true anomaly 140 deg (51,500 km out) and crosses the Moon's circular 384400 km orbit, coplanar with it,
# 3.20 days later. The Moon is phased to trail the crossing point by 18000 km along its orbit. On the
# two-body conic the spacecraft then passes the Moon at |v_inf| = 0.970 km/s with impact parameter
# b = 9514 km, and the patched-conic periapsis is r_p = -mu/v^2 + sqrt((mu/v^2)^2 + b^2) = 5633 km. Earth's
# tide over the ~1-day encounter (2.8e-7 against the Moon's 1.2e-5 km/s^2 at 20000 km) bends the approach
# by a few percent, so the truth must sit within 10 % of that. Measured 5812.2 km (+3.2 %).
#
# The engine's closest approach is extracted by *quintic* Hermite interpolation of its own step states
# (r, v and the model acceleration at each node): its error is O(h^6), so what is measured is RK4's.
# (A cubic Hermite gives a third-order velocity, and the closest-approach time then converges at ~6
# rather than 16 - the extraction, not the engine.) The Moon table steps at 600 s, (E0) 6.6e-9 km; at
# 1 h its 8.5e-6 km moved the finest end state by ~5e-5 km. Truth: DOP853 with the analytic Moon and a
# range-rate event.
# Measured, dt = 600 / 300 / 150 s: time error 4.56e-2, 2.94e-3, 1.83e-4 s (15.5, 16.0); distance error
# -6.26e-2, -3.94e-3, -2.46e-4 km (15.9, 16.0).

FLYBY_RP, FLYBY_RA = 6700.0, 450000.0
FLYBY_E = (FLYBY_RA - FLYBY_RP) / (FLYBY_RA + FLYBY_RP)
FLYBY_P = FLYBY_RP * (1.0 + FLYBY_E)
FLYBY_A = 0.5 * (FLYBY_RP + FLYBY_RA)
FLYBY_THETA0 = math.radians(140.0)
FLYBY_TRAIL_KM = 18000.0
MOON_TABLE_STEP = 600.0
FLYBY_STEPS = (600.0, 300.0, 150.0)
PATCHED_CONIC_REL = 0.10
CONIC_RK4_BOUND_KM = 5e-3


def _flyby_sim() -> Simulation:
    return scenarios.two_body(_new_session(), mu_primary=MU_E, p=FLYBY_P, e=FLYBY_E, i=math.radians(28.5),
                              raan=math.radians(20.0), arg_pe=math.radians(35.0), theta=FLYBY_THETA0)


def _time_from_perigee(theta: float) -> float:
    ecc_anom = 2.0 * math.atan(math.sqrt((1.0 - FLYBY_E) / (1.0 + FLYBY_E)) * math.tan(0.5 * theta))
    return (ecc_anom - FLYBY_E * math.sin(ecc_anom)) / math.sqrt(MU_E / FLYBY_A ** 3)


def _quintic(ts: ArrF, st: ArrF, k: int, t: float) -> Tuple[ArrF, ArrF]:
    """Quintic Hermite through (r, v, a) at nodes k, k+1 of `st` (columns r, v, a): position, velocity."""
    h = ts[k + 1] - ts[k]
    s = (t - ts[k]) / h
    c = [st[k, :3], st[k, 3:6] * h, st[k, 6:9] * h * h, st[k + 1, :3], st[k + 1, 3:6] * h, st[k + 1, 6:9] * h * h]
    basis = [1 - 10 * s**3 + 15 * s**4 - 6 * s**5, s - 6 * s**3 + 8 * s**4 - 3 * s**5,
             0.5 * s**2 - 1.5 * s**3 + 1.5 * s**4 - 0.5 * s**5, 10 * s**3 - 15 * s**4 + 6 * s**5,
             -4 * s**3 + 7 * s**4 - 3 * s**5, 0.5 * s**3 - s**4 + 0.5 * s**5]
    deriv = [-30 * s**2 + 60 * s**3 - 30 * s**4, 1 - 18 * s**2 + 32 * s**3 - 15 * s**4,
             s - 4.5 * s**2 + 6 * s**3 - 2.5 * s**4, 30 * s**2 - 60 * s**3 + 30 * s**4,
             -12 * s**2 + 28 * s**3 - 15 * s**4, 1.5 * s**2 - 4 * s**3 + 2.5 * s**4]
    pos: ArrF = sum(b * x for b, x in zip(basis, c))  # type: ignore[assignment]
    vel: ArrF = sum(b * x for b, x in zip(deriv, c)) / h  # type: ignore[assignment]
    return pos, vel


@pytest.fixture(scope="module")
def flyby() -> Iterator[Dict[str, Any]]:
    pytest.importorskip("scipy", reason="truth integration requires the [test] or [reference] extra")
    from scipy.optimize import brentq

    base = _flyby_sim()
    sc, earth = base.name_to_index["Secondary"], base.name_to_index["Primary"]
    g = base.global_states
    y0 = np.concatenate([g[sc, :3] - g[earth, :3], g[sc, 3:] - g[earth, 3:]])
    conic = _Kepler(y0[:3], y0[3:], MU_E)

    # The Moon: circular, in the spacecraft's plane, trailing the crossing point by FLYBY_TRAIL_KM.
    theta_c = math.acos((FLYBY_P / A_MOON - 1.0) / FLYBY_E)
    t_c = _time_from_perigee(theta_c) - _time_from_perigee(FLYBY_THETA0)
    n_moon = math.sqrt((MU_E + MU_M) / A_MOON ** 3)
    phi0 = theta_c - FLYBY_TRAIL_KM / A_MOON - n_moon * t_c
    u0 = math.cos(phi0) * conic.p_hat + math.sin(phi0) * conic.q_hat
    u1 = -math.sin(phi0) * conic.p_hat + math.cos(phi0) * conic.q_hat
    moon = _Kepler(A_MOON * u0, A_MOON * n_moon * u1, MU_E + MU_M)
    horizon = t_c + 1.5 * DAY
    table = _tabulate(moon, "Moon (analytic, 600 s)", horizon + 2.0 * MOON_TABLE_STEP, MOON_TABLE_STEP,
                      centre="Primary")

    def range_rate(r: ArrF, v: ArrF, t: Any) -> ArrF:
        rm, vm = moon.state(t)
        out: ArrF = np.einsum("ij,ij->i", np.atleast_2d(r) - rm, np.atleast_2d(v) - vm)
        return out

    def event(t: float, y: ArrF) -> float:
        return float(range_rate(y[:3], y[3:], t)[0])
    event.direction = 1.0  # type: ignore[attr-defined]

    def truth(mu_m: float) -> Tuple[float, float, Any]:
        sol = _local_truth(y0, horizon, MU_E, lambda t: moon.state(t)[0][0], mu_m, events=event, dense_output=True)
        assert sol.success and sol.t_events[0].size == 1, sol.t_events
        t_ca, y_ca = float(sol.t_events[0][0]), sol.y_events[0][0]
        return t_ca, float(np.linalg.norm(y_ca[:3] - moon.state(t_ca)[0][0])), sol

    def engine(dt: float, mu_m: float = MU_M, with_model: bool = True) -> Tuple[ArrF, ArrF]:
        sim = _flyby_sim()
        sim.record_history = False
        sim.use_compiled_kernel = False
        s, e = sim.name_to_index["Secondary"], sim.name_to_index["Primary"]
        sim.set_propagator(s, PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, s)
        if with_model:
            sim.enable_force_model(EPHEMERIS_MODEL, s, **ephemeris_coefficients([(table, mu_m)]))
        n = int(round(horizon / dt))
        ts = np.arange(n + 1, dtype=np.float64) * dt
        st = np.empty((n + 1, 9), dtype=np.float64)
        for k in range(n + 1):
            if k > 0:
                sim.step(dt)
            st[k, :6] = sim.global_states[s] - sim.global_states[e]
            st[k, 6:] = sim.accelerations(sim.t)[s]
        return ts, st

    def closest(ts: ArrF, st: ArrF) -> List[Tuple[float, float]]:
        f = range_rate(st[:, :3], st[:, 3:6], ts)
        found = []
        for k in np.flatnonzero((f[:-1] < 0.0) & (f[1:] >= 0.0)):
            def rr(t: float, k: int = int(k)) -> float:
                p, v = _quintic(ts, st, k, t)
                return float(range_rate(p, v, t)[0])
            t_ca = brentq(rr, ts[k], ts[k + 1], xtol=1e-10, rtol=4 * EPS)
            found.append((float(t_ca), float(np.linalg.norm(_quintic(ts, st, int(k), t_ca)[0] - moon.state(t_ca)[0][0]))))
        return found

    rc, vc = conic.state(t_c)
    rm, vm = moon.state(t_c)
    v_inf = vc[0] - vm[0]
    b = float(np.linalg.norm(np.cross(rc[0] - rm[0], v_inf / np.linalg.norm(v_inf))))
    k2 = MU_M / float(v_inf @ v_inf)
    yield {"truth": truth, "engine": engine, "closest": closest, "conic": conic, "horizon": horizon,
           "b": b, "rp_patched": -k2 + math.sqrt(k2 ** 2 + b ** 2)}


def test_flyby_truth_is_a_close_pass_near_the_patched_conic(flyby: Dict[str, Any]) -> None:
    t_ca, d_ca, _ = flyby["truth"](MU_M)
    assert abs(d_ca / flyby["rp_patched"] - 1.0) < PATCHED_CONIC_REL, (d_ca, flyby["rp_patched"])
    assert 1737.4 < d_ca < 10000.0, d_ca
    # Massless Moon: the conic misses by the impact parameter itself, no focusing (the conic's closest
    # approach is to b to O(b / R_moon_orbit) curvature, ~2 %).
    _, d_conic, _ = flyby["truth"](0.0)
    assert abs(d_conic / flyby["b"] - 1.0) < 0.02, (d_conic, flyby["b"])


def test_flyby_closest_approach_converges_at_fourth_order(flyby: Dict[str, Any]) -> None:
    t_ca, d_ca, _ = flyby["truth"](MU_M)
    t_err, d_err = [], []
    for dt in FLYBY_STEPS:
        found = flyby["closest"](*flyby["engine"](dt))
        assert len(found) == 1, found
        t_err.append(abs(found[0][0] - t_ca))
        d_err.append(abs(found[0][1] - d_ca))
    for errs in (t_err, d_err):
        ratios = [errs[k] / errs[k + 1] for k in range(len(errs) - 1)]
        assert all(ORDER_BAND[0] < q < ORDER_BAND[1] for q in ratios), (t_err, d_err)
    assert d_err[-1] < 1e-3 and t_err[-1] < 1e-3, (t_err, d_err)


def test_massless_moon_gives_the_two_body_conic_exactly(flyby: Dict[str, Any]) -> None:
    """`mu = 0` skips the slot, so the run is bit-identical to point-mass Cowell (both on the NumPy
    path). Against the analytic conic the difference is then RK4's alone: ~N x r (n h)^5/120 at the
    start (6e-8 km per step at 51,500 km and h = 150 s, ~2700 steps) = ~2e-4 km, a few times that with
    along-track growth - bound 5e-3 km. Measured 6.2e-4 km."""
    ts, with_zero = flyby["engine"](150.0, mu_m=0.0)
    _, point_mass = flyby["engine"](150.0, with_model=False)
    assert np.array_equal(with_zero[:, :6], point_mass[:, :6])
    conic_end = flyby["conic"].state(ts[-1])[0][0]
    assert float(np.linalg.norm(with_zero[-1, :3] - conic_end)) < CONIC_RK4_BOUND_KM


# ==================================================================================================
# f. Configuration
# ==================================================================================================

def _config_sim() -> Tuple[Simulation, int]:
    sim = scenarios.two_body(_new_session(), p=13000.0, e=0.3)
    sat = sim.name_to_index["Secondary"]
    sim.set_propagator(sat, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sat)
    return sim, sat


def _config_table(centre: str | None = "Primary", name: str = "cfg-moon") -> EphemerisTable:
    return _tabulate(_moon_like(0.0549), name, 2.0 * DAY, 3600.0, centre=centre)


def test_registered_with_its_coefficients_and_citation() -> None:
    model = registry.get_force_model(EPHEMERIS_MODEL)
    assert model.param_names == ("epoch_s", "table_1", "mu_1", "table_2", "mu_2", "table_3", "mu_3")
    assert MAX_PERTURBERS == 3
    assert "Montenbruck" in model.citation and "Hermite" in model.citation
    assert model.validate_bodies is not None and model.validate_coefficients is not None


def test_keys_are_content_hashes_idempotent_and_collision_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _config_table()
    b = _config_table()                              # same content, different object
    k = register_ephemeris(a)
    assert register_ephemeris(b) == k == ephemeris_key(b)
    assert k == float(int(k)) and 1.0 <= k < 2.0 ** 52
    assert ephemeris.registered_ephemeris(k) is a
    assert ephemeris_key(_config_table(name="other")) != k
    assert ephemeris_key(_config_table(centre=None)) != k
    shifted = EphemerisTable(a.name, a.t_s, a.position_km + 1e-9, a.velocity_km_s, centre=a.centre)
    assert ephemeris_key(shifted) != k

    monkeypatch.setattr(ephemeris, "_TABLES", {})
    monkeypatch.setattr(ephemeris, "ephemeris_key", lambda table: 7.0)
    register_ephemeris(a)
    with pytest.raises(RuntimeError, match="collision"):
        register_ephemeris(shifted)
    with pytest.raises(ValueError, match="at most 3"):
        ephemeris_coefficients([(a, 1.0)] * 4)


@pytest.mark.parametrize("coefficients, match", [
    ({"mu_1": -1.0}, "mu < 0"),
    ({"mu_1": float("nan")}, "not finite"),
    ({"epoch_s": float("inf")}, "not finite"),
    ({"table_1": 1.5, "mu_1": MU_M}, "not an ephemeris key"),
    ({"table_1": -3.0, "mu_1": MU_M}, "not an ephemeris key"),
    ({"table_1": 987654321.0, "mu_1": MU_M}, "no ephemeris table"),
    ({"mu_2": MU_M}, "no table"),
    ("duplicate", "counted twice"),
    ("wrong-centre", "centred on"),
    ("uncovered", "covers"),
])
def test_refusals_leave_the_arena_untouched(coefficients: Any, match: str) -> None:
    sim, sat = _config_sim()
    table = _config_table()
    key = register_ephemeris(table)
    if coefficients == "duplicate":
        coefficients = {"table_1": key, "mu_1": MU_M, "table_3": key, "mu_3": MU_M}
    elif coefficients == "wrong-centre":
        coefficients = ephemeris_coefficients([(_config_table(centre="Earth"), MU_M)])
    elif coefficients == "uncovered":
        coefficients = ephemeris_coefficients([(table, MU_M)], epoch_s=3.0 * DAY)
    mask_before = sim.force_model_mask.copy()
    with pytest.raises(ValueError, match=match):
        sim.enable_force_model(EPHEMERIS_MODEL, sat, **coefficients)
    assert np.array_equal(sim.force_model_mask, mask_before)
    assert EPHEMERIS_MODEL not in sim.force_model_params


def test_barycentre_parent_and_stored_rows_are_checked() -> None:
    sim, sat = _config_sim()
    coefficients = ephemeris_coefficients([(_config_table(centre=None), MU_M)])
    sim.parent_indices[sat] = sim.name_to_index["TB Barycenter"]
    with pytest.raises(ValueError, match="barycentre"):
        sim.enable_force_model(EPHEMERIS_MODEL, sat, **coefficients)

    # The effective row merges passed values over stored ones: enabling slot 2 later must still see
    # slot 1's table, so re-using it is caught as a duplicate.
    sim, sat = _config_sim()
    table = _config_table()
    sim.enable_force_model(EPHEMERIS_MODEL, sat, **ephemeris_coefficients([(table, MU_M)]))
    with pytest.raises(ValueError, match="counted twice"):
        sim.enable_force_model(EPHEMERIS_MODEL, sat, table_2=ephemeris_key(table), mu_2=MU_M)
    # A time the start of the next step no longer covers is caught against the live clock.
    sim.t = 2.5 * DAY
    with pytest.raises(ValueError, match="covers"):
        sim.enable_force_model(EPHEMERIS_MODEL, sat, mu_1=MU_M)


def test_ephemeris_bodies_fall_back_to_numpy_and_the_compiled_flag_changes_nothing() -> None:
    """The bit is foreign to `_refresh_cowell_plan`, so the fused compiled kernel is refused; with it
    refused, `use_compiled_kernel` touches only the Keplerian/global kernels, which on `two_body`
    (a fixed root) are bit-identical to NumPy - so the whole run must be too."""
    table = _config_table()

    def run(compiled: bool) -> ArrF:
        sim, sat = _config_sim()
        assert sim._cowell_fused_ok, "guard: point-mass Cowell alone qualifies for the fused kernel"
        sim.enable_force_model(EPHEMERIS_MODEL, sat, **ephemeris_coefficients([(table, MU_M)]))
        assert not sim._cowell_fused_ok
        sim.use_compiled_kernel = compiled
        sim.record_history = False
        for _ in range(100):
            sim.step(120.0)
        out: ArrF = sim.global_states[sat].copy()
        return out

    assert np.array_equal(run(True), run(False))


def test_sweep_config_carries_the_keys_as_plain_coefficients() -> None:
    sim, sat = _config_sim()
    table = _config_table()
    config = sweep.ModelConfig(
        name="moon-ephemeris", propagator=PropagatorType.COWELL, dt=60.0, bodies=["Secondary"],
        force_models=(sweep.ForceModelSpec(POINT_MASS_MODEL),
                      sweep.ForceModelSpec(EPHEMERIS_MODEL, coefficients=ephemeris_coefficients([(table, MU_M)]))),
    )
    applied = sweep.apply_config(sim, config)
    assert applied.tolist() == [sat]
    row = sim.force_model_params[EPHEMERIS_MODEL][sat]
    assert row[EPHEMERIS_PARAM_NAMES.index("table_1")] == ephemeris_key(table)
    assert row[EPHEMERIS_PARAM_NAMES.index("mu_1")] == MU_M
    assert not sim._cowell_fused_ok


def test_a_row_written_directly_with_an_unknown_key_raises_at_step() -> None:
    sim, sat = _config_sim()
    sim.enable_force_model(EPHEMERIS_MODEL, sat, **ephemeris_coefficients([(_config_table(), MU_M)]))
    sim.force_model_params[EPHEMERIS_MODEL][sat, EPHEMERIS_PARAM_NAMES.index("table_1")] = 31337.0
    with pytest.raises(LookupError, match="no ephemeris table"):
        sim.step(60.0)


def test_a_nonzero_epoch_aligns_table_time_with_the_sim_clock(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `epoch_s` maps sim time to table time (`query = epoch_s + t`). Every other test in this file uses
    `epoch_s = 0`, so in review a kernel querying `t - epoch_s` passed all 49 tests - and the Artemis
    replay aligns NASA's TDB tables to the sim clock through exactly this coefficient.

    A 550 km satellite under an analytic circular Moon tabulated at 600 s, with `epoch_s` = 123456 s
    (the Moon then sits 0.33 rad further along its orbit than at table time 0), stepped 6 h at 5 s,
    against a local DOP853 truth that evaluates the analytic Moon at `epoch_s + t` directly (never the
    table). Expected agreement: RK4 truncation scaled from this repo's measured 1.88 km per 24 h at
    60 s for this orbit: at dt = 5 s that is 1.88 / 12^4 = 9.1e-5 km per day, and the along-track
    error grows ~t^2, so ~6e-6 km over 6 h; Hermite interpolation of the Moon at 600 s ~1e-11 km;
    tolerance 1e-4 km. (A first draft of this test assumed ~1e-6 km at dt = 30 s and measured
    7.8e-3 km - exactly the scaled RK4 figure, 0.117 km/day x (1/4)^2 = 7.3e-3. The step, not the
    epoch, was wrong.) The lunar tide moves this
    satellite ~0.1 km over the 6 h, and a sign error on `epoch_s` moves the Moon by 0.66 rad, so a
    wrong alignment is visible at the 1e-2 km level - two orders above the tolerance (asserted).
    """
    from scipy.integrate import solve_ivp

    mu_e, mu_m = scenarios.MU_EARTH, 4902.800066
    r_m, inc, ph0, epoch = 384400.0, math.radians(5.1), 0.7, 123456.0
    n_m = math.sqrt((mu_e + mu_m) / r_m**3)

    def moon(tt: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        th = ph0 + n_m * tt
        p = r_m * np.stack([np.cos(th), np.sin(th) * math.cos(inc), np.sin(th) * math.sin(inc)], -1)
        v = r_m * n_m * np.stack([-np.sin(th), np.cos(th) * math.cos(inc), np.cos(th) * math.sin(inc)], -1)
        return p, v

    grid = np.arange(0.0, epoch + 86400.0, 600.0, dtype=np.float64)
    pos, vel = moon(grid)
    table = EphemerisTable("analytic-moon-epoch-test", grid, pos, vel, centre="Earth")

    sim = scenarios.earth_constellation(db_session_factory(), n_sats=1, n_planes=1)
    sat = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    earth = sim.name_to_index["Earth"]
    sim.set_propagator(sat, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sat)
    x0 = (sim.global_states[sat[0]] - sim.global_states[earth]).copy()
    sim.enable_force_model(
        EPHEMERIS_MODEL, sat, **ephemeris_coefficients([(table, mu_m)], epoch_s=epoch))
    sim.record_history = False
    horizon, dt = 6 * 3600.0, 5.0
    for _ in range(int(horizon / dt)):
        sim.step(dt)
    got = sim.global_states[sat[0], :3] - sim.global_states[earth, :3]

    def rhs(t: float, y: NDArray[np.float64], sign: float) -> NDArray[np.float64]:
        r = y[:3]
        pm = moon(np.array([epoch * sign + t]))[0][0]
        d = pm - r
        a = (-mu_e * r / np.linalg.norm(r) ** 3
             + mu_m * (d / np.linalg.norm(d) ** 3 - pm / np.linalg.norm(pm) ** 3))
        out: NDArray[np.float64] = np.concatenate([y[3:], a])
        return out

    truth = solve_ivp(rhs, (0.0, horizon), x0, method="DOP853", rtol=1e-13, atol=1e-10,
                      args=(1.0,)).y[:3, -1]
    wrong = solve_ivp(rhs, (0.0, horizon), x0, method="DOP853", rtol=1e-13, atol=1e-10,
                      args=(-1.0,)).y[:3, -1]
    assert np.linalg.norm(got - truth) < 1e-4, np.linalg.norm(got - truth)
    assert np.linalg.norm(wrong - truth) > 1e-2, np.linalg.norm(wrong - truth)
