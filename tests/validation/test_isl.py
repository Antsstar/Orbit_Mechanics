"""
Validation of `isl.py`: inter-satellite link visibility, its contact dataset, and the sweep metric.

Every expectation below is **derived before it is measured** (`CLAUDE.md`'s item 5), from plane
geometry that shares no code with `geometry.segment_clearance`:

a. **Same orbit, fixed phase.** Two satellites on one circular orbit of radius `a`, separated by
   central angle `dphi`, have a chord whose closest approach to the centre is its midpoint, at
   `a |cos(dphi / 2)|` - for all time. So the pair links **for the whole horizon** iff
   `a cos(dphi/2) > R + h_graze` and never otherwise; at 550 km with `h_graze = 100 km` the
   threshold is `dphi* = 2 acos(6471 / 6921) = 41.550 deg`. Asserted on both sides of it (0.01 deg
   away: +/-0.21 km of margin) and as the clearance *value*, to 1e-9 km.
b. **Coplanar circular orbits at two radii.** Relative phase `psi(t) = psi0 + (n1 - n2) t`. The
   segment's distance from the centre is `r1 r2 |sin psi| / |P1 - P2|` while the foot of the
   perpendicular is inside the segment (`cos psi <= r1 / r2`), else `r1`. It equals `rho = R +
   h_graze` exactly at the tangent configuration, `|psi*| = acos(rho / r1) + acos(rho / r2)`
   (52.047 deg at 550 / 1200 km), so every rise and set over any horizon is closed-form:
   `t = (2 pi k -/+ psi* - psi0) / (n1 - n2)`, a 13 152.4 s window every 45 485.9 s synodic period.
   **Edge error, derived:** linear inverse interpolation has the bias
   `t_hat - t* = -(f'' / (2 f')) a b` (`a`, `b` the crossing's offsets into its bracket; see
   `geometry.access_windows`, whose sign this module's docstring corrects). Here the clearance is
   **concave** at the crossing - `f'' = -3.58e-5 km/s^2` against `|f'| = 0.2087 km/s` - the opposite
   of elevation's convex horizon, so ISL rises read **late**, sets **early**, windows **short**, by
   up to `|f''| h^2 / (8 |f'|)` = 0.309 / 0.077 / 0.019 s at `h` = 120 / 60 / 30 s. Each measured
   edge is compared with `-(f'' / (2 f')) a b` for its own `a b` (tolerance: the `O(h^3)` remainder,
   5 %), and the halving ratio is measured on an edge phase-locked to 1/3 of its bracket, where
   `a b` scales by exactly 4.
c. **Range and range rate.** `|P1 - P2| = D(psi)`, `d D/dt = r1 r2 sin(psi) (n1 - n2) / D`: the
   engine's rate against that analytic derivative, against a central difference of its own range
   series (error `D''' h^2 / 6`, bound computed from the closed form), and its sign on a pair known
   to be closing (`psi < 0`, before conjunction) - **negative = closing, positive = opening**.
d. **Structure.** `N (N-1) / 2` pairs in lexicographic `a < b` order; symmetry under reversing the
   body order; invariance under a rigid rotation (the property the headline's Kepler result rests
   on); `max_range_km` monotonic and `inf` identical to none; a segment through the body is never
   visible; a radially stacked pair *is* (the clamp - the infinite line through them passes through
   the centre).
e. **Sweep.** `ErrorStats` and `access` bit-identical with `isl=` on; one dense truth per sweep,
   shared with `access` on the same grid; every truth call gets the same model keywords; the
   divisibility guard; an `ExternalTier` gets ISL metrics too; and a fast version of the headline
   (`benchmarks/isl_sweep.py`), with its expectation derived from the tier's along-track error.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import access, isl, scenarios, sweep
from orbital_engine.custom_types import PropagatorType
from orbital_engine.drag import EARTH_OMEGA
from orbital_engine.geometry import line_of_sight, segment_clearance
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.utilities import Transformations
from orbital_engine.viz import sample_states

R = scenarios.EARTH_RADIUS
MU = scenarios.MU_EARTH
H_GRAZE = 100.0
RHO = R + H_GRAZE                                     # the grazing sphere, 6471 km

INNER_ALT, OUTER_ALT = 550.0, 1200.0
R1, R2 = R + INNER_ALT, R + OUTER_ALT
N1, N2 = math.sqrt(MU / R1 ** 3), math.sqrt(MU / R2 ** 3)
DN = N1 - N2                                          # 1.38135e-4 rad/s
PSI_STAR = math.acos(RHO / R1) + math.acos(RHO / R2)  # 52.047 deg, the tangent configuration
SYNODIC_S = 2.0 * math.pi / DN                        # 45 485.9 s

SPEC = isl.IslSpec(central_body="Earth", body_radius_km=R, h_graze_km=H_GRAZE)


# --------------------------------------------------------------------------------------------------
# Closed forms for case (b), written from plane geometry - no engine code
# --------------------------------------------------------------------------------------------------

def _chord(psi: float) -> float:
    return math.sqrt(R1 ** 2 + R2 ** 2 - 2.0 * R1 * R2 * math.cos(psi))


def _segment_distance(psi: float) -> float:
    """Distance of the segment from the centre: the perpendicular foot while it is inside the
    segment (`cos psi <= r1 / r2`), else the inner endpoint's radius."""
    if math.cos(psi) > R1 / R2:
        return R1
    return R1 * R2 * abs(math.sin(psi)) / _chord(psi)


def _range_rate(psi: float) -> float:
    return R1 * R2 * math.sin(psi) * DN / _chord(psi)


def _clearance_derivatives() -> Tuple[float, float]:
    """`f'` and `f''` of the clearance in time at the **set** crossing (`psi = +psi*`), from central
    differences of the closed form in `psi` (step 1e-4 rad: ~1e-8 relative, far below any
    tolerance). At the rise `f'` flips sign and `f''` does not - the distance is even in `psi`."""
    e = 1e-4
    d0 = _segment_distance(PSI_STAR)
    dp = _segment_distance(PSI_STAR + e)
    dm = _segment_distance(PSI_STAR - e)
    return (dp - dm) / (2.0 * e) * DN, (dp - 2.0 * d0 + dm) / e ** 2 * DN ** 2


def _predicted_edges(psi0: float, horizon_s: float) -> List[Tuple[float, float]]:
    """Every (rise, set) with the rise inside `[0, horizon]`: `psi(t) = psi0 + DN t` crosses
    `-psi*` upward at a rise and `+psi*` at a set (mod 2 pi)."""
    edges = []
    k = math.ceil((psi0 + PSI_STAR) / (2.0 * math.pi))
    while True:
        rise = (2.0 * math.pi * k - PSI_STAR - psi0) / DN
        if rise > horizon_s:
            return edges
        edges.append((rise, (2.0 * math.pi * k + PSI_STAR - psi0) / DN))
        k += 1


def _coplanar(
    session_factory: Callable[[], Session],
    psi0: float,
    *,
    inclination_deg: float = 0.0,
    raan_deg: float = 0.0,
) -> Tuple[Simulation, List[int]]:
    sim = scenarios.coplanar_satellites(
        session_factory(), altitudes_km=[INNER_ALT, OUTER_ALT],
        phases_deg=[math.degrees(psi0), 0.0], inclination_deg=inclination_deg, raan_deg=raan_deg,
    )
    sim.record_history = False
    return sim, [sim.name_to_index[scenarios.coplanar_satellite_name(k)] for k in range(2)]


def _coplanar_windows(
    session_factory: Callable[[], Session], psi0: float, h: float, horizon_s: float, **kw: float,
) -> List[isl.IslWindow]:
    sim, bodies = _coplanar(session_factory, psi0, **kw)
    return isl.isl_windows_from_simulation(
        sim, bodies, access.access_grid(horizon_s, h), SPEC, max_dt=h,
    )


# The first rise phase-locked to 1/3 of a 120 s bracket at t = 6040 s: then at h = 120, 60, 30 s the
# crossing sits at offsets (40, 80), (40, 20), (10, 20) - `a b` = 3200, 800, 200, exactly 4x apart.
LOCKED_RISE_S = 6040.0
PSI0_LOCKED = -PSI_STAR - DN * LOCKED_RISE_S


# ==================================================================================================
# a. Same orbit, fixed phase: always or never
# ==================================================================================================

def test_same_orbit_pairs_link_for_all_time_iff_the_chord_clears_the_grazing_sphere(
    db_session_factory: Callable[[], Session],
) -> None:
    a = R + 550.0
    threshold_deg = math.degrees(2.0 * math.acos(RHO / a))
    assert threshold_deg == pytest.approx(41.550012, abs=1e-6)

    phases = [0.0, 30.0, threshold_deg - 0.01, threshold_deg + 0.01, 180.0, 300.0]
    sim = scenarios.coplanar_satellites(
        db_session_factory(), altitudes_km=[550.0] * len(phases), phases_deg=phases,
        inclination_deg=53.0, raan_deg=25.0,
    )
    sim.record_history = False
    bodies = [sim.name_to_index[scenarios.coplanar_satellite_name(k)] for k in range(len(phases))]
    horizon = 12_000.0
    times = access.access_grid(horizon, 60.0)
    states = sample_states(sim, bodies, times, relative_to=sim.name_to_index["Earth"], max_dt=60.0)
    positions = np.ascontiguousarray(states[..., :3])

    series = isl.link_series(positions, times, SPEC)
    windows = isl.isl_windows(positions, times, SPEC)
    by_pair = {(w.body_a, w.body_b): w for w in windows}
    assert len(by_pair) == len(windows), "a fixed-phase pair can have at most one window"

    for col, (ia, ib) in enumerate(zip(series.body_a.tolist(), series.body_b.tolist())):
        dphi = math.radians(phases[ib] - phases[ia])
        expected = a * abs(math.cos(0.5 * dphi)) - RHO
        # The value itself, at every sample: Keplerian circular orbits keep the radius to ~1e-12.
        np.testing.assert_allclose(series.clearance_km[:, col], expected, rtol=0.0, atol=1e-8)
        if expected > 0.0:
            w = by_pair[(ia, ib)]
            assert w.rise_clipped and w.set_clipped
            assert (w.rise_s, w.set_s) == (0.0, horizon)
        else:
            assert (ia, ib) not in by_pair

    # Both sides of the threshold, 0.01 deg apart: +/- a sin(dphi*/2) (0.005 deg) = +/-0.21 km.
    margin = a * math.sin(math.radians(threshold_deg) / 2.0) * math.radians(0.005)
    assert margin == pytest.approx(0.2141, abs=1e-3)
    assert (0, 2) in by_pair and (0, 3) not in by_pair


# ==================================================================================================
# b. Coplanar orbits at two radii: every edge in closed form
# ==================================================================================================

def test_segment_distance_closed_form_is_tangent_at_the_critical_phase() -> None:
    """The two closed forms agree: the perpendicular distance equals the grazing radius at
    `psi* = acos(rho/r1) + acos(rho/r2)`, the foot is inside the segment there, and the engine's
    clearance of the same two points is zero."""
    assert math.degrees(PSI_STAR) == pytest.approx(52.0474, abs=1e-4)
    assert math.cos(PSI_STAR) < R1 / R2
    assert _segment_distance(PSI_STAR) == pytest.approx(RHO, rel=1e-13)
    p1 = np.array([R1 * math.cos(PSI_STAR), R1 * math.sin(PSI_STAR), 0.0])
    p2 = np.array([R2, 0.0, 0.0])
    assert float(segment_clearance(p1, p2, body_radius_km=R, h_graze_km=H_GRAZE)) == pytest.approx(
        0.0, abs=1e-9)
    assert SYNODIC_S == pytest.approx(45_485.9, abs=0.1)
    assert 2.0 * PSI_STAR / DN == pytest.approx(13_152.36, abs=0.01)


def test_coplanar_windows_match_the_closed_form_edge_by_edge(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Every edge over two days against `t*` plus its derived interpolation bias.

    `f' = +/-0.2087 km/s`, `f'' = -3.58e-5 km/s^2` (concave), so the bias `-(f''/(2f')) a b` is
    **positive at a rise and negative at a set**: ISL windows read short. Tolerance: 5 % of the
    predicted bias (the `O(h^3)` remainder, ~`h |f'''/f''|` ~ 1-2 %) plus 1e-6 s for the propagator.
    """
    f1_set, f2 = _clearance_derivatives()
    assert f1_set == pytest.approx(-0.20873, abs=1e-4)
    assert f2 == pytest.approx(-3.581e-5, rel=1e-3)
    k_rise = -f2 / (2.0 * -f1_set)            # bias per unit (a b) at a rise: +8.58e-5 /s
    k_set = -f2 / (2.0 * f1_set)              # at a set: -8.58e-5 /s

    horizon = 2.0 * 86400.0
    psi0 = math.radians(-100.0)
    predicted = _predicted_edges(psi0, horizon)
    assert len(predicted) == 4

    for h in (120.0, 60.0):
        windows = _coplanar_windows(db_session_factory, psi0, h, horizon)
        assert [(w.body_a, w.body_b) for w in windows] == [(0, 1)] * 4
        for w, (t_rise, t_set) in zip(windows, predicted):
            assert not (w.rise_clipped or w.set_clipped)
            for measured, exact, k in ((w.rise_s, t_rise, k_rise), (w.set_s, t_set, k_set)):
                a = exact - h * math.floor(exact / h)
                bias = k * a * (h - a)
                assert measured - exact == pytest.approx(bias, rel=0.05, abs=1e-6)
                assert abs(measured - exact) <= abs(f2) * h ** 2 / (8.0 * abs(f1_set)) * 1.05
            assert w.rise_s >= t_rise and w.set_s <= t_set, "concave: rise late, set early"


def test_coplanar_edge_error_is_second_order(db_session_factory: Callable[[], Session]) -> None:
    """The phase-locked rise at 6040 s: `a b` = 3200, 800, 200 s^2 at h = 120, 60, 30 s, so the
    error must fall by 4.00 per halving and equal `8.58e-5 x a b` = 0.2745, 0.0686, 0.0172 s."""
    k_rise = -_clearance_derivatives()[1] / (2.0 * -_clearance_derivatives()[0])
    errors = []
    for h, ab in ((120.0, 3200.0), (60.0, 800.0), (30.0, 200.0)):
        [w] = _coplanar_windows(db_session_factory, PSI0_LOCKED, h, 20_160.0)
        errors.append(w.rise_s - LOCKED_RISE_S)
        assert errors[-1] == pytest.approx(k_rise * ab, rel=0.03)
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.03)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.03)


def test_a_tilted_plane_reproduces_the_equatorial_windows(
    db_session_factory: Callable[[], Session],
) -> None:
    """Clearance depends only on the two positions relative to a sphere, so tilting the common
    plane cannot move an edge; a frame or axis mistake would."""
    flat = _coplanar_windows(db_session_factory, PSI0_LOCKED, 60.0, 100_080.0)
    tilted = _coplanar_windows(
        db_session_factory, PSI0_LOCKED, 60.0, 100_080.0, inclination_deg=63.0, raan_deg=-140.0,
    )
    assert len(flat) == len(tilted) == 3
    for u, v in zip(flat, tilted):
        assert u.rise_s == pytest.approx(v.rise_s, abs=1e-6)
        assert u.set_s == pytest.approx(v.set_s, abs=1e-6)
        assert u.min_range_km == pytest.approx(v.min_range_km, abs=1e-6)


# ==================================================================================================
# c. Range and range rate
# ==================================================================================================

def _coplanar_states(
    session_factory: Callable[[], Session], psi0: float, times: NDArray[np.float64],
) -> NDArray[np.float64]:
    sim, bodies = _coplanar(session_factory, psi0, inclination_deg=30.0, raan_deg=40.0)
    return sample_states(sim, bodies, times, relative_to=sim.name_to_index["Earth"],
                         max_dt=float(times[1] - times[0]))


def test_range_and_range_rate_match_the_closed_form_and_a_central_difference(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Over one window at `h = 10 s`: range against `D(psi)`, rate against `r1 r2 sin(psi) dn / D`
    (both to floating-point precision - Keplerian), and rate against the central difference of the
    engine's *own* range, which shares no algebra with the rate: error `D''' h^2 / 6`, with `D'''`
    bounded from the closed form over the same span.
    """
    h = 10.0
    times = access.access_grid(20_160.0, h)
    states = _coplanar_states(db_session_factory, PSI0_LOCKED, times)
    series = isl.link_series(
        np.ascontiguousarray(states[..., :3]), times, SPEC,
        velocities_km_s=np.ascontiguousarray(states[..., 3:]),
    )
    assert series.range_rate_km_s is not None
    psi = PSI0_LOCKED + DN * times
    exact_range = np.array([_chord(p) for p in psi])
    exact_rate = np.array([_range_rate(p) for p in psi])
    np.testing.assert_allclose(series.range_km[:, 0], exact_range, rtol=1e-10)
    np.testing.assert_allclose(series.range_rate_km_s[:, 0], exact_rate, rtol=0.0, atol=1e-9)

    central = (series.range_km[2:, 0] - series.range_km[:-2, 0]) / (2.0 * h)
    rate_dd = np.abs(np.diff(exact_rate, 2)) / h ** 2           # D''' at the sample spacing
    bound = 1.5 * float(rate_dd.max()) * h ** 2 / 6.0
    worst = float(np.max(np.abs(central - series.range_rate_km_s[1:-1, 0])))
    assert worst <= bound, (worst, bound)
    assert bound < 1e-4, "the bound itself is small: the check is not vacuous"


def test_range_rate_is_negative_while_closing_and_positive_while_opening(
    db_session_factory: Callable[[], Session],
) -> None:
    """The contact dataset's edges: at the rise the pair is approaching conjunction (`psi < 0`) and
    the range shrinks - **negative**; at the set it grows - **positive**. The magnitudes are the
    closed form's `0.89387 km/s` at `|psi| = psi*`. The peak is the sampled closest approach: its
    range is within `D'' (h/2)^2 / 2` above `r2 - r1` and its rate within `D'' h / 2` of zero."""
    h = 60.0
    times = access.access_grid(20_160.0, h)
    states = _coplanar_states(db_session_factory, PSI0_LOCKED, times)
    positions = np.ascontiguousarray(states[..., :3])
    contacts = isl.isl_contacts(positions, np.ascontiguousarray(states[..., 3:]), times, SPEC)
    assert [c.window for c in contacts] == isl.isl_windows(positions, times, SPEC)
    [c] = contacts

    edge_rate = _range_rate(PSI_STAR)
    assert edge_rate == pytest.approx(0.893870, abs=1e-6)
    assert c.rise.range_rate_km_s < 0.0 < c.set.range_rate_km_s
    assert c.rise.range_rate_km_s == pytest.approx(-edge_rate, abs=1e-6)
    assert c.set.range_rate_km_s == pytest.approx(edge_rate, abs=1e-6)
    # The range at the reported (biased) instants is the true range there, to the linear
    # interpolation error `|D''| h^2 / 8`: `D'' = dn^2 d2D/dpsi2 = -2.89e-5 km/s^2` at the edge, so
    # 0.013 km at h = 60 s (the range is nearly linear in time there, not exactly).
    e = 1e-4
    d_dd_edge = (_chord(PSI_STAR + e) - 2.0 * _chord(PSI_STAR) + _chord(PSI_STAR - e)) / e ** 2 * DN ** 2
    assert d_dd_edge == pytest.approx(-2.89e-5, rel=1e-2)
    for sample in (c.rise, c.set):
        psi = PSI0_LOCKED + DN * sample.time_s
        assert sample.range_km == pytest.approx(_chord(psi), abs=1.05 * abs(d_dd_edge) * h ** 2 / 8.0)

    d_dd = R1 * R2 * DN ** 2 / (R2 - R1)                     # D'' at conjunction, 1.54e-3 km/s^2
    assert R2 - R1 <= c.peak.range_km <= (R2 - R1) + d_dd * (h / 2.0) ** 2 / 2.0
    assert abs(c.peak.range_rate_km_s) <= d_dd * h / 2.0
    assert c.rise_s < c.peak.time_s < c.set_s


# ==================================================================================================
# d. Structure
# ==================================================================================================

def test_pair_count_and_order() -> None:
    a, b = isl.pair_indices(5)
    assert a.size == b.size == 10 == 5 * 4 // 2
    assert list(zip(a.tolist(), b.tolist()))[:5] == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
    assert bool(np.all(a < b))
    assert isl.pair_indices(1)[0].size == 0 and isl.pair_indices(0)[0].size == 0


def _constellation_positions(
    session_factory: Callable[[], Session], horizon_s: float = 12_000.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sim = scenarios.earth_constellation(
        session_factory(), n_sats=12, n_planes=3, altitude_km=550.0, inclination_deg=53.0,
    )
    sim.record_history = False
    idx = sweep.eligible_bodies(sim)
    times = access.access_grid(horizon_s, 60.0)
    states = sample_states(sim, idx, times, relative_to=sim.name_to_index["Earth"], max_dt=60.0)
    return (np.ascontiguousarray(states[..., :3]), np.ascontiguousarray(states[..., 3:]), times)


def test_reversing_the_body_order_relabels_but_does_not_move_windows(
    db_session_factory: Callable[[], Session],
) -> None:
    positions, _, times = _constellation_positions(db_session_factory)
    n = positions.shape[1]
    forward = isl.isl_windows(positions, times, SPEC)
    reverse = isl.isl_windows(positions[:, ::-1, :], times, SPEC)
    assert len(forward) == len(reverse) > 0

    def key(a: int, b: int) -> Tuple[int, int]:
        return (min(a, b), max(a, b))

    relabelled = sorted(
        (key(n - 1 - w.body_a, n - 1 - w.body_b), w.rise_s, w.set_s) for w in reverse
    )
    original = sorted((key(w.body_a, w.body_b), w.rise_s, w.set_s) for w in forward)
    for (p, r0, s0), (q, r1, s1) in zip(original, relabelled):
        assert p == q
        assert r0 == pytest.approx(r1, abs=1e-6) and s0 == pytest.approx(s1, abs=1e-6)


def test_a_rigid_rotation_of_the_constellation_moves_no_window(
    db_session_factory: Callable[[], Session],
) -> None:
    """The clearance depends only on the two positions relative to a sphere, so rotating every
    satellite by the same (time-varying) rotation changes nothing. This is why a model that misses
    the *common* nodal regression (Kepler) is scored on its other errors only by the ISL metric."""
    positions, _, times = _constellation_positions(db_session_factory)
    angle = 1e-5 * times + 0.3                                   # a secular drift, like a node's
    rz = np.asarray(Transformations.Rz(angle), dtype=np.float64)  # (T, 3, 3)
    rotated = np.einsum("tij,tbj->tbi", rz, positions)
    base = isl.isl_windows(positions, times, SPEC)
    moved = isl.isl_windows(rotated, times, SPEC)
    assert len(base) == len(moved) > 0
    for u, v in zip(base, moved):
        assert (u.body_a, u.body_b) == (v.body_a, v.body_b)
        assert u.rise_s == pytest.approx(v.rise_s, abs=1e-6)
        assert u.set_s == pytest.approx(v.set_s, abs=1e-6)


def test_a_range_limit_never_adds_contact_and_infinity_is_no_limit(
    db_session_factory: Callable[[], Session],
) -> None:
    positions, _, times = _constellation_positions(db_session_factory)
    unlimited = isl.isl_windows(positions, times, SPEC)
    infinite = isl.isl_windows(
        positions, times, isl.IslSpec("Earth", R, H_GRAZE, max_range_km=math.inf))
    assert infinite == unlimited

    totals = []
    previous = unlimited
    for limit in (9000.0, 7000.0, 5000.0, 3000.0):
        limited = isl.isl_windows(
            positions, times, isl.IslSpec("Earth", R, H_GRAZE, max_range_km=limit))
        # Every limited window lies inside a window of the looser limit on the same pair.
        for w in limited:
            assert any(
                (p.body_a, p.body_b) == (w.body_a, w.body_b)
                and p.rise_s - 1e-9 <= w.rise_s and w.set_s <= p.set_s + 1e-9
                for p in previous
            )
            assert w.min_range_km < limit
        totals.append(sum(w.duration_s for w in limited))
        previous = limited
    total_unlimited = sum(w.duration_s for w in unlimited)
    assert all(t2 <= t1 for t1, t2 in zip([total_unlimited] + totals, totals))
    assert totals[-1] < total_unlimited, "the tightest limit must actually bind"


def test_a_segment_through_the_body_is_never_visible_and_a_radial_stack_always_is() -> None:
    """Antipodal points: the segment passes through the centre, clearance `-rho`. Two points on one
    radial line, one above the other: the infinite line passes through the centre too, but the
    *segment* stays at the inner radius - visible, by the clamp."""
    inner = np.array([0.0, 0.0, R1])
    assert float(segment_clearance(inner, -inner * R2 / R1, body_radius_km=R, h_graze_km=H_GRAZE)
                 ) == pytest.approx(-RHO)
    stacked = float(segment_clearance(inner, inner * R2 / R1, body_radius_km=R, h_graze_km=H_GRAZE))
    assert stacked == pytest.approx(R1 - RHO)

    times = np.array([0.0, 60.0, 120.0])
    positions = np.array([[inner, -inner, inner * R2 / R1]] * 3)
    windows = isl.isl_windows(positions, times, SPEC)
    assert [(w.body_a, w.body_b) for w in windows] == [(0, 2)]


def test_clearance_is_symmetric_and_its_sign_is_line_of_sight() -> None:
    rng = np.random.default_rng(7)
    r1 = rng.normal(size=(4000, 3)) * 9000.0
    r2 = rng.normal(size=(4000, 3)) * 9000.0
    c12 = segment_clearance(r1, r2, body_radius_km=R)
    c21 = segment_clearance(r2, r1, body_radius_km=R)
    np.testing.assert_allclose(c12, c21, rtol=0.0, atol=1e-8)
    clear = np.abs(c12) > 1e-6
    assert int(np.sum(clear & (c12 > 0.0))) > 100 and int(np.sum(clear & (c12 < 0.0))) > 100
    np.testing.assert_array_equal((c12 > 0.0)[clear], line_of_sight(r1, r2, body_radius_km=R)[clear])


def test_shape_guards() -> None:
    with pytest.raises(ValueError):
        isl.isl_windows(np.zeros((3, 2, 2)), np.array([0.0, 1.0, 2.0]), SPEC)
    with pytest.raises(ValueError):
        isl.isl_windows(np.zeros((3, 2, 3)), np.array([0.0, 1.0]), SPEC)
    with pytest.raises(ValueError):
        isl.isl_windows(np.zeros((1, 2, 3)), np.array([0.0]), SPEC)
    with pytest.raises(ValueError):
        isl.isl_windows(np.ones((3, 2, 3)), np.array([0.0, 2.0, 1.0]), SPEC)
    with pytest.raises(ValueError):
        isl.isl_contacts(np.ones((3, 2, 3)), np.ones((3, 1, 3)), np.array([0.0, 1.0, 2.0]), SPEC)


def test_blocking_over_pairs_does_not_change_the_answer(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory is bounded by evaluating pairs in blocks; a block boundary must be invisible."""
    positions, velocities, times = _constellation_positions(db_session_factory)
    whole = isl.isl_contacts(positions, velocities, times, SPEC)
    monkeypatch.setattr(isl, "_BLOCK_SAMPLES", 7 * times.size)     # 7 pairs per block, 10 blocks
    assert isl.isl_contacts(positions, velocities, times, SPEC) == whole


def test_the_metric_is_access_matching_per_pair() -> None:
    """`compare_isl_windows` is `access.compare_windows` on adapted records: overlap only, and
    windows of different pairs never match however well they line up."""
    def w(a: int, b: int, rise: float, set_: float) -> isl.IslWindow:
        return isl.IslWindow(a, b, rise, set_, set_ - rise, 1000.0, 0.5 * (rise + set_), False, False)

    truth = [w(0, 1, 0.0, 100.0), w(0, 2, 500.0, 600.0)]
    model = [w(0, 1, 3.0, 101.0), w(1, 2, 500.0, 600.0)]
    m = isl.compare_isl_windows(truth, model)
    assert (m.n_matched, m.passes_lost, m.passes_gained) == (1, 1, 1)
    assert m.rise.mean_s == pytest.approx(3.0) and m.duration.mean_s == pytest.approx(-2.0)
    [pair] = [p for p in m.matches if p.matched]
    assert (pair.station_index, pair.body_index) == (0, 1)
    assert isl.compare_isl_windows(truth, truth).rise.max_abs_s == 0.0


# ==================================================================================================
# e. The sweep
# ==================================================================================================

def _constellation_builder(session_factory: Callable[[], Session]) -> Callable[[], Simulation]:
    def build() -> Simulation:
        sim = scenarios.earth_constellation(
            session_factory(), n_sats=12, n_planes=3, altitude_km=550.0, inclination_deg=53.0,
        )
        sim.record_history = False
        return sim
    return build


def _access_spec(sample_dt_s: float = 60.0) -> access.AccessSpec:
    return access.AccessSpec(
        stations=[access.GroundStation("Wallops", math.radians(37.94), math.radians(-75.46), 0.01)],
        central_body="Earth", omega=EARTH_OMEGA, body_radius_km=R, sample_dt_s=sample_dt_s,
    )


KEPLER = sweep.ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=60.0)


def test_isl_metrics_leave_error_stats_and_access_bit_identical(
    db_session_factory: Callable[[], Session],
) -> None:
    build = _constellation_builder(db_session_factory)
    kw: Dict[str, Any] = dict(timing_batches=1, timing_warmup=0, access=_access_spec())
    [plain] = sweep.run_sweep(build, [KEPLER], 7200.0, **kw)
    [with_isl] = sweep.run_sweep(build, [KEPLER], 7200.0, isl=SPEC, **kw)
    assert plain.isl is None and with_isl.isl is not None
    assert with_isl.error == plain.error
    assert with_isl.access == plain.access
    m = with_isl.isl
    assert m.n_truth_windows > 0
    assert m.n_matched + m.passes_lost == m.n_truth_windows
    assert m.n_matched + m.passes_gained == m.n_model_windows


def _count_truth_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> List[Tuple[Tuple[Any, ...], Dict[str, Any]]]:
    calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    original = sweep.reference_for

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "reference_for", counting)
    return calls


@pytest.mark.parametrize("access_dt, expected_calls", [(None, 2), (60.0, 2), (120.0, 3)])
def test_one_dense_truth_per_sweep_shared_with_access_on_the_same_grid(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
    access_dt: float | None, expected_calls: int,
) -> None:
    calls = _count_truth_calls(monkeypatch)
    configs = [KEPLER, sweep.ModelConfig(name="k30", propagator=PropagatorType.KEPLERIAN, dt=30.0)]
    results = sweep.run_sweep(
        _constellation_builder(db_session_factory), configs, 3600.0, timing_batches=1, timing_warmup=0,
        access=None if access_dt is None else _access_spec(access_dt), isl=SPEC,
    )
    assert len(calls) == expected_calls
    assert all(r.isl is not None for r in results)


def test_every_truth_call_in_a_sweep_gets_the_same_model_keywords(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ISL truth is a separate `reference_for` call when its grid differs from access's. A truth
    model keyword forwarded to the endpoint and access truths but not to it would score the ISL
    metric against a different model - silently. So: identical keywords on every call."""
    calls = _count_truth_calls(monkeypatch)
    sweep.run_sweep(
        _constellation_builder(db_session_factory), [KEPLER], 3600.0, timing_batches=1, timing_warmup=0,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)}, access=_access_spec(60.0),
        isl=isl.IslSpec("Earth", R, H_GRAZE, sample_dt_s=120.0),
    )
    assert len(calls) == 3
    keywords = [kwargs for _, kwargs in calls]
    assert all(k == keywords[0] for k in keywords[1:]), keywords
    assert keywords[0]["oblateness"] is not None


def test_a_step_that_does_not_divide_the_isl_grid_raises(
    db_session_factory: Callable[[], Session],
) -> None:
    config = sweep.ModelConfig(name="odd", propagator=PropagatorType.KEPLERIAN, dt=90.0)
    with pytest.raises(ValueError, match="does not divide the ISL sample spacing"):
        sweep.run_sweep(
            _constellation_builder(db_session_factory), [config], 3600.0, timing_batches=1,
            timing_warmup=0, isl=SPEC,
        )


def test_an_external_tier_is_scored_on_isl_windows_too(
    db_session_factory: Callable[[], Session],
) -> None:
    """An `ExternalTier` that *is* the Keplerian solution, against point-mass truth: its link
    windows must match truth's to the truth's own tolerance - ~1e-9 km over `|c'|` ~ 1.7 km/s,
    i.e. well under a microsecond; 1e-4 s leaves room without admitting a real error."""
    build = _constellation_builder(db_session_factory)
    probe = build()
    names = [n for n, s in probe.name_to_index.items() if s in set(sweep.eligible_bodies(probe).tolist())]
    names.sort(key=lambda n: probe.name_to_index[n])

    def positions(times: NDArray[np.float64]) -> NDArray[np.float64]:
        sim = build()
        slots = [sim.name_to_index[n] for n in names]
        out: NDArray[np.float64] = sample_states(
            sim, slots, np.asarray(times, dtype=np.float64),
            relative_to=sim.name_to_index["Earth"], max_dt=60.0)[..., :3]
        return out

    tier = sweep.ExternalTier(name="kepler-external", dt=60.0, bodies=names, central_body="Earth",
                              positions=positions)
    [engine, external] = sweep.run_sweep(
        build, [KEPLER], 7200.0, timing_batches=1, timing_warmup=0, isl=SPEC, external=[tier],
    )
    assert external.isl is not None and engine.isl is not None
    m = external.isl
    assert m.n_truth_windows > 0 and m.passes_lost == 0 and m.passes_gained == 0
    assert m.rise.max_abs_s < 1e-4 and m.set.max_abs_s < 1e-4

    # A spec centred on another body than the tier's positions are relative to is refused.
    truth = sweep.reference_for(build(), np.array([0.0, 7200.0]))
    grid_truth = sweep.reference_for(build(), access.access_grid(7200.0, 60.0))
    elsewhere = isl.IslSpec(central_body=names[0], body_radius_km=R, h_graze_km=H_GRAZE)
    with pytest.raises(ValueError, match="central body"):
        sweep.score_external(tier, 7200.0, truth, timing_batches=1, timing_warmup=0,
                             isl=elsewhere, isl_truth=grid_truth)


def test_fast_headline_common_phase_lag_moves_every_link_window_early(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The benchmark's sharpest prediction, on 6 h against J2 truth. Cowell + j2 at 60 s against J2
    truth has one dominant error: RK4's phase lag, common to all twelve satellites (same `a`, same
    `n h`). A common along-track error is an exact time translation of every pair's geometry, so
    every ISL window moves by `-delta_s / v`, `v = r n = 7.589 km/s` - **early** (the model is
    ahead), nothing lost or gained. Expected worst shift: the horizon's error over `v`, taken from
    the sweep's own position error (0.175 km -> 0.023 s); tolerance 25 % for the residual edge bias
    (`C Delta h`) and the error's growth across the last windows. A window near `t = 0` has
    `Delta ~ 0`, where the small non-common part of the error can tip the sign: asserted as "no
    positive shift beyond 10 % of the worst negative one" rather than "every shift negative".
    Duration: RK4's common radial (energy) error closes the link margin by `0.93 delta_r` and so
    shortens windows at both ends; bounded here by half the worst shift, not claimed to be zero.

    Kepler, whose error is dominated by seed-dependent along-track drift and the nodal regression
    the ISL cannot see, must be far worse in ISL windows but, relative to the ground metric on the
    same run, *better*: its ISL mean |shift| below its ground mean |shift|.
    """
    horizon = 6.0 * 3600.0
    j2 = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}
    cowell = sweep.ModelConfig(
        name="cowell", propagator=PropagatorType.COWELL, dt=60.0,
        force_models=(sweep.ForceModelSpec(POINT_MASS_MODEL), sweep.ForceModelSpec(J2_MODEL, j2)))
    stations = access.AccessSpec(
        stations=[
            access.GroundStation("Kiruna", math.radians(67.86), math.radians(20.96), 0.40),
            access.GroundStation("Wallops", math.radians(37.94), math.radians(-75.46), 0.01),
            access.GroundStation("Santiago", math.radians(-33.15), math.radians(-70.67), 0.73),
        ],
        central_body="Earth", omega=EARTH_OMEGA, body_radius_km=R,
        mask_angle_rad=math.radians(5.0),
    )
    kepler_r, cowell_r = sweep.run_sweep(
        _constellation_builder(db_session_factory), [KEPLER, cowell], horizon,
        oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)}, timing_batches=1, timing_warmup=0,
        access=stations, isl=SPEC,
    )
    assert cowell_r.isl is not None and kepler_r.isl is not None and kepler_r.access is not None

    v = (R + 550.0) * math.sqrt(MU / (R + 550.0) ** 3)
    expected_max = cowell_r.error.max_km / v
    m = cowell_r.isl
    assert m.n_truth_windows > 50 and m.passes_lost == 0 and m.passes_gained == 0
    shifts = [s for p in m.matches for s in (p.rise_shift_s, p.set_shift_s) if s is not None]
    assert max(shifts) < 0.1 * expected_max, "a model ahead along-track links early"
    assert m.rise.mean_s < 0.0 and m.set.mean_s < 0.0
    assert max(abs(s) for s in shifts) == pytest.approx(expected_max, rel=0.25)
    assert abs(m.duration.mean_s) < 0.5 * expected_max

    k = kepler_r.isl
    assert k.rise.mean_abs_s > 50.0 * m.rise.mean_abs_s
    assert k.rise.mean_abs_s < kepler_r.access.rise.mean_abs_s
