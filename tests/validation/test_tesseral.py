"""
Validation of the `"tesseral"` force model (`tesseral.tesseral_kernel`, orders m >= 1 of degrees 2..4)
and of the matching truth option (`reference_for(..., tesseral=...)`).

Every tolerance is a named constant derived in the comment above it *before* measurement; the measured
value is quoted next to it. The checks, in order:

0. Registration, the EGM96 table's normalisation against an independently remembered unnormalised
   table, refusals, exact-zero behaviour, and fallback off the fused compiled plan.
a. Field: the kernel (Cunningham V/W recursion, rotation matrices) against `reference.tesseral_field`
   (explicit d^m P_n/ds^m polynomials, complex phase, monomial gradient) per pair and rotation angle;
   against a Richardson central difference of the potential built on `numpy.polynomial.legendre` (a
   third evaluation); against the closed-form equatorial tangential acceleration; and a body-fixed
   invariance check (a point co-rotating with the Earth sees a constant body-frame field).
b. Time: Cowell + tesseral converges on tesseral truth at fourth order - so every RK4 stage gets its
   own time - while freezing the rotation at the step's start is first order (negative control); and
   a step split by a manoeuvre starts its second half at its own clock.
c-f. GEO longitude drift, one shared 6-day run (`scenarios.geostationary_satellites`): the J22
   equilibria and the sign of the drift on either side; the drift acceleration against first-order
   theory at several longitudes; the full 4x4 stable points (verification against the closed form,
   comparison against the published figures); and the east-west station-keeping cost in m/s/year.
g. Truth: bit-identical when unused, argument errors, phase with a mid-run start, and conservation of
   momentum and the Jacobi-type integral `E - omega L_z` with a massive rotating body. Sweep wiring.

Notation: per pair the natural scale is `S_nm = mu |C_nm, S_nm| R^n / r^(n+2)`; field errors are
normalised by it. Unlike the zonal case `|a_nm| / S_nm` is not O(1): it reaches ~500 for (4,4), because
`P_nm` itself does (P_55 = 945 at the equator), so the worst-case bounds below carry `P_max` factors.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List, Mapping, Tuple

import numpy as np
import pytest
from numpy.polynomial import legendre
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import reference, registry, scenarios, tesseral
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import EARTH_OMEGA
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.tesseral import TESSERAL_MODEL, TESSERAL_PAIRS

ArrF = NDArray[np.float64]
Pair = Tuple[int, int]
EPS = float(np.finfo(np.float64).eps)
MU = scenarios.MU_EARTH
R = EARTH_R_EQ
TRUTH = dict(rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL)
EARTH_CS: Dict[Pair, Tuple[float, float]] = {
    (n, m): (tesseral.EARTH_TESSERALS[f"c{n}{m}"], tesseral.EARTH_TESSERALS[f"s{n}{m}"]) for n, m in TESSERAL_PAIRS
}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def _kernel_accel(rel: ArrF, t: float, cs: Mapping[Pair, Tuple[float, float]], *, theta0: float = 0.0,
                  omega: float = EARTH_OMEGA, r_eq: float = R) -> ArrF:
    """`tesseral_kernel` at bare relative positions: slot 0 is a root parent at the origin."""
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = MU
    params = np.zeros((n + 1, len(tesseral.TESSERAL_PARAM_NAMES)))
    params[1:, :3] = (r_eq, omega, theta0)
    for (deg, order), (c, s) in cs.items():
        params[1:, tesseral.TESSERAL_PARAM_NAMES.index(f"c{deg}{order}")] = c
        params[1:, tesseral.TESSERAL_PARAM_NAMES.index(f"s{deg}{order}")] = s
    out = np.zeros((n + 1, 3))
    tesseral.tesseral_kernel(np.arange(1, n + 1, dtype=np.int64), t, state, mu_array,
                             np.zeros(n + 1, dtype=np.int32), params, out)
    return out[1:]


def _points() -> ArrF:
    """LEO to GEO, both hemispheres, the equator and poles exactly and a hair off, many longitudes."""
    lat_deg = [0.0, 1e-6, 0.5, -0.5, 12.0, -20.0, 30.0, -39.23, 45.0, -60.0, 63.43, -75.0, 89.9, -89.9,
               90.0, -90.0]
    lon_deg = [0.0, 17.0, 45.0, 90.0, 133.0, 180.0, 215.0, 270.0, 301.0, 359.0, 12.0, 77.0, 160.0,
               240.0, 0.0, 0.0]
    pts = []
    for r in (6500.0, 7000.0, 7378.0, 20000.0, 42164.0):
        for la, lo in zip(np.radians(lat_deg), np.radians(lon_deg)):
            pts.append([r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo), r * math.sin(la)])
    return np.array(pts)


def _scale(rel: ArrF, pair: Pair, cs: Tuple[float, float]) -> ArrF:
    r = np.linalg.norm(rel, axis=1)
    return np.asarray(MU * math.hypot(*cs) * R ** pair[0] / r ** (pair[0] + 2))


def _p_max(n: int, m: int) -> float:
    """max |P_nm| on [-1, 1] is at most d^m P_n/ds^m at s = 1 = (n+m)!/(2^m m! (n-m)!)."""
    if m < 0 or m > n:
        return 0.0
    return math.factorial(n + m) / (2 ** m * math.factorial(m) * math.factorial(n - m))


def _kernel_side_bound(n: int, m: int) -> float:
    """
    Worst-case rounding of the kernel, in units of S_nm. Its terms are V/W of degree n + 1 and orders
    m - 1, m, m + 1, bounded by `_p_max` in these units, with weights 1/2, (n-m+1) and
    (n-m+2)(n-m+1)/2, and C, S combining (sqrt 2). Each recursion level adds ~4 eps relative and the
    deepest entry is n + m + 2 levels down, plus rotation in and out: 4 (n + m + 4) eps.
    """
    magnitude = (0.5 * _p_max(n + 1, m + 1) + (n - m + 1) * _p_max(n + 1, m)
                 + 0.5 * (n - m + 2) * (n - m + 1) * _p_max(n + 1, m - 1))
    return 4.0 * (n + m + 4) * math.sqrt(2.0) * magnitude * EPS


# ==================================================================================================
# 0. Registration, constants, refusals, zeros, dispatch
# ==================================================================================================

def test_tesseral_is_registered_with_its_coefficients_and_citation() -> None:
    model = registry.get_force_model(TESSERAL_MODEL)
    assert model.kernel is tesseral.tesseral_kernel
    assert model.param_names[:3] == ("r_eq", "omega", "theta0")
    assert model.param_names[3:] == tuple(f"{cs}{n}{m}" for n, m in TESSERAL_PAIRS for cs in "cs")
    assert "Montenbruck" in model.citation
    assert model.validate_bodies is not None and model.validate_coefficients is not None


# JGM-3 unnormalised coefficients as tabulated by Montenbruck & Gill (Table 3.2) - restated from memory,
# independently of the EGM96 normalised table the engine carries. Two modern fields differ in these
# terms by ~1e-3 relative (measured below: at most 1.5e-3, for S32); a normalisation error is a factor
# of sqrt(2) (the 2 - delta_m0) or of (n+m)!/(n-m)! - 0.4 to 5e3 - so 1 % separates the two cleanly.
JGM3_UNNORMALISED: Dict[Pair, Tuple[float, float]] = {
    (2, 2): (1.57446e-6, -0.90380e-6),
    (3, 1): (2.19264e-6, 0.26801e-6),
    (3, 2): (0.30990e-6, -0.21114e-6),
    (3, 3): (0.10056e-6, 0.19720e-6),
    (4, 1): (-0.50872e-6, -0.44945e-6),
    (4, 2): (0.07841e-6, 0.14815e-6),
    (4, 3): (0.05921e-6, -0.01201e-6),
    (4, 4): (-0.00398e-6, 0.00652e-6),
}
FIELD_TABLE_REL_TOL = 0.01
# The widely quoted J22 = 1.815e-6 and lambda22 = -14.93 deg (4 figures): half a unit in the last place.
J22_QUOTED, LAMBDA22_QUOTED_DEG = 1.8154e-6, -14.929


def test_egm96_table_normalises_onto_the_independent_unnormalised_table() -> None:
    """Measured worst relative difference 1.5e-3 (S32); (2,1) excluded - both fields put it at the
    ~1e-9 noise level where they legitimately disagree."""
    for pair, (c_want, s_want) in JGM3_UNNORMALISED.items():
        c_got, s_got = EARTH_CS[pair]
        for got, want in ((c_got, c_want), (s_got, s_want)):
            assert abs(got / want - 1.0) < FIELD_TABLE_REL_TOL, (pair, got, want)
    j22, lam22 = tesseral.j22_amplitude_and_longitude(*EARTH_CS[(2, 2)])
    assert abs(j22 - J22_QUOTED) < 0.5e-10 and abs(math.degrees(lam22) - LAMBDA22_QUOTED_DEG) < 5e-4
    assert abs(tesseral.normalisation_factor(2, 0) - math.sqrt(5.0)) < 4 * EPS   # zonal.py's factor
    assert dict(tesseral.EARTH_J22) == {"c22": EARTH_CS[(2, 2)][0], "s22": EARTH_CS[(2, 2)][1]}


def _constellation(session: Session, n_sats: int = 2) -> Tuple[Simulation, NDArray[np.int64]]:
    sim = scenarios.earth_constellation(session, n_sats=n_sats, n_planes=1)
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    return sim, sats


def test_refusals_leave_the_arena_untouched(db_session_factory: Callable[[], Session]) -> None:
    """Each refusal is a silent-physics failure otherwise: r_eq = 0 zeroes every term, a forgotten omega
    freezes the Earth's field in inertial space, a NaN poisons the row. A barycentre has no shape."""
    sim, sats = _constellation(db_session_factory())
    bit = np.uint64(1) << np.uint64(registry.get_force_model(TESSERAL_MODEL).bit)
    c22 = tesseral.EARTH_J22["c22"]
    for bad in ({"c22": c22, "omega": EARTH_OMEGA}, {"c22": c22, "r_eq": R},
                {"c22": c22, "r_eq": -R, "omega": EARTH_OMEGA},
                {"s33": float("nan"), "r_eq": R, "omega": EARTH_OMEGA}, {"theta0": float("inf")}):
        with pytest.raises(ValueError):
            sim.enable_force_model(TESSERAL_MODEL, sats, **bad)
        assert not np.any(sim.force_model_mask[sats] & bit), bad
        assert TESSERAL_MODEL not in sim.force_model_params, bad

    # Stored r_eq/omega are the effective values for a later coefficient-only call ...
    sim.enable_force_model(TESSERAL_MODEL, sats, r_eq=R, omega=EARTH_OMEGA)
    sim.enable_force_model(TESSERAL_MODEL, sats, **tesseral.EARTH_J22)
    # ... and zeroing omega while C22 is stored is refused.
    with pytest.raises(ValueError, match="omega"):
        sim.enable_force_model(TESSERAL_MODEL, sats, omega=0.0)
    assert np.all(sim.force_model_params[TESSERAL_MODEL][sats, 1] == EARTH_OMEGA)

    sim2, sats2 = _constellation(db_session_factory())
    sim2.parent_indices[sats2[0]] = sim2.name_to_index["Earth Barycenter"]
    with pytest.raises(ValueError, match="barycentre"):
        sim2.enable_force_model(TESSERAL_MODEL, sats2, r_eq=R, omega=EARTH_OMEGA, **tesseral.EARTH_J22)


def test_zero_coefficients_and_zero_separation_contribute_exactly_nothing() -> None:
    rel = _points()
    n = rel.shape[0]
    rng = np.random.default_rng(5)
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = MU
    parents = np.zeros(n + 1, dtype=np.int32)
    before = rng.normal(size=(n + 1, 3))
    width = len(tesseral.TESSERAL_PARAM_NAMES)

    # All-zero row, and a row with r_eq/omega/theta0 but no coefficients: `out` bit-identical.
    for head in ((0.0, 0.0, 0.0), (R, EARTH_OMEGA, 1.3)):
        params = np.zeros((n + 1, width))
        params[:, :3] = head
        out = before.copy()
        tesseral.tesseral_kernel(np.arange(1, n + 1, dtype=np.int64), 1234.5, state, mu_array, parents,
                                 params, out)
        assert np.array_equal(out, before)

    # The root slot (self-parented, zero separation) with the full field: exactly 0.0, no warning.
    params = np.zeros((n + 1, width))
    params[:, :3] = (R, EARTH_OMEGA, 0.2)
    for j, pair in enumerate(TESSERAL_PAIRS):
        params[:, 3 + 2 * j: 5 + 2 * j] = EARTH_CS[pair]
    out = before.copy()
    with np.errstate(all="raise"):
        tesseral.tesseral_kernel(np.array([0], dtype=np.int64), 99.0, state, mu_array, parents, params, out)
    assert np.array_equal(out, before)


def test_cowell_body_with_tesseral_falls_back_to_the_numpy_path(
    db_session_factory: Callable[[], Session],
) -> None:
    """The fused compiled Cowell kernel has no `t` and no tesseral term, so the bit is foreign to its
    plan: the whole Cowell set runs `RK4Integrator` over `forces.compose_accelerations`."""
    sim, sats = _constellation(db_session_factory())
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sats)
    sim.enable_force_model(J2_MODEL, sats, j2=EARTH_J2, r_eq=R)
    assert sim._cowell_fused_ok
    sim.enable_force_model(TESSERAL_MODEL, sats[:1], r_eq=R, omega=EARTH_OMEGA, **tesseral.EARTH_TESSERALS)
    assert not sim._cowell_fused_ok


# ==================================================================================================
# a. Field
# ==================================================================================================

def _truth_side_bound(n: int, m: int) -> float:
    """
    Worst-case rounding of `reference.tesseral_field`, units of S_nm: each monomial term c z^k w^m r^-q
    (q = n+m+k+1) and its gradient are at most |c| (q + k + m) in these units (|z|, |w| <= r), with
    ~(1.5 q + 2 m + 8) eps relative - r's 1.5 eps through the power, m complex products, a few more -
    and no cancellation credit.
    """
    table = {(2, 1): [(3.0, 1)], (2, 2): [(3.0, 0)], (3, 1): [(7.5, 2), (-1.5, 0)], (3, 2): [(15.0, 1)],
             (3, 3): [(15.0, 0)], (4, 1): [(17.5, 3), (-7.5, 1)], (4, 2): [(52.5, 2), (-7.5, 0)],
             (4, 3): [(105.0, 1)], (4, 4): [(105.0, 0)]}
    total = 0.0
    for c, k in table[(n, m)]:
        q = n + m + k + 1
        total += abs(c) * (q + k + m) * (1.5 * q + 2 * m + 8)
    return total * EPS


ROTATION_ANGLES = (0.0, 0.7, -2.1, math.pi, 5.9)


def test_kernel_agrees_with_the_truth_field_per_pair_and_rotation() -> None:
    """Bounds (kernel + truth side, units of S_nm): 2.8e-13 (2,1) up to 3.7e-11 (4,4). Measured (80
    points x 5 angles, the kernel reaching each angle through theta0 + omega t): 5.3e-15 (2,1),
    1.5e-14 (2,2), 1.4e-13 (3,3), 3.2e-13 (4,3), 1.5e-12 (4,4) - 25-95x inside, as for the zonal
    cross-check; |a|/S_nm reaches ~500 for (4,4), so that is ~3e-15 of the field itself. A transposed C/S, a wrong
    recursion coefficient or a rotation the wrong way round is O(1) here."""
    rel = _points()
    theta0 = 0.4
    for pair in TESSERAL_PAIRS:
        cs = EARTH_CS[pair]
        bound = _kernel_side_bound(*pair) + _truth_side_bound(*pair)
        worst = 0.0
        for theta in ROTATION_ANGLES:
            t = (theta - theta0) / EARTH_OMEGA
            got = _kernel_accel(rel, t, {pair: cs}, theta0=theta0)
            want = MU * reference.tesseral_field(rel, theta, R, {pair: cs})
            worst = max(worst, float(np.max(np.linalg.norm(got - want, axis=1) / _scale(rel, pair, cs))))
        assert worst < bound, (pair, worst, bound)

    # The full field at once: the sum of the per-pair bounds, against the sum of the scales.
    got = _kernel_accel(rel, 3.0e4, EARTH_CS, theta0=theta0)
    want = MU * reference.tesseral_field(rel, theta0 + EARTH_OMEGA * 3.0e4, R, EARTH_CS)
    scale = sum(_scale(rel, p, EARTH_CS[p]) for p in TESSERAL_PAIRS)
    bound = sum(_kernel_side_bound(*p) + _truth_side_bound(*p) for p in TESSERAL_PAIRS)
    assert float(np.max(np.linalg.norm(got - want, axis=1) / scale)) < bound


# P_nm(0) = (-1)^((n-m)/2) (n+m-1)!! / (n-m)!! for n - m even, 0 otherwise - a third route to the
# equatorial values, via the double factorial rather than any polynomial.
def _p_nm_equator(n: int, m: int) -> float:
    if (n - m) % 2:
        return 0.0

    def dfact(k: int) -> int:
        return 1 if k <= 0 else k * dfact(k - 2)
    return float((-1) ** ((n - m) // 2) * dfact(n + m - 1) / dfact(n - m))


def _equatorial_tangential(lam: float, a: float, cs: Mapping[Pair, Tuple[float, float]]) -> float:
    """Eastward acceleration on the equator at body-fixed longitude lam: (1/r) dU/dlambda =
    (mu/a^2) sum (R/a)^n P_nm(0) m (-C sin m lam + S cos m lam)."""
    return sum((MU / a ** 2) * (R / a) ** n * _p_nm_equator(n, m) * m * (-c * math.sin(m * lam) + s * math.cos(m * lam))
               for (n, m), (c, s) in cs.items())


EQUATOR_REL_TOL = 64 * EPS      # ~20 rounded operations on each side of a well-conditioned value


def test_equatorial_tangential_acceleration_matches_the_closed_form() -> None:
    """The quantity the GEO tests rest on, from the kernel with the Earth turned to theta = 0.9 rad:
    the eastward component at body-fixed longitude lam on the equator. For J22 alone it is
    -(mu/a^2)(R/a)^2 6 J22 sin 2(lam - lam22): **westward** just east of the long axis lam22, which is
    what makes lam22 unstable. Checked at sin 2(lam - lam22) = +-1 and in between; measured 2.6 eps
    (J22) and 2.7 eps (4x4) of the largest value."""
    a = scenarios.geostationary_radius_km()
    theta = 0.9
    j22, lam22 = tesseral.j22_amplitude_and_longitude(*EARTH_CS[(2, 2)])
    for cs in ({(2, 2): EARTH_CS[(2, 2)]}, EARTH_CS):
        lams = lam22 + np.radians([45.0, 135.0, 17.0, 250.0, 301.0])
        inertial = lams + theta
        rel = a * np.stack([np.cos(inertial), np.sin(inertial), np.zeros_like(lams)], axis=1)
        acc = _kernel_accel(rel, theta / EARTH_OMEGA, cs)
        east = -acc[:, 0] * np.sin(inertial) + acc[:, 1] * np.cos(inertial)
        want = np.array([_equatorial_tangential(float(x), a, cs) for x in lams])
        big = float(np.max(np.abs(want)))
        assert float(np.max(np.abs(east - want))) < EQUATOR_REL_TOL * big, (east, want)
        if len(cs) == 1:
            # 45 deg east of the long axis: westward, at exactly the J22 closed form.
            single = -(MU / a ** 2) * (R / a) ** 2 * 6.0 * j22
            assert east[0] < 0.0 and abs(east[0] / single - 1.0) < EQUATOR_REL_TOL, (east[0], single)


def _potential_clenshaw(rel: ArrF, theta: float, pair: Pair, cs: Tuple[float, float]) -> ArrF:
    """U_nm = mu R^n P_nm(sin phi)(C cos m lam + S sin m lam) / r^(n+1), geodesy sign, with
    P_nm = cos(phi)^m d^m P_n/ds^m: the derivative from `numpy.polynomial.legendre.legder` evaluated by
    Clenshaw (`legval`), and cos(phi) = sqrt(x^2 + y^2) / r - neither the kernel's recursion nor the
    truth's typed polynomials. (`scipy.special.lpmv` was the first choice and is unusable here: it
    rebuilds cos(phi) as sqrt(1 - s^2), which near the poles loses eps / cos^2(phi) - 7e-11 relative at
    89.9 deg - before the finite difference divides by h/r.)"""
    n, m = pair
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    rho = np.hypot(rel[:, 0], rel[:, 1])
    lam = np.arctan2(rel[:, 1], rel[:, 0]) - theta
    d_m = legendre.legval(rel[:, 2] / r, legendre.legder([0.0] * n + [1.0], m))
    p_nm = (rho / r) ** m * d_m
    return np.asarray(MU * R ** n * p_nm * (cs[0] * np.cos(m * lam) + cs[1] * np.sin(m * lam)) / r ** (n + 1))


FD_STEP_REL = 3e-4


def _fd_bound(n: int, m: int) -> float:
    """
    Richardson (4 D(h) - D(2h))/3 leaves -(h^4/30) f^(5). U_nm is (up to (n-m)!) the real or imaginary
    part of (d_x + i d_y)^m d_z^(n-m) (1/r): 2^m partial derivatives of order n, and every N-th partial
    of 1/r is bounded by N!/r^(N+1). So the normalised truncation is (h/r)^4 2^m (n+5)!/(30 (n-m)!).
    Rounding: U carries ~Q eps of up to P_max(n,m) in units of S_nm r, Q = 3n + 2m + 12 (Clenshaw, the
    power of cos(phi), the angle and cos/sin, the power of r); the quotient and Richardson weights make
    it 1.5 eps Q P_max / (h/r).
    At (4,4): truncation 1.6e-9, rounding 3.5e-9 - of a field reaching ~500 S_nm.
    """
    truncation = FD_STEP_REL ** 4 * 2 ** m * math.factorial(n + 5) / (30.0 * math.factorial(n - m))
    rounding = 1.5 * EPS * (3 * n + 2 * m + 12) * _p_max(n, m) / FD_STEP_REL
    return truncation + rounding


def test_kernel_is_the_gradient_of_the_tesseral_potential() -> None:
    """Measured (80 points, theta = 1.1, units of S_nm) 3.0e-12 (2,1) to 8.3e-10 (4,4), against
    bounds 6.9e-11 to 5.3e-9 - 6-30x inside, rounding-dominated."""
    rel = _points()
    theta = 1.1
    for pair in TESSERAL_PAIRS:
        cs = EARTH_CS[pair]
        r = np.linalg.norm(rel, axis=1)

        def fd(h_rel: float) -> ArrF:
            a = np.empty_like(rel)
            for axis in range(3):
                plus, minus = rel.copy(), rel.copy()
                plus[:, axis] += h_rel * r
                minus[:, axis] -= h_rel * r
                a[:, axis] = (_potential_clenshaw(plus, theta, pair, cs) - _potential_clenshaw(minus, theta, pair, cs)) \
                    / (plus[:, axis] - minus[:, axis])
            return a

        rich = (4.0 * fd(FD_STEP_REL) - fd(2 * FD_STEP_REL)) / 3.0
        got = _kernel_accel(rel, theta / EARTH_OMEGA, {pair: cs})
        err = float(np.max(np.linalg.norm(got - rich, axis=1) / _scale(rel, pair, cs)))
        assert err < _fd_bound(*pair), (pair, err, _fd_bound(*pair))


def test_field_is_fixed_in_the_rotating_body() -> None:
    """A point co-rotating with the Earth sees a constant body-frame field. Carried in and out through
    rotations by theta(t) = theta0 + omega t over a day; each adds a few eps, so the bound is twice the
    kernel-side rounding bound, 1.3e-10 of the summed scale; measured 1.5e-14 (the bound is worst-case
    aligned, the rotations are a handful of well-conditioned products). With the kernel's
    rotation reversed (-theta for +theta) the field would turn at -2 omega relative to the point, an O(1)
    change - this is the test that pins the rotation sense independently of the truth."""
    body = _points()
    theta0 = -0.3
    scale = sum(_scale(body, p, EARTH_CS[p]) for p in TESSERAL_PAIRS)
    bound = 2.0 * sum(_kernel_side_bound(*p) for p in TESSERAL_PAIRS)
    reference_bf = None
    for t in (0.0, 7200.0, 21600.0, 43210.0, 86164.0):
        th = theta0 + EARTH_OMEGA * t
        c, s = math.cos(th), math.sin(th)
        inertial = np.stack([c * body[:, 0] - s * body[:, 1], s * body[:, 0] + c * body[:, 1], body[:, 2]], 1)
        a = _kernel_accel(inertial, t, EARTH_CS, theta0=theta0)
        a_bf = np.stack([c * a[:, 0] + s * a[:, 1], -s * a[:, 0] + c * a[:, 1], a[:, 2]], 1)
        if reference_bf is None:
            reference_bf = a_bf
            continue
        err = float(np.max(np.linalg.norm(a_bf - reference_bf, axis=1) / scale))
        assert err < bound, (t, err, bound)


# ==================================================================================================
# b. Time: every RK4 stage at its own time
# ==================================================================================================

# An exaggerated, fast-turning field so that the time dependence is the dominant physics: EGM96 x 1000
# (C22 ~ 1.6e-3, J2-sized) turning at 10 x Earth rate, on the 550 km / 53 deg constellation orbit.
LOUD_CS = {p: (1000.0 * c, 1000.0 * s) for p, (c, s) in EARTH_CS.items()}
LOUD_OMEGA, LOUD_THETA0 = 10.0 * EARTH_OMEGA, 0.4
ORBIT_RADIUS = scenarios.EARTH_RADIUS + 550.0
PERIOD = 2.0 * math.pi * math.sqrt(ORBIT_RADIUS ** 3 / MU)
STEP_COUNTS = [256, 512, 1024]
# Fourth order: halving gives 16, and the zonal/J2 tests' band [12, 24] allows the leading term's
# neighbours. Frozen time is first order: effective stage lag h/2, so the acceleration is wrong by
# ~ m omega (h/2) |a| - at 1024 steps 4 x 7.3e-4 x 2.8 s x 2e-5 ~ 2e-7 km/s^2, ~km over an orbit if
# coherent - and halving gives 2; band [1.6, 2.6]. It must also sit >= 100x above the correct run.
ORDER_BAND = (12.0, 24.0)
FROZEN_BAND = (1.6, 2.6)


class _FrozenTimeIntegrator:
    """Negative control: the RK4 integrator with every stage handed the step's start time."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def step(self, provider: Callable[..., ArrF], t: float, state: ArrF, dt: float,
             indices: NDArray[np.int64], primaries: NDArray[np.int32]) -> None:
        self._inner.step(lambda _t, s: provider(t, s), t, state, dt, indices, primaries)  # type: ignore[attr-defined]


def _loud_sim(session: Session) -> Tuple[Simulation, NDArray[np.int64]]:
    sim, sats = _constellation(session, n_sats=1)
    sim.record_history = False
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sats)
    coefficients = {f"{cs}{n}{m}": v for (n, m), pair in LOUD_CS.items() for cs, v in zip("cs", pair)}
    sim.enable_force_model(TESSERAL_MODEL, sats, r_eq=R, omega=LOUD_OMEGA, theta0=LOUD_THETA0, **coefficients)
    return sim, sats


def _loud_error(session: Session, n_steps: int, truth: ArrF, frozen: bool = False) -> float:
    sim, sats = _loud_sim(session)
    if frozen:
        sim._cowell_integrator = _FrozenTimeIntegrator(sim._cowell_integrator)  # type: ignore[assignment]
    for _ in range(n_steps):
        sim.step(PERIOD / n_steps)
    return float(np.linalg.norm(sim.global_states[sats[0], :3] - truth))


def _loud_truth(session: Session) -> ArrF:
    sim, sats = _loud_sim(session)
    ref = reference.reference_for(sim, np.array([0.0, PERIOD]),
                                  tesseral={"Earth": (R, LOUD_OMEGA, LOUD_THETA0, LOUD_CS)}, **TRUTH)
    return ref.positions[-1, ref.index_of("SAT-00-000")].copy()


def test_cowell_tesseral_is_fourth_order_and_frozen_time_is_first_order(
    db_session_factory: Callable[[], Session],
) -> None:
    """Measured: 256/512/1024 steps -> 4.75e-4, 2.77e-5, 1.67e-6 km (ratios 17.1, 16.6 - the size of the
    J2 test's 4.2e-4, 2.4e-5, 1.5e-6, as the J2-sized field suggests); frozen time at 512/1024 ->
    14.2, 7.10 km (ratio 1.995), 4.2e6x the correct run and twice the coherent estimate."""
    truth = _loud_truth(db_session_factory())
    errors = [_loud_error(db_session_factory(), k, truth) for k in STEP_COUNTS]
    ratios = [errors[k] / errors[k + 1] for k in range(len(errors) - 1)]
    assert all(ORDER_BAND[0] < q < ORDER_BAND[1] for q in ratios), (ratios, errors)

    frozen = [_loud_error(db_session_factory(), k, truth, frozen=True) for k in STEP_COUNTS[1:]]
    assert FROZEN_BAND[0] < frozen[0] / frozen[1] < FROZEN_BAND[1], frozen
    assert frozen[1] > 100.0 * errors[-1], (frozen, errors)


# Splitting one step moves its RK4 nodes, changing that step's truncation by ~one local error:
# ~ E_global(512) / 512 = 1.1e-4 / 512 = 2.2e-7 km, and 64 splits add at most ~1.4e-5 km even if every
# one is coherent. A second half that restarted at the step's start time instead of the split epoch
# would evaluate a field lagging by omega x 0.4 h: ~ m omega (0.4 h) |a| (0.6 h)^2 / 2 ~ 1e-5 km per split,
# ~1e-3 km over 64. Tolerance 2e-5 km.
SPLIT_TOL_KM = 2e-5


def test_a_manoeuvre_split_substep_starts_at_its_own_time(db_session_factory: Callable[[], Session]) -> None:
    """Zero-Delta-v manoeuvres at 40 % of every 8th step cut 64 steps in two; the second half must start
    at the split epoch. Measured difference against the unsplit run 3.2e-6 km, inside the 1.4e-5 km
    all-coherent estimate (the unsplit run sits 2.8e-5 km from truth)."""
    n_steps = 512
    dt = PERIOD / n_steps

    def run(split: bool) -> ArrF:
        sim, sats = _loud_sim(db_session_factory())
        if split:
            for k in range(0, n_steps, 8):
                sim.schedule_delta_v(sats, [0.0, 0.0, 0.0], epoch_s=(k + 0.4) * dt)
        for _ in range(n_steps):
            sim.step(dt)
        assert not sim.pending_manoeuvres
        out: ArrF = sim.global_states[sats[0], :3].copy()
        return out

    diff = float(np.linalg.norm(run(True) - run(False)))
    assert 0.0 < diff < SPLIT_TOL_KM, diff


# ==================================================================================================
# c-f. GEO longitude drift - one shared run
# ==================================================================================================

GEO_A = scenarios.geostationary_radius_km()
J22, LAMBDA22 = tesseral.j22_amplitude_and_longitude(*EARTH_CS[(2, 2)])
# First-order drift constant, derived in tesseral.py: lambda_ddot = +K sin 2(lambda - lambda22).
K_J22 = 18.0 * EARTH_OMEGA ** 2 * (R / GEO_A) ** 2 * J22            # 3.976e-15 rad/s^2
SIDEREAL_S = 2.0 * math.pi / EARTH_OMEGA
STEPS_PER_DAY = 144                  # dt = 598.4 s; whole sidereal days so window means kill the wobble
GEO_DAYS = 6
J22_OFFSETS_DEG = (0.0, 10.0, -10.0, 45.0, 80.0, 90.0, 100.0, 135.0, 170.0, 180.0, 190.0, 260.0, 270.0, 280.0)
FULL = {p: EARTH_CS[p] for p in TESSERAL_PAIRS}
CONTROL_LON_DEG = 30.0


def _full_lddot(lam: float) -> float:
    """First-order drift under the full 4x4 field: lambda_ddot = -3 a_S / a (Gauss, near-circular)."""
    return -3.0 * _equatorial_tangential(lam, GEO_A, FULL) / GEO_A


def _full_roots() -> Dict[str, float]:
    """Closed-form equilibria of the full field, by bracketing on a 0.01 deg grid and bisection."""
    from scipy.optimize import brentq
    grid = np.radians(np.arange(-180.0, 180.0, 0.01))
    vals = np.array([_full_lddot(float(x)) for x in grid])
    roots: Dict[str, float] = {}
    for i in np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:])):
        x = brentq(_full_lddot, grid[i], grid[i + 1], xtol=1e-13)
        stable = vals[i] > 0.0          # lambda_ddot goes + to - : restoring
        key = ("stable" if stable else "unstable") + ("_east" if x > 0 else "_west")
        roots[key] = float(x)
    return roots


FULL_ROOTS = _full_roots()
WORST_FULL_DEG = 117.36              # max |lambda_ddot| of the 4x4 closed form (0.01 deg grid)
FULL_SITES_DEG = (math.degrees(FULL_ROOTS["stable_east"]) - 1.0, math.degrees(FULL_ROOTS["stable_east"]) + 1.0,
                  math.degrees(FULL_ROOTS["stable_west"]) - 1.0, math.degrees(FULL_ROOTS["stable_west"]) + 1.0,
                  WORST_FULL_DEG, 34.28)


def _window_fit(times: ArrF, lon: ArrF) -> Tuple[ArrF, ArrF]:
    """Means over whole sidereal days, then a quadratic in (t - centre): returns (lambda_ddot,
    lambda at the centre) per column. A quadratic fit's curvature is lambda_ddot at the window centre
    to first order; the window means remove the once-per-day wobble exactly."""
    n_win = (times.size - 1) // STEPS_PER_DAY
    tm = np.array([times[i * STEPS_PER_DAY:(i + 1) * STEPS_PER_DAY].mean() for i in range(n_win)])
    lm = np.array([lon[i * STEPS_PER_DAY:(i + 1) * STEPS_PER_DAY].mean(axis=0) for i in range(n_win)])
    centre = 0.5 * (tm[0] + tm[-1])
    coeffs = np.polyfit(tm - centre, lm, 2)
    return 2.0 * coeffs[0], coeffs[2]


def _theory_fit(times: ArrF, lam0: float, lddot: Callable[[float], float]) -> Tuple[float, float]:
    """The first-order pendulum lambda_ddot = f(lambda), from rest, through the same estimator - so
    the fit's own approximations (a quartic term leaks ~1e-4 K into a 16-day quadratic) cancel."""
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda _t, y: [y[1], lddot(y[0])], (0.0, float(times[-1])), [lam0, 0.0],
                    method="DOP853", t_eval=times, rtol=1e-12, atol=1e-16)
    ddot, lam_c = _window_fit(times, sol.y[0][:, np.newaxis])
    return float(ddot[0]), float(lam_c[0])


def _geo_session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def geo_run() -> Iterator[Dict[str, object]]:
    """
    One arena, 6 sidereal days at 144 steps per day (dt = 598.4 s), NumPy Cowell path:
    J22-only satellites at `lambda22 + J22_OFFSETS_DEG`, full-4x4 satellites at `FULL_SITES_DEG`, and a
    pm + J2 control. Every satellite is seeded at rest in the rotating frame.
    """
    lons = [math.degrees(LAMBDA22) + o for o in J22_OFFSETS_DEG] + list(FULL_SITES_DEG) + [CONTROL_LON_DEG]
    session = _geo_session()
    sim = scenarios.geostationary_satellites(session, longitudes_deg=lons)
    sim.record_history = False
    sats = np.array([sim.name_to_index[scenarios.geo_satellite_name(k)] for k in range(len(lons))])
    n22 = len(J22_OFFSETS_DEG)
    nfull = len(FULL_SITES_DEG)
    sim.enable_force_model(TESSERAL_MODEL, sats[:n22], r_eq=R, omega=EARTH_OMEGA, **tesseral.EARTH_J22)
    sim.enable_force_model(TESSERAL_MODEL, sats[n22:n22 + nfull], r_eq=R, omega=EARTH_OMEGA,
                           **tesseral.EARTH_TESSERALS)
    earth = sim.name_to_index["Earth"]
    n_steps = GEO_DAYS * STEPS_PER_DAY
    dt = SIDEREAL_S / STEPS_PER_DAY
    times = np.empty(n_steps + 1)
    lon = np.empty((n_steps + 1, len(lons)))
    for k in range(n_steps + 1):
        if k:
            sim.step(dt)
        rel = sim.global_states[sats, :3] - sim.global_states[earth, :3]
        times[k] = sim.t
        lon[k] = np.arctan2(rel[:, 1], rel[:, 0]) - EARTH_OMEGA * sim.t
    lon = np.unwrap(lon, axis=0)
    ddot, lam_c = _window_fit(times, lon)
    yield {"times": times, "lon0": lon[0], "ddot": ddot, "lam_c": lam_c, "control": float(ddot[-1]),
           "n22": n22, "nfull": nfull, "dt": dt}
    session.close()


def test_geo_control_drifts_at_the_rk4_energy_error(geo_run: Dict[str, object]) -> None:
    """The pm + J2 control has no longitude force, so its fitted lambda_ddot is RK4's: the Kepler
    energy error `(n h)^6 / 36` per step shrinks `a`, and `lambda_ddot = -(3/2) n da/dt / a` gives
    +(3/2) n (n h)^6 / (36 h) = 3.5041e-17 rad/s^2 at h = 598.4 s - 8.8e-3 K, which is why every
    measurement below subtracts it. Measured 3.5059e-17 (+0.05 %); 5 % allows the constant's O((nh)^2)
    correction. This is also the stated RK4 constant of CLAUDE.md, confirmed at GEO."""
    dt = float(geo_run["dt"])            # type: ignore[arg-type]
    nh = EARTH_OMEGA * dt
    predicted = 1.5 * EARTH_OMEGA * nh ** 6 / (36.0 * dt)
    assert abs(float(geo_run["control"]) / predicted - 1.0) < 0.05, (geo_run["control"], predicted)   # type: ignore[arg-type]


def _j22_measured(geo_run: Dict[str, object]) -> Tuple[ArrF, ArrF]:
    n22 = int(geo_run["n22"])            # type: ignore[call-overload]
    ddot = np.asarray(geo_run["ddot"])[:n22] - float(geo_run["control"])   # type: ignore[arg-type]
    return ddot, np.asarray(geo_run["lam_c"])[:n22]


# Exactly on an equilibrium the theory gives 0; the budget of the next test (<= 1.5e-4 K) bounds what
# the engine may show there. 1e-3 K is 340x below the drift 10 deg away (0.342 K).
EQUILIBRIUM_TOL = 1e-3


def test_geo_j22_equilibria_and_the_sign_of_the_drift_around_them(geo_run: Dict[str, object]) -> None:
    """Under J22 alone: satellites at lambda22 + 0/90/180/270 deg stay put (measured |lambda_ddot| <=
    3.4e-5 K); 10 deg either side of lambda22 and lambda22 + 180 they accelerate **away** (unstable), and
    either side of lambda22 + 90 and + 270 **toward** (stable) - so the stable slots are 75.07 E and
    104.93 W. With the textbook's minus sign on the drift formula the roles would swap."""
    ddot, _ = _j22_measured(geo_run)
    by_offset = dict(zip(J22_OFFSETS_DEG, ddot))
    for centre in (0.0, 90.0, 180.0, 270.0):
        assert abs(by_offset[centre]) < EQUILIBRIUM_TOL * K_J22, (centre, by_offset[centre] / K_J22)
    for centre, stable in ((0.0, False), (90.0, True), (180.0, False), (270.0, True)):
        east = by_offset[centre + 10.0 if centre + 10.0 in by_offset else centre - 350.0]
        west = by_offset[centre - 10.0 if centre - 10.0 in by_offset else centre + 350.0]
        if stable:
            assert east < 0.0 < west, (centre, east, west)
        else:
            assert west < 0.0 < east, (centre, east, west)
        assert min(abs(east), abs(west)) > 0.3 * K_J22           # sin 20 deg = 0.342


# Derived residual budget for the measured-vs-theory drift, in units of K, before measuring:
#  (i)   RK4 energy drift 8.8e-3 K, common to every satellite and removed with the control; what differs
#        between satellites is its dependence on the orbit, O(1e-6) of it: < 1e-7 K.
#  (ii)  J2 in the Gauss/Kepler relations (da/dt = 2 a_S/n and dn/da = -3n/2a both pick up O(J2 (R/a)^2)
#        = 2.5e-5 corrections, with coefficients of a few): ~1e-4 K.
#  (iii) second order in the tesseral field: J22 ~ 2e-6 relative.
#  (iv)  the once-per-day wobble (seed balances pm + J2 only; J22's own radial pull, 3.7e-7 of gravity,
#        leaves e ~ 4e-7): whole-sidereal-day window means remove it to ~(lambda_dot/omega) of itself.
#  (v)   that same unbalanced radial pull gives a linear drift (3/2) omega x 3.7e-7 over the run, moving
#        sin 2(lambda - lambda22) by <= 4e-5.
#  (vi)  the estimator: identical for engine and theory, so its own approximations cancel.
# Total <= ~1.5e-4 K; tolerance 5e-4 K. A sign error, a transposed C22/S22 or a factor-2 error is O(1).
DRIFT_TOL = 5e-4


def test_geo_j22_drift_acceleration_matches_first_order_theory(geo_run: Dict[str, object]) -> None:
    """Measured worst |engine - theory| 8.2e-5 K over the 14 J22 satellites, at lambda22 + 45 deg
    (budget item ii's size); there the engine reads 3.97635e-15 rad/s^2 against the theory's 3.97602e-15
    = 1.7006e-3 deg/day^2."""
    ddot, _ = _j22_measured(geo_run)
    times = np.asarray(geo_run["times"])
    lon0 = np.asarray(geo_run["lon0"])
    for k, offset in enumerate(J22_OFFSETS_DEG):
        want, _ = _theory_fit(times, float(lon0[k]), lambda x: K_J22 * math.sin(2.0 * (x - LAMBDA22)))
        assert abs(ddot[k] - want) < DRIFT_TOL * K_J22, (offset, (ddot[k] - want) / K_J22)


# Verification tolerance for the full-field stable points: the interpolation between the +-1 deg
# satellites is applied identically to the closed form, so only the measurement error enters: 1.5e-4 K
# (the budget above) over the local slope ~2 K per rad = 7.5e-5 rad = 0.004 deg. Tolerance 0.02 deg.
STABLE_POINT_TOL_DEG = 0.02
# Comparison: the published stable longitudes 75.1 E and 105.3 W (EGM-class fields, e.g. Soop, from
# memory). This 4x4 field omits degree >= 5, which by Kaula's rule moves each point by up to ~0.1 deg,
# the figures are printed to 0.1 deg, and published values scatter by ~0.3 deg with the field used. 0.4 deg.
PUBLISHED_STABLE_DEG = {"stable_east": 75.1, "stable_west": -105.3}
PUBLISHED_TOL_DEG = 0.4


def test_geo_full_field_stable_points(geo_run: Dict[str, object]) -> None:
    """Closed form (4x4 EGM96): stable 74.939 E and 105.094 W, unstable 11.519 W and 161.905 E (the
    J22-only points move by -0.13 and -0.17 deg, through J33 and degree 4). Measured by zero crossing
    between the satellites at +-1 deg: 74.9376 E and 105.0921 W - 1e-5 deg from the identically
    interpolated closed form, and 1.2e-3 / 1.8e-3 deg from its root (the interpolation's curvature
    error, as estimated). Against the published 75.1 E / 105.3 W: -0.16 and +0.21 deg - a comparison,
    not a verification."""
    n22 = int(geo_run["n22"])            # type: ignore[call-overload]
    ddot = np.asarray(geo_run["ddot"])[n22:] - float(geo_run["control"])   # type: ignore[arg-type]
    lam_c = np.asarray(geo_run["lam_c"])[n22:]
    for key, (i, j) in (("stable_east", (0, 1)), ("stable_west", (2, 3))):
        x0, x1 = float(lam_c[i]), float(lam_c[j])
        measured = x0 - ddot[i] * (x1 - x0) / (ddot[j] - ddot[i])
        f0, f1 = _full_lddot(x0), _full_lddot(x1)
        interpolated_theory = x0 - f0 * (x1 - x0) / (f1 - f0)
        assert abs(math.degrees(measured - interpolated_theory)) < STABLE_POINT_TOL_DEG, key
        assert abs(math.degrees(measured - FULL_ROOTS[key])) < STABLE_POINT_TOL_DEG, key
        assert ddot[i] > 0.0 > ddot[j]                                  # restoring, measured
        assert abs(math.degrees(FULL_ROOTS[key]) - PUBLISHED_STABLE_DEG[key]) < PUBLISHED_TOL_DEG, key


YEAR_S = 365.25 * 86400.0
# J22 closed form, derived before any run: a K / 3 per year = 1.7635 m/s/yr (K = 3.976e-15 rad/s^2).
# Full 4x4 closed form: 2.0659 m/s/yr at 117.36 E. The literature quotes ~1.7-2 m/s/yr at the worst
# longitudes (field and degree unstated) - a comparison, bracketed loosely: [1.6, 2.2].
J22_DV_PER_YEAR = 1.7635
LITERATURE_DV = (1.6, 2.2)


def test_geo_east_west_station_keeping_delta_v(geo_run: Dict[str, object]) -> None:
    """Holding a slot cancels the drift: Delta-v per unit time a |lambda_ddot| / 3 (Gauss). Measured:
    J22 at lambda22 + 45 / + 135 deg 1.7637 / 1.7636 m/s/yr against 1.7635; full 4x4 at 117.36 E
    2.0660 against 2.0659 m/s/yr - each within the drift test's 5e-4 of its closed form."""
    ddot, _ = _j22_measured(geo_run)
    closed_j22 = GEO_A * K_J22 / 3.0 * YEAR_S * 1e3
    assert abs(closed_j22 - J22_DV_PER_YEAR) < 1e-4, closed_j22
    for offset in (45.0, 135.0):
        dv = GEO_A * abs(ddot[J22_OFFSETS_DEG.index(offset)]) / 3.0 * YEAR_S * 1e3
        assert abs(dv / closed_j22 - 1.0) < DRIFT_TOL, (offset, dv)

    n22 = int(geo_run["n22"])            # type: ignore[call-overload]
    full = np.asarray(geo_run["ddot"])[n22:] - float(geo_run["control"])   # type: ignore[arg-type]
    worst = FULL_SITES_DEG.index(WORST_FULL_DEG)
    closed_full = GEO_A * abs(_full_lddot(float(np.asarray(geo_run["lam_c"])[n22 + worst]))) / 3.0 * YEAR_S * 1e3
    dv_full = GEO_A * abs(full[worst]) / 3.0 * YEAR_S * 1e3
    assert abs(dv_full - closed_full) < DRIFT_TOL * closed_j22, (dv_full, closed_full)
    # The worst site is the closed form's maximum (0.01 deg grid): nothing on the equator exceeds it.
    grid = np.radians(np.arange(-180.0, 180.0, 0.05))
    assert max(abs(_full_lddot(float(x))) for x in grid) <= abs(_full_lddot(math.radians(WORST_FULL_DEG))) * (1 + 1e-6)
    for dv in (closed_j22, dv_full):
        assert LITERATURE_DV[0] < dv < LITERATURE_DV[1], dv


# ==================================================================================================
# g. Truth: unused means bit-identical; phase; conservation. Sweep wiring.
# ==================================================================================================

def test_truth_without_tesseral_is_bit_identical(db_session_factory: Callable[[], Session]) -> None:
    """Omitted, `None`, empty and all-zero give today's truth bit for bit, with J2 and J3..J6 on."""
    sim, _ = _constellation(db_session_factory(), n_sats=3)
    times = np.arange(0.0, 1800.0 + 1.0, 300.0)
    kwargs: Dict[str, object] = dict(oblateness={"Earth": (EARTH_J2, R)},
                                     zonal={"Earth": (R, {3: -2.5326564853e-6, 4: -1.6196215914e-6})})
    base = reference.reference_for(sim, times, **kwargs)  # type: ignore[arg-type]
    zeros = {p: (0.0, 0.0) for p in TESSERAL_PAIRS}
    for variant in (None, {}, {"Earth": (R, EARTH_OMEGA, 0.3, {})}, {"Earth": (R, EARTH_OMEGA, 0.3, zeros)}):
        again = reference.reference_for(sim, times, tesseral=variant, **kwargs)  # type: ignore[arg-type]
        assert np.array_equal(again.positions, base.positions), variant
        assert np.array_equal(again.velocities, base.velocities), variant
        assert again.energy_drift == base.energy_drift
    with_field = reference.reference_for(sim, times, tesseral={"Earth": (R, EARTH_OMEGA, 0.3, EARTH_CS)},
                                         **kwargs)  # type: ignore[arg-type]
    assert not np.array_equal(with_field.positions, base.positions)


def test_truth_tesseral_arguments_and_phase(db_session_factory: Callable[[], Session]) -> None:
    sim, sats = _constellation(db_session_factory())
    t = np.array([0.0, 60.0])
    with pytest.raises(KeyError):
        reference.reference_for(sim, t, tesseral={"Earht": (R, EARTH_OMEGA, 0.0, {(2, 2): (1e-6, 0.0)})})
    for bad_pair in ((2, 0), (3, 0), (5, 1), (2, 3)):         # m = 0 has its own doors; out of range
        with pytest.raises(ValueError):
            reference.reference_for(sim, t, tesseral={"Earth": (R, EARTH_OMEGA, 0.0, {bad_pair: (1e-6, 0.0)})})
    with pytest.raises(ValueError):
        reference.reference_for(sim, t, tesseral={"Earth": (0.0, EARTH_OMEGA, 0.0, {(2, 2): (1e-6, 0.0)})})
    # A truth started mid-run carries theta0 + omega * sim.t, so it stays in phase with the engine.
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sats)
    sim.step(1000.0)
    ref = reference.reference_for(sim, t, tesseral={"Earth": (R, EARTH_OMEGA, 0.3, {(2, 2): EARTH_CS[(2, 2)]})})
    assert ref.tesseral is not None
    assert ref.tesseral[ref.index_of("Earth"), 2] == 0.3 + EARTH_OMEGA * 1000.0


# Momentum is a linear invariant (RK keeps it to rounding). E alone is NOT conserved - the rotating body
# does work - but E - omega L_z is (the field depends on t only through a rigid rotation about z, so
# dE/dt = omega dL_z/dt), held by DOP853 at rtol 1e-13 to ~1e-12 over a day, as the J2/zonal cases are.
# Exaggerated field (C, S = 1e-3, R = 3000 km, omega = 3e-4) so that a field inconsistent with its
# potential, or a Jacobi term with the wrong sign, moves the integral by ~1e-2 - ten orders above.
CONSERVATION_REL_TOL = 1e-11
ENERGY_ALONE_MIN_DRIFT = 1e-7


def test_massive_rotating_body_conserves_momentum_and_the_jacobi_integral(
    db_session_factory: Callable[[], Session],
) -> None:
    """Measured: momentum 2.1e-15, drift of E - omega L_z 1.4e-13; E alone drifts 2.3e-2."""
    sim = scenarios.two_body(db_session_factory(), mu_secondary=0.3 * MU, p=11000.0, e=0.2, i=0.9, raan=0.4)
    times = np.arange(0.0, 86400.0 + 1.0, 600.0)
    loud = {p: (1e-3, -1e-3 if p[1] % 2 else 1e-3) for p in TESSERAL_PAIRS}
    ref = reference.reference_for(sim, times, tesseral={"Primary": (3000.0, 3e-4, 0.2, loud)}, **TRUTH)
    mu = ref.mu
    momentum = np.einsum("i,tij->tj", mu, ref.velocities)
    scale = float(np.sum(mu * np.linalg.norm(ref.velocities[0], axis=1)))
    assert np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / scale < CONSERVATION_REL_TOL

    assert ref.tesseral is not None
    a = reference.tesseral_acceleration(ref.positions[0], mu, ref.tesseral, 0.0)
    assert np.linalg.norm(a[0]) > 1e-12                                     # the reaction is real
    assert np.linalg.norm(mu @ a) < 1e-12 * float(np.sum(mu * np.linalg.norm(a, axis=1)))
    assert ref.energy_drift < CONSERVATION_REL_TOL, ref.energy_drift

    # Without the omega L_z term the energy is visibly not conserved - the integral above has teeth.
    p_, s_ = ref.index_of("Primary"), ref.index_of("Secondary")

    def plain_energy(k: int) -> float:
        pos, vel = ref.positions[k], ref.velocities[k]
        kinetic = 0.5 * float(np.sum(mu * np.einsum("ij,ij->i", vel, vel)))
        rel = pos[s_] - pos[p_]
        phi = reference.tesseral_potential(rel[np.newaxis], 0.2 + 3e-4 * float(times[k]), 3000.0, loud)
        return kinetic - mu[p_] * mu[s_] / float(np.linalg.norm(rel)) + float(mu[p_] * mu[s_] * phi[0])
    assert abs(plain_energy(-1) / plain_energy(0) - 1.0) > ENERGY_ALONE_MIN_DRIFT


def test_run_sweep_forwards_tesseral_to_both_truths(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring: `run_sweep(tesseral=)` must reach the endpoint truth *and* the access truth - the gap the
    zonal review found (dropping `zonal=` from the access call passed every test at the time)."""
    from orbital_engine import access, sweep

    seen: List[object] = []
    real = sweep.reference_for

    def spy(*args: object, **kwargs: object) -> reference.ReferenceTrajectory:
        seen.append(kwargs.get("tesseral"))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sweep, "reference_for", spy)
    spec = access.AccessSpec(
        stations=[access.GroundStation("equator", 0.0, 0.0)], central_body="Earth",
        omega=EARTH_OMEGA, body_radius_km=scenarios.EARTH_RADIUS, sample_dt_s=60.0,
    )
    field = {"Earth": (R, EARTH_OMEGA, 0.0, {(2, 2): EARTH_CS[(2, 2)]})}
    sweep.run_sweep(
        lambda: _constellation(db_session_factory())[0],
        [sweep.ModelConfig("kepler", PropagatorType.KEPLERIAN, 60.0)],
        horizon_s=600.0, timing_batches=1, timing_warmup=0, tesseral=field, access=spec,
    )
    assert seen == [field, field], seen
