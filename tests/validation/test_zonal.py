"""
Validation of the `"zonal"` force model (`zonal.zonal_kernel`, J3..J6) and of the matching truth option
(`reference_for(..., zonal=...)`).

Every tolerance is a named constant derived in the comment above it *before* measurement; the measured
value is quoted next to it. The checks, in order:

0. Registration, constants (EGM96 J2 from the same table reproduces `geopotential.EARTH_J2`),
   refusals, exact-zero behaviour, and fused-plan membership.
a. Field, two derivations: the kernel (Legendre recursions, projected on r_hat / z_hat) against
   `reference.zonal_field` (explicit polynomials, monomial-by-monomial Cartesian gradient), per degree
   and summed.
b. Field is a gradient: the kernel against a Richardson-extrapolated central difference of the zonal
   potential, written once here through `numpy.polynomial.legendre` - a third evaluation of P_n.
c. Parity: odd degrees antisymmetric in z (horizontal components), even symmetric - bit for bit.
d. Truth: bit-identical when the option is unused; conservation with a massive oblate body.
e. Verification: engine Cowell + pm + j2 + zonal converges on truth with J2..J6 at fourth order.
f. Orbit dynamics, differential against a J2-only twin: J3's long-period eccentricity rate (and its
   sign flip with cos(omega)), and J4's secular node rate.

Notation: A_n = mu |J_n| R^n. Field errors are normalised by the degree's natural scale A_n / r^(n+2),
not by |a_n| - but |a_n| is never small against it: over latitude its minimum is 1.50, 1.73, 1.87 and
2.06 times the scale for n = 3..6, since the zeros of P_{n+1}' and P_n' interlace.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import pytest
from numpy.polynomial import legendre
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import geopotential, reference, registry, scenarios, zonal
from orbital_engine.custom_types import COEIndex, PropagatorType
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA
from orbital_engine.frames import ReferenceFrames
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.thrust import THRUST_MODEL
from orbital_engine.viz import sample_states
from orbital_engine.zonal import ZONAL_MODEL

ArrF = NDArray[np.float64]
EPS = float(np.finfo(np.float64).eps)
MU = scenarios.MU_EARTH
DEGREES = (3, 4, 5, 6)

# Restated literals (EGM96 unnormalised zonals, J_n = -sqrt(2n+1) Cbar_n0), so a change to the
# engine's constants is caught rather than mirrored.
J3, J4, J5, J6 = -2.5326564853e-6, -1.6196215914e-6, -2.2729608287e-7, 5.4068123911e-7
EARTH_ZONAL_TRUTH = {"Earth": (EARTH_R_EQ, {3: J3, 4: J4, 5: J5, 6: J6})}
TRUTH = dict(rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def _kernel_accel(rel: ArrF, r_eq: float, js: Mapping[int, float], mu: float = MU) -> ArrF:
    """`zonal_kernel` at bare relative positions: slot 0 is a root parent at the origin."""
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = mu
    params = np.zeros((n + 1, 5))
    params[1:, 0] = r_eq
    for degree, j_n in js.items():
        params[1:, degree - 2] = j_n
    out = np.zeros((n + 1, 3))
    zonal.zonal_kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, mu_array,
                       np.zeros(n + 1, dtype=np.int32), params, out)
    return out[1:]


def _points() -> ArrF:
    """A spread over radius (LEO to GEO), both hemispheres, the exact equator and poles, a hair off
    each, and the s^2 values where low-degree P_n' vanish."""
    lat_deg = [0.0, 1e-6, 0.5, -0.5, 12.0, -20.0, 30.0, -39.23, 45.0, -60.0, 63.43, -75.0, 89.9, -89.9,
               90.0, -90.0]
    lon_deg = [0.0, 17.0, 45.0, 90.0, 133.0, 180.0, 215.0, 270.0, 301.0, 359.0, 12.0, 77.0, 160.0,
               240.0, 0.0, 0.0]
    radii = [6500.0, 7000.0, 7378.0, 20000.0, 42164.0]
    pts = []
    for r in radii:
        for la, lo in zip(np.radians(lat_deg), np.radians(lon_deg)):
            pts.append([r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo), r * math.sin(la)])
    return np.array(pts)


def _scale(rel: ArrF, js: Mapping[int, float], r_eq: float = EARTH_R_EQ) -> ArrF:
    """sum_n mu |J_n| R^n / r^(n+2): the normalisation of every field error below."""
    r = np.linalg.norm(rel, axis=1)
    return np.asarray(sum(MU * abs(j) * r_eq ** n / r ** (n + 2) for n, j in js.items()))


def _normalised_error(got: ArrF, want: ArrF, scale: ArrF) -> float:
    return float(np.max(np.linalg.norm(got - want, axis=1) / scale))


# ==================================================================================================
# 0. Registration, constants, refusals, zeros, dispatch
# ==================================================================================================

def test_zonal_is_registered_with_its_coefficients_and_citation() -> None:
    model = registry.get_force_model(ZONAL_MODEL)
    assert model.kernel is zonal.zonal_kernel
    assert model.param_names == ("r_eq", "j3", "j4", "j5", "j6")
    assert "Montenbruck" in model.citation
    assert model.validate_bodies is not None and model.validate_coefficients is not None


# EARTH_J2 is printed to 11 significant figures (1.0826266835e-3), so "reproduces it to its printed
# digits" means within one unit of the last place, 1e-13 absolute. -sqrt(5) Cbar20 = 1.08262668355e-3
# is 5.5e-14 above it: EARTH_J2 is the truncation of that value, not its rounding (which would end
# ...836). Within one ulp-of-print either way; a wrong table entry would be off in the 3rd digit.
J2_PRINT_UNIT = 1e-13
# The restated literals are printed to 11 significant figures: half a unit is <= 5e-11 relative.
LITERAL_REL_TOL = 5e-11


def test_egm96_table_reproduces_the_engine_j2_and_the_restated_zonals() -> None:
    derived_j2 = -math.sqrt(5.0) * zonal.EGM96_CBAR_N0[2]
    assert abs(derived_j2 - EARTH_J2) < J2_PRINT_UNIT, derived_j2 - EARTH_J2
    engine = (zonal.EARTH_J3, zonal.EARTH_J4, zonal.EARTH_J5, zonal.EARTH_J6)
    for got, want in zip(engine, (J3, J4, J5, J6)):
        assert abs(got / want - 1.0) < LITERAL_REL_TOL, (got, want)
    assert dict(zonal.EARTH_ZONALS) == {"j3": zonal.EARTH_J3, "j4": zonal.EARTH_J4,
                                        "j5": zonal.EARTH_J5, "j6": zonal.EARTH_J6}


def _constellation(session: Session, n_sats: int = 2) -> Tuple[Simulation, NDArray[np.int64]]:
    sim = scenarios.earth_constellation(session, n_sats=n_sats, n_planes=1)
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    return sim, sats


def test_refusals_leave_the_arena_untouched(db_session_factory: Callable[[], Session]) -> None:
    """Each refusal is a silent-physics failure otherwise: r_eq = 0 zeroes every term, r_eq < 0 flips
    the odd degrees, a NaN poisons the row. A barycentre parent has no shape."""
    sim, sats = _constellation(db_session_factory())
    bit = np.uint64(1) << np.uint64(registry.get_force_model(ZONAL_MODEL).bit)
    for bad in ({"j3": J3}, {"j3": J3, "r_eq": 0.0}, {"j4": J4, "r_eq": -EARTH_R_EQ},
                {"j3": float("nan"), "r_eq": EARTH_R_EQ}, {"r_eq": float("inf")}):
        with pytest.raises(ValueError):
            sim.enable_force_model(ZONAL_MODEL, sats, **bad)
        assert not np.any(sim.force_model_mask[sats] & bit), bad
        assert ZONAL_MODEL not in sim.force_model_params, bad

    # r_eq written by an earlier call is the effective value for a later J_n-only call ...
    sim.enable_force_model(ZONAL_MODEL, sats, r_eq=EARTH_R_EQ)
    sim.enable_force_model(ZONAL_MODEL, sats, j4=J4)
    # ... and overwriting it with 0 while J4 is stored is refused.
    with pytest.raises(ValueError):
        sim.enable_force_model(ZONAL_MODEL, sats, r_eq=0.0)
    assert np.all(sim.force_model_params[ZONAL_MODEL][sats, 0] == EARTH_R_EQ)

    # Barycentre parent: the Earth's head bubble has no such body here, so re-point one satellite.
    sim2, sats2 = _constellation(db_session_factory())
    sim2.parent_indices[sats2[0]] = sim2.name_to_index["Earth Barycenter"]
    with pytest.raises(ValueError, match="barycentre"):
        sim2.enable_force_model(ZONAL_MODEL, sats2, r_eq=EARTH_R_EQ, j3=J3)


def test_zero_coefficients_and_zero_separation_contribute_exactly_nothing() -> None:
    rel = _points()
    rng = np.random.default_rng(3)
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = MU
    parents = np.zeros(n + 1, dtype=np.int32)
    before = rng.normal(size=(n + 1, 3))

    # An all-zero row (and one with r_eq but no J_n) leaves `out` bit-identical.
    for params_row in ([0.0] * 5, [EARTH_R_EQ, 0.0, 0.0, 0.0, 0.0]):
        params = np.tile(params_row, (n + 1, 1))
        out = before.copy()
        zonal.zonal_kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, mu_array, parents, params, out)
        assert np.array_equal(out, before)

    # The root slot (self-parented, zero separation) with full coefficients: exactly 0.0, no warning.
    params = np.tile([EARTH_R_EQ, J3, J4, J5, J6], (n + 1, 1))
    out = before.copy()
    with np.errstate(all="raise"):
        zonal.zonal_kernel(np.array([0], dtype=np.int64), 0.0, state, mu_array, parents, params, out)
    assert np.array_equal(out, before)

    # A zero degree adds nothing: J3-only equals the J3 term of a row whose other degrees are zero,
    # and a mixed row equals the sum of its single-degree rows to rounding (the accumulation order is
    # the only difference).
    single = {d: _kernel_accel(rel, EARTH_R_EQ, {d: j}) for d, j in zip(DEGREES, (J3, J4, J5, J6))}
    both = _kernel_accel(rel, EARTH_R_EQ, {3: J3, 4: J4})
    assert _normalised_error(both, single[3] + single[4], _scale(rel, {3: J3, 4: J4})) < 16 * EPS


def test_cowell_body_with_zonal_stays_on_the_fused_compiled_plan(
    db_session_factory: Callable[[], Session],
) -> None:
    """`kernels.cowell_rk4_step` fuses point_mass_gravity, j2, drag and zonal, so a zonal bit keeps
    the Cowell set on the compiled path (held equivalent to the NumPy one in
    `test_kernel_equivalence.py`), and Cowell + zonal timings are comparable with the other fused
    tiers - with drag on the same body too, since drag is fused. A zonal body that also carries a
    model outside the fused set - thrust here - still sends the whole set down the NumPy path."""
    sim, sats = _constellation(db_session_factory())
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sats)
    sim.enable_force_model(J2_MODEL, sats, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    assert sim._cowell_fused_ok
    sim.enable_force_model(ZONAL_MODEL, sats[:1], r_eq=EARTH_R_EQ, **zonal.EARTH_ZONALS)
    assert sim._cowell_fused_ok
    sim.enable_force_model(DRAG_MODEL, sats[:1], ballistic_coeff=0.01, rho0=1e-12, h0=500.0,
                           scale_height=60.0, r_ref=EARTH_R_EQ, omega=EARTH_OMEGA)
    assert sim._cowell_fused_ok, "zonal + drag are both fused; the body must stay compiled"
    sim.enable_force_model(THRUST_MODEL, sats[:1], thrust_n=0.1, isp_s=300.0, mass_kg=100.0,
                           dry_mass_kg=50.0, dir_s=1.0)
    assert not sim._cowell_fused_ok


# ==================================================================================================
# a. Kernel against the independently written truth field
# ==================================================================================================

# Monomial table of reference.py: degree n -> [(c_k, k)], term c_k z^k r^-(m_k), m_k = k + n + 1.
_MONOMIALS: Dict[int, List[Tuple[float, int]]] = {
    3: [(5 / 2, 3), (-3 / 2, 1)],
    4: [(35 / 8, 4), (-30 / 8, 2), (3 / 8, 0)],
    5: [(63 / 8, 5), (-70 / 8, 3), (15 / 8, 1)],
    6: [(231 / 16, 6), (-315 / 16, 4), (105 / 16, 2), (-5 / 16, 0)],
}


def _field_cross_bound(n: int) -> float:
    """
    Worst-case normalised disagreement, degree n, derived before measuring.

    Truth side: each gradient term is at most |c| (m + k) times the scale A_n / r^(n+2) (|z|, |x| <= r),
    and carries ~(1.5 (m + 2) + 5) eps relative - r's own ~1.5 eps amplified by the power, plus a few
    products - so with no cancellation credit the sum's error is eps sum |c| (m + k)(1.5 m + 8). This is
    the cancellation price of the monomial route: 16371 eps = 3.6e-12 at n = 6.
    Kernel side: the recursion carries ~4 (n + 1) eps relative on values up to P_{n+1}'(1) =
    (n+1)(n+2)/2, i.e. 2 (n+1)^2 (n+2) eps: 784 eps = 1.7e-13 at n = 6.
    Any wrong coefficient, index or sign is an O(1e-1..1) normalised error - ten orders above.
    """
    truth = sum(abs(c) * (k + n + 1 + k) * (1.5 * (k + n + 1) + 8.0) for c, k in _MONOMIALS[n])
    kernel = 2.0 * (n + 1) ** 2 * (n + 2)
    return (truth + kernel) * EPS


def test_kernel_agrees_with_the_truth_field_per_degree_and_summed() -> None:
    """Bounds 1.7e-13 / 4.8e-13 / 1.4e-12 / 3.8e-12 for n = 3..6. Measured (max over 80 points):
    1.5e-14, 1.7e-14, 2.5e-14, 8.9e-14; summed Earth J3..J6 1.6e-14 - 10-50x inside the worst-case
    bounds, as expected of uncorrelated rounding against an aligned-error bound."""
    rel = _points()
    earth = {3: J3, 4: J4, 5: J5, 6: J6}
    for n, j_n in earth.items():
        got = _kernel_accel(rel, EARTH_R_EQ, {n: j_n})
        want = MU * reference.zonal_field(rel, EARTH_R_EQ, {n: j_n})
        err = _normalised_error(got, want, _scale(rel, {n: j_n}))
        assert err < _field_cross_bound(n), (n, err, _field_cross_bound(n))

    got = _kernel_accel(rel, EARTH_R_EQ, earth)
    want = MU * reference.zonal_field(rel, EARTH_R_EQ, earth)
    err = _normalised_error(got, want, _scale(rel, earth))
    assert err < max(_field_cross_bound(n) for n in DEGREES), err


def test_kernel_reduces_to_the_known_values_on_the_axis() -> None:
    """Closed form on the spin axis. With P_m'(+-1) = (+-1)^(m+1) m(m+1)/2 and r_hat = +-z_hat, (Z)
    gives a_z = (mu J_n R^n / r^(n+2)) [(n+1)(n+2)/2 - n(n+1)/2] = (n+1) mu J_n R^n / r^(n+2) at the
    north pole and (-1)^(n+1) times that at the south, with a_x = a_y = 0 exactly. The kernel reaches
    it through ~10 rounded operations on exact small integers: 16 eps is ~2x headroom."""
    r = 8000.0
    for n, j_n in zip(DEGREES, (J3, J4, J5, J6)):
        a = _kernel_accel(np.array([[0.0, 0.0, r], [0.0, 0.0, -r]]), EARTH_R_EQ, {n: j_n})
        expected = (n + 1) * MU * j_n * EARTH_R_EQ ** n / r ** (n + 2)
        assert a[0, 0] == 0.0 and a[0, 1] == 0.0
        assert abs(a[0, 2] / expected - 1.0) < 16 * EPS, (n, a[0, 2], expected)
        assert abs(a[1, 2] / ((-1) ** (n + 1) * expected) - 1.0) < 16 * EPS, (n, a[1, 2])


# ==================================================================================================
# b. The kernel is the gradient of the potential
# ==================================================================================================

def _potential(rel: ArrF, n: int, j_n: float) -> ArrF:
    """Phi_n = mu J_n R^n P_n(z/r) / r^(n+1), sign a = -grad Phi. P_n from numpy's Legendre series
    (Clenshaw) - neither the kernel's recursion nor the truth's monomials."""
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    p_n = legendre.legval(rel[:, 2] / r, [0.0] * n + [1.0])
    return np.asarray(MU * j_n * EARTH_R_EQ ** n * p_n / r ** (n + 1))


def _fd_accel(rel: ArrF, n: int, j_n: float, h_rel: float) -> ArrF:
    a = np.empty_like(rel)
    r = np.linalg.norm(rel, axis=1)
    for axis in range(3):
        plus, minus = rel.copy(), rel.copy()
        plus[:, axis] += h_rel * r
        minus[:, axis] -= h_rel * r
        a[:, axis] = -(_potential(plus, n, j_n) - _potential(minus, n, j_n)) / (plus[:, axis] - minus[:, axis])
    return a


FD_STEP_REL = 3e-4


def _fd_bound(n: int) -> float:
    """
    Richardson (4 D(h) - D(2h))/3 leaves -(h^4/30) f^(5). Phi_n is a solid harmonic,
    Phi_n = (-1)^n A_n/n! d^n/dz^n (1/r), and every N-th directional derivative of 1/r is bounded by
    N!/r^(N+1), so |f^(5)| <= A_n (n+5)!/(n! r^(n+6)): normalised truncation (h/r)^4 (n+5)!/(30 n!).
    Rounding: Phi_n carries eps A_n/r^(n+1) times Q_n = n (Clenshaw) + n(n+1) (s = z/r's 2 eps through
    |P_n'| <= n(n+1)/2) + 1.5 (n+1) (the power of r); a quotient divides 2 such errors by 2h and the
    Richardson weights (4 + 1/2)/3 make it 1.5 eps Q_n (r/h).
    At h/r = 3e-4, n = 6: truncation 1.5e-11, rounding 6.4e-11.
    """
    truncation = FD_STEP_REL ** 4 * math.factorial(n + 5) / (30.0 * math.factorial(n))
    rounding = 1.5 * EPS * (n + n * (n + 1) + 1.5 * (n + 1)) / FD_STEP_REL
    return truncation + rounding


def test_kernel_is_minus_the_gradient_of_the_zonal_potential() -> None:
    """Bounds 2.5e-11 / 3.9e-11 / 5.7e-11 / 8.0e-11 for n = 3..6. Measured (max over 80 points,
    normalised): 6.3e-12, 8.2e-12, 1.5e-11, 2.1e-11 - rounding-dominated, 3-4x inside."""
    rel = _points()
    for n, j_n in zip(DEGREES, (J3, J4, J5, J6)):
        a = _kernel_accel(rel, EARTH_R_EQ, {n: j_n})
        rich = (4.0 * _fd_accel(rel, n, j_n, FD_STEP_REL) - _fd_accel(rel, n, j_n, 2 * FD_STEP_REL)) / 3.0
        err = _normalised_error(a, rich, _scale(rel, {n: j_n}))
        assert err < _fd_bound(n), (n, err, _fd_bound(n))


# ==================================================================================================
# c. Parity, bit for bit
# ==================================================================================================

def test_parity_odd_degrees_antisymmetric_even_symmetric() -> None:
    """P_n(-s) = (-1)^n P_n(s): under z -> -z, (a_x, a_y) pick up (-1)^n and a_z picks up (-1)^(n+1).
    Every operation in the recursion and the projection is sign-symmetric in IEEE arithmetic, so this
    holds exactly. A wrong Legendre index (P_n' where P_{n+1}' belongs) breaks it at O(1)."""
    rel = _points()
    mirrored = rel * np.array([1.0, 1.0, -1.0])
    for n, j_n in zip(DEGREES, (J3, J4, J5, J6)):
        a = _kernel_accel(rel, EARTH_R_EQ, {n: j_n})
        b = _kernel_accel(mirrored, EARTH_R_EQ, {n: j_n})
        sign = (-1.0) ** n
        assert np.array_equal(b[:, :2], sign * a[:, :2]), n
        assert np.array_equal(b[:, 2], -sign * a[:, 2]), n


# ==================================================================================================
# d. Truth: unused means bit-identical; conservation with a massive oblate body
# ==================================================================================================

def test_truth_without_zonal_is_bit_identical(db_session_factory: Callable[[], Session]) -> None:
    """Omitted, `None`, empty, and all-zero coefficients all give today's truth bit for bit - checked
    against a hand-built point-mass + J2 right-hand side integrated here, which is the pre-zonal
    `integrate_nbody` arithmetic."""
    from scipy.integrate import solve_ivp

    sim, _ = _constellation(db_session_factory(), n_sats=3)
    times = np.arange(0.0, 1800.0 + 1.0, 300.0)
    oblateness = {"Earth": (EARTH_J2, EARTH_R_EQ)}
    base = reference.reference_for(sim, times, oblateness=oblateness)

    variants: List[object] = [None, {}, {"Earth": (EARTH_R_EQ, {})},
                              {"Earth": (EARTH_R_EQ, {3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})}]
    for variant in variants:
        again = reference.reference_for(sim, times, oblateness=oblateness, zonal=variant)  # type: ignore[arg-type]
        assert np.array_equal(again.positions, base.positions), variant
        assert np.array_equal(again.velocities, base.velocities), variant

    physical = sim.active_mask & ~sim.is_system
    mu = sim.mu_array[physical].copy()
    n = mu.shape[0]
    j2 = np.where(np.arange(n) == 0, EARTH_J2, 0.0)
    r_eq = np.where(np.arange(n) == 0, EARTH_R_EQ, 0.0)
    assert base.names[0] == "Earth"

    def rhs(_t: float, y: ArrF) -> ArrF:
        r = y[: 3 * n].reshape(n, 3)
        v = y[3 * n:].reshape(n, 3)
        accel = reference.nbody_acceleration(r, mu) + reference.oblateness_acceleration(r, mu, j2, r_eq)
        return np.concatenate([v.ravel(), accel.ravel()])

    y0 = np.concatenate([sim.global_states[physical, :3].ravel(), sim.global_states[physical, 3:].ravel()])
    sol = solve_ivp(rhs, (0.0, float(times[-1])), y0, method="DOP853", t_eval=times,
                    rtol=reference.DEFAULT_RTOL, atol=reference.DEFAULT_ATOL)
    assert np.array_equal(sol.y.T[:, : 3 * n].reshape(-1, n, 3), base.positions)

    with_zonal = reference.reference_for(sim, times, oblateness=oblateness, zonal=EARTH_ZONAL_TRUTH)
    assert not np.array_equal(with_zonal.positions, base.positions)
    assert with_zonal.zonal is not None and with_zonal.zonal[0, 1] == J3


def test_truth_zonal_argument_errors(db_session_factory: Callable[[], Session]) -> None:
    sim, _ = _constellation(db_session_factory())
    t = np.array([0.0, 60.0])
    with pytest.raises(KeyError):
        reference.reference_for(sim, t, zonal={"Earht": (EARTH_R_EQ, {3: J3})})
    with pytest.raises(ValueError, match="oblateness"):     # J2 has exactly one door
        reference.reference_for(sim, t, zonal={"Earth": (EARTH_R_EQ, {2: EARTH_J2})})
    with pytest.raises(ValueError):
        reference.reference_for(sim, t, zonal={"Earth": (0.0, {3: J3})})


# Same budget as test_reference_j2's J2 case: momentum is a linear invariant (RK preserves it; only
# rounding, ~1e-15), energy and h_z are held by DOP853 at rtol 1e-13 to ~1e-12 over 2 days. The zonal
# potential here is exaggerated (J_n = 1e-3, R = 3000 km) so that a field inconsistent with the
# potential would move the energy by ~1e-3 x (Phi_zonal / E) ~ 1e-6, five orders above the tolerance.
CONSERVATION_REL_TOL = 1e-11


def test_massive_zonal_body_conserves_momentum_energy_and_h_z(
    db_session_factory: Callable[[], Session],
) -> None:
    """Measured: momentum 1.8e-15, energy_drift 6.9e-13 (7.1e-13 for J2 alone), h_z 4.4e-13."""
    sim = scenarios.two_body(db_session_factory(), mu_secondary=0.3 * MU, p=11000.0, e=0.2, i=0.9, raan=0.4)
    times = np.arange(0.0, 2.0 * 86400.0 + 1.0, 600.0)
    ref = reference.reference_for(
        sim, times, oblateness={"Primary": (EARTH_J2, EARTH_R_EQ)},
        zonal={"Primary": (3000.0, {3: 1e-3, 4: -1e-3, 5: 1e-3, 6: 1e-3})}, **TRUTH)
    mu = ref.mu
    momentum = np.einsum("i,tij->tj", mu, ref.velocities)
    scale = float(np.sum(mu * np.linalg.norm(ref.velocities[0], axis=1)))
    assert np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / scale < CONSERVATION_REL_TOL

    assert ref.zonal is not None
    a = reference.zonal_acceleration(ref.positions[0], mu, ref.zonal)
    assert np.linalg.norm(a[0]) > 1e-12                           # the reaction is real
    assert np.linalg.norm(mu @ a) < 1e-12 * float(np.sum(mu * np.linalg.norm(a, axis=1)))
    assert ref.energy_drift < CONSERVATION_REL_TOL, ref.energy_drift

    h_z = np.einsum("i,ti->t", mu, ref.positions[:, :, 0] * ref.velocities[:, :, 1]
                    - ref.positions[:, :, 1] * ref.velocities[:, :, 0])
    assert np.max(np.abs(h_z - h_z[0])) / abs(h_z[0]) < CONSERVATION_REL_TOL


# ==================================================================================================
# e. Verification: engine Cowell + pm + j2 + zonal against truth with J2..J6
# ==================================================================================================

ORBIT_RADIUS = scenarios.EARTH_RADIUS + 550.0
PERIOD = 2.0 * math.pi * math.sqrt(ORBIT_RADIUS ** 3 / MU)
CONVERGENCE_STEP_COUNTS = [256, 512, 1024]
ORDER_RATIO_LOW, ORDER_RATIO_HIGH = 12.0, 24.0       # the J2 test's band: 16 +- a model error / 2nd order
# J3..J6 add ~1e-3 of J2 to the force, so RK4 truncation is the J2 case's: 4.2e-4, 2.4e-5, 1.5e-6 km at
# these counts. Leaving zonal out of the engine against the same truth: the degree-4 radial component
# is (n+1) P_n(s) times the scale (9.7e-9 km/s^2 here), orbit-averaging to 5 <P4> = 5 x (-0.152) at
# 53 deg, f = +7.4e-9 km/s^2; Clohessy-Wiltshire for a constant radial f gives |4 pi f / n^2| = 0.078 km
# along-track after one orbit. (Written first as a cruder 0.16 km; either way an order-of-magnitude
# figure, since J3, J5, J6 and the orbit-varying parts are left out.) The assertion asks only for
# 1e-3 km, 700x the 1024-step error - a missing term cannot hide under convergence.
MODEL_GAP_MIN_KM = 1e-3


def _engine_error(session: Session, n_steps: int, truth: Dict[str, ArrF], with_zonal: bool) -> float:
    sim, sats = _constellation(session)
    sim.record_history = False
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sats)
    sim.enable_force_model(J2_MODEL, sats, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    if with_zonal:
        sim.enable_force_model(ZONAL_MODEL, sats, r_eq=EARTH_R_EQ, **zonal.EARTH_ZONALS)
    dt = PERIOD / n_steps
    for _ in range(n_steps):
        sim.step(dt)
    return max(float(np.linalg.norm(sim.global_states[sim.name_to_index[n], :3] - r)) for n, r in truth.items())


def test_cowell_zonal_converges_to_the_zonal_truth_at_fourth_order(
    db_session_factory: Callable[[], Session],
) -> None:
    """Measured: 256/512/1024 steps -> 4.17e-4, 2.45e-5, 1.48e-6 km (ratios 17.0, 16.5), the J2-only
    case's numbers as predicted; without `"zonal"` in the engine, 1024 steps -> 3.9e-2 km, the J3..J6
    signature, half the 0.078 km degree-4 estimate."""
    sim, _ = _constellation(db_session_factory())
    ref = reference.reference_for(sim, np.array([0.0, PERIOD]), oblateness={"Earth": (EARTH_J2, EARTH_R_EQ)},
                                  zonal=EARTH_ZONAL_TRUTH, **TRUTH)
    truth = {n: ref.positions[-1, ref.index_of(n)].copy() for n in ref.names if n.startswith("SAT-")}

    errors = [_engine_error(db_session_factory(), k, truth, True) for k in CONVERGENCE_STEP_COUNTS]
    ratios = [errors[k] / errors[k + 1] for k in range(len(errors) - 1)]
    assert all(ORDER_RATIO_LOW < q < ORDER_RATIO_HIGH for q in ratios), (ratios, errors)

    gap = _engine_error(db_session_factory(), CONVERGENCE_STEP_COUNTS[-1], truth, False)
    assert gap > MODEL_GAP_MIN_KM, gap
    assert gap > 100.0 * errors[-1], (gap, errors[-1])


# ==================================================================================================
# f. Orbit dynamics, measured differentially against a J2-only twin
# ==================================================================================================

SPAN_S = 3.0 * 86400.0
DT_S = 60.0


def _elements(sim: Simulation, names: List[str]) -> Tuple[ArrF, ArrF]:
    """Propagate `sim` over SPAN_S at DT_S and return (times, coe) with coe shaped (n_times, n_sats, 6)."""
    times = np.arange(0.0, SPAN_S + 0.5 * DT_S, DT_S)
    slots = [sim.name_to_index[n] for n in names]
    states = sample_states(sim, slots, times, relative_to=sim.name_to_index["Earth"], max_dt=DT_S)
    flat = states.reshape(-1, 6)
    coe, ok = ReferenceFrames.rv_to_coe(flat[:, :3], flat[:, 3:], MU)
    assert np.all(ok)
    return times, np.asarray(coe).reshape(times.size, len(names), 6)


def _window_means(times: ArrF, series: ArrF, period: float) -> Tuple[float, float, float, float]:
    """Means of `series` over the first and the last `period` of the run, and the mean sample time of
    each window - the time a linear trend's window mean belongs to."""
    first = times < times[0] + period
    last = times > times[-1] - period
    return (float(np.mean(times[first])), float(np.mean(series[first])),
            float(np.mean(times[last])), float(np.mean(series[last])))


# --- J3: long-period eccentricity rate -----------------------------------------------------------
#
# Derivation (first-order averaging, exact in e). U_3 = -mu J3 R^3 r^-4 P3(sin i sin u). With
# dM = r^2/(a^2 eta) dtheta and 1/r = (1 + e cos theta)/p, eta = sqrt(1 - e^2):
#     <r^-4 sin u>   = e sin w / (a^2 eta p^2),     <r^-4 sin^3 u> = (3/4) e sin w / (a^2 eta p^2)
# so <U_3> = (3/2) mu J3 R^3 e sin i (1 - (5/4) sin^2 i) sin w / (a^4 eta^5), and Lagrange's
# de/dt = -(eta / (n a^2 e)) d<U>/dw gives
#     de/dt = -(3/2) n J3 (R/p)^3 (1 - e^2) sin i (1 - (5/4) sin^2 i) cos w.
# That is the brief's formula times (1 - e^2) (4e-4 here). Cited as Kozai 1959 (AJ 64) / Vallado 4e
# Sec. 9.6 - from memory, unverified; the derivation above is what is checked.
# J2 turns w at w' = (3/4) n J2 (R/p)^2 (4 - 5 sin^2 i), 6.9 deg/day here, so over the run
#     Delta e = (K / w') [sin(w0 + w' t2) - sin(w0 + w' t1)],     K = the rate above at cos w = 1.
# The run is centred on w = 0 (and on 180 deg for the sign flip), where cos w is flat: an error dw in
# the initial *mean* w (osculating vs mean differ by ~J2 (R/p)^2 / e = 0.04 rad) moves the prediction
# by only dw^2/2 ~ 1e-3.
#
# Budget: (i) osculating a, e, i used where the theory wants mean: K ~ a^-4.5 and dK/di, O(J2 (R/p)^2)
# = 8.6e-4 times ~4.5 -> 0.4 %; (ii) J2 x J3 cross terms (J3 moving the mean elements J2's short-period
# terms depend on, and J2 moving the ones J3's rate depends on), second order: O(J2 (R/p)^2) relative,
# coefficients of a few -> <= 0.5 %; (iii) the omega-offset above, 0.1 %; (iv) averaging: the
# differential short-period signal is J3 (R/a)^3 ~ 2e-6 and a one-period window leaves ~1e-3 of it,
# 2e-9 against a 2.4e-4 signal, 1e-5 relative; RK4 truncation is common to both twins. Total ~1 %;
# tolerance 2 %.
ZN_A_KM, ZN_E, ZN_I_DEG = 7000.0, 0.02, 40.0
J3_RATE_REL_TOL = 0.02


def _j3_prediction(w0: float, t1: float, t2: float) -> float:
    p = ZN_A_KM * (1.0 - ZN_E ** 2)
    n = math.sqrt(MU / ZN_A_KM ** 3)
    s = math.sin(math.radians(ZN_I_DEG))
    k = -1.5 * n * J3 * (EARTH_R_EQ / p) ** 3 * (1.0 - ZN_E ** 2) * s * (1.0 - 1.25 * s * s)
    w_dot = _w_dot_j2()
    return k / w_dot * (math.sin(w0 + w_dot * t2) - math.sin(w0 + w_dot * t1))


def _w_dot_j2() -> float:
    p = ZN_A_KM * (1.0 - ZN_E ** 2)
    n = math.sqrt(MU / ZN_A_KM ** 3)
    s2 = math.sin(math.radians(ZN_I_DEG)) ** 2
    return 0.75 * n * EARTH_J2 * (EARTH_R_EQ / p) ** 2 * (4.0 - 5.0 * s2)


def test_j3_drives_eccentricity_at_the_first_order_long_period_rate(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Two perigees 180 deg apart, each with a J2-only control (copy 0) and a J2 + J3 twin (copy 1), all
    in one arena. Predicted Delta e between the first and last orbit-window: +2.4288e-4 (w ~ 0) and
    -2.4288e-4 (w ~ 180 deg). Measured: +2.4390e-4 and -2.4389e-4, both +0.42 %.

    The +0.42 % is budget item (i), and can be accounted for after the fact: the osculating seed at
    u = w ~ -10 deg sits ~6.0 km above the mean orbit (J2's short-period a, (3/2)(J2 R^2/a)[(2/3)(1 -
    (3/2) sin^2 i) + sin^2 i cos 2u]), and K ~ a^-4.5 then reads +0.39 % high with the mean a - leaving
    0.03 %. The control's own osculating e swings 2.0e-3 peak to peak over the run; the differenced
    series 2.5e-4, which is the signal.
    """
    w_centre = -_w_dot_j2() * SPAN_S / 2.0
    sim = scenarios.zonal_twins(
        db_session_factory(), arg_pe_deg=(math.degrees(w_centre), 180.0 + math.degrees(w_centre)),
        semi_major_axis_km=ZN_A_KM, eccentricity=ZN_E, inclination_deg=ZN_I_DEG)
    names = [scenarios.zonal_twin_name(o, c) for o in (0, 1) for c in (0, 1)]
    twins = [sim.name_to_index[scenarios.zonal_twin_name(o, 1)] for o in (0, 1)]
    sim.enable_force_model(ZONAL_MODEL, twins, r_eq=EARTH_R_EQ, j3=J3)
    times, coe = _elements(sim, names)

    period = 2.0 * math.pi * math.sqrt(ZN_A_KM ** 3 / MU)
    measured = []
    for orbit, w0 in ((0, w_centre), (1, w_centre + math.pi)):
        de = coe[:, 2 * orbit + 1, COEIndex.E] - coe[:, 2 * orbit, COEIndex.E]
        t1, m1, t2, m2 = _window_means(times, de, period)
        predicted = _j3_prediction(w0, t1, t2)
        measured.append((m2 - m1, predicted))
        assert abs((m2 - m1) / predicted - 1.0) < J3_RATE_REL_TOL, (orbit, m2 - m1, predicted)

    # The sign follows cos(w): the two perigees drive e in opposite directions.
    assert measured[0][0] > 0.0 > measured[1][0], measured


# --- J4: secular node rate -----------------------------------------------------------------------
#
# Derivation. U_4 = -mu J4 R^4 r^-5 P4(sin i sin u); its secular (w-independent) average uses
# <r^-5> = (1 + 3e^2/2)/(a^2 eta p^3), <sin^2 u> -> 1/2, <sin^4 u> -> 3/8 (the cos 2w parts are
# long-period), giving <U_4> = -(3/8) mu J4 R^4 (1 + 3e^2/2) [(35/8) sin^4 i - 5 sin^2 i + 1] /
# (a^2 eta p^3). Lagrange's dW/dt = (1/(n a^2 eta sin i)) d<U>/di then gives
#     dW/dt = (15/16) n J4 (R/p)^4 (1 + (3/2) e^2) cos i (4 - 7 sin^2 i).
# Merson 1961 / Kozai 1959 carry this term; cited from memory, unverified - the derivation is checked.
# Circular (e = 0) at 550 km, i = 30 deg: -2.33e-9 rad/s, -6.0e-4 rad over the 3-day run.
#
# Budget: J2^2 second-order terms are as large as J4 (J2^2 (R/p)^4 ~ 8e-7 vs J4 (R/p)^4 ~ 1.2e-6) but
# common to both twins. What survives: (i) osculating-vs-mean a in n (R/p)^4 ~ a^-5.5, 5.5 x
# J2 (R/p)^2 ~ 0.5 %; (ii) J2 x J4 cross terms - e.g. J4 shifts the twin's mean a by ~J4 (R/a)^4 a, which
# moves J2's own node rate (-1.3e-6 rad/s) by 3.5 J4 (R/a)^4 of itself, 2.3e-3 of the J4 rate per unit
# coefficient: ~0.5 %; (iii) averaging, as for J3, ~1e-5. Total ~1 %; tolerance 2 %.
NODE_A_KM, NODE_I_DEG = 6928.137, 30.0
J4_RATE_REL_TOL = 0.02


def test_j4_adds_the_first_order_secular_node_rate(db_session_factory: Callable[[], Session]) -> None:
    """Predicted Delta RAAN between the first and last orbit-window: -5.898e-4 rad. Measured
    -5.956e-4 rad (+0.97 %), inside the 2 % budget but not by much, so accounted for: the
    osculating-circular seed at u = 0 sits 6.4 km above the mean orbit by the same short-period formula
    as the J3 case, and n (R/p)^4 ~ a^-5.5 reads +0.50 % high with the mean a; osculating-vs-mean i
    (Delta i ~ (3/8) J2 (R/a)^2 sin 2i = 3.0e-4 rad against d ln(rate)/di = -3.3) is ~0.1 %. The
    remaining ~0.4 % is item (ii), the J2 x J4 cross terms, at the size estimated. For scale: J2 alone
    turns the control's node by -0.34 rad over the run, 570x the measured difference."""
    sim = scenarios.zonal_twins(db_session_factory(), semi_major_axis_km=NODE_A_KM, eccentricity=0.0,
                                inclination_deg=NODE_I_DEG)
    names = [scenarios.zonal_twin_name(0, c) for c in (0, 1)]
    sim.enable_force_model(ZONAL_MODEL, sim.name_to_index[names[1]], r_eq=EARTH_R_EQ, j4=J4)
    times, coe = _elements(sim, names)

    raan = np.unwrap(coe[:, :, COEIndex.RAAN], axis=0)
    period = 2.0 * math.pi * math.sqrt(NODE_A_KM ** 3 / MU)
    t1, m1, t2, m2 = _window_means(times, raan[:, 1] - raan[:, 0], period)

    n = math.sqrt(MU / NODE_A_KM ** 3)
    ci, s2 = math.cos(math.radians(NODE_I_DEG)), math.sin(math.radians(NODE_I_DEG)) ** 2
    rate = 15.0 / 16.0 * n * J4 * (EARTH_R_EQ / NODE_A_KM) ** 4 * ci * (4.0 - 7.0 * s2)
    predicted = rate * (t2 - t1)
    assert abs((m2 - m1) / predicted - 1.0) < J4_RATE_REL_TOL, (m2 - m1, predicted)


def test_run_sweep_forwards_zonal_to_both_truths(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring: `run_sweep(zonal=)` must reach the endpoint truth *and* the access truth. The headline
    that exercises it lives in `benchmarks/zonal_sweep.py`, not the suite, so in review dropping
    `zonal=` from the access-truth call passed all 43 zonal/sweep/access tests - every contact-window
    metric would then be scored against a J2-only truth while the km metric used J2..J6, silently."""
    from orbital_engine import access, sweep

    seen: List[object] = []
    real = sweep.reference_for

    def spy(*args: object, **kwargs: object) -> reference.ReferenceTrajectory:
        seen.append(kwargs.get("zonal"))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sweep, "reference_for", spy)
    spec = access.AccessSpec(
        stations=[access.GroundStation("equator", 0.0, 0.0)], central_body="Earth",
        omega=7.2921159e-5, body_radius_km=scenarios.EARTH_RADIUS, sample_dt_s=60.0,
    )
    sweep.run_sweep(
        lambda: _constellation(db_session_factory())[0],
        [sweep.ModelConfig("kepler", PropagatorType.KEPLERIAN, 60.0)],
        horizon_s=600.0, timing_batches=1, timing_warmup=0,
        zonal=EARTH_ZONAL_TRUTH, access=spec,
    )
    assert seen == [EARTH_ZONAL_TRUTH, EARTH_ZONAL_TRUTH], seen
