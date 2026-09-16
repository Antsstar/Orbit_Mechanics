"""
Validation of the RSW (radial, along-track, cross-track) frame transforms in `ReferenceFrames`.

    R = r / |r|        W = (r x v) / |r x v|        S = W x R

Definition: Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Sec. 3.3 (RSW).

What each check can and cannot see
----------------------------------
Orthonormality, determinant and round trip are *structural*. Any proper rotation passes them, including a wrong
one: a basis with S and W both negated is still a rotation, and a basis stored as columns instead of rows still
inverts itself exactly. The checks that pin the frame to the physics are the analytic velocity components and the
circular-equatorial axes. The negative control at the bottom of this file runs every check against deliberately
broken transforms and asserts which check catches which error, so that claim is tested rather than stated.

Analytic velocity components
----------------------------
Curtis, *Orbital Mechanics for Engineering Students*, 3rd ed., Sec. 2.5, Eqs. 2.48-2.49:

    v_S = (mu / h) (1 + e cos theta)        v_R = (mu / h) e sin theta        v_W = 0

Self-contained derivation, so the check does not rest on the equation numbers: in the RSW frame
v = r_dot R + r theta_dot S. With h = r^2 theta_dot and the orbit equation r = p / (1 + e cos theta), p = h^2 / mu,

    v_S = r theta_dot = h / r = (h / p)(1 + e cos theta) = (mu / h)(1 + e cos theta)
    v_R = r_dot = p e sin theta theta_dot / (1 + e cos theta)^2 = (h / p) e sin theta = (mu / h) e sin theta

Valid for every conic. States are built with `coe_to_rv`, which shares no code with the RSW transforms.

Expected error magnitudes
-------------------------
Each check is a short chain of correctly rounded operations on unit-scale quantities - a norm, a division, a cross
product, a three-term dot product - each contributing ~eps relative. The derived expectation is therefore a few eps
(~1e-15), and the analytic chain through `coe_to_rv`'s 3-1-3 rotation adds at most ~20 eps (~4e-15). Measured on
this suite's populations (97 conics + 6 near-radial states; numpy 2.4.6, x86-64):

    orthonormality ||B B^T - I||_max    4.4e-16   (2.0 eps)
    handedness     |det B - 1|          4.4e-16   (2.0 eps)
    round trip     relative             5.1e-16   (2.3 eps)
    v_R, v_S, v_W  relative to |v|      4.3e-16   (2.0 eps)
    circular equatorial axes            exactly 0

The budgets below sit two orders above the derived expectation. Every algebra error the negative control models
produces an O(1) error, and deleting the Gram-Schmidt step from `RSW_basis` raised orthonormality and round-trip
error to 1.9e-8 on this suite's sin(alpha) = 1e-9 states; both fail these budgets by many orders.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from orbital_engine.frames import RSW_RECTILINEAR_TOL, ReferenceFrames as rf

MU_EARTH = 398600.4418        # km^3/s^2
MU_SUN = 1.32712440018e11     # km^3/s^2
AU_KM = 1.495978707e8         # km

EPS = float(np.finfo(np.float64).eps)

ORTHONORMAL_TOL = 1e-13         # ||B B^T - I||_max; expected ~3 eps
HANDEDNESS_TOL = 1e-13          # |det B - 1|;        expected ~2 eps
ROUNDTRIP_TOL = 1e-13           # relative;           expected ~3 eps
VELOCITY_COMPONENT_TOL = 1e-13  # relative to |v|;    expected <= ~20 eps
AXES_TOL = 0.0                  # exact - see test_circular_equatorial_axes_are_exact

# Every mutant in the negative control is a sign flip, axis swap or transposition. On the anchor orbit each one's
# error is derived by hand as sqrt(2), 2 or 2*sqrt(2) (see MUTANTS). The floor sits below the smallest of those.
NEGATIVE_CONTROL_FLOOR = 1.0

# A fixed skew direction, not parallel to any axis, so no cross-product component is trivially zero.
_U = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)

Array = NDArray[np.float64]


# ==================================================================================================
# Test populations
# ==================================================================================================

ECCENTRICITIES = (0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.5, 4.0)
ORBITS_PER_ECCENTRICITY = 12
NEAR_RADIAL_SIN_ALPHA = (1e-3, 1e-6, 1e-9)


def _orbit_spread() -> tuple[Array, Array, Array, Array]:
    """
    (coe, mu, r, v) for a deterministic spread of conics.

    Every eccentricity class the engine represents; inclinations including exactly equatorial prograde and
    retrograde (W = +z and -z); three length scales from LEO to 1 AU; true anomaly kept 10% inside the asymptote
    for e >= 1, beyond which r = p / (1 + e cos theta) goes negative.

    Row 0 is the **anchor** for the negative control: circular, equatorial, theta = pi/2. There r = a y_hat and
    v = -|v| x_hat, so R = +y, S = -x, W = +z, and every mutant's error can be worked out by hand.
    """
    rng = np.random.default_rng(20260916)
    rows = [(7000.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, MU_EARTH)]

    for e in ECCENTRICITIES:
        theta_max = np.pi if e <= 1.0 else float(np.arccos(-1.0 / e))
        for k in range(ORBITS_PER_ECCENTRICITY):
            p_k, mu_k = ((7000.0, MU_EARTH), (42164.0, MU_EARTH), (AU_KM, MU_SUN))[k % 3]
            inc = (0.0, np.pi)[k] if k < 2 else rng.uniform(0.0, np.pi)
            rows.append((p_k, e, inc, rng.uniform(0.0, 2 * np.pi), rng.uniform(0.0, 2 * np.pi),
                         rng.uniform(-0.9, 0.9) * theta_max, mu_k))

    table = np.array(rows, dtype=np.float64)
    coe, mu = table[:, :6], table[:, 6]
    r, v, ok = rf.coe_to_rv(coe, mu)
    assert ok.all(), "coe_to_rv rejected part of the orbit spread"
    return coe, mu, r, v


def _near_radial_state(sin_alpha: float, r_mag: float, v_mag: float) -> tuple[Array, Array, Array]:
    """
    A state whose velocity makes a controlled angle alpha with the radius vector, plus its exact W direction.

    v = |v| (cos(alpha) u + sin(alpha) n) with n perpendicular to u, so r x v is parallel to u x n exactly.
    """
    n = np.cross(_U, np.array([0.0, 0.0, 1.0]))
    n /= np.linalg.norm(n)
    r = r_mag * _U
    v = v_mag * (np.sqrt(1.0 - sin_alpha ** 2) * _U + sin_alpha * n)
    return r, v, np.cross(_U, n)


def _all_states() -> tuple[Array, Array]:
    """The orbit spread plus near-radial states at LEO and heliocentric scale - the latter exercise Gram-Schmidt."""
    _, _, r, v = _orbit_spread()
    near = [_near_radial_state(s, rm, vm) for s in NEAR_RADIAL_SIN_ALPHA for rm, vm in ((7000.0, 7.5), (AU_KM, 29.78))]
    return (np.concatenate([r, np.array([state[0] for state in near])]),
            np.concatenate([v, np.array([state[1] for state in near])]))


# ==================================================================================================
# Error metrics. Each looks up the transforms at call time, so the negative control can substitute broken ones.
# ==================================================================================================

def orthonormality_error() -> float:
    r, v = _all_states()
    basis, valid = rf.RSW_basis(r, v)
    assert valid.all(), "a state in the spread was flagged rectilinear"
    return float(np.max(np.abs(basis @ np.swapaxes(basis, -1, -2) - np.eye(3))))


def handedness_error() -> float:
    r, v = _all_states()
    basis, _ = rf.RSW_basis(r, v)
    return float(np.max(np.abs(np.linalg.det(basis) - 1.0)))


def roundtrip_error() -> float:
    """Cartesian -> RSW -> Cartesian on the state's own r, v, and an arbitrary acceleration-scale vector."""
    r, v = _all_states()
    accel = np.random.default_rng(7).standard_normal(r.shape) * 1e-6
    ref_r, ref_v = np.concatenate([r, r, r]), np.concatenate([v, v, v])
    x = np.concatenate([r, v, accel])

    x_rsw, _ = rf.cart_to_RSW(ref_r, ref_v, x)
    x_back, _ = rf.RSW_to_cart(ref_r, ref_v, x_rsw)
    return float(np.max(np.linalg.norm(x_back - x, axis=-1) / np.linalg.norm(x, axis=-1)))


def velocity_component_error() -> float:
    """Distance between v expressed in its own RSW frame and Curtis's closed form, relative to |v|."""
    coe, mu, r, v = _orbit_spread()
    p, e, theta = coe[:, 0], coe[:, 1], coe[:, 5]
    mu_over_h = mu / np.sqrt(mu * p)
    expected = np.stack([mu_over_h * e * np.sin(theta),
                         mu_over_h * (1.0 + e * np.cos(theta)),
                         np.zeros_like(theta)], axis=-1)

    v_rsw, valid = rf.cart_to_RSW(r, v, v)
    assert valid.all()
    return float(np.max(np.linalg.norm(v_rsw - expected, axis=-1) / np.linalg.norm(v, axis=-1)))


def axes_error() -> float:
    """Frobenius distance of the basis from the identity for circular equatorial orbits at theta = 0."""
    worst = 0.0
    for a, mu in ((7000.0, MU_EARTH), (42164.0, MU_EARTH), (AU_KM, MU_SUN)):
        basis, _ = rf.RSW_basis(np.array([a, 0.0, 0.0]), np.array([0.0, np.sqrt(mu / a), 0.0]))
        worst = max(worst, float(np.linalg.norm(basis - np.eye(3))))
    return worst


# ==================================================================================================
# Positive checks
# ==================================================================================================

def test_basis_is_orthonormal_across_orbit_spread():
    err = orthonormality_error()
    assert err <= ORTHONORMAL_TOL, f"||B B^T - I||_max = {err:.3e} ({err / EPS:.1f} eps); expected a few eps"


def test_basis_is_right_handed_across_orbit_spread():
    err = handedness_error()
    assert err <= HANDEDNESS_TOL, f"|det B - 1| = {err:.3e} ({err / EPS:.1f} eps); expected a few eps"


def test_cartesian_rsw_roundtrip_recovers_input():
    err = roundtrip_error()
    assert err <= ROUNDTRIP_TOL, f"round-trip relative error {err:.3e} ({err / EPS:.1f} eps); expected a few eps"


def test_velocity_components_match_curtis_closed_form():
    """The independent analytic check: v in its own RSW frame is [(mu/h) e sin th, (mu/h)(1 + e cos th), 0]."""
    err = velocity_component_error()
    assert err <= VELOCITY_COMPONENT_TOL, f"relative error {err:.3e} ({err / EPS:.1f} eps); expected <= ~20 eps"


def test_circular_equatorial_axes_are_exact():
    """
    r = (a, 0, 0), v = (0, v_c, 0) must give R = x_hat, S = y_hat, W = z_hat with no rounding at all.

    Why exact rather than to tolerance: |r| = sqrt(a*a) = a exactly (correctly rounded sqrt of a correctly rounded
    square returns |a| in binary64), so R = a / a = 1. r x v = (0, 0, a v_c) and its norm is a v_c by the same
    argument, so W = z_hat. R.W = 0 exactly, so Gram-Schmidt subtracts nothing, and W x R = y_hat. Checked at three
    scales because the argument is scale-free.
    """
    assert axes_error() == AXES_TOL


def test_near_radial_w_direction_within_rounding_bound():
    """
    Close to the rectilinear threshold W is still defined, but only as well as rounding allows.

    Each component of the computed r x v is in error by at most ~eps |r||v|, so W's direction error is bounded by
    sqrt(3) eps / sin(alpha) to first order; asserted as 2 eps / sin(alpha). This is the bound that sets
    RSW_RECTILINEAR_TOL, so it is checked rather than assumed.
    """
    for sin_alpha in NEAR_RADIAL_SIN_ALPHA:
        for r_mag, v_mag in ((7000.0, 7.5), (AU_KM, 29.78)):
            r, v, w_exact = _near_radial_state(sin_alpha, r_mag, v_mag)
            basis, valid = rf.RSW_basis(r, v)
            assert valid[0]
            w_err = float(np.linalg.norm(basis[2] - w_exact))
            assert w_err < 2.0 * EPS / sin_alpha, f"sin(alpha)={sin_alpha:.0e}: W off by {w_err:.3e}"


# ==================================================================================================
# Rectilinear reference states: flagged by the mask, never NaN
# ==================================================================================================

@pytest.mark.parametrize(
    "r, v",
    [
        pytest.param(7000.0 * _U, 7.5 * _U, id="radial-outbound-leo"),
        pytest.param(7000.0 * _U, -7.5 * _U, id="radial-inbound-leo"),
        pytest.param(AU_KM * _U, 29.78 * _U, id="radial-heliocentric"),
        pytest.param(7000.0 * _U, np.zeros(3), id="zero-velocity"),
        pytest.param(np.zeros(3), np.array([0.0, 7.5, 0.0]), id="zero-position"),
    ],
)
def test_rectilinear_reference_is_flagged_without_nan(r: Array, v: Array):
    vec = np.array([1e-6, -2e-6, 3e-6])

    # Raise on any floating-point exception, so a 0/0 evaluated and then masked away still fails the test.
    with np.errstate(all="raise"):
        basis, valid_basis = rf.RSW_basis(r, v)
        x_rsw, valid_fwd = rf.cart_to_RSW(r, v, vec)
        x_cart, valid_inv = rf.RSW_to_cart(r, v, vec)

    for mask in (valid_basis, valid_fwd, valid_inv):
        assert mask.shape == (1,) and not mask[0]
    for out in (basis, x_rsw, x_cart):
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0), "an undefined frame must transform to zero, not to a plausible-looking vector"


def test_rectilinear_classification_is_scale_invariant():
    """
    Rounding noise in |r x v| scales with |r||v|, so an absolute threshold misclassifies at heliocentric scale.

    Precondition: this radial state's |h| is pure rounding noise yet exceeds 1e-9 km^2/s, the absolute threshold
    rv_to_coe uses. Without that the case would pass vacuously.
    """
    r, v = AU_KM * _U, 29.78 * _U
    assert np.linalg.norm(np.cross(r, v)) > 1e-9, "precondition lost: |h| noise no longer exceeds 1e-9"

    _, valid = rf.RSW_basis(r, v)
    assert not valid[0]


@pytest.mark.parametrize("r_mag, v_mag", [(7000.0, 7.5), (AU_KM, 29.78)], ids=["leo", "heliocentric"])
def test_threshold_separates_near_radial_from_rectilinear(r_mag: float, v_mag: float):
    """A factor of 100 either side of RSW_RECTILINEAR_TOL is classified correctly at both scales."""
    r_in, v_in, _ = _near_radial_state(100.0 * RSW_RECTILINEAR_TOL, r_mag, v_mag)
    r_out, v_out, _ = _near_radial_state(RSW_RECTILINEAR_TOL / 100.0, r_mag, v_mag)
    assert rf.RSW_basis(r_in, v_in)[1][0]
    assert not rf.RSW_basis(r_out, v_out)[1][0]


def test_mixed_batch_isolates_invalid_rows():
    """Invalid rows are zero and finite; valid rows are unaffected by their invalid neighbours; NaN is flagged."""
    r = np.array([[7000.0, 0.0, 0.0], 7000.0 * _U, [0.0, 8000.0, 100.0],
                  [0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [7000.0, 0.0, 0.0]])
    v = np.array([[0.0, 7.5, 0.0], 7.5 * _U, [-6.0, 0.0, 3.0],
                  [0.0, 7.5, 0.0], [0.0, 7.5, 0.0], [0.0, 0.0, 0.0]])
    thrust_rsw = np.array([0.0, 1e-6, 0.0])

    basis, valid = rf.RSW_basis(r, v)
    accel, valid_acc = rf.RSW_to_cart(r, v, thrust_rsw)

    np.testing.assert_array_equal(valid, [True, False, True, False, False, False])
    np.testing.assert_array_equal(valid_acc, valid)
    assert np.all(np.isfinite(basis)) and np.all(np.isfinite(accel))
    assert np.all(basis[~valid] == 0.0) and np.all(accel[~valid] == 0.0)

    for row in np.flatnonzero(valid):
        alone, _ = rf.RSW_basis(r[row], v[row])
        np.testing.assert_allclose(basis[row], alone, rtol=0.0, atol=ORTHONORMAL_TOL)


# ==================================================================================================
# Shape and buffer contracts
# ==================================================================================================

def test_single_state_shapes():
    r, v = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 1.0])
    basis, valid = rf.RSW_basis(r, v)
    x_rsw, _ = rf.cart_to_RSW(r, v, v)
    x, _ = rf.RSW_to_cart(r, v, x_rsw)
    assert basis.shape == (3, 3) and x_rsw.shape == (3,) and x.shape == (3,) and valid.shape == (1,)


def test_prograde_direction_broadcasts_over_many_states():
    """A single (3,) RSW direction applies to every reference state: +S is the basis S row and points along motion."""
    _, _, r, v = _orbit_spread()
    prograde, valid = rf.RSW_to_cart(r, v, np.array([0.0, 1.0, 0.0]))
    basis, _ = rf.RSW_basis(r, v)

    assert prograde.shape == r.shape and valid.all()
    np.testing.assert_array_equal(prograde, basis[:, 1, :])
    # v . S = v_S = h / r > 0 for every orbit: "prograde" never opposes the motion.
    assert np.all(np.einsum("ij,ij->i", prograde, v) > 0.0)


def test_many_vectors_in_one_frame():
    r, v = np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 1.0])
    vecs = np.random.default_rng(3).standard_normal((5, 3))
    x_rsw, valid = rf.cart_to_RSW(r, v, vecs)
    basis, _ = rf.RSW_basis(r, v)
    assert x_rsw.shape == (5, 3) and valid.shape == (1,)
    np.testing.assert_allclose(x_rsw, vecs @ basis.T, rtol=0.0, atol=ROUNDTRIP_TOL)


def test_out_buffers_are_written_in_place():
    _, _, r, v = _orbit_spread()
    vec = np.random.default_rng(11).standard_normal(r.shape)

    out_basis = np.empty((r.shape[0], 3, 3))
    basis, _ = rf.RSW_basis(r, v, out_basis=out_basis)
    assert basis is out_basis

    single_basis = np.empty((3, 3))
    b0, _ = rf.RSW_basis(r[0], v[0], out_basis=single_basis)
    assert np.shares_memory(b0, single_basis)
    np.testing.assert_allclose(single_basis, out_basis[0], rtol=0.0, atol=ORTHONORMAL_TOL)

    for transform in (rf.cart_to_RSW, rf.RSW_to_cart):
        out = np.empty_like(r)
        written, _ = transform(r, v, vec, out=out)
        fresh, _ = transform(r, v, vec)
        assert written is out
        np.testing.assert_array_equal(out, fresh)

        # Documented: `out` may alias the input, rotating in place. Overlap must not corrupt later rows.
        in_place = vec.copy()
        transform(r, v, in_place, out=in_place)
        np.testing.assert_array_equal(in_place, fresh)


@pytest.mark.parametrize(
    "r, v, kwargs",
    [
        pytest.param(np.ones((4, 2)), np.ones((4, 2)), {}, id="2-vectors"),
        pytest.param(np.ones((2, 4, 3)), np.ones((2, 4, 3)), {}, id="3d-stack"),
        pytest.param(np.ones((4, 3)), np.ones((4, 3)), {"out_basis": np.empty((3, 3, 3))}, id="out-basis-shape"),
    ],
)
def test_malformed_input_raises(r: Array, v: Array, kwargs: dict[str, Array]):
    with pytest.raises(ValueError):
        rf.RSW_basis(r, v, **kwargs)


# ==================================================================================================
# Negative control: the checks above must catch sign, axis-order and transposition errors
# ==================================================================================================

_TRUE_BASIS = rf.RSW_basis
_TRUE_TO_CART = rf.RSW_to_cart


def _negate_s(b: Array) -> Array:
    out = b.copy()
    out[..., 1, :] *= -1.0          # S = R x W instead of W x R
    return out


def _swap_r_s(b: Array) -> Array:
    return b[..., [1, 0, 2], :].copy()


def _negate_s_and_w(b: Array) -> Array:
    out = b.copy()                  # W = v x r, with S still formed as W x R: a *proper* rotation, det +1
    out[..., 1:, :] *= -1.0
    return out


def _transpose(b: Array) -> Array:
    return np.swapaxes(b, -1, -2).copy()   # basis stored as columns, both transforms self-consistent


CHECKS: dict[str, tuple[Callable[[], float], float]] = {
    "orthonormality": (orthonormality_error, ORTHONORMAL_TOL),
    "handedness": (handedness_error, HANDEDNESS_TOL),
    "roundtrip": (roundtrip_error, ROUNDTRIP_TOL),
    "velocity": (velocity_component_error, VELOCITY_COMPONENT_TOL),
    "axes": (axes_error, AXES_TOL),
}

# mutant id -> (basis mutation or None, one-sided transposition in cart_to_RSW?, checks expected to catch it)
#
# Derived magnitudes on the anchor row (R = +y, S = -x, W = +z; v = |v| S; r = a R). Each check reports its worst
# row, so these are lower bounds; measured values in brackets.
#   negate-S          v_rsw = (0, -|v|, 0)          velocity 2 [2.00]      det -1 -> 2 [2.00]   axes 2 [2.00]
#   swap-R-S          v_rsw = (|v|, 0, 0)           velocity sqrt2 [2.00]  det -1 -> 2 [2.00]   axes 2 [2.00]
#   negate-S-and-W    v_rsw = (0, -|v|, 0)          velocity 2 [2.00]      det +1               axes 2 sqrt2 [2.83]
#   transposed-basis  B^T v = (0, -|v|, 0)          velocity 2 [2.00]      det +1               axes 0 (B = I at theta = 0)
#   one-sided-T       B^T B^T r = -r                velocity 2 [2.00]      roundtrip 2 [2.00]   basis untouched
# (swap-R-S error is sqrt2 |v_S - v_R| / |v|, which approaches 2 on hyperbolic rows near v_R = -v_S.)
# Every mutant is orthogonal, so orthonormality sees none of them - it guards the numerics, not the convention.
MUTANTS = {
    "negate-S": (_negate_s, False, {"handedness", "velocity", "axes"}),
    "swap-R-S": (_swap_r_s, False, {"handedness", "velocity", "axes"}),
    "negate-S-and-W": (_negate_s_and_w, False, {"velocity", "axes"}),
    "transposed-basis": (_transpose, False, {"velocity"}),
    "one-sided-transpose": (None, True, {"roundtrip", "velocity"}),
}


@pytest.mark.parametrize("mutant_id", list(MUTANTS))
def test_negative_control_checks_catch_convention_errors(mutant_id: str, monkeypatch: pytest.MonkeyPatch):
    mutate, one_sided, expected_caught = MUTANTS[mutant_id]

    if mutate is not None:
        def broken_basis(r: Array, v: Array, **kwargs: Any) -> tuple[Array, NDArray[np.bool_]]:
            basis, valid = _TRUE_BASIS(r, v, **kwargs)
            return mutate(basis), valid
        monkeypatch.setattr(rf, "RSW_basis", staticmethod(broken_basis))

    if one_sided:
        # An index-order slip in cart_to_RSW alone: it applies B^T while RSW_to_cart still applies B^T as well.
        monkeypatch.setattr(rf, "cart_to_RSW", staticmethod(_TRUE_TO_CART))

    for name, (check, budget) in CHECKS.items():
        err = check()
        if name in expected_caught:
            assert err > NEGATIVE_CONTROL_FLOOR, f"{mutant_id}: '{name}' should catch it but measured {err:.3e}"
        else:
            assert err <= budget, f"{mutant_id}: '{name}' unexpectedly moved to {err:.3e}"

    # The mutant must be caught by at least one analytic check, not only by structure.
    assert expected_caught & {"velocity", "axes"}
