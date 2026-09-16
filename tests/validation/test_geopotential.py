"""
Validation of the `"j2"` force model (`geopotential.j2_kernel`).

Every tolerance below is a named constant whose comment derives its number *before* measurement. The
point is the class of bug `CLAUDE.md` warns about - a transposed coefficient that yields a plausible
orbit - so each check is chosen to be sensitive to a coefficient error, and the negative control at
the bottom proves it by running the same checks against a kernel with the z-term's 5 replaced by 3.

Notation: A = mu J2 R^2, r = |rel|, s = z/r. The kernel's acceleration has magnitude
(3/2)(A/r^4) sqrt(1 - 2 s^2 + 5 s^4), which ranges over [0.894, 2] x (3/2) A / r^4 - its minimum
over latitude is at s^2 = 1/5. That lower bound, 1.34 A/r^4, is the denominator of every relative
bound below.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import geopotential, registry, scenarios
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL

EPS = float(np.finfo(np.float64).eps)

# --------------------------------------------------------------------------------------------------
# Tolerances, derived.
# --------------------------------------------------------------------------------------------------

# Closed-form points use mu = 1e6, J2 = 1e-3, R = 1000, r = 2000, so A/r^4 = 1e12/1.6e13 = 1/16 exactly
# and the expected values -9.375e-5 and 1.875e-4 are exact binary fractions scaled by 1e-3. The kernel
# reaches them through ~10 rounded multiplications and a sqrt, and the operands (1/r^2 = 2.5e-7) are
# not exactly representable, so the result carries ~10 eps of relative error = 2.2e-15. 1e-14 is that
# with ~4x headroom; any coefficient error is >= O(1e-1).
CLOSED_FORM_REL_TOL = 1e-14

# Central difference on Phi_J2 with h = 1e-4 r. Error of D(h) = f' + (h^2/6) f''' + (h^4/120) f^(5).
# f''' of r^-n along an axis is bounded by n(n+2)(n+4) r^-(n+3): 105 for r^-3, 315 for r^-5 (times
# the 3 in 3 z^2, <= 945). With Phi = (A/2)(3 z^2 r^-5 - r^-3) and cross terms from differentiating z^2
# roughly doubling that, |f'''| <~ 1e3 A / r^6. Relative to |grad| >= 1.34 A/r^4:
#     truncation <~ (1e3 / (6 * 1.34)) (h/r)^2 = 125 * 1e-8 = 1.25e-6.
# Rounding: Phi is two terms of size 1.5 A/r^3 and 0.5 A/r^3, each carrying ~5 eps (r^5 multiplies
# the relative error of r by 5), so |dPhi| <~ 9 eps A/r^3 and the difference quotient's rounding is
# <~ 9 eps A/(r^3 h), relative 6.7 eps (r/h) = 1.5e-11. Truncation dominates by five orders, so the
# bound below is a deliberately crude worst case on f''' - it is not the sharp guard; that is
# RICHARDSON_REL_TOL, which removes the h^2 term instead of bounding it.
FD_STEP_REL = 1e-4
FD_TRUNCATION_REL_BOUND = 1.3e-6

# Richardson extrapolation (4 D(h) - D(2h)) / 3 cancels the h^2 term exactly, leaving -(h^4/30) f^(5).
# f^(5) of r^-5 is bounded by 5*7*9*11*13 = 45045 (x3 for 3 z^2, x~1.5 for cross terms) ~ 2e5 A/r^8,
# so truncation <~ (2e5 / (30 * 1.34)) (h/r)^4 = 5e3 * 1e-16 = 5e-13. Rounding: (4 + 1)/3 times the
# single-quotient bound, 1.67 * 1.5e-11 = 2.5e-11. Total ~3e-11; 5e-11 is that bound rounded up.
# A kernel coefficient error cannot hide under this: it would have to be below one part in 1e10.
RICHARDSON_REL_TOL = 5e-11

# Rotation about z by a general angle: the rotated input carries ~2 eps relative error, amplified
# ~5x through the r^-4 dependence, plus the kernel's own ~10 eps and the output rotation's ~2 eps:
# ~22 eps = 4.9e-15. 2e-14 is ~4x that.
AXISYMMETRY_REL_TOL = 2e-14

# Translating the whole state by the Earth's heliocentric offset (|d| = 1.5e8 km) before the kernel
# subtracts it back out costs ~2 eps |d| absolute in the Moon-Earth vector (|rel| = 3.8e5 km), i.e.
# 2 * 2.2e-16 * 1.5e8 / 3.8e5 = 1.7e-13 relative, amplified ~5x by r^-4 and angular terms: ~9e-13.
# 2e-12 is ~2x that. This is an upper bound: Moon and Earth coordinates lie within a factor of 2 of
# each other, so by Sterbenz's lemma the subtraction is often exact and the observed error is 0.0.
# The check's job is catching a kernel that reads absolute position, which would be off by ~1e5x.
TRANSLATION_REL_TOL = 2e-12

# LEO sanity ratio |a_J2| / (mu/r^2) = 1.5 J2 (R/r)^2 sqrt(1 - 2 s^2 + 5 s^4). Both sides are ~10
# rounded ops; 1e-13 relative is ~450 eps and far below any coefficient error.
RATIO_CLOSED_FORM_REL_TOL = 1e-13

# The negative control's z-term error, (5 s^2 - 3) -> (3 s^2 - 3), changes a_z by 2 (A/r^4)(3/2) s^3,
# i.e. a relative error 2|s|^3 / sqrt(1 - 2 s^2 + 5 s^4). It is smallest near the equator (0.01 at
# 10 deg), 0.63 at 45 deg and 0.98 at 80 deg; the checks take the worst sample point, so the mutant
# must show ~0.98. Requiring > 1e-2 is still eight orders above RICHARDSON_REL_TOL.
MUTANT_MIN_REL_ERROR = 1e-2


class _Kernel(Protocol):
    def __call__(
        self, indices: NDArray[np.int64], t: float, state: NDArray[np.float64],
        mu_array: NDArray[np.float64], parent_indices: NDArray[np.int32],
        params: NDArray[np.float64], out: NDArray[np.float64],
    ) -> None: ...


# --------------------------------------------------------------------------------------------------
# Helpers: evaluate a kernel at bare relative positions about a parent at the origin.
# --------------------------------------------------------------------------------------------------

def _accel(kernel: _Kernel, rel: NDArray[np.float64], mu: float, j2: float, r_eq: float) -> NDArray[np.float64]:
    """Slot 0 is the parent at the origin (a root); slots 1..N are bodies at `rel`."""
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = mu
    parent_indices = np.zeros(n + 1, dtype=np.int32)
    params = np.zeros((n + 1, 2))
    params[1:] = [j2, r_eq]
    out = np.zeros((n + 1, 3))
    kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, mu_array, parent_indices, params, out)
    return out[1:]


def _fd_points() -> NDArray[np.float64]:
    """LEO-radius points spanning latitude (both hemispheres) and longitude, off the exact equator and
    poles so that every component of the gradient is exercised."""
    r = EARTH_R_EQ + 600.0
    lat = np.radians([10.0, 26.57, 35.0, 60.0, 80.0, -45.0, -72.0])   # 26.57 deg ~ s^2 = 1/5
    lon = np.radians([0.0, 40.0, 135.0, 200.0, 290.0, 77.0, 330.0])
    return np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)], axis=1)


def _fd_gradient_accel(rel: NDArray[np.float64], h_rel: float) -> NDArray[np.float64]:
    """-grad Phi_J2 by central differences with step h = h_rel * r, dividing by the *realised* step
    (x+h) - (x-h) so the rounding of the step itself does not enter."""
    a = np.empty_like(rel)
    r = np.linalg.norm(rel, axis=1)
    for axis in range(3):
        plus, minus = rel.copy(), rel.copy()
        plus[:, axis] += h_rel * r
        minus[:, axis] -= h_rel * r
        phi_p = geopotential.j2_potential(plus, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
        phi_m = geopotential.j2_potential(minus, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
        a[:, axis] = -(phi_p - phi_m) / (plus[:, axis] - minus[:, axis])
    return a


def _rel_err(got: NDArray[np.float64], want: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1))


def _richardson_rel_err(kernel: _Kernel) -> float:
    rel = _fd_points()
    a = _accel(kernel, rel, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
    d_h = _fd_gradient_accel(rel, FD_STEP_REL)
    d_2h = _fd_gradient_accel(rel, 2.0 * FD_STEP_REL)
    return float(np.max(_rel_err(a, (4.0 * d_h - d_2h) / 3.0)))


def _mutant_z_coefficient_kernel(
    indices: NDArray[np.int64], t: float, state: NDArray[np.float64], mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32], params: NDArray[np.float64], out: NDArray[np.float64],
) -> None:
    """Negative control. Curtis' J2 acceleration written independently, with one deliberate error:
    the z-term's `5 z^2/r^2 - 3` is `3 z^2/r^2 - 3`. Not registered."""
    p = parent_indices[indices]
    rel = state[indices, :3] - state[p, :3]
    r = np.linalg.norm(rel, axis=1)
    k = 1.5 * params[indices, 0] * mu_array[p] * params[indices, 1] ** 2 / r ** 4
    s2 = (rel[:, 2] / r) ** 2
    out[indices, 0] += k * rel[:, 0] / r * (5.0 * s2 - 1.0)
    out[indices, 1] += k * rel[:, 1] / r * (5.0 * s2 - 1.0)
    out[indices, 2] += k * rel[:, 2] / r * (3.0 * s2 - 3.0)


# ==================================================================================================
# Registration
# ==================================================================================================

def test_j2_is_registered_with_its_coefficients_and_citation() -> None:
    model = registry.get_force_model(J2_MODEL)
    assert model.kernel is geopotential.j2_kernel
    assert model.param_names == ("j2", "r_eq")
    assert "Curtis" in model.citation


# ==================================================================================================
# 1. Closed-form values at the equator and the poles
# ==================================================================================================

def test_closed_form_equator_and_poles() -> None:
    """
    By hand, mu = 1e6, J2 = 1e-3, R = 1000, r = 2000: (3/2) J2 mu R^2 / r^4 = 1.5e-3 * 1e12 / 1.6e13
    = 9.375e-5.
      equator, s = 0: a = -(3/2) A/r^4 * r_hat          -> -9.375e-5 along r_hat (inward)
      pole,    s = +-1: a_z = (3/2) A/r^4 * s * (5 - 3)  -> +-1.875e-4 (outward)
    """
    rel = np.array([
        [2000.0, 0.0, 0.0],
        [0.0, 2000.0, 0.0],
        [0.0, 0.0, 2000.0],
        [0.0, 0.0, -2000.0],
    ])
    expected = np.array([
        [-9.375e-5, 0.0, 0.0],
        [0.0, -9.375e-5, 0.0],
        [0.0, 0.0, 1.875e-4],
        [0.0, 0.0, -1.875e-4],
    ])
    got = _accel(geopotential.j2_kernel, rel, mu=1e6, j2=1e-3, r_eq=1000.0)
    assert np.all(_rel_err(got, expected) < CLOSED_FORM_REL_TOL), _rel_err(got, expected)


def test_earth_surface_equatorial_magnitude() -> None:
    """
    (3/2) J2 mu / R^2 at r = R: 1.5 * 1.0826267e-3 * 3.986004418e5 / 6378.137^2
    = 1.6239e-3 * 3.986004e5 / 4.068063e7 = 1.59118e-5 km/s^2 (about 0.016 m/s^2). Literal quoted to
    6 significant figures, so the tolerance is the literal's own 5e-6 relative truncation.
    """
    got = _accel(geopotential.j2_kernel, np.array([[EARTH_R_EQ, 0.0, 0.0]]),
                 scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
    assert got[0, 0] == pytest.approx(-1.59118e-5, rel=5e-6)
    assert got[0, 1] == 0.0 and got[0, 2] == 0.0


# ==================================================================================================
# 2. Independent check: a = -grad Phi_J2, by central finite differences
# ==================================================================================================

def test_kernel_matches_negative_gradient_of_the_j2_potential() -> None:
    """The plain central difference must agree to within its truncation bound - and that residual must
    be h^2 truncation, which Richardson extrapolation then removes down to rounding. A residual that
    survived extrapolation would be a model error, not a finite-difference one."""
    rel = _fd_points()
    a = _accel(geopotential.j2_kernel, rel, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)

    plain = _rel_err(a, _fd_gradient_accel(rel, FD_STEP_REL))
    assert np.all(plain < FD_TRUNCATION_REL_BOUND), plain

    extrapolated = _richardson_rel_err(geopotential.j2_kernel)
    assert extrapolated < RICHARDSON_REL_TOL, extrapolated


# ==================================================================================================
# 3. Symmetry
# ==================================================================================================

def test_axisymmetry_about_the_spin_axis() -> None:
    rel = _fd_points()
    a = _accel(geopotential.j2_kernel, rel, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)

    # A quarter turn (x, y) -> (-y, x) is exact on the inputs, but it is *not* bit-identical on the
    # output: `np.einsum` does not promise a summation grouping for x^2 + y^2 + z^2, and swapping the
    # x and y columns was observed to move r^2 by one ulp. So both angles are held to rounding.
    for angle in (0.5 * np.pi, 0.7):
        c, s = np.cos(angle), np.sin(angle)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        a_rot = _accel(geopotential.j2_kernel, rel @ rz.T, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
        err = _rel_err(a_rot, a @ rz.T)
        assert np.all(err < AXISYMMETRY_REL_TOL), (angle, err)


def test_reflection_symmetry_across_the_equatorial_plane() -> None:
    """z enters only as z*z and as a linear factor of a_z, both exact under negation: bit-identical."""
    rel = _fd_points()
    a = _accel(geopotential.j2_kernel, rel, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
    mirrored = rel * np.array([1.0, 1.0, -1.0])
    a_m = _accel(geopotential.j2_kernel, mirrored, scenarios.MU_EARTH, EARTH_J2, EARTH_R_EQ)
    assert np.array_equal(a_m, a * np.array([1.0, 1.0, -1.0]))


# ==================================================================================================
# 4. LEO sanity ratio, through a real scenario
# ==================================================================================================

def test_leo_ratio_to_point_mass_gravity(db_session_factory: Callable[[], Session]) -> None:
    """
    At 550 km, (R/r)^2 = (6378.137/6928.137)^2 = 0.8475, so the equatorial ratio is
    1.5 * 1.0826e-3 * 0.8475 = 1.376e-3, and over all latitudes the ratio lies in
    [0.894, 2] * 1.376e-3 = [1.23e-3, 2.75e-3]. Inclined planes put the satellites at several latitudes.
    """
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=12, n_planes=2, altitude_km=550.0)
    earth = sim.name_to_index["Earth"]
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)

    sim.enable_force_model(J2_MODEL, bodies=sats, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    a = sim.accelerations(0.0)[sats]

    rel = sim.global_states[sats, :3] - sim.global_states[earth, :3]
    r = np.linalg.norm(rel, axis=1)
    s = rel[:, 2] / r
    ratio = np.linalg.norm(a, axis=1) / (scenarios.MU_EARTH / r ** 2)

    assert np.ptp(s) > 0.5, "scenario did not spread the satellites in latitude"
    assert np.all((1.23e-3 < ratio) & (ratio < 2.75e-3)), ratio

    closed = 1.5 * EARTH_J2 * (EARTH_R_EQ / r) ** 2 * np.sqrt(1.0 - 2.0 * s ** 2 + 5.0 * s ** 4)
    assert np.all(np.abs(ratio / closed - 1.0) < RATIO_CLOSED_FORM_REL_TOL)


# ==================================================================================================
# 5. Composition through the real layer; relative geometry
# ==================================================================================================

def test_composes_exactly_with_another_model(db_session_factory: Callable[[], Session]) -> None:
    """
    `compose_accelerations` zeroes the row, then each kernel adds. Floating-point addition is
    commutative and 0 + x == x, so (0 + c) + a_j2 and (0 + a_j2) + c are the same double: the
    composed result must equal the separately computed sum *bit for bit*, whichever model holds the
    lower registry bit.
    """
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=12, n_planes=2)
    earth = sim.name_to_index["Earth"]
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    both = sats[::2]

    sim.enable_force_model(J2_MODEL, bodies=sats, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    sim.enable_force_model("test_constant_accel", bodies=both, ax=1e-6, ay=-2e-7, az=3e-8)
    composed = sim.accelerations(0.0).copy()

    j2_only = np.zeros_like(composed)
    geopotential.j2_kernel(sats, 0.0, sim.global_states, sim.mu_array, sim.parent_indices,
                           sim.force_model_params[J2_MODEL], j2_only)
    expected = j2_only.copy()
    expected[both] += np.array([1e-6, -2e-7, 3e-8])

    assert np.array_equal(composed, expected)
    assert np.array_equal(composed[earth], [0.0, 0.0, 0.0])
    assert np.all(np.linalg.norm(composed[sats[1::2]], axis=1) > 0.0)


def test_uses_position_relative_to_the_parent(db_session_factory: Callable[[], Session]) -> None:
    """
    Sun-Earth-Moon puts Earth 1.5e8 km from the origin. Earth's J2 on the Moon evaluated from the
    committed state and from the same state re-centred on Earth must agree to TRANSLATION_REL_TOL.
    A kernel reading absolute rather than parent-relative position would differ by orders of magnitude.
    (The scenario is ecliptic-framed, so this is Earth's J2 about the ecliptic pole - a geometry check,
    not a physical lunar perturbation; see the frame note in geopotential.py.)

    Magnitude: (3/2) J2 (R/r)^2 mu/r^2 = 1.5 * 1.08e-3 * (6378/3.8e5)^2 * 2.7e-6 ~ 1e-12 km/s^2,
    asserted in [5e-13, 5e-12] to prove the row is populated with the right order.
    """
    sim = scenarios.sun_earth_moon(db_session_factory())
    earth, moon = sim.name_to_index["Earth"], sim.name_to_index["Moon"]
    assert sim.parent_indices[moon] == earth

    sim.enable_force_model(J2_MODEL, bodies=moon, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    committed = sim.accelerations(0.0)[moon].copy()
    recentred = sim.accelerations(0.0, state=sim.global_states - sim.global_states[earth])[moon].copy()

    assert np.linalg.norm(sim.global_states[earth, :3]) > 1e8
    assert 5e-13 < np.linalg.norm(committed) < 5e-12
    assert np.linalg.norm(committed - recentred) / np.linalg.norm(recentred) < TRANSLATION_REL_TOL


# ==================================================================================================
# 6. Empty indices and degenerate parents
# ==================================================================================================

def test_empty_indices_is_a_true_noop() -> None:
    out = np.full((3, 3), 7.0)
    state = np.arange(18, dtype=np.float64).reshape(3, 6)
    params = np.ones((3, 2))
    geopotential.j2_kernel(np.array([], dtype=np.int64), 0.0, state, np.ones(3),
                           np.zeros(3, dtype=np.int32), params, out)
    assert np.array_equal(out, np.full((3, 3), 7.0))


def test_root_and_coincident_bodies_contribute_exactly_zero(
    db_session_factory: Callable[[], Session],
) -> None:
    """A root self-parents, so its relative position is exactly zero; the kernel must add exactly 0.0
    and must not even evaluate an invalid operation (errstate='raise' would turn one into an error)."""
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=6)
    earth = sim.name_to_index["Earth"]
    assert sim.parent_indices[earth] == earth

    sim.enable_force_model(J2_MODEL, bodies=earth, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    with np.errstate(all="raise"):
        a = sim.accelerations(0.0)
    assert np.array_equal(a[earth], [0.0, 0.0, 0.0])

    # A non-root body sitting exactly on its parent.
    state = np.zeros((2, 6))
    out = np.zeros((2, 3))
    with np.errstate(all="raise"):
        geopotential.j2_kernel(np.array([1], dtype=np.int64), 0.0, state, np.array([1.0, 0.0]),
                               np.array([0, 0], dtype=np.int32), np.array([[0, 0], [EARTH_J2, EARTH_R_EQ]]), out)
    assert np.array_equal(out, np.zeros((2, 3)))


def test_barycentre_parent_is_flagged_at_configuration_and_never_nan(
    db_session_factory: Callable[[], Session],
) -> None:
    """The kernel cannot see `is_system` (not in the ForceKernel signature), so the barycentre case is
    caught by `barycentre_parented` at configuration time. The kernel itself must still stay finite if
    a caller ignores that guard. No shipped scenario parents a body to a barycentre, so the parent graph
    is re-pointed on a copy."""
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=6)
    bary = sim.name_to_index["Earth Barycenter"]
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    assert sim.is_system[bary]

    assert geopotential.barycentre_parented(sim.is_system, sim.parent_indices, sats).size == 0

    parents = sim.parent_indices.copy()
    parents[sats[:3]] = bary
    flagged = geopotential.barycentre_parented(sim.is_system, parents, sats)
    assert np.array_equal(flagged, sats[:3])

    params = np.zeros((sim.max_capacity, 2))
    params[sats] = [EARTH_J2, EARTH_R_EQ]
    out = np.zeros((sim.max_capacity, 3))
    geopotential.j2_kernel(sats, 0.0, sim.global_states, sim.mu_array, parents, params, out)
    assert np.all(np.isfinite(out))


def test_unset_coefficients_contribute_exactly_zero(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=6)
    sats = np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)
    sim.enable_force_model(J2_MODEL, bodies=sats)
    assert np.array_equal(sim.accelerations(0.0)[sats], np.zeros((sats.size, 3)))


# ==================================================================================================
# 7. Negative control: the checks above catch a wrong coefficient
# ==================================================================================================

def test_negative_control_wrong_z_coefficient_is_caught() -> None:
    """
    With the z-term's 5 replaced by 3: at the pole a_z becomes (3/2) A/r^4 * (3 - 3) = 0 instead of
    1.875e-4, and the gradient check fails by >= MUTANT_MIN_REL_ERROR. At the equator (z = 0) the
    mutant is indistinguishable - which is why the gradient check samples off-equator latitudes.
    Also confirms the mutant is otherwise faithful: it passes the equatorial closed form, so its
    failures are attributable to the one coefficient.
    """
    mutant = _mutant_z_coefficient_kernel

    equator = _accel(mutant, np.array([[2000.0, 0.0, 0.0]]), mu=1e6, j2=1e-3, r_eq=1000.0)
    assert _rel_err(equator, np.array([[-9.375e-5, 0.0, 0.0]]))[0] < CLOSED_FORM_REL_TOL

    pole = _accel(mutant, np.array([[0.0, 0.0, 2000.0]]), mu=1e6, j2=1e-3, r_eq=1000.0)
    assert _rel_err(pole, np.array([[0.0, 0.0, 1.875e-4]]))[0] > MUTANT_MIN_REL_ERROR

    assert _richardson_rel_err(mutant) > MUTANT_MIN_REL_ERROR
