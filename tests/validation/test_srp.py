"""
Validation of the `"srp"` force model (`srp.py`): cannonball solar radiation pressure with a
cylindrical or conical shadow.

Every check below states the number it expects **before** measuring it, per `CLAUDE.md`'s item 5.
The derivations live in the docstring of each section; the order is:

1. **Magnitude and direction.** The closed form recomputed in SI, and the direction as a *vector
   projection* onto the body-Sun line. A magnitude check cannot see a sign error, and a sign-flipped
   SRP produces a perfectly plausible orbit - so the projection, not the magnitude, is the load-bearing
   assertion here.
2. **Shadow geometry.** Exactly-zero umbra, exactly-one full sun, the terminator at the radius the
   geometry says, and the conical penumbra checked against a **numerical quadrature of the disc
   overlap** that shares no code with the closed form.
3. **SRP against drag.** The crossover altitude, derived as a density and then located in
   `atmosphere.py`'s table. This is the comparison the two models exist together to make.
4. **Orbit scale.** The secular drift of the eccentricity *vector*, against the orbit-averaged Gauss
   rate derived from first principles in `test_eccentricity_vector_matches_the_gauss_rate`.
5. **The cylinder's discontinuity.** It costs RK4 its convergence order; that is measured, and the
   conical model is shown to recover it.
6. **Configuration.** Coefficient validation, the fused-kernel fallback, and sweep name resolution.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import registry, scenarios, sweep
from orbital_engine.atmosphere import DENSITY_MODEL_LAYERED, layered_density
from orbital_engine.constants import AU2KM, C_SI
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import DRAG_MODEL, EARTH_OMEGA, drag_kernel
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.srp import (
    SHADOW_MODEL_CONICAL, SHADOW_MODEL_CYLINDRICAL, SOLAR_CONSTANT_1AU, SOLAR_PRESSURE_1AU,
    SRP_MODEL, SUN_RADIUS, shadow_factor, srp_kernel,
)

ArrF = NDArray[np.float64]

MU_E = scenarios.MU_EARTH
CR = 1.3            # a representative spacecraft radiation-pressure coefficient
AREA_MASS = 0.2     # m^2/kg - a high area-to-mass-ratio object, chosen so the orbit-scale effect
                    # sits well above the integrator's own noise; the model is linear in it.
CD = 2.2            # the free-molecular drag coefficient the crossover comparison assumes


def _session() -> Session:
    """An isolated in-memory database, built the way `tests/conftest.py` builds one. A local copy
    exists because several checks build more than one simulation and the conftest fixture is one."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# ==================================================================================================
# 1. Magnitude and direction
# ==================================================================================================

# Floating point only. `a = cr * p * (A/m) * (AU/d)^2 * 1e-3` is about eight rounded operations with
# no cancellation anywhere, so the relative error is a few ulp. Budget 1e-14, three orders of
# headroom over eps and still five orders tighter than any physical error this model has.
CLOSED_FORM_REL_TOL = 1e-14

# The direction is a normalised difference of two vectors that differ by 1 AU. The unit vector is
# accurate to ~eps and the projection to a few eps. Budget 8 eps.
DIRECTION_TOL = 8.0 * float(np.finfo(np.float64).eps)


def _bare_arena(n_bodies: int) -> Tuple[ArrF, ArrF, NDArray[np.int32], ArrF, ArrF]:
    """`(state, mu, parents, params, out)` for a hand-built arena: slot 0 is the occulting parent,
    slot 1 the light source, slots 2.. the bodies. Nothing here touches the ORM or `Simulation`."""
    state = np.zeros((n_bodies, 6))
    mu = np.zeros(n_bodies)
    parents = np.zeros(n_bodies, dtype=np.int32)
    params = np.zeros((n_bodies, 7))
    out = np.zeros((n_bodies, 3))
    return state, mu, parents, params, out


def test_solar_pressure_constant_is_S_over_c() -> None:
    """`P = S/c`. 1367 W/m^2 gives the 4.56e-6 N/m^2 quoted throughout the literature; the IAU 2015
    nominal 1361 W/m^2 gives 0.44 % less. Both statements are asserted so the fork is documented in
    a test rather than only in prose."""
    assert SOLAR_PRESSURE_1AU == pytest.approx(SOLAR_CONSTANT_1AU / C_SI, rel=0.0, abs=0.0)
    assert SOLAR_PRESSURE_1AU == pytest.approx(4.56e-6, rel=1e-3)
    assert 1361.0 / C_SI / SOLAR_PRESSURE_1AU == pytest.approx(1.0 - 0.0044, rel=1e-3)


def test_closed_form_magnitude_and_anti_sunward_direction() -> None:
    """
    Magnitude against the closed form recomputed in SI, and direction as a vector projection.

    Derivation, all in SI: `a = C_r P (A/m) (AU/d)^2` m/s^2, so at `d = 1 AU` with `C_r = 1.3`,
    `A/m = 0.2 m^2/kg`, `P = 4.5598e-6 N/m^2`, `a = 1.1856e-6 m/s^2 = 1.1856e-9 km/s^2`. Three
    bodies are placed at 1, 2 and 0.5 AU from the source along **non-axis-aligned** directions, so
    the inverse-square law is exercised and a transposed component cannot hide.

    The direction assertion is `a . u_hat / |a| == -1` exactly (to 8 eps), where `u_hat` points from
    the body **to** the source. This is the check that a sign flip cannot survive; the magnitude
    assertion is blind to it.
    """
    state, mu, parents, params, out = _bare_arena(5)
    state[1, :3] = [AU2KM, 0.0, 0.0]                      # source
    # Bodies at 1, 2 and 0.5 AU from the source, in three unrelated directions.
    offsets = np.array([
        [1.0, 1.0, 1.0],
        [-2.0, 1.0, 3.0],
        [0.3, -0.7, 0.2],
    ])
    distances = np.array([1.0, 2.0, 0.5]) * AU2KM
    units = offsets / np.linalg.norm(offsets, axis=1)[:, None]
    state[2:5, :3] = state[1, :3] - units * distances[:, None]
    params[2:5] = [CR, AREA_MASS, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS, SHADOW_MODEL_CYLINDRICAL]

    srp_kernel(np.arange(2, 5, dtype=np.int64), 0.0, state, mu, parents, params, out)

    a_si_1au = CR * SOLAR_PRESSURE_1AU * AREA_MASS                       # m/s^2 at 1 AU
    assert a_si_1au == pytest.approx(1.1856e-6, rel=1e-4), "guard on the derivation itself"
    expected_mag = a_si_1au * 1e-3 / (distances / AU2KM) ** 2            # km/s^2

    measured = np.linalg.norm(out[2:5], axis=1)
    assert np.all(np.abs(measured / expected_mag - 1.0) < CLOSED_FORM_REL_TOL), (measured, expected_mag)

    projection = np.einsum("ij,ij->i", out[2:5], units) / measured
    assert np.all(np.abs(projection + 1.0) < DIRECTION_TOL), (
        f"SRP must push anti-sunward; projection onto the body->Sun unit vector = {projection}")
    # And nothing off-axis: the residual after removing the anti-sunward part is pure round-off.
    residual = out[2:5] + measured[:, None] * units
    assert np.all(np.linalg.norm(residual, axis=1) / measured < DIRECTION_TOL)
    assert np.all(out[:2] == 0.0), "the kernel wrote rows outside `indices`"


def test_kernel_is_additive_and_a_no_op_on_empty_indices() -> None:
    state, mu, parents, params, out = _bare_arena(3)
    state[1, :3] = [AU2KM, 0.0, 0.0]
    state[2, :3] = [0.0, 7000.0, 0.0]
    params[2] = [CR, AREA_MASS, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS, SHADOW_MODEL_CYLINDRICAL]
    out[:] = 7.0

    srp_kernel(np.empty(0, dtype=np.int64), 0.0, state, mu, parents, params, out)
    assert np.all(out == 7.0)

    srp_kernel(np.array([2], dtype=np.int64), 0.0, state, mu, parents, params, out)
    d = float(np.linalg.norm(state[1, :3] - state[2, :3]))
    expected = CR * SOLAR_PRESSURE_1AU * AREA_MASS * (AU2KM / d) ** 2 * 1e-3
    assert out[2, 0] == pytest.approx(7.0 - expected, rel=1e-13)


def test_degenerate_rows_are_exactly_zero() -> None:
    """A body coincident with its source, and a body with zero `cr` or zero `A/m`, contribute
    **exactly** 0.0 - not 'small'. No division is evaluated in the first case."""
    state, mu, parents, params, out = _bare_arena(5)
    state[1, :3] = [AU2KM, 0.0, 0.0]
    state[2, :3] = state[1, :3]                                     # sitting on the source
    state[3, :3] = [0.0, 7000.0, 0.0]
    state[4, :3] = [0.0, 7000.0, 0.0]
    params[2] = [CR, AREA_MASS, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS, SHADOW_MODEL_CYLINDRICAL]
    params[3] = [0.0, AREA_MASS, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS, SHADOW_MODEL_CYLINDRICAL]
    params[4] = [CR, 0.0, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS, SHADOW_MODEL_CYLINDRICAL]

    with np.errstate(all="raise"):
        srp_kernel(np.arange(2, 5, dtype=np.int64), 0.0, state, mu, parents, params, out)
    assert np.all(out == 0.0)


# ==================================================================================================
# 2. Shadow geometry
# ==================================================================================================

def _geometry(
    body: ArrF, source: ArrF, occulter: ArrF,
) -> Tuple[ArrF, ArrF]:
    """`(to_source, to_occulter)` for `shadow_factor`, each shaped `(k, 3)`."""
    return source - body, occulter - body


def test_cylindrical_umbra_terminator_sits_exactly_at_the_occulter_radius() -> None:
    """
    Exactly zero behind the occulter, exactly one outside it, and the switch at `r_occ` to within a
    relative `1e-12` of the radius - the geometry's own definition, with no tolerance to tune.

    The occulter is at the origin with radius `R = 6378.137 km`; the source is at `+x`, 1 AU away.
    Bodies sit at `x = -42164 km` (behind) at perpendicular offsets `R(1 - 1e-12)` and `R(1 + 1e-12)`.
    The umbra is a *half*-infinite cylinder, so a body at the same offset but at `x = +42164 km` -
    between the source and the occulter - must be in full sun.
    """
    r_occ = EARTH_R_EQ
    source = np.array([AU2KM, 0.0, 0.0])
    occulter = np.zeros(3)
    inside, outside = r_occ * (1.0 - 1e-12), r_occ * (1.0 + 1e-12)
    bodies = np.array([
        [-42164.0, 0.0, 0.0],          # deep umbra, on the axis
        [-42164.0, inside, 0.0],        # just inside the terminator
        [-42164.0, outside, 0.0],       # just outside it
        [+42164.0, 0.0, 0.0],           # sunward side: the cylinder is half-infinite
        [-42164.0, 0.0, 2.0 * r_occ],   # well clear, in the z direction
    ])
    to_source, to_occulter = _geometry(bodies, source, occulter)
    nu = shadow_factor(to_source, to_occulter,
                       np.full(5, SUN_RADIUS), np.full(5, r_occ), np.zeros(5, dtype=np.bool_))

    assert nu.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0], nu


def test_no_occulter_radius_means_full_sun_under_either_law() -> None:
    """`r_occ <= 0` is the 'no occulter' convention (`drag.py`'s `scale_height <= 0` idiom). A body
    sitting exactly where the occulter is - which would be deep umbra with a real radius - reads
    exactly 1.0, and no division is evaluated."""
    bodies = np.array([[-42164.0, 0.0, 0.0], [-42164.0, 0.0, 0.0]])
    to_source, to_occulter = _geometry(bodies, np.array([AU2KM, 0.0, 0.0]), np.zeros(3))
    with np.errstate(all="raise"):
        nu = shadow_factor(to_source, to_occulter, np.full(2, SUN_RADIUS), np.zeros(2),
                           np.array([False, True]))
    assert nu.tolist() == [1.0, 1.0]


def _disc_overlap_fraction(alpha: float, beta: float, sep: float, n: int = 400001) -> float:
    """
    The fraction of a disc of radius `alpha` covered by a disc of radius `beta` whose centre is `sep`
    away, by **one-dimensional quadrature** over vertical chords.

    Deliberately a different derivation from `srp.shadow_factor`'s two-arccos circular-segment
    formula: at each `x` the two discs' half-chords are `sqrt(alpha^2 - x^2)` and
    `sqrt(beta^2 - (x - sep)^2)`, and the covered height is twice the smaller of the two where both
    exist. Trapezoidal integration of a function with square-root endpoints converges as `h^1.5`, so
    at `n = 4e5` over an interval of order `alpha` the relative error is of order `1e-8`; the test
    below budgets `1e-5`, three orders of slack.
    """
    lo = max(-alpha, sep - beta)
    hi = min(alpha, sep + beta)
    if hi <= lo:
        return 0.0
    x = np.linspace(lo, hi, n)
    half_a = np.sqrt(np.maximum(alpha * alpha - x * x, 0.0))
    half_b = np.sqrt(np.maximum(beta * beta - (x - sep) ** 2, 0.0))
    area = float(np.trapezoid(2.0 * np.minimum(half_a, half_b), x))
    return area / (math.pi * alpha * alpha)


@pytest.mark.parametrize("beta_over_alpha", [0.3, 1.0, 3.0, 30.0])
def test_conical_shadow_matches_an_independent_disc_overlap_quadrature(
    beta_over_alpha: float,
) -> None:
    """
    `1 - nu` from the kernel against the numerically integrated covered fraction, swept across the
    whole penumbra and beyond it at four occulter-to-source apparent-size ratios (annular through
    deeply total).

    Expected agreement: `1e-5` relative to the disc area, set by the quadrature (`~1e-8`) plus the
    small-angle difference between the planar overlap the formula assumes and the same planar overlap
    the quadrature computes - which is zero, because both are planar. So this is a pure algebra
    check, and `1e-5` is loose by three orders. Anything that fails it is a wrong formula, not a
    tolerance question.
    """
    d_source, d_occ = AU2KM, 42164.0
    alpha = math.asin(SUN_RADIUS / d_source)
    beta = alpha * beta_over_alpha
    r_occ = d_occ * math.sin(beta)

    seps = np.linspace(0.0, 1.2 * (alpha + beta), 25)
    body = np.zeros((seps.size, 3))
    to_source = np.tile(np.array([d_source, 0.0, 0.0]), (seps.size, 1))
    to_occulter = d_occ * np.stack([np.cos(seps), np.sin(seps), np.zeros_like(seps)], axis=1)

    nu = shadow_factor(to_source + body, to_occulter, np.full(seps.size, SUN_RADIUS),
                       np.full(seps.size, r_occ), np.ones(seps.size, dtype=np.bool_))
    expected = np.array([_disc_overlap_fraction(alpha, beta, float(s)) for s in seps])

    assert np.all(np.abs((1.0 - nu) - expected) < 1e-5), np.column_stack([seps, 1.0 - nu, expected])
    assert nu[0] == pytest.approx(max(0.0, 1.0 - beta_over_alpha ** 2), abs=1e-12), (
        "on-axis: annular below beta = alpha, total above it")
    assert nu[-1] == 1.0, "beyond alpha + beta the source is fully visible"


def test_penumbra_is_monotonic_and_brackets_the_cylindrical_terminator() -> None:
    """
    Across the penumbra `nu` rises monotonically from exactly 0 to exactly 1, and the cylindrical
    terminator falls strictly inside it.

    Derivation of the width: the penumbra spans angular separations `beta - alpha` to `beta + alpha`,
    a width of `2 alpha = 2 asin(R_sun / 1 AU) = 9.30e-3 rad`. At the test's 42164 km from the
    occulter centre that is `2 alpha d_occ = 392 km` of perpendicular offset. The cylindrical
    terminator is where the occulter's limb touches the line to the source's *centre*, which
    occults half the disc - so conical `nu` there must be near 0.5 and strictly between 0 and 1.
    """
    r_occ, d_occ = EARTH_R_EQ, 42164.0
    alpha = math.asin(SUN_RADIUS / AU2KM)
    half_width_km = alpha * d_occ
    assert half_width_km == pytest.approx(196.0, rel=0.02), "guard on the derived penumbra width"

    offsets = np.linspace(r_occ - 2.0 * half_width_km, r_occ + 2.0 * half_width_km, 81)
    bodies = np.stack([np.full_like(offsets, -d_occ), offsets, np.zeros_like(offsets)], axis=1)
    to_source, to_occulter = _geometry(bodies, np.array([AU2KM, 0.0, 0.0]), np.zeros(3))
    args = (np.full(offsets.size, SUN_RADIUS), np.full(offsets.size, r_occ))
    nu = shadow_factor(to_source, to_occulter, *args, np.ones(offsets.size, dtype=np.bool_))
    cyl = shadow_factor(to_source, to_occulter, *args, np.zeros(offsets.size, dtype=np.bool_))

    assert np.all(np.diff(nu) >= 0.0), "the penumbra must brighten monotonically outward"
    assert nu[0] == 0.0 and nu[-1] == 1.0
    assert set(np.unique(cyl)) == {0.0, 1.0}, "the cylinder is a step function by construction"

    at_terminator = shadow_factor(
        *_geometry(np.array([[-d_occ, r_occ, 0.0]]), np.array([AU2KM, 0.0, 0.0]), np.zeros(3)),
        np.array([SUN_RADIUS]), np.array([r_occ]), np.array([True]))[0]
    assert 0.0 < at_terminator < 1.0 and at_terminator == pytest.approx(0.5, abs=0.02)

    # The conical law is at least as dark as the cylinder nowhere: it is dark strictly deeper in and
    # lit strictly further out, so the two step boundaries bracket the penumbra.
    assert np.flatnonzero(nu > 0.0)[0] < np.flatnonzero(cyl > 0.0)[0]


# ==================================================================================================
# 3. SRP against drag: the crossover altitude
# ==================================================================================================

def _crossover_altitude_km(cr: float, cd: float, p_srp: float) -> float:
    """
    The altitude at which SRP at 1 AU equals drag on a circular orbit, from the two closed forms:

        C_r P (A/m) = (1/2) rho C_d (A/m) v_circ^2      =>      rho* = 2 C_r P / (C_d v_circ^2)

    `A/m` cancels **exactly** - the crossover is a property of `C_r/C_d` and the atmosphere alone,
    not of the spacecraft's area-to-mass ratio. `v_circ` depends weakly on altitude, so this solves
    the implicit equation by scanning `atmosphere.layered_density`.
    """
    h = np.linspace(200.0, 1400.0, 120001)
    v = np.sqrt(MU_E / (EARTH_R_EQ + h)) * 1e3                   # m/s
    rho_star = 2.0 * cr * p_srp / (cd * v * v)                   # kg/m^3
    return float(h[int(np.argmin(np.abs(layered_density(h) - rho_star)))])


def test_srp_overtakes_drag_at_the_derived_crossover_altitude() -> None:
    """
    Where SRP takes over from drag, measured by running **both shipped kernels** against each other
    on the same bodies - not by re-deriving either.

    Derived first: equality needs `rho* = 2 C_r P / (C_d v^2)`. With `C_r = 1.3`, `C_d = 2.2`,
    `P = 4.56e-6 N/m^2` and `v = 7.53 km/s`, `rho* = 9.5e-14 kg/m^3`. In `atmosphere.py`'s
    Vallado Table 8-4 that density sits at about **630 km**, not the 800 km rule of thumb: that
    table is a static, moderate-activity fit, and the 800 km figure assumes an atmosphere roughly
    eight times denser (elevated solar activity). The crossover moves about 140 km per decade of
    density, so the two statements are the same physics under different atmospheres. The assertion
    brackets 550-750 km, wide enough not to be a snapshot of the table's exact entries and narrow
    enough to fail on a factor-of-two error in either model.

    `A/m` cancels from the balance, so the crossover altitude is asserted **identical** for two
    area-to-mass ratios a decade apart. A model whose acceleration did not scale linearly in `A/m`
    would break that.
    """
    h_star = _crossover_altitude_km(CR, CD, SOLAR_PRESSURE_1AU)
    assert 550.0 < h_star < 750.0, h_star
    assert _crossover_altitude_km(CR, CD, SOLAR_PRESSURE_1AU) == h_star   # A/m-free by construction

    # Now the two kernels, on the same arena: Earth at the origin, the Sun at +x, and one satellite
    # per altitude at the sub-solar point moving in +y, so drag opposes +y and SRP pushes along -x.
    altitudes = np.array([h_star - 150.0, h_star - 50.0, h_star + 50.0, h_star + 150.0])
    n = altitudes.size + 2
    state, mu, parents, params, out = _bare_arena(n)
    state[1, :3] = [AU2KM, 0.0, 0.0]
    radii = EARTH_R_EQ + altitudes
    state[2:, 0] = radii
    state[2:, 4] = np.sqrt(MU_E / radii)
    mu[0] = MU_E
    idx = np.arange(2, n, dtype=np.int64)

    for area_mass in (0.02, 0.2):
        params[:] = 0.0
        params[2:] = [CR, area_mass, SOLAR_PRESSURE_1AU, 1.0, 0.0, SUN_RADIUS,
                      SHADOW_MODEL_CYLINDRICAL]
        out[:] = 0.0
        srp_kernel(idx, 0.0, state, mu, parents, params, out)
        a_srp = np.linalg.norm(out[2:], axis=1).copy()

        drag_params = np.zeros((n, 7))
        drag_params[2:] = [CD * area_mass, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_LAYERED]
        out[:] = 0.0
        drag_kernel(idx, 0.0, state, mu, parents, drag_params, out)
        a_drag = np.linalg.norm(out[2:], axis=1)

        ratio = a_srp / a_drag
        assert np.all(np.diff(ratio) > 0.0), "SRP/drag must grow with altitude"
        assert ratio[1] < 1.0 < ratio[2], (altitudes, ratio)
        # Linear in A/m on both sides, so the ratio is independent of it.
        assert a_srp[0] / area_mass == pytest.approx(
            CR * SOLAR_PRESSURE_1AU * (AU2KM / (AU2KM - radii[0])) ** 2 * 1e-3, rel=1e-12)


# ==================================================================================================
# 4. Orbit scale: the eccentricity vector
# ==================================================================================================

ORBITS = 20
ORBIT_DT = 40.0

# Derived budget for check 4, before measuring (see the test's docstring for each term):
#   short-period residual at the endpoint   (2/3)/(2 pi N) = 5.3e-3
#   the Sun's direction turning over 1.35 d  ~1.2e-2 (half of it enters the accumulated vector)
#   coupling with the seeded e0 = 1e-3       ~1e-3
#   RK4 + the frozen source at dt = 40 s     ~1e-4  (measured as the dt = 10 s vs 40 s difference)
# Sum of the first three, in quadrature or not, lands near 1-2 %. Budget 3 %.
GAUSS_REL_TOL = 0.03


def _eccentricity_vector(r: ArrF, v: ArrF, mu: float) -> ArrF:
    out: ArrF = ((v @ v - mu / float(np.linalg.norm(r))) * r - (r @ v) * v) / mu
    return out


def _run_leo_sat(with_srp: bool, dt: float = ORBIT_DT, orbits: int = ORBITS) -> Dict[str, ArrF]:
    """`scenarios.sun_earth_moon(moon_mu=0.0, leo_satellite=True)`, LEO-SAT on Cowell with
    `point_mass_gravity` and optionally `"srp"`, shadow disabled. Returns its Earth-relative state
    at both ends plus the run's geometry."""
    sim = scenarios.sun_earth_moon(_session(), moon_mu=0.0, leo_satellite=True)
    sat, sun, earth = (sim.name_to_index[n] for n in ("LEO-SAT", "Sun", "Earth"))
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sat)
    if with_srp:
        sim.enable_force_model(
            SRP_MODEL, sat, cr=CR, area_mass=AREA_MASS, p_srp=SOLAR_PRESSURE_1AU,
            source=float(sun), r_occ=0.0, r_source=SUN_RADIUS,
            shadow_model=SHADOW_MODEL_CYLINDRICAL)

    r0 = sim.global_states[sat, :3] - sim.global_states[earth, :3]
    v0 = sim.global_states[sat, 3:] - sim.global_states[earth, 3:]
    to_sun = sim.global_states[sun, :3] - sim.global_states[sat, :3]

    a_sma = 1.0 / (2.0 / float(np.linalg.norm(r0)) - float(v0 @ v0) / MU_E)
    n = math.sqrt(MU_E / a_sma ** 3)
    steps = int(round(orbits * 2.0 * math.pi / n / dt))
    for _ in range(steps):
        sim.step(dt)

    return dict(
        r0=r0, v0=v0, to_sun=to_sun, a=np.array(a_sma), n=np.array(n), T=np.array(steps * dt),
        r1=sim.global_states[sat, :3] - sim.global_states[earth, :3],
        v1=sim.global_states[sat, 3:] - sim.global_states[earth, 3:],
    )


@pytest.fixture(scope="module")
def gauss_runs() -> Tuple[Dict[str, ArrF], Dict[str, ArrF]]:
    """One run with SRP and one thrust-free control, identical in every other respect. Differencing
    them removes the seeded orbit's own evolution and RK4's drift, the same twin idiom
    `scenarios.powered_vessel` builds in for thrust."""
    return _run_leo_sat(True), _run_leo_sat(False)


def test_eccentricity_vector_matches_the_gauss_rate(
    gauss_runs: Tuple[Dict[str, ArrF], Dict[str, ArrF]],
) -> None:
    r"""
    Over 20 orbits, the SRP-induced change in the **eccentricity vector** against the orbit-averaged
    rate, derived here from first principles.

    Derivation. For a perturbing acceleration `f`, the eccentricity vector
    `e = (v x h)/mu - r_hat` obeys

        de/dt = [ f x h + v x (r x f) ] / mu = [ f x h + r (v . f) - f (v . r) ] / mu.

    Average over a circular orbit of radius `a`: write `r = a(cos E, sin E, 0)`,
    `v = na(-sin E, cos E, 0)`, `h = n a^2 z_hat`, and take `f = f x_hat` in the orbit plane. Then
    `f x h = -f n a^2 y_hat` (constant), `<r (v.f)> = -(1/2) f n a^2 y_hat` since
    `<sin^2 E> = 1/2` and `<sin E cos E> = 0`, and `v . r = 0` exactly. With `mu = n^2 a^3`,

        <de/dt> = -(3/2) f / (n a) y_hat,      i.e.   <de/dt> = -(3/2) (h_hat x f) / (n a)

    written frame-independently, which also projects out any component of `f` along `h_hat`. So the
    eccentricity vector grows **perpendicular to the force**, at `(3/2)|f_parallel|/(n a)`.

    Numbers for this case: `a = 7000 km`, `n = 1.0780e-3 rad/s`, `|f| = 1.1769e-9 km/s^2`
    (`C_r = 1.3`, `A/m = 0.2 m^2/kg`, at Earth's epoch distance from the Sun), `T = 20` orbits
    `= 1.166e5 s`. That gives `|de| = 2.68e-5` - a radial excursion of `a|de| = 0.19 km`, three
    orders above the integrator's noise on this run.

    Expected agreement `GAUSS_REL_TOL = 3 %`, budgeted term by term at its definition. The
    **direction** is asserted separately as `cos > 0.999`: a sign-flipped kernel would reproduce
    `|de|` exactly and reverse the vector, so the magnitude assertion alone would pass it.
    """
    on, off = gauss_runs
    measured = (_eccentricity_vector(on["r1"], on["v1"], MU_E)
                - _eccentricity_vector(off["r1"], off["v1"], MU_E))

    h_hat = np.cross(on["r0"], on["v0"])
    h_hat = h_hat / np.linalg.norm(h_hat)
    d = float(np.linalg.norm(on["to_sun"]))
    f = -CR * SOLAR_PRESSURE_1AU * AREA_MASS * (AU2KM / d) ** 2 * 1e-3 * on["to_sun"] / d
    predicted = -1.5 / (float(on["n"]) * float(on["a"])) * np.cross(h_hat, f) * float(on["T"])

    assert float(np.linalg.norm(f)) == pytest.approx(1.177e-9, rel=1e-3), "guard on the derivation"
    assert float(np.linalg.norm(predicted)) == pytest.approx(2.68e-5, rel=1e-2), "guard"

    cos = float(measured @ predicted / (np.linalg.norm(measured) * np.linalg.norm(predicted)))
    assert cos > 0.999, f"the eccentricity vector drifted the wrong way: cos = {cos:.6f}"
    rel = float(np.linalg.norm(measured - predicted) / np.linalg.norm(predicted))
    assert rel < GAUSS_REL_TOL, f"measured {measured}, predicted {predicted}, rel {rel:.4f}"


def test_flipping_the_srp_sign_is_caught_by_the_gauss_direction(
    gauss_runs: Tuple[Dict[str, ArrF], Dict[str, ArrF]],
) -> None:
    """Negative control for the check above: the *magnitude* of `de` is invariant under a sign flip
    of the acceleration, so only the direction assertion can see it. This asserts that explicitly,
    rather than leaving it to a mutation run."""
    on, off = gauss_runs
    measured = (_eccentricity_vector(on["r1"], on["v1"], MU_E)
                - _eccentricity_vector(off["r1"], off["v1"], MU_E))
    h_hat = np.cross(on["r0"], on["v0"])
    h_hat = h_hat / np.linalg.norm(h_hat)
    d = float(np.linalg.norm(on["to_sun"]))
    f = -CR * SOLAR_PRESSURE_1AU * AREA_MASS * (AU2KM / d) ** 2 * 1e-3 * on["to_sun"] / d
    predicted = -1.5 / (float(on["n"]) * float(on["a"])) * np.cross(h_hat, f) * float(on["T"])

    flipped = -predicted
    assert np.linalg.norm(flipped) == pytest.approx(float(np.linalg.norm(predicted)), rel=0.0)
    cos_flipped = float(measured @ flipped / (np.linalg.norm(measured) * np.linalg.norm(flipped)))
    assert cos_flipped < -0.999, "a sign flip must be visible as a reversed eccentricity drift"


# ==================================================================================================
# 5. What the cylinder's discontinuity costs RK4
# ==================================================================================================

CONV_TOTAL = 8800.0     # s; ~1.5 orbits at 7000 km, an exact multiple of every step size used
CONV_STEPS = (10.0, 5.0, 2.5, 1.25)


def _converge(r_occ: float, shadow: float, steps: Tuple[float, ...] = CONV_STEPS) -> Dict[float, ArrF]:
    """LEO-SAT's Earth-relative end state at each step size, over the same horizon."""
    out: Dict[float, ArrF] = {}
    for dt in steps:
        sim = scenarios.sun_earth_moon(_session(), moon_mu=0.0, leo_satellite=True)
        sat, sun, earth = (sim.name_to_index[n] for n in ("LEO-SAT", "Sun", "Earth"))
        sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
        sim.enable_force_model(POINT_MASS_MODEL, sat)
        sim.enable_force_model(
            SRP_MODEL, sat, cr=CR, area_mass=AREA_MASS, p_srp=SOLAR_PRESSURE_1AU,
            source=float(sun), r_occ=r_occ, r_source=SUN_RADIUS, shadow_model=shadow)
        for _ in range(int(round(CONV_TOTAL / dt))):
            sim.step(dt)
        out[dt] = sim.global_states[sat, :3] - sim.global_states[earth, :3]
    return out


@pytest.fixture(scope="module")
def convergence() -> Dict[str, Dict[float, ArrF]]:
    # Only the cylinder needs the full ladder: its asymptotic ratio is the measurement. The control
    # and the conical run are compared at the first step size alone, so they stop at 5 s.
    return {
        "none": _converge(0.0, SHADOW_MODEL_CYLINDRICAL, CONV_STEPS[:2]),
        "cylindrical": _converge(EARTH_R_EQ, SHADOW_MODEL_CYLINDRICAL),
        "conical": _converge(EARTH_R_EQ, SHADOW_MODEL_CONICAL, CONV_STEPS[:2]),
    }


def test_the_cylindrical_terminator_costs_rk4_its_convergence_order(
    convergence: Dict[str, Dict[float, ArrF]],
) -> None:
    """
    The cylindrical shadow makes the acceleration discontinuous, and an RK4 step that straddles the
    terminator is wrong by a first-order amount. This measures that, and shows the conical model
    recovers the smooth behaviour.

    Derivation. At the terminator the acceleration jumps by the full
    `Delta_a = C_r P (A/m) 1e-3 = 1.19e-9 km/s^2`. The four RK4 stages disagree about which side the
    body is on, so the switch is mistimed by up to `h`, giving a velocity error of about
    `Delta_a h / 2` per crossing, of essentially random sign, which then grows into position error
    linearly in the time remaining. Over this horizon there are 4 crossings and a mean remaining
    time of `T/2 = 4400 s`, so at `h = 10 s` the expected position error is roughly
    `4 x (1.19e-9)(5)(4400) = 1.0e-4 km` - **first order in `h`**, not fourth.

    The assertions are ratios rather than absolute values, because the per-crossing sign is
    effectively arbitrary:

    - the *asymptotic* step-halving difference ratio `|x(5) - x(2.5)| / |x(2.5) - x(1.25)|` must be
      **under 4** for the cylinder: a first-order error halves, a fourth-order one falls 16x. The
      first ratio in the sequence is not used, because RK4's own truncation is still decaying there
      and mixes the two orders;
    - the cylinder's `|x(10) - x(5)|` must exceed the no-shadow control's by at least 3x, i.e. the
      discontinuity is the dominant error source rather than a detail;
    - the conical model's must be within 2x of the no-shadow control: a continuous `nu` removes the
      penalty. That is the engineering reason to prefer it, not the penumbra's extra fidelity.
    """
    def diffs(key: str) -> Tuple[float, ...]:
        x = convergence[key]
        return tuple(float(np.linalg.norm(x[a] - x[b]))
                     for a, b in zip(CONV_STEPS, CONV_STEPS[1:]) if b in x)

    d10_none = diffs("none")[0]
    d10_cyl, d5_cyl, d25_cyl = diffs("cylindrical")
    d10_con = diffs("conical")[0]

    delta_a = CR * SOLAR_PRESSURE_1AU * AREA_MASS * 1e-3
    derived = 4.0 * delta_a * (10.0 / 2.0) * (CONV_TOTAL / 2.0)
    assert derived / 10.0 < d10_cyl < derived * 10.0, (
        f"crossing error {d10_cyl:.3e} km is not within a decade of the derived {derived:.3e} km")

    assert d5_cyl < 4.0 * d25_cyl, (
        f"the cylinder's error fell {d5_cyl / d25_cyl:.1f}x on halving h; a first-order error "
        f"falls 2x and a fourth-order one 16x, so this looks smooth and should not be")
    assert d10_cyl > 3.0 * d10_none, (d10_cyl, d10_none)
    assert d10_con < 2.0 * d10_none, (
        f"the conical shadow should cost about what no shadow costs: {d10_con:.3e} vs {d10_none:.3e}")


# ==================================================================================================
# 6. Configuration
# ==================================================================================================

_BASE_COEFFS = dict(cr=CR, area_mass=AREA_MASS, p_srp=SOLAR_PRESSURE_1AU,
                    r_occ=EARTH_R_EQ, r_source=SUN_RADIUS,
                    shadow_model=SHADOW_MODEL_CYLINDRICAL)


def _config_sim() -> Tuple[Simulation, Dict[str, int]]:
    sim = scenarios.sun_earth_moon(_session(), moon_mu=0.0, leo_satellite=True)
    return sim, dict(sim.name_to_index)


@pytest.mark.parametrize("override, source, match", [
    ({}, "__missing__", "needs a 'source'"),
    ({"p_srp": None}, "Sun", "needs a 'p_srp'"),
    ({"p_srp": 0.0}, "Sun", "must be positive"),
    ({"cr": 2.5}, "Sun", "physical range"),
    ({"cr": -0.1}, "Sun", "physical range"),
    ({"area_mass": -1.0}, "Sun", "must not be negative"),
    ({"r_occ": -1.0}, "Sun", "must not be negative"),
    ({"shadow_model": 2.0}, "Sun", "not a known shadow"),
    ({"shadow_model": SHADOW_MODEL_CONICAL, "r_source": 0.0}, "Sun", "positive 'r_source'"),
    ({}, "LEO-SAT", "lit by themselves"),
    ({}, "Earth", "Keplerian parent"),
    ({}, "EMB", "barycentre"),
    ({}, 2.5, "not an arena slot"),
    ({}, -1.0, "not an arena slot"),
])
def test_invalid_configuration_is_rejected_before_any_mutation(
    override: Dict[str, Optional[float]], source: object, match: str,
) -> None:
    sim, idx = _config_sim()
    coeffs: Dict[str, float] = dict(_BASE_COEFFS)
    for key, value in override.items():
        if value is None:
            coeffs.pop(key)
        else:
            coeffs[key] = value
    if source == "__missing__":
        pass
    elif isinstance(source, str):
        coeffs["source"] = float(idx[source])
    else:
        coeffs["source"] = float(source)  # type: ignore[arg-type]

    mask_before = sim.force_model_mask.copy()
    with pytest.raises(ValueError, match=match):
        sim.enable_force_model(SRP_MODEL, idx["LEO-SAT"], **coeffs)
    assert np.array_equal(sim.force_model_mask, mask_before)
    assert SRP_MODEL not in sim.force_model_params


def test_an_inactive_source_is_rejected() -> None:
    sim, idx = _config_sim()
    sim.active_mask[idx["Sun"]] = False
    with pytest.raises(ValueError, match="inactive"):
        sim.enable_force_model(SRP_MODEL, idx["LEO-SAT"], source=float(idx["Sun"]), **_BASE_COEFFS)


def test_a_shadow_needs_a_non_barycentre_parent_but_no_shadow_does_not() -> None:
    """The occulter is the Keplerian parent, so `r_occ > 0` needs one with a surface. Setting
    `r_occ = 0` is the documented escape, and it must still be accepted."""
    sim, idx = _config_sim()
    sat = idx["LEO-SAT"]
    sim.parent_indices[sat] = idx["EMB"]
    coeffs = dict(_BASE_COEFFS, source=float(idx["Sun"]))

    with pytest.raises(ValueError, match="cast a shadow"):
        sim.enable_force_model(SRP_MODEL, sat, **coeffs)
    assert SRP_MODEL not in sim.force_model_params

    sim.enable_force_model(SRP_MODEL, sat, **dict(coeffs, r_occ=0.0))
    bit = np.uint64(1) << np.uint64(registry.get_force_model(SRP_MODEL).bit)
    assert sim.force_model_mask[sat] & bit


def test_a_body_orbiting_the_source_may_have_srp_without_a_shadow() -> None:
    """Earth's parent *is* the Sun, so a shadow is nonsense there and is refused - but the model
    itself is not, once `r_occ` is zero. That is the heliocentric case."""
    sim, idx = _config_sim()
    with pytest.raises(ValueError, match="Keplerian parent"):
        sim.enable_force_model(SRP_MODEL, idx["Moon"], source=float(idx["Earth"]), **_BASE_COEFFS)
    sim.enable_force_model(SRP_MODEL, idx["Moon"],
                           **dict(_BASE_COEFFS, source=float(idx["Earth"]), r_occ=0.0))


def test_srp_is_registered_with_a_citation_and_is_foreign_to_the_fused_cowell_plan() -> None:
    """`kernels.cowell_rk4_step` fuses only `point_mass_gravity` and `j2`, so an `"srp"` body sends
    the whole Cowell set down the NumPy `RK4Integrator` path. Asserted, not assumed."""
    model = registry.get_force_model(SRP_MODEL)
    assert model.param_names == (
        "cr", "area_mass", "p_srp", "source", "r_occ", "r_source", "shadow_model")
    assert "Montenbruck" in model.citation and "unverified" in model.citation

    sim, idx = _config_sim()
    sat = idx["LEO-SAT"]
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sat)
    assert sim._cowell_fused_ok, "guard: point_mass_gravity alone does fuse"
    sim.enable_force_model(SRP_MODEL, sat, source=float(idx["Sun"]), **_BASE_COEFFS)
    assert not sim._cowell_fused_ok


def test_sweep_resolves_the_source_by_name() -> None:
    sim, idx = _config_sim()
    config = sweep.ModelConfig(
        name="leo+srp", propagator=PropagatorType.COWELL, dt=60.0, bodies=["LEO-SAT"],
        force_models=(
            sweep.ForceModelSpec(POINT_MASS_MODEL),
            sweep.ForceModelSpec(SRP_MODEL, coefficients=_BASE_COEFFS,
                                 body_coefficients={"source": "Sun"}),
        ),
    )
    applied = sweep.apply_config(sim, config)
    assert applied.tolist() == [idx["LEO-SAT"]]
    params = sim.force_model_params[SRP_MODEL]
    assert params[idx["LEO-SAT"], 3] == float(idx["Sun"])
    assert params[idx["LEO-SAT"], 0] == CR
    assert not sim._cowell_fused_ok

    bad = sweep.ModelConfig(
        name="bad", propagator=PropagatorType.COWELL, dt=60.0, bodies=["LEO-SAT"],
        force_models=(sweep.ForceModelSpec(SRP_MODEL, coefficients=_BASE_COEFFS,
                                           body_coefficients={"source": "Sirius"}),))
    with pytest.raises(KeyError, match="Sirius"):
        sweep.apply_config(_config_sim()[0], bad)


def test_drag_and_srp_compose_additively_through_the_simulation() -> None:
    """Both models on one body: `accelerations()` equals the sum of the two kernels called directly.
    Exact, because every kernel only adds."""
    sim, idx = _config_sim()
    sat = idx["LEO-SAT"]
    sim.set_propagator(np.array([sat], dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, sat)
    sim.enable_force_model(SRP_MODEL, sat, source=float(idx["Sun"]), **_BASE_COEFFS)
    sim.enable_force_model(DRAG_MODEL, sat, ballistic_coeff=CD * AREA_MASS, r_ref=EARTH_R_EQ,
                           omega=EARTH_OMEGA, density_model=DENSITY_MODEL_LAYERED)

    total = sim.accelerations(0.0).copy()
    expected = np.zeros_like(total)
    indices = np.array([sat], dtype=np.int64)
    for name, kernel in ((POINT_MASS_MODEL, registry.get_force_model(POINT_MASS_MODEL).kernel),
                         (SRP_MODEL, srp_kernel), (DRAG_MODEL, drag_kernel)):
        kernel(indices, 0.0, sim.global_states, sim.mu_array, sim.parent_indices,
               sim.force_model_params[name], expected)
    assert np.array_equal(total, expected)
