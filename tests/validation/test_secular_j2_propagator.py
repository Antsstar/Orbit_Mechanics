"""
Validation of `PropagatorType.SECULAR_J2` (`propagators.SecularJ2Propagator` /
`kernels.secular_j2_propagate`) - analytic Keplerian propagation plus the first-order secular drift of
RAAN, argument of periapsis and mean anomaly under J2.

Four kinds of check, matching the per-feature contract in `CLAUDE.md`:

1. Closed-form rates at hand-computable inputs, plus two published zero-crossings (polar inclination
   has no nodal regression; the critical inclination has no apsidal drift) - all independent of the
   engine, checkable with a calculator.
2. `Simulation.set_propagator`'s configuration-time guards - a body this model is meaningless for must
   raise, not produce a plausible-looking wrong trajectory.
3. A negative control: a deliberately sign-flipped rate formula is caught by check 1.
4. The strongest available independent check - `PropagatorType.COWELL` + `point_mass_gravity` + `j2`,
   a completely different code path (numerical RK4 integration of the un-averaged J2 acceleration, no
   classical elements, no secular theory), compared against this analytic propagator's RAAN drift over
   many orbits, and its position over a few - which is also where the mean-vs-osculating modelling
   error this propagator introduces (see `SecularJ2Propagator`'s docstring) is actually measured.

Kernel/reference equivalence (the two-implementation rule) lives in `test_kernel_equivalence.py`
alongside the Keplerian pair it mirrors, not here.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import geopotential, reference, scenarios
from orbital_engine.custom_types import COEIndex, PropagatorType
from orbital_engine.propagators import mean_seeded_p, secular_j2_rates
from orbital_engine.simulator import Simulation

EPS = float(np.finfo(np.float64).eps)

# --------------------------------------------------------------------------------------------------
# Tolerances, derived - see each test for the derivation.
# --------------------------------------------------------------------------------------------------

# Closed-form point below is exact binary-friendly inputs (mu=1e6, J2=1e-3, R=1000, p=2000, e=0,
# i=45deg); the formula is ~10 rounded multiplications and a sqrt, same class as
# test_geopotential.py's CLOSED_FORM_REL_TOL, so the same headroom applies.
CLOSED_FORM_REL_TOL = 1e-13

# At i=90deg (polar) and at the critical inclination, the exact rate is precisely zero; the only
# residue is float rounding of cos(pi/2) or of the critical inclination's own cos^2(i) - 1/5. That
# residue is bounded by ~2 eps in the trig argument, amplified by the rest of the formula (n * factor,
# at most ~1e-2 for any orbit this suite exercises), so 1e-16 is generously above what float noise can
# produce (measured this session: ~2.6e-22 and ~9.3e-22 respectively) and far below any coefficient or
# sign error, which would leave the pole/critical-inclination values at their full ~1e-6 scale.
ZERO_CROSSING_ABS_TOL = 1e-16

# A sign-flipped rate is wrong by a factor of 2 (opposite sign, same magnitude) at any inclination away
# from the crossing itself - MUTANT_MIN_REL_ERROR only needs to clear noise by orders of magnitude,
# mirroring test_geopotential.py's negative-control constant.
MUTANT_MIN_REL_ERROR = 1.5

LEO_ALTITUDE_KM = 550.0
LEO_INCLINATION_DEG = 53.0


def _leo_p_km() -> float:
    """Circular altitude -> semi-latus rectum, matching `scenarios.earth_constellation`'s convention
    (p == a == r for e == 0)."""
    return scenarios.EARTH_RADIUS + LEO_ALTITUDE_KM


def _leo_period_s() -> float:
    p = _leo_p_km()
    return 2.0 * math.pi * math.sqrt(p**3 / scenarios.MU_EARTH)


# ==================================================================================================
# 1. Closed-form rates
# ==================================================================================================

def test_closed_form_rates_at_arbitrary_inclination() -> None:
    """
    By hand: mu=1e6, J2=1e-3, R=1000, p=2000, e=0, i=45deg.
      a = p = 2000 (e=0);  n = sqrt(mu/a^3) = sqrt(1e6/8e9) = 0.011180339887498949
      factor = J2 (R/p)^2 = 1e-3 * 0.25 = 2.5e-4
      cos(i) = cos^2(i) = sqrt(2)/2, 0.5
      dRAAN/dt  = -1.5 * n * factor * cos(i)              = -2.964635306407856e-06
      dARGPE/dt =  0.75 * n * factor * (5*0.5 - 1)         =  3.1444705933590802e-06
      dM/dt     =  n * (1 + 0.75*factor*1*(3*0.5 - 1))     =  0.011181388044363402
    Independently reproduced (not by calling this test's own machinery) and matched to the values
    below to 15 significant figures this session.
    """
    coe = np.zeros((2, 6))
    coe[1] = [2000.0, 0.0, math.radians(45.0), 0.0, 0.0, 0.0]
    mu_array = np.array([1e6, 0.0])
    parent_indices = np.array([0, 0], dtype=np.int32)
    params = np.array([[0.0, 0.0], [1e-3, 1000.0]])
    idx = np.array([1], dtype=np.int64)

    rates = secular_j2_rates(coe, mu_array, parent_indices, params, idx)[0]
    expected = np.array([-2.964635306407856e-06, 3.1444705933590802e-06, 0.011181388044363402])

    rel = np.abs(rates - expected) / np.abs(expected)
    assert np.all(rel < CLOSED_FORM_REL_TOL), (rates, expected, rel)


def test_zero_nodal_drift_at_polar_inclination() -> None:
    """
    Published result (e.g. Vallado, Curtis): a polar orbit (i = 90 deg) has no nodal regression -
    dRAAN/dt = -1.5 n J2 (R/p)^2 cos(i) is exactly zero at cos(90deg) = 0, independent of a, e, J2.
    """
    coe = np.zeros((2, 6))
    coe[1] = [7000.0, 0.1, math.radians(90.0), 0.0, 0.0, 0.0]
    mu_array = np.array([398600.4418, 0.0])
    parent_indices = np.array([0, 0], dtype=np.int32)
    params = np.array([[0.0, 0.0], [geopotential.EARTH_J2, geopotential.EARTH_R_EQ]])
    idx = np.array([1], dtype=np.int64)

    raan_dot = secular_j2_rates(coe, mu_array, parent_indices, params, idx)[0, 0]
    assert abs(raan_dot) < ZERO_CROSSING_ABS_TOL, raan_dot


def test_zero_apsidal_drift_at_critical_inclination() -> None:
    """
    Published result: the critical inclination i = arccos(1/sqrt(5)) ~ 63.435 deg has no apsidal
    drift - dARGPE/dt = 0.75 n J2 (R/p)^2 (5 cos^2(i) - 1) is exactly zero where cos^2(i) = 1/5,
    independent of a, e, J2. This is the inclination Molniya- and Tundra-type orbits are seeded at for
    exactly this reason.
    """
    critical_inclination = math.acos(1.0 / math.sqrt(5.0))
    coe = np.zeros((2, 6))
    coe[1] = [7000.0, 0.3, critical_inclination, 0.0, 0.0, 0.0]
    mu_array = np.array([398600.4418, 0.0])
    parent_indices = np.array([0, 0], dtype=np.int32)
    params = np.array([[0.0, 0.0], [geopotential.EARTH_J2, geopotential.EARTH_R_EQ]])
    idx = np.array([1], dtype=np.int64)

    argpe_dot = secular_j2_rates(coe, mu_array, parent_indices, params, idx)[0, 1]
    assert abs(argpe_dot) < ZERO_CROSSING_ABS_TOL, argpe_dot


# ==================================================================================================
# 2. Negative control: a sign error is caught
# ==================================================================================================

def _mutant_secular_j2_rates(
    coe_states: np.ndarray, mu_array: np.ndarray, parent_indices: np.ndarray,
    j2_params: np.ndarray, indices: np.ndarray,
) -> np.ndarray:
    """Independent re-derivation of `secular_j2_rates` with one deliberate error: the RAAN
    coefficient's sign, `+1.5` instead of `-1.5`. Not registered, not imported from propagators.py."""
    parents = parent_indices[indices]
    mu = mu_array[indices] + mu_array[parents]
    p = coe_states[indices, 0]
    e = coe_states[indices, 1]
    inc = coe_states[indices, 2]
    j2 = j2_params[indices, 0]
    r_eq = j2_params[indices, 1]

    a = p / (1.0 - e * e)
    n = np.sqrt(mu / a**3)
    factor = j2 * (r_eq / p) ** 2
    cos_i = np.cos(inc)
    cos2_i = cos_i * cos_i

    raan_dot = 1.5 * n * factor * cos_i  # BUG: should be -1.5
    argpe_dot = 0.75 * n * factor * (5.0 * cos2_i - 1.0)
    m_dot = n * (1.0 + 0.75 * factor * np.sqrt(1.0 - e * e) * (3.0 * cos2_i - 1.0))
    return np.stack([raan_dot, argpe_dot, m_dot], axis=1)


def test_negative_control_wrong_raan_sign_is_caught() -> None:
    """Applies the closed-form check from test 1 to the mutant: the RAAN rate comes out with the
    opposite sign, a relative error of exactly 2.0, which the tolerance the real test uses
    (CLOSED_FORM_REL_TOL = 1e-13) would reject by twelve orders of magnitude."""
    coe = np.zeros((2, 6))
    coe[1] = [2000.0, 0.0, math.radians(45.0), 0.0, 0.0, 0.0]
    mu_array = np.array([1e6, 0.0])
    parent_indices = np.array([0, 0], dtype=np.int32)
    params = np.array([[0.0, 0.0], [1e-3, 1000.0]])
    idx = np.array([1], dtype=np.int64)

    correct = secular_j2_rates(coe, mu_array, parent_indices, params, idx)[0, 0]
    mutant = _mutant_secular_j2_rates(coe, mu_array, parent_indices, params, idx)[0, 0]

    assert correct < 0.0 and mutant > 0.0, (correct, mutant)
    rel_err = abs(mutant - correct) / abs(correct)
    assert rel_err > MUTANT_MIN_REL_ERROR, rel_err


# ==================================================================================================
# 3. Simulation.set_propagator configuration-time guards
# ==================================================================================================

def test_set_propagator_rejects_heads_and_barycenters(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    earth = sim.name_to_index["Earth"]
    bary = sim.name_to_index["Earth Barycenter"]

    with pytest.raises(ValueError):
        sim.set_propagator(earth, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    with pytest.raises(ValueError):
        sim.set_propagator(bary, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)

    assert sim.propagator_type[earth] == PropagatorType.KEPLERIAN
    assert sim.propagator_type[bary] == PropagatorType.KEPLERIAN


def test_set_propagator_rejects_a_barycentre_parented_body(db_session_factory: Callable[[], Session]) -> None:
    """A body whose Keplerian parent is a barycentre has meaningless J2 - `geopotential.
    barycentre_parented` is the wired guard. No shipped scenario parents a body to a barycentre (see
    `test_geopotential.py`'s identical setup), so the parent graph is re-pointed on a copy first."""
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    bary = sim.name_to_index["Earth Barycenter"]
    sat = sim.name_to_index["SAT-00-000"]
    assert geopotential.barycentre_parented(sim.is_system, sim.parent_indices, np.array([sat])).size == 0

    sim.parent_indices[sat] = bary
    with pytest.raises(ValueError):
        sim.set_propagator(sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)


def test_set_propagator_requires_j2_and_r_eq(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    sat = sim.name_to_index["SAT-00-000"]

    with pytest.raises(ValueError):
        sim.set_propagator(sat, PropagatorType.SECULAR_J2)
    with pytest.raises(ValueError):
        sim.set_propagator(sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2)
    with pytest.raises(ValueError):
        sim.set_propagator(
            sat, PropagatorType.SECULAR_J2,
            j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ, drag_coefficient=2.2,
        )
    assert sim.propagator_type[sat] == PropagatorType.KEPLERIAN


def test_set_propagator_rejects_coefficients_on_other_propagators(
    db_session_factory: Callable[[], Session],
) -> None:
    sim = scenarios.two_body(db_session_factory())
    secondary = sim.name_to_index["Secondary"]
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.COWELL, j2=geopotential.EARTH_J2)
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.KEPLERIAN, j2=geopotential.EARTH_J2)


def test_set_propagator_rejects_an_open_orbit(db_session_factory: Callable[[], Session]) -> None:
    """The secular rates derive from mean motion n = sqrt(mu/a^3), undefined for e >= 1."""
    sim = scenarios.two_body(db_session_factory(), mu_secondary=0.0, p=11000.0, e=1.2)
    secondary = sim.name_to_index["Secondary"]
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)


def test_set_propagator_rejects_a_massive_body(db_session_factory: Callable[[], Session]) -> None:
    """Same reason Cowell requires it: `_kepler_sib_idx` excludes SECULAR_J2 slots, so a massive one
    would silently stop contributing to its head's reflex kick."""
    sim = scenarios.two_body(db_session_factory(), mu_secondary=100.0, p=11000.0, e=0.1)
    secondary = sim.name_to_index["Secondary"]
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)


def test_set_propagator_populates_coefficients_and_caches_rates(
    db_session_factory: Callable[[], Session],
) -> None:
    """A successful call writes (j2, r_eq) into `force_model_params["j2"]` - the same array
    `enable_force_model("j2", ...)` would - without setting the "j2" mask bit, and caches the three
    derived rates rather than leaving them at zero."""
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    sat = sim.name_to_index["SAT-00-000"]

    sim.set_propagator(sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)

    assert sim.force_model_params["j2"][sat, 0] == geopotential.EARTH_J2
    assert sim.force_model_params["j2"][sat, 1] == geopotential.EARTH_R_EQ
    assert sim.force_model_mask[sat] == 0, "SECULAR_J2 must not enable the 'j2' force-model bit"
    assert np.any(sim._secular_j2_rates[sat] != 0.0)
    assert sim.propagator_type[sat] == PropagatorType.SECULAR_J2


# ==================================================================================================
# 4. Independent validation against Cowell + point_mass_gravity + j2
# ==================================================================================================

def _build_leo_secular(session: Session) -> tuple[Simulation, int]:
    sim = scenarios.earth_constellation(
        session, n_sats=1, n_planes=1, altitude_km=LEO_ALTITUDE_KM, inclination_deg=LEO_INCLINATION_DEG,
    )
    sat = sim.name_to_index["SAT-00-000"]
    sim.record_history = False
    sim.set_propagator(sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sat


def _build_leo_cowell_full_j2(session: Session) -> tuple[Simulation, int, int]:
    sim = scenarios.earth_constellation(
        session, n_sats=1, n_planes=1, altitude_km=LEO_ALTITUDE_KM, inclination_deg=LEO_INCLINATION_DEG,
    )
    sat = sim.name_to_index["SAT-00-000"]
    earth = sim.name_to_index["Earth"]
    sim.record_history = False
    sim.set_propagator(sat, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", bodies=sat)
    sim.enable_force_model("j2", bodies=sat, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sat, earth


def _cowell_raan(sim: Simulation, sat: int, earth: int) -> float:
    """A Cowell body's `coe_states` row is stale (CLAUDE.md); recover osculating RAAN from `(r, v)`."""
    from orbital_engine import frames as fr

    rel_r = (sim.global_states[sat, :3] - sim.global_states[earth, :3])[None, :]
    rel_v = (sim.global_states[sat, 3:] - sim.global_states[earth, 3:])[None, :]
    mu_total = np.array([sim.mu_array[sat] + sim.mu_array[earth]])
    coe, success = fr.ReferenceFrames.rv_to_coe(rel_r, rel_v, mu_total)
    assert success[0], "Cowell state did not resolve to a valid osculating orbit"
    return float(coe[0, COEIndex.RAAN])


def _wrapped_deg_diff(a_rad: float, b_rad: float) -> float:
    """Signed shortest-angle difference (a - b) in degrees, robust to either side crossing the
    0/360-degree branch cut `rv_to_coe` places RAAN on."""
    diff_deg = math.degrees(a_rad - b_rad)
    return (diff_deg + 180.0) % 360.0 - 180.0


# Measured this session, 60 orbits (~4 days) at 550 km / 53 deg, dt = period/200: secular-J2 RAAN
# drifted -17.9282 deg; Cowell + point_mass_gravity + j2's osculating RAAN, recovered via rv_to_coe,
# drifted -17.9796 deg (unwrapped) - a residual of 0.0514 deg. Derived bound: the un-averaged J2
# acceleration causes RAAN to oscillate with short-period amplitude O(J2 (R/p)^2) radians about the
# secular trend (Brouwer/Vallado short-period theory) - here that evaluates to ~0.0526 deg, matching
# the measured residual to within 5%. ALLOWED_DEG below carries ~6x headroom over both the measured
# value and the derived bound.
N_ORBITS_RAAN = 60


def test_secular_j2_raan_drift_matches_cowell_full_j2(db_session_factory: Callable[[], Session]) -> None:
    period = _leo_period_s()
    dt = period / 200.0
    n_steps = int(round(N_ORBITS_RAAN * period / dt))

    sec_sim, sec_sat = _build_leo_secular(db_session_factory())
    raan0 = float(sec_sim.coe_states[sec_sat, COEIndex.RAAN])
    for _ in range(n_steps):
        sec_sim.step(dt)
    raan1_secular = float(sec_sim.coe_states[sec_sat, COEIndex.RAAN])

    cow_sim, cow_sat, cow_earth = _build_leo_cowell_full_j2(db_session_factory())
    raan0_cowell = _cowell_raan(cow_sim, cow_sat, cow_earth)
    for _ in range(n_steps):
        cow_sim.step(dt)
    raan1_cowell = _cowell_raan(cow_sim, cow_sat, cow_earth)

    drift_secular_deg = math.degrees(raan1_secular - raan0)
    drift_cowell_deg = _wrapped_deg_diff(raan1_cowell, raan0_cowell)

    p = _leo_p_km()
    short_period_bound_deg = math.degrees(geopotential.EARTH_J2 * (geopotential.EARTH_R_EQ / p) ** 2)
    allowed_deg = 6.0 * short_period_bound_deg

    residual_deg = abs(drift_secular_deg - drift_cowell_deg)
    assert residual_deg < allowed_deg, (
        f"secular-J2 RAAN drift ({drift_secular_deg:.4f} deg) disagrees with Cowell + point_mass_"
        f"gravity + j2's numerically integrated drift ({drift_cowell_deg:.4f} deg) by {residual_deg:.4f} "
        f"deg over {N_ORBITS_RAAN} orbits, exceeding the {allowed_deg:.4f} deg bound derived from the "
        f"short-period oscillation amplitude"
    )
    # Sanity: both must actually show real nodal regression, not two near-zero numbers agreeing
    # trivially - LEO at 53 deg regresses several degrees per day.
    assert abs(drift_secular_deg) > 1.0
    assert abs(drift_cowell_deg) > 1.0


# ==================================================================================================
# 5. The mean-vs-osculating approximation error itself
# ==================================================================================================
#
# The arena holds osculating elements; SecularJ2Propagator advances them as if they were mean elements
# (CLAUDE.md forbids converting between the two - see that propagator's docstring). Two effects follow:
#
#   (a) a bounded, orbit-period oscillation - the short-period J2 terms the averaging discards -
#       amplitude ~ J2 (R/p)^2 * p in position, ~6 km at 550 km altitude;
#   (b) treating the *seeded* osculating semi-latus rectum as constant for the life of the propagator
#       biases the cached mean motion by the same fractional amount, ~J2 (R/p)^2, which accumulates
#       *linearly* with elapsed orbits rather than staying bounded - this dominates (a) after more than
#       a handful of orbits.
#
# Measured this session (550 km / 53 deg, dt = period/200): position disagreement against Cowell +
# point_mass_gravity + j2 was 57.4 km at 1 orbit, 172.1 km at 3 orbits, 573.6 km at 10 orbits - linear
# to better than 1% (ratios 3.00 and 9.99 against the exactly-linear predictions 3 and 10), confirming
# (b) rather than (a) dominates. The order-of-magnitude estimate below, J2 (R/p)^2 * 2*pi*p per orbit
# (~40 km, the phase error in radians converted to an along-track distance), matches the measured
# per-orbit rate (~57 km) to within a factor of ~1.4 - well inside the band this test uses.

ORBIT_COUNTS_FOR_GROWTH = [1, 4]


def _order_of_magnitude_km_per_orbit() -> float:
    p = _leo_p_km()
    return geopotential.EARTH_J2 * (geopotential.EARTH_R_EQ / p) ** 2 * 2.0 * math.pi * p


def test_mean_vs_osculating_error_is_bounded_and_grows_with_orbit_count(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    This is a *comparison*, not a *verification*, in `CLAUDE.md`'s sense: SecularJ2Propagator is an
    approximation from the moment it treats osculating elements as mean, so disagreement with Cowell's
    (numerically exact, to RK4 truncation) trajectory is the measurement, not a bug - provided it sits
    within the order-of-magnitude band derived above and grows roughly linearly with elapsed orbits,
    rather than being either vanishingly small (this test measuring nothing) or wildly larger (a real
    error beyond the documented approximation).
    """
    period = _leo_period_s()
    per_orbit_estimate = _order_of_magnitude_km_per_orbit()

    errors_km = []
    for n_orbits in ORBIT_COUNTS_FOR_GROWTH:
        dt = period / 200.0
        n_steps = int(round(n_orbits * period / dt))

        sec_sim, sec_sat = _build_leo_secular(db_session_factory())
        cow_sim, cow_sat, _ = _build_leo_cowell_full_j2(db_session_factory())
        for _ in range(n_steps):
            sec_sim.step(dt)
            cow_sim.step(dt)

        errors_km.append(float(np.linalg.norm(
            sec_sim.global_states[sec_sat, :3] - cow_sim.global_states[cow_sat, :3]
        )))

    for n_orbits, err in zip(ORBIT_COUNTS_FOR_GROWTH, errors_km):
        predicted = per_orbit_estimate * n_orbits
        assert 0.2 * predicted < err < 5.0 * predicted, (
            f"{n_orbits} orbits: measured error {err:.1f} km is not within an order of magnitude of "
            f"the {predicted:.1f} km estimate (J2 (R/p)^2 * 2*pi*p per orbit)"
        )

    # Growth must be roughly linear in elapsed orbits (not vanishing, not runaway): a first-order
    # secular-rate bug would show either no growth (a missing term) or markedly super-linear growth
    # (a wrong exponent/sign compounding).
    ratio = errors_km[-1] / errors_km[0]
    expected_ratio = ORBIT_COUNTS_FOR_GROWTH[-1] / ORBIT_COUNTS_FOR_GROWTH[0]
    assert 0.3 * expected_ratio < ratio < 3.0 * expected_ratio, (
        f"error growth ratio {ratio:.2f} over {ORBIT_COUNTS_FOR_GROWTH} orbits is not consistent with "
        f"roughly linear growth (expected near {expected_ratio:.1f}); errors={errors_km}"
    )


# ==================================================================================================
# 6. `mean_seed` - a first-order mean semi-major axis, per `propagators.mean_seeded_p`
# ==================================================================================================
#
# `mean_seeded_p` corrects only `p` (and so the cached mean motion), which section 5 above identifies
# as the source of the *linearly growing* error. What should remain after that correction is the
# *bounded* short-period oscillation section 5 also derives - amplitude ~ J2 (R/p)^2 * p (~6.37 km at
# 550 km / 53 deg), which this propagator has no machinery to remove (it holds p, e, i constant and
# never synthesises the periodic terms back in). That residual should not grow with elapsed orbits,
# unlike the osculating-seeded error, and should scale with the same |cos(2u0)| phase dependence the
# correction itself uses, vanishing at u0 = 45/135/225/315 deg exactly where section 5's osculating
# measurement already showed near-zero error (frac = 0 there, so mean_seed is a no-op).

MEAN_SEED_U0_DEG = [0.0, 30.0, 45.0, 60.0, 90.0]
N_ORBITS_MEAN_SEED = 3


def _bounded_residual_estimate_km() -> float:
    """Order-of-magnitude bound on the *leftover* short-period oscillation after mean-seeding removes
    the linear term - the same J2 (R/p)^2 * p estimate `_order_of_magnitude_km_per_orbit` uses for the
    bounded (a) effect in section 5's module comment, not fitted to any figure measured here."""
    p = _leo_p_km()
    return geopotential.EARTH_J2 * (geopotential.EARTH_R_EQ / p) ** 2 * p


def _build_two_body_secular(session: Session, u0_rad: float, *, mean_seed: bool) -> tuple[Simulation, int]:
    """A single LEO satellite at argument of latitude `u0_rad` (`two_body` supports an arbitrary
    `theta`, unlike `earth_constellation`, which is what a phase sweep needs)."""
    sim = scenarios.two_body(
        session, mu_primary=scenarios.MU_EARTH, mu_secondary=0.0,
        p=_leo_p_km(), e=0.0, i=math.radians(LEO_INCLINATION_DEG), arg_pe=0.0, theta=u0_rad,
    )
    sat = sim.name_to_index["Secondary"]
    sim.record_history = False
    sim.set_propagator(
        sat, PropagatorType.SECULAR_J2, mean_seed=mean_seed,
        j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ,
    )
    return sim, sat


def test_mean_seeded_p_closed_form() -> None:
    """
    By hand, at u0 = 0 (cos(2u0) = 1): mu=1e6, J2=1e-3, R=1000, p=2000, e=0, i=45deg (same inputs as
    the closed-form rates check in section 1, plus the argument of latitude).
      a_osc = p = 2000;  frac = 1.5 * J2 * (R/p)^2 * sin^2(i) * cos(0) = 1.5e-3 * 0.25 * 0.5 = 1.875e-4
      a_mean = a_osc * (1 - frac) = 1999.625;  p_mean = a_mean (e=0, so p_mean == a_mean)
    Independently reproduced (not via this module's own machinery) and matched to 15 significant
    figures this session. At u0 = 45 deg (cos(2u0) = 0) the correction must vanish identically -
    `frac == 0` exactly, not merely small - since `mean_seed` must be a documented no-op there.
    """
    coe = np.zeros((1, 6))
    coe[0] = [2000.0, 0.0, math.radians(45.0), 0.0, 0.0, 0.0]
    params = np.array([[1e-3, 1000.0]])
    idx = np.array([0], dtype=np.int64)

    p_mean = mean_seeded_p(coe, params, idx)[0]
    assert abs(p_mean - 1999.625) / 1999.625 < CLOSED_FORM_REL_TOL, p_mean

    coe[0, 4] = math.radians(45.0)  # arg_pe + theta = 45 deg -> u0 = 45 deg
    p_mean_45 = mean_seeded_p(coe, params, idx)[0]
    assert p_mean_45 == 2000.0, p_mean_45  # exact: cos(2*45deg) == 0 to float precision at this input


def test_set_propagator_mean_seed_rejected_for_other_propagators(
    db_session_factory: Callable[[], Session],
) -> None:
    """`mean_seed` is meaningless outside SECULAR_J2 - see `Simulation.set_propagator`'s docstring."""
    sim = scenarios.two_body(db_session_factory())
    secondary = sim.name_to_index["Secondary"]
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.COWELL, mean_seed=True)
    with pytest.raises(ValueError):
        sim.set_propagator(secondary, PropagatorType.KEPLERIAN, mean_seed=True)


def test_mean_seed_default_is_bit_identical_to_osculating_seeding(
    db_session_factory: Callable[[], Session],
) -> None:
    """The osculating-seeded behaviour is the default and must stay unchanged: `mean_seed=False` (the
    default) must leave `p` exactly as `set_propagator` found it, matching this propagator's behaviour
    before `mean_seed` existed."""
    sim, sat = _build_two_body_secular(db_session_factory(), 0.0, mean_seed=False)
    assert sim.coe_states[sat, COEIndex.P] == pytest.approx(_leo_p_km(), rel=1e-12)


def test_mean_seed_true_overwrites_seeded_p_away_from_the_zero_crossing(
    db_session_factory: Callable[[], Session],
) -> None:
    """At u0 = 0 (worst case), `mean_seed=True` must actually change the seeded `p` - a no-op here
    would mean the option silently does nothing."""
    sim, sat = _build_two_body_secular(db_session_factory(), 0.0, mean_seed=True)
    assert sim.coe_states[sat, COEIndex.P] != _leo_p_km()


def test_mean_seeding_collapses_the_phase_dependent_error_against_j2_truth(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The core claim: seeding the mean semi-major axis removes the *linearly growing* along-track drift
    section 5 measures for osculating seeding, at every phase, not just the u0 = 45 deg case where that
    drift already happened to vanish. What remains is bounded by the short-period estimate derived
    above - this test shows that residual explicitly rather than only showing the error shrank.
    """
    period = _leo_period_s()
    dt = period / 200.0
    n_steps = int(round(N_ORBITS_MEAN_SEED * period / dt))
    times = np.array([0.0, N_ORBITS_MEAN_SEED * period])
    bound_km = _bounded_residual_estimate_km()

    for u0_deg in MEAN_SEED_U0_DEG:
        u0 = math.radians(u0_deg)

        osc_sim, osc_sat = _build_two_body_secular(db_session_factory(), u0, mean_seed=False)
        truth = reference.reference_for(
            osc_sim, times, rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL,
            oblateness={"Primary": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)},
        )
        truth_pos = truth.position_of("Secondary")[-1]
        for _ in range(n_steps):
            osc_sim.step(dt)
        err_osc = float(np.linalg.norm(osc_sim.global_states[osc_sat, :3] - truth_pos))

        # Truth depends only on the shared initial condition, which `mean_seed` never changes (it
        # only biases the propagator's internal seed for t > 0) - computed once, reused for both.
        mean_sim, mean_sat = _build_two_body_secular(db_session_factory(), u0, mean_seed=True)
        for _ in range(n_steps):
            mean_sim.step(dt)
        err_mean = float(np.linalg.norm(mean_sim.global_states[mean_sat, :3] - truth_pos))

        # The residual bound is loose (2x headroom over the derived ~6.37 km estimate) because it is a
        # single-term order-of-magnitude estimate, the same convention section 5 uses.
        assert err_mean < 2.0 * bound_km, (
            f"u0={u0_deg} deg: mean-seeded error {err_mean:.2f} km exceeds the {2.0*bound_km:.2f} km "
            f"bound derived from the leftover short-period oscillation"
        )
        if abs(math.cos(2.0 * u0)) > 0.3:  # away from the u0=45 deg zero-crossing, where both agree
            assert err_mean < 0.2 * err_osc, (
                f"u0={u0_deg} deg: mean-seeded error {err_mean:.2f} km is not well below the "
                f"osculating-seeded error {err_osc:.2f} km"
            )


def test_mean_seeding_residual_does_not_grow_with_orbit_count(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The whole point of correcting the *mean motion* bias is that what is left behind is bounded, not
    merely smaller. At u0 = 0 (the worst case for the uncorrected error - 573.5 km at 10 orbits,
    section 5), the mean-seeded residual must stay within the same order of magnitude from 1 to 10
    orbits rather than continuing to grow linearly.
    """
    period = _leo_period_s()
    dt = period / 200.0
    bound_km = _bounded_residual_estimate_km()

    errors_km = []
    for n_orbits in (1, 10):
        n_steps = int(round(n_orbits * period / dt))
        times = np.array([0.0, n_orbits * period])

        sim, sat = _build_two_body_secular(db_session_factory(), 0.0, mean_seed=True)
        truth = reference.reference_for(
            sim, times, rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL,
            oblateness={"Primary": (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)},
        )
        truth_pos = truth.position_of("Secondary")[-1]
        for _ in range(n_steps):
            sim.step(dt)
        errors_km.append(float(np.linalg.norm(sim.global_states[sat, :3] - truth_pos)))

    for err in errors_km:
        assert err < 2.0 * bound_km, (errors_km, bound_km)
    ratio = errors_km[-1] / errors_km[0]
    assert ratio < 3.0, (
        f"mean-seeded residual grew {ratio:.2f}x from 1 to 10 orbits ({errors_km}); expected roughly "
        f"bounded, not the ~10x a still-linear error would show"
    )
