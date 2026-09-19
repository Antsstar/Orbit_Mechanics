"""
Validation of impulsive manoeuvres (`manoeuvres.py`, `Simulation.apply_delta_v` /
`schedule_delta_v`) - an instantaneous Delta-v, in RSW, under every propagator.

The headline case is a **Hohmann transfer**, because it is the manoeuvre with exact closed-form
answers for both burns, for the intermediate orbit and for the transfer time, so the engine can be
checked against arithmetic instead of against itself.

Derivation of everything asserted below (vis-viva, `v^2 = mu (2/r - 1/a)`; no text needed)
------------------------------------------------------------------------------------------
Circular orbit `r1`, target circular orbit `r2`, transfer ellipse with `a_t = (r1 + r2)/2` touching
both:

    v_c1   = sqrt(mu / r1)                                  circular speed at r1
    v_t1   = sqrt(mu (2/r1 - 2/(r1 + r2))) = v_c1 sqrt(2 r2 / (r1 + r2))
    dv1    = v_t1 - v_c1 = sqrt(mu/r1) (sqrt(2 r2/(r1 + r2)) - 1)
    dv2    = sqrt(mu/r2) (1 - sqrt(2 r1/(r1 + r2)))
    e_t    = (r2 - r1) / (r2 + r1),   p_t = a_t (1 - e_t^2),   r_apo = p_t/(1 - e_t) = r2
    t_t    = pi sqrt(a_t^3 / mu)                            half the transfer period

Both burns are prograde and both are at an apsis, where the flight-path angle is zero and the RSW
along-track axis S *is* the velocity direction - so `(0, dv, 0)` is exactly the scalar `dv` above.
That is only true at an apsis; see `frames.RSW_basis`.

The engine is never asked for these values. They are recomputed here from `mu`, `r1` and `r2` alone.

Expected error magnitudes, and where each comes from
----------------------------------------------------
- **Elements straight after a burn** (`test_hohmann_first_impulse_matches_closed_form`): the impulse
  is one vector addition and `rv_to_coe` is ~20 arithmetic operations, so the answer is exact to a
  few eps. Derived: `~1e-15` relative. Measured 5.6e-16 (e), 7.6e-16 (p), 2.6e-15 (apoapsis). Asserted
  at `1e-12`, three orders of headroom over the measurement and nine below any plausible error.
- **Keplerian transfer** (`test_hohmann_keplerian_reaches_the_target_orbit`): analytic propagation is
  exact, so the residue is the Kepler solver's. `Anomalies`/`kernels.solve_kepler_scalar` stop when
  the Newton correction falls below `1e-5`; Newton is quadratically convergent, so the *anomaly* is
  then right to `O(1e-10)` rad per solve. A phase error `dth` at the transfer apoapsis means the
  circularisation burn is not quite at the apsis, leaving a radial velocity `(mu/h) e dth` and so a
  residual eccentricity `~(mu/h) e dth / v_c2 = 1.3 dth`. Over the ~320 solves of the transfer, summed
  incoherently, that bounds `e_res` by `sqrt(320) * 1.3e-10 ~ 2.4e-9` and the radius error by
  `r2 e_res ~ 1e-4 km`. Measured `e_res = 3.6e-10` and `|r| - r2 = -3.2e-6 km`.
- **Cowell transfer** (`test_hohmann_cowell_matches_the_analytic_transfer`): RK4's own truncation, and
  nothing else - the impulse is exact for both propagators. RK4's local phase error per step is
  `(w h)^5 / 120` for instantaneous angular rate `w`, so the accumulated phase over the transfer is
  `(h^4/120) INT w^5 dt = (h^4/120) (H^4/p_t^8) INT_0^pi (1 + e_t cos th)^8 dth` with `H = sqrt(mu p_t)`
  the angular momentum (using `dth = w dt`, `w = H/r^2`, `r = p/(1 + e cos th)`). The integral is 50.96
  for this transfer, and the perigee end dominates it - the instantaneous rate there is 8.6x the mean
  motion, which is why using `n` instead would under-predict by four orders. That gives `1.2e-2 km` at
  `dt = 60 s`; measured `9.8e-2 km`, 8.0x the estimate, the order-unity constants dropped along the
  way (the `1/120` is the principal-error coefficient of a scalar linear model, and an accumulated
  energy error amplifies the along-track term further). The assertion is therefore an order of
  magnitude either side of the estimate **plus** the property that pins the disagreement as RK4
  truncation rather than a manoeuvre error: it falls by 16x per halving of `dt`.
- **Secular-J2 nodal rate** (`test_impulse_changes_the_secular_j2_nodal_rate`): derived in that test's
  docstring. A prograde kick `dv = d * v_c` on a circular orbit gives, exactly,
  `p -> r (1+d)^2`, `a -> r / (1 - 2d - d^2)`, `e -> 2d + d^2`, so the nodal rate
  `-1.5 n J2 (R/p)^2 cos i` scales by `(1 - 2d - d^2)^{3/2} (1 + d)^{-4} = 1 - 7d + 22 d^2 + ...`.
  A stale cached rate gives exactly 1.0 and fails by `7d`.
- **Step splitting** (`test_split_step_does_not_disturb_a_non_manoeuvring_body`): derived there.

Negative controls run by hand against this suite (mutate `manoeuvres.py`, run, restore):
skipping the element re-derivation for analytic bodies, and applying the Delta-v in the inertial frame
as though it were RSW. Both results are recorded in `docs/engineering-log.md`.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import geopotential, manoeuvres, scenarios
from orbital_engine.custom_types import COEIndex, PropagatorType
from orbital_engine.frames import ReferenceFrames
from orbital_engine.simulator import Simulation

# --------------------------------------------------------------------------------------------------
# The closed form. Recomputed from mu, r1, r2 - never read back from the engine.
# --------------------------------------------------------------------------------------------------
MU = scenarios.MU_EARTH
R1 = 7000.0                     # km, departure circular radius (LEO-ish)
R2 = 42164.0                    # km, target circular radius (geostationary)

A_T = 0.5 * (R1 + R2)
E_T = (R2 - R1) / (R2 + R1)
P_T = A_T * (1.0 - E_T * E_T)
DV1 = math.sqrt(MU / R1) * (math.sqrt(2.0 * R2 / (R1 + R2)) - 1.0)
DV2 = math.sqrt(MU / R2) * (1.0 - math.sqrt(2.0 * R1 / (R1 + R2)))
T_TRANSFER = math.pi * math.sqrt(A_T**3 / MU)

DT = 60.0                       # s, the reference step size for the transfer runs

# --------------------------------------------------------------------------------------------------
# Derived tolerances. Each is justified in the module docstring; the measurement is in the comment.
# --------------------------------------------------------------------------------------------------

# One vector addition plus rv_to_coe's ~20 operations. Derived ~1e-15; measured 5.6e-16 to 2.6e-15.
ELEMENT_REL_TOL = 1e-12

# sqrt(320) solves x 1.3 x O(solver tol^2 = 1e-10) rad. Derived 2.4e-9; measured 3.6e-10.
CIRCULARISATION_E_TOL = 1e-8
# r2 * the above. Derived ~1e-4 km; measured 3.2e-6 km.
CIRCULARISATION_R_TOL_KM = R2 * CIRCULARISATION_E_TOL

# RK4 truncation over the transfer, from the local-error integral in the module docstring.
_H_ANG = math.sqrt(MU * P_T)
_INT_1_PLUS_E_COS_8 = 50.962025      # INT_0^pi (1 + e_t cos th)^8 dth for e_t = 0.7152..., by quadrature


def rk4_transfer_error_km(dt: float) -> float:
    """Derived RK4 position error over the Hohmann transfer at step `dt`. See the module docstring."""
    phase = (dt**4 / 120.0) * (_H_ANG**4 / P_T**8) * _INT_1_PLUS_E_COS_8
    return R2 * phase


# The estimate drops order-unity constants; it is asserted to within a decade either way, with the
# dt^4 convergence below carrying the weight. Measured ratio 8.0 at dt = 60 s.
TRUNCATION_ESTIMATE_BAND = (0.1, 10.0)


def _transfer_profile(sim: Simulation) -> tuple[int, int, int, int]:
    """Schedule both Hohmann burns on the two `-SAT` vessels and return the four vessel slots."""
    ks = sim.name_to_index["KEPLER-SAT"]
    kt = sim.name_to_index["KEPLER-TWIN"]
    cs = sim.name_to_index["COWELL-SAT"]
    ct = sim.name_to_index["COWELL-TWIN"]
    sim.schedule_delta_v([ks, cs], (0.0, DV1, 0.0), 0.0, label="departure")
    sim.schedule_delta_v([ks, cs], (0.0, DV2, 0.0), T_TRANSFER, label="circularisation")
    return ks, kt, cs, ct


def _fly(sim: Simulation, dt: float, until: float) -> None:
    for _ in range(int(math.ceil(until / dt))):
        sim.step(dt)


def _relative(sim: Simulation, slot: int) -> np.ndarray:
    """State of `slot` relative to its Keplerian parent - the frame every element here is measured in."""
    rel: np.ndarray = sim.global_states[slot] - sim.global_states[sim.parent_indices[slot]]
    return rel


def _elements(sim: Simulation, slot: int) -> np.ndarray:
    """Elements re-derived from the Cartesian state, so a Cowell body (whose `coe_states` row is
    deliberately stale) is read the same way a Keplerian one is."""
    rel = _relative(sim, slot)
    coe, valid = ReferenceFrames.rv_to_coe(rel[:3], rel[3:], MU)
    assert bool(np.all(valid))
    return coe


# ==================================================================================================
# 1. The closed form, one burn at a time
# ==================================================================================================

def test_hohmann_first_impulse_matches_closed_form(db_session_factory: Callable[[], Session]) -> None:
    """
    After `dv1`, the intermediate orbit must *be* the Hohmann transfer ellipse: `e = e_t`, `p = p_t`,
    and an apoapsis of exactly `r2`. This is the element re-derivation on the Keplerian path - without
    it, `coe_states` still describes the departure circle and the next `step()` silently puts the
    vessel back on it.
    """
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    ks = sim.name_to_index["KEPLER-SAT"]

    sim.apply_delta_v(ks, (0.0, DV1, 0.0))
    coe = sim.coe_states[ks]
    e, p = float(coe[COEIndex.E]), float(coe[COEIndex.P])

    assert e == pytest.approx(E_T, rel=ELEMENT_REL_TOL)
    assert p == pytest.approx(P_T, rel=ELEMENT_REL_TOL)
    assert p / (1.0 - e) == pytest.approx(R2, rel=ELEMENT_REL_TOL)          # apoapsis
    assert p / (1.0 + e) == pytest.approx(R1, rel=ELEMENT_REL_TOL)          # periapsis: the burn point

    # The impulse is instantaneous: position is continuous across it.
    assert float(np.linalg.norm(_relative(sim, ks)[:3])) == pytest.approx(R1, rel=ELEMENT_REL_TOL)


def test_impulse_leaves_the_engine_agreeing_with_itself(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The elements written by the impulse must be the elements the arena's own Cartesian state implies.
    `coe_states` is written by `manoeuvres.apply_delta_v` from the parent-relative state, and
    `global_states`/`local_states` by the same call; if either used the wrong reference the two would
    disagree. Independent of the Hohmann numbers.
    """
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    ks = sim.name_to_index["KEPLER-SAT"]

    sim.apply_delta_v(ks, (0.2, 0.35, -0.1))
    stored = sim.coe_states[ks]
    rederived = _elements(sim, ks)
    assert stored == pytest.approx(rederived, rel=1e-12, abs=1e-12)

    # local_states is measured against body_sys_map (the barycentre, which sits on Earth here because
    # the vessels are massless); global_states against the root. Both must have moved by the same dv.
    assert sim.local_states[ks, 3:] == pytest.approx(
        sim.global_states[ks, 3:] - sim.global_states[sim.body_sys_map[ks], 3:], rel=0.0, abs=1e-14)


# ==================================================================================================
# 2. The full transfer, on both propagators
# ==================================================================================================

def test_hohmann_keplerian_reaches_the_target_orbit(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Fly the whole profile analytically: burn, coast `t_t`, burn, coast. The vessel must end on a
    circular orbit of radius `r2`.

    The second burn is scheduled at `t_t = 19178.154 s`, which is **not** a multiple of `dt = 60 s`
    (319.64 steps). Quantising it to a step boundary would place it 21 s from apoapsis and leave
    `e ~ 1e-4`, four orders above what is asserted here - so this also tests that `step()` really does
    split at the epoch.
    """
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    ks, _, _, _ = _transfer_profile(sim)
    _fly(sim, DT, 1.1 * T_TRANSFER)

    coe = _elements(sim, ks)
    e = float(coe[COEIndex.E])
    a = float(coe[COEIndex.P]) / (1.0 - e * e)
    r = float(np.linalg.norm(_relative(sim, ks)[:3]))

    assert e < CIRCULARISATION_E_TOL
    assert abs(a - R2) < CIRCULARISATION_R_TOL_KM
    assert abs(r - R2) < CIRCULARISATION_R_TOL_KM
    assert not sim.pending_manoeuvres


def test_hohmann_cowell_reaches_the_target_orbit(db_session_factory: Callable[[], Session]) -> None:
    """
    The same profile on `PropagatorType.COWELL` + `point_mass_gravity`. The bound is RK4's, derived in
    the module docstring: the whole transfer carries `~1e-1 km` of truncation at `dt = 60 s`, so the
    final radius and semi-major axis are asked to be right to one part in `1e5`, not to rounding.
    A Cowell body's `coe_states` row stays stale by design, so the elements are re-derived from its
    Cartesian state here - and that stale row is asserted, so the day it starts being maintained this
    test says so rather than silently passing.
    """
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    _, _, cs, _ = _transfer_profile(sim)
    stale_before = sim.coe_states[cs].copy()
    _fly(sim, DT, 1.1 * T_TRANSFER)

    coe = _elements(sim, cs)
    e = float(coe[COEIndex.E])
    a = float(coe[COEIndex.P]) / (1.0 - e * e)
    r = float(np.linalg.norm(_relative(sim, cs)[:3]))

    bound = 10.0 * rk4_transfer_error_km(DT)      # 1.2e-1 km; measured |a - r2| = 4.6e-2 km
    assert abs(a - R2) < bound
    assert abs(r - R2) < bound
    assert e < bound / R2 * 2.0                   # an along-track error leaves the shape nearly circular
    assert sim.coe_states[cs] == pytest.approx(stale_before, rel=0.0, abs=0.0)


def test_cowell_and_keplerian_transfers_agree_within_rk4_truncation(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The two propagators fly the *same* manoeuvre in the same arena, so their difference is RK4's
    truncation and nothing else. Two independent statements, because either alone is weak:

    1. the difference is within a decade of the derived local-error integral (module docstring);
    2. it falls by 16x for every halving of `dt` - fourth order, which identifies it as truncation.
       An impulse applied wrongly (wrong frame, wrong sign, wrong body) would leave a difference that
       does not converge at all.

    Measured: 1.83, 9.81e-2, 5.61e-3 km at dt = 120, 60, 30 s - ratios 18.7 and 17.5 against an ideal
    16. The excess over 16 is the step grid moving relative to perigee, where the local error is
    concentrated (the integrand above is 8.6^5 times larger there than at apoapsis).
    """
    errors = []
    for dt in (120.0, 60.0, 30.0):
        sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
        ks, _, cs, _ = _transfer_profile(sim)
        _fly(sim, dt, 1.1 * T_TRANSFER)
        diff = float(np.linalg.norm(_relative(sim, ks)[:3] - _relative(sim, cs)[:3]))
        errors.append(diff)

        lo, hi = TRUNCATION_ESTIMATE_BAND
        estimate = rk4_transfer_error_km(dt)
        assert lo * estimate < diff < hi * estimate, f"dt={dt}: {diff} km against {estimate} km derived"

    for coarse, fine in zip(errors, errors[1:]):
        assert 14.0 < coarse / fine < 22.0, f"expected ~16x (fourth order), got {coarse / fine}"


# ==================================================================================================
# 3. Step splitting
# ==================================================================================================

def test_split_step_does_not_disturb_a_non_manoeuvring_body(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    A manoeuvre cuts the step at its epoch for the *whole* arena, so a body that is not manoeuvring
    must not care.

    - **Analytic bodies**: propagating `h1` then `h2` is the same closed-form advance as `h1 + h2`, so
      the difference is the rounding of that sum plus the solver's fixed-point quantisation -
      `r * few eps ~ 1e-11 km`. Measured **exactly 0.0**; asserted below 1e-9 km, since bit-identity
      is a property of this solver's fixed point and not something the maths guarantees.
    - **Cowell bodies**: it *cannot* be exactly nothing. RK4 evaluates its stages at nodes fixed by the
      step size, so cutting one step in two changes that step's truncation by an amount of order the
      local error itself, `r (n h)^5 / 120 = 6.6e-5 km` here, which then grows along-track over the
      remaining ~2.3 orbits by at most `~3 pi N_orb = 22`, giving `1.5e-3 km`. Measured 4.7e-4 km -
      2.4e-8 of the manoeuvring vessel's own 8810 km displacement, which is the number that matters.

    The epoch is deliberately off the step grid (`37.5 dt + 11.3 s`) so the split is real.
    """
    epoch = 37.5 * DT + 11.3
    n_steps = 200

    def run(with_manoeuvre: bool) -> tuple[np.ndarray, np.ndarray]:
        sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
        ks, kt, cs, ct = (sim.name_to_index[n] for n in
                          ("KEPLER-SAT", "KEPLER-TWIN", "COWELL-SAT", "COWELL-TWIN"))
        if with_manoeuvre:
            sim.schedule_delta_v([ks, cs], (0.0, 0.3, 0.0), epoch)
        for _ in range(n_steps):
            sim.step(DT)
        return sim.global_states[[kt, ct]].copy(), sim.global_states[[ks, cs]].copy()

    split_twins, split_sats = run(True)
    plain_twins, plain_sats = run(False)

    kepler_twin_shift = float(np.linalg.norm(split_twins[0, :3] - plain_twins[0, :3]))
    cowell_twin_shift = float(np.linalg.norm(split_twins[1, :3] - plain_twins[1, :3]))
    manoeuvred_shift = float(np.linalg.norm(split_sats[0, :3] - plain_sats[0, :3]))

    n0 = math.sqrt(MU / R1**3)
    local_error_km = R1 * (n0 * DT) ** 5 / 120.0                      # 6.6e-5 km
    orbits = n0 * n_steps * DT / (2.0 * math.pi)
    cowell_bound = local_error_km * (1.0 + 3.0 * math.pi * orbits)    # 1.5e-3 km

    assert kepler_twin_shift < 1e-9
    assert cowell_twin_shift < cowell_bound
    assert manoeuvred_shift > 1e3                                     # the manoeuvre itself did happen
    assert cowell_twin_shift < 1e-6 * manoeuvred_shift


def test_scheduled_epoch_is_exact_not_quantised(db_session_factory: Callable[[], Session]) -> None:
    """
    A burn scheduled mid-step must land at its epoch, not at the step boundary. Flown on the analytic
    propagator, where the only error is the solver's, the state after a Delta-v scheduled at
    `t = 1.5 dt` must equal the state of a vessel given the same Delta-v by two explicit half-steps.

    Without splitting the burn would be late by `0.5 dt = 30 s`, which on this orbit is 3.2 degrees of
    true anomaly and ~390 km of position - eight orders above the tolerance asserted.
    """
    epoch = 1.5 * DT

    sim_a = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    a_sat = sim_a.name_to_index["KEPLER-SAT"]
    sim_a.schedule_delta_v(a_sat, (0.0, DV1, 0.0), epoch)
    for _ in range(6):
        sim_a.step(DT)

    sim_b = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    b_sat = sim_b.name_to_index["KEPLER-SAT"]
    sim_b.step(DT)
    sim_b.step(0.5 * DT)
    sim_b.apply_delta_v(b_sat, (0.0, DV1, 0.0))
    sim_b.step(0.5 * DT)
    for _ in range(4):
        sim_b.step(DT)

    assert sim_a.t == pytest.approx(sim_b.t, rel=0.0, abs=1e-12)
    assert _relative(sim_a, a_sat) == pytest.approx(_relative(sim_b, b_sat), rel=0.0, abs=1e-9)


def test_manoeuvres_inside_one_step_are_applied_in_epoch_order(
    db_session_factory: Callable[[], Session],
) -> None:
    """Several impulses inside a single step each get their own sub-step, in epoch order, and the
    clock lands exactly on the step boundary regardless of how the sub-intervals rounded."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]

    dt = 900.0
    sim.schedule_delta_v(sat, (0.0, 0.10, 0.0), 700.0, label="second")
    sim.schedule_delta_v(sat, (0.0, 0.05, 0.0), 200.0, label="first")
    assert [m.label for m in sim.pending_manoeuvres] == ["first", "second"]

    sim.step(dt)
    assert sim.t == dt
    assert not sim.pending_manoeuvres

    # Same two impulses, hand-timed: the queue must be no different from doing it manually.
    manual = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    m_sat = manual.name_to_index["KEPLER-SAT"]
    manual.step(200.0)
    manual.apply_delta_v(m_sat, (0.0, 0.05, 0.0))
    manual.step(500.0)
    manual.apply_delta_v(m_sat, (0.0, 0.10, 0.0))
    manual.step(200.0)

    assert _relative(sim, sat) == pytest.approx(_relative(manual, m_sat), rel=0.0, abs=1e-9)


def test_epoch_zero_burns_before_any_propagation(db_session_factory: Callable[[], Session]) -> None:
    """`epoch_s = 0` on a fresh simulation is a departure burn: applied before the first step's
    propagation, so the first step is already flown on the new orbit."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]
    sim.schedule_delta_v(sat, (0.0, DV1, 0.0), 0.0)
    sim.step(DT)

    immediate = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    i_sat = immediate.name_to_index["KEPLER-SAT"]
    immediate.apply_delta_v(i_sat, (0.0, DV1, 0.0))
    immediate.step(DT)

    assert _relative(sim, sat) == pytest.approx(_relative(immediate, i_sat), rel=0.0, abs=1e-12)


def test_empty_queue_leaves_step_bit_identical(db_session_factory: Callable[[], Session]) -> None:
    """The manoeuvre machinery must cost an existing caller nothing, numerically: with no manoeuvre
    scheduled, `step()` is `_advance()` and the trajectory is bit-identical to one flown by
    `_advance` alone."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    other = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    for _ in range(50):
        sim.step(DT)
        other._advance(DT)
    assert np.array_equal(sim.global_states, other.global_states)


# ==================================================================================================
# 4. Secular J2: the cached rates must follow the new orbit
# ==================================================================================================

def test_impulse_changes_the_secular_j2_nodal_rate(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    A prograde kick `dv = d v_c` on a circular orbit of radius `r`, exactly (no expansion):

        v -> v(1 + d)      =>   h -> h(1 + d)   =>   p = h^2/mu -> r (1 + d)^2
        1/a = 2/r - v^2/mu = (2 - (1 + d)^2)/r  =>   a -> r / (1 - 2d - d^2)
        e^2 = 1 - p/a = 1 - (1 + d)^2 (1 - 2d - d^2) = (2d + d^2)^2   =>   e = 2d + d^2

    The nodal rate is `-1.5 n J2 (R/p)^2 cos i` with `n = sqrt(mu/a^3)` and `i` unchanged (a prograde
    burn is in-plane), so

        RAAN_dot_after / RAAN_dot_before = (a_new/r)^{-3/2} (r/p_new)^2
                                         = (1 - 2d - d^2)^{3/2} (1 + d)^{-4}
                                         = 1 - 7d + 22 d^2 + O(d^3).

    At `d = 1e-3` that is a 0.70 % change in the nodal rate. **A stale cached rate gives exactly 1.0**,
    and no other assertion in this file would notice: the position, the elements and the energy would
    all still be right. This is the test the whole secular-J2 branch of `apply_delta_v` exists for.

    The rate is measured as an observed RAAN difference over 400 steps, not read from
    `_secular_j2_rates` - the cache is what is under test. Measured ratio 0.993021952088 against a
    closed form of 0.993021952088 (2.2e-13 relative); the first-order `1 - 7d` sits 2.195e-5 away,
    matching the derived second-order term `22 d^2 = 2.200e-5` to 0.2 %.
    """
    d = 1.0e-3
    p_km, inc, raan0 = 7000.0, math.radians(45.0), 1.0
    steps, dt = 400, 60.0

    sim = scenarios.two_body(
        db_session_factory(), p=p_km, e=0.0, i=inc, raan=raan0, mu_secondary=0.0)
    sat = sim.name_to_index["Secondary"]
    sim.set_propagator(
        sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)

    def observed_rate() -> float:
        t0, raan = float(sim.t), float(sim.coe_states[sat, COEIndex.RAAN])
        for _ in range(steps):
            sim.step(dt)
        return (float(sim.coe_states[sat, COEIndex.RAAN]) - raan) / (float(sim.t) - t0)

    before = observed_rate()
    sim.apply_delta_v(sat, (0.0, d * math.sqrt(MU / p_km), 0.0))

    # The new orbit is exactly the one derived above - checked before the rate, so a failure says
    # which half is wrong.
    assert float(sim.coe_states[sat, COEIndex.E]) == pytest.approx(2 * d + d * d, rel=1e-9)
    assert float(sim.coe_states[sat, COEIndex.P]) == pytest.approx(p_km * (1 + d) ** 2, rel=1e-9)
    assert float(sim.coe_states[sat, COEIndex.I]) == pytest.approx(inc, rel=1e-12)

    after = observed_rate()
    exact = (1.0 - 2 * d - d * d) ** 1.5 / (1.0 + d) ** 4

    assert before < 0.0                                   # prograde orbit: the node regresses
    assert after / before == pytest.approx(exact, rel=1e-9)
    assert exact - (1.0 - 7.0 * d) == pytest.approx(22.0 * d * d, rel=0.01)
    assert abs(after / before - 1.0) > 1e-3               # a stale rate would give exactly 1.0


def test_secular_j2_impulse_onto_an_open_orbit_is_refused(
    db_session_factory: Callable[[], Session],
) -> None:
    """`set_propagator` requires `0 <= e < 1` because the cached rates are built on `n = sqrt(mu/a^3)`.
    An impulse is the other way that restriction can be broken, so it is refused there too - and the
    body is left untouched rather than half-updated."""
    sim = scenarios.two_body(db_session_factory(), p=7000.0, e=0.0, mu_secondary=0.0)
    sat = sim.name_to_index["Secondary"]
    sim.set_propagator(
        sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    before_coe = sim.coe_states[sat].copy()
    before_state = sim.global_states[sat].copy()

    with pytest.raises(ValueError, match="open orbit"):
        sim.apply_delta_v(sat, (0.0, 5.0, 0.0))           # escape speed at 7000 km is ~10.7 km/s

    assert np.array_equal(sim.coe_states[sat], before_coe)
    assert np.array_equal(sim.global_states[sat], before_state)


# ==================================================================================================
# 5. Frame: the Delta-v is RSW, about the *parent*
# ==================================================================================================

def test_impulse_is_parent_relative_with_a_moving_parent(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `sun_earth_moon(leo_satellite=True)` is the only shipped scenario whose satellite has a parent that
    is genuinely moving (Earth at ~30 km/s about the Sun) *and* a system bubble that is not that parent
    (the Earth-Moon barycentre, ~4700 km away). The RSW frame must come from the satellite's motion
    about **Earth**.

    Building the frame from the absolute state instead would point the same 0.05 km/s Delta-v 0.063
    km/s away - 126 % of its own magnitude, essentially a random direction - and nothing else in this
    file would catch it, because `hohmann_pair`'s Earth is at rest at the origin.
    """
    sim = scenarios.sun_earth_moon(db_session_factory(), leo_satellite=True)
    sat = sim.name_to_index["LEO-SAT"]
    earth = sim.name_to_index["Earth"]
    assert sim.body_sys_map[sat] != sim.parent_indices[sat]        # the two graphs really do diverge

    rel = sim.global_states[sat] - sim.global_states[earth]
    basis, valid = ReferenceFrames.RSW_basis(rel[:3], rel[3:])
    assert bool(valid[0] if valid.ndim else valid)
    dv = 0.05
    expected = dv * basis[1]                                       # row 1 is S, the along-track axis

    absolute_basis, _ = ReferenceFrames.RSW_basis(
        sim.global_states[sat, :3], sim.global_states[sat, 3:])
    wrong = dv * absolute_basis[1]

    v_before = sim.global_states[sat, 3:].copy()
    sim.apply_delta_v(sat, (0.0, dv, 0.0))
    delivered = sim.global_states[sat, 3:] - v_before

    assert delivered == pytest.approx(expected, rel=0.0, abs=1e-14)
    assert float(np.linalg.norm(wrong - expected)) > 0.5 * dv      # the control really is different

    # local_states is measured against the Earth-Moon barycentre, not Earth, and must have taken the
    # same Cartesian kick.
    assert sim.local_states[sat, 3:] == pytest.approx(
        sim.global_states[sat, 3:] - sim.global_states[sim.body_sys_map[sat], 3:], rel=0.0, abs=1e-14)


def test_radial_impulse_does_not_raise_the_orbit(db_session_factory: Callable[[], Session]) -> None:
    """
    Axis control for the RSW frame. On a circular orbit `v` is perpendicular to `r`, so a **radial**
    kick does no work to first order: `d(v^2)/2 = v.dv + dv^2/2 = dv^2/2`, and the semi-major axis
    rises only by `2 a^2 (dv^2/2) / mu = a (dv/v)^2` - second order - while the eccentricity rises to
    `dv/v`, first order. Transposing R and S in the kernel would swap the two: `a` would jump by
    `2 a dv/v` (first order, 2000x larger here) and `e` would reach `2 dv/v`.
    """
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]
    v_c = math.sqrt(MU / R1)
    d = 1.0e-3
    dv = d * v_c

    sim.apply_delta_v(sat, (dv, 0.0, 0.0))
    coe = sim.coe_states[sat]
    e = float(coe[COEIndex.E])
    a = float(coe[COEIndex.P]) / (1.0 - e * e)

    assert e == pytest.approx(d, rel=1e-6)                  # first order in dv/v
    assert (a - R1) / R1 == pytest.approx(d * d, rel=1e-3)  # second order - the axis control


def test_per_body_delta_v_rows_follow_the_named_order(
    db_session_factory: Callable[[], Session],
) -> None:
    """An `(n, 3)` Delta-v pairs row `k` with the `k`-th *named* body. Named in reverse slot order, the
    pairing must not silently re-sort - which is what makes `_manoeuvre_slots` refuse to sort."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    first = sim.name_to_index["KEPLER-SAT"]
    second = sim.name_to_index["KEPLER-TWIN"]
    assert second > first

    sim.apply_delta_v([second, first], [[0.0, 0.4, 0.0], [0.0, 0.1, 0.0]])
    assert float(sim.coe_states[second, COEIndex.E]) > float(sim.coe_states[first, COEIndex.E])


# ==================================================================================================
# 6. Restrictions - a manoeuvre that cannot mean anything must raise, not look plausible
# ==================================================================================================

def test_rejects_heads_barycentres_roots_and_inactive_slots(
    db_session_factory: Callable[[], Session],
) -> None:
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    earth = sim.name_to_index["Earth"]                     # head of the Earth system
    bary = int(sim.body_sys_map[earth])                    # its barycentre
    free_slot = sim.free_indices[-1]                       # never allocated: inactive

    assert sim.is_head[earth] and sim.is_system[bary]
    for slot in (earth, bary, free_slot):
        with pytest.raises(ValueError, match="restricted to active"):
            sim.apply_delta_v(int(slot), (0.0, 0.1, 0.0))


def test_rejects_a_massive_body(db_session_factory: Callable[[], Session]) -> None:
    """A massive body's impulse would not reach its barycentre's own orbit - see manoeuvres.py."""
    sim = scenarios.sun_earth_moon(db_session_factory())
    moon = sim.name_to_index["Moon"]
    with pytest.raises(ValueError, match="requires mu == 0"):
        sim.apply_delta_v(moon, (0.0, 0.1, 0.0))


def test_rejects_a_repeated_body(db_session_factory: Callable[[], Session]) -> None:
    """Two Delta-vs on one slot in one call is a many-to-one fancy-indexed write: one would vanish."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]
    with pytest.raises(ValueError, match="at most once"):
        sim.apply_delta_v([sat, sat], [[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])


def test_rejects_a_misshapen_delta_v(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    ks, kt = sim.name_to_index["KEPLER-SAT"], sim.name_to_index["KEPLER-TWIN"]
    with pytest.raises(ValueError, match="dv_rsw must be"):
        sim.apply_delta_v(ks, (0.0, 0.1))
    with pytest.raises(ValueError, match="dv_rsw must be"):
        sim.apply_delta_v([ks, kt], [[0.0, 0.1, 0.0]] * 3)
    with pytest.raises(ValueError, match="dv_rsw must be"):
        sim.schedule_delta_v(ks, np.zeros((2, 3)), 10.0)


def test_rejects_an_out_of_range_slot(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    with pytest.raises(ValueError, match="outside the arena"):
        sim.apply_delta_v(sim.max_capacity, (0.0, 0.1, 0.0))
    with pytest.raises(ValueError, match="at least one body"):
        sim.apply_delta_v(np.empty(0, dtype=np.int64), (0.0, 0.1, 0.0))


def test_a_refused_manoeuvre_leaves_the_arena_untouched(
    db_session_factory: Callable[[], Session],
) -> None:
    """The kernel computes the whole transaction before committing any of it, so a call that raises -
    here one body valid, one a head - changes nothing at all, including the valid body."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]
    earth = sim.name_to_index["Earth"]
    before = sim.global_states.copy()

    with pytest.raises(ValueError):
        sim.apply_delta_v([sat, earth], (0.0, 0.1, 0.0))
    assert np.array_equal(sim.global_states, before)

    # And the same for a degenerate frame reaching the kernel: a zero relative state has no RSW basis.
    sim.global_states[sat] = sim.global_states[sim.parent_indices[sat]]
    guarded = sim.global_states.copy()
    with pytest.raises(ValueError, match="RSW frame"):
        sim.apply_delta_v(sat, (0.0, 0.1, 0.0))
    assert np.array_equal(sim.global_states, guarded)


def test_clear_manoeuvres_drops_the_queue(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    sat = sim.name_to_index["KEPLER-SAT"]
    sim.schedule_delta_v(sat, (0.0, 0.1, 0.0), 100.0)
    assert len(sim.pending_manoeuvres) == 1
    sim.clear_manoeuvres()
    assert sim.pending_manoeuvres == ()
    before = sim.global_states[sat].copy()
    sim.step(200.0)
    assert not np.array_equal(sim.global_states[sat], before)       # it propagated, it just did not burn


def test_manoeuvre_kernel_reports_rather_than_raises(
    db_session_factory: Callable[[], Session],
) -> None:
    """`manoeuvres.apply_delta_v` is the array-in/array-out half: it reports validity through a mask and
    leaves the invalid rows alone. `Simulation.apply_delta_v` is what turns that into a `ValueError`."""
    sim = scenarios.hohmann_pair(db_session_factory(), r1_km=R1)
    good = sim.name_to_index["KEPLER-SAT"]
    bad = sim.name_to_index["KEPLER-TWIN"]
    sim.global_states[bad] = sim.global_states[sim.parent_indices[bad]]     # no RSW frame
    idx = np.array([good, bad], dtype=np.int64)
    before_bad = sim.global_states[bad].copy()

    valid = manoeuvres.apply_delta_v(
        idx, np.array([0.0, 0.1, 0.0]), sim.global_states, sim.local_states, sim.coe_states,
        sim.mu_array, sim.parent_indices, sim.propagator_type,
    )
    assert valid.tolist() == [True, False]
    assert np.array_equal(sim.global_states[bad], before_bad)
    assert float(sim.coe_states[good, COEIndex.E]) > 1e-3
