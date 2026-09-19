"""
Validation of the `"thrust"` force model (`thrust.thrust_kernel`, `thrust.deplete_mass`).

Every tolerance below is derived before it was measured. The class of bug this file guards against is
the one `CLAUDE.md` calls out: a transposed RSW axis, a flipped sign or a dropped unit conversion
produces a burn that still looks like a burn and is wrong by a constant factor. Each check is
therefore stated as a magnitude, and each was run against real-file mutants of `thrust.py` - see
"Negative controls" below.

Derivations
-----------
**Closed form.** `a = (T/m) (d_R R + d_S S + d_W W)`, with the basis `R = r/|r|`,
`W = (r x v)/|r x v|`, `S = W x R` built in the test from that definition rather than by calling
`frames.RSW_basis`. `T/m` is m/s^2; the engine is km/s^2, hence the single factor 1e-3.

**Rocket equation.** Integrating `dv/dt = T/m` with `m = m0 - m_dot t` along a *fixed* direction gives
Tsiolkovsky, `Delta v = (T/m_dot) ln(m0/m1) = Isp g0 ln(m0/m1)`. Two things separate that from what a
powered orbit actually does:

1. *Gravity.* The speed of a thrusting satellite is not `v0 + Delta v` - the orbit turns. The
   separation used here is a **thrust-free twin**: two co-located, identical, massless vessels in one
   arena, one thrusting, differenced. Their difference obeys
   `d(delta_v)/dt = a_T + G delta_r`, with `G = d g/d r` the gravity gradient, so the twin difference
   is the pure thrust integral plus a correction bounded by `|G| |delta_r| tau`. With
   `|G| ~ 2 mu/r^3 = 2 n^2`, `|delta_r| ~ Delta v tau / 2`, that correction is `~ (n tau)^2 Delta v`.
   The burn is therefore placed on a **400 000 km orbit** (`n = 2.4956e-6 /s`) and run for
   `tau = 600 s`: `(n tau)^2 = 2.2e-6`. Numerically, `|delta_r| = 93 km`, `|G| = 1.25e-11 /s^2`,
   giving `3.5e-7 km/s` against `Delta v = 0.31 km/s`, i.e. **1.1e-6 relative**.
2. *Direction rotation.* `S` turns with the orbit, so `|int a_T dt| < int |a_T| dt`. The position
   sweeps `v tau / r = 1.5e-3 rad` over the burn, and a uniformly rotating unit vector loses
   `phi^2/24 = 9.4e-8` relative. Below the gradient term.

So the twin difference must reproduce the **discrete** delta-v the scheme actually computes,
`sum_n (T/m_n) dt`, to 1.1e-6 relative. `ROCKET_DISCRETE_REL_TOL = 1e-5` is 9x that.

**The mass-update bias, asserted as a prediction.** Mass is frozen across a step's four RK4 stages
(`thrust.py`), so the integrated delta-v is a *left* Riemann sum of `T/m(t)` - and `T/m` *increases*
through a burn, so the left rule **under**-delivers: the whole step is flown at the heavier
start-of-step mass. Euler-Maclaurin gives the bias as `-(dt/2)(a_end - a_start)`. For the burn below,
`a_start = 4.9033e-4`, `a_end = 5.4481e-4 km/s^2`, `dt = 1 s`, so the bias is `-2.724e-5 km/s`,
**-8.79e-5 of `Delta v`**. The test asserts that predicted bias against the measured shortfall to 5%
- not as a tolerance on the answer but as a check that the first-order term is the size and *sign*
theory says. (Getting that sign wrong first, and being caught by the assertion, is what item 5 of the
per-feature contract is for.) Halving `dt` must halve it, so `ROCKET_EXACT_REL_TOL = 2e-4` (bias
8.8e-5 plus the 1.1e-6 geometric term, doubled for headroom) is a statement about the scheme, not a
snapshot.

**Orbit raising.** Specific energy `E = v^2/2 - mu/r = -mu/(2a)`, and an along-track acceleration `f`
does work `dE/dt = f v` on a circular orbit (where `S` *is* the velocity direction). With
`dE/da = mu/(2a^2)` and `v = sqrt(mu/a)`,

    da/dt = (2 a^2 / mu) f sqrt(mu/a) = 2 f sqrt(a^3 / mu) = 2 f / n

the `e -> 0` limit of Gauss's variational equation for `a`. `f = T 1e-3 / m(t)` falls as fuel burns,
so the prediction integrates the coupled scalar ODE with RK4 in the test rather than multiplying a
constant rate: over the run `m` drops 0.084%, which is 8e-3 of the answer and 100x the tolerance.

A **radial** burn of the same magnitude is run as the axis control. Gauss gives
`da/dt = (2/(n sqrt(1-e^2)))[e sin(theta) a_R + (p/r) a_S]`, so a pure `a_R` raises `a` only through
the induced eccentricity - second order, `O(e)` with `e ~ 2 f/(v n) = 2.4e-5`. Transposing R and S in
the kernel would move `Delta a` by a factor of ~1/e. The bound on the radial run is therefore
`e = 2.4e-5` of the tangential one, plus the integrator drift the thrust-free twin measures
(6.8e-6); `RADIAL_AXIS_MAX_FRACTION = 1e-3` is 30x that and still fails by three orders of magnitude
if R and S are transposed.

Error budget, orbit raising
---------------------------
`a0 = 6921 km`, `n = 1.0965e-3 /s`, `T = 5730.7 s`, `dt = 20 s`, 5 orbits (1433 steps),
`f = 8.667e-8 km/s^2`, `Delta a = 4.53 km`.

- *RK4 drift.* Measured directly by the thrust-free twin in the same arena at the same `dt`.
  `test_drag.py` measured 3.0e-5 km of spurious `|Delta a|` on the same orbit and step size; against
  4.53 km that is **6.8e-6 relative**.
- *Osculating scatter.* Tangential thrust drives `e ~ 2f/(v n) = 2.4e-5`. Through the `(p/r)` factor
  that puts a once-per-orbit ripple of relative size `e` on `da/dt`, so the osculating `a` carries an
  oscillation of amplitude `|da/dt| e / n = 3.5e-6 km`, **8e-7 relative** at the endpoints.
- *Mass quantisation.* The scheme's stepwise mass against the ODE's continuous one is the same left
  Riemann bias, `(dt/2)(f_end - f_start)/int f = 1.5e-7` relative.
- *Scalar-ODE quadrature.* RK4 with 2000 steps on a rate varying by 0.1% - below 1e-12.

Budget: 6.8e-6 + 8e-7 is about 8e-6. `ORBIT_RAISE_REL_TOL = 5e-5` is 6x that.

Negative controls
-----------------
Each was applied to `src/orbital_engine/thrust.py` itself, one at a time, and reverted with
`git checkout`:

- *Dropping `_M_PER_S2_TO_KM_PER_S2`* (a factor 1e3): 5 failures - both closed-form checks, both
  rocket-equation checks, and orbit raising (8447 km of raising against 4.53 predicted).
- *Flipping the direction sign* (`out[indices] -= ...`): 4 failures - both closed-form checks, the
  rocket equation and orbit raising (`Delta a = -4.53 km`). The burnout, registration and
  depletion-arithmetic tests still pass, which is correct: they never look at the sign.
  **This mutant initially escaped the rocket-equation test**, because `|v_powered - v_coasting|` is a
  magnitude and cannot see a sign. That test now asserts the projection of the twin difference onto
  `S` as well, before its magnitude.
- *Transposing the R and S direction columns*: 5 failures, including
  `test_radial_burn_is_the_axis_control`, which exists for exactly this mutation and which neither of
  the other two mutants trips.
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import registry, scenarios, thrust
from orbital_engine.gravity import POINT_MASS_MODEL
from orbital_engine.simulator import Simulation
from orbital_engine.thrust import (
    STANDARD_GRAVITY, THRUST_MODEL, THRUST_PARAM_NAMES, deplete_mass, thrust_kernel,
)

ArrF = NDArray[np.float64]

MU = scenarios.MU_EARTH

# --------------------------------------------------------------------------------------------------
# Tolerances, derived - see the module docstring.
# --------------------------------------------------------------------------------------------------

# Closed form, direct kernel call. The kernel does ~10 rounded operations after the parent
# subtraction (a cross product, three norms, a divide, three scaled sums) and the test's reference
# does the same again, so about 20 eps = 4.4e-15. 1e-13 is 20x that. Any unit slip is 1e3 and a sign
# flip is 2.
CLOSED_FORM_REL_TOL = 1e-13
# Same case with the parent translated to 1.5e8 km. Subtracting that offset from a 7000 km separation
# loses `eps * 1.5e8 / 7000` = 4.8e-12 of the relative position, which tilts the RSW basis by the
# same relative amount. 1e-11 is ~2x.
CLOSED_FORM_TRANSLATED_REL_TOL = 1e-11

# Twin-differenced delta-v against the discrete sum the scheme computes. Derived 1.1e-6 (gravity
# gradient across the twins' 93 km separation), 9x margin.
ROCKET_DISCRETE_REL_TOL = 1e-5
# ... and against the exact Tsiolkovsky value. Dominated by the first-order mass update's
# left-Riemann bias, derived -8.79e-5; doubled for headroom.
ROCKET_EXACT_REL_TOL = 2e-4
# The bias itself, `-(dt/2)(a_end - a_start)`, against the measured shortfall against Tsiolkovsky.
# This is the prediction, not a tolerance on the answer: the residual after removing it is ~1e-6 of
# `Delta v`, i.e. ~1% of the bias, so 5% is a modest margin on a first-order truncation estimate.
ROCKET_BIAS_REL_TOL = 0.05

# Orbit raising against the energy-derived `da/dt = 2f/n`. Derived 8e-6, 6x margin.
ORBIT_RAISE_REL_TOL = 5e-5
# A radial burn raises `a` only at O(e), e ~ 2.4e-5, on top of the 6.8e-6 integrator drift. 30x
# margin on that, and still three orders of magnitude clear of a transposed R/S (which gives 1).
RADIAL_AXIS_MAX_FRACTION = 1e-3


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def _rsw_basis_by_hand(r: ArrF, v: ArrF) -> Tuple[ArrF, ArrF, ArrF]:
    """R, S, W from the definition, without touching `frames.py` - the closed-form check's reference."""
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    W = h / np.linalg.norm(h)
    S = np.cross(W, R)
    return R, S, W


def _semi_major_axis(sim: Simulation, slot: int) -> float:
    """Osculating `a` from the vessel's state relative to its Keplerian parent (its COE row is stale)."""
    parent = int(sim.parent_indices[slot])
    rel = sim.global_states[slot] - sim.global_states[parent]
    r = float(np.linalg.norm(rel[:3]))
    v2 = float(rel[3:] @ rel[3:])
    mu = float(sim.mu_array[slot] + sim.mu_array[parent])
    return 1.0 / (2.0 / r - v2 / mu)


def _relative_state(sim: Simulation, slot: int) -> ArrF:
    parent = int(sim.parent_indices[slot])
    return np.asarray(sim.global_states[slot] - sim.global_states[parent], dtype=np.float64)


def _rk4_scalar(
    f: Callable[[float, float], float], y0: float, t0: float, t1: float, steps: int,
) -> float:
    """Classical RK4 on one scalar ODE. The mean-element prediction's quadrature."""
    h = (t1 - t0) / steps
    y, t = y0, t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


# ==================================================================================================
# 1. Registration and composition
# ==================================================================================================

def test_model_is_registered_with_its_parameters() -> None:
    model = registry.get_force_model(THRUST_MODEL)
    assert model.kernel is thrust_kernel
    assert model.param_names == THRUST_PARAM_NAMES
    assert model.citation
    assert model.validate_bodies is not None and model.validate_coefficients is not None
    assert THRUST_MODEL in [m.name for m in registry.all_force_models()]


def test_thrust_bit_is_foreign_to_the_fused_cowell_plan(db_session: Session) -> None:
    """
    `kernels.cowell_rk4_step` is fused for `point_mass_gravity` and `j2` only, so a Cowell body with
    thrust must send the **whole** Cowell set down the NumPy `RK4Integrator` path - as `"drag"` and
    `"third_body"` do. Nothing else in this file is meaningful if the compiled path silently ran and
    ignored the thrust term.
    """
    sim = scenarios.powered_vessel(db_session, n_vessels=2, n_powered=1, thrust_n=1.0)
    assert sim._cowell_fused_ok is False

    # Remove the thrust bit and the same arena becomes eligible again: it is the bit that disqualifies
    # it, not the scenario.
    slot = sim.name_to_index["THRUSTER-00"]
    sim.force_model_mask[slot] &= ~registry.mask_for([THRUST_MODEL])
    sim.resolve_force_models()
    assert sim._cowell_fused_ok is True


def test_empty_index_set_is_a_true_no_op() -> None:
    """`forces.ForceKernel`'s contract: called directly with an empty index set, touch nothing."""
    state = np.zeros((3, 6))
    out = np.full((3, 3), 7.0)
    thrust_kernel(
        np.array([], dtype=np.int64), 0.0, state, np.zeros(3), np.zeros(3, dtype=np.int32),
        np.zeros((3, len(THRUST_PARAM_NAMES))), out,
    )
    assert np.array_equal(out, np.full((3, 3), 7.0))


def test_massive_bodies_and_bad_coefficients_are_refused(db_session: Session) -> None:
    sim = scenarios.powered_vessel(db_session, n_vessels=1, n_powered=0)
    earth = sim.name_to_index["Earth"]
    vessel = sim.name_to_index["THRUSTER-00"]

    with pytest.raises(ValueError, match="massless"):
        sim.enable_force_model(THRUST_MODEL, earth, thrust_n=1.0)
    assert sim.force_model_mask[earth] & registry.mask_for([THRUST_MODEL]) == 0

    for kwargs in (
        {"thrust_n": -1.0},
        {"thrust_n": 1.0, "isp_s": 0.0},
        {"mass_kg": 0.0},
        {"dry_mass_kg": -1.0},
        {"mass_kg": 100.0, "dry_mass_kg": 200.0},
    ):
        with pytest.raises(ValueError):
            sim.enable_force_model(THRUST_MODEL, vessel, **kwargs)
        assert sim.force_model_mask[vessel] & registry.mask_for([THRUST_MODEL]) == 0


# ==================================================================================================
# 2. Closed form: the RSW rotation and the newton-to-km/s^2 conversion
# ==================================================================================================

def _closed_form_case(
    r: ArrF, v: ArrF, direction: ArrF, thrust_n: float, mass_kg: float, offset: ArrF,
) -> Tuple[ArrF, ArrF]:
    """Run the kernel on a two-row hand-built arena and return (measured, independently computed)."""
    state = np.zeros((2, 6))
    state[0, :3] = offset                      # the parent
    state[1, :3] = offset + r
    state[1, 3:] = v
    parent_indices = np.array([0, 0], dtype=np.int32)
    params = np.zeros((2, len(THRUST_PARAM_NAMES)))
    params[1] = [thrust_n, 300.0, mass_kg, 0.0, direction[0], direction[1], direction[2]]
    out = np.zeros((2, 3))

    thrust_kernel(np.array([1], dtype=np.int64), 0.0, state, np.zeros(2), parent_indices, params, out)

    R, S, W = _rsw_basis_by_hand(r, v)
    # Independently: T/m in m/s^2, converted to km/s^2 by dividing by 1000, times the direction law
    # re-expressed in Cartesian from the basis built above.
    magnitude_m_per_s2 = thrust_n / mass_kg
    expected = (magnitude_m_per_s2 / 1000.0) * (direction[0] * R + direction[1] * S + direction[2] * W)
    return np.asarray(out[1]), expected


def test_closed_form_axis_aligned() -> None:
    """
    `r` along +x, `v` along +y: R = x_hat, S = y_hat, W = z_hat exactly, so the expected answer is
    arithmetic done by hand. 500 N on 1000 kg is 0.5 m/s^2 = 5e-4 km/s^2; the direction (0.6, 0.8, 0)
    splits it 3:4.
    """
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    measured, expected = _closed_form_case(
        r, v, np.array([0.6, 0.8, 0.0]), 500.0, 1000.0, np.zeros(3))

    hand = np.array([3.0e-4, 4.0e-4, 0.0])     # written out, not computed from the model
    assert np.allclose(measured, hand, rtol=CLOSED_FORM_REL_TOL, atol=0.0)
    assert np.allclose(measured, expected, rtol=CLOSED_FORM_REL_TOL, atol=0.0)


def test_closed_form_general_geometry() -> None:
    """A state with no axis alignment and a non-zero flight-path angle, so all three RSW rows matter."""
    r = np.array([3000.0, -4500.0, 5200.0])
    v = np.array([4.1, 5.3, -1.7])
    direction = np.array([0.1, -0.5, 0.8])
    measured, expected = _closed_form_case(r, v, direction, 1234.5, 789.0, np.zeros(3))

    assert np.allclose(measured, expected, rtol=CLOSED_FORM_REL_TOL, atol=0.0)
    # Magnitude is |direction| * T/m: the direction vector is used as given, so its norm throttles.
    assert float(np.linalg.norm(measured)) == pytest.approx(
        (1234.5 / 789.0) * 1e-3 * float(np.linalg.norm(direction)), rel=CLOSED_FORM_REL_TOL)


def test_closed_form_is_translation_invariant() -> None:
    """The same geometry with the parent 1 AU from the origin: only relative state may enter."""
    r = np.array([3000.0, -4500.0, 5200.0])
    v = np.array([4.1, 5.3, -1.7])
    direction = np.array([0.1, -0.5, 0.8])
    at_origin, _ = _closed_form_case(r, v, direction, 1234.5, 789.0, np.zeros(3))
    translated, _ = _closed_form_case(
        r, v, direction, 1234.5, 789.0, np.array([1.5e8, 0.0, 0.0]))

    assert np.allclose(translated, at_origin, rtol=CLOSED_FORM_TRANSLATED_REL_TOL, atol=0.0)


def test_empty_tanks_and_unconfigured_rows_contribute_exactly_zero() -> None:
    """Not "small": exactly 0.0, so a burnt-out or unconfigured body cannot bias an accumulator."""
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    state = np.zeros((2, 6))
    state[1, :3] = r
    state[1, 3:] = v
    parent_indices = np.array([0, 0], dtype=np.int32)
    idx = np.array([1], dtype=np.int64)

    for row in (
        [500.0, 300.0, 260.0, 260.0, 0.0, 1.0, 0.0],   # empty tanks: mass == dry
        [0.0, 300.0, 300.0, 260.0, 0.0, 1.0, 0.0],     # no thruster
        [0.0] * len(THRUST_PARAM_NAMES),               # bit set, coefficients never written
    ):
        params = np.zeros((2, len(THRUST_PARAM_NAMES)))
        params[1] = row
        out = np.zeros((2, 3))
        thrust_kernel(idx, 0.0, state, np.zeros(2), parent_indices, params, out)
        assert np.array_equal(out, np.zeros((2, 3)))


# ==================================================================================================
# 3. The rocket equation, separated from the gravity turn by a thrust-free twin
# ==================================================================================================

ROCKET_P_KM = 400_000.0
ROCKET_DT = 1.0
ROCKET_STEPS = 600
ROCKET_MDOT = 0.05                                   # kg/s, chosen so 600 s burns 30 of 40 kg
ROCKET_ISP = 300.0
ROCKET_THRUST_N = ROCKET_MDOT * ROCKET_ISP * STANDARD_GRAVITY


def test_rocket_equation_against_a_thrust_free_twin(db_session: Session) -> None:
    sim = scenarios.powered_vessel(
        db_session, n_vessels=2, n_powered=1, p_km=ROCKET_P_KM,
        dry_mass=260.0, fuel_mass=40.0,
        thrust_n=ROCKET_THRUST_N, isp_s=ROCKET_ISP, direction=(0.0, 1.0, 0.0),
    )
    powered = sim.name_to_index["THRUSTER-00"]
    coasting = sim.name_to_index["THRUSTER-01"]
    params = sim.force_model_params[THRUST_MODEL]

    m0 = float(params[powered, 2])
    masses: List[float] = []
    for _ in range(ROCKET_STEPS):
        masses.append(float(params[powered, 2]))
        sim.step(ROCKET_DT)
    m1 = float(params[powered, 2])

    # The tanks must not have run dry - that would make this a different test.
    assert m1 > float(params[powered, 3])
    assert m0 - m1 == pytest.approx(ROCKET_MDOT * ROCKET_DT * ROCKET_STEPS, rel=1e-12)

    dv_vector = _relative_state(sim, powered)[3:] - _relative_state(sim, coasting)[3:]
    dv_measured = float(np.linalg.norm(dv_vector))

    # Direction, before magnitude: the delta-v must lie along +S, the direction that was commanded.
    # A norm alone cannot see a sign flip, so assert the projection too. S turns by phi = 1.5e-3 rad
    # over the burn, costing cos(phi) - 1 = -1.1e-6 of the projection; 1e-5 is 9x that.
    s_hat = _rsw_basis_by_hand(
        _relative_state(sim, coasting)[:3], _relative_state(sim, coasting)[3:])[1]
    assert float(dv_vector @ s_hat) / dv_measured == pytest.approx(1.0, abs=1e-5)

    dv_exact = ROCKET_ISP * STANDARD_GRAVITY * math.log(m0 / m1) * 1e-3          # km/s
    dv_discrete = float(np.sum(ROCKET_THRUST_N * 1e-3 / np.array(masses)) * ROCKET_DT)

    # (a) The twin difference reproduces the sum the scheme actually forms, to the gravity-gradient
    #     bound. This is the check that the model integrates what it claims to.
    assert dv_measured == pytest.approx(dv_discrete, rel=ROCKET_DISCRETE_REL_TOL)

    # (b) ... and therefore Tsiolkovsky, to the first-order mass bias.
    assert dv_measured == pytest.approx(dv_exact, rel=ROCKET_EXACT_REL_TOL)

    # (c) The bias is a prediction, not slack: a left Riemann sum on an increasing `T/m` is LOW by
    #     (dt/2)(f_end - f_start), because the whole step flies at the heavier start-of-step mass.
    predicted_bias = -0.5 * ROCKET_DT * (ROCKET_THRUST_N * 1e-3) * (1.0 / m1 - 1.0 / m0)
    assert dv_measured - dv_exact == pytest.approx(predicted_bias, rel=ROCKET_BIAS_REL_TOL)
    assert predicted_bias / dv_exact == pytest.approx(-8.79e-5, rel=0.05)    # the derived magnitude


def test_mass_bias_halves_with_the_step(db_session_factory: Callable[[], Session]) -> None:
    """
    The excess over Tsiolkovsky is first order in `dt`, so halving `dt` must halve it. That is what
    makes the tolerance above a statement about the scheme rather than a recorded number.
    """
    excesses: List[float] = []
    for dt, steps in ((2.0, 300), (1.0, 600)):
        sim = scenarios.powered_vessel(
            db_session_factory(), n_vessels=2, n_powered=1, p_km=ROCKET_P_KM,
            dry_mass=260.0, fuel_mass=40.0,
            thrust_n=ROCKET_THRUST_N, isp_s=ROCKET_ISP, direction=(0.0, 1.0, 0.0),
        )
        powered = sim.name_to_index["THRUSTER-00"]
        coasting = sim.name_to_index["THRUSTER-01"]
        params = sim.force_model_params[THRUST_MODEL]
        m0 = float(params[powered, 2])
        for _ in range(steps):
            sim.step(dt)
        m1 = float(params[powered, 2])
        dv = float(np.linalg.norm(
            _relative_state(sim, powered)[3:] - _relative_state(sim, coasting)[3:]))
        excesses.append(dv - ROCKET_ISP * STANDARD_GRAVITY * math.log(m0 / m1) * 1e-3)

    assert excesses[1] == pytest.approx(0.5 * excesses[0], rel=0.02)


# ==================================================================================================
# 4. Orbit raising: measured mean `a` against the energy-derived da/dt = 2f/n
# ==================================================================================================

RAISE_A0 = scenarios.EARTH_RADIUS + 550.0            # 6921 km, circular
RAISE_THRUST_N = 0.026                               # ~1e-4 m/s^2 on 300 kg
RAISE_ISP = 300.0
RAISE_DT = 20.0
RAISE_ORBITS = 5
RAISE_PERIOD = 2.0 * math.pi * math.sqrt(RAISE_A0 ** 3 / MU)
RAISE_STEPS = int(round(RAISE_ORBITS * RAISE_PERIOD / RAISE_DT))


def _run_raise(session: Session, direction: Tuple[float, float, float]) -> Tuple[float, float]:
    """Run the raising scenario; return (Delta a of the powered vessel, Delta a of the twin)."""
    sim = scenarios.powered_vessel(
        session, n_vessels=2, n_powered=1, p_km=RAISE_A0,
        dry_mass=260.0, fuel_mass=40.0,
        thrust_n=RAISE_THRUST_N, isp_s=RAISE_ISP, direction=direction,
    )
    powered = sim.name_to_index["THRUSTER-00"]
    control = sim.name_to_index["THRUSTER-01"]
    a0_powered = _semi_major_axis(sim, powered)
    a0_control = _semi_major_axis(sim, control)
    for _ in range(RAISE_STEPS):
        sim.step(RAISE_DT)
    return (_semi_major_axis(sim, powered) - a0_powered,
            _semi_major_axis(sim, control) - a0_control)


def test_tangential_burn_raises_a_at_the_derived_rate(db_session: Session) -> None:
    delta_a, control_drift = _run_raise(db_session, (0.0, 1.0, 0.0))

    # Prediction: da/dt = 2 f(t) sqrt(a^3/mu), f = T*1e-3/m(t), m falling at T/(Isp g0).
    m0 = 300.0
    mdot = RAISE_THRUST_N / (RAISE_ISP * STANDARD_GRAVITY)

    def rate(t: float, a: float) -> float:
        return 2.0 * (RAISE_THRUST_N * 1e-3 / (m0 - mdot * t)) * math.sqrt(a ** 3 / MU)

    predicted = _rk4_scalar(rate, RAISE_A0, 0.0, RAISE_STEPS * RAISE_DT, 2000) - RAISE_A0

    assert delta_a > 0.0
    assert delta_a == pytest.approx(predicted, rel=ORBIT_RAISE_REL_TOL)
    # The thrust-free twin measures the integrator's own drift in `a` in the same arena.
    assert abs(control_drift) < ORBIT_RAISE_REL_TOL * abs(predicted)
    # Sanity on the derivation itself: ~4.5 km over 5 orbits.
    assert predicted == pytest.approx(4.53, rel=0.02)


def test_radial_burn_is_the_axis_control(db_session_factory: Callable[[], Session]) -> None:
    """
    Pure radial thrust raises `a` only through the eccentricity it induces (O(e), e ~ 2.4e-5), so it
    must be negligible against the same magnitude applied along-track. R and S transposed in the
    kernel would make these two runs swap. The measured radial change is indistinguishable from the
    thrust-free twin's own drift, i.e. the secular term really is absent.
    """
    tangential, _ = _run_raise(db_session_factory(), (0.0, 1.0, 0.0))
    radial, _ = _run_raise(db_session_factory(), (1.0, 0.0, 0.0))
    assert abs(radial) < RADIAL_AXIS_MAX_FRACTION * abs(tangential)


# ==================================================================================================
# 5. Fuel exhaustion
# ==================================================================================================

BURNOUT_DT = 10.0
BURNOUT_MDOT = 0.01
BURNOUT_THRUST_N = BURNOUT_MDOT * 300.0 * STANDARD_GRAVITY
BURNOUT_FUEL = 3.995                      # 39.95 steps' worth: the 40th step clamps at dry mass
BURNOUT_STEPS = 40


def test_fuel_exhaustion_stops_the_burn_and_the_body_then_coasts(db_session: Session) -> None:
    """
    Two identical powered vessels burn to exhaustion. Afterwards: the mass is *exactly* the dry mass
    and stops changing, the kernel's contribution is *exactly* zero, and disabling the model on one
    of them changes its trajectory not at all - bit for bit. That last check is the coasting twin:
    "out of fuel" and "model not enabled" must be the same dynamics.
    """
    sim = scenarios.powered_vessel(
        db_session, n_vessels=2, n_powered=2, p_km=RAISE_A0,
        dry_mass=260.0, fuel_mass=BURNOUT_FUEL,
        thrust_n=BURNOUT_THRUST_N, isp_s=300.0, direction=(0.0, 1.0, 0.0),
    )
    burning = sim.name_to_index["THRUSTER-00"]
    twin = sim.name_to_index["THRUSTER-01"]
    params = sim.force_model_params[THRUST_MODEL]
    dry = float(params[burning, 3])

    masses: List[float] = []
    for _ in range(BURNOUT_STEPS):
        sim.step(BURNOUT_DT)
        masses.append(float(params[burning, 2]))

    # Never crossed, and landed exactly on the floor rather than near it.
    assert min(masses) >= dry
    assert float(params[burning, 2]) == dry
    assert float(params[twin, 2]) == dry
    # The burn really did most of its work before the clamp: the last step is the only clamped one.
    assert masses[-2] > dry

    # Thrust is now exactly zero, not merely small.
    out = np.zeros_like(sim.accel_accum)
    thrust_kernel(
        np.array([burning, twin], dtype=np.int64), sim.t, sim.global_states, sim.mu_array,
        sim.parent_indices, params, out,
    )
    assert np.array_equal(out, np.zeros_like(out))

    # Mass stops changing.
    sim.step(BURNOUT_DT)
    assert float(params[burning, 2]) == dry

    # The coasting twin: drop the model from one vessel and the two must stay bit-identical.
    assert np.array_equal(sim.global_states[burning], sim.global_states[twin])
    sim.force_model_mask[twin] &= ~registry.mask_for([THRUST_MODEL])
    sim.resolve_force_models()
    assert sim._cowell_fused_ok is False          # the other vessel still carries the foreign bit
    for _ in range(50):
        sim.step(BURNOUT_DT)
    assert np.array_equal(sim.global_states[burning], sim.global_states[twin])


def test_deplete_mass_is_vectorised_over_the_thrusting_set() -> None:
    """
    Direct check of the depletion arithmetic on a hand-built parameter array: three bodies with
    different thrust levels, one already dry, one with `isp_s = 0` (unconfigured).
    """
    params = np.zeros((5, len(THRUST_PARAM_NAMES)))
    params[1] = [100.0, 300.0, 500.0, 200.0, 0.0, 1.0, 0.0]
    params[2] = [0.0, 0.0, 400.0, 400.0, 0.0, 1.0, 0.0]       # already dry, no thruster
    params[3] = [50.0, 0.0, 300.0, 100.0, 0.0, 1.0, 0.0]      # isp 0: burns nothing
    idx = np.array([1, 2, 3], dtype=np.int64)

    deplete_mass(idx, params, 10.0)

    assert params[1, 2] == pytest.approx(500.0 - 100.0 / (300.0 * STANDARD_GRAVITY) * 10.0, rel=1e-15)
    assert params[2, 2] == 400.0
    assert params[3, 2] == 300.0
    assert np.array_equal(params[0], np.zeros(len(THRUST_PARAM_NAMES)))   # untouched rows
    assert np.array_equal(params[4], np.zeros(len(THRUST_PARAM_NAMES)))

    # The clamp: a long step cannot burn past the dry mass.
    deplete_mass(idx, params, 1.0e9)
    assert params[1, 2] == 200.0


def test_step_does_not_deplete_when_nothing_thrusts(db_session: Session) -> None:
    """`_thrust_idx` stays empty when no body carries the model, so `step()` skips the depletion."""
    sim = scenarios.powered_vessel(db_session, n_vessels=1, n_powered=0)
    assert sim._thrust_idx.size == 0
    sim.step(60.0)
    assert THRUST_MODEL not in sim.force_model_params
