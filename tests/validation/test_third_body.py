"""
Validation of the `"third_body"` force model (`thirdbody.py`): the point-mass pull of a named perturber on a
body integrated relative to its parent, in direct-minus-indirect form.

The headline check turns the Moon from a *comparison* into a *verification*. In
`scenarios.sun_earth_moon(moon_mu=0.0)` Earth's heliocentric orbit is exactly Keplerian on both sides,
because the Moon is massless. So the Moon on Cowell + `point_mass_gravity` + `third_body`(Sun) models
exactly the physics `reference.py`'s N-body truth integrates. Every remaining difference is numerical.
The checks, in order:

1. Pointwise. A hand-computed geometry, including the indirect term's sign. Then the solar tide on the real
   Moon against `reference.nbody_acceleration`, which shares no code with the kernel.
2. Verification against truth over 30 days. The engine freezes the perturber within a step
   (`thirdbody.py`'s derivation), so the error is **first order** in dt, and is predicted *vectorially* as
   `(h/2) dr/dtau`. Here `dr/dtau` is the sensitivity of the truth's Moon to lagging the Sun, computed
   entirely from `reference.py`.
3. The freeze really is the only residual. A test-local oracle feeds the kernel the Sun at each RK4
   stage's true time, and convergence becomes fourth order.
4. Without `third_body` the error is back at comparison size.
5. Negative control. A 2 % error in the perturbation fails check 2 by two orders of magnitude.
6. Configuration: perturber validation, the fused-kernel fallback, and name resolution in the sweep.

Positions are compared **relative to Earth**. The Moon's absolute error also contains Earth's own
heliocentric Kepler-versus-DOP853 difference, measured at 2.3e-3 km (1.5e-11 relative). That is
unrelated to this model, and it would floor the fourth-order scan in check 3.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, Iterator, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import reference, scenarios, sweep
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.simulator import Simulation
from orbital_engine.thirdbody import THIRD_BODY_MODEL, third_body_kernel

ArrF = NDArray[np.float64]
EPS = float(np.finfo(np.float64).eps)

DAY = 86400.0
HORIZON = 30.0 * DAY
TRUTH = dict(rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL)


# ==================================================================================================
# 1. Pointwise
# ==================================================================================================

# Perturber mu_s = 1e6 at distance D = 1e4 from the parent, body at R = 1e3. Collinear, each written as
# a scalar formula independent of the kernel's vector code:
#   near side r = +R x:  mu_s (1/(D-R)^2 - 1/D^2) = +2.3457e-3 x   (toward the perturber)
#   far side  r = -R x:  mu_s (1/(D+R)^2 - 1/D^2) = -1.7355e-3 x   (away from it: the tide stretches)
# Without the indirect term the far side would read +8.26e-3, pointing the *wrong way*. With its
# sign flipped the near side would read 2.2e-2, about ten times too large. Perpendicular, r = +R y:
#   mu_s (D/|d|^3 - 1/D^2) x  -  mu_s R/|d|^3 y,   |d| = sqrt(D^2 + R^2)   (compressive)
# The whole geometry is offset by a non-zero parent position, so a kernel reading absolute rather than
# parent-relative rows fails too. Rounding: the collinear terms cancel to 0.19 relative, which costs
# under one digit on top of about ten rounded operations, so about 1e-14. Budget 1e-13.
CLOSED_FORM_REL_TOL = 1e-13


def test_closed_form_geometry_including_the_indirect_sign() -> None:
    mu_s, big_d, r = 1e6, 1e4, 1e3
    offset = np.array([5.0e3, -7.0e3, 2.0e3])
    # Slots: 0 parent, 1 perturber, 2..4 bodies.
    state = np.zeros((5, 6))
    state[:, :3] = offset
    state[1, :3] += [big_d, 0.0, 0.0]
    state[2, :3] += [r, 0.0, 0.0]
    state[3, :3] += [-r, 0.0, 0.0]
    state[4, :3] += [0.0, r, 0.0]
    mu = np.array([3.0e5, mu_s, 0.0, 0.0, 0.0])
    parents = np.zeros(5, dtype=np.int32)
    params = np.zeros((5, 1))
    params[2:, 0] = 1.0
    out = np.zeros((5, 3))

    third_body_kernel(np.arange(2, 5, dtype=np.int64), 0.0, state, mu, parents, params, out)

    d_perp3 = (big_d ** 2 + r ** 2) ** 1.5
    expected = np.array([
        [mu_s * (1.0 / (big_d - r) ** 2 - 1.0 / big_d ** 2), 0.0, 0.0],
        [mu_s * (1.0 / (big_d + r) ** 2 - 1.0 / big_d ** 2), 0.0, 0.0],
        [mu_s * (big_d / d_perp3 - 1.0 / big_d ** 2), -mu_s * r / d_perp3, 0.0],
    ])
    assert expected[0, 0] > 0.0 > expected[1, 0], "guard: the tide stretches along the perturber line"

    err = np.linalg.norm(out[2:] - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert np.all(err < CLOSED_FORM_REL_TOL), (out[2:], expected, err)
    assert np.all(out[:2] == 0.0), "the kernel wrote rows outside `indices`"


def test_kernel_is_additive_and_a_no_op_on_empty_indices() -> None:
    state = np.zeros((3, 6))
    state[1, :3] = [1e4, 0.0, 0.0]
    state[2, :3] = [1e3, 0.0, 0.0]
    mu = np.array([1.0, 1e6, 0.0])
    parents = np.zeros(3, dtype=np.int32)
    params = np.array([[0.0], [0.0], [1.0]])
    out = np.full((3, 3), 7.0)

    third_body_kernel(np.empty(0, dtype=np.int64), 0.0, state, mu, parents, params, out)
    assert np.all(out == 7.0)

    third_body_kernel(np.array([2], dtype=np.int64), 0.0, state, mu, parents, params, out)
    expected = 7.0 + 1e6 * (1.0 / 9e3 ** 2 - 1.0 / 1e4 ** 2)
    assert out[2, 0] == pytest.approx(expected, rel=1e-13) and out[2, 1] == 7.0


# The real Moon at epoch. The engine's `third_body` row is compared against the N-body truth's relative
# acceleration with Earth's pull removed: (a_Moon - a_Earth)_nbody + mu_E r/r^3. That isolates the solar
# tide, about 2e-8 km/s^2. It is formed by cancelling two 5.9e-6 km/s^2 terms on each side, and that
# cancellation costs 5.9e-6/2e-8 ~ 300, i.e. about 3e2 * 10 eps ~ 7e-13 relative. Budget 1e-11. A
# dropped indirect term is O(3e2) relative, and a 1 % coefficient error is 1e-2.
# Magnitude band: the exact tide lies between mu_s r/d^3 (perpendicular) and
# 2 (1 + 1.5 r/d) mu_s r/d^3 = 2.0077 mu_s r/d^3 (collinear), so asserting (0.99, 2.01) is a true bound.
# The epoch Moon sits close to the Sun line, at 1.975.
# Measured: tide 1.2e-14, total acceleration 2.1e-16.
TIDE_POINTWISE_REL_TOL = 1e-11


# ==================================================================================================
# Shared scenario and truth
# ==================================================================================================

def _new_session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _moon_on_cowell(third_body: bool = True) -> Simulation:
    sim = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    sim.record_history = False
    moon = sim.name_to_index["Moon"]
    sim.set_propagator(moon, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", moon)
    if third_body:
        sim.enable_force_model(THIRD_BODY_MODEL, moon, perturber=float(sim.name_to_index["Sun"]))
    return sim


def _moon_rel_earth(sim: Simulation) -> ArrF:
    g = sim.global_states
    out: ArrF = g[sim.name_to_index["Moon"], :3] - g[sim.name_to_index["Earth"], :3]
    return out


def _run(sim: Simulation, dt: float) -> ArrF:
    for _ in range(int(round(HORIZON / dt))):
        sim.step(dt)
    return _moon_rel_earth(sim)


def _lagged_truth(tau: float) -> ArrF:
    """
    Truth Moon-relative-to-Earth at the horizon, with the Sun running `tau` seconds *behind*, so that
    Sun - Earth follows r_s(t - tau). Built only from `reference.py`: Sun and Earth are integrated
    back by `tau` with DOP853 under time reversal (negated velocities), and the Sun's state relative
    to Earth is replaced by the result. With a massless Moon, Sun-Earth is an isolated two-body pair,
    so every later r_s(t) is shifted by exactly `tau`.
    """
    sim = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    sun, earth = sim.name_to_index["Sun"], sim.name_to_index["Earth"]
    g = sim.global_states
    if tau != 0.0:
        pair = [sun, earth]
        back = reference.integrate_nbody(
            sim.mu_array[pair].copy(), g[pair, :3].copy(), -math.copysign(1.0, tau) * g[pair, 3:],
            np.array([0.0, abs(tau)]), names=["Sun", "Earth"], **TRUTH,
        )
        rel_pos = back.position_of("Sun")[-1] - back.position_of("Earth")[-1]
        rel_vel = back.velocity_of("Sun")[-1] - back.velocity_of("Earth")[-1]
        g[sun, :3] = g[earth, :3] + rel_pos
        g[sun, 3:] = g[earth, 3:] - math.copysign(1.0, tau) * rel_vel
    ref = reference.reference_for(sim, np.array([0.0, HORIZON]), **TRUTH)
    out: ArrF = ref.position_of("Moon")[-1] - ref.position_of("Earth")[-1]
    return out


@pytest.fixture(scope="module")
def truth() -> Iterator[Dict[str, ArrF]]:
    """The 30-day truth, and its sensitivity dr/dtau to lagging the Sun (central difference, 1 h).
    A 15-minute difference agreed to 1.2e-6 relative, so the difference step is not a factor."""
    pytest.importorskip("scipy", reason="reference integration requires the [test] or [reference] extra")
    lag = 3600.0
    yield {
        "rel": _lagged_truth(0.0),
        "dr_dtau": (_lagged_truth(lag) - _lagged_truth(-lag)) / (2.0 * lag),
    }


def test_solar_tide_on_the_moon_matches_the_nbody_acceleration() -> None:
    sim = _moon_on_cowell()
    moon, earth = sim.name_to_index["Moon"], sim.name_to_index["Earth"]

    engine_total = sim.accelerations(0.0)[moon].copy()
    params = sim.force_model_params[THIRD_BODY_MODEL]
    engine_tide = np.zeros_like(sim.accel_accum)
    third_body_kernel(np.array([moon], dtype=np.int64), 0.0, sim.global_states, sim.mu_array,
                      sim.parent_indices, params, engine_tide)

    physical = np.flatnonzero(sim.active_mask & ~sim.is_system)
    row = {int(s): k for k, s in enumerate(physical)}
    a = reference.nbody_acceleration(sim.global_states[physical, :3], sim.mu_array[physical])
    nbody_rel = a[row[moon]] - a[row[earth]]
    r = _moon_rel_earth(sim)
    nbody_tide = nbody_rel + sim.mu_array[earth] * r / np.linalg.norm(r) ** 3

    # Order of magnitude first: between mu_s r/d^3 and 2 mu_s r/d^3 (1.5e-8 .. 3.1e-8 km/s^2).
    d = np.linalg.norm(sim.global_states[sim.name_to_index["Sun"], :3] - sim.global_states[earth, :3])
    scale = scenarios.MU_SUN * np.linalg.norm(r) / d ** 3
    assert 0.99 * scale < np.linalg.norm(engine_tide[moon]) < 2.01 * scale

    tide_err = np.linalg.norm(engine_tide[moon] - nbody_tide) / np.linalg.norm(nbody_tide)
    assert tide_err < TIDE_POINTWISE_REL_TOL, tide_err
    total_err = np.linalg.norm(engine_total - nbody_rel) / np.linalg.norm(nbody_rel)
    assert total_err < TIDE_POINTWISE_REL_TOL, total_err


# ==================================================================================================
# 2. Verification: first order, predicted as a vector
# ==================================================================================================

# Derivation (thirdbody.py, "The approximation"). The perturber is frozen at step start, and RK4's
# effective evaluation time is mid-step (sum b_i c_i = 1/2). The engine therefore integrates the
# Sun lagged by h/2, and to leading order
#
#     err(h) = r_engine - r_truth = (h/2) dr/dtau + O(h^2)
#
# where dr/dtau is the truth's own sensitivity to a Sun lag (fixture above, reference.py only). It is
# 1.50e-3 km/s, giving 2.71 km at h = 3600 s. RK4's own error at 3600 s is about 5e-4 km, from
# check 3's 2.1e-4 km at 2700 s scaled by (4/3)^4. That is 2e-4 of the freeze term and negligible.
#
# The O(h^2) remainder. Per step, frozen RK4's position error is -h^3 a'/6, where a pure h/2 lag gives
# -h^3 a'/4. The difference, +h^3 a'/12, sums over the run to (h^2/12)|Delta a_p|. |Delta a_p| is
# at most twice the tide, 6e-8 km/s^2, so the direct part is 0.065 km at h = 3600 s, 2.4 % of the leading
# term. Allowing a factor of 3 for the orbit's dynamical growth of that offset gives
# |err - pred| / |pred| < 0.08 at 3600 s. The ratio falls as h, so at 1800 s it must be under 0.6 times
# its 3600 s value (0.5 plus rounding headroom).
#
# Convergence order. err(3600)/err(1800) = 2 (1 - m)/(1 - m/2) with m = 0.08 gives 1.92, and m = 0
# gives 2. Band (1.8, 2.2). Second order would be 4 and RK4's native order 16, so this band is
# a statement that the freeze dominates.
#
# Measured: |err| = 2.6379 and 1.3360 km (ratio 1.974). Mismatch against the prediction 2.64e-2 and
# 1.33e-2, and cos(err, pred) > 0.997 at every h tried. At 21600 s the mismatch is 9.7 %, where RK4's
# 1.0 km is no longer negligible.
LAG_MISMATCH_TOL_3600 = 0.08
LAG_MISMATCH_SHRINK = 0.6
FIRST_ORDER_RATIO = (1.8, 2.2)


def test_moon_with_third_body_converges_to_truth_at_the_derived_first_order(
    truth: Dict[str, ArrF],
) -> None:
    errors = {}
    mismatch = {}
    for dt in (3600.0, 1800.0):
        sim = _moon_on_cowell()
        assert not sim._cowell_fused_ok, "guard: third_body must take the NumPy RK4Integrator path"
        err = _run(sim, dt) - truth["rel"]
        pred = 0.5 * dt * truth["dr_dtau"]
        errors[dt] = float(np.linalg.norm(err))
        mismatch[dt] = float(np.linalg.norm(err - pred) / np.linalg.norm(pred))

    ratio = errors[3600.0] / errors[1800.0]
    assert FIRST_ORDER_RATIO[0] < ratio < FIRST_ORDER_RATIO[1], (errors, ratio)
    assert mismatch[3600.0] < LAG_MISMATCH_TOL_3600, (errors, mismatch)
    assert mismatch[1800.0] < LAG_MISMATCH_SHRINK * mismatch[3600.0], mismatch


# ==================================================================================================
# 3. The freeze is the only residual: a per-stage oracle restores fourth order
# ==================================================================================================

# The same run, with `sim.accelerations` wrapped so that the Sun's row, relative to Earth's frozen row,
# holds its value at each stage's *true* time. The values come from a Keplerian twin stepped at h/2.
# Nothing in the engine changes. If the freeze were the only approximation, the residual would be RK4's
# own truncation, which is fourth order.
#
# Predicted magnitude: the two-body Cowell scan measured 1.0e-2 km per orbit at 128 steps/orbit on
# a = 8000 km. At 21600 s the Moon gets 109 steps/orbit, so 1.0e-2 (128/109)^4 (3.84e5/8e3) ~ 0.9 km
# per orbit, over about 1.1 orbits. Measured 1.04 km. The floor is the truth's tolerance error (1.8e-8 km
# from rtol 1e-13 against 1e-12) plus rebasing rounding on Earth's 1.5e8 km coordinates (~1e-7 km,
# test_cowell_propagator.py). The scan stops four decades above that.
# Measured: 1.04, 5.84e-2, 3.43e-3, 2.07e-4 km, ratios 17.9, 17.0, 16.5. At 675 s the error is still
# 8.5e-7 km, ratio 15.1. Band (12, 24), matching test_cowell_propagator.py.
ORACLE_STEPS = (21600.0, 10800.0, 5400.0, 2700.0)
ORACLE_RATIO = (12.0, 24.0)
ORACLE_ERR_AT_21600_KM = (0.3, 3.0)


def _oracle_error(dt: float, rel_truth: ArrF) -> float:
    n = int(round(HORIZON / dt))
    kepler = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    kepler.record_history = False
    sun, earth = kepler.name_to_index["Sun"], kepler.name_to_index["Earth"]
    table = np.empty((2 * n + 1, 3))
    table[0] = kepler.global_states[sun, :3] - kepler.global_states[earth, :3]
    for k in range(1, 2 * n + 1):
        kepler.step(0.5 * dt)
        table[k] = kepler.global_states[sun, :3] - kepler.global_states[earth, :3]

    sim = _moon_on_cowell()
    real = sim.accelerations

    def staged(t: float, state: ArrF) -> ArrF:
        saved = state[sun].copy()
        state[sun, :3] = state[earth, :3] + table[int(round(t / (0.5 * dt)))]
        out = real(t, state)
        state[sun] = saved
        return out

    sim.accelerations = staged  # type: ignore[method-assign]
    return float(np.linalg.norm(_run(sim, dt) - rel_truth))


def test_perturber_oracle_restores_fourth_order(truth: Dict[str, ArrF]) -> None:
    errors = [_oracle_error(dt, truth["rel"]) for dt in ORACLE_STEPS]
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert ORACLE_ERR_AT_21600_KM[0] < errors[0] < ORACLE_ERR_AT_21600_KM[1], errors
    assert all(ORACLE_RATIO[0] < q < ORACLE_RATIO[1] for q in ratios), (errors, ratios)


# ==================================================================================================
# 4. Without third_body: back to comparison size
# ==================================================================================================

# Same bound as test_reference_agreement.py's comparison: the solar tide acting coherently for 30
# days gives 0.5 * 3.05e-8 * (2.59e6)^2 = 1.0e5 km as a ceiling, and 1e3 km as a floor. The
# third_body run must sit at least 1e3 times below it. Measured: 2.48e4 km without, 2.64 km with.
def test_moon_without_third_body_is_comparison_sized(truth: Dict[str, ArrF]) -> None:
    without = float(np.linalg.norm(_run(_moon_on_cowell(third_body=False), 3600.0) - truth["rel"]))
    assert 1.0e3 < without < 1.02e5, without
    with_model = 0.5 * 3600.0 * float(np.linalg.norm(truth["dr_dtau"])) * (1.0 + LAG_MISMATCH_TOL_3600)
    assert without > 1.0e3 * with_model, (without, with_model)


# ==================================================================================================
# 5. Negative control: a few-percent error in the perturbation must fail check 2
# ==================================================================================================

# The perturbation displaces the Moon by 2.48e4 km over 30 days (check 4), so a 2 % error in it
# adds about 500 km against a 2.7 km prediction. That is a mismatch of roughly 180, far above 0.08.
# Measured: 558 km, mismatch 205. A physics error does not shrink with h, so check 2's ratio would
# fail too (about 1.0). The mismatch check is the one that also bounds errors too small to move that ratio.
def test_two_percent_error_in_the_perturbation_is_caught(truth: Dict[str, ArrF]) -> None:
    sim = _moon_on_cowell()

    def inflated(indices: NDArray[np.int64], t: float, state: ArrF, mu_array: ArrF,
                 parent_indices: NDArray[np.int32], params: ArrF, out: ArrF) -> None:
        before = out[indices].copy()
        third_body_kernel(indices, t, state, mu_array, parent_indices, params, out)
        out[indices] += 0.02 * (out[indices] - before)

    sim._resolved_force_models = [
        dataclasses.replace(rm, kernel=inflated) if rm.name == THIRD_BODY_MODEL else rm
        for rm in sim._resolved_force_models
    ]
    err = _run(sim, 3600.0) - truth["rel"]
    pred = 1800.0 * truth["dr_dtau"]
    mismatch = float(np.linalg.norm(err - pred) / np.linalg.norm(pred))
    assert mismatch > 10.0 * LAG_MISMATCH_TOL_3600, f"2 % perturbation error went undetected: {mismatch:.3e}"


# ==================================================================================================
# 6. Configuration
# ==================================================================================================

def _config_sim() -> Tuple[Simulation, Dict[str, int]]:
    sim = scenarios.sun_earth_moon(_new_session(), moon_mu=0.0)
    return sim, dict(sim.name_to_index)


@pytest.mark.parametrize("perturber, match", [
    ("Moon", "themselves"),
    ("Earth", "Keplerian parent"),
    ("EMB", "barycentre"),
    (None, "needs a 'perturber'"),
    (2.5, "not an arena slot"),
    (-1.0, "not an arena slot"),
])
def test_invalid_perturbers_are_rejected_before_any_mutation(perturber: object, match: str) -> None:
    sim, idx = _config_sim()
    moon = idx["Moon"]
    mask_before = sim.force_model_mask.copy()
    kwargs = {} if perturber is None else {
        "perturber": float(idx[perturber]) if isinstance(perturber, str) else perturber}
    with pytest.raises(ValueError, match=match):
        sim.enable_force_model(THIRD_BODY_MODEL, moon, **kwargs)  # type: ignore[arg-type]
    assert np.array_equal(sim.force_model_mask, mask_before)
    assert THIRD_BODY_MODEL not in sim.force_model_params


def test_inactive_and_massless_perturbers_are_rejected() -> None:
    sim, idx = _config_sim()
    sun = idx["Sun"]
    sim.active_mask[sun] = False
    with pytest.raises(ValueError, match="inactive"):
        sim.enable_force_model(THIRD_BODY_MODEL, idx["Moon"], perturber=float(sun))

    sat = scenarios.two_body(_new_session())   # massless secondary: as a perturber it would add nothing
    with pytest.raises(ValueError, match="massless"):
        sat.enable_force_model(THIRD_BODY_MODEL, sat.name_to_index["Primary"],
                               perturber=float(sat.name_to_index["Secondary"]))


def test_sweep_resolves_the_perturber_by_name() -> None:
    sim, idx = _config_sim()
    config = sweep.ModelConfig(
        name="moon+sun", propagator=PropagatorType.COWELL, dt=3600.0, bodies=["Moon"],
        force_models=(
            sweep.ForceModelSpec("point_mass_gravity"),
            sweep.ForceModelSpec(THIRD_BODY_MODEL, body_coefficients={"perturber": "Sun"}),
        ),
    )
    applied = sweep.apply_config(sim, config)
    assert applied.tolist() == [idx["Moon"]]
    assert sim.force_model_params[THIRD_BODY_MODEL][idx["Moon"], 0] == float(idx["Sun"])
    assert not sim._cowell_fused_ok

    bad = dataclasses.replace(config, force_models=(
        sweep.ForceModelSpec(THIRD_BODY_MODEL, body_coefficients={"perturber": "Jupiter"}),))
    with pytest.raises(KeyError, match="Jupiter"):
        sweep.apply_config(_config_sim()[0], bad)
