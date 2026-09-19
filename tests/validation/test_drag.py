"""
Validation of the `"drag"` force model (`drag.drag_kernel`).

Each tolerance below is derived before it was measured. The class of bug this file guards against
is a unit slip, a dropped 1/2, or a sign error in `w x r`. Each of those produces a plausible,
smoothly decaying orbit that is wrong by a constant factor. Each check is therefore stated as a
magnitude. Each check was run against real-file mutants of `drag.py` (co-rotation sign flipped, the
1/2 dropped, the 1e3 unit conversion dropped), and each mutant fails the decay, energy and closed-form
checks.

Derivations
-----------
**Decay of a circular orbit.** The specific energy is `E = v^2/2 - mu/r = -mu/(2a)`, and drag does work
`dE/dt = a_drag . v`. With no co-rotation, `a_drag = -(1/2) rho B' |v| v`, where `B' = B * 1e3` is
the ballistic coefficient in 1/km per kg/m^3. That gives `dE/dt = -(1/2) rho B' v^3`. Since
`dE/da = mu/(2 a^2)`,

    da/dt = (2 a^2 / mu) dE/dt = -rho B' a^2 v^3 / mu = -rho B' sqrt(mu a)      (v^2 = mu/a)

This is also Gauss's `da/dt = (2/n) a_T` with `a_T = -(1/2) rho B' v^2`, so the form in the brief is
correct as stated. One orbit loses `2 pi rho B' a^2`.

**Co-rotation.** On a prograde equatorial circular orbit, `w x r` is parallel to `v` with magnitude
`omega r`, so `v_rel = (1 - omega r / v) v`. Drag stays anti-parallel to `v`, with magnitude
`(1/2) rho B' (v - omega r)^2`, so the rate is multiplied by `f = (1 - omega a / v)^2`. On a
retrograde orbit, `w x r` is anti-parallel to `v`, which gives `f = (1 + omega a / v)^2`. At
`a = 6921 km`, where `omega a / v = 0.5047 / 7.5890 = 0.0665`, that is 0.8714 and 1.1374.

**Why the prediction is an ODE, not a product.** Density rises by `exp(|da|/H)`, about 1.7% over
the run, and a prograde body samples less of that rise because it decays more slowly. So the
predicted `Delta a` integrates the mean equation `da/dt = -rho(a) B' sqrt(mu a) f(a)` with a scalar
RK4 in the test. The measured prograde ratio is 0.87047. `f(a0)` alone would say 0.8714, which is
1e-3 away and larger than the ratio tolerance.

Error budget, for the scenario below
-----------------------------------
The scenario is a = 6921 km, T = 5730 s, n = 1.097e-3 /s, dt = 20 s (n dt = 0.0219), 5 orbits
(1433 steps), and `Delta a` of about -1.03 km.

- *RK4 energy drift.* A linear oscillator loses a fraction `(n dt)^6 / 72` of its energy per RK4 step,
  which gives 1.5e-12 per step, 2.2e-9 over the run, and 1.5e-5 km in `a`. That is 1.5e-5 of `Delta a`.
  Kepler motion is not linear, so this only fixes the order of magnitude. Allowing a factor of 5 gives
  7.5e-5. A drag-free control satellite in the same arena measures it directly.
- *Osculating vs mean.* Drag drives an eccentricity oscillation of amplitude
  `e ~ 2 a_T / (v n) = rho B' a`, which is 4.7e-6. Over that radial excursion `a e`, density varies by
  `a e / H` = 5.4e-4 and `v^3` by `3e`. Both vary once per orbit, so they average out of the secular
  rate to first order. What remains is an oscillation in the osculating `a` of amplitude about
  `|da/dt| (a e / H) / n` = 1.7e-5 km, or 1.7e-5 of `Delta a`, and endpoint values carry that scatter.
  Second-order terms are `(a e / H)^2`, about 3e-7.
- *Mean-ODE quadrature.* A scalar RK4 with 2000 steps on a rate that varies by 1% over the run has
  error far below 1e-10.

Budget: 7.5e-5 + 1.7e-5 is about 1e-4, and doubling that for headroom gives `DECAY_REL_TOL` = 2e-4.
Holding the density at `rho(a0)` misses the 0.85% rise, which is 40 times that tolerance.
`test_secular_decay_matches_orbit_averaged_rate` asserts that sensitivity, so the tolerance
provably resolves the density term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import drag, geopotential, gravity, registry, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.drag import (
    DENSITY_MODEL_EXPONENTIAL, DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA,
)
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ
from orbital_engine.simulator import Simulation

ArrF = NDArray[np.float64]

MU = scenarios.MU_EARTH

# --------------------------------------------------------------------------------------------------
# Tolerances, derived (see module docstring for the decay budget).
# --------------------------------------------------------------------------------------------------

# Closed-form case, direct kernel call: about 15 rounded operations (subtractions against the parent
# row, a cross product, two sqrts, an exp, and products), each adding about eps of relative error.
# Subtracting the parent row, which is offset by 1.5e8 km, loses `eps * 1.5e8 / 6938` = 4.8e-12 in
# relative position. That changes the altitude by 3.3e-8 km, the density by
# `3.3e-8 / 60` = 5.5e-10 relative, and the velocity term by a similar ulp-sized amount. So the
# translated case is limited by the offset, and 2e-9 sits about 4 times above it (measured 1.4e-10).
# The un-offset case carries ~15 eps from the kernel plus ~15 eps from the SI reference computation,
# so about 30 eps = 7e-15 (measured 1.5e-14). 1e-13 is 7 to 14 times that. Any unit slip is 1e3, a
# dropped 1/2 is 2, and a flipped co-rotation sign is 1.14.
CLOSED_FORM_REL_TOL = 1e-13
CLOSED_FORM_TRANSLATED_REL_TOL = 2e-9

# See the module docstring. Measured: 4.6e-5 non-rotating, 5.0e-5 prograde, 4.2e-5 retrograde.
DECAY_REL_TOL = 2e-4
# The drag-free control's |Delta a| relative to the drag decay. Derived 1.5e-5 (x5 is 7.5e-5), and
# measured 3.0e-5.
INTEGRATOR_DRIFT_REL_TOL = 7.5e-5
# Holding density constant misses the 0.85% rise over the run. It must fail DECAY_REL_TOL by a clear
# margin, or this test cannot resolve the density term at all.
DENSITY_VARIATION_MIN_REL = 5e-3

# Energy balance: `Delta E` against the trapezoid integral of `a_drag . v` over the step samples.
# RK4 drift is the same 1.5e-5 (x5 = 7.5e-5) relative to the drag work, since `Delta E / E` equals
# `Delta a / a`. The trapezoid rule's error is `(dt^2/12) * mean(P'')`. `P` is constant apart from a
# once-per-orbit ripple of relative size 5.4e-4, so `P'' ~ n^2 * 5.4e-4 * P`, and over whole orbits
# the ripple integrates to ~0 anyway. That leaves (n dt)^2 / 12 * 5.4e-4, about 2e-8. Doubling
# 7.5e-5 gives 1.5e-4, and 2e-4 matches the decay budget. Measured 4.5e-5 and 5.0e-5.
ENERGY_REL_TOL = 2e-4

# Ratio of Delta a (co-rotating) to Delta a (non-rotating) against the same ratio from the mean ODE.
# Both bodies share dt and altitude, so the RK4 drift (3e-5 of each decay) cancels to about
# `3e-5 * |1 - f|`, which is 4e-6. The osculating scatter of 1.7e-5 per endpoint does not cancel,
# giving about 2.4e-5 in quadrature. 1e-4 is 4 times that. A flipped `w x r` sign moves the ratio
# by 30%, and dropping the term moves it by 13%. Measured 4.5e-6 prograde and 3.7e-6 retrograde.
COROTATION_RATIO_REL_TOL = 1e-4

# --------------------------------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------------------------------

ALTITUDE_KM = 550.0
A0_KM = scenarios.EARTH_RADIUS + ALTITUDE_KM        # the constellation's circular radius, 6921 km
# B = C_d A / m for the constellation's own vessel (drag_area 4 m^2, dry_mass 260 kg) with C_d = 2.2.
BALLISTIC_COEFF = 2.2 * 4.0 / 260.0                 # m^2/kg
# rho0 = 2e-11 kg/m^3 is roughly the real density near 300 km, not 550 km. It is chosen so the decay,
# 2 pi rho B' a^2 = 0.204 km per orbit, is about 6e4 times the RK4 drift, while the density still
# changes by under 2% over the run.
RHO0 = 2e-11                                        # kg/m^3
SCALE_HEIGHT_KM = 60.0
H0_KM = A0_KM - EARTH_R_EQ                          # so rho(a0) == RHO0
DT = 20.0
PERIOD = 2.0 * math.pi * math.sqrt(A0_KM**3 / MU)
N_ORBITS = 5
N_STEPS = int(round(N_ORBITS * PERIOD / DT))

_DRAG_COEFFS: Dict[str, float] = dict(
    ballistic_coeff=BALLISTIC_COEFF, rho0=RHO0, h0=H0_KM, scale_height=SCALE_HEIGHT_KM, r_ref=EARTH_R_EQ,
)


def _session() -> Session:
    """An isolated in-memory database, built the way `tests/conftest.py` builds one. A local copy
    exists because the propagation runs are module-scoped and the conftest factory is not."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sat_slots(sim: Simulation) -> List[int]:
    return sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))


# --------------------------------------------------------------------------------------------------
# Independent physics used by the assertions. None of this imports the kernel.
# --------------------------------------------------------------------------------------------------

def _semi_major_axis(y: ArrF) -> ArrF:
    r = np.linalg.norm(y[..., :3], axis=-1)
    v2 = np.einsum("...i,...i->...", y[..., 3:], y[..., 3:])
    a: ArrF = -MU / (2.0 * (0.5 * v2 - MU / r))
    return a


def _specific_energy(y: ArrF) -> ArrF:
    e: ArrF = 0.5 * np.einsum("...i,...i->...", y[..., 3:], y[..., 3:]) - MU / np.linalg.norm(y[..., :3], axis=-1)
    return e


def _drag_accel_si(y: ArrF, omega: float) -> ArrF:
    """The drag acceleration, written from scratch in SI: metres, m/s, kg/m^3 and m^2/kg throughout,
    converted to km/s^2 only at the end. It shares no unit conversion with the kernel."""
    r_m = y[..., :3] * 1e3
    v_m = y[..., 3:] * 1e3
    v_rel = v_m - np.cross(np.array([0.0, 0.0, omega]), r_m)
    h_m = np.linalg.norm(r_m, axis=-1) - EARTH_R_EQ * 1e3
    rho = RHO0 * np.exp(-(h_m - H0_KM * 1e3) / (SCALE_HEIGHT_KM * 1e3))
    a_si = -0.5 * rho[..., None] * BALLISTIC_COEFF * np.linalg.norm(v_rel, axis=-1)[..., None] * v_rel
    out: ArrF = a_si * 1e-3
    return out


def _mean_decay(a0: float, t_end: float, corotation: Callable[[float], float], m: int = 2000) -> float:
    """Integrate `da/dt = -rho(a) B' sqrt(mu a) f(a)` with scalar RK4. B' carries the 1e3."""
    def rate(a: float) -> float:
        rho = RHO0 * math.exp(-((a - EARTH_R_EQ) - H0_KM) / SCALE_HEIGHT_KM)
        return -rho * BALLISTIC_COEFF * 1e3 * math.sqrt(MU * a) * corotation(a)

    h = t_end / m
    a = a0
    for _ in range(m):
        k1 = rate(a)
        k2 = rate(a + 0.5 * h * k1)
        k3 = rate(a + 0.5 * h * k2)
        k4 = rate(a + h * k3)
        a += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return a - a0


def _no_rotation(a: float) -> float:
    return 1.0


def _prograde(a: float) -> float:
    return (1.0 - EARTH_OMEGA * a / math.sqrt(MU / a)) ** 2


def _retrograde(a: float) -> float:
    return (1.0 + EARTH_OMEGA * a / math.sqrt(MU / a)) ** 2


# --------------------------------------------------------------------------------------------------
# The propagation runs, shared by the decay, energy and co-rotation tests.
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayRuns:
    # (N_STEPS + 1, 3, 6) parent-relative states for the equatorial prograde arena:
    # [drag with omega = 0, drag with omega = EARTH_OMEGA, point mass only (control)]
    prograde: ArrF
    # (N_STEPS + 1, 1, 6) for a retrograde equatorial satellite, drag with omega = EARTH_OMEGA.
    retrograde: ArrF
    fused_ok: List[bool]


def _run(inclination_deg: float, omegas: List[float | None]) -> tuple[ArrF, bool]:
    sim = scenarios.earth_constellation(
        _session(), n_sats=len(omegas), n_planes=len(omegas),
        altitude_km=ALTITUDE_KM, inclination_deg=inclination_deg,
    )
    sim.record_history = False
    earth = sim.name_to_index["Earth"]
    sats = _sat_slots(sim)
    for slot, omega in zip(sats, omegas):
        sim.set_propagator(slot, PropagatorType.COWELL)
        sim.enable_force_model(gravity.POINT_MASS_MODEL, bodies=slot)
        if omega is not None:
            sim.enable_force_model(DRAG_MODEL, bodies=slot, omega=omega, **_DRAG_COEFFS)

    track = np.empty((N_STEPS + 1, len(sats), 6), dtype=np.float64)
    track[0] = sim.global_states[sats] - sim.global_states[earth]
    for k in range(1, N_STEPS + 1):
        sim.step(DT)
        track[k] = sim.global_states[sats] - sim.global_states[earth]
    return track, sim._cowell_fused_ok


@pytest.fixture(scope="module")
def runs() -> Iterator[DecayRuns]:
    prograde, ok_p = _run(0.0, [0.0, EARTH_OMEGA, None])
    retrograde, ok_r = _run(180.0, [EARTH_OMEGA])
    yield DecayRuns(prograde=prograde, retrograde=retrograde, fused_ok=[ok_p, ok_r])


# ==================================================================================================
# Closed form, registration and configuration
# ==================================================================================================

# Hand-computed in SI, independently of the kernel. The parent-relative state is
# r = 6938.137 km * (2/3, 2/3, 1/3), so h = 560 km = h0 + H and rho = rho0 / e = 3.6787944117e-13
# kg/m^3. The velocity is v = (-5, 4, 3) km/s. With w x r = omega (-y, x, 0) = (-0.337291, 0.337291, 0),
# v_rel = (-4.662709, 3.662709, 3.0) km/s and |v_rel| = 6.645020 km/s. Then
# a = -0.5 * rho * 0.02 m^2/kg * |v_rel| * v_rel in SI (m/s^2), and dividing by 1e3 gives km/s^2.
_CF_R_KM = (EARTH_R_EQ + 500.0 + 60.0) * np.array([2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0])
_CF_V = np.array([-5.0, 4.0, 3.0])
_CF_PARAMS = np.array([0.02, 1e-12, 500.0, 60.0, EARTH_R_EQ, EARTH_OMEGA,
                       DENSITY_MODEL_EXPONENTIAL])
_CF_EXPECTED = np.array([1.1398299925610358e-10, -8.953733767913337e-11, -7.33369847309106e-11])


def _closed_form_accel(parent_offset: ArrF) -> ArrF:
    state = np.zeros((2, 6))
    state[0] = parent_offset
    state[1, :3] = parent_offset[:3] + _CF_R_KM
    state[1, 3:] = parent_offset[3:] + _CF_V
    params = np.zeros((2, len(DRAG_PARAM_NAMES)))
    params[1] = _CF_PARAMS
    out = np.zeros((2, 3))
    with np.errstate(all="raise"):
        drag.drag_kernel(np.array([1], dtype=np.int64), 0.0, state, np.array([MU, 0.0]),
                         np.array([0, 0], dtype=np.int32), params, out)
    assert np.array_equal(out[0], np.zeros(3)), "the kernel wrote a row outside `indices`"
    result: ArrF = out[1]
    return result


def test_closed_form_acceleration_matches_hand_computed_si_value() -> None:
    """Pins the unit conversion, the 1/2, the density law and the sign of `w x r` in a single state. The
    state is off the equator and has a velocity component along every axis, so no term of
    `w x r` drops out. The second call puts the parent on Earth's heliocentric orbit (1.5e8 km,
    30 km/s), which proves the kernel reads position and velocity relative to the parent."""
    at_origin = _closed_form_accel(np.zeros(6))
    rel = np.abs(at_origin - _CF_EXPECTED) / np.abs(_CF_EXPECTED)
    assert np.all(rel < CLOSED_FORM_REL_TOL), f"relative error {rel} against the SI hand computation"

    translated = _closed_form_accel(np.array([1.2e8, -9.0e7, 3.0e3, 18.0, 24.0, -0.1]))
    rel_t = np.abs(translated - _CF_EXPECTED) / np.abs(_CF_EXPECTED)
    assert np.all(rel_t < CLOSED_FORM_TRANSLATED_REL_TOL), (
        f"relative error {rel_t} with a moving parent: the kernel must use parent-relative r and v")

    # Sensitivity: the check above must be able to see the co-rotation term at all.
    no_rot = -0.5 * (1e-12 / math.e) * 0.02 * 1e3 * math.sqrt(50.0) * _CF_V
    assert np.max(np.abs(no_rot - _CF_EXPECTED) / np.abs(_CF_EXPECTED)) > 0.1


def test_degenerate_rows_contribute_exactly_zero_and_the_kernel_adds() -> None:
    """Three degenerate rows each add exactly 0.0, with floating-point errors set to raise: a row whose
    coefficients were never written (H = 0), a root body (parent is itself, r = 0, and would overflow
    exp if evaluated), and a zero ballistic coefficient. A valid row adds to what `out` already
    holds instead of overwriting it."""
    state = np.zeros((5, 6))
    state[1:, :3] = [7000.0, 0.0, 0.0]
    state[1:, 3:] = [0.0, 7.5, 0.0]
    params = np.zeros((5, len(DRAG_PARAM_NAMES)))
    # Trailing DENSITY_MODEL_EXPONENTIAL: these rows pre-date the layered law and must keep
    # selecting the single exponential, which is also what an unwritten (all-zero) row selects.
    params[2] = [0.02, 1e-12, 0.0, 8.0, EARTH_R_EQ, EARTH_OMEGA,
                 DENSITY_MODEL_EXPONENTIAL]                         # root body: parent is itself
    params[3] = [0.0, 1e-12, 500.0, 60.0, EARTH_R_EQ, EARTH_OMEGA,
                 DENSITY_MODEL_EXPONENTIAL]                         # B = 0
    params[4] = [0.02, 1e-12, 500.0, 60.0, EARTH_R_EQ, 0.0,
                 DENSITY_MODEL_EXPONENTIAL]                         # valid
    parents = np.array([0, 0, 2, 0, 0], dtype=np.int32)
    out = np.full((5, 3), 1.0)
    with np.errstate(all="raise"):
        drag.drag_kernel(np.array([1, 2, 3, 4], dtype=np.int64), 0.0, state, np.zeros(5),
                         parents, params, out)
    assert np.array_equal(out[:4], np.ones((4, 3)))
    assert out[4, 1] < 1.0 and out[4, 0] == 1.0 and out[4, 2] == 1.0

    empty_out = np.zeros((5, 3))
    drag.drag_kernel(np.array([], dtype=np.int64), 0.0, state, np.zeros(5), parents, params, empty_out)
    assert np.array_equal(empty_out, np.zeros((5, 3)))


def test_drag_is_registered_with_its_coefficient_layout() -> None:
    model = registry.get_force_model(DRAG_MODEL)
    assert model.param_names == ("ballistic_coeff", "rho0", "h0", "scale_height", "r_ref", "omega",
                                 "density_model")
    assert model.kernel is drag.drag_kernel
    assert model.validate_bodies is not None
    assert "Vallado" in model.citation


def test_enable_force_model_refuses_drag_on_a_barycentre_parented_body() -> None:
    sim = scenarios.earth_constellation(_session(), n_sats=4, n_planes=4)
    bary = sim.name_to_index["Earth Barycenter"]
    sats = np.array(_sat_slots(sim), dtype=np.int64)
    sim.parent_indices[sats[:2]] = bary

    mask_before = sim.force_model_mask.copy()
    with pytest.raises(ValueError, match="barycentre"):
        sim.enable_force_model(DRAG_MODEL, sats, omega=EARTH_OMEGA, **_DRAG_COEFFS)
    assert np.array_equal(sim.force_model_mask, mask_before)
    assert DRAG_MODEL not in sim.force_model_params

    sim.enable_force_model(DRAG_MODEL, sats[2:], omega=EARTH_OMEGA, **_DRAG_COEFFS)
    bit = np.uint64(1) << np.uint64(registry.get_force_model(DRAG_MODEL).bit)
    assert np.all(sim.force_model_mask[sats[2:]] & bit)


def test_drag_composes_with_point_mass_and_j2() -> None:
    """With all three models enabled through `enable_force_model`, `accelerations()` equals the sum
    of the three kernels called directly. The sum is exact, because every kernel only adds."""
    sim = scenarios.earth_constellation(_session(), n_sats=6, n_planes=3)
    sats = np.array(_sat_slots(sim), dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats)
    sim.enable_force_model(geopotential.J2_MODEL, sats, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    sim.enable_force_model(DRAG_MODEL, sats, omega=EARTH_OMEGA, **_DRAG_COEFFS)

    total = sim.accelerations(0.0).copy()
    expected = np.zeros_like(total)
    args = (sim.global_states, sim.mu_array, sim.parent_indices)
    gravity.point_mass_gravity_kernel(sats, 0.0, *args, np.zeros((sim.max_capacity, 0)), expected)
    geopotential.j2_kernel(sats, 0.0, *args, sim.force_model_params[geopotential.J2_MODEL], expected)
    drag_only = np.zeros_like(total)
    drag.drag_kernel(sats, 0.0, *args, sim.force_model_params[DRAG_MODEL], drag_only)
    expected += drag_only

    assert np.array_equal(total[sats], expected[sats])
    # Drag is nonzero and opposes the inertial velocity on these prograde orbits.
    v_rel_earth = sim.global_states[sats, 3:] - sim.global_states[sim.name_to_index["Earth"], 3:]
    assert np.all(np.linalg.norm(drag_only[sats], axis=1) > 0.0)
    power = np.einsum("ij,ij->i", drag_only[sats], v_rel_earth)
    assert np.all(power < 0.0)


def test_cowell_bodies_with_drag_fall_back_to_the_numpy_path() -> None:
    """`kernels.cowell_rk4_step` is fused for `point_mass_gravity` and `j2` only. A drag bit must
    disqualify it, or the compiled path would silently drop drag."""
    sim = scenarios.earth_constellation(_session(), n_sats=2, n_planes=2)
    sats = _sat_slots(sim)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats)
    assert sim._cowell_fused_ok
    sim.enable_force_model(DRAG_MODEL, sats[1], omega=0.0, **_DRAG_COEFFS)
    assert not sim._cowell_fused_ok


# ==================================================================================================
# Orbit-level validation: decay, energy, co-rotation
# ==================================================================================================

def test_secular_decay_matches_orbit_averaged_rate(runs: DecayRuns) -> None:
    """`Delta a` over 5 orbits under Cowell with `point_mass_gravity` and non-rotating drag, compared
    with `da/dt = -rho(a) B' sqrt(mu a)`. The budget is in the module docstring."""
    assert runs.fused_ok == [False, False]
    a = _semi_major_axis(runs.prograde)
    assert np.allclose(a[0], A0_KM, rtol=1e-12)
    t_end = N_STEPS * DT
    measured = float(a[-1, 0] - a[0, 0])
    predicted = _mean_decay(float(a[0, 0]), t_end, _no_rotation)

    assert predicted == pytest.approx(-1.028, rel=1e-3), "scenario drifted from the derived budget"
    err = abs(measured - predicted) / abs(predicted)
    assert err < DECAY_REL_TOL, (
        f"Delta a = {measured:.9f} km, orbit-averaged theory {predicted:.9f} km, "
        f"relative error {err:.3e} (tolerance {DECAY_REL_TOL:.1e})")

    control = abs(float(a[-1, 2] - a[0, 2])) / abs(predicted)
    assert control < INTEGRATOR_DRIFT_REL_TOL, f"drag-free RK4 drift is {control:.3e} of the decay"

    constant_density = -RHO0 * BALLISTIC_COEFF * 1e3 * math.sqrt(MU * float(a[0, 0])) * t_end
    assert abs(constant_density - predicted) / abs(predicted) > DENSITY_VARIATION_MIN_REL
    assert abs(measured - constant_density) / abs(constant_density) > DECAY_REL_TOL


@pytest.mark.parametrize("column, omega", [(0, 0.0), (1, EARTH_OMEGA)], ids=["non_rotating", "co_rotating"])
def test_energy_change_equals_integrated_drag_power(runs: DecayRuns, column: int, omega: float) -> None:
    """`E_end - E_start` must equal the trapezoid integral of `a_drag . v` over the step samples,
    with `a_drag` recomputed in SI. The power uses the inertial velocity, not `v_rel`: the
    atmosphere does work on the body in the inertial frame."""
    y = runs.prograde[:, column]
    energy = _specific_energy(y)
    power = np.einsum("ij,ij->i", _drag_accel_si(y, omega), y[:, 3:])
    work = float(DT * (0.5 * power[0] + power[1:-1].sum() + 0.5 * power[-1]))
    delta_e = float(energy[-1] - energy[0])

    err = abs(delta_e - work) / abs(work)
    assert err < ENERGY_REL_TOL, (
        f"Delta E = {delta_e:.9e}, integral of drag power = {work:.9e}, relative error {err:.3e}")


def test_corotating_atmosphere_scales_decay_prograde_down_and_retrograde_up(runs: DecayRuns) -> None:
    """A prograde equatorial orbit moves through a co-rotating atmosphere more slowly than its inertial
    speed, so it decays less. A retrograde orbit moves through it faster, so it decays more. Each
    ratio to the non-rotating decay matches the mean-ODE ratio. `f = (1 -/+ omega a / v)^2` evaluated
    at a0 is only a first approximation to that ratio (see the module docstring)."""
    a_p = _semi_major_axis(runs.prograde)
    a_r = _semi_major_axis(runs.retrograde)
    t_end = N_STEPS * DT
    hz = np.cross(runs.retrograde[0, 0, :3], runs.retrograde[0, 0, 3:])[2]
    assert hz < 0.0, "the 180 deg constellation must actually be retrograde"

    nonrot = float(a_p[-1, 0] - a_p[0, 0])
    pred_nonrot = _mean_decay(float(a_p[0, 0]), t_end, _no_rotation)

    for label, delta, a0, law, f0 in (
        ("prograde", float(a_p[-1, 1] - a_p[0, 1]), float(a_p[0, 1]), _prograde, 0.8714),
        ("retrograde", float(a_r[-1, 0] - a_r[0, 0]), float(a_r[0, 0]), _retrograde, 1.1374),
    ):
        predicted = _mean_decay(a0, t_end, law)
        assert abs(predicted - _mean_decay(a0, t_end, _no_rotation) * law(a0)) / abs(predicted) < 3e-3
        assert law(A0_KM) == pytest.approx(f0, abs=1e-4)

        assert abs(delta - predicted) / abs(predicted) < DECAY_REL_TOL, (
            f"{label}: Delta a = {delta:.9f} km against theory {predicted:.9f} km")

        ratio = delta / nonrot
        ratio_pred = predicted / pred_nonrot
        err = abs(ratio - ratio_pred) / ratio_pred
        assert err < COROTATION_RATIO_REL_TOL, (
            f"{label}: decay ratio {ratio:.7f} against theory {ratio_pred:.7f}, relative error {err:.3e}")
