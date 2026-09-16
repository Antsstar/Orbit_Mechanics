"""
Validation of the J2 term in the DOP853 reference truth (`reference.py`), and of the engine's Cowell +
`point_mass_gravity` + `j2` configuration against it.

`reference.py` writes the J2 field from a spherical-coordinate gradient, projected onto `r_hat` and the
spin axis; `geopotential.j2_kernel` writes Curtis' Cartesian components. Neither imports the other. The
checks below are, in order:

1. Invariants of the truth itself: specific energy including the J2 potential, and h_z, for a massless
   satellite about a fixed oblate Earth; linear momentum when the oblate body is massive.
2. Secular nodal regression against the first-order theory -(3/2) n J2 (R/p)^2 cos i, with the
   short-period amplitude derived and asserted too.
3. Verification: engine Cowell + point_mass_gravity + j2 converges to the truth at fourth order, down
   to a floor predicted from the truth's own tolerance error and RK4 rounding.
4. Negative control: the truth with its spin-axis coefficient doubled - exactly Curtis' z-term
   `(5 s^2 - 3)` -> `(5 s^2 - 5)` - fails check 3 by six orders of magnitude.

Every tolerance is derived in the comment above it before any measurement; the measured value is
quoted next to it. Earth constants are restated as literals here rather than imported from
`geopotential.py`, so the truth is not silently re-pointed if the engine's constants change; one test
checks the two sets still agree.

Geometry used throughout: `earth_constellation` puts satellites on circular orbits of radius
r = 6371 + 550 = 6921 km (so p = a = r), inclination 53 deg. With R = 6378.137 km:
(R/r)^2 = 0.84928, n = sqrt(mu/r^3) = 1.0965e-3 rad/s, period T = 5730 s,
J2 (R/r)^2 = 9.1945e-4.
"""
from __future__ import annotations

import ast
import inspect
import math
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import geopotential, reference, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.database import Base
from orbital_engine.simulator import Simulation

pytest.importorskip("scipy", reason="reference integration requires the [test] or [reference] extra")

ArrF = NDArray[np.float64]

# Restated, not imported (see module docstring). EGM96 J2 and WGS-84 equatorial radius.
J2 = 1.0826266835e-3
R_EQ = 6378.137
MU = scenarios.MU_EARTH
EARTH_OBLATENESS = {"Earth": (J2, R_EQ)}

EPS = float(np.finfo(np.float64).eps)
ORBIT_RADIUS = scenarios.EARTH_RADIUS + 550.0
INCLINATION = math.radians(53.0)
MEAN_MOTION = math.sqrt(MU / ORBIT_RADIUS ** 3)
PERIOD = 2.0 * math.pi / MEAN_MOTION
J2_R2 = J2 * (R_EQ / ORBIT_RADIUS) ** 2                       # 9.19e-4, the small parameter

TRUTH = dict(rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL)


def _sat_indices(sim: Simulation) -> NDArray[np.int64]:
    return np.array([i for n, i in sim.name_to_index.items() if n.startswith("SAT-")], dtype=np.int64)


# ==================================================================================================
# 0. The field itself, pointwise
# ==================================================================================================

# gm = 1e6, J2 = 1e-3, R = 1000, r = 2000: 3 gm J2 R^2 / r^4 = 3e9 / 1.6e13 = 1.875e-4.
# Equator: radial coefficient (5*0 - 1)/2 = -1/2 -> -9.375e-5 along r_hat. Pole: (5 - 1)/2 - 1 = +1
# -> +1.875e-4 along z. About ten rounded operations on non-representable intermediates: ~10 eps =
# 2.2e-15 relative; budget 1e-14.
FIELD_CLOSED_FORM_REL_TOL = 1e-14

# Two independent transcriptions of the same function, each ~10 eps (the reference normalises r_hat
# first, the engine does not, so their rounding is uncorrelated): ~4.4e-15; budget 1e-14. Any real
# coefficient disagreement is O(1e-1). Measured: 1.2e-15 over 200 random directions.
FIELD_CROSS_REL_TOL = 1e-14


def test_field_closed_form_equator_and_poles() -> None:
    rel = np.array([[2000.0, 0.0, 0.0], [0.0, 2000.0, 0.0], [0.0, 0.0, 2000.0], [0.0, 0.0, -2000.0]])
    expected = np.array([[-9.375e-5, 0, 0], [0, -9.375e-5, 0], [0, 0, 1.875e-4], [0, 0, -1.875e-4]])
    got = 1e6 * reference.j2_field(rel, 1e-3, 1000.0)
    err = np.linalg.norm(got - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert np.all(err < FIELD_CLOSED_FORM_REL_TOL), err


def test_field_agrees_with_the_engine_kernel_written_independently() -> None:
    rng = np.random.default_rng(20260916)
    rel = rng.normal(size=(200, 3))
    rel *= (ORBIT_RADIUS / np.linalg.norm(rel, axis=1))[:, np.newaxis]

    truth = MU * reference.j2_field(rel, J2, R_EQ)

    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[1:, :3] = rel
    mu = np.zeros(n + 1)
    mu[0] = MU
    params = np.zeros((n + 1, 2))
    params[1:] = [geopotential.EARTH_J2, geopotential.EARTH_R_EQ]
    out = np.zeros((n + 1, 3))
    geopotential.j2_kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, mu,
                           np.zeros(n + 1, dtype=np.int32), params, out)

    err = np.linalg.norm(truth - out[1:], axis=1) / np.linalg.norm(out[1:], axis=1)
    assert np.max(err) < FIELD_CROSS_REL_TOL, np.max(err)


def test_restated_constants_match_the_engine() -> None:
    """The literals here and the engine's constants must describe the same Earth, or check 3 below is a
    comparison rather than a verification."""
    assert (J2, R_EQ) == (geopotential.EARTH_J2, geopotential.EARTH_R_EQ)


# ==================================================================================================
# API contract: explicit, independent, bit-identical when unused
# ==================================================================================================

def test_reference_module_imports_no_engine_physics() -> None:
    """Independence is structural: the only intra-package import is `Simulation`, for typing."""
    tree = ast.parse(inspect.getsource(reference))
    relative = [node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.level > 0]
    assert relative == ["simulator"], relative


def test_oblateness_is_not_read_from_the_engine_configuration(
    db_session_factory: Callable[[], Session],
) -> None:
    """Enabling `j2` in the engine must not change the truth; only the explicit argument may. And no
    argument, an empty mapping, or zero J2 must all give the historical point-mass result bit for bit."""
    times = np.arange(0.0, 1200.0 + 1.0, 60.0)
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=4, n_planes=2)
    baseline = reference.reference_for(sim, times)

    sim.enable_force_model("j2", bodies=_sat_indices(sim), j2=J2, r_eq=R_EQ)
    variants: List[Optional[Dict[str, Tuple[float, float]]]] = [None, {}, {"Earth": (0.0, R_EQ)}]
    for oblateness in variants:
        again = reference.reference_for(sim, times, oblateness=oblateness)
        assert np.array_equal(again.positions, baseline.positions), oblateness
        assert np.array_equal(again.velocities, baseline.velocities), oblateness

    oblate = reference.reference_for(sim, times, oblateness=EARTH_OBLATENESS)
    assert not np.array_equal(oblate.positions, baseline.positions)


def test_unknown_oblate_body_is_an_error(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=2, n_planes=1)
    with pytest.raises(KeyError):
        reference.reference_for(sim, np.array([0.0, 60.0]), oblateness={"Earht": (J2, R_EQ)})
    with pytest.raises(KeyError):   # a barycentre is not an integrated body
        reference.reference_for(sim, np.array([0.0, 60.0]), oblateness={"Earth Barycenter": (J2, R_EQ)})
    with pytest.raises(ValueError):
        reference.reference_for(sim, np.array([0.0, 60.0]), oblateness={"Earth": (J2, 0.0)})


# ==================================================================================================
# 1. Invariants of the truth
# ==================================================================================================

# Momentum is a *linear* invariant, and every Runge-Kutta method preserves linear invariants exactly;
# only rounding remains. Per step, sum mu_i v_i picks up ~ a few eps * sum mu_i |v_i|; over the ~1e3
# steps of this run that is ~1e3 * 4 eps = 1e-12 relative to sum mu_i |v_i|. Budget 1e-11. Without the
# reaction term on the oblate body, momentum would change by ~ mu_sat * |a_J2| * t ~ 1e-3 relative.
MOMENTUM_REL_TOL = 1e-11

# Energy including the J2 pair potential, and total h_z: not linear, so DOP853 conserves them only to
# its truncation error. The suite already holds point-mass DOP853 energy drift to 1e-11
# (test_reference_agreement.REFERENCE_ENERGY_DRIFT_TOL). J2 changes the right-hand side and its
# derivatives by ~J2 (R/r)^2 ~ 1e-3 relative, so step sizes and local errors change by ~1e-3 too: the
# same budget applies. A field that is not the gradient of the stated potential breaks energy by the
# size of the J2 potential itself, ~1e-3 relative.
ENERGY_REL_TOL = 1e-11
H_Z_REL_TOL = 1e-11


def test_massive_oblate_body_conserves_momentum_energy_and_h_z(
    db_session_factory: Callable[[], Session],
) -> None:
    """Two massive bodies (secondary mu = 0.3 mu_Earth), an inclined orbit so J2 does work, the primary
    oblate. Checks the reaction term and the J2 pair energy together.

    Measured: momentum 2.0e-15, energy_drift 7.1e-13 (7.0e-13 for the same case without J2), h_z
    4.7e-13. sum mu_i a_i is exactly 0.0: the two terms are the same product in a different order."""
    sim = scenarios.two_body(db_session_factory(), mu_secondary=0.3 * MU, p=11000.0, e=0.2, i=0.9, raan=0.4)
    times = np.arange(0.0, 2.0 * 86400.0 + 1.0, 600.0)
    ref = reference.reference_for(sim, times, oblateness={"Primary": (J2, R_EQ)}, **TRUTH)

    mu = ref.mu
    momentum = np.einsum("i,tij->tj", mu, ref.velocities)
    scale = float(np.sum(mu * np.linalg.norm(ref.velocities[0], axis=1)))
    assert np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / scale < MOMENTUM_REL_TOL

    # The reaction is real, not vacuously zero: sum mu_i a_i vanishes only because it is included.
    assert ref.j2 is not None and ref.r_eq is not None
    a = reference.oblateness_acceleration(ref.positions[0], mu, ref.j2, ref.r_eq)
    assert np.linalg.norm(a[0]) > 1e-9
    assert np.linalg.norm(mu @ a) < 1e-12 * float(np.sum(mu * np.linalg.norm(a, axis=1)))

    assert ref.energy_drift < ENERGY_REL_TOL, ref.energy_drift

    h_z = np.einsum("i,ti->t", mu, ref.positions[:, :, 0] * ref.velocities[:, :, 1]
                    - ref.positions[:, :, 1] * ref.velocities[:, :, 0])
    assert np.max(np.abs(h_z - h_z[0])) / abs(h_z[0]) < H_Z_REL_TOL


def _new_session() -> Tuple[Session, Callable[[], None]]:
    """Module-scoped fixtures cannot use the function-scoped `db_session_factory`; same construction."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    def close() -> None:
        session.close()
        engine.dispose()
    return session, close


# One day at 60 s sampling, one plane of 8 satellites. SAT-00-001 starts at argument of latitude
# u0 = 45 deg, where the first-order short-period terms in a and i (both ~ cos 2u) vanish, so its
# osculating initial a and i equal their mean values to first order in J2 - see check 2.
DAY = 86400.0
TRACKED = "SAT-00-001"


@pytest.fixture(scope="module")
def satellite_day() -> Iterator[Dict[str, ArrF]]:
    out: Dict[str, ArrF] = {}
    for label, oblateness in (("j2", EARTH_OBLATENESS), ("point_mass", None)):
        session, close = _new_session()
        try:
            sim = scenarios.earth_constellation(session, n_sats=8, n_planes=1)
            times = np.arange(0.0, DAY + 1.0, 60.0)
            ref = reference.reference_for(sim, times, oblateness=oblateness, **TRUTH)
            out[f"{label}_r"] = ref.position_of(TRACKED).copy()
            out[f"{label}_v"] = ref.velocity_of(TRACKED).copy()
            out[f"{label}_earth"] = ref.position_of("Earth").copy()
            out["t"] = times
        finally:
            close()
    yield out


def _test_potential(r: ArrF) -> ArrF:
    """Written here, in latitude form, from the potential rather than the field: -mu/r + (mu J2 R^2 /
    2 r^3)(3 sin^2 phi - 1)."""
    rn = np.linalg.norm(r, axis=1)
    sin_lat = np.sin(np.arcsin(r[:, 2] / rn))
    return np.asarray(-MU / rn + MU * J2 * R_EQ ** 2 / (2.0 * rn ** 3) * (3.0 * sin_lat ** 2 - 1.0))


# The Keplerian energy v^2/2 - mu/r of a J2 orbit is *not* conserved: it trades against Phi_J2, whose
# range along a circular orbit is (mu J2 R^2 / 2 r^3) * 3 sin^2 i (sin^2 u from 0 to 1)
# = 0.02648 * 3 * 0.6378 = 0.05066 km^2/s^2, i.e. 1.759e-3 of |E| = mu/2r = 28.797. The orbit radius
# itself varies by O(J2) (~6 km in 6921), which moves that by O(1e-3) relative, plus the extremes of
# sin^2 u are sampled at 60 s (u moves 3.8 deg per sample, cos(3.8 deg)^2 misses the extreme by
# < 0.5%). 5% covers both. This proves the energy check below is sensitive: dropping the J2 term
# from the energy would fail it by eight orders of magnitude.
KEPLER_ENERGY_SWING_REL = 1.759e-3
KEPLER_ENERGY_SWING_TOL = 0.05


def test_satellite_energy_with_j2_potential_and_h_z_are_conserved(satellite_day: Dict[str, ArrF]) -> None:
    """Measured at TRUTH tolerances: dE/E 1.51e-13 with J2 against 1.35e-13 point-mass (at the
    defaults: 4.60e-13 against 4.59e-13, the ratio ~1 predicted above); dh_z/h_z 7.5e-14; Kepler-energy
    swing 1.7598e-3 against the derived 1.759e-3. Earth stays exactly at the origin (massless satellites)."""
    r, v = satellite_day["j2_r"], satellite_day["j2_v"]
    energy = 0.5 * np.einsum("ij,ij->i", v, v) + _test_potential(r)
    e_drift = float(np.max(np.abs(energy - energy[0])) / abs(energy[0]))

    rp, vp = satellite_day["point_mass_r"], satellite_day["point_mass_v"]
    energy_pm = 0.5 * np.einsum("ij,ij->i", vp, vp) - MU / np.linalg.norm(rp, axis=1)
    e_drift_pm = float(np.max(np.abs(energy_pm - energy_pm[0])) / abs(energy_pm[0]))

    h_z = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    h_drift = float(np.max(np.abs(h_z - h_z[0])) / abs(h_z[0]))

    assert e_drift < ENERGY_REL_TOL, (e_drift, e_drift_pm)
    assert e_drift_pm < ENERGY_REL_TOL, e_drift_pm
    assert h_drift < H_Z_REL_TOL, h_drift
    assert np.all(satellite_day["j2_earth"] == 0.0)

    kepler = 0.5 * np.einsum("ij,ij->i", v, v) - MU / np.linalg.norm(r, axis=1)
    swing = float(np.ptp(kepler)) / abs(float(kepler[0]))
    assert abs(swing / KEPLER_ENERGY_SWING_REL - 1.0) < KEPLER_ENERGY_SWING_TOL, swing


# ==================================================================================================
# 2. Secular nodal regression
# ==================================================================================================

# Gauss' equation for Omega with the J2 normal component a_W = -(3 mu J2 R^2 / r^4) sin i cos i sin u,
# on a circular orbit, gives
#     dOmega/dt = -3 n J2 (R/r)^2 cos i sin^2 u = -(3/2) n J2 (R/r)^2 cos i (1 - cos 2u),
# so the secular rate is the textbook -(3/2) n J2 (R/p)^2 cos i = -9.101e-7 rad/s (-4.51 deg/day), and
# the short-period term integrates to (3/4) J2 (R/r)^2 cos i sin 2u:
NODAL_RATE = -1.5 * MEAN_MOTION * J2_R2 * math.cos(INCLINATION)
SHORT_PERIOD_AMPLITUDE = 0.75 * J2_R2 * math.cos(INCLINATION)          # 4.15e-4 rad

# The rate is a least-squares slope of Omega(t) over T = 1 day. A sinusoid of amplitude A and
# frequency w = 2n biases an LSQ slope by at most 12 A / (w T^2) = 3.3e-10 rad/s, 3.6e-4 relative.
# The theory is first order in J2 and is stated for *mean* elements; the satellite starts at u = 45 deg
# where the first-order osculating-minus-mean offsets in a and i (both proportional to cos 2u) are
# zero, so what remains is second order: J2 (R/p)^2 * O(1) relative. Budgeting O(1) as 3:
# 3.6e-4 + 3 * 9.19e-4 = 3.1e-3. The negative control's doubled spin-axis term doubles a_W and hence
# the rate: relative error 1.0.
NODAL_LSQ_BIAS_REL = 12.0 * SHORT_PERIOD_AMPLITUDE / (2.0 * MEAN_MOTION * DAY ** 2 * abs(NODAL_RATE))
NODAL_RATE_REL_TOL = NODAL_LSQ_BIAS_REL + 3.0 * J2_R2

# The residual about the fitted line peaks at A, shifted by the slope bias over half the window:
# 6 A / (w T) = 3.2% of A. Second-order terms add O(J2) ~ 0.1%. Budget 5%.
SHORT_PERIOD_REL_TOL = 0.05


def test_mean_node_regresses_at_the_first_order_j2_rate(satellite_day: Dict[str, ArrF]) -> None:
    """Measured: rate 5.2e-4 relative to theory (budget 3.1e-3); residual peak 4.23e-4 rad against the
    derived 4.15e-4 (+1.9%, inside the 3.2% slope-tilt allowance). The point-mass run's node moves by
    3.6e-15 rad, which rules out the frame."""
    t = satellite_day["t"]

    def node(r: ArrF, v: ArrF) -> ArrF:
        h = np.cross(r, v)
        return np.asarray(np.unwrap(np.arctan2(h[:, 0], -h[:, 1])))

    raan = node(satellite_day["j2_r"], satellite_day["j2_v"])
    slope, intercept = np.polyfit(t, raan, 1)
    rel = abs(slope / NODAL_RATE - 1.0)
    assert rel < NODAL_RATE_REL_TOL, (slope, NODAL_RATE, rel, NODAL_RATE_REL_TOL)

    residual_peak = float(np.max(np.abs(raan - (slope * t + intercept))))
    assert abs(residual_peak / SHORT_PERIOD_AMPLITUDE - 1.0) < SHORT_PERIOD_REL_TOL, residual_peak

    raan_pm = node(satellite_day["point_mass_r"], satellite_day["point_mass_v"])
    assert float(np.ptp(raan_pm)) < 1e-9


# ==================================================================================================
# 3. Verification: engine Cowell + point_mass_gravity + j2 against the truth
# ==================================================================================================

# One orbit of a 6-satellite, 2-plane constellation; the error is the worst satellite's position.
CONVERGENCE_STEP_COUNTS = [256, 512, 1024, 2048]
# Same band as test_cowell_propagator.py: centred on 16, rejects 2nd order (4x) and a model error (1x).
ORDER_RATIO_LOW = 12.0
ORDER_RATIO_HIGH = 24.0

# The floor at dt -> 0 is the sum of
#  (a) the truth's own error. A DOP853 run holding local error to rtol |y| accumulates of order
#      rtol * r per orbit times a small factor, 6.9e-10 * O(1..10) km. test_truth_error_... measures
#      it directly (against rtol 2.5e-14, atol 1e-16) and asserts the O(10) end, 7e-9 km;
#  (b) RK4 rounding in the engine, at most ~ eps * r per step accumulating linearly:
#      8192 * 2.2e-16 * 6921 = 1.3e-8 km at the finest step count.
# Floor bound 7e-9 + 1.3e-8 = 2.0e-8 km. RK4 truncation at 8192 steps extrapolates from 2048 as
# 16^-2 * 9e-8 = 3.6e-10 km, below both, so 8192 steps is at the floor.
FLOOR_STEP_COUNT = 8192
TRUTH_ERROR_BOUND_KM = 7e-9
ENGINE_FLOOR_BOUND_KM = TRUTH_ERROR_BOUND_KM + FLOOR_STEP_COUNT * EPS * ORBIT_RADIUS


def _build_cowell_j2(session: Session) -> Tuple[Simulation, NDArray[np.int64]]:
    sim = scenarios.earth_constellation(session, n_sats=6, n_planes=2)
    sim.record_history = False
    sats = _sat_indices(sim)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", bodies=sats)
    sim.enable_force_model(geopotential.J2_MODEL, bodies=sats, j2=J2, r_eq=R_EQ)
    return sim, sats


def _truth_final_positions(
    db_session_factory: Callable[[], Session], total_time: float, **tolerances: float,
) -> Dict[str, ArrF]:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    ref = reference.reference_for(sim, np.array([0.0, total_time]), oblateness=EARTH_OBLATENESS,
                                  **tolerances)
    return {n: ref.positions[-1, ref.index_of(n)].copy() for n in ref.names if n.startswith("SAT-")}


def _engine_errors(
    db_session_factory: Callable[[], Session], truth: Dict[str, ArrF], step_counts: List[int],
    total_time: float,
) -> List[float]:
    errors = []
    for n_steps in step_counts:
        sim, _ = _build_cowell_j2(db_session_factory())
        dt = total_time / n_steps
        for _ in range(n_steps):
            sim.step(dt)
        errors.append(max(float(np.linalg.norm(sim.global_states[sim.name_to_index[n], :3] - r))
                          for n, r in truth.items()))
    return errors


def test_truth_error_is_below_its_derived_bound(db_session_factory: Callable[[], Session]) -> None:
    """Measured at the end of one orbit: 8.6e-10 km at TRUTH tolerances, 3.5e-9 km at the defaults.
    Over 30 s samples during the orbit (dense output included): 1.0e-9 and 3.9e-9 km, then 1.5e-8 at
    rtol 1e-12 and 1.6e-7 at 1e-11, which is proportional to rtol, as expected once atol stops limiting."""
    tight = _truth_final_positions(db_session_factory, PERIOD, rtol=2.5e-14, atol=1e-16)
    truth = _truth_final_positions(db_session_factory, PERIOD, **TRUTH)
    err = max(float(np.linalg.norm(truth[n] - tight[n])) for n in truth)
    assert err < TRUTH_ERROR_BOUND_KM, err


def test_cowell_j2_converges_to_the_truth_at_fourth_order_down_to_the_floor(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Identical physics on both sides (point mass + J2 about a fixed Earth, massless satellites), so this
    is verification: the error is RK4 truncation, then the floor.

    Measured: 256..2048 steps -> 4.17e-4, 2.44e-5, 1.48e-6, 9.20e-8 km (ratios 17.0, 16.5, 16.1);
    4096 -> 7.06e-9 (ratio 13.0, leaving the asymptote); 8192 -> 1.62e-9 km (ratio 4.4, at the floor),
    against the derived floor bound 2.0e-8 km. The floor matches the truth's own error (8.6e-10 km)
    plus rounding well below the linear worst case.
    """
    truth = _truth_final_positions(db_session_factory, PERIOD, **TRUTH)
    errors = _engine_errors(db_session_factory, truth, CONVERGENCE_STEP_COUNTS + [FLOOR_STEP_COUNT], PERIOD)
    order_errors, floor_error = errors[:-1], errors[-1]

    ratios = [order_errors[k] / order_errors[k + 1] for k in range(len(order_errors) - 1)]
    assert all(ORDER_RATIO_LOW < q < ORDER_RATIO_HIGH for q in ratios), (ratios, order_errors)
    assert floor_error < ENGINE_FLOOR_BOUND_KM, (floor_error, ENGINE_FLOOR_BOUND_KM)


# ==================================================================================================
# 4. Negative control
# ==================================================================================================

# The mutation doubles the spin-axis term of (*) in reference.py - identical to Curtis' z-component
# (5 s^2 - 3) -> (5 s^2 - 5) - adding a_z = -3 (mu J2 R^2 / r^4) s. Two effects over one orbit:
#  - cross-track: its W component equals the true one, so the node regresses twice as fast. Omega is off
#    by |NODAL_RATE| T = 5.2e-3 rad, displacing the satellite by up to r sin i * 5.2e-3 = 29 km.
#  - along-track: its radial projection -3 (mu J2 R^2/r^4) s^2 averages, with <s^2> = sin^2 i / 2 =
#    0.319, to f = -0.957 * 7.65e-6 = -7.3e-6 km/s^2. A constant radial acceleration f on a circular
#    orbit drifts along-track at 3 f / n per unit time (Clohessy-Wiltshire), i.e. 6 pi f / n^2 = 114 km
#    per orbit.
# Combined sqrt(114^2 + 29^2) = 118 km. This estimate was completed only after measuring: the first
# version counted the node shift alone (29 km) and the real-file mutant gave 136 km, which is what
# exposed the omitted radial term. 136 km sits 15% above the completed estimate, which ignores the
# short-period terms. The error is model error, so it does not shrink with dt: ratios ~1.
# Asserted: > 1 km (4e4 x the correct error at 512 steps, 2.4e-5 km), inside [0.5, 2] x 118 km so the
# in-suite mutant is shown to be the intended mutation, and ratio outside the fourth-order band.
MUTANT_MIN_ERROR_KM = 1.0
MUTANT_PREDICTED_KM = math.hypot(
    6.0 * math.pi * 3.0 * (MU * J2 * R_EQ ** 2 / ORBIT_RADIUS ** 4) * 0.5 * math.sin(INCLINATION) ** 2
    / MEAN_MOTION ** 2,
    ORBIT_RADIUS * math.sin(INCLINATION) * abs(NODAL_RATE) * PERIOD,
)
MUTANT_CHECK_STEP_COUNTS = [256, 512]


def test_negative_control_doubled_spin_axis_term_fails_verification(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    In-suite version: wraps the real `j2_field` and adds the mutant's extra term. The same mutation made
    as a one-token edit of `reference.py` itself (`field[:, 2] -= k * s` -> `-= 2.0 * k * s`) was run
    against this file when it was written. That run failed 6 of the 12 tests in this file: the closed
    form (pole), the engine cross-check, both energy checks (satellite dE/E 8.9e-4, massive-case drift
    1.4e-3), the nodal rate (relative error 1.007) and this verification (136.2 km at 256, 512, 1024
    and 2048 steps, ratios 1.000). The wrapper below reproduces the verification failure without
    editing the module: 136.2 km at 256 and 512 steps, agreeing with the real-file mutant to 1e-11 km.
    """
    original = reference.j2_field

    def mutant(rel: ArrF, j2: float, r_eq: float) -> ArrF:
        field: ArrF = original(rel, j2, r_eq).copy()
        r = np.linalg.norm(rel, axis=1)
        field[:, 2] -= 3.0 * j2 * r_eq * r_eq * rel[:, 2] / r ** 5
        return field

    monkeypatch.setattr(reference, "j2_field", mutant)
    truth = _truth_final_positions(db_session_factory, PERIOD, **TRUTH)
    monkeypatch.undo()

    errors = _engine_errors(db_session_factory, truth, MUTANT_CHECK_STEP_COUNTS, PERIOD)
    assert all(e > MUTANT_MIN_ERROR_KM for e in errors), errors
    assert all(0.5 < e / MUTANT_PREDICTED_KM < 2.0 for e in errors), (errors, MUTANT_PREDICTED_KM)
    assert not (ORDER_RATIO_LOW < errors[0] / errors[1] < ORDER_RATIO_HIGH), errors
