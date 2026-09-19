"""
Validation for `orbital_engine.viz`, the plot-data layer.

Two halves, deliberately separated:

**The transforms, against analytic geometry.** `ground_track` and `altitude_series` are pure
functions of a position array, so they are fed a closed-form circular orbit written here -
`sin(lat) = sin(i) sin(u)` for the inclined case - and never a propagated one. That makes the
assertions about the *transform* rather than about the propagator: an orbit at inclination `i` must
reach latitude `+/-i` and no further, an equatorial orbit must stay at latitude 0, and a ground
track must drift **west** by `omega_earth * T` per orbit. That last one is the assertion that earns
its keep: an inertial-to-body-fixed rotation with the sign reversed produces a perfectly smooth
ground track that drifts east instead, which no shape-based check would catch. See `viz.py`'s module
docstring on why the sign is easy to get wrong.

**The curve, against `sweep.run_sweep`.** `error_curve`'s final value must equal the number
`sweep.run_sweep` reports for the same `ModelConfig` - that is what ties the new figures to the
numbers already in `README.md`, and it holds each tier's sub-stepping (`max_dt`) equal to the
sweep's `dt`. The tolerance is `AGREEMENT_RTOL` below, with the reasoning for its size there.

No matplotlib. `viz.py` does not import it and neither does this file; the figure scripts in
`benchmarks/` are the only consumers that do.
"""
from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import geopotential, scenarios, viz
from orbital_engine.custom_types import PropagatorType
from orbital_engine.drag import EARTH_OMEGA
from orbital_engine.geopotential import EARTH_J2, EARTH_R_EQ
from orbital_engine.reference import TRUTH_ATOL, TRUTH_RTOL, reference_for
from orbital_engine.simulator import Simulation
from orbital_engine.sweep import ForceModelSpec, ModelConfig, apply_config, run_sweep

ArrF = NDArray[np.float64]

MU = scenarios.MU_EARTH
R_CIRC = 7000.0                                      # km, the analytic orbit's radius
PERIOD = 2.0 * math.pi * math.sqrt(R_CIRC ** 3 / MU)  # 5828.5 s
N_SAMPLES = 501                                      # over exactly one period


def _circular_orbit(inclination_deg: float, n_samples: int = N_SAMPLES) -> tuple[ArrF, ArrF]:
    """
    A closed-form circular orbit of radius `R_CIRC` with its ascending node on +x, sampled over one
    period. Returns `(positions (n, 1, 3), times (n,))`.

    Written out rather than taken from `coe_to_rv`, so a bug shared between the two could not cancel:
    the position is `R (cos u, sin u cos i, sin u sin i)`, whose latitude satisfies
    `sin(lat) = sin(i) sin(u)` by inspection.
    """
    times = np.linspace(0.0, PERIOD, n_samples)
    u = 2.0 * math.pi * times / PERIOD
    i = math.radians(inclination_deg)
    pos = R_CIRC * np.stack([np.cos(u), np.sin(u) * math.cos(i), np.sin(u) * math.sin(i)], axis=1)
    return pos[:, np.newaxis, :], times


def _max_latitude_tolerance_deg(inclination_deg: float, n_samples: int = N_SAMPLES) -> float:
    """
    How far below `i` the *sampled* maximum latitude may fall, purely from the sample spacing.

    Near the northernmost point `lat(u)` has zero first derivative and `lat'' = -tan(i)`, so
    `lat_max - lat(u) ~ (1/2) tan(i) du^2`, and the worst-case offset of the nearest sample from the
    extremum is half a sample, `du = pi / (n - 1)`. The polar case is exact instead of quadratic -
    `sin(lat) = sin(u)` gives `lat = u`, so the shortfall is just `du` itself - and `tan(90 deg)` is
    infinite, so it is handled separately.
    """
    du = math.pi / (n_samples - 1)
    if abs(inclination_deg - 90.0) < 1e-9:
        return math.degrees(du)
    return math.degrees(0.5 * math.tan(math.radians(inclination_deg)) * du ** 2)


# ==================================================================================================
# Ground-track transform, against analytic geometry
# ==================================================================================================

def test_equatorial_orbit_stays_on_the_equator() -> None:
    """An orbit in the xy-plane has zero z, so every sub-satellite latitude is exactly zero. The
    body-fixed rotation is about +z and cannot move it off the equator."""
    pos, times = _circular_orbit(0.0)
    track = viz.ground_track(pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)
    assert np.max(np.abs(track.latitude_deg)) < 1e-12


@pytest.mark.parametrize("inclination_deg", [0.0, 28.5, 53.0, 90.0])
def test_latitude_is_bounded_by_the_inclination(inclination_deg: float) -> None:
    """Maximum latitude equals the inclination and minimum equals its negative, to the sampling
    tolerance derived in `_max_latitude_tolerance_deg`. The *upper* bound is exact in exact
    arithmetic - `|sin(lat)| = |sin i sin u| <= sin i` - so it is asserted at 1e-9, floating-point
    slack only, while the shortfall below `i` is the sampled quantity."""
    pos, times = _circular_orbit(inclination_deg)
    track = viz.ground_track(pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)

    tol = _max_latitude_tolerance_deg(inclination_deg)
    assert np.max(track.latitude_deg) <= inclination_deg + 1e-9
    assert np.min(track.latitude_deg) >= -inclination_deg - 1e-9
    assert np.max(track.latitude_deg) >= inclination_deg - tol - 1e-12
    assert np.min(track.latitude_deg) <= -inclination_deg + tol + 1e-12


def test_ground_track_drifts_west_by_the_earth_rotation_per_orbit() -> None:
    """
    The negative control's target.

    Over one orbital period the satellite advances a full `2 pi` in inertial longitude while the
    body turns `omega * T` beneath it, so the *unwrapped* body-fixed longitude advances
    `2 pi - omega * T`: a net westward shift of `omega * T = 24.352 deg` at this radius. Asserting
    the signed residual, not its magnitude, is what makes a reversed rotation sign fail - drop the
    rotation entirely and the residual is 0 instead of -24.352 deg.
    """
    pos, times = _circular_orbit(0.0)
    track = viz.ground_track(pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)

    unwrapped = np.unwrap(np.radians(track.longitude_deg[:, 0]))
    drift = (unwrapped[-1] - unwrapped[0]) - 2.0 * math.pi
    assert drift == pytest.approx(-EARTH_OMEGA * PERIOD, abs=1e-9)
    assert math.degrees(drift) == pytest.approx(-24.352, abs=1e-3)


def test_theta0_offsets_the_whole_track_rigidly() -> None:
    """The prime-meridian angle at the first sample is a rigid longitude offset: it shifts every
    longitude by `-theta0` and touches neither latitude nor altitude."""
    pos, times = _circular_orbit(53.0)
    base = viz.ground_track(pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)
    shifted = viz.ground_track(
        pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ, theta0=math.radians(30.0)
    )
    delta = np.radians(shifted.longitude_deg - base.longitude_deg)
    wrapped = (delta + math.pi) % (2.0 * math.pi) - math.pi
    assert np.allclose(wrapped, math.radians(-30.0), atol=1e-12)
    assert np.allclose(shifted.latitude_deg, base.latitude_deg, atol=1e-12)
    assert np.allclose(shifted.altitude_km, base.altitude_km, atol=1e-12)


def test_altitude_is_the_radius_above_the_spherical_body() -> None:
    """Both `ground_track` and `altitude_series` report `|r| - body_radius_km`, and they agree."""
    pos, times = _circular_orbit(53.0)
    track = viz.ground_track(pos, times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)
    assert np.allclose(track.altitude_km, R_CIRC - EARTH_R_EQ, atol=1e-9)
    assert np.allclose(
        viz.altitude_series(pos, body_radius_km=EARTH_R_EQ), track.altitude_km, atol=0.0
    )


def test_ground_track_rejects_a_mismatched_time_grid() -> None:
    pos, times = _circular_orbit(0.0)
    with pytest.raises(ValueError, match="incompatible"):
        viz.ground_track(pos, times[:-1], omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ)


# ==================================================================================================
# Sampling
# ==================================================================================================

def _constellation(session: Session, n_sats: int = 3) -> Simulation:
    sim = scenarios.earth_constellation(
        session, n_sats=n_sats, n_planes=1, altitude_km=550.0, inclination_deg=53.0
    )
    sim.record_history = False
    return sim


def _sat_slots(sim: Simulation) -> List[int]:
    return sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))


def test_sample_states_substeps_exactly_like_a_step_loop(
    db_session_factory: Callable[[], Session]
) -> None:
    """
    `max_dt` is the propagation step size, not a sampling convenience: sampling every 300 s with
    `max_dt=60.0` must reproduce, state for state, a plain loop of 60 s steps. If it did not, an
    error curve's final point would not be the same run `sweep.run_sweep` times and scores.
    """
    horizon, dt, stride = 1200.0, 60.0, 300.0

    sampled_sim = _constellation(db_session_factory())
    slots = _sat_slots(sampled_sim)
    for slot in slots:
        sampled_sim.set_propagator(slot, PropagatorType.COWELL)
        sampled_sim.enable_force_model("point_mass_gravity", bodies=slot)
    times = np.arange(0.0, horizon + 0.5 * stride, stride)
    sampled = viz.sample_states(sampled_sim, slots, times, max_dt=dt)

    looped_sim = _constellation(db_session_factory())
    for slot in _sat_slots(looped_sim):
        looped_sim.set_propagator(slot, PropagatorType.COWELL)
        looped_sim.enable_force_model("point_mass_gravity", bodies=slot)
    for _ in range(int(horizon / dt)):
        looped_sim.step(dt)

    assert np.array_equal(sampled[-1], looped_sim.global_states[slots])
    assert sampled.shape == (times.size, len(slots), 6)


def test_sample_states_relative_to_differences_against_that_slot(
    db_session_factory: Callable[[], Session]
) -> None:
    """`relative_to` subtracts one slot's global row at the same instant - for
    `earth_constellation` the root is the Earth *barycentre*, so an Earth-centred track needs it."""
    sim = _constellation(db_session_factory())
    slots = _sat_slots(sim)
    earth = sim.name_to_index["Earth"]
    times = np.array([0.0, 600.0])

    relative = viz.sample_states(sim, slots, times, relative_to=earth)
    assert np.allclose(
        relative[-1], sim.global_states[slots] - sim.global_states[earth], atol=0.0
    )


def test_propagated_ground_track_regresses_at_the_earth_rotation_rate(
    db_session_factory: Callable[[], Session]
) -> None:
    """
    End to end through the engine, and the second half of the negative control's target.

    A Keplerian satellite's orbit plane is inertially fixed, so its ascending node crosses the
    equator at a longitude exactly `omega_earth * T` further west each revolution - no J2 nodal
    regression, no draconitic-period correction, nothing else in the number. Measured by linear
    interpolation between the two samples bracketing each northbound crossing, which at a 10 s step
    is good to well under the 1e-3 deg asserted here; the engine gives -23.9409 deg against the
    predicted -23.9409 deg. `benchmarks/figures.py` annotates the same quantity on the ground-track
    figure, where J2 is on and the measured shift is 0.25 deg larger.
    """
    sim = _constellation(db_session_factory(), n_sats=1)
    slots = _sat_slots(sim)
    earth = sim.name_to_index["Earth"]
    a_km = scenarios.EARTH_RADIUS + 550.0
    period = 2.0 * math.pi * math.sqrt(a_km ** 3 / MU)

    times = np.arange(0.0, 3.0 * period, 10.0)
    states = viz.sample_states(sim, slots, times, relative_to=earth, max_dt=10.0)
    track = viz.ground_track(
        states[..., :3], times, omega=EARTH_OMEGA, body_radius_km=EARTH_R_EQ
    )

    lat, lon = track.latitude_deg[:, 0], track.longitude_deg[:, 0]
    rising = np.flatnonzero((lat[:-1] < 0.0) & (lat[1:] >= 0.0))
    nodes = []
    for k in rising:
        if abs(lon[k + 1] - lon[k]) > 180.0:
            continue
        frac = -lat[k] / (lat[k + 1] - lat[k])
        nodes.append(lon[k] + frac * (lon[k + 1] - lon[k]))

    assert len(nodes) >= 2
    expected = -math.degrees(EARTH_OMEGA * period)
    assert nodes[1] - nodes[0] == pytest.approx(expected, abs=1e-3)
    assert np.max(np.abs(lat)) <= 53.0 + 1e-6


def test_sample_states_refuses_to_run_backwards(db_session_factory: Callable[[], Session]) -> None:
    sim = _constellation(db_session_factory())
    sim.step(100.0)
    with pytest.raises(ValueError, match="only moves forward"):
        viz.sample_states(sim, _sat_slots(sim), np.array([0.0, 50.0]))


# ==================================================================================================
# Error curve, tied back to `sweep.run_sweep`
# ==================================================================================================

HORIZON_S = 3600.0
CURVE_SAMPLES = 7                       # 0 .. 3600 s in 600 s steps
COWELL_DT = 60.0
N_SATS = 3
_J2_COEFFS = {"j2": EARTH_J2, "r_eq": EARTH_R_EQ}

# `error_curve`'s last point and `run_sweep`'s reported median come from the same engine run and the
# same DOP853 solve, differing only in that the curve asks `solve_ivp` for seven output samples and
# the sweep for two. `t_eval` is dense-output interpolation and does not change the steps the solver
# takes, so the two agree to round-off; 1e-9 relative is round-off with three orders of headroom, and
# is far tighter than any tier's spread (the tiers here differ from one another by 10x to 1000x).
AGREEMENT_RTOL = 1e-9

_CONFIGS = [
    ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=HORIZON_S),
    ModelConfig(name="secular-j2 (osculating-seeded)", propagator=PropagatorType.SECULAR_J2,
                dt=HORIZON_S, propagator_coefficients=_J2_COEFFS),
    ModelConfig(name="secular-j2 (mean-seeded)", propagator=PropagatorType.SECULAR_J2,
                dt=HORIZON_S, mean_seed=True, propagator_coefficients=_J2_COEFFS),
    ModelConfig(name="cowell", propagator=PropagatorType.COWELL, dt=COWELL_DT,
                force_models=(ForceModelSpec("point_mass_gravity"),
                              ForceModelSpec("j2", _J2_COEFFS))),
]


@pytest.mark.parametrize("config", _CONFIGS, ids=[c.name for c in _CONFIGS])
def test_error_curve_endpoint_matches_run_sweep(
    config: ModelConfig, db_session_factory: Callable[[], Session]
) -> None:
    """
    The last point of a tier's error curve is the number the sweep reports for the same tier.

    The analytic tiers are closed-form in time, so sampling them every 600 s costs nothing and
    changes nothing (`benchmarks/frontier_plot.py`'s docstring records the measurement); Cowell is
    sub-stepped at `max_dt = config.dt`, which is the sweep's own step size. Both sides use the same
    truth tolerances, `TRUTH_RTOL` / `TRUTH_ATOL`, and the same J2 oblateness.
    """
    oblateness = {"Earth": (EARTH_J2, EARTH_R_EQ)}

    def build() -> Simulation:
        sim = _constellation(db_session_factory(), n_sats=N_SATS)
        return sim

    curve_sim = build()
    idx = apply_config(curve_sim, config)
    slot_to_name = {slot: name for name, slot in curve_sim.name_to_index.items()}
    names = [slot_to_name[int(s)] for s in idx]

    times = np.linspace(0.0, HORIZON_S, CURVE_SAMPLES)
    truth = reference_for(
        build(), times, rtol=TRUTH_RTOL, atol=TRUTH_ATOL, oblateness=oblateness
    )
    max_dt = COWELL_DT if config.propagator == PropagatorType.COWELL else None
    sampled = viz.sample_states(curve_sim, idx.tolist(), times, max_dt=max_dt)
    curve = viz.error_curve(sampled[..., :3], truth, names)

    swept = run_sweep(
        build, [config], HORIZON_S, oblateness=oblateness, timing_batches=1, timing_warmup=0
    )[0]

    assert curve.per_body_km.shape == (CURVE_SAMPLES, N_SATS)
    assert float(curve.median_km[-1]) == pytest.approx(
        swept.error.median_km, rel=AGREEMENT_RTOL
    )
    assert float(curve.max_km[-1]) == pytest.approx(swept.error.max_km, rel=AGREEMENT_RTOL)
    # The curve starts from the shared initial condition, so it opens at zero error, and error grows.
    assert float(curve.median_km[0]) < 1e-9
    assert float(curve.median_km[-1]) > float(curve.median_km[1])


def test_position_error_rejects_a_truth_on_a_different_grid(
    db_session_factory: Callable[[], Session]
) -> None:
    """A truth trajectory sampled on a different grid is refused rather than interpolated: a silent
    resample would produce a plausible-looking curve from a mismatched comparison."""
    sim = _constellation(db_session_factory())
    slots = _sat_slots(sim)
    names = [n for n in sim.name_to_index if n.startswith("SAT-")]
    truth = reference_for(_constellation(db_session_factory()), np.array([0.0, 600.0]))
    sampled = viz.sample_states(sim, slots, np.array([0.0, 300.0, 600.0]))
    with pytest.raises(ValueError, match="same time grid"):
        viz.position_error(sampled[..., :3], truth, sorted(names))
