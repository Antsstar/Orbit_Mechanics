"""
Validation for `orbital_engine.geometry`, the observation-geometry layer.

Three halves, deliberately separated - the same split `test_viz.py` uses, for the same reason.

**The conventions, against hand-computed vectors.** `station_position_fixed` and the SEZ triad are
pinned to sites and targets whose answers can be written down by inspection: a station at
`(0 deg, 0 deg)` sits on `+x`, its zenith is `+x`, its east is `+y` and its north is `+z`, so a
target displaced due north on the horizon must read azimuth 0 and elevation 0. The rotation-sign
anchor is `test_zenith_at_a_rotated_epoch`: with the body turned 90 deg, a station on the prime
meridian has moved to inertial `+y`, so a satellite at `(0, r, 0)` is at its zenith. Under the
flipped sign the station is at `-y` and the same satellite reads -90 deg - not a subtle shift, which
is the point of choosing a 90 deg epoch rather than a small one.

**The pass, against closed-form horizon geometry.** For a circular orbit of radius `r` over a
station in its plane on a body of radius `R`, everything about the pass is exact:

    lambda_0 = acos(R / r)                       horizon half-angle at the body's centre
    lambda_e = acos((R / r) cos e) - e           the same at mask elevation e
    rho_0    = sqrt(r^2 - R^2)                   slant range at the horizon
    Omega    = n - omega                         station-relative angular rate
    duration = 2 lambda_e / Omega                pass length above the mask

`Omega` is where the body's own rotation enters: dropping it under-states a 550 km pass by 6.7 %,
which is far larger than any tolerance here, and is one of the two negative controls recorded in
`docs/engineering-log.md`.

**The interpolation, against its derived error term.** `access_windows` interpolates linearly, so a
crossing carries a *bias* of `-(f'' / (2 f')) a b` with `a`, `b` the offsets of the bracketing
samples. `_edge_bias` writes that out from the closed-form elevation curve, and the tests compare
measured minus exact against it - not against a tolerance. The grids are phase-locked (the rise
always falls 30 % of the way into its interval) so that halving `h` scales `a b` by exactly 4 and
the convergence order is measured rather than asserted by eye.

**Range rate, four ways.** `rho_dot = r R sin(u) Omega / rho` is the closed form of the same
coplanar pass, so the analytic instants pin it to floating point (6.5213 km/s at the horizon, and
*exactly* zero at closest approach - asserted at the analytic instant, because the nearest grid
sample is off by `rho'' h / 2`, 1.26 km/s at `h = 30 s`). The independent check is a central
difference of the range series, which shares no algebra with the analytic rate and is compared
against `rho''' h^2 / 6` in the equatorial case and against its own halving ratio in the inclined
one. The transport term `omega x r_station` is asserted as a *number*: at the horizon of an
equatorial pass it is exactly `R omega = 0.4646 km/s` of 6.5213 km/s, 7.1 %. And the sign - positive
= opening - is asserted as two signs rather than an absolute value, because an inverted convention
passes every magnitude check here and inverts every downstream Doppler.

The inclined case exists because the equatorial one is degenerate: an equatorial station watching an
equatorial orbit has an identically zero SEZ *south* component, so it cannot discriminate a
transposed south/east axis. `test_inclined_pass_matches_the_general_triangle` checks the general
station-satellite-centre triangle instead, with the station's inertial position written out
independently of `geometry.py`.
"""
from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import geometry, scenarios, viz
from orbital_engine.drag import EARTH_OMEGA

ArrF = NDArray[np.float64]

# --------------------------------------------------------------------------------------------------
# The one geometry every closed form below refers to: `scenarios.ground_station_pass`'s default.
# --------------------------------------------------------------------------------------------------
R_BODY = scenarios.EARTH_RADIUS                      # 6371.0 km, the sphere the station sits on
ALT_KM = 550.0
R_ORBIT = R_BODY + ALT_KM                            # 6921.0 km
MU = scenarios.MU_EARTH
N_MEAN = math.sqrt(MU / R_ORBIT ** 3)                # 1.0965e-3 rad/s
OMEGA_REL = N_MEAN - EARTH_OMEGA                     # 1.0236e-3 rad/s, station-relative
C_RATIO = R_BODY / R_ORBIT                           # cos(lambda_0)
LAMBDA_0 = math.acos(C_RATIO)                        # 0.40136 rad = 23.0 deg
RANGE_HORIZON = math.sqrt(R_ORBIT ** 2 - R_BODY ** 2)  # 2703.81 km

# Range rate of the coplanar pass at the horizon, `r R sin(lambda_0) Omega / rho_0`. It is also the
# largest |range rate| of the pass: `rho'' = 0` there exactly, because `r cos lambda_0 = R` makes
# the two terms of `rho'' = (r R Omega^2 cos u - rho_dot^2) / rho` cancel.
RANGE_RATE_HORIZON = R_ORBIT * R_BODY * math.sin(LAMBDA_0) * OMEGA_REL / RANGE_HORIZON  # 6.5213
# Curvature of the range at closest approach, `r R Omega^2 / (r - R)`: 0.0840 km/s^2.
RHO_DDOT_CLOSEST = R_ORBIT * R_BODY * OMEGA_REL ** 2 / (R_ORBIT - R_BODY)

# The satellite is seeded at true anomaly 180 deg and the station sits on the prime meridian, so the
# two share a longitude - the satellite is exactly overhead - when Omega t = pi.
T_OVERHEAD = math.pi / OMEGA_REL                     # 3069.17 s
T_RISE = T_OVERHEAD - LAMBDA_0 / OMEGA_REL           # 2677.07 s
T_SET = T_OVERHEAD + LAMBDA_0 / OMEGA_REL            # 3461.28 s
DURATION_EXACT = T_SET - T_RISE                      # 784.21 s

STATION_LAT = math.radians(scenarios.STATION_LATITUDE_DEG)
STATION_LON = math.radians(scenarios.STATION_LONGITUDE_DEG)

# Machine-precision bounds. The Keplerian propagation of a circular orbit and every transform here
# are closed-form double-precision arithmetic, so "agrees analytically" means a few ulp of the
# quantities involved, not a fitted number. Elevations near the horizon lose digits to the
# cancellation in `r cos(lambda) - R`; 1e-11 rad is ~5 decades of headroom over what is measured.
EXACT_ANGLE_TOL = 1e-11      # rad
EXACT_RANGE_RTOL = 1e-12


# --------------------------------------------------------------------------------------------------
# Closed-form elevation for the coplanar circular pass, and the interpolation bias it implies
# --------------------------------------------------------------------------------------------------

def _elevation_of_central_angle(u: float) -> float:
    """
    Elevation seen from a station at radius `R_BODY` of a satellite at radius `R_ORBIT`, `u` radians
    away at the body's centre. Straight from the triangle: the satellite's height above the station's
    horizontal plane is `r cos u - R` and its horizontal offset is `r sin u`.
    """
    return math.atan2(R_ORBIT * math.cos(u) - R_BODY, R_ORBIT * math.sin(u))


def _elevation_derivatives(u: float) -> Tuple[float, float]:
    """
    `(de/du, d2e/du2)` of `_elevation_of_central_angle`, in closed form.

    With `c = R/r` and `e = atan f`, `f = (cos u - c) / sin u`:

        f'    = (c cos u - 1) / sin^2 u
        1+f^2 = (1 - 2 c cos u + c^2) / sin^2 u
        de/du = (c cos u - 1) / (1 - 2 c cos u + c^2)                       = N / D
        d2e   = (N' D - N D') / D^2 = -c sin u (D + 2N) / D^2
              =  c sin u (1 - c^2) / D^2                                    since D + 2N = c^2 - 1

    At the horizon (`cos u = c`) these reduce to `-1` and `cot(lambda_0)`: the elevation leaves the
    horizon at exactly the station-relative angular rate, and curves *upward* from there.
    """
    c = C_RATIO
    denom = 1.0 - 2.0 * c * math.cos(u) + c * c
    first = (c * math.cos(u) - 1.0) / denom
    second = c * math.sin(u) * (1.0 - c * c) / denom ** 2
    return first, second


def _edge_bias(u_cross: float, a: float, b: float, *, setting: bool) -> float:
    """
    The `O(h^2)` bias of a linearly interpolated crossing, from `access_windows`' derivation:

        t_hat - t* = -(f'' / (2 f')) a b,     a = t* - t_i,  b = t_i+1 - t*

    The central angle runs as `u = Omega |t - t*|`, so `f' = +/- Omega de/du` (minus before the
    overhead moment) and `f'' = Omega^2 d2e/du2`. `de/du < 0` and `d2e/du2 > 0` at both crossings,
    which makes a rise come out **early** and a set **late** - the duration is over-estimated, and
    the sign of this function is as much the prediction as its magnitude.
    """
    first, second = _elevation_derivatives(u_cross)
    sign = 1.0 if setting else -1.0
    return -sign * OMEGA_REL * second / (2.0 * first) * a * b


def _bracket_offsets(grid: ArrF, t_cross: float) -> Tuple[float, float]:
    """`(a, b)`: how far the exact crossing sits from the samples that bracket it."""
    i = int(np.searchsorted(grid, t_cross)) - 1
    return t_cross - float(grid[i]), float(grid[i + 1]) - t_cross


def _phase_locked_grid(h: float, t_anchor: float, t_end: float) -> ArrF:
    """
    A uniform grid of spacing `h` on which `t_anchor` always falls 30 % into its interval.

    Halving `h` then scales the bracketing product `a b = 0.21 h^2` by exactly 4, which is what makes
    the convergence *order* measurable: on an arbitrary grid the crossing's position inside its
    interval jumps about, and the error ratio with it.
    """
    phase = (t_anchor - 0.3 * h) % h
    return phase + h * np.arange(int((t_end - phase) / h) + 1)


# --------------------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------------------

def _sample_states(
    session: Session, times: ArrF, *, inclination_deg: float = 0.0,
) -> ArrF:
    """Earth-relative inertial states of the single pass satellite, shape `(n_times, 1, 6)`."""
    sim = scenarios.ground_station_pass(
        session, altitude_km=ALT_KM, inclination_deg=inclination_deg,
    )
    earth = sim.name_to_index["Earth"]
    sat = sim.name_to_index["PASS-SAT-00"]
    states: ArrF = viz.sample_states(sim, [sat], times, relative_to=earth)
    return states


def _sample_positions(
    session: Session, times: ArrF, *, inclination_deg: float = 0.0,
) -> ArrF:
    """Earth-relative inertial positions of the single pass satellite, shape `(n_times, 1, 3)`."""
    return np.ascontiguousarray(
        _sample_states(session, times, inclination_deg=inclination_deg)[:, :, :3]
    )


def _look_angles(
    positions: ArrF, times: ArrF, *, lat: float = STATION_LAT, lon: float = STATION_LON,
) -> geometry.Topocentric:
    return geometry.elevation_azimuth(
        positions, times,
        latitude_rad=lat, longitude_rad=lon, altitude_km=0.0,
        omega=EARTH_OMEGA, body_radius_km=R_BODY,
    )


def _look_with_velocity(
    states: ArrF, times: ArrF, *, lat: float = STATION_LAT, lon: float = STATION_LON,
) -> geometry.Topocentric:
    """The same look angles, plus range rate, from a `(n_times, n_bodies, 6)` state array."""
    return geometry.elevation_azimuth(
        np.ascontiguousarray(states[:, :, :3]), times,
        velocities_km_s=np.ascontiguousarray(states[:, :, 3:]),
        latitude_rad=lat, longitude_rad=lon, altitude_km=0.0,
        omega=EARTH_OMEGA, body_radius_km=R_BODY,
    )


def _rho(t: float) -> float:
    """Closed-form slant range of the coplanar equatorial pass, `sqrt(r^2 + R^2 - 2 r R cos u)`."""
    u = OMEGA_REL * (t - T_OVERHEAD)
    return math.sqrt(R_ORBIT ** 2 + R_BODY ** 2 - 2.0 * R_ORBIT * R_BODY * math.cos(u))


def _rho_dot(t: float) -> float:
    """Its derivative, `r R sin(u) Omega / rho` - positive after the overhead moment (opening)."""
    u = OMEGA_REL * (t - T_OVERHEAD)
    return R_ORBIT * R_BODY * math.sin(u) * OMEGA_REL / _rho(t)


# ==================================================================================================
# Conventions, against hand-computed vectors
# ==================================================================================================

def test_station_position_matches_hand_computed_sites() -> None:
    """Three sites whose body-fixed vectors can be written down, plus the altitude offset."""
    lats = np.array([0.0, 0.0, 0.5 * math.pi])
    lons = np.array([0.0, 0.5 * math.pi, 0.0])
    sites = geometry.station_position_fixed(lats, lons, 0.0, body_radius_km=R_BODY)

    expected = R_BODY * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(sites, expected, atol=1e-9)

    raised = geometry.station_position_fixed(0.0, 0.0, 100.0, body_radius_km=R_BODY)
    assert raised.shape == (1, 3)
    assert abs(float(raised[0, 0]) - (R_BODY + 100.0)) < 1e-9


def test_sez_triad_is_right_handed_and_azimuth_runs_clockwise_from_north() -> None:
    """
    At `(0 deg, 0 deg)` with the body unrotated: zenith `+x`, east `+y`, north `+z`.

    A target 100 km due north of the station on its local horizontal must read azimuth 0 and
    elevation 0; east 90 deg, south 180 deg, west 270 deg. This is the assertion that fails if S and
    E are transposed, or if azimuth is measured anticlockwise.
    """
    station = np.array([R_BODY, 0.0, 0.0])
    offsets = {0.0: (0.0, 0.0, 1.0), 90.0: (0.0, 1.0, 0.0),
               180.0: (0.0, 0.0, -1.0), 270.0: (0.0, -1.0, 0.0)}
    for expected_az_deg, direction in offsets.items():
        target = (station + 100.0 * np.array(direction))[np.newaxis, np.newaxis, :]
        look = geometry.elevation_azimuth(
            target, np.array([0.0]),
            latitude_rad=0.0, longitude_rad=0.0, omega=0.0, body_radius_km=R_BODY,
        )
        assert abs(float(np.degrees(look.azimuth_rad[0, 0, 0])) - expected_az_deg) < 1e-9
        assert abs(float(look.elevation_rad[0, 0, 0])) < 1e-12
        assert abs(float(look.range_km[0, 0, 0]) - 100.0) < 1e-9


def test_zenith_at_a_rotated_epoch() -> None:
    """
    The rotation-sign anchor. See `geometry.py`'s module docstring and `docs/engineering-log.md`.

    A station on the prime meridian with the body turned 90 deg has moved to inertial `+y`, so a
    satellite at `(0, r, 0)` is exactly overhead: elevation 90 deg, range `r - R`. With the sign of
    the rotation flipped the station is at `-y`, the satellite is on the far side of the body, and
    the elevation reads -90 deg. Nothing about the magnitude of either answer is plausible-looking
    enough to survive, which is why the epoch is 90 deg and not 1 deg.
    """
    sat = np.array([[[0.0, R_ORBIT, 0.0]]])
    look = geometry.elevation_azimuth(
        sat, np.array([0.0]),
        latitude_rad=0.0, longitude_rad=0.0, omega=0.0, body_radius_km=R_BODY,
        theta0=0.5 * math.pi,
    )
    assert abs(float(look.elevation_rad[0, 0, 0]) - 0.5 * math.pi) < 1e-12
    assert abs(float(look.range_km[0, 0, 0]) - ALT_KM) < 1e-9


def test_the_rotation_epoch_is_absolute_not_grid_relative() -> None:
    """
    Two grids over the same instant must give the same look angles.

    `theta0` is referenced to `epoch_s` (default 0.0), not to `times_s[0]`. Were it grid-relative -
    which is `viz.ground_track`'s convention - the same satellite at the same absolute time would be
    seen from a station displaced by `omega * times_s[0]`, silently moving every reported rise and
    set. That is a real mistake made while building this module; the log records it.
    """
    sat = np.array([[[0.0, R_ORBIT, 0.0]], [[R_ORBIT, 0.0, 0.0]]])
    early = geometry.elevation_azimuth(
        sat, np.array([1000.0, 2000.0]),
        latitude_rad=0.3, longitude_rad=-0.7, omega=EARTH_OMEGA, body_radius_km=R_BODY,
    )
    late = geometry.elevation_azimuth(
        sat[1:], np.array([2000.0]),
        latitude_rad=0.3, longitude_rad=-0.7, omega=EARTH_OMEGA, body_radius_km=R_BODY,
    )
    assert abs(float(early.elevation_rad[1, 0, 0]) - float(late.elevation_rad[0, 0, 0])) < 1e-15


def test_shape_and_ordering_contracts() -> None:
    """A single-body position array is promoted; mismatched grids and shapes raise, not broadcast."""
    times = np.array([0.0, 60.0])
    pos = np.array([[R_ORBIT, 0.0, 0.0], [0.0, R_ORBIT, 0.0]])
    look = geometry.elevation_azimuth(
        pos, times, latitude_rad=[0.0, 0.5], longitude_rad=[0.0, 1.0],
        omega=EARTH_OMEGA, body_radius_km=R_BODY,
    )
    assert look.elevation_rad.shape == (2, 2, 1)

    with pytest.raises(ValueError):
        geometry.elevation_azimuth(
            pos, np.array([0.0]), latitude_rad=0.0, longitude_rad=0.0,
            omega=0.0, body_radius_km=R_BODY,
        )
    with pytest.raises(ValueError):
        geometry.access_windows(np.array([0.0, 0.0]), np.zeros(2))
    with pytest.raises(ValueError):
        geometry.line_of_sight(np.zeros(2), np.zeros(3), body_radius_km=R_BODY)


# ==================================================================================================
# The pass, against closed-form horizon geometry
# ==================================================================================================

def test_overhead_pass_hits_ninety_degrees_and_the_horizon_exactly(db_session: Session) -> None:
    """
    Sampled at the three instants the geometry names, the engine must reproduce them exactly.

    At `T_OVERHEAD` the satellite shares the station's longitude and both are equatorial, so the
    elevation is exactly 90 deg. At `T_RISE` and `T_SET` the central angle is exactly `lambda_0`, so
    the elevation is 0 and the slant range is `sqrt(r^2 - R^2)` - the other half of what the horizon
    half-angle fixes. Nothing here is interpolated: the instants are analytic and the propagation is
    closed form, so the only error is floating point.
    """
    times = np.array([T_RISE, T_OVERHEAD, T_SET])
    look = _look_angles(_sample_positions(db_session, times), times)
    elevation = look.elevation_rad[:, 0, 0]

    assert abs(float(elevation[1]) - 0.5 * math.pi) < EXACT_ANGLE_TOL
    assert abs(float(elevation[0])) < EXACT_ANGLE_TOL
    assert abs(float(elevation[2])) < EXACT_ANGLE_TOL

    assert float(look.range_km[1, 0, 0]) == pytest.approx(ALT_KM, rel=EXACT_RANGE_RTOL)
    assert float(look.range_km[0, 0, 0]) == pytest.approx(RANGE_HORIZON, rel=EXACT_RANGE_RTOL)
    assert float(look.range_km[2, 0, 0]) == pytest.approx(RANGE_HORIZON, rel=EXACT_RANGE_RTOL)


def test_a_prograde_satellite_rises_in_the_west_and_sets_in_the_east(db_session: Session) -> None:
    """
    Azimuth 270 deg at the horizon crossing before the overhead moment, 90 deg after.

    `n > omega`, so the satellite gains on the station and moves eastward across its sky. This is
    the only assertion in the suite that would notice azimuth being measured from south, or the
    satellite's motion being reversed; the elevation checks are symmetric in time and would not.
    """
    times = np.array([T_RISE, T_SET])
    look = _look_angles(_sample_positions(db_session, times), times)
    azimuth_deg = np.degrees(look.azimuth_rad[:, 0, 0])
    assert abs(float(azimuth_deg[0]) - 270.0) < 1e-9
    assert abs(float(azimuth_deg[1]) - 90.0) < 1e-9


@pytest.mark.parametrize("mask_deg", [0.0, 10.0])
def test_window_duration_matches_the_horizon_half_angle(
    db_session_factory: Callable[[], Session], mask_deg: float,
) -> None:
    """
    The reported window equals `2 lambda_e / Omega` plus the interpolation bias, and nothing else.

    `lambda_e = acos((R/r) cos e) - e` is the central half-angle at mask elevation `e`; `Omega` is
    the station-relative rate `n - omega`, which is where the body's rotation enters - using `n`
    alone would shorten the 0 deg window from 784.2 s to 732.1 s, a 6.7 % error.

    The measured duration is *not* compared to the closed form directly. It is compared to the closed
    form **plus** `_edge_bias`, evaluated at the actual bracketing offsets. The residual is the
    `O(h^3)` remainder of the same expansion; asserting it is under 5 % of the bias is what stops
    this from being a snapshot, and the bias itself is ~0.1 % of the duration.
    """
    mask = math.radians(mask_deg)
    lambda_mask = math.acos(C_RATIO * math.cos(mask)) - mask
    duration_exact = 2.0 * lambda_mask / OMEGA_REL
    t_rise = T_OVERHEAD - lambda_mask / OMEGA_REL
    t_set = T_OVERHEAD + lambda_mask / OMEGA_REL

    grid = np.arange(0.0, 4200.0, 15.0)
    look = _look_angles(_sample_positions(db_session_factory(), grid), grid)
    windows = geometry.access_windows(grid, look.elevation_rad, mask_angle_rad=mask)

    assert len(windows) == 1
    window = windows[0]
    assert not window.rise_clipped and not window.set_clipped
    assert window.station_index == 0 and window.body_index == 0

    bias = (
        _edge_bias(lambda_mask, *_bracket_offsets(grid, t_set), setting=True)
        - _edge_bias(lambda_mask, *_bracket_offsets(grid, t_rise), setting=False)
    )
    assert bias > 0.0, "a linearly interpolated pass must come out long, not short"
    residual = window.duration_s - (duration_exact + bias)
    assert abs(residual) < 0.05 * abs(bias)

    # The mask shortens the window by exactly what the geometry says, to the same accuracy.
    assert window.duration_s - bias == pytest.approx(duration_exact, abs=0.05 * abs(bias))


def test_a_mask_angle_shortens_the_window_by_the_predicted_amount(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    10 deg of mask costs 273.8 s of a 784.2 s pass - a third of it - and that is a derived number.

    Both windows are measured on one grid geometry so the comparison is of two closed forms against
    two measurements, not of a measurement against itself.
    """
    mask = math.radians(10.0)
    lambda_mask = math.acos(C_RATIO * math.cos(mask)) - mask
    shortening_exact = 2.0 * (LAMBDA_0 - lambda_mask) / OMEGA_REL
    assert shortening_exact == pytest.approx(273.79, abs=0.01)

    grid = np.arange(0.0, 4200.0, 15.0)
    look = _look_angles(_sample_positions(db_session_factory(), grid), grid)
    open_window = geometry.access_windows(grid, look.elevation_rad)[0]
    masked_window = geometry.access_windows(grid, look.elevation_rad, mask_angle_rad=mask)[0]

    # Each edge carries its own bias, so the difference of durations carries their difference too.
    bias_open = (
        _edge_bias(LAMBDA_0, *_bracket_offsets(grid, T_SET), setting=True)
        - _edge_bias(LAMBDA_0, *_bracket_offsets(grid, T_RISE), setting=False)
    )
    t_rise_m = T_OVERHEAD - lambda_mask / OMEGA_REL
    t_set_m = T_OVERHEAD + lambda_mask / OMEGA_REL
    bias_masked = (
        _edge_bias(lambda_mask, *_bracket_offsets(grid, t_set_m), setting=True)
        - _edge_bias(lambda_mask, *_bracket_offsets(grid, t_rise_m), setting=False)
    )
    measured = (open_window.duration_s - bias_open) - (masked_window.duration_s - bias_masked)
    assert measured == pytest.approx(shortening_exact, abs=0.05 * (bias_open + bias_masked))

    # The masked window sits strictly inside the open one, on both edges.
    assert masked_window.rise_s > open_window.rise_s
    assert masked_window.set_s < open_window.set_s


def test_window_edges_converge_second_order(db_session_factory: Callable[[], Session]) -> None:
    """
    Halving the sample spacing quarters the rise-time error, and the error equals its derived bias.

    The grids are phase-locked (`_phase_locked_grid`) so `a b = 0.21 h^2` exactly at every spacing;
    without that the ratio wanders between 1 and 10 with where the crossing happens to land, and the
    order cannot be read off three numbers. Expected ratio 4, with an `O(h)` correction of a few
    percent from the next term in the expansion.
    """
    errors: List[float] = []
    for h in (30.0, 15.0, 7.5):
        grid = _phase_locked_grid(h, T_RISE, 4200.0)
        look = _look_angles(_sample_positions(db_session_factory(), grid), grid)
        window = geometry.access_windows(grid, look.elevation_rad)[0]

        a, b = _bracket_offsets(grid, T_RISE)
        assert a == pytest.approx(0.3 * h, rel=1e-9)
        predicted = _edge_bias(LAMBDA_0, a, b, setting=False)
        measured = window.rise_s - T_RISE
        assert measured < 0.0, "a linearly interpolated rise must come out early"
        assert measured == pytest.approx(predicted, rel=0.05)
        errors.append(abs(measured))

    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 3.6 < coarse / fine < 4.4


def test_a_clipped_window_is_flagged_rather_than_trusted(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    A grid that starts inside the pass reports the window from its first sample, and says so.

    An access metric that averaged durations without reading `rise_clipped` would silently mix a
    truncated pass in with whole ones; the flag is the whole reason the field exists.
    """
    grid = np.arange(T_OVERHEAD - 100.0, 4200.0, 15.0)
    look = _look_angles(_sample_positions(db_session_factory(), grid), grid)
    window = geometry.access_windows(grid, look.elevation_rad)[0]

    assert window.rise_clipped and not window.set_clipped
    assert window.rise_s == float(grid[0])
    assert window.duration_s < DURATION_EXACT


def test_inclined_pass_matches_the_general_triangle(db_session: Session) -> None:
    """
    The general case, against a station-satellite-centre triangle built independently of the module.

    The station's *inertial* position is written out here as
    `R (cos(lon + theta) cos lat, sin(lon + theta) cos lat, sin lat)` with `theta = omega t` - the
    forward statement that the site's inertial longitude advances with the body - and the elevation
    and range follow from the central angle `lambda` alone. That is an independent derivation of both
    the SEZ rotation and the rotation *sign*: a station carried the wrong way round would disagree
    by up to `2 omega t`, which over 6000 s is 50 deg of longitude.

    The orbit is inclined 53 deg and the station is at 35 N, 20 E, so the SEZ south component is
    non-zero throughout - the equatorial case cannot see a transposed S/E axis, this one can.
    """
    lat, lon = math.radians(35.0), math.radians(20.0)
    times = np.linspace(0.0, 6000.0, 601)
    positions = _sample_positions(db_session, times, inclination_deg=53.0)
    look = _look_angles(positions, times, lat=lat, lon=lon)

    theta = EARTH_OMEGA * times
    station_inertial = R_BODY * np.stack([
        np.cos(lon + theta) * math.cos(lat),
        np.sin(lon + theta) * math.cos(lat),
        np.full_like(theta, math.sin(lat)),
    ], axis=1)

    sat = positions[:, 0, :]
    radii = np.linalg.norm(sat, axis=1)
    central = np.arccos(
        np.clip(np.sum(sat * station_inertial, axis=1) / (radii * R_BODY), -1.0, 1.0)
    )
    elevation_exact = np.arctan2(radii * np.cos(central) - R_BODY, radii * np.sin(central))
    range_exact = np.sqrt(radii ** 2 + R_BODY ** 2 - 2.0 * radii * R_BODY * np.cos(central))

    assert np.max(np.abs(elevation_exact - look.elevation_rad[:, 0, 0])) < EXACT_ANGLE_TOL
    assert np.allclose(range_exact, look.range_km[:, 0, 0], rtol=EXACT_RANGE_RTOL)
    # The case is only non-degenerate if the satellite actually rises there.
    assert float(np.max(look.elevation_rad)) > math.radians(20.0)


# ==================================================================================================
# Range rate
# ==================================================================================================

def test_range_rate_is_absent_unless_velocities_are_given(db_session: Session) -> None:
    """
    The field is `None` without velocities - the contract that keeps every existing caller working -
    and the look angles are bit-identical with and without them, since velocity enters nothing else.
    """
    times = np.array([T_RISE, T_OVERHEAD, T_SET])
    states = _sample_states(db_session, times)
    without = _look_angles(np.ascontiguousarray(states[:, :, :3]), times)
    with_v = _look_with_velocity(states, times)

    assert without.range_rate_km_s is None
    assert with_v.range_rate_km_s is not None
    assert np.array_equal(without.elevation_rad, with_v.elevation_rad)
    assert np.array_equal(without.range_km, with_v.range_km)

    with pytest.raises(ValueError):
        geometry.elevation_azimuth(
            np.zeros((3, 1, 3)), times, velocities_km_s=np.zeros((3, 2, 3)),
            latitude_rad=0.0, longitude_rad=0.0, omega=0.0, body_radius_km=R_BODY,
        )


def test_range_rate_matches_the_coplanar_closed_form_and_signs_opening_positive(
    db_session: Session,
) -> None:
    """
    The three named instants against `rho_dot = r R sin(u) Omega / rho`, and the sign convention.

    Differentiating `rho^2 = r^2 + R^2 - 2 r R cos u` with `u = Omega (t - t_overhead)` gives that
    closed form directly; `Omega = n - omega` is the **station-relative** rate, which is where the
    transport term lives (see the next test). At the horizon it is 6.5213 km/s, and at the analytic
    instant the only error is floating point - nothing here is interpolated.

    **Sign: positive = opening.** Before the overhead moment `sin u < 0` and the satellite is
    closing, after it the range is growing. An inverted convention passes every magnitude check in
    this file and inverts every downstream Doppler shift, so it is asserted as two signs, not as an
    absolute value.
    """
    times = np.array([T_RISE, T_OVERHEAD, T_SET])
    rate = _look_with_velocity(_sample_states(db_session, times), times).range_rate_km_s
    assert rate is not None
    measured = rate[:, 0, 0]

    assert float(measured[0]) == pytest.approx(-RANGE_RATE_HORIZON, rel=EXACT_RANGE_RTOL)
    assert float(measured[2]) == pytest.approx(+RANGE_RATE_HORIZON, rel=EXACT_RANGE_RTOL)
    assert float(measured[0]) < 0.0 < float(measured[2])


def test_range_rate_vanishes_at_closest_approach(db_session_factory: Callable[[], Session]) -> None:
    """
    Exactly zero at the overhead instant, and *not* resolvable to zero on a sampled grid.

    `rho_dot = r R Omega sin(u) / rho` vanishes at `u = 0`, so the analytic instant must return zero
    up to floating point. How much is that? An angular position error `du` shows up as
    `r R Omega du / rho = 82.1 km/s per radian`; the propagation is closed form, so `du` is a few
    ulp of a `pi`-sized angle, ~2e-15 rad, giving **~1.6e-13 km/s**. The assertion is 1e-11, and the
    measured value is 5.4e-14.

    The grid half of it is the reason that instant has to be asked for explicitly: the sample of
    *minimum range* sits up to `h/2` from closest approach, where the rate has already grown to
    `rho'' h / 2` with `rho''(0) = r R Omega^2 / (r - R) = 0.0840 km/s^2` - 1.26 km/s at `h = 30 s`.
    A test that sampled a grid and asserted "about zero" would be asserting nothing.
    """
    times = np.array([T_OVERHEAD])
    rate = _look_with_velocity(_sample_states(db_session_factory(), times), times).range_rate_km_s
    assert rate is not None
    assert abs(float(rate[0, 0, 0])) < 1e-11

    h = 30.0
    grid = np.arange(T_RISE - 100.0, T_SET + 100.0, h)
    look = _look_with_velocity(_sample_states(db_session_factory(), grid), grid)
    assert look.range_rate_km_s is not None
    nearest = int(np.argmin(look.range_km[:, 0, 0]))
    bound = RHO_DDOT_CLOSEST * h / 2.0
    assert abs(float(look.range_rate_km_s[nearest, 0, 0])) <= bound
    # ... and it is nowhere near zero, which is what makes the analytic instant the only honest test.
    assert abs(float(look.range_rate_km_s[nearest, 0, 0])) > 0.1 * bound


def test_the_transport_term_is_the_stations_own_speed(db_session: Session) -> None:
    """
    Dropping `omega x r_station` would replace `Omega = n - omega` by `n`: 7.1 % at the horizon.

    The two closed forms differ by `r R omega sin(lambda_0) / rho_0`, which for a station on the
    equator and a satellite on its horizon is **exactly `R omega = 0.4646 km/s`** - the station's
    own ground speed, projected entirely onto the line of sight because that line is tangent to the
    sphere at the station and the station's velocity is due east, along it. So the assertion is a
    number, not a tolerance: the measured rate must sit `R omega` away from the no-transport answer,
    and on the correct side of it.

    0.4646 km/s against LEO range rates of 6.5 km/s is not a rounding effect, and it is the class of
    error that produces a plausible pass and a wrong Doppler.
    """
    times = np.array([T_SET])
    rate = _look_with_velocity(_sample_states(db_session, times), times).range_rate_km_s
    assert rate is not None
    measured = float(rate[0, 0, 0])

    no_transport = R_ORBIT * R_BODY * math.sin(LAMBDA_0) * N_MEAN / RANGE_HORIZON
    assert no_transport - measured == pytest.approx(R_BODY * EARTH_OMEGA, rel=1e-9)
    assert no_transport - measured == pytest.approx(0.464581, abs=1e-6)
    # Without the term the engine would read 6.9859 km/s where the geometry says 6.5213 km/s.
    assert measured == pytest.approx(RANGE_RATE_HORIZON, rel=EXACT_RANGE_RTOL)
    assert abs(measured - no_transport) / abs(measured) == pytest.approx(0.0712, abs=1e-4)


def test_range_rate_matches_a_central_difference_of_the_range(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The independent check: the range *series* differenced numerically, sharing no algebra with the
    analytic rate.

    A central difference of `rho` has error `rho''' h^2 / 6`. Two things are asserted, in the two
    geometries:

    * **Equatorial.** The engine's own error, `(rho(t+h) - rho(t-h)) / 2h - rho_dot(t)`, is compared
      with the *same* difference taken on the closed-form `rho(t)` of the coplanar pass - a
      prediction, not a fitted tolerance. At `t = T_RISE + 100 s` and `h = 8 s` both are
      +1.8146e-4 km/s (`rho''' = 1.700e-5 km/s^3` gives `rho''' h^2 / 6 = 1.814e-4`), agreeing to
      1.6e-9 relative.
    * **Inclined 53 deg, station at 35 N / 20 E.** Nothing closed-form here, so what is asserted is
      the *order*: halving `h` must quarter the error. Measured ratios are 4.000, 4.000, 4.000.

    A missing transport term would break both, because the range series it differences already
    contains the station's motion while the analytic rate would not.
    """
    t0 = T_RISE + 100.0
    for h in (8.0, 4.0, 2.0):
        grid = np.array([t0 - h, t0, t0 + h])
        look = _look_with_velocity(_sample_states(db_session_factory(), grid), grid)
        assert look.range_rate_km_s is not None
        ranges = look.range_km[:, 0, 0]
        engine_error = float((ranges[2] - ranges[0]) / (2.0 * h) - look.range_rate_km_s[1, 0, 0])
        closed_error = (_rho(t0 + h) - _rho(t0 - h)) / (2.0 * h) - _rho_dot(t0)
        assert engine_error == pytest.approx(closed_error, rel=1e-5)

    lat, lon = math.radians(35.0), math.radians(20.0)
    errors: List[float] = []
    for h in (8.0, 4.0, 2.0, 1.0):
        grid = np.array([2500.0 - h, 2500.0, 2500.0 + h])
        look = _look_with_velocity(
            _sample_states(db_session_factory(), grid, inclination_deg=53.0), grid,
            lat=lat, lon=lon,
        )
        assert look.range_rate_km_s is not None
        ranges = look.range_km[:, 0, 0]
        errors.append(float((ranges[2] - ranges[0]) / (2.0 * h) - look.range_rate_km_s[1, 0, 0]))

    assert abs(errors[0]) > 1e-6, "the inclined case must have a measurable error to halve"
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, abs=0.05)


# ==================================================================================================
# Access-window bookkeeping, on a series with no propagation in it
# ==================================================================================================

def test_linear_crossings_are_exact_and_multiple_windows_separate() -> None:
    """
    Linear inverse interpolation is *exact* for a piecewise-linear elevation - the `O(h^2)` term
    carries `f''`, which is zero here - so a triangular series pins the arithmetic with no tolerance
    to argue about. Two disjoint triangles must come back as two windows in time order.
    """
    times = np.arange(0.0, 40.0, 1.0)
    elevation = np.concatenate([
        np.linspace(-5.0, 5.0, 10), np.linspace(5.0, -5.0, 10),
        np.linspace(-5.0, 5.0, 10), np.linspace(5.0, -5.0, 10),
    ])
    windows = geometry.access_windows(times, elevation)
    assert len(windows) == 2
    assert windows[0].set_s < windows[1].rise_s

    # The first ramp crosses zero where the line does, to the last bit.
    slope = 10.0 / 9.0
    assert windows[0].rise_s == pytest.approx(5.0 / slope, abs=1e-12)
    assert windows[0].peak_elevation_rad == pytest.approx(5.0)

    # Raising the mask must strictly shrink both windows, never reorder or merge them.
    masked = geometry.access_windows(times, elevation, mask_angle_rad=2.5)
    assert len(masked) == 2
    for tight, wide in zip(masked, windows):
        assert tight.rise_s > wide.rise_s and tight.set_s < wide.set_s


def test_windows_are_reported_per_station_and_body() -> None:
    """The station and body axes are kept distinct, and indices identify which pair each window is."""
    times = np.arange(0.0, 20.0, 1.0)
    ramp = np.concatenate([np.linspace(-2.0, 2.0, 10), np.linspace(2.0, -2.0, 10)])
    elevation = np.full((times.size, 2, 3), -1.0)
    elevation[:, 1, 2] = ramp
    windows = geometry.access_windows(times, elevation)
    assert len(windows) == 1
    assert (windows[0].station_index, windows[0].body_index) == (1, 2)


# ==================================================================================================
# Line of sight
# ==================================================================================================

def test_line_of_sight_boundary_on_a_circular_orbit() -> None:
    """
    Two satellites on a circular orbit of radius `r` are blocked exactly when the central angle
    between them exceeds `2 acos(R / r)`.

    The chord between them passes `r cos(phi/2)` from the centre, which equals `R` precisely at
    `phi = 2 lambda_0`. Both sides of the boundary are checked at 0.1 %, which at 550 km is 2.6 km of
    closest approach - far outside any rounding, and far inside anything a wrong formula would give.
    """
    phi_boundary = 2.0 * LAMBDA_0
    for factor, expected in ((0.999, True), (1.001, False)):
        phi = phi_boundary * factor
        first = R_ORBIT * np.array([1.0, 0.0, 0.0])
        second = R_ORBIT * np.array([math.cos(phi), math.sin(phi), 0.0])
        assert bool(geometry.line_of_sight(first, second, body_radius_km=R_BODY)) is expected

    # Exactly tangent counts as visible, by the documented `>=`.
    tangent_first = R_ORBIT * np.array([1.0, 0.0, 0.0])
    tangent_second = R_ORBIT * np.array([math.cos(phi_boundary), math.sin(phi_boundary), 0.0])
    assert bool(geometry.line_of_sight(tangent_first, tangent_second, body_radius_km=R_BODY))


def test_the_segment_clamp_keeps_a_station_link_visible() -> None:
    """
    A ground station and a satellite 10 deg away must see each other; an unclamped test says they do
    not.

    The infinite line through them leaves the station climbing, so its closest approach to the centre
    lies *behind* the station at `tau < 0` and is well inside the body. Only the clamp to `[0, 1]`
    makes this a segment test. Checked by asserting the visibility that the clamp provides, together
    with the fact that the unclamped minimiser really is out of range - otherwise the case would be
    vacuous.
    """
    station = R_BODY * np.array([1.0, 0.0, 0.0])
    lam = math.radians(10.0)
    satellite = R_ORBIT * np.array([math.cos(lam), math.sin(lam), 0.0])

    delta = satellite - station
    tau_unclamped = -float(station @ delta) / float(delta @ delta)
    assert tau_unclamped < 0.0
    assert np.linalg.norm(station + tau_unclamped * delta) < R_BODY

    assert bool(geometry.line_of_sight(station, satellite, body_radius_km=R_BODY))


def test_line_of_sight_broadcasts_and_is_symmetric() -> None:
    """`(N,3)` against `(3,)` broadcasts to `(N,)`, and swapping the endpoints changes nothing."""
    phi = np.linspace(0.0, math.pi, 37)
    ring = R_ORBIT * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    anchor = R_ORBIT * np.array([1.0, 0.0, 0.0])

    visible = geometry.line_of_sight(ring, anchor, body_radius_km=R_BODY)
    assert visible.shape == (37,)
    assert np.array_equal(visible, geometry.line_of_sight(anchor, ring, body_radius_km=R_BODY))
    # Visibility is a contiguous run from the anchor outwards, ending at the boundary angle.
    assert bool(visible[0]) and not bool(visible[-1])
    last_visible = int(np.flatnonzero(visible)[-1])
    assert phi[last_visible] <= 2.0 * LAMBDA_0 < phi[last_visible + 1]
