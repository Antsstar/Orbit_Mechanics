"""
Observation geometry: where a body is in a ground station's sky, when it is in view, and whether two
points can see each other past a spherical body.

**Why this module exists.** The engine reports model error in kilometres, and no user's decision is
in kilometres. For a constellation or network study the decision is in *contact windows* - when a
link opens, how long it lasts, whether a marginal pass is visible at all. These three primitives are
what a later sweep-level access metric is built on; nothing here knows about `Simulation`, exactly as
`viz.py`'s transforms do not, so every function below can be checked against closed-form geometry
with no propagation involved.

**Not a force model and not a propagator.** Nothing here is registered in `registry.py`: there is no
acceleration to compose and no state to advance. `manoeuvres.py` sets the same precedent for physics
that is neither.

References - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.:

* Sec. 3.2, site position from geodetic/geocentric coordinates (Alg. 51, `SITE`). We use the
  **spherical** specialisation; see "Coordinates" below.
* Sec. 4.4, the topocentric-horizon (SEZ) system, and Alg. 27 (`RAZEL`) for range/azimuth/elevation.
  The ECEF -> SEZ rotation `ROT2(90 deg - phi) ROT3(lambda)` is written out and re-derived in
  `_sez_rotations` rather than taken on trust.
* Sec. 5.3, Alg. 35 (`SIGHT`), the line-of-sight test. Our `line_of_sight` is the spherical form of
  it, with the parameter clamped to the segment (see there).

Algorithm and section numbers are from memory and **unverified against the text**; each formula is
derived in place, so the derivation - not the citation - is what the tests hold.

## Coordinates, stated rather than implied

* **Latitude is geocentric (spherical), not geodetic.** A station at geocentric latitude `phi`,
  east longitude `lambda` and altitude `h` sits at radius `body_radius_km + h` from the centre. This
  engine carries no reference ellipsoid - its J2 is a gravity-field coefficient with no flattening
  attached, and `viz.GroundTrack` already reports spherical latitude for the same reason. On Earth
  the two latitudes differ by up to 0.19 deg, which is ~21 km of surface position: material for a
  real station, and out of scope for a model-fidelity comparison whose satellites are seeded from
  spherical elements anyway. Feed geodetic values here and you get a consistently displaced station,
  not a crash.
* **Longitude is positive east**, matching `frames.fixed_to_longlat`'s convention.
* **Azimuth is measured from local north, positive clockwise (toward east)**, in `[0, 2 pi)`.
  Elevation is measured up from the local horizontal plane, in `[-pi/2, +pi/2]`. Range is in km.
  All angles in and out are **radians**, unlike `viz.ground_track`, which reports degrees - radians
  are the engine-wide convention and a plotting layer is the right place to convert.
* **The central body rotates about the frame's +z axis** at a constant rate `omega`, with prime
  meridian angle `theta(t) = theta0 + omega * (t - epoch_s)`. Rotation about +z is the same
  assumption `geopotential.py`, `drag.py` and `viz.ground_track` already make. No precession, no
  nutation, no polar motion: `pyerfa` is the boundary to reach for if those ever matter.
* **`epoch_s` defaults to 0.0, i.e. absolute simulation time**, and this is where `elevation_azimuth`
  deliberately *differs* from `viz.ground_track`, which references `theta0` to `times_s[0]`. A ground
  track is a shape and an arbitrary rotation of it is still a valid figure; a pass time is not, and
  referencing the rotation to the first sample makes the answer depend on where the grid happens to
  start. Sampling the same pass on a grid beginning at `t = 28 s` instead of `t = 0` moved every
  reported rise and set by `omega * 28 / (n - omega) = 2.0 s` - larger than the interpolation error
  the edges are refined to, and entirely silent. With the default the station's position is a
  function of simulation time alone, which is what `Simulation.t` means.

## Range rate, and its sign

`elevation_azimuth` optionally returns **range rate**, the quantity a downstream link or network
study needs for Doppler. It is `d|rho| / dt` and nothing else - a kinematic scalar. Carrier
frequency, Doppler shift and link margin are properties of a *radio*, not of an orbit, and are
deliberately on the far side of this module's boundary.

**Sign: positive = opening (the range is growing, the body is receding); negative = closing.** That
is the plain time derivative of the reported `range_km`, so `d(range)/dt > 0` and "range rate > 0"
say the same thing, and a consumer computing a received frequency applies its own minus sign
(`f_rx ~ f_tx (1 - rho_dot / c)`: a positive range rate is a red shift). Stating it here because an
inverted convention is a sign error no plot of a pass would reveal.

**The transport term is the whole content of the calculation.** The station is fixed in the
*rotating* frame, so its inertial velocity is `omega x r_station`, and

    rho      = r_body - r_station
    rho_dot  = v_body - omega x r_station                       <- the transport term
    d|rho|/dt = (rho . rho_dot) / |rho|

Omitting `omega x r_station` leaves a smooth, plausible, wrong answer. For the coplanar equatorial
550 km pass of `scenarios.ground_station_pass` the closed form is
`d|rho|/dt = r R sin(u) (n - omega) / |rho|` with `u` the central angle, and dropping the transport
term replaces the station-relative rate `n - omega` by `n`: at the horizon crossing that is
**0.4646 km/s out of 6.521 km/s, 7.1 %** - and 0.4646 km/s is exactly `R omega`, the station's own
speed, because at the horizon the line of sight is tangent to the sphere at the station and the
station's due-east velocity lies along it. `tests/validation/test_geometry.py` asserts that
difference as a number rather than asserting a tolerance.

Two equivalent formulations exist and the module uses the first: differentiate in the inertial
frame and rotate the result (`A v_body - omega x r_station_fixed`, with `A` the body-fixed
rotation), or work in the rotating frame (`A v_body - omega x A r_body`). Their difference is
`omega x rho`, which is perpendicular to `rho`, so the projection onto `rho` - the only thing range
rate uses - is identical. The first is used because `omega x r_station_fixed` is one constant vector
per station rather than one per sample.

## The rotation sign

`ReferenceFrames.inertia_to_fixed` applies `Transformations.Rz(theta)`, an **active** rotation, so
expressing an inertial vector in a frame whose axes have themselves turned by `+theta` takes
`-theta` - the inverse. `elevation_azimuth` therefore passes `-theta`, exactly as `viz.ground_track`
does, and `docs/engineering-log.md` records the failure this causes when reversed: nothing raises,
the geometry stays smooth and plausible, and the station is simply in the wrong place by `2 omega t`.
`tests/validation/test_geometry.py::test_zenith_at_a_rotated_epoch` is the assertion that pins it -
a station at longitude 0 with the body turned 90 deg must see a satellite at inertial `(0, r, 0)`
directly overhead, and under the flipped sign it sees it 180 deg away, far below the horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, ArraySeconds, Numeric, ScalarFloat
from .frames import ReferenceFrames
from .utilities import Transformations

__all__ = [
    "Topocentric", "AccessWindow",
    "station_position_fixed", "elevation_azimuth", "access_windows", "line_of_sight",
]


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Topocentric:
    """
    Look angles from each station to each body at each sample, shape `(n_times, n_stations, n_bodies)`.

    `elevation_rad` is measured up from the station's local horizontal, `azimuth_rad` clockwise from
    local north in `[0, 2 pi)`, `range_km` is the straight-line distance. A body at the station's
    exact zenith or nadir has an undefined azimuth; `np.arctan2(0, 0)` returns 0.0 there rather than
    raising, so azimuth at 90 deg elevation is meaningless, not wrong.

    The station axis is always present, even for a single scalar station, so downstream code never
    has to branch on it.

    `range_rate_km_s` is `None` unless `elevation_azimuth` was given velocities, and otherwise has
    the same shape as `range_km`. **Positive means opening (receding)** - see the module docstring;
    it is `d(range_km)/dt` including the station's own motion. A body at the station's exact
    position has an undefined range rate and reports exactly 0.0, the same convention azimuth uses
    at the zenith.
    """
    times_s: ArraySeconds
    elevation_rad: ArrayFloat
    azimuth_rad: ArrayFloat
    range_km: ArrayFloat
    range_rate_km_s: Optional[ArrayFloat] = None


@dataclass(frozen=True)
class AccessWindow:
    """
    One contiguous interval during which a body was above a station's mask angle.

    `rise_s` and `set_s` are **interpolated between samples**, not snapped to the grid - see
    `access_windows` for the interpolation and its error order. `peak_elevation_rad` is the largest
    *sampled* elevation inside the window and is therefore a lower bound, with **no** refinement
    applied - unlike the edges. It is short of the true maximum by `O(h^2)` for a general pass, but
    only `O(h)` for one that passes through the zenith, where the elevation has a *corner* rather
    than a smooth maximum (the azimuth flips by 180 deg there). An exactly overhead 550 km pass
    sampled at `h = 30 s` peaks at 82 deg, not 90: read this field as "at least this high".

    `rise_clipped` (`set_clipped`) marks a window already open at the first sample (still open at the
    last). Its edge is the grid endpoint, its duration is a lower bound, and an access metric that
    averages durations must either drop or flag it - which is why the flag exists rather than the
    window being silently truncated or silently discarded.

    `peak_time_s` is the **sample time** at which `peak_elevation_rad` was attained, carrying the
    same "sampled, not refined" caveat: it is one of the grid times, within `h/2` of the true
    maximum for a smooth pass. It exists so a consumer can evaluate another series - range and range
    rate, in `access.ContactWindow` - at the same instant the peak elevation belongs to, rather than
    re-deriving the argmax and risking a different tie-break. It defaults to `nan` so a hand-built
    record that only cares about the interval can omit it.
    """
    station_index: int
    body_index: int
    rise_s: float
    set_s: float
    duration_s: float
    peak_elevation_rad: float
    rise_clipped: bool
    set_clipped: bool
    peak_time_s: float = math.nan


# --------------------------------------------------------------------------------------------------
# Station geometry
# --------------------------------------------------------------------------------------------------

def station_position_fixed(
    latitude_rad: Numeric,
    longitude_rad: Numeric,
    altitude_km: Numeric = 0.0,
    *,
    body_radius_km: ScalarFloat,
) -> ArrayFloat:
    """
    Body-fixed Cartesian position of one or more stations, shape `(n_stations, 3)`, in km.

    Spherical (geocentric) latitude, east longitude, altitude above a sphere of `body_radius_km` -
    see the module docstring on why there is no ellipsoid here. The three arguments broadcast against
    each other, so a row of stations at one altitude is `station_position_fixed(lats, lons, 0.0, ...)`.

    This is `frames.longlat_to_fixed` on `[r, lon, lat]`, i.e. exactly the inverse of the
    `fixed_to_longlat` that `viz.ground_track` reports a sub-satellite point with; going through the
    same pair keeps a station and a ground track in provably the same convention.
    """
    lat = np.asarray(latitude_rad, dtype=np.float64)
    lon = np.asarray(longitude_rad, dtype=np.float64)
    alt = np.asarray(altitude_km, dtype=np.float64)
    if lat.ndim > 1 or lon.ndim > 1 or alt.ndim > 1:
        raise ValueError(
            f"station coordinates must be scalars or 1-D, got shapes "
            f"{lat.shape}, {lon.shape}, {alt.shape}."
        )
    lat_b, lon_b, alt_b = np.broadcast_arrays(
        np.atleast_1d(lat), np.atleast_1d(lon), np.atleast_1d(alt)
    )
    radius = float(body_radius_km) + alt_b
    fixed: ArrayFloat = ReferenceFrames.longlat_to_fixed(
        np.stack([radius, lon_b, lat_b], axis=-1)
    )
    return fixed


def _sez_rotations(latitude_rad: ArrayFloat, longitude_rad: ArrayFloat) -> ArrayFloat:
    """
    The body-fixed -> SEZ rotation for each station, shape `(n_stations, 3, 3)`.

    Vallado's form is `ROT2(90 deg - phi) ROT3(lambda)`, where `ROTn` are *frame* (passive) rotations.
    `Transformations.Ry` and `Rz` are **active**, so `ROT3(lambda) = Rz(-lambda)` and
    `ROT2(90 deg - phi) = Ry(phi - 90 deg)`, and the product is built from those. Written out, the
    rows are

        S = ( sin phi cos lam,  sin phi sin lam, -cos phi)     south
        E = (        -sin lam,          cos lam,       0.0)    east
        Z = ( cos phi cos lam,  cos phi sin lam,  sin phi)     zenith (the station's own unit vector)

    a right-handed triad in the order (S, E, Z): `S x E = Z`. The `(phi, lam) = (0, 0)` case is worth
    checking by hand - zenith is `+x`, east is `+y`, south is `-z`.
    """
    product: ArrayFloat = np.asarray(
        Transformations.Ry(latitude_rad - 0.5 * math.pi) @ Transformations.Rz(-longitude_rad),
        dtype=np.float64,
    )
    return product


# --------------------------------------------------------------------------------------------------
# Look angles
# --------------------------------------------------------------------------------------------------

def elevation_azimuth(
    positions_km: ArrayFloat,
    times_s: ArraySeconds,
    *,
    velocities_km_s: Optional[ArrayFloat] = None,
    latitude_rad: Numeric,
    longitude_rad: Numeric,
    altitude_km: Numeric = 0.0,
    omega: ScalarFloat,
    body_radius_km: ScalarFloat,
    theta0: ScalarFloat = 0.0,
    epoch_s: ScalarFloat = 0.0,
) -> Topocentric:
    """
    Elevation, azimuth, range and - given velocities - **range rate** of each body from each station
    over a time grid.

    `positions_km` is `(n_times, n_bodies, 3)` (a `(n_times, 3)` single-body array is accepted and
    given one body column), **central-body-relative inertial** positions - i.e. `viz.sample_states`
    with `relative_to` set to the central body, the same input `viz.ground_track` takes. `times_s` is
    `(n_times,)`. The station arguments broadcast to `(n_stations,)`; see `station_position_fixed`
    for their meaning and the module docstring for `omega` / `theta0` / `epoch_s` - in particular for
    why `theta0` is referenced to `epoch_s` (default 0.0, absolute simulation time) and not to
    `times_s[0]` the way `viz.ground_track` references it.

    Method, in three steps:

    1. Rotate the bodies into the body-fixed frame with
       `ReferenceFrames.inertia_to_fixed(r, 0, -theta)`. The station is then *stationary* in that
       frame and is built once, which is why the rotation is applied to the satellites rather than to
       the stations. See the module docstring for the sign.
    2. Form the topocentric range vector `rho = r_body_fixed - r_station`, shape
       `(n_times, n_stations, n_bodies, 3)`.
    3. Rotate `rho` into SEZ by `_sez_rotations` and read off

           range     = |rho|
           elevation = atan2(rho_Z, hypot(rho_S, rho_E))
           azimuth   = atan2(rho_E, -rho_S)    wrapped to [0, 2 pi)

       `atan2` rather than `asin(rho_Z / |rho|)`: the two agree analytically, but `asin` loses
       relative precision exactly where the elevation is highest, which is where a pass metric cares
       most, and it needs a clip to survive `rho_Z / |rho|` rounding past 1 at the zenith.

    **Range rate** is computed only when `velocities_km_s` is given - `(n_times, n_bodies, 3)` or
    `(n_times, 3)`, the **same frame as `positions_km`**, i.e. central-body-relative *inertial*
    velocities, which is columns 3:6 of `viz.sample_states(..., relative_to=central)`. It is then
    `Topocentric.range_rate_km_s`, **positive = opening**, and `None` otherwise, so every existing
    caller is unchanged.

    Why this extends `elevation_azimuth` rather than living in its own function: range rate is the
    derivative of the `rho` this function already builds, through the same station, the same
    `-theta` rotation and the same sign convention - a second entry point would have to reproduce
    all three, and the module docstring's whole complaint about that rotation is how quietly it goes
    wrong when duplicated.

    Step 4, then, given velocities: rotate them with the same `-theta` (a pure rotation `A v`, *not*
    a body-fixed velocity - no transport term is applied to the body), subtract the station's
    rotated inertial velocity `omega x r_station_fixed`, and project:

        rho_dot = A v_body - omega x r_station_fixed
        d|rho|/dt = (rho . rho_dot) / |rho|

    The station term is one constant vector per station, `omega * (-y_fixed, +x_fixed, 0)`, because
    the rotation is about `+z` (the same assumption `j2` and `drag` make). The module docstring
    derives why projecting `rho_dot` in this "rotate the inertial derivative" form gives the same
    scalar as differentiating in the rotating frame, and what the term is worth: 0.4646 km/s of
    6.521 km/s at the horizon of a 550 km equatorial pass.

    A body exactly at the station gives `|rho| = 0`; the numerator is then 0 too, and the quotient
    is defined to 0.0 rather than `nan`, matching the zenith-azimuth convention above.
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[:, np.newaxis, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions_km must have shape (n_times, n_bodies, 3), got {pos.shape}.")
    times = np.asarray(times_s, dtype=np.float64)
    if times.shape != (pos.shape[0],):
        raise ValueError(
            f"times_s has shape {times.shape}, incompatible with {pos.shape[0]} position samples."
        )

    lat, lon = np.broadcast_arrays(
        np.atleast_1d(np.asarray(latitude_rad, dtype=np.float64)),
        np.atleast_1d(np.asarray(longitude_rad, dtype=np.float64)),
    )
    station_fixed = station_position_fixed(lat, lon, altitude_km, body_radius_km=body_radius_km)
    rotations = _sez_rotations(lat, lon)

    # One rotation angle per sample. `Rz` builds an `(n_times, 1, 3, 3)` tensor from an
    # `(n_times, 1)` angle array, which broadcasts across the body axis of the
    # `(n_times, n_bodies, 3, 1)` column vectors - so the matrices are built once per sample, not
    # once per sample per body.
    theta = (float(theta0) + float(omega) * (times - float(epoch_s)))[:, np.newaxis]
    r_fixed, _ = ReferenceFrames.inertia_to_fixed(
        pos[..., np.newaxis], np.zeros_like(pos)[..., np.newaxis], -theta
    )

    # (n_times, 1, n_bodies, 3) - (1, n_stations, 1, 3) -> (n_times, n_stations, n_bodies, 3).
    rho = r_fixed[..., 0][:, np.newaxis, :, :] - station_fixed[np.newaxis, :, np.newaxis, :]
    rho_sez = np.einsum("sij,tsbj->tsbi", rotations, rho)

    south = rho_sez[..., 0]
    east = rho_sez[..., 1]
    up = rho_sez[..., 2]
    elevation: ArrayFloat = np.arctan2(up, np.hypot(south, east))
    azimuth: ArrayFloat = np.arctan2(east, -south) % (2.0 * math.pi)
    ranges: ArrayFloat = np.linalg.norm(rho, axis=-1)

    range_rate: Optional[ArrayFloat] = None
    if velocities_km_s is not None:
        vel = np.asarray(velocities_km_s, dtype=np.float64)
        if vel.ndim == 2:
            vel = vel[:, np.newaxis, :]
        if vel.shape != pos.shape:
            raise ValueError(
                f"velocities_km_s has shape {vel.shape}, incompatible with positions {pos.shape}."
            )
        # `A v`, a pure rotation of the inertial velocity - deliberately not the body-fixed
        # velocity, which would also carry `-omega x r_body`. See the docstring: the two differ by
        # `omega x rho`, which is orthogonal to `rho` and so drops out of the projection anyway.
        v_rot, _ = ReferenceFrames.inertia_to_fixed(
            vel[..., np.newaxis], np.zeros_like(vel)[..., np.newaxis], -theta
        )
        # omega x r_station, expressed in the body-fixed frame: rotation is about +z.
        station_vel = float(omega) * np.stack(
            [-station_fixed[:, 1], station_fixed[:, 0],
             np.zeros(station_fixed.shape[0], dtype=np.float64)],
            axis=-1,
        )
        rho_dot = v_rot[..., 0][:, np.newaxis, :, :] - station_vel[np.newaxis, :, np.newaxis, :]
        # |rho| = 0 forces the numerator to 0 as well, so dividing by 1.0 there returns exactly 0.0.
        rate: ArrayFloat = (
            np.sum(rho * rho_dot, axis=-1) / np.where(ranges > 0.0, ranges, 1.0)
        )
        range_rate = rate

    return Topocentric(
        times_s=times, elevation_rad=elevation, azimuth_rad=azimuth, range_km=ranges,
        range_rate_km_s=range_rate,
    )


# --------------------------------------------------------------------------------------------------
# Access windows
# --------------------------------------------------------------------------------------------------

def access_windows(
    times_s: ArraySeconds,
    elevation_rad: ArrayFloat,
    *,
    mask_angle_rad: ScalarFloat = 0.0,
) -> List[AccessWindow]:
    """
    Rise/set intervals where elevation exceeds `mask_angle_rad`, with **interpolated** edges.

    `elevation_rad` is `(n_times, n_stations, n_bodies)` as `Topocentric` reports it; a `(n_times,)`
    or `(n_times, n_bodies)` array is promoted with a single station axis. `times_s` must be strictly
    increasing. Windows come back ordered by station, then body, then time.

    **Why the edges are interpolated.** Snapping a crossing to the nearest sample quantises the
    reported duration by up to one sample interval, which for a 60 s grid and an 800 s LEO pass is a
    7 % error on precisely the quantity this module exists to report - and it is a *biased* error,
    not noise. So the crossing is found by linear inverse interpolation on `e(t) - mask` between the
    bracketing samples:

        t_cross = t_i + (t_i+1 - t_i) * (mask - e_i) / (e_i+1 - e_i)

    **Error order: second, `O(h^2)`.** Expanding `f = e - mask` about its root `t*` and solving the
    secant through the bracketing samples for its zero gives

        t_hat - t* = -(f'' / (2 f')) (t_i - t*) (t_i+1 - t*) + O(h^3)
                   =  (f'' / (2 f')) a b,        a = t* - t_i >= 0,  b = t_i+1 - t* >= 0

    so the bound is `|f''| h^2 / (8 |f'|)`, attained at `a = b = h/2`. Two consequences the tests
    assert rather than assume. First, it is a *bias*, not a scatter. The elevation curve is **convex
    at the horizon** - it leaves the horizon at rate `Omega` and reaches far more than `Omega * T/2`
    by mid-pass - so `f'' > 0` at both crossings, while `f' > 0` at a rise and `f' < 0` at a set.
    Rises therefore come out **early**, sets **late**, and **durations are systematically
    over-estimated**. (The intuitive "concave, so durations are short" reading is wrong, and the
    tests assert the sign for that reason.) Second, halving the grid quarters the error.

    For a circular 550 km pass over an equatorial station the coefficient is `Omega cot(lambda_0)/2`
    with `Omega = n - omega` the station-relative angular rate and `lambda_0 = acos(R/r)` the horizon
    half-angle: 1.21e-3 per second, so 0.23 s of edge error at `h = 30 s` against a 784 s window.
    Two orders below the sample interval, which is why a coarse sweep grid can still report a
    defensible duration.

    A quadratic (three-point) edge would be `O(h^3)` and is deliberately not built: it needs a third
    sample that may not exist at a clipped edge, and it can place the root outside its bracket.

    The time axis is vectorised; the `(station, body)` pairs are walked in a Python loop, because the
    output is a variable-length list of records per pair and each pair's run-length decomposition is
    independent. That is post-processing over a handful of columns, not an arena-sized inner loop.
    """
    times = np.asarray(times_s, dtype=np.float64)
    elevation = np.asarray(elevation_rad, dtype=np.float64)
    if elevation.ndim == 1:
        elevation = elevation[:, np.newaxis, np.newaxis]
    elif elevation.ndim == 2:
        elevation = elevation[:, np.newaxis, :]
    if elevation.ndim != 3:
        raise ValueError(
            f"elevation_rad must have shape (n_times, n_stations, n_bodies), got {elevation.shape}."
        )
    if times.shape != (elevation.shape[0],):
        raise ValueError(
            f"times_s has shape {times.shape}, incompatible with {elevation.shape[0]} samples."
        )
    if times.size < 2:
        raise ValueError("access_windows needs at least two samples to bracket a crossing.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing.")

    mask = float(mask_angle_rad)
    last = times.size - 1
    windows: List[AccessWindow] = []

    for s in range(elevation.shape[1]):
        for b in range(elevation.shape[2]):
            excess = elevation[:, s, b] - mask
            above = np.flatnonzero(excess > 0.0)
            if above.size == 0:
                continue
            # Contiguous runs of in-view samples: a gap of more than one index ends a window.
            breaks = np.flatnonzero(np.diff(above) > 1)
            starts = np.concatenate((above[:1], above[breaks + 1]))
            ends = np.concatenate((above[breaks], above[-1:]))

            for i0, i1 in zip(starts.tolist(), ends.tolist()):
                rise_clipped = i0 == 0
                set_clipped = i1 == last
                rise = float(times[0]) if rise_clipped else _crossing(times, excess, i0 - 1)
                set_t = float(times[last]) if set_clipped else _crossing(times, excess, i1)
                # One argmax serves both the peak elevation and the instant it belongs to, so the
                # two can never disagree about which sample the peak is.
                peak = i0 + int(np.argmax(excess[i0:i1 + 1]))
                windows.append(AccessWindow(
                    station_index=s,
                    body_index=b,
                    rise_s=rise,
                    set_s=set_t,
                    duration_s=set_t - rise,
                    peak_elevation_rad=float(excess[peak]) + mask,
                    rise_clipped=rise_clipped,
                    set_clipped=set_clipped,
                    peak_time_s=float(times[peak]),
                ))
    return windows


def _crossing(times: ArrayFloat, excess: ArrayFloat, i: int) -> float:
    """
    Linear inverse interpolation of the zero of `excess` between samples `i` and `i + 1`.

    The caller only ever passes a bracketing pair, one non-positive and one strictly positive, so the
    denominator is strictly positive and the result lies inside the bracket. A sample sitting exactly
    on the mask returns that sample's own time.
    """
    f0 = float(excess[i])
    f1 = float(excess[i + 1])
    t0 = float(times[i])
    return t0 + (float(times[i + 1]) - t0) * (-f0) / (f1 - f0)


# --------------------------------------------------------------------------------------------------
# Line of sight
# --------------------------------------------------------------------------------------------------

def line_of_sight(
    r1_km: ArrayFloat,
    r2_km: ArrayFloat,
    *,
    body_radius_km: ScalarFloat,
) -> NDArray[np.bool_]:
    """
    Whether two positions can see each other past a sphere of `body_radius_km` at the origin.

    Both arrays are `(..., 3)`, measured from the **occulting body's centre**, and broadcast against
    each other; the result has the broadcast shape with the trailing axis removed. This is the
    inter-satellite link test, and the same primitive an eclipse/shadow model needs - there the two
    endpoints are the satellite and the Sun.

    Vallado Alg. 35 (`SIGHT`), spherical form. The segment is `p(tau) = r1 + tau (r2 - r1)`; its
    closest approach to the centre is at

        tau* = -(r1 . (r2 - r1)) / |r2 - r1|^2

    **clamped to `[0, 1]`**. The clamp is what makes the algebra a *segment* test rather than an
    infinite-line test: with both endpoints on the same side of the body the minimiser falls outside
    the segment, and an unclamped test would report a spurious occultation for two satellites a few
    degrees apart on the near side. Visible iff `|p(tau*)| >= body_radius_km`, so a ray exactly
    tangent to the surface counts as visible; refraction and any atmosphere are not modelled, and a
    real link budget wants a positive grazing altitude added to the radius.

    Consequence worth stating, because it is the test case: two satellites on a circular orbit of
    radius `r` separated by central angle `phi` have a chord whose closest approach is `r cos(phi/2)`,
    so they are blocked exactly when `phi > 2 acos(body_radius_km / r)`.

    Comparison is on squared magnitudes - no `sqrt`, and no change to the decision, since both sides
    are non-negative. Coincident endpoints give `tau* = 0`, hence the visibility of `r1` itself.
    """
    r1 = np.asarray(r1_km, dtype=np.float64)
    r2 = np.asarray(r2_km, dtype=np.float64)
    if r1.shape[-1] != 3 or r2.shape[-1] != 3:
        raise ValueError(f"positions must have a trailing axis of 3, got {r1.shape}, {r2.shape}.")

    delta = r2 - r1
    denom = np.sum(delta * delta, axis=-1)
    # Coincident endpoints: the segment is a point, so evaluate at tau = 0 rather than dividing by 0.
    safe = np.where(denom > 0.0, denom, 1.0)
    tau = np.clip(np.where(denom > 0.0, -np.sum(r1 * delta, axis=-1) / safe, 0.0), 0.0, 1.0)

    closest = r1 + tau[..., np.newaxis] * delta
    visible: NDArray[np.bool_] = np.sum(closest * closest, axis=-1) >= float(body_radius_km) ** 2
    return visible
