"""
Attitude as data, and what it decides: where a vessel's body axes point, impulses given in those axes,
and visibility cones (an antenna's or a sensor's field of view) for link visibility.

**Why.** A link or a burn often depends on how a vessel is oriented, not only where it is. A nadir
antenna sees the ground and not a relay overhead; a sensor pointed at the Moon cannot see a satellite
behind it; a thruster fixed in the body fires wherever the body points. Here an attitude is a small,
named rule - a configuration, like everything else in the engine - rather than an integrated rotational
state. Attitude *dynamics* (torques, rates, slews) are not modelled: a vessel is assumed to hold its
pointing law exactly.

**Body frame.** `+z` is the boresight; `+x` is a secondary direction made perpendicular to `+z`;
`+y = z x x`. The secondary is the vessel's velocity relative to the reference for `nadir`, `zenith`
and `target`; the direction *to* the reference for `velocity` (whose boresight already is the
velocity); and a fixed inertial axis for `inertial`, so an inertial law holds its roll too, not only
its boresight. Only if the secondary is parallel to `+z` does a fixed fallback axis stand in, and that
fallback switches axes discontinuously - a roll jump a tracking controller would chase. (Until
2026-10-10 the secondary was always the velocity: under `velocity` that is the boresight itself, so
every target came from the switching fallback and rolled abruptly twice an orbit, and under
`inertial` the roll followed the orbit.) Modes:

- `nadir` / `zenith`: boresight toward / away from the `reference` body's centre (an Earth-observing
  payload; an antenna looking up at relays).
- `target`: boresight toward the `reference` body (the Sun for a solar array, the Earth for a lunar
  relay's trunk antenna).
- `inertial`: boresight along the fixed `vector` (a star tracker, a deep-space antenna on a fixed line).
- `velocity`: boresight along the velocity relative to the `reference` body (a ram-facing instrument).

**Cones.** `Cone(attitude, half_angle_deg)` is a field of view about the boresight. The margin a link
window is cut on (`isl_scale.LinkSpec.cones`) is `range * sin(half_angle - off_axis_angle)`, clipped to a
quarter turn either way: zero exactly on the cone's edge, positive inside, continuous, and in km like
the occultation and range margins it is combined with by `min`, so window edges are interpolated on one
scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .frames import ReferenceFrames

__all__ = ["Attitude", "Cone", "ATTITUDE_MODES", "boresights", "attitude_matrix", "cone_margin_km", "body_dv_to_rsw"]

ATTITUDE_MODES = ("nadir", "zenith", "target", "inertial", "velocity")
Vec = NDArray[np.float64]


@dataclass(frozen=True)
class Attitude:
    """A pointing law (module docstring). `reference` names a body for every mode but `inertial`, which
    takes the fixed boresight `vector` instead."""

    mode: str
    reference: Optional[str] = None
    vector: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.mode not in ATTITUDE_MODES:
            raise ValueError(f"attitude mode {self.mode!r}: use one of {ATTITUDE_MODES}")
        if self.mode == "inertial":
            if self.vector is None or not np.linalg.norm(self.vector) > 0.0:
                raise ValueError("an inertial attitude needs a non-zero boresight vector")
        elif self.reference is None:
            raise ValueError(f"a {self.mode!r} attitude needs a reference body")


@dataclass(frozen=True)
class Cone:
    """A field of view of `half_angle_deg` (0, 180] about the boresight of `attitude`."""

    attitude: Attitude
    half_angle_deg: float

    def __post_init__(self) -> None:
        if not 0.0 < self.half_angle_deg <= 180.0:
            raise ValueError(f"cone half-angle {self.half_angle_deg!r} must be in (0, 180] degrees")


def boresights(attitude: Attitude, pos: Vec, vel: Vec, ref_pos: Optional[Vec] = None,
               ref_vel: Optional[Vec] = None) -> Vec:
    """Unit boresights `(N, 3)` for bodies at `pos`, `vel` `(N, 3)`; `ref_pos` / `ref_vel` `(3,)` are the
    reference body's state in the same frame (unused by `inertial`)."""
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    vel = np.atleast_2d(np.asarray(vel, dtype=np.float64))
    if attitude.mode == "inertial":
        b = np.broadcast_to(np.asarray(attitude.vector, dtype=np.float64), pos.shape).copy()
    else:
        assert ref_pos is not None
        to_ref = np.asarray(ref_pos, dtype=np.float64) - pos
        if attitude.mode in ("nadir", "target"):
            b = to_ref
        elif attitude.mode == "zenith":
            b = -to_ref
        else:                                                         # velocity, relative to the reference
            b = vel - (np.zeros(3) if ref_vel is None else np.asarray(ref_vel, dtype=np.float64))
    out: Vec = b / np.linalg.norm(b, axis=1)[:, None]
    return out


def attitude_matrix(attitude: Attitude, r: Vec, v: Vec, ref_pos: Optional[Vec] = None,
                    ref_vel: Optional[Vec] = None) -> NDArray[np.float64]:
    """The 3x3 rotation taking body components to inertial ones (columns are the body x, y, z axes) for
    one vessel at `r`, `v`."""
    z = boresights(attitude, r, v, ref_pos, ref_vel)[0]
    if attitude.mode == "velocity":
        assert ref_pos is not None
        second = np.asarray(ref_pos, dtype=np.float64) - np.asarray(r, dtype=np.float64)
    elif attitude.mode == "inertial":
        second = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        second = np.asarray(v, dtype=np.float64) - (np.zeros(3) if ref_vel is None else np.asarray(ref_vel, dtype=np.float64))
    x = second - (second @ z) * z
    if float(np.linalg.norm(x)) < 1e-12 * max(float(np.linalg.norm(second)), 1.0):
        trial = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = trial - (trial @ z) * z
    x = x / np.linalg.norm(x)
    out: NDArray[np.float64] = np.column_stack([x, np.cross(z, x), z])
    return out


def body_dv_to_rsw(attitude: Attitude, r: Vec, v: Vec, dv_body: Vec, r_parent: Vec, v_parent: Vec,
                   ref_pos: Optional[Vec] = None, ref_vel: Optional[Vec] = None) -> Vec:
    """
    An impulse fixed in the body frame (`dv_body`, km/s - a thruster's axis) re-expressed in the vessel's
    RSW axes about its Keplerian parent (`r_parent`, `v_parent`), which is what `Simulation.apply_delta_v`
    takes. All states in one inertial frame.
    """
    dv_inertial = attitude_matrix(attitude, r, v, ref_pos, ref_vel) @ np.asarray(dv_body, dtype=np.float64)
    q, ok = ReferenceFrames.RSW_basis(np.asarray(r, dtype=np.float64) - r_parent,
                                      np.asarray(v, dtype=np.float64) - v_parent)
    if not bool(np.all(ok)):
        raise ValueError("the vessel's RSW frame about its parent is undefined")
    out: Vec = np.asarray(q, dtype=np.float64).reshape(3, 3) @ dv_inertial
    return out


def cone_margin_km(bore: Vec, direction: Vec, ranges: Vec, half_angle_rad: Vec) -> Vec:
    """`range * sin(clip(half_angle - angle, -pi/2, pi/2))` per row: the km margin of `direction` (unit,
    `(P, 3)`) inside the cone of boresight `bore` `(P, 3)`. Positive inside, zero on the edge."""
    cosang = np.clip(np.einsum("ij,ij->i", bore, direction), -1.0, 1.0)
    angle = np.arccos(cosang)
    out: Vec = ranges * np.sin(np.clip(half_angle_rad - angle, -0.5 * math.pi, 0.5 * math.pi))
    return out
