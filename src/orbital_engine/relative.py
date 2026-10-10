"""
Vessel-relative frames and relative-motion targeting: one vessel seen from another.

**Why.** Impulses in this engine are given in a vessel's own RSW frame (`manoeuvres.py`), which is the
frame for orbit corrections of a single vessel. Rendezvous, formation keeping and pointing need the
*other* view: where the chaser is, and how it moves, in the **target's** local frame, and which burns
bring it there. That is the Hill (LVLH) frame of classical relative motion. Here it is the target's RSW
basis (`frames.ReferenceFrames.RSW_basis`: R radial out, S along-track, W along the angular momentum),
*rotating* with the target's orbital angular velocity.

Definitions
-----------
For target `(r_t, v_t)` and chaser `(r_c, v_c)` about the same primary, with `Q` the target's RSW
rotation (rows R, S, W) and `omega = |r_t x v_t| / |r_t|^2` along W (the instantaneous orbital rate):

    rho     = Q (r_c - r_t)
    rho_dot = Q (v_c - v_t) - omega_W x rho                (velocity *seen in the rotating frame*)

and `lvlh_to_inertial` inverts it exactly. An impulse leaves `rho` and the frame alone, so a change in
`rho_dot` is the chaser's inertial velocity change expressed in the target's RSW axes; `lvlh_dv_to_rsw`
re-expresses it in the chaser's own RSW axes, which is what `Simulation.apply_delta_v` takes.

**Clohessy-Wiltshire** (Hill-Clohessy-Wiltshire): for a circular target of mean motion `n` and
separations small against the orbit radius, the linearised relative motion has the closed-form
state-transition matrix `cw_stm(n, t)` (Clohessy & Wiltshire, *J. Aerospace Sciences* 27, 1960;
Curtis, *Orbital Mechanics for Engineering Students*, ch. 7 - **equation numbers from memory,
unverified**; the tests check the matrix against the engine's own two-body propagation instead):

    x(t)  = (4 - 3c) x0 + (s/n) xd0 + (2/n)(1 - c) yd0
    y(t)  = 6(s - nt) x0 + y0 + (2/n)(c - 1) xd0 + ((4s - 3nt)/n) yd0
    z(t)  = c z0 + (s/n) zd0
    xd(t) = 3n s x0 + c xd0 + 2s yd0
    yd(t) = 6n(c - 1) x0 - 2s xd0 + (4c - 3) yd0
    zd(t) = -n s z0 + c zd0                     (x radial, y along-track, z cross-track; c, s of n t)

`cw_rendezvous` is the two-impulse transfer from it: the first burn sets `rho_dot0+` so the chaser
reaches the target's origin after `tof`, `rho_dot0+ = -Phi_rv^{-1} Phi_rr rho0`, and the second cancels
the arrival velocity. Its error is the linearisation, which grows as `rho^2 / r` (and with any
eccentricity of the target); Lambert (`iod.lambert`) is the exact two-body alternative.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .frames import ReferenceFrames

__all__ = ["lvlh_state", "lvlh_to_inertial", "cw_stm", "cw_rendezvous", "lvlh_dv_to_rsw", "inertial_dv_to_rsw"]

Vec = NDArray[np.float64]


def _basis(r: Vec, v: Vec) -> Tuple[NDArray[np.float64], float]:
    q, ok = ReferenceFrames.RSW_basis(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64))
    if not bool(np.all(ok)):
        raise ValueError("the target's RSW frame is undefined (rectilinear or zero state)")
    h = float(np.linalg.norm(np.cross(r, v)))
    out: NDArray[np.float64] = np.asarray(q, dtype=np.float64).reshape(3, 3)
    return out, h / float(np.dot(r, r))


def lvlh_state(r_t: Vec, v_t: Vec, r_c: Vec, v_c: Vec) -> Tuple[Vec, Vec]:
    """`(rho, rho_dot)` of the chaser in the target's rotating RSW (Hill) frame, km and km/s."""
    q, omega = _basis(r_t, v_t)
    rho: Vec = q @ (np.asarray(r_c, dtype=np.float64) - r_t)
    w = np.array([0.0, 0.0, omega])
    rho_dot: Vec = q @ (np.asarray(v_c, dtype=np.float64) - v_t) - np.cross(w, rho)
    return rho, rho_dot


def lvlh_to_inertial(r_t: Vec, v_t: Vec, rho: Vec, rho_dot: Vec) -> Tuple[Vec, Vec]:
    """The chaser's inertial `(r_c, v_c)` from its state in the target's rotating RSW frame."""
    q, omega = _basis(r_t, v_t)
    w = np.array([0.0, 0.0, omega])
    r_c: Vec = np.asarray(r_t, dtype=np.float64) + q.T @ rho
    v_c: Vec = np.asarray(v_t, dtype=np.float64) + q.T @ (np.asarray(rho_dot, dtype=np.float64) + np.cross(w, rho))
    return r_c, v_c


def cw_stm(n: float, t: float) -> NDArray[np.float64]:
    """The 6x6 Clohessy-Wiltshire state-transition matrix (x radial, y along, z cross; module docstring)."""
    c, s, nt = math.cos(n * t), math.sin(n * t), n * t
    out: NDArray[np.float64] = np.array([
        [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
        [6 * (s - nt), 1, 0, 2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0],
        [0, 0, c, 0, 0, s / n],
        [3 * n * s, 0, 0, c, 2 * s, 0],
        [6 * n * (c - 1), 0, 0, -2 * s, 4 * c - 3, 0],
        [0, 0, -n * s, 0, 0, c],
    ], dtype=np.float64)
    return out


def cw_rendezvous(rho0: Vec, rho_dot0: Vec, n: float, tof: float) -> Tuple[Vec, Vec]:
    """
    The two Clohessy-Wiltshire impulses `(dv1, dv2)`, in the target's RSW axes (km/s), that take the
    chaser from `(rho0, rho_dot0)` to the target's origin after `tof` and stop it there. Singular when
    `Phi_rv(tof)` is (a whole number of orbits, among others): refused.
    """
    phi = cw_stm(n, tof)
    rr, rv, vr, vv = phi[:3, :3], phi[:3, 3:], phi[3:, :3], phi[3:, 3:]
    if abs(np.linalg.det(rv)) < 1e-12 * float(np.max(np.abs(rv))) ** 3:
        raise ValueError(f"CW rendezvous is singular for tof={tof} s (n tof = {n * tof:.3f} rad).")
    rho0 = np.asarray(rho0, dtype=np.float64)
    plus: Vec = np.asarray(np.linalg.solve(rv, -rr @ rho0), dtype=np.float64)
    dv1: Vec = plus - np.asarray(rho_dot0, dtype=np.float64)
    dv2: Vec = -(vr @ rho0 + vv @ plus)
    return dv1, dv2


def inertial_dv_to_rsw(r_c: Vec, v_c: Vec, dv_inertial: Vec) -> Vec:
    """An inertial velocity change in the chaser's own RSW axes (what `Simulation.apply_delta_v` takes)."""
    q, _ = _basis(r_c, v_c)
    out: Vec = q @ np.asarray(dv_inertial, dtype=np.float64)
    return out


def lvlh_dv_to_rsw(r_t: Vec, v_t: Vec, r_c: Vec, v_c: Vec, dv_target_rsw: Vec) -> Vec:
    """A velocity change given in the *target's* RSW axes (as `cw_rendezvous` returns it), re-expressed in
    the *chaser's* RSW axes for `Simulation.apply_delta_v`."""
    q_t, _ = _basis(r_t, v_t)
    return inertial_dv_to_rsw(r_c, v_c, q_t.T @ np.asarray(dv_target_rsw, dtype=np.float64))
