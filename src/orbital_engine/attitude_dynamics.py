"""
Attitude dynamics: vessels that *track* their pointing law instead of holding it exactly, so a cone or a
body-fixed thruster can lag where it should point.

**Why.** `attitude.py` treats attitude as a law a vessel obeys exactly. A real vessel has inertia and
limited torque: it slews to a new pointing over seconds to minutes, overshoots, and tracks a rotating
target with an error its controller decides. For visibility that is a time shift and a gap; for an
impulse it is a direction error. This module gives the comparison engine the second rung -
"ideal pointing" against "rigid body under a controller" - as data.

Model
-----
Per vessel, a quaternion `q = [w, x, y, z]` (body -> inertial; body axes are the principal axes) and a
body angular velocity `omega`, rad/s:

    q_dot     = 1/2 q (x) [0, omega]                                   (kinematics)
    I omega_dot = tau - omega x (I omega)                              (Euler's equations, principal axes)

Hughes, *Spacecraft Attitude Dynamics* (1986), and Wie, *Space Vehicle Dynamics and Control* (2008),
ch. 7 - **chapter references from memory, unverified**; the tests check conservation laws and a
closed-form precession instead. No external torques (gravity gradient, drag, SRP): the controller's
torque is the only one.

**Control.** A quaternion PD law on the error to the target attitude (from the vessel's pointing law,
`attitude.attitude_matrix`):

    q_e   = q_target* (x) q                    (rotation from target to current, body frame)
    tau   = -k_p sign(q_e,w) q_e,vec - k_d (omega - omega_target)
    tau   = clip(tau, -tau_max, tau_max)       per axis

with the target's own rate `omega_target` fed forward (`PointingController.feed_forward`, default on).
Without it, the derivative acts on the measured rate: at steady state the body turns at the target's
rate `n` with no net torque, so `k_p |q_e,vec| = k_d n`, and since `q_e,vec ~ angle / 2` the target is
tracked with a steady error of `2 k_d n / k_p` rad. With it, the loop - plant `1/(I s^2)`,
controller `k_p + k_d s` on the error - is type 2, and a target at constant rate is tracked with zero
steady-state error.
For small angles each axis is a second-order system: `omega_n = sqrt(k_p / (2 I))`,
`zeta = k_d / (2 sqrt(k_p I / 2))`. The `/2` is because `q_e,vec ~ angle / 2`.

**Stepping.** Attitude never feeds back into the orbits here, so it is advanced once per real
`Simulation.step` (`AttitudeTracker.advance`, called by the simulation after its orbit propagation, not
during event trial propagations). Within the step it takes RK4 sub-steps of at most `max_substep_s`,
with the target slerped between its start-of-step and end-of-step values and `omega_target` constant
across the step. The pointing law's target at each end is exact; between them the slerp is the
interpolation error (the target turns at most `n dt` in a step).

Body-fixed thrust
-----------------
`AttitudeTracker.steer_thrust(sim, axis_body)` fixes the tracked vessels' `"thrust"` direction to a body
axis. At the start of every real step the simulation writes `R(q) axis_body`, re-expressed in each
vessel's RSW frame about its Keplerian parent, into the thrust model's direction columns, scaled by
the norm the columns already held: **the configured direction's norm stays the throttle, the attitude
supplies the direction**, so a coast (`(0, 0, 0)`) stays a coast and an event that sets a throttle is
not overwritten.

The direction is then held fixed *in RSW* for the step, exactly as the thrust model holds any
direction, so a vessel that tracks an RSW-fixed law (`velocity` on a circular orbit) reproduces the
ideal-direction burn. It is the start-of-step attitude, because the attitude is advanced after the
orbits (it needs their end-of-step state for its target): the direction lags by half a step on
average, which is **first order in `dt`** while the body turns relative to RSW - during a slew, an
error of ~`omega_slew dt / 2` rad in the direction. Like the frozen mass, halve `dt` to halve it.
`body_dv_rsw` gives the same re-expression for an impulse along a body axis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .attitude import Attitude, attitude_matrix
from .frames import ReferenceFrames

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = ["RigidBody", "PointingController", "AttitudeTracker", "quat_from_matrix", "matrix_from_quat",
           "quat_mul", "quat_conj", "slerp"]

Q = NDArray[np.float64]


def quat_mul(a: Q, b: Q) -> Q:
    """Hamilton product, scalar first, broadcasting over leading axes."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    out: Q = np.stack([aw * bw - ax * bx - ay * by - az * bz,
                       aw * bx + ax * bw + ay * bz - az * by,
                       aw * by - ax * bz + ay * bw + az * bx,
                       aw * bz + ax * by - ay * bx + az * bw], axis=-1)
    return out


def quat_conj(q: Q) -> Q:
    out: Q = q * np.array([1.0, -1.0, -1.0, -1.0])
    return out


def matrix_from_quat(q: Q) -> NDArray[np.float64]:
    """Rotation matrices `(..., 3, 3)` (body -> inertial) of unit quaternions `(..., 4)`."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    out: NDArray[np.float64] = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)], axis=-2)
    return out


def quat_from_matrix(m: NDArray[np.float64]) -> Q:
    """A unit quaternion (scalar first, `w >= 0`) of one rotation matrix (Shepperd's method)."""
    tr = float(np.trace(m))
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + m[i, i] - m[j, j] - m[k, k], 0.0))
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = q / np.linalg.norm(q)
    out: Q = -q if q[0] < 0.0 else q
    return out


def slerp(q0: Q, q1: Q, f: float) -> Q:
    """Spherical interpolation between unit quaternions `(N, 4)` at fraction `f` (shortest path)."""
    dot = np.einsum("ij,ij->i", q0, q1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    small = theta < 1e-9
    s = np.where(small, 1.0, np.sin(theta))
    a = np.where(small, 1.0 - f, np.sin((1.0 - f) * theta) / s)
    b = np.where(small, f, np.sin(f * theta) / s)
    out: Q = a[:, None] * q0 + b[:, None] * q1
    out = out / np.linalg.norm(out, axis=1)[:, None]
    return out


@dataclass(frozen=True)
class RigidBody:
    """Principal moments of inertia, kg m^2 (body axes are principal)."""

    inertia: Tuple[float, float, float]

    def __post_init__(self) -> None:
        a, b, c = self.inertia
        if min(a, b, c) <= 0.0 or a + b < c or b + c < a or a + c < b:
            raise ValueError(f"principal moments {self.inertia} must be positive and satisfy the triangle "
                             f"inequality (a physical body)")


@dataclass(frozen=True)
class PointingController:
    """Quaternion PD gains (`k_p` N m, `k_d` N m s) and a per-axis torque limit, N m."""

    kp: float
    kd: float
    max_torque_nm: float = math.inf
    feed_forward: bool = True

    def __post_init__(self) -> None:
        if self.kp < 0.0 or self.kd < 0.0 or not self.max_torque_nm > 0.0:
            raise ValueError("PD gains must be non-negative and the torque limit positive")


class AttitudeTracker:
    """
    Rigid-body attitude for `vessels` (names) of a `Simulation`, each driven toward its pointing `law`
    by `controller`. Attach with `Simulation.attach_attitude_tracker(tracker)`; each real step then
    advances it (see the module docstring). Starts aligned with the law and **at rest** unless `q0` /
    `omega0` are given, so a law that turns (nadir, velocity: at the orbit rate `n`) begins with a
    transient of ~`n / omega_n` rad; pass the law's rate as `omega0` to start on it. `record=True` keeps every step's boresights for the link scan
    (`isl_scale.link_contact_table(..., boresights=...)`).
    """

    def __init__(self, sim: "Simulation", vessels: Sequence[str], law: Attitude, body: RigidBody,
                 controller: Optional[PointingController], *, q0: Optional[Q] = None,
                 omega0: Optional[NDArray[np.float64]] = None, max_substep_s: float = 1.0,
                 record: bool = False) -> None:
        missing = [v for v in vessels if v not in sim.name_to_index]
        if missing:
            raise KeyError(f"no vessels named {missing}")
        if law.reference is not None and law.reference not in sim.name_to_index:
            raise KeyError(f"the pointing law's reference {law.reference!r} is not in the simulation")
        self.names = list(vessels)
        self.slots = np.asarray([sim.name_to_index[v] for v in vessels], dtype=np.int64)
        self.law = law
        self.inertia = np.asarray(body.inertia, dtype=np.float64)
        self.controller = controller
        self.max_substep_s = float(max_substep_s)
        self._target = self._targets(sim)
        self.q: Q = self._target.copy() if q0 is None else np.asarray(q0, dtype=np.float64).copy()
        self.q /= np.linalg.norm(self.q, axis=1)[:, None]
        self.omega: NDArray[np.float64] = (np.zeros((len(vessels), 3)) if omega0 is None
                                           else np.asarray(omega0, dtype=np.float64).copy())
        self.record = record
        self.thrust_axis: Optional[NDArray[np.float64]] = None
        self.times: List[float] = []
        self.history: List[NDArray[np.float64]] = []
        if record:
            self._record(float(sim.t))

    # -- the target ---------------------------------------------------------------------------------
    def _targets(self, sim: "Simulation") -> Q:
        ref = sim.name_to_index.get(self.law.reference) if self.law.reference is not None else None
        out = np.empty((self.slots.size, 4))
        for j, k in enumerate(self.slots.tolist()):
            g = sim.global_states[k]
            rp = None if ref is None else sim.global_states[ref, :3]
            rv = None if ref is None else sim.global_states[ref, 3:]
            out[j] = quat_from_matrix(attitude_matrix(self.law, g[:3], g[3:], rp, rv))
        return out

    # -- the dynamics --------------------------------------------------------------------------------
    def _derivative(self, q: Q, w: NDArray[np.float64], q_t: Q, w_t: NDArray[np.float64]) -> Tuple[Q, NDArray[np.float64]]:
        q_dot: Q = 0.5 * quat_mul(q, np.concatenate([np.zeros((q.shape[0], 1)), w], axis=1))
        tau = np.zeros_like(w)
        if self.controller is not None:
            q_e = quat_mul(quat_conj(q_t), q)
            sign = np.where(q_e[:, 0] < 0.0, -1.0, 1.0)[:, None]
            # The target's rate is in the target's body frame; in the vessel's, it is rotated by q_e*.
            w_t_body = np.einsum("nij,nj->ni", np.transpose(matrix_from_quat(q_e), (0, 2, 1)), w_t)
            w_ref = w_t_body if self.controller.feed_forward else 0.0
            tau = -self.controller.kp * sign * q_e[:, 1:] - self.controller.kd * (w - w_ref)
            tau = np.clip(tau, -self.controller.max_torque_nm, self.controller.max_torque_nm)
        i = self.inertia
        w_dot = (tau - np.cross(w, w * i)) / i
        return q_dot, w_dot

    def advance(self, sim: "Simulation", dt: float) -> None:
        """Advance from the previous call's target (the start of this step) to `sim`'s current state
        (its end), over `dt`. Called by `Simulation.step` after it has propagated the orbits."""
        q_t0 = self._target
        q_t1 = self._targets(sim)
        # The target's rate, constant over the step, in the target's body frame: the rotation from the
        # start target to the end target is q_t0* (x) q_t1, of angle theta about axis u.
        rel = quat_mul(quat_conj(q_t0), np.where((np.einsum("ij,ij->i", q_t0, q_t1) < 0.0)[:, None], -q_t1, q_t1))
        vec_norm = np.linalg.norm(rel[:, 1:], axis=1)
        angle = 2.0 * np.arctan2(vec_norm, rel[:, 0])
        axis = rel[:, 1:] / np.where(vec_norm > 0.0, vec_norm, 1.0)[:, None]
        w_t = axis * (angle / dt)[:, None] if dt > 0.0 else np.zeros_like(axis)
        n = max(1, int(math.ceil(dt / self.max_substep_s - 1e-12)))
        h = dt / n
        q, w = self.q, self.omega
        for k in range(n):
            f0, fm, f1 = k / n, (k + 0.5) / n, (k + 1.0) / n
            t0, tm, t1 = slerp(q_t0, q_t1, f0), slerp(q_t0, q_t1, fm), slerp(q_t0, q_t1, f1)
            k1q, k1w = self._derivative(q, w, t0, w_t)
            k2q, k2w = self._derivative(q + 0.5 * h * k1q, w + 0.5 * h * k1w, tm, w_t)
            k3q, k3w = self._derivative(q + 0.5 * h * k2q, w + 0.5 * h * k2w, tm, w_t)
            k4q, k4w = self._derivative(q + h * k3q, w + h * k3w, t1, w_t)
            q = q + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
            w = w + h / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
            q = q / np.linalg.norm(q, axis=1)[:, None]
        self.q, self.omega, self._target = q, w, q_t1
        if self.record:
            self._record(float(sim.t))

    # -- body-fixed thrust and impulses --------------------------------------------------------------
    def steer_thrust(self, sim: "Simulation", axis_body: Sequence[float] = (0.0, 0.0, 1.0)) -> None:
        """Point the tracked vessels' `"thrust"` force model along the body axis `axis_body` (normalised)
        from now on; see "Body-fixed thrust" in the module docstring. Every vessel must already carry
        `"thrust"`, and the tracker must be attached (`Simulation.attach_attitude_tracker`)."""
        from .registry import get_force_model
        from .thrust import THRUST_MODEL
        axis = np.asarray(axis_body, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or not norm > 0.0:
            raise ValueError(f"thrust axis {axis_body!r} must be a non-zero 3-vector")
        bit = np.uint64(1) << np.uint64(get_force_model(THRUST_MODEL).bit)
        bare = [n for n, k in zip(self.names, self.slots.tolist()) if not sim.force_model_mask[k] & bit]
        if bare:
            raise ValueError(f"vessels {bare} do not carry the {THRUST_MODEL!r} force model; enable it "
                             f"first (its direction norm is the throttle)")
        if self not in sim._attitude_trackers:
            raise ValueError("attach the tracker (Simulation.attach_attitude_tracker) before steering thrust")
        self.thrust_axis = axis / norm

    def _rsw(self, sim: "Simulation") -> NDArray[np.float64]:
        parents = sim.parent_indices[self.slots]
        rel = sim.global_states[self.slots] - sim.global_states[parents]
        basis, ok = ReferenceFrames.RSW_basis(rel[:, :3], rel[:, 3:])
        if not bool(np.all(ok)):
            raise ValueError("a tracked vessel's RSW frame about its parent is undefined")
        out: NDArray[np.float64] = np.asarray(basis, dtype=np.float64).reshape(-1, 3, 3)
        return out

    def body_dv_rsw(self, sim: "Simulation", dv_body: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """An impulse fixed in the body frame (km/s) re-expressed, per tracked vessel, in its RSW axes about
        its Keplerian parent, at the vessel's *current* attitude - what `Simulation.apply_delta_v` takes."""
        inertial = matrix_from_quat(self.q) @ np.asarray(dv_body, dtype=np.float64)
        out: NDArray[np.float64] = np.einsum("nij,nj->ni", self._rsw(sim), inertial)
        return out

    def write_thrust_direction(self, sim: "Simulation") -> None:
        """Write the body-fixed thrust axis, in RSW and at the current throttle, into the thrust model's
        direction columns. Called by `Simulation.step` at the start of each real step."""
        if self.thrust_axis is None:
            return
        from .thrust import THRUST_MODEL, THRUST_PARAM_NAMES
        params = sim.force_model_params[THRUST_MODEL]
        c = THRUST_PARAM_NAMES.index("dir_r")
        throttle = np.linalg.norm(params[self.slots, c:c + 3], axis=1)
        params[self.slots, c:c + 3] = throttle[:, None] * self.body_dv_rsw(sim, self.thrust_axis)

    # -- what it decides ------------------------------------------------------------------------------
    def boresights(self) -> NDArray[np.float64]:
        """Inertial body `+z` of each vessel, `(N, 3)`."""
        out: NDArray[np.float64] = matrix_from_quat(self.q)[:, :, 2]
        return out

    def pointing_error_rad(self) -> NDArray[np.float64]:
        """Angle between each vessel's attitude and its law's target now, rad."""
        q_e = quat_mul(quat_conj(self._target), self.q)
        out: NDArray[np.float64] = 2.0 * np.arctan2(np.linalg.norm(q_e[:, 1:], axis=1), np.abs(q_e[:, 0]))
        return out

    def angular_momentum_inertial(self) -> NDArray[np.float64]:
        """`R(q) I omega`, `(N, 3)`: conserved when no torque acts."""
        out: NDArray[np.float64] = np.einsum("nij,nj->ni", matrix_from_quat(self.q), self.omega * self.inertia)
        return out

    def _record(self, t: float) -> None:
        self.times.append(t)
        self.history.append(self.boresights())
