"""
Fixed-step integration for Cowell propagation: classical RK4 (the default) and two symplectic methods, leapfrog and Yoshida's fourth-order composition (see
`_KickDriftKick`), selected by name (`make_integrator`, `Simulation.set_cowell_integrator`).

**What this is for.** The engine's only propagator until now is the analytic hierarchical Keplerian
one (`propagators.py` / `kernels.py`): closed-form advance of classical elements, with no accumulated
numerical state. Cowell propagation is the opposite approach - direct numerical integration of the
Cartesian equations of motion `d^2r/dt^2 = a(t, r, v)` - and this module is the integrator that drives
it. `a(t, r, v)` is whatever `forces.py`'s composition layer currently has enabled
(`forces.AccelerationProvider`); this module knows nothing about gravity, J2, or any other physics -
only how to advance a state given a black-box acceleration function.

**Why a class, not a bare function.** `Simulation.accelerations` returns `Simulation.accel_accum`
itself - a single shared buffer, reused on every call, not a copy (see its docstring). Classical RK4
needs *four* stage accelerations alive at once to form the weighted combination at the end of a step;
holding all four means each one must be copied out of the shared buffer before the next stage's call
overwrites it, and those four copies need somewhere to live between calls. An `Integrator` therefore
owns persistent scratch sized to `max_capacity`, allocated once at construction, exactly like
`Simulation`'s own `_kick` / `_accum` / `accel_accum` - never re-allocated inside `step`. A bare
function would either allocate that scratch fresh on every call (defeating the point) or require the
caller to pass it in on every call, which is what the `Integrator` object *is*, just written out.

**Why this is a Protocol.** `Integrator.step(provider, t, state, dt, indices, primaries)` is
deliberately the whole contract: advance `state[indices]` in place, from `t` by `dt`, using `provider`
for acceleration, relative to each body's own gravitating parent (`primaries`). Nothing in the
signature mentions RK4, fixed steps, or four stages - an adaptive embedded method (RK45,
Dormand-Prince) or a symplectic method (velocity Verlet, leapfrog) satisfies the identical signature
and needs no change to any caller. An adaptive method would additionally want to report the step it
actually took and an error estimate; that is a strict *extension* of this contract (a subclass, or a
richer return value), not a change to it - `Simulation.step` calling `self._cowell_integrator.step(...)`
would not itself need to change to gain a fixed-step-RK4-to-adaptive upgrade, only the object
constructed in `Simulation.__init__` would.

**Why `primaries` is part of the contract, not an afterthought.** A body bound to a parent (the only
kind `Simulation.set_propagator` allows onto Cowell - see its docstring) has to be integrated in a
frame that does not silently assume the parent stands still. The Cartesian equations of motion this
module integrates use whatever `provider` returns, and every registered force kernel so far
(`gravity.point_mass_gravity`, `geopotential.j2_kernel`) computes an acceleration that depends *only*
on the body's position relative to `parent_indices[body]` - never on the parent's absolute position.
That makes the *relative* state `y = (r_body - r_parent, v_body - v_parent)` obey a self-contained
ODE, `dy/dt = f(y)`, with no reference anywhere to where the parent actually is or how it is moving.
Integrating that relative state, rather than the body's absolute position, is what makes a Cowell body
correctly follow a parent that is itself accelerating - see `docs/architecture.md`'s Cowell section
for the derivation and `docs/engineering-log.md` for the bug this replaced (a body's absolute state
integrated against only its parent-relative acceleration silently assumed a stationary parent, and the
Moon flew off unbound around a real, accelerating Earth).

**How the relative frame is built without changing the force-kernel contract.** `provider` still needs
a full, absolute-frame `(C, 6)` array to evaluate every enabled model on every dispatched body - a
kernel reads `state[primaries]` and `state[indices]` and expects them in one common frame (see
`forces.ForceKernel`). At each RK4 sub-stage this integrator reconstructs `state[indices]` as
`state[primaries] + candidate_relative_state`, calls `provider`, and discards the reconstruction once
the acceleration is read back. Because `point_mass_gravity`, `j2` and `drag` depend only on
`state[primaries] - state[indices]`, and that difference is unchanged by adding the *same* offset to
both sides, the acceleration returned is exactly the correct relative acceleration regardless of what
value `state[primaries]` happens to hold - it is read, never written, throughout one Cowell step, so
its value is simply whatever `global_states` held when the step began. `Simulation.step` is
responsible for translating the integrator's result - which is therefore expressed relative to the
parent's *start-of-step* position - into the parent's freshly Keplerian-propagated position once that
is known; see its docstring for exactly where that happens.

**What remains an approximation, and what does not.** For a force that depends only on a body's
state relative to its own parent (`point_mass_gravity`, `j2`, `drag`), this formulation carries *no*
approximation from the parent's motion during the step. The relative ODE is exact however the parent
accelerates, because the parent's absolute trajectory never enters it. A force that depends on some
*other* body's position is different. `third_body` is one; sibling-sibling coupling would be another.
This integrator does not advance any row outside `indices` between stages (`primaries` is read but never
advanced), so such a model sees the other body frozen at its start-of-step position. For `third_body`
that makes Cowell **first order** in dt rather than fourth: the tide is effectively applied h/2 late.
The Moon's 30-day error is 2.64 km at 3600 s and 1.34 km at 1800 s; see `thirdbody.py` and
`tests/validation/test_third_body.py`. Removing the freeze would mean advancing perturbers per stage.

References
----------
Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*, 3rd ed., section 17.1 (the classical
  fourth-order Runge-Kutta method, RK4).
"""
from __future__ import annotations

import math
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .forces import AccelerationProvider

__all__ = ["Integrator", "RK4Integrator", "LeapfrogIntegrator", "Yoshida4Integrator", "EnckeIntegrator",
           "INTEGRATOR_NAMES", "make_integrator", "kepler_advance", "battin_f"]


class Integrator(Protocol):
    """
    The contract `Simulation.step` drives Cowell propagation through. See the module docstring for why
    this is shaped as a stateful object rather than a function, why `primaries` is part of the
    signature, and why it supports a later adaptive or symplectic method with no change to callers.
    """

    def step(
        self,
        provider: AccelerationProvider,
        t: ScalarSeconds,
        state: NDArray[np.float64],
        dt: ScalarSeconds,
        indices: NDArray[np.int64],
        primaries: NDArray[np.int32],
    ) -> None:
        """
        Advance `state[indices]` in place from `t` by `dt`, integrated relative to `state[primaries]`.

        `state` is shape (C, 6) - `[x y z vx vy vz]`, arena-indexed exactly like
        `Simulation.global_states`. `indices[k]`'s gravitating parent is `primaries[k]` (same length,
        typically `parent_indices[indices]`); `state[primaries]` is read repeatedly but never written.
        Only `state[indices]` rows are mutated, and the value left there is `state[primaries]` *as it
        stood when this call began* plus the newly integrated relative state - not necessarily a
        physically meaningful absolute position on its own if `state[primaries]` changes afterward, by
        design (see the module docstring). `indices` may be empty, in which case this is a no-op.
        """
        ...


class RK4Integrator:
    """
    Classical fixed-step fourth-order Runge-Kutta, applied to the second-order relative system
    `d^2(r_body - r_parent)/dt^2 = a(r_body - r_parent, ...)` by treating `dv_rel/dt = a` and
    `dr_rel/dt = v_rel` as one six-component first-order system per body. See the module docstring for
    why the state integrated is the *relative* one and why that is what makes this correct for a body
    whose parent is itself accelerating.

    Scratch is allocated once, sized `(max_capacity, ...)`, at construction. Every operation inside
    `step` touches only the rows named by `indices` (and reads, never writes, `primaries`), so heap
    traffic and cost track the active Cowell set, never `max_capacity` - the same discipline
    `kernels.py` documents for `_kick` / `_accum`. Small `(indices.size, ...)`-shaped temporaries
    produced by ordinary NumPy arithmetic on `indices`-selected slices are unavoidable (fancy indexing
    always copies) and match the existing reference-tier style used elsewhere in this codebase
    (`propagators.py`, `forces.py`'s demonstration kernels); what this class specifically avoids is any
    allocation sized by `max_capacity` inside `step`, which is the failure pattern this project has hit
    before (see `docs/engineering-log.md`, "Optimisation order was wrong until it was measured", and
    the scaling-invariants tests).
    """

    def __init__(self, max_capacity: int) -> None:
        self._y0 = np.zeros((max_capacity, 6), dtype=np.float64)
        self._a1 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._a2 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._a3 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._a4 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._v1 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._v2 = np.zeros((max_capacity, 3), dtype=np.float64)
        self._v3 = np.zeros((max_capacity, 3), dtype=np.float64)

    def step(
        self,
        provider: AccelerationProvider,
        t: ScalarSeconds,
        state: NDArray[np.float64],
        dt: ScalarSeconds,
        indices: NDArray[np.int64],
        primaries: NDArray[np.int32],
    ) -> None:
        if indices.size == 0:
            return

        dt = float(dt)
        y0, a1, a2, a3, a4 = self._y0, self._a1, self._a2, self._a3, self._a4
        v1, v2, v3 = self._v1, self._v2, self._v3

        # Initial RELATIVE state: r_rel0 = r_body0 - r_parent0, v_rel0 = v_body0 - v_parent0. This is
        # the step that fixes the "missing parent acceleration" bug - integrating the absolute state
        # y0 = state[indices] directly (the earlier, incorrect version) implicitly assumes the parent
        # never moves, because nothing in that formulation ever subtracts the parent's own velocity.
        y0[indices, :3] = state[indices, :3] - state[primaries, :3]
        y0[indices, 3:] = state[indices, 3:] - state[primaries, 3:]
        r0 = y0[indices, :3]
        v0 = y0[indices, 3:]

        # Every stage below reconstructs the ABSOLUTE candidate row `state[primaries] +
        # candidate_relative` before calling `provider`, so force kernels see a genuine common-frame
        # position pair. `state[primaries]` is never written during this loop, so it is safe to read
        # fresh each time; its value does not affect the acceleration returned (see the module
        # docstring's translation-invariance argument) but is needed to give `provider` a real frame.

        # Stage 1: acceleration at (t, y0). state[indices] already equals state[primaries] + r0/v0 (the
        # untouched original absolute value), so no reconstruction is needed before this first call.
        # Copied out of the shared `accel_accum` immediately - the next call below overwrites that same
        # buffer in place, so `a1` must hold its own values, not a reference to a buffer about to change.
        a1[indices] = provider(t, state)[indices]
        v1[indices] = v0 + 0.5 * dt * a1[indices]
        state[indices, :3] = state[primaries, :3] + (r0 + 0.5 * dt * v0)
        state[indices, 3:] = state[primaries, 3:] + v1[indices]

        # Stage 2: acceleration at (t + dt/2, y1).
        a2[indices] = provider(t + 0.5 * dt, state)[indices]
        v2[indices] = v0 + 0.5 * dt * a2[indices]
        state[indices, :3] = state[primaries, :3] + (r0 + 0.5 * dt * v1[indices])
        state[indices, 3:] = state[primaries, 3:] + v2[indices]

        # Stage 3: acceleration at (t + dt/2, y2).
        a3[indices] = provider(t + 0.5 * dt, state)[indices]
        v3[indices] = v0 + dt * a3[indices]
        state[indices, :3] = state[primaries, :3] + (r0 + dt * v2[indices])
        state[indices, 3:] = state[primaries, 3:] + v3[indices]

        # Stage 4: acceleration at (t + dt, y3).
        a4[indices] = provider(t + dt, state)[indices]

        # Weighted combination on the RELATIVE state: y_rel_new = y_rel0 + dt/6 * (k1+2k2+2k3+k4), then
        # converted back to the absolute row Simulation.step expects, still offset by `state[primaries]`
        # as it stood at the *start* of this call (not necessarily its current value below, since nothing
        # here has changed it) - Simulation.step re-bases this onto the parent's fresh end-of-step
        # position once that is available. See the module docstring.
        r_rel_new = r0 + (dt / 6.0) * (v0 + 2.0 * v1[indices] + 2.0 * v2[indices] + v3[indices])
        v_rel_new = v0 + (dt / 6.0) * (a1[indices] + 2.0 * a2[indices] + 2.0 * a3[indices] + a4[indices])
        state[indices, :3] = state[primaries, :3] + r_rel_new
        state[indices, 3:] = state[primaries, 3:] + v_rel_new


# --------------------------------------------------------------------------------------------------
# Symplectic integrators: Stormer-Verlet (leapfrog) and Yoshida's fourth-order composition
# --------------------------------------------------------------------------------------------------

class _KickDriftKick:
    """
    Shared machinery of the symplectic methods: the Stormer-Verlet substep in kick-drift-kick form
    (Hairer, Lubich & Wanner, *Geometric Numerical Integration*, 2nd ed., Ch. I.3; section cited from memory, equation numbers not checked), on the
    same parent-relative state as `RK4Integrator` and through the same `provider` contract.

        v_half = v + (h/2) a(t, r)
        r'     = r + h v_half
        v'     = v_half + (h/2) a(t + h, r')

    **Symplectic only for forces that do not depend on velocity** (`point_mass_gravity`, `j2`,
    `zonal`, `tesseral` - the last is time-dependent, which the method handles in extended phase
    space). `drag` depends on velocity: the provider is then evaluated at `v_half`, the method stays
    second-order consistent, but the long-horizon guarantees do not apply - and drag is dissipative,
    so there is nothing for them to preserve anyway. The acceleration at the end of one substep is the
    start of the next (the "first same as last" saving), so `n` substeps cost `n + 1` evaluations.
    """

    def __init__(self, max_capacity: int) -> None:
        self._r = np.zeros((max_capacity, 3), dtype=np.float64)
        self._v = np.zeros((max_capacity, 3), dtype=np.float64)
        self._a = np.zeros((max_capacity, 3), dtype=np.float64)

    def _compose(
        self,
        provider: AccelerationProvider,
        t: ScalarSeconds,
        state: NDArray[np.float64],
        dt: ScalarSeconds,
        indices: NDArray[np.int64],
        primaries: NDArray[np.int32],
        weights: tuple[float, ...],
    ) -> None:
        if indices.size == 0:
            return
        dt = float(dt)
        r, v, a = self._r, self._v, self._a
        r[indices] = state[indices, :3] - state[primaries, :3]
        v[indices] = state[indices, 3:] - state[primaries, 3:]
        a[indices] = provider(t, state)[indices]          # state[indices] is still the original row
        tau = float(t)
        for w in weights:
            h = w * dt
            v_half = v[indices] + (0.5 * h) * a[indices]
            r[indices] = r[indices] + h * v_half
            tau += h
            state[indices, :3] = state[primaries, :3] + r[indices]
            state[indices, 3:] = state[primaries, 3:] + v_half
            a[indices] = provider(tau, state)[indices]
            v[indices] = v_half + (0.5 * h) * a[indices]
        state[indices, :3] = state[primaries, :3] + r[indices]
        state[indices, 3:] = state[primaries, 3:] + v[indices]


class LeapfrogIntegrator(_KickDriftKick):
    """Stormer-Verlet / leapfrog: second order, symplectic, time-reversible. Two force evaluations per
    step (half of RK4's four). See `_KickDriftKick` for the method and its scope."""

    def step(self, provider: AccelerationProvider, t: ScalarSeconds, state: NDArray[np.float64],
             dt: ScalarSeconds, indices: NDArray[np.int64], primaries: NDArray[np.int32]) -> None:
        self._compose(provider, t, state, dt, indices, primaries, (1.0,))


#: Yoshida's triple-jump weights, `w1 = 1 / (2 - 2^(1/3))`, `w0 = -2^(1/3) / (2 - 2^(1/3))`:
#: Yoshida, Phys. Lett. A 150 (1990) 262; Hairer, Lubich & Wanner Ch. II.4 (cited from memory; the
#: weights are checked by the order test, which fails unless they make the composition fourth order).
#: The middle step runs backwards (`w0 < 0`); `2 w1 + w0 = 1`.
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


class Yoshida4Integrator(_KickDriftKick):
    """Yoshida's fourth-order symplectic composition of three leapfrog substeps (`w1, w0, w1`). Four
    force evaluations per step, the same as RK4. See `_KickDriftKick` for its scope."""

    def step(self, provider: AccelerationProvider, t: ScalarSeconds, state: NDArray[np.float64],
             dt: ScalarSeconds, indices: NDArray[np.int64], primaries: NDArray[np.int32]) -> None:
        self._compose(provider, t, state, dt, indices, primaries, (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1))


#: Cowell integrators by name - what a configuration names (`Simulation.set_cowell_integrator`,
#: `sweep.ModelConfig.integrator`). Each has a fused compiled twin in `kernels.py`.
INTEGRATOR_NAMES = ("rk4", "leapfrog", "yoshida4", "encke")


def make_integrator(name: str, max_capacity: int,
                    mu_array: Optional[NDArray[np.float64]] = None) -> Integrator:
    """A fresh integrator of the named kind, scratch sized to `max_capacity`. `"encke"` needs the
    arena's `mu_array` (read live) for its reference conics."""
    if name == "rk4":
        return RK4Integrator(max_capacity)
    if name == "leapfrog":
        return LeapfrogIntegrator(max_capacity)
    if name == "yoshida4":
        return Yoshida4Integrator(max_capacity)
    if name == "encke":
        if mu_array is None:
            raise ValueError("the Encke integrator needs the arena's mu_array")
        return EnckeIntegrator(max_capacity, mu_array)
    raise ValueError(f"unknown integrator {name!r}; have {INTEGRATOR_NAMES}")


# --------------------------------------------------------------------------------------------------
# Encke: integrate only the deviation from the osculating conic, re-anchored every step
# --------------------------------------------------------------------------------------------------

def _stumpff(z: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised Stumpff `C(z)`, `S(z)` (series near 0, so no cancellation)."""
    c = np.empty_like(z)
    s = np.empty_like(z)
    small = np.abs(z) < 1e-3
    pos = (z > 0.0) & ~small
    neg = (z < 0.0) & ~small
    zs = z[small]
    c[small] = 0.5 - zs / 24.0 + zs * zs / 720.0 - zs ** 3 / 40320.0
    s[small] = 1.0 / 6.0 - zs / 120.0 + zs * zs / 5040.0 - zs ** 3 / 362880.0
    sp = np.sqrt(z[pos])
    c[pos] = (1.0 - np.cos(sp)) / z[pos]
    s[pos] = (sp - np.sin(sp)) / sp ** 3
    sn = np.sqrt(-z[neg])
    c[neg] = (np.cosh(sn) - 1.0) / (-z[neg])
    s[neg] = (np.sinh(sn) - sn) / sn ** 3
    return c, s


def kepler_advance(r0: NDArray[np.float64], v0: NDArray[np.float64], dt: float,
                   mu: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    `(r, v)` `(k, 3)` after `dt` on each body's two-body conic through `(r0, v0)` about `mu` `(k,)`:
    the universal-variable Kepler equation (Curtis Alg. 3.3, f and g with their derivatives), solved
    by safeguarded Newton for every body at once to 1e-12 in the universal anomaly. Any conic type.

    Plain Newton fails above escape speed (measured: v = 10.7 km/s at 7,000 km), because F(chi) grows
    exponentially on a hyperbola: from above it descends in constant steps of about sqrt(-a), and
    from below it overshoots into cosh overflow. So the solve is bracketed, as Numerical Recipes'
    `rtsafe`. `dchi/dt = sqrt(mu) / r` and r never falls below periapsis, so the root lies between 0
    and `sqrt(mu) dt / r_p`. F is strictly increasing (`F' = r > 0`), so the root is unique, and any
    Newton step that leaves the bracket, overflows, or fails to halve the step before last becomes a
    bisection step. The start is Vallado's hyperbolic guess (*Fundamentals*, 4th ed., Alg. 8),
    `chi0 = sign(dt) sqrt(-a) ln[-2 mu alpha dt / (r0.v0 + sign(dt) sqrt(-mu a)(1 - r0 alpha))]`,
    where its logarithm is valid and has the sign of dt, and `sqrt(mu) |alpha| dt` elsewhere.
    *Measured* on 3,240 cases (1-100 km/s, flight-path angles -89 to 89 deg, dt from -6e4 to 6e5 s,
    starting at 7,000 and 1e6 km): every case converges, median 5 iterations, worst 48. The 1e-12
    tolerance matches `iod.kepler_universal_fg`. The earlier 1e-14 sat below the residual's own
    rounding floor (~1.5e-14 relative at chi ~ 50). Convergence is quadratic, so the step that
    passes 1e-12 lands at round-off anyway.
    """
    rn = np.sqrt(np.einsum("ij,ij->i", r0, r0))
    vr = np.einsum("ij,ij->i", r0, v0) / rn
    alpha = 2.0 / rn - np.einsum("ij,ij->i", v0, v0) / mu
    sq = np.sqrt(mu)
    a0 = rn * vr / sq
    b0 = 1.0 - alpha * rn
    h = np.cross(r0, v0)
    p = np.einsum("ij,ij->i", h, h) / mu
    r_p = p / (1.0 + np.sqrt(np.maximum(0.0, 1.0 - p * alpha)))
    sg = math.copysign(1.0, dt)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bound = 1.01 * sq * abs(dt) / r_p
        den = rn * vr + sg * np.sqrt(-mu / alpha) * b0
        arg = -2.0 * mu * alpha * dt / den
        chi_h = sg * np.sqrt(-1.0 / alpha) * np.log(arg)
    lo = -bound if dt < 0.0 else np.zeros_like(rn)
    hi = np.zeros_like(rn) if dt < 0.0 else bound
    chi = np.where(alpha != 0.0, sq * np.abs(alpha) * dt, sq * dt / rn)
    chi = np.where((alpha < 0.0) & (arg > 0.0) & np.isfinite(chi_h) & (chi_h * dt > 0.0), chi_h, chi)
    chi = np.clip(chi, lo, hi)
    dx_old = hi - lo
    dx = dx_old.copy()
    done = np.zeros(rn.shape, dtype=bool)   # a converged body is frozen: a later safeguard step could move it
    bisected = np.zeros(rn.shape, dtype=bool)
    for _ in range(100):
        with np.errstate(over="ignore", invalid="ignore"):
            z = alpha * chi * chi
            c, s = _stumpff(z)
            f_val = a0 * chi * chi * c + b0 * chi ** 3 * s + rn * chi - sq * dt
            f_der = a0 * chi * (1.0 - z * s) + b0 * chi * chi * c + rn
            finite = np.isfinite(f_val)
            lo = np.where(np.where(finite, f_val < 0.0, chi < 0.0), chi, lo)   # overflow: chi is too far out
            hi = np.where(np.where(finite, f_val > 0.0, chi > 0.0), chi, hi)
            newton = chi - f_val / f_der
            ok = finite & (newton > lo) & (newton < hi) & (np.abs(2.0 * f_val) <= np.abs(dx_old * f_der))
            new = np.where(done, chi, np.where(ok, newton, 0.5 * (lo + hi)))
        bisected = np.where(done, bisected, ~ok)
        step = new - chi
        chi = new
        dx_old, dx = dx, np.abs(step)
        done |= np.abs(step) <= 1e-12 * np.maximum(1.0, np.abs(chi))
        if bool(np.all(done)):
            break
    else:
        raise RuntimeError("kepler_advance: universal Kepler equation did not converge in 100 iterations")
    if bool(np.any(bisected)):
        # A bisection step stops at the bracket's width (1e-12), not at round-off; one Newton step from
        # there does. Without it, which path an ulp sends a body down shows in the result at ~1e-11.
        z = alpha * chi * chi
        c, s = _stumpff(z)
        f_val = a0 * chi * chi * c + b0 * chi ** 3 * s + rn * chi - sq * dt
        f_der = a0 * chi * (1.0 - z * s) + b0 * chi * chi * c + rn
        chi = np.where(bisected, chi - f_val / f_der, chi)
    z = alpha * chi * chi
    c, s = _stumpff(z)
    f = 1.0 - chi * chi / rn * c
    g = dt - chi ** 3 * s / sq
    r = f[:, None] * r0 + g[:, None] * v0
    rr = np.sqrt(np.einsum("ij,ij->i", r, r))
    f_dot = sq / (rr * rn) * (z * chi * s - chi)
    g_dot = 1.0 - chi * chi / rr * c
    v = f_dot[:, None] * r0 + g_dot[:, None] * v0
    return r, v


def battin_f(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """`(1 + q)^(3/2) - 1` without cancellation: `q (3 + 3q + q^2) / (1 + (1 + q)^(3/2))`."""
    out: NDArray[np.float64] = q * (3.0 + 3.0 * q + q * q) / (1.0 + (1.0 + q) ** 1.5)
    return out


class EnckeIntegrator:
    """
    Encke's method with the reference re-anchored every step: the reference is the osculating two-body
    conic through each body's state at the start of the step, advanced exactly (`kepler_advance`), and
    only the deviation `dr = r - r_ref` is integrated, by classical RK4, under

        d2(dr)/dt2 = -(mu / rho^3) (f(q) r + dr) + a_p,     rho = |r_ref|,
        q = dr . (dr - 2 r) / r^2,    f(q) = (1 + q)^(3/2) - 1 (`battin_f`),

    derived from `-mu r / r^3 + mu r_ref / rho^3` with `rho^2 / r^2 = 1 + q` (Battin, *An Introduction
    to the Mathematics and Methods of Astrodynamics*, Sec. 9.3, cited from memory; the derivation is in
    `docs/architecture.md` and the identity is tested against the direct difference). `a_p` is the
    provider's total acceleration minus the central term `-mu r / r^3`, so **every body it advances must
    carry `point_mass_gravity`** (`Simulation` refuses to step otherwise), with `mu` the summed
    `mu_array[body] + mu_array[parent]` that model uses.

    **Why re-anchor every step.** Nothing persists between steps, so a step the engine splits at a
    manoeuvre or an event, or rewinds during an event's root find, needs no reference bookkeeping; and
    for pure two-body motion the deviation is identically zero, so the method is exact at any step.
    Under a perturbation RK4's truncation acts only on the deviation, driven by `a_p` (~1e-3 of the
    central term for J2 in LEO) rather than on the whole orbit.

    Two `kepler_advance` solves per step (to `h/2` and `h`) and four provider evaluations, like RK4.
    Compiled twin: `kernels.cowell_encke_step`, held to this to 1e-12 (the NumPy path is the reference).
    """

    def __init__(self, max_capacity: int, mu_array: NDArray[np.float64]) -> None:
        self._mu_array = mu_array            # the arena's own array, read live (summed with the parent's)

    def _deviation_rate(self, provider: AccelerationProvider, tau: float, state: NDArray[np.float64],
                        indices: NDArray[np.int64], primaries: NDArray[np.int32], mu: NDArray[np.float64],
                        r_ref: NDArray[np.float64], v_ref: NDArray[np.float64],
                        dy: NDArray[np.float64]) -> NDArray[np.float64]:
        dr, dv = dy[:, :3], dy[:, 3:]
        r = r_ref + dr
        state[indices, :3] = state[primaries, :3] + r
        state[indices, 3:] = state[primaries, 3:] + v_ref + dv
        a_tot = provider(tau, state)[indices]
        r2 = np.einsum("ij,ij->i", r, r)
        rn = np.sqrt(r2)
        a_p = a_tot + (mu / (r2 * rn))[:, None] * r
        rho = np.sqrt(np.einsum("ij,ij->i", r_ref, r_ref))
        q = np.einsum("ij,ij->i", dr, dr - 2.0 * r) / r2
        dd = -(mu / rho ** 3)[:, None] * (battin_f(q)[:, None] * r + dr) + a_p
        out: NDArray[np.float64] = np.concatenate([dv, dd], axis=1)
        return out

    def step(self, provider: AccelerationProvider, t: ScalarSeconds, state: NDArray[np.float64],
             dt: ScalarSeconds, indices: NDArray[np.int64], primaries: NDArray[np.int32]) -> None:
        if indices.size == 0:
            return
        h = float(dt)
        t0 = float(t)
        mu = self._mu_array[indices] + self._mu_array[primaries]
        r0 = state[indices, :3] - state[primaries, :3]
        v0 = state[indices, 3:] - state[primaries, 3:]
        r_mid, v_mid = kepler_advance(r0, v0, 0.5 * h, mu)
        r_end, v_end = kepler_advance(r0, v0, h, mu)
        zero = np.zeros((indices.size, 6), dtype=np.float64)
        k1 = self._deviation_rate(provider, t0, state, indices, primaries, mu, r0, v0, zero)
        k2 = self._deviation_rate(provider, t0 + 0.5 * h, state, indices, primaries, mu, r_mid, v_mid, 0.5 * h * k1)
        k3 = self._deviation_rate(provider, t0 + 0.5 * h, state, indices, primaries, mu, r_mid, v_mid, 0.5 * h * k2)
        k4 = self._deviation_rate(provider, t0 + h, state, indices, primaries, mu, r_end, v_end, h * k3)
        dy = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[indices, :3] = state[primaries, :3] + r_end + dy[:, :3]
        state[indices, 3:] = state[primaries, 3:] + v_end + dy[:, 3:]
