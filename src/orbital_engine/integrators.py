"""
Fixed-step integration for Cowell propagation: classical RK4 (the default, and the only one with a
fused compiled twin) and two symplectic methods, leapfrog and Yoshida's fourth-order composition (see
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

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .forces import AccelerationProvider

__all__ = ["Integrator", "RK4Integrator", "LeapfrogIntegrator", "Yoshida4Integrator", "INTEGRATOR_NAMES",
           "make_integrator"]


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
#: `sweep.ModelConfig.integrator`). Only `"rk4"` has a fused compiled twin.
INTEGRATOR_NAMES = ("rk4", "leapfrog", "yoshida4")


def make_integrator(name: str, max_capacity: int) -> Integrator:
    """A fresh integrator of the named kind, scratch sized to `max_capacity`."""
    if name == "rk4":
        return RK4Integrator(max_capacity)
    if name == "leapfrog":
        return LeapfrogIntegrator(max_capacity)
    if name == "yoshida4":
        return Yoshida4Integrator(max_capacity)
    raise ValueError(f"unknown integrator {name!r}; have {INTEGRATOR_NAMES}")
