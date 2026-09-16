"""
Fixed-step Runge-Kutta integration for Cowell propagation.

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

**Why this is a Protocol.** `Integrator.step(provider, t, state, dt, indices)` is deliberately the
whole contract: advance `state[indices]` in place, from `t` by `dt`, using `provider` for
acceleration. Nothing in the signature mentions RK4, fixed steps, or four stages - an adaptive
embedded method (RK45, Dormand-Prince) or a symplectic method (velocity Verlet, leapfrog) satisfies the
identical signature and needs no change to any caller. An adaptive method would additionally want to
report the step it actually took and an error estimate; that is a strict *extension* of this contract
(a subclass, or a richer return value), not a change to it - `Simulation.step` calling
`self._cowell_integrator.step(...)` would not itself need to change to gain a fixed-step-RK4-to-adaptive
upgrade, only the object constructed in `Simulation.__init__` would.

**Frame and background-field assumption.** `state` is whatever the caller passes - for
`propagators.CowellPropagator` that is `Simulation.global_states`, the simulation-root inertial frame
(see `docs/architecture.md`'s Cowell section for why that frame was chosen). Every force kernel indexes
`state` by absolute arena slot, so a body's own parent's position is read from the same array. This
integrator does *not* re-evaluate any row outside `indices` between stages: the acceleration felt by a
Cowell body at each of its four RK4 sub-stages is computed against whatever the *other* bodies' rows
held when `step` was called, not their true position at that sub-stage's time. For the validation
scenarios in this project (an isolated or fixed two-body primary, which by construction never moves)
this is exact. For a Cowell body whose own gravitating parent is *itself* moving substantially within
one macro-step (a satellite integrated with Cowell while its planet is Keplerian-propagated around the
Sun in the same step) this introduces an additional error on top of RK4's own fourth-order truncation
error, which would show up as a lower observed convergence order. Handling that properly - resampling
or interpolating the background field's motion across the sub-stages - is future work; see
`docs/engineering-log.md`'s entry on this for the reasoning and where a fix would go if it is ever
needed. `Simulation.set_propagator` also restricts Cowell to bodies that are not themselves a system
head or barycenter, for a related but distinct reason documented there.

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

__all__ = ["Integrator", "RK4Integrator"]


class Integrator(Protocol):
    """
    The contract `Simulation.step` drives Cowell propagation through. See the module docstring for why
    this is shaped as a stateful object rather than a function, and why it supports a later adaptive or
    symplectic method with no change to callers.
    """

    def step(
        self,
        provider: AccelerationProvider,
        t: ScalarSeconds,
        state: NDArray[np.float64],
        dt: ScalarSeconds,
        indices: NDArray[np.int64],
    ) -> None:
        """
        Advance `state[indices]` in place from `t` by `dt`.

        `state` is shape (C, 6) - `[x y z vx vy vz]`, arena-indexed exactly like
        `Simulation.global_states` - and is mutated at `indices` rows only; every other row may be
        read (by `provider`) but is never written. `indices` is a sorted, duplicate-free arena slot
        array owned by the caller and may be empty, in which case this is a no-op.
        """
        ...


class RK4Integrator:
    """
    Classical fixed-step fourth-order Runge-Kutta, applied to the second-order system
    `d^2r/dt^2 = a(t, r, v)` by treating `dv/dt = a` and `dr/dt = v` as one six-component first-order
    system per body.

    Scratch is allocated once, sized `(max_capacity, ...)`, at construction. Every operation inside
    `step` touches only the rows named by `indices`, so heap traffic and cost track the active Cowell
    set, never `max_capacity` - the same discipline `kernels.py` documents for `_kick` / `_accum`.
    Small `(indices.size, ...)`-shaped temporaries produced by ordinary NumPy arithmetic on
    `indices`-selected slices are unavoidable (fancy indexing always copies) and match the existing
    reference-tier style used elsewhere in this codebase (`propagators.py`, `forces.py`'s demonstration
    kernels); what this class specifically avoids is any allocation sized by `max_capacity` inside
    `step`, which is the failure pattern this project has hit before (see `docs/engineering-log.md`,
    "Optimisation order was wrong until it was measured", and the scaling-invariants tests).
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
    ) -> None:
        if indices.size == 0:
            return

        dt = float(dt)
        y0, a1, a2, a3, a4 = self._y0, self._a1, self._a2, self._a3, self._a4
        v1, v2, v3 = self._v1, self._v2, self._v3

        y0[indices] = state[indices]
        r0 = y0[indices, :3]
        v0 = y0[indices, 3:]

        # Stage 1: acceleration at (t, y0). Copied out of the shared `accel_accum` immediately - the
        # next call below overwrites that same buffer in place, so `a1` must hold its own values, not
        # a reference to a buffer that is about to change underneath it.
        a1[indices] = provider(t, state)[indices]
        v1[indices] = v0 + 0.5 * dt * a1[indices]
        state[indices, :3] = r0 + 0.5 * dt * v0
        state[indices, 3:] = v1[indices]

        # Stage 2: acceleration at (t + dt/2, y1).
        a2[indices] = provider(t + 0.5 * dt, state)[indices]
        v2[indices] = v0 + 0.5 * dt * a2[indices]
        state[indices, :3] = r0 + 0.5 * dt * v1[indices]
        state[indices, 3:] = v2[indices]

        # Stage 3: acceleration at (t + dt/2, y2).
        a3[indices] = provider(t + 0.5 * dt, state)[indices]
        v3[indices] = v0 + dt * a3[indices]
        state[indices, :3] = r0 + dt * v2[indices]
        state[indices, 3:] = v3[indices]

        # Stage 4: acceleration at (t + dt, y3).
        a4[indices] = provider(t + dt, state)[indices]

        # Weighted combination: y_new = y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4), split into position
        # (velocity-weighted) and velocity (acceleration-weighted) halves.
        state[indices, :3] = r0 + (dt / 6.0) * (
            v0 + 2.0 * v1[indices] + 2.0 * v2[indices] + v3[indices]
        )
        state[indices, 3:] = v0 + (dt / 6.0) * (
            a1[indices] + 2.0 * a2[indices] + 2.0 * a3[indices] + a4[indices]
        )
