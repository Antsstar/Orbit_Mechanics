"""
Central point-mass (two-body) Newtonian gravity as a registered force model.

**What this is for.** Cowell propagation (`integrators.py`) integrates the Cartesian equations of
motion directly, so it needs its own force to integrate against - unlike the analytic Keplerian
propagator, which solves the two-body problem in closed form and never evaluates an acceleration at
all. This module supplies exactly that term: the same two-body physics the Keplerian propagator
already assumes, expressed as a `forces.ForceKernel` so Cowell can consume it through the same
composition layer as every other force.

A Cowell body running under *only* this model is solving the identical equation of motion the
Keplerian propagator solves analytically - same mu convention, same primary - so the two should trace
the same orbit to numerical-integration error, not to modelling error. That equivalence is what
`tests/validation/test_cowell_propagator.py`'s verification tests check.

**Convention.** Acceleration is computed toward each body's Keplerian parent (`parent_indices`, not
`body_sys_map` - see `docs/architecture.md`'s note on why the two graphs diverge and are read
differently), using the same two-body mass sum `Simulation._rehydrate_coes` and `KeplerianPropagator`
already use: `mu = mu_array[body] + mu_array[parent_indices[body]]`. A root body
(`parent_indices[i] == i`) has no primary and receives no acceleration from this model - the same
"root means no relative vector" convention `forces._radial_bias_kernel` already establishes.

**Scope.** This is a two-body term only: each body feels gravity from its own parent alone, never from
every other active body. That is deliberate, not a simplification pending a general N-body force -
Cowell here is a drop-in *numerical* alternative to the existing *analytic* two-body physics, and
nothing in the current validation asks for more. A genuine N-body or third-body force would be a
different, separately registered model (see `forces.py`'s note on `np.add.at` for what that would
need to scatter onto a repeatable target).

References
----------
Vallado, D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 1-35 (the two-body
  equation of motion, `d^2r/dt^2 = -mu r / |r|^3`).
"""
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .registry import register_force_model

__all__ = ["POINT_MASS_MODEL", "point_mass_gravity_kernel"]

POINT_MASS_MODEL: Final[str] = "point_mass_gravity"


@register_force_model(
    POINT_MASS_MODEL,
    param_names=(),
    citation="Vallado, Fundamentals of Astrodynamics and Applications, 4th ed., Eq. 1-35; mu is the "
             "parent+body GM sum, matching KeplerianPropagator's / _rehydrate_coes's two-body "
             "convention.",
)
def point_mass_gravity_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Newtonian two-body gravitational acceleration on each body toward its Keplerian parent. See the
    module docstring for the mu convention and why this reads `parent_indices` rather than
    `body_sys_map`.

    Takes no per-body parameters (`params` is present only to satisfy `forces.ForceKernel`'s shape and
    is otherwise unused) - unlike J2's oblateness coefficient, the two-body term needs nothing beyond
    what `mu_array` / `parent_indices` already hold.
    """
    primaries = parent_indices[indices]
    rel = state[primaries, :3] - state[indices, :3]  # points from the body toward its primary
    r2 = np.einsum("ij,ij->i", rel, rel)
    r = np.sqrt(r2)

    mu_total = mu_array[indices] + mu_array[primaries]

    accel = np.zeros_like(rel)
    safe = r > 0.0
    accel[safe] = (mu_total[safe] / (r2[safe] * r[safe]))[:, None] * rel[safe]

    out[indices] += accel
