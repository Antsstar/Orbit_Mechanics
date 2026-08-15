"""
Independent reference trajectories by high-order numerical integration.

**What this is for.** Everything the validation suite asserts so far is either a closed-form
two-body result or a conserved quantity. Both are necessary and neither is sufficient: a propagator
can conserve energy and angular momentum exactly while following the wrong trajectory, and closed
form only exists for cases simple enough not to be interesting. This module supplies the third leg -
a trajectory computed by a completely different method, against which the engine's output can be
differenced.

The method is direct Newtonian N-body integration with `scipy`'s DOP853 (Dormand-Prince 8(5,3)) at
`rtol=1e-13`. It shares no code with the engine: no classical elements, no Kepler solver, no
hierarchy, no barycenters. It integrates

    d2r_i/dt2 = sum_{j != i} mu_j (r_j - r_i) / |r_j - r_i|^3

in inertial Cartesian coordinates and nothing else. An error in `frames.py` or in the anomaly stack
therefore cannot appear on both sides of a comparison.

**Two distinct uses, which must not be confused.**

1. *Verification.* Where the engine's model is exact - a massless secondary about a single primary
   is exactly Keplerian - the two must agree to integration tolerance. Disagreement is an engine
   bug. `tests/validation/test_reference_agreement.py` asserts this.

2. *Comparison.* Where the engine's model is an approximation - hierarchical two-body motion plus a
   reflex kick neglects, for instance, the Sun's perturbation of the lunar orbit - the two *will*
   diverge, and the divergence is the quantity of interest. It is the modelling error of the
   approximation, measured rather than assumed, which is the entire premise of this project. A
   disagreement here is a result, not a defect.

Knowing which case you are in is the whole skill. Treating a case-2 divergence as a bug leads to
"fixing" a correct engine; treating a case-1 divergence as a result hides a real one.

**On mass.** `mu_array` holds *summed* system mass on barycenter rows after
`_recalculate_all_barycenters`, so barycenters are excluded here - including them would double-count
every body they aggregate. Only rows with `is_system == False` are integrated.

References
----------
Hairer, Norsett & Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., section II.5
  (the DOP853 coefficients and the 8(5,3) embedded error estimate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = ["ReferenceTrajectory", "nbody_acceleration", "integrate_nbody", "reference_for"]

# DOP853 at these tolerances sits near the double-precision floor. rtol=1e-13 is about two decimal
# digits short of machine epsilon, which is as tight as an adaptive integrator can usefully run
# before step-size control starts chasing rounding noise. atol is well below the smallest physically
# meaningful quantity in km / km/s, so rtol governs throughout.
DEFAULT_RTOL = 1e-13
DEFAULT_ATOL = 1e-9


@dataclass(frozen=True)
class ReferenceTrajectory:
    """
    Positions and velocities of the integrated bodies, sampled at `times`.

    `positions` and `velocities` have shape `(n_times, n_bodies, 3)`, ordered to match `names`.
    """

    names: List[str]
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    mu: NDArray[np.float64]
    energy_drift: float

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"'{name}' is not in this reference trajectory; have {self.names}") from None

    def position_of(self, name: str) -> NDArray[np.float64]:
        """Position history of one body, shape `(n_times, 3)`."""
        return self.positions[:, self.index_of(name), :]

    def velocity_of(self, name: str) -> NDArray[np.float64]:
        """Velocity history of one body, shape `(n_times, 3)`."""
        return self.velocities[:, self.index_of(name), :]


def nbody_acceleration(
    positions: NDArray[np.float64], mu: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Newtonian gravitational acceleration on each body, shape `(n, 3)` in, `(n, 3)` out.

    Written as the full O(n^2) pairwise sum with no softening and no cutoff. This is the reference:
    it is meant to be transparently correct rather than fast, and any approximation introduced here
    would be an approximation the engine is then measured against.

    Self-interaction is removed by setting the diagonal separation to infinity before inverting,
    which avoids a divide-by-zero without a branch or a mask.
    """
    # separation[i, j] = r_j - r_i
    separation = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    square_distance = np.einsum("ijk,ijk->ij", separation, separation)
    np.fill_diagonal(square_distance, np.inf)

    inv_cube = square_distance ** -1.5
    # einsum is typed as returning Any, so the cast is what preserves the signature under --strict.
    return cast(NDArray[np.float64], np.einsum("j,ijk,ij->ik", mu, separation, inv_cube))


def _specific_energy(
    positions: NDArray[np.float64], velocities: NDArray[np.float64], mu: NDArray[np.float64]
) -> float:
    """
    Total energy per unit G, in km^2/s^2 weighted by mu. Used only as a self-check on the integrator.

    Computed here rather than imported from the test helpers so that this module has no dependency
    on the suite that consumes it.
    """
    kinetic = 0.5 * float(np.sum(mu * np.einsum("ij,ij->i", velocities, velocities)))

    separation = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", separation, separation))
    np.fill_diagonal(distance, np.inf)

    # Each pair counted twice by the full matrix, hence the halving.
    potential = -0.5 * float(np.sum(np.outer(mu, mu) / distance))
    return kinetic + potential


def integrate_nbody(
    mu: NDArray[np.float64],
    positions0: NDArray[np.float64],
    velocities0: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    names: Optional[List[str]] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ReferenceTrajectory:
    """
    Integrate an isolated N-body system with DOP853 and sample it at `times`.

    `times` must start at 0.0, be strictly increasing, and are seconds from the initial state.

    Raises `RuntimeError` if the integrator fails, rather than returning a partial trajectory. A
    truncated reference silently compared against a full engine run would report enormous
    disagreement at the tail and look like an engine bug.
    """
    from scipy.integrate import solve_ivp  # imported lazily: scipy is an optional extra

    n = mu.shape[0]
    if positions0.shape != (n, 3) or velocities0.shape != (n, 3):
        raise ValueError(
            f"expected positions and velocities of shape ({n}, 3), "
            f"got {positions0.shape} and {velocities0.shape}")
    if times[0] != 0.0:
        raise ValueError(f"times must start at 0.0, got {times[0]}")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        r = y[: 3 * n].reshape(n, 3)
        v = y[3 * n:].reshape(n, 3)
        return np.concatenate([v.ravel(), nbody_acceleration(r, mu).ravel()])

    y0 = np.concatenate([positions0.ravel(), velocities0.ravel()])

    solution = solve_ivp(
        rhs, (0.0, float(times[-1])), y0,
        method="DOP853", t_eval=times, rtol=rtol, atol=atol, dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(f"DOP853 reference integration failed: {solution.message}")

    n_times = times.shape[0]
    states = solution.y.T                                   # (n_times, 6n)
    pos = states[:, : 3 * n].reshape(n_times, n, 3)
    vel = states[:, 3 * n:].reshape(n_times, n, 3)

    # Energy drift is the integrator's own error estimate, independent of its step-size control.
    # An adaptive method can satisfy its local tolerance while accumulating global error, so this
    # is reported alongside the trajectory rather than assumed acceptable.
    e0 = _specific_energy(pos[0], vel[0], mu)
    e1 = _specific_energy(pos[-1], vel[-1], mu)
    drift = abs((e1 - e0) / e0) if e0 != 0.0 else abs(e1 - e0)

    return ReferenceTrajectory(
        names=list(names) if names is not None else [f"body_{k}" for k in range(n)],
        times=np.asarray(times, dtype=np.float64),
        positions=pos,
        velocities=vel,
        mu=mu.copy(),
        energy_drift=float(drift),
    )


def reference_for(
    sim: "Simulation",
    times: NDArray[np.float64],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ReferenceTrajectory:
    """
    Integrate the physical bodies of a built `Simulation` from its current state.

    Initial conditions are taken from `global_states`, so the reference starts from exactly the
    configuration the engine is about to propagate - the comparison then isolates propagation error
    rather than mixing in any difference in how the scenario was set up.

    Barycenters are excluded: their `mu_array` rows hold summed system mass, so integrating them
    alongside their members would double-count every body.
    """
    physical = sim.active_mask & ~sim.is_system
    slots = np.flatnonzero(physical)

    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    names = [slot_to_name.get(int(s), f"slot_{int(s)}") for s in slots]

    return integrate_nbody(
        sim.mu_array[physical].copy(),
        sim.global_states[physical, :3].copy(),
        sim.global_states[physical, 3:].copy(),
        times,
        names=names,
        rtol=rtol,
        atol=atol,
    )
