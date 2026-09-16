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
hierarchy, no barycenters, no force-model registry. It integrates

    d2r_i/dt2 = sum_{j != i} mu_j (r_j - r_i) / |r_j - r_i|^3   [+ optional J2 terms, below]

in inertial Cartesian coordinates and nothing else. An error in `frames.py`, in the anomaly stack or
in `geopotential.py` therefore cannot appear on both sides of a comparison.

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

J2 oblateness (optional)
------------------------
Passed **explicitly** - `reference_for(..., oblateness={"Earth": (j2, r_eq)})` - and never read from
the simulation's `force_model_mask` / `force_model_params`, so a bug in how the engine stores its
force-model configuration cannot be copied into the truth it is judged against. With no oblateness
argument (or all-zero J2) the right-hand side is exactly the point-mass one, bit for bit.

*Physical assumptions, the same ones `geopotential.j2_kernel` makes:* the oblate body's spin
(symmetry) axis is the +z axis of the integration frame and does not move; the field is
time-invariant in that frame. The Earth constants a caller should pass are the EGM96/WGS-84 pair
`J2 = 1.0826266835e-3`, `R_eq = 6378.137 km`, numerically equal to `geopotential.EARTH_J2` /
`EARTH_R_EQ`. They are **not imported** from there (nor defined here): the caller supplies them, and
the tests restate the literals so a change to the engine's constants is not silently mirrored.

*The field, derived from a different starting point than the engine.* `geopotential.py` differentiates
the J2 potential in Cartesian components (Curtis' form). Here the gradient is taken in **spherical
coordinates** - radius `r` and geocentric latitude `phi`, with `s = sin(phi) = z/r` - and projected onto
the radial unit vector and the spin axis. With the potential energy per unit mass of the J2 term

    Phi_J2(r, phi) = + gm J2 R^2 P2(sin phi) / r^3,        P2(s) = (3 s^2 - 1) / 2,

and `a = -grad Phi = -(dPhi/dr) r_hat - (1/r)(dPhi/dphi) phi_hat`:

    -(dPhi/dr)           = 3 gm J2 R^2 P2(s) / r^4
    -(1/r)(dPhi/dphi)    = -(gm J2 R^2 / r^4) * 3 s cos(phi)          [dP2(sin phi)/dphi = 3 s cos phi]
    phi_hat              = (z_hat - s r_hat) / cos(phi)                [northward unit vector]

The `cos(phi)` cancels (so the poles are not singular), and collecting terms along `r_hat` and `z_hat`:

    a_J2 = (3 gm J2 R^2 / r^4) * [ (5 s^2 - 1)/2 * r_hat  -  s * z_hat ]            (*)

That is the form implemented. Expanding (*) into components reproduces Curtis' Cartesian expression
(`z`: `(3/2)(gm J2 R^2/r^4) s (5 s^2 - 3)`), which is the cross-check - not the source.
Citation: the spherical-partials route (dU/dr, dU/dphi_gc, dU/dlambda, then rotate to Cartesian) is
the one Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 8.7 uses for the
general geopotential, and Montenbruck & Gill, *Satellite Orbits* (2000), Section 3.2 give the zonal
potential `U = (GM/r) sum_n -J_n (R/r)^n P_n(sin phi)` it starts from (their Eq. 3.28 for the
expansion). **Section and equation numbers are from memory and unverified.** The derivation above is
self-contained and is what should be checked.

*Sign sanity.* Equator (`s = 0`): `a = -(3/2)(gm J2 R^2/r^4) r_hat`, inward. North pole (`s = 1`,
`r_hat = z_hat`): `a = (3 gm J2 R^2/r^4)(2 - 1) z_hat`, outward, twice the equatorial magnitude.

*Reaction and conservation.* The oblate body `j` exerts `mu_i * mu_j * G(r_i - r_j)` (per unit G) on
every other body `i`, where `G` is (*) per unit `gm`. The equal-and-opposite force is applied to `j`,
so total linear momentum is conserved. The field is axisymmetric about an axis through `j` parallel
to `z`, so the z-component of total angular momentum is conserved too; the other two components are
not, because the matching torque would act on the oblate body's spin, which is not modelled (the axis
is held fixed). Total energy including the J2 pair potential is conserved, and `energy_drift` includes
that term when oblateness is present. The field acts on *every* other integrated body, as physics does;
the engine applies `j2` only to bodies it is enabled on, relative to their own parent, so a comparison
where those differ is a comparison, not a verification.

References
----------
Hairer, Norsett & Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., section II.5
  (the DOP853 coefficients and the 8(5,3) embedded error estimate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Mapping, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "ReferenceTrajectory", "nbody_acceleration", "j2_field", "oblateness_acceleration",
    "integrate_nbody", "reference_for",
    "DEFAULT_RTOL", "DEFAULT_ATOL", "TRUTH_RTOL", "TRUTH_ATOL",
]

# DOP853 at these tolerances sits near the double-precision floor. rtol=1e-13 is about two decimal
# digits short of machine epsilon, which is as tight as an adaptive integrator can usefully run
# before step-size control starts chasing rounding noise.
#
# atol is NOT negligible everywhere, contrary to what this comment used to say. scipy scales each
# component's error by atol + rtol*|y|. For positions of 1e3-1e8 km, rtol*|y| >= 1e-10 km and rtol
# governs; for velocities of a few km/s, rtol*|v| ~ 1e-12 km/s, so atol = 1e-9 km/s governs the
# velocity components by three orders of magnitude. These defaults are kept unchanged so existing
# callers stay bit-identical; for frontier-plot truth use TRUTH_RTOL / TRUTH_ATOL below.
DEFAULT_RTOL = 1e-13
DEFAULT_ATOL = 1e-9

# Recommended for truth runs that judge small Cowell errors (the model-fidelity frontier plot).
# atol is lowered until it no longer dominates a LEO velocity (rtol*|v| = 7.6e-13 km/s at 7.6 km/s).
# Measured on earth_constellation(6 sats, 550 km) with J2, against a run at rtol=2.5e-14, atol=1e-16:
# position error after one orbit 3.9e-9 km at the defaults, 1.0e-9 km here; after one day 2.5e-8 km
# and 9.1e-9 km. atol=1e-11 and 1e-14 measured the same as 1e-12, so atol is no longer the limit and
# rtol (error roughly proportional to it: 1.5e-8 km at 1e-12, 1.6e-7 km at 1e-11) is. Cost is about
# 15% more right-hand-side evaluations than the defaults. See tests/validation/test_reference_j2.py.
TRUTH_RTOL = 1e-13
TRUTH_ATOL = 1e-12


@dataclass(frozen=True)
class ReferenceTrajectory:
    """
    Positions and velocities of the integrated bodies, sampled at `times`.

    `positions` and `velocities` have shape `(n_times, n_bodies, 3)`, ordered to match `names`.
    `j2` and `r_eq` record the oblateness the truth was integrated with, per body (zeros when none),
    so a stored result carries its own model configuration.
    """

    names: List[str]
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    mu: NDArray[np.float64]
    energy_drift: float
    j2: Optional[NDArray[np.float64]] = None
    r_eq: Optional[NDArray[np.float64]] = None

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


def j2_field(rel: NDArray[np.float64], j2: float, r_eq: float) -> NDArray[np.float64]:
    """
    J2 acceleration **per unit gm of the oblate body** at positions `rel` `(m,3)` relative to it,
    in 1/km^2 (multiply by gm in km^3/s^2 for km/s^2). Equation (*) of the module docstring:

        G = (3 J2 R^2 / r^4) * [ (5 s^2 - 1)/2 * r_hat - s * z_hat ],     s = z / r

    Every row of `rel` must be non-zero; callers exclude the oblate body itself.
    """
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    r_hat = rel / r[:, np.newaxis]
    s = r_hat[:, 2]
    k = 3.0 * j2 * r_eq * r_eq / r ** 4

    field: NDArray[np.float64] = (k * 0.5 * (5.0 * s * s - 1.0))[:, np.newaxis] * r_hat   # radial part
    field[:, 2] -= k * s                                                                  # spin-axis part
    return field


def _j2_potential_per_unit_mu(rel: NDArray[np.float64], j2: float, r_eq: float) -> NDArray[np.float64]:
    """`Phi_J2 / gm = J2 R^2 P2(z/r) / r^3`, 1/km. Used only for the `energy_drift` self-check."""
    r = np.sqrt(np.einsum("ij,ij->i", rel, rel))
    s = rel[:, 2] / r
    potential: NDArray[np.float64] = j2 * r_eq * r_eq * 0.5 * (3.0 * s * s - 1.0) / r ** 3
    return potential


def oblateness_acceleration(
    positions: NDArray[np.float64],
    mu: NDArray[np.float64],
    j2: NDArray[np.float64],
    r_eq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    J2 accelerations on every body from every oblate body (`j2[j] != 0`), shape `(n,3)`, km/s^2.

    Body `i != j` receives `mu[j] * G(r_i - r_j)`; the oblate body `j` receives the reaction
    `-sum_i mu[i] * G(r_i - r_j)`, so `sum_i mu[i] * a_i` over the J2 terms is zero (momentum is
    conserved). Loops over the oblate bodies only - there are few - and is vectorised over the rest.
    """
    n = positions.shape[0]
    accel = np.zeros((n, 3), dtype=np.float64)
    for j in np.flatnonzero(j2 != 0.0):
        others = np.arange(n) != j
        field = j2_field(positions[others] - positions[j], float(j2[j]), float(r_eq[j]))
        accel[others] += mu[j] * field
        accel[j] -= np.einsum("i,ij->j", mu[others], field)
    return accel


def _specific_energy(
    positions: NDArray[np.float64], velocities: NDArray[np.float64], mu: NDArray[np.float64]
) -> float:
    """
    Total energy per unit G, in km^2/s^2 weighted by mu. Used only as a self-check on the integrator.

    Computed here rather than imported from the test helpers so that this module has no dependency
    on the suite that consumes it. Massless bodies carry zero weight, so for a constellation of
    massless satellites this measures the primary alone and certifies nothing about the satellites;
    check their specific energies directly.
    """
    kinetic = 0.5 * float(np.sum(mu * np.einsum("ij,ij->i", velocities, velocities)))

    separation = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", separation, separation))
    np.fill_diagonal(distance, np.inf)

    # Each pair counted twice by the full matrix, hence the halving.
    potential = -0.5 * float(np.sum(np.outer(mu, mu) / distance))
    return kinetic + potential


def _oblateness_energy(
    positions: NDArray[np.float64], mu: NDArray[np.float64],
    j2: NDArray[np.float64], r_eq: NDArray[np.float64],
) -> float:
    """J2 pair potential energy per unit G: sum over oblate j, other i, of mu_i mu_j Phi_J2/gm_j."""
    n = positions.shape[0]
    total = 0.0
    for j in np.flatnonzero(j2 != 0.0):
        others = np.arange(n) != j
        phi = _j2_potential_per_unit_mu(positions[others] - positions[j], float(j2[j]), float(r_eq[j]))
        total += float(mu[j] * np.sum(mu[others] * phi))
    return total


def integrate_nbody(
    mu: NDArray[np.float64],
    positions0: NDArray[np.float64],
    velocities0: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    names: Optional[List[str]] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    j2: Optional[NDArray[np.float64]] = None,
    r_eq: Optional[NDArray[np.float64]] = None,
) -> ReferenceTrajectory:
    """
    Integrate an isolated N-body system with DOP853 and sample it at `times`.

    `times` must start at 0.0, be strictly increasing, and are seconds from the initial state.

    `j2` and `r_eq`, both shape `(n,)` or both `None`, give each body's J2 coefficient and the
    equatorial radius (km) it is normalised to; a zero `j2` means a point mass. See the module
    docstring for the field, its frame assumption and conservation properties.

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

    if (j2 is None) != (r_eq is None):
        raise ValueError("j2 and r_eq must be given together")
    j2_arr = np.zeros(n) if j2 is None else np.asarray(j2, dtype=np.float64)
    r_eq_arr = np.zeros(n) if r_eq is None else np.asarray(r_eq, dtype=np.float64)
    if j2_arr.shape != (n,) or r_eq_arr.shape != (n,):
        raise ValueError(f"j2 and r_eq must have shape ({n},), got {j2_arr.shape} and {r_eq_arr.shape}")
    if not (np.all(np.isfinite(j2_arr)) and np.all(np.isfinite(r_eq_arr))):
        raise ValueError("j2 and r_eq must be finite")
    if np.any((j2_arr != 0.0) & ~(r_eq_arr > 0.0)):
        raise ValueError("every body with non-zero j2 needs a positive r_eq")
    oblate = bool(np.any(j2_arr != 0.0))

    if oblate:
        def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            r = y[: 3 * n].reshape(n, 3)
            v = y[3 * n:].reshape(n, 3)
            accel = nbody_acceleration(r, mu) + oblateness_acceleration(r, mu, j2_arr, r_eq_arr)
            return np.concatenate([v.ravel(), accel.ravel()])
    else:
        # Exactly the historical right-hand side, so point-mass callers stay bit-identical.
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
    if oblate:
        e0 += _oblateness_energy(pos[0], mu, j2_arr, r_eq_arr)
        e1 += _oblateness_energy(pos[-1], mu, j2_arr, r_eq_arr)
    drift = abs((e1 - e0) / e0) if e0 != 0.0 else abs(e1 - e0)

    return ReferenceTrajectory(
        names=list(names) if names is not None else [f"body_{k}" for k in range(n)],
        times=np.asarray(times, dtype=np.float64),
        positions=pos,
        velocities=vel,
        mu=mu.copy(),
        energy_drift=float(drift),
        j2=j2_arr,
        r_eq=r_eq_arr,
    )


def reference_for(
    sim: "Simulation",
    times: NDArray[np.float64],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    oblateness: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> ReferenceTrajectory:
    """
    Integrate the physical bodies of a built `Simulation` from its current state.

    Initial conditions are taken from `global_states`, so the reference starts from exactly the
    configuration the engine is about to propagate - the comparison then isolates propagation error
    rather than mixing in any difference in how the scenario was set up.

    Barycenters are excluded: their `mu_array` rows hold summed system mass, so integrating them
    alongside their members would double-count every body.

    `oblateness` maps body name to `(j2, r_eq_km)`, e.g. `{"Earth": (1.0826266835e-3, 6378.137)}`.
    It is deliberately independent of the simulation's own force-model configuration - see the module
    docstring. A name that is not an integrated body raises `KeyError` rather than being ignored: a
    typo that silently dropped J2 would make the truth point-mass and look like an engine error.
    Only `mu_array` and `global_states` are read from `sim`.
    """
    physical = sim.active_mask & ~sim.is_system
    slots = np.flatnonzero(physical)

    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    names = [slot_to_name.get(int(s), f"slot_{int(s)}") for s in slots]

    j2: Optional[NDArray[np.float64]] = None
    r_eq: Optional[NDArray[np.float64]] = None
    if oblateness is not None:
        j2 = np.zeros(len(names))
        r_eq = np.zeros(len(names))
        for name, (j2_value, r_eq_value) in oblateness.items():
            if name not in names:
                raise KeyError(f"oblateness given for '{name}', which is not an integrated body; have {names}")
            k = names.index(name)
            j2[k] = j2_value
            r_eq[k] = r_eq_value

    return integrate_nbody(
        sim.mu_array[physical].copy(),
        sim.global_states[physical, :3].copy(),
        sim.global_states[physical, 3:].copy(),
        times,
        names=names,
        rtol=rtol,
        atol=atol,
        j2=j2,
        r_eq=r_eq,
    )
