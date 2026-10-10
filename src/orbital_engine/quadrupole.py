"""
The quadrupole of a two-body system, seen from outside: registered as the force model
`"system_quadrupole"`.

**Why.** A body orbiting a system's barycentre with the system's summed `mu` (`hierarchy.reparent` to
a barycentre) sees the system as one point mass: the **monopole**. The leading error of that picture
is the system's quadrupole, of relative size `(d / r)^2` for an inner separation `d` and a distance
`r`. This model adds that term, in two fidelities, so the step from "the Earth-Moon system as a
point" to "Earth and Moon resolved" is a ladder of configurations, each measured against N-body truth:

    monopole  ->  + averaged quadrupole  ->  + instantaneous quadrupole  ->  resolved members

Physics
-------
Masses `m_a`, `m_b` at `x_a = -(m_b / M) d` and `x_b = (m_a / M) d` from their barycentre, with
`d = x_b - x_a`. Their second moment about the barycentre is `sum m x x^T = mu_r d d^T`, with the reduced
mass `mu_r = m_a m_b / M` (here as `G mu_r = mu_a mu_b / (mu_a + mu_b)`). The dipole vanishes about
the barycentre, so beyond the monopole the potential at `r` is the quadrupole

    Phi = -(G / (2 r^5)) (3 r^T S r - tr(S) r^2),      S = mu_r d d^T,

and the acceleration `a = -grad Phi` is

    a = (G mu_r / (2 r^5)) [ 6 M r - 15 (r^T M r) r / r^2 + 3 tr(M) r ],      M = d d^T.

On the pair's axis this is `-3 G mu_r d^2 / r^4` along `r_hat`: an extra pull, the pair looking like a
rod. (Standard multipole expansion; e.g. Murray & Dermott, *Solar System Dynamics*, 1999, ch. 6, for
the disturbing function of a hierarchical system. **Section from memory, unverified**; the derivation
above is self-contained and is what the tests check.)

**Two fidelities** (`mode` coefficient):

- `mode = 1`, **instantaneous**: `M = d d^T` with the live separation. The pair is propagated by the
  engine, so `d` is read from the arena; within a step it is carried along the pair's own two-body
  conic (`integrators.kepler_advance`, `mu_a + mu_b`) to each stage time, from the engine-owned `t0`
  column (the same scheme as `thirdbody.py`'s `staged=1`). This is the full periodic forcing: every
  Fourier harmonic of the inner orbit that the quadrupole carries (`2n` for a circular pair; `n`,
  `2n`, `3n`, ... for an eccentric one) is in it, because it is evaluated, not expanded.
- `mode = 0`, **averaged**: `M = <d d^T>`, the time average over the inner orbit. For an ellipse with
  focus at the origin, semi-major axis `a`, eccentricity `e`, periapsis direction `e_hat` and
  `q_hat = n_hat x e_hat` in the orbit plane, `<d d^T> = (a^2/2)[(1 + 4e^2) e_hat e_hat^T + (1 - e^2)
  q_hat q_hat^T]` (verified numerically against a mean-anomaly average to 6 digits at e = 0..0.7). For
  a circular pair this is exactly an oblate body, `J2 R^2 = mu_r d^2 / (2 M)` about the inner orbit's
  normal: the "ring" the secular theory of hierarchical systems uses. It keeps the secular
  (orbit-averaged) effect and drops every periodic one.

**Configuration** (`validate_coefficients`): the body's Keplerian parent must be a barycentre whose two
active members are exactly `primary` and `secondary` (slots, given by name in sweeps through
`ForceModelSpec.body_coefficients`), both massive; `mode` is 0 or 1; `t0` is engine-owned and refused.
Compose with `point_mass_gravity` (the monopole, the parent's summed `mu`); this model adds only the
quadrupole. Not fused: Cowell runs on the NumPy path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final, Mapping

import numpy as np
from numpy.typing import NDArray

from .custom_types import ScalarSeconds
from .integrators import kepler_advance
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = ["QUADRUPOLE_MODEL", "QUADRUPOLE_PARAM_NAMES", "quadrupole_kernel", "averaged_second_moment"]

QUADRUPOLE_MODEL: Final = "system_quadrupole"
QUADRUPOLE_PARAM_NAMES: Final = ("mode", "primary", "secondary", "t0")
_MODE, _PRIMARY, _SECONDARY, _T0 = 0, 1, 2, 3


def _validate(sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float]) -> None:
    if "t0" in coefficients:
        raise ValueError(f"force model '{QUADRUPOLE_MODEL}': 't0' is engine-owned state, not a coefficient.")
    mode = float(coefficients.get("mode", 0.0))
    if mode not in (0.0, 1.0):
        raise ValueError(f"force model '{QUADRUPOLE_MODEL}': mode={mode!r} must be 0 (averaged) or 1 "
                         f"(instantaneous).")
    for key in ("primary", "secondary"):
        if key not in coefficients:
            raise ValueError(f"force model '{QUADRUPOLE_MODEL}' needs '{key}' (a member slot of the "
                             f"body's parent system); a missing one would silently read slot 0.")
    a, b = int(coefficients["primary"]), int(coefficients["secondary"])
    for body in bodies.tolist():
        s = int(sim.parent_indices[body])
        if not sim.is_system[s]:
            raise ValueError(f"force model '{QUADRUPOLE_MODEL}': slot {body}'s parent is not a barycentre; "
                             f"the quadrupole is of the parent system, about its barycentre.")
        members = np.flatnonzero(sim.active_mask & (sim.body_sys_map == s) & ~sim.is_system)
        massive = set(int(m) for m in members if sim.mu_array[m] > 0.0)
        if massive != {a, b}:
            raise ValueError(f"force model '{QUADRUPOLE_MODEL}': primary/secondary {a}, {b} are not exactly "
                             f"the massive members {sorted(massive)} of slot {body}'s parent system.")


def averaged_second_moment(d: NDArray[np.float64], v: NDArray[np.float64],
                           mu: NDArray[np.float64]) -> NDArray[np.float64]:
    """`<d d^T>` over each pair's orbit, `(k, 3, 3)`, from relative states `(k, 3)` and `mu = mu_a + mu_b`."""
    rn = np.linalg.norm(d, axis=1)
    h = np.cross(d, v)
    n_hat = h / np.linalg.norm(h, axis=1)[:, None]
    e_vec = np.cross(v, h) / mu[:, None] - d / rn[:, None]
    e = np.linalg.norm(e_vec, axis=1)
    a = 1.0 / (2.0 / rn - np.einsum("ij,ij->i", v, v) / mu)
    # A circular pair has no periapsis; any in-plane direction serves, since the tensor is then isotropic.
    e_hat = np.where((e > 1e-12)[:, None], e_vec / np.maximum(e, 1e-300)[:, None], d / rn[:, None])
    q_hat = np.cross(n_hat, e_hat)
    out: NDArray[np.float64] = 0.5 * (a * a)[:, None, None] * (
        (1.0 + 4.0 * e * e)[:, None, None] * np.einsum("ki,kj->kij", e_hat, e_hat)
        + (1.0 - e * e)[:, None, None] * np.einsum("ki,kj->kij", q_hat, q_hat))
    return out


@register_force_model(
    QUADRUPOLE_MODEL,
    param_names=QUADRUPOLE_PARAM_NAMES,
    validate_coefficients=_validate,
    citation=("Multipole expansion of a two-body system about its barycentre (derived in quadrupole.py); "
              "orbit-averaged second moment verified numerically. Murray & Dermott (1999) ch. 6, from "
              "memory, unverified."),
)
def quadrupole_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Add the parent system's quadrupole acceleration on each body to `out[indices]`, km/s^2. See the
    module docstring for the field, the two modes and the staging of the live separation."""
    if indices.size == 0:
        return
    p = params[indices]
    a_idx = p[:, _PRIMARY].astype(np.int64)
    b_idx = p[:, _SECONDARY].astype(np.int64)
    mu_a, mu_b = mu_array[a_idx], mu_array[b_idx]
    mu_pair = mu_a + mu_b
    g_mu_r = mu_a * mu_b / mu_pair

    d = state[b_idx, :3] - state[a_idx, :3]
    v = state[b_idx, 3:] - state[a_idx, 3:]
    live = p[:, _MODE] == 1.0
    m = np.empty((indices.size, 3, 3), dtype=np.float64)
    if live.any():
        rows = np.flatnonzero(live)
        elapsed = float(t) - float(p[rows[0], _T0])           # one t0 per step, for all rows
        d_live = d[rows]
        if elapsed != 0.0:
            d_live, _ = kepler_advance(d[rows], v[rows], elapsed, mu_pair[rows])
        m[rows] = np.einsum("ki,kj->kij", d_live, d_live)
    if (~live).any():
        rows = np.flatnonzero(~live)
        m[rows] = averaged_second_moment(d[rows], v[rows], mu_pair[rows])

    r = state[indices, :3] - state[parent_indices[indices], :3]       # about the barycentre
    r2 = np.einsum("ij,ij->i", r, r)
    mr = np.einsum("kij,kj->ki", m, r)
    rmr = np.einsum("ki,ki->k", r, mr)
    tr = np.einsum("kii->k", m)
    coef = g_mu_r / (2.0 * r2 * r2 * np.sqrt(r2))
    out[indices] += coef[:, None] * (6.0 * mr - 15.0 * (rmr / r2)[:, None] * r + 3.0 * tr[:, None] * r)
