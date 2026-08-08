"""
Conserved-quantity helpers for propagator validation.

Every propagator this project adds gets checked against the same invariants, so the arithmetic for
them lives here rather than being re-derived per test. These are deliberately independent of the
engine's own maths: they take raw Cartesian state and recompute from first principles, so an error
in `frames.py` cannot hide itself by being used on both sides of an assertion.

Units follow the engine: km, km/s, radians, seconds, mu in km^3/s^2.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrF = NDArray[np.float64]


# ----------------------------------------------------------------------------------------------
# Tolerances
#
# These are budgets, not observations. Each is set roughly three orders of magnitude above the
# residual actually measured on the two-body case, so ordinary float64 noise passes while a real
# regression fails. Tighten them if a propagator turns out to do better; never loosen one to make
# a failing test pass without recording why.
# ----------------------------------------------------------------------------------------------

# Specific orbital energy and angular momentum are exactly conserved in two-body motion, so the
# only error is floating point. Measured ~3e-16 relative over one period; budget 1e-12.
ENERGY_REL_TOL = 1e-12
ANGULAR_MOMENTUM_REL_TOL = 1e-12

# Position residual after an integer number of periods. Measured ~7e-10 km over 1000 steps of a
# 7200 km orbit; budget 1e-5 km (1 cm).
CLOSURE_POS_TOL_KM = 1e-5
CLOSURE_VEL_TOL_KMS = 1e-8

# Centre of mass and total linear momentum of an isolated system are conserved exactly. Measured
# ~2e-19 km over a simulated year; budget 1e-9 km.
COM_DRIFT_TOL_KM = 1e-9
MOMENTUM_DRIFT_TOL_KMS = 1e-9


# ----------------------------------------------------------------------------------------------
# Per-body quantities
# ----------------------------------------------------------------------------------------------

def specific_energy(r: ArrF, v: ArrF, mu: float) -> float:
    """Specific orbital energy, v^2/2 - mu/|r| (km^2/s^2). Constant for unperturbed two-body motion."""
    return float(0.5 * np.dot(v, v) - mu / np.linalg.norm(r))


def specific_angular_momentum(r: ArrF, v: ArrF) -> ArrF:
    """Specific angular momentum vector r x v (km^2/s). Constant in magnitude *and* direction."""
    return np.cross(r, v)


# ----------------------------------------------------------------------------------------------
# System-wide quantities
# ----------------------------------------------------------------------------------------------

def centre_of_mass(mu: ArrF, r: ArrF) -> ArrF:
    """Mass-weighted centroid (km). mu is used as a mass proxy - the common factor G cancels."""
    total = mu.sum()
    if total <= 0.0:
        raise ValueError("Cannot take a centre of mass of zero total mass")
    return np.asarray((mu[:, None] * r).sum(axis=0) / total, dtype=np.float64)


def total_linear_momentum(mu: ArrF, v: ArrF) -> ArrF:
    """Mass-weighted mean velocity (km/s). Zero for a system built about a stationary barycenter."""
    total = mu.sum()
    if total <= 0.0:
        raise ValueError("Cannot take momentum of zero total mass")
    return np.asarray((mu[:, None] * v).sum(axis=0) / total, dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------

def relative_drift(initial: float, final: float) -> float:
    """
    Fractional change |final - initial| / |initial|.

    Raises on a zero reference rather than returning inf, because a zero initial invariant means the
    test case is degenerate and the number that comes out would be meaningless either way.
    """
    if initial == 0.0:
        raise ValueError("Relative drift is undefined against a zero reference value")
    return abs((final - initial) / initial)


def vector_drift(initial: ArrF, final: ArrF) -> float:
    """Euclidean norm of the change in a vector quantity, in the vector's own units."""
    return float(np.linalg.norm(np.asarray(final) - np.asarray(initial)))
