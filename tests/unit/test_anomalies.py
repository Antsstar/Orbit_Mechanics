"""
Anomaly conversion stack: true <-> eccentric/hyperbolic <-> mean.

Kepler's equation has no closed-form inverse, so `mean_to_eccentric` is the only iterative solver in
the engine's analytic path and the only place a convergence failure can appear. These tests pin it
from both directions - round-trip identity, and agreement between the two independent solvers.

The round-trip is the sharper of the two. `eccentric_to_mean` evaluates M = E - e sin E directly,
with no iteration, so it is an independent check rather than the inverse of the same code: if the
solver converges to the wrong root, the residual shows up immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital_engine.exceptions import ConvergenceError
from orbital_engine.utilities import Anomalies

# Newton-Raphson terminates on |dE| < tol. Convergence is quadratic, so the error remaining in E
# after the final accepted step is O(tol^2) ~ 1e-10, and the residual it leaves in
# M = E - e sin E is of the same order. 1e-8 is that bound with two orders of headroom; it is not
# a measured value rounded up. Successive substitution converges only linearly, so it is held to
# the solver tolerance itself rather than its square.
NR_ROUNDTRIP_TOL = 1e-8
SS_ROUNDTRIP_TOL = 1e-4

ELLIPTIC_E = [0.0, 0.01, 0.3, 0.54, 0.55, 0.56, 0.8, 0.94, 0.95, 0.96, 0.99]
HYPERBOLIC_E = [1.05, 1.5, 3.0, 10.0]


# ==================================================================================================
# Round-trip identity
# ==================================================================================================

@pytest.mark.parametrize("e", ELLIPTIC_E)
def test_elliptic_roundtrip_recovers_mean_anomaly(e: float) -> None:
    """
    M -> E -> M across a full revolution. The eccentricity list deliberately straddles the seed-band
    boundaries at 0.55 and 0.95, since a wrong band assignment shows up as a divergent or
    slow-converging solve rather than a wrong answer, and only near the boundary.
    """
    M = np.linspace(-np.pi, np.pi, 61)
    e_arr = np.full_like(M, e)

    E = Anomalies.mean_to_eccentric(M, e_arr)
    M_back = Anomalies.eccentric_to_mean(E, e_arr)

    assert np.max(np.abs(M_back - M)) < NR_ROUNDTRIP_TOL


@pytest.mark.parametrize("e", HYPERBOLIC_E)
def test_hyperbolic_roundtrip_recovers_mean_anomaly(e: float) -> None:
    """M -> H -> M on the hyperbolic branch, where M is unbounded rather than periodic."""
    M = np.linspace(-5.0, 5.0, 41)
    e_arr = np.full_like(M, e)

    H = Anomalies.mean_to_eccentric(M, e_arr)
    M_back = Anomalies.eccentric_to_mean(H, e_arr)

    assert np.max(np.abs(M_back - M)) < NR_ROUNDTRIP_TOL


def test_mixed_elliptic_and_hyperbolic_populations_solve_together() -> None:
    """
    The solver splits the population by branch and solves each separately. A single array holding
    both is the case that exercises the scatter back into the original ordering - if the two halves
    were recombined in the wrong order the round-trip fails even though each branch solved fine.
    """
    e_arr = np.array([0.1, 2.0, 0.7, 1.5, 0.96, 3.0, 0.0, 1.05])
    M = np.array([0.5, 1.2, -2.0, 0.3, 2.9, -1.1, 1.7, 0.05])

    E = Anomalies.mean_to_eccentric(M, e_arr)
    M_back = Anomalies.eccentric_to_mean(E, e_arr)

    assert np.max(np.abs(M_back - M)) < NR_ROUNDTRIP_TOL

    # And each element must match what it gets when solved entirely on its own, which no ordering
    # bug can survive.
    for k in range(e_arr.size):
        alone = Anomalies.mean_to_eccentric(
            np.array([M[k]]), np.array([e_arr[k]]))
        assert E[k] == pytest.approx(float(np.asarray(alone)[0]), abs=1e-12)


# ==================================================================================================
# Solver agreement - the check that catches a branch that is wrong but self-consistent
# ==================================================================================================

@pytest.mark.parametrize("e", [0.0, 0.05, 0.2, 0.4, 0.5])
def test_successive_substitution_agrees_with_newton_raphson(e: float) -> None:
    """
    Two solvers with no shared iteration code must land on the same root.

    This is the test that was missing: the `"S.S"` branch indexed an active-length array with a
    global-length mask, and since nothing exercised it, the defect survived. Successive substitution
    only converges for e well below 1 (its iteration map has derivative e cos E), so the range here
    stops at 0.5.
    """
    M = np.linspace(-np.pi, np.pi, 41)
    e_arr = np.full_like(M, e)

    E_nr = Anomalies.mean_to_eccentric(M, e_arr, solver="N-R")
    E_ss = Anomalies.mean_to_eccentric(M, e_arr, solver="S.S", tol=1e-10, max_ite=5000)

    assert np.max(np.abs(np.asarray(E_nr) - np.asarray(E_ss))) < SS_ROUNDTRIP_TOL


def test_successive_substitution_handles_a_partially_converged_population() -> None:
    """
    Elements converge at different rates, so the active set shrinks unevenly. That shrinking is
    exactly what the old indexing bug mishandled: it sliced a full-length mask against the
    already-reduced active arrays. A spread of eccentricities in one call reproduces it.
    """
    e_arr = np.array([0.0, 0.01, 0.1, 0.25, 0.4, 0.5])
    M = np.full_like(e_arr, 2.5)

    E_ss = Anomalies.mean_to_eccentric(M, e_arr, solver="S.S", tol=1e-10, max_ite=5000)
    M_back = Anomalies.eccentric_to_mean(E_ss, e_arr)

    assert np.max(np.abs(M_back - M)) < SS_ROUNDTRIP_TOL


# ==================================================================================================
# Domain and shape contracts
# ==================================================================================================

def test_scalar_input_returns_a_scalar() -> None:
    """A 0-d input must not come back as an array; callers index the result directly."""
    E = Anomalies.mean_to_eccentric(0.75, 0.2)
    assert isinstance(E, float)
    assert Anomalies.eccentric_to_mean(E, 0.2) == pytest.approx(0.75, abs=NR_ROUNDTRIP_TOL)


def test_broadcast_scalar_eccentricity_against_array_mean_anomaly() -> None:
    """A single eccentricity applied across many anomalies is the common constellation case."""
    M = np.linspace(0.0, 2.0 * np.pi, 17)
    E = np.asarray(Anomalies.mean_to_eccentric(M, 0.3))

    assert E.shape == M.shape
    assert np.max(np.abs(Anomalies.eccentric_to_mean(E, 0.3) - M)) < NR_ROUNDTRIP_TOL


def test_unknown_solver_is_rejected_before_any_work() -> None:
    with pytest.raises(ValueError, match="not recognised"):
        Anomalies.mean_to_eccentric(np.array([0.5]), np.array([0.1]), solver="bisection")


def test_parabolic_eccentricity_is_a_domain_error() -> None:
    """e = 1 has no eccentric anomaly; Barker's equation covers that case instead."""
    with pytest.raises(ValueError, match="[Pp]arabolic"):
        Anomalies.mean_to_eccentric(np.array([0.5]), np.array([1.0]))


def test_negative_eccentricity_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        Anomalies.mean_to_eccentric(np.array([0.5]), np.array([-0.1]))


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_non_convergence_raises_rather_than_returning_a_wrong_answer() -> None:
    """
    Successive substitution on the hyperbolic rearrangement H <- e sinh H - M has derivative
    e cosh H > 1 everywhere, so it is formally divergent. The contract is that this raises rather
    than silently returning the unconverged iterate.

    Regression guard for a specific silent-NaN path. Divergence here overflows sinh to inf and then
    evaluates inf - inf = nan; under a `abs(delta) > tol` convergence test NaN compares False, so
    the element was dropped from the active set and the solver returned NaN *reporting success*.
    The overflow warnings are the expected symptom of the divergence being probed, not a defect.
    """
    with pytest.raises(ConvergenceError, match="hyperbolic"):
        Anomalies.mean_to_eccentric(
            np.array([2.0]), np.array([1.5]), solver="S.S", max_ite=50)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solver_never_returns_nan_while_reporting_success() -> None:
    """
    The general form of the above: whatever the solver returns, it must be finite. A NaN that
    propagates into `coe_to_rv` yields a NaN position, which then contaminates the barycenter and
    every body downstream of it - and no assertion on energy or momentum catches NaN, because NaN
    comparisons are all False. Cheaper to refuse at the source.
    """
    e_arr = np.array([0.0, 0.5, 0.99, 1.5, 8.0])
    M = np.array([0.1, 3.0, -2.5, 4.0, -9.0])

    E = np.asarray(Anomalies.mean_to_eccentric(M, e_arr))
    assert np.all(np.isfinite(E)), f"solver returned non-finite values: {E}"
