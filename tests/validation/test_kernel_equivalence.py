"""
Equivalence of the compiled kernel with the NumPy reference implementation.

`propagators.KeplerianPropagator` is the readable definition of the physics; `kernels.kepler_propagate`
is an optimisation of it. An optimisation that disagrees with its reference is simply wrong, so this
is the gate that makes the compiled path usable at all.

**Why this is the sharp test.** The conserved-quantity suite in `test_barycentric_dynamics.py` checks
that the engine obeys physics, but both implementations could obey physics while disagreeing - a
transposed rotation column still conserves energy and angular momentum magnitude, and a reflex kick
applied with the wrong mass ratio still leaves the centre of mass at the origin. Only an elementwise
comparison against the reference catches those. Conversely this test says nothing about whether the
*reference* is right; that is what the validation suite is for. The two are complementary and neither
substitutes for the other.

Tolerance
---------
Both implementations run the same Newton iteration from the same seed with the same tolerance, so
they differ only in libm rounding of the transcendental calls - under an ulp each, over order tens of
operations. Newton's iteration is self-correcting, and with a 1e-5 stopping tolerance and quadratic
convergence the final residual is ~1e-20, far below double precision, so both land on the true root
to machine epsilon. Expected agreement is therefore ~1e-15 relative, and the bound below is 1e-12 -
three orders of headroom, derived rather than measured.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import kernels, scenarios
from orbital_engine.propagators import KeplerianPropagator
from orbital_engine.simulator import Simulation

# See the module docstring: derived from double-precision rounding, not fitted to observation.
KERNEL_AGREEMENT_REL_TOL = 1e-12

DT = 3600.0


def _snapshot(sim: Simulation) -> dict[str, np.ndarray]:
    return {
        "coe": sim.coe_states.copy(),
        "local": sim.local_states.copy(),
        "global": sim.global_states.copy(),
        "mu": sim.mu_array.copy(),
    }


def _restore(sim: Simulation, snap: dict[str, np.ndarray]) -> None:
    sim.coe_states[:] = snap["coe"]
    sim.local_states[:] = snap["local"]
    sim.global_states[:] = snap["global"]
    sim.mu_array[:] = snap["mu"]


def _run_reference(sim: Simulation, dt: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(steps):
        KeplerianPropagator.propagate(
            dt=dt, primary_states=sim.coe_states, secondary_states=sim.local_states,
            mu_array=sim.mu_array, parent_indices=sim.parent_indices,
            active_mask=sim.active_mask, is_head=sim.is_head, is_system=sim.is_system,
            body_sys_map=sim.body_sys_map, sys_head_map=sim.sys_head_map,
        )
    return sim.local_states.copy(), sim.coe_states.copy()


def _run_kernel(sim: Simulation, dt: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    sib_idx = np.flatnonzero(sim.active_mask & ~sim.is_head).astype(np.int64)
    head_idx = np.flatnonzero(sim.active_mask & sim.is_head).astype(np.int64)
    kick = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    accum = np.zeros((sim.max_capacity, 6), dtype=np.float64)

    for _ in range(steps):
        kernels.kepler_propagate(
            dt, sim.coe_states, sim.local_states, sim.mu_array, sim.parent_indices,
            sim.body_sys_map, sim.sys_head_map, sim.is_system, sib_idx, head_idx, kick, accum,
        )
    return sim.local_states.copy(), sim.coe_states.copy()


def _relative_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Scaled by the reference magnitude, so a 1 km disagreement at 1 AU is not called large."""
    scale = np.maximum(np.abs(a[mask]), 1.0)
    return float(np.max(np.abs(a[mask] - b[mask]) / scale))


SCENARIOS: list[tuple[str, Callable[[Session], Simulation]]] = [
    ("two_body", lambda s: scenarios.two_body(s)),
    ("eccentric", lambda s: scenarios.two_body(s, p=11000.0, e=0.7)),
    ("inclined", lambda s: scenarios.two_body(s, p=9000.0, e=0.3, i=1.1, raan=2.0, arg_pe=0.8)),
    ("sun_earth_moon", lambda s: scenarios.sun_earth_moon(s)),
    ("constellation", lambda s: scenarios.earth_constellation(s, n_sats=24, n_planes=4)),
]


@pytest.mark.parametrize("name,build", SCENARIOS, ids=[n for n, _ in SCENARIOS])
@pytest.mark.parametrize("steps", [1, 50])
def test_kernel_matches_reference_state(
    name: str, build: Callable[[Session], Simulation], steps: int,
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Both implementations advanced from an identical arena must produce identical arenas.

    Run over multiple steps as well as one: a discrepancy in the reflex kick is a fixed fraction of
    the state each step, so it compounds, and a single step can hide it inside the tolerance.
    """
    reference_sim = build(db_session_factory())
    kernel_sim = build(db_session_factory())

    # Guard the premise: if the two builds differ, the comparison afterwards means nothing.
    assert np.array_equal(reference_sim.coe_states, kernel_sim.coe_states)
    assert np.array_equal(reference_sim.local_states, kernel_sim.local_states)

    ref_local, ref_coe = _run_reference(reference_sim, DT, steps)
    ker_local, ker_coe = _run_kernel(kernel_sim, DT, steps)

    active = reference_sim.active_mask

    assert np.all(np.isfinite(ref_local[active])), "reference produced non-finite state"
    assert np.all(np.isfinite(ker_local[active])), "kernel produced non-finite state"

    local_diff = _relative_difference(ref_local, ker_local, active)
    coe_diff = _relative_difference(ref_coe, ker_coe, active)

    assert local_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{name}: local state disagrees by {local_diff:.3e} after {steps} steps")
    assert coe_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{name}: elements disagree by {coe_diff:.3e} after {steps} steps")


def test_comparison_would_detect_a_perturbed_kernel(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Verifies the test above can fail.

    An equivalence test whose tolerance is looser than the effect it is meant to catch passes
    regardless, exactly like the vacuous mask recorded in the engineering log. Nudging one element
    by a relative 1e-9 - a thousand times below the smallest error worth caring about, and a
    thousand times above the 1e-12 bound - must be detected.
    """
    sim = build_sim = scenarios.sun_earth_moon(db_session_factory())
    baseline = _snapshot(sim)

    ref_local, _ = _run_reference(sim, DT, 10)

    _restore(build_sim, baseline)
    build_sim.coe_states[build_sim.name_to_index["Moon"], 0] *= (1.0 + 1e-9)
    ker_local, _ = _run_kernel(build_sim, DT, 10)

    diff = _relative_difference(ref_local, ker_local, sim.active_mask)
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"a deliberately perturbed kernel was not detected (diff {diff:.3e}); "
        f"the tolerance is too loose to be meaningful")


# ==================================================================================================
# Global-state accumulation
# ==================================================================================================

@pytest.mark.parametrize("name,build", SCENARIOS, ids=[n for n, _ in SCENARIOS])
def test_calc_global_paths_agree_exactly(
    name: str, build: Callable[[Session], Simulation],
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The compiled and NumPy paths of `calc_global` perform the same additions in the same
    topological order, so they must agree *bit for bit* - not merely to a tolerance. Floating-point
    addition is deterministic; only a different order or a different operand set can perturb it.
    Asserting exact equality therefore makes this test maximally sensitive.
    """
    sim = build(db_session_factory())
    sim.record_history = False

    # Advance first, so local_states holds something non-trivial rather than the build-time values.
    for _ in range(5):
        sim.step(DT)

    sim.use_compiled_kernel = True
    sim.global_states[:] = 0.0
    sim.calc_global()
    compiled = sim.global_states.copy()

    sim.use_compiled_kernel = False
    sim.global_states[:] = 0.0
    sim.calc_global()
    reference = sim.global_states.copy()

    assert np.array_equal(compiled, reference), (
        f"{name}: max divergence "
        f"{np.max(np.abs(compiled - reference)):.3e} km between calc_global paths")


def test_topological_order_resolves_every_parent_before_its_child(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The compiled path drops the tier boundaries and relies purely on the flattened ordering, so the
    ordering property has to hold on its own. If it ever failed, a body would be accumulated against
    a stale parent - producing a trajectory that is smooth, plausible and wrong.

    Note this checks `body_sys_map`, the kinematic graph, not `parent_indices`.
    """
    sim = scenarios.sun_earth_moon(db_session_factory())

    seen: set[int] = set()
    for position, slot in enumerate(sim._topo_order.tolist()):
        parent = int(sim.body_sys_map[slot])
        if position < sim._n_roots:
            assert parent == slot, f"root slot {slot} is not self-referencing"
        else:
            assert parent in seen, f"slot {slot} resolved before its bubble {parent}"
        seen.add(slot)

    assert seen == set(np.flatnonzero(sim.active_mask).tolist()), (
        "topological order does not cover exactly the active slots")


# ==================================================================================================
# Scalar helpers against closed-form values
# ==================================================================================================

@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("M", [0.0, 0.5, 1.5, 3.0, -2.0])
def test_scalar_kepler_solver_satisfies_keplers_equation(e: float, M: float) -> None:
    """
    Residual of M = E - e sin E, evaluated directly. This checks the solver against the *equation*
    rather than against the NumPy solver, so an error common to both would still be caught.
    """
    E = kernels.solve_kepler_scalar(M, e, 1e-12, 200)
    assert np.isfinite(E)
    assert abs((E - e * np.sin(E)) - M) < 1e-10


@pytest.mark.parametrize("e", [1.5, 3.0])
@pytest.mark.parametrize("M", [0.5, 2.0, -4.0])
def test_scalar_kepler_solver_handles_the_hyperbolic_branch(e: float, M: float) -> None:
    H = kernels.solve_kepler_scalar(M, e, 1e-12, 200)
    assert np.isfinite(H)
    assert abs((e * np.sinh(H) - H) - M) < 1e-9


def test_scalar_coe_to_rv_reproduces_a_circular_orbit() -> None:
    """
    A circular equatorial orbit has a closed form: r = p everywhere, speed = sqrt(mu/p), and the two
    vectors are perpendicular. Independent of both implementations.
    """
    mu, p = 398600.4418, 7000.0
    for theta in np.linspace(0.0, 2.0 * np.pi, 13):
        rx, ry, rz, vx, vy, vz = kernels.coe_to_rv_scalar(p, 0.0, 0.0, 0.0, 0.0, theta, mu)
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])

        assert np.linalg.norm(r) == pytest.approx(p, rel=1e-14)
        assert np.linalg.norm(v) == pytest.approx(np.sqrt(mu / p), rel=1e-14)
        assert float(np.dot(r, v)) == pytest.approx(0.0, abs=1e-9)
        assert rz == pytest.approx(0.0, abs=1e-12), "equatorial orbit left the xy-plane"


def test_scalar_coe_to_rv_places_an_inclined_orbit_correctly() -> None:
    """
    At the ascending node (arg_pe + theta = 0) the body sits on the node line, so its position must
    lie along the RAAN direction in the xy-plane regardless of inclination. This pins the rotation
    columns against each other - a transposed 3-1-3 matrix passes a magnitude check but fails here.
    """
    mu, p, inc, raan = 398600.4418, 8000.0, 0.9, 1.3
    rx, ry, rz, _, _, _ = kernels.coe_to_rv_scalar(p, 0.0, inc, raan, 0.0, 0.0, mu)

    assert rz == pytest.approx(0.0, abs=1e-9), "body is not on the node line"
    assert float(np.arctan2(ry, rx)) == pytest.approx(raan, abs=1e-12)
