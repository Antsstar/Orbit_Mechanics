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

from orbital_engine import geopotential, gravity, kernels, scenarios
from orbital_engine.custom_types import PropagatorType
from orbital_engine.integrators import RK4Integrator
from orbital_engine.propagators import KeplerianPropagator, SecularJ2Propagator
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


# ==================================================================================================
# propagators.SecularJ2Propagator vs kernels.secular_j2_propagate
# ==================================================================================================
#
# Same relationship as KeplerianPropagator / kepler_propagate above: the NumPy implementation is the
# readable definition, the scalar kernel is an optimisation of it, and both share the same underlying
# anomaly stack (Anomalies.true_to_mean/mean_to_true vs. true_to_mean_scalar/mean_to_true_scalar)
# already held equivalent by the tests above - so the same 1e-12 relative bound applies here for the
# same reason: both land on the same Kepler-equation root to within double-precision rounding, and the
# only new arithmetic (the linear rate advance and coe_to_rv/coe_to_rv_scalar) is a handful of
# multiplications, well inside that budget.

SECULAR_DT = 60.0


def _build_secular_j2_sim(
    session: Session, *, n_sats: int = 8, n_planes: int = 2,
) -> tuple[Simulation, list[int]]:
    """A flat LEO constellation with every satellite assigned SECULAR_J2 and Earth's real coefficients -
    exercises several inclinations/RAANs/phases at once, the same reason `earth_constellation` is used
    throughout the suite rather than a single body."""
    sim = scenarios.earth_constellation(session, n_sats=n_sats, n_planes=n_planes, altitude_km=550.0)
    sats = [idx for name, idx in sim.name_to_index.items() if name.startswith("SAT-")]
    sim.set_propagator(sats, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sats


def _run_secular_j2_reference(sim: Simulation, sats: list[int], dt: float, steps: int) -> np.ndarray:
    idx = np.asarray(sats, dtype=np.int64)
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    for _ in range(steps):
        SecularJ2Propagator.propagate(
            dt=dt, coe_states=sim.coe_states, mu_array=sim.mu_array,
            parent_indices=sim.parent_indices, rates=sim._secular_j2_rates, indices=idx, rel_out=rel_out,
        )
    return rel_out[idx].copy()


def _run_secular_j2_kernel(sim: Simulation, sats: list[int], dt: float, steps: int) -> np.ndarray:
    idx = np.asarray(sats, dtype=np.int64)
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    for _ in range(steps):
        kernels.secular_j2_propagate(
            dt, sim.coe_states, sim.mu_array, sim.parent_indices, sim._secular_j2_rates, idx, rel_out,
        )
    return rel_out[idx].copy()


@pytest.mark.parametrize("steps", [1, 50])
def test_secular_j2_kernel_matches_reference_state(
    steps: int, db_session_factory: Callable[[], Session],
) -> None:
    """
    Both implementations advanced from an identical arena (same cached rates, same seed elements) must
    produce identical `coe_states` and identical parent-relative state vectors - run over several steps,
    not just one, for the same reason `test_kernel_matches_reference_state` above does: a discrepancy
    compounds with repeated advances and a single step can hide it inside the tolerance.
    """
    reference_sim, ref_sats = _build_secular_j2_sim(db_session_factory())
    kernel_sim, ker_sats = _build_secular_j2_sim(db_session_factory())

    assert np.array_equal(reference_sim.coe_states, kernel_sim.coe_states)
    assert np.array_equal(reference_sim._secular_j2_rates, kernel_sim._secular_j2_rates)

    ref_rel = _run_secular_j2_reference(reference_sim, ref_sats, SECULAR_DT, steps)
    ker_rel = _run_secular_j2_kernel(kernel_sim, ker_sats, SECULAR_DT, steps)

    assert np.all(np.isfinite(ref_rel)), "reference produced non-finite state"
    assert np.all(np.isfinite(ker_rel)), "kernel produced non-finite state"

    rel_diff = _relative_difference(ref_rel, ker_rel, np.ones(len(ref_sats), dtype=bool))
    coe_diff = _relative_difference(
        reference_sim.coe_states, kernel_sim.coe_states,
        np.isin(np.arange(reference_sim.max_capacity), ref_sats),
    )

    assert rel_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"secular-J2 relative state disagrees by {rel_diff:.3e} after {steps} steps")
    assert coe_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"secular-J2 elements disagree by {coe_diff:.3e} after {steps} steps")


def test_secular_j2_comparison_would_detect_a_perturbed_kernel(
    db_session_factory: Callable[[], Session],
) -> None:
    """Same negative control as `test_comparison_would_detect_a_perturbed_kernel` above, applied to the
    secular-J2 pair: a relative-1e-9 nudge to one body's cached RAAN rate - a thousand times below any
    effect worth caring about and a thousand times above the 1e-12 bound - must be detected."""
    ref_sim, ref_sats = _build_secular_j2_sim(db_session_factory())
    ref_rel = _run_secular_j2_reference(ref_sim, ref_sats, SECULAR_DT, 10)

    ker_sim, ker_sats = _build_secular_j2_sim(db_session_factory())
    ker_sim._secular_j2_rates[ker_sats[0], 0] *= (1.0 + 1e-9)
    ker_rel = _run_secular_j2_kernel(ker_sim, ker_sats, SECULAR_DT, 10)

    diff = _relative_difference(ref_rel, ker_rel, np.ones(len(ref_sats), dtype=bool))
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"a deliberately perturbed rate was not detected (diff {diff:.3e}); "
        f"the tolerance is too loose to be meaningful")


def test_secular_j2_bodies_do_not_change_plain_keplerian_bit_identity(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `CLAUDE.md`: plain Keplerian bodies must be bit-identical to `main` whether or not a secular-J2 body
    exists in the arena, on both paths - the arithmetic risk called out there is that
    `M + (n + dM)*dt` rounds differently from `M + n*dt`, so a shared code path computing both would be
    exactly wrong in the way this test would catch. Here the two propagators are structurally
    independent (`_kepler_sib_idx` excludes SECULAR_J2 slots, exactly as it excludes Cowell ones), so
    this is confirming that isolation holds bit-for-bit, on both the compiled and reference paths.
    """
    for use_compiled in (True, False):
        reference_sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
        reference_sim.record_history = False
        reference_sim.use_compiled_kernel = use_compiled

        mixed_sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
        mixed_sim.record_history = False
        mixed_sim.use_compiled_kernel = use_compiled
        secular_name = "SAT-00-000"
        secular_idx = mixed_sim.name_to_index[secular_name]
        mixed_sim.set_propagator(
            secular_idx, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ,
        )

        for _ in range(30):
            reference_sim.step(DT)
            mixed_sim.step(DT)

        keplerian_names = [n for n in reference_sim.name_to_index if n != secular_name]
        assert len(keplerian_names) > 1, "guard: the scenario must have Keplerian bodies to compare"

        for name in keplerian_names:
            ref_idx = reference_sim.name_to_index[name]
            mix_idx = mixed_sim.name_to_index[name]
            assert np.array_equal(
                reference_sim.global_states[ref_idx], mixed_sim.global_states[mix_idx]
            ), f"{name}'s global state diverged merely because {secular_name!r} carries SECULAR_J2 (use_compiled_kernel={use_compiled})"
            assert np.array_equal(
                reference_sim.local_states[ref_idx], mixed_sim.local_states[mix_idx]
            ), f"{name}'s local state diverged (use_compiled_kernel={use_compiled})"


# ==================================================================================================
# Cowell: RK4 + point_mass_gravity + j2
# ==================================================================================================
#
# `kernels.cowell_rk4_step` is the fused twin of `integrators.RK4Integrator.step` driving
# `Simulation.accelerations` with `point_mass_gravity` and/or `j2`, plus the subtraction `step()` makes
# afterwards. Same tolerance as the Keplerian pair and for the same reason: identical arithmetic in the
# same order, differing only in the summation order inside `np.einsum` for the three squared
# components, so an ulp or so per force evaluation.

COWELL_DT = 60.0


def _build_cowell_constellation(
    session: Session, *, j2_on: str, n_sats: int = 8, n_planes: int = 2,
) -> tuple[Simulation, np.ndarray]:
    """Every satellite on Cowell + `point_mass_gravity`; `j2_on` is "all", "none" or "half" - the
    last leaves the per-body J2 flag genuinely mixed inside one kernel call."""
    sim = scenarios.earth_constellation(session, n_sats=n_sats, n_planes=n_planes, altitude_km=550.0)
    sats = np.asarray(
        [idx for name, idx in sim.name_to_index.items() if name.startswith("SAT-")], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    with_j2 = {"all": sats, "none": sats[:0], "half": sats[::2]}[j2_on]
    if with_j2.size:
        sim.enable_force_model(
            geopotential.J2_MODEL, with_j2.tolist(), j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sats


def _build_cowell_two_body(session: Session, **kwargs: float) -> tuple[Simulation, np.ndarray]:
    sim = scenarios.two_body(session, **kwargs)
    sats = np.asarray([sim.name_to_index["Secondary"]], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    sim.enable_force_model(
        geopotential.J2_MODEL, sats.tolist(), j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sats


def _run_cowell_reference(
    sim: Simulation, idx: np.ndarray, dt: float, steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The definition: `RK4Integrator` against `Simulation.accelerations`, then the subtraction
    `Simulation.step` applies to recover the parent-relative result."""
    integrator = RK4Integrator(sim.max_capacity)
    primaries = sim.parent_indices[idx]
    rel = np.zeros((idx.size, 6))
    for _ in range(steps):
        parent_start = sim.global_states[primaries].copy()
        integrator.step(sim.accelerations, sim.t, sim.global_states, dt, idx, primaries)
        rel = sim.global_states[idx] - parent_start
    return sim.global_states[idx].copy(), rel


def _run_cowell_kernel(
    sim: Simulation, idx: np.ndarray, dt: float, steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    for _ in range(steps):
        kernels.cowell_rk4_step(
            dt, sim.global_states, sim.mu_array, sim.parent_indices, idx,
            sim._cowell_has_point_mass, sim._cowell_has_j2, sim._cowell_j2_params, rel_out,
        )
    return sim.global_states[idx].copy(), rel_out[idx].copy()


COWELL_SCENARIOS: list[tuple[str, Callable[[Session], tuple[Simulation, np.ndarray]]]] = [
    ("point_mass", lambda s: _build_cowell_constellation(s, j2_on="none")),
    ("point_mass+j2", lambda s: _build_cowell_constellation(s, j2_on="all")),
    ("mixed_j2", lambda s: _build_cowell_constellation(s, j2_on="half")),
    ("eccentric+j2", lambda s: _build_cowell_two_body(s, p=11000.0, e=0.7)),
    ("inclined+j2", lambda s: _build_cowell_two_body(s, p=9000.0, e=0.3, i=1.1, raan=2.0, arg_pe=0.8)),
]


@pytest.mark.parametrize("name,build", COWELL_SCENARIOS, ids=[n for n, _ in COWELL_SCENARIOS])
@pytest.mark.parametrize("steps", [1, 50])
def test_cowell_kernel_matches_reference_state(
    name: str, build: Callable[[Session], tuple[Simulation, np.ndarray]], steps: int,
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Direct calls, isolating the kernel from the rest of `step()`: the integrated absolute rows and the
    parent-relative results must agree. The parent is at the origin in every scenario here, so the
    relative result is compared at its own scale with no rounding floor from re-basing (see the
    whole-step test below for the heliocentric case, where that floor exists and is derived).
    """
    reference_sim, ref_idx = build(db_session_factory())
    kernel_sim, ker_idx = build(db_session_factory())
    assert np.array_equal(reference_sim.global_states, kernel_sim.global_states)
    assert kernel_sim._cowell_fused_ok, "guard: this scenario must qualify for the fused kernel"

    ref_abs, ref_rel = _run_cowell_reference(reference_sim, ref_idx, COWELL_DT, steps)
    ker_abs, ker_rel = _run_cowell_kernel(kernel_sim, ker_idx, COWELL_DT, steps)

    assert np.all(np.isfinite(ref_rel)), "reference produced non-finite state"
    assert np.all(np.isfinite(ker_rel)), "kernel produced non-finite state"

    every = np.ones(ref_idx.size, dtype=bool)
    abs_diff = _relative_difference(ref_abs, ker_abs, every)
    rel_diff = _relative_difference(ref_rel, ker_rel, every)
    assert abs_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{name}: Cowell absolute state disagrees by {abs_diff:.3e} after {steps} steps")
    assert rel_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{name}: Cowell parent-relative state disagrees by {rel_diff:.3e} after {steps} steps")


def test_cowell_comparison_would_detect_a_perturbed_kernel(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Same negative control as the two above, for the Cowell pair: a relative-1e-9 nudge to Earth's mu on
    the kernel side changes every acceleration by 1e-9, which over 10 steps of 60 s displaces a 550 km
    satellite by ~0.5 * 8e-12 km/s^2 * (600 s)^2 ~ 1.5e-6 km, 2e-10 of its radius - a thousand times
    below any effect worth caring about and well above the 1e-12 bound - and must be detected.
    """
    ref_sim, ref_idx = _build_cowell_constellation(db_session_factory(), j2_on="all")
    _, ref_rel = _run_cowell_reference(ref_sim, ref_idx, COWELL_DT, 10)

    ker_sim, ker_idx = _build_cowell_constellation(db_session_factory(), j2_on="all")
    ker_sim.mu_array[ker_sim.name_to_index["Earth"]] *= (1.0 + 1e-9)
    _, ker_rel = _run_cowell_kernel(ker_sim, ker_idx, COWELL_DT, 10)

    diff = _relative_difference(ref_rel, ker_rel, np.ones(ref_idx.size, dtype=bool))
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"a deliberately perturbed kernel was not detected (diff {diff:.3e}); "
        f"the tolerance is too loose to be meaningful")


def _build_cowell_moon(session: Session) -> tuple[Simulation, np.ndarray]:
    """The accelerating-parent case: a massless Moon on Cowell + `point_mass_gravity` around an Earth
    that is itself Keplerian about the Sun, so re-basing onto a moving parent is exercised."""
    sim = scenarios.sun_earth_moon(session, moon_mu=0.0)
    moon = np.asarray([sim.name_to_index["Moon"]], dtype=np.int64)
    sim.set_propagator(moon, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, moon.tolist())
    return sim, moon


WHOLE_STEP_SCENARIOS: list[tuple[str, Callable[[Session], tuple[Simulation, np.ndarray]]]] = [
    ("constellation+j2", lambda s: _build_cowell_constellation(s, j2_on="all")),
    ("moon_about_moving_earth", _build_cowell_moon),
]


@pytest.mark.parametrize("name,build", WHOLE_STEP_SCENARIOS, ids=[n for n, _ in WHOLE_STEP_SCENARIOS])
@pytest.mark.parametrize("steps", [1, 50])
def test_cowell_step_paths_agree(
    name: str, build: Callable[[Session], tuple[Simulation, np.ndarray]], steps: int,
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The wired path: `Simulation.step` with `use_compiled_kernel` on and off must leave every active
    body's global state within the bound, so the frontier plot's compiled Cowell tier is the same
    physics as the NumPy one. (The Keplerian and secular twins this also toggles are certified above.)

    The Cowell rows' parent-relative state is checked too, with a floor derived from re-basing rather
    than fitted: each step both paths compute `global[parent] + rel` and later subtract the parent
    again, and because the two sims' parents differ at the Keplerian twins' own ~1e-15, that rounding
    is not identical between them - up to one ulp of the parent's position per step, 8e-14 of the lunar
    distance for a heliocentric Earth and exactly zero for an Earth at the origin.
    """
    numpy_sim, numpy_idx = build(db_session_factory())
    compiled_sim, compiled_idx = build(db_session_factory())
    numpy_sim.use_compiled_kernel = False
    compiled_sim.use_compiled_kernel = True
    assert compiled_sim._cowell_fused_ok, "guard: this scenario must qualify for the fused kernel"
    for sim in (numpy_sim, compiled_sim):
        sim.record_history = False

    for _ in range(steps):
        numpy_sim.step(COWELL_DT)
        compiled_sim.step(COWELL_DT)

    active = numpy_sim.active_mask
    global_diff = _relative_difference(numpy_sim.global_states, compiled_sim.global_states, active)
    assert global_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{name}: global state disagrees by {global_diff:.3e} after {steps} steps")

    parents = numpy_sim.parent_indices[numpy_idx]
    numpy_rel = numpy_sim.global_states[numpy_idx] - numpy_sim.global_states[parents]
    compiled_rel = compiled_sim.global_states[compiled_idx] - compiled_sim.global_states[parents]
    parent_scale = float(np.max(np.abs(numpy_sim.global_states[parents, :3])))
    rel_scale = float(np.min(np.linalg.norm(numpy_rel[:, :3], axis=1)))
    rebase_floor = steps * np.finfo(np.float64).eps * parent_scale / rel_scale
    rel_diff = _relative_difference(numpy_rel, compiled_rel, np.ones(numpy_idx.size, dtype=bool))
    assert rel_diff < KERNEL_AGREEMENT_REL_TOL + rebase_floor, (
        f"{name}: Cowell parent-relative state disagrees by {rel_diff:.3e} after {steps} steps "
        f"(bound {KERNEL_AGREEMENT_REL_TOL:.1e} + re-base floor {rebase_floor:.1e})")


def test_cowell_fused_kernel_is_selected_only_for_point_mass_and_j2(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The fallback is data: a Cowell body carrying any other model sends the set down the NumPy path,
    and enabling that model after the plan was built must re-plan. Observed by spying on the name
    `Simulation.step` calls, so this fails if the wiring silently stops reaching the kernel too.
    """
    import orbital_engine.simulator as simulator_module

    calls: list[int] = []
    real = kernels.cowell_rk4_step

    def spy(*args: object, **kwargs: object) -> None:
        calls.append(1)
        real(*args, **kwargs)

    monkeypatch.setattr(simulator_module, "cowell_rk4_step", spy)

    sim, sats = _build_cowell_constellation(db_session_factory(), j2_on="half")
    sim.record_history = False
    sim.use_compiled_kernel = True
    assert sim._cowell_fused_ok
    sim.step(COWELL_DT)
    assert calls == [1], "the fused kernel was not used for a point_mass_gravity + j2 configuration"

    sim.enable_force_model("test_constant_accel", [int(sats[0])], ax=0.0, ay=0.0, az=0.0)
    assert not sim._cowell_fused_ok, "a foreign force model must disqualify the fused kernel"
    sim.step(COWELL_DT)
    assert calls == [1], "the fused kernel ran despite a model it does not implement"

    sim.use_compiled_kernel = False
    sim.force_model_mask[:] = np.uint64(0)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    assert sim._cowell_fused_ok, "removing the foreign model must re-qualify the fused kernel"
    sim.step(COWELL_DT)
    assert calls == [1], "use_compiled_kernel=False must select the NumPy path"


# ==================================================================================================
# Re-basing Cowell and secular-J2 rows onto their parents' end-of-step states
# ==================================================================================================


def _build_secular_j2_moon(session: Session) -> tuple[Simulation, np.ndarray]:
    """A massless Moon on SECULAR_J2 about an Earth that is itself Keplerian about the Sun."""
    sim = scenarios.sun_earth_moon(session, moon_mu=0.0)
    moon = np.asarray([sim.name_to_index["Moon"]], dtype=np.int64)
    sim.set_propagator(moon, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, moon


def _leo_sat_in_massive_earth_moon(session: Session, propagator: str) -> tuple[Simulation, np.ndarray]:
    """A massless satellite about Earth with the real, massive Moon: its parent (Earth) and its bubble
    (EMB) have different global states, so a re-base that used the wrong one for `local_states` shows."""
    sim = scenarios.sun_earth_moon(session, leo_satellite=True)
    sat = np.asarray([sim.name_to_index["LEO-SAT"]], dtype=np.int64)
    if propagator == "cowell":
        sim.set_propagator(sat, PropagatorType.COWELL)
        sim.enable_force_model(gravity.POINT_MASS_MODEL, sat.tolist())
    else:
        sim.set_propagator(sat, PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    return sim, sat


def _secular_constellation(session: Session) -> tuple[Simulation, np.ndarray]:
    sim, sats = _build_secular_j2_sim(session)
    return sim, np.asarray(sats, dtype=np.int64)


REBASE_SCENARIOS: list[tuple[str, Callable[[Session], tuple[Simulation, np.ndarray]], str]] = [
    ("cowell_constellation+j2", lambda s: _build_cowell_constellation(s, j2_on="all"), "_cowell_rel"),
    ("cowell_moon_about_moving_earth", _build_cowell_moon, "_cowell_rel"),
    ("secular_constellation", _secular_constellation, "_secular_j2_rel"),
    ("secular_moon_about_moving_earth", _build_secular_j2_moon, "_secular_j2_rel"),
    ("cowell_leo_sat_parent_not_bubble", lambda s: _leo_sat_in_massive_earth_moon(s, "cowell"), "_cowell_rel"),
    ("secular_leo_sat_parent_not_bubble", lambda s: _leo_sat_in_massive_earth_moon(s, "secular"), "_secular_j2_rel"),
]


@pytest.mark.parametrize("name,build,rel_attr", REBASE_SCENARIOS, ids=[n for n, _, _ in REBASE_SCENARIOS])
def test_rebase_paths_agree_exactly(
    name: str, build: Callable[[Session], tuple[Simulation, np.ndarray]], rel_attr: str,
    db_session_factory: Callable[[], Session],
) -> None:
    """
    `Simulation._rebase`'s compiled and NumPy paths perform one addition and one subtraction per
    component on the same operands, so like `calc_global` they must agree *bit for bit*. Exercised on
    a state several steps in, with the parent-relative scratch holding that step's real result, on an
    Earth at the origin and on a heliocentric Earth whose 1.5e8 km position is where rounding would
    show if the two ever took a different route.
    """
    sim, idx = build(db_session_factory())
    sim.record_history = False
    sim.use_compiled_kernel = False
    for _ in range(5):
        sim.step(COWELL_DT)
    rel = getattr(sim, rel_attr)
    assert np.any(rel[idx] != 0.0), "guard: the relative scratch must hold a real result"
    assert sim._rebase_compiled_ok, "guard: this scenario must qualify for the compiled re-base"

    # Disturb the rows the re-base writes, so agreement cannot come from both paths being no-ops.
    before = _snapshot(sim)
    sim.global_states[idx] = 0.0
    sim.local_states[idx] = 0.0
    disturbed = _snapshot(sim)

    sim.use_compiled_kernel = True
    sim._rebase(idx, rel)
    compiled_global, compiled_local = sim.global_states.copy(), sim.local_states.copy()

    _restore(sim, disturbed)
    sim.use_compiled_kernel = False
    sim._rebase(idx, rel)
    reference_global, reference_local = sim.global_states.copy(), sim.local_states.copy()

    assert np.array_equal(reference_global[idx], before["global"][idx]), (
        "guard: re-basing the committed state must reproduce it")
    assert np.array_equal(compiled_global, reference_global), (
        f"{name}: max divergence {np.max(np.abs(compiled_global - reference_global)):.3e} km "
        f"between re-base paths (global)")
    assert np.array_equal(compiled_local, reference_local), (
        f"{name}: max divergence {np.max(np.abs(compiled_local - reference_local)):.3e} km "
        f"between re-base paths (local)")

    # The arena invariant, on the NumPy definition: local_states is relative to the bubble.
    bubbles = sim.body_sys_map[idx]
    assert np.array_equal(reference_local[idx], reference_global[idx] - reference_global[bubbles])
    if "parent_not_bubble" in name:
        parents = sim.parent_indices[idx]
        assert not np.array_equal(reference_global[parents, :3], reference_global[bubbles, :3]), (
            "guard: this scenario exists because the parent and bubble must not coincide")


def test_rebase_kernel_is_selected_by_use_compiled_kernel(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring: one compiled re-base per propagator per step when enabled, none when disabled."""
    import orbital_engine.simulator as simulator_module

    calls: list[int] = []
    real = kernels.rebase_relative_states

    def spy(*args: object) -> None:
        calls.append(1)
        real(*args)

    monkeypatch.setattr(simulator_module, "rebase_relative_states", spy)

    sim, sats = _build_cowell_constellation(db_session_factory(), j2_on="all")
    sim.record_history = False
    sim.set_propagator(
        int(sats[0]), PropagatorType.SECULAR_J2, j2=geopotential.EARTH_J2, r_eq=geopotential.EARTH_R_EQ)
    assert sim._cowell_idx.size > 0 and sim._secular_j2_idx.size > 0

    sim.use_compiled_kernel = True
    sim.step(COWELL_DT)
    assert calls == [1, 1], "expected one compiled re-base for the Cowell set and one for the secular set"

    sim.use_compiled_kernel = False
    sim.step(COWELL_DT)
    assert calls == [1, 1], "use_compiled_kernel=False must select the NumPy re-base"
