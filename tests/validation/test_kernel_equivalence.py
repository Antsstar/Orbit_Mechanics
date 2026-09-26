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

from orbital_engine import drag, geopotential, gravity, kernels, scenarios, zonal
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED,
)
from orbital_engine.custom_types import PropagatorType
from orbital_engine.msis_bridge import (
    SOLAR_ACTIVITY_HIGH, SOLAR_ACTIVITY_LOW, SOLAR_ACTIVITY_MODERATE,
    msis_coefficients, msis_profile,
)
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
# Cowell: RK4 + point_mass_gravity + j2 + drag + zonal
# ==================================================================================================
#
# `kernels.cowell_rk4_step` is the fused twin of `integrators.RK4Integrator.step` driving
# `Simulation.accelerations` with any subset of `point_mass_gravity`, `j2`, `drag` and `zonal`, plus
# the subtraction `step()` makes afterwards. Same tolerance as the Keplerian pair and for the same
# reason: identical arithmetic in the same order, differing only in the summation order inside
# `np.einsum` for the three squared components, so an ulp or so per force evaluation. The zonal and
# drag sections below say what that means for those terms specifically.

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


def _fused_plan_args(sim: Simulation) -> tuple[object, ...]:
    """The per-body flags, coefficient arrays and density tables `Simulation._refresh_cowell_plan`
    built, in `kernels.cowell_rk4_step`'s argument order (between `indices` and `rel_out`)."""
    tables = sim._cowell_drag_tables
    return (
        sim._cowell_has_point_mass, sim._cowell_has_j2, sim._cowell_j2_params,
        sim._cowell_has_zonal, sim._cowell_zonal_params,
        sim._cowell_has_drag, sim._cowell_drag_params, sim._cowell_drag_table_of,
        tables.altitude_km, tables.density_kg_m3, tables.scale_height_km, tables.n_nodes,
        tables.activity,
    )


def _run_cowell_kernel(
    sim: Simulation, idx: np.ndarray, dt: float, steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    for _ in range(steps):
        stale = kernels.cowell_rk4_step(
            dt, sim.global_states, sim.mu_array, sim.parent_indices, idx, *_fused_plan_args(sim),
            rel_out,
        )
        assert stale == -1, f"the fused kernel reported a stale MSIS plan for slot {stale}"
    return sim.global_states[idx].copy(), rel_out[idx].copy()


_J2 = {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ}
_ZONAL_FULL = {"r_eq": geopotential.EARTH_R_EQ, **zonal.EARTH_ZONALS}
_ZONAL_J3_ONLY = {"r_eq": geopotential.EARTH_R_EQ, "j3": zonal.EARTH_J3}


def _build_cowell_zonal(session: Session, *, variant: str) -> tuple[Simulation, np.ndarray]:
    """
    Cowell + `point_mass_gravity` satellites at 550 km / 53 deg (so `s = z/r` sweeps +/-0.8 and every
    degree's odd and even parts are exercised) carrying `"zonal"`:

    - `"j2+zonal"`: every satellite pm + j2 + J3..J6, all four degrees non-zero.
    - `"j3_only"`: pm + j2 + a zonal row with only J3 non-zero (J4..J6 exactly 0.0).
    - `"zonal_no_j2"`: pm + J3..J6 without j2, so the zonal term is composed straight onto pm.
    - `"mixed"`: all of the above plus a pm-only body inside **one** kernel call, so the per-body
      `has_zonal` / `has_j2` flags and the zonal row indexing are genuinely mixed.
    """
    sim = scenarios.earth_constellation(session, n_sats=8, n_planes=2, altitude_km=550.0)
    sats = np.asarray(
        [idx for name, idx in sim.name_to_index.items() if name.startswith("SAT-")], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    if variant == "j2+zonal":
        sim.enable_force_model(geopotential.J2_MODEL, sats.tolist(), **_J2)
        sim.enable_force_model(zonal.ZONAL_MODEL, sats.tolist(), **_ZONAL_FULL)
    elif variant == "j3_only":
        sim.enable_force_model(geopotential.J2_MODEL, sats.tolist(), **_J2)
        sim.enable_force_model(zonal.ZONAL_MODEL, sats.tolist(), **_ZONAL_J3_ONLY)
    elif variant == "zonal_no_j2":
        sim.enable_force_model(zonal.ZONAL_MODEL, sats.tolist(), **_ZONAL_FULL)
    elif variant == "mixed":
        sim.enable_force_model(geopotential.J2_MODEL, sats[:5].tolist(), **_J2)
        sim.enable_force_model(zonal.ZONAL_MODEL, sats[:3].tolist(), **_ZONAL_FULL)
        sim.enable_force_model(zonal.ZONAL_MODEL, sats[3:5].tolist(), **_ZONAL_J3_ONLY)
        sim.enable_force_model(zonal.ZONAL_MODEL, sats[5:7].tolist(), **_ZONAL_FULL)
        # sats[7]: point_mass_gravity only.
    else:
        raise ValueError(variant)
    return sim, sats


def _build_cowell_zonal_eccentric(session: Session) -> tuple[Simulation, np.ndarray]:
    """An eccentric (perigee 6471 km), steeply inclined orbit on pm + j2 + J3..J6: `r` spans 6471 to
    36667 km, so `(R/r)^n` spans six decades at n = 6, and `s` reaches +/-0.89."""
    sim, sats = _build_cowell_two_body(session, p=11000.0, e=0.7, i=1.1, raan=2.0, arg_pe=0.8)
    sim.enable_force_model(zonal.ZONAL_MODEL, sats.tolist(), **_ZONAL_FULL)
    return sim, sats


# Drag at B = 0.2 m^2/kg - four times a typical satellite's - so the term moves the state enough for
# the state comparison to see it (the drag section below quantifies that). Co-rotating atmosphere.
_DRAG_COMMON = {"ballistic_coeff": 0.2, "r_ref": geopotential.EARTH_R_EQ, "omega": drag.EARTH_OMEGA}
_DRAG_EXPONENTIAL = {"density_model": DENSITY_MODEL_EXPONENTIAL, "rho0": 2.418e-11, "h0": 300.0,
                     "scale_height": 53.628}
_DRAG_LAYERED = {"density_model": DENSITY_MODEL_LAYERED}


def _msis_law(level: str) -> dict[str, float]:
    """MSIS drag coefficients at an ECSS level; skips the test when the `[msis]` extra is absent."""
    pytest.importorskip("pymsis")
    activity = {"low": SOLAR_ACTIVITY_LOW, "moderate": SOLAR_ACTIVITY_MODERATE,
                "high": SOLAR_ACTIVITY_HIGH}[level]
    return msis_coefficients(activity)


def _build_cowell_drag(session: Session, *, variant: str) -> tuple[Simulation, np.ndarray]:
    """
    Cowell + `point_mass_gravity` satellites 300 km above `scenarios.EARTH_RADIUS` (293 km above
    `EARTH_R_EQ`) at 51.6 deg, carrying `"drag"`:

    - `"exponential"` / `"layered"` / `"msis"`: every satellite on one density law.
    - `"msis_two_triples"`: MSIS moderate and MSIS high alternating, so the plan stacks two profiles
      and the kernel reads both within one call.
    - `"mixed"`: the laws crossed with j2 and zonal inside **one** kernel call - layered + j2,
      exponential + j2 + zonal, MSIS moderate + j2 + zonal, MSIS high alone, an exponential row with
      `scale_height = 0` (drag's silent zero) + j2, a drag-free pm + j2 body, layered + zonal without
      j2, and MSIS low + j2 - three MSIS triples in all.
    """
    sim = scenarios.earth_constellation(session, n_sats=8, n_planes=2, altitude_km=300.0,
                                        inclination_deg=51.6)
    sats = np.asarray(
        [idx for name, idx in sim.name_to_index.items() if name.startswith("SAT-")], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    every = sats.tolist()
    if variant == "exponential":
        sim.enable_force_model(drag.DRAG_MODEL, every, **_DRAG_COMMON, **_DRAG_EXPONENTIAL)
    elif variant == "layered":
        sim.enable_force_model(drag.DRAG_MODEL, every, **_DRAG_COMMON, **_DRAG_LAYERED)
    elif variant == "msis":
        sim.enable_force_model(drag.DRAG_MODEL, every, **_DRAG_COMMON, **_msis_law("moderate"))
    elif variant == "msis_two_triples":
        sim.enable_force_model(drag.DRAG_MODEL, sats[::2].tolist(), **_DRAG_COMMON, **_msis_law("moderate"))
        sim.enable_force_model(drag.DRAG_MODEL, sats[1::2].tolist(), **_DRAG_COMMON, **_msis_law("high"))
    elif variant == "mixed":
        s = [int(i) for i in sats]
        sim.enable_force_model(geopotential.J2_MODEL, [s[0], s[1], s[2], s[4], s[5], s[7]], **_J2)
        sim.enable_force_model(zonal.ZONAL_MODEL, [s[1], s[2], s[6]], **_ZONAL_FULL)
        sim.enable_force_model(drag.DRAG_MODEL, [s[0], s[6]], **_DRAG_COMMON, **_DRAG_LAYERED)
        sim.enable_force_model(drag.DRAG_MODEL, s[1], **_DRAG_COMMON, **_DRAG_EXPONENTIAL)
        sim.enable_force_model(drag.DRAG_MODEL, s[2], **_DRAG_COMMON, **_msis_law("moderate"))
        sim.enable_force_model(drag.DRAG_MODEL, s[3], **_DRAG_COMMON, **_msis_law("high"))
        sim.enable_force_model(drag.DRAG_MODEL, s[4], **_DRAG_COMMON,
                               **{**_DRAG_EXPONENTIAL, "scale_height": 0.0})
        sim.enable_force_model(drag.DRAG_MODEL, s[7], **_DRAG_COMMON, **_msis_law("low"))
        # s[5]: point_mass_gravity + j2, no drag.
    else:
        raise ValueError(variant)
    return sim, sats


def _build_cowell_drag_eccentric(session: Session, *, law: str) -> tuple[Simulation, np.ndarray]:
    """One satellite on pm + j2 + drag whose orbit spans 250 km to 1500 km above `EARTH_R_EQ`
    (p = 7199.3 km, e = 0.0862, 63 deg): over the 500-step test it crosses every layered band edge
    from 250 to 1000 km and the extrapolated top band, and every MSIS node above 250 km."""
    sim, sats = _build_cowell_two_body(session, p=7199.3, e=0.0862, i=1.1, raan=2.0, arg_pe=0.8)
    coefficients = _DRAG_LAYERED if law == "layered" else _msis_law("moderate")
    sim.enable_force_model(drag.DRAG_MODEL, sats.tolist(), **_DRAG_COMMON, **coefficients)
    return sim, sats


COWELL_SCENARIOS: list[tuple[str, Callable[[Session], tuple[Simulation, np.ndarray]]]] = [
    ("point_mass", lambda s: _build_cowell_constellation(s, j2_on="none")),
    ("point_mass+j2", lambda s: _build_cowell_constellation(s, j2_on="all")),
    ("mixed_j2", lambda s: _build_cowell_constellation(s, j2_on="half")),
    ("eccentric+j2", lambda s: _build_cowell_two_body(s, p=11000.0, e=0.7)),
    ("inclined+j2", lambda s: _build_cowell_two_body(s, p=9000.0, e=0.3, i=1.1, raan=2.0, arg_pe=0.8)),
    ("j2+zonal", lambda s: _build_cowell_zonal(s, variant="j2+zonal")),
    ("j2+zonal_j3_only", lambda s: _build_cowell_zonal(s, variant="j3_only")),
    ("zonal_no_j2", lambda s: _build_cowell_zonal(s, variant="zonal_no_j2")),
    ("mixed_zonal", lambda s: _build_cowell_zonal(s, variant="mixed")),
    ("eccentric_inclined+j2+zonal", _build_cowell_zonal_eccentric),
    ("drag_exponential", lambda s: _build_cowell_drag(s, variant="exponential")),
    ("drag_layered", lambda s: _build_cowell_drag(s, variant="layered")),
    ("drag_msis", lambda s: _build_cowell_drag(s, variant="msis")),
    ("drag_msis_two_triples", lambda s: _build_cowell_drag(s, variant="msis_two_triples")),
    ("drag_mixed", lambda s: _build_cowell_drag(s, variant="mixed")),
    ("drag_layered_eccentric+j2", lambda s: _build_cowell_drag_eccentric(s, law="layered")),
    ("drag_msis_eccentric+j2", lambda s: _build_cowell_drag_eccentric(s, law="msis")),
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


COWELL_LONG_STEPS = 500


@pytest.mark.parametrize("name,build", COWELL_SCENARIOS, ids=[n for n, _ in COWELL_SCENARIOS])
def test_cowell_kernel_matches_reference_over_several_orbits(
    name: str, build: Callable[[Session], tuple[Simulation, np.ndarray]],
    db_session_factory: Callable[[], Session],
) -> None:
    """
    500 steps of 60 s - 5.2 orbits at 550 km - with the difference measured per body as `|dr|/|r|` and
    `|dv|/|v|`, not elementwise.

    **Why not elementwise here.** `_relative_difference` divides each component by `max(|a|, 1)`, and
    over several orbits some position component is always passing near zero - measured: the elementwise
    maximum at 500 steps sits on a component of 300 km within a 6921 km orbit, where one ulp of the
    orbit-scale arithmetic is already 1e-13 of the component. That is a property of the metric, and the
    pre-existing point_mass scenario shows it too (1.1e-12 elementwise at 500 steps against 9.9e-14 by
    norm), so the elementwise test stays at the 1 and 50 steps it was derived for.

    **Bound, derived.** Each step both sides round differently by ~eps = 1.1e-16 of the state (the
    einsum order, and a J2/zonal ulp). Those roundings random-walk the semi-major axis, `da/a ~
    sqrt(N) eps`, and Kepler shear turns that into along-track phase `~ n t sqrt(N) eps` (the integral of
    `(3/2) n da/a` over a random walk). At N = 500, `n t` = 32.8 rad: **8e-14**. Measured 6e-14 to
    1.4e-13 across these scenarios, zonal or not - the zonal term adds no divergence of its own. The
    bound is 1e-12: an order above the random-walk estimate, and below the coherent worst case
    (`1.5 n t N eps` = 2.7e-12), so a systematic per-step bias the size of an ulp would fail it.
    """
    reference_sim, ref_idx = build(db_session_factory())
    kernel_sim, ker_idx = build(db_session_factory())
    assert kernel_sim._cowell_fused_ok, "guard: this scenario must qualify for the fused kernel"

    _, ref_rel = _run_cowell_reference(reference_sim, ref_idx, COWELL_DT, COWELL_LONG_STEPS)
    _, ker_rel = _run_cowell_kernel(kernel_sim, ker_idx, COWELL_DT, COWELL_LONG_STEPS)
    assert np.all(np.isfinite(ref_rel)) and np.all(np.isfinite(ker_rel))

    dr = float(np.max(np.linalg.norm(ref_rel[:, :3] - ker_rel[:, :3], axis=1)
                      / np.linalg.norm(ref_rel[:, :3], axis=1)))
    dv = float(np.max(np.linalg.norm(ref_rel[:, 3:] - ker_rel[:, 3:], axis=1)
                      / np.linalg.norm(ref_rel[:, 3:], axis=1)))
    assert dr < KERNEL_AGREEMENT_REL_TOL, f"{name}: |dr|/|r| = {dr:.3e} after {COWELL_LONG_STEPS} steps"
    assert dv < KERNEL_AGREEMENT_REL_TOL, f"{name}: |dv|/|v| = {dv:.3e} after {COWELL_LONG_STEPS} steps"


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


# --- The fused zonal term ---------------------------------------------------------------------------
#
# Why 1e-12 holds for J3..J6. `kernels._cowell_accel`'s zonal block is `zonal.zonal_kernel` written one
# scalar at a time: the same two Legendre recursions (Bonnet's for P_n, `P_n' = n P_{n-1} + s
# P_{n-1}'`) run from the same seeds over the same n = 2..7 with the same integer coefficients
# (converted to float exactly), each product and sum evaluated left to right as NumPy evaluates the
# array expression, and the final projection `((k * radial) * x) * inv_r` / `k * (radial * s - axial)`
# unchanged. Two places differ:
#
# 1. `r^2`. The reference forms it with `np.einsum("ij,ij->i")`, whose inner summation order is
#    NumPy's to choose; the twin writes `x*x + y*y + z*z`. At most an ulp or two of `r^2`, which
#    propagates to a few ulp of the term - ~1e-15 relative.
# 2. Composition. `compose_accelerations` adds the models into `out` in registration order
#    (`point_mass_gravity`, `j2`, ..., `zonal`), and the twin adds them in that same order into a
#    scalar starting at 0.0; with the fixture and other models absent from the fused set, the chain of
#    additions is the same. (Were it not, the difference would be one rounding of the sum.)
#
# Everything else, RK4 included, is the arithmetic the pm + j2 pair above already certifies. So the
# expected disagreement is ~1e-15 of the state per step - measured 5.6e-15 to 4.1e-14 of the term's
# scale at field level, with 26-36 of 240 components not bit-identical - accumulating along-track
# through Kepler shear over many steps (`test_cowell_kernel_matches_reference_over_several_orbits`).
#
# **What the state comparison cannot see.** The J3..J6 acceleration is ~5e-8 km/s^2 against ~8e-3 for
# the central term, and over 50 steps of 60 s it moves a 550 km satellite by ~0.2 km, 3e-5 of its
# radius. A relative error `delta` in the zonal term therefore shows in the state as ~3e-5 delta, and is
# detected by a 1e-12 state bound only above delta ~ 3e-8. That is why the zonal term also has a
# field-level comparison below, where the bound applies to the term itself and a 1e-9 perturbation of
# one coefficient is visible (its negative control proves it), and why
# `test_zonal_term_is_visible_to_the_state_comparison` guards that the state tests are not vacuous.

_NO_TABLES = drag.density_tables(np.empty((0, 3)))[0]


def _fused_accel(
    px: float, py: float, pz: float, cx: float, cy: float, cz: float,
    pv: tuple[float, float, float] = (0.0, 0.0, 0.0), cv: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *, mu: float = 0.0, has_zonal: bool = False, zonal_params: np.ndarray | None = None, row: int = 0,
    drag_row: np.ndarray | None = None, drag_table: int = kernels.LAYERED_TABLE_ROW,
    tables: drag.DensityTables = _NO_TABLES,
) -> tuple[float, float, float]:
    """`kernels._cowell_accel` with every term off except the zonal term or the drag term - the
    field-level twin both the zonal and the drag comparisons below call. `drag_row` is one row of
    `force_model_params["drag"]`, decoded exactly as `cowell_rk4_step` decodes it."""
    zp = np.zeros((1, len(zonal.ZONAL_PARAM_NAMES))) if zonal_params is None else zonal_params
    has_drag = drag_row is not None
    d = np.zeros(len(drag.DRAG_PARAM_NAMES)) if drag_row is None else drag_row
    law = kernels._drag_law(float(d[drag.DENSITY_MODEL_COL]))
    return kernels._cowell_accel(
        px, py, pz, cx, cy, cz, pv[0], pv[1], pv[2], cv[0], cv[1], cv[2], mu, mu,
        False, False, 0.0, 0.0,
        has_drag, law, float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5]),
        drag_table if law == kernels.DRAG_LAW_MSIS else kernels.LAYERED_TABLE_ROW,
        tables.altitude_km, tables.density_kg_m3, tables.scale_height_km, tables.n_nodes,
        has_zonal, zp, row)


def _zonal_field_points() -> np.ndarray:
    """Relative positions spanning LEO to GEO, both hemispheres, the equator, a hair off it, the poles
    and points near them - every sign combination of `s` and of each `P_n'`."""
    lat_deg = [0.0, 1e-7, 3.0, -3.0, 17.0, -25.0, 35.26, -39.23, 49.1, -55.0, 63.43, -70.0, 85.0,
               -89.99, 90.0, -90.0]
    lon_deg = [0.0, 31.0, 77.0, 90.0, 141.0, 180.0, 203.0, 266.0, 299.0, 330.0, 12.0, 58.0, 160.0,
               240.0, 0.0, 0.0]
    pts = []
    for r in (6500.0, 6921.0, 8000.0, 26560.0, 42164.0):
        for la, lo in zip(np.radians(lat_deg), np.radians(lon_deg)):
            pts.append([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)])
    return np.asarray(pts)


ZONAL_ROWS: list[tuple[str, dict[str, float]]] = [
    ("j3..j6", _ZONAL_FULL),
    ("j3_only", _ZONAL_J3_ONLY),
    ("j6_only", {"r_eq": geopotential.EARTH_R_EQ, "j6": zonal.EARTH_J6}),
    ("j4_j5", {"r_eq": geopotential.EARTH_R_EQ, "j4": zonal.EARTH_J4, "j5": zonal.EARTH_J5}),
]


def _zonal_params(coefficients: dict[str, float], n: int) -> np.ndarray:
    params = np.zeros((n, len(zonal.ZONAL_PARAM_NAMES)))
    for col, name in enumerate(zonal.ZONAL_PARAM_NAMES):
        params[:, col] = coefficients.get(name, 0.0)
    return params


def _zonal_fields(
    rel: np.ndarray, coefficients: dict[str, float], parent: np.ndarray, *, perturb_j3: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    `(reference, twin, scale)` at every relative position in `rel`, with the parent at `parent`.
    Reference: `zonal.zonal_kernel` over an arena whose slot 0 is the parent. Twin: `_cowell_accel`
    with only its zonal flag set. `scale` is the term's natural size `sum_n mu |J_n| R^n / r^(n+2)`,
    which `|a|` never falls far below (`test_zonal.py`'s module docstring: its minimum over latitude
    is 1.5..2.1 times the scale). `perturb_j3` multiplies J3 on the twin's side only.
    """
    mu = scenarios.MU_EARTH
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[0, :3] = parent
    state[1:, :3] = parent + rel
    mu_array = np.zeros(n + 1)
    mu_array[0] = mu
    params = _zonal_params(coefficients, n + 1)
    ref = np.zeros((n + 1, 3))
    zonal.zonal_kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, mu_array,
                       np.zeros(n + 1, dtype=np.int32), params, ref)

    twin_params = params.copy()
    twin_params[:, 1] *= perturb_j3
    twin = np.zeros((n, 3))
    px, py, pz = (float(c) for c in state[0, :3])
    for k in range(n):
        cx, cy, cz = (float(c) for c in state[k + 1, :3])
        twin[k] = _fused_accel(px, py, pz, cx, cy, cz, mu=mu, has_zonal=True, zonal_params=twin_params,
                               row=k + 1)

    r = np.linalg.norm(rel, axis=1)
    r_eq = coefficients["r_eq"]
    scale = sum(
        mu * abs(coefficients.get(f"j{deg}", 0.0)) * r_eq ** deg / r ** (deg + 2) for deg in zonal.ZONAL_DEGREES)
    return ref[1:], twin, np.asarray(scale)


PARENTS: list[tuple[str, np.ndarray]] = [
    ("origin", np.zeros(3)),
    ("heliocentric", np.array([1.495978707e8, -2.3e6, 4.1e4])),
]


@pytest.mark.parametrize("row_name,coefficients", ZONAL_ROWS, ids=[n for n, _ in ZONAL_ROWS])
@pytest.mark.parametrize("parent_name,parent", PARENTS, ids=[n for n, _ in PARENTS])
def test_fused_zonal_term_matches_zonal_kernel(
    row_name: str, coefficients: dict[str, float], parent_name: str, parent: np.ndarray,
) -> None:
    """
    The J3..J6 term alone, compiled against `zonal.zonal_kernel`, elementwise to 1e-12 of the term's
    own scale - the sharp test the state comparison cannot be (see the section comment). The
    heliocentric parent puts the body-minus-parent subtraction on a 1 AU grid, where both sides make
    the same rounding of `rel` because they perform the same subtraction.
    """
    rel = _zonal_field_points()
    ref, twin, scale = _zonal_fields(rel, coefficients, parent)
    assert np.all(np.isfinite(twin)) and np.all(np.isfinite(ref))
    diff = float(np.max(np.abs(twin - ref) / scale[:, None]))
    assert diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{row_name} / {parent_name}: fused zonal term disagrees with zonal_kernel by {diff:.3e} of scale")


def test_fused_zonal_comparison_would_detect_a_perturbed_coefficient() -> None:
    """
    Negative control for the field-level test: J3 scaled by (1 + 1e-9) on the twin's side only. J3 is
    the largest of the four terms at every radius here (its share of `scale` is 0.52 at 6500 km
    and grows outward), and `|a_3|` is at least 1.5 times its own scale, so the change is >= ~8e-10 of
    the combined scale somewhere - three orders above the bound, and must be detected.
    """
    rel = _zonal_field_points()
    ref, twin, scale = _zonal_fields(rel, _ZONAL_FULL, np.zeros(3), perturb_j3=1.0 + 1e-9)
    diff = float(np.max(np.abs(twin - ref) / scale[:, None]))
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"a relative-1e-9 change to J3 was not detected (diff {diff:.3e}); the bound is vacuous")


def test_zonal_term_is_visible_to_the_state_comparison(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The Cowell state tests above certify the fused zonal path only if dropping the zonal term would
    fail them. Over 50 steps of 60 s the J3..J6 term moves each 550 km satellite by ~0.2 km (section
    comment), ~3e-5 of the radius; the same compiled run with every `has_zonal` flag cleared must
    therefore differ from the zonal run by far more than the 1e-12 bound - asserted at 1e-6.
    """
    with_sim, idx = _build_cowell_zonal(db_session_factory(), variant="j2+zonal")
    without_sim, _ = _build_cowell_zonal(db_session_factory(), variant="j2+zonal")
    without_sim._cowell_has_zonal[:] = False
    _, with_rel = _run_cowell_kernel(with_sim, idx, COWELL_DT, 50)
    _, without_rel = _run_cowell_kernel(without_sim, idx, COWELL_DT, 50)
    diff = _relative_difference(with_rel, without_rel, np.ones(idx.size, dtype=bool))
    assert diff > 1e-6, f"the zonal term moved the state by only {diff:.3e}; the state test is blind to it"


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
    ("constellation+mixed_zonal", lambda s: _build_cowell_zonal(s, variant="mixed")),
    ("moon_about_moving_earth", _build_cowell_moon),
    ("constellation+mixed_drag", lambda s: _build_cowell_drag(s, variant="mixed")),
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


def test_cowell_fused_kernel_is_selected_only_for_point_mass_j2_drag_and_zonal(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The fallback is data: a Cowell body carrying any model outside `{point_mass_gravity, j2, drag,
    zonal}` sends the set down the NumPy path, and enabling that model after the plan was built must
    re-plan.
    Observed by spying on the name `Simulation.step` calls, so this fails if the wiring silently stops
    reaching the kernel too.
    """
    import orbital_engine.simulator as simulator_module

    calls: list[int] = []
    real = kernels.cowell_rk4_step

    def spy(*args: object, **kwargs: object) -> int:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(simulator_module, "cowell_rk4_step", spy)

    sim, sats = _build_cowell_constellation(db_session_factory(), j2_on="half")
    sim.record_history = False
    sim.use_compiled_kernel = True
    assert sim._cowell_fused_ok
    sim.step(COWELL_DT)
    assert calls == [1], "the fused kernel was not used for a point_mass_gravity + j2 configuration"

    sim.enable_force_model(zonal.ZONAL_MODEL, [int(sats[1])], **_ZONAL_FULL)
    assert sim._cowell_fused_ok, "zonal is fused; enabling it must keep the compiled plan"
    assert sim._cowell_has_zonal[sats[1]] and not sim._cowell_has_zonal[sats[0]]
    assert sim._cowell_zonal_params is sim.force_model_params[zonal.ZONAL_MODEL]
    sim.step(COWELL_DT)
    assert calls == [1, 1], "the fused kernel was not used for a point_mass_gravity + j2 + zonal configuration"

    sim.enable_force_model(drag.DRAG_MODEL, [int(sats[1]), int(sats[2])], **_DRAG_COMMON, **_DRAG_LAYERED)
    assert sim._cowell_fused_ok, "drag is fused; enabling it must keep the compiled plan"
    assert sim._cowell_has_drag[sats[1]] and not sim._cowell_has_drag[sats[0]]
    assert sim._cowell_drag_params is sim.force_model_params[drag.DRAG_MODEL]
    sim.step(COWELL_DT)
    assert calls == [1, 1, 1], "the fused kernel was not used for a configuration with drag"

    sim.enable_force_model("test_constant_accel", [int(sats[0])], ax=0.0, ay=0.0, az=0.0)
    assert not sim._cowell_fused_ok, "a foreign force model must disqualify the fused kernel"
    sim.step(COWELL_DT)
    assert calls == [1, 1, 1], "the fused kernel ran despite a model it does not implement"

    sim.use_compiled_kernel = False
    sim.force_model_mask[:] = np.uint64(0)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    assert sim._cowell_fused_ok, "removing the foreign model must re-qualify the fused kernel"
    sim.step(COWELL_DT)
    assert calls == [1, 1, 1], "use_compiled_kernel=False must select the NumPy path"


# --- The fused drag term ----------------------------------------------------------------------------
#
# Why 1e-12 holds for drag. `kernels._cowell_accel`'s drag block is `drag.drag_kernel` written one row
# at a time: `rel = candidate - parent` for position *and* velocity, `v_rel = (vx + w y, vy - w x,
# vz)`, `|v_rel|` summed left to right, `h = sqrt(r^2) - r_ref`, the three masked density terms summed
# in the reference's order (`rho0 * factor + layered + msis`, the absent two exactly +0.0), and `k =
# ((0.5 rho) B) 1e3 |v_rel|` subtracted component-wise. The band lookup of the layered table and of
# every MSIS profile is a scalar binary search returning `np.searchsorted(side="right") - 1`, clipped
# the same way, on copies of the same float64 tables. Three places can differ:
#
# 1. `r^2`, from `np.einsum` there and `x*x + y*y + z*z` here: an ulp of `r^2` is half an ulp of `r`
#    (~5e-13 km at 7000 km), which the exponential turns into `dh / H` of the density - **1e-13** at the
#    smallest scale height anywhere in the tables (5.38 km, the 90 km band), ~1e-14 in LEO.
# 2. `exp`: NumPy's SIMD loop against the C library's, each under an ulp - ~2e-16.
# 3. Composition: drag is bit 2, between `j2` and `zonal`, and the twin adds it there.
#
# So the field-level agreement is <= ~1e-13 relative to `|a_drag|` in the worst band, far less in LEO,
# and the bound is 1e-12. The density law never changes the arithmetic, only which table row is read.
#
# **Velocity is new to the fused kernel.** `RK4Integrator` writes `state[primaries, 3:] + v_k` into
# every candidate row before each stage (k = 0 committed, then v1, v2, v3), and `drag_kernel` takes
# the parent's velocity back off it; `cowell_rk4_step` passes exactly that stage velocity,
# `pv + v_k`, and the drag block subtracts `pv` again. Evaluating every stage at `v0` instead is a
# first-order error in the drag stage accelerations - `test_drag_stage_velocity_is_visible_to_the_state_
# comparison` shows the state tests see it.
#
# **What the state comparison sees.** At 293 km and B = 0.2 m^2/kg drag is ~1.3e-7 km/s^2 against
# ~8.9e-3 for gravity, and over 50 steps of 60 s it displaces a satellite by ~0.6 km, ~1e-4 of its
# radius, so a relative error `delta` in the drag term shows in the state as ~1e-4 delta: a 1e-12 state
# bound sees drag errors above ~1e-8. That is why drag, like zonal, also has the field-level comparison
# below, whose negative control proves a 1e-9 change to `B` is visible.

_DRAG_R_REF_EXACT = 6378.0      # integer, so on-axis `h = x - r_ref` is exact and lands *on* a boundary


def _drag_field_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    `(rel_pos, rel_vel, params)` - one row per evaluation point:

    - layered: every one of the 28 base altitudes **exactly** (on the +x axis with an integer `r_ref`,
      so `sqrt(x*x) - r_ref` is the base altitude to the bit), one ulp either side of each, the bottom
      band extrapolated (-0.5, -5, -50 km) and the top band (1000 exactly, 1200, 2500, 20000 km);
    - MSIS moderate and MSIS high: nodes exactly (0, 1, 150, 199, 200, 202, 400, 998, 1000 km), one ulp
      either side of 200 km (where the grid spacing changes), a mid-band point (201 km), both
      extrapolated regions (-3, 1100, 1600 km);
    - exponential: around `h0`, plus `scale_height` = 0 and < 0 (the silent zero);
    - generic directions at LEO altitudes for every law, with `omega` = Earth's, 0 and negative, so the
      co-rotation term is exercised with every sign;
    - zero separation under each law (exactly zero, never `exp` of `-r_ref / H`).
    """
    rows_r: list[list[float]] = []
    rows_v: list[list[float]] = []
    rows_p: list[np.ndarray] = []
    width = len(drag.DRAG_PARAM_NAMES)

    def add(r: list[float], v: list[float], **coefficients: float) -> None:
        row = np.zeros(width)
        row[0] = 0.2
        row[4] = _DRAG_R_REF_EXACT
        row[5] = drag.EARTH_OMEGA
        for name, value in coefficients.items():
            row[drag.DRAG_PARAM_NAMES.index(name)] = value
        rows_r.append(r)
        rows_v.append(v)
        rows_p.append(row)

    def on_axis(h_values: list[float], **coefficients: float) -> None:
        for h in h_values:
            x = _DRAG_R_REF_EXACT + h
            for xx in (x, np.nextafter(x, -np.inf), np.nextafter(x, np.inf)):
                add([float(xx), 0.0, 0.0], [0.013, 7.6, 1.1], **coefficients)

    layered = {"density_model": DENSITY_MODEL_LAYERED}
    on_axis([float(h) for h in BASE_ALTITUDE_KM] + [-0.5, -5.0, -50.0, 1200.0, 2500.0, 20000.0], **layered)

    msis_levels: list[dict[str, float]] = []
    try:
        import pymsis  # noqa: F401
    except ImportError:
        pass
    else:
        msis_levels = [msis_coefficients(SOLAR_ACTIVITY_MODERATE), msis_coefficients(SOLAR_ACTIVITY_HIGH)]
    for level in msis_levels:
        for f107, f107a, ap in [(level["f107"], level["f107a"], level["ap"])]:
            msis_profile(f107, f107a, ap)           # configuration time: evaluated (or memoised) here
        on_axis([0.0, 1.0, 150.0, 199.0, 200.0, 201.0, 202.0, 400.0, 998.0, 1000.0, -3.0, 1100.0, 1600.0],
                **level)

    exponential = {"density_model": DENSITY_MODEL_EXPONENTIAL, "rho0": 2.418e-11, "h0": 300.0,
                   "scale_height": 53.628}
    on_axis([250.0, 300.0, 350.0, 700.0], **exponential)
    for bad_h in (0.0, -10.0):
        add([_DRAG_R_REF_EXACT + 300.0, 0.0, 0.0], [0.0, 7.7, 0.0], **{**exponential, "scale_height": bad_h})

    directions = [(0.3, -0.5, 0.81), (-0.7, 0.2, -0.68), (0.05, 0.99, 0.13), (-0.6, -0.6, 0.53)]
    laws = [layered, exponential] + msis_levels
    for law in laws:
        for k, d in enumerate(directions):
            u = np.asarray(d) / np.linalg.norm(d)
            r = (_DRAG_R_REF_EXACT + 180.0 + 97.0 * k) * u
            v = 7.7 * np.cross([0.1, 0.3, 0.95], u) / np.linalg.norm(np.cross([0.1, 0.3, 0.95], u))
            for omega in (drag.EARTH_OMEGA, 0.0, -3.1e-4):
                add(r.tolist(), v.tolist(), **{**law, "omega": omega})
        add([0.0, 0.0, 0.0], [0.0, 7.7, 0.0], **law)                  # zero separation
    return np.asarray(rows_r), np.asarray(rows_v), np.asarray(rows_p)


def _drag_fields(
    rel: np.ndarray, vel: np.ndarray, params: np.ndarray, parent: np.ndarray, parent_v: np.ndarray,
    *, perturb_b: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """`(reference, twin)` drag accelerations at every row. Reference: `drag.drag_kernel` over an arena
    whose slot 0 is the parent. Twin: `_cowell_accel` with only its drag flag set, the MSIS rows pointed
    at stacked tables built exactly as `Simulation._refresh_cowell_plan` builds them. `perturb_b`
    multiplies `B` on the twin's side only."""
    n = rel.shape[0]
    state = np.zeros((n + 1, 6))
    state[0, :3] = parent
    state[0, 3:] = parent_v
    state[1:, :3] = parent + rel
    state[1:, 3:] = parent_v + vel
    arena = np.zeros((n + 1, params.shape[1]))
    arena[1:] = params
    ref = np.zeros((n + 1, 3))
    drag.drag_kernel(np.arange(1, n + 1, dtype=np.int64), 0.0, state, np.zeros(n + 1),
                     np.zeros(n + 1, dtype=np.int32), arena, ref)

    msis_rows = np.flatnonzero(params[:, drag.DENSITY_MODEL_COL] >= 1.5)
    tables, table_of = drag.density_tables(params[msis_rows][:, drag.SOLAR_COLS])
    table_row = np.full(n, kernels.LAYERED_TABLE_ROW, dtype=np.int64)
    table_row[msis_rows] = table_of

    twin = np.zeros((n, 3))
    px, py, pz = (float(c) for c in state[0, :3])
    pv = (float(state[0, 3]), float(state[0, 4]), float(state[0, 5]))
    for k in range(n):
        cx, cy, cz = (float(c) for c in state[k + 1, :3])
        cv = (float(state[k + 1, 3]), float(state[k + 1, 4]), float(state[k + 1, 5]))
        row = params[k].copy()
        row[0] *= perturb_b
        twin[k] = _fused_accel(px, py, pz, cx, cy, cz, pv, cv, drag_row=row, drag_table=int(table_row[k]),
                               tables=tables)
    return ref[1:], twin


DRAG_PARENTS: list[tuple[str, np.ndarray, np.ndarray]] = [
    ("origin", np.zeros(3), np.zeros(3)),
    ("heliocentric", np.array([1.495978707e8, -2.3e6, 4.1e4]), np.array([0.44, 29.78, -0.002])),
]


def _drag_disagreement(ref: np.ndarray, twin: np.ndarray) -> float:
    """Largest `|twin - ref| / |ref|` over rows with non-zero drag. Rows where the reference is exactly
    zero (the silent-zero and zero-separation rows) must be exactly zero in the twin too."""
    ref_norm = np.linalg.norm(ref, axis=1)
    zero = ref_norm == 0.0
    assert np.array_equal(twin[zero], np.zeros_like(twin[zero])), "a zero-contribution row is non-zero"
    return float(np.max(np.linalg.norm(twin[~zero] - ref[~zero], axis=1) / ref_norm[~zero]))


@pytest.mark.parametrize("parent_name,parent,parent_v", DRAG_PARENTS, ids=[n for n, _, _ in DRAG_PARENTS])
def test_fused_drag_term_matches_drag_kernel(
    parent_name: str, parent: np.ndarray, parent_v: np.ndarray,
) -> None:
    """
    The drag term alone, compiled against `drag.drag_kernel`, to 1e-12 of `|a_drag|` per row - every
    density law, every layered band edge exactly and an ulp either side, MSIS nodes and the 200 km grid
    change, both extrapolated regions of both tables, every zero-contribution rule. The heliocentric
    parent (moving at 29.8 km/s) puts both subtractions - position and velocity - on a coarse grid,
    where both sides round `rel` identically because they perform the same subtraction.
    """
    rel, vel, params = _drag_field_rows()
    ref, twin = _drag_fields(rel, vel, params, parent, parent_v)
    assert np.all(np.isfinite(ref)) and np.all(np.isfinite(twin))
    assert int(np.count_nonzero(np.linalg.norm(ref, axis=1) == 0.0)) >= 4, (
        "guard: the silent-zero and zero-separation rows must be present")
    diff = _drag_disagreement(ref, twin)
    assert diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{parent_name}: fused drag term disagrees with drag_kernel by {diff:.3e} of |a_drag|")


def test_layered_band_edges_are_exact_boundaries() -> None:
    """Guard for the test above: the on-axis rows really do sit *on* every band base altitude (so the
    search's side convention is exercised, not approximated), and the band either side of a boundary
    gives a measurably different density there - otherwise a wrong side could hide inside 1e-12."""
    for h in BASE_ALTITUDE_KM:
        x = _DRAG_R_REF_EXACT + float(h)
        assert np.sqrt(x * x) - _DRAG_R_REF_EXACT == h
    rel, vel, params = _drag_field_rows()
    ref, _ = _drag_fields(rel, vel, params, np.zeros(3), np.zeros(3))
    n_edges = BASE_ALTITUDE_KM.size
    on_edge = np.linalg.norm(ref[0:3 * n_edges:3], axis=1)[1:]     # exactly on 25, 30, ..., 1000 km
    below = np.linalg.norm(ref[1:3 * n_edges:3], axis=1)[1:]       # an ulp below each (band k - 1)
    assert np.all(np.abs(on_edge - below) / on_edge > 1e-10), (
        "a band boundary where both bands agree to 1e-10 would not expose a side error")


def test_fused_drag_comparison_would_detect_a_perturbed_coefficient() -> None:
    """Negative control for the field-level test: `B` scaled by (1 + 1e-9) on the twin's side only.
    `a_drag` is linear in `B`, so every non-zero row moves by exactly 1e-9 of itself - three orders
    above the bound - and must be detected."""
    rel, vel, params = _drag_field_rows()
    ref, twin = _drag_fields(rel, vel, params, np.zeros(3), np.zeros(3), perturb_b=1.0 + 1e-9)
    diff = _drag_disagreement(ref, twin)
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"a relative-1e-9 change to B was not detected (diff {diff:.3e}); the bound is vacuous")


@pytest.mark.parametrize("variant", ["layered", "mixed"])
def test_drag_term_is_visible_to_the_state_comparison(
    variant: str, db_session_factory: Callable[[], Session],
) -> None:
    """
    The Cowell state tests certify the fused drag path only if dropping drag would fail them. Over 50
    steps of 60 s drag moves each satellite by ~0.6 km (section comment), ~1e-4 of its radius; the
    same compiled run with every `has_drag` flag cleared must differ by far more than the 1e-12 bound
    - asserted at 1e-6.
    """
    with_sim, idx = _build_cowell_drag(db_session_factory(), variant=variant)
    without_sim, _ = _build_cowell_drag(db_session_factory(), variant=variant)
    without_sim._cowell_has_drag[:] = False
    _, with_rel = _run_cowell_kernel(with_sim, idx, COWELL_DT, 50)
    _, without_rel = _run_cowell_kernel(without_sim, idx, COWELL_DT, 50)
    diff = _relative_difference(with_rel, without_rel, np.ones(idx.size, dtype=bool))
    assert diff > 1e-6, f"drag moved the state by only {diff:.3e}; the state test is blind to it"


def test_drag_stage_velocity_is_visible_to_the_state_comparison(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The check that each RK4 stage's drag sees its *own* velocity is the state comparison itself - it
    must be able to fail. Here the NumPy reference is driven with an acceleration provider that
    evaluates drag at the stage-1 (committed) velocity for all four stages, which is what a twin that
    forgot to pass `pv + v_k` would compute. That differs from the true reference in the drag stage
    accelerations by `(dt/2..dt) |a_grav| / |v|` ~ 3-7 % of drag, and over 50 steps must exceed the 1e-12 bound
    the equivalence tests use - so a twin making that mistake fails them.
    """
    true_sim, idx = _build_cowell_drag(db_session_factory(), variant="layered")
    frozen_sim, _ = _build_cowell_drag(db_session_factory(), variant="layered")
    _, true_rel = _run_cowell_reference(true_sim, idx, COWELL_DT, 50)

    integrator = RK4Integrator(frozen_sim.max_capacity)
    primaries = frozen_sim.parent_indices[idx]
    committed_v = np.zeros((frozen_sim.max_capacity, 3))

    def frozen_velocity(t: float, state: np.ndarray) -> np.ndarray:
        staged = state[idx, 3:].copy()
        state[idx, 3:] = committed_v[idx]
        out = frozen_sim.accelerations(t, state)
        state[idx, 3:] = staged
        return out

    for _ in range(50):
        parent_start = frozen_sim.global_states[primaries].copy()
        committed_v[idx] = frozen_sim.global_states[idx, 3:]
        # Gravity reads position only, so only the drag stage accelerations change.
        integrator.step(frozen_velocity, frozen_sim.t, frozen_sim.global_states, COWELL_DT, idx, primaries)
        frozen_rel = frozen_sim.global_states[idx] - parent_start
    diff = _relative_difference(true_rel, frozen_rel, np.ones(idx.size, dtype=bool))
    assert diff > 100 * KERNEL_AGREEMENT_REL_TOL, (
        f"freezing drag's stage velocity moved the state by only {diff:.3e}; the state comparison "
        f"could not tell a twin that did so")


def test_msis_indices_written_after_planning_replan_the_fused_step(
    db_session_factory: Callable[[], Session],
) -> None:
    """`drag_kernel` reads a row's `(f107, f107a, ap)` live, so indices written straight into
    `force_model_params` take effect on the next step without `resolve_force_models`. The compiled step
    must do the same: the kernel sees the mismatch with its planned profile before writing anything,
    `step()` re-plans, and the result still matches the NumPy path to the bound."""
    pytest.importorskip("pymsis")
    high = msis_coefficients(SOLAR_ACTIVITY_HIGH)
    msis_profile(high["f107"], high["f107a"], high["ap"])          # evaluated, as configuration would
    sims = []
    for compiled in (False, True):
        sim, idx = _build_cowell_drag(db_session_factory(), variant="msis")
        sim.record_history = False
        sim.use_compiled_kernel = compiled
        sim.step(COWELL_DT)
        sim.force_model_params[drag.DRAG_MODEL][idx[0], drag.SOLAR_COLS] = (high["f107"], high["f107a"], high["ap"])
        for _ in range(20):
            sim.step(COWELL_DT)
        sims.append(sim)
    numpy_sim, compiled_sim = sims
    assert compiled_sim._cowell_fused_ok, "a memoised triple must keep the compiled plan"
    row = compiled_sim._cowell_drag_table_of[idx[0]]
    assert tuple(compiled_sim._cowell_drag_tables.activity[row]) == (high["f107"], high["f107a"], high["ap"])
    diff = _relative_difference(numpy_sim.global_states, compiled_sim.global_states, numpy_sim.active_mask)
    assert diff < KERNEL_AGREEMENT_REL_TOL, f"re-planned compiled step disagrees by {diff:.3e}"


def test_an_unevaluated_msis_triple_written_after_planning_still_raises(
    db_session_factory: Callable[[], Session],
) -> None:
    """The reference raises `LookupError` on a triple nobody evaluated rather than calling `pymsis`
    inside a step. The compiled path must not turn that into a silent stale density: it re-plans, finds
    no profile, falls back, and the reference raises."""
    pytest.importorskip("pymsis")
    sim, idx = _build_cowell_drag(db_session_factory(), variant="msis")
    sim.record_history = False
    sim.use_compiled_kernel = True
    sim.step(COWELL_DT)
    sim.force_model_params[drag.DRAG_MODEL][idx[0], drag.SOLAR_COLS] = (71.25, 83.5, 7.0)
    with pytest.raises(LookupError, match="configuration time"):
        sim.step(COWELL_DT)
    assert not sim._cowell_fused_ok, "a triple with no profile must disqualify the compiled plan"


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
