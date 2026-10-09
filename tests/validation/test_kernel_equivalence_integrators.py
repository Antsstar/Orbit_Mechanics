"""
Equivalence of the compiled leapfrog, Yoshida-4 and Encke kernels with their NumPy references
(`integrators.LeapfrogIntegrator`, `Yoshida4Integrator`, `EnckeIntegrator`).

The same standard as `test_kernel_equivalence.py` gives `kernels.cowell_rk4_step`: the reference is the
definition, the twin runs the same operations in the same order, and the two differ only by libm
rounding and by `np.einsum`'s summation order for the three squared components - an ulp or so per
force evaluation, so ~1e-15 of the state per step. Asserted at 1e-12 elementwise after 1 and 50 steps,
and at 1e-12 per body by norm after 500 (the elementwise metric floors on near-zero components over
several orbits; see the long-run test in `test_kernel_equivalence.py` for that derivation, which these
methods share since each is the same force composition stepped differently).

**Encke has two further sources of difference, both rounding-level.** The reference's vectorised
`kepler_advance` iterates every body until the slowest converges, so an early-converged body takes
extra Newton steps there that the scalar twin does not; and the reference's `chi ** 3` is libm `pow`
where the twin multiplies out. Each is within an ulp or two of `chi`, and Newton's iteration
self-corrects, so they do not change the 1e-12 bound's derivation - the measured agreement is reported
by the tests that fail if it ever stops holding.

The scenarios reuse `test_kernel_equivalence`'s builders (`mixed` drag/zonal/tesseral arenas whose
per-body flags differ inside one kernel call) plus one that crosses all four terms and all three drag
laws. `"tesseral"`'s `mixed` variant has a body with no central term, which Encke refuses by design,
so Encke runs the arenas where every body carries `point_mass_gravity`.
"""
from __future__ import annotations

import math

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import integrators, kernels, tesseral
from orbital_engine.integrators import make_integrator
from orbital_engine.simulator import Simulation

from .test_kernel_equivalence import (
    COWELL_DT, COWELL_LONG_STEPS, KERNEL_AGREEMENT_REL_TOL, _build_cowell_constellation,
    _build_cowell_drag, _build_cowell_tesseral, _build_cowell_two_body, _build_cowell_zonal,
    _build_cowell_zonal_eccentric, _fused_plan_args, _relative_difference, _TESSERAL_FULL,
)

Build = Callable[[Session], tuple[Simulation, np.ndarray]]

KERNELS: dict[str, Callable[..., int]] = {
    "leapfrog": kernels.cowell_leapfrog_step,
    "yoshida4": kernels.cowell_yoshida4_step,
    "encke": kernels.cowell_encke_step,
}


def _build_two_body_point_mass(session: Session) -> tuple[Simulation, np.ndarray]:
    """The e = 0.2, p = 11,000 km two-body orbit of `test_symplectic` / `test_encke`: point mass only."""
    from orbital_engine import gravity, scenarios
    from orbital_engine.custom_types import PropagatorType
    sim = scenarios.two_body(session, p=11000.0, e=0.2)
    sats = np.asarray([sim.name_to_index["Secondary"]], dtype=np.int64)
    sim.set_propagator(sats, PropagatorType.COWELL)
    sim.enable_force_model(gravity.POINT_MASS_MODEL, sats.tolist())
    return sim, sats


def _build_mixed_all(session: Session) -> tuple[Simulation, np.ndarray]:
    """One arena crossing every fused term and all three drag laws: `_build_cowell_drag`'s `mixed`
    (layered + j2, exponential + j2 + zonal, MSIS moderate + j2 + zonal, MSIS high, an exponential row
    with `scale_height = 0` + j2, a drag-free pm + j2 body, layered + zonal, MSIS low + j2), with the
    tesseral field added to four of them at different phases and rates. Every body has point mass."""
    sim, sats = _build_cowell_drag(session, variant="mixed")
    s = [int(i) for i in sats]
    field = {**_TESSERAL_FULL}
    sim.enable_force_model(tesseral.TESSERAL_MODEL, [s[0], s[5]], **{**field, "theta0": 0.4})
    sim.enable_force_model(tesseral.TESSERAL_MODEL, s[3], **{**field, "omega": 7.0882181e-5, "theta0": -2.0})
    sim.enable_force_model(tesseral.TESSERAL_MODEL, s[6], **{**field, "omega": -2.9924e-7})
    return sim, sats


# Scenarios every integrator runs (every body has point mass), and the extra ones the symplectic
# methods run too (a tesseral body with no central term, which Encke refuses).
SCENARIOS: list[tuple[str, Build]] = [
    ("two_body_e0.2", _build_two_body_point_mass),
    ("eccentric+j2", lambda s: _build_cowell_two_body(s, p=11000.0, e=0.7)),
    ("mixed_j2", lambda s: _build_cowell_constellation(s, j2_on="half")),
    ("mixed_zonal", lambda s: _build_cowell_zonal(s, variant="mixed")),
    ("eccentric_inclined+j2+zonal", _build_cowell_zonal_eccentric),
    ("drag_mixed", lambda s: _build_cowell_drag(s, variant="mixed")),
    ("mixed_all", _build_mixed_all),
]
SYMPLECTIC_ONLY: list[tuple[str, Build]] = [
    ("mixed_tesseral_no_central_term", lambda s: _build_cowell_tesseral(s, variant="mixed")),
]


def _cases() -> list[tuple[str, str, Build]]:
    out = [(n, name, b) for n in KERNELS for name, b in SCENARIOS]
    out += [(n, name, b) for n in ("leapfrog", "yoshida4") for name, b in SYMPLECTIC_ONLY]
    return out


CASES = _cases()
CASE_IDS = [f"{n}-{name}" for n, name, _ in CASES]


def _run_reference(sim: Simulation, name: str, idx: np.ndarray, dt: float, steps: int,
                   t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The definition: the named NumPy integrator against `Simulation.accelerations`, then the
    subtraction `Simulation.step` applies to recover the parent-relative result."""
    integrator = make_integrator(name, sim.max_capacity, sim.mu_array)
    primaries = sim.parent_indices[idx]
    rel = np.zeros((idx.size, 6))
    t = float(t0)
    for _ in range(steps):
        parent_start = sim.global_states[primaries].copy()
        integrator.step(sim.accelerations, t, sim.global_states, dt, idx, primaries)
        rel = sim.global_states[idx] - parent_start
        t += dt
    return sim.global_states[idx].copy(), rel


def _run_kernel(sim: Simulation, name: str, idx: np.ndarray, dt: float, steps: int,
                t0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    t = float(t0)
    for _ in range(steps):
        stale = KERNELS[name](
            dt, t, sim.global_states, sim.mu_array, sim.parent_indices, idx, *_fused_plan_args(sim),
            rel_out,
        )
        assert stale == -1, f"{name}: the fused kernel reported a stale MSIS plan for slot {stale}"
        t += dt
    return sim.global_states[idx].copy(), rel_out[idx].copy()


def test_kernel_yoshida_weights_are_the_references_bit_for_bit() -> None:
    assert kernels._YOSHIDA_W1 == integrators._YOSHIDA_W1
    assert kernels._YOSHIDA_W0 == integrators._YOSHIDA_W0


@pytest.mark.parametrize("integrator,name,build", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("steps", [1, 50])
def test_kernel_matches_reference_state(
    integrator: str, name: str, build: Build, steps: int, db_session_factory: Callable[[], Session],
) -> None:
    ref_sim, ref_idx = build(db_session_factory())
    ker_sim, ker_idx = build(db_session_factory())
    assert np.array_equal(ref_sim.global_states, ker_sim.global_states)
    assert ker_sim._cowell_fused_ok, "guard: this scenario must qualify for the fused kernel"

    ref_abs, ref_rel = _run_reference(ref_sim, integrator, ref_idx, COWELL_DT, steps)
    ker_abs, ker_rel = _run_kernel(ker_sim, integrator, ker_idx, COWELL_DT, steps)
    assert np.all(np.isfinite(ref_rel)) and np.all(np.isfinite(ker_rel))

    every = np.ones(ref_idx.size, dtype=bool)
    abs_diff = _relative_difference(ref_abs, ker_abs, every)
    rel_diff = _relative_difference(ref_rel, ker_rel, every)
    assert abs_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{integrator}/{name}: absolute state disagrees by {abs_diff:.3e} after {steps} steps")
    assert rel_diff < KERNEL_AGREEMENT_REL_TOL, (
        f"{integrator}/{name}: parent-relative state disagrees by {rel_diff:.3e} after {steps} steps")


@pytest.mark.parametrize("integrator,name,build", CASES, ids=CASE_IDS)
def test_kernel_matches_reference_over_several_orbits(
    integrator: str, name: str, build: Build, db_session_factory: Callable[[], Session],
) -> None:
    """500 steps of 60 s, measured per body as `|dr|/|r|` and `|dv|/|v|` - the horizon and metric of
    `test_cowell_kernel_matches_reference_over_several_orbits`, whose derivation applies unchanged."""
    ref_sim, ref_idx = build(db_session_factory())
    ker_sim, ker_idx = build(db_session_factory())
    _, ref_rel = _run_reference(ref_sim, integrator, ref_idx, COWELL_DT, COWELL_LONG_STEPS)
    _, ker_rel = _run_kernel(ker_sim, integrator, ker_idx, COWELL_DT, COWELL_LONG_STEPS)
    assert np.all(np.isfinite(ref_rel)) and np.all(np.isfinite(ker_rel))
    dr = float(np.max(np.linalg.norm(ref_rel[:, :3] - ker_rel[:, :3], axis=1)
                      / np.linalg.norm(ref_rel[:, :3], axis=1)))
    dv = float(np.max(np.linalg.norm(ref_rel[:, 3:] - ker_rel[:, 3:], axis=1)
                      / np.linalg.norm(ref_rel[:, 3:], axis=1)))
    assert dr < KERNEL_AGREEMENT_REL_TOL, f"{integrator}/{name}: |dr|/|r| = {dr:.3e}"
    assert dv < KERNEL_AGREEMENT_REL_TOL, f"{integrator}/{name}: |dv|/|v| = {dv:.3e}"


# --- Encke's scalar pieces against the vectorised references, at function level -----------------------
#
# The state comparison above cannot see a defect in the Stumpff series' high-order term: at the
# step sizes the arenas use `|z| < 1e-3`, the `z^3 / 40320` term is ~2.5e-14 of C, far below 1e-12 in
# the state (found by mutation: flipping its sign passed every state test). These compare the pieces
# themselves, where the bound can be as tight as the arithmetic allows (a few ulp: 1e-14 relative).

Z_VALUES = [0.0, 1e-12, -1e-9, 3e-4, -3e-4, 9.99e-4, -9.99e-4, 1.0e-3, -1.0e-3, 0.7, -0.7, 5.0, -5.0, 40.0, -40.0]


def test_scalar_stumpff_matches_the_reference() -> None:
    z = np.array(Z_VALUES)
    c_ref, s_ref = integrators._stumpff(z)
    for k, zk in enumerate(Z_VALUES):
        c, s = kernels._stumpff_scalar(zk)
        assert abs(c - c_ref[k]) <= 1e-14 * abs(c_ref[k]), (zk, c, c_ref[k])
        assert abs(s - s_ref[k]) <= 1e-14 * abs(s_ref[k]), (zk, s, s_ref[k])


def test_scalar_battin_f_matches_the_reference() -> None:
    q = np.array([0.0, 1e-12, -3e-9, 1e-6, -1e-6, 1e-3, -0.2, 0.5, 3.0])
    ref = integrators.battin_f(q)
    for k, qk in enumerate(q):
        got = kernels._battin_f_scalar(float(qk))
        assert abs(got - ref[k]) <= 1e-14 * max(abs(ref[k]), 1e-300), (qk, got, ref[k])


@pytest.mark.parametrize("label,r0,v0,dt", [
    ("elliptic_leo", [6921.0, 0.0, 0.0], [0.0, 5.5, 5.5], 60.0),
    ("elliptic_long", [7000.0, 1000.0, -500.0], [-0.5, 6.9, 1.0], 3000.0),
    ("eccentric", [6471.0, 0.0, 0.0], [0.0, 9.5, 3.0], 900.0),
    ("hyperbolic", [8000.0, 0.0, 0.0], [0.0, 9.0, 4.0], 1200.0),
    ("near_parabolic", [7000.0, 0.0, 0.0], [0.0, 10.6, 0.0], 600.0),
    ("just_above_escape", [7000.0, 0.0, 0.0], [0.0, 10.7, 0.0], 600.0),
    ("fast_hyperbolic", [7000.0, 0.0, 0.0], [0.0, 20.0, 0.0], 6000.0),
    ("inbound_hyperbolic", [7000.0, 0.0, 0.0], [-14.1, 10.0, 0.0], -6000.0),
    ("backward", [6921.0, 0.0, 0.0], [0.0, 5.5, 5.5], -60.0),
])
def test_scalar_kepler_advance_matches_the_reference(
    label: str, r0: list[float], v0: list[float], dt: float,
) -> None:
    mu = 398600.4418
    r_ref, v_ref = integrators.kepler_advance(
        np.array([r0]), np.array([v0]), dt, np.array([mu]))
    out = kernels._kepler_advance_scalar(*r0, *v0, dt, mu)
    got = np.array(out)
    ref = np.concatenate([r_ref[0], v_ref[0]])
    scale = np.maximum(np.abs(ref), 1e-3)
    assert float(np.max(np.abs(got - ref) / scale)) < 1e-12, label


def test_scalar_kepler_advance_matches_the_reference_on_every_conic() -> None:
    """
    The safeguarded solve branches (Newton, bisection, overflow side), so the twin is held to the
    reference over the grid of `test_encke.test_kepler_advance_converges_on_every_conic`: to 1e-12
    where the problem is well conditioned (periapsis above the surface, |dt| <= 6e4 s), and to 1e-10
    elsewhere. *Measured* 1.0e-11 there, on a near-parabolic arc over 6e5 s and on passes tens of
    km from the point mass, where an ulp of difference in cosh is amplified.
    """
    mu = 398600.4418
    worst_clear = worst_all = 0.0
    for radius in (7000.0, 1.0e6):
        for speed in (1.0, 7.5, 10.6717, 10.7, 15.0, 20.0, 50.0, 100.0):
            for fpa in np.radians([-89.0, -45.0, 0.0, 10.0, 80.0]):
                r0 = [radius, 0.0, 0.0]
                v0 = [speed * math.sin(fpa), speed * math.cos(fpa), 0.0]
                p = (radius * v0[1]) ** 2 / mu
                e = math.sqrt(max(0.0, 1.0 - p * (2.0 / radius - speed * speed / mu)))
                for dt in (-60000.0, -60.0, 10.0, 600.0, 6000.0, 600000.0):
                    r_ref, v_ref = integrators.kepler_advance(np.array([r0]), np.array([v0]), dt, np.array([mu]))
                    ref = np.concatenate([r_ref[0], v_ref[0]])
                    got = np.array(kernels._kepler_advance_scalar(*r0, *v0, dt, mu))
                    scale = np.maximum(np.abs(ref), 1e-3 * np.max(np.abs(ref)))
                    err = float(np.max(np.abs(got - ref) / scale))
                    worst_all = max(worst_all, err)
                    if p / (1.0 + e) >= 6378.137 and abs(dt) <= 6.0e4:
                        worst_clear = max(worst_clear, err)
    assert worst_clear < 1e-12
    assert worst_all < 1e-10


# --- Wired path: Simulation.step on and off the compiled kernel, with manoeuvre splits ---------------

@pytest.mark.parametrize("integrator", list(KERNELS))
@pytest.mark.parametrize("name,build", [SCENARIOS[0], SCENARIOS[-1]], ids=["two_body_e0.2", "mixed_all"])
def test_step_paths_agree_across_manoeuvre_splits(
    integrator: str, name: str, build: Build, db_session_factory: Callable[[], Session],
) -> None:
    """`Simulation.step` with `use_compiled_kernel` on and off: 50 steps with a real RSW Delta-v
    scheduled at 40 % of every fifth step (ten steps cut in two, the second half starting at the split
    epoch, which tesseral reads). Parent at the origin, so no re-base floor."""
    numpy_sim, idx = build(db_session_factory())
    compiled_sim, _ = build(db_session_factory())
    for sim, compiled in ((numpy_sim, False), (compiled_sim, True)):
        sim.use_compiled_kernel = compiled
        sim.record_history = False
        sim.set_cowell_integrator(integrator)
        for k in range(0, 50, 5):
            sim.schedule_delta_v(idx, [1.0e-4, 2.0e-4, -1.0e-4], epoch_s=(k + 0.4) * COWELL_DT)
    assert compiled_sim._cowell_fused_ok and compiled_sim.cowell_integrator == integrator
    for _ in range(50):
        numpy_sim.step(COWELL_DT)
        compiled_sim.step(COWELL_DT)
    assert not numpy_sim.pending_manoeuvres and not compiled_sim.pending_manoeuvres
    diff = _relative_difference(numpy_sim.global_states, compiled_sim.global_states, numpy_sim.active_mask)
    assert diff < KERNEL_AGREEMENT_REL_TOL, f"{integrator}/{name}: split-step state disagrees by {diff:.3e}"


def test_every_integrator_is_fused_under_the_same_conditions(
    db_session_factory: Callable[[], Session],
) -> None:
    for integrator in integrators.INTEGRATOR_NAMES:
        sim, _ = _build_cowell_constellation(db_session_factory(), j2_on="all")
        sim.set_cowell_integrator(integrator)
        assert sim._cowell_fused_ok, integrator


def test_fused_encke_still_refuses_a_body_without_point_mass(
    db_session_factory: Callable[[], Session],
) -> None:
    sim, _ = _build_cowell_tesseral(db_session_factory(), variant="mixed")   # one body has no pm bit
    sim.record_history = False
    sim.set_cowell_integrator("encke")
    assert sim._cowell_fused_ok
    for compiled in (True, False):
        sim.use_compiled_kernel = compiled
        with pytest.raises(ValueError, match="point_mass_gravity"):
            sim.step(COWELL_DT)


# --- Negative controls -------------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", list(KERNELS))
def test_comparison_would_detect_a_perturbed_mu(
    integrator: str, db_session_factory: Callable[[], Session],
) -> None:
    """A relative-1e-9 nudge to Earth's mu on the kernel side changes every acceleration by 1e-9; over
    10 steps of 60 s that displaces a 550 km satellite by ~1.5e-6 km (2e-10 of its radius), far above
    the 1e-12 bound, so it must be detected by each kernel's comparison."""
    ref_sim, ref_idx = _build_cowell_constellation(db_session_factory(), j2_on="all")
    _, ref_rel = _run_reference(ref_sim, integrator, ref_idx, COWELL_DT, 10)
    ker_sim, ker_idx = _build_cowell_constellation(db_session_factory(), j2_on="all")
    ker_sim.mu_array[ker_sim.name_to_index["Earth"]] *= (1.0 + 1e-9)
    _, ker_rel = _run_kernel(ker_sim, integrator, ker_idx, COWELL_DT, 10)
    diff = _relative_difference(ref_rel, ker_rel, np.ones(ref_idx.size, dtype=bool))
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"{integrator}: a deliberately perturbed kernel was not detected (diff {diff:.3e})")


@pytest.mark.parametrize("flag_name", ["COWELL_J2", "COWELL_ZONAL", "COWELL_DRAG", "COWELL_TESSERAL"])
@pytest.mark.parametrize("integrator", list(KERNELS))
def test_comparison_would_detect_a_dropped_term(
    integrator: str, flag_name: str, db_session_factory: Callable[[], Session],
) -> None:
    """Clearing one term's per-body flag on the kernel side of `mixed_all` (which every term appears
    in) must move the 50-step state far beyond the bound - the equivalence tests are not blind to it."""
    ref_sim, idx = _build_mixed_all(db_session_factory())
    _, ref_rel = _run_reference(ref_sim, integrator, idx, COWELL_DT, 50)
    ker_sim, _ = _build_mixed_all(db_session_factory())
    ker_sim._cowell_flags &= ~getattr(kernels, flag_name)
    _, ker_rel = _run_kernel(ker_sim, integrator, idx, COWELL_DT, 50)
    diff = _relative_difference(ref_rel, ker_rel, np.ones(idx.size, dtype=bool))
    assert diff > 1e-9, f"{integrator}: dropping {flag_name} moved the state by only {diff:.3e}"
