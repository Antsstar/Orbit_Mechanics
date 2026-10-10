"""
Equivalence of the fused `third_body` term (`kernels._third_body_rel` / `_third_body_term`, inside every
`cowell_*_step`) with its NumPy reference (`thirdbody.third_body_kernel` under the matching integrator),
in both modes: `staged=0` (perturber frozen at its start-of-step state) and `staged=1` (perturber carried
along its two-body conic to each stage time).

The scenario is the verification case of `test_third_body.py`: `sun_earth_moon(moon_mu=0.0)` with the Moon
on Cowell + `point_mass_gravity` + `third_body`(Sun). The Moon's parent is Earth, which accelerates toward
the Sun, so a kernel that forgot the indirect term or read the wrong parent row would be visibly wrong.

**The bound.** Both sides run the same operations in the same order, so they differ only by libm rounding
(under an ulp per call) and by `np.einsum`'s summation order for the three squared components, giving
~1e-16 relative per force evaluation; the staged conic adds a Newton solve that stops at the same
iterate in both (the reference freezes a converged body, the scalar twin breaks), so it adds the same
order. Asserted at `KERNEL_AGREEMENT_REL_TOL` = 1e-12 per body by norm (`|dr|/|r|`, `|dv|/|v|`) after 1, 30
and 300 steps of 3600 s (12.5 days, 0.46 lunar orbits). The heliocentric grid is the one place rounding can
grow: an absolute coordinate near 1.5e8 km has an ulp of 3e-8 km, 8e-14 of the lunar distance, so a
single flipped rounding costs 8e-14 and the horizon is kept short enough that a few of them stay inside
the bound. *Measured* worst case over every integrator x mode x horizon: see the log of the run that
added this file (about 1e-16, mostly bit-identical).

**Negative controls.** The Moon's solar tide is ~3e-8 km/s^2 against ~2.7e-3 for the Earth's pull, so the
comparison is only as sensitive to the third-body term as the tide's share of the motion. Dropping the
term is a 1e-4-level change and is detected at once; a relative 1e-9 nudge to the Sun's mu is a 1e-9
nudge to the tide, which over 300 steps moves the Moon by 5e-11 of its distance, five times the bound,
so the horizon of that control is 300 steps (30 steps would give 5e-13 and miss).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import kernels, scenarios
from orbital_engine import simulator as simulator_module
from orbital_engine.custom_types import PropagatorType
from orbital_engine.integrators import INTEGRATOR_NAMES, make_integrator
from orbital_engine.simulator import Simulation
from orbital_engine.thirdbody import (
    THIRD_BODY_MODEL, THIRD_BODY_PARAM_NAMES, _PERTURBER_COL, _STAGED_COL, _T0_COL,
)

from .test_kernel_equivalence import KERNEL_AGREEMENT_REL_TOL, _fused_plan_args

DT = 3600.0
HORIZONS = (1, 30, 300)

KERNELS: dict[str, Callable[..., int]] = {
    "rk4": kernels.cowell_rk4_step,
    "leapfrog": kernels.cowell_leapfrog_step,
    "yoshida4": kernels.cowell_yoshida4_step,
    "encke": kernels.cowell_encke_step,
}
MODES = [("frozen", False), ("staged", True)]


def _moon(session: Session, staged: bool) -> tuple[Simulation, np.ndarray]:
    sim = scenarios.sun_earth_moon(session, moon_mu=0.0)
    sim.record_history = False
    moon = np.asarray([sim.name_to_index["Moon"]], dtype=np.int64)
    sim.set_propagator(moon, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", moon.tolist())
    sim.enable_force_model(THIRD_BODY_MODEL, moon.tolist(),
                           perturber=float(sim.name_to_index["Sun"]), staged=1.0 if staged else 0.0)
    return sim, moon


def _run_reference(sim: Simulation, name: str, idx: np.ndarray, steps: int) -> np.ndarray:
    """The definition: the NumPy integrator against `Simulation.accelerations`. `Simulation._advance`
    writes each step's start time into the engine-owned `t0` column; done by hand here, as the kernel
    runner does too, since neither goes through `_advance`."""
    integrator = make_integrator(name, sim.max_capacity, sim.mu_array)
    primaries = sim.parent_indices[idx]
    params = sim.force_model_params[THIRD_BODY_MODEL]
    rel = np.zeros((idx.size, 6))
    t = 0.0
    for _ in range(steps):
        params[idx, _T0_COL] = t
        start = sim.global_states[primaries].copy()
        integrator.step(sim.accelerations, t, sim.global_states, DT, idx, primaries)
        rel = sim.global_states[idx] - start
        t += DT
    return rel


def _run_kernel(sim: Simulation, name: str, idx: np.ndarray, steps: int) -> np.ndarray:
    rel_out = np.zeros((sim.max_capacity, 6), dtype=np.float64)
    params = sim.force_model_params[THIRD_BODY_MODEL]
    t = 0.0
    for _ in range(steps):
        params[idx, _T0_COL] = t
        stale = KERNELS[name](DT, t, sim.global_states, sim.mu_array, sim.parent_indices, idx,
                              *_fused_plan_args(sim), rel_out)
        assert stale == -1
        t += DT
    return rel_out[idx].copy()


def _norm_difference(ref: np.ndarray, twin: np.ndarray) -> float:
    """Worst of `|dr|/|r|` and `|dv|/|v|` over the bodies (the metric of the long-run Cowell tests)."""
    dr = np.linalg.norm(ref[:, :3] - twin[:, :3], axis=1) / np.linalg.norm(ref[:, :3], axis=1)
    dv = np.linalg.norm(ref[:, 3:] - twin[:, 3:], axis=1) / np.linalg.norm(ref[:, 3:], axis=1)
    return float(max(dr.max(), dv.max()))


def test_column_constants_match_the_model() -> None:
    assert THIRD_BODY_PARAM_NAMES[_PERTURBER_COL] == "perturber"
    assert THIRD_BODY_PARAM_NAMES[_STAGED_COL] == "staged"
    assert THIRD_BODY_PARAM_NAMES[_T0_COL] == "t0"
    assert (kernels._THIRD_PERTURBER_COL, kernels._THIRD_STAGED_COL, kernels._THIRD_T0_COL) == (
        _PERTURBER_COL, _STAGED_COL, _T0_COL)


def test_every_integrator_name_has_a_kernel() -> None:
    assert set(KERNELS) == set(INTEGRATOR_NAMES)


# --- 1. Fused against reference ---------------------------------------------------------------------

@pytest.mark.parametrize("steps", HORIZONS)
@pytest.mark.parametrize("mode,staged", MODES, ids=[m for m, _ in MODES])
@pytest.mark.parametrize("integrator", list(KERNELS))
def test_fused_matches_reference(
    integrator: str, mode: str, staged: bool, steps: int, db_session_factory: Callable[[], Session],
) -> None:
    ref_sim, idx = _moon(db_session_factory(), staged)
    ker_sim, _ = _moon(db_session_factory(), staged)
    assert ker_sim._cowell_fused_ok
    ref = _run_reference(ref_sim, integrator, idx, steps)
    twin = _run_kernel(ker_sim, integrator, idx, steps)
    assert np.all(np.isfinite(ref)) and np.all(np.isfinite(twin))
    diff = _norm_difference(ref, twin)
    assert diff < KERNEL_AGREEMENT_REL_TOL, f"{integrator}/{mode}: {diff:.3e} after {steps} steps"


# --- 2. The plan accepts the model ------------------------------------------------------------------

def test_fused_plan_accepts_third_body_and_sets_its_flag(db_session_factory: Callable[[], Session]) -> None:
    for staged in (False, True):
        sim, idx = _moon(db_session_factory(), staged)
        assert sim._cowell_fused_ok, "third_body was foreign to the fused plan before this twin existed"
        flags = sim._cowell_flags[idx]
        assert np.all((flags & kernels.COWELL_THIRD_BODY) != 0)
        assert np.all((flags & kernels.COWELL_POINT_MASS) != 0)
        assert sim._cowell_third_params is sim.force_model_params[THIRD_BODY_MODEL]
        for integrator in INTEGRATOR_NAMES:
            sim.set_cowell_integrator(integrator)
            assert sim._cowell_fused_ok, integrator


def test_a_set_without_the_model_gets_the_dummy_params_and_no_flag(
    db_session_factory: Callable[[], Session],
) -> None:
    sim = scenarios.sun_earth_moon(db_session_factory(), moon_mu=0.0)
    moon = sim.name_to_index["Moon"]
    sim.set_propagator(moon, PropagatorType.COWELL)
    sim.enable_force_model("point_mass_gravity", moon)
    assert sim._cowell_fused_ok
    assert sim._cowell_third_params is sim._no_third_params
    assert (sim._cowell_flags[moon] & kernels.COWELL_THIRD_BODY) == 0


# --- 3. Wired path, adaptive sub-stepping: the t0 column, not the kernel's step-start argument ---------

ADAPTIVE_TOL_KM = {"rk4": 1.0e-3, "leapfrog": 1.0e-2, "yoshida4": 1.0e-3, "encke": 1.0e-5}   # all force real sub-stepping
ADAPTIVE_DT = 43200.0
ADAPTIVE_STEPS = 6


def _adaptive_run(session: Session, integrator: str, compiled: bool) -> tuple[np.ndarray, int]:
    sim, idx = _moon(session, True)
    sim.use_compiled_kernel = compiled
    sim.set_cowell_integrator(integrator)
    sim.set_cowell_tolerance(ADAPTIVE_TOL_KM[integrator])
    for _ in range(ADAPTIVE_STEPS):
        sim.step(ADAPTIVE_DT)
    earth = sim.name_to_index["Earth"]
    rel: np.ndarray = sim.global_states[idx] - sim.global_states[earth]
    return rel, sim.cowell_substeps_taken


@pytest.mark.parametrize("integrator", list(KERNELS))
def test_staged_third_body_agrees_under_adaptive_substepping(
    integrator: str, db_session_factory: Callable[[], Session],
) -> None:
    ref, n_ref = _adaptive_run(db_session_factory(), integrator, compiled=False)
    twin, n_twin = _adaptive_run(db_session_factory(), integrator, compiled=True)
    assert n_ref > 3 * ADAPTIVE_STEPS, "guard: the tolerance must force real sub-stepping"
    assert n_ref == n_twin
    diff = _norm_difference(ref, twin)
    assert diff < KERNEL_AGREEMENT_REL_TOL, f"{integrator}: {diff:.3e}"


@pytest.mark.parametrize("integrator", list(KERNELS))
def test_adaptive_comparison_would_detect_a_kernel_that_used_its_step_start_as_t0(
    integrator: str, db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control. A wrapper rewrites the `t0` column to the *kernel call's own* start time before
    each call - the bug this twin avoids by reading the column. Sub-steps then see `stage time - their
    own start` instead of `stage time - the arena step's start`, the perturber is advanced too little,
    and the comparison with the NumPy path must fail."""
    ref, _ = _adaptive_run(db_session_factory(), integrator, compiled=False)
    real = simulator_module._COWELL_KERNELS[integrator]

    def wrong_t0(dt: float, t: float, state: Any, mu: Any, parents: Any, idx: Any, flags: Any,
                 j2: Any, zonal: Any, drag_p: Any, drag_of: Any, tables: Any, meta: Any,
                 tess: Any, vw: Any, third: Any, rel_out: Any) -> int:
        third[idx, _T0_COL] = t
        return int(real(dt, t, state, mu, parents, idx, flags, j2, zonal, drag_p, drag_of, tables, meta,
                        tess, vw, third, rel_out))

    monkeypatch.setitem(simulator_module._COWELL_KERNELS, integrator, wrong_t0)
    wrong, _ = _adaptive_run(db_session_factory(), integrator, compiled=True)
    diff = _norm_difference(ref, wrong)
    assert diff > 1e3 * KERNEL_AGREEMENT_REL_TOL, (
        f"{integrator}: a kernel using the wrong t0 was not detected (diff {diff:.3e})")


def test_adaptive_substepping_really_moves_the_kernel_start(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control above is only meaningful if the kernel is in fact called with start times later than
    the arena step's, while the t0 column holds the arena step's."""
    seen: list[tuple[float, float]] = []
    real = simulator_module._COWELL_KERNELS["rk4"]

    def spy(dt: float, t: float, state: Any, mu: Any, parents: Any, idx: Any, flags: Any, j2: Any,
            zonal: Any, drag_p: Any, drag_of: Any, tables: Any, meta: Any, tess: Any, vw: Any,
            third: Any, rel_out: Any) -> int:
        seen.append((t, float(third[idx[0], _T0_COL])))
        return int(real(dt, t, state, mu, parents, idx, flags, j2, zonal, drag_p, drag_of, tables, meta,
                        tess, vw, third, rel_out))

    monkeypatch.setitem(simulator_module._COWELL_KERNELS, "rk4", spy)
    _adaptive_run(db_session_factory(), "rk4", compiled=True)
    assert any(t > t0 for t, t0 in seen), "no sub-step started after the arena step's start"
    assert all(t >= t0 for t, t0 in seen)


# --- 4. Negative controls on the fixed-step comparison ------------------------------------------------

@pytest.mark.parametrize("integrator", list(KERNELS))
@pytest.mark.parametrize("mode,staged", MODES, ids=[m for m, _ in MODES])
def test_comparison_would_detect_a_dropped_third_body_term(
    integrator: str, mode: str, staged: bool, db_session_factory: Callable[[], Session],
) -> None:
    ref_sim, idx = _moon(db_session_factory(), staged)
    ref = _run_reference(ref_sim, integrator, idx, 30)
    ker_sim, _ = _moon(db_session_factory(), staged)
    ker_sim._cowell_flags &= ~kernels.COWELL_THIRD_BODY
    twin = _run_kernel(ker_sim, integrator, idx, 30)
    diff = _norm_difference(ref, twin)
    assert diff > 1e3 * KERNEL_AGREEMENT_REL_TOL, f"{integrator}/{mode}: dropped term moved state by {diff:.3e}"


@pytest.mark.parametrize("integrator", list(KERNELS))
@pytest.mark.parametrize("mode,staged", MODES, ids=[m for m, _ in MODES])
def test_comparison_would_detect_a_perturbed_perturber_mu(
    integrator: str, mode: str, staged: bool, db_session_factory: Callable[[], Session],
) -> None:
    """A relative 1e-9 nudge to the Sun's mu on the kernel side only; see the module docstring for why
    this needs the 300-step horizon."""
    ref_sim, idx = _moon(db_session_factory(), staged)
    ref = _run_reference(ref_sim, integrator, idx, 300)
    ker_sim, _ = _moon(db_session_factory(), staged)
    ker_sim.mu_array[ker_sim.name_to_index["Sun"]] *= (1.0 + 1e-9)
    twin = _run_kernel(ker_sim, integrator, idx, 300)
    diff = _norm_difference(ref, twin)
    assert diff > KERNEL_AGREEMENT_REL_TOL, (
        f"{integrator}/{mode}: a 1e-9 perturbed perturber mu was not detected (diff {diff:.3e})")
