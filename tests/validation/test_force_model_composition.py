"""
The force-model composition layer: `registry.py`'s force-model section and `forces.py`.

Most of these tests operate on plain NumPy arrays rather than a built `Simulation`, the same way
`test_kernel_equivalence.py`'s scalar-helper tests exercise `kernels.py` directly - `resolve_force_models`
and `compose_accelerations` are pure functions of their arguments and do not need a database or an
arena to be meaningfully tested. A handful go through a real `Simulation` (via `scenarios.py`) to prove
the arena hooks (`force_model_mask`, `force_model_params`, `accel_accum`) and the public methods
(`enable_force_model`, `accelerations`) are wired together correctly end to end.

The two demonstration kernels registered at the bottom of `forces.py` - `test_constant_accel` and
`test_radial_bias` - are the fixtures used throughout. Neither is physics; see their docstrings.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from orbital_engine import forces, registry, scenarios
from orbital_engine.exceptions import RegistryError
from orbital_engine.simulator import Simulation

CONST_MODEL = "test_constant_accel"
RADIAL_MODEL = "test_radial_bias"


def _bit(name: str) -> np.uint64:
    return np.uint64(1) << np.uint64(registry.get_force_model(name).bit)


# ==================================================================================================
# Pure array-level tests: resolve_force_models / compose_accelerations directly.
# ==================================================================================================

def test_masks_compose_via_bitwise_or() -> None:
    """
    Enabling two independent models on the same body is `mask |= bit_a; mask |= bit_b`; the resolved
    body must appear in *both* models' index sets, and a body carrying only one bit must appear in
    only that one. This is the composability the whole layer exists for - a body's physics is
    whichever bits are set, and setting two is no different in kind from setting one.
    """
    active_mask = np.ones(3, dtype=np.bool_)
    mask = np.zeros(3, dtype=np.uint64)
    mask[0] = _bit(CONST_MODEL)
    mask[1] = _bit(RADIAL_MODEL)
    mask[2] = _bit(CONST_MODEL) | _bit(RADIAL_MODEL)

    assert int(mask[2] & _bit(CONST_MODEL)) != 0
    assert int(mask[2] & _bit(RADIAL_MODEL)) != 0

    resolved, dispatch_idx = forces.resolve_force_models(active_mask, mask, 3, {})
    by_name = {rm.name: rm for rm in resolved}

    assert set(by_name) == {CONST_MODEL, RADIAL_MODEL}
    assert list(by_name[CONST_MODEL].indices) == [0, 2]
    assert list(by_name[RADIAL_MODEL].indices) == [1, 2]
    assert list(dispatch_idx) == [0, 1, 2]


def test_zero_mask_bodies_are_never_touched() -> None:
    """
    A body with mask 0 is not merely *reset* to zero - it is never written at all. Pre-filling its row
    with a sentinel and confirming the sentinel survives `compose_accelerations` proves the "no-op" is
    architectural (the row is outside `dispatch_idx`, so no code path can reach it), not incidental.
    """
    active_mask = np.ones(2, dtype=np.bool_)
    mask = np.zeros(2, dtype=np.uint64)
    mask[0] = _bit(CONST_MODEL)  # body 1 stays at mask 0

    params: dict[str, NDArray[np.float64]] = {}
    resolved, dispatch_idx = forces.resolve_force_models(active_mask, mask, 2, params)
    assert 1 not in dispatch_idx.tolist()

    params[CONST_MODEL][0] = [1.0, 2.0, 3.0]

    out = np.full((2, 3), 9.0)  # sentinel: compose_accelerations must never touch row 1
    state = np.zeros((2, 6))
    mu = np.zeros(2)
    parent_indices = np.array([0, 1], dtype=np.int32)

    forces.compose_accelerations(resolved, dispatch_idx, 0.0, state, mu, parent_indices, out)

    assert np.array_equal(out[0], [1.0, 2.0, 3.0])
    assert np.array_equal(out[1], [9.0, 9.0, 9.0]), "a zero-mask body's row was written to"


def test_accumulation_is_correct_with_several_models_enabled() -> None:
    """
    Body 0 carries both demo models with known coefficients and known geometry, so the composed
    result is analytically predictable: `test_constant_accel` contributes a fixed vector, and
    `test_radial_bias` contributes `k` along the unit vector from body 0's parent (body 2, at the
    origin) to body 0, at (10, 0, 0) - i.e. (k, 0, 0). Body 1 carries only the constant model and must
    not pick up any radial contribution, proving accumulation does not leak across bodies.
    """
    active_mask = np.ones(3, dtype=np.bool_)  # 0: both models, 1: constant only, 2: parent (no model)
    mask = np.zeros(3, dtype=np.uint64)
    mask[0] = _bit(CONST_MODEL) | _bit(RADIAL_MODEL)
    mask[1] = _bit(CONST_MODEL)

    params: dict[str, NDArray[np.float64]] = {}
    resolved, dispatch_idx = forces.resolve_force_models(active_mask, mask, 3, params)

    params[CONST_MODEL][0] = [1.0, 2.0, 3.0]
    params[CONST_MODEL][1] = [0.0, 0.0, 0.0]
    params[RADIAL_MODEL][0, 0] = 5.0

    state = np.zeros((3, 6))
    state[0, :3] = [10.0, 0.0, 0.0]
    parent_indices = np.array([2, 2, 2], dtype=np.int32)  # everyone's parent is body 2, at the origin
    mu = np.zeros(3)

    out = np.zeros((3, 3))
    forces.compose_accelerations(resolved, dispatch_idx, 0.0, state, mu, parent_indices, out)

    assert out[0] == pytest.approx([6.0, 2.0, 3.0], abs=1e-12), "constant + radial did not sum"
    assert out[1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12), "constant-only body picked up radial"
    assert out[2] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12), "parent body was never enabled"


def test_a_kernel_given_an_empty_index_array_is_a_true_noop() -> None:
    """
    `ForceKernel`'s contract requires correctness on an empty `indices`, independent of the
    composition layer's own skip-if-unused optimisation. Exercises both demo kernels directly.
    """
    out = np.full((2, 3), 7.0)
    state = np.zeros((2, 6))
    mu = np.zeros(2)
    parent_indices = np.array([0, 1], dtype=np.int32)
    empty = np.array([], dtype=np.int64)

    const_model = registry.get_force_model(CONST_MODEL)
    radial_model = registry.get_force_model(RADIAL_MODEL)
    const_model.kernel(empty, 0.0, state, mu, parent_indices, np.zeros((2, 3)), out)
    radial_model.kernel(empty, 0.0, state, mu, parent_indices, np.zeros((2, 1)), out)

    assert np.array_equal(out, np.full((2, 3), 7.0))


def test_resolve_drops_models_with_no_active_body() -> None:
    """
    A registered model that no body has enabled must not appear in `resolved` at all - dispatch cost
    is meant to track models *in use*, not models *registered*.
    """
    active_mask = np.ones(2, dtype=np.bool_)
    mask = np.zeros(2, dtype=np.uint64)  # nobody has any bit set

    resolved, dispatch_idx = forces.resolve_force_models(active_mask, mask, 2, {})
    assert resolved == []
    assert dispatch_idx.size == 0


def test_get_unregistered_force_model_raises_registry_error() -> None:
    with pytest.raises(RegistryError):
        registry.get_force_model("does_not_exist")


def test_register_force_model_rejects_a_duplicate_name() -> None:
    """
    Does not consume a bit: the registry rejects the name before incrementing its counter, so this is
    safe to run without permanently reducing the 64-bit budget for the rest of the test session -
    unlike deliberately exhausting the registry, which is not tested here for exactly that reason.
    """
    with pytest.raises(RegistryError):
        registry.register_force_model(CONST_MODEL)(lambda *a: None)  # type: ignore[arg-type]


def test_mask_for_ors_bits_together() -> None:
    expected = _bit(CONST_MODEL) | _bit(RADIAL_MODEL)
    assert registry.mask_for([CONST_MODEL, RADIAL_MODEL]) == expected
    assert registry.mask_for([]) == np.uint64(0)


# ==================================================================================================
# Dispatch cost: calls, not timing. A direct count of kernel invocations is exact and immune to
# system noise, unlike a wall-clock ratio - see test_scaling_invariants.py for the complementary
# timing-based check of the same property.
# ==================================================================================================

@registry.register_force_model("test_call_counter")
def _counting_kernel(
    indices: NDArray[np.int64],
    t: float,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Test instrumentation, not a force model: counts how many times it is called, via a module-level
    list rather than a closure variable so the count survives being read from outside. A production
    `ForceKernel` must be stateless (see forces.py); this one deliberately is not, because its whole
    job is to observe the composition layer's own call count.
    """
    _CALL_COUNT.append(len(indices))


_CALL_COUNT: List[int] = []


@pytest.mark.parametrize("n_bodies", [8, 4096])
def test_dispatch_calls_the_kernel_exactly_once_regardless_of_body_count(n_bodies: int) -> None:
    """
    One registered, active model must be dispatched with exactly one Python-level call, whether it
    applies to 8 bodies or 4096. The vectorised arithmetic inside that one call naturally costs more
    at 4096 than at 8 - that is expected and is not what this test is about - but the *call count*,
    which is what CLAUDE.md's "dispatch cost scales with models, never bodies" is a claim about, must
    not move at all.
    """
    _CALL_COUNT.clear()
    active_mask = np.ones(n_bodies, dtype=np.bool_)
    mask = np.full(n_bodies, _bit("test_call_counter"), dtype=np.uint64)

    resolved, dispatch_idx = forces.resolve_force_models(active_mask, mask, n_bodies, {})
    assert len(resolved) == 1

    out = np.zeros((n_bodies, 3))
    state = np.zeros((n_bodies, 6))
    mu = np.zeros(n_bodies)
    parent_indices = np.arange(n_bodies, dtype=np.int32)

    forces.compose_accelerations(resolved, dispatch_idx, 0.0, state, mu, parent_indices, out)

    assert len(_CALL_COUNT) == 1, f"kernel dispatched {len(_CALL_COUNT)} times, expected exactly 1"
    assert _CALL_COUNT[0] == n_bodies, "the one call did not see the whole active set"


# ==================================================================================================
# Simulation-level: the arena hooks and public methods, end to end.
# ==================================================================================================

def test_a_freshly_built_simulation_has_no_force_models_enabled(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The default, unconfigured state is exactly today's engine: pure Keplerian, zero overhead from
    this layer. Existing scenarios and tests that never call `enable_force_model` must be completely
    unaffected by its existence - this is the property that lets all pre-existing tests pass unchanged.
    """
    sim = scenarios.two_body(db_session_factory())
    assert sim._resolved_force_models == []
    assert np.array_equal(sim.force_model_mask, np.zeros(sim.max_capacity, dtype=np.uint64))

    accel = sim.accelerations(0.0)
    assert np.array_equal(accel, np.zeros((sim.max_capacity, 3)))


def test_enable_force_model_end_to_end(db_session_factory: Callable[[], Session]) -> None:
    """
    `enable_force_model` is the whole sweep surface: name in, arena state out. Confirms the mask bit,
    the parameter array, and the composed result all agree with what was asked for, through the public
    API rather than by reaching into `forces.py` directly.
    """
    sim = scenarios.two_body(db_session_factory())
    secondary = sim.name_to_index["Secondary"]

    sim.enable_force_model(CONST_MODEL, bodies=secondary, ax=1.5, ay=-2.0, az=0.0)

    assert int(sim.force_model_mask[secondary] & _bit(CONST_MODEL)) != 0
    assert sim.force_model_params[CONST_MODEL][secondary] == pytest.approx([1.5, -2.0, 0.0])

    accel = sim.accelerations(0.0)
    assert accel[secondary] == pytest.approx([1.5, -2.0, 0.0])

    primary = sim.name_to_index["Primary"]
    assert accel[primary] == pytest.approx([0.0, 0.0, 0.0])


def test_disabling_a_model_leaves_no_stale_acceleration(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Regression guard for a stale-row bug found in review, before this layer first landed.

    `compose_accelerations` only writes rows in the current dispatch set, so when a body's last
    enabled model is switched off its row drops out of that set and is never written again. Unless
    the arena clears the accumulator on re-resolve, the body keeps reporting the disabled model's
    final acceleration. A sweep toggles models off as routinely as on: "Keplerian + J2" followed by
    "Keplerian" would silently still include J2, producing a plausible trajectory for the wrong
    configuration. The expected value is exactly zero, not approximately - nothing may remain.
    """
    sim = scenarios.two_body(db_session_factory())
    secondary = sim.name_to_index["Secondary"]

    sim.enable_force_model(CONST_MODEL, bodies=secondary, ax=7.0, ay=0.0, az=0.0)
    assert sim.accelerations(0.0)[secondary] == pytest.approx([7.0, 0.0, 0.0])

    sim.force_model_mask[secondary] = 0
    sim.resolve_force_models()

    assert np.array_equal(sim.accelerations(0.0)[secondary], [0.0, 0.0, 0.0]), (
        "a disabled model's acceleration survived re-resolve"
    )


def test_enable_force_model_rejects_an_unknown_coefficient(
    db_session_factory: Callable[[], Session],
) -> None:
    sim = scenarios.two_body(db_session_factory())
    with pytest.raises(ValueError):
        sim.enable_force_model(CONST_MODEL, bodies=0, not_a_real_param=1.0)


def test_accelerations_honours_an_explicit_state_not_just_the_committed_one(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The `AccelerationProvider` contract exists specifically so an integrator can evaluate the force
    field at an intermediate state that is not `sim.global_states` - an RK-family sub-stage, most
    obviously. `test_radial_bias` depends on `state`, so moving the body in a caller-supplied state
    array (leaving `global_states` untouched) must move the result the same way.
    """
    sim = scenarios.two_body(db_session_factory())
    secondary = sim.name_to_index["Secondary"]
    primary = sim.name_to_index["Primary"]

    sim.enable_force_model(RADIAL_MODEL, bodies=secondary, k=2.0)

    committed_accel = sim.accelerations(0.0).copy()

    candidate_state = sim.global_states.copy()
    candidate_state[secondary, :3] = candidate_state[primary, :3] + np.array([0.0, 100.0, 0.0])

    candidate_accel = sim.accelerations(0.0, state=candidate_state)

    assert not np.allclose(candidate_accel[secondary], committed_accel[secondary]), (
        "accelerations() ignored the explicit state argument")
    assert candidate_accel[secondary] == pytest.approx([0.0, 2.0, 0.0], abs=1e-12)
    # global_states itself must be untouched by evaluating a candidate state.
    assert not np.allclose(sim.global_states[secondary], candidate_state[secondary])
