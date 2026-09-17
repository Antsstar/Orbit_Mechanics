"""
The sweep harness: one scenario, N model configurations, diffed against a common reference.

**Why this is a library module, not a script.** `CLAUDE.md`'s Direction states the sweep is this
project's main loop - "the benchmark sweep is the primary use case, not a script layered on top" - so
it lives in `src/orbital_engine`, importable and testable like any other module, rather than in
`benchmarks/`. `benchmarks/frontier_plot.py` is a *consumer* of this module, not its home.

**What a model configuration is.** `ModelConfig` is plain data: a name, a propagator
(`custom_types.PropagatorType`), that propagator's coefficients (mandatory for `SECULAR_J2`, unused
otherwise - matching `Simulation.set_propagator`'s own contract), an optional `mean_seed` flag
(`SECULAR_J2` only, see `propagators.mean_seeded_p`), zero or more force models with their own
coefficients, and a step size. A sweep is a list of these applied, one at a time, to independent
builds of the same scenario. Nothing here is a code path a new tier needs to branch on - a new
configuration is a new `ModelConfig` value, never a new function.

**What `bodies=None` means.** Most sweeps compare satellite fidelity, not a scenario's Keplerian
primary, so a config's default target is every *eligible* body - active, not a head, not a barycenter,
not its own kinematic bubble, and massless - exactly the set `Simulation.set_propagator`'s `COWELL` and
`SECULAR_J2` branches already require, so a default-targeted config can never fail that validation for
a reason the caller did not choose. Passing `bodies` explicitly (a sequence of body names) narrows the
target for a specific comparison.

**Truth, computed once.** `run_sweep` builds the scenario exactly once to establish truth
(`reference.reference_for`, at `TRUTH_RTOL`/`TRUTH_ATOL` by default) from the shared initial condition,
then builds a fresh `Simulation` per configuration - so no configuration's mutations (mask bits,
propagator assignment, seeded elements) can leak into another's, and truth generation is excluded from
every configuration's timing.

**What is reported.** Position error against truth at the horizon, as **statistics over bodies**
(median, RMS, max) - never a single satellite's figure. `docs/architecture.md`'s secular-J2 section
measured a single satellite's error varying by two orders of magnitude with initial phase alone; a
sweep that reported one body's number would rank tiers by which phase happened to be sampled, not by
which model is more accurate. Wall time is minimum-of-batches (`benchmark.measure`), timing only the
step loop to the horizon - truth generation and `Simulation` construction are excluded, both here and
in each batch's `setup`.

**No plotting dependency.** This module never imports `matplotlib`; `benchmarks/frontier_plot.py`
does that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .custom_types import PropagatorType
from .reference import TRUTH_ATOL, TRUTH_RTOL, ReferenceTrajectory, reference_for
from .benchmark import measure
from .simulator import Simulation

__all__ = [
    "ForceModelSpec", "ModelConfig", "ErrorStats", "SweepResult",
    "eligible_bodies", "apply_config", "run_sweep",
]


@dataclass(frozen=True)
class ForceModelSpec:
    """One `Simulation.enable_force_model` call: a registered model name and its coefficients, if any
    (`registry.py`'s `param_names`) - e.g. `ForceModelSpec("j2", {"j2": EARTH_J2, "r_eq": EARTH_R_EQ})`.
    """
    name: str
    coefficients: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig(object):
    """
    One model-fidelity tier, expressed entirely as data - see the module docstring.

    `propagator_coefficients` is forwarded to `Simulation.set_propagator(**coefficients)`; it is
    mandatory (`j2`, `r_eq`) for `PropagatorType.SECULAR_J2` and must be empty for every other
    propagator, the same strictness `set_propagator` itself enforces. `mean_seed` is likewise only
    meaningful for `SECULAR_J2` - `apply_config` raises before ever calling into `Simulation` if either
    is set inconsistently, rather than letting `set_propagator`'s own error surface out of context.
    """
    name: str
    propagator: PropagatorType
    dt: float
    propagator_coefficients: Mapping[str, float] = field(default_factory=dict)
    mean_seed: bool = False
    force_models: Sequence[ForceModelSpec] = field(default_factory=tuple)
    bodies: Optional[Sequence[str]] = None


@dataclass(frozen=True)
class ErrorStats:
    """Position error against truth at the sweep horizon, in km, over every body the configuration
    governed. Never a single body's figure - see the module docstring."""
    median_km: float
    rms_km: float
    max_km: float


@dataclass(frozen=True)
class SweepResult:
    """One configuration's outcome: its name, error statistics, minimum-of-batches wall time (us) for
    the propagation alone, and how many bodies the statistics were taken over."""
    config_name: str
    error: ErrorStats
    wall_time_us: float
    n_bodies: int


def eligible_bodies(sim: Simulation) -> NDArray[np.int64]:
    """
    Every active body a `ModelConfig` may legally target by default: not a head, not a system
    barycenter, not its own kinematic bubble root, and massless (`mu == 0`) - exactly the restrictions
    `Simulation.set_propagator` already enforces for `COWELL` and `SECULAR_J2` (see that method's
    docstring), reused here as the harness's "every satellite" default so a config never needs to name
    a scenario's bodies explicitly to be valid for every propagator tier at once.
    """
    is_root = sim.body_sys_map == np.arange(sim.max_capacity, dtype=np.int32)
    eligible = (
        sim.active_mask & ~sim.is_head & ~sim.is_system & ~is_root & (sim.mu_array == 0.0)
    )
    return np.flatnonzero(eligible).astype(np.int64)


def _resolve_bodies(sim: Simulation, config: ModelConfig) -> NDArray[np.int64]:
    if config.bodies is None:
        return eligible_bodies(sim)
    return np.asarray([sim.name_to_index[name] for name in config.bodies], dtype=np.int64)


def apply_config(sim: Simulation, config: ModelConfig) -> NDArray[np.int64]:
    """
    Apply one `ModelConfig` to a freshly built `Simulation`: resolve its target bodies, assign the
    propagator (with `mean_seed` and coefficients where applicable), and enable every listed force
    model. Returns the resolved body slots, so a caller does not have to re-derive them to compute
    error statistics against the same set. Exposed standalone so a caller can use it without
    `run_sweep`'s truth generation and timing.

    Raises `ValueError` if `mean_seed` is set for any propagator other than `SECULAR_J2` - the same
    strictness `Simulation.set_propagator` applies to an unexpected coefficient, checked before it
    would otherwise be silently ignored by a propagator with no use for it.
    """
    idx = _resolve_bodies(sim, config)
    if idx.size == 0:
        return idx

    if config.mean_seed and config.propagator != PropagatorType.SECULAR_J2:
        raise ValueError(
            f"config '{config.name}': mean_seed=True is only meaningful for PropagatorType.SECULAR_J2, "
            f"got {config.propagator.name}."
        )

    # `mean_seed` is passed positionally, not by keyword: `set_propagator`'s remaining parameter is
    # `**coefficients: float`, and a keyword `mean_seed=...` alongside `**config.propagator_coefficients`
    # (an arbitrary-keyed `Mapping[str, float]`) would make mypy --strict conservatively check the
    # mapping's float values against the `bool` parameter, since it cannot prove the mapping never
    # contains a "mean_seed" key. Binding it positionally removes the ambiguity.
    sim.set_propagator(idx, config.propagator, config.mean_seed, **config.propagator_coefficients)

    for fm in config.force_models:
        # `.tolist()` rather than the `NDArray[np.int64]` itself: `enable_force_model`'s `bodies`
        # parameter is typed `int | Sequence[int] | NDArray[np.bool_]`, which does not include an
        # integer array even though the implementation accepts one via ordinary fancy indexing - a
        # pre-existing signature narrowness in `Simulation.enable_force_model`, not something this
        # module should widen.
        sim.enable_force_model(fm.name, idx.tolist(), **fm.coefficients)

    return idx


def _position_errors_km(
    sim: Simulation, idx: NDArray[np.int64], truth: ReferenceTrajectory,
) -> NDArray[np.float64]:
    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    truth_positions = np.stack([truth.position_of(slot_to_name[int(s)])[-1] for s in idx])
    return cast(
        NDArray[np.float64],
        np.linalg.norm(sim.global_states[idx, :3] - truth_positions, axis=1),
    )


def _error_stats(errors_km: NDArray[np.float64]) -> ErrorStats:
    return ErrorStats(
        median_km=float(np.median(errors_km)),
        rms_km=float(np.sqrt(np.mean(errors_km ** 2))),
        max_km=float(np.max(errors_km)),
    )


def _time_propagation(
    build_scenario: Callable[[], Simulation],
    config: ModelConfig,
    n_steps: int,
    *,
    batches: int,
    warmup: int,
) -> float:
    """
    Minimum-of-batches wall time (us) for stepping a fresh build of `config` to the horizon, excluding
    scenario construction and `apply_config` from the timed region - `benchmark.measure`'s `setup`
    rebuilds the simulation before every batch, and `inner=1` because one call already performs the
    full `n_steps` propagation (unlike a per-step microbenchmark, where a large `inner` amortises timer
    overhead over many cheap calls).
    """
    holder: dict[str, Simulation] = {}

    def setup() -> None:
        sim = build_scenario()
        sim.record_history = False
        apply_config(sim, config)
        holder["sim"] = sim

    def run() -> None:
        sim = holder["sim"]
        for _ in range(n_steps):
            sim.step(config.dt)

    setup()  # populate `holder` before `measure`'s own warmup loop, which does not call `setup` first.
    return measure(run, batches=batches, inner=1, warmup=warmup, setup=setup).best


def run_sweep(
    build_scenario: Callable[[], Simulation],
    configs: Sequence[ModelConfig],
    horizon_s: float,
    *,
    truth_rtol: float = TRUTH_RTOL,
    truth_atol: float = TRUTH_ATOL,
    oblateness: Optional[Mapping[str, Tuple[float, float]]] = None,
    timing_batches: int = 5,
    timing_warmup: int = 2,
) -> List[SweepResult]:
    """
    Run every configuration in `configs` against one scenario and return one `SweepResult` each, in
    order.

    `build_scenario` takes no arguments and returns a fresh `Simulation` from a fresh session - the
    same shape as a `scenarios.py` builder partially applied over its session argument, e.g.
    `lambda: scenarios.earth_constellation(db_session_factory(), n_sats=60)`. It is called once for
    truth and once per configuration, so no configuration's state can leak into another's or into
    truth.

    `oblateness` is forwarded unchanged to `reference.reference_for` - pass it whenever any configured
    tier includes a J2 model, or the truth is point-mass and every J2 tier is being judged against the
    wrong reference (see `reference.py`'s module docstring on why this argument is never inferred from
    a `Simulation`'s own configuration).
    """
    truth_sim = build_scenario()
    times = np.array([0.0, horizon_s], dtype=np.float64)
    truth = reference_for(
        truth_sim, times, rtol=truth_rtol, atol=truth_atol, oblateness=oblateness,
    )

    results: List[SweepResult] = []
    for config in configs:
        n_steps = max(1, int(round(horizon_s / config.dt)))

        sim = build_scenario()
        sim.record_history = False
        idx = apply_config(sim, config)
        for _ in range(n_steps):
            sim.step(config.dt)
        errors_km = _position_errors_km(sim, idx, truth)

        wall_time_us = _time_propagation(
            build_scenario, config, n_steps, batches=timing_batches, warmup=timing_warmup,
        )

        results.append(SweepResult(
            config_name=config.name,
            error=_error_stats(errors_km),
            wall_time_us=wall_time_us,
            n_bodies=int(idx.size),
        ))

    return results
