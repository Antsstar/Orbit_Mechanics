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

**Contact windows, optionally.** Position error in km is not the unit anyone's decision is in. Pass
`access=AccessSpec(...)` and every `SweepResult` additionally carries `access.AccessMetrics`: rise
and set shifts against truth, duration and total-contact error, and passes gained or lost. It is
strictly additive - the error statistics and the timing are computed by exactly the code path they
were before, and `access is None` (the default) does not build a grid, a station or a second truth.
See `access.py` for the metric definitions and the matching rule.

**External tiers.** Some models the sweep must rank are not something the arena can step: SGP4 is
an analytic theory evaluated by a stateful third-party object (`sgp4_bridge.py`), and `CLAUDE.md`
forbids stateful dependencies inside `step()`. `ExternalTier` is the second kind of tier for that:
plain data naming the bodies and a pure function `times_s -> (n_times, n_bodies, 3)` positions
relative to a named central body. `run_sweep(..., external=[...])` scores it against the **same**
truth, the same way - error at the horizon as statistics over bodies, minimum-of-batches timing of
producing one state per `dt` up to the horizon, and `AccessMetrics` from the same access grid - and
appends its `SweepResult` after the `ModelConfig` results. Nothing about an engine configuration's
path changes when `external` is empty or not.

**Station-keeping Delta-v, optionally.** For drag the decision unit is propellant. Pass
`station_keeping=StationKeepingSpec(...)` with `delta_v_baseline="<config name>"` and every engine
`SweepResult` carries `stationkeeping.DeltaVMetrics`: per-body total Delta-v, raises and steady
m/s/day, and the signed relative budget error against the **named baseline configuration** - not
against truth, which has no drag and so no Delta-v. It is the same kind of additive measurement as
`access`: one extra propagation per configuration from a fresh build, at that configuration's own
`dt`, and nothing about the error or timing runs changes. See `station_keeping_for`.

**Inter-satellite links, optionally.** Pass `isl=IslSpec(...)` and every `SweepResult` (engine and
external) carries `isl`, an `access.AccessMetrics` over satellite *pairs*: the same overlap matching,
shift statistics and lost/gained counts as `access`, computed by the same code (`isl.py`). It is
wired exactly like `access` - one dense truth for the sweep (shared with `access` when the two grids
coincide), one extra propagation per configuration, the same divisibility rule - and changes nothing
else.

**No plotting dependency.** This module never imports `matplotlib`; `benchmarks/frontier_plot.py`
does that.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .access import (
    AccessMetrics, AccessSpec, access_grid, compare_windows, windows_from_positions,
    windows_from_simulation, windows_from_truth,
)
from .custom_types import PropagatorType
from .isl import (
    IslSpec, compare_isl_windows, isl_windows, isl_windows_from_simulation, isl_windows_from_truth,
)
from .stationkeeping import (
    MIN_SAMPLES_PER_ORBIT, BodyDeltaV, DeltaVMetrics, StationKeepingSpec, delta_v_metrics,
    run_station_keeping, window_period_s,
)
from .reference import TRUTH_ATOL, TRUTH_RTOL, ReferenceTrajectory, reference_for
from .benchmark import measure
from .simulator import Simulation

__all__ = [
    "ForceModelSpec", "ModelConfig", "ExternalTier", "ErrorStats", "SweepResult",
    "eligible_bodies", "apply_config", "run_sweep", "access_metrics_for", "score_external",
    "check_station_keeping_dt", "station_keeping_for", "isl_metrics_for",
]


@dataclass(frozen=True)
class ForceModelSpec:
    """One `Simulation.enable_force_model` call: a registered model name and its coefficients, if any
    (`registry.py`'s `param_names`) - e.g. `ForceModelSpec("j2", {"j2": EARTH_J2, "r_eq": EARTH_R_EQ})`.

    `body_coefficients` holds coefficients whose value is a *body*, given by name, e.g.
    `ForceModelSpec("third_body", body_coefficients={"perturber": "Sun"})`. `apply_config` resolves
    each name to its arena slot for the simulation being configured, so a config never carries a slot
    number, which is not stable across builds. A key may appear in only one of the two mappings.
    """
    name: str
    coefficients: Mapping[str, float] = field(default_factory=dict)
    body_coefficients: Mapping[str, str] = field(default_factory=dict)


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
class ExternalTier:
    """
    A model tier whose trajectory is produced outside the arena - see the module docstring.

    `positions(times_s)` takes seconds since the scenario's `t = 0` and returns `(n_times,
    len(bodies), 3)` km, **relative to `central_body`**, in the scenario's inertial frame. It must be
    a pure function: `run_sweep` calls it repeatedly to time it. `bodies` are body names in the
    scenario, which is how its rows are matched to truth. `dt` is the cadence at which the tier is
    asked for states when it is timed, so its cost is comparable to an engine tier stepping at `dt`;
    it does not affect the error, which is evaluated at the horizon itself.

    Truth for an external tier is the scenario's truth, which was seeded from the scenario's initial
    state - so the tier is only meaningfully scored if that state is the tier's own state at `t = 0`
    (`scenarios.tle_satellites` guarantees this for SGP4).

    An external tier's `SweepResult.delta_v` is always `None`, with or without `station_keeping`: it
    is a pure function of time, so it cannot take a burn, and a station-keeping budget needs the
    trajectory to respond to the controller's impulses.
    """
    name: str
    dt: float
    bodies: Sequence[str]
    central_body: str
    positions: Callable[[NDArray[np.float64]], NDArray[np.float64]]


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
    the propagation alone, and how many bodies the statistics were taken over.

    `access` is `None` unless `run_sweep` was given an `AccessSpec`, and `delta_v` is `None` unless it
    was given a `StationKeepingSpec` (and always for an `ExternalTier`); neither affects the four
    fields above, which are computed by the same code either way.
    """
    config_name: str
    error: ErrorStats
    wall_time_us: float
    n_bodies: int
    access: Optional[AccessMetrics] = None
    delta_v: Optional[DeltaVMetrics] = None
    isl: Optional[AccessMetrics] = None


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
        coefficients = dict(fm.coefficients)
        for key, body_name in fm.body_coefficients.items():
            if key in coefficients:
                raise ValueError(
                    f"config '{config.name}', force model '{fm.name}': coefficient '{key}' is given both "
                    f"as a number and as a body name."
                )
            if body_name not in sim.name_to_index:
                raise KeyError(
                    f"config '{config.name}', force model '{fm.name}': {key}={body_name!r} is not a body "
                    f"in this simulation; have {sorted(sim.name_to_index)}"
                )
            coefficients[key] = float(sim.name_to_index[body_name])
        sim.enable_force_model(fm.name, idx.tolist(), **coefficients)

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


def _access_body_names(
    sim: Simulation, idx: NDArray[np.int64], spec: AccessSpec | IslSpec,
) -> Tuple[List[str], List[int]]:
    """
    The body names and slots the access metrics are reported for, in a single fixed order.

    Default is the configuration's own target set, so the window metrics and the position-error
    statistics describe the same satellites; `AccessSpec.bodies` narrows it by name. The order is
    what `geometry.AccessWindow.body_index` refers to, and it is the *same* list for truth and for
    the model - which is what makes a `body_index` comparable across the two.
    """
    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    if spec.bodies is None:
        slots = [int(s) for s in idx]
        return [slot_to_name[s] for s in slots], slots
    slots = [sim.name_to_index[name] for name in spec.bodies]
    return list(spec.bodies), slots


def _sample_grid(
    config: ModelConfig, horizon_s: float, sample_dt_s: float, metric: str, spec_name: str,
) -> NDArray[np.float64]:
    """
    The shared sample grid for a window metric, refusing a configuration whose `dt` does not divide
    its spacing - see `access_metrics_for` for why that is a silent wrong answer rather than a
    rounding detail. Used by both `access_metrics_for` and `isl_metrics_for`.
    """
    grid = access_grid(horizon_s, sample_dt_s)
    spacing = float(grid[1] - grid[0])
    steps_per_sample = spacing / config.dt
    if abs(steps_per_sample - round(steps_per_sample)) > 1e-9 or round(steps_per_sample) < 1:
        raise ValueError(
            f"config '{config.name}': dt={config.dt} s does not divide the {metric} sample spacing "
            f"{spacing} s (horizon {horizon_s} s at sample_dt_s={sample_dt_s} s). The {metric} "
            f"metric would then propagate this configuration at a step it was never run with. "
            f"Raise {spec_name}.sample_dt_s to a multiple of every config's dt."
        )
    return grid


def access_metrics_for(
    build_scenario: Callable[[], Simulation],
    config: ModelConfig,
    horizon_s: float,
    spec: AccessSpec,
    access_truth: ReferenceTrajectory,
) -> AccessMetrics:
    """
    Contact-window error for one configuration against an already-integrated truth trajectory.

    `access_truth` must have been sampled on `access_grid(horizon_s, spec.sample_dt_s)` - the same
    grid this function propagates the model onto. Sharing the grid is not an optimisation: the
    linear edge interpolation in `geometry.access_windows` has an `O(h^2)` *bias*, and it only
    cancels out of a difference if both sides carry it (see `access.py`'s "Sampling step").

    A fresh scenario is built and propagated here rather than reusing `run_sweep`'s own run, because
    that run steps straight to the horizon and keeps no history. The trajectory is nonetheless the
    same one: `viz.sample_states(max_dt=config.dt)` sub-steps each sample interval at the
    configuration's own step size.

    That last sentence is only true when `config.dt` **divides** the sample spacing, because
    `sample_states` splits an interval into sub-steps of *at most* `max_dt`: a 10 s grid asked of a
    `dt = 60 s` configuration would take one 10 s step per sample and score a sixfold finer model
    than the one `run_sweep` timed and reported an error for. That is a silent, entirely plausible
    wrong answer, so it raises `ValueError` instead.
    """
    grid = _sample_grid(config, horizon_s, spec.sample_dt_s, "access", "AccessSpec")

    sim = build_scenario()
    sim.record_history = False
    idx = apply_config(sim, config)
    names, slots = _access_body_names(sim, idx, spec)

    model = windows_from_simulation(sim, slots, grid, spec, max_dt=config.dt)
    truth = windows_from_truth(access_truth, names, spec)
    return compare_windows(truth, model)


def isl_metrics_for(
    build_scenario: Callable[[], Simulation],
    config: ModelConfig,
    horizon_s: float,
    spec: IslSpec,
    isl_truth: ReferenceTrajectory,
) -> AccessMetrics:
    """
    Inter-satellite-link window error for one configuration against an already-integrated truth -
    `access_metrics_for` with satellite pairs in place of `(station, body)`, and every argument of
    that function's docstring carries over: `isl_truth` sampled on `access_grid(horizon_s,
    spec.sample_dt_s)`, a fresh build propagated at `config.dt` onto the same grid, and a
    `ValueError` if `config.dt` does not divide the sample spacing.

    The body list - the configuration's own targets unless `spec.bodies` narrows them - is the same
    list, in the same order, for truth and model, so a pair `(a, b)` names the same two satellites
    on both sides.
    """
    grid = _sample_grid(config, horizon_s, spec.sample_dt_s, "ISL", "IslSpec")

    sim = build_scenario()
    sim.record_history = False
    idx = apply_config(sim, config)
    names, slots = _access_body_names(sim, idx, spec)

    model = isl_windows_from_simulation(sim, slots, grid, spec, max_dt=config.dt)
    truth = isl_windows_from_truth(isl_truth, names, spec)
    return compare_isl_windows(truth, model)


def check_station_keeping_dt(
    sim: Simulation, config: ModelConfig, spec: StationKeepingSpec,
) -> None:
    """
    Refuse a configuration whose `dt` the controller cannot observe at: fewer than
    `stationkeeping.MIN_SAMPLES_PER_ORBIT` steps per averaging window (the Keplerian period at the
    band's midpoint, ~5420 s at 292 km, so `dt` must be at most ~1355 s).

    Checked against `sim` - any build of the scenario, read and never mutated - before anything is
    propagated, so a sweep fails in milliseconds rather than after its truth. `StationKeeper` would
    raise the same thing later, without the configuration's name. The step is never substituted:
    the Delta-v must come from the model at the step `run_sweep` timed and scored. An analytic tier's
    position error does not depend on `dt`, so the caller is free to give it a controller-compatible
    one; a Cowell tier's does, and the budget then carries its truncation error too.
    """
    idx = _resolve_bodies(sim, config)
    if idx.size == 0:
        return
    mu = sim.mu_array[idx] + sim.mu_array[sim.parent_indices[idx]]
    if not bool(np.all(mu > 0.0)):
        raise ValueError(f"config {config.name!r}: every station-kept body needs a massive parent")
    period = window_period_s(spec, float(np.max(mu)))
    if period / config.dt < MIN_SAMPLES_PER_ORBIT:
        raise ValueError(
            f"config {config.name!r}: dt={config.dt} s gives {period / config.dt:.2f} samples per "
            f"station-keeping window ({period:.0f} s); the controller's one-period mean needs at least "
            f"{MIN_SAMPLES_PER_ORBIT:.0f}. Give this config a dt <= {period / MIN_SAMPLES_PER_ORBIT:.0f} "
            f"s - the Delta-v is never computed at a different step than the config's own."
        )


def station_keeping_for(
    build_scenario: Callable[[], Simulation],
    config: ModelConfig,
    horizon_s: float,
    spec: StationKeepingSpec,
) -> Tuple[BodyDeltaV, ...]:
    """
    One configuration's station-keeping budget: a fresh build with `apply_config`, stepped to
    `horizon_s` at `config.dt` under `stationkeeping.run_station_keeping` - the same function the
    standalone study calls, so the result is that study's to the bit for the same configuration and
    arena. Every body the configuration targets is station-kept. A configuration with no drag makes
    no raise and returns zero Delta-v: a real prediction, not an error.

    `horizon_s` must already be a whole number of steps (`run_sweep` checks it).
    """
    sim = build_scenario()
    sim.record_history = False
    idx = apply_config(sim, config)
    if idx.size == 0:
        return ()
    run = run_station_keeping(sim, idx, spec, horizon_s, config.dt)
    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    return tuple(
        BodyDeltaV(
            body=slot_to_name[s.body], total_dv_m_s=s.total_dv_km_s * 1e3, n_raises=s.n_burns,
            steady_rate_m_s_per_day=s.steady_rate_km_s_per_s * 1e3 * 86400.0,
        )
        for s in run.summary
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


def score_external(
    tier: ExternalTier,
    horizon_s: float,
    truth: ReferenceTrajectory,
    *,
    timing_batches: int = 5,
    timing_warmup: int = 2,
    access: Optional[AccessSpec] = None,
    access_truth: Optional[ReferenceTrajectory] = None,
    isl: Optional[IslSpec] = None,
    isl_truth: Optional[ReferenceTrajectory] = None,
) -> SweepResult:
    """
    One `ExternalTier` against an already-integrated truth - the external counterpart of one
    configuration's pass through `run_sweep`'s loop, exposed so a caller holding a truth can score a
    tier without re-integrating it.

    `truth` must end at `horizon_s`. The error is `|tier - truth|` at the horizon with both sides
    taken relative to `tier.central_body`. Timing is one call producing a state every `tier.dt` up
    to the horizon. With `access`, `access_truth` must be sampled on `access_grid(horizon_s,
    access.sample_dt_s)`; the tier is evaluated on exactly that grid, so there is no step-size
    divisibility condition to check - an external tier samples, it does not step. `isl` /
    `isl_truth` work the same way on the ISL grid; the tier's positions are relative to
    `tier.central_body`, so `isl.central_body` must name the same body.
    """
    if abs(float(truth.times[-1]) - horizon_s) > 1e-9 * max(1.0, horizon_s):
        raise ValueError(f"truth ends at {truth.times[-1]} s, not at the horizon {horizon_s} s")

    central = truth.position_of(tier.central_body)[-1]
    truth_rel = np.stack([truth.position_of(name)[-1] - central for name in tier.bodies])
    at_horizon = tier.positions(np.array([horizon_s], dtype=np.float64))[0]
    errors_km: NDArray[np.float64] = np.linalg.norm(at_horizon - truth_rel, axis=1)

    n_steps = max(1, int(round(horizon_s / tier.dt)))
    grid: NDArray[np.float64] = np.asarray(
        tier.dt * np.arange(1, n_steps + 1, dtype=np.float64), dtype=np.float64,
    )
    wall_time_us = measure(
        lambda: tier.positions(grid), batches=timing_batches, inner=1, warmup=timing_warmup,
    ).best

    access_metrics: Optional[AccessMetrics] = None
    if access is not None:
        if access_truth is None:
            raise ValueError("access metrics need access_truth sampled on the access grid")
        names = list(tier.bodies) if access.bodies is None else list(access.bodies)
        columns = [list(tier.bodies).index(name) for name in names]
        agrid = access_grid(horizon_s, access.sample_dt_s)
        model = windows_from_positions(tier.positions(agrid)[:, columns, :], agrid, access)
        access_metrics = compare_windows(windows_from_truth(access_truth, names, access), model)

    isl_metrics: Optional[AccessMetrics] = None
    if isl is not None:
        if isl_truth is None:
            raise ValueError("ISL metrics need isl_truth sampled on the ISL grid")
        if isl.central_body != tier.central_body:
            raise ValueError(
                f"tier {tier.name!r} gives positions relative to {tier.central_body!r}, but the ISL "
                f"spec's central body is {isl.central_body!r}")
        isl_names = list(tier.bodies) if isl.bodies is None else list(isl.bodies)
        isl_columns = [list(tier.bodies).index(name) for name in isl_names]
        igrid = access_grid(horizon_s, isl.sample_dt_s)
        isl_model = isl_windows(tier.positions(igrid)[:, isl_columns, :], igrid, isl)
        isl_metrics = compare_isl_windows(
            isl_windows_from_truth(isl_truth, isl_names, isl), isl_model,
        )

    return SweepResult(
        config_name=tier.name,
        error=_error_stats(errors_km),
        wall_time_us=wall_time_us,
        n_bodies=len(tier.bodies),
        access=access_metrics,
        isl=isl_metrics,
    )


def run_sweep(
    build_scenario: Callable[[], Simulation],
    configs: Sequence[ModelConfig],
    horizon_s: float,
    *,
    truth_rtol: float = TRUTH_RTOL,
    truth_atol: float = TRUTH_ATOL,
    oblateness: Optional[Mapping[str, Tuple[float, float]]] = None,
    zonal: Optional[Mapping[str, Tuple[float, Mapping[int, float]]]] = None,
    timing_batches: int = 5,
    timing_warmup: int = 2,
    access: Optional[AccessSpec] = None,
    external: Sequence[ExternalTier] = (),
    station_keeping: Optional[StationKeepingSpec] = None,
    delta_v_baseline: Optional[str] = None,
    isl: Optional[IslSpec] = None,
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
    a `Simulation`'s own configuration). `zonal` is forwarded the same way (J3..J6, see
    `reference.reference_for`); omitting it leaves truth bit-identical to a sweep without the option.

    `access`, when given, adds `SweepResult.access` - contact-window error against the same truth
    model (see `access.py`). It costs **one** extra `reference_for` call for the whole sweep, on the
    dense access grid, and one extra propagation per configuration. The endpoint truth above is left
    alone rather than being read off the dense one: `solve_ivp`'s dense output at the horizon is not
    bit-identical to a run that stops there, and the position-error statistics must not move because
    an unrelated metric was switched on.

    `external` tiers (`ExternalTier`, e.g. `sgp4_bridge.sgp4_tier`) are scored by `score_external`
    against the same two truths and appended, in order, after the `configs` results.

    `station_keeping`, when given, adds `SweepResult.delta_v` to every engine configuration: one extra
    propagation each (`station_keeping_for`), scored against the configuration named by
    `delta_v_baseline`, which is **required** and must name a config in `configs` - it is never
    inferred, because which model is the reference is the comparison's premise, not a default.
    Every config's `dt` is checked against the controller first (`check_station_keeping_dt`), before
    truth is integrated. `delta_v_baseline` without `station_keeping` is refused too, since it would
    otherwise be silently ignored.

    `isl`, when given, adds `SweepResult.isl` - inter-satellite link window error, per satellite
    pair, against the same truth model (see `isl.py`). Same economy as `access`: one dense truth for
    the whole sweep - **the access truth itself** when `access` is also given on the same grid - and
    one extra propagation per configuration. `ErrorStats` and `access` are untouched by it.
    """
    truth_sim = build_scenario()

    if station_keeping is not None:
        names = [c.name for c in configs]
        if delta_v_baseline is None:
            raise ValueError(
                "station_keeping needs delta_v_baseline: the name of the config every Delta-v budget "
                "is scored against (truth has no drag, so there is no truth Delta-v).")
        if len(set(names)) != len(names):
            raise ValueError(f"station_keeping needs unique config names to score by; have {names}")
        if delta_v_baseline not in names:
            raise ValueError(
                f"delta_v_baseline={delta_v_baseline!r} names no config in this sweep; have {names}")
        for config in configs:
            check_station_keeping_dt(truth_sim, config, station_keeping)
    elif delta_v_baseline is not None:
        raise ValueError("delta_v_baseline was given without station_keeping; it would be ignored")

    times = np.array([0.0, horizon_s], dtype=np.float64)
    truth = reference_for(
        truth_sim, times, rtol=truth_rtol, atol=truth_atol, oblateness=oblateness, zonal=zonal,
    )

    access_truth: Optional[ReferenceTrajectory] = None
    if access is not None:
        access_truth = reference_for(
            build_scenario(), access_grid(horizon_s, access.sample_dt_s),
            rtol=truth_rtol, atol=truth_atol, oblateness=oblateness, zonal=zonal,
        )

    isl_truth: Optional[ReferenceTrajectory] = None
    if isl is not None:
        isl_grid = access_grid(horizon_s, isl.sample_dt_s)
        if access_truth is not None and np.array_equal(access_truth.times, isl_grid):
            isl_truth = access_truth
        else:
            # Every truth-model keyword the dense access truth above receives must be forwarded
            # here too - `tests/validation/test_isl.py` asserts that all truth calls in one sweep
            # share their model keywords, which is what catches a pass-through added to one only.
            isl_truth = reference_for(
                build_scenario(), isl_grid,
                rtol=truth_rtol, atol=truth_atol, oblateness=oblateness, zonal=zonal,
            )

    budgets: dict[str, Tuple[BodyDeltaV, ...]] = {}
    results: List[SweepResult] = []
    for config in configs:
        # The horizon must be a whole number of steps. Rounding it silently would score the config
        # at n*dt rather than at horizon_s - against a truth sampled at horizon_s - so the error
        # would include however far the body moves in the difference (199 km for a 5554 s horizon
        # at dt = 60 s, which quietly runs to 5580 s). Refuse instead, as access_metrics_for does.
        ratio = horizon_s / config.dt
        n_steps = int(round(ratio))
        if n_steps < 1 or abs(ratio - n_steps) > 1e-9 * max(1.0, ratio):
            raise ValueError(
                f"config {config.name!r}: horizon_s={horizon_s} is not a whole number of steps of "
                f"dt={config.dt} (ratio {ratio:.6f}); the config would be scored at the wrong time."
            )

        sim = build_scenario()
        sim.record_history = False
        idx = apply_config(sim, config)
        for _ in range(n_steps):
            sim.step(config.dt)
        errors_km = _position_errors_km(sim, idx, truth)

        wall_time_us = _time_propagation(
            build_scenario, config, n_steps, batches=timing_batches, warmup=timing_warmup,
        )

        access_metrics: Optional[AccessMetrics] = None
        if access is not None and access_truth is not None:
            access_metrics = access_metrics_for(
                build_scenario, config, horizon_s, access, access_truth,
            )

        isl_metrics: Optional[AccessMetrics] = None
        if isl is not None and isl_truth is not None:
            isl_metrics = isl_metrics_for(build_scenario, config, horizon_s, isl, isl_truth)

        if station_keeping is not None:
            budgets[config.name] = station_keeping_for(
                build_scenario, config, horizon_s, station_keeping,
            )

        results.append(SweepResult(
            config_name=config.name,
            error=_error_stats(errors_km),
            wall_time_us=wall_time_us,
            n_bodies=int(idx.size),
            access=access_metrics,
            isl=isl_metrics,
        ))

    if station_keeping is not None and delta_v_baseline is not None:
        # Scored after the loop, when the baseline's budget exists whatever its position in `configs`.
        base = budgets[delta_v_baseline]
        results = [
            replace(r, delta_v=delta_v_metrics(budgets[r.config_name], base, delta_v_baseline))
            if budgets[r.config_name] else r
            for r in results
        ]

    for tier in external:
        results.append(score_external(
            tier, horizon_s, truth, timing_batches=timing_batches, timing_warmup=timing_warmup,
            access=access, access_truth=access_truth, isl=isl, isl_truth=isl_truth,
        ))

    return results
