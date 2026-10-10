"""
Running requests: build the scenario in a private in-memory database, apply the configurations, and
answer in JSON.

**Validation runs the engine's own checks.** Parsing catches what the document alone shows; whether a
configuration fits a scenario (a Cowell body with mass, a perturber name that is not in the scenario,
a SECULAR_J2 config without coefficients) is decided by `sweep.apply_config` and the force models'
validators. `validate_sweep` builds the scenario and applies each configuration to a fresh build, so
those refusals come back as `ApiError`s against `configs[i]` before anything long runs.

**Limits.** A service exposes the engine to requests it did not write, so `Limits` caps the work one
request can ask for: configurations, steps per configuration, bodies, and values returned. They are
checked before running; the defaults suit a workstation.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..simulator import Simulation
from ..sweep import ModelConfig, apply_config, eligible_bodies, run_sweep
from .requests import (API_VERSION, ApiError, Errors, ScenarioRequest, SweepRequest, parse_scenario,
                       parse_simulate, parse_sweep)

__all__ = ["Limits", "memory_session", "build", "describe_scenario", "validate_sweep", "run_sweep_request",
           "simulate"]

JsonDict = Dict[str, Any]
_BUILD_ERRORS = (ValueError, KeyError, TypeError, ImportError, FileNotFoundError, IndexError)


@dataclass(frozen=True)
class Limits:
    max_configs: int = 32
    max_steps: int = 2_000_000          # horizon / dt, per configuration
    max_bodies: int = 20_000            # active slots in the built scenario
    max_values: int = 5_000_000         # floats in a simulate response


def memory_session() -> Any:
    """A fresh, private in-memory database (the `tests/conftest.py` recipe: StaticPool keeps the one
    connection, so the schema created here is the one every query sees)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from ..database import Base
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build(scenario: ScenarioRequest) -> Simulation:
    """A fresh simulation of the scenario; builder failures become `ApiError` at `scenario`."""
    try:
        sim = scenario.builder.function(memory_session(), **scenario.params)
    except _BUILD_ERRORS as exc:
        raise ApiError([{"path": "scenario", "message": f"{type(exc).__name__}: {exc}"}]) from exc
    sim.record_history = False
    return sim


def _check_bodies(sim: Simulation, limits: Limits) -> None:
    n = int(np.count_nonzero(sim.active_mask))
    if n > limits.max_bodies:
        raise ApiError([{"path": "scenario", "message": f"{n} bodies exceeds the limit of {limits.max_bodies}"}])


def _kind(sim: Simulation, k: int) -> str:
    if sim.is_system[k]:
        return "barycentre"
    if sim.is_head[k]:
        return "head"
    return "body"


def describe_scenario(doc: Any, limits: Limits = Limits()) -> JsonDict:
    """Build a scenario document and list its bodies: what a config's `bodies` and `body_coefficients`
    may name, and which bodies a config targets by default (`eligible`)."""
    errors = Errors()
    req = parse_scenario(doc, "scenario", errors)
    errors.raise_if_any()
    assert req is not None
    sim = build(req)
    _check_bodies(sim, limits)
    eligible = set(eligible_bodies(sim).tolist())
    names = {k: n for n, k in sim.name_to_index.items()}
    bodies = []
    for name, k in sorted(sim.name_to_index.items(), key=lambda kv: kv[1]):
        if not sim.active_mask[k]:
            continue
        parent = int(sim.parent_indices[k])
        bodies.append({"name": name, "kind": _kind(sim, k), "mu_km3_s2": float(sim.mu_array[k]),
                       "parent": None if parent == k else names.get(parent),
                       "default_target": k in eligible,
                       "state_km_km_s": [float(x) for x in sim.global_states[k]]})
    return {"api_version": API_VERSION, "scenario": req.builder.name, "bodies": bodies}


def _config_errors(req: SweepRequest, limits: Limits) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if len(req.configs) > limits.max_configs:
        out.append({"path": "configs", "message": f"{len(req.configs)} configs exceeds the limit of {limits.max_configs}"})
    for k, config in enumerate(req.configs):
        steps = req.horizon_s / config.dt
        if steps > limits.max_steps:
            out.append({"path": f"configs[{k}].dt",
                        "message": f"horizon / dt = {steps:.0f} steps exceeds the limit of {limits.max_steps}"})
    return out


def _apply(sim: Simulation, config: ModelConfig, path: str) -> Optional[Dict[str, str]]:
    try:
        apply_config(sim, config)
    except (ValueError, KeyError, TypeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return {"path": path, "message": str(message)}
    return None


def _warnings(req: SweepRequest) -> List[str]:
    """Truth that cannot judge a configuration: a tier with a field the truth omits."""
    out: List[str] = []
    for config in req.configs:
        names = {fm.name for fm in config.force_models}
        uses_j2 = "j2" in names or bool(config.propagator_coefficients)
        if uses_j2 and not req.truth.oblateness:
            out.append(f"config {config.name!r} models J2 but truth.oblateness is empty: it is judged against "
                       f"point-mass truth")
        if "zonal" in names and not req.truth.zonal:
            out.append(f"config {config.name!r} models J3..J6 but truth.zonal is empty")
        for model in sorted(names & {"tesseral", "drag", "srp", "thrust", "third_body", "system_quadrupole"}):
            if model in ("third_body", "system_quadrupole"):
                continue                                         # truth integrates every massive body
            out.append(f"config {config.name!r} models {model!r}, which this API version's truth omits; "
                       f"its error includes that model's whole effect")
    return out


def validate_sweep(doc: Any, limits: Limits = Limits()) -> JsonDict:
    """Parse a sweep request and apply each configuration to a fresh build; `ApiError` lists every
    problem. Returns the warnings a run would carry."""
    req = parse_sweep(doc)
    problems = _config_errors(req, limits)
    if problems:
        raise ApiError(problems)
    first = build(req.scenario)
    _check_bodies(first, limits)
    for k, config in enumerate(req.configs):
        err = _apply(first if k == 0 else build(req.scenario), config, f"configs[{k}]")
        if err is not None:
            problems.append(err)
    if problems:
        raise ApiError(problems)
    return {"api_version": API_VERSION, "valid": True, "warnings": _warnings(req)}


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    return value


def run_sweep_request(doc: Any, limits: Limits = Limits()) -> JsonDict:
    """Run a sweep document: every configuration against one truth (`sweep.run_sweep`)."""
    checked = validate_sweep(doc, limits)
    req = parse_sweep(doc)
    kwargs: Dict[str, Any] = {"timing_batches": req.timing_batches, "timing_warmup": req.timing_warmup}
    if req.truth.rtol is not None:
        kwargs["truth_rtol"] = req.truth.rtol
    if req.truth.atol is not None:
        kwargs["truth_atol"] = req.truth.atol
    if req.truth.oblateness:
        kwargs["oblateness"] = dict(req.truth.oblateness)
    if req.truth.zonal:
        kwargs["zonal"] = dict(req.truth.zonal)
    start = time.perf_counter()
    try:
        results = run_sweep(lambda: build(req.scenario), list(req.configs), req.horizon_s, **kwargs)
    except ImportError as exc:
        raise ApiError([{"path": "truth", "message": f"truth needs scipy (the [reference] extra): {exc}"}]) from exc
    return {"api_version": API_VERSION, "scenario": req.scenario.builder.name, "horizon_s": req.horizon_s,
            "warnings": checked["warnings"], "elapsed_s": time.perf_counter() - start,
            "results": [_json({k: v for k, v in asdict(r).items() if v is not None}) for r in results]}


def _sample_slots(sim: Simulation, names: Optional[Tuple[str, ...]]) -> Tuple[List[str], List[int]]:
    if names is None:
        pairs = sorted(((n, k) for n, k in sim.name_to_index.items()
                        if sim.active_mask[k] and not sim.is_system[k]), key=lambda nk: nk[1])
        return [n for n, _ in pairs], [k for _, k in pairs]
    missing = [n for n in names if n not in sim.name_to_index]
    if missing:
        raise ApiError([{"path": "bodies", "message": f"not in the scenario: {missing}"}])
    return list(names), [sim.name_to_index[n] for n in names]


def simulate(doc: Any, limits: Limits = Limits()) -> JsonDict:
    """Propagate one scenario (under one config, or as built) and return sampled global states:
    `states[time][body] = [x, y, z, vx, vy, vz]`, km and km/s, relative to the simulation root."""
    req = parse_simulate(doc)
    n_steps = int(np.ceil(req.horizon_s / req.dt - 1e-9))
    every = int(round(req.sample_every_s / req.dt))
    if n_steps > limits.max_steps:
        raise ApiError([{"path": "horizon_s", "message": f"{n_steps} steps exceeds the limit of {limits.max_steps}"}])
    sim = build(req.scenario)
    _check_bodies(sim, limits)
    if req.config is not None:
        err = _apply(sim, req.config, "config")
        if err is not None:
            raise ApiError([err])
    names, slots = _sample_slots(sim, req.bodies)
    n_samples = n_steps // every + 1 + (1 if n_steps % every else 0)
    if n_samples * len(slots) * 6 > limits.max_values:
        raise ApiError([{"path": "sample_every_s", "message": f"{n_samples} samples x {len(slots)} bodies exceeds "
                         f"the limit of {limits.max_values} values; sample less often or name fewer bodies"}])
    idx = np.asarray(slots, dtype=np.int64)
    times: List[float] = [float(sim.t)]
    states: List[Any] = [sim.global_states[idx].tolist()]
    start = time.perf_counter()
    for step in range(1, n_steps + 1):
        sim.step(req.dt)
        if step % every == 0 or step == n_steps:
            times.append(float(sim.t))
            states.append(sim.global_states[idx].tolist())
    return {"api_version": API_VERSION, "scenario": req.scenario.builder.name,
            "config": None if req.config is None else req.config.name, "frame": "global (simulation root)",
            "bodies": names, "times_s": times, "states": states, "elapsed_s": time.perf_counter() - start}

