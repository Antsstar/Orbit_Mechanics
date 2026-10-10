"""
Request documents -> engine objects, strictly, with every error reported against its JSON path.

Documents (all keys optional unless marked *required*; unknown keys are errors):

    scenario  = {"builder"*: str, "params": {name: value}}
    config    = {"name"*: str, "propagator"*: "KEPLERIAN" | "COWELL" | "SECULAR_J2", "dt"*: s,
                 "propagator_coefficients": {"j2", "r_eq"}, "mean_seed": bool,
                 "force_models": [{"name"*: str, "coefficients": {str: number},
                                   "body_coefficients": {str: body name}}],
                 "bodies": [body name] | null, "compiled": bool | null,
                 "integrator": "rk4" | "leapfrog" | "yoshida4" | "encke" | null,
                 "cowell_tolerance_km": number | null}
    sweep     = {"api_version": 1, "scenario"*, "configs"*: [config], "horizon_s"*: s,
                 "truth": {"rtol", "atol", "oblateness": {body: [j2, r_eq]},
                           "zonal": {body: [r_eq, {"3": j3, ...}]}},
                 "timing": {"batches": int >= 1, "warmup": int >= 0}}
    simulate  = {"api_version": 1, "scenario"*, "config": config | null, "dt": s (without a config),
                 "horizon_s"*: s, "sample_every_s": s, "bodies": [body name] | null}

The config fields are `sweep.ModelConfig`'s; the ones not listed (encounters, patches, regimes,
station keeping) are not offered in this version and are refused by `config_to_json`, rather than
dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..sweep import ForceModelSpec, ModelConfig
from .catalog import (BODY_PARAMS, ENGINE_OWNED, PROPAGATORS, ScenarioBuilder, exposed_force_models,
                      propagator_type, scenario_builders)

__all__ = ["API_VERSION", "ApiError", "Errors", "ScenarioRequest", "SweepRequest", "SimulateRequest",
           "parse_scenario", "parse_config", "parse_sweep", "parse_simulate", "config_from_json",
           "config_to_json"]

API_VERSION = 1
JsonDict = Dict[str, Any]


class ApiError(ValueError):
    """A request that cannot be run: `errors` is a list of `{"path", "message"}`, every one found."""

    def __init__(self, errors: Sequence[Mapping[str, str]]) -> None:
        self.errors: List[Dict[str, str]] = [dict(e) for e in errors]
        super().__init__("; ".join(f"{e['path'] or '<request>'}: {e['message']}" for e in self.errors))

    def to_json(self) -> JsonDict:
        return {"api_version": API_VERSION, "errors": self.errors}


class Errors:
    """Collects errors while a document is read; `raise_if_any` at the end."""

    def __init__(self) -> None:
        self.items: List[Dict[str, str]] = []

    def add(self, path: str, message: str) -> None:
        self.items.append({"path": path, "message": message})

    def raise_if_any(self) -> None:
        if self.items:
            raise ApiError(self.items)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _object(doc: Any, path: str, errors: Errors, allowed: Sequence[str], required: Sequence[str] = ()
            ) -> Optional[Mapping[str, Any]]:
    if not isinstance(doc, Mapping):
        errors.add(path, f"expected an object, got {type(doc).__name__}")
        return None
    for key in doc:
        if key not in allowed:
            errors.add(_join(path, str(key)), f"unknown key; allowed: {sorted(allowed)}")
    for key in required:
        if key not in doc:
            errors.add(_join(path, key), "required")
    return doc


def _number(value: Any, path: str, errors: Errors, *, positive: bool = False, minimum: Optional[float] = None
            ) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.add(path, f"expected a finite number, got {value!r}")
        return None
    x = float(value)
    if positive and not x > 0.0:
        errors.add(path, f"must be > 0, got {x}")
        return None
    if minimum is not None and x < minimum:
        errors.add(path, f"must be >= {minimum}, got {x}")
        return None
    return x


def _integer(value: Any, path: str, errors: Errors, minimum: int) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.add(path, f"expected an integer >= {minimum}, got {value!r}")
        return None
    return int(value)


def _string(value: Any, path: str, errors: Errors, choices: Optional[Sequence[str]] = None) -> Optional[str]:
    if not isinstance(value, str) or not value:
        errors.add(path, f"expected a non-empty string, got {value!r}")
        return None
    if choices is not None and value not in choices:
        errors.add(path, f"{value!r} is not one of {list(choices)}")
        return None
    return value


def _names(value: Any, path: str, errors: Errors) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        errors.add(path, "expected a non-empty list of body names, or null")
        return None
    if len(set(value)) != len(value):
        errors.add(path, "body names repeat")
    return tuple(value)


# -- scenario ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioRequest:
    builder: ScenarioBuilder
    params: Mapping[str, Any]


def _coerce(value: Any, schema: Mapping[str, Any], path: str, errors: Errors) -> Any:
    """`value` checked against the builder parameter's schema (`catalog.schema_for`'s subset)."""
    if "anyOf" in schema:
        return None if value is None else _coerce(value, schema["anyOf"][0], path, errors)
    kind = schema.get("type")
    if kind == "number":
        return _number(value, path, errors)
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.add(path, f"expected an integer, got {value!r}")
            return None
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            errors.add(path, f"expected true or false, got {value!r}")
        return value
    if kind == "string":
        return _string(value, path, errors)
    if kind == "array":
        if not isinstance(value, list):
            errors.add(path, f"expected a list, got {value!r}")
            return None
        if "prefixItems" in schema:
            items = schema["prefixItems"]
            if len(value) != len(items):
                errors.add(path, f"expected exactly {len(items)} items, got {len(value)}")
                return None
            return tuple(_coerce(v, s, f"{path}[{k}]", errors) for k, (v, s) in enumerate(zip(value, items)))
        return tuple(_coerce(v, schema["items"], f"{path}[{k}]", errors) for k, v in enumerate(value))
    errors.add(path, "unsupported parameter type")                   # pragma: no cover - catalog filters these
    return None


def parse_scenario(doc: Any, path: str, errors: Errors) -> Optional[ScenarioRequest]:
    obj = _object(doc, path, errors, ("builder", "params"), ("builder",))
    if obj is None or "builder" not in obj:
        return None
    builders = scenario_builders()
    name = _string(obj["builder"], _join(path, "builder"), errors, sorted(builders))
    if name is None:
        return None
    builder = builders[name]
    specs = {p.name: p for p in builder.params}
    raw = obj.get("params", {})
    pobj = _object(raw, _join(path, "params"), errors, list(specs), [p.name for p in builder.params if p.required])
    params: Dict[str, Any] = {}
    if pobj is not None:
        for key, value in pobj.items():
            if key in specs:
                params[key] = _coerce(value, specs[key].schema, _join(_join(path, "params"), key), errors)
    return ScenarioRequest(builder, params)


# -- model configuration ----------------------------------------------------------------------------
_CONFIG_KEYS = ("name", "propagator", "dt", "propagator_coefficients", "mean_seed", "force_models", "bodies",
                "compiled", "integrator", "cowell_tolerance_km")


def _force_model(doc: Any, path: str, errors: Errors) -> Optional[ForceModelSpec]:
    obj = _object(doc, path, errors, ("name", "coefficients", "body_coefficients"), ("name",))
    if obj is None or "name" not in obj:
        return None
    models = exposed_force_models()
    name = _string(obj["name"], _join(path, "name"), errors, sorted(models))
    if name is None:
        return None
    owned, body_keys = ENGINE_OWNED.get(name, ()), BODY_PARAMS.get(name, ())
    numeric = [p for p in models[name].param_names if p not in owned and p not in body_keys]
    coefficients: Dict[str, float] = {}
    cobj = _object(obj.get("coefficients", {}), _join(path, "coefficients"), errors, numeric)
    for key, value in (cobj or {}).items():
        if key in numeric:
            x = _number(value, _join(_join(path, "coefficients"), key), errors)
            if x is not None:
                coefficients[key] = x
    bodies: Dict[str, str] = {}
    bobj = _object(obj.get("body_coefficients", {}), _join(path, "body_coefficients"), errors, body_keys)
    for key, value in (bobj or {}).items():
        if key in body_keys:
            s = _string(value, _join(_join(path, "body_coefficients"), key), errors)
            if s is not None:
                bodies[key] = s
    return ForceModelSpec(name, coefficients, bodies)


def parse_config(doc: Any, path: str, errors: Errors) -> Optional[ModelConfig]:
    obj = _object(doc, path, errors, _CONFIG_KEYS, ("name", "propagator", "dt"))
    if obj is None or not all(k in obj for k in ("name", "propagator", "dt")):
        return None
    n = len(errors.items)
    name = _string(obj["name"], _join(path, "name"), errors)
    prop = _string(obj["propagator"], _join(path, "propagator"), errors, PROPAGATORS)
    dt = _number(obj["dt"], _join(path, "dt"), errors, positive=True)
    pcoeff: Dict[str, float] = {}
    pobj = _object(obj.get("propagator_coefficients", {}), _join(path, "propagator_coefficients"), errors,
                   ("j2", "r_eq"))
    for key, value in (pobj or {}).items():
        if key in ("j2", "r_eq"):
            x = _number(value, _join(_join(path, "propagator_coefficients"), key), errors)
            if x is not None:
                pcoeff[key] = x
    mean_seed = obj.get("mean_seed", False)
    if not isinstance(mean_seed, bool):
        errors.add(_join(path, "mean_seed"), "expected true or false")
    fms_doc = obj.get("force_models", [])
    fms: List[ForceModelSpec] = []
    if not isinstance(fms_doc, list):
        errors.add(_join(path, "force_models"), "expected a list")
    else:
        for k, fm_doc in enumerate(fms_doc):
            fm = _force_model(fm_doc, f"{_join(path, 'force_models')}[{k}]", errors)
            if fm is not None:
                fms.append(fm)
    bodies = _names(obj.get("bodies"), _join(path, "bodies"), errors)
    compiled = obj.get("compiled")
    if compiled is not None and not isinstance(compiled, bool):
        errors.add(_join(path, "compiled"), "expected true, false or null")
    integrator = obj.get("integrator")
    if integrator is not None:
        from ..integrators import INTEGRATOR_NAMES
        integrator = _string(integrator, _join(path, "integrator"), errors, INTEGRATOR_NAMES)
    tol = obj.get("cowell_tolerance_km")
    if tol is not None:
        tol = _number(tol, _join(path, "cowell_tolerance_km"), errors, positive=True)
    if len(errors.items) > n or name is None or prop is None or dt is None:
        return None
    return ModelConfig(name=name, propagator=propagator_type(prop), dt=dt, propagator_coefficients=pcoeff,
                       mean_seed=bool(mean_seed), force_models=tuple(fms), bodies=bodies,
                       compiled=compiled, integrator=integrator, cowell_tolerance_km=tol)


def config_from_json(doc: Any) -> ModelConfig:
    """One config document -> `ModelConfig`, or `ApiError`."""
    errors = Errors()
    config = parse_config(doc, "", errors)
    errors.raise_if_any()
    assert config is not None
    return config


def config_to_json(config: ModelConfig) -> JsonDict:
    """A `ModelConfig` -> its config document. Refuses fields this API version does not carry."""
    unsupported = [f for f in ("encounters", "patches", "regimes") if getattr(config, f)]
    if config.station_keeping is not None:
        unsupported.append("station_keeping")
    if unsupported:
        raise ValueError(f"config {config.name!r}: {unsupported} are not expressible in API version {API_VERSION}")
    doc: JsonDict = {"name": config.name, "propagator": config.propagator.name, "dt": float(config.dt)}
    if config.propagator_coefficients:
        doc["propagator_coefficients"] = {k: float(v) for k, v in config.propagator_coefficients.items()}
    if config.mean_seed:
        doc["mean_seed"] = True
    if config.force_models:
        doc["force_models"] = [
            {"name": fm.name, **({"coefficients": {k: float(v) for k, v in fm.coefficients.items()}}
                                 if fm.coefficients else {}),
             **({"body_coefficients": dict(fm.body_coefficients)} if fm.body_coefficients else {})}
            for fm in config.force_models]
    for key in ("bodies", "compiled", "integrator", "cowell_tolerance_km"):
        value = getattr(config, key)
        if value is not None:
            doc[key] = list(value) if key == "bodies" else value
    return doc


# -- requests ---------------------------------------------------------------------------------------
def _version(obj: Mapping[str, Any], errors: Errors) -> None:
    if "api_version" in obj and obj["api_version"] != API_VERSION:
        errors.add("api_version", f"this engine speaks version {API_VERSION}, got {obj['api_version']!r}")


@dataclass(frozen=True)
class Truth:
    rtol: Optional[float] = None
    atol: Optional[float] = None
    oblateness: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    zonal: Mapping[str, Tuple[float, Mapping[int, float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepRequest:
    scenario: ScenarioRequest
    configs: Tuple[ModelConfig, ...]
    horizon_s: float
    truth: Truth
    timing_batches: int
    timing_warmup: int


def _truth(doc: Any, errors: Errors) -> Truth:
    obj = _object(doc, "truth", errors, ("rtol", "atol", "oblateness", "zonal"))
    if obj is None:
        return Truth()
    rtol = None if "rtol" not in obj else _number(obj["rtol"], "truth.rtol", errors, positive=True)
    atol = None if "atol" not in obj else _number(obj["atol"], "truth.atol", errors, positive=True)
    obl: Dict[str, Tuple[float, float]] = {}
    oobj = obj.get("oblateness", {})
    if not isinstance(oobj, Mapping):
        errors.add("truth.oblateness", "expected {body: [j2, r_eq]}")
    else:
        for body, pair in oobj.items():
            p = f"truth.oblateness.{body}"
            if not isinstance(pair, list) or len(pair) != 2:
                errors.add(p, "expected [j2, r_eq]")
                continue
            j2, req = _number(pair[0], p + "[0]", errors), _number(pair[1], p + "[1]", errors, positive=True)
            if j2 is not None and req is not None:
                obl[str(body)] = (j2, req)
    zon: Dict[str, Tuple[float, Mapping[int, float]]] = {}
    zobj = obj.get("zonal", {})
    if not isinstance(zobj, Mapping):
        errors.add("truth.zonal", "expected {body: [r_eq, {degree: J}]}")
    else:
        for body, pair in zobj.items():
            p = f"truth.zonal.{body}"
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[1], Mapping):
                errors.add(p, "expected [r_eq, {\"3\": j3, ...}]")
                continue
            req = _number(pair[0], p + "[0]", errors, positive=True)
            coeffs: Dict[int, float] = {}
            for deg, value in pair[1].items():
                if str(deg) not in ("3", "4", "5", "6"):
                    errors.add(f"{p}[1].{deg}", "degree must be 3..6")
                    continue
                x = _number(value, f"{p}[1].{deg}", errors)
                if x is not None:
                    coeffs[int(deg)] = x
            if req is not None:
                zon[str(body)] = (req, coeffs)
    return Truth(rtol, atol, obl, zon)


def parse_sweep(doc: Any) -> SweepRequest:
    errors = Errors()
    obj = _object(doc, "", errors, ("api_version", "scenario", "configs", "horizon_s", "truth", "timing"),
                  ("scenario", "configs", "horizon_s"))
    if obj is None:
        errors.raise_if_any()
    assert obj is not None
    _version(obj, errors)
    scenario = parse_scenario(obj["scenario"], "scenario", errors) if "scenario" in obj else None
    configs: List[ModelConfig] = []
    raw = obj.get("configs")
    if "configs" in obj:
        if not isinstance(raw, list) or not raw:
            errors.add("configs", "expected a non-empty list of configs")
        else:
            for k, c in enumerate(raw):
                config = parse_config(c, f"configs[{k}]", errors)
                if config is not None:
                    configs.append(config)
            names: List[str] = [str(c["name"]) for c in raw if isinstance(c, Mapping) and isinstance(c.get("name"), str)]
            for dup in sorted({n for n in names if names.count(n) > 1}):
                errors.add("configs", f"config name {dup!r} repeats; names identify results")
    horizon = _number(obj["horizon_s"], "horizon_s", errors, positive=True) if "horizon_s" in obj else None
    truth = _truth(obj.get("truth", {}), errors)
    tobj = _object(obj.get("timing", {}), "timing", errors, ("batches", "warmup")) or {}
    batches = _integer(tobj.get("batches", 1), "timing.batches", errors, 1)
    warmup = _integer(tobj.get("warmup", 0), "timing.warmup", errors, 0)
    errors.raise_if_any()
    assert scenario is not None and horizon is not None and batches is not None and warmup is not None
    return SweepRequest(scenario, tuple(configs), horizon, truth, batches, warmup)


@dataclass(frozen=True)
class SimulateRequest:
    scenario: ScenarioRequest
    config: Optional[ModelConfig]
    dt: float
    horizon_s: float
    sample_every_s: float
    bodies: Optional[Tuple[str, ...]]


def parse_simulate(doc: Any) -> SimulateRequest:
    errors = Errors()
    obj = _object(doc, "", errors, ("api_version", "scenario", "config", "dt", "horizon_s", "sample_every_s",
                                    "bodies"), ("scenario", "horizon_s"))
    if obj is None:
        errors.raise_if_any()
    assert obj is not None
    _version(obj, errors)
    scenario = parse_scenario(obj["scenario"], "scenario", errors) if "scenario" in obj else None
    config = None if obj.get("config") is None else parse_config(obj["config"], "config", errors)
    dt: Optional[float] = None
    if config is not None:
        if "dt" in obj:
            errors.add("dt", "give dt in the config when there is one")
        dt = config.dt
    elif "dt" not in obj:
        errors.add("dt", "required when there is no config")
    else:
        dt = _number(obj["dt"], "dt", errors, positive=True)
    horizon = _number(obj["horizon_s"], "horizon_s", errors, positive=True) if "horizon_s" in obj else None
    every = _number(obj["sample_every_s"], "sample_every_s", errors, positive=True) if "sample_every_s" in obj else dt
    if every is not None and dt is not None:
        ratio = every / dt
        if abs(ratio - round(ratio)) > 1e-9 * max(ratio, 1.0) or round(ratio) < 1:
            errors.add("sample_every_s", f"must be a whole multiple of dt ({dt} s), got {every}")
    bodies = _names(obj.get("bodies"), "bodies", errors)
    errors.raise_if_any()
    assert scenario is not None and dt is not None and horizon is not None and every is not None
    return SimulateRequest(scenario, config, dt, horizon, every, bodies)
