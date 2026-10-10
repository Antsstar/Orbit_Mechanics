"""
What the engine offers, read from the code: scenario builders with a JSON Schema for their arguments,
force models with their coefficients, propagators and integrators.

A builder is exposed when it returns a `Simulation` and every parameter it *requires* has a JSON
representation (number, integer, boolean, string, a nullable one of those, or a list / fixed tuple of
them). An optional parameter without one (`tle_satellites`' `tles`) is left out and keeps its default.
"""
from __future__ import annotations

import inspect
import math
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .. import scenarios
from ..custom_types import PropagatorType
from ..integrators import INTEGRATOR_NAMES
from ..registry import ForceModel, all_force_models
from ..simulator import Simulation

__all__ = ["ParamSpec", "ScenarioBuilder", "scenario_builders", "exposed_force_models", "catalog",
           "PROPAGATORS", "BODY_PARAMS", "ENGINE_OWNED", "HIDDEN_MODELS"]

JsonDict = Dict[str, Any]

#: Propagators that drive dispatch (`CLAUDE.md`: every other `PropagatorType` is unimplemented).
PROPAGATORS: Tuple[str, ...] = ("KEPLERIAN", "COWELL", "SECULAR_J2")

#: Coefficients whose value is an arena slot: given by body *name* in `body_coefficients`.
BODY_PARAMS: Mapping[str, Tuple[str, ...]] = {
    "third_body": ("perturber",), "srp": ("source",), "system_quadrupole": ("primary", "secondary"),
}
#: Columns the engine writes; a request may not set them.
ENGINE_OWNED: Mapping[str, Tuple[str, ...]] = {
    "third_body": ("t0",), "system_quadrupole": ("t0",), "srp": ("shadow_latch",),
}
#: Registered but not offered: test fixtures, and a model whose coefficients are tables, not numbers.
HIDDEN_MODELS: Tuple[str, ...] = ("test_constant_accel", "test_radial_bias", "ephemeris_third_body")


@dataclass(frozen=True)
class ParamSpec:
    """One builder parameter: its JSON Schema, whether it is required, and its default."""
    name: str
    schema: JsonDict
    required: bool
    default: Any
    hint: Any


@dataclass(frozen=True)
class ScenarioBuilder:
    name: str
    function: Callable[..., Simulation]
    summary: str
    params: Tuple[ParamSpec, ...]

    def json_schema(self) -> JsonDict:
        return {"type": "object", "additionalProperties": False,
                "properties": {p.name: p.schema for p in self.params},
                "required": [p.name for p in self.params if p.required]}


_SCALARS: Mapping[Any, JsonDict] = {float: {"type": "number"}, int: {"type": "integer"},
                                     bool: {"type": "boolean"}, str: {"type": "string"}}


def schema_for(hint: Any) -> Optional[JsonDict]:
    """The JSON Schema of a type hint, or `None` if it has no JSON representation."""
    if hint in _SCALARS:
        return dict(_SCALARS[hint])
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union:
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and len(args) == 2:
            inner = schema_for(rest[0])
            return None if inner is None else {"anyOf": [inner, {"type": "null"}]}
        return None
    if origin in (list, tuple) or (origin is not None and getattr(origin, "__name__", "") == "Sequence"):
        if origin is tuple and args and args[-1] is not Ellipsis:
            items = [schema_for(a) for a in args]
            if any(i is None for i in items):
                return None
            return {"type": "array", "prefixItems": items, "minItems": len(items), "maxItems": len(items)}
        inner = schema_for(args[0]) if args else None
        return None if inner is None else {"type": "array", "items": inner}
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(doc: Optional[str]) -> str:
    text = inspect.cleandoc(doc or "")
    return " ".join(text.split("\n\n", 1)[0].split())


def scenario_builders() -> Dict[str, ScenarioBuilder]:
    """Every exposable builder in `scenarios.py`, by name."""
    out: Dict[str, ScenarioBuilder] = {}
    for name, fn in inspect.getmembers(scenarios, inspect.isfunction):
        if fn.__module__ != scenarios.__name__ or name.startswith("_"):
            continue
        try:
            hints = typing.get_type_hints(fn)
        except Exception:                                             # pragma: no cover - unresolvable hint
            continue
        if hints.get("return") is not Simulation:
            continue
        params: List[ParamSpec] = []
        usable = True
        for pname, p in inspect.signature(fn).parameters.items():
            if pname == "session":
                continue
            required = p.default is inspect.Parameter.empty
            schema = schema_for(hints.get(pname))
            if schema is None:
                if required:
                    usable = False
                continue
            default = None if required else _json_default(p.default)
            if not required:
                schema = {**schema, "default": default}
            params.append(ParamSpec(pname, schema, required, default, hints.get(pname)))
        if usable:
            out[name] = ScenarioBuilder(name, fn, _summary(fn.__doc__), tuple(params))
    return out


def exposed_force_models() -> Dict[str, ForceModel]:
    from .. import quadrupole  # noqa: F401  (registers "system_quadrupole", which nothing else imports)
    return {m.name: m for m in all_force_models() if m.name not in HIDDEN_MODELS}


def _force_model_entry(m: ForceModel) -> JsonDict:
    owned = ENGINE_OWNED.get(m.name, ())
    bodies = BODY_PARAMS.get(m.name, ())
    return {"name": m.name, "citation": m.citation,
            "coefficients": [p for p in m.param_names if p not in owned and p not in bodies],
            "body_coefficients": list(bodies), "engine_owned": list(owned)}


def catalog() -> JsonDict:
    """Everything a request can name, as JSON."""
    from .requests import API_VERSION
    return {
        "api_version": API_VERSION,
        "scenarios": [{"name": b.name, "summary": b.summary, "params": b.json_schema()}
                      for b in scenario_builders().values()],
        "force_models": [_force_model_entry(m) for m in exposed_force_models().values()],
        "propagators": list(PROPAGATORS),
        "propagator_coefficients": {"SECULAR_J2": ["j2", "r_eq"]},
        "integrators": list(INTEGRATOR_NAMES),
        "units": {"length": "km", "velocity": "km/s", "angle": "rad (degrees where a parameter says _deg)",
                  "time": "s", "mu": "km^3/s^2"},
    }


def propagator_type(name: str) -> PropagatorType:
    return PropagatorType[name]
