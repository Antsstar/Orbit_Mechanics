"""
The engine as JSON documents: a catalog of what can be run, and requests that run it.

**Why a layer of its own.** Everything the engine does is already data - a scenario is a builder and
its arguments, a model fidelity is a `sweep.ModelConfig` - but that data is Python objects. A UI, an
HTTP service and an MCP server all need the same thing: plain JSON in, plain JSON out, with every
mistake reported against the field that caused it. This package is that one translation, written
once; the HTTP and MCP adapters are thin shells over it and contain no engine logic.

**Three rules it keeps.**

- *Generated, not maintained.* The catalog is read from the code: scenario parameters from the
  builders' signatures and type hints, force-model coefficients from the registry. A builder or a
  model added to the engine appears here without an edit, and the two cannot drift.
- *Strict.* An unknown key, a wrong type or an out-of-range value is an error, never ignored. A
  silently dropped coefficient is the kind of mistake that produces a plausible wrong orbit.
- *All errors at once.* Parsing collects every problem with its JSON path (`configs[1].dt`) before
  raising `ApiError`, so a form can mark every bad field in one round trip.

Entry points: `catalog()`, `describe_scenario(doc)`, `validate_sweep(doc)`, `run_sweep_request(doc)`,
`simulate(doc)`. `API_VERSION` is the document version; a request may state it and must match.
"""
from __future__ import annotations

from .catalog import catalog, exposed_force_models, scenario_builders
from .requests import API_VERSION, ApiError, config_from_json, config_to_json
from .service import Limits, describe_scenario, run_sweep_request, simulate, validate_sweep

__all__ = ["API_VERSION", "ApiError", "Limits", "catalog", "config_from_json", "config_to_json",
           "describe_scenario", "exposed_force_models", "run_sweep_request", "scenario_builders",
           "simulate", "validate_sweep"]
