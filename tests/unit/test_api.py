"""
The JSON layer (`orbital_engine.api`): a translation, so the checks are that it translates exactly and
refuses loudly.

- **No physics of its own.** A sweep or a simulation run through a document gives bit-identical
  numbers to the same run through `sweep.run_sweep` / `Simulation.step` directly.
- **The catalog is the code.** Every builder returning a `Simulation` with JSON-expressible required
  parameters is listed, and every listed builder with all-default parameters builds through the API.
  Hidden models (test fixtures, table-driven ephemeris) are not offered; engine-owned columns are not
  settable; body-valued coefficients are taken by name.
- **Strict, and all errors at once.** Unknown keys, wrong types, bad ranges and engine refusals each
  come back with their JSON path, every one in the same `ApiError`.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pytest

from orbital_engine import api, scenarios
from orbital_engine.api.catalog import scenario_builders
from orbital_engine.api.service import memory_session
from orbital_engine.custom_types import PropagatorType
from orbital_engine.sweep import ForceModelSpec, ModelConfig, run_sweep

J2, R_EQ = 1.08262668e-3, 6378.137
SCENARIO = {"builder": "earth_constellation", "params": {"n_sats": 4, "n_planes": 2}}
CONFIGS = [
    {"name": "kepler", "propagator": "KEPLERIAN", "dt": 60.0},
    {"name": "secular", "propagator": "SECULAR_J2", "dt": 60.0, "propagator_coefficients": {"j2": J2, "r_eq": R_EQ}},
    {"name": "cowell-j2", "propagator": "COWELL", "dt": 30.0, "integrator": "rk4",
     "force_models": [{"name": "point_mass_gravity"}, {"name": "j2", "coefficients": {"j2": J2, "r_eq": R_EQ}}]},
]


def _sweep_doc(**overrides: Any) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"api_version": 1, "scenario": SCENARIO, "configs": CONFIGS, "horizon_s": 1800.0,
                           "truth": {"oblateness": {"Earth": [J2, R_EQ]}}}
    doc.update(overrides)
    return doc


def _paths(exc: pytest.ExceptionInfo[api.ApiError]) -> set:  # type: ignore[type-arg]
    return {e["path"] for e in exc.value.errors}


# -- the catalog ------------------------------------------------------------------------------------
def test_the_catalog_is_json_and_lists_what_the_code_has() -> None:
    cat = json.loads(json.dumps(api.catalog()))
    names = {s["name"] for s in cat["scenarios"]}
    assert {"two_body", "earth_constellation", "walker_constellation", "earth_moon_constellations"} <= names
    models = {m["name"]: m for m in cat["force_models"]}
    assert "test_constant_accel" not in models and "ephemeris_third_body" not in models
    assert models["third_body"]["body_coefficients"] == ["perturber"]
    assert "t0" not in models["third_body"]["coefficients"] and models["third_body"]["engine_owned"] == ["t0"]
    assert cat["propagators"] == ["KEPLERIAN", "COWELL", "SECULAR_J2"]
    emc = next(s for s in cat["scenarios"] if s["name"] == "earth_moon_constellations")["params"]
    assert emc["properties"]["earth"]["minItems"] == 5 and emc["properties"]["earth"]["default"] == [24, 3, 1, 56.0, 1200.0]
    walker = next(s for s in cat["scenarios"] if s["name"] == "walker_constellation")["params"]
    assert set(walker["required"]) == {"total", "planes", "phasing", "inclination_deg", "altitude_km"}


def test_every_default_builder_builds_through_the_api() -> None:
    built = 0
    for name, builder in scenario_builders().items():
        if any(p.required for p in builder.params) or name == "planet_flyby":     # flyby: slow, needs scipy
            continue
        bodies = api.describe_scenario({"builder": name})["bodies"]
        assert bodies and all(math.isfinite(x) for b in bodies for x in b["state_km_km_s"])
        built += 1
    assert built >= 12


def test_describe_marks_default_targets() -> None:
    bodies = {b["name"]: b for b in api.describe_scenario({"builder": "two_body"})["bodies"]}
    assert bodies["Secondary"]["default_target"] and not bodies["Primary"]["default_target"]
    assert bodies["Primary"]["kind"] == "head" and bodies["Secondary"]["parent"] == "Primary"


# -- configs ----------------------------------------------------------------------------------------
def test_configs_round_trip() -> None:
    configs = [
        ModelConfig(name="a", propagator=PropagatorType.KEPLERIAN, dt=60.0),
        ModelConfig(name="b", propagator=PropagatorType.SECULAR_J2, dt=30.0, mean_seed=True,
                    propagator_coefficients={"j2": J2, "r_eq": R_EQ}, bodies=("SAT-00-000",)),
        ModelConfig(name="c", propagator=PropagatorType.COWELL, dt=10.0, integrator="yoshida4", compiled=False,
                    cowell_tolerance_km=1e-4,
                    force_models=(ForceModelSpec("point_mass_gravity"),
                                  ForceModelSpec("third_body", {"staged": 1.0}, {"perturber": "Moon"}))),
    ]
    for config in configs:
        doc = json.loads(json.dumps(api.config_to_json(config)))
        assert api.config_from_json(doc) == config


def test_config_fields_this_version_lacks_are_refused_not_dropped() -> None:
    from orbital_engine.hierarchy import EncounterPolicy, EncounterSpec
    config = ModelConfig(name="e", propagator=PropagatorType.KEPLERIAN, dt=60.0,
                         encounters=(EncounterSpec("A", "B", EncounterPolicy(10.0, 20.0)),))
    with pytest.raises(ValueError, match="encounters"):
        api.config_to_json(config)


# -- errors -----------------------------------------------------------------------------------------
def test_every_error_comes_back_with_its_path() -> None:
    doc = {"api_version": 2, "scenario": {"builder": "earth_constellation", "params": {"n_sats": "6", "bogus": 1}},
           "horizon_s": -1, "extra": True,
           "configs": [{"name": "a", "propagator": "WARP", "dt": 0},
                       {"name": "a", "propagator": "COWELL", "dt": 10,
                        "force_models": [{"name": "j2", "coefficients": {"j3": 1}},
                                         {"name": "third_body", "coefficients": {"t0": 0.0}}]}]}
    with pytest.raises(api.ApiError) as exc:
        api.validate_sweep(doc)
    assert _paths(exc) >= {"api_version", "extra", "scenario.params.bogus", "scenario.params.n_sats", "horizon_s",
                           "configs[0].propagator", "configs[0].dt", "configs",
                           "configs[1].force_models[0].coefficients.j3", "configs[1].force_models[1].coefficients.t0"}
    assert json.loads(json.dumps(exc.value.to_json()))["errors"]


def test_engine_refusals_come_back_against_the_config() -> None:
    doc = _sweep_doc(configs=[CONFIGS[0], {"name": "x", "propagator": "COWELL", "dt": 30.0, "force_models": [
        {"name": "third_body", "body_coefficients": {"perturber": "Moon"}}]},
        {"name": "y", "propagator": "SECULAR_J2", "dt": 60.0}])
    with pytest.raises(api.ApiError) as exc:
        api.validate_sweep(doc)
    assert _paths(exc) == {"configs[1]", "configs[2]"}
    assert "Moon" in exc.value.errors[0]["message"]


def test_limits_are_checked_before_running() -> None:
    with pytest.raises(api.ApiError, match="steps exceeds"):
        api.validate_sweep(_sweep_doc(horizon_s=1e9))
    with pytest.raises(api.ApiError, match="bodies exceeds"):
        api.validate_sweep(_sweep_doc(), api.Limits(max_bodies=3))
    with pytest.raises(api.ApiError, match="values"):
        api.simulate({"scenario": SCENARIO, "dt": 10.0, "horizon_s": 1000.0}, api.Limits(max_values=100))


def test_truth_that_cannot_judge_a_config_is_warned() -> None:
    doc = _sweep_doc(truth={})
    warnings = api.validate_sweep(doc)["warnings"]
    assert any("secular" in w and "oblateness" in w for w in warnings)
    assert any("cowell-j2" in w for w in warnings) and not any("kepler'" in w for w in warnings)


# -- no physics of its own --------------------------------------------------------------------------
def test_a_sweep_document_is_bit_identical_to_run_sweep() -> None:
    result = api.run_sweep_request(_sweep_doc())
    direct = run_sweep(lambda: scenarios.earth_constellation(memory_session(), n_sats=4, n_planes=2),
                       [api.config_from_json(c) for c in CONFIGS], 1800.0, timing_batches=1, timing_warmup=0,
                       oblateness={"Earth": (J2, R_EQ)})
    assert [r["config_name"] for r in result["results"]] == ["kepler", "secular", "cowell-j2"]
    for got, want in zip(result["results"], direct):
        assert got["error"] == {"median_km": want.error.median_km, "rms_km": want.error.rms_km,
                                "max_km": want.error.max_km}
        assert got["n_bodies"] == want.n_bodies
    errs = {r["config_name"]: r["error"]["max_km"] for r in result["results"]}
    assert errs["cowell-j2"] < 0.01 < errs["kepler"]                      # J2 truth judges the tiers
    json.dumps(result)


def test_a_simulate_document_is_bit_identical_to_stepping() -> None:
    config = CONFIGS[2]
    out = api.simulate({"scenario": SCENARIO, "config": config, "horizon_s": 650.0, "sample_every_s": 300.0,
                        "bodies": ["SAT-00-000", "Earth"]})
    assert out["times_s"] == [0.0, 300.0, 600.0, 660.0]                     # the last step closes the horizon
    sim = scenarios.earth_constellation(memory_session(), n_sats=4, n_planes=2)
    from orbital_engine.sweep import apply_config
    apply_config(sim, api.config_from_json(config))
    k = sim.name_to_index["SAT-00-000"]
    for _ in range(22):
        sim.step(30.0)
    assert np.array_equal(np.asarray(out["states"][-1][0]), sim.global_states[k])
    assert out["bodies"] == ["SAT-00-000", "Earth"]
    with pytest.raises(api.ApiError, match="whole multiple"):
        api.simulate({"scenario": SCENARIO, "dt": 60.0, "horizon_s": 600.0, "sample_every_s": 90.0})
    with pytest.raises(api.ApiError, match="not in the scenario"):
        api.simulate({"scenario": SCENARIO, "dt": 60.0, "horizon_s": 60.0, "bodies": ["Mars"]})
