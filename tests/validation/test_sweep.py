"""
Validation of `sweep.py`, the model-fidelity sweep harness.

Four behaviours, matching the task this module exists for:

1. Configurations enumerate as plain data - building a list of `ModelConfig` needs nothing but the
   dataclass itself.
2. `eligible_bodies` picks out exactly the bodies a config may legally target by default.
3. A trivial one-configuration sweep matches an independently hand-run propagation and truth build -
   `run_sweep` adds no hidden transformation of its own.
4. Truth is computed once per sweep and reused across every configuration, not recomputed per config.

Physics correctness (secular-J2, Cowell, J2) is validated in each feature's own test file; this file
is about the harness wiring, not the models it composes.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import geopotential, reference, scenarios, sweep
from orbital_engine.custom_types import PropagatorType

DT = 100.0
HORIZON = 300.0  # 3 steps at DT


def _two_body_builder(db_session_factory: Callable[[], Session]) -> Callable[[], Any]:
    """Massless secondary about a fixed primary - exactly Keplerian, so a `KEPLERIAN` sweep's error
    against truth is near the integration floor rather than a modelling divergence (`CLAUDE.md`'s
    verification case)."""
    def build() -> Any:
        return scenarios.two_body(db_session_factory(), mu_secondary=0.0, p=11000.0, e=0.2)
    return build


# ==================================================================================================
# 1. Configurations enumerate
# ==================================================================================================

def test_configs_enumerate() -> None:
    configs = [
        sweep.ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=60.0),
        sweep.ModelConfig(
            name="secular-j2-osc", propagator=PropagatorType.SECULAR_J2, dt=60.0,
            propagator_coefficients={"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ},
        ),
        sweep.ModelConfig(
            name="secular-j2-mean", propagator=PropagatorType.SECULAR_J2, dt=60.0, mean_seed=True,
            propagator_coefficients={"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ},
        ),
        sweep.ModelConfig(
            name="cowell-j2", propagator=PropagatorType.COWELL, dt=10.0,
            force_models=(
                sweep.ForceModelSpec("point_mass_gravity"),
                sweep.ForceModelSpec("j2", {"j2": geopotential.EARTH_J2, "r_eq": geopotential.EARTH_R_EQ}),
            ),
        ),
    ]
    names = [c.name for c in configs]
    assert len(names) == len(set(names)) == 4
    assert configs[2].mean_seed is True
    assert configs[3].force_models[1].coefficients["j2"] == geopotential.EARTH_J2


# ==================================================================================================
# 2. `eligible_bodies`
# ==================================================================================================

def test_eligible_bodies_excludes_head_and_barycentre(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.earth_constellation(db_session_factory(), n_sats=6, n_planes=2)
    idx = sweep.eligible_bodies(sim)
    slots = set(idx.tolist())
    names = {n for n, s in sim.name_to_index.items() if s in slots}

    assert "Earth" not in names
    assert "Earth Barycenter" not in names
    assert names == {n for n in sim.name_to_index if n.startswith("SAT-")}
    assert len(names) == 6


def test_apply_config_rejects_mean_seed_outside_secular_j2(
    db_session_factory: Callable[[], Session],
) -> None:
    sim = scenarios.two_body(db_session_factory())
    config = sweep.ModelConfig(name="bad", propagator=PropagatorType.KEPLERIAN, dt=60.0, mean_seed=True)
    with pytest.raises(ValueError):
        sweep.apply_config(sim, config)


# ==================================================================================================
# 3. A trivial sweep matches a hand-run propagation
# ==================================================================================================

def test_trivial_sweep_matches_hand_run_propagation(db_session_factory: Callable[[], Session]) -> None:
    build = _two_body_builder(db_session_factory)
    config = sweep.ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=DT)

    [result] = sweep.run_sweep(build, [config], HORIZON, timing_batches=1, timing_warmup=0)

    # Independently hand-run: a fresh build, stepped by hand (no `sweep.apply_config` involved - the
    # default propagator is already Keplerian for every slot), and a fresh truth build.
    hand_sim = build()
    hand_sim.record_history = False
    secondary = hand_sim.name_to_index["Secondary"]
    n_steps = int(round(HORIZON / DT))
    for _ in range(n_steps):
        hand_sim.step(DT)

    truth = reference.reference_for(
        build(), np.array([0.0, HORIZON]), rtol=reference.TRUTH_RTOL, atol=reference.TRUTH_ATOL,
    )
    expected_err_km = float(np.linalg.norm(
        hand_sim.global_states[secondary, :3] - truth.position_of("Secondary")[-1]
    ))

    assert result.config_name == "kepler"
    assert result.n_bodies == 1
    for value in (result.error.median_km, result.error.rms_km, result.error.max_km):
        assert value == pytest.approx(expected_err_km, rel=1e-9, abs=1e-12)

    # Verification case (CLAUDE.md): massless secondary, fixed primary, exactly Keplerian - the error
    # should sit at the propagation/integration floor, not show a modelling divergence.
    assert expected_err_km < 1e-6, expected_err_km


# ==================================================================================================
# 4. Truth is computed once, reused across configurations
# ==================================================================================================

def test_truth_is_computed_once_across_multiple_configs(
    db_session_factory: Callable[[], Session], monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _two_body_builder(db_session_factory)

    calls = {"n": 0}
    original = sweep.reference_for

    def counting_reference_for(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "reference_for", counting_reference_for)

    configs = [
        sweep.ModelConfig(name="kepler-a", propagator=PropagatorType.KEPLERIAN, dt=100.0),
        sweep.ModelConfig(name="kepler-b", propagator=PropagatorType.KEPLERIAN, dt=50.0),
        sweep.ModelConfig(name="kepler-c", propagator=PropagatorType.KEPLERIAN, dt=25.0),
    ]
    results = sweep.run_sweep(build, configs, HORIZON, timing_batches=1, timing_warmup=0)

    assert calls["n"] == 1, "reference_for must be called exactly once per sweep, not once per config"
    assert len(results) == 3
    assert [r.config_name for r in results] == ["kepler-a", "kepler-b", "kepler-c"]


def test_a_horizon_that_is_not_a_whole_number_of_steps_is_refused(
    db_session_factory: Callable[[], Session],
) -> None:
    """Rounding horizon/dt silently would score the config at n*dt, not at the horizon the truth is
    sampled at, and report the gap as model error - 199 km for a 5554 s horizon at dt = 60 s."""
    build = _two_body_builder(db_session_factory)
    config = sweep.ModelConfig(name="kepler", propagator=PropagatorType.KEPLERIAN, dt=DT)
    with pytest.raises(ValueError, match="whole number of steps"):
        sweep.run_sweep(build, [config], HORIZON + 0.5 * DT, timing_batches=1, timing_warmup=0)
