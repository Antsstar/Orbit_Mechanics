"""
History recording contract.

`Simulation.history` is the engine's output surface - it is what the notebooks and any downstream
analysis consume - so its shape and column set are a contract, not an implementation detail. The
recorder was rewritten from row-wise dict appends to columnar array slices for performance; these
tests pin the observable behaviour so that rewrite, and any future one, cannot quietly change it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios

EXPECTED_COLUMNS = [
    "timestamp", "seconds", "body",
    "g_x", "g_y", "g_z", "g_vx", "g_vy", "g_vz",
    "x", "y", "z", "vx", "vy", "vz",
    "e", "theta",
]

DT = 3600.0


def test_history_columns_and_row_count(db_session_factory: Callable[[], Session]) -> None:
    """One row per body per snapshot, in long format."""
    sim = scenarios.sun_earth_moon(db_session_factory())
    n_bodies = len(sim.name_to_index)

    sim.run(10 * DT, DT)
    df = sim.history

    # run() records once at t=0 and once per step thereafter.
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 11 * n_bodies
    assert set(df["body"].unique()) == set(sim.name_to_index.keys())


def test_history_values_match_the_arena_at_the_final_step(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    The last snapshot must equal the live arena. This is what catches an off-by-one in the recording
    order, or a buffer that stores a view instead of a copy - the latter would make every historical
    row silently equal to the final state.
    """
    sim = scenarios.sun_earth_moon(db_session_factory())
    sim.run(5 * DT, DT)
    df = sim.history

    final = df[df["seconds"] == df["seconds"].max()]
    for name, slot in sim.name_to_index.items():
        row = final[final["body"] == name].iloc[0]
        assert row["g_x"] == pytest.approx(sim.global_states[slot, 0], rel=1e-15)
        assert row["g_vz"] == pytest.approx(sim.global_states[slot, 5], rel=1e-15)
        assert row["x"] == pytest.approx(sim.local_states[slot, 0], rel=1e-15)
        assert row["theta"] == pytest.approx(sim.coe_states[slot, 5], rel=1e-15)


def test_snapshots_are_independent_copies(db_session_factory: Callable[[], Session]) -> None:
    """
    Distinct snapshots must hold distinct values. A recorder that appended array *views* rather than
    copies would produce a history in which every row equals the final state - which still has the
    right shape, the right columns and the right final values, so only this test catches it.
    """
    sim = scenarios.sun_earth_moon(db_session_factory())
    sim.run(20 * DT, DT)
    df = sim.history

    moon = df[df["body"] == "Moon"].sort_values("seconds")
    assert moon["theta"].nunique() > 15, "true anomaly is not advancing between snapshots"
    assert moon["g_x"].nunique() > 15, "position is not changing between snapshots"


def test_timestamps_track_simulation_seconds(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.sun_earth_moon(db_session_factory())
    sim.run(4 * DT, DT)
    df = sim.history

    offsets = (df["timestamp"] - sim.start_epoch).dt.total_seconds()
    assert np.allclose(offsets.to_numpy(), df["seconds"].to_numpy())


def test_history_is_empty_but_well_formed_before_any_run(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    An empty history must still carry the full column set, so downstream code that selects columns
    does not fail differently depending on whether the simulation has run.
    """
    sim = scenarios.sun_earth_moon(db_session_factory())
    df = sim.history

    assert len(df) == 0
    assert list(df.columns) == EXPECTED_COLUMNS


def test_clear_history_empties_the_buffer(db_session_factory: Callable[[], Session]) -> None:
    sim = scenarios.sun_earth_moon(db_session_factory())
    sim.run(3 * DT, DT)
    assert len(sim.history) > 0

    sim.clear_history()
    assert len(sim.history) == 0

    sim.run(2 * DT, DT)
    assert len(sim.history) == 2 * len(sim.name_to_index)


def test_record_history_flag_suppresses_recording(
    db_session_factory: Callable[[], Session],
) -> None:
    """
    Disabling recording must not change the physics - only whether it is stored. A sweep that reads
    final state has no use for the buffer and should not pay for it.
    """
    recorded = scenarios.sun_earth_moon(db_session_factory())
    silent = scenarios.sun_earth_moon(db_session_factory())
    silent.record_history = False

    recorded.run(10 * DT, DT)
    silent.run(10 * DT, DT)

    assert len(silent.history) == 0
    assert len(recorded.history) > 0
    assert np.array_equal(recorded.global_states, silent.global_states)
