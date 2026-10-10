"""
Streaming history (`history.HistorySink`, `Simulation.attach_history_sink`, `history.read_history`).

Expected: a streamed recording holds exactly what the in-memory history holds (global states,
bit for bit), across chunk boundaries, with decimation and subsetting applied as specified; memory is
one chunk whatever the length; a run that dies keeps every chunk it flushed; and a temporary system
forming mid-run does not disturb the recorded set.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import scenarios
from orbital_engine.history import HistorySink, read_history


def _constellation(session: Session):  # type: ignore[no-untyped-def]
    return scenarios.walker_constellation(session, total=12, planes=3, phasing=1, inclination_deg=53.0,
                                          altitude_km=550.0)


def test_streamed_equals_in_memory(db_session_factory: Callable[[], Session], tmp_path: Path) -> None:
    mem = _constellation(db_session_factory())
    streamed = _constellation(db_session_factory())
    mem._record_state()                                 # the t = 0 snapshot the sink takes on attach
    with HistorySink(tmp_path / "run", streamed, chunk_snapshots=7) as sink:
        streamed.attach_history_sink(sink)
        for _ in range(20):
            mem.step(60.0)
            streamed.step(60.0)
    times, names, states = read_history(tmp_path / "run")
    assert len(json.loads((tmp_path / "run" / "manifest.json").read_text())["chunks"]) == 3   # 21 = 7+7+7
    assert names == sink.names and times.shape == (21,) and states.shape == (21, len(names), 6)
    h = mem.history
    for j, name in enumerate(names):
        rows = h[h.body == name][["g_x", "g_y", "g_z", "g_vx", "g_vy", "g_vz"]].to_numpy()
        assert np.array_equal(rows, states[:, j])
    assert np.array_equal(times, np.arange(21) * 60.0)
    assert streamed._hist_seconds == []                 # nothing kept in memory


def test_decimation_subset_and_window(db_session: Session, tmp_path: Path) -> None:
    sim = _constellation(db_session)
    sink = HistorySink(tmp_path / "run", sim, bodies=["SAT-00-000", "SAT-02-003"], chunk_snapshots=4)
    sim.attach_history_sink(sink, every=5)
    for _ in range(50):
        sim.step(60.0)
    sink.close()
    sink.close()                                        # idempotent
    times, names, states = read_history(tmp_path / "run")
    assert names == ["SAT-00-000", "SAT-02-003"]
    assert np.array_equal(times, np.arange(0, 3001, 300.0))
    t2, n2, s2 = read_history(tmp_path / "run", bodies=["SAT-02-003"], t_range=(600.0, 1500.0))
    assert np.array_equal(t2, [600.0, 900.0, 1200.0, 1500.0]) and n2 == ["SAT-02-003"]
    assert np.array_equal(s2[:, 0], states[2:6, 1])


def test_memory_is_one_chunk_and_a_dead_run_keeps_its_chunks(db_session: Session, tmp_path: Path) -> None:
    sim = _constellation(db_session)
    sink = HistorySink(tmp_path / "run", sim, chunk_snapshots=8)
    sim.attach_history_sink(sink)
    for _ in range(30):
        sim.step(60.0)
    assert sink._g.shape[0] == 8                        # the buffer never grows
    times, _, _ = read_history(tmp_path / "run")        # not closed: the run "died" here
    assert np.array_equal(times, np.arange(24) * 60.0)  # three full chunks of 8, the partial one lost


def test_refusals(db_session_factory: Callable[[], Session], tmp_path: Path) -> None:
    sim = scenarios.sun_earth_moon(db_session_factory())
    with pytest.raises(ValueError, match="barycentres"):
        HistorySink(tmp_path / "a", sim, bodies=["Earth", "EMB"])
    with pytest.raises(KeyError, match="Mars"):
        HistorySink(tmp_path / "b", sim, bodies=["Mars"])
    HistorySink(tmp_path / "c", sim).close()
    with pytest.raises(FileExistsError):
        HistorySink(tmp_path / "c", sim)
    with pytest.raises(ValueError, match="every"):
        sim.attach_history_sink(HistorySink(tmp_path / "d", sim), every=0)


def test_a_restructure_mid_run_leaves_the_recorded_set_alone(db_session: Session, tmp_path: Path) -> None:
    sim = scenarios.sun_earth_moon(db_session)
    with HistorySink(tmp_path / "run", sim) as sink:
        sim.attach_history_sink(sink)
        sim.step(86400.0)
        sim.dissolve_system(sim.name_to_index["EMB"])
        sim.step(86400.0)
    times, names, states = read_history(tmp_path / "run")
    assert names == ["Earth", "Moon", "Sun"] and states.shape == (3, 3, 6)
