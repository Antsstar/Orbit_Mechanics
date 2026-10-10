"""
ISL contacts at constellation scale (`isl_scale.py`): pruned, streamed, columnar - and bit-identical to
the dense `isl.isl_contacts`.

Expected: the pruning radius `min(max_range, L_los) + 2 v_max dt + 1 km` never drops a pair in view at
either end of an interval, and every edge, interpolated value and closest approach is computed in the
dense path's floating-point order, so the two produce **identical** records (exact equality, every
field), with and without a range limit. Measured on a Walker 53 deg 40/5/1 shell at 1,200 km, 2 h at
30 s: 816 / 384 / 186 windows at no limit / 4,000 / 2,500 km, all identical. A pruning radius that is
too small (the motion allowance dropped) is a negative control: it loses edges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import isl, isl_scale, scenarios
from orbital_engine.history import HistorySink
from orbital_engine.isl_scale import isl_contact_table, isl_contact_table_from_recording


def _shell(session: Session) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    sim = scenarios.walker_constellation(session, total=40, planes=5, phasing=1, inclination_deg=53.0,
                                         altitude_km=1200.0)
    sim.record_history = False
    earth = sim.name_to_index["Earth"]
    names = [n for n in sim.name_to_index if n.startswith("SAT-")]
    sats = [sim.name_to_index[n] for n in names]
    t, p, v = [], [], []
    for k in range(241):
        if k:
            sim.step(30.0)
        g = sim.global_states[sats] - sim.global_states[earth]
        t.append(sim.t)
        p.append(g[:, :3].copy())
        v.append(g[:, 3:].copy())
    return np.array(t), np.array(p), np.array(v), names


@pytest.mark.parametrize("max_range", [None, 4000.0, 2500.0])
def test_identical_to_the_dense_path(db_session: Session, max_range: float) -> None:
    t, p, v, names = _shell(db_session)
    spec = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=max_range)
    dense = isl.isl_contacts(p, v, t, spec)
    table = isl_contact_table(p, v, t, spec, names=names)
    assert len(dense) > 100
    assert table.to_contacts() == dense


def test_too_small_a_pruning_radius_loses_edges(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    t, p, v, names = _shell(db_session)
    spec = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=2500.0)
    dense = isl.isl_contacts(p, v, t, spec)
    real = isl_scale._Scanner._radius

    def no_motion(self, p0, v0, p1, v1, dt, *occ):  # type: ignore[no-untyped-def]
        return real(self, p0, v0 * 0.0, p1, v1 * 0.0, dt, *occ)    # drop the 2 v_max dt allowance
    monkeypatch.setattr(isl_scale._Scanner, "_radius", no_motion)
    try:
        wrong = isl_contact_table(p, v, t, spec, names=names).to_contacts()
    except RuntimeError as err:                                   # a set whose rise was pruned
        assert "never opened" in str(err)
    else:
        assert wrong != dense


def test_from_a_recording(db_session_factory, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    t, p, v, names = _shell(db_session_factory())
    sim = scenarios.walker_constellation(db_session_factory(), total=40, planes=5, phasing=1,
                                         inclination_deg=53.0, altitude_km=1200.0)
    with HistorySink(tmp_path / "run", sim, chunk_snapshots=50) as sink:
        sim.attach_history_sink(sink)
        for _ in range(240):
            sim.step(30.0)
    spec = isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, max_range_km=4000.0)
    streamed = isl_contact_table_from_recording(tmp_path / "run", spec)
    assert streamed.names == names
    assert streamed.to_contacts() == isl_contact_table(p, v, t, spec, names=names).to_contacts()
    streamed.save(tmp_path / "contacts", spec)
    assert np.array_equal(np.load(tmp_path / "contacts" / "rise_s.npy"), streamed.rise_s)
    with pytest.raises(KeyError, match="central body"):
        isl_contact_table_from_recording(tmp_path / "run", isl.IslSpec("Moon", 1737.4, 0.0))
