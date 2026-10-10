"""
Cislunar link visibility (`isl_scale.LinkSpec`): several occulters, link ends in two groups.

Scenario: `scenarios.earth_moon_constellations` defaults (Walker 24/3/1 at 1,200 km about Earth, 6/2/1 at
3,000 km about the Moon), recorded with a `HistorySink` at 60 s for 12 h. Occulters: Earth (100 km
grazing) and the Moon (0 km).

Expected, derived before measuring:
- **One occulter, no groups, is `IslSpec`.** Positions are taken relative to the first occulter, and its
  clearance is computed on them exactly as `isl.py` does, so the records are bit-identical.
- **Groups and two occulters against a dense reference.** Every Earth x Moon pair evaluated at every
  sample with the same margin (the minimum of both clearances and the range margin), windows cut by
  `isl._extract`: identical edges and closest approaches.
- **The Moon only removes visibility.** A link must clear every occulter, so with the Moon added each
  window lies inside an Earth-only window, and the total in-view time falls. By how much: a satellite
  at r = 4,737 km from the Moon's centre is behind it (as seen from Earth, ~384,000 km away) for at most
  2 asin(R_M / r) / 2 pi = 12 % of its orbit, when the orbit plane contains the Earth direction; 60 deg
  planes see less. Bound: between 2 % and 12 % removed. *Measured* 5.3 % (1,256 -> 1,189 pair-hours),
  and 1,050 -> 1,120 windows, as blocking splits windows in two.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import isl, scenarios
from orbital_engine.geometry import segment_clearance
from orbital_engine.history import HistorySink, read_history
from orbital_engine.isl_scale import (
    IslContactTable, LinkSpec, Occulter, isl_contact_table_from_recording, link_contact_table_from_recording,
)

EARTH = Occulter("Earth", scenarios.EARTH_RADIUS, 100.0)
MOON = Occulter("Moon", scenarios.MOON_RADIUS, 0.0)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, list, list]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sim = scenarios.earth_moon_constellations(sessionmaker(bind=engine)())
    path = tmp_path_factory.mktemp("cislunar") / "run"
    with HistorySink(path, sim, chunk_snapshots=100) as sink:
        sim.attach_history_sink(sink)
        for _ in range(720):
            sim.step(60.0)
    earth_sats = [n for n in sim.name_to_index if n.startswith("E-SAT")]
    moon_sats = [n for n in sim.name_to_index if n.startswith("L-SAT")]
    return path, earth_sats, moon_sats


def test_one_occulter_is_the_isl_spec(recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, _ = recording
    old = isl_contact_table_from_recording(path, isl.IslSpec("Earth", scenarios.EARTH_RADIUS, 100.0, 4000.0),
                                           bodies=earth_sats)
    new = link_contact_table_from_recording(path, LinkSpec([EARTH], max_range_km=4000.0), bodies=earth_sats)
    assert len(old) > 50 and new.to_contacts() == old.to_contacts()


def _dense(path: Path, a_names: list, b_names: list, spec: LinkSpec) -> isl._Edges:
    times, names, states = read_history(path)
    idx = {n: k for k, n in enumerate(names)}
    origin = states[:, idx[spec.occulters[0].body], :3]
    pos_a = states[:, [idx[n] for n in a_names], :3] - origin[:, None, :]
    pos_b = states[:, [idx[n] for n in b_names], :3] - origin[:, None, :]
    ia, jb = np.meshgrid(np.arange(len(a_names)), np.arange(len(b_names)), indexing="ij")
    r_a, r_b = pos_a[:, ia.ravel()], pos_b[:, jb.ravel()]
    margin = segment_clearance(r_a, r_b, body_radius_km=spec.occulters[0].radius_km,
                               h_graze_km=spec.occulters[0].h_graze_km)
    for o in spec.occulters[1:]:
        c = (states[:, idx[o.body], :3] - origin)[:, None, :]
        margin = np.minimum(margin, segment_clearance(r_a - c, r_b - c, body_radius_km=o.radius_km,
                                                      h_graze_km=o.h_graze_km))
    rel = r_b - r_a
    ranges = np.sqrt(np.sum(rel * rel, axis=-1))
    if spec.max_range_km is not None:
        margin = np.minimum(margin, spec.max_range_km - ranges)
    return isl._extract(times, margin, ranges)


def test_groups_and_two_occulters_match_a_dense_scan(recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, moon_sats = recording
    spec = LinkSpec([EARTH, MOON], group_a=earth_sats, group_b=moon_sats)
    table = link_contact_table_from_recording(path, spec)
    dense = _dense(path, earth_sats, moon_sats, spec)
    n_b = len(moon_sats)
    assert len(table) == dense.pair.size > 20
    order = np.lexsort((dense.rise, dense.pair))
    assert np.array_equal(table.body_a, (dense.pair // n_b)[order])
    assert np.array_equal(table.body_b, (dense.pair % n_b)[order] + len(earth_sats))
    assert np.array_equal(table.rise_s, dense.rise[order]) and np.array_equal(table.set_s, dense.set[order])
    assert table.names == earth_sats + moon_sats


def _intervals(table: IslContactTable) -> dict:
    out: dict = {}
    for a, b, r, s in zip(table.body_a.tolist(), table.body_b.tolist(), table.rise_s.tolist(), table.set_s.tolist()):
        out.setdefault((a, b), []).append((r, s))
    return out


def test_the_moon_only_removes_visibility(recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, moon_sats = recording
    earth_only = link_contact_table_from_recording(path, LinkSpec([EARTH], group_a=earth_sats, group_b=moon_sats))
    both = link_contact_table_from_recording(path, LinkSpec([EARTH, MOON], group_a=earth_sats, group_b=moon_sats))
    outer = _intervals(earth_only)
    for pair, windows in _intervals(both).items():
        for r, s in windows:
            assert any(r0 <= r + 1e-6 and s <= s0 + 1e-6 for r0, s0 in outer[pair])
    removed = 1.0 - both.duration_s.sum() / earth_only.duration_s.sum()
    assert 0.02 < removed < 0.12 and len(both) > len(earth_only)


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="at least one occulter"):
        LinkSpec([])
    with pytest.raises(ValueError, match="pairs"):
        LinkSpec([EARTH], group_a=["x"])
    with pytest.raises(ValueError, match="disjoint"):
        LinkSpec([EARTH], group_a=["x", "y"], group_b=["y"])
    with pytest.raises(ValueError, match="occulters"):
        LinkSpec([EARTH], group_a=["Earth"], group_b=["y"])
