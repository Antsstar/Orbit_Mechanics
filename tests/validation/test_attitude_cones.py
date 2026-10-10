"""
Attitude modes, body-frame impulses and visibility cones (`attitude.py`, `isl_scale.LinkSpec.cones`).

Expected, derived before measuring:
- **The body frame** is a right-handed rotation with `+z` on the boresight; a nadir boresight points at
  the reference body's centre; on a circular orbit a nadir vessel's `+x` is the along-track direction.
  So a burn along body `+x` is `(0, dv, 0)` in RSW and along body `+z` is `(-dv, 0, 0)`, exactly.
- **The cone margin** is `range * sin(half - angle)`: `range * sin(half)` on axis, zero on the edge,
  negative outside, never more than `range`.
- **Cones only remove visibility.** With a narrow Earth-pointing cone on each lunar satellite (0.5 deg,
  while the Earth constellation's orbits span ~1.1 deg seen from the Moon), every window lies inside a
  no-cone window and the in-view time falls a lot. *Measured* 4.3 % of the no-cone time at 0.5 deg,
  49.7 % at 1.0 deg (the orbits span ~1.13 deg from the Moon).
- **A cone that removes nothing still enters the interpolation.** Window edges are interpolated on the
  minimum of the margins, so a 180 deg cone (margin = range, never binding) leaves the same windows
  but can shift an edge where it is the smallest positive term just before the edge: within the edge
  interpolation error, here bounded by one sample (60 s). *Measured* 0 s: the range (~384,000 km) is
  never the smallest term near an edge here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from orbital_engine import scenarios
from orbital_engine.attitude import Attitude, Cone, attitude_matrix, body_dv_to_rsw, cone_margin_km
from orbital_engine.history import HistorySink
from orbital_engine.isl_scale import LinkSpec, Occulter, link_contact_table_from_recording

MU = scenarios.MU_EARTH


def test_the_body_frame_and_body_impulses() -> None:
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, math.sqrt(MU / 7000.0), 0.0])
    nadir = Attitude("nadir", reference="Earth")
    m = attitude_matrix(nadir, r, v, np.zeros(3), np.zeros(3))
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-14) and np.isclose(np.linalg.det(m), 1.0)
    assert np.allclose(m[:, 2], [-1.0, 0.0, 0.0])
    assert np.allclose(body_dv_to_rsw(nadir, r, v, np.array([1e-3, 0, 0]), np.zeros(3), np.zeros(3),
                                      np.zeros(3), np.zeros(3)), [0.0, 1e-3, 0.0], atol=1e-15)
    assert np.allclose(body_dv_to_rsw(nadir, r, v, np.array([0, 0, 1e-3]), np.zeros(3), np.zeros(3),
                                      np.zeros(3), np.zeros(3)), [-1e-3, 0.0, 0.0], atol=1e-15)
    with pytest.raises(ValueError, match="reference"):
        Attitude("nadir")
    with pytest.raises(ValueError, match="vector"):
        Attitude("inertial")
    with pytest.raises(ValueError, match="half-angle"):
        Cone(nadir, 0.0)


def test_the_cone_margin() -> None:
    bore = np.array([[0.0, 0.0, 1.0]] * 4)
    half = np.full(4, math.radians(10.0))
    angles = np.radians([0.0, 10.0, 20.0, 179.0])
    direction = np.stack([np.zeros(4), np.sin(angles), np.cos(angles)], axis=1)
    m = cone_margin_km(bore, direction, np.full(4, 100.0), half)
    assert np.isclose(m[0], 100.0 * math.sin(math.radians(10.0)))
    assert abs(m[1]) < 1e-12 and m[2] < 0.0 and np.isclose(m[3], -100.0)


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, list, list]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sim = scenarios.earth_moon_constellations(sessionmaker(bind=engine)())
    path = tmp_path_factory.mktemp("cones") / "run"
    with HistorySink(path, sim, chunk_snapshots=100) as sink:
        sim.attach_history_sink(sink)
        for _ in range(720):
            sim.step(60.0)
    return path, [n for n in sim.name_to_index if n.startswith("E-SAT")], \
        [n for n in sim.name_to_index if n.startswith("L-SAT")]


OCC = [Occulter("Earth", scenarios.EARTH_RADIUS, 100.0), Occulter("Moon", scenarios.MOON_RADIUS, 0.0)]


def test_a_narrow_cone_only_removes_visibility(recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, moon_sats = recording
    free = link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats))
    cone = Cone(Attitude("target", reference="Earth"), 0.5)
    narrow = link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats,
                                                              cones={n: cone for n in moon_sats}))
    outer: dict = {}
    for a, b, r, s in zip(free.body_a, free.body_b, free.rise_s, free.set_s):
        outer.setdefault((int(a), int(b)), []).append((float(r), float(s)))
    for a, b, r, s in zip(narrow.body_a, narrow.body_b, narrow.rise_s, narrow.set_s):
        assert any(r0 <= r + 1e-6 and s <= s0 + 1e-6 for r0, s0 in outer[(int(a), int(b))])
    assert 0.0 < narrow.duration_s.sum() < 0.8 * free.duration_s.sum()


def test_a_cone_that_removes_nothing_moves_edges_only_within_interpolation(
        recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, moon_sats = recording
    free = link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats))
    whole = Cone(Attitude("target", reference="Earth"), 180.0)
    wide = link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats,
                                                            cones={n: whole for n in moon_sats}))
    assert len(wide) == len(free)
    assert np.array_equal(wide.body_a, free.body_a) and np.array_equal(wide.body_b, free.body_b)
    shift = max(float(np.max(np.abs(wide.rise_s - free.rise_s))), float(np.max(np.abs(wide.set_s - free.set_s))))
    assert shift < 60.0


def test_cones_must_name_link_ends_and_recorded_references(recording: Tuple[Path, list, list]) -> None:
    path, earth_sats, moon_sats = recording
    with pytest.raises(KeyError, match="not link ends"):
        link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats,
                                                         cones={"Sun": Cone(Attitude("target", reference="Earth"), 5.0)}))
    with pytest.raises(KeyError, match="does not include"):
        link_contact_table_from_recording(path, LinkSpec(OCC, group_a=earth_sats, group_b=moon_sats,
                                                         cones={moon_sats[0]: Cone(Attitude("target", reference="Mars"), 5.0)}))
