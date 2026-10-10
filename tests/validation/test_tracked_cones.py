"""
Link cones driven by a tracked attitude (`isl_scale.link_contact_table_from_recording(..., attitude=)`).

Scenario: `scenarios.earth_moon_constellations`; each lunar satellite carries a 1 deg Earth-pointing cone
(its antenna), and an `AttitudeTracker` starts it 90 deg away from that pointing, at rest, with a
torque limit, so it must slew before the antenna can see Earth. Recorded at 60 s for 12 h alongside a
`HistorySink`.

Expected, derived before measuring:
- **While slewing, no link.** The Earth constellation spans ~1.13 deg seen from the Moon, so a lunar
  satellite cannot close a link before its pointing error is below 1 deg + 1.13 deg. No tracked window
  rises before the first sample at which that holds.
- **A mis-pointed cone sees different things, not a subset.** (The first version of this test claimed a
  subset and failed: overshooting, the antenna's axis swings past Earth and briefly sees satellites the
  ideal cone does not.) What holds is that the slewing vessels are in view less in total, and once the
  slew has settled their windows are the ideal law's, same pairs and same counts, with edges shifted by
  the residual pointing error. That residual is not zero: seen from a satellite orbiting the Moon, the
  Earth direction turns at a varying rate, and rate feed-forward cancels a constant rate but not the
  target's angular acceleration, leaving ~alpha / omega_n^2. *Measured* 1e-4..2e-4 rad at k_p = 0.05,
  and 20x less at k_p = 1 (the gain ratio). Earth satellites cross the 1 deg cone at ~7.3 km/s /
  384,000 km = 1.9e-5 rad/s, so 2e-4 rad shifts an edge by ~11 s. Bound 15 s; *measured* 3.3 s at most
  (median 0) over 144 pairs after settling below 1e-3 rad at 1,020 s, and 580 h of link time against
  591 h ideal. A tracker that starts on the law and holds it at k_p = 1 reproduces the ideal windows.
- **Times must match.** A tracker that recorded on a different grid is refused, not silently misaligned.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from orbital_engine import scenarios
from orbital_engine.attitude import Attitude, Cone
from orbital_engine.attitude_dynamics import AttitudeTracker, PointingController, RigidBody, quat_mul
from orbital_engine.history import HistorySink
from orbital_engine.isl_scale import LinkSpec, Occulter, link_contact_table_from_recording

OCC = [Occulter("Earth", scenarios.EARTH_RADIUS, 100.0), Occulter("Moon", scenarios.MOON_RADIUS, 0.0)]
LAW = Attitude("target", reference="Earth")
CONE = Cone(LAW, 1.0)


def _session():  # type: ignore[no-untyped-def]
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from orbital_engine.database import Base
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, object]:
    sim = scenarios.earth_moon_constellations(_session())
    earth_sats = [n for n in sim.name_to_index if n.startswith("E-SAT")]
    moon_sats = [n for n in sim.name_to_index if n.startswith("L-SAT")]
    ideal_q = AttitudeTracker(sim, moon_sats, LAW, RigidBody((50.0,) * 3), None).q
    off = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])        # 90 deg about body x
    slewing = AttitudeTracker(sim, moon_sats, LAW, RigidBody((50.0,) * 3),
                              PointingController(kp=0.05, kd=1.5, max_torque_nm=0.002),
                              q0=quat_mul(ideal_q, np.repeat(off, len(moon_sats), axis=0)), record=True)
    holding = AttitudeTracker(sim, moon_sats, LAW, RigidBody((50.0,) * 3), PointingController(1.0, 3.0),
                              record=True)
    sim.attach_attitude_tracker(slewing)
    sim.attach_attitude_tracker(holding)
    path = tmp_path_factory.mktemp("tracked") / "run"
    errors = [slewing.pointing_error_rad().copy()]
    with HistorySink(path, sim, chunk_snapshots=100) as sink:
        sim.attach_history_sink(sink)
        for _ in range(720):
            sim.step(60.0)
            errors.append(slewing.pointing_error_rad().copy())
    return {"path": path, "earth": earth_sats, "moon": moon_sats, "slewing": slewing, "holding": holding,
            "errors": np.array(errors), "times": np.array(slewing.times)}


def _spec(r: Dict[str, object]) -> LinkSpec:
    return LinkSpec(OCC, group_a=r["earth"], group_b=r["moon"], cones={n: CONE for n in r["moon"]})  # type: ignore[arg-type]


def test_no_link_while_slewing(run: Dict[str, object]) -> None:
    table = link_contact_table_from_recording(run["path"], _spec(run), attitude=run["slewing"])  # type: ignore[arg-type]
    errors, times = run["errors"], run["times"]
    assert len(table) > 0
    limit = math.radians(1.0 + 1.2)
    for j, name in enumerate(run["moon"]):  # type: ignore[arg-type]
        ready = times[np.argmax(errors[:, j] < limit)]  # type: ignore[index]
        rises = table.rise_s[table.body_b == len(run["earth"]) + j]  # type: ignore[arg-type]
        assert ready >= 120.0                                        # the slew takes real time
        assert rises.size == 0 or float(rises.min()) >= ready - 60.0


def _after(table, t0: float) -> Dict[Tuple[int, int], list]:  # type: ignore[no-untyped-def]
    out: Dict[Tuple[int, int], list] = {}
    for a, b, r, s in zip(table.body_a, table.body_b, table.rise_s, table.set_s):
        if r >= t0 and not s >= table.set_s.max():                  # whole windows after t0, not clipped
            out.setdefault((int(a), int(b)), []).append((float(r), float(s)))
    return out


def test_after_the_slew_the_windows_are_the_laws(run: Dict[str, object]) -> None:
    ideal = link_contact_table_from_recording(run["path"], _spec(run))  # type: ignore[arg-type]
    slewing = link_contact_table_from_recording(run["path"], _spec(run), attitude=run["slewing"])  # type: ignore[arg-type]
    holding = link_contact_table_from_recording(run["path"], _spec(run), attitude=run["holding"])  # type: ignore[arg-type]
    assert slewing.duration_s.sum() < ideal.duration_s.sum()
    errors, times = run["errors"], run["times"]
    late = np.flatnonzero(np.max(errors, axis=1) > 1e-3)  # type: ignore[call-overload]
    settled = float(times[late[-1] + 1]) + 60.0  # type: ignore[index]
    assert settled < 0.5 * float(times[-1])  # type: ignore[index]
    a, b = _after(slewing, settled), _after(ideal, settled)
    assert a.keys() == b.keys() and len(a) > 10
    for pair in a:
        assert len(a[pair]) == len(b[pair])
        assert all(abs(r0 - r1) < 15.0 and abs(s0 - s1) < 15.0 for (r0, s0), (r1, s1) in zip(a[pair], b[pair]))
    assert len(holding) == len(ideal)
    assert float(np.max(np.abs(holding.rise_s - ideal.rise_s))) < 1.0


def test_mismatched_times_are_refused(run: Dict[str, object]) -> None:
    sim = scenarios.earth_moon_constellations(_session())
    late = AttitudeTracker(sim, run["moon"], LAW, RigidBody((50.0,) * 3), None, record=True)  # type: ignore[arg-type]
    sim.attach_attitude_tracker(late)
    sim.step(30.0)                                                   # a different grid
    with pytest.raises(ValueError, match="do not match"):
        link_contact_table_from_recording(run["path"], _spec(run), attitude=late)  # type: ignore[arg-type]
