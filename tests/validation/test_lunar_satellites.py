"""
Satellites of the Moon inside the Earth-Moon system (`scenarios.earth_moon_constellations`).

A regression test for a build bug found while adding cislunar visibility. A body whose parent is a
*plain member* of a barycentric bubble (a satellite of the Moon, not of the head Earth) was given the
bubble - the Earth-Moon barycentre - as its kinematic frame, while its Keplerian state is about the
Moon. One step later it had moved a lunar distance: 3,000 km from the Moon at build, 390,415 km
after 600 s. Nothing raised. The build now gives such a body its parent as its bubble, as
`hierarchy._bubble_for_parent` does at runtime.

Expected: the satellite follows its two-body conic about the Moon (the Keplerian propagator *is* that
conic) to round-off, on both kernel paths. *Measured* 6.9e-9 km after 600 s. Earth satellites, on
the head's reflex-kick path, are unaffected (their conic about Earth, to the same order).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from sqlalchemy.orm import Session

from orbital_engine import iod, scenarios


@pytest.mark.parametrize("compiled", [True, False])
def test_lunar_and_earth_satellites_follow_their_conics(
        db_session_factory: Callable[[], Session], compiled: bool) -> None:
    sim = scenarios.earth_moon_constellations(db_session_factory())
    sim.use_compiled_kernel = compiled
    sim.record_history = False
    moon, earth = sim.name_to_index["Moon"], sim.name_to_index["Earth"]
    luna, terra = sim.name_to_index["L-SAT-00-000"], sim.name_to_index["E-SAT-01-002"]
    assert sim.body_sys_map[luna] == moon and sim.parent_indices[luna] == moon
    x_l = sim.global_states[luna] - sim.global_states[moon]
    x_e = sim.global_states[terra] - sim.global_states[earth]
    assert abs(float(np.linalg.norm(x_l[:3])) - (scenarios.MOON_RADIUS + 3000.0)) < 1e-6
    for _ in range(10):
        sim.step(60.0)
    r_l, _ = iod.kepler_universal(x_l[:3], x_l[3:], 600.0, float(sim.mu_array[moon]))
    r_e, _ = iod.kepler_universal(x_e[:3], x_e[3:], 600.0, float(sim.mu_array[earth]))
    assert float(np.linalg.norm(sim.global_states[luna, :3] - sim.global_states[moon, :3] - r_l)) < 1e-6
    assert float(np.linalg.norm(sim.global_states[terra, :3] - sim.global_states[earth, :3] - r_e)) < 1e-3
