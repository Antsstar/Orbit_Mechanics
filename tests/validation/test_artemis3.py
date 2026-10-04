"""
Artemis III rendezvous (`orbital_engine.artemis3`), a prospective scenario: each tier plans Orion's
final transfer to the lander, the plan is flown in the truth tier.

Expected magnitudes (the module docstring derives them, written before the first run):
- the two-body plan **is** Lambert's: zero shooting iterations;
- the truth's own plan flown in the truth: zero miss, to the 1 cm shooting tolerance;
- two-body planner: 0.1-10 km (measured 642 m); J2 planner: tens of metres at most (3.0 m);
  geopotential planner, which lacks only drag: 1-5 m estimated (0.42 m);
- the half-way correction for the two-body miss is about miss / (remaining time), 0.64 km / 1,316 s
  ~ 0.5 m/s (derived after the first run, measured 0.46 m/s, ratio 0.93): ratio held to 0.5-2. I then
  claimed a correction at the start would need half that; a mutation run measured 0.42 m/s, so that
  claim was wrong (relative-motion coupling, not distance over time). The burn's epoch is pinned
  instead by `test_midcourse_is_at_half_way_against_lambert`;
- RK4 at the 10 s step bound: halving it moves a planned miss by far less than a centimetre, since
  local error is ~(n h)^5 r / 120 ~ 1e-8 km per step.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital_engine import artemis3 as A


@pytest.mark.parametrize("tier, models", [
    (A.TIERS[0], {"point_mass_gravity"}),
    (A.TIERS[1], {"point_mass_gravity", "j2"}),
    (A.TIERS[2], {"point_mass_gravity", "j2", "zonal", "tesseral"}),
    (A.TIERS[3], {"point_mass_gravity", "j2", "zonal", "tesseral", "drag"}),
], ids=lambda x: getattr(x, "model_id", ""))
def test_tiers_enable_exactly_their_models(tier: A.Tier, models: set) -> None:
    sim, i, k = A.build(tier)
    assert set(sim.force_model_params) == models
    if "drag" in models:
        b = sim.force_model_params["drag"][[i, k], 0]
        np.testing.assert_allclose(b, [A.ORION_BALLISTIC, A.LANDER_BALLISTIC])


def test_two_body_plan_is_lambert() -> None:
    plan = A.plan_transfer(A.TIERS[0])
    assert plan.iterations == 0
    np.testing.assert_allclose(np.asarray(plan.burn1_rsw_m_s) * 1e-3, A.lambert_plan(), atol=1e-9)


def test_truth_plan_flown_in_truth_does_not_miss() -> None:
    out = A.fly_in_truth(A.plan_transfer(A.TRUTH))
    assert out.miss_km < 1e-5
    assert out.midcourse_m_s < 1e-3


def test_miss_ladder_matches_the_estimates() -> None:
    two_body, j2, geo = (A.fly_in_truth(A.plan_transfer(t)) for t in A.TIERS[:3])
    assert 0.1 < two_body.miss_km < 10.0
    assert j2.miss_km < 0.02
    assert geo.miss_km < 0.005
    assert two_body.miss_km > 50.0 * j2.miss_km > 50.0 * geo.miss_km
    tof, _ = A.transfer_geometry()
    ratio = two_body.midcourse_m_s / (two_body.miss_km * 1e3 / (tof / 2.0))
    assert 0.5 < ratio < 2.0


def test_midcourse_is_at_half_way_against_lambert() -> None:
    """In a two-body world the half-way correction has a closed answer: Lambert from Orion's two-body
    state at T/2 to the lander's two-body position at T, minus Orion's velocity there. Pins both the
    correction and its epoch, independently of the shooting (agreement: the 1 cm shooting tolerance
    over 1,300 s, ~1e-5 m/s, plus RK4's ~1e-6 m/s)."""
    from orbital_engine import iod, scenarios
    tof, _ = A.transfer_geometry()
    off = np.asarray(A.lambert_plan()) + np.array([0.3e-3, -0.2e-3, 0.1e-3])        # a deliberately bad first burn
    plan = A.PlannedTransfer(A.TIERS[0], tuple(off * 1e3), 0.0, 0.0, 0)
    got = A.fly_in_truth(plan, truth=A.TIERS[0])
    sim, i, k = A.build(A.TIERS[0])
    xo, xl = sim.global_states[i].copy(), sim.global_states[k].copy()
    sim.apply_delta_v([i], off)
    xo = sim.global_states[i].copy()
    r_h, v_h = iod.kepler_universal(xo[:3], xo[3:], tof / 2.0, scenarios.MU_EARTH)
    r_t, _ = iod.kepler_universal(xl[:3], xl[3:], tof, scenarios.MU_EARTH)
    v1, _ = iod.lambert(r_h, r_t, tof / 2.0, scenarios.MU_EARTH)
    rr = r_h / np.linalg.norm(r_h)
    w = np.cross(r_h, v_h) / np.linalg.norm(np.cross(r_h, v_h))
    want = np.array([(v1 - v_h) @ rr, (v1 - v_h) @ np.cross(w, rr), (v1 - v_h) @ w]) * 1e3
    np.testing.assert_allclose(got.midcourse_rsw_m_s, want, atol=1e-3)


def test_step_halving_leaves_the_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    coarse = A.fly_in_truth(A.plan_transfer(A.TIERS[1])).miss_km
    monkeypatch.setattr(A, "DT_S", A.DT_S / 2.0)
    fine = A.fly_in_truth(A.plan_transfer(A.TIERS[1])).miss_km
    assert abs(coarse - fine) < 1e-5
