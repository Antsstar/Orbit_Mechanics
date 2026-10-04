"""
Artemis III (2027, low Earth orbit), a **prospective** scenario: Orion's final rendezvous transfer to
Blue Origin's lander test vehicle, planned by model tiers of increasing fidelity and flown in the
highest one. The question, in decision units: how far does each planner's transfer miss the lander,
and what does that cost?

What is public (NASA's restructured Artemis III, announced February 2026; secondary sources)
---------------------------------------------------------------------------------------------
A ~2-week crewed flight in low Earth orbit, NET mid-2027: Blue Origin's lander test vehicle launches
first, Orion follows on SLS, rendezvous and docks for ~2 days, then docks with a SpaceX Starship for
~1 day. The orbit is quoted at about 230 nmi (~430 km) and 33 deg inclination (Wikipedia's infobox; the
inclination was not traced to a primary source). No phasing plan or burn timeline is public.

Assumed here (every number below is a choice, not NASA's)
---------------------------------------------------------
- Target: circular, `TARGET_ALTITUDE_KM` above the equatorial radius, `INCLINATION_DEG`.
- Orion: coplanar, circular, `CHASER_ALTITUDE_KM` (30 km below), at the start of its final transfer:
  a two-impulse transfer of `TRANSFER_ANGLE_DEG` (170 deg, Hohmann-like but clear of Lambert's 180 deg
  singularity) whose time of flight is the Hohmann half-period scaled by 170/180. The lander's lead
  angle is whatever makes that transfer arrive on it.
- Ballistic coefficients `C_d A / m`: Orion 2.2 x 30 m^2 / 26,500 kg = 2.5e-3 m^2/kg (capsule plus
  service module with arrays, a rough figure); the lander `LANDER_BALLISTIC` = 5e-3 m^2/kg (its mass is
  not public; a guess). The 28-band layered atmosphere (`atmosphere.py`), no solar-activity input.
- Earth orientation at the epoch is arbitrary (`theta0 = 0`): the epoch is not fixed.

Tiers (`TIERS`)
---------------
| id | the planner's physics |
|---|---|
| `two_body` | Earth point mass. Its plan is exactly Lambert's (`iod.lambert`) |
| `j2` | + J2 |
| `geopotential` | + J3..J6 (`zonal`) and the 4x4 tesserals |
| `full` | + drag. The truth every plan is flown in |

Each tier predicts **both** vehicles with its own physics and shoots Orion's first burn so that Orion
arrives on its prediction of the lander (`plan_transfer`). The plan is then flown in the truth
(`fly_in_truth`): the miss is truth Orion minus truth lander at the planned arrival.

Expected magnitudes (written before the first run)
--------------------------------------------------
Transfer ~46 min (2,750 s), the two vehicles 0-30 km apart radially and up to ~170 deg apart in
argument of latitude at the start.
- **Two-body plan.** J2's acceleration is ~1.1e-5 km/s^2 at 6,800 km, but most of it is common to both
  vehicles (same plane, 30 km apart in radius). What is not common: J2 shortens the period differently
  at the two radii and acts on the transfer ellipse's changing radius. A differential of ~1e-3 of J2
  (the 30 km / 6,800 km ratio, times a few) gives `0.5 a t^2` = 0.5 x 1e-8 x 2750^2 ~ 40 m; but the
  common-mode cancellation is imperfect over a 170 deg transfer, so **0.1-10 km**.
- **J2 plan.** J3..J6 and the tesserals are ~1e-3 of J2: **tens of metres at most**, likely metres.
- **Geopotential plan** (no drag). Differential drag: `0.5 rho dB v^2` with rho ~ 3e-12 kg/m^3 at
  430 km and dB = 2.5e-3 m^2/kg: 2.2e-10 km/s^2, `0.5 a t^2` ~ 1 m, along-track growth a few times that:
  **~1-5 m**.
- **Full plan** flown in itself: zero to the shooting tolerance (a control).

Measured (`tests/validation/test_artemis3.py`)
----------------------------------------------
Transfer 2,631 s, lander 0.56 deg ahead at the start; every plan totals ~17.2 m/s (Hohmann-like).

| planner | first burn | truth miss | half-way fix |
|---|---|---|---|
| two-body | 8.59 m/s (R 1.48, S 8.46, W 0) | **642 m** | 0.46 m/s |
| + J2 | 8.65 m/s (W **0.41**) | 3.0 m | 1.5 mm/s |
| + J3-J6, 4x4 | 8.65 m/s | 0.42 m | 0.2 mm/s |
| + drag (truth) | 8.65 m/s | 0 | 0 |

All within the estimates; the drag-only gap (0.42 m) came out below the 1-5 m estimated. The two-body
planner budgets the Delta-v to 0.3 % but aims 642 m off: R -130, **S +625**, W -60 m. That is mostly
timing: J2 changes how long the transfer takes relative to the lander. The J2 plan's 0.41 m/s
cross-track component removes the 60 m out of plane, and it is expensive because 170 deg from arrival
a cross-track burn moves Orion by only about (dv / n) sin 170 deg = 62 m. Correction
cost is not "miss over remaining time": a correction at the start would cost 0.42 m/s, nearly the same
as at half-way (0.46), because radial and along-track motion are coupled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from . import iod, scenarios
from .atmosphere import DENSITY_MODEL_LAYERED
from .custom_types import ArrayFloat, PropagatorType
from .database import Base
from .drag import DRAG_MODEL, EARTH_OMEGA
from .geopotential import EARTH_J2, EARTH_R_EQ, J2_MODEL
from .gravity import POINT_MASS_MODEL
from .simulator import Simulation
from .tesseral import EARTH_TESSERALS, TESSERAL_MODEL
from .zonal import EARTH_ZONALS, ZONAL_MODEL

__all__ = [
    "TARGET_ALTITUDE_KM", "CHASER_ALTITUDE_KM", "INCLINATION_DEG", "TRANSFER_ANGLE_DEG", "ORION_BALLISTIC",
    "LANDER_BALLISTIC", "DT_S", "Tier", "TIERS", "TRUTH", "transfer_geometry", "build", "fly", "lambert_plan",
    "PlannedTransfer", "plan_transfer", "TruthOutcome", "fly_in_truth", "relative_track", "run", "export_demo",
]

TARGET_ALTITUDE_KM: Final[float] = 430.0
CHASER_ALTITUDE_KM: Final[float] = 400.0
INCLINATION_DEG: Final[float] = 33.0
TRANSFER_ANGLE_DEG: Final[float] = 170.0
ORION_BALLISTIC: Final[float] = 2.5e-3      # m^2/kg, C_d A / m
LANDER_BALLISTIC: Final[float] = 5.0e-3     # m^2/kg, a guess: the test vehicle's mass is not public
DT_S: Final[float] = 10.0                   # RK4 step bound; the transfer is cut into equal steps


@dataclass(frozen=True)
class Tier:
    """One planner: the physics it predicts both vehicles with."""
    model_id: str
    label: str
    j2: bool
    geopotential: bool
    drag: bool


TIERS: Final[Tuple[Tier, ...]] = (
    Tier("two_body", "Two-body (Lambert)", False, False, False),
    Tier("j2", "+ J2", True, False, False),
    Tier("geopotential", "+ J3-J6 and 4x4 tesserals", True, True, False),
    Tier("full", "+ drag", True, True, True),
)
TRUTH: Final[Tier] = TIERS[-1]


def transfer_geometry() -> Tuple[float, float]:
    """`(time of flight s, lander lead deg)`: the Hohmann half-period scaled to `TRANSFER_ANGLE_DEG`,
    and the lead that puts the lander at the transfer's end then (two-body)."""
    r1, r2 = EARTH_R_EQ + CHASER_ALTITUDE_KM, EARTH_R_EQ + TARGET_ALTITUDE_KM
    a_tr = 0.5 * (r1 + r2)
    tof = math.pi * math.sqrt(a_tr ** 3 / scenarios.MU_EARTH) * TRANSFER_ANGLE_DEG / 180.0
    n_target = math.sqrt(scenarios.MU_EARTH / r2 ** 3)
    return tof, TRANSFER_ANGLE_DEG - math.degrees(n_target * tof)


def _session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def build(tier: Tier) -> Tuple[Simulation, int, int]:
    """`(sim, Orion slot, lander slot)` at the start of the transfer, both Cowell with `tier`'s physics."""
    _, lead = transfer_geometry()
    sim = scenarios.artemis3_rendezvous(
        _session(), chaser_radius_km=EARTH_R_EQ + CHASER_ALTITUDE_KM,
        target_radius_km=EARTH_R_EQ + TARGET_ALTITUDE_KM, inclination_deg=INCLINATION_DEG, target_lead_deg=lead)
    sim.record_history = False
    i, k = sim.name_to_index[scenarios.ORION_NAME], sim.name_to_index[scenarios.LANDER_NAME]
    both = [i, k]
    sim.set_propagator(np.array(both, dtype=np.int64), PropagatorType.COWELL)
    sim.enable_force_model(POINT_MASS_MODEL, both)
    if tier.j2:
        sim.enable_force_model(J2_MODEL, both, j2=EARTH_J2, r_eq=EARTH_R_EQ)
    if tier.geopotential:
        sim.enable_force_model(ZONAL_MODEL, both, r_eq=EARTH_R_EQ, **EARTH_ZONALS)
        sim.enable_force_model(TESSERAL_MODEL, both, r_eq=EARTH_R_EQ, omega=EARTH_OMEGA, theta0=0.0,
                               **EARTH_TESSERALS)
    if tier.drag:
        for body, b in ((i, ORION_BALLISTIC), (k, LANDER_BALLISTIC)):
            sim.enable_force_model(DRAG_MODEL, [body], ballistic_coeff=b, r_ref=EARTH_R_EQ, omega=EARTH_OMEGA,
                                   density_model=DENSITY_MODEL_LAYERED)
    return sim, i, k


def fly(tier: Tier, burn_rsw_km_s: ArrayFloat, midcourse_rsw_km_s: Optional[ArrayFloat] = None, *,
        samples: Optional[List[Tuple[float, ArrayFloat, ArrayFloat]]] = None) -> Tuple[ArrayFloat, ArrayFloat]:
    """`(Orion state, lander state)` `(6,)` each at the end of the transfer, Orion having burned
    `burn_rsw_km_s` (RSW of its own state) at the start and, if given, `midcourse_rsw_km_s` at the
    transfer's half-way time, under `tier`'s physics. `samples`, if given, receives `(t, Orion,
    lander)` after the first burn and after every step."""
    tof, _ = transfer_geometry()
    sim, i, k = build(tier)
    sim.apply_delta_v([i], np.asarray(burn_rsw_km_s, dtype=np.float64))
    n = 2 * int(math.ceil(tof / (2.0 * DT_S)))          # even, so half-way falls between steps
    if samples is not None:
        samples.append((0.0, sim.global_states[i].copy(), sim.global_states[k].copy()))
    for step in range(n):
        if step == n // 2 and midcourse_rsw_km_s is not None:
            sim.apply_delta_v([i], np.asarray(midcourse_rsw_km_s, dtype=np.float64))
        sim.step(tof / n)
        if samples is not None:
            samples.append(((step + 1) * tof / n, sim.global_states[i].copy(), sim.global_states[k].copy()))
    return sim.global_states[i].copy(), sim.global_states[k].copy()


def relative_track(tier: Tier, burn_rsw_km_s: ArrayFloat,
                   midcourse_rsw_km_s: Optional[ArrayFloat] = None) -> Tuple[ArrayFloat, ArrayFloat]:
    """`(t s (n,), Orion relative to the lander (n, 3) m)` in the lander's RSW frame (R radial out, S
    along-track, W orbit normal): the view from the lander, as rendezvous plots are drawn."""
    samples: List[Tuple[float, ArrayFloat, ArrayFloat]] = []
    fly(tier, burn_rsw_km_s, midcourse_rsw_km_s, samples=samples)
    t = np.array([s[0] for s in samples], dtype=np.float64)
    rel = np.array([_to_rsw(s[2][:3], s[2][3:], s[1][:3] - s[2][:3]) for s in samples], dtype=np.float64) * 1e3
    return t, rel


def _to_rsw(r: ArrayFloat, v: ArrayFloat, dv: ArrayFloat) -> ArrayFloat:
    rr = r / np.linalg.norm(r)
    w = np.cross(r, v)
    w = w / np.linalg.norm(w)
    out: ArrayFloat = np.array([dv @ rr, dv @ np.cross(w, rr), dv @ w])
    return out


def lambert_plan() -> ArrayFloat:
    """The two-body first burn, RSW km/s: `iod.lambert` from Orion now to the lander's two-body position
    at arrival (`iod.kepler_universal`)."""
    tof, _ = transfer_geometry()
    sim, i, k = build(TIERS[0])
    xo, xl = sim.global_states[i].copy(), sim.global_states[k].copy()
    r_arrive, _ = iod.kepler_universal(xl[:3], xl[3:], tof, scenarios.MU_EARTH)
    v1, _ = iod.lambert(xo[:3], r_arrive, tof, scenarios.MU_EARTH, prograde=float(np.cross(xo[:3], xo[3:])[2]) > 0.0)
    return _to_rsw(xo[:3], xo[3:], v1 - xo[3:])


@dataclass(frozen=True)
class PlannedTransfer:
    """A tier's plan: the first burn (RSW m/s), the matching burn at arrival in that tier's own
    prediction (m/s), and the shooting's residual (km) and iterations."""
    tier: Tier
    burn1_rsw_m_s: Tuple[float, float, float]
    burn2_m_s: float
    residual_km: float
    iterations: int

    @property
    def total_m_s(self) -> float:
        return float(np.linalg.norm(self.burn1_rsw_m_s)) + self.burn2_m_s


def _shoot(miss: Callable[[ArrayFloat], ArrayFloat], x0: ArrayFloat, step_km_s: float, tol_km: float,
           max_iter: int) -> Tuple[ArrayFloat, ArrayFloat, int]:
    """Newton on a 3-vector burn (forward-difference Jacobian), then Broyden updates."""
    x = np.asarray(x0, dtype=np.float64).copy()
    res = miss(x)
    jac = np.empty((3, 3))
    for c in range(3):
        e = np.zeros(3)
        e[c] = step_km_s
        jac[:, c] = (miss(x + e) - res) / step_km_s
    it = 0
    while float(np.linalg.norm(res)) > tol_km and it < max_iter:
        dx = np.linalg.solve(jac, -res)
        new = miss(x + dx)
        jac += np.outer(new - res - jac @ dx, dx) / float(dx @ dx)
        x, res = x + dx, new
        it += 1
    if float(np.linalg.norm(res)) > tol_km:
        raise RuntimeError(f"shooting did not converge: {np.linalg.norm(res):.3g} km")
    return x, res, it


def plan_transfer(tier: Tier, *, step_km_s: float = 1e-5, tol_km: float = 1e-5, max_iter: int = 12) -> PlannedTransfer:
    """Shoot Orion's first burn so that, **in `tier`'s own prediction**, Orion arrives on the lander,
    starting from `lambert_plan`."""
    def miss(x: ArrayFloat) -> ArrayFloat:
        xo, xl = fly(tier, x)
        out: ArrayFloat = xo[:3] - xl[:3]
        return out

    x, res, it = _shoot(miss, lambert_plan(), step_km_s, tol_km, max_iter)
    xo, xl = fly(tier, x)
    return PlannedTransfer(tier, (float(x[0]) * 1e3, float(x[1]) * 1e3, float(x[2]) * 1e3),
                           float(np.linalg.norm(xl[3:] - xo[3:])) * 1e3, float(np.linalg.norm(res)), it)


@dataclass(frozen=True)
class TruthOutcome:
    """A plan flown in the truth: the miss distance and closing speed at the planned arrival, and the
    half-way correction (RSW m/s, re-targeted in the truth) that would have removed the miss."""
    plan: PlannedTransfer
    miss_km: float
    closing_m_s: float
    midcourse_rsw_m_s: Tuple[float, float, float]

    @property
    def midcourse_m_s(self) -> float:
        return float(np.linalg.norm(self.midcourse_rsw_m_s))


def fly_in_truth(plan: PlannedTransfer, *, truth: Tier = TRUTH, tol_km: float = 1e-5) -> TruthOutcome:
    """Fly `plan`'s first burn in `truth`, measure Orion minus the lander at the planned arrival, and
    shoot the half-way correction that puts Orion on the lander in `truth`."""
    burn1 = np.asarray(plan.burn1_rsw_m_s) * 1e-3
    xo, xl = fly(truth, burn1)

    def miss(x: ArrayFloat) -> ArrayFloat:
        a, b = fly(truth, burn1, x)
        out: ArrayFloat = a[:3] - b[:3]
        return out

    mcc, _, _ = _shoot(miss, np.zeros(3), 1e-5, tol_km, 12)
    return TruthOutcome(plan, float(np.linalg.norm(xo[:3] - xl[:3])), float(np.linalg.norm(xo[3:] - xl[3:])) * 1e3,
                        (float(mcc[0]) * 1e3, float(mcc[1]) * 1e3, float(mcc[2]) * 1e3))


def run() -> List[TruthOutcome]:
    """Every tier's plan, flown in the truth."""
    return [fly_in_truth(plan_transfer(t)) for t in TIERS]


def export_demo(outcomes: Sequence[TruthOutcome], out_dir: str, *, every_s: float = 30.0) -> Dict[str, int]:
    """
    Write `approach.csv` (`t_s, planner, case, r_m, s_m, w_m`: Orion relative to the lander in its RSW
    frame, every plan flown in the truth `as_planned`, and the two-body plan `corrected` by its half-way
    burn) and `summary.json` (assumptions, sources, the per-planner table) to `out_dir`.
    """
    import csv
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    tof, lead = transfer_geometry()
    with open(os.path.join(out_dir, "approach.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["t_s", "planner", "case", "r_m", "s_m", "w_m"])
        cases: List[Tuple[str, str, TruthOutcome, Optional[ArrayFloat]]] = [
            (o.plan.tier.model_id, "as_planned", o, None) for o in outcomes]
        cases += [(o.plan.tier.model_id, "corrected", o, np.asarray(o.midcourse_rsw_m_s) * 1e-3)
                  for o in outcomes if o.plan.tier.model_id == TIERS[0].model_id]
        for mid, case, o, mcc in cases:
            t, rel = relative_track(TRUTH, np.asarray(o.plan.burn1_rsw_m_s) * 1e-3, mcc)
            stride = max(1, int(round(every_s / float(t[1] - t[0]))))
            keep = (np.arange(t.size) % stride == 0) | (np.arange(t.size) == t.size - 1)
            for tt, (r, s, ww) in zip(t[keep], rel[keep]):
                w.writerow([f"{tt:.1f}", mid, case, f"{r:.3f}", f"{s:.3f}", f"{ww:.3f}"])
    summary = {
        "title": "Artemis III: Orion's final rendezvous transfer, planned by models of rising fidelity",
        "prospective": True,
        "public_facts": ["~2-week crewed flight in low Earth orbit, NET mid-2027 (NASA, February 2026)",
                         "Blue Origin's lander test vehicle launches first; Orion docks ~2 days, then with SpaceX Starship ~1 day",
                         "Orbit ~230 nmi (~430 km), 33 deg inclination (Wikipedia infobox; inclination not traced to a primary source)"],
        "assumptions": [f"Lander circular at {TARGET_ALTITUDE_KM:g} km, {INCLINATION_DEG:g} deg; Orion coplanar, circular, "
                        f"{CHASER_ALTITUDE_KM:g} km",
                        f"Two-impulse transfer of {TRANSFER_ANGLE_DEG:g} deg, {tof:.0f} s; lander {lead:.2f} deg ahead at the start",
                        f"Ballistic coefficients C_d A/m: Orion {ORION_BALLISTIC:g}, lander {LANDER_BALLISTIC:g} m^2/kg (the lander's is a guess)",
                        "Truth: Earth point mass + J2 + J3-J6 + 4x4 tesserals (EGM96) + drag (28-band layered atmosphere); RK4 at <= 10 s"],
        "transfer_s": tof,
        "planners": [{"id": o.plan.tier.model_id, "label": o.plan.tier.label,
                      "burn1_m_s": float(np.linalg.norm(o.plan.burn1_rsw_m_s)), "burn1_rsw_m_s": list(o.plan.burn1_rsw_m_s),
                      "total_m_s": o.plan.total_m_s, "miss_m": o.miss_km * 1e3, "closing_m_s": o.closing_m_s,
                      "midcourse_m_s": o.midcourse_m_s} for o in outcomes],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, indent=2) + "\n")
    return {n: os.path.getsize(os.path.join(out_dir, n)) for n in ("approach.csv", "summary.json")}
