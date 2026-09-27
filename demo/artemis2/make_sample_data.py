"""Generate the SYNTHETIC sample data set for the Artemis II dashboard.

This is a layout fixture, not physics output from OrbitalEngine. It exists so
the front end can be built and reviewed before the engine side produces the
real files. Everything it writes follows ``SCHEMA.md`` exactly, and
``meta.json`` carries ``"synthetic": true`` so the page shows its
SYNTHETIC SAMPLE DATA banner.

What it does, briefly:

* The Moon moves on a fixed Keplerian ellipse (e = 0.0549, i = 28.6 deg to the
  equator, the 2025-26 major lunar standstill), not an ephemeris.
* "nasa" is a free-return trajectory integrated here with RK4 under Earth
  point mass + J2 + Moon + Sun + a small solar-pressure term, targeted by
  Newton iteration so its lunar periapsis altitude and time land near the
  figures NASA reported for the real flyby (6,545 km, 6 April ~23:00 UTC).
  Its outbound and return correction burns are invented. Its
  ``closest_lunar_approach`` and ``max_earth_distance`` rows in events.csv
  carry NASA's *reported* figures, as the real file will, not the fixture's
  own minima (6,547.6 km and 407,093 km).
* The model tiers (kepler, j2, moon, moon_sun) are seeded from that state an
  hour after translunar injection and flown with fewer force terms, with the
  same correction burns applied. Their divergence is genuine for the toy
  dynamics here, and meaningless for the real mission.

Only NumPy is needed.  Run from anywhere:

    python demo/artemis2/make_sample_data.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --- constants (km, s) ------------------------------------------------------
MU_E = 398600.4418
MU_M = 4902.800066
MU_S = 1.32712440018e11
R_E = 6378.137
R_M = 1737.4
J2 = 1.08262668e-3
AU = 149597870.7
OMEGA_E = 7.2921150e-5
EI_ALT = 121.92  # entry interface, km (400,000 ft)

EPOCH = datetime(2026, 4, 1, 22, 35, 0, tzinfo=timezone.utc)  # launch
TT_MINUS_UTC = 69.184  # s, 37 leap seconds + 32.184; TDB - TT < 2 ms

# Figures NASA reported for the real flyby, used only as targets so the
# fixture looks plausible. The page reads NASA's numbers from events.csv.
TARGET_PERILUNE_ALT = 6545.0
TARGET_PERILUNE_UTC = datetime(2026, 4, 6, 23, 0, 0, tzinfo=timezone.utc)
TARGET_RETURN_PERIGEE_ALT = 40.0  # vacuum perigee for a shallow entry

PARKING_R = R_E + 185.0


def t_of(dt: datetime) -> float:
    return (dt - EPOCH).total_seconds()


# --- Moon and Sun (fixture ephemerides) -------------------------------------
MOON_A = 384748.0
MOON_E = 0.0549
MOON_I = math.radians(28.58)
MOON_RAAN = math.radians(0.0)
MOON_ARGP = math.radians(318.0)
MOON_N = math.sqrt((MU_E + MU_M) / MOON_A**3)
# Chosen so the Moon is ~398,000 km out (past mean distance) at the flyby.
MOON_NU_AT_FLYBY = math.radians(134.0)


def _kepler_E(M: float, e: float) -> float:
    E = M if e < 0.8 else math.pi
    for _ in range(30):
        d = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-14:
            break
    return E


def _perifocal_to_eci(i: float, raan: float, argp: float) -> np.ndarray:
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )


_MOON_Q = _perifocal_to_eci(MOON_I, MOON_RAAN, MOON_ARGP)
_E_fb = 2 * math.atan(math.sqrt((1 - MOON_E) / (1 + MOON_E)) * math.tan(MOON_NU_AT_FLYBY / 2))
_M_fb = _E_fb - MOON_E * math.sin(_E_fb)
MOON_M0 = _M_fb - MOON_N * t_of(TARGET_PERILUNE_UTC)


def moon_pos(t: float) -> tuple[float, float, float]:
    M = MOON_M0 + MOON_N * t
    E = _kepler_E(math.fmod(M, 2 * math.pi), MOON_E)
    x = MOON_A * (math.cos(E) - MOON_E)
    y = MOON_A * math.sqrt(1 - MOON_E**2) * math.sin(E)
    q = _MOON_Q
    return (q[0, 0] * x + q[0, 1] * y, q[1, 0] * x + q[1, 1] * y, q[2, 0] * x + q[2, 1] * y)


OBLIQUITY = math.radians(23.4393)
SUN_LON0 = math.radians(11.9)  # apparent ecliptic longitude on 1 April 2026, approx.
SUN_RATE = 2 * math.pi / (365.2422 * 86400.0)


def sun_pos(t: float) -> tuple[float, float, float]:
    lam = SUN_LON0 + SUN_RATE * t
    r = 1.0 * AU
    return (r * math.cos(lam), r * math.sin(lam) * math.cos(OBLIQUITY), r * math.sin(lam) * math.sin(OBLIQUITY))


# --- dynamics ---------------------------------------------------------------
@dataclass(frozen=True)
class ForceModel:
    j2: bool = False
    moon: bool = False
    sun: bool = False
    srp: float = 0.0  # acceleration at 1 AU, km/s^2 (anti-sunward)


def accel(t: float, x: float, y: float, z: float, fm: ForceModel) -> tuple[float, float, float]:
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    k = -MU_E / (r2 * r)
    ax, ay, az = k * x, k * y, k * z
    if fm.j2:
        f = 1.5 * J2 * MU_E * R_E * R_E / (r2 * r2 * r)
        zz = 5.0 * z * z / r2
        ax += f * x * (zz - 1.0)
        ay += f * y * (zz - 1.0)
        az += f * z * (zz - 3.0)
    if fm.moon:
        mx, my, mz = moon_pos(t)
        dx, dy, dz = mx - x, my - y, mz - z
        d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
        m3 = (mx * mx + my * my + mz * mz) ** 1.5
        ax += MU_M * (dx / d3 - mx / m3)
        ay += MU_M * (dy / d3 - my / m3)
        az += MU_M * (dz / d3 - mz / m3)
    if fm.sun or fm.srp:
        sx, sy, sz = sun_pos(t)
        dx, dy, dz = sx - x, sy - y, sz - z
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        if fm.sun:
            d3 = dn**3
            s3 = (sx * sx + sy * sy + sz * sz) ** 1.5
            ax += MU_S * (dx / d3 - sx / s3)
            ay += MU_S * (dy / d3 - sy / s3)
            az += MU_S * (dz / d3 - sz / s3)
        if fm.srp:
            a = fm.srp * (AU / dn) ** 2 / dn
            ax -= a * dx
            ay -= a * dy
            az -= a * dz
    return ax, ay, az


def rk4_step(t: float, s: list[float], h: float, fm: ForceModel) -> list[float]:
    x, y, z, vx, vy, vz = s
    a1 = accel(t, x, y, z, fm)
    h2 = 0.5 * h
    x2, y2, z2 = x + h2 * vx, y + h2 * vy, z + h2 * vz
    v2 = (vx + h2 * a1[0], vy + h2 * a1[1], vz + h2 * a1[2])
    a2 = accel(t + h2, x2, y2, z2, fm)
    x3, y3, z3 = x + h2 * v2[0], y + h2 * v2[1], z + h2 * v2[2]
    v3 = (vx + h2 * a2[0], vy + h2 * a2[1], vz + h2 * a2[2])
    a3 = accel(t + h2, x3, y3, z3, fm)
    x4, y4, z4 = x + h * v3[0], y + h * v3[1], z + h * v3[2]
    v4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
    a4 = accel(t + h, x4, y4, z4, fm)
    h6 = h / 6.0
    return [
        x + h6 * (vx + 2 * v2[0] + 2 * v3[0] + v4[0]),
        y + h6 * (vy + 2 * v2[1] + 2 * v3[1] + v4[1]),
        z + h6 * (vz + 2 * v2[2] + 2 * v3[2] + v4[2]),
        vx + h6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]),
        vy + h6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]),
        vz + h6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2]),
    ]


@dataclass
class Burn:
    t: float
    name: str
    dv_kms: tuple[float, float, float] = (0.0, 0.0, 0.0)  # inertial
    rsw: tuple[float, float, float] | None = None  # (radial, along, cross) m/s, resolved at burn time


def _rsw_to_inertial(s: list[float], rsw_ms: tuple[float, float, float]) -> tuple[float, float, float]:
    r = np.array(s[:3])
    v = np.array(s[3:])
    R = r / np.linalg.norm(r)
    W = np.cross(r, v)
    W /= np.linalg.norm(W)
    S = np.cross(W, R)
    d = (rsw_ms[0] * R + rsw_ms[1] * S + rsw_ms[2] * W) * 1e-3
    return (float(d[0]), float(d[1]), float(d[2]))


def propagate(
    t0: float,
    s0: list[float],
    t1: float,
    fm: ForceModel,
    h: float = 30.0,
    burns: list[Burn] | None = None,
    stop_below: float | None = None,
    stop_after: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 with impulsive burns landed on exact epochs.

    ``stop_below``: stop at the first sample with r below this radius after
    ``stop_after`` (entry interface).
    """
    burns = sorted(burns or [], key=lambda b: b.t)
    ts = [t0]
    ss = [list(s0)]
    t, s = t0, list(s0)
    bi = 0
    while t < t1 - 1e-9:
        step = min(h, t1 - t)
        if bi < len(burns) and t + step > burns[bi].t - 1e-9:
            step = burns[bi].t - t
        if step > 1e-9:
            s = rk4_step(t, s, step, fm)
            t += step
            ts.append(t)
            ss.append(list(s))
        if bi < len(burns) and abs(t - burns[bi].t) < 1e-6:
            b = burns[bi]
            d = _rsw_to_inertial(s, b.rsw) if b.rsw is not None else b.dv_kms
            s = [s[0], s[1], s[2], s[3] + d[0], s[4] + d[1], s[5] + d[2]]
            ss[-1] = list(s)
            bi += 1
        if stop_below is not None and t > stop_after:
            if math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) < stop_below:
                break
    return np.array(ts), np.array(ss)


def _refine_min(ts: np.ndarray, d: np.ndarray, k: int) -> tuple[float, float]:
    """Quadratic refinement of a sampled minimum at index k."""
    if k <= 0 or k >= len(d) - 1:
        return float(ts[k]), float(d[k])
    y0, y1, y2 = d[k - 1], d[k], d[k + 1]
    h = ts[k + 1] - ts[k]
    den = y0 - 2 * y1 + y2
    if den <= 0:
        return float(ts[k]), float(y1)
    u = 0.5 * (y0 - y2) / den
    return float(ts[k] + u * h), float(y1 - 0.25 * (y0 - y2) * u)


def moon_positions(ts: np.ndarray) -> np.ndarray:
    return np.array([moon_pos(float(t)) for t in ts])


def perilune(ts: np.ndarray, ss: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(ss[:, :3] - moon_positions(ts), axis=1)
    return _refine_min(ts, d, int(np.argmin(d)))


def return_perigee(ts: np.ndarray, ss: np.ndarray, after: float) -> tuple[float, float]:
    r = np.linalg.norm(ss[:, :3], axis=1)
    m = ts > after
    idx = np.nonzero(m)[0]
    k = idx[int(np.argmin(r[m]))]
    if k == idx[0] or k == len(r) - 1:
        return math.nan, math.nan  # no local minimum inside the window
    return _refine_min(ts, r, int(k))


# --- the synthetic "nasa" trajectory ------------------------------------------
TRUTH = ForceModel(j2=True, moon=True, sun=True, srp=1.5e-11)


def make_burns(t_tli: float) -> list[Burn]:
    """Invented correction burns (RSW, m/s), timed relative to TLI."""
    return [
        Burn(t_tli + 20 * 3600, "outbound_correction_1", rsw=(0.6, 1.1, -0.3)),
        Burn(t_tli + 60 * 3600, "outbound_correction_2", rsw=(-0.2, 0.4, 0.15)),
        Burn(t_tli + 140 * 3600, "return_correction_1", rsw=(0.3, -0.5, 0.2)),
    ]


def tli_state(t_tli: float, phi: float, v: float) -> list[float]:
    """Perigee state in the Moon's orbital plane at in-plane angle ``phi``."""
    q = _MOON_Q
    # in-plane unit vectors (perifocal frame of the lunar orbit)
    p_hat = q[:, 0]
    q_hat = q[:, 1]
    r_hat = math.cos(phi) * p_hat + math.sin(phi) * q_hat
    t_hat = -math.sin(phi) * p_hat + math.cos(phi) * q_hat
    r = PARKING_R * r_hat
    vv = v * t_hat
    return [*map(float, r), *map(float, vv)]


def fly_translunar(t_tli: float, phi: float, v: float, h: float = 60.0, far_side: bool = False):  # type: ignore[no-untyped-def]
    s0 = tli_state(t_tli, phi, v)
    ts, ss = propagate(t_tli, s0, t_tli + 9.5 * 86400, TRUTH, h=h, burns=make_burns(t_tli))
    t_pl, d_pl = perilune(ts, ss)
    t_rp, r_rp = return_perigee(ts, ss, t_pl)
    out = (t_pl, d_pl - R_M, t_rp, r_rp - R_E)
    if not far_side:
        return out
    k = int(np.argmin(np.abs(ts - t_pl)))
    mp = np.array(moon_pos(float(ts[k])))
    return (*out, bool(np.dot(ss[k, :3] - mp, mp) > 0))


def target_free_return() -> tuple[float, float, float]:
    t_fb = t_of(TARGET_PERILUNE_UTC)
    t_tli = t_fb - 4.0 * 86400
    # Moon's in-plane angle at the flyby; TLI roughly opposite it.
    m = np.array(moon_pos(t_fb))
    ang_moon = math.atan2(float(m @ _MOON_Q[:, 1]), float(m @ _MOON_Q[:, 0]))
    # Coarse scan for a far-side pass that comes back to Earth; Newton from there.
    best = None
    for v in np.arange(10.935, 10.9501, 0.005):
        for dphi in np.arange(-4.0, 25.0, 2.0):
            phi = ang_moon + math.pi + math.radians(dphi)
            t_pl, h_pl, t_rp, h_rp, far = fly_translunar(t_tli, phi, float(v), h=120.0, far_side=True)
            if not far or not math.isfinite(h_rp):
                continue
            score = abs(h_pl - TARGET_PERILUNE_ALT) / 1000 + abs(h_rp - TARGET_RETURN_PERIGEE_ALT) / 1000
            if best is None or score < best[0]:
                best = (score, phi, float(v))
    assert best is not None, "no far-side free return in the scan window"
    _, phi, v = best
    for outer in range(4):
        for _ in range(8):
            t_pl, h_pl, t_rp, h_rp = fly_translunar(t_tli, phi, v)
            f = np.array([h_pl - TARGET_PERILUNE_ALT, h_rp - TARGET_RETURN_PERIGEE_ALT])
            if np.all(np.abs(f) < [5.0, 1.0]):
                break
            J = np.zeros((2, 2))
            for j, (dphi, dv) in enumerate([(1e-5, 0.0), (0.0, 1e-5)]):
                _, h1, _, r1 = fly_translunar(t_tli, phi + dphi, v + dv)
                J[:, j] = (np.array([h1 - TARGET_PERILUNE_ALT, r1 - TARGET_RETURN_PERIGEE_ALT]) - f) / 1e-5
            step = np.linalg.solve(J, -f)
            scale = min(1.0, 0.02 / max(abs(step[0]), 1e-12), 0.02 / max(abs(step[1]), 1e-12))
            phi += step[0] * scale
            v += step[1] * scale
        shift = t_fb - t_pl
        print(f"  outer {outer}: perilune {h_pl:9.1f} km  return perigee {h_rp:7.1f} km  timing {shift:+8.0f} s")
        if abs(shift) < 60:
            break
        # Shift the whole transfer in time; rotate phi with the Moon's mean motion.
        t_tli += shift
        phi += MOON_N * shift
    return t_tli, phi, v


# --- products -----------------------------------------------------------------
STATIONS = {
    "Goldstone": (35.4267, -116.8900),
    "Madrid": (40.4314, -4.2481),
    "Canberra": (-35.4014, 148.9817),
}
DSN_MASK_DEG = 10.0


def gmst(t: float) -> float:
    d = EPOCH + timedelta(seconds=t)
    jd = d.timestamp() / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T
    return math.radians(g % 360.0)


def station_eci(name: str, t: np.ndarray) -> np.ndarray:
    lat, lon = map(math.radians, STATIONS[name])
    th = np.array([gmst(float(x)) for x in t]) + lon
    c = math.cos(lat)
    return np.stack([R_E * c * np.cos(th), R_E * c * np.sin(th), np.full_like(th, R_E * math.sin(lat))], axis=1)


def windows_from_mask(t: np.ndarray, m: np.ndarray) -> list[tuple[float, float]]:
    out = []
    start = None
    for i, on in enumerate(m):
        if on and start is None:
            start = t[i]
        if not on and start is not None:
            out.append((float(start), float(t[i - 1])))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1])))
    return out


@dataclass
class Run:
    model_id: str
    ts: np.ndarray
    ss: np.ndarray
    events: list[dict] = field(default_factory=list)


def resample(ts: np.ndarray, ss: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(grid, ts, ss[:, k]) for k in range(ss.shape[1])], axis=1)


def adaptive_grid(ts: np.ndarray, ss: np.ndarray, fine: float, coarse: float) -> np.ndarray:
    """Every ``fine`` s near Earth or the Moon, every ``coarse`` s elsewhere."""
    grid = [ts[0]]
    t = ts[0]
    end = ts[-1]
    while t < end:
        s = resample(ts, ss, np.array([t]))[0]
        r = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        m = moon_pos(t)
        dm = math.dist(s[:3], m)
        step = fine if (r < 60000.0 or dm < 40000.0) else coarse
        t = min(end, (math.floor(t / step) + 1) * step)
        grid.append(t)
    return np.array(grid)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("Targeting the synthetic free return ...")
    t_tli, phi, v_tli = target_free_return()

    # Pre-TLI: 185 km parking orbit, apogee raise at its perigee, one HEO revolution.
    s_tli_post = tli_state(t_tli, phi, v_tli)
    t_arb = 6480.0  # T+1h48m
    heo_T = t_tli - t_arb
    heo_a = (MU_E * (heo_T / (2 * math.pi)) ** 2) ** (1 / 3)
    v_heo_p = math.sqrt(MU_E * (2 / PARKING_R - 1 / heo_a))
    v_circ = math.sqrt(MU_E / PARKING_R)
    r_hat = np.array(s_tli_post[:3]) / PARKING_R
    t_hat = np.array(s_tli_post[3:]) / v_tli
    print(f"  HEO apogee {2 * heo_a - PARKING_R - R_E:,.0f} km altitude, period {heo_T / 3600:.2f} h")
    print(f"  TLI dv {1000 * (v_tli - v_heo_p):.1f} m/s at T+{t_tli / 3600:.2f} h")

    t_ins = 600.0
    # parking orbit, two-body, ends exactly at the ARB point
    pk_t = np.arange(t_ins, t_arb, 60.0)
    pk_t = np.append(pk_t, t_arb)
    ang = (pk_t - t_arb) * v_circ / PARKING_R
    pk = np.stack(
        [
            *(PARKING_R * (np.cos(ang)[:, None] * r_hat + np.sin(ang)[:, None] * t_hat)).T,
            *(v_circ * (-np.sin(ang)[:, None] * r_hat + np.cos(ang)[:, None] * t_hat)).T,
        ],
        axis=1,
    )
    s_arb = [*map(float, PARKING_R * r_hat), *map(float, v_heo_p * t_hat)]
    heo_ts, heo_ss = propagate(t_arb, s_arb, t_tli, ForceModel(), h=30.0)

    burns = make_burns(t_tli)
    t_end_max = t_tli + 9.5 * 86400
    tr_ts, tr_ss = propagate(
        t_tli, s_tli_post, t_end_max, TRUTH, h=30.0, burns=burns, stop_below=R_E + EI_ALT, stop_after=t_tli + 86400
    )
    t_ei = float(tr_ts[-1])
    print(f"  truth reaches entry interface at T+{t_ei / 86400:.2f} d")

    truth_t = np.concatenate([pk_t, heo_ts[1:], tr_ts[1:]])
    truth_s = np.concatenate([pk, heo_ss[1:], tr_ss[1:]])

    t_seed = t_tli + 3600.0
    s_seed = resample(truth_t, truth_s, np.array([t_seed]))[0].tolist()
    t_final = t_ei

    tiers = [
        ("kepler", ForceModel()),
        ("j2", ForceModel(j2=True)),
        ("moon", ForceModel(j2=True, moon=True)),
        ("moon_sun", ForceModel(j2=True, moon=True, sun=True)),
    ]
    runs = [Run("nasa", truth_t, truth_s)]
    for mid, fm in tiers:
        # Six hours past NASA's entry interface, so a model that misses Earth still shows its return perigee.
        ts, ss = propagate(
            t_seed, s_seed, t_final + 6 * 3600, fm, h=30.0, burns=[b for b in burns if b.t > t_seed],
            stop_below=R_E + EI_ALT, stop_after=t_seed + 86400,
        )
        runs.append(Run(mid, ts, ss))
        print(f"  {mid:9s} flown")

    write_all(runs, burns, t_tli, t_arb, v_tli - v_heo_p, v_heo_p - v_circ, t_ins, t_ei, t_seed)


def fmt(x: float, nd: int) -> str:
    s = f"{x:.{nd}f}"
    return "0" if s in ("-0", "-0." + "0" * nd) else s


def write_all(runs, burns, t_tli, t_arb, dv_tli, dv_arb, t_ins, t_ei, t_seed) -> None:
    truth = runs[0]

    # models.csv
    models = [
        ("nasa", "NASA navigation", "Orion's reconstructed trajectory as reported by NASA. The reference every model is scored against.", "#17212e", "#f2f5f8", 1),
        ("kepler", "Earth only", "Two-body Kepler: Earth as a point mass, nothing else. Blind to the Moon.", "#2a78d6", "#3987e5", 0),
        ("j2", "Earth + oblateness", "Earth point mass plus J2, the equatorial bulge. Still blind to the Moon.", "#eb6834", "#d95926", 0),
        ("moon", "Earth + Moon", "Adds the Moon's gravity. The first tier that can fly a lunar flyby.", "#1baf7a", "#199e70", 0),
        ("moon_sun", "Earth + Moon + Sun", "Adds the Sun's tidal pull on the Earth-Moon-Orion system.", "#eda100", "#c98500", 0),
    ]
    with open(DATA / "models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "label", "description", "colour", "colour_dark", "is_truth"])
        w.writerows(models)

    # trajectory.csv
    n_rows = 0
    with open(DATA / "trajectory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "model_id", "body", "x_km", "y_km", "z_km"])
        for run in runs:
            grid = adaptive_grid(run.ts, run.ss, 60.0, 300.0)
            p = resample(run.ts, run.ss, grid)
            for t, s in zip(grid, p):
                w.writerow([fmt(t, 1), run.model_id, "orion", fmt(s[0], 3), fmt(s[1], 3), fmt(s[2], 3)])
                n_rows += 1
        mgrid = np.arange(0.0, t_ei + 600.0, 600.0)
        for t in mgrid:
            m = moon_pos(float(t))
            w.writerow([fmt(t, 1), "nasa", "moon", fmt(m[0], 3), fmt(m[1], 3), fmt(m[2], 3)])
            n_rows += 1
    print(f"  trajectory.csv: {n_rows} rows")

    # metrics.csv on each run's adaptive grid (120 s near bodies, 600 s elsewhere)
    truth_interp = lambda g: resample(truth.ts, truth.ss, g)  # noqa: E731
    burn_times = [(t_arb, dv_arb * 1000), (t_tli, dv_tli * 1000)] + [
        (b.t, float(np.linalg.norm(b.rsw))) for b in burns
    ]
    n_rows = 0
    with open(DATA / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "model_id", "metric", "value", "unit"])
        for run in runs:
            grid = adaptive_grid(run.ts, run.ss, 120.0, 600.0)
            s = resample(run.ts, run.ss, grid)
            moon = moon_positions(grid)
            er = np.linalg.norm(s[:, :3], axis=1)
            mr = np.linalg.norm(s[:, :3] - moon, axis=1)
            sp = np.linalg.norm(s[:, 3:], axis=1)
            rows = []
            if run.model_id != "nasa":
                cover = grid <= truth.ts[-1]
                err = np.full(grid.shape, np.nan)
                err[cover] = np.linalg.norm(s[cover, :3] - truth_interp(grid[cover])[:, :3], axis=1)
                rows.append(("position_error_km", "km", err, 3))
            rows += [("earth_range_km", "km", er, 1), ("moon_range_km", "km", mr, 1), ("speed_km_s", "km/s", sp, 5)]
            if run.model_id == "nasa":
                dv = np.array([sum(d for tb, d in burn_times if tb <= t + 1e-6) for t in grid])
                rows.append(("delta_v_m_s", "m/s", dv, 2))
            for name, unit, vals, nd in rows:
                for t, val in zip(grid, vals):
                    if not math.isfinite(val):
                        continue
                    w.writerow([fmt(t, 1), run.model_id, name, fmt(val, nd), unit])
                    n_rows += 1
    print(f"  metrics.csv: {n_rows} rows")

    # events.csv
    ev = []
    ev.append((0.0, "nasa", "launch", "milestone", "", "", "Liftoff from Kennedy Space Center, LC-39B"))
    ev.append((t_arb, "nasa", "apogee_raise_burn", "burn", round(dv_arb * 1000, 1), "m/s", "Upper stage raises apogee into a high Earth orbit"))
    ev.append((t_tli, "nasa", "translunar_injection", "burn", round(dv_tli * 1000, 1), "m/s", "Service module burn that sends Orion to the Moon"))
    ev.append((t_seed, "nasa", "model_seed", "milestone", "", "", "Every model starts from NASA's state here"))
    for b in burns:
        ev.append((b.t, "nasa", b.name, "burn", round(float(np.linalg.norm(b.rsw)), 2), "m/s", "Small trajectory correction"))
    for run in runs:
        s = run.ss
        pts = run.ts >= (t_seed if run.model_id != "nasa" else t_tli)
        ts, ss = run.ts[pts], s[pts]
        t_pl, d_pl = perilune(ts, ss)
        er = np.linalg.norm(ss[:, :3], axis=1)
        k = int(np.argmax(er))
        t_ma, r_ma = _refine_min(ts, -er, k)
        if run.model_id == "nasa":
            # Reported events carry NASA's published figures, as the real file will.
            ev.append((t_of(TARGET_PERILUNE_UTC), "nasa", "closest_lunar_approach", "apsis", TARGET_PERILUNE_ALT, "km",
                       "Reported by NASA: 6,545 km above the lunar surface, about 23:00 UTC on 6 April"))
            ev.append((t_ma, "nasa", "max_earth_distance", "apsis", 406740.0, "km", "Reported by NASA: 406,740 km from Earth"))
        else:
            ev.append((t_pl, run.model_id, "closest_lunar_approach", "apsis", round(d_pl - R_M, 1), "km",
                       "Predicted closest approach to the Moon (altitude above the mean lunar radius)"))
            ev.append((t_ma, run.model_id, "max_earth_distance", "apsis", round(-r_ma, 1), "km", "Farthest distance from Earth's centre"))
        r_end = float(np.linalg.norm(ss[-1, :3]))
        if r_end < R_E + EI_ALT + 50:
            ev.append((float(ts[-1]), run.model_id, "entry_interface", "milestone", EI_ALT, "km", "Reaches the top of the atmosphere (400,000 ft)"))
        elif run.model_id != "nasa":
            t_rp, r_rp = return_perigee(ts, ss, t_pl)
            if math.isfinite(t_rp):
                ev.append((t_rp, run.model_id, "return_perigee", "apsis", round(r_rp - R_E, 1), "km", "Closest return to Earth: misses the atmosphere"))
    ev.sort(key=lambda e: (e[0], e[1]))
    with open(DATA / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "model_id", "event", "kind", "value", "unit", "note"])
        for e in ev:
            w.writerow([fmt(e[0], 1), *e[1:]])

    # windows.csv on a 60 s grid
    wrows = []
    for run in runs:
        t0 = run.ts[0] if run.model_id != "nasa" else t_ins
        grid = np.arange(math.ceil(t0 / 60.0) * 60.0, run.ts[-1], 60.0)
        s = resample(run.ts, run.ss, grid)[:, :3]
        for st in STATIONS:
            g = station_eci(st, grid)
            up = g / np.linalg.norm(g, axis=1)[:, None]
            rel = s - g
            el = np.degrees(np.arcsin(np.sum(rel * up, axis=1) / np.linalg.norm(rel, axis=1)))
            for a, b in windows_from_mask(grid, el > DSN_MASK_DEG):
                wrows.append((run.model_id, "dsn_contact", st, a, b))
        moon = moon_positions(grid)
        rs = np.linalg.norm(s, axis=1)
        rm = np.linalg.norm(moon, axis=1)
        cosang = np.sum(s * moon, axis=1) / (rs * rm)
        ang = np.arccos(np.clip(cosang, -1, 1))
        hidden = (ang < np.arcsin(R_M / rm)) & (rs > rm)
        for a, b in windows_from_mask(grid, hidden):
            wrows.append((run.model_id, "lunar_blackout", "", a, b))
    with open(DATA / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_id", "kind", "station", "start_s", "end_s"])
        for r in wrows:
            w.writerow([r[0], r[1], r[2], fmt(r[3], 1), fmt(r[4], 1)])

    # meta.json
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    tdb = EPOCH + timedelta(seconds=TT_MINUS_UTC)
    meta = {
        "synthetic": True,
        "title": "Artemis II lunar flyby",
        "epoch_utc": EPOCH.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "epoch_tdb": tdb.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + " TDB",
        "epoch_label": "Launch (T+0)",
        "frame": "Earth-centred ICRF (J2000 axes), km",
        "source": "SYNTHETIC layout fixture from demo/artemis2/make_sample_data.py: toy dynamics, invented burns, a Keplerian Moon. Not NASA data and not OrbitalEngine output.",
        "truth_source": "Synthetic stand-in for NASA's reconstructed Orion trajectory",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_commit": commit,
        "engine_version": None,
        "notes": [
            "NASA-reported reference figures (6,545 km lunar periapsis altitude, 406,740 km from Earth) are targets the fixture was steered toward, not measurements.",
        ],
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
