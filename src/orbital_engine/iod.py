"""
Initial orbit determination and two-point boundary values: Lambert's problem, Gibbs, Herrick-Gibbs and
Gauss's angles-only method. All are two-body (one central `mu`), in any inertial frame, km / s.

These are boundary tools, not step code: each call is a handful of scalar operations on `(3,)` vectors,
made once per observation set or per burn. They are written for clarity and conditioning rather than
vectorised over many problems.

References
----------
- Curtis, *Orbital Mechanics for Engineering Students*, Ch. 5: Gibbs (Alg. 5.1), Lambert by universal
  variables (Alg. 5.2), Gauss's method (Alg. 5.5) with iterative improvement (Alg. 5.6); Stumpff
  functions and the universal Kepler equation, Ch. 3 (Alg. 3.3). Validation: Examples 5.1 and 5.2.
- Vallado, *Fundamentals of Astrodynamics and Applications*, Herrick-Gibbs (Alg. 55): the Taylor-series
  replacement for Gibbs when the three positions are closely spaced.

Which to use
------------
- `gibbs`: three position vectors, any spacing above a few degrees. Exact on a conic; ill-conditioned as
  the separation shrinks, because it divides by cross products of nearly parallel vectors.
- `herrick_gibbs`: three **closely spaced, timed** positions (around a degree or less of arc). Its error
  is a truncation of order `(n dt)^4` relative (`n` the mean motion), so it improves as Gibbs worsens.
- `lambert`: two positions and a time of flight. It gives the velocities at both ends, which makes it
  the transfer and targeting tool. Zero-revolution, short way by default (`prograde`).
- `gauss`: three timed **lines of sight** from known observer positions (angles only). Its first
  estimate truncates f and g, an error of order `(n tau)^3` (2e-3 at two-minute spacing in LEO);
  `improve=True` iterates it to the exact two-body solution.

Expected accuracy, on exact two-body data: `gibbs`, `lambert` and improved `gauss` agree with the
truth to round-off and solver tolerance (better than 1e-9 relative). `herrick_gibbs` and unimproved
`gauss` carry truncation error that shrinks as the spacing does, at the orders above.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .custom_types import ArrayFloat

__all__ = ["stumpff_c", "stumpff_s", "lambert", "gibbs", "herrick_gibbs", "gauss", "kepler_universal_fg",
           "kepler_universal"]


# --------------------------------------------------------------------------------------------------
# Stumpff functions and the universal Kepler equation (Curtis Ch. 3)
# --------------------------------------------------------------------------------------------------

def stumpff_c(z: float) -> float:
    """C(z) = (1 - cos sqrt z) / z, continued through z <= 0; series near 0 (no cancellation)."""
    if abs(z) < 1e-3:
        return 0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0 + z ** 4 / 3628800.0
    if z > 0.0:
        return (1.0 - math.cos(math.sqrt(z))) / z
    return (math.cosh(math.sqrt(-z)) - 1.0) / (-z)


def stumpff_s(z: float) -> float:
    """S(z) = (sqrt z - sin sqrt z) / sqrt(z)^3, continued through z <= 0; series near 0."""
    if abs(z) < 1e-3:
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0 + z ** 4 / 39916800.0
    if z > 0.0:
        s = math.sqrt(z)
        return (s - math.sin(s)) / s ** 3
    s = math.sqrt(-z)
    return (math.sinh(s) - s) / s ** 3


def kepler_universal_fg(r0: ArrayFloat, v0: ArrayFloat, dt: float, mu: float, *, tol: float = 1e-12,
                        max_iter: int = 100) -> Tuple[float, float, float]:
    """
    `(f, g, chi)`: Lagrange coefficients `r(t0 + dt) = f r0 + g v0` from the universal Kepler equation
    (Curtis Alg. 3.3, Newton on the universal anomaly `chi`, km^0.5), for any conic.
    """
    rn = float(np.linalg.norm(r0))
    vr = float(r0 @ v0) / rn
    alpha = 2.0 / rn - float(v0 @ v0) / mu
    sq = math.sqrt(mu)
    chi = sq * abs(alpha) * dt if alpha != 0.0 else sq * dt / rn
    for _ in range(max_iter):
        z = alpha * chi * chi
        c, s = stumpff_c(z), stumpff_s(z)
        f_val = rn * vr / sq * chi * chi * c + (1.0 - alpha * rn) * chi ** 3 * s + rn * chi - sq * dt
        f_der = rn * vr / sq * chi * (1.0 - z * s) + (1.0 - alpha * rn) * chi * chi * c + rn
        step = f_val / f_der
        chi -= step
        if abs(step) <= tol * max(1.0, abs(chi)):
            break
    else:
        raise RuntimeError(f"universal Kepler equation did not converge in {max_iter} iterations")
    z = alpha * chi * chi
    f = 1.0 - chi * chi / rn * stumpff_c(z)
    g = dt - chi ** 3 * stumpff_s(z) / sq
    return f, g, chi


def kepler_universal(r0: ArrayFloat, v0: ArrayFloat, dt: float, mu: float) -> Tuple[ArrayFloat, ArrayFloat]:
    """`(r, v)` after `dt` on the two-body conic through `(r0, v0)` (Curtis Alg. 3.4: f, g, f-dot, g-dot)."""
    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    f, g, chi = kepler_universal_fg(r0, v0, dt, mu)
    r: ArrayFloat = f * r0 + g * v0
    rn, r0n = float(np.linalg.norm(r)), float(np.linalg.norm(r0))
    alpha = 2.0 / r0n - float(v0 @ v0) / mu
    z = alpha * chi * chi
    f_dot = math.sqrt(mu) / (rn * r0n) * (z * chi * stumpff_s(z) - chi)
    g_dot = 1.0 - chi * chi / rn * stumpff_c(z)
    v: ArrayFloat = f_dot * r0 + g_dot * v0
    return r, v


# --------------------------------------------------------------------------------------------------
# Lambert's problem (Curtis Alg. 5.2)
# --------------------------------------------------------------------------------------------------

def lambert(r1: ArrayFloat, r2: ArrayFloat, tof: float, mu: float, *,
            prograde: bool = True) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    `(v1, v2)`: the zero-revolution conic from `r1` to `r2` in `tof` seconds. `prograde` picks the
    transfer whose angular momentum has a positive z component (the short way for a prograde orbit).

    Universal variables: the time-of-flight equation `F(z) = 0` is monotonic in `z`, so it is solved by
    bisection on a bracket `(z_lo, 4 pi^2)`; there is no starting guess to fail. Raises `ValueError`
    for a transfer angle of 0 or 180 deg, where the plane is undefined.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    if tof <= 0.0:
        raise ValueError("lambert: time of flight must be positive")
    n1, n2 = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    cos_t = float(r1 @ r2) / (n1 * n2)
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = math.acos(cos_t)
    hz = float(np.cross(r1, r2)[2])
    if (prograde and hz < 0.0) or (not prograde and hz >= 0.0):
        theta = 2.0 * math.pi - theta
    if abs(math.sin(theta)) < 1e-10:
        raise ValueError("lambert: transfer angle is 0 or 180 deg, the transfer plane is undefined")
    a_par = math.sin(theta) * math.sqrt(n1 * n2 / (1.0 - math.cos(theta)))
    sq_mu = math.sqrt(mu)

    def y_of(z: float) -> float:
        c = stumpff_c(z)
        return n1 + n2 + a_par * (z * stumpff_s(z) - 1.0) / math.sqrt(c) if c > 0.0 else math.inf

    def f_of(z: float) -> float:
        c = stumpff_c(z)
        if c <= 0.0:                       # at the zero-revolution limit z = 4 pi^2: unbounded time
            return math.inf
        y = y_of(z)
        if y <= 0.0:                       # below the feasible region: shorter than any transfer
            return -math.inf
        return math.pow(y / c, 1.5) * stumpff_s(z) + a_par * math.sqrt(y) - sq_mu * tof

    z_hi = 4.0 * math.pi ** 2 * (1.0 - 1e-6)
    z_lo = -4.0 * math.pi ** 2
    while f_of(z_lo) > 0.0:
        z_lo = 2.0 * z_lo - 1.0
        if z_lo < -1e6:
            raise ValueError("lambert: no zero-revolution solution bracketed")
    if f_of(z_hi) < 0.0:
        raise ValueError("lambert: time of flight exceeds the zero-revolution limit")
    for _ in range(400):
        z_mid = 0.5 * (z_lo + z_hi)
        if f_of(z_mid) > 0.0:
            z_hi = z_mid
        else:
            z_lo = z_mid
        if z_hi - z_lo <= 1e-15 * max(1.0, abs(z_mid)):
            break
    z = 0.5 * (z_lo + z_hi)
    y = y_of(z)
    f = 1.0 - y / n1
    g = a_par * math.sqrt(y / mu)
    g_dot = 1.0 - y / n2
    v1: ArrayFloat = (r2 - f * r1) / g
    v2: ArrayFloat = (g_dot * r2 - r1) / g
    return v1, v2


# --------------------------------------------------------------------------------------------------
# Gibbs and Herrick-Gibbs
# --------------------------------------------------------------------------------------------------

def gibbs(r1: ArrayFloat, r2: ArrayFloat, r3: ArrayFloat, mu: float, *,
          coplanar_tol_rad: float = 1e-3) -> ArrayFloat:
    """
    Velocity at `r2` of the conic through three positions (Curtis Alg. 5.1). Raises `ValueError` if the
    three are further than `coplanar_tol_rad` from one plane (the angle of `r1` out of the plane of
    `r2`, `r3`), where no single conic passes through them.
    """
    r1, r2, r3 = (np.asarray(x, dtype=np.float64) for x in (r1, r2, r3))
    n1, n2, n3 = (float(np.linalg.norm(x)) for x in (r1, r2, r3))
    c23 = np.cross(r2, r3)
    off_plane = abs(math.asin(min(1.0, abs(float(r1 @ c23)) / (n1 * float(np.linalg.norm(c23))))))
    if off_plane > coplanar_tol_rad:
        raise ValueError(f"gibbs: positions are {off_plane:.2e} rad from coplanar (tolerance {coplanar_tol_rad:g})")
    n_vec = n1 * c23 + n2 * np.cross(r3, r1) + n3 * np.cross(r1, r2)
    d_vec = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    s_vec = r1 * (n2 - n3) + r2 * (n3 - n1) + r3 * (n1 - n2)
    out: ArrayFloat = math.sqrt(mu / float(n_vec @ d_vec)) * (np.cross(d_vec, r2) / n2 + s_vec)
    return out


def herrick_gibbs(r1: ArrayFloat, r2: ArrayFloat, r3: ArrayFloat, t1: float, t2: float, t3: float,
                  mu: float) -> ArrayFloat:
    """
    Velocity at `r2` from three closely spaced, timed positions (Vallado Alg. 55): a Taylor-series fit
    of the trajectory with the two-body acceleration folded in. Truncation error of order `(n dt)^4`
    relative; use `gibbs` once the arcs exceed a few degrees.
    """
    r1, r2, r3 = (np.asarray(x, dtype=np.float64) for x in (r1, r2, r3))
    n1, n2, n3 = (float(np.linalg.norm(x)) for x in (r1, r2, r3))
    dt21, dt31, dt32 = t2 - t1, t3 - t1, t3 - t2
    out: ArrayFloat = (-dt32 * (1.0 / (dt21 * dt31) + mu / (12.0 * n1 ** 3)) * r1
                       + (dt32 - dt21) * (1.0 / (dt21 * dt32) + mu / (12.0 * n2 ** 3)) * r2
                       + dt21 * (1.0 / (dt32 * dt31) + mu / (12.0 * n3 ** 3)) * r3)
    return out


# --------------------------------------------------------------------------------------------------
# Gauss's angles-only method (Curtis Alg. 5.5, 5.6)
# --------------------------------------------------------------------------------------------------

def _positive_roots(a: float, b: float, c: float) -> List[float]:
    """Positive real roots of `x^8 + a x^6 + b x^3 + c = 0`."""
    roots = np.roots([1.0, 0.0, a, 0.0, 0.0, b, 0.0, 0.0, c])
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8 * max(1.0, abs(r)) and r.real > 0.0)


def gauss(t: Tuple[float, float, float], observer: ArrayFloat, los: ArrayFloat, mu: float, *,
          improve: bool = True, root: Optional[float] = None, tol: float = 1e-12,
          max_iter: int = 200) -> Tuple[ArrayFloat, ArrayFloat]:
    """
    `(r2, v2)` at the middle observation from three lines of sight (Curtis Alg. 5.5).

    `t` the three times (s), `observer` `(3, 3)` the observer's inertial positions at them (km), `los`
    `(3, 3)` the unit vectors from observer to object. The eighth-degree equation for `r2` can have
    more than one positive root; if more than one gives positive slant ranges, `root` must pick it
    (the error lists them). With `improve`, the truncated f and g series are replaced by exact ones from
    the universal Kepler equation and iterated to `tol` in the slant ranges (Alg. 5.6).
    """
    t1, t2, t3 = t
    obs = np.asarray(observer, dtype=np.float64)
    rho = np.asarray(los, dtype=np.float64)
    rho = rho / np.linalg.norm(rho, axis=1)[:, None]
    tau1, tau3 = t1 - t2, t3 - t2
    tau = tau3 - tau1
    p1 = np.cross(rho[1], rho[2])
    p2 = np.cross(rho[0], rho[2])
    p3 = np.cross(rho[0], rho[1])
    d0 = float(rho[0] @ p1)
    if abs(d0) < 1e-14:
        raise ValueError("gauss: the three lines of sight are coplanar; the method is singular")
    d = np.array([[float(obs[i] @ p) for p in (p1, p2, p3)] for i in range(3)])
    a_c = (-d[0, 1] * tau3 / tau + d[1, 1] + d[2, 1] * tau1 / tau) / d0
    b_c = (d[0, 1] * (tau3 ** 2 - tau ** 2) * tau3 / tau + d[2, 1] * (tau ** 2 - tau1 ** 2) * tau1 / tau) / (6.0 * d0)
    e_c = float(obs[1] @ rho[1])
    r2sq = float(obs[1] @ obs[1])
    a = -(a_c ** 2 + 2.0 * a_c * e_c + r2sq)
    b = -2.0 * mu * b_c * (a_c + e_c)
    c = -(mu ** 2) * b_c ** 2

    def ranges(x: float) -> Tuple[float, float, float]:
        x3 = x ** 3
        rho1 = ((6.0 * (d[2, 0] * tau1 / tau3 + d[1, 0] * tau / tau3) * x3
                 + mu * d[2, 0] * (tau ** 2 - tau1 ** 2) * tau1 / tau3)
                / (6.0 * x3 + mu * (tau ** 2 - tau3 ** 2)) - d[0, 0]) / d0
        rho2 = a_c + mu * b_c / x3
        rho3 = ((6.0 * (d[0, 2] * tau3 / tau1 - d[1, 2] * tau / tau1) * x3
                 + mu * d[0, 2] * (tau ** 2 - tau3 ** 2) * tau3 / tau1)
                / (6.0 * x3 + mu * (tau ** 2 - tau1 ** 2)) - d[2, 2]) / d0
        return rho1, rho2, rho3

    if root is None:
        valid = [x for x in _positive_roots(a, b, c) if min(ranges(x)) > 0.0]
        if len(valid) != 1:
            raise ValueError(f"gauss: {len(valid)} physically valid roots for r2 ({valid}); pass `root`")
        root = valid[0]
    rho1, rho2, rho3 = ranges(root)
    r = [obs[0] + rho1 * rho[0], obs[1] + rho2 * rho[1], obs[2] + rho3 * rho[2]]
    x3 = root ** 3
    f1 = 1.0 - 0.5 * mu * tau1 ** 2 / x3
    f3 = 1.0 - 0.5 * mu * tau3 ** 2 / x3
    g1 = tau1 - mu * tau1 ** 3 / (6.0 * x3)
    g3 = tau3 - mu * tau3 ** 3 / (6.0 * x3)
    v2: ArrayFloat = (-f3 * r[0] + f1 * r[2]) / (f1 * g3 - f3 * g1)
    if not improve:
        out_r: ArrayFloat = r[1]
        return out_r, v2

    for _ in range(max_iter):
        f1n, g1n, _ = kepler_universal_fg(r[1], v2, tau1, mu)
        f3n, g3n, _ = kepler_universal_fg(r[1], v2, tau3, mu)
        # Averaging the old and new coefficients damps the iteration (Curtis Alg. 5.6, step 4).
        f1, g1, f3, g3 = 0.5 * (f1 + f1n), 0.5 * (g1 + g1n), 0.5 * (f3 + f3n), 0.5 * (g3 + g3n)
        den = f1 * g3 - f3 * g1
        c1, c3 = g3 / den, -g1 / den
        new = ((-d[0, 0] + d[1, 0] / c1 - d[2, 0] * c3 / c1) / d0,
               (-c1 * d[0, 1] + d[1, 1] - c3 * d[2, 1]) / d0,
               (-d[0, 2] * c1 / c3 + d[1, 2] / c3 - d[2, 2]) / d0)
        change = max(abs(n - o) / max(1.0, abs(n)) for n, o in zip(new, (rho1, rho2, rho3)))
        rho1, rho2, rho3 = new
        r = [obs[0] + rho1 * rho[0], obs[1] + rho2 * rho[1], obs[2] + rho3 * rho[2]]
        v2 = (-f3 * r[0] + f1 * r[2]) / den
        if change < tol:
            break
    else:
        raise RuntimeError(f"gauss: iterative improvement did not converge in {max_iter} iterations")
    out_r2: ArrayFloat = r[1]
    return out_r2, v2
