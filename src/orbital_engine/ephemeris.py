"""
Tabulated ephemerides and the force model `"ephemeris_third_body"`: point-mass perturbers (Moon, Sun)
whose positions come from a table interpolated at **each Runge-Kutta stage's own time**.

Why this exists
---------------
`thirdbody.py`'s `"third_body"` takes its perturber from the arena. The perturber is then an analytic
Keplerian body, one per perturbed body, and `RK4Integrator` holds it at its start-of-step position for
all four stages, so a Cowell body under it converges at **first order** (`thirdbody.py` derives it:
the Moon under the Sun is 2.64 km off after 30 days at 3600 s). A real trajectory - a crewed lunar
flyby compared against a navigation solution - needs the Moon and Sun at their *real* positions, which
are data, not Keplerian orbits, and needs them at the right time in every stage. This module supplies
both: the perturber is a function of time, evaluated at the time the integrator hands the kernel, so
nothing is frozen and **RK4 keeps its fourth order** (`tests/validation/test_ephemeris.py`: ratios
~16 per halving where `"third_body"` in the same geometry gives 2).

Data contract (`EphemerisTable`)
--------------------------------
`EphemerisTable(name, t_s, position_km, velocity_km_s, centre=None)`: `t_s` `(n,)` seconds on the
table's own time scale (TDB for a JPL Horizons export), strictly increasing, `n >= 2`;
`position_km` / `velocity_km_s` `(n, 3)`, the perturber **relative to the body it is centred on** -
Earth-centred ICRF for a geocentric Horizons vector table - in the engine's inertial frame. All
finite. The arrays are copied and made read-only; the table is immutable. `centre`, optional, names
the arena body the table is centred on; when given, `enable_force_model` checks it (see Refusals).

Nothing in this module reads a file, a network or `jplephem` - the table is plain arrays handed over
at configuration time. That is the boundary `CLAUDE.md` asks for: no stateful third-party object, no
I/O and no ephemeris *model* inside a step, only arithmetic on immutable data.

Interpolation: piecewise cubic Hermite
--------------------------------------
On `[t_k, t_{k+1}]`, `h = t_{k+1} - t_k`, `s = (t - t_k)/h`, the unique cubic matching position and
velocity at both nodes:

    p(t) = H00(s) x_k + H10(s) h v_k + H01(s) x_{k+1} + H11(s) h v_{k+1}
    H00 = 2s^3 - 3s^2 + 1,  H10 = s^3 - 2s^2 + s,  H01 = -2s^3 + 3s^2,  H11 = s^3 - s^2

It is C1 across nodes (value and derivative match the data at both ends of every interval), so the
force the integrator sees is continuous with a continuous rate - no kink for RK4 to trip on. Using the
tabulated velocities makes it fourth order from two nodes, local to one interval, with no end
conditions (a cubic spline needs those and couples every node).

*Error bound.* The Hermite remainder (Burden & Faires, *Numerical Analysis*, Sec. 3.4, Thm 3.9 in
the 9th ed. - **from memory, unverified**; the argument is the Lagrange remainder with each node
doubled) is, per component,

    x(t) - p(t) = x''''(xi)/4! * (t - t_k)^2 (t - t_{k+1})^2,      xi in (t_k, t_{k+1})

and `max s^2 (1 - s)^2 = 1/16` at the midpoint, so

    |x - p| <= h^4 max|x''''| / 384                                                       (E0)

nearly attained at the midpoint, where the error is `x''''(t_mid) h^4/384` to leading order - a
*prediction*, not only a bound. For a circular orbit (`x'''' = n^4 x`) the midpoint error is exactly
`(n h)^4/384 * r_mid`, radial and outward (the interpolant cuts inside the circle), less a relative
`(n h/2)^2/15`. Magnitudes, with the Moon's `n = 2.66e-6 rad/s` at 384400 km: `h = 1 h` gives
8.8e-6 km, `h = 1 day` 2.9 km. The Sun as seen from Earth (`n = 1.99e-7 rad/s`, 1.5e8 km) at 1 h:
1.1e-7 km. A Horizons table at 1 h is therefore interpolation-exact for any purpose here; halving
the step divides the error by 16.

The interpolated *velocity* (`EphemerisTable.velocity`, the derivative of `p`) is third order, with
leading error `x''''(t) w'(t)/24`, `w = (t - t_k)^2 (t - t_{k+1})^2`, whose maximum over the interval
gives

    |v - p'| <= sqrt(3) h^3 max|x''''| / 216                                              (E1)

(Birkhoff & Priver 1967, *J. Math. Phys.* 46, the optimal constants of cubic Hermite interpolation
`1/384, sqrt(3)/216, 1/12, 1/2` - **from memory, unverified**; the leading-term derivation just
given is self-contained). The force model reads positions only; the velocity is for callers, such as
a closest-approach range rate.

`EphemerisTable.error_estimate_km()` applies (E0) to a table's own data, estimating `max|x''''|`
per interval from third divided differences of the tabulated velocities - how to check a real
table's step without an analytic truth. It is an estimate (the divided difference samples `v'''` at
an interior point), accurate to `O(h)` relative.

*Out of range raises.* A query outside `[t_s[0], t_s[-1]]` raises `ValueError`: extrapolating a
cubic is silently wrong by more than any bound above, and a table that does not cover the run is a
configuration error. So does a non-finite query time.

*Time precision.* Query time is `epoch_s + t` (see Coefficients). At TDB seconds since J2000 in 2026
(8.3e8 s) one ulp is 1.2e-7 s - 1.2e-7 km of lunar motion - far below (E0).

Tables at the boundary: registration and lifetime
-------------------------------------------------
A force kernel receives only a float `params` row per body (`forces.ForceKernel`), so a body cannot
hold a table; it holds a **key**. `register_ephemeris(table)` stores the table in a module-level memo
and returns its key, a float (exact integer below 2^52) derived from a BLAKE2b digest of the table's
name, centre, times, positions and velocities. The key is therefore a **pure function of content**:

- registering the same table twice (in one process or two) returns the same key, so a sweep config
  that stores a key is reproducible whenever its table is registered again;
- a key can never be re-bound to different data. A digest collision between two different tables
  (probability ~n^2 / 2^53) raises rather than overwriting.

That memo is the only module state. It lives for the process, holds only immutable tables, and cannot
leak behaviour between simulations: two simulations that share a key share, by construction, the same
data, exactly as two simulations that share an MSIS triple share one profile (`msis_bridge.py`, the
pattern this follows). The kernel reads the memo through `registered_ephemeris`, which **raises
`LookupError`** on an unknown key (a row written by direct assignment, bypassing `enable_force_model`)
rather than guessing. There is no unregister; a table is tens of kB.

Force model `"ephemeris_third_body"`
------------------------------------
Physics - `thirdbody.py`'s derivation, unchanged: for a body at `r` relative to its Keplerian parent
`P`, and a perturber of mass `mu_s` at `r_s` **relative to the same parent**,

    a = mu_s [ (r_s - r)/|r_s - r|^3  -  r_s/|r_s|^3 ]
               '----- direct -----'    '- indirect -'

The indirect term is the perturber's pull on the parent, which the parent-relative frame inherits. For
an Earth satellite at 42000 km under the Moon, the two terms are each ~3.3e-8 km/s^2 and their
difference is ~7e-9 km/s^2 (the lunar tide): dropping the indirect term is a 5x error; with the Sun
in the same geometry (5.9e-6 against a 3.3e-9 km/s^2 tide) it is ~1800x. **The table must be centred on the parent** - `r_s` is used as
given, never re-centred, since the parent's own ephemeris is not in the arena - so an Earth-centred
Moon table on a body whose parent is Earth. `centre` is how to have that checked.

Montenbruck, O. & Gill, E., *Satellite Orbits* (2000), Sec. 3.3, the point-mass form
`r'' = GM (s - r)/|s - r|^3 - GM s/|s|^3` (**equation number 3.37 from memory, unverified**); the
five-line derivation in `thirdbody.py` is what should be checked.

*Stage time.* The kernel evaluates `r_s` at `epoch_s + t`, where `t` is the evaluation time the
integrator passes - `t, t + h/2, t + h/2, t + h` under `RK4Integrator`, and a split sub-step's own
start after a manoeuvre or event split (`tesseral.py` established both). Every term then depends only
on the parent-relative position and on time, so the relative ODE `r'' = f(t, r)` is exactly what RK4
integrates, with no frozen quantity: **fourth order**. The parent's absolute motion never enters.

Coefficients (`EPHEMERIS_PARAM_NAMES`)
--------------------------------------
`("epoch_s", "table_1", "mu_1", "table_2", "mu_2", "table_3", "mu_3")` - up to
`MAX_PERTURBERS = 3` perturbers per body (Moon, Sun and one spare), each a `(key, mu km^3/s^2)` pair.
`epoch_s` is the **table time at simulation time 0**, shared by the body's perturbers (they must share
a time scale), so the query time is `epoch_s + sim.t`. `ephemeris_coefficients([(table, mu), ...],
epoch_s=...)` registers the tables and builds the keyword set, which is also what a
`sweep.ForceModelSpec`'s `coefficients` takes.

**Zero means absent, exactly.** A slot with `mu == 0` is skipped - its table is never looked up and
nothing is added, bit for bit - so `mu_k = 0` switches a perturber off without touching the key. A
zero separation in either term adds exactly `0.0` for that term, as in `third_body`.

Refusals (before any bit is set)
--------------------------------
`validate_bodies`: a barycentre parent (a barycentre is not where any table is centred, and its `mu`
row is summed system mass). `validate_coefficients`, on each body's *effective* row (passed values
over stored ones): a non-finite coefficient; `mu < 0`; a non-zero `mu` with key 0 or an unregistered
key; a non-integral or unregistered non-zero key; the same table twice among one body's live slots
(double counting); a table whose `centre` is not the body's parent; and a live table that does not
cover `epoch_s + sim.t`, the time the next step starts at.

Not in the fused compiled Cowell plan: the bit is foreign to `Simulation._refresh_cowell_plan`, so a
Cowell set carrying it runs on the NumPy `RK4Integrator`. No compiled twin; one would need the tables
passed into `kernels.cowell_rk4_step` (a `kernel-twin` job, not this module's).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Final, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat, ArrayKilometers, ArraySeconds, ScalarSeconds
from .geopotential import barycentre_parented
from .registry import register_force_model

if TYPE_CHECKING:
    from .simulator import Simulation

__all__ = [
    "EPHEMERIS_MODEL", "EPHEMERIS_PARAM_NAMES", "MAX_PERTURBERS",
    "HERMITE_POSITION_CONSTANT", "HERMITE_VELOCITY_CONSTANT",
    "EphemerisTable", "ephemeris_key", "register_ephemeris", "registered_ephemeris",
    "is_registered", "ephemeris_coefficients", "hermite_position_bound", "hermite_velocity_bound",
    "ephemeris_third_body_kernel",
]

EPHEMERIS_MODEL: Final[str] = "ephemeris_third_body"
MAX_PERTURBERS: Final[int] = 3
_EPOCH_COL: Final[int] = 0

EPHEMERIS_PARAM_NAMES: Final[Tuple[str, ...]] = ("epoch_s",) + tuple(
    f"{kind}_{k + 1}" for k in range(MAX_PERTURBERS) for kind in ("table", "mu")
)

#: (E0): `|x - p| <= HERMITE_POSITION_CONSTANT * h^4 * max|x''''|`.
HERMITE_POSITION_CONSTANT: Final[float] = 1.0 / 384.0
#: (E1): `|v - p'| <= HERMITE_VELOCITY_CONSTANT * h^3 * max|x''''|`.
HERMITE_VELOCITY_CONSTANT: Final[float] = math.sqrt(3.0) / 216.0

# Keys are integers below 2^52, exactly representable in a float64 coefficient row, and never 0
# (0 is the empty slot).
_KEY_BITS: Final[int] = 52


def _key_col(k: int) -> int:
    return 1 + 2 * k


def _mu_col(k: int) -> int:
    return 2 + 2 * k


def hermite_position_bound(h: ArraySeconds, max_fourth_derivative: ArrayFloat) -> ArrayFloat:
    """(E0): `h^4 max|x''''| / 384`, km for `h` in s and `x''''` in km/s^4."""
    out: ArrayFloat = HERMITE_POSITION_CONSTANT * np.asarray(h, dtype=np.float64) ** 4 * np.asarray(
        max_fourth_derivative, dtype=np.float64)
    return out


def hermite_velocity_bound(h: ArraySeconds, max_fourth_derivative: ArrayFloat) -> ArrayFloat:
    """(E1): `sqrt(3) h^3 max|x''''| / 216`, km/s."""
    out: ArrayFloat = HERMITE_VELOCITY_CONSTANT * np.asarray(h, dtype=np.float64) ** 3 * np.asarray(
        max_fourth_derivative, dtype=np.float64)
    return out


def _readonly(a: ArrayFloat) -> ArrayFloat:
    out: ArrayFloat = np.array(a, dtype=np.float64, copy=True, order="C")
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class EphemerisTable:
    """
    One perturber's tabulated state relative to the body the table is centred on. See the module
    docstring for the data contract, the interpolation and its error bound. Immutable: the arrays
    are copied to read-only float64 on construction. Equality is identity; compare `ephemeris_key`s
    to compare content.
    """
    name: str
    t_s: ArraySeconds
    position_km: ArrayKilometers
    velocity_km_s: ArrayFloat
    centre: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"an ephemeris table needs a non-empty name, got {self.name!r}")
        if self.centre is not None and (not isinstance(self.centre, str) or not self.centre):
            raise ValueError(f"table '{self.name}': centre must be a body name or None, got {self.centre!r}")
        t = _readonly(np.asarray(self.t_s, dtype=np.float64))
        pos = _readonly(np.asarray(self.position_km, dtype=np.float64))
        vel = _readonly(np.asarray(self.velocity_km_s, dtype=np.float64))
        if t.ndim != 1 or t.size < 2:
            raise ValueError(f"table '{self.name}': t_s must be 1-D with at least 2 nodes, got shape {t.shape}")
        n = t.size
        if pos.shape != (n, 3) or vel.shape != (n, 3):
            raise ValueError(
                f"table '{self.name}': position_km and velocity_km_s must have shape ({n}, 3), got "
                f"{pos.shape} and {vel.shape}")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError(f"table '{self.name}': times, positions and velocities must all be finite")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError(f"table '{self.name}': t_s must be strictly increasing")
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "position_km", pos)
        object.__setattr__(self, "velocity_km_s", vel)

    @property
    def t_min(self) -> float:
        return float(self.t_s[0])

    @property
    def t_max(self) -> float:
        return float(self.t_s[-1])

    def covers(self, times: ArraySeconds) -> bool:
        """True if every query time is finite and inside `[t_min, t_max]`. (A NaN makes `min`/`max`
        NaN, and every comparison with NaN is false, so one pair of reductions checks all three.)"""
        q = np.asarray(times, dtype=np.float64)
        if q.size == 0:
            return True
        return bool(q.min() >= self.t_s[0] and q.max() <= self.t_s[-1])

    def _locate(self, times: ArraySeconds) -> Tuple[NDArray[np.int64], ArrayFloat, ArrayFloat]:
        """Interval index `k`, step `h` and fraction `s` for each query; raises outside the table."""
        q = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if not self.covers(q):
            bad = q[~(np.isfinite(q) & (q >= self.t_s[0]) & (q <= self.t_s[-1]))]
            raise ValueError(
                f"ephemeris '{self.name}' covers t = [{self.t_min!r}, {self.t_max!r}] s; queried "
                f"{bad[:3].tolist()} - out of range (or non-finite). Tables never extrapolate: extend the "
                f"table, or check the body's epoch_s coefficient.")
        # side="right" puts a query equal to node j in interval j, and so the last node in interval
        # n - 1, which does not exist: it belongs to interval n - 2 at s = 1. In range, k >= 0 always.
        k: NDArray[np.int64] = np.searchsorted(self.t_s, q, side="right").astype(np.int64) - 1
        np.minimum(k, self.t_s.size - 2, out=k)
        t0 = self.t_s[k]
        h: ArrayFloat = self.t_s[k + 1] - t0
        s: ArrayFloat = (q - t0) / h
        return k, h, s

    def position(self, times: ArraySeconds) -> ArrayKilometers:
        """
        Cubic Hermite position at each query time, shape `(m, 3)` km (a scalar query gives `(1, 3)`).
        Exactly the tabulated position at a node. Error bound (E0) of the module docstring.
        """
        k, h, s = self._locate(times)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = (s3 - 2.0 * s2 + s) * h
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = (s3 - s2) * h
        p, v = self.position_km, self.velocity_km_s
        out: ArrayKilometers = (h00[:, None] * p[k] + h10[:, None] * v[k]
                                + h01[:, None] * p[k + 1] + h11[:, None] * v[k + 1])
        return out

    def velocity(self, times: ArraySeconds) -> ArrayFloat:
        """
        Derivative of the Hermite position, shape `(m, 3)` km/s. Exactly the tabulated velocity at a
        node; continuous across nodes. Error bound (E1) of the module docstring.
        """
        k, h, s = self._locate(times)
        s2 = s * s
        g = (6.0 * s2 - 6.0 * s) / h          # dH00/dt = -dH01/dt
        d10 = 3.0 * s2 - 4.0 * s + 1.0         # d(H10 h)/dt
        d11 = 3.0 * s2 - 2.0 * s               # d(H11 h)/dt
        p, v = self.position_km, self.velocity_km_s
        out: ArrayFloat = g[:, None] * (p[k] - p[k + 1]) + d10[:, None] * v[k] + d11[:, None] * v[k + 1]
        return out

    def error_estimate_km(self) -> ArrayFloat:
        """
        Estimated (E0) for each interval, shape `(n - 1,)` km: `h_k^4 M_k / 384` with `M_k` the norm of
        the per-component max `|x''''|` estimated as `6 x` the third divided difference of the
        tabulated velocities over the (one or two) four-node windows containing the interval. Needs
        `n >= 4`. An `O(h)`-accurate estimate of the bound, not a rigorous bound.
        """
        t, v = self.t_s, self.velocity_km_s
        n = t.size
        if n < 4:
            raise ValueError(f"table '{self.name}': error_estimate_km needs at least 4 nodes, has {n}")
        d1 = (v[1:] - v[:-1]) / (t[1:] - t[:-1])[:, None]
        d2 = (d1[1:] - d1[:-1]) / (t[2:] - t[:-2])[:, None]
        d3 = (d2[1:] - d2[:-1]) / (t[3:] - t[:-3])[:, None]          # (n - 3, 3): window j = nodes j..j+3
        fourth = 6.0 * np.abs(d3)
        # Interval k lies inside windows k - 1 and k (where they exist).
        m = np.zeros((n - 1, 3), dtype=np.float64)
        m[: n - 3] = fourth
        m[1: n - 2] = np.maximum(m[1: n - 2], fourth)
        m[n - 2] = fourth[n - 4]
        h = np.diff(t)
        out: ArrayFloat = hermite_position_bound(h, np.sqrt(np.einsum("ij,ij->i", m, m)))
        return out


# ==================================================================================================
# Registration memo
# ==================================================================================================

_TABLES: Dict[float, EphemerisTable] = {}


def ephemeris_key(table: EphemerisTable) -> float:
    """
    The table's key: an integer in `[1, 2^52)`, as a float, from a BLAKE2b digest of name, centre,
    times, positions and velocities. Deterministic across processes and platforms (the arrays are
    hashed as little-endian float64).
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(table.name.encode("utf-8"))
    digest.update(b"\x00" + (table.centre or "").encode("utf-8") + b"\x00")
    for arr in (table.t_s, table.position_km, table.velocity_km_s):
        digest.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    value = int.from_bytes(digest.digest(), "little") >> (64 - _KEY_BITS)
    return float(value if value != 0 else 1)


def _same_content(a: EphemerisTable, b: EphemerisTable) -> bool:
    return (a.name == b.name and a.centre == b.centre and np.array_equal(a.t_s, b.t_s)
            and np.array_equal(a.position_km, b.position_km)
            and np.array_equal(a.velocity_km_s, b.velocity_km_s))


def register_ephemeris(table: EphemerisTable) -> float:
    """
    Register `table` for use by `"ephemeris_third_body"` and return its key. **Configuration time
    only.** Idempotent: the same content always gives the same key. Raises `RuntimeError` on a digest
    collision with different content, never overwriting. See the module docstring for the lifetime.
    """
    key = ephemeris_key(table)
    held = _TABLES.get(key)
    if held is None:
        _TABLES[key] = table
    elif held is not table and not _same_content(held, table):
        raise RuntimeError(
            f"ephemeris key collision: '{table.name}' and the registered '{held.name}' share key "
            f"{key!r} with different content")
    return key


def is_registered(key: float) -> bool:
    """True if `key` names a registered table."""
    return float(key) in _TABLES


def registered_ephemeris(key: float) -> EphemerisTable:
    """The step-time lookup: the registered table, or `LookupError`. Never registers anything."""
    try:
        return _TABLES[float(key)]
    except KeyError:
        raise LookupError(
            f"no ephemeris table is registered under key {key!r}. Tables are registered at "
            f"configuration time: ephemeris.register_ephemeris(table) or ephemeris_coefficients(...), "
            f"then enable '{EPHEMERIS_MODEL}' through Simulation.enable_force_model.") from None


def ephemeris_coefficients(
    perturbers: Sequence[Tuple[EphemerisTable, float]], *, epoch_s: float = 0.0,
) -> Dict[str, float]:
    """
    Register each table and return the keyword set for `enable_force_model(EPHEMERIS_MODEL, ...)` or a
    `sweep.ForceModelSpec`: `{"epoch_s": epoch_s, "table_1": key, "mu_1": mu, ...}`, one pair per
    `(table, mu)` in order. At most `MAX_PERTURBERS`. Unused slots are left out (they read 0).
    """
    if len(perturbers) > MAX_PERTURBERS:
        raise ValueError(f"at most {MAX_PERTURBERS} perturbers per body, got {len(perturbers)}")
    out: Dict[str, float] = {"epoch_s": float(epoch_s)}
    for k, (table, mu) in enumerate(perturbers):
        out[EPHEMERIS_PARAM_NAMES[_key_col(k)]] = register_ephemeris(table)
        out[EPHEMERIS_PARAM_NAMES[_mu_col(k)]] = float(mu)
    return out


# ==================================================================================================
# Configuration-time checks
# ==================================================================================================

def _reject_barycentre_parents(sim: "Simulation", bodies: NDArray[np.int64]) -> None:
    """`validate_bodies` hook: a barycentre is not where any table is centred."""
    offending = barycentre_parented(sim.is_system, sim.parent_indices, bodies)
    if offending.size > 0:
        raise ValueError(
            f"force model '{EPHEMERIS_MODEL}': slot(s) {offending.tolist()} have a barycentre as their "
            f"Keplerian parent; an ephemeris table is centred on a body, and the perturbation is taken "
            f"relative to the parent. See ephemeris.py's module docstring.")


def _validate_ephemeris_coefficients(
    sim: "Simulation", bodies: NDArray[np.int64], coefficients: Mapping[str, float],
) -> None:
    """`validate_coefficients` hook. See "Refusals" in the module docstring for each rule."""
    for key, value in coefficients.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"force model '{EPHEMERIS_MODEL}': {key}={value!r} is not finite")

    stored = sim.force_model_params.get(EPHEMERIS_MODEL)

    def effective(col: int) -> ArrayFloat:
        name = EPHEMERIS_PARAM_NAMES[col]
        if name in coefficients:
            return np.full(bodies.shape, float(coefficients[name]), dtype=np.float64)
        if stored is None:
            return np.zeros(bodies.shape, dtype=np.float64)
        column: ArrayFloat = stored[bodies, col]
        return column

    slot_to_name = {slot: name for name, slot in sim.name_to_index.items()}
    start = float(sim.t) + effective(_EPOCH_COL)
    seen: list[ArrayFloat] = []
    for k in range(MAX_PERTURBERS):
        keys, mus = effective(_key_col(k)), effective(_mu_col(k))
        label = f"slot {k + 1} ({EPHEMERIS_PARAM_NAMES[_key_col(k)]}, {EPHEMERIS_PARAM_NAMES[_mu_col(k)]})"
        if np.any(mus < 0.0):
            raise ValueError(f"force model '{EPHEMERIS_MODEL}': {label} has mu < 0 on slot(s) "
                             f"{bodies[mus < 0.0].tolist()}")
        bad_key = (keys != np.floor(keys)) | (keys < 0.0) | (keys >= 2.0 ** _KEY_BITS)
        if np.any(bad_key):
            raise ValueError(f"force model '{EPHEMERIS_MODEL}': {label} key {keys[bad_key][0]!r} is not "
                             f"an ephemeris key (use register_ephemeris / ephemeris_coefficients)")
        live = mus > 0.0
        empty = live & (keys == 0.0)
        if np.any(empty):
            raise ValueError(
                f"force model '{EPHEMERIS_MODEL}': {label} has mu > 0 and no table on slot(s) "
                f"{bodies[empty].tolist()}")
        for key in np.unique(keys[keys != 0.0]):
            try:
                table = registered_ephemeris(float(key))
            except LookupError as exc:
                raise ValueError(f"force model '{EPHEMERIS_MODEL}': {label}: {exc}") from None
            rows = live & (keys == key)
            if not np.any(rows):
                continue
            if table.centre is not None:
                centre_slot = sim.name_to_index.get(table.centre)
                parents = sim.parent_indices[bodies[rows]]
                wrong = bodies[rows][parents != (-1 if centre_slot is None else centre_slot)]
                if wrong.size > 0:
                    raise ValueError(
                        f"force model '{EPHEMERIS_MODEL}': table '{table.name}' is centred on "
                        f"'{table.centre}', but slot(s) {wrong.tolist()} have parent(s) "
                        f"{[slot_to_name.get(int(p), int(p)) for p in sim.parent_indices[wrong]]}; the "
                        f"perturbation is relative to the parent, so the table must be centred on it")
            if not table.covers(start[rows]):
                raise ValueError(
                    f"force model '{EPHEMERIS_MODEL}': table '{table.name}' covers "
                    f"[{table.t_min!r}, {table.t_max!r}] s but the next step starts at table time "
                    f"{start[rows].tolist()[:3]} (epoch_s + sim.t)")
        for earlier in seen:
            dup = live & (keys != 0.0) & (keys == earlier)
            if np.any(dup):
                raise ValueError(
                    f"force model '{EPHEMERIS_MODEL}': {label} repeats a table already live on slot(s) "
                    f"{bodies[dup].tolist()} - it would be counted twice")
        seen.append(np.where(live, keys, np.nan))


# ==================================================================================================
# Kernel
# ==================================================================================================

def _perturber_positions(keys: ArrayFloat, times: ArraySeconds) -> ArrayKilometers:
    """Position of each row's table at its own query time, `(m, 3)`. One lookup per distinct key."""
    first = keys[0]
    if np.all(keys == first):
        return registered_ephemeris(float(first)).position(times)
    out: ArrayKilometers = np.empty((keys.size, 3), dtype=np.float64)
    for key in np.unique(keys):
        sel = keys == key
        out[sel] = registered_ephemeris(float(key)).position(times[sel])
    return out


@register_force_model(
    EPHEMERIS_MODEL,
    param_names=EPHEMERIS_PARAM_NAMES,
    validate_bodies=_reject_barycentre_parents,
    validate_coefficients=_validate_ephemeris_coefficients,
    citation=(
        "Montenbruck & Gill, Satellite Orbits (2000), Sec. 3.3 (Eq. 3.37 from memory, unverified): "
        "direct minus indirect point-mass form, derived in thirdbody.py; perturbers from tabulated "
        "ephemerides by cubic Hermite interpolation (error h^4 max|x''''|/384) at each stage's time"
    ),
)
def ephemeris_third_body_kernel(
    indices: NDArray[np.int64],
    t: ScalarSeconds,
    state: NDArray[np.float64],
    mu_array: NDArray[np.float64],
    parent_indices: NDArray[np.int32],
    params: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """
    Add each body's tabulated perturbers' direct-minus-indirect pull to `out[indices]`, km/s^2.

    For body `i` with parent `P`, `r = state[i,:3] - state[P,:3]`, and each live slot `k`
    (`mu_k != 0`): `r_s = table_k.position(epoch_s + t)` (parent-centred by contract) and

        a += mu_k [ (r_s - r)/|r_s - r|^3 - r_s/|r_s|^3 ]

    **`t` is used** and must be each RK stage's own time. A slot with `mu == 0` is never looked up
    and adds nothing; an unregistered key raises `LookupError`; a time outside a table raises
    `ValueError`. `mu_array` is unused - the perturbers are not arena bodies.
    """
    if indices.size == 0:
        return
    parents = parent_indices[indices]
    r = state[indices, :3] - state[parents, :3]
    query = float(t) + params[indices, _EPOCH_COL]
    for k in range(MAX_PERTURBERS):
        mu = params[indices, _mu_col(k)]
        rows = np.flatnonzero(mu != 0.0)
        if rows.size == 0:
            continue
        r_s = _perturber_positions(params[indices[rows], _key_col(k)], query[rows])
        d = r_s - r[rows]                                     # perturber relative to the body
        d2 = np.einsum("ij,ij->i", d, d)
        s2 = np.einsum("ij,ij->i", r_s, r_s)
        # Exactly-zero separations contribute exactly zero, as in third_body.
        inv_d3 = np.divide(1.0, d2 * np.sqrt(d2), out=np.zeros_like(d2), where=d2 > 0.0)
        inv_s3 = np.divide(1.0, s2 * np.sqrt(s2), out=np.zeros_like(s2), where=s2 > 0.0)
        out[indices[rows]] += mu[rows, None] * (d * inv_d3[:, None] - r_s * inv_s3[:, None])
