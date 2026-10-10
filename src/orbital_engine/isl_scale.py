"""
Inter-satellite link contacts at constellation scale: the same windows and samples as
`isl.isl_contacts`, computed without visiting every pair, streamed sample by sample, and returned as
columns.

**Why.** `isl.py` evaluates all `N (N - 1) / 2` pairs at every sample. Its memory is bounded (it works
in blocks of pairs) but its work is not: 10,000 satellites is 5e7 pairs per sample. Most of them can
never see each other at that instant: they are farther apart than any link could reach. This module
only evaluates pairs that could be in view at the next sample, and it never needs the whole history
in memory, so a day of a mega-constellation becomes a stream over samples - from arrays, or from a
`history.HistorySink` recording on disk.

**The same answer, by construction.** On any constellation small enough to run both, this produces
*bit-identical* records to `isl.isl_contacts` (`tests/validation/test_isl_scale.py`): the same margin
(`geometry.segment_clearance`, the range limit), the same linear inverse interpolation of each edge in
the same floating-point order, the same range and range-rate interpolation onto rise and set, the
same closest approach (the first sampled minimum of range while in view), and the same ordering, by
`(body_a, body_b)` then time. Only the set of pairs evaluated differs, and the pruning below never
drops a pair that is in view at either end of an interval.

**Pruning, and why it is exact.** For the interval between samples `k` and `k + 1`, the candidate
pairs are those closer than

    L = min(max_range_km, L_los) + 2 v_max (t_{k+1} - t_k) + 1 km            at sample k,

where `L_los = 2 sqrt(r_max^2 - (R + h_graze)^2)` is the longest segment that can clear the grazing
sphere between two satellites no farther than `r_max` from the centre (two tangent segments), and
`v_max` the fastest satellite at either sample. A pair in view at `k` is closer than `min(max_range,
L_los)`, and a pair in view at `k + 1` was at most `2 v_max dt` farther at `k` (each end moved at most
`v_max dt`), so every pair whose margin is positive at either sample is a candidate, and every edge in
the interval is found. The 1 km absorbs rounding. Candidates come from `scipy.spatial.cKDTree.query_pairs`
when scipy is installed, and from a dense, blocked distance test otherwise (exact, just O(N^2)).

**Open windows** are carried from interval to interval as arrays sorted by pair key `a N + b`, with
their rise and their running closest approach, and closed when the margin falls through zero or
the samples end. Nothing scales with the number of samples except the output.

**Output.** `IslContactTable`, one numpy column per field of `isl.IslContact` (window edges, clip
flags, closest approach, and range and range rate at rise, peak and set; **range rate positive =
opening**). `to_contacts()` converts it to `isl.IslContact` records for small cases, and `save()`
writes `.npy` columns plus a JSON manifest (body names, the spec, units) for a downstream link-budget
or network project to ingest without importing this engine.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import kernels
from .access import ContactSample
from .attitude import Cone, boresights, cone_margin_km
from .geometry import segment_clearance
from .isl import IslContact, IslSpec, IslWindow

__all__ = ["IslContactTable", "isl_contact_table", "isl_contact_table_from_recording", "candidate_pairs",
           "Occulter", "LinkSpec", "link_contact_table", "link_contact_table_from_recording"]

# Candidate pairs evaluated per block: a few (P, 3) float temporaries each, so ~1e6 keeps a block to
# tens of MB however many pairs are in view.
_PAIR_BLOCK = 1_000_000


@dataclass(frozen=True)
class Occulter:
    """A sphere a link must clear: `body` by name, `radius_km`, and the grazing altitude `h_graze_km`
    the segment must keep above it (`geometry.segment_clearance`)."""

    body: str
    radius_km: float
    h_graze_km: float = 0.0


@dataclass(frozen=True)
class LinkSpec:
    """
    Link visibility between bodies anywhere - an Earth constellation and a lunar one, a probe drifting
    through cislunar space - generalising `isl.IslSpec` in two ways:

    - **Several occulters.** A link must clear *every* sphere in `occulters` (Earth *and* the Moon, each
      with its own radius and grazing altitude): the margin is the minimum of the clearances and the
      range margin. Positions are taken relative to `occulters[0]`, the frame origin, so with one
      occulter this is `isl.IslSpec`'s geometry exactly.
    - **Groups.** With `group_a` and `group_b` (disjoint, by name) only pairs `(a in A, b in B)` are
      evaluated: Earth satellites to lunar satellites, or one probe to a whole constellation, without
      the pairs inside each group. Without groups, every pair of the chosen bodies.

    `max_range_km` closes the link beyond that range. Pruning (see the module docstring) uses
    `min(max_range, L_los of each occulter)`; with no range limit and occulters far from the bodies,
    that bound is loose and most pairs are candidates, which is fine for small groups.

    `cones` gives a link end a field of view (`attitude.Cone`, by body name): the link then also needs
    each coned end to see the other inside its cone. Cones only remove visibility, so pruning stays
    exact. The reference bodies the attitudes name must be in the recording (or in the arrays API's
    `reference_states_km`).
    """

    occulters: Sequence[Occulter]
    max_range_km: Optional[float] = None
    group_a: Optional[Sequence[str]] = None
    group_b: Optional[Sequence[str]] = None
    cones: Mapping[str, Cone] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.occulters:
            raise ValueError("LinkSpec needs at least one occulter (the first is the frame origin).")
        if (self.group_a is None) != (self.group_b is None):
            raise ValueError("LinkSpec groups come in pairs: give both group_a and group_b, or neither.")
        if self.group_a is not None and self.group_b is not None:
            both = set(self.group_a) & set(self.group_b)
            if both:
                raise ValueError(f"LinkSpec groups must be disjoint; {sorted(both)} are in both.")
            occ = {o.body for o in self.occulters} & (set(self.group_a) | set(self.group_b))
            if occ:
                raise ValueError(f"{sorted(occ)} are occulters and cannot also be link ends.")


@dataclass(frozen=True)
class _Model:
    """What the scan needs from either spec: occulter spheres (index 0 is the frame origin) and grazing
    altitudes, the range limit, and the A/B split of the body list (`None`: all pairs)."""

    radius: Tuple[float, ...]
    graze: Tuple[float, ...]
    max_range: Optional[float]
    split: Optional[int]

    @staticmethod
    def of(spec: Union[IslSpec, LinkSpec], split: Optional[int] = None) -> "_Model":
        if isinstance(spec, IslSpec):
            return _Model((spec.body_radius_km,), (spec.h_graze_km,), spec.max_range_km, None)
        return _Model(tuple(o.radius_km for o in spec.occulters), tuple(o.h_graze_km for o in spec.occulters),
                      spec.max_range_km, split)


@dataclass
class IslContactTable:
    """The ISL contact dataset as columns, one row per window, ordered by `(body_a, body_b, rise_s)`.
    `body_a < body_b` index `names`. Field meanings are `isl.IslWindow`'s and `isl.IslContact`'s."""

    names: List[str]
    body_a: NDArray[np.int64]
    body_b: NDArray[np.int64]
    rise_s: NDArray[np.float64]
    set_s: NDArray[np.float64]
    duration_s: NDArray[np.float64]
    min_range_km: NDArray[np.float64]
    peak_time_s: NDArray[np.float64]
    rise_clipped: NDArray[np.bool_]
    set_clipped: NDArray[np.bool_]
    rise_range_km: NDArray[np.float64]
    rise_rate_km_s: NDArray[np.float64]
    peak_rate_km_s: NDArray[np.float64]
    set_range_km: NDArray[np.float64]
    set_rate_km_s: NDArray[np.float64]

    def __len__(self) -> int:
        return int(self.body_a.size)

    def to_contacts(self) -> List[IslContact]:
        """`isl.IslContact` records, for comparison with the dense path. Small tables only."""
        out = []
        for k in range(len(self)):
            w = IslWindow(
                body_a=int(self.body_a[k]), body_b=int(self.body_b[k]), rise_s=float(self.rise_s[k]),
                set_s=float(self.set_s[k]), duration_s=float(self.duration_s[k]),
                min_range_km=float(self.min_range_km[k]), peak_time_s=float(self.peak_time_s[k]),
                rise_clipped=bool(self.rise_clipped[k]), set_clipped=bool(self.set_clipped[k]))
            out.append(IslContact(
                window=w,
                rise=ContactSample(w.rise_s, float(self.rise_range_km[k]), float(self.rise_rate_km_s[k])),
                peak=ContactSample(w.peak_time_s, w.min_range_km, float(self.peak_rate_km_s[k])),
                set=ContactSample(w.set_s, float(self.set_range_km[k]), float(self.set_rate_km_s[k]))))
        return out

    def save(self, directory: Union[str, Path], spec: Union[IslSpec, LinkSpec]) -> None:
        """`<field>.npy` per column plus `manifest.json` (names, spec, units, conventions)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        columns = [f.name for f in fields(self) if f.name != "names"]
        for name in columns:
            np.save(directory / f"{name}.npy", getattr(self, name))
        manifest = {
            "dataset": "isl_contacts", "format_version": 1, "rows": len(self), "names": self.names,
            "columns": columns,
            "spec": ({"central_body": spec.central_body, "body_radius_km": spec.body_radius_km,
                      "h_graze_km": spec.h_graze_km, "max_range_km": spec.max_range_km}
                     if isinstance(spec, IslSpec) else
                     {"occulters": [{"body": o.body, "radius_km": o.radius_km, "h_graze_km": o.h_graze_km}
                                    for o in spec.occulters],
                      "frame_origin": spec.occulters[0].body, "max_range_km": spec.max_range_km,
                      "group_a": list(spec.group_a) if spec.group_a is not None else None,
                      "group_b": list(spec.group_b) if spec.group_b is not None else None}),
            "units": {"time": "s (simulation time)", "range": "km", "range_rate": "km/s"},
            "conventions": {"pair": "body_a < body_b, indices into names", "range_rate": "positive = opening",
                            "edges": "linearly interpolated on the link margin; clipped = grid endpoint",
                            "peak": "first sampled minimum of range while in view"},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))


def candidate_pairs(pos: NDArray[np.float64], radius_km: float,
                    split: Optional[int] = None) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Every pair `(a < b)` of `(N, 3)` positions closer than `radius_km`, sorted by `a N + b`; with
    `split`, only pairs with `a < split <= b` (group A is `[0, split)`, group B the rest)."""
    n = pos.shape[0]
    if split is not None:
        return _bipartite_pairs(pos, radius_km, split)
    try:
        from scipy.spatial import cKDTree
    except ImportError:                                             # exact, O(N^2), blocked
        ia_parts, ib_parts = [], []
        step = max(1, _PAIR_BLOCK // max(1, n))
        for s in range(0, n, step):
            d = np.linalg.norm(pos[s:s + step, None, :] - pos[None, :, :], axis=2)
            a, b = np.nonzero(d < radius_km)
            a = a + s
            keep = a < b
            ia_parts.append(a[keep])
            ib_parts.append(b[keep])
        ia = np.concatenate(ia_parts).astype(np.int64) if ia_parts else np.empty(0, np.int64)
        ib = np.concatenate(ib_parts).astype(np.int64) if ib_parts else np.empty(0, np.int64)
    else:
        pairs = cKDTree(pos).query_pairs(radius_km, output_type="ndarray")
        lo = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        hi = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
        ia, ib = lo, hi
    order = np.argsort(ia * n + ib, kind="stable")
    out_a: NDArray[np.int64] = ia[order]
    out_b: NDArray[np.int64] = ib[order]
    return out_a, out_b


def _bipartite_pairs(pos: NDArray[np.float64], radius_km: float,
                     split: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    n = pos.shape[0]
    a_pos, b_pos = pos[:split], pos[split:]
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        d = np.linalg.norm(a_pos[:, None, :] - b_pos[None, :, :], axis=2)
        ia, jb = np.nonzero(d < radius_km)
    else:
        if math.isinf(radius_km):
            ia, jb = np.nonzero(np.ones((split, n - split), dtype=bool))
        else:
            m = cKDTree(a_pos).sparse_distance_matrix(cKDTree(b_pos), radius_km, output_type="ndarray")
            keep = m["v"] < radius_km
            ia, jb = m["i"][keep], m["j"][keep]
    a = np.asarray(ia, dtype=np.int64)
    b = np.asarray(jb, dtype=np.int64) + split
    order = np.argsort(a * n + b, kind="stable")
    out_a: NDArray[np.int64] = a[order]
    out_b: NDArray[np.int64] = b[order]
    return out_a, out_b


def _geometry(pos: NDArray[np.float64], vel: NDArray[np.float64], ia: NDArray[np.int64], ib: NDArray[np.int64],
              model: _Model, occ: Optional[NDArray[np.float64]] = None,
              bore: Optional[NDArray[np.float64]] = None, half: Optional[NDArray[np.float64]] = None,
              compiled: Optional[bool] = None,
              ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Margin, range and range rate for pairs at one sample, by `_geometry_numpy` (the definition) or its
    compiled twin `kernels.isl_pair_geometry`. `compiled=None` takes `kernels.NUMBA_AVAILABLE`; without
    numba the NumPy path always runs (the kernel interpreted would be slower)."""
    if not (kernels.NUMBA_AVAILABLE if compiled is None else compiled):
        return _geometry_numpy(pos, vel, ia, ib, model, occ, bore, half)
    p = np.asarray(pos, dtype=np.float64)
    v = np.asarray(vel, dtype=np.float64)
    n_occ = len(model.radius)
    if n_occ > 1:
        assert occ is not None
        centres = np.asarray(occ, dtype=np.float64)
    else:
        centres = np.zeros((1, 3))
    has_cone = bore is not None and half is not None
    margin = np.empty(ia.size)
    ranges = np.empty(ia.size)
    rates = np.empty(ia.size)
    kernels.isl_pair_geometry(
        p, v, ia, ib, np.asarray(model.radius, dtype=np.float64), np.asarray(model.graze, dtype=np.float64),
        centres, model.max_range is not None, 0.0 if model.max_range is None else float(model.max_range),
        has_cone, bore if bore is not None and has_cone else np.zeros((1, 3)),
        half if half is not None and has_cone else np.zeros(1), margin, ranges, rates)
    return margin, ranges, rates


def _geometry_numpy(pos: NDArray[np.float64], vel: NDArray[np.float64], ia: NDArray[np.int64],
                    ib: NDArray[np.int64], model: _Model, occ: Optional[NDArray[np.float64]] = None,
                    bore: Optional[NDArray[np.float64]] = None, half: Optional[NDArray[np.float64]] = None,
                    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Margin, range and range rate for pairs at one sample - `isl._pair_block`'s arithmetic, per pair.
    Occulter 0 is the frame origin (its clearance is computed on the positions as given, exactly as
    `isl.py` does); `occ` holds the centres of occulters 1.. in that frame, and each adds a clearance."""
    r_a, r_b = pos[ia], pos[ib]
    rel = r_b - r_a
    ranges: NDArray[np.float64] = np.sqrt(np.sum(rel * rel, axis=-1))
    margin = segment_clearance(r_a, r_b, body_radius_km=model.radius[0], h_graze_km=model.graze[0])
    for k in range(1, len(model.radius)):
        assert occ is not None
        c = occ[k - 1]
        margin = np.minimum(margin, segment_clearance(r_a - c, r_b - c, body_radius_km=model.radius[k],
                                                      h_graze_km=model.graze[k]))
    if model.max_range is not None:
        margin = np.minimum(margin, float(model.max_range) - ranges)
    if bore is not None and half is not None:                         # fields of view, either end
        unit = rel / np.where(ranges > 0.0, ranges, 1.0)[:, None]
        for end, sign in ((ia, 1.0), (ib, -1.0)):
            coned = ~np.isnan(half[end])
            if coned.any():
                margin = margin.copy()
                margin[coned] = np.minimum(margin[coned], cone_margin_km(
                    bore[end[coned]], sign * unit[coned], ranges[coned], half[end[coned]]))
    rel_v = vel[ib] - vel[ia]
    rates: NDArray[np.float64] = np.sum(rel * rel_v, axis=-1) / np.where(ranges > 0.0, ranges, 1.0)
    return np.asarray(margin, dtype=np.float64), ranges, rates


class _Scanner:
    """The streaming state: the previous sample and the open windows. See the module docstring."""

    _OPEN = ("key", "rise", "rise_clipped", "rise_range", "rise_rate", "min_range", "peak_t", "peak_rate")

    def __init__(self, n: int, model: _Model, compiled: Optional[bool] = None) -> None:
        self.n = n
        self.model = model
        self.compiled = compiled
        self.prev: Optional[Tuple[float, NDArray[np.float64], NDArray[np.float64],
                                  Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]] = None
        self.half: Optional[NDArray[np.float64]] = None              # cone half-angles per body, NaN = none
        self.open: Dict[str, NDArray[Any]] = {k: np.empty(0) for k in self._OPEN}
        self.open["key"] = np.empty(0, dtype=np.int64)
        self.closed: List[Dict[str, NDArray[Any]]] = []
        self.samples = 0

    def _radius(self, p0: NDArray[np.float64], v0: NDArray[np.float64], p1: NDArray[np.float64],
                v1: NDArray[np.float64], dt: float, o0: Optional[NDArray[np.float64]] = None,
                o1: Optional[NDArray[np.float64]] = None) -> float:
        # A visible link clears every occulter, so it is no longer than the longest segment that can clear
        # each one: the bound is the minimum over occulters (occulter 0 sits at the origin).
        reach = math.inf
        for k, (radius, graze) in enumerate(zip(self.model.radius, self.model.graze)):
            c0 = np.zeros(3) if k == 0 or o0 is None else o0[k - 1]
            c1 = np.zeros(3) if k == 0 or o1 is None else o1[k - 1]
            rg = radius + graze
            r_max = float(max(np.max(np.linalg.norm(p0 - c0, axis=1)), np.max(np.linalg.norm(p1 - c1, axis=1))))
            reach = min(reach, 2.0 * math.sqrt(max(r_max * r_max - rg * rg, 0.0)))
        if self.model.max_range is not None:
            reach = min(reach, float(self.model.max_range))
        v_max = float(max(np.max(np.linalg.norm(v0, axis=1)), np.max(np.linalg.norm(v1, axis=1))))
        return reach + 2.0 * v_max * dt + 1.0

    def feed(self, t: float, pos: NDArray[np.float64], vel: NDArray[np.float64],
             occ: Optional[NDArray[np.float64]] = None, bore: Optional[NDArray[np.float64]] = None) -> None:
        if self.prev is None:
            self.prev = (t, pos, vel, occ, bore)
            self.samples = 1
            return
        t0, p0, v0, o0, b0 = self.prev
        if not t > t0:
            raise ValueError("ISL samples must be strictly increasing in time.")
        ca, cb = candidate_pairs(p0, self._radius(p0, v0, pos, vel, t - t0, o0, occ), self.model.split)
        for s in range(0, ca.size, _PAIR_BLOCK):
            self._interval(t0, p0, v0, o0, b0, t, pos, vel, occ, bore, ca[s:s + _PAIR_BLOCK],
                           cb[s:s + _PAIR_BLOCK], first=self.samples == 1)
        self.prev = (t, pos, vel, occ, bore)
        self.samples += 1

    def _interval(self, t0: float, p0: NDArray[np.float64], v0: NDArray[np.float64],
                  o0: Optional[NDArray[np.float64]], b0: Optional[NDArray[np.float64]], t1: float,
                  p1: NDArray[np.float64], v1: NDArray[np.float64], o1: Optional[NDArray[np.float64]],
                  b1: Optional[NDArray[np.float64]], ia: NDArray[np.int64], ib: NDArray[np.int64],
                  first: bool) -> None:
        m0, r0, q0 = _geometry(p0, v0, ia, ib, self.model, o0, b0, self.half, self.compiled)
        m1, r1, q1 = _geometry(p1, v1, ia, ib, self.model, o1, b1, self.half, self.compiled)
        key = ia * self.n + ib
        in0, in1 = m0 > 0.0, m1 > 0.0

        if first:                                                     # windows open at the grid start
            self._open(key[in0], np.full(int(in0.sum()), t0), np.ones(int(in0.sum()), bool),
                       r0[in0], q0[in0], r0[in0], np.full(int(in0.sum()), t0), q0[in0])

        rise = ~in0 & in1                                             # isl._extract / _edge_value order
        if rise.any():
            f0, f1 = m0[rise], m1[rise]
            t_r = t0 + (t1 - t0) * (-f0) / (f1 - f0)
            span = t0 - t1                                            # t_out - t_in, inside = k + 1
            w = (t_r - t1) / span
            self._open(key[rise], t_r, np.zeros(int(rise.sum()), bool),
                       (1.0 - w) * r1[rise] + w * r0[rise], (1.0 - w) * q1[rise] + w * q0[rise],
                       np.full(int(rise.sum()), np.inf), np.zeros(int(rise.sum())), np.zeros(int(rise.sum())))

        # Closest approach: every in-view sample k + 1 updates the running first minimum.
        if in1.any():
            pos_open = self._locate(key[in1])
            better = r1[in1] < self.open["min_range"][pos_open]
            idx = pos_open[better]
            self.open["min_range"][idx] = r1[in1][better]
            self.open["peak_t"][idx] = t1
            self.open["peak_rate"][idx] = q1[in1][better]

        fall = in0 & ~in1
        if fall.any():
            g0, g1 = m0[fall], m1[fall]
            t_s = t0 + (t1 - t0) * (-g0) / (g1 - g0)
            w = (t_s - t0) / (t1 - t0)                                # inside = k, outside = k + 1
            self._close(key[fall], t_s, np.zeros(int(fall.sum()), bool),
                        (1.0 - w) * r0[fall] + w * r1[fall], (1.0 - w) * q0[fall] + w * q1[fall])

    def _open(self, key: NDArray[np.int64], rise: NDArray[np.float64], clipped: NDArray[np.bool_],
              rise_range: NDArray[np.float64], rise_rate: NDArray[np.float64], min_range: NDArray[np.float64],
              peak_t: NDArray[np.float64], peak_rate: NDArray[np.float64]) -> None:
        if key.size == 0:
            return
        new: Dict[str, NDArray[Any]] = {"key": key, "rise": rise, "rise_clipped": clipped, "rise_range": rise_range,
               "rise_rate": rise_rate, "min_range": min_range, "peak_t": peak_t, "peak_rate": peak_rate}
        merged = {k: np.concatenate([self.open[k], new[k]]) for k in self._OPEN}
        order = np.argsort(merged["key"], kind="stable")
        self.open = {k: v[order] for k, v in merged.items()}

    def _locate(self, key: NDArray[np.int64]) -> NDArray[np.int64]:
        """Rows of the open windows for `key`, all of which must be open: a pair in view whose window is
        not open means a pair in view was pruned at its rise, which the candidate radius rules out."""
        at: NDArray[np.int64] = np.searchsorted(self.open["key"], key).astype(np.int64)
        if bool(np.any(at >= self.open["key"].size)) or not np.array_equal(self.open["key"][at], key):
            raise RuntimeError("ISL scan: a pair in view has no open window (it was never opened) - a pair "
                               "in view was pruned. The candidate radius must cover every pair in view at "
                               "either end of an interval.")
        return at

    def _close(self, key: NDArray[np.int64], set_t: NDArray[np.float64], clipped: NDArray[np.bool_],
               set_range: NDArray[np.float64], set_rate: NDArray[np.float64]) -> None:
        at = self._locate(key)
        rows: Dict[str, NDArray[Any]] = {k: v[at] for k, v in self.open.items()}
        rows.update({"set": set_t, "set_clipped": clipped, "set_range": set_range, "set_rate": set_rate})
        self.closed.append(rows)
        keep = np.ones(self.open["key"].size, dtype=bool)
        keep[at] = False
        self.open = {k: v[keep] for k, v in self.open.items()}

    def finish(self, names: Sequence[str]) -> IslContactTable:
        if self.samples < 2:
            raise ValueError("ISL windows need at least two samples to bracket a crossing.")
        assert self.prev is not None
        t_end, p_end, v_end, o_end, b_end = self.prev
        if self.open["key"].size:                                     # still in view at the last sample
            keys: NDArray[np.int64] = self.open["key"].astype(np.int64)
            a, b = keys // self.n, keys % self.n
            _, r_end, q_end = _geometry(p_end, v_end, a.astype(np.int64), b.astype(np.int64), self.model, o_end,
                                        b_end, self.half, self.compiled)
            self._close(keys, np.full(keys.size, t_end), np.ones(keys.size, bool), r_end, q_end)
        rows: Dict[str, NDArray[Any]]
        if self.closed:
            rows = {k: np.concatenate([c[k] for c in self.closed]) for k in self.closed[0]}
        else:
            rows = {k: np.empty(0) for k in (*self._OPEN, "set", "set_clipped", "set_range", "set_rate")}
            rows["key"] = np.empty(0, dtype=np.int64)
        key = rows["key"].astype(np.int64)
        order = np.lexsort((rows["rise"], key))
        a = (key // self.n)[order]
        b = (key % self.n)[order]
        rise, set_t = rows["rise"][order], rows["set"][order]
        return IslContactTable(
            names=list(names), body_a=a.astype(np.int64), body_b=b.astype(np.int64),
            rise_s=rise.astype(np.float64), set_s=set_t.astype(np.float64),
            duration_s=(set_t - rise).astype(np.float64),
            min_range_km=rows["min_range"][order].astype(np.float64),
            peak_time_s=rows["peak_t"][order].astype(np.float64),
            rise_clipped=rows["rise_clipped"][order].astype(bool), set_clipped=rows["set_clipped"][order].astype(bool),
            rise_range_km=rows["rise_range"][order].astype(np.float64),
            rise_rate_km_s=rows["rise_rate"][order].astype(np.float64),
            peak_rate_km_s=rows["peak_rate"][order].astype(np.float64),
            set_range_km=rows["set_range"][order].astype(np.float64),
            set_rate_km_s=rows["set_rate"][order].astype(np.float64))


def _scan(samples: Iterable[Tuple[float, NDArray[np.float64], NDArray[np.float64]]], n: int,
          names: Sequence[str], spec: IslSpec, compiled: Optional[bool] = None) -> IslContactTable:
    scanner = _Scanner(n, _Model.of(spec), compiled)
    for t, pos, vel in samples:
        scanner.feed(float(t), pos, vel)
    return scanner.finish(names)


_Sample = Tuple[float, NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]],
                Mapping[str, NDArray[np.float64]]]


def _link_scan(samples: Iterable[_Sample], names: Sequence[str], spec: LinkSpec,
               compiled: Optional[bool] = None) -> IslContactTable:
    """Each sample: `(t, pos, vel, occulter centres 1.., {reference body: (6,) state})`, all relative to
    `spec.occulters[0]`."""
    split = len(spec.group_a) if spec.group_a is not None else None
    scanner = _Scanner(len(names), _Model.of(spec, split), compiled)
    unknown = sorted(set(spec.cones) - set(names))
    if unknown:
        raise KeyError(f"cones given for {unknown}, which are not link ends")
    if spec.cones:
        scanner.half = np.full(len(names), np.nan)
        for name, cone in spec.cones.items():
            scanner.half[list(names).index(name)] = math.radians(cone.half_angle_deg)
    coned = [(list(names).index(n), c.attitude) for n, c in spec.cones.items()]
    for t, pos, vel, occ, refs in samples:
        bore: Optional[NDArray[np.float64]] = None
        if coned:
            bore = np.full((len(names), 3), np.nan)
            for k, att in coned:
                ref = refs.get(att.reference) if att.reference is not None else None
                bore[k] = boresights(att, pos[k], vel[k], None if ref is None else ref[:3],
                                     None if ref is None else ref[3:])[0]
        scanner.feed(float(t), pos, vel, occ, bore)
    return scanner.finish(names)


def _references(spec: LinkSpec) -> List[str]:
    return sorted({c.attitude.reference for c in spec.cones.values() if c.attitude.reference is not None})


def _link_names(spec: LinkSpec, available: Sequence[str], bodies: Optional[Sequence[str]]) -> List[str]:
    if spec.group_a is not None and spec.group_b is not None:
        names = list(spec.group_a) + list(spec.group_b)
    else:
        occulters = {o.body for o in spec.occulters}
        names = list(bodies) if bodies is not None else [n for n in available if n not in occulters]
    missing = [n for n in names + [o.body for o in spec.occulters] if n not in available]
    if missing:
        raise KeyError(f"no bodies named {missing}")
    return names


def link_contact_table(positions_km: NDArray[np.float64], velocities_km_s: NDArray[np.float64],
                       times_s: NDArray[np.float64], names: Sequence[str],
                       occulter_positions_km: Mapping[str, NDArray[np.float64]], spec: LinkSpec,
                       reference_states_km: Optional[Mapping[str, NDArray[np.float64]]] = None,
                       compiled: Optional[bool] = None) -> IslContactTable:
    """
    Link contacts for `(T, N, 3)` positions and velocities of the bodies `names` in any common inertial
    frame, with each occulter's `(T, 3)` positions in the same frame (`occulter_positions_km` by name).
    `compiled` picks the per-pair geometry: the numba twin (`None`: when numba is installed) or NumPy.
    Positions are re-expressed relative to `spec.occulters[0]`. With groups, `names` must hold both
    groups; the table's body list is `group_a + group_b`. `reference_states_km` gives the `(T, 6)` states
    of the bodies the cones' attitudes name, in the same frame.
    """
    pos_all = np.asarray(positions_km, dtype=np.float64)
    vel_all = np.asarray(velocities_km_s, dtype=np.float64)
    order = _link_names(spec, list(names) + [o.body for o in spec.occulters], None)
    if spec.group_a is None:
        order = [n for n in names if n not in {o.body for o in spec.occulters}]
    cols = np.asarray([list(names).index(n) for n in order], dtype=np.int64)
    origin = np.asarray(occulter_positions_km[spec.occulters[0].body], dtype=np.float64)
    others = [np.asarray(occulter_positions_km[o.body], dtype=np.float64) for o in spec.occulters[1:]]

    refs_in = {n: np.asarray(v, dtype=np.float64) for n, v in (reference_states_km or {}).items()}
    missing = [n for n in _references(spec) if n not in refs_in]
    if missing:
        raise KeyError(f"the cones' attitudes name {missing}; give their states in reference_states_km")

    def samples() -> Iterator[_Sample]:
        for k, t in enumerate(np.asarray(times_s, dtype=np.float64)):
            occ = np.stack([o[k] - origin[k] for o in others]) if others else None
            refs = {n: np.concatenate([v[k, :3] - origin[k], v[k, 3:]]) for n, v in refs_in.items()}
            yield float(t), pos_all[k, cols] - origin[k], np.ascontiguousarray(vel_all[k, cols]), occ, refs
    return _link_scan(samples(), order, spec, compiled)


def link_contact_table_from_recording(directory: Union[str, Path], spec: LinkSpec,
                                      bodies: Optional[Sequence[str]] = None,
                                      compiled: Optional[bool] = None) -> IslContactTable:
    """
    Link contacts from a `history.HistorySink` recording (`compiled` as in `link_contact_table`), streamed one chunk at a time. The recording
    must include every occulter. Positions and velocities are taken relative to `spec.occulters[0]`.
    Without groups, `bodies` (default: every recorded body that is not an occulter) are paired among
    themselves.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    recorded: List[str] = manifest["names"]
    names = _link_names(spec, recorded, bodies)
    origin = recorded.index(spec.occulters[0].body)
    others = np.asarray([recorded.index(o.body) for o in spec.occulters[1:]], dtype=np.int64)
    cols = np.asarray([recorded.index(n) for n in names], dtype=np.int64)
    missing = [n for n in _references(spec) if n not in recorded]
    if missing:
        raise KeyError(f"the cones' attitudes name {missing}, which the recording does not include")
    ref_cols = {n: recorded.index(n) for n in _references(spec)}

    def samples() -> Iterator[_Sample]:
        for stem in manifest["chunks"]:
            t = np.load(directory / f"{stem}_t.npy")
            g = np.load(directory / f"{stem}_global.npy", mmap_mode="r")
            for k in range(t.size):
                snap = np.asarray(g[k], dtype=np.float64)
                rel = snap[cols] - snap[origin]
                occ = (snap[others, :3] - snap[origin, :3]) if others.size else None
                refs = {n: snap[j] - snap[origin] for n, j in ref_cols.items()}
                yield float(t[k]), np.ascontiguousarray(rel[:, :3]), np.ascontiguousarray(rel[:, 3:]), occ, refs
    return _link_scan(samples(), names, spec, compiled)


def isl_contact_table(positions_km: NDArray[np.float64], velocities_km_s: NDArray[np.float64],
                      times_s: NDArray[np.float64], spec: IslSpec,
                      names: Optional[Sequence[str]] = None,
                      compiled: Optional[bool] = None) -> IslContactTable:
    """
    The ISL contact dataset from `(T, N, 3)` **central-body-relative inertial** positions and velocities
    sampled at `times_s`: `isl.isl_contacts`'s records, bit for bit, as columns. `names` defaults to
    `body0 .. bodyN-1`. `compiled` picks the per-pair geometry (see `link_contact_table`).
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    vel = np.asarray(velocities_km_s, dtype=np.float64)
    n = pos.shape[1]
    names = list(names) if names is not None else [f"body{k}" for k in range(n)]
    return _scan(((float(t), pos[k], vel[k]) for k, t in enumerate(np.asarray(times_s, dtype=np.float64))),
                 n, names, spec, compiled)


def isl_contact_table_from_recording(directory: Union[str, Path], spec: IslSpec,
                                     bodies: Optional[Sequence[str]] = None,
                                     compiled: Optional[bool] = None) -> IslContactTable:
    """
    The ISL contact dataset from a `history.HistorySink` recording (`compiled` as in `isl_contact_table`), streamed one chunk at a time so a
    day of a mega-constellation never sits in memory. The recording must include `spec.central_body`;
    positions are taken relative to it. `bodies` (default: every recorded body except the central one)
    fixes the pair order.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    recorded: List[str] = manifest["names"]
    if spec.central_body not in recorded:
        raise KeyError(f"the recording does not include the central body {spec.central_body!r}")
    names = list(bodies) if bodies is not None else [n for n in recorded if n != spec.central_body]
    missing = [n for n in names if n not in recorded]
    if missing:
        raise KeyError(f"the recording has no bodies named {missing}")
    centre = recorded.index(spec.central_body)
    cols = np.asarray([recorded.index(n) for n in names], dtype=np.int64)

    def samples() -> Iterator[Tuple[float, NDArray[np.float64], NDArray[np.float64]]]:
        for stem in manifest["chunks"]:
            t = np.load(directory / f"{stem}_t.npy")
            g = np.load(directory / f"{stem}_global.npy", mmap_mode="r")
            for k in range(t.size):
                snap = np.asarray(g[k], dtype=np.float64)
                rel = snap[cols] - snap[centre]
                yield float(t[k]), np.ascontiguousarray(rel[:, :3]), np.ascontiguousarray(rel[:, 3:])

    return _scan(samples(), len(names), names, spec, compiled)
