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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .access import ContactSample
from .geometry import segment_clearance
from .isl import IslContact, IslSpec, IslWindow

__all__ = ["IslContactTable", "isl_contact_table", "isl_contact_table_from_recording", "candidate_pairs"]

# Candidate pairs evaluated per block: a few (P, 3) float temporaries each, so ~1e6 keeps a block to
# tens of MB however many pairs are in view.
_PAIR_BLOCK = 1_000_000


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

    def save(self, directory: Union[str, Path], spec: IslSpec) -> None:
        """`<field>.npy` per column plus `manifest.json` (names, spec, units, conventions)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        columns = [f.name for f in fields(self) if f.name != "names"]
        for name in columns:
            np.save(directory / f"{name}.npy", getattr(self, name))
        manifest = {
            "dataset": "isl_contacts", "format_version": 1, "rows": len(self), "names": self.names,
            "columns": columns,
            "spec": {"central_body": spec.central_body, "body_radius_km": spec.body_radius_km,
                     "h_graze_km": spec.h_graze_km, "max_range_km": spec.max_range_km},
            "units": {"time": "s (simulation time)", "range": "km", "range_rate": "km/s"},
            "conventions": {"pair": "body_a < body_b, indices into names", "range_rate": "positive = opening",
                            "edges": "linearly interpolated on the link margin; clipped = grid endpoint",
                            "peak": "first sampled minimum of range while in view"},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))


def candidate_pairs(pos: NDArray[np.float64], radius_km: float) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Every pair `(a < b)` of `(N, 3)` positions closer than `radius_km`, sorted by `a N + b`."""
    n = pos.shape[0]
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


def _geometry(pos: NDArray[np.float64], vel: NDArray[np.float64], ia: NDArray[np.int64], ib: NDArray[np.int64],
              spec: IslSpec) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Margin, range and range rate for pairs at one sample - `isl._pair_block`'s arithmetic, per pair."""
    r_a, r_b = pos[ia], pos[ib]
    rel = r_b - r_a
    ranges: NDArray[np.float64] = np.sqrt(np.sum(rel * rel, axis=-1))
    margin = segment_clearance(r_a, r_b, body_radius_km=spec.body_radius_km, h_graze_km=spec.h_graze_km)
    if spec.max_range_km is not None:
        margin = np.minimum(margin, float(spec.max_range_km) - ranges)
    rel_v = vel[ib] - vel[ia]
    rates: NDArray[np.float64] = np.sum(rel * rel_v, axis=-1) / np.where(ranges > 0.0, ranges, 1.0)
    return np.asarray(margin, dtype=np.float64), ranges, rates


class _Scanner:
    """The streaming state: the previous sample and the open windows. See the module docstring."""

    _OPEN = ("key", "rise", "rise_clipped", "rise_range", "rise_rate", "min_range", "peak_t", "peak_rate")

    def __init__(self, n: int, spec: IslSpec) -> None:
        self.n = n
        self.spec = spec
        self.prev: Optional[Tuple[float, NDArray[np.float64], NDArray[np.float64]]] = None
        self.open: Dict[str, NDArray[Any]] = {k: np.empty(0) for k in self._OPEN}
        self.open["key"] = np.empty(0, dtype=np.int64)
        self.closed: List[Dict[str, NDArray[Any]]] = []
        self.samples = 0

    def _radius(self, p0: NDArray[np.float64], v0: NDArray[np.float64], p1: NDArray[np.float64],
                v1: NDArray[np.float64], dt: float) -> float:
        rg = self.spec.body_radius_km + self.spec.h_graze_km
        r_max = float(max(np.max(np.linalg.norm(p0, axis=1)), np.max(np.linalg.norm(p1, axis=1))))
        reach = 2.0 * math.sqrt(max(r_max * r_max - rg * rg, 0.0))
        if self.spec.max_range_km is not None:
            reach = min(reach, float(self.spec.max_range_km))
        v_max = float(max(np.max(np.linalg.norm(v0, axis=1)), np.max(np.linalg.norm(v1, axis=1))))
        return reach + 2.0 * v_max * dt + 1.0

    def feed(self, t: float, pos: NDArray[np.float64], vel: NDArray[np.float64]) -> None:
        if self.prev is None:
            self.prev = (t, pos, vel)
            self.samples = 1
            return
        t0, p0, v0 = self.prev
        if not t > t0:
            raise ValueError("ISL samples must be strictly increasing in time.")
        ca, cb = candidate_pairs(p0, self._radius(p0, v0, pos, vel, t - t0))
        for s in range(0, ca.size, _PAIR_BLOCK):
            self._interval(t0, p0, v0, t, pos, vel, ca[s:s + _PAIR_BLOCK], cb[s:s + _PAIR_BLOCK], first=self.samples == 1)
        self.prev = (t, pos, vel)
        self.samples += 1

    def _interval(self, t0: float, p0: NDArray[np.float64], v0: NDArray[np.float64], t1: float,
                  p1: NDArray[np.float64], v1: NDArray[np.float64], ia: NDArray[np.int64], ib: NDArray[np.int64],
                  first: bool) -> None:
        m0, r0, q0 = _geometry(p0, v0, ia, ib, self.spec)
        m1, r1, q1 = _geometry(p1, v1, ia, ib, self.spec)
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
        t_end, p_end, v_end = self.prev
        if self.open["key"].size:                                     # still in view at the last sample
            keys: NDArray[np.int64] = self.open["key"].astype(np.int64)
            a, b = keys // self.n, keys % self.n
            _, r_end, q_end = _geometry(p_end, v_end, a.astype(np.int64), b.astype(np.int64), self.spec)
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
          names: Sequence[str], spec: IslSpec) -> IslContactTable:
    scanner = _Scanner(n, spec)
    for t, pos, vel in samples:
        scanner.feed(float(t), pos, vel)
    return scanner.finish(names)


def isl_contact_table(positions_km: NDArray[np.float64], velocities_km_s: NDArray[np.float64],
                      times_s: NDArray[np.float64], spec: IslSpec,
                      names: Optional[Sequence[str]] = None) -> IslContactTable:
    """
    The ISL contact dataset from `(T, N, 3)` **central-body-relative inertial** positions and velocities
    sampled at `times_s`: `isl.isl_contacts`'s records, bit for bit, as columns. `names` defaults to
    `body0 .. bodyN-1`.
    """
    pos = np.asarray(positions_km, dtype=np.float64)
    vel = np.asarray(velocities_km_s, dtype=np.float64)
    n = pos.shape[1]
    names = list(names) if names is not None else [f"body{k}" for k in range(n)]
    return _scan(((float(t), pos[k], vel[k]) for k, t in enumerate(np.asarray(times_s, dtype=np.float64))),
                 n, names, spec)


def isl_contact_table_from_recording(directory: Union[str, Path], spec: IslSpec,
                                     bodies: Optional[Sequence[str]] = None) -> IslContactTable:
    """
    The ISL contact dataset from a `history.HistorySink` recording, streamed one chunk at a time so a
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

    return _scan(samples(), len(names), names, spec)
