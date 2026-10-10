"""
Streaming history: snapshots written to disk in fixed-size chunks, so a recording's memory does not
grow with its length.

**Why.** `Simulation`'s in-memory history (`_record_state`) keeps every snapshot. That is right for
analysis runs and wrong for a constellation: 10,000 satellites every 10 s for a day is 8,640 snapshots
x 10,000 x 18 floats = 12.4 GB. A `HistorySink` keeps one chunk of `chunk_snapshots` snapshots in a
preallocated buffer, writes it to disk when full, and reuses the buffer, so memory is
`chunk_snapshots x n_bodies x 6 x 8` bytes whatever the duration.

**Format** (no dependency beyond numpy): a directory holding

    manifest.json              names, units, columns, dtype, the chunk list, the decimation
    chunk_00000_t.npy          (k,)        float64 seconds of simulation time
    chunk_00000_global.npy     (k, n, 6)   global states [x y z vx vy vz], km and km/s
    ...

`.npy` rather than `.npz` so a reader can memory-map a chunk instead of loading it. The manifest is
rewritten at every flush, so a run that dies keeps everything up to its last chunk.

**What is recorded.** Global states (the arena's state of record for position) of a fixed list of
*bodies*, by name. Barycentres are refused: a temporary one (`hierarchy.form_system`) can vanish
mid-run, while a body never does, so the recorded set is fixed for the run's life.

`read_history(directory, bodies=None, t_range=None)` returns `(times, names, states)` for a subset of
bodies and a time window, reading chunk by chunk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .simulator import Simulation

FORMAT_VERSION = 1


class HistorySink:
    """
    Snapshots of `bodies`' global states, streamed to `directory` in chunks. Attach it with
    `Simulation.attach_history_sink(sink, every=...)`; `step()` then writes here instead of to the
    in-memory history. Call `close()` (or use it as a context manager) to flush the last, partial chunk.
    """

    def __init__(self, directory: Union[str, Path], sim: "Simulation", bodies: Optional[Sequence[str]] = None,
                 chunk_snapshots: int = 1024) -> None:
        if chunk_snapshots < 1:
            raise ValueError(f"chunk_snapshots must be at least 1, got {chunk_snapshots}")
        names = list(bodies) if bodies is not None else [
            n for n, k in sim.name_to_index.items() if not sim.is_system[k]]
        missing = [n for n in names if n not in sim.name_to_index]
        if missing:
            raise KeyError(f"history sink: no bodies named {missing}")
        systems = [n for n in names if sim.is_system[sim.name_to_index[n]]]
        if systems:
            raise ValueError(f"history sink: {systems} are barycentres; a temporary one can vanish mid-run, "
                             f"so only bodies are recorded")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists():
            raise FileExistsError(f"{self.directory} already holds a recording; give each run its own directory")
        self.names: List[str] = names
        self.slots: NDArray[np.int64] = np.asarray([sim.name_to_index[n] for n in names], dtype=np.int64)
        self.chunk_snapshots = int(chunk_snapshots)
        self._t: NDArray[np.float64] = np.empty(self.chunk_snapshots, dtype=np.float64)
        self._g: NDArray[np.float64] = np.empty((self.chunk_snapshots, len(names), 6), dtype=np.float64)
        self._fill = 0
        self._chunks: List[str] = []
        self.every = 1
        self.closed = False
        self.snapshots = 0

    def record(self, t: float, global_states: NDArray[np.float64]) -> None:
        """Append one snapshot (`global_states` is the arena's whole array; the sink takes its rows)."""
        if self.closed:
            raise ValueError("history sink is closed")
        self._t[self._fill] = t
        np.take(global_states, self.slots, axis=0, out=self._g[self._fill])
        self._fill += 1
        self.snapshots += 1
        if self._fill == self.chunk_snapshots:
            self._flush()

    def _flush(self) -> None:
        if self._fill == 0:
            return
        stem = f"chunk_{len(self._chunks):05d}"
        np.save(self.directory / f"{stem}_t.npy", self._t[:self._fill])
        np.save(self.directory / f"{stem}_global.npy", self._g[:self._fill])
        self._chunks.append(stem)
        self._fill = 0
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "format_version": FORMAT_VERSION,
            "names": self.names,
            "columns": ["x", "y", "z", "vx", "vy", "vz"],
            "units": {"t": "s (simulation time)", "position": "km", "velocity": "km/s"},
            "frame": "simulation root, inertial (Simulation.global_states)",
            "dtype": "float64",
            "every": self.every,
            "chunks": self._chunks,
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=1))

    def close(self) -> None:
        """Flush the last, partial chunk and write the manifest. Idempotent."""
        if not self.closed:
            self._flush()
            self._write_manifest()
            self.closed = True

    def __enter__(self) -> "HistorySink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_history(
    directory: Union[str, Path],
    bodies: Optional[Sequence[str]] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> Tuple[NDArray[np.float64], List[str], NDArray[np.float64]]:
    """
    `(times (k,), names, states (k, n, 6))` from a recording, for `bodies` (default: all) and the
    closed window `t_range` (default: everything). Chunks are memory-mapped and only the selected rows
    and columns are copied out.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    all_names: List[str] = manifest["names"]
    names = list(bodies) if bodies is not None else all_names
    missing = [n for n in names if n not in all_names]
    if missing:
        raise KeyError(f"recording has no bodies named {missing}")
    cols = np.asarray([all_names.index(n) for n in names], dtype=np.int64)
    times: List[NDArray[np.float64]] = []
    states: List[NDArray[np.float64]] = []
    for stem in manifest["chunks"]:
        t = np.load(directory / f"{stem}_t.npy")
        keep = np.ones(t.shape, dtype=bool) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
        if not keep.any():
            continue
        g = np.load(directory / f"{stem}_global.npy", mmap_mode="r")
        times.append(np.asarray(t[keep], dtype=np.float64))
        states.append(np.asarray(g[np.flatnonzero(keep)][:, cols], dtype=np.float64))
    if not times:
        return np.empty(0, dtype=np.float64), names, np.empty((0, len(names), 6), dtype=np.float64)
    out_t: NDArray[np.float64] = np.concatenate(times)
    out_g: NDArray[np.float64] = np.concatenate(states)
    return out_t, names, out_g
