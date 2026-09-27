"""
JPL Horizons at the boundary: VECTORS output in, **plain arrays** out.

**What this is.** A parser for the text that the JPL Horizons system returns for an
`EPHEM_TYPE='VECTORS'` request (both the default text layout and `CSV_FORMAT='YES'`), a compact
`.npz` store for what it parses, a loader for the copies committed under
`src/orbital_engine/data/`, and a `fetch` used **only** by the download script
(`scripts/fetch_artemis2.py`). Nothing in the engine and nothing in the test suite touches the
network: `fetch` is the one function here that imports `urllib`, it imports it lazily, and
`tests/validation/test_horizons_bridge.py` runs the loaders with sockets disabled.

**Why a bridge and not an ephemeris.** `CLAUDE.md`: planetary ephemerides are not reimplemented,
and no stateful third-party object lives inside a step. Horizons is the ephemeris; this module turns
its output into `(n,)` / `(n, 3)` float64 arrays once, at ingest. Interpolating those arrays inside a
force model is a separate module's job (`ephemeris.py`), which consumes `t_s`, `position_km` and
`velocity_km_s` directly - there is deliberately no interpolation object here.

**Reference.** Giorgini, J. D. et al., *JPL's On-Line Solar System Data Service*, BAAS 28(3), 1158
(1996); the Horizons API is documented at https://ssd-api.jpl.nasa.gov/doc/horizons.html and the
user manual at https://ssd.jpl.nasa.gov/horizons/manual.html. The planetary and lunar states come
from the DE44x ephemeris Horizons names in its `{source: ...}` tag (DE441 at retrieval, recorded in
each file's `header`). Spacecraft states are whatever the mission supplied to JPL, also named there.

Time scales
-----------
Horizons VECTORS tables are always tabulated in **TDB** (the `JDTDB` column and the
`Calendar Date (TDB)` column; the text layout ends each epoch line with `TDB`). `t_s` is seconds from
`epoch_tdb`, an ISO-8601 instant **on the TDB scale**, computed from the calendar string's integer
0.1 ms field in `int64` nanoseconds - never from the printed Julian date, whose 9 decimals resolve
only 86 us. The printed `jd_tdb` is kept alongside and checked against the calendar to 1e-8 d.

The UTC <-> TDB offset: `TDB - UTC = (TAI - UTC) + 32.184 s + (TDB - TT)`. `TAI - UTC` has been
37 s since 2017-01-01 (no leap second since, to the date of this module - verify against IERS
Bulletin C for epochs after it), so `TT - UTC = 69.184 s` (`TT_MINUS_UTC_S`). `TDB - TT` is periodic,
+-1.66 ms, dominated by the Earth's orbital eccentricity:

    TDB - TT = 0.001657 s sin g + 0.000014 s sin 2g,  g = 357.53 deg + 0.98560028 deg (JD_TT - 2451545)

(USNO Circular 179, Kaplan 2005, Eq. 2.6 - **from memory, unverified against the text**;
`tests/validation/test_horizons_bridge.py` checks it against Horizons' own `TDB - UT` column, which the
download script stores). `utc_to_tdb` applies both. Earth rotation needs UT1, not UTC; the difference
DUT1 is under 0.9 s by construction of UTC and is not modelled here.

Frame
-----
`REF_SYSTEM='ICRF'`, `REF_PLANE='FRAME'` gives the ICRF axes (aligned with the J2000 mean equator
and equinox to ~0.02 arcsec). **Its +z is not the Earth's spin axis in 2026**: the mean pole of date
has precessed ~0.146 deg away since J2000. The engine's J2, drag and tesseral models assume the frame
+z is the spin axis. The size and the cost are in `artemis2.py` and `docs/architecture.md`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from .custom_types import ArrayFloat

__all__ = [
    "HORIZONS_API_URL", "DATA_DIR", "TT_MINUS_UTC_S", "SECONDS_PER_DAY",
    "HorizonsVectors", "HorizonsError",
    "parse_vectors", "format_vectors_csv", "save_npz", "load_npz", "load_cached", "cached_path",
    "build_query", "fetch", "stack_times",
    "as_ns", "parse_calendar_tdb", "julian_date", "utc_to_tdb", "tdb_minus_tt_s", "tdb_minus_utc_s",
]

HORIZONS_API_URL: Final[str] = "https://ssd.jpl.nasa.gov/api/horizons.api"

#: Committed data sets live here, one sub-folder per data set (`artemis2/`).
DATA_DIR: Final[Path] = Path(__file__).resolve().parent / "data"

SECONDS_PER_DAY: Final[float] = 86400.0

#: TT - UTC in seconds: TAI - UTC = 37 s (since 2017-01-01) plus TT - TAI = 32.184 s.
TT_MINUS_UTC_S: Final[float] = 69.184

# J2000.0 = JD 2451545.0 = 2000-01-01T12:00 (on whatever scale the Julian date is read in).
_J2000_JD: Final[float] = 2451545.0
_J2000_INSTANT: Final = np.datetime64("2000-01-01T12:00:00", "ns")
_NS_PER_S: Final[int] = 1_000_000_000

_MONTHS: Final[Dict[str, int]] = {
    m: k + 1 for k, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}

# "A.D. 2026-Apr-02 01:59:00.0000" (seconds to 1e-4, the precision Horizons prints by default).
_CAL_RE: Final = re.compile(
    r"A\.D\.\s+(\d{4})-([A-Z][a-z]{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?"
)
_NUM: Final[str] = r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?"
_KV_RE: Final = re.compile(r"\b(X|Y|Z|VX|VY|VZ)\s*=\s*(" + _NUM + r")")


class HorizonsError(ValueError):
    """Horizons returned text that is not a VECTORS table this module can read (an error message,
    a missing `$$SOE`, units other than km and km/s, a time scale other than TDB)."""


@dataclass(frozen=True)
class HorizonsVectors:
    """
    One Horizons VECTORS table as plain arrays.

    `jd_tdb` `(n,)` is the printed Julian date (TDB). `t_s` `(n,)` is **seconds from `epoch_tdb`**,
    TDB, exact to the table's 0.1 ms calendar field. `position_km` / `velocity_km_s` are `(n, 3)`,
    geometric states of `target` relative to `center` in `frame` (no light-time or aberration).
    `header` is the Horizons preamble above `$$SOE` verbatim - it names the data sources
    (`{source: ...}`), the ephemeris revision and the EOP file, and is the provenance record.
    """
    target: str
    center: str
    frame: str
    time_scale: str
    epoch_tdb: str
    jd_tdb: ArrayFloat
    t_s: ArrayFloat
    position_km: ArrayFloat
    velocity_km_s: ArrayFloat
    header: str

    def __post_init__(self) -> None:
        n = self.t_s.shape[0]
        if self.t_s.shape != (n,) or self.jd_tdb.shape != (n,):
            raise ValueError("t_s and jd_tdb must both be (n,)")
        if self.position_km.shape != (n, 3) or self.velocity_km_s.shape != (n, 3):
            raise ValueError("position_km and velocity_km_s must both be (n, 3)")

    @property
    def epoch(self) -> np.datetime64:
        """`epoch_tdb` as a `datetime64[ns]` (the value is a TDB instant)."""
        return as_ns(self.epoch_tdb)

    @property
    def state(self) -> ArrayFloat:
        """`(n, 6)` `[x y z vx vy vz]`, a copy."""
        out: ArrayFloat = np.concatenate([self.position_km, self.velocity_km_s], axis=1)
        return out

    def at_tdb(self, when: Union[str, np.datetime64]) -> int:
        """Index of the sample exactly at TDB instant `when`; `KeyError` if none."""
        target_ns = (as_ns(when) - self.epoch).astype(np.int64)
        hit = np.flatnonzero(np.round(self.t_s * 1e9).astype(np.int64) == int(target_ns))
        if hit.size == 0:
            raise KeyError(f"no sample at {when} TDB")
        return int(hit[0])


# --------------------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------------------

def as_ns(when: Union[str, np.datetime64]) -> np.datetime64:
    """`when` (an ISO-8601 string or a `datetime64`) as a `datetime64[ns]`; the scale is the caller's."""
    return cast(np.datetime64, np.asarray(when, dtype="datetime64[ns]")[()])


def parse_calendar_tdb(text: str) -> np.datetime64:
    """`"A.D. 2026-Apr-02 01:59:00.0000"` -> `datetime64[ns]`, exact (the scale is the caller's)."""
    m = _CAL_RE.search(text)
    if m is None:
        raise HorizonsError(f"not a Horizons calendar date: {text!r}")
    year, mon, day, hh, mm, ss, frac = m.groups()
    if mon not in _MONTHS:
        raise HorizonsError(f"unknown month {mon!r} in {text!r}")
    iso = f"{year}-{_MONTHS[mon]:02d}-{day}T{hh}:{mm}:{ss}"
    base = np.datetime64(iso, "ns")
    ns = int((frac or "0").ljust(9, "0")[:9])
    return base + np.timedelta64(ns, "ns")


def julian_date(when: Union[str, np.datetime64]) -> float:
    """Julian date of `when`, on the same scale as `when` (JD 2451545.0 = 2000-01-01T12:00)."""
    ns = (as_ns(when) - _J2000_INSTANT).astype(np.int64)
    return _J2000_JD + float(ns) / (SECONDS_PER_DAY * _NS_PER_S)


def tdb_minus_tt_s(jd_tt: Union[float, ArrayFloat]) -> ArrayFloat:
    """`TDB - TT` in seconds, the two-term periodic series in the module docstring."""
    g = np.radians(357.53 + 0.98560028 * (np.asarray(jd_tt, dtype=np.float64) - _J2000_JD))
    out: ArrayFloat = np.asarray(0.001657 * np.sin(g) + 0.000014 * np.sin(2.0 * g), dtype=np.float64)
    return out


def tdb_minus_utc_s(utc: Union[str, np.datetime64]) -> float:
    """`TDB - UTC` in seconds at UTC instant `utc` (2017-01-01 onward; see `TT_MINUS_UTC_S`)."""
    when = as_ns(utc)
    if when < np.datetime64("2017-01-01T00:00:00", "ns"):
        raise ValueError("TT_MINUS_UTC_S = 69.184 s holds from 2017-01-01; earlier epochs need the leap-second table")
    jd_tt = julian_date(when) + TT_MINUS_UTC_S / SECONDS_PER_DAY
    return TT_MINUS_UTC_S + float(tdb_minus_tt_s(jd_tt))


def utc_to_tdb(utc: Union[str, np.datetime64]) -> np.datetime64:
    """The TDB instant (as `datetime64[ns]`) of UTC instant `utc`, to the nanosecond of the series."""
    when = as_ns(utc)
    return when + np.timedelta64(int(round(tdb_minus_utc_s(when) * 1e9)), "ns")


# --------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------

def _header_field(header: str, label: str) -> str:
    m = re.search(r"^" + re.escape(label) + r"\s*:\s*(.+?)\s*$", header, flags=re.MULTILINE)
    if m is None:
        raise HorizonsError(f"Horizons header has no '{label}' line")
    return m.group(1)


def _strip_source(value: str) -> str:
    """`"Earth (399)                     {source: DE441}"` -> `"Earth (399)"`."""
    return re.sub(r"\s*\{source:[^}]*\}\s*$", "", value).strip()


def parse_vectors(text: str, *, epoch_tdb: Optional[Union[str, np.datetime64]] = None) -> HorizonsVectors:
    """
    Parse a Horizons VECTORS response (text or CSV layout, `VEC_TABLE` 2 or 3) into
    `HorizonsVectors`. `epoch_tdb` is the TDB instant `t_s` counts from; the default is the first
    sample. Raises `HorizonsError` on anything that is not a km / km/s TDB vector table - including
    Horizons' own error messages, which arrive as ordinary text with HTTP 200.
    """
    if "$$SOE" not in text or "$$EOE" not in text:
        first = next((ln for ln in text.splitlines() if ln.strip() and "API" not in ln), "")
        raise HorizonsError(f"no $$SOE/$$EOE block in Horizons output ({first.strip()!r})")
    pre, rest = text.split("$$SOE", 1)
    body = rest.split("$$EOE", 1)[0]

    units = _header_field(pre, "Output units")
    if units.split()[0] != "KM-S":
        raise HorizonsError(f"expected OUT_UNITS='KM-S', got {units!r}")
    target = _strip_source(_header_field(pre, "Target body name"))
    center = _strip_source(_header_field(pre, "Center body name"))
    frame = _header_field(pre, "Reference frame").split()[0]

    jd: List[float] = []
    when: List[np.datetime64] = []
    rows: List[List[float]] = []
    csv = "Calendar Date (TDB)" in pre or re.search(r"^\s*\d+\.\d+,\s*A\.D\.", body, re.MULTILINE) is not None
    if csv:
        if "Calendar Date (TDB)" not in pre:
            raise HorizonsError("CSV table is not tabulated in TDB")
        for line in body.splitlines():
            if not line.strip():
                continue
            cells = [c.strip() for c in line.split(",")]
            jd.append(float(cells[0]))
            when.append(parse_calendar_tdb(cells[1]))
            rows.append([float(c) for c in cells[2:8]])
    else:
        # Text layout: "2461132.583333333 = A.D. 2026-Apr-02 02:00:00.0000 TDB" then X= ... lines.
        records = re.split(r"^(?=\s*\d+\.\d+\s*=\s*A\.D\.)", body, flags=re.MULTILINE)
        for rec in records:
            if not rec.strip():
                continue
            head = rec.strip().splitlines()[0]
            if not head.rstrip().endswith("TDB"):
                raise HorizonsError(f"epoch line is not TDB: {head!r}")
            jd.append(float(head.split("=")[0]))
            when.append(parse_calendar_tdb(head))
            rest_lines = "\n".join(rec.strip().splitlines()[1:])
            kv = dict(_KV_RE.findall(rest_lines))
            if kv:  # VEC_LABELS='YES': "X =-2.46E+04 Y = ..."
                try:
                    rows.append([float(kv[k]) for k in ("X", "Y", "Z", "VX", "VY", "VZ")])
                except KeyError as exc:
                    raise HorizonsError(f"record lacks {exc}; VEC_TABLE must include velocity") from exc
            else:  # VEC_LABELS='NO': bare numbers, position line then velocity line
                numbers = re.findall(_NUM, rest_lines)
                if len(numbers) < 6:
                    raise HorizonsError(f"record has {len(numbers)} numbers; VEC_TABLE must include velocity")
                rows.append([float(x) for x in numbers[:6]])
    if not rows:
        raise HorizonsError("empty $$SOE/$$EOE block")

    instants = np.array(when, dtype="datetime64[ns]")
    epoch = as_ns(epoch_tdb) if epoch_tdb is not None else instants[0]
    t_s: ArrayFloat = np.asarray((instants - epoch).astype(np.int64), dtype=np.float64) / 1e9
    jd_arr: ArrayFloat = np.asarray(jd, dtype=np.float64)
    from_cal = np.array([julian_date(w) for w in instants], dtype=np.float64)
    if np.max(np.abs(jd_arr - from_cal)) > 1e-8:
        raise HorizonsError("printed JDTDB disagrees with the calendar column by more than 1e-8 d")
    states: ArrayFloat = np.asarray(rows, dtype=np.float64)
    return HorizonsVectors(
        target=target, center=center, frame=frame, time_scale="TDB",
        epoch_tdb=str(np.datetime_as_string(epoch, unit="ns")),
        jd_tdb=jd_arr, t_s=t_s,
        position_km=np.ascontiguousarray(states[:, :3]),
        velocity_km_s=np.ascontiguousarray(states[:, 3:]),
        header=pre.strip("\n"),
    )


def format_vectors_csv(vec: HorizonsVectors) -> str:
    """
    `vec` written back in Horizons' CSV layout (header, `$$SOE`, rows, `$$EOE`), with every float at
    17 significant digits so that `parse_vectors(format_vectors_csv(v), epoch_tdb=v.epoch_tdb)`
    reproduces `v` bit for bit. The calendar column keeps 9 fractional digits of seconds.
    """
    lines = [vec.header, "$$SOE"]
    instants = vec.epoch + np.round(vec.t_s * 1e9).astype(np.int64).astype("timedelta64[ns]")
    for k in range(vec.t_s.shape[0]):
        stamp = str(np.datetime_as_string(instants[k], unit="ns"))
        date, clock = stamp.split("T")
        y, mo, d = date.split("-")
        month = [m for m, v in _MONTHS.items() if v == int(mo)][0]
        values = ", ".join(f"{x:.16E}" for x in (*vec.position_km[k], *vec.velocity_km_s[k]))
        lines.append(f"{vec.jd_tdb[k]:.17g}, A.D. {y}-{month}-{d} {clock}, {values},")
    lines.append("$$EOE")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------------

_STR_FIELDS: Final[Tuple[str, ...]] = ("target", "center", "frame", "time_scale", "epoch_tdb", "header")
_ARR_FIELDS: Final[Tuple[str, ...]] = ("jd_tdb", "t_s", "position_km", "velocity_km_s")


def save_npz(vec: HorizonsVectors, path: Union[str, Path]) -> None:
    """Write `vec` as a compressed `.npz` - arrays as float64, metadata as 0-d unicode arrays, so
    `load_npz` needs no pickle."""
    payload: Dict[str, NDArray[np.generic]] = {f: np.asarray(getattr(vec, f)) for f in _ARR_FIELDS}
    payload.update({f: np.asarray(str(getattr(vec, f))) for f in _STR_FIELDS})
    np.savez_compressed(Path(path), **payload)  # type: ignore[arg-type]


def load_npz(path: Union[str, Path]) -> HorizonsVectors:
    """Read a file written by `save_npz` (no pickle, no network)."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {f: np.asarray(data[f], dtype=np.float64) for f in _ARR_FIELDS}
        strings = {f: str(data[f][()]) for f in _STR_FIELDS}
    return HorizonsVectors(**strings, **arrays)  # type: ignore[arg-type]


def cached_path(name: str, dataset: str = "artemis2") -> Path:
    """Where `load_cached(name, dataset)` reads from: `DATA_DIR / dataset / f"{name}.npz"`."""
    return DATA_DIR / dataset / f"{name}.npz"


def load_cached(name: str, dataset: str = "artemis2") -> HorizonsVectors:
    """The committed copy of one Horizons table (`"orion"`, `"moon"`, `"sun"` for `artemis2`)."""
    path = cached_path(name, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"no cached Horizons table '{name}' in {path.parent}")
    return load_npz(path)


# --------------------------------------------------------------------------------------------------
# Network - the download script only
# --------------------------------------------------------------------------------------------------

def build_query(
    command: str,
    *,
    center: str = "500@399",
    start: str = "",
    stop: str = "",
    step: str = "1m",
    ephemeris: bool = True,
    obj_data: bool = False,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    The Horizons API parameters for an Earth-centred (by default) ICRF km/s VECTORS table - pure, so
    the exact query recorded in a data set's README is the one this function builds. Values are
    quoted the way the API documentation shows (`COMMAND='-1024'`).
    """
    params: Dict[str, str] = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'YES'" if obj_data else "'NO'",
        "MAKE_EPHEM": "'YES'" if ephemeris else "'NO'",
    }
    if ephemeris:
        params.update({
            "EPHEM_TYPE": "'VECTORS'", "CENTER": f"'{center}'",
            "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
            "VEC_TABLE": "'2'", "REF_PLANE": "'FRAME'", "REF_SYSTEM": "'ICRF'",
            "OUT_UNITS": "'KM-S'", "CSV_FORMAT": "'YES'", "VEC_LABELS": "'NO'",
        })
    if extra:
        params.update(extra)
    return params


def fetch(params: Dict[str, str], *, timeout_s: float = 120.0) -> Tuple[str, str]:
    """
    **Network.** GET the Horizons API with `params` (see `build_query`) and return
    `(url, response_text)`. Used only by `scripts/fetch_artemis2.py`; never called by the engine or
    the tests. `urllib` is imported here, lazily, so importing this module opens nothing.
    """
    import urllib.parse
    import urllib.request

    url = HORIZONS_API_URL + "?" + urllib.parse.urlencode(params, safe="'@")
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - fixed https host
        text = resp.read().decode("utf-8")
    return url, text


def stack_times(tables: Sequence[HorizonsVectors]) -> HorizonsVectors:
    """Concatenate tables fetched in pieces (Horizons caps rows per request); they must share
    target, center, frame and epoch, and the result must be strictly increasing in time."""
    first = tables[0]
    for t in tables[1:]:
        if (t.target, t.center, t.frame, t.epoch_tdb) != (first.target, first.center, first.frame, first.epoch_tdb):
            raise ValueError("tables differ in target, center, frame or epoch")
    t_s: ArrayFloat = np.concatenate([t.t_s for t in tables])
    if np.any(np.diff(t_s) <= 0.0):
        raise ValueError("stacked tables are not strictly increasing in time")
    return HorizonsVectors(
        target=first.target, center=first.center, frame=first.frame, time_scale=first.time_scale,
        epoch_tdb=first.epoch_tdb,
        jd_tdb=np.concatenate([t.jd_tdb for t in tables]), t_s=t_s,
        position_km=np.concatenate([t.position_km for t in tables]),
        velocity_km_s=np.concatenate([t.velocity_km_s for t in tables]),
        header=first.header,
    )
