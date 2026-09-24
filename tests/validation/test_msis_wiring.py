"""
`DENSITY_MODEL_MSIS` wiring that must hold **without `pymsis` installed** - the half of the NRLMSIS
work that the no-numba CI job, a bare `pip install orbital_engine`, and every pre-existing
configuration depend on. The physics of the MSIS law itself is `test_msis.py` (which skips without
`pymsis`); nothing here evaluates MSIS.

1. **Pre-existing rows are bit-identical.** Adding the third law added three columns to `"drag"`'s
   parameters and a third masked term to the kernel. The guarantee is that a row on the single
   exponential or the table - written before MSIS existed, i.e. with columns 7-9 zero, or never
   written at all - produces **exactly** the acceleration it did before. That is asserted against a
   frozen copy of the pre-MSIS kernel (below, verbatim from commit 7d384b5, including its own copy of
   the old table lookup, so a change to `atmosphere.layered_density`'s arithmetic would show too), on
   a mixed arena with random states, with `np.array_equal`. There is no tolerance to derive: the
   masks are the same booleans, and the new term adds exactly `+0.0`.
2. **No network, no step-time MSIS.** The selector refuses to be set without all three indices in
   the same call, so `pymsis` is never left to download them; solar indices on a non-MSIS row are
   refused rather than silently ignored; out-of-range indices are refused. All before any bit or
   coefficient is written. The kernel's MSIS term reads a memo and raises `LookupError` on a miss -
   with `pymsis` made unimportable, to prove the step path never reaches for it.
3. **Optional dependency.** `import orbital_engine` succeeds with `pymsis` unimportable, and asking
   for MSIS then fails at configuration time with an `ImportError` naming the extra.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from orbital_engine import drag, scenarios
from orbital_engine.atmosphere import (
    BASE_ALTITUDE_KM, BASE_DENSITY_KG_M3, DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED,
    DENSITY_MODEL_MSIS, SCALE_HEIGHT_KM,
)
from orbital_engine.database import Base
from orbital_engine.drag import DRAG_MODEL, DRAG_PARAM_NAMES, EARTH_OMEGA
from orbital_engine.geopotential import EARTH_R_EQ
from orbital_engine.simulator import Simulation

ArrF = NDArray[np.float64]
N_LEGACY_COLUMNS = 7          # "drag"'s parameter width before f107, f107a, ap were appended

# A triple no test anywhere evaluates, so it is guaranteed absent from msis_bridge's memo.
NEVER_EVALUATED = {"f107": 171.25, "f107a": 163.5, "ap": 11.75}


def _session() -> Session:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sim() -> tuple[Simulation, List[int]]:
    sim = scenarios.earth_constellation(_session(), n_sats=3, n_planes=3)
    sats = sorted(i for n, i in sim.name_to_index.items() if n.startswith("SAT-"))
    return sim, sats


@pytest.fixture
def no_pymsis(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make `import pymsis` raise `ImportError` for the duration of one test."""
    monkeypatch.setitem(sys.modules, "pymsis", None)
    yield


# ==================================================================================================
# 1. The frozen pre-MSIS kernel, verbatim apart from the inlined helpers
# ==================================================================================================

def _frozen_two_law_kernel(indices: NDArray[np.int64], state: ArrF, parent_indices: NDArray[np.int32],
                           params: ArrF, out: ArrF) -> None:
    primaries = parent_indices[indices]
    rel = state[indices] - state[primaries]
    r2 = np.einsum("ij,ij->i", rel[:, :3], rel[:, :3])
    coeff = params[indices]
    omega = coeff[:, 5]
    scale_height = coeff[:, 3]
    vrx = rel[:, 3] + omega * rel[:, 1]
    vry = rel[:, 4] - omega * rel[:, 0]
    vrz = rel[:, 5]
    speed = np.sqrt(vrx * vrx + vry * vry + vrz * vrz)
    altitude = np.sqrt(r2) - coeff[:, 4]
    separated = r2 > 0.0
    use_layered = coeff[:, 6] >= 0.5

    # atmosphere.exponential_density at 7d384b5, masked branch.
    valid_e = separated & ~use_layered & (scale_height > 0.0)
    exponent_e = np.divide(altitude - coeff[:, 2], scale_height, out=np.zeros_like(altitude),
                           where=valid_e)
    rho_e = coeff[:, 1] * np.exp(-exponent_e, out=np.zeros_like(altitude), where=valid_e)
    # atmosphere.layered_density at 7d384b5, masked branch.
    valid_l = separated & use_layered
    band = np.clip(np.searchsorted(BASE_ALTITUDE_KM, altitude, side="right") - 1, 0,
                   BASE_ALTITUDE_KM.size - 1)
    exponent_l = (altitude - BASE_ALTITUDE_KM[band]) / SCALE_HEIGHT_KM[band]
    rho_l = BASE_DENSITY_KG_M3[band] * np.exp(-exponent_l, out=np.zeros_like(exponent_l),
                                               where=valid_l)
    density = rho_e + rho_l

    k = 0.5 * density * coeff[:, 0] * 1.0e3 * speed
    out[indices, 0] -= k * vrx
    out[indices, 1] -= k * vry
    out[indices, 2] -= k * vrz


def _legacy_arena(n: int, seed: int) -> tuple[ArrF, NDArray[np.int32], ArrF]:
    """`n` bodies about slot 0: random LEO-to-re-entry states, a root row, and every legacy flavour -
    single band, table, unwritten selector, `H = 0` no-op, `B = 0`, and a NaN selector (reachable only
    by direct assignment, and still the single band) - in the first seven columns."""
    rng = np.random.default_rng(seed)
    state = np.zeros((n, 6))
    radius = EARTH_R_EQ + rng.uniform(-20.0, 1200.0, n)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    state[:, :3] = radius[:, None] * direction
    state[:, 3:] = rng.normal(scale=4.0, size=(n, 3))
    state[0] = 0.0                                                  # the parent / root
    parents = np.zeros(n, dtype=np.int32)
    params = np.zeros((n, N_LEGACY_COLUMNS))
    flavour = np.arange(n) % 6
    params[:, 0] = rng.uniform(0.0, 0.1, n)
    params[:, 4] = EARTH_R_EQ
    params[:, 5] = rng.choice([0.0, EARTH_OMEGA, -EARTH_OMEGA], n)
    single = flavour == 0
    params[single, 1] = rng.uniform(1e-13, 1e-10, single.sum())
    params[single, 2] = rng.uniform(200.0, 600.0, single.sum())
    params[single, 3] = rng.uniform(20.0, 80.0, single.sum())
    params[single, 6] = DENSITY_MODEL_EXPONENTIAL
    params[flavour == 1, 6] = DENSITY_MODEL_LAYERED
    params[flavour == 2, 1:4] = [1e-12, 400.0, 60.0]                 # selector never written: 0.0
    params[flavour == 3, 6] = DENSITY_MODEL_EXPONENTIAL             # H = 0: the silent no-op
    params[flavour == 4, 0] = 0.0                                   # B = 0
    params[flavour == 4, 6] = DENSITY_MODEL_LAYERED
    params[flavour == 5, 1:4] = [2e-12, 350.0, 55.0]               # a NaN selector, written directly:
    params[flavour == 5, 6] = np.nan                                # the two-law kernel read it as 0.0
    return state, parents, params


@pytest.mark.parametrize("width", ["legacy 7-column array", "current array, columns 7-9 zero",
                                   "current array, junk in columns 7-9"])
def test_pre_existing_rows_are_bit_identical_to_the_two_law_kernel(width: str) -> None:
    """Every legacy flavour, 400 random states, the root row included (`indices` starts at 0), with
    floating-point errors raising. Bitwise equality with the frozen kernel, and the result is not
    trivially zero. 'Junk in columns 7-9' is what a direct write could leave on a non-MSIS row: the
    kernel must not read those columns unless the selector says MSIS."""
    n = 400
    state, parents, legacy = _legacy_arena(n, seed=20260924)
    if width == "legacy 7-column array":
        params = legacy
    else:
        params = np.zeros((n, len(DRAG_PARAM_NAMES)))
        params[:, :N_LEGACY_COLUMNS] = legacy
        if "junk" in width:
            params[:, 7:10] = np.random.default_rng(1).uniform(-1e3, 1e3, (n, 3))
    idx = np.arange(n, dtype=np.int64)
    expected = np.full((n, 3), 0.25)
    actual = np.full((n, 3), 0.25)
    with np.errstate(all="raise"):
        _frozen_two_law_kernel(idx, state, parents, params, expected)
        drag.drag_kernel(idx, 0.0, state, np.zeros(n), parents, params, actual)
    assert np.array_equal(actual, expected)
    assert np.count_nonzero(actual != 0.25) > n, "the arena must actually produce drag"


def test_every_existing_all_zero_row_still_means_the_single_exponential() -> None:
    """The convention `docs/architecture.md` states for selector columns: zero means what the model did
    before. The new selector value is 2.0, the old ones are unchanged, and the parameter layout only
    grew at the end."""
    assert DENSITY_MODEL_EXPONENTIAL == 0.0 and DENSITY_MODEL_LAYERED == 1.0
    assert DENSITY_MODEL_MSIS == 2.0
    assert DRAG_PARAM_NAMES[:N_LEGACY_COLUMNS] == (
        "ballistic_coeff", "rho0", "h0", "scale_height", "r_ref", "omega", "density_model")
    assert DRAG_PARAM_NAMES[N_LEGACY_COLUMNS:] == ("f107", "f107a", "ap")


# ==================================================================================================
# 2. Refusals - all before any bit or coefficient is written, none needing pymsis
# ==================================================================================================

_COMMON = {"ballistic_coeff": 0.02, "r_ref": EARTH_R_EQ, "omega": EARTH_OMEGA}


@pytest.mark.parametrize("given", [{}, {"f107": 150.0}, {"f107": 150.0, "f107a": 150.0},
                                   {"f107a": 150.0, "ap": 15.0}])
def test_msis_refuses_to_be_selected_without_all_three_indices(given: dict[str, float]) -> None:
    """The no-network guarantee at its root: `pymsis.calculate` downloads CelesTrak's space-weather
    file whenever an index is `None`. Refusing the selector without all three means the wrap never has
    a `None` to pass."""
    sim, sats = _sim()
    mask = sim.force_model_mask.copy()
    with pytest.raises(ValueError, match="needs f107, f107a and ap"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS, **given, **_COMMON)
    assert np.array_equal(sim.force_model_mask, mask)
    assert DRAG_MODEL not in sim.force_model_params


@pytest.mark.parametrize("law", [DENSITY_MODEL_EXPONENTIAL, DENSITY_MODEL_LAYERED, None])
def test_solar_indices_on_a_non_msis_row_are_refused_not_ignored(law: float | None) -> None:
    """Otherwise a sweep over F10.7 on the table would 'measure' a solar-activity axis that does
    nothing - a silent no-op of exactly the kind item 5 guards against."""
    sim, sats = _sim()
    selector = {} if law is None else {"density_model": law}
    with pytest.raises(ValueError, match="read only under density_model=DENSITY_MODEL_MSIS"):
        sim.enable_force_model(DRAG_MODEL, sats, f107=150.0, **selector, **_COMMON)
    assert DRAG_MODEL not in sim.force_model_params


@pytest.mark.parametrize("bad", [{"f107": 0.0}, {"f107": -5.0}, {"f107a": float("nan")},
                                 {"ap": -1.0}, {"ap": 401.0}, {"f107": float("inf")}])
def test_out_of_range_indices_are_refused(bad: dict[str, float], no_pymsis: None) -> None:
    """Range-checked before `pymsis` is touched - which is why this passes with it unimportable."""
    sim, sats = _sim()
    activity = {**NEVER_EVALUATED, **bad}
    with pytest.raises(ValueError, match="MSIS"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS, **activity,
                               **_COMMON)
    assert DRAG_MODEL not in sim.force_model_params


def test_an_unknown_selector_is_refused() -> None:
    sim, sats = _sim()
    for bad in (3.0, 1.5, -1.0):
        with pytest.raises(ValueError, match="not a known density law"):
            sim.enable_force_model(DRAG_MODEL, sats, density_model=bad, **_COMMON)


# ==================================================================================================
# 3. The kernel never evaluates MSIS; the engine runs without pymsis
# ==================================================================================================

def test_the_kernel_raises_on_an_unevaluated_profile_and_never_imports_pymsis(no_pymsis: None) -> None:
    """A row switched to MSIS by direct assignment bypasses `enable_force_model`, so no profile was
    evaluated for it. The kernel must say so - `LookupError`, not a silent zero and not a lazy
    `pymsis` call inside the step. `pymsis` is unimportable here, so an `ImportError` would mean the
    kernel tried."""
    state = np.zeros((2, 6))
    state[1, :3] = [EARTH_R_EQ + 400.0, 0.0, 0.0]
    state[1, 3:] = [0.0, 7.6, 0.0]
    params = np.zeros((2, len(DRAG_PARAM_NAMES)))
    params[1] = [0.02, 0.0, 0.0, 0.0, EARTH_R_EQ, 0.0, DENSITY_MODEL_MSIS,
                 NEVER_EVALUATED["f107"], NEVER_EVALUATED["f107a"], NEVER_EVALUATED["ap"]]
    with pytest.raises(LookupError, match="configuration time"):
        drag.drag_kernel(np.array([1], dtype=np.int64), 0.0, state, np.zeros(2),
                         np.zeros(2, dtype=np.int32), params, np.zeros((2, 3)))


def test_selecting_msis_without_pymsis_fails_at_configuration_with_the_extra_named(
    no_pymsis: None,
) -> None:
    sim, sats = _sim()
    mask = sim.force_model_mask.copy()
    with pytest.raises(ImportError, match=r"orbital_engine\[msis\]"):
        sim.enable_force_model(DRAG_MODEL, sats, density_model=DENSITY_MODEL_MSIS,
                               **NEVER_EVALUATED, **_COMMON)
    assert np.array_equal(sim.force_model_mask, mask)
    assert DRAG_MODEL not in sim.force_model_params


def test_the_engine_imports_and_steps_drag_without_pymsis() -> None:
    """In a fresh interpreter, because this one has already imported everything: `pymsis` blocked,
    the package imported, and a layered-drag Cowell satellite stepped. The optional-dependency claim
    of `pyproject.toml`'s `[msis]` extra, checked rather than asserted."""
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; sys.modules['pymsis'] = None\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "import numpy as np\n"
        "from sqlalchemy import create_engine\n"
        "from sqlalchemy.orm import sessionmaker\n"
        "from sqlalchemy.pool import StaticPool\n"
        "import orbital_engine\n"
        "from orbital_engine import scenarios, msis_bridge\n"
        "from orbital_engine.database import Base\n"
        "from orbital_engine.custom_types import PropagatorType\n"
        "e = create_engine('sqlite:///:memory:', poolclass=StaticPool)\n"
        "Base.metadata.create_all(e)\n"
        "sim = scenarios.earth_constellation(sessionmaker(bind=e)(), n_sats=1, n_planes=1)\n"
        "i = next(v for k, v in sim.name_to_index.items() if k.startswith('SAT-'))\n"
        "sim.set_propagator([i], PropagatorType.COWELL)\n"
        "sim.enable_force_model('point_mass_gravity', [i])\n"
        "sim.enable_force_model('drag', [i], ballistic_coeff=0.02, r_ref=6378.137, density_model=1.0)\n"
        "sim.step(30.0)\n"
        "assert np.all(np.isfinite(sim.global_states[i]))\n"
        "assert 'pymsis' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
        "print('ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]
