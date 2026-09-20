from __future__ import annotations
import logging
from typing import List, Optional, Any, Dict, Sequence, Union, cast
from .custom_types import ScalarSeconds, PropagatorType, ForceModelMask, COEIndex
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .propagators import (
    Propagator, KeplerianPropagator, SecularJ2Propagator, secular_j2_rates, mean_seeded_p,
)
from .kernels import (
    NUMBA_AVAILABLE, calc_global_states, cowell_rk4_step, kepler_propagate, rebase_relative_states,
    secular_j2_propagate,
)
from .database import get_session, CelestialBodyORM, BaseBodyORM, VesselORM, VirtualBodyORM, SystemORM
from .body import BodyHandle
from .registry import get_force_model, get_propagators
from .integrators import Integrator, RK4Integrator
from . import frames as fr
from . import forces
from . import gravity  # noqa: F401 - import registers "point_mass_gravity" as a force model (gravity.py)
from . import geopotential  # registers "j2"; also supplies barycentre_parented and J2_MODEL below
from . import thrust  # registers "thrust"; also supplies deplete_mass, called from step() below
from . import manoeuvres  # impulsive Delta-v: Manoeuvre, apply_delta_v, used by the API below
from . import events  # event-driven step splitting: Event, locate_crossing, used by the API below

# A library must not write to stdout. Build-time diagnostics go to the logger, where an application
# can opt in with logging.getLogger("orbital_engine").setLevel(logging.DEBUG).
logger = logging.getLogger(__name__)


class Simulation:
    """
    Initializes the simulation with a list of body names, maximum capacity, and an optional start epoch.
    Written using Data-Oriented Design (DOD) principles for performance, parallelization, and memory efficiency.
    Free-index stack is used to efficiently manage memory and allow for dynamic addition/removal of bodies.
    All results are stored in a history buffer for later analysis or visualization, exported as a pandas DataFrame.
    """
    def __init__(self,
                 body_names: List[str],
                 system_names: List[str],
                 max_capacity: int = 10000,
                 start_epoch: Optional[datetime] = None,
                 session: Optional[Session] = None,
                 default_propagator: Optional[Propagator] | int = PropagatorType.NONE,
                 use_compiled_kernel: Optional[bool] = None,
                 record_history: bool = True) -> None:
        """
        `use_compiled_kernel` defaults to whether numba is importable. The choice cannot simply be
        "compiled if available" spelled as `True`, because without numba the kernel still *runs* -
        as interpreted Python - and is then slower than the vectorised NumPy path it replaces.
        Passing an explicit bool overrides the detection, which is what the equivalence tests use to
        exercise both paths in one process.

        `record_history` gates the per-body, per-step dict append in `_record_state`. It costs about
        14 us per step at five bodies and grows linearly with both bodies and steps, which is fine
        for analysis and pure waste for a benchmark or a parameter sweep that only reads final state.
        """
        self.start_epoch: datetime = start_epoch if start_epoch is not None else datetime.now()

        # --- Runtime State Varables ---
        self.t: ScalarSeconds = 0.0                                               # Global Simulation time
        self.bodies: List[BodyHandle] = []                                  # Body Handles for UI use on simulation objects.

        # Columnar history buffers. One entry per snapshot, each an (n_recorded, 6) array, rather
        # than one dict per body per step. See `_record_state`.
        self._hist_seconds: List[float] = []
        self._hist_global: List[NDArray[np.float64]] = []
        self._hist_local: List[NDArray[np.float64]] = []
        self._hist_coe: List[NDArray[np.float64]] = []
        self._recorded_names: List[str] = []
        self._recorded_slots: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        self.max_capacity = max_capacity                                    # Simulation rated capacity

        self.mu_array = np.zeros(max_capacity, dtype=np.float64)            # mu values for each body
        self.local_states = np.zeros((max_capacity, 6), dtype=np.float64)   # [x, y, z, vx, vy, vz]
        self.coe_states = np.zeros((max_capacity, 6), dtype=np.float64)     # [p, e, i, Omega, omega, theta]
        self.parent_indices = np.full(max_capacity, -1, dtype=np.int32)     # indices of parent bodies
        self.global_states = np.zeros((max_capacity, 6), dtype=np.float64)  # We will store both systems and bodies.

        # The free-list stack.
        self.free_indices = list(range(max_capacity-1, -1, -1))             # Reverse order for efficient pop()

        # Name to integer mapping
        self.name_to_index: dict[str, int] = {}                             # Maps body names to their indices

        # Filtering masks
        self.active_mask = np.zeros(max_capacity, dtype=np.bool_)           # Active indices mask for bodies currently loaded.
        self.is_system = np.zeros(max_capacity, dtype=np.bool_)             # Checks if a body index is a system barycenter.
        self.is_head = np.zeros(max_capacity, dtype=np.bool_)               # Checks if a body index is the head of its local system bubble.
        self.body_sys_map = np.full(max_capacity, -1, dtype=np.int32)       # Maps body index to their local system index (supports barycenters and point masses). Systems map to their parent system, or -1 if none.
        self.propagator_type = np.full(max_capacity, PropagatorType.KEPLERIAN,dtype=np.uint8)       # Likely won't be implemented yet
        self.sys_head_map = np.full(max_capacity, -1, dtype=np.int32)       # Systems refer to sibling heads, while bodies refer to siblings head

        # --- Compiled-kernel support ---
        self.use_compiled_kernel: bool = NUMBA_AVAILABLE if use_compiled_kernel is None else use_compiled_kernel
        self.record_history: bool = record_history

        # Scratch for the reflex kick, owned by the arena and reused every step so the kernel itself
        # allocates nothing. Only the active head slots are zeroed per step, so the cost tracks the
        # active set rather than max_capacity.
        self._kick = np.zeros((max_capacity, 6), dtype=np.float64)
        self._accum = np.zeros((max_capacity, 6), dtype=np.float64)

        # Active sibling/head slots as integer indices. These change only when the active set does,
        # so they are cached rather than recomputed from the masks on every step.
        #
        # Annotated explicitly rather than inferred: `np.empty(0, ...)` infers the *narrow* shape
        # type `ndarray[tuple[int], ...]` under some numpy versions, which then rejects the
        # `ndarray[tuple[int, ...], ...]` that `flatnonzero(...).astype(...)` returns in
        # `_refresh_active_indices`. It resolved differently on 3.10 than on 3.11/3.12 and only CI
        # caught it.
        self._sib_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._head_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        # Per-body propagator selection (data-driven; see `set_propagator`). Cowell-designated slots
        # are excluded from both Keplerian dispatch sets below, so a Cowell body's dynamics come only
        # from `_cowell_integrator` and a Keplerian sibling's dynamics are unaffected by whether any
        # Cowell body exists elsewhere in the arena. Rebuilt in `_refresh_active_indices`, same
        # convention and same reason as `_sib_idx` / `_head_idx` above.
        self._cowell_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._kepler_sib_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._kepler_sib_mask: NDArray[np.bool_] = np.zeros(max_capacity, dtype=np.bool_)

        # Cowell's integrator, owned here rather than constructed per call so its scratch (sized to
        # max_capacity) is allocated exactly once - see integrators.py's module docstring for why an
        # integrator must be a stateful object rather than a bare function.
        self._cowell_integrator: Integrator = RK4Integrator(max_capacity)

        # Secular-J2 propagator scratch (see propagators.SecularJ2Propagator). `_secular_j2_idx` is
        # the cached active-slot list, same convention as `_cowell_idx` above. `_secular_j2_rates`
        # holds the three cached per-body rates (`[dRAAN/dt, dARGPE/dt, dM/dt]`, radians/second),
        # computed once by `set_propagator` rather than every step - p, e, i (and so mean motion) are
        # constant under this theory, so nothing per-step depends on recomputing them.
        # `_secular_j2_rel` is the per-step parent-relative state vector `propagate` writes into;
        # `step()` re-bases it onto the parent's end-of-step global position once `calc_global()` has
        # produced it, exactly as it does for a Cowell body - see that propagator's docstring.
        self._secular_j2_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._secular_j2_rates = np.zeros((max_capacity, 3), dtype=np.float64)
        self._secular_j2_rel = np.zeros((max_capacity, 6), dtype=np.float64)

        # --- Force-model composition arena hooks ---
        # A body's enabled physics is data: one bit per registered force model (registry.py), OR'd
        # together per slot. Composing two models on one body is `mask |= bit_a; mask |= bit_b` - no
        # branch, no subclass, no lookup keyed on a string at call time. See forces.py for the
        # composition layer this feeds and the contract a force model implements.
        self.force_model_mask: ForceModelMask = np.zeros(max_capacity, dtype=np.uint64)

        # Shared acceleration accumulator, km/s^2, arena-indexed like global_states' position
        # columns. Caller-owned scratch, exactly like `_kick`/`_accum` above: `compose_accelerations`
        # zeros only the rows some model will touch and allocates nothing itself.
        self.accel_accum: NDArray[np.float64] = np.zeros((max_capacity, 3), dtype=np.float64)

        # Per-model per-body coefficient arrays, keyed by model name and allocated lazily (shape
        # (max_capacity, n_params)) the first time that model is resolved with at least one active
        # body. Populated through `enable_force_model`; read by that model's kernel as `params`.
        self.force_model_params: Dict[str, NDArray[np.float64]] = {}

        # Resolved dispatch list and the union of slots it touches, rebuilt by
        # `resolve_force_models()` whenever `force_model_mask` or `active_mask` changes. Holding
        # these as a cache rather than recomputing per call is what keeps `accelerations()` bounded
        # by registered-model count rather than by `max_capacity` - see forces.py.
        self._resolved_force_models: List[forces.ResolvedForceModel] = []
        self._force_dispatch_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)

        # Thrust is the one force model whose coefficients are *state*: a burning vessel's mass falls,
        # so `force_model_params["thrust"]`'s mass column has to be advanced once per step. These two
        # are the cached handles `step()` needs to do that without searching the resolved list or
        # reducing over the mask - same caching convention, and same obligation to rebuild on any
        # configuration change, as `_force_dispatch_idx` above. A one-row dummy stands in until the
        # model is resolved for the first time; `_thrust_idx` is empty until then, so it is never read.
        # See thrust.py's module docstring for why the mass lives in the parameter array at all.
        self._thrust_idx: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._thrust_params: NDArray[np.float64] = np.zeros(
            (1, len(thrust.THRUST_PARAM_NAMES)), dtype=np.float64)

        # Scheduled impulsive manoeuvres, kept sorted by epoch (see `schedule_delta_v`). A plain list
        # rather than a registry axis: an impulse is neither an acceleration nor a way of advancing
        # time, so neither half of `registry.py` fits it - see `manoeuvres.py`'s module docstring.
        # `step()` tests this list's truthiness once per step and does nothing else when it is empty,
        # so an arena with no mission profile pays nothing.
        self._manoeuvres: List[manoeuvres.Manoeuvre] = []

        # Event-driven step splitting (`events.py`). Same shape as the manoeuvre queue above and for
        # the same reason: an event is neither an acceleration nor a way of advancing time, so
        # neither half of `registry.py` fits it. `step()` tests this list's truthiness once per step
        # and does nothing else when it is empty, so an arena with no events is bit-identical - and
        # the snapshot buffers below stay unallocated until `add_event` is called.
        self._events: List[events.Event] = []
        self._event_directions: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._event_tolerances: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.max_event_splits: int = events.MAX_SPLITS_PER_STEP
        self._event_splits: int = 0
        self._event_evaluations: int = 0
        # Snapshot of everything `_advance` mutates, so a trial propagation can be undone exactly.
        # `accel_accum`, `_kick`, `_accum`, `_cowell_rel` and `_secular_j2_rel` are deliberately
        # absent: every one of them is fully rewritten at its dispatch rows before being read
        # (`forces.compose_accelerations` zeroes `out[dispatch_idx]` first), so restoring them would
        # be copying scratch that has no readable state to restore.
        self._event_snap_t: ScalarSeconds = 0.0
        self._event_snap_global: NDArray[np.float64] = np.empty((0, 6), dtype=np.float64)
        self._event_snap_local: NDArray[np.float64] = np.empty((0, 6), dtype=np.float64)
        self._event_snap_coe: NDArray[np.float64] = np.empty((0, 6), dtype=np.float64)
        self._event_snap_thrust: NDArray[np.float64] = np.empty(
            (0, len(thrust.THRUST_PARAM_NAMES)), dtype=np.float64)

        # Cowell dispatch plan, rebuilt by `_refresh_cowell_plan` from both `_refresh_active_indices`
        # (the Cowell set changed) and `resolve_force_models` (the enabled models changed). The
        # compiled twin `kernels.cowell_rk4_step` is fused for `point_mass_gravity` and `j2` only,
        # so `_cowell_fused_ok` records whether the current masks let `step()` use it; any other
        # model on a Cowell body sends the whole Cowell set down the NumPy `RK4Integrator` path.
        # `_cowell_rel` is the per-step parent-relative result both paths hand to the re-base in
        # `step()`, the same role `_secular_j2_rel` plays for that propagator.
        self._cowell_primaries: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        # Whether `_rebase` may use its compiled twin: false only when a re-based body's parent is
        # in the same re-base set, where the scalar loop would read the parent live and the NumPy
        # block reads it gathered. Rebuilt in `_refresh_active_indices`.
        self._rebase_compiled_ok: bool = True
        self._cowell_rel = np.zeros((max_capacity, 6), dtype=np.float64)
        self._cowell_has_point_mass = np.zeros(max_capacity, dtype=np.bool_)
        self._cowell_has_j2 = np.zeros(max_capacity, dtype=np.bool_)
        self._no_j2_params = np.zeros((1, len(geopotential.J2_PARAM_NAMES)), dtype=np.float64)
        self._cowell_j2_params: NDArray[np.float64] = self._no_j2_params
        self._cowell_fused_ok: bool = False

        self._build_universe(body_names, system_names, session=session)     # Initialize the simulation by building the universe from the database.

    def _build_universe(self, body_names: List[str], system_names: List[str], session: Optional[Session] = None) -> None:
        """Queries the DB, topologically sorts the bodies, and populates the arrays."""
        local_session = False                                               # Framework guard for test cases.
        if session is None:
            session = get_session()
            local_session = True
            

        orm_bodies = session.query(BaseBodyORM).filter(BaseBodyORM.name.in_(body_names)).all()  # All queried bodies
        orm_systems = session.query(SystemORM).filter(SystemORM.name.in_(system_names)).all()   # All queried systems

        if len(orm_bodies) != len(body_names):
            found = [b.name for b in orm_bodies]
            missing = set(body_names) - set(found)
            raise ValueError(f"Could not find bodies in database: {missing}")

        if len(orm_systems) != len(system_names):
            found = [b.name for b in orm_systems]
            missing = set(system_names) - set(found)
            raise ValueError(f"Could not find systems in database: {missing}")
        
        all_orm_bodies: List[tuple[BaseBodyORM, Optional[SystemORM]]] = []  # Will attempt to pair each body with a respective system.

        for orm_body in orm_bodies:
            if orm_body.system in orm_systems: # Pair up the body and system
                all_orm_bodies.append((orm_body, orm_body.system))
            else:
                reparent = getattr(orm_body.parent, 'system', None) # Check the parent body's system
                all_orm_bodies.append((orm_body, reparent if reparent in orm_systems else None)) # If still not found, orphan the body
        for orm_sys in orm_systems: # Auto add all the barycenters for each system.
            if orm_sys.barycenter:
                reparent = getattr(orm_sys.barycenter, 'system', None)
                all_orm_bodies.append((orm_sys.barycenter, reparent if reparent in orm_systems else None)) # Pair systems with parent systems

        # print(all_orm_bodies)
        sys_barycenters = set(getattr(id, 'barycenter_id', -1) for _, id in all_orm_bodies)     # Index of all barycenters in simulation.
        logger.debug("Barycenter ids in scenario: %s", sys_barycenters)

        for orm_body, associated_sys in all_orm_bodies:
            idx = self.free_indices.pop()
            self.name_to_index[orm_body.name] = idx

            self.active_mask[idx]   = True
            self.mu_array[idx]      = getattr(orm_body, "mu", 0.0)
            self.coe_states[idx, 0] = getattr(orm_body, "p", 0.0)
            self.coe_states[idx, 1] = getattr(orm_body, "e", 0.0)
            self.coe_states[idx, 2] = getattr(orm_body, "i", 0.0)
            self.coe_states[idx, 3] = getattr(orm_body, "raan", 0.0)
            self.coe_states[idx, 4] = getattr(orm_body, "arg_pe", 0.0)
            self.coe_states[idx, 5] = getattr(orm_body, "theta", 0.0)


            if orm_body.id in sys_barycenters:
                self.is_system[idx] = True
            
            if getattr(associated_sys, 'head_body_id', None) == orm_body.id:
                self.is_head[idx] = True
        
        # Note: an unreferenced barycenter is not flagged as a system, because nothing points at it.
        logger.debug("Slot map: %s", self.name_to_index)
        logger.debug("is_system: %s", self.is_system[self.active_mask])
        logger.debug("is_head:   %s", self.is_head[self.active_mask])

        for orm_body, associated_sys in all_orm_bodies:
            idx = self.name_to_index[orm_body.name]

            parent = getattr(getattr(orm_body, 'parent', None), 'name', None)
            bary_name = getattr(getattr(associated_sys, 'barycenter', None), 'name', None) # Check that barycenters are members of parent barycenters

            system_idx = self.name_to_index.get(bary_name, idx) if bary_name else idx # Orphaned or no system means self-reference.

            self.parent_indices[idx] = self.name_to_index.get(parent, idx) if parent else idx # Similar logic for parent.

            if system_idx == idx and not self.is_system[idx]: # If not a root system, then...
                system_idx = self.parent_indices[idx] # Define the parent body as a system. (Can also be self-referencing if no parent exists)

            self.body_sys_map[idx] = system_idx # Bubbled logic


        self._resolve_circular() # Resolve any circular dependencies in the parent graph, ensuring a valid hierarchy.

        if local_session:   # Test case guard framework
            session.close()

        self._build_sys_head_map()  # Build a mapping from each body to the head of its local system bubble, allowing for efficient lookups.

        logger.debug("parent_indices: %s", self.parent_indices[self.active_mask])
        logger.debug("body_sys_map:   %s", self.body_sys_map[self.active_mask])
        logger.debug("sys_head_map:   %s", self.sys_head_map[self.active_mask])

        self._topological_sort(self.parent_indices) # Topological sort based on DB parent_indices. Provides an schematic for the initial positions of all bodies.
        self._unfold_database_to_global() # Using our topological map, compute global positions for all bodies based on local states and reference parents.
        logger.debug("parent_indices after unfold: %s", self.parent_indices[self.active_mask])

        # Re-sort based on system mapping, This means barycentric referencing (and point-mass in extremes) for all bodies.
        self._topological_sort(self.body_sys_map)   # This ensures all siblings (+head) are processed at the same step, allowing for proper barycentric calculations.
        self._recalculate_all_barycenters()         # Barycenters wouldn't have positions or coes defined in the DB, so we first compute their global vectors.
        self._zero_roots()                          # We want to ensure the root system is at the origin, so we shift all bodies in the same ancestry to that root, but its vector.
        self._rehydrate_coes()                      # Recompute the local states and coes based on new global positions and parenting. Also provides barycenters coe values.

        
        self._refresh_active_indices()
        self.resolve_force_models()  # Empty mask -> empty dispatch list; see resolve_force_models.

        loaded_count = int(np.count_nonzero(self.active_mask))
        logger.info("Universe built: %d slots loaded.", loaded_count)
        logger.debug("mu_array: %s", self.mu_array[:loaded_count])

    def _refresh_active_indices(self) -> None:
        """
        Rebuild the cached sibling/head index arrays from the boolean masks.

        Must be called after anything that changes `active_mask` or `is_head`. There is currently no
        despawn path, so in practice that is once, at the end of the build - but the kernel reads
        these instead of the masks, so a future spawn that forgets this call would propagate a stale
        body set rather than failing loudly.
        """
        self._sib_idx = np.flatnonzero(self.active_mask & ~self.is_head).astype(np.int64)
        self._head_idx = np.flatnonzero(self.active_mask & self.is_head).astype(np.int64)

        # Per-body propagator selection, read from data (`propagator_type`) rather than inferred.
        # Cowell-designated slots come out of the Keplerian dispatch sets so `step()` never hands them
        # to the analytic path - both the compiled kernel's index array and the NumPy reference's
        # sibling mask, so the two propagation paths exclude them identically (see `KeplerianPropagator
        # .propagate`'s `sib_mask` override and `tests/validation/test_cowell_propagator.py`'s
        # isolation test).
        is_cowell = self.propagator_type == np.uint8(PropagatorType.COWELL)
        is_secular_j2 = self.propagator_type == np.uint8(PropagatorType.SECULAR_J2)
        self._cowell_idx = np.flatnonzero(self.active_mask & ~self.is_head & is_cowell).astype(np.int64)
        self._secular_j2_idx = np.flatnonzero(
            self.active_mask & ~self.is_head & is_secular_j2
        ).astype(np.int64)
        self._kepler_sib_mask = self.active_mask & ~self.is_head & ~is_cowell & ~is_secular_j2
        self._kepler_sib_idx = np.flatnonzero(self._kepler_sib_mask).astype(np.int64)
        self._rebase_compiled_ok = not (
            bool(np.isin(self.parent_indices[self._cowell_idx], self._cowell_idx).any())
            or bool(np.isin(self.parent_indices[self._secular_j2_idx], self._secular_j2_idx).any())
        )
        self._refresh_cowell_plan()

        # Flatten the tier list into a single topologically ordered slot array. `calc_global_states`
        # needs only the ordering, not the tier boundaries, since a forward pass over a topological
        # order already resolves every parent before its children.
        #
        # This must be built from the tiers produced by the *second* `_topological_sort`, the one
        # keyed on `body_sys_map`. The first sort is keyed on `parent_indices` and is only used to
        # unfold database elements into initial global states.
        self._topo_order = np.concatenate(self.topological_tiers).astype(np.int64)
        self._n_roots = int(len(self.topological_tiers[0]))

        # Recording order is fixed here so `_record_state` needs no dict iteration per step, and so
        # the row order of `history` stays stable across snapshots. Insertion order of
        # `name_to_index` is preserved, matching the previous recorder's output ordering.
        self._recorded_names = list(self.name_to_index.keys())
        self._recorded_slots = np.asarray(
            [self.name_to_index[n] for n in self._recorded_names], dtype=np.int64)

    def resolve_force_models(self) -> None:
        """
        Rebuild the resolved force-model dispatch list from current mask state.

        Must be called after anything that changes `force_model_mask` or `active_mask` - exactly the
        same obligation `_refresh_active_indices` documents for the sibling/head caches, and for the
        same reason: `accelerations()` reads the cache built here instead of the masks, so a change
        that forgets this call would silently keep dispatching last configuration's models rather than
        failing loudly. `enable_force_model` calls this automatically; `_build_universe` calls it once
        at the end of build.
        """
        self._resolved_force_models, self._force_dispatch_idx = forces.resolve_force_models(
            self.active_mask, self.force_model_mask, self.max_capacity, self.force_model_params,
        )

        # Clear the whole accumulator on every re-resolve. `compose_accelerations` zeroes only the
        # rows it is about to write, so a body whose last model was just disabled would otherwise
        # keep reporting that model's final acceleration forever - its row is outside the new
        # dispatch set and nothing writes it again. In a sweep that runs "Keplerian + J2" then
        # "Keplerian", the second configuration would silently still carry J2. This is O(capacity),
        # but it runs once per configuration change, never per step, so step cost is unaffected.
        self.accel_accum.fill(0.0)

        # Bind the thrust set for `step()`'s per-step mass depletion. The loop is over resolved
        # models (tens at most, see forces.py), runs only on a configuration change, and leaves
        # `_thrust_idx` empty - so `step()` skips the depletion entirely - when nothing thrusts.
        self._thrust_idx = np.empty(0, dtype=np.int64)
        for rm in self._resolved_force_models:
            if rm.name == thrust.THRUST_MODEL:
                self._thrust_idx, self._thrust_params = rm.indices, rm.params

        self._refresh_cowell_plan()

    def _refresh_cowell_plan(self) -> None:
        """
        Decide, from data, whether `step()` may run the Cowell set through the fused compiled kernel.

        `kernels.cowell_rk4_step` hard-codes `point_mass_gravity` and `j2` (numba cannot dispatch
        over the Python kernel list `forces.compose_accelerations` walks), with per-body flags so a
        mixed arena - some Cowell bodies with J2, some without, some with neither - still qualifies.
        Two conditions disqualify the whole set, and the fallback is then the NumPy path for every
        Cowell body, not a per-body split: any Cowell body carrying a bit outside those two models,
        and any Cowell body whose parent is itself Cowell (the kernel reads a parent's row as fixed
        across the four stages; `RK4Integrator` would see the parent's stage candidates instead).

        Cached here, at configuration time, for the same reason `_force_dispatch_idx` is: the step
        must not pay a NumPy reduction over the mask to discover a configuration that only changes
        when `set_propagator`, `enable_force_model` or `resolve_force_models` is called. The
        `"j2"` coefficient array is bound here too, since `forces.resolve_force_models` allocates it
        lazily; a one-row dummy stands in until it exists, and the kernel only reads a row behind
        that body's `has_j2` flag.
        """
        idx = self._cowell_idx
        self._cowell_primaries = self.parent_indices[idx]

        pm_bit = np.uint64(1) << np.uint64(get_force_model(gravity.POINT_MASS_MODEL).bit)
        j2_bit = np.uint64(1) << np.uint64(get_force_model(geopotential.J2_MODEL).bit)
        np.not_equal(self.force_model_mask & pm_bit, np.uint64(0), out=self._cowell_has_point_mass)
        np.not_equal(self.force_model_mask & j2_bit, np.uint64(0), out=self._cowell_has_j2)

        foreign = (self.force_model_mask[idx] & ~(pm_bit | j2_bit)) != np.uint64(0)
        parent_is_cowell = np.isin(self._cowell_primaries, idx)
        self._cowell_fused_ok = bool(idx.size > 0 and not foreign.any() and not parent_is_cowell.any())

        j2_params = self.force_model_params.get(geopotential.J2_MODEL)
        self._cowell_j2_params = self._no_j2_params if j2_params is None else j2_params

    def enable_force_model(
        self,
        name: str,
        bodies: Union[int, Sequence[int], NDArray[np.integer[Any]], NDArray[np.bool_]],
        **coefficients: float,
    ) -> None:
        """
        Turn on a registered force model for the given bodies, and set any of its per-body
        coefficients.

        This is the sweep surface: two simulations that differ only in which models are enabled, or
        in what coefficients they carry, differ only in what this was called with. There is no other
        API a comparison needs - `force_model_mask` and `force_model_params` are both plain arrays,
        so a sweep harness can equally well assign them directly and call `resolve_force_models()`.

        `bodies` is an arena slot index, a sequence of slot indices, or a boolean mask shaped
        `(max_capacity,)`. Coefficients are matched against the model's declared `param_names` by
        keyword; an unknown keyword raises rather than silently being dropped.

        If the model registered a `validate_bodies` hook (`registry.BodyValidator`), it runs first,
        and a `ValueError` from it leaves the mask and coefficients untouched. `"j2"` uses this to
        refuse bodies whose parent is a barycentre. A `validate_coefficients` hook
        (`registry.CoefficientValidator`) runs next with the same guarantee, and also receives
        `coefficients`. `"third_body"` uses it to check its perturber slot.
        """
        model = get_force_model(name)
        idx: Any = bodies if isinstance(bodies, (int, np.integer)) else np.asarray(bodies)

        if model.validate_bodies is not None or model.validate_coefficients is not None:
            arr = np.asarray(idx)
            slots = np.flatnonzero(arr) if arr.dtype == np.bool_ else np.atleast_1d(arr).astype(np.int64)
            if model.validate_bodies is not None:
                model.validate_bodies(self, slots)
            if model.validate_coefficients is not None:
                model.validate_coefficients(self, slots, coefficients)

        bit_value = np.uint64(1) << np.uint64(model.bit)
        self.force_model_mask[idx] |= bit_value

        if coefficients:
            if not model.param_names:
                raise ValueError(f"force model '{name}' takes no parameters, got {list(coefficients)}")
            params = self.force_model_params.get(name)
            if params is None:
                params = np.zeros((self.max_capacity, model.n_params), dtype=np.float64)
                self.force_model_params[name] = params
            for key, value in coefficients.items():
                try:
                    col = model.param_names.index(key)
                except ValueError:
                    raise ValueError(
                        f"force model '{name}' has no parameter '{key}'; "
                        f"available: {model.param_names}"
                    ) from None
                params[idx, col] = value

        self.resolve_force_models()

    def accelerations(
        self, t: ScalarSeconds, state: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Total accumulated acceleration from every enabled force model, km/s^2, arena-indexed like
        `global_states`. Implements the `forces.AccelerationProvider` contract: an integrator holds a
        reference to this bound method and calls `sim.accelerations(t, state)` without knowing which
        models produced the result, how many there were, or how they were combined.

        `state` defaults to `self.global_states` (the arena's committed state) for direct inspection
        and testing; an integrator evaluating an intermediate sub-stage passes its own candidate state
        instead. See `forces.AccelerationProvider` for why that argument exists at all.

        The returned array is `self.accel_accum` itself, reused every call - copy it if a value must
        survive past the next call.
        """
        if state is None:
            state = self.global_states
        return forces.compose_accelerations(
            self._resolved_force_models, self._force_dispatch_idx, t, state,
            self.mu_array, self.parent_indices, self.accel_accum,
        )

    def _resolve_circular(self) -> None:
        """Identify and resolve DB circular dependencies (Binary Systems) by electing a head body based on mass and/or index."""
        idx = np.arange(self.max_capacity)
        p_idx = self.parent_indices
        gp_idx = p_idx[p_idx]

        circular_mask = self.active_mask & (gp_idx == idx) & (p_idx != idx)
        if np.any(circular_mask):
            logger.debug("Circular parent pairs detected: %s", circular_mask[self.active_mask])
            mu_self = self.mu_array
            mu_parent = self.mu_array[p_idx]

            head_election_mask = circular_mask & (
                (mu_self > mu_parent) | 
                ( (mu_self == mu_parent) & (idx < p_idx) )
            )

            self.is_head[head_election_mask] = True
            self.coe_states[head_election_mask, :] = 0.0

            self.parent_indices[circular_mask] = self.body_sys_map[circular_mask]

    def _build_sys_head_map(self) -> None:
        """
        Constructs an O(1) lookup array mapping every index to the head of their local system bubble.
        """
        # 1. Self reference for all entries
        idx = np.arange(self.max_capacity, dtype=np.int32)
        self.sys_head_map[:] = idx

        # 2. Find active heads
        heads_m = self.is_head & self.active_mask
        heads = idx[heads_m]

        if len(heads) > 0:
            # 3. What systems do these heads belong to, make a temp mapping
            bubbles = self.body_sys_map[heads_m]
            Bar_to_Hea = np.copy(idx)
            Bar_to_Hea[bubbles] = heads


            # 4. Route Siblings to heads (can be systems!)
            sibs_m = self.active_mask & (self.body_sys_map != idx) & ~self.is_head

            bar_sibs = self.body_sys_map[sibs_m]

            self.sys_head_map[sibs_m] = Bar_to_Hea[bar_sibs]

    def _topological_sort(self, array: NDArray[np.int32]) -> None:
        active_indices = np.where(self.active_mask)[0].astype(np.int32)
        # root = self.parent_indices[active_indices] == active_indices
        root = array[active_indices] == active_indices

        current_tier = active_indices[root]

        if len(current_tier) == 0:
            raise RuntimeError("Topology Error: No root detected amongst active indices")
        
        self.topological_tiers: List[NDArray[np.int32]] = [current_tier]

        processed_count = len(current_tier)
        total_active = len(active_indices)
  
        while processed_count < total_active:
            # next_layer_mask = np.isin(self.parent_indices[active_indices], current_tier)
            next_layer_mask = np.isin(array[active_indices], current_tier)
            already_processed = np.isin(active_indices, np.concatenate(self.topological_tiers))
            next_tier = active_indices[next_layer_mask & ~already_processed]

            if len(next_tier) == 0:
                raise RuntimeError("Topology Error: Disconnected Graph or circular dependency detected.")
            
            self.topological_tiers.append(next_tier)
            processed_count += len(next_tier)
            current_tier = next_tier

        logger.debug("Topological tiers: %s", self.topological_tiers)

    def _unfold_database_to_global(self) -> None:
        tier_0 = self.topological_tiers[0]
        self.global_states[tier_0] = 0.0

        # print(self.topological_tiers)

        for tier in self.topological_tiers[1:]:

            parents = self.parent_indices[tier]
            parent_sys = self.is_system[parents]
            mu_parents = self.mu_array[parents]

            m1 = ~parent_sys # Any body without a system as a parent
            # print(m1)
            if np.any(m1):
                idx_m1 = tier[m1]
                p_m1 = parents[m1]

                local_r, local_v, _ = fr.ReferenceFrames.coe_to_rv(self.coe_states[idx_m1], mu_parents[m1] + self.mu_array[idx_m1]) # Likely should include child mass too.
                self.global_states[idx_m1, :3] = self.global_states[p_m1, :3] + local_r
                self.global_states[idx_m1, 3:] = self.global_states[p_m1, 3:] + local_v

            m2 = parent_sys # Bodies with systems as parents
            # print(m2)
            if np.any(m2): # Needs to be clever, only collapse bodies in that system i.e. if there are multiple parent systems, don't sum all bodies to 1 parent
                idx_m2 = tier[m2]
                p_m2 = parents[m2]

                # 1. Filter between heads and siblings
                is_h = self.is_head[idx_m2]
                m2_heads = idx_m2[is_h]
                m2_sibs = idx_m2[~is_h]

                # 2. Pre-allocate local relative vectors
                local_r_rel = np.zeros((len(idx_m2), 3), dtype=np.float64)
                local_v_rel = np.zeros_like(local_r_rel)

                # 3. Compute True Relative Orbit for siblings

                if len(m2_sibs) > 0:
                    heads_of_sibs = self.sys_head_map[m2_sibs]
                    mu_calc = self.mu_array[m2_sibs] + self.mu_array[heads_of_sibs]

                    sib_r, sib_v, _ = fr.ReferenceFrames.coe_to_rv(self.coe_states[m2_sibs], mu_calc)

                    # Slot back into sibling positions
                    local_r_rel[~is_h] = sib_r
                    local_v_rel[~is_h] = sib_v
                

                # 4. Collapse multiple bodies into N-Systems
                system_r_sums = np.zeros((self.max_capacity, 3), dtype=np.float64)
                system_v_sums = np.zeros((self.max_capacity, 3), dtype=np.float64)
                system_masses = np.zeros(self.max_capacity, dtype=np.float64)

                body_masses = self.mu_array[idx_m2]

                np.add.at(system_r_sums, p_m2, local_r_rel * body_masses[:, None])
                np.add.at(system_v_sums, p_m2, local_v_rel * body_masses[:, None])
                np.add.at(system_masses, p_m2, body_masses)

                # 5. Compute reflexive kick for each valid system
                valid_sys = system_masses > 0
                kick_r = np.zeros_like(system_r_sums)
                kick_v = np.zeros_like(system_v_sums)

                kick_r[valid_sys] = -system_r_sums[valid_sys] / system_masses[valid_sys, None]
                kick_v[valid_sys] = -system_v_sums[valid_sys] / system_masses[valid_sys, None]

                # 6. Apply kick to all bodies in tier
                shifted_local_r = local_r_rel + kick_r[p_m2]
                shifted_local_v = local_v_rel + kick_v[p_m2]

                # 7. Global position by adding parent system global vector
                self.global_states[idx_m2, :3] = self.global_states[p_m2, :3] + shifted_local_r
                self.global_states[idx_m2, 3:] = self.global_states[p_m2, 3:] + shifted_local_v

                # 8. Dynamic reparenting, point siblings to head
                if len(m2_sibs) > 0:
                    self.parent_indices[m2_sibs] = self.sys_head_map[m2_sibs]

    def _recalculate_all_barycenters(self) -> None:
        """
        Calculate the true global Cartesian center of mass for system based on updated
        absolute positions of its member bodies.
        """
        # 1. Identify all systems and bodies
        sys_mask = self.is_system & self.active_mask
        body_mask = ~self.is_system & self.active_mask

        # Reset system states and mu values to 0.0
        self.global_states[sys_mask] = 0.0
        self.mu_array[sys_mask] = 0.0

        # Temporary mass-weighted array for all bodies.
        mass_weighted_states = np.zeros_like(self.global_states)
        mass_weighted_states[body_mask] = self.global_states[body_mask] * self.mu_array[body_mask, None]

        for tier in reversed(self.topological_tiers):
            # 2. Identify bodies that belong to a system (do not self reference)
            has_sys = self.body_sys_map[tier] != tier
            idx = tier[has_sys]

            if len(idx) > 0:
                # 3. Check systems are actual barycenters, then prepare to calculate their states.
                parent_sys = self.body_sys_map[idx]
                valid_mask = self.is_system[parent_sys] # Check the body's assigned system is an actual barycenter!
                valid_idx = idx[valid_mask]             # Filter
                valid_parents = parent_sys[valid_mask]

                if len(valid_idx) > 0:
                    # 4. Compute the mass-weighted sum of all bodies in the system, and the total mass of the system.
                    np.add.at(self.mu_array, valid_parents, self.mu_array[valid_idx])
                    np.add.at(mass_weighted_states, valid_parents, mass_weighted_states[valid_idx])

        # 5. Normalize the mass-weighted states to get the true barycenter positions and velocities.
        valid_sys = (self.mu_array > 0) & sys_mask  # Avoid divide by zeros
        self.global_states[valid_sys] = mass_weighted_states[valid_sys] / self.mu_array[valid_sys, None]    # Their mu should be a sum of all mu now.

    def _rehydrate_coes(self) -> None:
        """
        Calculate normalized local Cartesian vectors then generates mathematically accurate Classical Orbital Elements.
        Current Definition uses Two-Body mass sum for mu (mu_parent + mu_child).
        """
        # 1. Compute local states, clean up heads and roots data.
        # Self-reference made this very simple, if the node self references then the local state will be 0.0
        self.local_states[self.active_mask] = (                     # All local states are relative to their system bubble barycenter.
            self.global_states[self.active_mask] - 
            self.global_states[self.body_sys_map[self.active_mask]]
        )

        self.coe_states[self.is_head, :] = 0.0 # Heads move reflexively to their siblings within the system. Coe's are thus meaningless.
        self.coe_states[self.active_mask & (self.parent_indices == np.arange(self.max_capacity))] = 0.0 # All root nodes haven no parent hence no orbit parameters.
        valid_mask = self.active_mask & ~self.is_head & (self.parent_indices != np.arange(self.max_capacity))   # Any body that actually has an orbit.
        # Pick anything that is active, not a head and isn't a self referenced object (root)

        # print(rel_r, rel_v)
        if np.any(valid_mask): # This needs to be refactored. They need to compute their distance from the head instead!
            # 2. Find parent and child indices, then compute relative positions, velocites, and summed orbital paramaters between them.
            parents = self.parent_indices[valid_mask]
            children = np.where(valid_mask)[0].astype(np.uint32)

            rel_r = self.local_states[children, :3].copy()      # Copy the current local states.
            rel_v = self.local_states[children, 3:].copy()

            barycentric = self.body_sys_map[children] != parents    # If they belong to a system, which is not explicitly the head body...

            np.add.at(rel_r, barycentric, -self.local_states[parents[barycentric], :3]) # Then find their vectors relative to the system head instead.
            np.add.at(rel_v, barycentric, -self.local_states[parents[barycentric], 3:])

            mu_calc = self.mu_array[parents] + self.mu_array[children]  # Two-body mass sum for mu, as per standard definition.

            # 3. Compute the Classical Orbital Elements from the relative vectors and mu values and update the succesful entries.
            new_coes, success = fr.ReferenceFrames.rv_to_coe(
                rel_r,
                rel_v,
                mu_calc# mu_two_body
            )


            valid_idx = np.where(valid_mask)[0][success]
            self.coe_states[valid_idx] = new_coes[success]


    def _zero_roots(self) -> None:
        """Adjust global state such that the root system is at the origin, shifting all bodies in the same ancestry by the same vector."""
        # 1. Identify all active root nodes (self-referencing), Store their position vectors.
        roots = self.body_sys_map == np.arange(self.max_capacity)
        active_roots = roots & self.active_mask

        if not np.any(active_roots):
            return
        
        shifts = np.zeros_like(self.global_states)
        shifts[active_roots] = self.global_states[active_roots]

        for tier in self.topological_tiers:
            # 2. Identify all children of the current tier, copy the shift vector from their parent to them.
            has_system = self.body_sys_map[tier] != tier
            children = tier[has_system]

            if len(children) > 0:
                systems = self.body_sys_map[children]
                shifts[children] = shifts[systems]

        # 3. Apply the shift to all active bodies, effectively moving the root system to the origin.
        self.global_states[self.active_mask] -= shifts[self.active_mask]

    def calc_global(self) -> None:
        """
        Recalculate the global states for all bodies based on their local positions and system bubble barycenters.
        This is done topologically, ensuring that parents are processed before their children.

        Dispatches to the compiled kernel when available. The two are equivalent by construction -
        the kernel performs the same additions in the same topological order - and that equivalence
        is asserted in `tests/validation/test_kernel_equivalence.py`.
        """
        if self.use_compiled_kernel:
            calc_global_states(
                self._topo_order, self._n_roots, self.body_sys_map,
                self.local_states, self.global_states,
            )
            return

        tier_0 = self.topological_tiers[0]
        self.global_states[tier_0] = 0.0

        for tier in self.topological_tiers[1:]:

            parents = self.body_sys_map[tier]

            self.global_states[tier, :3] = self.global_states[parents, :3] + self.local_states[tier, :3]
            self.global_states[tier, 3:] = self.global_states[parents, 3:] + self.local_states[tier, 3:]

        return

    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)

    def set_propagator(
        self,
        bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
        propagator_type: PropagatorType,
        mean_seed: bool = False,
        **coefficients: float,
    ) -> None:
        """
        Assign a propagator to the given bodies, per body, as data.

        `step()` reads `propagator_type` (via the cached index arrays this rebuilds) rather than
        selecting Keplerian unconditionally - `registry._PROPAGATOR_REGISTRY` is what it dispatches
        Cowell through (`propagators.CowellPropagator`). `PropagatorType.KEPLERIAN` (the default for
        every slot), `PropagatorType.COWELL` and `PropagatorType.SECULAR_J2` currently drive that
        dispatch.

        `**coefficients` is only meaningful for `SECULAR_J2` (below); passing any for another
        propagator type raises, the same strictness `enable_force_model` applies to an unknown
        keyword.

        **Cowell is restricted to active bodies that are not a system head, not a system barycenter,
        and not their own kinematic bubble** (`body_sys_map[i] != i`). A head or barycenter's motion
        is the reflex kick `kepler_propagate` / `_recalculate_all_barycenters` compute for the *whole*
        bubble from *every* sibling's mass and position; replacing it with an independently integrated
        Cartesian state would corrupt every other body in that bubble, not just the one reassigned. A
        kinematic root has no bubble to be relative to at all: `calc_global` unconditionally zeroes
        tier-0 slots every step, which would silently discard a root Cowell body's motion rather than
        raise. Both are enforced here, explicitly, rather than left to produce a trajectory that looks
        plausible and is not - see `docs/architecture.md`'s Cowell section for the full reasoning.

        **Cowell also requires `mu_array[body] == 0.0`.** A body's mass only reaches the rest of the
        arena through `kepler_propagate`'s barycentric accumulation (pass 1: every active, non-Cowell
        sibling's mass-weighted position is summed onto its head), which a Cowell body never passes
        through - `_kepler_sib_idx` excludes it. A massive Cowell body would therefore silently stop
        contributing to its own head's reflex kick the moment it was reassigned, corrupting every other
        sibling in the bubble exactly like the head/barycenter case above, just via mass instead of
        position. Rather than leave that as a documented-but-permitted gap, it is rejected here; a
        massless body (`mu=0`, the common case for a satellite or a test secondary) is unaffected by
        this restriction since it never contributed to a reflex kick in the first place.

        **`SECULAR_J2` carries the same head/barycenter/kinematic-bubble and `mu == 0` restrictions as
        Cowell, for the same reasons** - see `propagators.SecularJ2Propagator`'s docstring, which is
        this propagator's counterpart of `docs/architecture.md`'s Cowell section. Two further, specific
        restrictions:

        - **The Keplerian parent must not be a barycentre** (`geopotential.barycentre_parented`): a
          barycentre's J2 is physically meaningless, and `geopotential.j2_kernel` cannot detect the case
          itself (no `is_system` in its signature) - the same guard `enable_force_model("j2", ...)`
          documents as the caller's responsibility, enforced here instead since this propagator's
          coefficients are never optional.
        - **`0 <= e < 1` (a closed, elliptical orbit)**: the secular rates derive from mean motion
          `n = sqrt(mu/a^3)`, undefined for a parabolic or hyperbolic orbit.

        `**coefficients` must supply exactly `j2` and `r_eq` - unlike `enable_force_model("j2", ...)`,
        where omitting them is a documented silent no-op, `SECULAR_J2` has no meaning without them, so
        they are mandatory here. They are written into `force_model_params["j2"]` - the same array and
        column layout `geopotential.j2_kernel` reads (`param_names = ("j2", "r_eq")`) - without setting
        the `"j2"` force-model mask bit, so this never adds the body to `accelerations()`'s dispatch set;
        see `propagators.SecularJ2Propagator`'s docstring for why that storage choice was made. The three
        secular rates are then derived once, by `propagators.secular_j2_rates`, and cached in
        `self._secular_j2_rates` rather than recomputed every step.

        `mean_seed` (`SECULAR_J2` only; a sweep-configuration flag, not a coefficient) replaces the
        seeded semi-latus rectum `p` with a first-order estimate of the *mean* value before the rates
        above are derived from it - see `propagators.mean_seeded_p` for the derivation and citation.
        Defaults to `False`, which leaves `p` at its osculating, as-given value: bit-identical to this
        propagator's behaviour before this option existed. `e` and `i` are never converted; only `p`
        (and so the cached mean motion) is - `CLAUDE.md`'s relaxed osculating<->mean rule permits only
        this propagator's own first-order theory correcting its own seed, not a general conversion.
        """
        idx = np.atleast_1d(np.asarray(bodies, dtype=np.int64))

        if mean_seed and propagator_type != PropagatorType.SECULAR_J2:
            raise ValueError(
                f"mean_seed is only meaningful for PropagatorType.SECULAR_J2, got "
                f"{PropagatorType(propagator_type).name}."
            )

        if propagator_type == PropagatorType.COWELL:
            if coefficients:
                raise ValueError(
                    f"Cowell propagation takes no coefficients, got {sorted(coefficients)}."
                )
            disallowed = (
                ~self.active_mask[idx] | self.is_head[idx] | self.is_system[idx] |
                (self.body_sys_map[idx] == idx)
            )
            if np.any(disallowed):
                raise ValueError(
                    f"Cowell propagation is restricted to active, non-head, non-barycenter bodies "
                    f"that are not their own kinematic bubble; slot(s) {idx[disallowed].tolist()} do "
                    f"not qualify. See Simulation.set_propagator's docstring."
                )
            massive = idx[self.mu_array[idx] != 0.0]
            if massive.size > 0:
                raise ValueError(
                    f"Cowell propagation requires mu == 0; slot(s) {massive.tolist()} carry nonzero "
                    f"mass and would silently stop contributing to their head's reflex kick. See "
                    f"Simulation.set_propagator's docstring."
                )

        elif propagator_type == PropagatorType.SECULAR_J2:
            disallowed = (
                ~self.active_mask[idx] | self.is_head[idx] | self.is_system[idx] |
                (self.body_sys_map[idx] == idx)
            )
            if np.any(disallowed):
                raise ValueError(
                    f"secular-J2 propagation is restricted to active, non-head, non-barycenter bodies "
                    f"that are not their own kinematic bubble; slot(s) {idx[disallowed].tolist()} do "
                    f"not qualify. See Simulation.set_propagator's docstring."
                )

            bary_parented = geopotential.barycentre_parented(self.is_system, self.parent_indices, idx)
            if bary_parented.size > 0:
                raise ValueError(
                    f"secular-J2 propagation requires a non-barycentre Keplerian parent - a "
                    f"barycentre's J2 is meaningless, and the kernel cannot detect it (see "
                    f"geopotential.py); slot(s) {bary_parented.tolist()} are parented to a barycentre."
                )

            missing = {"j2", "r_eq"} - set(coefficients)
            if missing:
                raise ValueError(
                    f"secular-J2 propagation requires coefficients {sorted(missing)}, matching the "
                    f"'j2' force model's (j2, r_eq); got {sorted(coefficients)}."
                )
            extra = set(coefficients) - {"j2", "r_eq"}
            if extra:
                raise ValueError(
                    f"secular-J2 propagation takes only 'j2' and 'r_eq'; got unexpected "
                    f"{sorted(extra)}."
                )

            e = self.coe_states[idx, COEIndex.E]
            bad_e = idx[(e < 0.0) | (e >= 1.0) | ~np.isfinite(e)]
            if bad_e.size > 0:
                raise ValueError(
                    f"secular-J2 propagation requires a closed elliptical orbit (0 <= e < 1) - the "
                    f"secular rates derive from mean motion n = sqrt(mu/a^3), undefined for an open "
                    f"orbit; slot(s) {bad_e.tolist()} do not qualify."
                )

            massive = idx[self.mu_array[idx] != 0.0]
            if massive.size > 0:
                raise ValueError(
                    f"secular-J2 propagation requires mu == 0; slot(s) {massive.tolist()} carry "
                    f"nonzero mass and would silently stop contributing to their head's reflex kick, "
                    f"for the same reason Cowell requires it. See Simulation.set_propagator's "
                    f"docstring."
                )

            j2_model = get_force_model(geopotential.J2_MODEL)
            params = self.force_model_params.get(geopotential.J2_MODEL)
            if params is None:
                params = np.zeros((self.max_capacity, j2_model.n_params), dtype=np.float64)
                self.force_model_params[geopotential.J2_MODEL] = params
            params[idx, 0] = coefficients["j2"]
            params[idx, 1] = coefficients["r_eq"]

            if mean_seed:
                self.coe_states[idx, COEIndex.P] = mean_seeded_p(self.coe_states, params, idx)

            self._secular_j2_rates[idx] = secular_j2_rates(
                self.coe_states, self.mu_array, self.parent_indices, params, idx,
            )

        elif coefficients:
            raise ValueError(
                f"propagator {PropagatorType(propagator_type).name} takes no coefficients, got "
                f"{sorted(coefficients)}."
            )

        self.propagator_type[idx] = np.uint8(propagator_type)
        self._refresh_active_indices()

    def _advance(self, dt: ScalarSeconds) -> None:
        """
        Advance the arena by `dt` and the clock with it. The whole of the physics of one step.

        Split out of `step()` so a scheduled impulsive manoeuvre can cut a step in two at its exact
        epoch (`step()` calls this once per sub-interval); `step()` owns the manoeuvre queue and the
        history snapshot, and this owns everything else. It is not a public entry point: it records no
        history, so calling it directly leaves a gap in `history()`.

        Keplerian siblings and heads propagate exactly as before - `_kepler_sib_idx` / `_head_idx`
        hold the same slots `_sib_idx` / `_head_idx` would if no Cowell body existed, so the two
        propagation paths remain held elementwise equivalent by
        `tests/validation/test_kernel_equivalence.py`, and a Keplerian body's result is bit-identical
        regardless of whether a Cowell body is present elsewhere in the arena
        (`tests/validation/test_cowell_propagator.py`'s isolation test).

        Cowell siblings (`propagator_type == COWELL`, set via `set_propagator`) integrate their state
        **relative to their own gravitating parent** (`parent_indices`), through
        `registry.get_propagators()`'s entry for `PropagatorType.COWELL`
        (`propagators.CowellPropagator`), which delegates to `self._cowell_integrator`. This runs
        *before* the Keplerian propagation and `calc_global()` below, using the arena state as
        committed at the *start* of this step as the frame `provider` evaluates forces against - see
        `integrators.py`'s module docstring for why that is exact for every force model registered so
        far (each depends only on position relative to `parent_indices`, so the parent's own motion
        never enters the relative equation of motion at all) and what would remain an approximation for
        a hypothetical future model that is not. The integrator's result is therefore expressed
        relative to the parent's *start-of-step* position; it is saved, and once `calc_global()` below
        has propagated every Keplerian body (including the parent) to its true end-of-step position,
        the saved relative state is re-based onto that fresh position - this is what makes a Cowell
        body correctly follow a parent that is itself accelerating, rather than silently assuming a
        stationary one. `local_states` is then rebuilt from the difference against `body_sys_map` (the
        kinematic bubble reference, generally *not* the same slot as `parent_indices` - the Moon is the
        canonical case, see `docs/architecture.md`) so the arena's stated invariant - `local_states[i]`
        relative to `global_states[body_sys_map[i]]` - holds for Cowell bodies exactly as it does for
        Keplerian ones once `step()` returns.

        Secular-J2 siblings (`propagator_type == SECULAR_J2`) are analytic, like the Keplerian path,
        so there is no integrator and no state to snapshot: `propagators.SecularJ2Propagator.propagate`
        (or its compiled twin `kernels.secular_j2_propagate`) advances RAAN, argument of periapsis and
        mean anomaly by their cached rates and writes the resulting state, relative to `parent_indices`,
        into `self._secular_j2_rel`. Once `calc_global()` has produced every Keplerian parent's true
        end-of-step position, that relative state is added onto it and `local_states` rebuilt against
        `body_sys_map` - the same re-basing pattern Cowell uses above, for the same reason (the two
        parent graphs diverge; see `docs/architecture.md`), just without a `parent_state_at_start` to
        subtract first, since nothing here was integrated relative to a snapshot.

        Finally, any body carrying the `"thrust"` force model burns propellant - `thrust.deplete_mass`
        on the cached `_thrust_idx`, *after* the propagation, so all four RK4 stages of this step read
        the mass the step began with. See `thrust.py` for why the mass lives in that model's parameter
        array and what freezing it across a step costs.
        """
        cowell_idx = self._cowell_idx
        if cowell_idx.size > 0:
            if self.use_compiled_kernel and self._cowell_fused_ok:
                # Fused twin: leaves global_states[cowell_idx] exactly as RK4Integrator would and
                # writes the parent-relative result straight into _cowell_rel. See
                # `_refresh_cowell_plan` for when this branch is available.
                cowell_rk4_step(
                    float(dt), self.global_states, self.mu_array, self.parent_indices, cowell_idx,
                    self._cowell_has_point_mass, self._cowell_has_j2, self._cowell_j2_params,
                    self._cowell_rel,
                )
            else:
                parent_state_at_start = self.global_states[self._cowell_primaries].copy()
                cowell_propagator = get_propagators()[int(PropagatorType.COWELL)]
                cowell_propagator.propagate(
                    dt=dt, integrator=self._cowell_integrator, provider=self.accelerations,
                    t=self.t, state=self.global_states, indices=cowell_idx,
                    primaries=self._cowell_primaries,
                )
                # The integrator's result is (parent_state_at_start + relative_state); subtracting
                # the start-of-step parent reference recovers the pure relative state.
                self._cowell_rel[cowell_idx] = self.global_states[cowell_idx] - parent_state_at_start

        if self._secular_j2_idx.size > 0:
            if self.use_compiled_kernel:
                secular_j2_propagate(
                    float(dt), self.coe_states, self.mu_array, self.parent_indices,
                    self._secular_j2_rates, self._secular_j2_idx, self._secular_j2_rel,
                )
            else:
                SecularJ2Propagator.propagate(
                    dt=dt, coe_states=self.coe_states, mu_array=self.mu_array,
                    parent_indices=self.parent_indices, rates=self._secular_j2_rates,
                    indices=self._secular_j2_idx, rel_out=self._secular_j2_rel,
                )

        if self.use_compiled_kernel:
            kepler_propagate(
                float(dt), self.coe_states, self.local_states, self.mu_array,
                self.parent_indices, self.body_sys_map, self.sys_head_map, self.is_system,
                self._kepler_sib_idx, self._head_idx, self._kick, self._accum,
            )
        else:
            KeplerianPropagator.propagate(dt=dt, primary_states=self.coe_states, secondary_states=self.local_states, mu_array=self.mu_array,
                                          parent_indices=self.parent_indices, active_mask=self.active_mask, is_head=self.is_head, is_system=self.is_system,
                                          body_sys_map=self.body_sys_map, sys_head_map=self.sys_head_map, sib_mask=self._kepler_sib_mask)
        self.calc_global()

        # Adding each parent's now-fresh (post calc_global) position to the saved relative state
        # places the body correctly regardless of how far its parent moved this step. Cowell first,
        # then secular J2, so a secular body parented by a Cowell one reads the re-based parent.
        if cowell_idx.size > 0:
            self._rebase(cowell_idx, self._cowell_rel)
        if self._secular_j2_idx.size > 0:
            self._rebase(self._secular_j2_idx, self._secular_j2_rel)

        # Propellant burn, after the propagation: every RK4 stage of this step saw the mass the step
        # began with, which is what makes the mass half of the scheme first-order while the position
        # half stays fourth-order. See thrust.py's module docstring and the Delta-v bias asserted in
        # `tests/validation/test_thrust.py`.
        if self._thrust_idx.size > 0:
            thrust.deplete_mass(self._thrust_idx, self._thrust_params, dt)

        self.t += dt

    def step(self, dt: ScalarSeconds) -> None:
        """
        Advance the arena by `dt`, applying any scheduled impulsive manoeuvre at its exact epoch, and
        record one history snapshot.

        With an empty manoeuvre queue **and** no registered events this is `_advance(dt)` plus the
        snapshot - the behaviour every existing caller already has, bit for bit.

        **Event splitting.** Registered events (`add_event`, `events.py`) cut the step wherever a
        continuous event function changes sign - the same cut this method already makes at a scheduled
        impulse's epoch, but at an epoch that has to be *found* rather than read off a queue. The
        search is per sub-interval, so events and manoeuvres compose: each manoeuvre sub-step is
        itself scanned for crossings. A sub-interval with no crossing is advanced exactly as it is
        today, bit for bit - the detection is two pure reads of the arena around the same single
        `_advance` call. See `_advance_with_events`.

        **Step splitting.** A manoeuvre scheduled at an epoch inside `(t, t + dt]` does not wait for
        the step boundary: the step is cut at that instant, so the timing of an impulse is exact rather
        than quantised to `dt`, which is what lets a Hohmann second burn be placed at the transfer
        apoapsis instead of at the nearest multiple of the step size. Several manoeuvres inside one
        step are applied in epoch order, each with its own sub-step; one scheduled at or before `t`
        (including at `t = 0`, before the first step) is applied immediately with no sub-step. `self.t`
        is set to `t + dt` exactly at the end, so splitting cannot make the clock drift by the rounding
        of the sub-intervals.

        **What splitting costs a body that is not manoeuvring.** For an analytic body - Keplerian or
        secular-J2 - nothing: propagating `h1` then `h2` is the same closed-form advance as propagating
        `h1 + h2`, to the Kepler solver's convergence tolerance and rounding (~1e-10 km on a LEO orbit,
        asserted in `tests/validation/test_manoeuvres.py`). For a **Cowell** body it is not exactly
        nothing and cannot be: RK4 evaluates its stages at quadrature nodes fixed by the step size, so
        cutting one step in two moves those nodes and changes that step's truncation error by an amount
        of the order of the local truncation error itself, `~r (n h)^5 / 120`. That is far below the
        integration error the body already carries, it is a one-off at the split rather than a per-step
        bias, and it is asserted against that derived bound in the same test file rather than assumed
        negligible.
        """
        if not self._manoeuvres and not self._events:
            self._advance(dt)
            if self.record_history:
                self._record_state()
            return

        t_end = float(self.t) + float(dt)
        due = manoeuvres.due_before(self._manoeuvres, t_end)
        for _ in range(due):
            m = self._manoeuvres.pop(0)
            sub = m.epoch_s - float(self.t)
            if sub >= manoeuvres.MIN_SUBSTEP_S:
                self._advance_with_events(sub)
            self.apply_delta_v(m.bodies, m.dv_rsw)

        remaining = t_end - float(self.t)
        if remaining >= manoeuvres.MIN_SUBSTEP_S:
            self._advance_with_events(remaining)

        # The sub-intervals sum to `dt` only up to rounding; pin the clock so a long run with many
        # split steps stays on the same time grid an unsplit run would.
        self.t = t_end
        if self.record_history:
            self._record_state()

    def schedule_delta_v(
        self,
        bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
        dv_rsw: Sequence[float] | NDArray[np.float64],
        epoch_s: ScalarSeconds,
        label: str = "",
    ) -> manoeuvres.Manoeuvre:
        """
        Schedule an impulsive Delta-v (RSW, km/s) to be applied at simulation time `epoch_s` seconds.

        `step()` cuts its step at that instant, so the impulse lands at the epoch asked for rather than
        at the next step boundary - see `step()` for what that costs a body that is not manoeuvring,
        and `manoeuvres.py` for the model, the frame convention (`(0, dv, 0)` is prograde) and the
        per-propagator handling.

        The bodies are validated **now**, not at the epoch, so a profile that names a head or a
        barycentre fails at the point the mistake was made. `dv_rsw` is `(3,)` for one Delta-v shared by
        every named body, or `(len(bodies), 3)` for one each.

        An epoch at or before the current time is applied on the next `step()` with no sub-step - so
        `schedule_delta_v(..., epoch_s=0.0)` on a fresh simulation burns before any propagation, which
        is how a departure burn is expressed. Returns the queued `Manoeuvre` so a caller can hold onto
        it (it is frozen plain data; a mission profile is a list of them).
        """
        idx = self._manoeuvre_slots(bodies)
        m = manoeuvres.Manoeuvre(
            epoch_s=float(epoch_s), bodies=idx, dv_rsw=self._manoeuvre_dv(dv_rsw, idx.size), label=label,
        )
        self._manoeuvres.append(m)

        # Sorted on insert rather than searched on use: a mission profile is tens of entries, the sort
        # is stable (so two manoeuvres at one epoch keep their scheduling order), and `step()` then
        # only has to look at the front of the list.
        self._manoeuvres.sort(key=lambda entry: entry.epoch_s)
        return m

    @property
    def pending_manoeuvres(self) -> tuple[manoeuvres.Manoeuvre, ...]:
        """Scheduled manoeuvres not yet applied, in epoch order. A tuple: the queue is `step()`'s."""
        return tuple(self._manoeuvres)

    def clear_manoeuvres(self) -> None:
        """Drop every scheduled manoeuvre. Applied ones are already in the state and are unaffected."""
        self._manoeuvres.clear()

    # ==============================================================================================
    # Event-driven step splitting - see events.py
    # ==============================================================================================

    def add_event(self, event: events.Event) -> events.Event:
        """
        Register an `events.Event`, so `step()` cuts its step wherever that event's function changes
        sign.

        The canonical use is the cylindrical shadow terminator, which is a genuine discontinuity in
        the acceleration and costs RK4 its convergence order when a step straddles it::

            sim.enable_force_model("srp", sat, ..., shadow_model=SHADOW_MODEL_CYLINDRICAL)
            sim.add_event(events.shadow_event(sim))

        Events are **opt-in and cost nothing when none are registered**: `step()` tests the list once
        and falls straight through to `_advance`. The first `add_event` allocates the snapshot buffers
        the root find needs; they are sized to `max_capacity` and never re-allocated, like every other
        piece of arena scratch.

        Returns the event, so a caller can hold onto it. See `events.py` for the interface, for why
        detection is per body while the split is per arena, and for what a crossing costs.
        """
        if event.bodies.size == 0:
            raise ValueError(f"event {event.name!r} names no bodies; it could never fire.")
        if bool(((event.bodies < 0) | (event.bodies >= self.max_capacity)).any()):
            raise ValueError(
                f"event {event.name!r} names slot(s) {event.bodies.tolist()} outside the arena "
                f"(capacity {self.max_capacity})."
            )
        if event.direction not in (-1, 0, 1):
            raise ValueError(
                f"event {event.name!r}: direction={event.direction!r} must be -1 (positive to "
                f"negative), +1 (negative to positive) or 0 (both)."
            )
        if not event.tol_s > 0.0:
            raise ValueError(
                f"event {event.name!r}: tol_s={event.tol_s!r} must be positive; it is the bracket "
                f"width in seconds the crossing time is converged to "
                f"(events.DEFAULT_EVENT_TOL_S is {events.DEFAULT_EVENT_TOL_S})."
            )

        if self._event_snap_global.size == 0:
            self._event_snap_global = np.zeros((self.max_capacity, 6), dtype=np.float64)
            self._event_snap_local = np.zeros((self.max_capacity, 6), dtype=np.float64)
            self._event_snap_coe = np.zeros((self.max_capacity, 6), dtype=np.float64)
            self._event_snap_thrust = np.zeros(
                (self.max_capacity, len(thrust.THRUST_PARAM_NAMES)), dtype=np.float64)

        self._events.append(event)
        # One concatenated direction vector, in the same block order `_event_values` returns, cached
        # here rather than rebuilt per step: the block lengths are fixed at construction.
        self._event_directions = np.concatenate(
            [np.full(e.bodies.size, float(e.direction), dtype=np.float64) for e in self._events])
        self._event_tolerances = np.concatenate(
            [np.full(e.bodies.size, float(e.tol_s), dtype=np.float64) for e in self._events])
        return event

    @property
    def registered_events(self) -> tuple[events.Event, ...]:
        """The registered events, in registration order. A tuple: the list is `step()`'s."""
        return tuple(self._events)

    def clear_events(self) -> None:
        """Drop every registered event. Steps stop splitting; nothing already integrated changes."""
        self._events.clear()
        self._event_directions = np.empty(0, dtype=np.float64)
        self._event_tolerances = np.empty(0, dtype=np.float64)

    @property
    def event_splits(self) -> int:
        """How many times an event has cut a step since this simulation was built. A diagnostic."""
        return self._event_splits

    @property
    def event_evaluations(self) -> int:
        """
        Trial propagations spent locating crossings since this simulation was built.

        This is the cost of the feature: each one is a full `_advance` of the arena over a candidate
        sub-interval, plus the snapshot restore. `event_evaluations / event_splits` is the mean
        iteration count of the root find - about 10 to 12 at the default tolerance from a 10 s step.
        """
        return self._event_evaluations

    def _event_values(self) -> NDArray[np.float64]:
        """
        Every registered event's function, evaluated on the current arena and concatenated.

        One block per event, in registration order, matching `_event_directions`. A **pure read**:
        this is called at trial times during a root find, and any mutation here would leak into the
        restored state. Allocates one `(K,)` array per call, where `K` is the total event-body count
        - not an arena-sized quantity, and not a per-step cost unless events are registered.
        """
        return np.concatenate([e.function(self, e.bodies) for e in self._events])

    def _set_event_latches(self, clearance_sign: Optional[NDArray[np.float64]]) -> None:
        """
        Pin every latching event's model to the branch it starts the sub-interval on, or release.

        `clearance_sign` is the concatenated `sign(g)` vector in `_event_values` order, sliced back
        into each event's own block; `None` releases. Events with no `latch` are skipped - only a
        model with a genuine discontinuity needs one. See `_advance_with_events` for why.
        """
        offset = 0
        for e in self._events:
            k = int(e.bodies.size)
            if e.latch is not None:
                e.latch(self, e.bodies,
                        None if clearance_sign is None else clearance_sign[offset:offset + k])
            offset += k

    def _capture_event_state(self) -> None:
        """
        Snapshot everything `_advance` mutates, into the pre-allocated buffers, so a trial
        propagation can be undone exactly.

        `global_states`, `local_states` and `coe_states` are the arena's state of record; the clock is
        the fourth; and `force_model_params["thrust"]`'s mass column is the fifth, because
        `thrust.deplete_mass` is state mutated once per `_advance` (see `CLAUDE.md`). Scratch is not
        captured - see the comment in `__init__` for which arrays and why.
        """
        np.copyto(self._event_snap_global, self.global_states)
        np.copyto(self._event_snap_local, self.local_states)
        np.copyto(self._event_snap_coe, self.coe_states)
        self._event_snap_t = float(self.t)
        n = self._thrust_idx.size
        if n > 0:
            np.take(self._thrust_params, self._thrust_idx, axis=0, out=self._event_snap_thrust[:n])

    def _restore_event_state(self) -> None:
        """Undo every `_advance` since the last `_capture_event_state`, exactly. The inverse of it."""
        np.copyto(self.global_states, self._event_snap_global)
        np.copyto(self.local_states, self._event_snap_local)
        np.copyto(self.coe_states, self._event_snap_coe)
        self.t = self._event_snap_t
        n = self._thrust_idx.size
        if n > 0:
            self._thrust_params[self._thrust_idx] = self._event_snap_thrust[:n]

    def _advance_with_events(self, dt: ScalarSeconds) -> None:
        """
        `_advance(dt)`, cut at every event crossing inside the interval.

        With no registered events this **is** `_advance(dt)` - one branch, no snapshot, no evaluation.
        With events registered but no crossing in this interval the arena still takes exactly one
        `_advance(dt)`, bit for bit: the detection is `_event_values()` before and after, both pure
        reads, and the speculative advance is *kept* rather than repeated. That is the property every
        other test in the suite depends on, and it is asserted directly in
        `tests/validation/test_events.py`.

        When a crossing is found the speculative advance is rolled back and the crossing *bracket*
        located by `events.locate_crossing` on the scalar reduction `events.reduce_to_scalar` - see
        `events.py` for why one scalar suffices for any number of crossing bodies. The arena is then
        advanced across that bracket in the three pieces described at the call site (the reason there
        are three, and not two, is the load-bearing detail of this method), and the remainder is
        re-scanned, so several crossings in one interval are all resolved.

        Raises `ValueError` if the interval needs more than `max_event_splits` cuts. Silently dropping
        the remaining crossings would reintroduce exactly the non-raising, plausible-looking error
        this machinery exists to remove; the arena is left part-advanced, and the fix is a smaller
        `dt` or a larger `max_event_splits`.
        """
        if not self._events:
            self._advance(dt)
            return

        remaining = float(dt)
        splits = 0
        while remaining >= manoeuvres.MIN_SUBSTEP_S:
            self._capture_event_state()
            g0 = self._event_values()
            self._advance(remaining)
            crossed = events.crossing_indices(g0, self._event_values(), self._event_directions)
            if crossed.size == 0:
                return                      # the speculative advance is the real one: bit-identical

            if splits >= self.max_event_splits:
                raise ValueError(
                    f"more than max_event_splits={self.max_event_splits} event crossings in one "
                    f"step of {float(dt)} s (still {crossed.size} unresolved). Use a smaller dt, or "
                    f"raise Simulation.max_event_splits. Dropping them would silently restore the "
                    f"first-order error event splitting exists to remove - see events.py."
                )

            # H(tau) = max_j -sign(g_j(0)) g_j(tau): negative until the earliest crossing, positive
            # after it. Evaluated by trial propagation from the snapshot, which `h_at` restores to.
            start_sign = np.sign(g0[crossed])
            h_lo = events.reduce_to_scalar(g0[crossed], start_sign)

            def h_at(tau: float, _sign0: NDArray[np.float64] = start_sign,
                     _crossed: NDArray[np.int64] = crossed) -> float:
                self._restore_event_state()
                self._advance(tau)
                return events.reduce_to_scalar(self._event_values()[_crossed], _sign0)

            h_hi = events.reduce_to_scalar(self._event_values()[crossed], start_sign)
            self._restore_event_state()

            # The tightest tolerance among the events that actually crossed - not among all of them,
            # so a loose event elsewhere in the list never relaxes the crossing being located here.
            tol_s = float(np.min(self._event_tolerances[crossed]))
            tau_lo, tau_hi, n_evals = events.locate_crossing(
                h_at, 0.0, remaining, h_lo, h_hi, tol_s)
            self._event_evaluations += n_evals

            # Splitting the step is necessary and *not sufficient*, and this is where the difference
            # lives. RK4's stages sample points off the trajectory - stage 4 by `O(h^3 |da/dt|)` in
            # position, which is 5.6e-4 km at h = 5 s in LEO. A sub-step that ends exactly at the
            # terminator therefore evaluates its fourth stage on whichever side of the surface that
            # off-trajectory point happens to fall, and when that is the far side the stage carries
            # weight 1/6 of a full `Delta_a h`. The result is **still first order**: measured
            # 3.38e-6 km against a derived `n Delta_a (h/6) T_rem` = 3.26e-6 km, which is *worse*
            # than not splitting at all at h = 1.25 s. Two things fix it together:
            #
            #   1. The **latch**. For the duration of a sub-interval known to contain no crossing,
            #      the event's model is pinned to the branch it starts on (`Event.latch`,
            #      `srp.latch_shadow_branch`). The right-hand side is then genuinely smooth over
            #      that interval - one branch, all four stages - which is the only condition under
            #      which RK4's order theorem applies at all.
            #   2. **Three advances, not two.** Up to `tau_lo` (latched to the pre-crossing branch,
            #      exact); then one micro-step across the bracket itself, no wider than `tol_s`,
            #      which is the only interval that straddles the jump and so contributes at most
            #      `Delta_a tol_s`; and the arena is left at `tau_hi`, strictly past the crossing, so
            #      the next advance's first stage reads the post-crossing branch unaided.
            #
            # The residual is then the *tolerance*, not the step - `n Delta_a tol_s T_rem` - which
            # is what makes it vanish from the convergence ladder instead of dominating it.
            sign0 = np.sign(g0)
            self._set_event_latches(sign0)
            try:
                self._restore_event_state()
                if tau_lo >= manoeuvres.MIN_SUBSTEP_S:
                    self._advance(tau_lo)
                if tau_hi > tau_lo:
                    self._advance(tau_hi - tau_lo)

                # Postcondition: the arena must now be strictly past the crossing, or the next
                # sub-interval's first stage reads the pre-crossing branch and the whole exercise
                # is undone. It is *checked*, because the bracket was measured on the root find's
                # trial trajectory (one advance of `tau` from the interval start) and the split
                # followed a different one (an advance to `tau_lo`, then the micro-step): within a
                # tolerance of the surface those two can land on opposite sides. Nudging by one
                # more tolerance costs `Delta_a tol_s` - the same order as the tolerance residual
                # already accepted - and is measured at zero or one nudge per crossing.
                #
                # The test is `<= 0`, not `< 0`, and that is not a style choice. A converging root
                # find lands *on* the root, and the event function quantises: `umbra_clearance` is
                # `sqrt(...) - r_occ`, whose value rounds to **exactly 0.0** for every position
                # within one ulp of 6378 km of the surface. `H` is then `-0.0`, and `-0.0 < 0.0` is
                # false in IEEE 754, so a strict test declares the crossing resolved while the body
                # sits exactly on a surface whose own membership test (`perp < r_occ`) is strict and
                # reads *lit*. That is how the first-order error survived the split, undetected and
                # unraised, until a full-precision trace of one crossing showed `g = +0.000000e+00`.
                advanced = tau_hi
                crossed_sign = sign0[crossed]
                nudges = 0
                while events.reduce_to_scalar(self._event_values()[crossed], crossed_sign) <= 0.0:
                    if nudges >= events.MAX_CROSSING_NUDGES:
                        raise ValueError(
                            f"event crossing located at t={float(self.t)} s could not be stepped "
                            f"past in {nudges} nudges of {tol_s} s. The event function is moving "
                            f"faster than the tolerance resolves, or is not continuous. Loosen "
                            f"tol_s, or check the event function. See events.py."
                        )
                    self._advance(tol_s)
                    advanced += tol_s
                    nudges += 1
            finally:
                self._set_event_latches(None)
            self._event_splits += 1
            splits += 1
            remaining -= advanced

    def apply_delta_v(
        self,
        bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
        dv_rsw: Sequence[float] | NDArray[np.float64],
    ) -> None:
        """
        Apply an impulsive Delta-v (RSW components, km/s) to `bodies` **now**, between steps.

        `(0, dv, 0)` is a prograde (along-track) burn, `(dv, 0, 0)` radial-out, `(0, 0, dv)` normal -
        the RSW frame of the body's state relative to its Keplerian parent, the same convention
        `thrust.py`'s direction law uses. `dv_rsw` is `(3,)` for one Delta-v shared by every named
        body, or `(len(bodies), 3)` for one each.

        What changes depends on the body's propagator, and getting that wrong is the whole difficulty
        of this operation (see `manoeuvres.py`):

        - **Keplerian**: the Cartesian state *and* `coe_states`, re-derived from the post-burn
          `(r, v)`. The elements are this propagator's state of record - `KeplerianPropagator` rewrites
          `local_states` from them every step - so an impulse that only touched the Cartesian state
          would be silently erased by the next `step()`.
        - **Secular J2**: as Keplerian, plus `self._secular_j2_rates`, recomputed here from the new
          elements through `propagators.secular_j2_rates`. Those rates are cached at `set_propagator`
          time and depend on `p`, `e` and `i`, every one of which an impulse changes; left stale, the
          body would keep regressing its node at the *old* orbit's rate and nothing would raise.
        - **Cowell**: the Cartesian state only. Its `coe_states` row is documented stale, and this
          deliberately does not revive it.

        Raises `ValueError`, before mutating anything, for an inactive slot, a system head, a
        barycentre, a root, a body carrying mass (`mu != 0`), a body whose pre-burn RSW frame is
        undefined, or an impulse that would leave a secular-J2 body on an open orbit. The restrictions
        and their reasons are `manoeuvres.py`'s module docstring; they mirror `set_propagator`'s.
        """
        idx = self._manoeuvre_slots(bodies)
        dv = self._manoeuvre_dv(dv_rsw, idx.size)

        valid = manoeuvres.apply_delta_v(
            idx, dv, self.global_states, self.local_states, self.coe_states,
            self.mu_array, self.parent_indices, self.propagator_type,
        )
        if not bool(valid.all()):
            raise ValueError(
                f"impulsive Delta-v refused for slot(s) {idx[~valid].tolist()}: the pre-burn RSW frame "
                f"is undefined (zero, rectilinear or non-finite state relative to the Keplerian "
                f"parent), the post-burn orbit could not be classified, or a SECULAR_J2 body would be "
                f"left on an open orbit. No body was modified. See manoeuvres.py."
            )

        # The secular-J2 rates are a *cache* of a function of (p, e, i), all three of which just
        # changed. Recomputing them is the whole reason this method exists rather than a bare call to
        # the kernel - see this method's docstring and `propagators.secular_j2_rates`.
        secular = idx[self.propagator_type[idx] == np.uint8(PropagatorType.SECULAR_J2)]
        if secular.size > 0:
            j2_params = self.force_model_params.get(geopotential.J2_MODEL)
            if j2_params is not None:
                self._secular_j2_rates[secular] = secular_j2_rates(
                    self.coe_states, self.mu_array, self.parent_indices, j2_params, secular,
                )

    def _manoeuvre_slots(
        self, bodies: Union[int, Sequence[int], NDArray[np.integer[Any]]],
    ) -> NDArray[np.int64]:
        """
        Normalise a manoeuvre's `bodies` argument to an integer slot array and enforce the restrictions
        in `manoeuvres.py`'s module docstring.

        **Caller order is preserved**, so row `k` of a per-body `(n, 3)` Delta-v belongs to body `k` as
        written; sorting here would silently re-pair them. A repeated slot is rejected rather than
        de-duplicated, because `manoeuvres.apply_delta_v` writes through fancy indexing: a slot named
        twice would be a many-to-one scatter, and one of the two Delta-vs would vanish with no error at
        all. Two impulses on one body are two calls, or two scheduled manoeuvres.
        """
        idx = np.atleast_1d(np.asarray(bodies, dtype=np.int64))

        if idx.size == 0:
            raise ValueError("an impulsive Delta-v needs at least one body.")
        if np.unique(idx).size != idx.size:
            raise ValueError(
                f"an impulsive Delta-v names each body at most once; got {idx.tolist()}. Two impulses "
                f"on one body are two calls - one fancy-indexed write cannot deliver both."
            )
        if bool(((idx < 0) | (idx >= self.max_capacity)).any()):
            raise ValueError(f"slot(s) {idx.tolist()} are outside the arena (capacity {self.max_capacity}).")

        disallowed = (
            ~self.active_mask[idx] | self.is_head[idx] | self.is_system[idx] |
            (self.body_sys_map[idx] == idx) | (self.parent_indices[idx] == idx)
        )
        if np.any(disallowed):
            raise ValueError(
                f"an impulsive Delta-v is restricted to active, non-head, non-barycentre bodies with a "
                f"Keplerian parent and a system bubble of their own to sit in; slot(s) "
                f"{idx[disallowed].tolist()} do not qualify. A head's motion is its bubble's reflex "
                f"kick, recomputed every step, so an impulse on it would be silently overwritten; a "
                f"barycentre is a mass-weighted mean, not an object; a root has no frame to burn "
                f"relative to. See manoeuvres.py."
            )

        massive = idx[self.mu_array[idx] != 0.0]
        if massive.size > 0:
            raise ValueError(
                f"an impulsive Delta-v requires mu == 0; slot(s) {massive.tolist()} carry mass, and "
                f"their system barycentre's own orbit - a separate slot this call does not touch - "
                f"would not see the momentum change, leaving the bubble inconsistent while looking "
                f"entirely plausible. The same masslessness Cowell, secular-J2 and thrust require. "
                f"See manoeuvres.py."
            )
        return idx

    @staticmethod
    def _manoeuvre_dv(
        dv_rsw: Sequence[float] | NDArray[np.float64], n_bodies: int,
    ) -> NDArray[np.float64]:
        """
        Validate a Delta-v argument and return it as float64, `(3,)` or `(n_bodies, 3)`.

        Shape is checked here rather than left to broadcasting: a `(2, 3)` Delta-v against three bodies
        would raise deep inside an einsum with a shape message that names neither, and a `(3, 3)`
        against three bodies is ambiguous only if the check is absent - here it means one Delta-v each.
        """
        dv: NDArray[np.float64] = np.asarray(dv_rsw, dtype=np.float64)
        if dv.shape != (3,) and dv.shape != (n_bodies, 3):
            raise ValueError(
                f"dv_rsw must be (3,) - one Delta-v for every named body - or ({n_bodies}, 3), one "
                f"each, in RSW components and km/s; got shape {dv.shape} for {n_bodies} body(ies)."
            )
        return dv

    def _rebase(self, indices: NDArray[np.int64], rel: NDArray[np.float64]) -> None:
        """
        Write `global_states[indices] = global_states[parent] + rel[indices]` and rebuild
        `local_states[indices]` against `body_sys_map`, so the arena's invariant - `local_states[i]`
        relative to `global_states[body_sys_map[i]]` - holds for a Cowell or secular-J2 body exactly
        as it does for a Keplerian one. Called by `step()` after `calc_global()`, once per propagator.

        The NumPy block below is the definition. `kernels.rebase_relative_states` is its compiled twin,
        held bit-identical (not merely within a tolerance) by
        `tests/validation/test_kernel_equivalence.py`, and selected by `use_compiled_kernel` like every
        other twin - except when `_rebase_compiled_ok` is false, see `_refresh_active_indices`.
        """
        if self.use_compiled_kernel and self._rebase_compiled_ok:
            rebase_relative_states(
                indices, self.parent_indices, self.body_sys_map, rel,
                self.global_states, self.local_states,
            )
            return

        parents = self.parent_indices[indices]
        self.global_states[indices] = self.global_states[parents] + rel[indices]
        self.local_states[indices] = (
            self.global_states[indices] - self.global_states[self.body_sys_map[indices]]
        )

    def run(self, duration: ScalarSeconds, dt: ScalarSeconds) -> None:
        if self.t == 0 and self.record_history:
            self._record_state()

        steps = int(duration/dt)
        for _ in range(steps):
            self.step(dt)

    def _record_state(self) -> None:
        """
        Snapshot the current arena state.

        Records **columnar**: three array slices per step, rather than one dict per body per step.
        The previous row-wise form built `n_bodies` dictionaries of sixteen keys on every step, which
        cost 1875 us per step at 600 bodies - roughly ten times the entire physics step it was
        recording. Copying three contiguous slices instead is a handful of microseconds and
        independent of body count in call overhead.

        The DataFrame is assembled lazily in `history`, so a run that never inspects its history
        never pays for the long-format expansion at all.
        """
        self._hist_seconds.append(float(self.t))
        self._hist_global.append(self.global_states[self._recorded_slots].copy())
        self._hist_local.append(self.local_states[self._recorded_slots].copy())
        self._hist_coe.append(self.coe_states[self._recorded_slots].copy())

    @property
    def history(self) -> pd.DataFrame:
        """
        Recorded history in long format: one row per body per snapshot.

        Columns are unchanged from the original row-wise recorder, so existing analysis and the
        notebooks continue to work. Built on demand from the columnar buffers.
        """
        if not self._hist_seconds:
            return pd.DataFrame(columns=[
                "timestamp", "seconds", "body",
                "g_x", "g_y", "g_z", "g_vx", "g_vy", "g_vz",
                "x", "y", "z", "vx", "vy", "vz", "e", "theta",
            ])

        n_snaps = len(self._hist_seconds)
        n_bodies = len(self._recorded_names)

        g = np.stack(self._hist_global).reshape(n_snaps * n_bodies, 6)
        loc = np.stack(self._hist_local).reshape(n_snaps * n_bodies, 6)
        coe = np.stack(self._hist_coe).reshape(n_snaps * n_bodies, 6)

        seconds = np.repeat(np.asarray(self._hist_seconds, dtype=np.float64), n_bodies)

        return pd.DataFrame({
            "timestamp": [self.start_epoch + timedelta(seconds=s) for s in seconds],
            "seconds": seconds,
            "body": self._recorded_names * n_snaps,
            "g_x": g[:, 0], "g_y": g[:, 1], "g_z": g[:, 2],
            "g_vx": g[:, 3], "g_vy": g[:, 4], "g_vz": g[:, 5],
            "x": loc[:, 0], "y": loc[:, 1], "z": loc[:, 2],
            "vx": loc[:, 3], "vy": loc[:, 4], "vz": loc[:, 5],
            "e": coe[:, 1],
            "theta": coe[:, 5],
        })

    def clear_history(self) -> None:
        self._hist_seconds = []
        self._hist_global = []
        self._hist_local = []
        self._hist_coe = []

if __name__ == "__main__":
    # sim = Simulation(body_names=["Earth", "Moon", "Sun"])

    # sim.run(24*(60**2), 0.5*(60.0**2.0))
    # print(sim.history)
    from .database import seed_test_universe

    seed_test_universe()

    print("\n[ ENGINE BOOTING ]")
    # system_query = ["Solar System", "Earth-Moon System", "Alpha Centauri System"]
    system_query = ["Solar System", "Alpha Centauri System"]
    # system_query = ["Solar System"]


    sim = Simulation(body_names=["Sun", "Earth", "Moon", "Alpha Centauri A", "Alpha Centauri B", "Jupiter", "Saturn"], system_names=system_query)
    # sim = Simulation(body_names=["Sun", "Earth", "Moon"], system_names=system_query)

    idx_to_name = {idx: name for name, idx in sim.name_to_index.items()}

    print("\n=======================================================")
    print("     RUNTIME TOPOLOGICAL EXECUTION TIERS")
    print("=======================================================")
    for i, tier in enumerate(sim.topological_tiers):
        names = [idx_to_name[idx] for idx in tier]
        print(f"Tier {i} (Depth): {names}")

    print("\n=======================================================")
    print("             DOD MEMORY ARENA DUMP")
    print("=======================================================")

    for idx in range(sim.max_capacity):
        if sim.active_mask[idx]:
            name = idx_to_name[idx]
            
            # Resolve System Name
            sys_idx = sim.body_sys_map[idx]
            sys_name = idx_to_name.get(sys_idx, "None") if sys_idx != -1 else "None"
            head_idx = sim.sys_head_map[idx]
            sys_head = idx_to_name.get(head_idx, "None") if head_idx != -1 else "None"
                
            # Resolve Parent Name
            parent_idx = sim.parent_indices[idx]
            parent_name = idx_to_name.get(parent_idx, "Universal Root") if parent_idx != idx else "Universal Root"
            
            print(f"[{name}]")
            print(f"  Type      : {'System Barycenter' if sim.is_system[idx] else 'Physical Body'}")
            print(f"  Is Head?  : {sim.is_head[idx]}")
            print(f"  System    : {sys_name}")
            print(f"  Sys Head  : {'***' if sim.is_head[idx] else sys_head}")
            print(f"  Parent    : {parent_name}")
            print(f"  Mass (mu) : {sim.mu_array[idx]:.4e}")
            print(f"  Global R  : {sim.global_states[idx, :3]}")
            print(f"  Local R   : {sim.local_states[idx, :3]}")
            print(f"  COE (e)   : {sim.coe_states[idx, 1]:.5f}")
            print("-" * 55)

            # Current bug is whether barycenters belong to their parent's system, or the system they define and how this effects the code! RESOLVED

    sim.run(24*(60**2), 0.5*(60.0**2.0))
    with pd.option_context('display.max_rows', 100):
        print(sim.history.head(100))
    # print(sim.history.head(100).to_string())