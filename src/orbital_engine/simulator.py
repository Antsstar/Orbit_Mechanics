from __future__ import annotations
from typing import List, Optional, Any, cast
from .custom_types import ScalarSeconds, PropagatorType
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .propagators import Propagator, KeplerianPropagator
from .database import get_session, CelestialBodyORM, BaseBodyORM, VesselORM, VirtualBodyORM, SystemORM
from .body import BodyHandle
from . import frames as fr

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
                 default_propagator: Optional[Propagator] | int = PropagatorType.NONE) -> None:

        self.start_epoch: datetime = start_epoch if start_epoch is not None else datetime.now()

        # --- Runtime State Varables ---
        self.t: ScalarSeconds = 0.0                                               # Global Simulation time
        self.bodies: List[BodyHandle] = []                                  # Body Handles for UI use on simulation objects.
        self._history_buffer: List[dict[str, Any]] = []                     # Simulation data history buffer.

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
        print(sys_barycenters)

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
        
        print(self.name_to_index)
        print(self.is_system[self.active_mask]) # Interesting behaviour. An empty barycenter isn't seen as a system (because noone references it!)
        print(self.is_head[self.active_mask])   # I think that's fine honestly.

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

        print(self.parent_indices[self.active_mask])
        print(self.body_sys_map[self.active_mask])
        print(self.sys_head_map[self.active_mask])
        # print(self.name_to_index)
        self._topological_sort(self.parent_indices) # Topological sort based on DB parent_indices. Provides an schematic for the initial positions of all bodies.
        self._unfold_database_to_global() # Using our topological map, compute global positions for all bodies based on local states and reference parents.
        print(self.parent_indices[self.active_mask])

        # Re-sort based on system mapping, This means barycentric referencing (and point-mass in extremes) for all bodies.
        self._topological_sort(self.body_sys_map)   # This ensures all siblings (+head) are processed at the same step, allowing for proper barycentric calculations.
        self._recalculate_all_barycenters()         # Barycenters wouldn't have positions or coes defined in the DB, so we first compute their global vectors.
        self._zero_roots()                          # We want to ensure the root system is at the origin, so we shift all bodies in the same ancestry to that root, but its vector.
        self._rehydrate_coes()                      # Recompute the local states and coes based on new global positions and parenting. Also provides barycenters coe values.

        
        loaded_count = sum(self.active_mask)
        print(f"Universe Built: {loaded_count} bodies loaded.")
        print(f"Mass Array: {self.mu_array[:loaded_count]}")
    

    def _resolve_circular(self) -> None:
        """Identify and resolve DB circular dependencies (Binary Systems) by electing a head body based on mass and/or index."""
        idx = np.arange(self.max_capacity)
        p_idx = self.parent_indices
        gp_idx = p_idx[p_idx]

        circular_mask = self.active_mask & (gp_idx == idx) & (p_idx != idx)
        if np.any(circular_mask):
            print(circular_mask[self.active_mask])
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
        
        print(self.topological_tiers)

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
        """
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
    

    def step(self, dt: ScalarSeconds) -> None:

        KeplerianPropagator.propagate(dt=dt, primary_states=self.coe_states, secondary_states=self.local_states, mu_array=self.mu_array,
                                      parent_indices=self.parent_indices, active_mask=self.active_mask, is_head=self.is_head, is_system=self.is_system,
                                      body_sys_map=self.body_sys_map, sys_head_map=self.sys_head_map)
        self.calc_global()
        # exit() 

        self.t += dt
        self._record_state()

    def run(self, duration: ScalarSeconds, dt: ScalarSeconds) -> None:
        if self.t == 0:
            self._record_state()

        steps = int(duration/dt)
        for _ in range(steps):
            self.step(dt)

    def _record_state(self) -> None:
        """Internal helper to snap the current state of all bodies."""
        # current_dt = self.start_epoch + timedelta(seconds=self.t)

        for body in self.name_to_index:
            index = self.name_to_index[body]
            self._history_buffer.append({
                "timestamp": self.current_epoch,
                "seconds": self.t,
                "body": body,
                "g_x": self.global_states[index, 0], "g_y": self.global_states[index, 1], "g_z": self.global_states[index, 2],
                "g_vx": self.global_states[index, 3], "g_vy": self.global_states[index, 4], "g_vz": self.global_states[index, 5],
                "x": self.local_states[index, 0], "y": self.local_states[index, 1], "z": self.local_states[index, 2],
                "vx": self.local_states[index, 3], "vy": self.local_states[index, 4], "vz": self.local_states[index, 5],
                # "p": self.coe_states[index, 0],
                "e": self.coe_states[index, 1],
                # "i": self.coe_states[index, 2],
                # "raan": self.coe_states[index, 3],
                # "arg_pe": self.coe_states[index, 4],
                "theta": self.coe_states[index, 5]
                })

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history_buffer)
    
    def clear_history(self) -> None:
        self._history_buffer = []

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