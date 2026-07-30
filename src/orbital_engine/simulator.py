from __future__ import annotations
from typing import List, Optional, Any
from .custom_types import Seconds, PropagatorType

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
                 system_names, 
                 max_capacity: int = 10000, 
                 start_epoch: Optional[datetime] = None,
                 session: Optional[Session] = None,
                 default_propagator: Optional[Propagator] | int = PropagatorType.NONE) -> None:

        self.start_epoch: datetime = start_epoch if start_epoch is not None else datetime.now()

        # --- Runtime State Varables ---
        self.t: Seconds = 0.0
        self.bodies: List[BodyHandle] = []
        self._history_buffer: List[dict[str, Any]] = []

        self.max_capacity = max_capacity

        self.local_states = np.zeros((max_capacity, 6), dtype=np.float64)   # [x, y, z, vx, vy, vz]
        self.coe_states = np.zeros((max_capacity, 6), dtype=np.float64)     # [p, e, i, Omega, omega, theta]
        self.mu_array = np.zeros(max_capacity, dtype=np.float64)            # mu values for each body
        self.parent_indices = np.full(max_capacity, -1, dtype=np.int32)     # indices of parent bodies
        self.global_states = np.zeros((max_capacity, 6), dtype=np.float64)  # We will store both systems and bodies.
        # self.global_depth = np.full(max_capacity, -1, dtype=np.int32)       # Same as active mask( for now -1 but change later)

        # The free-list stack.
        self.free_indices = list(range(max_capacity-1, -1, -1))             # Reverse order for efficient pop()

        # Name to integer mapping
        self.name_to_index: dict[str, int] = {}

        # Filtering masks
        self.active_mask = np.zeros(max_capacity, dtype=np.bool_)           # Currently isn't necessary as unactive bodies point to a parent of -1, change this
        self.is_system = np.zeros(max_capacity, dtype=np.bool_)
        self.is_head = np.zeros(max_capacity, dtype=np.bool_)
        self.body_sys_map = np.full(max_capacity, -1, dtype=np.int32)
        self.propagator_type = np.full(max_capacity, PropagatorType.KEPLERIAN,dtype=np.uint8)       # Likely won't be implemented yet
        self.sys_head_map = np.full(max_capacity, -1, dtype=np.int32)       # Systems refer to children heads, while bodies refer to siblings head


        self._build_universe(body_names, system_names, session=session)                   # Our goal is to remove the top sort and implement an execution order

    def _build_universe(self, body_names: List[str], system_names: List[str], session: Optional[Session] = None) -> None:
        """Queries the DB, topologically sorts the bodies, and populates the arrays."""
        local_session = False
        if session is None:
            session = get_session()
            local_session = True
            

        orm_bodies = session.query(BaseBodyORM).filter(BaseBodyORM.name.in_(body_names)).all()
        orm_systems = session.query(SystemORM).filter(SystemORM.name.in_(system_names)).all()

        if len(orm_bodies) != len(body_names):
            found = [b.name for b in orm_bodies]
            missing = set(body_names) - set(found)
            raise ValueError(f"Could not find bodies in database: {missing}")
        
        all_orm_bodies: List[tuple[BaseBodyORM, Optional[SystemORM]]] = []

        for orm_body in orm_bodies:
            if orm_body.system in orm_systems: # Doesn't allow bodies without systems!
                all_orm_bodies.append((orm_body, orm_body.system))
            else:
                reparent = getattr(orm_body.parent, 'system', None)
                all_orm_bodies.append((orm_body, reparent if reparent in orm_systems else None))
        for orm_sys in orm_systems:
            if orm_sys.barycenter: #Changed from orm_sys to the parent's system
                all_orm_bodies.append((orm_sys.barycenter, getattr(orm_sys.barycenter.parent, 'system', None))) # This is creating the issue!

        print(all_orm_bodies)
        sys_barycenters = set(getattr(id, 'barycenter_id', -1) for _, id in all_orm_bodies)
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

            # if associated_sys.barycenter_id == orm_body.id:
            if orm_body.id in sys_barycenters:
                self.is_system[idx] = True
            # if associated_sys.head_body_id == orm_body.id:
            if getattr(associated_sys, 'head_body_id', None) == orm_body.id:
                self.is_head[idx] = True
        
        print(self.name_to_index)
        print(self.is_system[self.active_mask]) # Interesting behaviour. An empty barycenter isn't seen as a system (because noone references it!)
        print(self.is_head[self.active_mask])   # I think that's fine honestly.

        for orm_body, associated_sys in all_orm_bodies:
            idx = self.name_to_index[orm_body.name]

            parent = getattr(getattr(orm_body, 'parent', None), 'name', None)
            bary_name = getattr(getattr(associated_sys, 'barycenter', None), 'name', None) # Check that barycenters are members of parent barycenters
            system_idx = self.name_to_index.get(bary_name, -1) # Check that barycenters are members of parent barycenters
            self.body_sys_map[idx] = system_idx # I know they're parent is an outer body, but im not sure if they are also members of parent barycenters
            # Insane foresight this was the problem, whether their system should be self referenced or to the parent barycenter. Been updated to parent barycenter
            
            if orm_body.parent_id and parent in self.name_to_index:
                self.parent_indices[idx] = self.name_to_index[parent]
            else:
                self.parent_indices[idx] = idx

            if system_idx == -1 and not self.is_system[idx]: # Need to update all of the is_system logic.
                system_idx = self.parent_indices[idx] # Follow same self reference or parent logic.
            self.body_sys_map[idx] = system_idx # Bubbled logic

            if not self.is_system[idx] and getattr(associated_sys, 'head_body_id', orm_body.id) is None: # None is only returned if nothing is found(at least thats the idea)
                # self.parent_indices[idx] = self.name_to_index[associated_sys.barycenter.name] # Check this line
                self.parent_indices[idx] = self.name_to_index[bary_name]
                

        if local_session:
            session.close()

        # Continue from here by doing global vectors!
        print(self.parent_indices[self.active_mask])
        print(self.body_sys_map[self.active_mask])
        # print(self.name_to_index)
        self._topological_sort()
        self._unfold_database_to_global() # Something is either wrong for barycenter calcs here
        self._normalize_runtime_graph() # Reparenting function which works I believe
        self._topological_sort()
        self._recalculate_all_barycenters() # Or here!
        self._zero_roots()
        self._rehydrate_coes()

        head_mask = self.is_head & self.active_mask
        head_idx = np.where(head_mask)[0].astype(np.int32)
        sys_head_idx = self.body_sys_map[head_mask]
        self.sys_head_map[sys_head_idx] = head_idx

        child_mask = ~self.is_head & ~self.is_system & self.active_mask
        child_idx = np.where(child_mask)[0].astype(np.int32)
        child_bubbles = self.body_sys_map[child_mask]
        valid_cbubbles_mask = child_bubbles != -1
        valid_children = child_idx[valid_cbubbles_mask]
        valid_cbubbles = child_bubbles[valid_cbubbles_mask]

        self.sys_head_map[valid_children] = self.sys_head_map[valid_cbubbles]
        
        loaded_count = sum(self.active_mask)
        print(f"Universe Built: {loaded_count} bodies loaded.")
        print(f"Mass Array: {self.mu_array[:loaded_count]}")
    

    def _topological_sort(self) -> None:
        active_indices = np.where(self.active_mask)[0].astype(np.uint32)
        root = self.parent_indices[active_indices] == active_indices

        current_tier = active_indices[root]

        if len(current_tier) == 0:
            raise RuntimeError("Topology Error: No root detected amongst active indices")
        
        self.topological_tiers: List[np.ndarray[np.uint32]] = [current_tier]

        processed_count = len(current_tier)
        total_active = len(active_indices)
  
        while processed_count < total_active:
            next_layer_mask = np.isin(self.parent_indices[active_indices], current_tier)
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

                local_r, local_v, _ = fr.ReferenceFrames.coe_to_rv(self.coe_states[idx_m1], mu_parents[m1])
                self.global_states[idx_m1, :3] = self.global_states[p_m1, :3] + local_r
                self.global_states[idx_m1, 3:] = self.global_states[p_m1, 3:] + local_v

            m2 = parent_sys # Bodies with systems as parents
            # print(m2)
            if np.any(m2): # Needs to be clever, only collapse bodies in that system i.e. if there are multiple parent systems, don't sum all bodies to 1 parent
                idx_m2 = tier[m2]
                p_m2 = parents[m2]

                local_r, local_v, _ = fr.ReferenceFrames.coe_to_rv(self.coe_states[idx_m2], mu_parents[m2]) 
                # I need to collapse this into n-systems rather than m bodies n systems.

                system_r_sums = np.zeros((self.max_capacity, 3), dtype=np.float64)
                system_v_sums = np.zeros((self.max_capacity, 3), dtype=np.float64)
                system_masses = np.zeros(self.max_capacity, dtype=np.float64)

                body_masses = self.mu_array[idx_m2]

                np.add.at(system_r_sums, p_m2, local_r * body_masses[:, None])
                np.add.at(system_v_sums, p_m2, local_v * body_masses[:, None])
                np.add.at(system_masses, p_m2, body_masses)

                valid_sys = system_masses > 0
                kick_r = np.zeros_like(system_r_sums)
                kick_v = np.zeros_like(system_v_sums)

                kick_r[valid_sys] = -system_r_sums[valid_sys] / system_masses[valid_sys, None]
                kick_v[valid_sys] = -system_v_sums[valid_sys] / system_masses[valid_sys, None]

                shifted_local_r = local_r + kick_r[p_m2]
                shifted_local_v = local_v + kick_v[p_m2]

                self.global_states[idx_m2, :3] = self.global_states[p_m2, :3] + shifted_local_r
                self.global_states[idx_m2, 3:] = self.global_states[p_m2, 3:] + shifted_local_v

    def _recalculate_all_barycenters(self) -> None:
        """
        Calculate the true global Cartesian center of mass for system based on updated
        absolute positions of its member bodies.
        """
        sys_mask = self.is_system & self.active_mask
        body_mask = ~self.is_system & self.active_mask

        self.global_states[sys_mask] = 0.0
        self.mu_array[sys_mask] = 0.0

        # print(self.global_states[self.name_to_index["Solar System Barycenter"]])
        mass_weighted_states = np.zeros_like(self.global_states)
        mass_weighted_states[body_mask] = self.global_states[body_mask] * self.mu_array[body_mask, None]

        for tier in reversed(self.topological_tiers):
            has_sys = self.body_sys_map[tier] != -1
            idx = tier[has_sys]

            if len(idx) > 0:
                parent_sys = self.body_sys_map[idx]
                valid_mask = self.is_system[parent_sys]
                valid_idx = idx[valid_mask]
                valid_parents = parent_sys[valid_mask]

                # np.add.at(self.mu_array, parent_sys, self.mu_array[idx]) # This line needs to change to not add mass to "bodies" which don't return True from is_system

                # mass_weighted_states = self.global_states[idx] * self.mu_array[idx, None]
                # np.add.at(self.global_states, parent_sys, mass_weighted_states)

                # np.add.at(mass_weighted_states, parent_sys, mass_weighted_states[idx])
                if len(valid_idx) > 0:
                    np.add.at(self.mu_array, valid_parents, self.mu_array[valid_idx])
                    np.add.at(mass_weighted_states, valid_parents, mass_weighted_states[valid_idx])

        valid_sys = (self.mu_array > 0) & sys_mask
        # self.global_states[valid_sys] /= self.mu_array[valid_sys, None]
        self.global_states[valid_sys] = mass_weighted_states[valid_sys] / self.mu_array[valid_sys, None]


    def _normalize_runtime_graph(self) -> None:
        """
        Rewires the parent graph to be strictly Barycentric.
        Bodies orbit their System Barycenter. Systems orbit their Parent System's Barycenter.
        """

        active_bodies = ~self.is_system & self.active_mask
        active_systems = self.is_system & self.active_mask

        has_system = active_bodies & (self.body_sys_map != -1)
        self.parent_indices[has_system] = self.body_sys_map[has_system] # Ensure all bodies (except root body) point to their system (barycenter or point mass)

        sys_db_parents = self.parent_indices[active_systems]

        # parent_bubbles = self.body_sys_map[sys_db_parents] # Ah so if the parent exists but isn't in a system (its not flagged)

        # valid_bubbles = parent_bubbles != -1

        # sys_id_up = np.where(active_systems)[0][valid_bubbles]
        # self.parent_indices[sys_id_up] = parent_bubbles[valid_bubbles]

        has_db_parent = sys_db_parents != -1

        sys_id_up = np.where(active_systems)[0][has_db_parent]
        valid_sys_parents = sys_db_parents[has_db_parent]

        parent_bubbles = self.body_sys_map[valid_sys_parents]

        final_targets = np.where(parent_bubbles != -1, parent_bubbles, valid_sys_parents)

        # self.parent_indices[sys_id_up] = valid_sys_parents
        self.parent_indices[sys_id_up] = final_targets

        roots = self.parent_indices == -1
        self.parent_indices[roots] = np.where(roots)[0].astype(np.uint32)
        # print("Normalize: ", self.body_sys_map[self.active_mask]) # -1 roots
        # print(self.parent_indices[self.active_mask]) # Self reference

    def _rehydrate_coes(self) -> None:
        """
        Calculate normalized local Cartesian vectors then generates mathematically accurate Classical Orbital Elements.
        Current Definition uses Two-Body mass sum for mu (mu_parent + mu_child).
        """
        # Self-reference made this very simple, if the node self references then the local state will be 0.0
        self.local_states[self.active_mask] = (
            self.global_states[self.active_mask] - 
            self.global_states[self.parent_indices[self.active_mask]]
        )

        self.coe_states[self.is_head, :] = 0.0 # Question if this should have values.
        self.coe_states[self.active_mask & (self.parent_indices == np.arange(self.max_capacity))] = 0.0
        valid_mask = self.active_mask & ~self.is_head & (self.parent_indices != np.arange(self.max_capacity))
        # Pick anything that is active, not a head and isn't a self referenced object (root)

        # print(rel_r, rel_v)
        if np.any(valid_mask):
            rel_r = self.local_states[valid_mask, :3].copy()
            rel_v = self.local_states[valid_mask, 3:].copy() # New copy

            parents = self.parent_indices[valid_mask]
            children = np.where(valid_mask)[0].astype(np.uint32)

            # mu_two_body = self.mu_array[parents] + self.mu_array[children]
            parent_is_sys = self.is_system[parents]

            if np.any(parent_is_sys):
                m_c = self.mu_array[children[parent_is_sys]]
                m_s = self.mu_array[parents[parent_is_sys]]
                m_h = m_s - m_c

                scale = m_s / (m_h  + 1e-20) # avoid div by 0
                rel_r[parent_is_sys] *= scale[:, None]
                rel_v[parent_is_sys] *= scale[:, None]

            # mu_head = self.mu_array[parents] - self.mu_array[children]
            # mu_sys  = self.mu_array[parents]

            mu_calc = np.where(
                parent_is_sys,
                self.mu_array[parents],
                # (mu_head ** 3) / ((mu_sys ** 2) + 1e-20),
                # self.mu_array[parents] + self.mu_array[children]
                self.mu_array[parents]
            )


            new_coes, success = fr.ReferenceFrames.rv_to_coe(
                rel_r,
                rel_v,
                mu_calc# mu_two_body
            )


            valid_idx = np.where(valid_mask)[0][success]
            self.coe_states[valid_idx] = new_coes[success]


    def _zero_roots(self) -> None:
        roots = self.parent_indices == np.arange(self.max_capacity)
        active_roots = roots & self.active_mask

        if not np.any(active_roots):
            return
        
        shifts = np.zeros_like(self.global_states)
        shifts[active_roots] = self.global_states[active_roots]

        for tier in self.topological_tiers:
            has_parent = self.parent_indices[tier] != tier
            children = tier[has_parent]

            if len(children) > 0:
                parents = self.parent_indices[children]
                shifts[children] = shifts[parents]

        self.global_states[self.active_mask] -= shifts[self.active_mask]

    def calc_global(self) -> None:
        tier_0 = self.topological_tiers[0]
        self.global_states[tier_0] = 0.0

        # print(self.topological_tiers)

        for tier in self.topological_tiers[1:]:

            parents = self.parent_indices[tier]
            # parent_sys = self.is_system[parents]
            # mu_parents = self.mu_array[parents]

            # m1 = ~parent_sys # Any body without a system as a parent
            # # print(m1)
            # if np.any(m1):
            #     idx_m1 = tier[m1]
            #     p_m1 = parents[m1]

            #     local_r, local_v = fr.ReferenceFrames.coe_to_rv(self.coe_states[idx_m1], mu_parents[m1])
            #     self.global_states[idx_m1, :3] = self.global_states[p_m1, :3] + local_r
            #     self.global_states[idx_m1, 3:] = self.global_states[p_m1, 3:] + local_v
            # parent_sys = self.is_system[parents]
            # mu_calc = np.where(
            #     parent_sys,
            #     self.mu_array[parents],
            #     self.mu_array[parents] + self.mu_array[tier]
            # )
            # local_r, local_v, _ = fr.ReferenceFrames.coe_to_rv(self.coe_states[tier], mu_calc)
            self.global_states[tier, :3] = self.global_states[parents, :3] + self.local_states[tier, :3]
            self.global_states[tier, 3:] = self.global_states[parents, 3:] + self.local_states[tier, 3:]

        return

    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)
    

    def step(self, dt: Seconds) -> None:

        KeplerianPropagator.propagate(dt=dt, primary_states=self.coe_states, secondary_states=self.local_states, mu_array=self.mu_array,
                                      parent_indices=self.parent_indices, active_mask=self.active_mask, is_head=self.is_head, is_system=self.is_system,
                                      body_sys_map=self.body_sys_map, sys_head_map=self.sys_head_map)
        self.calc_global() 

        self.t += dt
        self._record_state()

    def run(self, duration: Seconds, dt: Seconds) -> None:
        if self.t == 0:
            self._record_state()

        steps = int(duration/dt)
        for _ in range(steps):
            self.step(dt)

    def _record_state(self) -> None:
        """Internal helper to snap the current state of all bodies."""
        current_dt = self.start_epoch + timedelta(seconds=self.t)

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
    system_query = ["Solar System", "Earth-Moon System", "Alpha Centauri System"]
    # system_query = ["Solar System", "Alpha Centauri System"]
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