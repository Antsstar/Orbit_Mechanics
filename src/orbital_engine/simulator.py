from __future__ import annotations
from typing import List, Optional, Any
from .custom_types import Seconds

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .propagators import KeplerianPropagator
from .database import get_session, CelestialBodyORM, BaseBodyORM, VesselORM, VirtualBodyORM
from .body import BodyHandle
from . import frames as fr

class Simulation:
    """
    Initializes the simulation with a list of body names, maximum capacity, and an optional start epoch.
    Written using Data-Oriented Design (DOD) principles for performance, parallelization, and memory efficiency.
    Free-index stack is used to efficiently manage memory and allow for dynamic addition/removal of bodies.
    All results are stored in a history buffer for later analysis or visualization, exported as a pandas DataFrame.
    """
    def __init__(self, body_names: List[str], max_capacity: int = 10000, start_epoch: Optional[datetime] = None,
                 session: Optional[Session] = None) -> None:

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

        # The free-list stack.
        self.free_indices = list(range(max_capacity-1, -1, -1))             # Reverse order for efficient pop()

        # Name to integer mapping
        self.name_to_index: dict[str, int] = {}

        self._build_universe(body_names, session=session)

    def _build_universe(self, body_names: List[str], session: Optional[Session] = None) -> None:
        """Queries the DB, topologically sorts the bodies, and populates the arrays."""
        local_session = False
        if session is None:
            session = get_session()
            local_session = True

        orm_bodies = session.query(BaseBodyORM).filter(BaseBodyORM.name.in_(body_names)).all()

        if len(orm_bodies) != len(body_names):
            found = [b.name for b in orm_bodies]
            missing = set(body_names) - set(found)
            raise ValueError(f"Could not find bodies in database: {missing}")
        
        # --- Pass 1: Topological Sort ---
        # Ensure that parent bodies are instantiated before their children to maintain a valid hierarchy.
        sorted_bodies = self._topological_sort(orm_bodies)

        # --- Pass 2: Populate Arrays ---
        # Set the parent indices and populate the COE and mu arrays based on the sorted order.
        for orm_body in sorted_bodies:
            idx = self.free_indices.pop()
            self.name_to_index[orm_body.name] = idx
            self.mu_array[idx] = orm_body.mu

            if orm_body.parent and (orm_body.parent in self.name_to_index):
                self.coe_states[idx, 0] = orm_body.p or 0.0
                self.coe_states[idx, 1] = orm_body.e or 0.0
                self.coe_states[idx, 2] = orm_body.i or 0.0
                self.coe_states[idx, 3] = orm_body.raan or 0.0
                self.coe_states[idx, 4] = orm_body.arg_pe or 0.0
                self.coe_states[idx, 5] = orm_body.theta or 0.0

                self.parent_indices[idx] = self.name_to_index[orm_body.parent]
            else:
                self.parent_indices[idx] = -1
                self.coe_states[idx] = 0.0
        
        if local_session:
            session.close()

        loaded_count = len(sorted_bodies)
        root_count = np.sum(self.parent_indices[:loaded_count] == -1)
        if root_count > 1:
            raise ValueError(f"Disconnected system detected! The simulation has {root_count} root nodes. Ensure all bodies trace back to a single parent.")
        
        # --- Pass 3: Dynamic mass aggregation for barycenters ---
        # Iterate BACKWARDS (Bottom-Up) through toplogically sorted list.
        # Ensures that inner systems calculate their mass before outer systems.

        for orm_body in reversed(sorted_bodies):
            if isinstance(orm_body, VirtualBodyORM):
                idx = self.name_to_index[orm_body.name]

                # Find all bodies whose parent pointer is this Barycenter
                child_mask = (self.parent_indices[:loaded_count] == idx) # NumPy C-Level cast optimization

                # Sum their masses and assign it to the Barycenter's mu array slot
                system_mu = np.sum(self.mu_array[:loaded_count][child_mask])
                self.mu_array[idx] = system_mu
        
        print(f"Universe Built: {loaded_count} bodies loaded.")
        print(f"Mass Array: {self.mu_array[:loaded_count]}")
    
    def _topological_sort(self, orm_bodies: List[BaseBodyORM]) -> List[BaseBodyORM]:
        """Ensures a parent body is instantiated before it's children"""
        sorted_bodies: List[BaseBodyORM] = []
        processed_names = set()

        # 1. Defer Vessels for Patched Conics
        system_bodies = [b for b in orm_bodies if not isinstance(b, VesselORM)]
        vessels = [b for b in orm_bodies if isinstance(b, VesselORM)]

        # 2. Track what the user actually loaded
        loaded_names = {b.name for b in system_bodies}

        while len(sorted_bodies) < len(system_bodies):
            start_len = len(sorted_bodies)

            for body in system_bodies:
                if body.name in processed_names:
                    continue

                # A body is ready to sort if:
                # A. It has no parent
                # B. It's parent exists in DB, but wasn't loaded in the sim
                # C. It's parent has already been sorted
                if body.parent is None or \
                   body.parent not in loaded_names or \
                   body.parent in processed_names:

                    sorted_bodies.append(body)
                    processed_names.add(body.name)
                
            if start_len == len(sorted_bodies):
                raise ValueError("Circular dependency detected in body hierachy. Ensure all bodies trace back to a single parent.")
            
        # 3. Safely append all vessels at end of memory block.
        sorted_bodies.extend(vessels)

        return sorted_bodies

    # def _allocate_body(self, orm_body: BaseBodyORM) -> BodyHandle:
    #     pass



        

    # @property
    # def current_epoch(self) -> datetime:
    #     return self.start_epoch + timedelta(seconds=self.t)
    
    # # def add_body(self, body: BaseBody) -> None:
    # #     self.bodies.append(body)

    # #     if body.parent and body.elements is None:
    # #         body.sync_elements()

    # def step(self, dt: Seconds) -> None:
    #     for body in self.bodies:
    #         KeplerianPropagator.propagate(body, dt)
        
    #     self.t += dt
    #     self._record_state()

    # def run(self, duration: Seconds, dt: Seconds) -> None:
    #     if self.t == 0:
    #         self._record_state()

    #     steps = int(duration/dt)
    #     for _ in range(steps):
    #         self.step(dt)

    # def _record_state(self) -> None:
    #     """Internal helper to snap the current state of all bodies."""
    #     current_dt = self.start_epoch + timedelta(seconds=self.t)
    #     for body in self.bodies:
    #         self._history_buffer.append({
    #             "timestamp": current_dt,
    #             "seconds": self.t,
    #             "body": body.name,
    #             "x": body.r[0], "y": body.r[1], "z": body.r[2],
    #             "vx": body.v[0], "vy": body.v[1], "vz": body.v[2],
    #             "e": body.elements.e if body.elements else None,
    #             "theta": body.elements.theta if body.elements else None
    #         })

    # @property
    # def history(self) -> pd.DataFrame:
    #     return pd.DataFrame(self._history_buffer)
    
    # def clear_history(self) -> None:
    #     self._history_buffer = []


    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)
    
    # # def add_body(self, body: BaseBody) -> None:
    # #     self.bodies.append(body)

    # #     if body.parent and body.elements is None:
    # #         body.sync_elements()

    def step(self, dt: Seconds) -> None:
        # for body in self.bodies:
        #     KeplerianPropagator.propagate(body, dt)
        mask = (self.parent_indices != -1)
        # KeplerianPropagator.propagate(self.coe_states[mask], self.mu_array[mask], self.parent_indices[mask], dt)
        KeplerianPropagator.propagate(self.coe_states, self.mu_array, self.parent_indices, dt)
        self.local_states[mask, 0:3], self.local_states[mask, 3:6] = fr.ReferenceFrames.coe_to_rv(self.coe_states[mask], self.mu_array[mask])
        
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
        # for body in self.bodies:
        #     self._history_buffer.append({
        #         "timestamp": current_dt,
        #         "seconds": self.t,
        #         "body": body.name,
        #         "x": body.r[0], "y": body.r[1], "z": body.r[2],
        #         "vx": body.v[0], "vy": body.v[1], "vz": body.v[2],
        #         "e": body.elements.e if body.elements else None,
        #         "theta": body.elements.theta if body.elements else None
        #     })
        for body in self.name_to_index:
            index = self.name_to_index[body]
            self._history_buffer.append({
                "timestamp": self.current_epoch,
                "seconds": self.t,
                "body": body,
                "x": self.local_states[index, 0], "y": self.local_states[index, 1], "z": self.local_states[index, 2],
                "vx": self.local_states[index, 3], "vy": self.local_states[index, 4], "vz": self.local_states[index, 5],
                "e": self.coe_states[index, 1],
                "theta": self.coe_states[index, 5]
                })

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history_buffer)
    
    def clear_history(self) -> None:
        self._history_buffer = []

if __name__ == "__main__":
    sim = Simulation(body_names=["Earth", "Moon", "Sun"])

    sim.run(24*(60**2), 0.5*(60.0**2.0))
    print(sim.history)
