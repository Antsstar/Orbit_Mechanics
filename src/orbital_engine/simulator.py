from __future__ import annotations
from typing import List, Optional, Any
from .custom_types import Seconds

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# from .body import BaseBody
from .propagators import KeplerianPropagator
from .database import get_session, CelestialBodyORM, BaseBodyORM, VesselORM
from .body import BodyHandle

class Simulation:
    # def __init__(self, start_epoch: Optional[datetime] = None) -> None:
    #     self.start_epoch: datetime = start_epoch if start_epoch else datetime.now()

    #     self.t: Seconds = 0.0
    #     self.bodies: List[BaseBody] = []
    #     self._history_buffer: List[dict[str, Any]] = []
    def __init__(self, body_names: List[str], max_capacity: int = 10000, start_epoch: Optional[datetime] = None) -> None:
        self.start_epoch: datetime = start_epoch if start_epoch else datetime.now()
        self.max_capacity = max_capacity

        self.local_states = np.zeros((max_capacity, 6), dtype=np.float64)
        self.coe_states = np.zeros((max_capacity, 6), dtype=np.float64)
        self.mu_array = np.zeros(max_capacity, dtype=np.float64)
        self.parent_indices = np.full(max_capacity, -1, dtype=np.int32)

        # The free-list stack.
        self.free_indices = list(range(max_capacity-1, -1, -1))

        # Name to integer mapping
        self.name_to_index: dict[str, int] = {}

        self._build_universe(body_names)

    def _build_universe(self, body_names: List[str]) -> None:
        """Queries the DB, topologically sorts the bodies, and populates the arrays."""
        session = get_session()

        orm_bodies = session.query(BaseBodyORM).filter(BaseBodyORM.name.in_(body_names)).all()

        if len(orm_bodies) != len(body_names):
            found = [b.name for b in orm_bodies]
            missing = set(body_names) - set(found)
            # session.close()
            raise ValueError(f"Could not find bodies in database: {missing}")
        
        sorted_bodies = self._topological_sort(orm_bodies)


        # for orm_body in sorted_bodies:
        #     idx = self.free_indices.pop()
        #     self.name_to_index[orm_body.name] = idx
        #     self.mu_array[idx] = orm_body.mu

        for orm_body in sorted_bodies:
            idx = self.free_indices.pop()
            self.name_to_index[orm_body.name] = idx
            self.mu_array[idx] = orm_body.mu

            # child_idx = self.name_to_index[orm_body.name]

            if orm_body.parent and (orm_body.parent in self.name_to_index):
            #     parent_idx = self.name_to_index[orm_body.parent]
            #     self.parent_indices[child_idx] = parent_idx
            # else:
            #     self.parent_indices[child_idx] = -1
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
        
        session.close()

        loaded_count = len(sorted_bodies)
        root_count = np.sum(self.parent_indices[:loaded_count] == -1)
        if root_count > 1:
            raise ValueError(f"Disconnected system detected! The simulation has {root_count} root nodes. Ensure all bodies trace back to a single parent.")
        
        # print(self.coe_states[:loaded_count])
        # print(loaded_count)
        # print(sorted_bodies)
    
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

    def _allocate_body(self, orm_body: BaseBodyORM) -> BodyHandle:
        pass



        

    @property
    def current_epoch(self) -> datetime:
        return self.start_epoch + timedelta(seconds=self.t)
    
    # def add_body(self, body: BaseBody) -> None:
    #     self.bodies.append(body)

    #     if body.parent and body.elements is None:
    #         body.sync_elements()

    def step(self, dt: Seconds) -> None:
        for body in self.bodies:
            KeplerianPropagator.propagate(body, dt)
        
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
        for body in self.bodies:
            self._history_buffer.append({
                "timestamp": current_dt,
                "seconds": self.t,
                "body": body.name,
                "x": body.r[0], "y": body.r[1], "z": body.r[2],
                "vx": body.v[0], "vy": body.v[1], "vz": body.v[2],
                "e": body.elements.e if body.elements else None,
                "theta": body.elements.theta if body.elements else None
            })

    @property
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history_buffer)
    
    def clear_history(self) -> None:
        self._history_buffer = []


if __name__ == "__main__":
    sim = Simulation(body_names=["Earth", "Moon", "Sun"])
