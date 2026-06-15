from __future__ import annotations
from typing import Optional, Any
from .types import Radians, Kilometers, Seconds
from numpy.typing import NDArray

import numpy as np
# from .utilities import Units
from .constants import G
from .frames import ReferenceFrames, OrbitalElements
from .database import get_session, BaseBodyORM
from .registry import get_model

class BaseBody:
    def __init__(self, name: str, radius: Optional[float] = None, mu: Optional[float] = None, rotation_rate: float = 0, parent: Optional[BaseBody] = None, *, mass: Optional[float] = None) -> None:
        self.name = name
        self.rotation_rate = rotation_rate
        self.parent = parent
        self.children: list[BaseBody] = []
        self.physics_models: dict[str, Any] = {}

        # Initialisation routing
        if radius is None and mu is None and mass is None:
            self._load_from_database()
        else:
            # if radius is None:
            #     raise ValueError(f"Body '{name}' requires a radius for initialisation.") # Not necessary at the moment but will be if collision detection is implemented.
            self.radius = radius

            if mu is not None:
                self.mu_self = mu
                self.mass = mu / G
            elif mass is not None:
                self.mass = mass
                self.mu_self = G * mass
            else:
                raise ValueError(f"Body '{name}' requires either mu or mass for initialisation.")

        # State Vectors and Orbital Elements
        self.r: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.v: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self.elements: Optional[OrbitalElements] = None

        #Reference Frames
        self.ref_x: NDArray[np.float64] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.ref_z: NDArray[np.float64] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ref_y: NDArray[np.float64] = np.cross(self.ref_z, self.ref_x)

        self._system_mu = self.mu_self
        self._dirty = True
    
    def _load_from_database(self) -> None:
        """ Querries Database for body parameters."""
        session = get_session()
        db_record = session.query(BaseBodyORM).filter_by(name=self.name).first()

        if not db_record:
            session.close()
            raise ValueError(f"Body '{self.name}' not found in database."
                             f" Please provide mu or mass for initialisation manually.")
        
        self.radius = db_record.radius # type: ignore
        self.mu_self = db_record.mu # type: ignore
        self.mass = self.mu_self / G

        if db_record.physics_models:
            # self.physics_models = db_record.physics_models
            for model_type, model_name in db_record.physics_models.items(): # Will outline if it is a AtmosphericModel, GravityModel, etc. and the name of the model to load.
                try:
                    model_class = get_model(model_name)
                    self.physics_models[model_type] = model_class()
                except ValueError as e:
                    raise ValueError(f"Failed to load physics model for {self.name}: {e}")

        session.close()

    def invalidate_cache(self) -> None:
        self._dirty = True
        if self.parent:
            if self.mu_self / self.parent.mu_self > 1e-9:
                self.parent.invalidate_cache()

    def add_child(self, child: 'BaseBody') -> None:
        self.children.append(child)
        child.parent = self
        self.invalidate_cache()

    @property
    def mu_system(self) -> float:
        if self._dirty:
            divisor = self.mu_self if self.mu_self > 0 else 1e-20
            children_mu = sum(c.mu_system for c in self.children if c.mu_self / divisor > 1e-9)
            self._system_mu = self.mu_self + children_mu
            self._dirty = False
        return self._system_mu
    
    @property
    def mu_orbit(self) -> Optional[float]:
        if not self.parent:
            return None
        return self.parent.mu_self + self.mu_system

    def sync_state(self) -> None:
        if not self.parent:
            return
        assert self.elements is not None, "Orbital elements must be defined to sync state."
        assert self.mu_orbit is not None, "Parent body must have a defined mu to sync state."
        r, v = ReferenceFrames.coe_to_rv(self.elements, self.mu_orbit)
        self.r, self.v = r, v
        return

    def sync_elements(self) -> None:
        if not self.parent:
            return
        assert self.mu_orbit is not None, "Parent body must have a defined mu to sync elements."
        elements = ReferenceFrames.rv_to_coe(self.r, self.v, self.mu_orbit)
        self.elements = elements
        return