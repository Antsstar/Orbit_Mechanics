from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast
import numpy as np
from numpy.typing import NDArray

from .custom_types import (
    Numeric,
    ScalarFloat,
    ArrayFloat,
    ScalarKilometers,
    KmPerSec,
    ScalarGravitationalParameter
)
from .exceptions import TopologyError

if TYPE_CHECKING:
    from .simulator import Simulation

@dataclass(frozen=True)
class BodyHandle:
    """
    Lightweight, Immutable UI pointer to identify data in the central Data-Oriented Design (DOD) memory arena.
    
    *Architecture Note*
    - This class owns no kinematic data. All of this data is held in the host simulation arena, and is availiable on demand
    using self.index to slice the array.
    - Helps us avoid memory duplication, pointer chasing, and memory reuse efficiently optimising code execution.
    """
    name: str
    index: int
    _sim: Simulation
    parent_name: Optional[str] = None

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Safe Guarding
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Checks if this entity is currently alive in the simulation arena."""
        return bool(self._sim.active_mask[self.index] and self.name in self._sim.name_to_index)

    @property
    def is_head(self) -> bool:
        """Check if this entity is the head body of its allocated system."""
        return bool(self._sim.is_head[self.index] and self.is_active)

    @property
    def is_barycenter(self) -> bool:
        """Check if this entity is defined as a system barycenter."""
        return bool(self._sim.is_system[self.index] and self.is_active)

    @property
    def is_vessel(self) -> None:
        """Check if this entity is an active vessel."""
        pass

    @property
    def parent_index(self) -> Optional[int]:
        """Return the integer arena index of this body's gravitational parent."""
        return int(self._sim.parent_indices[self.index]) if self.is_active else None

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Dynamic invariant properties (Mass, S.O.I, Physical Scale)
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    @property
    def mu(self) -> Optional[ScalarGravitationalParameter]:
        """Gravitational Parameter G*M (km^3 / s^2)."""
        return self._sim.mu_array[self.index] if self.is_active else None
        # return cast(ScalarFloat, float)

    @property
    def soi_radius(self) -> Optional[ScalarKilometers]:
        """
        Static Sphere of Influence (S.O.I) or Hill Sphere radius in Kilometers.
        Used for high-speed BVH boundary routing during propagation.
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Kinematic properties (Displacement, Velocity)
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    @property
    def r_local(self) -> Optional[ArrayFloat]:
        """
        Local Cartesian Position Vector respective to the entities allocated system barycenter.
        **Returns a NumPy View (not copy) of the data**.
        """
        return self._sim.local_states[self.index, :3] if self.is_active else None

    @property
    def v_local(self) -> Optional[ArrayFloat]:
        """
        Local Cartesian Velocity Vector respective to the entities allocated system barycenter.
        **Returns a NumPy View (not copy) of the data**.
        """
        return self._sim.local_states[self.index, 3:] if self.is_active else None

    @property
    def r_global(self) -> Optional[ArrayFloat]:
        """
        Global Cartesian Position Vector respective to the Simulation root.
        **Returns a NumPy View (not copy) of the data**.
        """
        return self._sim.global_states[self.index, :3] if self.is_active else None

    @property
    def v_global(self) -> Optional[ArrayFloat]:
        """
        Global Cartesian Velocity Vector respective to the Simulation root.
        **Returns a NumPy View (not copy) of the data**.
        """
        return self._sim.global_states[self.index, 3:] if self.is_active else None

    @property
    def coe(self) -> Optional[ArrayFloat]:
        """
        Classical Orbital Elements [p, e, i, Omega, omega, theta] relative to local parent.
        **Returns a NumPy View (not copy) of the data.**
        """
        return self._sim.coe_states[self.index] if self.is_active else None

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Dynamic in-place state mutators
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def apply_impulsive_delta_v(self, del_v: ArrayFloat) -> None:
        pass

    def __repr__(self) -> str:
        status = "ACTIVE" if self.is_active else "DEPRICATED"
        role = "BARYCENTER" if self.is_barycenter else ("HEAD" if self.is_head else "Body")
        return f"<BodyHandle [{status}] '{self.name}' (idx={self.index}, role={role}, parent='{self.parent_name}')>"