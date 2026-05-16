import numpy as np
from .utilities import Units
from .frames import ReferenceFrames

class BaseBody:
    def __init__(self, name, radius, mu=None, rotation_rate = 0, parent=None, *, mass=None):
        self.name = name
        self.radius = radius
        self.rotation_rate = rotation_rate
        self.parent = parent
        self.children = []

        if mass != None:
            self.mass = mass
            self.mu_self = Units.G * mass
            # pass #Calculate mu from Units.G
        elif mu != None:
            self.mu_self = mu
            self.mass = mu / Units.G
        else:
            # print("Please input a value for mu or mass of body")
            # return
            raise ValueError(f"Body '{name}' requires either mu or mass for initialisation.")

        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.elements = None

        #Reference Frames
        self.ref_x = np.array([1, 0, 0])
        self.ref_z = np.array([0, 0, 1])
        self.ref_y = np.cross(self.ref_z, self.ref_x)

        self._system_mu = self.mu_self
        self._dirty = True

    def invalidate_cache(self):
        self._dirty = True
        if self.parent:
            if self.mu_self / self.parent.mu_self > 1e-9:
                self.parent.invalidate_cache()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self.invalidate_cache()

    @property
    def mu_system(self):
        if self._dirty:
            divisor = self.mu_self if self.mu_self > 0 else 1e-20
            children_mu = sum(c.mu_system for c in self.children if c.mu_self / divisor > 1e-9)
            self._system_mu = self.mu_self + children_mu
            self._dirty = False
        return self._system_mu
    
    @property
    def mu_orbit(self):
        if not self.parent:
            return None
        return self.parent.mu_self + self.mu_system

    def sync_state(self):
        if not self.parent:
            return
        r, v = ReferenceFrames.coe_to_rv(self.elements, self.mu_orbit)
        self.r, self.v = r, v
        return

    def sync_elements(self):
        if not self.parent:
            return
        elements = ReferenceFrames.rv_to_coe(self.r, self.v, self.mu_orbit)
        self.elements = elements
        return