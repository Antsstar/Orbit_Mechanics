import numpy as np
# from .utilities import Units
from .constants import G
from .frames import ReferenceFrames
from .database import get_session, BaseBodyORM

class BaseBody:
    def __init__(self, name: str, radius: float = None, mu: float = None, rotation_rate: float = 0, parent=None, *, mass=None):
        self.name = name
        self.rotation_rate = rotation_rate
        self.parent = parent
        self.children = []
        self.physics_models = {}

        # Initialisation routing
        if radius == None and mu == None and mass == None:
            self._load_from_database()
        else:
            # if radius == None:
            #     raise ValueError(f"Body '{name}' requires a radius for initialisation.") # Not necessary at the moment but will be if collision detection is implemented.
            self.radius = radius

            if mu != None:
                self.mu_self = mu
                self.mass = mu / G
            elif mass != None:
                self.mass = mass
                self.mu_self = G * mass
            else:
                raise ValueError(f"Body '{name}' requires either mu or mass for initialisation.")

        # if mass != None:
        #     self.mass = mass
        #     # self.mu_self = Units.G * mass
        #     # # pass #Calculate mu from Units.G
        #     self.mu_self = G * mass
        #     # pass #Calculate mu from constants.G
        # elif mu != None:
        #     self.mu_self = mu
        #     # self.mass = mu / Units.G
        #     self.mass = mu / G
        # else:
        #     # print("Please input a value for mu or mass of body")
        #     # return
        #     raise ValueError(f"Body '{name}' requires either mu or mass for initialisation.")

        # State Vectors and Orbital Elements
        self.r = np.zeros(3)
        self.v = np.zeros(3)
        self.elements = None

        #Reference Frames
        self.ref_x = np.array([1, 0, 0])
        self.ref_z = np.array([0, 0, 1])
        self.ref_y = np.cross(self.ref_z, self.ref_x)

        self._system_mu = self.mu_self
        self._dirty = True
    
    def _load_from_database(self):
        """ Querries Database for body parameters."""
        session = get_session()
        db_record = session.query(BaseBodyORM).filter_by(name=self.name).first()

        if not db_record:
            session.close()
            raise ValueError(f"Body '{self.name}' not found in database."
                             f" Please provide mu or mass for initialisation manually.")
        
        self.radius = db_record.radius
        self.mu_self = db_record.mu
        self.mass = self.mu_self / G

        if db_record.physics_models:
            self.physics_models = db_record.physics_models

        session.close()

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