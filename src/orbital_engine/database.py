from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional, Any, List, Dict

from sqlalchemy import create_engine, Integer, String, Float, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session, relationship, Mapped, mapped_column

from .constants import G
from .custom_types import Kilograms, Kilometers, Radians, SquareMeters

class Base(DeclarativeBase):
    pass

class BaseBodyORM(Base):
    __tablename__ = 'bodies'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    parent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    mu: Mapped[float] = mapped_column(Float, default=0.0)
    physics_models: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    system_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('systems.id'), nullable=True)

    # Classical Orbital Elements (COE) stored at Epoch
    p: Mapped[Optional[Kilometers]] = mapped_column(Float, nullable=True)   # Semi-latus rectum (km)
    e: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # Eccentricity
    i: Mapped[Optional[Radians]] = mapped_column(Float, nullable=True)      # Inclination (rad)
    raan: Mapped[Optional[Radians]] = mapped_column(Float, nullable=True)   # Right Ascension of Ascending Node (rad)
    arg_pe: Mapped[Optional[Radians]] = mapped_column(Float, nullable=True) # Argument of Periapsis (rad)
    theta: Mapped[Optional[Radians]] = mapped_column(Float, nullable=True)  # True Anomaly (rad)

    # Identifier for the type of celestial body (e.g., 'planet', 'star', 'moon')
    body_type: Mapped[str] = mapped_column(String(50), nullable=False)
    __mapper_args__ = {
        'polymorphic_on': body_type,
        'polymorphic_identity': 'base_body'
    }

    @property
    def mass(self) -> Kilograms:
        return self.mu / G
    
    @mass.setter
    def mass(self, new_mass: Kilograms) -> None:
        """ Forceful mass update and recalculation of mu. Use with caution! """
        self.mu = new_mass * G

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', mu={self.mu})>"
    
# Child Classes
class CelestialBodyORM(BaseBodyORM):
    """Planets, Moons, Stars, Asteroids"""
    __mapper_args__ = {
        'polymorphic_identity': 'celestial_body'
    }

    radius: Mapped[Optional[Kilometers]] = mapped_column(Float, nullable=True)
    classification: Mapped[Optional[str]] = mapped_column(String(50), nullable=True) # Stellar, Gas-giant, Planetary, Asteroid.

class VesselORM(BaseBodyORM):
    """Spacecraft and Satellites"""
    __mapper_args__ = {
        'polymorphic_identity': 'vessel'
    }

    dry_mass: Mapped[Kilograms] = mapped_column(Float, default=0.0)
    fuel_mass: Mapped[Optional[Kilograms]] = mapped_column(Float, nullable=True)
    drag_area: Mapped[Optional[SquareMeters]] = mapped_column(Float, nullable=True)

class VirtualBodyORM(BaseBodyORM):
    """Represents Barycenters and Lagrange Points."""
    __mapper_args__ = {
        'polymorphic_identity': 'virtual_node'
    }

# --- Systems table ---
class SystemORM(Base):
    __tablename__ = 'systems'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    body_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    averaged_j2_effect: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    barycenter_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('bodies.id', use_alter=True, name="fk_barycenter_id")
                                                         , nullable=True)

    bodies: Mapped[List["BaseBodyORM"]] = relationship("BaseBodyORM", foreign_keys="[BaseBodyORM.system_id]")

# ----------------------------------------------------------------
# Session and Engine Setup
# ----------------------------------------------------------------

# 1. Get the absolute path to the directory where database.py lives
BASE_DIR = Path(__file__).resolve().parent

# 2. Define the path to the 'data' directory and force it to exist
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 3. Create the absolute path to the database file
DB_PATH = DATA_DIR / 'planets.db'

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_session() -> Session:
    """ Querrying function for database session."""
    return SessionLocal()

def inspect_registry() -> None:
    """Queries the database and prints a formatted summary of available bodies."""
    session = get_session()
    bodies = session.query(CelestialBodyORM).all()

    if not bodies:
        print("Database is currently empty.")
        session.close()
        return
    
    print("\n" + "="*80)
    print(f"{'🌟 ORBITAL ENGINE DATABASE REGISTRY 🌟':^80}")
    print("="*80)

    print(f"{'ID':<4} | {'Name':<12} | {'Parent':<12} | {'Mu':<15} | {'Radius':<12} | {'Physics Models'}")
    print("-"*80)
    
    for b in bodies:
        parent_str = b.parent if b.parent else "None (Root)"

        models = list(b.physics_models.keys()) if b.physics_models else ["None"]
        models_str =  ", ".join(models)

        print(f"{b.id:<4} | {b.name:<12} | {parent_str:<12} | {b.mu:<15.4e} | {b.radius:< 12.1f} | {models_str}")

    print("="*80 + "\n")
    session.close()


if __name__ == "__main__":
    Base.metadata.create_all(engine)

    session = get_session()

    print("--- Running Orbital Engine Database Setup Wizard ---")

    seed_data: List[Dict[str, Any]] = [
        {
            "name": "Sun",
            "parent": None,
            "mu": 1.32712440042e11,
            "radius": 696340.0,
            "classification": "Star",
            "physics_models": {},
            "p": 0.0, "e": 0.0, "i": 0.0, "raan": 0.0, "arg_pe": 0.0, "theta": 0.0
        },
        {
            "name": "Earth",
            "parent": "Sun",
            "mu": 3.986004418e5,
            "radius": 6371.0,
            "classification": "Planet",
            "physics_models": {
                "atmosphere": {
                    "ExponentialDrag": {"rho_0": 1.225e-9, "H": 8.5, "h_0": 0.0}
                }
            },
            "p": 149556260.0,             # p = a * (1 - e^2)
            "e": 0.0167086,               
            "i": 0.0,                     # Earth defines the ecliptic plane
            "raan": math.radians(-11.26), # Longitude of Ascending Node
            "arg_pe": math.radians(114.2),# Argument of Perihelion
            "theta": math.radians(102.34) # True Anomaly at J2000
        },
        {
            "name": "Moon",
            "parent": "Earth",
            "mu": 4.9048695e3,
            "radius": 1737.4,
            "classification": "Moon",
            "physics_models": {},
            "p": 383241.0,                 # p = a * (1 - e^2)
            "e": 0.0549,
            "i": math.radians(5.145),      # Inclination to the ecliptic
            "raan": math.radians(125.08),  # RAAN
            "arg_pe": math.radians(318.15),# Argument of Perigee
            "theta": math.radians(115.0)   # Approximate True Anomaly at J2000
        }
    ]

    for data in seed_data:
        existing_body = session.query(CelestialBodyORM).filter_by(name=data["name"]).first()

        if not existing_body:
            new_body = CelestialBodyORM(**data) #Dictionary unpacking technique to list as keywrds=param structure.
            session.add(new_body)
            print(f"[+] Inserted: {data['name']} (Parent: {data['parent']})")
        else:
            print(f"[~] Skipped: {data['name']} already exists.")

    # Commit and close
    session.commit()
    session.close()

    print("--- Database Setup Complete ---")

    inspect_registry()
