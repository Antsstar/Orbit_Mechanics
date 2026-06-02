import json
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from .constants import G

Base = declarative_base()

class BaseBodyORM(Base):
    __tablename__ = 'bodies'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)

    # Identifier for the type of celestial body (e.g., 'planet', 'star', 'moon')
    body_type = Column(String(50), nullable=False)

    mu = Column(Float, nullable=False)
    radius = Column(Float, nullable=False)
    physics_models = Column(JSON, nullable=True)

    __mapper_args__ = {
        'polymorphic_on': body_type,
        'polymorphic_identity': 'base_body'
    }

    @property
    def mass(self):
        return self.mu / G
    
    @mass.setter
    def mass(self, new_mass):
        """ Forceful mass update and recalculation of mu. Use with caution! """
        self.mu = new_mass * G

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', mu={self.mu})>"
    
# Child Classes
class CelestialBodyORM(BaseBodyORM):
    __mapper_args__ = {
        'polymorphic_identity': 'celestial_body'
    }

class StellarObjectORM(BaseBodyORM):
    __mapper_args__ = {
        'polymorphic_identity': 'stellar_Object'
    }

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

def get_session():
    """ Querrying function for database session."""
    return SessionLocal()