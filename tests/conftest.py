import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orbital_engine.database import Base

@pytest.fixture(scope="function") # Every test function will get a new database session
def db_session():
    """
    Creates a fresh, isolated in-memory SQLite database for every test.
    """

    engine = create_engine("sqlite:///:memory:", echo=False) # Define a memory location for database
    Base.metadata.create_all(engine) # Create the database schema at that memory location

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session # <--- This is where testing happens,

    # - Cleanup after testing has finished
    session.close()
    Base.metadata.drop_all(engine)