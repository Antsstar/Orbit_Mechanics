from orbital_engine.database import CelestialBodyORM, VesselORM, VirtualBodyORM

def test_polymorphic_inheritance(db_session):
    """Verify that different body types correctly insert and query."""

    # 1. Create one of each type
    earth = CelestialBodyORM(name="Earth", mu=398600.4418, radius=6371.0)
    barycenter = VirtualBodyORM(name="Earth-Moon Barycenter")
    apollo = VesselORM(name="Apollo 11", dry_mass=5000.0)

    # 2. Add to our RAM database
    db_session.add_all([earth, barycenter, apollo])
    db_session.commit()

    # 3. Query them using their specific classes
    queried_earth = db_session.query(CelestialBodyORM).filter_by(name="Earth").first()
    queried_barycenter = db_session.query(VirtualBodyORM).filter_by(name="Earth-Moon Barycenter").first()
    queried_apollo = db_session.query(VesselORM).filter_by(name="Apollo 11").first()

    assert queried_earth is not None
    assert queried_earth.mu == 398600.4418

    assert queried_barycenter is not None
    assert queried_barycenter.name == "Earth-Moon Barycenter"

    assert queried_apollo is not None
    assert queried_apollo.dry_mass == 5000.0
    
