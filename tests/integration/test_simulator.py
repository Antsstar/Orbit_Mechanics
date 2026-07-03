from orbital_engine.simulator import Simulation
from orbital_engine.database import CelestialBodyORM
import math as m

def test_simulation_initialization(db_session):
    """Verify that the Simulation class initializes correctly with a set of bodies from the database."""

    # 1. Seed the database with some celestial bodies
    earth = CelestialBodyORM(name="Earth", mu=398600.4418, radius=6371.0, parent="Sun",
                            p=149597870.7*(1 - 0.0167**2), e=0.0167, i=m.radians(0.00005), raan=0.0, arg_pe=0.0, theta=0.0)
    moon = CelestialBodyORM(name="Moon", mu=4902.800066, radius=1737.1, parent="Earth", 
                            p=384400.0 * (1 - 0.0549**2), e=0.0549, i=m.radians(5.145), raan=0.0, arg_pe=0.0, theta=0.0)
    db_session.add(earth)
    db_session.add(moon)
    db_session.commit()

    # 2. Initialize the simulation with these bodies
    sim = Simulation(body_names=["Earth", "Moon"], session=db_session)

    # 3. Verify that the simulation has been initialized correctly
    assert sim is not None

    assert len(sim.name_to_index) == 2 # We only loaded 2 bodies
    assert "Earth" in sim.name_to_index
    assert "Moon" in sim.name_to_index

    assert sim.local_states.shape == (10000, 6) # Basic shape checks
    assert sim.coe_states.shape == (10000, 6)
    assert sim.mu_array.shape == (10000,)

    assert all(x == 0.0 for x in sim.coe_states[sim.name_to_index["Earth"]]) # Earth is the root body, so its COE should be filled with zeros
    assert any(x != 0.0 for x in sim.coe_states[sim.name_to_index["Moon"]])  # Moon has a parent, so its COE should be populated with its orbital elements

    assert sim.parent_indices[sim.name_to_index["Earth"]] == -1 # Earth has no parent
    assert sim.parent_indices[sim.name_to_index["Moon"]] == sim.name_to_index["Earth"] # Moon's parent index should point to Earth

    assert sim.mu_array[sim.name_to_index["Earth"]] == 398600.4418
    assert sim.mu_array[sim.name_to_index["Moon"]] == 4902.800066


