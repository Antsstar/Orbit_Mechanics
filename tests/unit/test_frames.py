import pytest
import numpy as np
from datetime import datetime
from orbital_engine.frames import OrbitalElements as oe, ReferenceFrames as rf


def test_orbit_classifications():
    mu = 3.986004418e5
    r = np.array([0.0, 12000.0, 0.0])
    v = np.array([-np.sqrt(mu/r[1]), 0.0, 0.0])

    coe, _ = rf.rv_to_coe(r, v, mu)

    assert coe[0] == pytest.approx( (np.linalg.norm(r)*np.linalg.norm(v))**2 / mu, abs=1e-9)
    assert coe[1] == pytest.approx(0, abs=1e-13)
    assert coe[2] == pytest.approx(0, abs=1e-13)
    assert coe[5] == pytest.approx(np.pi/2, abs=1e-9)

    r = np.array([0.0, 0.0, 16000.0])
    v = np.array([0.0, np.sqrt(2*mu/r[2]), 0.0])

    coe, _ = rf.rv_to_coe(r, v, mu)
 
    assert coe[1] == pytest.approx(1, abs=1e-13)
    assert coe[2] == pytest.approx(np.pi/2, abs=1e-13)
    assert coe[3] == pytest.approx(3*np.pi/2, abs=1e-13)
    assert coe[4] == pytest.approx(np.pi/2, abs=1e-13)
    assert coe[5] == pytest.approx(0, abs=1e-9)

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, -np.sqrt(0.75*mu / r[0]), np.sqrt(0.75*mu / r[0])])

    coe, _ = rf.rv_to_coe(r, v, mu)

    assert coe[0] == pytest.approx((np.linalg.norm(r)*np.linalg.norm(v))**2 / mu, abs=1e-6)
    assert coe[1] == pytest.approx(0.5, abs=1e-12)
    assert coe[2] == pytest.approx(3*np.pi / 4, abs=1e-12)
    assert coe[3] == pytest.approx(0.0, abs=1e-12)
    assert coe[4] == pytest.approx(0.0, abs=1e-12)
    assert coe[5] == pytest.approx(0.0, abs=1e-12)

    r = np.array([10000.0, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(1.5*mu/r[0]), np.sqrt(1.5*mu/r[0])])

    coe, _ = rf.rv_to_coe(r, v, mu)

    assert coe[0] == pytest.approx((np.linalg.norm(r)*np.linalg.norm(v))**2 / mu, abs=1e-6)
    assert coe[1] == pytest.approx(2.0, abs=1e-12)
    assert coe[2] == pytest.approx(np.pi/4, abs=1e-12)
    assert coe[3] == pytest.approx(0.0, abs=1e-12)
    assert coe[4] == pytest.approx(0.0, abs=1e-12)
    assert coe[5] == pytest.approx(0.0, abs=1e-12)


def test_rv_coe_conversions_vectorized_fuzzing():
    mu = 398600.4418

    # np.random.seed(301)
    np.random.seed(int((datetime.now().timestamp() * 1000) % (2**32)))

    r_dir = np.random.rand(100, 3)
    r_dir /= np.linalg.norm(r_dir, axis=-1, keepdims=True)
    r_mag = np.random.uniform(6500.0, 50000.0, 100)
    r_in = r_dir * r_mag[..., np.newaxis]

    v_dir = np.random.rand(100, 3)
    v_dir /= np.linalg.norm(v_dir, axis=-1, keepdims=True)
    v_mag = np.random.uniform(1.0, 15.0, 100)
    v_in = v_dir * v_mag[..., np.newaxis]

    h = np.cross(r_in, v_in)
    mask = (np.linalg.norm(h, axis=-1) < 1e-3)
    coe, _ = rf.rv_to_coe(r_in, v_in, mu*np.ones(100, dtype=np.float64))
    r_out, v_out, _ = rf.coe_to_rv(coe, mu)

    assert np.all(np.abs(r_out[mask] - r_in[mask]) < 1e-7)
    assert np.all(np.abs(v_out[mask] - v_in[mask]) < 1e-7)


def test_rv_coe_conversions_edgecase():
    mu = 398600.4418

    np.random.seed(10)  # Distinct seed for edge cases

    for _ in range(25):  # Run 25 variations of each edge case
        r_mag = np.random.uniform(6500.0, 42000.0)
        
        # ----------------------------------------------------
        # CASE 1: Forced Equatorial Orbit (Z = 0, Vz = 0)
        # ----------------------------------------------------
        # Constraining positions and velocities entirely to the XY plane
        r_eq = np.array([r_mag, 0.0, 0.0])  # Simplest alignment on X-axis
        v_mag = np.random.uniform(3.0, 7.0)
        v_eq = np.random.choice([-1, 1]) * np.array([0.0, v_mag, 0.0])  # Perpendicular in XY plane

        coe_eq, _ = rf.rv_to_coe(r_eq, v_eq, mu)
        assert coe_eq[2] == pytest.approx(0.0, abs=1e-12) or coe_eq[2] == pytest.approx(np.pi, abs=1e-12)
        r_out, v_out, _ = rf.coe_to_rv(coe_eq, mu)
        assert r_out == pytest.approx(r_eq, abs=1e-4)
        assert v_out == pytest.approx(v_eq, abs=1e-4)

        # ----------------------------------------------------
        # CASE 2: Forced Circular Orbit (v = sqrt(mu/r) and r ⊥ v)
        # ----------------------------------------------------
        # Generate a random 3D position vector
        r_dir = np.random.randn(3)
        r_dir /= np.linalg.norm(r_dir)
        r_circ = r_dir * r_mag
        
        # Create a perfectly orthogonal velocity vector
        v_dir = np.random.randn(3)
        v_dir = np.cross(r_circ, v_dir)  # Cross product forces orthogonality
        v_dir /= np.linalg.norm(v_dir)
        v_circ = v_dir * np.sqrt(mu / r_mag)  # Exact circular velocity magnitude

        coe_circ, _ = rf.rv_to_coe(r_circ, v_circ, mu)
        assert coe_circ[1] == pytest.approx(0.0, abs=1e-6)
        r_out, v_out, _ = rf.coe_to_rv(coe_circ, mu)
        assert r_out == pytest.approx(r_circ, abs=1e-4)
        assert v_out == pytest.approx(v_circ, abs=1e-4)

        # ----------------------------------------------------
        # CASE 3: Forced Polar Orbit (Angular Momentum Hz = 0)
        # ----------------------------------------------------
        # Position on Equator, Velocity pointing straight north (along Z)
        r_pol = np.array([r_mag, 0.0, 0.0])
        v_mag_pol = np.random.uniform(3.0, 7.0)
        v_pol = np.array([0.0, 0.0, v_mag_pol])
        
        coe_pol, _ = rf.rv_to_coe(r_pol, v_pol, mu)
        assert coe_pol[2] == pytest.approx(np.pi / 2, abs=1e-12)
        r_out, v_out, _ = rf.coe_to_rv(coe_pol, mu)
        assert r_out == pytest.approx(r_pol, abs=1e-4)
        assert v_out == pytest.approx(v_pol, abs=1e-4)

        # ----------------------------------------------------
        # CASE 4: Forced Parabolic Orbit (v = sqrt(2*mu/r) and r ⊥ v)
        # ----------------------------------------------------
        r_dir_para = np.random.randn(3)
        r_dir_para /= np.linalg.norm(r_dir_para)
        r_para = r_dir_para * r_mag
        
        v_dir_para = np.random.randn(3)
        v_dir_para = np.cross(r_para, v_dir_para)
        v_dir_para /= np.linalg.norm(v_dir_para)
        v_para = v_dir_para * np.sqrt(2.0 * mu / r_mag)  # Exact escape velocity magnitude
        

        coe_para, _ = rf.rv_to_coe(r_para, v_para, mu)
        assert coe_para[1] == pytest.approx(1.0, abs=1e-6)
        r_out, v_out, _ = rf.coe_to_rv(coe_para, mu)
        assert r_out == pytest.approx(r_para, abs=1e-3)
        assert v_out == pytest.approx(v_para, abs=1e-3)
