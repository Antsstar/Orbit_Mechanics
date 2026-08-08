import numpy as np

print("Hello World")
MU = 3.986004418 * 10**5 #398600 km^3/s^2 Earth
MU_Sun = 1.32712440042 * 10**11


def rv(r=np.array([]), v=np.array([]), *, mu):
    return r, v

def angle(a, b):
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))



def main():
    # r = np.array([1, 2, 3])
    # v = np.array([4, 5, 6])
    r = np.array([1.023, 1.076, 1.011])
    v = np.array([.62, .7, -.25])

    # r = r / np.linalg.norm(r)
    # v = v / np.linalg.norm(v)

    h = np.cross(r, v)

    print("Vector h: ", h)

    z = np.array([0, 0, 1])
    
    # i = np.arccos(np.dot(r, v) / np.linalg.norm(r) * np.linalg.norm(v))
    i = np.arccos(np.dot(h, z) / (np.linalg.norm(h) * np.linalg.norm(z)))

    print("Inclination: ", np.degrees(i))

    N = np.cross(z, h)

    print("Ascending Node Vector: ", N)

    x = np.array([1, 0, 0])

    Omega = np.arccos(np.dot(N, x) / (np.linalg.norm(N) * np.linalg.norm(x)))

    if N[1] < 0:
        Omega = 2*np.pi - Omega

    print("RAAN: ", np.degrees(Omega))

    e = np.cross(v, h) - r / np.linalg.norm(r)
    print("Eccentricity vector: ", e)

    omega = np.arccos(np.dot(e, N) / (np.linalg.norm(e) * np.linalg.norm(N)))

    if e[2] < 0:
        omega = 2*np.pi - omega

    print("Argument of Perigee: ", np.degrees(omega))

    theta = np.arccos(np.dot(r, e) / (np.linalg.norm(r) * np.linalg.norm(e)))

    if np.dot(r, v) < 0:
        theta = 2*np.pi - theta
    
    print("True anomaly: ", np.degrees(theta))
    print("Eccenticity: ", np.linalg.norm(e))
    # a = np.linalg.norm(h)**2
    
    print("\n\n New formatting! \n\n")

    r = np.array([-145510750, 39268690, 10500])
    v = np.array([-6.995, -29.215, -0.00025])

    # elements = coe_from_rv(r, v, mu=1)
    elements = coe_from_rv(r, v, mu=MU_Sun)
    # print(elements)
    for i, (key, value) in enumerate(elements.items()):
        if value:
            if i > 2:
                value = np.degrees(value)
            if i > 0:
                value = f"{value:.3f}"
            print(f"{key:<15} :   {value}")

    
    M = eccen_to_mean(2.45, elements["Eccentricity"])
    E = mean_to_eccen(M, elements["Eccentricity"])
    print(np.linalg.norm(r))
    print(np.sqrt(MU_Sun / np.linalg.norm(r)))
    print(E)
    print(kepler_equation(MU, elements["Semi-major axis"], elements["Eccentricity"], 1, E))
    # print(np.degrees(E))
    # print(np.degrees(kepler_equation(MU, elements["Semi-major axis"], elements["Eccentricity"], 1, E)) % 360)
    return

def coe_from_rv(r, v, *, mu = MU, ref_x = np.array([1, 0, 0]), ref_z = np.array([0, 0, 1])):
    if np.dot(ref_x, ref_z) != 0:
        print("Reference directions are not orthogonal!")
        return
    
    ref_y = np.cross(ref_z, ref_x)

    orb_case    = ""   # Classification of orbit
    a           = None # Semi-major axis > Semi-latus rectum "p" = h^2 / mu
    e_mag       = None # Eccentricity
    i           = None # Inclination
    Omega       = None # Right Ascension of the Ascending Node (RAAN)
    omega       = None # Argument of Perigee
    theta       = None # True anomaly
    tor         = None # Time of periapsis passage
    omega_true  = None # Omega + omega (x_ref.e) Non-Circular Equatorial, True longitude of periapsis
    u           = None # omega + theta (N.r) Circular Inclined, True argument of latitude
    lambda_true = None # Omega + omega + theta (x_ref . r) Circular Equatorial, True longitude

    
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    if h_mag == 0:
        print("Velocity and displacement are parallel. Entity is not in orbit")
        return
    
    e = (np.cross(v, h) / mu) - (r / np.linalg.norm(r))
    e_mag = np.linalg.norm(e)

    a = (h_mag**2 / mu) * (1 / (1 - e_mag**2))

    # match e_mag:
    #     case 0:
    #         orb_case = "Circular"
    #     case 1:

    i = np.arccos(np.dot(h, ref_z) / (np.linalg.norm(h) * np.linalg.norm(ref_z)))

    # if [i, e_mag] == [0, 0]:
    #     orb_case = "Circular, Equatorial"
    #     lambda_true = angle(ref_x, r)

    match [e_mag, i]:
        case [0, 0]:
            # orb_case = "Circular, Equatorial"
            e_type = "Circular"
            i_type = "Equatorial"
            lambda_true = angle(ref_x, r)
            if np.dot(ref_y, r) < 0:
                lambda_true = 2*np.pi - lambda_true

        case [0, _]:
            # orb_case = "Circular"
            e_type = "Circular"
            N = np.cross(ref_z, h)

            Omega = angle(N, ref_x)
            if np.dot(ref_y, N) < 0:
                Omega = 2*np.pi - Omega

            u = angle(r, N)
            if np.dot(ref_z, r) < 0:
                u = 2*np.pi - u

        case [_, 0]:
            # orb_case = "Equatorial"
            i_type = "Equatorial"
            omega_true = angle(ref_x, e)
            if np.dot(ref_y, e) < 0:
                omega_true = 2*np.pi - omega_true

        case _:
            N = np.cross(ref_z, h)

            Omega = angle(N, ref_x)
            if np.dot(ref_y, N) < 0:
                Omega = 2*np.pi - Omega

            omega = angle(e, N)
            if np.dot(ref_z, e) < 0:
                omega = 2*np.pi - omega

    if e_mag != 0:
        theta = angle(r, e)
        if np.dot(r, v) < 0:
            theta = 2*np.pi - theta

        if e_mag < 1:
            e_type = "Elliptic"
        elif e_mag == 1:
            e_type = "Parabolic"
        else:
            e_type = "Hyperbolic"

        # orb_case = ", ".join([e_type, orb_case])
    
    if i != 0:
        if np.degrees(i) < 90:
            i_type = "Pro-grade"
        elif np.degrees(i) == 90:
            i_type = "Polar"
        else:
            i_type = "Retro-grade"

        # orb_case = ", ".join([orb_case, i_type])

    orb_case = ", ".join([e_type, i_type])

    elements = {
        "Class": orb_case,
        "Semi-major axis": a,
        "Eccentricity": e_mag,
        "Inclination": i,
        "RAAN": Omega,
        "Arg of Perigee": omega,
        "True anomaly": theta,
        "True long of periapsis": omega_true,
        "True arg of Lat": u,
        "True long": lambda_true
    }

    return elements
    # return [orb_case, a, e_mag, i, Omega, omega, theta, omega_true, u, lambda_true]

def true_to_eccen(theta, e):
    tanE_2 = np.sqrt( (1 - e) / (1 + e) ) * np.tan(theta / 2)
    return 2*np.atan(tanE_2)

def eccen_to_true(E, e):
    tanTheta_2 = np.sqrt( (1 + e) / (1 - e)) * np.tan(E / 2)
    return 2*np.atan(tanTheta_2)

def eccen_to_mean(E, e):
    return E - e*np.sin(E)

def mean_to_eccen(M, e, *, tol=1e-5, solver="N-R", max_ite = 1000):
    E_0 = 0

    if e <= 0.55:
        E_0 = M
    elif 0.55 < e <= 0.95:
        E_0 = np.cbrt(6*M)
    elif 0.95 < e <= 1:
        E_0 = np.pi
    else:
        print("Switching to Hyperbolic solver!")
        E_1 = "Placeholder"
        return E_1
    
    ite = 0
    if solver == "N-R":
        # E_1 = E_0 - (M - E_0 + e*np.sin(E_0)) / (1 + e*np.cos(E_0))
        def func(E):
            return E - ((M - E + e*np.sin(E)) / (e*np.cos(E) - 1))
    elif solver == "S.S":
        def func(E):
            return M + e*np.sin(E)
        
    else:
        print("Not a valid solver option")
        return None
    

    while True:
        # if solver == "N-R":
        #     E_1 = E_0 - (M - E_0 + e*np.sin(E_0)) / (1 + e*np.cos(E_0))
        E_1 = func(E_0)
        error = abs(E_1 - E_0)
        ite += 1
        
        if error < tol:
            break

        if ite >= max_ite:
            print("Did not converge!")
            return None

        E_0 = E_1

    print(f"Converged in {ite}/{max_ite} iterations!")
    return E_1

def kepler_equation(mu, a, e, delta_t, E_0):
    print(2*np.pi*np.sqrt(a**3 / mu))
    delta_M = np.sqrt(mu / (a**3)) * delta_t
    
    M = delta_M + E_0 - e*np.sin(E_0)
    E_1 = mean_to_eccen(M, e)
    return E_1

        

if __name__ == "__main__":
    main()
