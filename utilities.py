# Defining functions and tools for a orb sandbox.

import numpy as np

class Units:
    DEG2RAD = np.pi / 180
    RAD2DEG = 180 / np.pi
    AU2KM   = 1.495978707e8
    KM2AU   = 1 / AU2KM
    C       = 2.99792458e8
    JD2SEC  = 60*60*365.25
    LY2KM   = C * JD2SEC

class Transformations:
    """ Rotation Matrix Toolbox, takes radians as inputs for euler angles about classic X,Y or Z definitions.  """



    @staticmethod
    def Rx(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
            ])
    
    @staticmethod
    def Ry(angle):
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
            ])
    
    @staticmethod
    def Rz(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
            ])
    
    @staticmethod
    def Rxyz(alpha, beta, gamma):
        return Transformations.Rz(gamma) @ Transformations.Ry(beta) @ Transformations.Rx(alpha)
    
    @staticmethod
    def Rzyx(alpha, beta, gamma):
        return Transformations.Rx(gamma) @ Transformations.Ry(beta) @ Transformations.Rz(alpha)
    
    @staticmethod
    def Rzxz(alpha, beta, gamma):
        return Transformations.Rz(gamma) @ Transformations.Rx(beta) @ Transformations.Rz(alpha)

class Anomalies:
    """
    Anomaly Toolbox
    """
    @staticmethod
    def true_to_eccentric(theta, e):
        if theta == np.pi:
            return np.pi
        if e < 0:
            return None
        if e <= 1:
            tanE_2 = np.sqrt( (1 - e) / (1 + e) ) * np.tan(theta / 2)
            return 2*np.atan(tanE_2)
        else:
            tanhE_2 = np.sqrt( (e - 1) / (e + 1) ) * np.tan(theta / 2)
            return 2*np.atanh(tanhE_2)
    

    @staticmethod
    def eccentric_to_true(E, e):
        if E == np.pi:
            return np.pi
        if e < 0:
            return None
        if e <= 1:
            tanTheta_2 = np.sqrt( (1 + e) / (1 - e)) * np.tan(E / 2)
        else:
            tanTheta_2 = np.sqrt( (e + 1) / (e - 1)) * np.tanh(E / 2)
        return 2*np.atan(tanTheta_2)

    @staticmethod
    def eccentric_to_mean(E, e):
        if e < 0:
            return None
        if e <= 1:
            return E - e*np.sin(E)
        else:
            return e*np.sinh(E) - E
    
    @staticmethod
    def mean_to_eccentric(M, e, *, tol=1e-5, solver="N-R", max_ite = 1000):
        E_0 = 0

        if e <= 0.55:
            E_0 = M
        elif 0.55 < e <= 0.95:
            E_0 = np.cbrt(6*M)
        elif 0.95 < e <= 1:
            E_0 = np.pi
        else:
            E_0 = np.log(2*M / (e+1))
        
        ite = 0
        if solver == "N-R":
            # E_1 = E_0 - (M - E_0 + e*np.sin(E_0)) / (1 + e*np.cos(E_0))
            def func(E):
                if e <= 1:
                    return E - ((M - E + e*np.sin(E)) / (e*np.cos(E) - 1))
                else:
                    return E - ((M + E - e*np.sinh(E)) / (1 - e*np.cosh(E)))
            
        elif solver == "S.S":
            def func(E):
                if e <= 1:
                    return M + e*np.sin(E)
                else:
                    return e*np.sinh(E) - M
            
        else:
            print("Not a valid solver option")
            return None
        

        while True:
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
    
class Kepler:
    """
    Kepler's Equation relating orbital position and orbital period. Inputs and outputs are in terms of seconds
    and/or radians.
    """
    @staticmethod
    def t_to_M(mu, a, delta_t):
        return np.sqrt(mu / (a**3)) * delta_t
        
    @staticmethod
    def M_to_t(mu, a, delta_M):
        return np.sqrt((a**3) / mu) * delta_M