import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import math
import quaternion
import my_utils as my_utils

class Orbit():
    altitude = 0
    inclination = 0
    v = 0
    radius = 0
    latitude = 0
    mu = 398600e3

    def __init__(self, altitude):
        if altitude > 100000:
            self.altitude = altitude
        else:
            raise(Exception("invalid altitude given: altitude must be greater than 100km"))
        # place into circular sun synchronous orbit
        RAAN_rate = 2*np.pi/(365.26*24*3600)
        earth_radius = 6378e3
        
        J = 1.08263e-3 # Oblateness factor
        e = 0
        m_earth = 5.972e24
        G = 6.67430e-11
        self.radius = altitude + earth_radius
        
        self.inclination = arccos(np.power(RAAN_rate*(1-np.power(e,2)),2)*np.power(self.radius,7/2)/((-3/2)*
                                    np.sqrt(self.mu)*J*np.power(earth_radius,2)))
        
        self.v = np.sqrt(G*m_earth/self.radius)
        self.latitude = 0

    
    def calc_orbit_state():

        return 
        
class Disturbances():
    orbit : Orbit = None
    def __init__(self):
        self.orbit = Orbit(altitude=500000)
        
    def calc_aero_torque(self, satellite, q):
        aero_angle = np.pi - np.arctan(10)
        w_earth = 7.272e-5

        rotation_obj = my_utils.conv_numpy_to_Rotation_obj_q(q)
        dcm = rotation_obj.as_matrix()
        va_b = dcm@[-self.orbit.v + w_earth*self.orbit.radius*cos(self.orbit.latitude*cos(aero_angle)),
                -w_earth*self.orbit.radius*cos(self.orbit.latitude)*sin(aero_angle),
                0]
        density_0 = 1.585e-12
        density = density_0*np.exp(-(self.orbit.altitude-450e3)/60.828e3)
        sigma_n = 0.8
        sigma_t = 0.8
        T_aero = np.zeros(3)
        # All calcs further assume sbc frame for vectors
        for i, face in enumerate(satellite.faces):
            cos_alpha = (-1)*(face.norm_vec)@va_b

            cos_alpha_h = np.heaviside(cos_alpha, 1)
            # print(f"face {i} cos_alpha {cos_alpha} cos_alpha_h {cos_alpha_h}")
            # print(cos_alpha)
            A_p = cos_alpha_h*cos_alpha*face.area
            # print(f"face {i}  {density*np.power(np.linalg.norm(va_b,2),2)*A_p}")
            # print(f"face {i}  {(sigma_t*np.cross(face.r_cop_to_com,va_b)) + 
            #                 ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
            #                 np.cross(face.r_cop_to_com,-1*face.norm_vec)}")
            T_aero_tmp = (density*np.power(np.linalg.norm(va_b,2),2)*A_p)*(
                        (sigma_t*np.cross(face.r_cop_to_com,va_b)) + 
                            ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
                            np.cross(face.r_cop_to_com,-1*face.norm_vec)
                        )
            T_aero += T_aero_tmp
            # print(T_aero_tmp)
        # print(np.cross(face.r_cop_to_com,va_b))
        return T_aero
    
    def calc_grav_torque(self, satellite, q):
        rotation_obj = my_utils.conv_numpy_to_Rotation_obj_q(q)
        dcm = rotation_obj.as_matrix()
        u_e = dcm@np.array([0,0,1])
        
        T_grav = 3*self.orbit.mu/pow(self.orbit.radius,3)*np.cross(u_e,satellite.M_inertia@u_e)
        # print(np.cross(u_e,satellite.M_inertia@u_e))
        # print(satellite.M_inertia@u_e)
        return T_grav

    def calc_dist_torque_Shen(self, t):
        return np.array([-1, 1, -1])*-0.005*np.sin(t)