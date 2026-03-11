import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import math
import quaternion
import my_utils as my_utils
from sgp4.api import Satrec, WGS72
from astropy.coordinates import get_body, solar_system_ephemeris
from astropy.time import Time

MU = 398600 # km^3/s^2

class Orbit():
    altitude = 0 # km
    inclination = 0 # rad
    radius = 0 # km

    v_eci = np.zeros(3) # velocity vector in TEME coordinates
    s_eci = np.zeros(3) # position vector in TEME coordinates
    RAAN = 0
    v_norm = 0 # km/s
    period = 0 # s
    arg_perigee = 0 # rad
    true_anomaly = 0 # rad
    jd=np.float64(2461041.5000000) # 2026-01-01 00:00:00 UTC
    fr=np.float64(0.0)

    sgp4_sat : Satrec = None

    nBS_B = np.zeros(3) # normalized sun vector
    nBG_B = np.zeros(3) # normalized earth vector
    TBO_B = np.zeros((3,3)) # body to orbit frame DCM

    def __init__(self, config):
        altitude = config['orbit']['altitude_km']
        if altitude > 100:
            self.altitude = altitude
        else:
            raise(Exception("invalid altitude given: altitude must be greater than 100km"))
        # place into circular sun synchronous orbit
        RAAN_rate = 2*np.pi/(365.26*24*3600)
        self.RAAN = 15 * my_utils.DEG_TO_RAD
        earth_radius = 6378 # km
        
        J = 1.08263e-3 # Oblateness factor
        e = 0
        m_earth = 5.972e24 # kg
        G = 6.67430e-11
        self.radius = altitude + earth_radius
        self.h = math.sqrt(MU*self.radius)
        # print("Orbit radius set to: ", self.radius)
        
        self.inclination = arccos(np.power(RAAN_rate*(1-np.power(e,2)),2)*np.power(self.radius,7/2)/((-3/2)*
                                    np.sqrt(MU)*J*np.power(earth_radius,2)))
        
        self.period = 2*np.pi*self.radius/np.sqrt(MU/self.radius)

        self.arg_perigee = np.pi/2

        # self.v = np.sqrt(G*m_earth/self.radius)
        # self.v_norm = np.linalg.norm(self.v)
        self.mean_anomaly = 0
        self.true_anomaly = 0

        self.sgp4_sat = Satrec()

        print("Initializing SGP4 satellite with the following parameters:")
        print("  Altitude:", self.altitude, "km")
        print("  Radius:", self.radius, "km")
        print("  JD:", self.jd)
        print("  RAAN:", self.RAAN)
        print("  Arg Perigee:", self.arg_perigee)
        print("  Inclination:", self.inclination)
        print("  Mean Anomaly:", self.mean_anomaly)
        print("  Period:", self.period/60, "minutes")
        self.sgp4_sat.sgp4init(
            WGS72,
            'i',
            1,
            self.jd,
            0.0,
            0.0,
            0.0,
            e,
            self.arg_perigee, # argument of perigee (rad)
            self.inclination, # inclination (rad)
            self.mean_anomaly, # mean anomaly (rad)
            2*np.pi/(self.period/60), # mean motion (rad/min)
            self.RAAN # rad
        )
        with solar_system_ephemeris.set('de440'):
            r_GS_G = get_body('sun', Time(self.jd, format='jd'), None, 'jpl').cartesian.xyz.to_value()
            # r_GS_G = get_body('sun', Time("2026-01-01 00:00:00", format='iso', scale='utc'), None, 'jpl').cartesian.xyz.to_value()
        self.n_BS_B = r_GS_G/np.linalg.norm(r_GS_G)
        # self.s = 

    # orbital parameters update
    def calc_orbit_state(self, t_runtime):
        # self.mean_anomaly = 2*np.pi*t_runtime/self.period
        # self.true_anomaly = self.mean_anomaly  # circular orbit assumption

        # s_orf = self.h**2/MU*(1/(1 + self.e*cos(self.true_anomaly)))*np.array(cos(self.true_anomaly),
        #                                 sin(self.true_anomaly), 0)
        
        # v_orf = MU/self.h*np.array(-sin(self.true_anomaly), self.e + cos(self.true_anomaly), 0)

        # # Update RAAN
        # self.RAAN = self.RAAN_rate*t % (2*np.pi)
        
        # R_orf_to_ecef = np.array([[-sin(self.RAAN)*cos(self.inclination)*sin(self.arg_perigee) + cos(self.RAAN)*cos(self.arg_perigee),
        # cos(self.RAAN)],
        # []
        # []])

        # R_3_Omega = np.array([[cos(self.arg_perigee), sin(self.arg_perigee), 0],
        #                     [-sin(self.arg_perigee), cos(self.arg_perigee), 0],
        #                     [0, 0, 1]])
        # R_1_i = np.array([[1, 0, 0],
        #                 [0, cos(self.inclination), sin(self.inclination)],
        #                 [0, -sin(self.inclination), cos(self.inclination)]])
        
        # R_3_omega = np.array([[cos(self.RAAN), sin(self.RAAN), 0],
        #                     [-sin(self.RAAN), cos(self.RAAN), 0],
        #                     [0, 0, 1]])

        # self.s_ecef = R_3_omega @ R_1_i @ R_3_Omega @ s_orf


        # self.jd += t_runtime/86400
        # jd_sans_fr = math.floor(self.jd)
        self.fr = t_runtime/86400
        jd_total = self.jd + self.fr
        # print(self.fr, self.jd, jd_total, Time(jd_total, format='jd').datetime.strftime('%Y-%m-%d %H:%M:%S.%f'), sep=" | ")
        # exit()
        error, sBT_T , DTsBT_T = self.sgp4_sat.sgp4(self.jd, self.fr) # sBT_T refers to body position in TEME frame, DTsBT_T refers to velocity in TEME frame
        # error, sBT_T , DTsBT_T = self.sgp4_sat.sgp4(self.jd, 0.0)
        if error != 0:
            raise Exception("SGP4 propagation error")
        self.s_eci = np.array(sBT_T)  # km
        self.v_eci = np.array(DTsBT_T)  # km/s
        self.nTB_B = -self.s_eci/np.linalg.norm(self.s_eci)
        x = self.v_eci/np.linalg.norm(self.v_eci)
        z = self.nTB_B
        y = my_utils.cross_product(x, z)
        self.TBO_B = np.array([x, y, z]).T # here the T in TBO_B means tensor and O refers to the orbit frame

        # r_GS_G = self.calc_sun_vector_update(t_runtime)
        # r_BS_B = r_GS_G - r_GB_G
        # self.n_BS_B = r_BS_B/np.linalg.norm(r_BS_B)

    
    def calc_sun_vector_update(self, t):
        with solar_system_ephemeris.set('de440'):
            r_GS_G = get_body('sun', Time(self.jd + t/86400, format='jd'), 'itrs').cartesian.xyz.to_value()
            # r_GS_G = get_body('sun', Time("2026-01-01 00:00:00", format='iso', scale='utc'), None, 'jpl').cartesian.xyz.to_value()
        return r_GS_G

    r_GS_G = np.zeros(3)
    def check_sun_eclipse(self):

        # return True
        R_earth = 6378e3
        r_mag = np.linalg.norm(self.s_eci)
        r_hat = self.s_eci/r_mag

        theta_earth = np.arcsin(R_earth/r_mag)
        theta_sun = np.arcsin(696340e3/np.linalg.norm(self.r_GS_G))

        cos_phi = r_hat @ self.r_GS_G_hat
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))

        if phi < theta_earth - theta_sun:
            return True  # in eclipse
        else:
            return False  # not in eclipse
        
class Disturbances():
    orbit : Orbit = None
    def __init__(self, orbit):
        self.orbit = orbit
        
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
            # print(f"face {i}  {(sigma_t*my_utils.cross_product(face.r_cop_to_com,va_b)) + 
            #                 ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
            #                 my_utils.cross_product(face.r_cop_to_com,-1*face.norm_vec)}")
            T_aero_tmp = (density*np.power(np.linalg.norm(va_b,2),2)*A_p)*(
                        (sigma_t*my_utils.cross_product(face.r_cop_to_com,va_b)) + 
                            ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
                            my_utils.cross_product(face.r_cop_to_com,-1*face.norm_vec)
                        )
            T_aero += T_aero_tmp
            # print(T_aero_tmp)
        # print(my_utils.cross_product(face.r_cop_to_com,va_b))
        return T_aero
    
    def calc_grav_torque(self, satellite, q):
        rotation_obj = my_utils.conv_numpy_to_Rotation_obj_q(q)
        dcm = rotation_obj.as_matrix()
        u_e = dcm@np.array([0,0,1])
        
        T_grav = 3*self.orbit.mu/pow(self.orbit.radius,3)*my_utils.cross_product(u_e,satellite.M_inertia@u_e)
        # print(my_utils.cross_product(u_e,satellite.M_inertia@u_e))
        # print(satellite.M_inertia@u_e)
        return T_grav

    def calc_dist_torque_Shen(self, t):
        return np.array([-1, 1, -1])*-0.005*np.sin(t)
    
    def calc_dist_torque_Nadafi(self, t):
        return np.array([0.1 + 9*np.sin(0.5*t), 0.1 + 7.5*np.sin(0.8*t), 0])*7.5e-3