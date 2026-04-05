from typing import Tuple

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import math
import quaternion
import my_utils
from sgp4.api import Satrec, WGS72
from astropy.coordinates import get_body, solar_system_ephemeris
from astropy.time import Time
from skyfield.api import Timescale, EarthSatellite, load

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

    TOI_I = np.eye(3) # body to orbit frame DCM
    nSI_I = np.zeros(3) # normalized sun vector in inertial frame
    nOB_I = np.zeros(3) # normalized orbit vector in inertial frame

    eclipse = False
    t_sample = 1.0 # seconds

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

        if config['simulation']['verbose']:
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

    next_t_sample: float = 0.0
    # orbital parameters update
    def calc_orbit_state(self, t_runtime):
        if t_runtime >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return

        self.fr = t_runtime/86400

        error, sBT_T , DTsBT_T = self.sgp4_sat.sgp4(self.jd, self.fr) # sBT_T refers to body position in TEME frame, DTsBT_T refers to velocity in TEME frame

        # Use astropy or manual rotation matrix for TEME->ICRF conversion
        # For now, keeping in TEME as exact conversion requires additional ephemeris data
        # error, sBT_T , DTsBT_T = self.sgp4_sat.sgp4(self.jd, 0.0)
        if error != 0:
            raise Exception("SGP4 propagation error")
        self.s_eci = np.array(sBT_T)  # km
        self.v_eci = np.array(DTsBT_T)  # km/s
        self.nOB_I = -self.s_eci/np.linalg.norm(self.s_eci)
        x = self.v_eci/np.linalg.norm(self.v_eci)
        z = self.nOB_I
        y = my_utils.cross_product(z, x)
        self.TOI_I = np.column_stack([x, y, z]) # here the T in TBO_B means transformation and O refers to the orbit frame
        rST_T, _, _ = self.calc_sun_vector_update()
        self.nSI_I = rST_T/np.linalg.norm(rST_T)
        # rSB_T = rST_T - self.s_eci
        # self.nSB_T = rSB_T/np.linalg.norm(rSB_T)

        self.check_sun_eclipse(self.s_eci, rST_T)


    def sun_ecliptic_parameters(self, t: float) -> Tuple[float, float, float]:
        """Compute the mean longitude, mean anomaly, and ecliptic longitude of the Sun.

        References:
            Vallado: 2022, pp. 283-284

        Args:
            t (float): Time since J2000 in Julian centuries (e.g. 'tut1' or 'ttdb')

        Returns:
            tuple: (mean_lon, mean_anomaly, ecliptic_lon)
                mean_lon (float): Mean longitude of the Sun in radians
                mean_anomaly (float): Mean anomaly of the Sun in radians
                ecliptic_lon (float): Ecliptic longitude of the Sun in radians
        """
        mean_lon = np.radians(280.46 + 36000.771285 * t) % my_utils.TWOPI
        mean_anomaly = np.radians(357.528 + 35999.050957 * t) % my_utils.TWOPI
        ecliptic_lon = (
            np.radians(
                np.degrees(mean_lon)
                + 1.915 * np.sin(mean_anomaly)
                + 0.02 * np.sin(2 * mean_anomaly)
            )
            % my_utils.TWOPI
        )

        return float(mean_lon), float(mean_anomaly), ecliptic_lon

    def obliquity_ecliptic(self, t: float) -> float:
        """Compute the obliquity of the ecliptic.

        Args:
            t (float): Time since J2000 in Julian centuries (e.g. 'tut1' or 'ttdb')

        Returns:
            float: Obliquity of the ecliptic in radians
        """
        return float(np.radians(np.degrees(my_utils.OBLIQUITYEARTH) - 0.0130042 * t))
        
    def get_sun_position(self, jd: float) -> Tuple[np.ndarray, float, float]:
        """Calculates the geocentric equatorial position vector of the Sun.

        This is the low precision formula and is valid for years from 1950 to 2050. The
        accuaracy of apparent coordinates is about 0.01 degrees.  notice many of the
        calculations are performed in degrees, and are not changed until later. This is due
        to the fact that the almanac uses degrees exclusively in their formulations.

        Sergey K (2022) has noted that improved results are found assuming the oputput is in
        a precessing frame (TEME) and converting to ICRF.

        References:
            Vallado: 2022, p. 285, Algorithm 29

        Args:
            jd (float): Julian date (days from 4713 BC)

        Returns:
            tuple: (rsun, rtasc, decl)
                rsun (np.ndarray): Inertial sun position vector in km
                rtasc (float): Right ascension of the sun in radians
                decl (float): Declination of the sun in radians
        """
        # Julian centuries from J2000
        tut1 = (jd - my_utils.J2000) / my_utils.CENT2DAY

        # Mean anomaly and ecliptic longitude of the sun in radians
        _, meananomaly, eclplong = self.sun_ecliptic_parameters(tut1)

        # Obliquity of the ecliptic in radians
        obliquity = self.obliquity_ecliptic(tut1)

        # Magnitude of the Sun vector in AU
        magr = (
            1.000140612
            - 0.016708617 * np.cos(meananomaly)
            - 0.000139589 * np.cos(2 * meananomaly)
        )

        # Sun position vector in geocentric equatorial coordinates
        rsun = np.array(
            [
                magr * np.cos(eclplong),
                magr * np.cos(obliquity) * np.sin(eclplong),
                magr * np.sin(obliquity) * np.sin(eclplong),
            ]
        )

        # Right ascension in radians
        rtasc = np.arctan(np.cos(obliquity) * np.tan(eclplong))

        # Ensure right ascension is in the same quadrant as ecliptic longitude
        if eclplong < 0:
            eclplong += my_utils.TWOPI
        if abs(eclplong - rtasc) > np.pi / 2:
            rtasc += 0.5 * np.pi * round((eclplong - rtasc) / (0.5 * np.pi))

        # Declination (radians)
        decl = np.arcsin(np.sin(obliquity) * np.sin(eclplong))

        return rsun * my_utils.AU2KM, rtasc, decl

    def calc_sun_vector_update(self):
        rST_T = self.get_sun_position(self.jd + self.fr)
        # nST_T = rST_T/np.linalg.norm(rST_T)
        return rST_T 

        # with solar_system_ephemeris.set('de440'):
            # r_GS_G = get_body('sun', Time(self.jd + t/86400, format='jd'), 'itrs').cartesian.xyz.to_value()
            # r_GS_G = get_body('sun', Time("2026-01-01 00:00:00", format='iso', scale='utc'), None, 'jpl').cartesian.xyz.to_value()
        # return r_GS_G

    r_GS_G = np.zeros(3)

    def check_sun_eclipse(self, r_sat: np.array, r_sun: np.array) -> bool:
        """Check if the satellite is in Earth's shadow.

        References:
            Curtis, H.D.: Orbit Mechanics for Engineering Students, 2014, Algorithm 12.3

        Args:
            r_sat (array_like): Satellite position vector in km
            r_sun (array_like): Sun position vector in km

        Returns:
            bool: Whether satellite is in attracting body's shadow
        """
        # Calculate angles
        sun_sat_angle = my_utils.angle_vec(r_sun, r_sat)
        angle1 = np.arccos(my_utils.RE / np.linalg.norm(r_sat))
        angle2 = np.arccos(my_utils.RE / np.linalg.norm(r_sun))

        # Check line of sight (no LOS = eclipse)
        if (angle1 + angle2) <= sun_sat_angle:
            return True

        return False
        
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