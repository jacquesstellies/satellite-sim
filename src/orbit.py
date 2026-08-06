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
from logger import Logger

MU = 398600 # km^3/s^2

class Orbit():
    logger : Logger = None
    config : dict = None
    DIsBI_I = np.zeros(3) # velocity vector in TEME coordinates
    sBI_I = np.zeros(3) # position vector in TEME coordinates

    latitude = 0.0
    longditude = 0.0
    Dlat = 0.0
    Dlong = 0.0


    altitude = 0 # km
    inclination = 0 # rad
    radius = 0 # km
    RAAN = 0
    v_norm = 0 # km/s
    period = 0 # s
    arg_perigee = 0 # rad
    # true_anomaly = 0 # rad
    jd=np.float64(2461041.5000000) # 2026-01-01 00:00:00 UTC
    fr=np.float64(0.0)

    propagator : Satrec = None

    T_OI = np.eye(3) # inertial to orbit frame DCM

    # rSB_I = np.zeros(3) # sun vector in inertial frame
    # rIB_I = np.zeros(3) # orbit vector in inertial frame
    nSB_I = np.zeros(3) # normalized sun vector in inertial frame
    nIB_I = np.zeros(3) # normalized orbit vector in inertial frame

    eclipse = False
    t_sample = 0.1 # seconds

    mean_anomaly = 0.0
    enable = True

    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        if self.config['simulation']['tuning'] and self.config['satellite']['mode'] == "ref_pointing":
            self.enable = False
        altitude = self.config['orbit']['altitude_km']
        if 't_sample' in self.config['orbit']:
            self.t_sample = self.config['orbit']['t_sample']
        if 'jd_start' in self.config['orbit']:
            self.jd = np.float64(self.config['orbit']['jd_start'])
        
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

        self.arg_perigee = 0

        if 'mean_anomaly_deg' in self.config['orbit']:
            self.mean_anomaly = self.config['orbit']['mean_anomaly_deg'] * np.pi / 180

        self.propagator = Satrec()

        self.logger.log(f"Initializing SGP4 satellite with the following parameters:", to_results_file=True, to_console=False)
        self.logger.log(f"  Altitude: {self.altitude} km", to_results_file=True, to_console=False)
        self.logger.log(f"  Radius: {self.radius} km", to_results_file=True, to_console=False)
        self.logger.log(f"  JD: {self.jd}", to_results_file=True, to_console=False)
        self.logger.log(f"  RAAN: {self.RAAN}", to_results_file=True, to_console=False)
        self.logger.log(f"  Arg Perigee: {self.arg_perigee}", to_results_file=True, to_console=False)
        self.logger.log(f"  Inclination: {self.inclination}", to_results_file=True, to_console=False)
        self.logger.log(f"  Mean Anomaly: {self.mean_anomaly}", to_results_file=True, to_console=False)
        self.logger.log(f"  Period: {self.period/60} minutes", to_results_file=True, to_console=False)

        self.propagator.sgp4init(
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


    next_t_sample: float = 0.0
    # orbital parameters update
    def calc_orbit_state(self, t_runtime):
        if not self.enable:
            return
        if t_runtime >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return

        self.fr = t_runtime/86400

        error, sBT_T , DTsBT_T = self.propagator.sgp4(self.jd, self.fr) # sBT_T refers to body position in TEME frame, DTsBT_T refers to velocity in TEME frame

        # Use astropy or manual rotation matrix for TEME->ICRF conversion
        # For now, keeping in TEME as exact conversion requires additional ephemeris data
        # error, sBT_T , DTsBT_T = self.propagator.sgp4(self.jd, 0.0)
        if error != 0:
            raise Exception("SGP4 propagation error")
        self.sBI_I = np.array(sBT_T)  # km
        self.DIsBI_I = np.array(DTsBT_T)  # km/s
        self.nIB_I = -self.sBI_I/np.linalg.norm(self.sBI_I)
        # Orthonormal LVLH/orbit (O) frame: z = nadir (-r_hat), y = -orbit-normal
        # (-h/|h|), x = y x z (~ velocity direction). Building y from h = r x v keeps
        # T_OI an exact rotation even when r.v != 0 under SGP4/J2 (v_hat is not exactly
        # perpendicular to r_hat), so dcm_to_quat(T_OI) is faithful.
        h = my_utils.cross_product_M31M31(self.sBI_I, self.DIsBI_I)
        z = self.nIB_I
        y = -h/np.linalg.norm(h)
        x = my_utils.cross_product_M31M31(y, z)
        # Rows of the passive DCM T_OI are the O-frame axes in I coords (v_O = T_OI v_I).
        self.T_OI = np.row_stack([x, y, z])
        rSI_I, _, _ = self.calc_sun_vector_update()

        rSB_I = rSI_I - self.sBI_I
        self.nSB_I = rSB_I/np.linalg.norm(rSB_I)

        self.eclipse = self.check_sun_eclipse(self.sBI_I, rSI_I)

        lat_prev = self.latitude
        long_prev = self.longditude
        self.latitude, self.longitude = self.get_lat_lon(self.sBI_I, self.jd + self.fr)
        self.Dlat = (self.latitude - lat_prev)/self.t_sample
        self.Dlong = (self.longditude - long_prev)/self.t_sample



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
    
    def get_lat_lon(self, sBI_I: np.ndarray, jd: float) -> Tuple[float, float]:
        """Convert ECI position to geodetic latitude and longitude.
        
        Args:
            sBI_I (np.ndarray): Satellite position in ECI frame (km)
            jd (float): Julian date
            
        Returns:
            tuple: (latitude, longitude) in radians
        """
        from astropy.coordinates import GCRS, CartesianRepresentation, EarthLocation
        from astropy.time import Time
        
        # Create GCRS coordinate from ECI position
        coord = GCRS(CartesianRepresentation(sBI_I[0]*1000, sBI_I[1]*1000, sBI_I[2]*1000, unit='m'),
                     obstime=Time(jd, format='jd'))
        
        # Convert to EarthLocation (geodetic coordinates)
        loc = EarthLocation.from_geocentric(*coord.cartesian.xyz)
        
        return float(loc.lat.rad), float(loc.lon.rad)
        
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
        rSI_I = self.get_sun_position(self.jd + self.fr)
        return rSI_I 

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
    T : np.array = np.zeros(3)
    enable = False
    model = ""
    models = ['realistic', 'Nadafi', 'Zarourati']
    t_sample = 0.1

    def __init__(self, orbit, config):
        self.orbit = orbit
        self.config = config
        self.enable = config['disturbances']['enable']
        if self.enable:
            self.model = config['disturbances']['model']
            if self.model not in self.models:
                raise Exception('Disturbace model not in list')
        self.sigma_n = 0.8
        self.sigma_t = 0.8
        
        self.dipole_vec = ([0,0,1])
        self.t_sample = orbit.t_sample

        W_s = 1361 # W/m^2 at 1 AU
        c = 299792458 # m/s
        self.P_solar = W_s/c # N/m^2 at 1 AU
        self.rho_s = config['disturbances'].get('rho_s', 0.6) # specular reflectivity (aluminised Mylar)

    sigma_n = 0.8
    sigma_t = 0.8
    init = True
    def calc_aero_torque(self, satellite, T_BI):

        w_earth = 7.272e-5
        if self.orbit.Dlong < 0:
            ratio = self.orbit.Dlat/self.orbit.Dlong
            aero_angle = np.pi - np.arctan(ratio)
        elif self.orbit.Dlong == 0:
            aero_angle = np.pi/2
        elif self.orbit.Dlong > 0:
            ratio = self.orbit.Dlat/self.orbit.Dlong
            aero_angle = -1*np.arctan(ratio)

        T_BO = T_BI@np.linalg.inv(self.orbit.T_OI)

        va_b = T_BO@[-np.linalg.norm(self.orbit.DIsBI_I*1e3) + w_earth*(self.orbit.radius*1e3)*cos(self.orbit.latitude)*cos(aero_angle),
                -w_earth*self.orbit.radius*1e3*cos(self.orbit.latitude)*sin(aero_angle),
                0]
        
        density_0 = 1.585e-12
        if self.orbit.eclipse:
            density_0 = density_0*0.5

        density = density_0*np.exp(-(self.orbit.altitude-450)/60.828e3)
        T_aero = np.zeros(3)

        va_b_norm = np.linalg.norm(va_b)
        va_b_unit = va_b/va_b_norm

        # All calcs further assume sbc frame for vectors
        for i, face in enumerate(satellite.faces):
            cos_alpha = (-1)*(face.norm_vec)@va_b_unit
            cos_alpha_h = np.heaviside(cos_alpha, 1)
            A_p = cos_alpha_h*cos_alpha*face.area
            T_aero_tmp = (density*np.linalg.norm(va_b)**2*A_p)*(
                        (self.sigma_t*my_utils.cross_product_M31M31(face.r_com_to_cop,va_b_unit)) + 
                            ((self.sigma_n*0.05) + (2-self.sigma_n-self.sigma_t)*cos_alpha)*
                            my_utils.cross_product_M31M31(face.r_com_to_cop,-1*face.norm_vec)
                        )
            T_aero += T_aero_tmp

        return T_aero
    
    def calc_grav_torque(self, satellite, T_BI):
        # Gravity-gradient torque uses the nadir (local vertical) unit vector in the
        # body frame. nadir in inertial is nIB_I (= -r_hat); rotate it into body.
        u_e = T_BI @ self.orbit.nIB_I

        T_grav = 3*MU/pow(self.orbit.radius,3)*my_utils.cross_product_M31M31(u_e,satellite.M_inertia@u_e)
        return T_grav

    def calc_solar_radiation_pressure_torque(self, satellite, T_BI):
        if self.orbit.eclipse:
            return np.zeros(3)

        s_B = -1*(T_BI @ self.orbit.nSB_I) # photon travel direction (sun -> satellite) in body frame
        T_solar = np.zeros(3)

        # Wie solar thrust model (Jordaan 2016, eq. 3.3.29-3.3.32):
        # F_n = P*A*(1+rho_s)*cos^2(xi), F_t = P*A*(1-rho_s)*cos(xi)*sin(xi)
        # combined into vector form F = P*A*cos(xi)*((1-rho_s)*s - 2*rho_s*cos(xi)*n)
        for i, face in enumerate(satellite.faces):
            cos_xi = (-1)*(face.norm_vec)@s_B
            cos_xi_h = np.heaviside(cos_xi, 0)
            F_solar = (self.P_solar*face.area*cos_xi_h*cos_xi)*(
                        (1-self.rho_s)*s_B - 2*self.rho_s*cos_xi*face.norm_vec)
            T_solar += my_utils.cross_product_M31M31(face.r_com_to_cop, F_solar)

        return T_solar


    # def calc_mag_torque(self, satellite, dcm):
    #     m = 
    #     B = self.orbit.calc_magnetic_field(satellite, dcm)
    #     T_mag = np.cross(m, B)
    #     return T_mag

    def calc_torque_realistic(self, satellite, T_BI, t):
        T_aero = self.calc_aero_torque(satellite, T_BI)
        T_solar = self.calc_solar_radiation_pressure_torque(satellite, T_BI)
        T_grav = self.calc_grav_torque(satellite, T_BI)
        T_dist = T_aero + T_solar + T_grav
        return T_dist

    t_sample_next = 0.0
    def calc_torque(self, satellite, T_BI, t):
        if t >= self.t_sample_next:
            self.t_sample_next += self.t_sample
        else:
            return self.T
        if self.init:
            self.init = False
            return np.zeros(3)
        
        if self.enable:
            if self.model == "Nadafi":
                self.T = self.calc_dist_torque_Nadafi(t)
            elif self.model == "Zarourati":
                self.T = self.calc_dist_torque_Zarourati(t, satellite.w_BI_B)
            elif self.model == "realistic":
                self.T = self.calc_torque_realistic(satellite, T_BI, t)
                # self.T = self.calc_torque_realistic(satellite, T_BI, t)
            else:
                raise Exception("disturbance model not specified")
        else:
            self.T = np.zeros(3)
        return self.T
        
    def calc_dist_torque_Shen(self, t):
        return np.array([-1, 1, -1])*-0.005*np.sin(t)
    
    def calc_dist_torque_Nadafi(self, t):
        return np.array([0.1 + 9*np.sin(0.5*t), 0.1 + 7.5*np.sin(0.8*t), 0])*7.5e-3
    
    def calc_dist_torque_Zarourati(self, t, w_sat):
        # Paper Eq. (38): d = 1e-5 * [...], bounded by d_bar = 1.1e-4 N.m. (The previous 7.5e-3
        # scale was copied from the Nadafi model and made the disturbance ~750x the paper's.)
        n = 0.0011
        return np.array(
                    [-3 + 4*cos(n*t) - cos(n*t) + 2*w_sat[0]*sin(n*t),
                    4 + 3*sin(n*t) - 2*cos(n*t) + w_sat[1]*cos(n*t),
                    -3 + 4*sin(n*t) - 3*sin(n*t) - 2*w_sat[2]*cos(n*t)]
                        )*1e-5