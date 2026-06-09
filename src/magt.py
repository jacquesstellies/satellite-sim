import numpy as np
from orbit import Orbit
import my_utils

EARTH_MAG_DIPOLE = 7.96e15

class MagtModule():
    config : dict = None
    orbit : Orbit = None
    T = np.zeros(3)
    mode = None

    B_field = np.zeros(3)
    def __init__(self, config, orbit):
        self.config = config
        self.orbit = orbit
        self.enable = self.config['magt']['enable']
        if self.config['magt']['enable']:
            self.mode = self.config['magt']['mode']
        
    
    def calc_B_field(self, t):
        # Placeholder for magnetic field calculation
        # self.B_field = np.array([0.0, 0.0, 0.0])  # Replace with actual model
        alpha = self.orbit.arg_perigee + self.orbit.true_anomaly
        w_0 = 1/self.orbit.period
        eta_m = w_0 * t % (2*np.pi)  # magnetic dipole rotation
        xi_m = self.orbit.inclination - np.radians(17) # mag_field_inclination
        self.B_field = EARTH_MAG_DIPOLE / (self.orbit.radius)**3 * np.array([np.cos(alpha-eta_m)*np.sin(xi_m),
                                                                        np.cos(xi_m),
                                                                        -2*np.sin(alpha-eta_m)*np.sin(xi_m)])
    prev = 0
    def calc_torque(self, q_err_vec, w_sat, H_sat, t):
        if not self.enable:
            return
        ## Calculate magnetic torque
        # self.calc_B_field(t)  # t should be passed appropriately
        # m = np.ones(3)*self.config['magt']['mag_moment_max']
        # T_magt = np.array([[0, self.B_field[2], -self.B_field[1]], [-self.B_field[2], 0, self.B_field[0]], [self.B_field[1], -self.B_field[0], 0]]) @ m
        # T_magt[0], T_magt[1] = 0, 0
        match self.mode:
            case "z-axis_simple":
                self.T = np.zeros(3)
                k = 1
                self.T[2] = -1 * my_utils._sign(w_sat[2]) * k
                self.T[2] = my_utils.low_pass_filter(self.T[2], self.prev, 0.5)
            case "momentum_dumping":
                m = np.ones(3)*self.config['magt']['mag_moment_max']
                k = 1
                self.T = -1 * my_utils.sat_vec(k*H_sat, 0.01)
            case "momentum_dump_w_z_axis_simple":
                km = 1
                kz = 0.001

                self.T = my_utils.sat_vec(km*H_sat, 0.01)
                self.T[2] = -1 * my_utils._sign(q_err_vec[2]) * kz
        # T_magt[2] = T_magt[2] * my_utils._sign(q_err[2]) * k

        ## Testing Stuff
        # T_magt = np.cross(m, self.B_field)
        # T_magt = my_utils.mat_multiply_3x3_vec(np.array([[0, self.B_field[2], -self.B_field[1]], [-self.B_field[2], 0, self.B_field[0]], [self.B_field[1], -self.B_field[0], 0]]), m)
        # T_magt = np.zeros(3)
        # T_magt[2] = T_magt[2] * -1 * my_utils._sign(w_sat[2])
        # for i in range(3):
        #     T_magt[i] = T_magt[i] * -1 * my_utils._sign(w_sat[i])

        # T_magt = np.zeros(3)
        # T_magt[2] = 10 * 1e-3 * -1*my_utils._sign (w_sat[2])