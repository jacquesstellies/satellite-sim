import numpy as np
from orbit import Orbit
import my_utils

class MagtModule():
    config : dict = None
    orbit : Orbit = None

    B_field = np.zeros(3)
    def __init__(self, config, orbit):
        self.config = config
        self.orbit = orbit
        self.enable = self.config['magt']['enable']
    
    def calc_B_field(self, position, t):
        # Placeholder for magnetic field calculation
        self.B_field = np.array([0.0, 0.0, 0.0])  # Replace with actual model
    
    def calc_torque(self, w_sat, H_sat):
        # m = self.config['magnetorquer']['mag_moment_max'] * (-w_sat / np.linalg.norm(w_sat))
        # T_magt = np.cross(m, B_field)
        T_magt = np.zeros(3)
        T_magt[2] = 10 * 1e-3 * -1*my_utils._sign (w_sat[2])
        return T_magt