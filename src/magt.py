import numpy as np
from orbit import Orbit
import my_utils

class MagtModule():
    config : dict = None
    orbit : Orbit = None
    T = np.zeros(3)
    m = np.zeros(3)
    B_B = np.zeros(3)
    mode = None

    def __init__(self, config, orbit):
        self.config = config
        self.orbit = orbit
        magt_cfg = self.config['magt']
        self.enable = magt_cfg['enable']
        self.m_max = magt_cfg.get('mag_moment_max', 0.0)
        self.t_sample = magt_cfg.get('t_sample', 0.0)
        physical_cfg = magt_cfg.get('physical', {})
        self.physical = bool(physical_cfg.get('enable', False)) if isinstance(physical_cfg, dict) else False
        if self.enable:
            self.mode = magt_cfg['mode']
        self.T = np.zeros(3)
        self.m = np.zeros(3)
        self.B_B = np.zeros(3)
        self.next_t_sample = 0.0
        self.prev = 0

    def _physical_torque(self, T_des, B_B):
        """Map a desired torque onto the achievable magnetorquer set τ = m × B.

        The unconstrained moment that realises the component of T_des perpendicular
        to B is m = (B × T_des) / |B|². Each body axis is then saturated at
        mag_moment_max [A.m²] and the physical torque is recomputed.
        """
        B2 = float(B_B @ B_B)
        if B2 < 1e-24:
            return np.zeros(3), np.zeros(3)
        m = my_utils.cross_product_M31M31(B_B, T_des) / B2
        m = my_utils.sat_vec(np.array(m, dtype=float, copy=True), self.m_max)
        T = my_utils.cross_product_M31M31(m, B_B)
        return T, m

    def calc_torque(self, q_err_vec, w_sat, H_sat, t, T_BI=None):
        if not self.enable:
            self.T = np.zeros(3)
            self.m = np.zeros(3)
            return
        if self.physical and self.t_sample > 0.0 and t < self.next_t_sample:
            return
        if self.physical and self.t_sample > 0.0:
            self.next_t_sample = t + self.t_sample

        ## Calculate magnetic torque command (desired, possibly unphysical)
        match self.mode:
            case "z-axis_simple":
                self.T = np.zeros(3)
                k = 1
                self.T[2] = -1 * my_utils._sign(w_sat[2]) * k
                self.T[2] = my_utils.low_pass_filter(self.T[2], self.prev, 0.5)
            case "momentum_dump_xyz":
                k = 1.0
                self.T = -1 * my_utils.sat_vec(k*H_sat, 0.1)
                self.T = my_utils.low_pass_filter(self.T, self.prev, 0.2)
            case "momentum_dump_xy_axis":
                k = 1.0
                self.T = -1 * my_utils.sat_vec(k*H_sat, 0.1)
                self.T = my_utils.low_pass_filter(self.T, self.prev, 0.2)
                self.T[2] = 0
            case "momentum_dump_w_z_axis_simple":
                km = 1
                kz = 0.001

                self.T = my_utils.sat_vec(km*H_sat, 0.01)
                self.T[2] = -1 * my_utils._sign(q_err_vec[2]) * kz

        if self.physical:
            if T_BI is None:
                raise ValueError("physical magnetorquer torque requires T_BI")
            self.B_B = T_BI @ self.orbit.B_I
            self.T, self.m = self._physical_torque(self.T, self.B_B)
        else:
            self.m = np.zeros(3)
