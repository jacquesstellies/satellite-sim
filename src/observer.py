from wheels import WheelModule
from fault import Fault

import my_utils as my_utils
import my_globals

import numpy as np

class WheelExtendedStateObserver():
    t_sample = 0
    index = 0
    gain = np.zeros(2)
    M_inertia = None
    M_inertia_inv = None
    wheel = None
    def __init__(self, config, wheel: WheelModule):
        self.config = config
        self.t_sample = config['controller']['t_sample']
        self.index = 0
        self.gain = config['controller']['gain_obs']
        self.friction_coef = wheel.friction_coef
        self.M_inertia = wheel.M_inertia[2][2]
        self.wheel = wheel

        # if len(self.gain) != 2:
        #     raise ValueError("Gain must be a 2-element vector.")
        
        self.calc_system_matrices()
    
    F = None
    g = None
    A = None
    b = None
    def calc_system_matrices(self):
        # Calculate the system matrices based on the wheel's inertia
        self.M_inertia_inv = 1/self.M_inertia
        self.A = np.array([[-self.friction_coef * self.M_inertia_inv, self.M_inertia_inv],
                           [0, 0]])
        self.b = np.array([[self.M_inertia_inv],
                           [0]])

        self.F = np.eye(2) + self.A * self.t_sample + 0.5 * self.A**2 * self.t_sample**2 + 1/6 * self.A**3 * self.t_sample**3
        self.g = self.t_sample * (np.eye(2) + 0.5 * self.A * self.t_sample + 1/6 * self.A**2 * self.t_sample**2) @ self.b

    e_prev = 0
    def calc_state_rates(self, t, state, u):

        w_est = state[0] # wheel speed
        f_est = state[1] # disturbance
        x_est = state
        y = u[1] # measured output
        y_est = w_est

        e = y - y_est  # estimation error
        
        k_w = self.gain[0]
        k_f = self.gain[1]
        k_fd = self.gain[2] if len(self.gain) > 2 else 0

        # dx_est = (self.A @ x_est).flatten() + self.b.flatten() * u[0] + np.array([k_w * e, k_f * e])

        # dw_est = dx_est[0]
        # df_est = dx_est[1]

        de = e - self.e_prev / self.t_sample
        self.e_prev = e

        dw_est = -(self.friction_coef * self.M_inertia_inv) * w_est + u[0] * self.M_inertia_inv + f_est * self.M_inertia_inv + e*k_w
        df_est = k_f * e + k_fd*de

        T_est = self.M_inertia * dw_est
        if np.abs(T_est) > self.wheel.T_max:
            dw_est = np.sign(dw_est) * self.wheel.T_max / self.M_inertia


        w_est += dw_est * self.t_sample
        f_est += df_est * self.t_sample


        if np.abs(w_est) >= self.wheel.w_max:
            w_est = np.sign(w_est) * self.wheel.w_max

        return  [w_est, f_est, dw_est]

class WheelObserver():
    t_sample = 0
    index = 0
    gain = np.zeros(2)
    M_inertia = None
    M_inertia_inv = None
    wheel = None
    def __init__(self, config, wheel: WheelModule):
        self.config = config
        self.t_sample = config['controller']['t_sample']
        self.index = 0
        self.gain = config['controller']['gain_obs']
        self.friction_coef = wheel.friction_coef
        self.M_inertia = wheel.M_inertia[2][2]
        self.wheel = wheel

        if len(self.gain) != 2:
            raise ValueError("Gain must be a 2-element vector.")
        
        self.M_inertia_inv = 1/self.M_inertia
        
    def calc_state_rates(self, t, state, u):

        w_est = state # wheel speed
        y = u[1] # measured output
        y_est = w_est

        e = y - y_est  # estimation error
        
        k_w = self.gain[0]

        dw_est = -(self.friction_coef * self.M_inertia_inv) * w_est + u[0] * self.M_inertia_inv + e*k_w

        # T_est = self.M_inertia * dw_est
        # if np.abs(T_est) > self.wheel.T_max:
        #     dw_est = np.sign(dw_est) * self.wheel.T_max / self.M_inertia


        w_est += dw_est * self.t_sample

        # if np.abs(w_est) >= self.wheel.w_max:
        #     w_est = np.sign(w_est) * self.wheel.w_max

        return w_est