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
        self.t_sample = config['observer']['t_sample']
        self.index = wheel.index
        self.gain = config['observer']['gain']
        self.friction_coef = wheel.friction_coef
        self.M_inertia_inv = wheel.M_inertia_inv_fast
        self.M_inertia = wheel.M_inertia_fast
        self.wheel = wheel

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

        # self.F = np.eye(2) + self.A * self.t_sample + 0.5 * self.A**2 * self.t_sample**2 + 1/6 * self.A**3 * self.t_sample**3
        # self.g = self.t_sample * (np.eye(2) + 0.5 * self.A * self.t_sample + 1/6 * self.A**2 * self.t_sample**2) @ self.b

    e_prev = 0
    t_prev = 0
    def calc_state_estimates(self, t, state, u):

        w_est = state[0] # wheel speed
        f_est = state[1] # disturbance
        y = u[1] # measured output
        y_est = w_est

        e = y - y_est  # estimation error

        # print(f"observer {self.index} y = {y}, y_est = {y_est}, e = {e}, u = {u}, t = {t}")
        k_w = self.gain[0]
        k_f = self.gain[1]
        k_fd = self.gain[2] if len(self.gain) > 2 else 0
        k_wd = self.gain[3] if len(self.gain) > 3 else 0
        k_wi = self.gain[4] if len(self.gain) > 4 else 0
        k_fi = self.gain[5] if len(self.gain) > 5 else 0

        # dx_est = (self.A @ x_est).flatten() + self.b.flatten() * u[0] + np.array([k_w * e, k_f * e])

        # dw_est = dx_est[0]
        # df_est = dx_est[1]

        if t - self.t_prev == 0:
            de = 0
        else:
            de = e - self.e_prev / (t - self.t_prev)
        e_int = e + self.e_prev * (t - self.t_prev)
        self.e_prev = e

        if abs(u[0]) > self.wheel.T_max:
            u[0] = self.wheel.T_max*my_utils._sign(u[0])

        dw_est = (-self.friction_coef * w_est + u[0]) * self.M_inertia_inv + f_est * self.M_inertia_inv + e*k_w + de*k_wd + e_int*k_wi
        df_est = k_f * e + k_fd*de + k_fi*e_int

        if (w_est >= self.wheel.w_max and dw_est > 0) or (w_est <= -self.wheel.w_max and dw_est < 0):
            dw_est = 0
            df_est = 0

        w_est += dw_est * (t - self.t_prev)
        f_est += df_est * (t - self.t_prev)
        self.t_prev = t
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
        
    def calc_state_estimates(self, t, state, u):

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


class ObserverModule():

    wheel_module: WheelModule = None
    config : dict = None
    results_data : dict = None
    fault : Fault = None
    t_sample : float = 0
    wheel_extended_state_observers : list[WheelExtendedStateObserver] = []
    E_mul : np.array = None # multiplicative actuator effectiveness matrix

    w_wheels_est : np.array = None # wheel speed estimate
    f_wheels_est : np.array = None # disturbance torque estimate
    dw_wheels_est : np.array = None # wheel acceleration estimate


    def __init__(self, config : dict, results_data : dict, wheel_module: WheelModule, fault: Fault=None):
        self.config = config
        self.results_data = results_data
        self.fault = fault
        self.wheel_module = wheel_module
        self.t_sample : float = config['observer']['t_sample']
        self.wheel_extended_state_observers = []
        self.enable = config['observer']['enable']

        for i, wheel in enumerate(wheel_module.wheels):
            eso = WheelExtendedStateObserver(config, wheel)
            self.wheel_extended_state_observers.append(eso)

        self.w_wheels_est = np.zeros(wheel_module.num_wheels)
        self.f_wheels_est = np.zeros(wheel_module.num_wheels)
        self.dw_wheels_est = np.zeros(wheel_module.num_wheels)
        self.E_mul = np.eye(self.wheel_module.num_wheels)
        
    next_t_sample = 0
    dE = 0
    def calc_state_estimates(self, t : float, state : list[float], u_wheels : list[float]):
        if t >= self.next_t_sample:
            if self.config['simulation']['test_mode_en'] is True:
                print("Observer t:", t)
            self.next_t_sample += self.t_sample
        else:
            return self.E_mul
        
        for i, wheel_extended_state_observer in enumerate(self.wheel_extended_state_observers):
            [self.w_wheels_est[i], self.f_wheels_est[i], self.dw_wheels_est[i]] \
                = wheel_extended_state_observer.calc_state_estimates(t, [self.w_wheels_est[i], self.f_wheels_est[i]], [u_wheels[i], state[i]])

            # Only update E if there was a control input
            if u_wheels[i] != 0:
                E_mul_temp = (u_wheels[i] + self.f_wheels_est[i])/u_wheels[i]
                E_mul_temp_prev = self.E_mul[i][i]
                dE_prev = self.dE
                self.dE = (E_mul_temp - E_mul_temp_prev)/self.t_sample
                ddE = (self.dE - dE_prev)/self.t_sample
                if np.abs(ddE) < 10 and E_mul_temp < 1 and E_mul_temp > 0:
                    # self.E_mul[i][i] = my_utils.low_pass_filter(E_mul_temp, E_mul_temp_prev, 0.9)
                    self.E_mul[i][i] = E_mul_temp

        return self.E_mul