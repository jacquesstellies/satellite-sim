import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
import math
from state_provider import StateProvider

import my_utils as my_utils
import my_globals

from fault import Fault

class Wheel():
    mass = 0
    M_inertia = np.zeros((3,3))
    M_inertia_inv = np.zeros((3,3))
    M_inertia_fast : float = 0
    M_inertia_inv_fast : float = 0

    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    d = np.zeros(3) # direction vector
    friction_coef = 0.0
    config = None
    index = 0

    _faults : list[Fault] = None 
    w_max = 0
    T_max = 0

    w : float = 0.0
    dw : float = 0.0
    H : float = 0.0
    dH : float = 0.0

    controller_enable = True
    t_sample = None

    _noise_std = 0

    def __init__(self, config, faults, index=0):
        self.config = config
        self.mass = self.config['wheels']['mass']
        self.dimensions['radius'] = self.config['wheels']['radius']
        self.dimensions['height'] = self.config['wheels']['height']
        self.w_max = my_utils.conv_rpm_to_rads_per_sec(self.config['wheels']['max_speed_rpm'])
        self.T_max = self.config['wheels']['max_torque']
        self._faults = faults
        self.index = index
        self.t_sample = self.config['controller']['t_sample']
        self.friction_coef = self.config['wheels']['friction_coef']
        self.calc_M_inertia()

        target_noise_db = 2
        target_noise_watts = 10 ** (target_noise_db / 10)
        self._noise_std = math.sqrt(target_noise_watts)
    
    def calc_M_inertia(self):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)

        self.M_inertia_inv = np.linalg.inv(self.M_inertia)
        self.M_inertia_fast = self.M_inertia[2][2]  # For fast calculations, we only need the z-axis inertia
        self.M_inertia_inv_fast = self.M_inertia_inv[2][2]  # For fast calculations, we only need the z-axis inertia


    def calc_state_rates(self, t : float, state : float, u : float):
        self.w = state
        # Set fault torque limit
        T_limit = self.T_max
    
        # Apply fault if enabled (self updated in fault class)
        for fault in self._faults:
            u *= fault.E[self.index][self.index]
            u += fault.u_a[self.index]

        # Check torque limit exceeded
        if abs(u) >= T_limit:
            u = T_limit*my_utils._sign(u)

            
        # Calculate the new wheel speed derivative
        self.dw = (u - self.friction_coef*self.w)*self.M_inertia_inv_fast
        # self.dw = u*self.M_inertia_inv_fast
        
        if (self.w >= self.w_max and self.dw > 0) or (self.w <= -self.w_max and self.dw < 0):
            self.dw = 0
        
        self.dH = self.dw*self.M_inertia_fast
        self.H = self.w*self.M_inertia_fast
        # return self.calc_sensor_output(np.array([dw,w]))
        # return dw

    def calc_sensor_output(self, state: np.array):
        noise = np.random.normal(0, self._noise_std, size=len(state))
        return state + noise

    
    def calc_state_outputs(self, t, state):
        dw = state[0]
        w = state[1]
        H = w*self.M_inertia_fast
        dH = dw*self.M_inertia_fast
        return np.array([dH, H])

    e_prev = 0
    def calc_state_speed_control(self, t, state, w_ref):
        w = state[0]
        kp = self.config['controller']['kp_wheel_w']
        kd = self.config['controller']['kd_wheel_w']
        ki = self.config['controller']['ki_wheel_w']

        e = w_ref - w
        de = (e - self.e_prev)/self.t_sample
        e_int = e+self.e_prev*self.t_sample
        u = (kp*e + kd*de + ki*e_int)*self.M_inertia_fast

        if abs(u) > self.T_max:
            u = self.T_max*my_utils._sign(u)

        self.e_prev = e
        state = self.calc_state_rates(t, state, u)

        return state, u

class WheelModule():
    wheels = None
    layout = None
    config = None
    D = None
    D_psuedo_inv = None
    _D_psuedo_inv_local = None
    num_wheels = 0
    
    def __init__(self, config,faults=None):
        self.layout = config['wheels']['config']
        self.config = config
        if self.layout == "ortho":
            self.num_wheels = 3
            self.D = np.eye(3)
        elif self.layout == "pyramid":
            self.num_wheels = 4
            self.D = np.array([ [ -1, -1, 1, 1,], [ 1, -1, -1, 1,], [ 1, 1, 1, 1,],])

        elif self.layout ==  "tetra":
            self.num_wheels = 4
            self.D = np.array([[0.9428,-0.4714,-0.4714,0],
                            [0,0.8165,-0.8165,0],
                            [-0.3333,-0.3333,-0.3333,1]])
        elif self.layout == "custom":
            self.num_wheels = config['wheels']['num_wheels']
            self.D = np.array(config['wheels']['D'])
            if self.D.shape[1] != self.num_wheels or self.D.shape[0] != 3:
                raise(Exception(f"invalid D matrix shape {self.D.shape}"))
        else:
            raise(Exception(f"{self.layout} is not a valid wheel layout. \nerror unable to set up wheel layout"))
        
        self.wheels = [Wheel(config,faults,i) for wheel, i in enumerate(range(self.num_wheels))]
        for i, wheel in enumerate(self.wheels):
            wheel.w_max = my_utils.conv_rpm_to_rads_per_sec(config['wheels']['max_speed_rpm'])
            wheel.T_max = config['wheels']['max_torque']
            wheel.d = np.transpose(self.D)[i]

        # self.D_psuedo_inv = np.linalg.pinv(self.D)
        if self.D.shape[1] != 3:
            self.D_psuedo_inv = np.linalg.pinv(self.D)
        else:
            self.D_psuedo_inv = np.linalg.inv(self.D)

        self._D_psuedo_inv_local = self.D_psuedo_inv

        if config['controller']['type'] == "backstepping" and config['controller']['sub_type'] == "Shen":
            self._D_psuedo_inv_local = np.eye(self.num_wheels)
        
        self.w_wheels = np.array(self.num_wheels)
        self.dw_wheels = np.array(self.num_wheels)
        self.H_wheels = np.array(self.num_wheels)
        self.dH_wheels = np.array(self.num_wheels)
        self.H_vec = np.zeros(3)
        self.dH_vec = np.zeros(3)

    w_wheels = None
    dw_wheels = None
    H_wheels = None
    dH_wheels = None
    H_vec = None
    dH_vec = None
    def calc_state_rates(self, t, state, u : np.array):
        # dw = state[:self.num_wheels]
        # w = state[self.num_wheels:]
        if len(u) != self.num_wheels:
            raise Exception(f"invalid control torque length {len(u)}")
            
        for wheel in self.wheels:
            wheel.calc_state_rates(t, state[wheel.index], u[wheel.index])

        self.w_wheels = np.array([wheel.w for wheel in self.wheels])
        self.dw_wheels = np.array([wheel.dw for wheel in self.wheels])
        self.H_wheels = np.array([wheel.H for wheel in self.wheels])
        self.dH_wheels = np.array([wheel.dH for wheel in self.wheels])

        self.H_vec = self.D @ self.H_wheels
        self.dH_vec = self.D @ self.dH_wheels
    
    # calc_state_rates_optmized

    def calc_state_outputs(self, t, state):
        dw_wheels = state[:self.num_wheels]
        w_wheels = state[self.num_wheels:]
        dH_module = np.zeros(3)
        H_module = np.zeros(3)
        for wheel in self.wheels:
            y = wheel.calc_state_outputs(t, [dw_wheels[wheel.index], w_wheels[wheel.index]])
            # dH_wheels = np.zeros(self.num_wheels)
            # H_wheels = np.zeros(self.num_wheels)
            dH_module += y[0]*wheel.d 
            H_module += y[1]*wheel.d
        # return np.concatenate([dH_module, H_module, dH_wheels, H_wheels])
        return np.concatenate([dH_module, H_module])