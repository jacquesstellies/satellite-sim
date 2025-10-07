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

    _fault : Fault = None 
    w_max = 0
    T_max = 0

    controller_enable = True
    t_sample = None

    _noise_std = 0

    def __init__(self, config, fault, index=0):
        self.config = config
        self.mass = self.config['wheels']['mass']
        self.dimensions['radius'] = self.config['wheels']['radius']
        self.dimensions['height'] = self.config['wheels']['height']
        self.w_max = my_utils.conv_rpm_to_rads_per_sec(self.config['wheels']['max_speed_rpm'])
        self.T_max = self.config['wheels']['max_torque']
        self._fault = fault
        self.index = index
        self.t_sample = self.config['controller']['t_sample']
        self.friction_coef = self.config['wheels']['friction_coef']
        print("Initializing wheel with friction coef:", self.friction_coef)
        self.calc_M_inertia()

        # print(f"Wheel {self.index} inertia:\n", self.M_inertia_fast)
        print(f"Wheel {self.index} inertia inv:", self.M_inertia_inv_fast)

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


    def calc_state_rates(self, t, state, u):
        w = state[0]

        # Check wheel speed limit exceeded
        # if abs(w) >= self.w_max:
        #     w = self.w_max*my_utils._sign(w)
            # u = 0
            # if u == 0:
            #     u = 0
            #     dw = 0
            #     return np.array([dw,w])

        # Set fault torque limit
        T_limit = self.T_max
    
        # Check torque limit exceeded
        if abs(u) >= T_limit:
            # print("Wheel torque input saturated at:", u)
            u = T_limit*my_utils._sign(u)

        # Apply fault if enabled (self updated in fault class)
        u = self._fault.E[self.index][self.index] * u
            
        # Calculate the new wheel speed derivative
        dw = (u - self.friction_coef*w)*self.M_inertia_inv_fast

        if (w >= self.w_max and dw > 0) or (w <= -self.w_max and dw < 0):
            dw = 0
        
        # return self.calc_sensor_output(np.array([dw,w]))
        return np.array([dw,w])

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
    
    def __init__(self, config,fault=None):
        self.layout = config['wheels']['config']
        self.config = config
        if self.layout == "ortho":
            self.num_wheels = 3
            self.D = np.eye(3)
        elif self.layout == "pyramid":
            self.num_wheels = 4
            beta = np.radians(30)
            self.D = np.array([cos(beta)*np.array([1.0,0.0,-1.0,0.0]),
                            cos(beta)*np.array([0.0,1.0,0.0,-1.0]),
                            sin(beta)*np.array([1.0,1.0,1.0,1.0])])

        elif self.layout ==  "tetrahedron":
            self.num_wheels = 4
            self.D = np.array([[0.9428,-0.4714,-0.4714,0],
                            [0,0.8165,-0.8165,0],
                            [-0.3333,-0.3333,-0.3333,1]]) 
            # elif len(wheels) == 1:
            #     self.wheels[0].position[0] = self.dimensions[wheel_axes]-wheel_offset
        elif self.layout == "custom":
            self.num_wheels = config['wheels']['num_wheels']
            self.D = np.array(config['wheels']['D'])
            if self.D.shape[1] != self.num_wheels or self.D.shape[0] != 3:
                raise(Exception(f"invalid D matrix shape {self.D.shape}"))
        else:
            raise(Exception(f"{self.layout} is not a valid wheel layout. \nerror unable to set up wheel layout"))
        
        self.wheels = [Wheel(config,fault,i) for wheel, i in enumerate(range(self.num_wheels))]
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


    def calc_state_rates(self, t, state, u : np.array):
        # dw = state[:self.num_wheels]
        # w = state[self.num_wheels:]

        T_c = self._D_psuedo_inv_local@u
        for wheel in self.wheels:
            [dw_wheel, w_wheel] = wheel.calc_state_rates(t, [state[wheel.index],state[wheel.index + self.num_wheels]], T_c[wheel.index])
            state[wheel.index] = dw_wheel
            state[wheel.index + self.num_wheels] = w_wheel
        return state
    
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