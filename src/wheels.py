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

    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    d = np.zeros(3) # direction vector
    friction_coef = 0.0001
    config = None
    index = 0

    _fault : Fault = None 
    w_max = 0
    T_max = 0

    controller_enable = True
    t_sample = None

    def __init__(self, config, fault, index=0):
        self.config = config
        self.mass = self.config['wheels']['mass']
        self.dimensions['radius'] = self.config['wheels']['radius']
        self.dimensions['height'] = self.config['wheels']['height']
        self.w_max = my_utils.conv_rpm_to_rads_per_sec(self.config['wheels']['max_speed_rpm'])
        self.T_max = self.config['wheels']['max_torque']
        self._fault = fault
        self.index = index
        # self.t_sample = self.config['wheels']['t_sample']
        self.calc_M_inertia()
    
    def calc_M_inertia(self):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)

        self.M_inertia_inv = np.linalg.inv(self.M_inertia)
    # def controller_output(self, t, state, u):
    #     kp = 1.0
    #     w_i = state[0]
    #     w_delta = u/self.M_inertia[2][2]
    #     w_result = w_i

    def calc_state_rates(self, t, state, u):
        
        T_c = u
        w = state[0]

        self._fault.update(t)
        # Check wheel speed limit exceeded
        if np.abs(w) >= self.w_max:
            # print(f"Wheel {self.index} speed limit exceeded: {w} > {self.w_max}")
            w = self.w_max*np.sign(w)
            dw = 0
            return [dw, w]
         
        # Set fault torque limit
        T_limit = self.T_max
        # if self._fault.enabled:
        #     T_limit = self._fault.torque_limit_mul * self.T_max
        #     if T_limit > self.T_max:
        #         raise(Exception(f"Fault torque limit {T_limit} exceeds max torque {self.T_max}"))
    
        # Check torque limit exceeded
        if np.abs(T_c) > T_limit:
            T_c = T_limit*np.sign(T_c)

        if self._fault.enabled:
            T_c = self._fault.torque_limit_mul * T_c
            
        # Calculate the new wheel speed derivative
        dw = (T_c - self.friction_coef*w)*self.M_inertia_inv[2][2]
        
        return [dw, w]
    
    def calc_state_outputs(self, t, state):
        dw = state[0]
        w = state[1]
        H = w*self.M_inertia[2][2]
        dH = dw*self.M_inertia[2][2] + self.friction_coef*w
        return [dH, H]

class WheelModule():
    wheels = None
    layout = None
    config = None
    D = None
    D_psuedo_inv = None
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
            my_globals.results_data['wheel_speed_' + str(i)] = []
            my_globals.results_data['wheel_torque_' + str(i)] = []
            wheel.d = np.transpose(self.D)[i]

        self.D_psuedo_inv = np.linalg.pinv(self.D)

    def calc_state_rates(self, t, state, u : np.array):
        dw = state[:self.num_wheels]
        w = state[self.num_wheels:]
        if len(u) == self.D.shape[1]:
           T_c = u
        elif len(u) == 3:
            T_c = self.D_psuedo_inv@u
        else:
            raise(Exception(f"invalid control input shape: {u.shape}"))
        dw_wheels = np.zeros(self.num_wheels)
        w_wheels = np.zeros(self.num_wheels)
        for wheel in self.wheels:
            [dw_wheel, w_wheel] = wheel.calc_state_rates(t, [dw[wheel.index],w[wheel.index]], T_c[wheel.index])
            dw_wheels[wheel.index] = dw_wheel
            w_wheels[wheel.index] = w_wheel
        return np.concatenate([dw_wheels, w_wheels])
    
    def calc_state_outputs(self, t, state):
        dw_wheels = state[:self.num_wheels]
        w_wheels = state[self.num_wheels:]
        dH_module = np.zeros(3)
        H_module = np.zeros(3)
        for wheel in self.wheels:
            y = wheel.calc_state_outputs(t, [w_wheels[wheel.index], dw_wheels[wheel.index]])
            dH_wheels = np.zeros(self.num_wheels)
            H_wheels = np.zeros(self.num_wheels)
            dH_module += y[0]*wheel.d 
            H_module += y[1]*wheel.d
        # return np.concatenate([dH_module, H_module, dH_wheels, H_wheels])
        return np.concatenate([dH_module, H_module])