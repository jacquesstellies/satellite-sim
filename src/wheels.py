import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
import math

import my_utils as my_utils
import my_globals

from fault import Fault

class Wheel():
    mass = 0
    M_inertia = np.zeros((3,3))
    M_inertia_inv = np.zeros((3,3))

    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    speed = 0
    T = 0
    H = 0
    dir_vector = np.zeros(3)
    friction_coef = 0.0001
    config = None
    index = 0

    _fault : Fault = None 
    max_speed = 0
    max_torque = 0
    t_sample = 0
    def __init__(self, config, fault):
        self.mass = config['wheels']['mass']
        self.dimensions['radius'] = config['wheels']['radius']
        self.dimensions['height'] = config['wheels']['height']
        self.max_speed = my_utils.conv_rpm_to_rads_per_sec(config['wheels']['max_speed_rpm'])
        self.max_torque = config['wheels']['max_torque']
        self.t_sample = config['controller']['time_step']
        self._fault = fault
    
    def calc_M_inertia(self):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)
    
    # returns angular momentum
    # def calc_angular_momentum(self) -> np.array:
    #     # angular_momentum = self.M_inertia@self.angular_v
    #     # print(f"angular v {self.angular_v}")
    #     if np.isnan(angular_momentum[0]):
    #         raise("Nan error")
    #     return self.speed*self.dir_vector
    flag = False
    def calc_state_rates(self, new_speed):
        speed_prev = self.speed

        if np.abs(new_speed) > self.max_speed:
            self.speed = self.max_speed*np.sign(new_speed)
            if not self.flag:
                # print(f"speed limit reached {self.speed} {self.index}")
                self.flag = True
        else:
            self.speed = new_speed

        self.T = self.M_inertia[2][2]*((self.speed - speed_prev)/self.t_sample)
        wheel_torque_limit = self.max_torque

        
        if self._fault.master_enable:
            wheel_torque_limit = self._fault.mul_fault_matrix[self.index][self.index]*self.max_torque
            # if self._fault.type == "torque_limit":
            #     self.T = self._fault.torque_limit*self.max_torque
            # elif self._fault.type == "catastrophic":
            #     self.T = 0
            #     self.speed = 0
            #     self.H = 0
            #     return 0, 0, 0
        

        if np.abs(self.T) > wheel_torque_limit:
            sign = np.sign(self.T)
            self.speed = (((sign)*(wheel_torque_limit)*self.t_sample)/self.M_inertia[2][2])+speed_prev
            # if(wheel_torque_limit==0):
            #     self.speed = 0
            if np.abs(new_speed) > self.max_speed:
                self.speed = self.max_speed*np.sign(new_speed)
            self.T = wheel_torque_limit

        self.speed = my_utils.low_pass_filter(self.speed, speed_prev, 0.8)
        self.H = self.M_inertia[2][2]*self.speed
        return self.speed, self.T, self.H

class WheelModule():
    H = np.zeros(3)
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
        
        self.wheels = [Wheel(config,fault) for wheel in range(self.num_wheels)]
        for i, wheel in enumerate(self.wheels):
            wheel.max_speed = my_utils.conv_rpm_to_rads_per_sec(config['wheels']['max_speed_rpm'])
            wheel.max_torque = config['wheels']['max_torque']
            my_globals.results_data['wheel_speed_' + str(i)] = []
            my_globals.results_data['wheel_torque_' + str(i)] = []
            wheel.dir_vector = np.transpose(self.D)[i]
            wheel.calc_M_inertia()
            wheel.index = i

        self.D_psuedo_inv = np.linalg.pinv(self.D)

    def get_angular_momentum(self):
        return self.angular_momentum
    
    def low_pass_filter(self, value, value_prev, coeff):
        return (coeff)*value_prev + (1 - coeff)*value
    
    H_dot_prev = np.zeros(3)
    
    def calc_state_rates(self, u_c : np.array, sampling_time):
        H_vec_init = np.zeros(3)
        H_vec_result = np.zeros(3)
        H_dot = np.zeros(3)
        if len(u_c) == self.D.shape[1]:
           T_c = u_c
           pass
        elif len(u_c) == 3:
            T_c = self.D_psuedo_inv@u_c
        else:
            raise(Exception(f"invalid control input shape: {u_c.shape}"))

        for wheel in self.wheels:
            H_vec_init = H_vec_init + wheel.H*wheel.dir_vector
            dH = T_c[wheel.index]*sampling_time
            # print('dH ', dH)
            speed, H, T = wheel.calc_state_rates((wheel.H+dH)/wheel.M_inertia[2][2])
            # print('wheel speed ', wheel.speed)
            # wheel.H = wheel.speed*wheel.M_inertia[2][2]
            # print('wheel H ', wheel.H)
            H_vec_result = H_vec_result + wheel.H*wheel.dir_vector
            # print('H_vec_result ', H_vec_result)
            # print('')
        
        H_dot = (H_vec_result - H_vec_init)/sampling_time
        self.H = H_vec_result
        # print("H_dot = ",H_dot)

        # @TODO make torque zero when acc is zero
        return H_dot, H_vec_result
