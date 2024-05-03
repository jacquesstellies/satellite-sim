import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import math
import quaternion
import my_utils as my_utils


class Fault():
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 0

    def __init__(self, time, wheel_num, type):
        self.time = time
        self.wheel_num = wheel_num
        self.type=type

class Body:
    dir_init_inertial = Rotation.from_quat([0,0,0,1])

    mass = 0
    dimensions = {'x':0, 'y':0, 'z':0} # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))
    
    def calc_M_inertia_inv(self):
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)

class Wheel(Body):
    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    angular_v = np.zeros(3)
    speed = 0
    shaft_axis_mask = None

    fault : Fault = None 
    speed_min = 0.1
    def __init__(self, mass, radius, height, dir_init_inertial=None):
        self.mass = mass
        self.dimensions['radius'] = radius
        self.dimensions['height'] = height
        if dir_init_inertial is None:
            self.dir_init_inertial = Rotation.from_quat([0,0,0,1])
    
    def calc_M_inertia(self, direction):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)
        self.dir_init_inertial = Rotation.from_euler('xyz', direction, degrees=True) 
        self.M_inertia = my_utils.rotate_M_inertia(self.M_inertia, self.dir_init_inertial) # @TBD look at floating point error here
        self.M_inertia *= np.eye(1)
        self.calc_M_inertia_inv()
    
    # returns angular momentum
    def calc_angular_momentum(self) -> np.array:
        angular_momentum = self.M_inertia@self.angular_v
        # print(f"angular v {self.angular_v}")
        if np.isnan(angular_momentum[0]):
            raise("Nan error")
        return angular_momentum*self.shaft_axis_mask
    
    def update_angular_velocity(self, angular_v: np.array):
        if (self.fault is not None and self.fault.enabled) or self.speed >= self.speed_min:
            self.speed = 0
            self.angular_v = np.zeros(3)
        else:
            self.angular_v = angular_v*self.shaft_axis_mask
            self.speed = my_utils.magnitude(angular_v)


    def update_speed(self, speed):
        self.speed = speed
        self.angular_v = speed        

class Controller:
    type = "torque"
    types = ["torque", "reaction_wheel"]
    M_max = None
    M_min = 0
    time_step = np.longdouble(0.1)
    filter_coef = 0
    k=1
    c=1
    fault : Fault = None
    def __init__(self, k, c, fault=None, M_max = None, M_min = 0, type = "torque", filter_coef = 0.5, time_step=0.1):
        self.type = type
        self.M_max = M_max
        self.M_min = M_min
        if filter_coef <= 1 or filter_coef >= 0:
            self.filter_coef = filter_coef
        else:
            raise Exception("filter coef must be between 0 and 1")
        self.time_step = np.longdouble(time_step)
        self.k = k
        if len(c) != 3:
            raise("error c must be a list of length 3")
        self.c = c
        self.fault = fault

    M_output_prev = 0
    def calc_torque_control_output(self, curr_q : np.quaternion,  angular_v : list[float], ref_q : np.quaternion) -> np.array:
        K = np.diag(np.full(3,self.k))
        C = np.diag(self.c)

        angular_v = np.array(angular_v)
        
        q_error = curr_q.inverse() * my_utils.conv_scipy_to_numpy_q(ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        M_output = + K@q_error_vec - C@angular_v

        M_output = self.limit_torque(M_output)
        
        M_output = self.low_pass_filter(M_output, self.M_output_prev, self.filter_coef)

        M_output = self.inject_fault(M_output)
        return M_output
    
    def limit_torque(self, M):
        for i in range(3):
            if (self.M_max is not None) and (np.abs(M[i]) >= self.M_max):
                M[i] = np.sign(M[i])*self.M_max
            # if np.abs(M[i]) < self.M_min:
            #     M[i] = 0
        return M

    def low_pass_filter(self, value, value_prev, coeff):
        return (coeff)*value_prev + (1 - coeff)*value

    def calc_wheel_control_output(self, torque, angular_momentum) -> list[float]:
        ref_angular_momentum = torque*self.control_time_step + angular_momentum
        return ref_angular_momentum

    _torque_prev = 0
    def inject_fault(self, M):
        if self.fault.wheel_num > 2:
            raise("invaled fault injection wheel_num")
        if self.fault.enabled is False:
            return M
        
        if self.fault.type == "catastrophic":
            M[self.fault.wheel_num] = 0
        elif self.fault.type == "comm_delay":
            self._torque_prev = M[self.fault.wheel_num]
            M[self.fault.wheel_num] = self.low_pass_filter(M[self.fault.wheel_num], self._torque_prev, 0.99)
            limit = 0.005*self.M_max
            if abs(M[self.fault.wheel_num]) > limit:
                M[self.fault.wheel_num] == limit*np.sign(M[self.fault.wheel_num])
        else:
            raise("invalid fault type")
        return M
        

class WheelModule():
    angular_momentum = np.zeros(3)
    angular_velocity = np.zeros(3)
    wheels = None
    config = ""
    def __init__(self, mass, radius, height, config="standard"):
        self.wheels = [Wheel(mass, radius, height) for wheel in range(3)]
        if config == "standard":
            if len(self.wheels) == 3:
                self.wheels[0].calc_M_inertia([0,90,0])
                self.wheels[1].calc_M_inertia([90,0,0])
                self.wheels[2].calc_M_inertia([0,0,0])

                self.wheels[0].shaft_axis_mask = [1,0,0]
                self.wheels[1].shaft_axis_mask = [0,1,0]
                self.wheels[2].shaft_axis_mask = [0,0,1]
            # elif len(wheels) == 1:
            #     self.wheels[0].position[0] = self.dimensions[wheel_axes]-wheel_offset
            else:
                raise(Exception("error unable to set up wheel configuration"))

    def get_angular_momentum(self):
        return self.angular_momentum

    def update_angular_momentum(self, torque : np.array, time_step : float):
        wheels_H_delta_com = torque * time_step
        self.angular_momentum = wheels_H_delta_com + self.angular_momentum

        for wheel in self.wheels:
            wheel.update_angular_velocity(wheel.M_inertia_inv@(self.angular_momentum*wheel.shaft_axis_mask))
        
        return self.angular_momentum