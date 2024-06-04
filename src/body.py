import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import math
import quaternion
import my_utils as my_utils
import my_globals

class Fault():
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 'x'
    torque_limit = 1
    types = ["catastrophic", "comm_delay", "torque_limit"]
    filter_coeff = 0

    def __init__(self, time, wheel_axis, type, torque_limit):
        self.time = time
        self.wheel_num = my_utils.xyz_axes.index(wheel_axis)
        self.type=type
        self.torque_limit = torque_limit

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
    # angular_v = np.zeros(3)
    speed = 0
    acceleration = 0
    dir_vector = np.zeros(3)
    friction_coef = 0.0001
    

    fault : Fault = None 
    min_speed = 0.1
    max_speed = 10000
    max_torque = 0.016
    def __init__(self, mass, radius, height, dir_init_inertial=None):
        self.mass = mass
        self.dimensions['radius'] = radius
        self.dimensions['height'] = height
        if dir_init_inertial is None:
            self.dir_init_inertial = Rotation.from_quat([0,0,0,1])
        my_globals.results_data['adaptive_model_output'] = []
    
    def calc_M_inertia(self):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)
        self.calc_M_inertia_inv()
    
    # returns angular momentum
    # def calc_angular_momentum(self) -> np.array:
    #     # angular_momentum = self.M_inertia@self.angular_v
    #     # print(f"angular v {self.angular_v}")
    #     if np.isnan(angular_momentum[0]):
    #         raise("Nan error")
    #     return self.speed*self.dir_vector
    
    def update_angular_velocity(self, angular_v: np.array):
        if (self.fault is not None and self.fault.enabled) or self.speed >= self.min_speed:
            self.speed = 0
            self.angular_v = np.zeros(3)
        else:
            self.angular_v = angular_v*self.dir_vector
            self.speed = my_utils.magnitude(angular_v)


    def update_speed(self, speed):
        if np.abs(speed) > self.max_speed:
            self.speed = self.max_speed*np.sign(speed)
        else:
            self.speed = speed
        # self.speed = speed
        
    def update_acceleration(self, acceleration):
        if self.speed < self.max_speed:
            self.acceleration = acceleration
        else:
            self.acceleration = 0 
        

class Integrator:
    prev_val = None
    init_val = 0
    time_step = 0
    def __init__(self, time_step, init_val=0):
        self.init_val = init_val
        self.time_step = time_step
    
    def integrate(self, val):
        if self.prev_val == None:
            self.prev_val = self.init_val
        integrand = self.prev_val + val*self.time_step
        self.prev_val = integrand
        return integrand
    
class Controller:
    T_max = None
    T_min = 0
    time_step = 0.1
    filter_coef = 0
    k=1
    c=1
    fault : Fault = None
    type="standard"
    types = ["standard", "adaptive"]
    adaptive_gain = 0

    def __init__(self, k, c, fault=None, T_max = None, T_min = 0, 
                 filter_coef = 0.5, time_step=0.1, type="standard", adaptive_model_coeff=[0,0,0],
                 angular_v_init=None, quaternion_init=None, adaptive_gain=0):
        self.T_max = T_max
        self.T_min = T_min
        if filter_coef <= 1 or filter_coef >= 0:
            self.filter_coef = filter_coef
        else:
            raise Exception("filter coef must be between 0 and 1")
        self.time_step = time_step
        self.k = k
        if len(c) != 3:
            raise("error c must be a list of length 3")
        self.c = c
        self.fault = fault
        self.type = type
        if type not in self.types:
            raise Exception(f"controller type {type} type is not a valid type")
        if type == "adaptive":
            if angular_v_init is None or quaternion_init is None:
                raise Exception("initialization of angular velocity and quaternion is required for adaptive model")
            self.angular_v_prev = angular_v_init
            self.quaternion_prev = quaternion_init
            self.adaptive_gain = adaptive_gain

    T_output_prev = 0
    def calc_torque_control_output(self, q_curr : np.quaternion,  angular_v : list[float], ref_q : np.quaternion) -> np.array:
        K = np.diag(np.full(3,self.k))
        C = np.diag(self.c)

        angular_v = np.array(angular_v)
        
        q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        T_output = + K@q_error_vec - C@angular_v

        T_output = self.limit_torque(T_output)
        
        T_output = self.low_pass_filter(T_output, self.T_output_prev, self.filter_coef)
        T_output = self.inject_fault(T_output)
        
        if self.type == "adaptive":
            T_output = self.calc_adaptive_control_torque_output(T_output, q_curr)

        return T_output
    
    
    def update_M_inertia_inv_model(self, M_inertia_inv_model):
        self.M_inertia_inv_model = M_inertia_inv_model 

    M_inertia_inv_model = None
    angular_v_prev = None
    q_prev : np.quaternion = None
    
    def calc_adaptive_model_output(self, T_com):
        
        angular_v_result = self.angular_v_prev + self.M_inertia_inv_model@(T_com)*self.time_step
        self.angular_v_prev = angular_v_result
        inertial_v_q = np.quaternion(0, angular_v_result[0], angular_v_result[1], angular_v_result[2])
        # print(self.q_prev)
        # print(inertial_v_q)
        q_result = self.q_prev + 0.5*self.q_prev*inertial_v_q*self.time_step
        self.q_prev = q_result
        return q_result

    theta_prev = 0
    def calc_adaptive_control_torque_output(self, T_com, q_meas : np.quaternion):
        q_model = self.calc_adaptive_model_output(T_com)
        y_model = my_utils.conv_numpy_to_Rotation_obj_q(q_model).as_euler("xyz",degrees=True) 
        my_globals.results_data['adaptive_model_output'].append(my_utils.conv_Rotation_obj_to_euler_int(my_utils.conv_numpy_to_Rotation_obj_q(q_model)))
        
        y_meas = my_utils.conv_numpy_to_Rotation_obj_q(q_meas).as_euler("xyz",degrees=True) 
        error = y_meas - y_model
        print(f"y_meas {y_meas}")
        print(f"y_model {y_model}")
        # print(f"error {error}")
        
        theta = self.theta_prev - error*self.adaptive_gain*y_model*error
        print(f"theta {theta}")
        self.theta_prev = theta
        # theta=[1,1,1]
        return T_com*theta

    def limit_torque(self, M):
        for i in range(3):
            if (self.T_max is not None) and (np.abs(M[i]) >= self.T_max):
                M[i] = np.sign(M[i])*self.T_max
            if np.abs(M[i]) < self.T_min:
                M[i] = 0
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
            M[self.fault.wheel_num] = self.low_pass_filter(M[self.fault.wheel_num], self._torque_prev, self.fault.filter_coeff)
            self._torque_prev = M[self.fault.wheel_num]
        elif self.fault.type == "torque_limit":
            limit = self.fault.torque_limit*self.T_max
            if abs(M[self.fault.wheel_num]) > limit:
                M[self.fault.wheel_num] = limit*np.sign(M[self.fault.wheel_num])
        elif self.fault.type == "update_rate":
            pass
        else:
            raise("invalid fault type")
        return M
    
    integrator_list : list[Integrator] = []
    # curr_integrator = 0
    # def reset_integrator_list(self):
    #     self.curr_integrator = 0

    def integrate(self, value, integrator_number : int):
        if integrator_number not in self.integrator_list:
            self.integrator_list.append(Integrator())
        
        return self.integrator_list[integrator_number].integrate(value)


class WheelModule():
    angular_momentum = np.zeros(3)
    angular_velocity = np.zeros(3)
    wheels = None
    config = ""
    def __init__(self, mass, radius, height, config="standard", max_speed=0, max_torque=0):
        self.wheels = [Wheel(mass, radius, height) for wheel in range(3)]
        for i, wheel in enumerate(self.wheels):
            wheel.max_speed = max_speed
            wheel.max_torque = max_torque
            my_globals.results_data['wheel_speed_' + str(i)] = []
        if config == "standard":
            self.wheels[0].calc_M_inertia()
            self.wheels[1].calc_M_inertia()
            self.wheels[2].calc_M_inertia()

            self.wheels[0].dir_vector = np.array([1,0,0])
            self.wheels[1].dir_vector = np.array([0,1,0])
            self.wheels[2].dir_vector = np.array([0,0,1])
            # elif len(wheels) == 1:
            #     self.wheels[0].position[0] = self.dimensions[wheel_axes]-wheel_offset
        else:
            raise(Exception("error unable to set up wheel configuration"))

    def get_angular_momentum(self):
        return self.angular_momentum
    
    def low_pass_filter(self, value, value_prev, coeff):
        return (coeff)*value_prev + (1 - coeff)*value
    
    H_rate_prev = np.zeros(3)
    def calc_state_rates(self, T_com : np.array, sampling_time):
        H_input = np.zeros(3)
        dH = np.zeros(3)
        H_rate = np.zeros(3)

        for wheel in self.wheels:
            H_input = H_input + wheel.speed*wheel.M_inertia[2][2]*wheel.dir_vector
            
        dH = sampling_time*(T_com)
        H_result = dH + H_input
        # H_rate = self.low_pass_filter(H_result, self.H_rate_prev, 0)
        H_rate = H_result

        for wheel in self.wheels:
            wheel.update_speed((H_result@wheel.dir_vector)/wheel.M_inertia[2][2])
            # acc = ((T_com@wheel.dir_vector)-wheel.friction_coef*wheel.speed)/wheel.M_inertia[2][2]
            # wheel.update_acceleration(acc)
            # H_rate = H_rate + wheel.acceleration*wheel.M_inertia[2][2]*wheel.dir_vector
        # @TODO make torque zero when speed is zero
        return H_rate, H_result
