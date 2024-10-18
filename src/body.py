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
    wheel_num = 0
    torque_limit = 1
    types = ["catastrophic", "comm_delay", "torque_limit"]
    filter_coeff = 0

    def __init__(self, config):
        self.time = config['fault']['time']
        self.wheel_num = config['fault']['wheel_num']
        self.type=config['fault']['type']
        self.torque_limit = config['fault']['torque_limit']
        self.master_enable = config['fault']['master_enable']
        self.filter_coeff = config['fault']['filter_coef']


class Body:

    mass = 0
    dimensions = {'x':0, 'y':0, 'z':0} # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))
    
    def calc_M_inertia_inv(self):
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)

class Wheel(Body):
    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    speed = 0
    T = 0
    H = 0
    dir_vector = np.zeros(3)
    friction_coef = 0.0001
    config = None
    index = 0

    fault : Fault = None 
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
        self.fault = fault
    
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
    flag = False
    flag_2 = False
    def calc_state_rates(self, new_speed):
        speed_prev = self.speed

        if np.abs(new_speed) > self.max_speed:
            self.speed = self.max_speed*np.sign(new_speed)
            if not self.flag:
                print(f"speed limit reached {self.speed} {self.index}")
                self.flag = True
        else:
            self.speed = new_speed

        self.T = self.M_inertia[2][2]*((self.speed - speed_prev)/self.t_sample)
        wheel_torque_limit = self.max_torque

        
        if self.fault.master_enable and self.index == self.fault.wheel_num:
            if self.flag == False:
                print("catastrophic fault")
                self.flag = True
            if self.fault.type == "torque_limit":
                self.T = self.fault.torque_limit*self.max_torque
            elif self.fault.type == "catastrophic":
                self.T = 0
                self.speed = 0
                self.H = 0
                return 0, 0, 0
        

        if np.abs(self.T) > wheel_torque_limit:
            # if not self.flag:
            #     print("torque limit reached")
            #     self.flag = True
            sign = np.sign(self.T)
            self.speed = (((sign)*(self.max_torque)*self.t_sample)/self.M_inertia[2][2])+speed_prev
            if np.abs(new_speed) > self.max_speed:
                self.speed = self.max_speed*np.sign(new_speed)

        self.speed = my_utils.low_pass_filter(self.speed, speed_prev, 0.8)

        # if not self.flag:
            
        #     print(f"wheel {self.index} speed {self.speed} torque {self.T}")
        self.H = self.M_inertia[2][2]*self.speed
        return self.speed, self.T, self.H
        
        

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
    enable = True
    T_max = None
    T_min = 0
    time_step = 0.1
    filter_coef = 0
    k=1
    c=1
    fault : Fault = None
    type="q_feedback"
    types = ["q_feedback", "backstepping", "adaptive"]
    adaptive_gain = 0
    config = None

    def __init__(self, fault=None,
                 angular_v_init=None, quaternion_init=None, config=None):
        self.enable = config['controller']['enable']
        self.T_max = config['controller']['T_max']
        self.T_min = config['controller']['T_min']
        filter_coef = config['controller']['filter_coef']
        if filter_coef <= 1 or filter_coef >= 0:
            self.filter_coef = filter_coef
        else:
            raise Exception("filter coef must be between 0 and 1")
        self.time_step = config['controller']['time_step']
        self.k = config['controller']['k']
        self.c = config['controller']['c']
        if len(self.c) != 3:
            raise("error c must be a list of length 3")
        self.fault = fault
        self.type = config['controller']['type']
        if self.type not in self.types:
            raise Exception(f"controller type {type} type is not a valid type")
        if self.type == "adaptive":
            if angular_v_init is None or quaternion_init is None:
                raise Exception("initialization of angular velocity and quaternion is required for adaptive model")
            self.angular_v_prev = angular_v_init
            self.quaternion_prev = quaternion_init
            self.adaptive_gain = config['controller']['adaptive_gain']
            my_globals.results_data['control_adaptive_model_output'] = []
            for i, axis in enumerate(my_utils.xyz_axes):
                my_globals.results_data[f'control_theta_{axis}'] = []
        print("controller type ", self.type)
        if self.type == "adaptive":
            print("adaptive gain ", self.adaptive_gain)
        my_globals.results_data['control_time'] = []
        my_globals.results_data['control_time'].append(0)
        self.config = config['controller']

    T_output_prev = 0

    def calc_q_feedback_control_torque(self, q_curr, angular_v, ref_q, M_inertia, wheels_H):
        K = np.diag(np.full(3,self.k))
        C = np.diag(self.c)
        angular_v = np.array(angular_v)
        
        q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        Hnet = M_inertia@(angular_v) + wheels_H

        return + K@q_error_vec - C@angular_v + np.cross(angular_v, Hnet)
    
    sub_types = ["linear", "arctan"]
    def calc_backstepping_control_torque(self, q_curr: np.quaternion, angular_v, ref_q, satellite):
        I1 = satellite.M_inertia[0][0]
        I2 = satellite.M_inertia[1][1]
        I3 = satellite.M_inertia[2][2]
        p1 = (I2 - I3)/I1
        p2 = (I3 - I1)/I2
        p3 = (I1 - I2)/I3
        alpha = 0.75
        beta = 8
        sub_type = self.config["sub_type"]
        if sub_type not in self.sub_types:
            raise Exception(f"sub_type {sub_type} is not a valid sub_type")
        if self.config["sub_type"] == "linear":
            s = 1
            g = 10
            e = angular_v - s*quaternion.as_vector_part(q_curr)
            u1 = -1/2*q_curr.x-p1*angular_v[1]*angular_v[2] - s*0.5*(q_curr.w*angular_v[0]+q_curr.y*angular_v[2]-q_curr.z*angular_v[1]) \
                - g*(e[0])
            u2 = -1/2*q_curr.y-p2*angular_v[0]*angular_v[2] - s*0.5*(q_curr.w*angular_v[1]+q_curr.z*angular_v[0]-q_curr.x*angular_v[2]) \
                - g*(e[1])
            u3 = -1/2*q_curr.z-p3*angular_v[0]*angular_v[1] - s*0.5*(q_curr.w*angular_v[2]+q_curr.x*angular_v[1]-q_curr.y*angular_v[0]) \
                - g*(e[2])
        # elif self.config["sub_type"] == "arctan":
        #     a  = [1, 1, 1]
        #     b  = [1, 1, 1]
        #     c  = [1, 1, 1]
        #     s = 1
        #     g = 10

        #     phi = alpha*np.arctan(beta*quaternion.as_vector_part(q_curr))
        #     phi_dot = alpha*beta/(1+np.power(beta*quaternion.as_vector_part(q_curr),2))
        #     e = angular_v - s*phi
        #     u1 = -1/2*q_curr.x-p1*angular_v[1]*angular_v[2] - s*0.5*phi_dot*(q_curr.w*angular_v[0]+q_curr.y*angular_v[2]-q_curr.z*angular_v[1]) \
        #         - g*(e[0])
        #     u2 = -1/2*q_curr.y-p2*angular_v[0]*angular_v[2] - s*0.5*phi_dot*(q_curr.w*angular_v[1]+q_curr.z*angular_v[0]-q_curr.x*angular_v[2]) \
        #         - g*(e[1])
        #     u3 = -1/2*q_curr.z-p3*angular_v[0]*angular_v[1] - s*0.5*phi_dot*(q_curr.w*angular_v[2]+q_curr.x*angular_v[1]-q_curr.y*angular_v[0]) \
        #         - g*(e[2])
            
        u = np.array([u1, u2, u3])
        return u

    def calc_torque_control_output(self, q_curr : np.quaternion,  angular_v : list[float], ref_q : np.quaternion, satellite, wheels_H, t) -> np.array:

        my_globals.results_data['control_time'].append(t)
        if self.type == "q_feedback":
            T_output = self.calc_q_feedback_control_torque(q_curr, angular_v, ref_q, satellite.M_inertia, wheels_H)
            # T_output = self.inject_fault(T_output)
            if self.type == "adaptive":
                T_output = self.calc_adaptive_control_torque_output(T_output, q_curr, ref_q)

        elif self.type == "backstepping":    
            T_output = self.calc_backstepping_control_torque(q_curr, angular_v, ref_q, satellite)

        T_output = self.limit_torque(T_output)
        
        T_output = self.low_pass_filter(T_output, self.T_output_prev, self.filter_coef)
        

        return T_output
    
    
    def update_M_inertia_inv_model(self, M_inertia_inv_model):
        self.M_inertia_inv_model = M_inertia_inv_model 

    M_inertia_inv_model = None
    angular_v_prev = None
    q_prev : np.quaternion = None
    
    def calc_adaptive_model_output(self, ref_q):
        T_model = self.calc_q_feedback_control_torque(self.q_prev, self.angular_v_prev, ref_q)
        angular_v_result = self.angular_v_prev + self.M_inertia_inv_model@(T_model)*self.time_step
        self.angular_v_prev = angular_v_result
        inertial_v_q = np.quaternion(0, angular_v_result[0], angular_v_result[1], angular_v_result[2])
        q_result = self.q_prev + 0.5*self.q_prev*inertial_v_q*self.time_step
        self.q_prev = q_result
        return q_result

    theta_prev = 0
    def calc_adaptive_control_torque_output(self, T_com, q_meas : np.quaternion, q_ref : np.quaternion):
        q_model = self.calc_adaptive_model_output(q_ref)
        my_globals.results_data['control_adaptive_model_output'].append(my_utils.conv_Rotation_obj_to_alpha(my_utils.conv_numpy_to_Rotation_obj_q(q_model)))
        q_error = q_meas.inverse()*q_model
        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        q_model_vec = np.array([q_model.x, q_model.y, q_model.z])
        
        theta = self.theta_prev - q_error_vec*self.adaptive_gain*q_model_vec*self.time_step
        for i, axis in enumerate(my_utils.xyz_axes):
            my_globals.results_data[f'control_theta_{axis}'].append(theta[i])
        self.theta_prev = theta
        return T_com + T_com*theta

    def limit_torque(self, M):
        # print(self.T_max)
        for i in range(3):
            if (self.T_max is not None) and (np.abs(M[i]) >= self.T_max):
                # print('limiting torque')
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
    

    def integrate(self, value, integrator_number : int):
        if integrator_number not in self.integrator_list:
            self.integrator_list.append(Integrator())
        
        return self.integrator_list[integrator_number].integrate(value)


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
            self.D = 0.5774*np.array([[1,-1,-1,1],
                            [1,1,-1,-1],
                            [1,1,1,1]])

        elif self.layout ==  "tetrahedron":
            self.num_wheels = 4
            self.D = np.array([[0.9428,-0.4714,-0.4714,0],
                            [0,0.8165,-0.8165,0],
                            [-0.3333,-0.3333,-0.3333,1]]) 
            # elif len(wheels) == 1:
            #     self.wheels[0].position[0] = self.dimensions[wheel_axes]-wheel_offset
        else:
            raise(Exception(f"{self.layout} is not a valid wheel layout. \nerror unable to set up wheel layout"))
        
        self.wheels = [Wheel(config,fault) for wheel in range(self.num_wheels)]
        for i, wheel in enumerate(self.wheels):
            wheel.max_speed = my_utils.conv_rpm_to_rads_per_sec(config['wheels']['max_speed_rpm'])
            wheel.max_torque = config['wheels']['max_torque']
            my_globals.results_data['wheel_speed_' + str(i)] = []
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
        T_c = self.D_psuedo_inv@u_c

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

class Orbit():
    altitude = 0
    inclination = 0
    v = 0
    radius = 0
    latitude = 0
    mu = 398600e3

    def __init__(self, altitude):
        if altitude > 100000:
            self.altitude = altitude
        else:
            raise(Exception("invalid altitude given: altitude must be greater than 100km"))
        print(f"altitude {altitude}m")
        # place into circular sun synchronous orbit
        RAAN_rate = 2*np.pi/(365.26*24*3600)
        earth_radius = 6378e3
        
        J = 1.08263e-3 # Oblateness factor
        e = 0
        m_earth = 5.972e24
        G = 6.67430e-11
        self.radius = altitude + earth_radius
        
        self.inclination = arccos(np.power(RAAN_rate*(1-np.power(e,2)),2)*np.power(self.radius,7/2)/((-3/2)*
                                    np.sqrt(self.mu)*J*np.power(earth_radius,2)))
        
        print("inclination ", np.degrees(self.inclination), "deg")

        self.v = np.sqrt(G*m_earth/self.radius)
        self.latitude = 0

    
    def calc_orbit_state():

        return 
        

class Disturbances():
    orbit : Orbit = None
    def __init__(self):
        self.orbit = Orbit(altitude=500000)
        
    def calc_aero_torque(self, satellite, q):
        aero_angle = np.pi - np.arctan(10)
        angular_v_earth = 7.272e-5

        rotation_obj = my_utils.conv_numpy_to_Rotation_obj_q(q)
        dcm = rotation_obj.as_matrix()
        va_b = dcm@[-self.orbit.v + angular_v_earth*self.orbit.radius*cos(self.orbit.latitude*cos(aero_angle)),
                -angular_v_earth*self.orbit.radius*cos(self.orbit.latitude)*sin(aero_angle),
                0]
        density_0 = 1.585e-12
        density = density_0*np.exp(-(self.orbit.altitude-450e3)/60.828e3)
        sigma_n = 0.8
        sigma_t = 0.8
        T_aero = np.zeros(3)
        # All calcs further assume sbc frame for vectors
        for i, face in enumerate(satellite.faces):
            cos_alpha = (-1)*(face.norm_vec)@va_b

            cos_alpha_h = np.heaviside(cos_alpha, 1)
            # print(f"face {i} cos_alpha {cos_alpha} cos_alpha_h {cos_alpha_h}")
            # print(cos_alpha)
            A_p = cos_alpha_h*cos_alpha*face.area
            # print(f"face {i}  {density*np.power(np.linalg.norm(va_b,2),2)*A_p}")
            # print(f"face {i}  {(sigma_t*np.cross(face.r_cop_to_com,va_b)) + 
            #                 ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
            #                 np.cross(face.r_cop_to_com,-1*face.norm_vec)}")
            T_aero_tmp = (density*np.power(np.linalg.norm(va_b,2),2)*A_p)*(
                        (sigma_t*np.cross(face.r_cop_to_com,va_b)) + 
                            ((sigma_n*0.05) + (2-sigma_n-sigma_t)*cos_alpha)*
                            np.cross(face.r_cop_to_com,-1*face.norm_vec)
                        )
            T_aero += T_aero_tmp
            # print(T_aero_tmp)
        # print(np.cross(face.r_cop_to_com,va_b))
        return T_aero
    
    def calc_grav_torque(self, satellite, q):
        rotation_obj = my_utils.conv_numpy_to_Rotation_obj_q(q)
        dcm = rotation_obj.as_matrix()
        u_e = dcm@np.array([0,0,1])
        
        T_grav = 3*self.orbit.mu/pow(self.orbit.radius,3)*np.cross(u_e,satellite.M_inertia@u_e)
        # print(np.cross(u_e,satellite.M_inertia@u_e))
        # print(satellite.M_inertia@u_e)
        return T_grav
