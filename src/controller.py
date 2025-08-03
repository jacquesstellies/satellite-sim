import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
import quaternion
import my_utils as my_utils
import my_globals
from fault import Fault

class Controller:
    enable = True
    T_max = None
    T_min = 0
    t_sample = 0.1
    filter_coef = 0
    k=1
    c=1
    _fault : Fault = None
    type="q_feedback"
    types = ["q_feedback", "backstepping", "adaptive"]
    adaptive_gain = 0
    config = None

    def __init__(self, fault=None,
                 w_init=None, q_sat_init=None, config=None):
        self.enable = config['controller']['enable']
        self.T_max = config['controller']['T_max']
        self.T_min = config['controller']['T_min']
        filter_coef = config['controller']['filter_coef']
        if filter_coef <= 1 or filter_coef >= 0:
            self.filter_coef = filter_coef
        else:
            raise Exception("filter coef must be between 0 and 1")
        self.t_sample = config['controller']['t_sample']
        self.k = config['controller']['k']
        self.c = config['controller']['c']
        if len(self.c) != 3:
            raise("error c must be a list of length 3")
        self._fault = fault
        self.type = config['controller']['type']
        if self.type not in self.types:
            raise Exception(f"controller type {type} type is not a valid type")
        if self.type == "adaptive":
            if w_init is None or quaternion_init is None:
                raise Exception("initialization of angular velocity and quaternion is required for adaptive model")
            self.w_prev = w_init
            self.quaternion_prev = quaternion_init
            self.adaptive_gain = config['controller']['adaptive_gain']
            my_globals.results_data['control_adaptive_model_output'] = []
            for i, axis in enumerate(my_utils.xyz_axes):
                my_globals.results_data[f'control_theta_{axis}'] = []
        print("controller type: ", self.type, " sub_type: ", config["controller"]["sub_type"])
        if self.type == "adaptive":
            print("adaptive gain ", self.adaptive_gain)
        my_globals.results_data['control_time'] = []
        my_globals.results_data['control_time'].append(0)
        self.h = 0.1
        self.config = config

    T_output_prev = 0

    def calc_q_feedback_control_torque(self, q_curr, w, ref_q, M_inertia, wheels_H):
        K = np.diag(np.full(3,self.k))
        C = np.diag(self.c)
        w = np.array(w)
        
        q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        Hnet = M_inertia@(w) + wheels_H

        # return + K@q_error_vec - C@w + np.cross(w, Hnet)

        return + self.config['controller']['kj']*M_inertia@q_error_vec - self.config['controller']['kd']*M_inertia@w
    
    h = 0
    u_prev = np.zeros(4)
    sub_types = ["linear", "arctan", "Shen"]
    def calc_backstepping_control_torque(self, q_curr: np.quaternion, w, ref_q, satellite):
        I1 = satellite.M_inertia[0][0]
        I2 = satellite.M_inertia[1][1]
        I3 = satellite.M_inertia[2][2]
        p1 = (I2 - I3)/I1
        p2 = (I3 - I1)/I2
        p3 = (I1 - I2)/I3
        alpha = 0.75
        beta = 8
        sub_type = self.config["controller"]["sub_type"]
        u = np.zeros(3)
        if sub_type not in self.sub_types:
            raise Exception(f"sub_type {sub_type} is not a valid sub_type")
        if self.config["controller"]["sub_type"] == "linear":
            s = 1
            g = 10
            e = w - s*quaternion.as_vector_part(q_curr)
            u1 = -1/2*q_curr.x-p1*w[1]*w[2] - s*0.5*(q_curr.w*w[0]+q_curr.y*w[2]-q_curr.z*w[1]) \
                - g*(e[0])
            u2 = -1/2*q_curr.y-p2*w[0]*w[2] - s*0.5*(q_curr.w*w[1]+q_curr.z*w[0]-q_curr.x*w[2]) \
                - g*(e[1])
            u3 = -1/2*q_curr.z-p3*w[0]*w[1] - s*0.5*(q_curr.w*w[2]+q_curr.x*w[1]-q_curr.y*w[0]) \
                - g*(e[2])
            u = np.array([u1, u2, u3])
        elif self.config["controller"]["sub_type"] == "arctan":
            raise(Exception("arctan not implemented"))
        #     a  = [1, 1, 1]
        #     b  = [1, 1, 1]
        #     c  = [1, 1, 1]
        #     s = 1
        #     g = 10

        #     phi = alpha*np.arctan(beta*quaternion.as_vector_part(q_curr))
        #     phi_dot = alpha*beta/(1+np.power(beta*quaternion.as_vector_part(q_curr),2))
        #     e = w - s*phi
        #     u1 = -1/2*q_curr.x-p1*w[1]*w[2] - s*0.5*phi_dot*(q_curr.w*w[0]+q_curr.y*w[2]-q_curr.z*w[1]) \
        #         - g*(e[0])
        #     u2 = -1/2*q_curr.y-p2*w[0]*w[2] - s*0.5*phi_dot*(q_curr.w*w[1]+q_curr.z*w[0]-q_curr.x*w[2]) \
        #         - g*(e[1])
        #     u3 = -1/2*q_curr.z-p3*w[0]*w[1] - s*0.5*phi_dot*(q_curr.w*w[2]+q_curr.x*w[1]-q_curr.y*w[0]) \
        #         - g*(e[2])
        elif self.config["controller"]["sub_type"] == "Shen":
            u_max = self.config["wheels"]["max_torque"]
            D_plus = satellite._wheel_module.D_psuedo_inv
            eta_0 = np.linalg.norm(D_plus)
            alpha = 0.2
            beta = 1.8
            k = 100
            eta_1 = 0.1
            upsilon = 0.01
            c_1 = 0.01
            c_2 = 0.1

            Omega = np.linalg.norm(w)**2 + np.linalg.norm(w) + 1
            s = w - alpha*np.arctan(beta*quaternion.as_vector_part(q_curr))
            s_norm = np.linalg.norm(s)
            s_norm_pow2 = s_norm**2
            eta_2 = upsilon/Omega

            # Update h
            h_dot = -c_1*self.h + c_2*(Omega*s_norm_pow2)/(s_norm + eta_2) 
            self.h = self.h + h_dot*self.t_sample

            D = satellite._wheel_module.D
            E = satellite._fault.mul_fault_matrix
            if True:
                A = D_plus
                B = np.atleast_2d(s).T
                c = -1*(k + self.h*Omega/(s_norm + eta_2))
                C = D_plus@np.atleast_2d(s).T*c
                phi = -1*(1 / (s_norm**2 + eta_1**2))
                F = np.atleast_2d(s)@D@E*phi
            
                u = C + (F@C/(1 - F@A@B))*A@B
                u = u.reshape(-1)
            else:
                if not satellite._fault.master_enable:
                    f = np.ones(3)
                else:
                    f = -1*satellite._wheel_module.D@satellite._fault.mul_fault_matrix@self.u_prev

                Gamma = k + s@f.reshape(-1,1)/(s_norm_pow2+eta_1**2) + self.h*Omega/(s_norm+eta_2)
                test = u_max/(eta_0*Gamma)
                if(s_norm >= test):
                    u = (D_plus*u_max)@s/(s_norm*eta_0)
                else:
                    u = -1*Gamma*D_plus@s

            if(np.shape(u)[0] != satellite._wheel_module.num_wheels):
                raise(Exception("invalid shape of u"))
                
            self.u_prev = u

        return u


    def calc_torque_control_output(self, q_curr : np.quaternion,  w : list[float], ref_q : np.quaternion, satellite, wheels_H, t) -> np.array:

        my_globals.results_data['control_time'].append(t)
        if self.type == "q_feedback":
            T_output = self.calc_q_feedback_control_torque(q_curr, w, ref_q, satellite.M_inertia, wheels_H)
            # T_output = self.inject_fault(T_output)
            if self.type == "adaptive":
                T_output = self.calc_adaptive_control_torque_output(T_output, q_curr, ref_q)

        elif self.type == "backstepping": 
            # if count_debug < COUNT_DEBUG_MAX:
            # print(f"backstepping control")
                # count_debug += 1
            T_output = self.calc_backstepping_control_torque(q_curr, w, ref_q, satellite)
        else:
            raise Exception(f"controller type {self.type} is not a valid type")
        # print(f"T_output {T_output.shape}")
        T_output = self.limit_torque(T_output)
        
        # T_output = self.low_pass_filter(T_output, self.T_output_prev, self.filter_coef)
        
        return T_output
    
    
    def update_M_inertia_inv_model(self, M_inertia_inv_model):
        self.M_inertia_inv_model = M_inertia_inv_model 

    M_inertia_inv_model = None
    w_prev = None
    q_prev : np.quaternion = None
    
    def calc_adaptive_model_output(self, ref_q):
        T_model = self.calc_q_feedback_control_torque(self.q_prev, self.w_prev, ref_q)
        w_result = self.w_prev + self.M_inertia_inv_model@(T_model)*self.t_sample
        self.w_prev = w_result
        inertial_v_q = np.quaternion(0, w_result[0], w_result[1], w_result[2])
        q_result = self.q_prev + 0.5*self.q_prev*inertial_v_q*self.t_sample
        self.q_prev = q_result
        return q_result

    theta_prev = 0
    def calc_adaptive_control_torque_output(self, T_com, q_meas : np.quaternion, q_ref : np.quaternion):
        q_model = self.calc_adaptive_model_output(q_ref)
        my_globals.results_data['control_adaptive_model_output'].append(my_utils.conv_Rotation_obj_to_euler_axis_angle(my_utils.conv_numpy_to_Rotation_obj_q(q_model)))
        q_error = q_meas.inverse()*q_model
        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        q_model_vec = np.array([q_model.x, q_model.y, q_model.z])
        
        theta = self.theta_prev - q_error_vec*self.adaptive_gain*q_model_vec*self.t_sample
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
        ref_angular_momentum = torque*self.control_t_sample + angular_momentum
        return ref_angular_momentum

    _torque_prev = 0
    def inject_fault(self, M):
        if self._fault.wheel_num > 2:
            raise("invaled fault injection wheel_num")
        if self._fault.enabled is False:
            return M
        
        if self._fault.type == "catastrophic":
            M[self._fault.wheel_num] = 0
        elif self._fault.type == "comm_delay":
            M[self._fault.wheel_num] = self.low_pass_filter(M[self._fault.wheel_num], self._torque_prev, self._fault.filter_coeff)
            self._torque_prev = M[self._fault.wheel_num]
        elif self._fault.type == "torque_limit":
            limit = self._fault.torque_limit*self.T_max
            if abs(M[self._fault.wheel_num]) > limit:
                M[self._fault.wheel_num] = limit*np.sign(M[self._fault.wheel_num])
        elif self._fault.type == "update_rate":
            pass
        else:
            raise("invalid fault type")
        return M