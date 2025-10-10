import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
import quaternion
import my_utils as my_utils
from fault import Fault
import observer

class Controller:
    enable : bool = True
    T_max : float = None
    T_min : float = 0.0
    t_sample : float = 0.1
    filter_coef : float = 0.0
    k : float = 1.0
    c : float = 1.0
    _fault : Fault = None
    type="q_feedback"
    types = ["q_feedback", "backstepping", "adaptive"]
    adaptive_gain = 0
    config = None
    wheel_module = None
    wheel_extended_state_observers = []
    results_data = None

    u_vec_prev : np.array = np.zeros(3)
    u_wheels_prev : np.array = None

    def __init__(self, fault=None, wheel_module=None, results_data=None,
                 w_sat_init=None, q_sat_init=None, config=None):
        self.results_data = results_data
        self._fault = fault
        self.config = config
        self.wheel_module = wheel_module

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
        self.type = config['controller']['type']
        if self.type not in self.types:
            raise Exception(f"controller type {type} type is not a valid type")
        if self.type == "adaptive":
            if w_sat_init is None or q_sat_init is None:
                raise Exception("initialization of angular velocity and quaternion is required for adaptive model")
            self.w_prev = w_sat_init
            self.quaternion_prev = q_sat_init
            self.adaptive_gain = config['controller']['adaptive_gain']
            self.results_data['control_adaptive_model_output'] = []
            for i, axis in enumerate(my_utils.xyz_axes):
                self.results_data[f'control_theta_{axis}'] = []
        if self.type == "adaptive":
            print("adaptive gain ", self.adaptive_gain)
        self.h = 0.1
        
        if config['simulation']['tuning'] is False:
            self.alpha = self.config['backstepping']['alpha']
            self.beta = self.config['backstepping']['beta']
            self.k = self.config['backstepping']['k']
            self.eta_1 = self.config['backstepping']['eta_1']
            self.upsilon = self.config['backstepping']['upsilon']
            self.c_1 = self.config['backstepping']['c_1']
            self.c_2 = self.config['backstepping']['c_2']
        
        self.u_vec_prev = np.zeros(3)
        self.u_wheels_prev = np.zeros(self.wheel_module.num_wheels)

    def calc_q_feedback_control_torque(self, q_curr, w, q_ref, M_inertia, H_wheels):
        # K = np.diag(np.full(3,self.k))
        # C = np.diag(self.c)
        w = np.array(w)
        
        q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(q_ref) # q_ref is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        # Hnet = M_inertia@(w) + H_wheels

        # return + K@q_error_vec - C@w + my_utils.cross_product(w, Hnet)

        return + self.config['controller']['kj']*M_inertia@q_error_vec - self.config['controller']['kd']*M_inertia@w
    
    h = 0
    sub_types = ["linear", "arctan", "Shen"]

    # Shen variables
    alpha = None
    beta = None
    k = None
    eta_1 = None
    upsilon = None
    c_1 = None
    c_2 = None
    def calc_backstepping_control_torque(self, q_curr: np.quaternion, w, satellite, E: np.array):
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
            D_plus = satellite.wheel_module.D_psuedo_inv
            # eta_0 = np.linalg.norm(D_plus)

            Omega = np.linalg.norm(w)**2 + np.linalg.norm(w) + 1
            s = w - self.alpha*np.arctan(self.beta*quaternion.as_vector_part(q_curr))
            s_norm = np.linalg.norm(s)
            s_norm_pow2 = s_norm**2
            eta_2 = self.upsilon/Omega

            # Update h
            h_dot = -self.c_1*self.h + self.c_2*(Omega*s_norm_pow2)/(s_norm + eta_2) 
            self.h = self.h + h_dot*self.t_sample

            D = satellite.wheel_module.D
            if True:
                A = D_plus
                B = np.atleast_2d(s).T
                c = -1*(self.k + self.h*Omega/(s_norm + eta_2))
                C = D_plus@np.atleast_2d(s).T * c
                phi = -1 * (1 / (s_norm**2 + self.eta_1**2))
                F = np.atleast_2d(s) @ D @ E * phi
            
                u = C + (F@C/(1 - F@A@B))*A@B
                u = u.reshape(-1)
            else:
                if not satellite._fault.master_enable:
                    f = np.ones(3)
                else:
                    f = -1*satellite.wheel_module.D@satellite._fault.mul_fault_matrix@self.u_wheels_prev

                Gamma = self.k + s@f.reshape(-1,1)/(s_norm_pow2+self.eta_1**2) + self.h*Omega/(s_norm+eta_2)
                test = u_max/(eta_0*Gamma)
                if(s_norm >= test):
                    u = (D_plus*u_max)@s/(s_norm*eta_0)
                else:
                    u = -1*Gamma*D_plus@s
            if(np.shape(u)[0] != satellite.wheel_module.num_wheels):
                raise(Exception("invalid shape of u"))
                

        return u

    next_t_sample : float = 0
    def calc_torque_control_output(self, t, q_curr : np.quaternion,  w : list[float], q_ref : np.quaternion, satellite, w_wheels, E) -> np.array:
        if t >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return self.u_vec_prev, self.u_wheels_prev
        
        q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(q_ref)
        H_wheels = satellite.wheel_module.D@w_wheels # @TODO check this!!!
        if self.type == "q_feedback":
            u = self.calc_q_feedback_control_torque(q_curr, w, q_ref, satellite.M_inertia, H_wheels)
            if self.type == "adaptive":
                u = self.calc_adaptive_control_torque_output(u, q_curr, q_ref)
            u_vec = u
            u_wheels = satellite.wheel_module.D_psuedo_inv@u_vec

        elif self.type == "backstepping":
            u = self.calc_backstepping_control_torque(q_error, w, satellite, E)
            u_wheels = u
            u_vec = satellite.wheel_module.D@u_wheels
        else:
            raise Exception(f"controller type {self.type} is not a valid type")
        # print(f"u {u.shape}")
        # u = self.limit_torque(u)
        
        # u = self.low_pass_filter(u, self.u_vec_prev, self.filter_coef)
        
        self.u_vec_prev, self.u_wheels_prev = u_vec, u_wheels

        return u_vec, u_wheels
    
    
    def update_M_inertia_inv_model(self, M_inertia_inv_model):
        self.M_inertia_inv_model = M_inertia_inv_model 

    M_inertia_inv_model = None
    w_prev = None
    q_prev : np.quaternion = None
    
    def calc_adaptive_model_output(self, q_ref):
        T_model = self.calc_q_feedback_control_torque(self.q_prev, self.w_prev, q_ref)
        w_result = self.w_prev + self.M_inertia_inv_model@(T_model)*self.t_sample
        self.w_prev = w_result
        inertial_v_q = np.quaternion(0, w_result[0], w_result[1], w_result[2])
        q_result = self.q_prev + 0.5*self.q_prev*inertial_v_q*self.t_sample
        self.q_prev = q_result
        return q_result

    theta_prev = 0
    def calc_adaptive_control_torque_output(self, T_com, q_meas : np.quaternion, q_ref : np.quaternion):
        q_model = self.calc_adaptive_model_output(q_ref)
        self.results_data['control_adaptive_model_output'].append(my_utils.conv_Rotation_obj_to_euler_axis_angle(my_utils.conv_numpy_to_Rotation_obj_q(q_model)))
        q_error = q_meas.inverse()*q_model
        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        q_model_vec = np.array([q_model.x, q_model.y, q_model.z])
        
        theta = self.theta_prev - q_error_vec*self.adaptive_gain*q_model_vec*self.t_sample
        for i, axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'control_theta_{axis}'].append(theta[i])
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