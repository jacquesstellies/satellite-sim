import numpy as np
from numpy import cos, diag, sin, tan, arccos, arcsin, arctan
import quaternion
import my_utils as my_utils
from fault import Fault
import observer
from scipy.spatial.transform import Rotation as R

def calc_rotation_matrix_from_quaternion(q: np.quaternion) -> np.array:
    q_vec = np.array([q.x, q.y, q.z])
    q_vec_sqr = q.vec@q.vec.reshape(-1)

    r = (q.w**2 - q_vec.reshape(-1)@q_vec)*np.eye(3) + 2*np.asmatrix(q_vec).T@np.asmatrix(q_vec) - 2*q.w*my_utils.skew_symmetric(q_vec)
    return r


class Controller:
    enable : bool = True
    T_max : float = None
    T_min : float = 0.0
    t_sample : float = 0.1
    filter_coef : float = 0.0
    k : float = 1.0
    c : float = 1.0
    _faults : list[Fault] = None
    type="pid"
    types = ["pid", "backstepping", "adaptive"]
    adaptive_gain = 0
    config = None
    wheel_module = None
    wheel_extended_state_observers = []
    results_data = None

    u_vec_prev : np.array = np.zeros(3)
    u_wheels_prev : np.array = None

    def __init__(self, faults=None, wheel_module=None, results_data=None,
                 w_sat_init=None, q_sat_init=None, config=None):
        self.results_data = results_data
        self._faults = faults
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

        self.q_prev = np.zeros(4)

        self.sub_type = config['controller']['sub_type']

    q_prev : np.quaternion = None
    def calc_pid_torque(self, q_error: np.quaternion, w : np.array, M_inertia, H_wheels):
        # K = np.diag(np.full(3,self.k))
        # C = np.diag(self.c)
        
        # q_error = q_curr.inverse() * my_utils.conv_Rotation_obj_to_numpy_q(q_ref) # q_ref is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        q_int = q_error + (self.q_prev * self.t_sample)
        self.q_prev = q_int
        q_int_vec = np.array([q_int.x, q_int.y, q_int.z])
        # Hnet = M_inertia@(w) + H_wheels

        # return + K@q_error_vec - C@w + my_utils.cross_product(w, Hnet)

        return + self.config['controller']['kj']*M_inertia@q_error_vec - self.config['controller']['kd']*M_inertia@w + self.config['controller']['ki']*q_int_vec
    
    h = 0
    sub_types = ["linear", "arctan", "Shen", "Nafadi_FNDO", "Nafadi_MFNDO"]
    sub_type = None

    # Shen variables
    alpha = None
    beta = None
    k = None
    eta_1 = None
    upsilon = None
    c_1 = None
    c_2 = None

    v_0_prev = np.zeros(2)
    chi_0_prev = np.zeros(2)
    chi_1_prev = np.zeros(2)
    # mu_prev = np.zeros(2)
    mu_prev = np.ones(2) * 0.00001

    def calc_backstepping_control_torque(self, q_err: np.quaternion, w, satellite, f_est: np.array):
        I1 = satellite.M_inertia[0][0]
        I2 = satellite.M_inertia[1][1]
        I3 = satellite.M_inertia[2][2]
        p1 = (I2 - I3)/I1
        p2 = (I3 - I1)/I2
        p3 = (I1 - I2)/I3
        alpha = 0.75
        beta = 8
        
        u = np.zeros(3)
        if self.sub_type not in self.sub_types:
            raise Exception(f"sub_type {self.sub_type} is not a valid sub_type")
        if self.sub_type == "linear":
            s = 1
            g = 10
            e = w - s*quaternion.as_vector_part(q_err)
            u1 = -1/2*q_err.x-p1*w[1]*w[2] - s*0.5*(q_err.w*w[0]+q_err.y*w[2]-q_err.z*w[1]) \
                - g*(e[0])
            u2 = -1/2*q_err.y-p2*w[0]*w[2] - s*0.5*(q_err.w*w[1]+q_err.z*w[0]-q_err.x*w[2]) \
                - g*(e[1])
            u3 = -1/2*q_err.z-p3*w[0]*w[1] - s*0.5*(q_err.w*w[2]+q_err.x*w[1]-q_err.y*w[0]) \
                - g*(e[2])
            u = np.array([u1, u2, u3])
        elif self.config["controller"]["sub_type"] == "arctan":
            raise(Exception("arctan not implemented"))
        #     a  = [1, 1, 1]
        #     b  = [1, 1, 1]
        #     c  = [1, 1, 1]
        #     s = 1
        #     g = 10

        #     phi = alpha*np.arctan(beta*quaternion.as_vector_part(q_err))
        #     phi_dot = alpha*beta/(1+np.power(beta*quaternion.as_vector_part(q_err),2))
        #     e = w - s*phi
        #     u1 = -1/2*q_err.x-p1*w[1]*w[2] - s*0.5*phi_dot*(q_err.w*w[0]+q_err.y*w[2]-q_err.z*w[1]) \
        #         - g*(e[0])
        #     u2 = -1/2*q_err.y-p2*w[0]*w[2] - s*0.5*phi_dot*(q_err.w*w[1]+q_err.z*w[0]-q_err.x*w[2]) \
        #         - g*(e[1])
        #     u3 = -1/2*q_err.z-p3*w[0]*w[1] - s*0.5*phi_dot*(q_err.w*w[2]+q_err.x*w[1]-q_err.y*w[0]) \
        #         - g*(e[2])
        elif self.config["controller"]["sub_type"] == "Shen":
            u_max = self.config["wheels"]["max_torque"]
            D_plus = satellite.wheel_module.D_psuedo_inv
            eta_0 = np.linalg.norm(D_plus)

            Omega = np.linalg.norm(w)**2 + np.linalg.norm(w) + 1
            s = w - self.alpha*np.arctan(self.beta*quaternion.as_vector_part(q_err))
            s_norm = np.linalg.norm(s)
            s_norm_pow2 = s_norm**2
            eta_2 = self.upsilon/Omega

            # Update h
            h_dot = -self.c_1*self.h + self.c_2*(Omega*s_norm_pow2)/(s_norm + eta_2) 
            self.h = self.h + h_dot*self.t_sample

            D = satellite.wheel_module.D
            if False:
                A = D_plus
                B = np.atleast_2d(s).T
                c = -1*(self.k + self.h*Omega/(s_norm + eta_2))
                C = D_plus@np.atleast_2d(s).T * c
                phi = -1 * (1 / (s_norm**2 + self.eta_1**2))
                F = np.atleast_2d(s) @ D @ E * phi


                # test = u_max/(eta_0*Gamma)
                # if(s_norm >= test):
                #     u = (D_plus*u_max)@s/(s_norm*eta_0)
                # else:
                u = C + (F@C/(1 - F@A@B))*A@B
                u = u.reshape(-1)


            else:
                Gamma = self.k + s@f_est.reshape(-1,1)/(s_norm_pow2+self.eta_1**2) + self.h*Omega/(s_norm+eta_2)
                test = u_max/(eta_0*Gamma)
                if(s_norm >= test):
                    u = -1*(D_plus*u_max)@s/(s_norm*eta_0)
                else:
                    u = -1*Gamma*D_plus@s
            if(np.shape(u)[0] != satellite.wheel_module.num_wheels):
                raise(Exception("invalid shape of u"))
                
        elif self.config["controller"]["sub_type"] == "Nafadi_FNDO":
            Gamma_z =  1.5*np.array([[1, 0], [0, 0.8]])
            L = 10*np.eye(2)
            k_0 = 0.9
            k_1 = 0.1
            Lambda = np.diag([3,3,6])
            # Lambda = np.diag([10,10,10])
            lambda_1 = Lambda[0,0]
            lambda_2 = Lambda[1,1]
            lambda_3 = Lambda[2,2]
            J_0 = satellite.M_inertia
            a = Lambda[1,1] - Lambda[2,2]
            b = Lambda[2,2] - Lambda[0,0]

            f_idx = 2 # fault index
            nf_idx = [i for i in range(3) if i != f_idx] # no fault indices

            C = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
            C_r = np.array([[C[nf_idx[0],nf_idx[0]], C[nf_idx[0],nf_idx[1]]], [C[nf_idx[1],nf_idx[0]], C[nf_idx[1],nf_idx[1]]]])

            Aq = np.array([[q_err.w, -q_err.z], [q_err.z, q_err.w], [-q_err.y, q_err.x]]) # @TODO generalize for any nf_idx
            q_err_vec = np.array([q_err.x, q_err.y, q_err.z])

            w_d = np.zeros(3) # desired angular velocity
            w_d_r = w_d

            w_r = np.array([w[nf_idx[0]], w[nf_idx[1]]])

            w_err = w - C@w_d
            w_err_r = np.array([w_err[nf_idx[0]], w_err[nf_idx[1]]])

            dw_d = np.zeros(3)
            dw_d_r = np.array([dw_d[nf_idx[0]], dw_d[nf_idx[1]]])

            F = -C_r@dw_d_r + (C[2,0]*w_d[2] + C[2,0]*w_d[1])*np.array([w_err[1], -w_err[0]])
            # F = np.zeros(2)

            chi_0 = self.chi_0_prev
            chi_1 = self.chi_1_prev
            v_0 = -k_0*diag(np.sqrt(L)@np.sqrt(np.abs(chi_0 - w_err_r)))@np.sign(chi_0 - w_err_r) + chi_1

            dchi_0 = v_0 + self.u_wheels_prev[nf_idx] + F
            dchi_1 = -k_1 * L @ np.sign(chi_1 - v_0)

            chi_0 = chi_0 + dchi_0*self.t_sample
            chi_1 = chi_1 + dchi_1*self.t_sample
            self.chi_0_prev = chi_0
            self.chi_1_prev = chi_1

            self.v_0_prev = v_0
            phi = -2*np.array([a*q_err.y*q_err.z + q_err.w*q_err.x*lambda_1, b*q_err.x *q_err.z + q_err.w*q_err.y*lambda_2])
            dphi = -2*np.array([[lambda_1*q_err.w, a*q_err.z, a*q_err.y], [b*q_err.z, lambda_2*q_err.w, b*q_err.x]])

            Z = w_err_r - phi

            u_r = - F - chi_1 + dphi@Aq@(Z+phi) + (q_err_vec.T @ Lambda @ Aq).T + Gamma_z @ Z

            u = np.array([u_r[0], u_r[1], 0])

        elif self.sub_type == "Nafadi_MFNDO":
            Gamma_z =  1.5*np.array([[1, 0], [0, 0.8]])
            Gamma_mu = 0.5*np.array([[0.01, 0], [0, 0.01]])
            L = 10*np.eye(2)
            k_0 = 0.1
            k = 0.15
            alpha_1 = 0.2
            alpha_2 = 0.3
            Lambda = np.diag([3,3,6])
            lambda_1 = Lambda[0,0]
            lambda_2 = Lambda[1,1]
            lambda_3 = Lambda[2,2]
            J_0 = satellite.M_inertia
            a = lambda_2 - lambda_3
            b = lambda_3 - lambda_1

            f_idx = 2 # fault index
            nf_idx = [i for i in range(3) if i != f_idx] # no fault indices

            C = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
            C_r = np.array([[C[nf_idx[0],nf_idx[0]], C[nf_idx[0],nf_idx[1]]], [C[nf_idx[1],nf_idx[0]], C[nf_idx[1],nf_idx[1]]]])

            Aq = 0.5*np.array([[q_err.w, -q_err.z], [q_err.z, q_err.w], [-q_err.y, q_err.x]]) # @TODO generalize for any nf_idx
            q_err_vec = np.array([q_err.x, q_err.y, q_err.z])

            w_d = np.zeros(3) # desired angular velocity
            w_d_r = w_d

            w_r = np.array([w[nf_idx[0]], w[nf_idx[1]]])

            w_err = w - C@w_d
            w_err_r = np.array([w_err[nf_idx[0]], w_err[nf_idx[1]]])

            dw_d = np.zeros(3)
            dw_d_r = np.array([dw_d[nf_idx[0]], dw_d[nf_idx[1]]])

            F = -C_r@dw_d_r + (C[2,0]*w_d[2] + C[2,0]*w_d[1])*np.array([w_err[1], -w_err[0]])
            # F = np.zeros(2)

            chi_0 = self.chi_0_prev
            chi_1 = self.chi_1_prev
            # v_0 = -k_0*diag(np.sqrt(L)@np.sqrt(np.abs(my_utils.col_vec(chi_0) - my_utils.col_vec(w_err_r))))@np.sign(my_utils.col_vec(chi_0) - my_utils.col_vec(w_err_r)) + my_utils.col_vec(chi_1)
            # v_0 = np.array([v_0[0,0], v_0[1,0]])
            v_0 = -k_0*diag(np.sqrt(L)@np.sqrt(np.abs(chi_0 - w_err_r)))@np.sign(chi_0 - w_err_r) + chi_1

            dchi_0 = v_0 + self.u_wheels_prev[nf_idx] + F
            dchi_1 = -k * L @ np.sign(chi_1 - v_0)

            chi_0 = chi_0 + dchi_0*self.t_sample
            chi_1 = chi_1 + dchi_1*self.t_sample
            self.chi_0_prev = chi_0
            self.chi_1_prev = chi_1

            self.v_0_prev = v_0
            phi = -2*np.array([a*q_err.y*q_err.z + q_err.w*q_err.x*lambda_1, b*q_err.x*q_err.z + q_err.w*q_err.y*lambda_2])
            dphi_qe = -2*np.array([[lambda_1*q_err.w, a*q_err.z, a*q_err.y], [b*q_err.z, lambda_2*q_err.w, b*q_err.x]])

            Z = w_err_r - phi
            temp_1 = Aq@(my_utils.col_vec(Z+phi))
            temp_2 = dphi_qe@temp_1

            temp_3 = my_utils.col_vec(self.mu_prev)@my_utils.row_vec(Z)/np.linalg.norm(self.mu_prev)**2
            temp_4 = (my_utils.col_vec(my_utils.sat_delta_vec(self.mu_prev))*alpha_1)

            dmu = temp_3 @ \
                (temp_4 - my_utils.col_vec(F) - my_utils.col_vec(chi_1) + temp_2 + (my_utils.row_vec(q_err_vec) @ Lambda @ Aq).T + Gamma_z @ my_utils.col_vec(Z)) \
                    - Gamma_mu@my_utils.col_vec(self.mu_prev)
            dmu = np.array(dmu[0,0], dmu[1,0])
            

            mu = self.mu_prev + dmu*self.t_sample
            self.mu_prev = mu

            u_r = -1*alpha_1*my_utils.sat_delta_vec(mu) - alpha_2*my_utils.sat_delta_vec(Z)
        
            u = np.array([u_r[0], u_r[1], 0])
        return u

    next_t_sample : float = 0
    def calc_torque_control_output(self, t, q_curr : np.quaternion,  w_sat : np.array, q_ref : np.quaternion, satellite, w_wheels : np.array, f_est : np.array) -> np.array:
        if t >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return self.u_vec_prev, self.u_wheels_prev
        
        # q_sat_error =  q_ref * q_curr.inverse()
        q_sat_error =  q_curr.inverse() * q_ref
        H_wheels = satellite.wheel_module.D@w_wheels # @TODO check this!!!
        if self.type == "pid":
            u = self.calc_pid_torque(q_sat_error, w_sat, satellite.M_inertia, H_wheels)
            if self.type == "adaptive":
                u = self.calc_adaptive_control_torque_output(u, q_curr, q_ref)
            u_vec = u
            u_wheels = satellite.wheel_module.D_psuedo_inv@u_vec

        elif self.type == "backstepping":
            u = self.calc_backstepping_control_torque(q_sat_error, w_sat, satellite, f_est)
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
        T_model = self.calc_pid_torque(self.q_prev, self.w_prev, q_ref)
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