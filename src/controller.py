import sys

import numpy as np
from numpy import cos, diag, sin, tan, arccos, arcsin, arctan
import quaternion
import my_utils as my_utils
from my_utils import col_vec, row_vec
from fault import Fault
import observer
from scipy.spatial.transform import Rotation as R

def calc_rotation_matrix_from_quaternion(q: np.quaternion) -> np.array:
    q_vec = np.array([q.x, q.y, q.z])
    q_vec_sqr = q.vec@q.vec.reshape(-1)

    r = (q.w**2 - q_vec.reshape(-1)@q_vec)*np.eye(3) + 2*np.asmatrix(q_vec).T@np.asmatrix(q_vec) - 2*q.w*my_utils.skew_symmetric(q_vec)
    return r


class NadafiController:

    Gamma_z11 = 0.0
    Gamma_z22 = 0.0
    Gamma_z = None
    L11 = 0.0
    L22 = 0.0
    L = None
    kappa_0 = None
    kappa_1 = None
    Lambda = None
    lambda_1 = None
    lambda_2 = None
    lambda_3 = None
    a = None
    b = None
    alpha_1 = None
    alpha_2 = None
    Gamma_mu11 = None
    Gamma_mu22 = None
    t_sample = None

    chi_0 = np.asmatrix(np.zeros(2)).T
    chi_1 = np.asmatrix(np.zeros(2)).T
    v_0 = np.asmatrix(np.zeros(2)).T
    mu = np.asmatrix(np.ones(2)*0.00001).T

    config = None
    
    def __init__(self, config):
        self.config = config
        if not config['controller']['sub_type'].startswith("Nadafi"):
            return
        Nadafi_config = self.config['Nadafi']
        self.Gamma_z11 = np.array(Nadafi_config['Gamma_z11'])
        self.Gamma_z22 = np.array(Nadafi_config['Gamma_z22'])

        self.lambda_1 = Nadafi_config['lambda_1']
        self.lambda_2 = Nadafi_config['lambda_2']
        self.lambda_3 = Nadafi_config['lambda_3']

        self.a = self.lambda_2 - self.lambda_3
        self.b = self.lambda_3 - self.lambda_1
        self.t_sample = config['controller']['t_sample']
        if config['simulation']['verbose'] is True:
            print("Nadafi Controller Gains:")
            print("Gamma_z11 ", self.Gamma_z11)
            print("Gamma_z22 ", self.Gamma_z22)
            print("lambda_1 ", self.lambda_1)
            print("lambda_2 ", self.lambda_2)
            print("lambda_3 ", self.lambda_3)

        if self.config['controller']['sub_type'] == 'Nadafi_FNDO':
            self.L11 = np.array(Nadafi_config['L11'])
            self.L22 = np.array(Nadafi_config['L22'])

            self.kappa_0 = Nadafi_config['kappa_0']
            self.kappa_1 = Nadafi_config['kappa_1']

        if self.config['controller']['sub_type'] == 'Nadafi_MFNDO':
            self.L11 = np.array(Nadafi_config['L11'])
            self.L22 = np.array(Nadafi_config['L22'])

            self.kappa_0 = Nadafi_config['kappa_0']
            self.kappa_1 = Nadafi_config['kappa_1']

            self.alpha_1 = Nadafi_config['alpha_1']
            self.alpha_2 = Nadafi_config['alpha_2']
            self.Gamma_mu11 = Nadafi_config['Gamma_mu11']
            self.Gamma_mu22 = Nadafi_config['Gamma_mu22']

    def calc_output_BS(self, q_err: np.quaternion, w : np.array, w_d : np.array):
        f_idx = 2 # fault index
        nf_idx = [i for i in range(3) if i != f_idx] # no fault indices

        J_0_r = np.array(self.config['satellite']['M_Inertia'])[nf_idx][:, nf_idx]

        C = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
        # C_r = np.array([[C[nf_idx[0],nf_idx[0]], C[nf_idx[0],nf_idx[1]]], [C[nf_idx[1],nf_idx[0]], C[nf_idx[1],nf_idx[1]]]])
        # C = my_utils.conv_quat_to_dcm_nadafi(q_err)
        C_r = np.array([[C[0,0], C[0,1]], [C[1,0], C[1,1]]])

        Aq = 0.5*np.array([[q_err.w, -q_err.z], [q_err.z, q_err.w], [-q_err.y, q_err.x]]) # @TODO generalize for any nf_idx
        q_err_vec = my_utils.col_vec(np.array([q_err.x, q_err.y, q_err.z]))

        w_d = my_utils.col_vec(w_d)
        w = my_utils.col_vec(w)

        w_r = np.array([w[nf_idx[0],0], w[nf_idx[1],0]])

        w_err = w - C@w_d
        # w_err_r = my_utils.col_vec(np.array([w_err[nf_idx[0]], w_err[nf_idx[1]]]))
        w_err_r = my_utils.col_vec(np.array([w_err[0], w_err[1]]))
        # w_err_r = np.array([w_err[0], w_err[1]])

        dw_d = my_utils.col_vec(np.zeros(3))
        # dw_d_r = my_utils.col_vec(np.array([dw_d[nf_idx[0]], dw_d[nf_idx[1]]]))
        dw_d_r = my_utils.col_vec(np.array([[dw_d[0]], [dw_d[1]]]))

        F = -C_r@dw_d_r + (C[2,0]*w_d[2,0] + C[2,1]*w_d[1,0])*my_utils.col_vec(np.array([w_err[1,0], -w_err[0,0]]))
        # F = -C_r@dw_d_r + (C[2,0]*w_d[2] + C[2,0]*w_d[1])*np.array([w_err[1], -w_err[0]])
        # F = np.zeros(2)

        phi = -2*my_utils.col_vec([self.a*q_err.y*q_err.z + q_err.w*q_err.x*self.lambda_1, 
                                   self.b*q_err.x *q_err.z + q_err.w*q_err.y*self.lambda_2])
        # phi = -2*np.array([self.a*q_err.y*q_err.z + q_err.w*q_err.x*self.lambda_1, self.b*q_err.x *q_err.z + q_err.w*q_err.y*self.lambda_2])
        dphi = np.array([[-2*self.lambda_1*q_err.w, -2*self.a*q_err.z, -2*self.a*q_err.y], 
                         [-2*self.b*q_err.z, -2*self.lambda_2*q_err.w, -2*self.b*q_err.x]])

        Z = w_err_r - phi

        # u_r = - F + dphi@Aq@(w_err_r) - (q_err_vec.T @ self.Lambda @ Aq).T - self.Gamma_z @ Z
        u_r = - F + dphi@Aq@(Z + phi) - (q_err_vec.T @ np.diag([self.lambda_1, self.lambda_2, self.lambda_3]) @ Aq).T - np.diag([self.Gamma_z11, self.Gamma_z22]) @ Z
        # print(f"t: qe=[{q_err.w:.4f},{q_err.x:.4f},{q_err.y:.4f},{q_err.z:.4f}] "
        # f"|Z|={np.linalg.norm(Z):.4f} u_r={u_r.flatten()}"
        # f"Z={Z.flatten()}"
        # , file=sys.stderr)

        h_w = -1*J_0_r @ u_r #+ np  .skew_symmetric(w_r) @ J_0_r @ w_r
        return np.array([h_w[0,0], h_w[1,0], 0])
        # return np.array([u_r[0], u_r[1], 0])
    
    ##############################################################################################
    def calc_output_BS_FNDO(self, q_err: np.quaternion, w : np.array, u_wheels_prev : np.array, w_d : np.array):
        f_idx = 2 # fault index
        nf_idx = [i for i in range(3) if i != f_idx] # no fault indices

        J_0_r = np.array(self.config['satellite']['M_Inertia'])[nf_idx][:, nf_idx]
        J_0_inv_r = np.array(self.satellite.M_inertia_inv)[nf_idx][:, nf_idx]

        C = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
        # C_r = np.array([[C[nf_idx[0],nf_idx[0]], C[nf_idx[0],nf_idx[1]]], [C[nf_idx[1],nf_idx[0]], C[nf_idx[1],nf_idx[1]]]])
        # C = my_utils.conv_quat_to_dcm_nadafi(q_err)
        C_r = np.array([[C[0,0], C[0,1]], [C[1,0], C[1,1]]])

        Aq = 0.5*np.array([[q_err.w, -q_err.z], [q_err.z, q_err.w], [-q_err.y, q_err.x]]) # @TODO generalize for any nf_idx
        q_err_vec = my_utils.col_vec(np.array([q_err.x, q_err.y, q_err.z]))

        w_d = my_utils.col_vec(w_d)
        w = my_utils.col_vec(w)

        w_r = np.array([w[nf_idx[0],0], w[nf_idx[1],0]])

        w_err = w - C@w_d
        # w_err_r = my_utils.col_vec(np.array([w_err[nf_idx[0]], w_err[nf_idx[1]]]))
        w_err_r = my_utils.col_vec(np.array([w_err[0], w_err[1]]))
        # w_err_r = np.array([w_err[0], w_err[1]])

        dw_d = my_utils.col_vec(np.zeros(3))
        # dw_d_r = my_utils.col_vec(np.array([dw_d[nf_idx[0]], dw_d[nf_idx[1]]]))
        dw_d_r = my_utils.col_vec(np.array([[dw_d[0]], [dw_d[1]]]))

        F = -C_r@dw_d_r + (C[2,0]*w_d[2,0] + C[2,1]*w_d[1,0])*my_utils.col_vec(np.array([w_err[1,0], -w_err[0,0]]))
        # F = -C_r@dw_d_r + (C[2,0]*w_d[2] + C[2,0]*w_d[1])*np.array([w_err[1], -w_err[0]])
        # F = np.zeros(2)

        phi = -2*my_utils.col_vec([self.a*q_err.y*q_err.z + q_err.w*q_err.x*self.lambda_1, 
                                   self.b*q_err.x *q_err.z + q_err.w*q_err.y*self.lambda_2])
        dphi = np.array([[-2*self.lambda_1*q_err.w, -2*self.a*q_err.z, -2*self.a*q_err.y], 
                         [-2*self.b*q_err.z, -2*self.lambda_2*q_err.w, -2*self.b*q_err.x]])

        Z = w_err_r - phi

        L = np.array([[self.L11, 0], [0, self.L22]])

        v_temp = np.sqrt(L)@np.sqrt(np.abs(self.chi_0 - w_err_r))
        self.v_0 = -self.kappa_0*np.diag([v_temp[0,0], v_temp[1,0]])@np.sign(self.chi_0 - w_err_r) + self.chi_1
        u_wheels_prev =  my_utils.col_vec(u_wheels_prev)
        dchi_0 = self.v_0 + -1 * J_0_inv_r @ u_wheels_prev[nf_idx,] + F
        dchi_1 = -self.kappa_1 * L @ np.sign(self.chi_1 - self.v_0)
        # dchi_1 = np.zeros(2)

        self.chi_0 = self.chi_0 + dchi_0*self.t_sample
        self.chi_1 = self.chi_1 + dchi_1*self.t_sample

        u_r = - self.chi_1 - F + dphi@Aq@(Z + phi) - (q_err_vec.T @ np.diag([self.lambda_1, self.lambda_2, self.lambda_3]) @ Aq).T - np.diag([self.Gamma_z11, self.Gamma_z22]) @ Z
        h_w = -1*J_0_r @ u_r #+ np  .skew_symmetric(w_r) @ J_0_r @ w_r
        
        return np.array([h_w[0,0], h_w[1,0], 0])
    
    def calc_output_BS_MFNDO(self, q_err: np.quaternion, w : np.array, u_wheels_prev : np.array):
        # @TODO implement MFNDO version of Nadafi controller
        
        u_wheels_prev = my_utils.col_vec(u_wheels_prev)
        w = my_utils.col_vec(w)

        f_idx = 2 # fault index
        nf_idx = [i for i in range(3) if i != f_idx] # no fault indices

        C = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
        C_r = np.array([[C[nf_idx[0],nf_idx[0]], C[nf_idx[0],nf_idx[1]]], [C[nf_idx[1],nf_idx[0]], C[nf_idx[1],nf_idx[1]]]])

        Aq = 0.5*np.array([[q_err.w, -q_err.z], [q_err.z, q_err.w], [-q_err.y, q_err.x]]) # @TODO generalize for any nf_idx
        q_err_vec = my_utils.col_vec(np.array([q_err.x, q_err.y, q_err.z]))

        w_d = my_utils.col_vec(np.array([0, 0, 0])) # desired angular velocity
        w_d_r = w_d
        w_r = my_utils.col_vec(np.array([w[nf_idx[0]], w[nf_idx[1]]]))
        
        w_err = w - C@w_d
        w_err_r = my_utils.col_vec(np.array([w_err[nf_idx[0]], w_err[nf_idx[1]]]))

        dw_d = np.zeros(3)
        dw_d_r = my_utils.col_vec(np.array([dw_d[nf_idx[0]], dw_d[nf_idx[1]]]))

        # F = -C_r@dw_d_r + (C[2,0]*w_d[2] + C[2,0]*w_d[1])*np.array([w_err[1], -w_err[0]])
        F = my_utils.col_vec(np.zeros(2))

        # v_0 = -k_0*diag(np.sqrt(L)@np.sqrt(np.abs(my_utils.col_vec(chi_0) - my_utils.col_vec(w_err_r))))@np.sign(my_utils.col_vec(chi_0) - my_utils.col_vec(w_err_r)) + my_utils.col_vec(chi_1)
        # v_0 = np.array([v_0[0,0], v_0[1,0]])
        L = np.diag([self.L11, self.L22])
        # print(self.chi_0.shape)
        # print(w_err_r.shape)
        # print(np.sqrt(L).shape)
        # print(np.sqrt(np.abs(self.chi_0 - w_err_r)).shape)
        # print(self.kappa_0)
        # print(np.sqrt(L)@np.sqrt(np.abs(self.chi_0 - w_err_r)))
        v_temp = np.sqrt(L)@np.sqrt(np.abs(self.chi_0 - w_err_r))
        self.v_0 = -self.kappa_0*np.diag([v_temp[0,0], v_temp[1,0]])@np.sign(self.chi_0 - w_err_r) + self.chi_1
        
        dchi_0 = self.v_0 + u_wheels_prev[nf_idx] + F
        dchi_1 = -self.kappa_1 * L @ np.sign(self.chi_1 - self.v_0)
        # dchi_1 = np.zeros(2)

        self.chi_0 = self.chi_0 + dchi_0*self.t_sample
        self.chi_1 = self.chi_1 + dchi_1*self.t_sample

        phi = -2*my_utils.col_vec([self.a*q_err.y*q_err.z + q_err.w*q_err.x*self.lambda_1, 
                                   self.b*q_err.x *q_err.z + q_err.w*q_err.y*self.lambda_2])
        # phi = -2*np.array([self.a*q_err.y*q_err.z + q_err.w*q_err.x*self.lambda_1, self.b*q_err.x *q_err.z + q_err.w*q_err.y*self.lambda_2])
        dphi = np.array([[-2*self.lambda_1*q_err.w, -2*self.a*q_err.z, -2*self.a*q_err.y], 
                         [-2*self.b*q_err.z, -2*self.lambda_2*q_err.w, -2*self.b*q_err.x]])
        Z = w_err_r - phi
        temp_1 = Aq@(Z+phi)
        temp_2 = dphi@temp_1

        temp_3 = (self.mu@Z.T)/np.linalg.norm(self.mu)**2
        temp_4 = my_utils.sat_delta_vec(self.mu)*self.alpha_1

        dmu = temp_3 @ \
            (temp_4 - F - self.chi_1 + temp_2 - \
             (q_err_vec.T @ np.diag([self.lambda_1, self.lambda_2, self.lambda_3]) @ Aq).T - \
             np.diag([self.Gamma_z11, self.Gamma_z22]) @ Z) \
                    - np.diag([self.Gamma_mu11, self.Gamma_mu22])@self.mu      

        self.mu = self.mu + dmu*self.t_sample

        u_r = -1*temp_4 - self.alpha_2*my_utils.sat_delta_vec(Z)

        return np.array([u_r[0,0], u_r[1,0], 0])

class ZarouratiController:
        def __init__(self, config):
            self.config = config
            self.h = 0.1

class UrakuboController:
    J = None
    alpha = 0.0
    beta = 0.0
    def __init__(self, config):
        self.config = config
        # self.J_t = config['satellite']['M_inertia']
        self.alpha = config['Urakubo']['alpha']
        self.beta = config['Urakubo']['beta']
        # self.J = np.array([[0, -1], [1, 0]])
    
    def calc_output(self, q_err: np.quaternion):
        e1, e2, e3 = quaternion.as_vector_part(q_err)
        e0 = q_err.w
        
        # Matrix B from Equation 23
        B = 0.5 * np.array([
            [e0, -e3],
            [e3,  e0],
            [-e2, e1]
        ])
        
        # Gradient of Lyapunov function V = 0.5 * (e1^2 + e2^2 + e3^2)
        # grad_V = [dV/de1, dV/de2, dV/de3]^T = [e1, e2, e3]^T
        grad_V = np.array([e1, e2, e3])
        
        # Helper term: B^T * grad_V
        BT_gradV = B.T @ grad_V
        
        # g = |B^T * grad_V| (Equation 27)
        g = 0.5 * e0 * np.sqrt(e1**2 + e2**2)
        
        # Define I and J (Equation 26)
        I_mat = np.eye(2)
        J_mat = np.array([[0, -1], [1, 0]])
        
        # Handle singularity where g = 0 (Equation 25/30)
        # if g < 1e-12:
        #     return np.zeros(2)
        
        # if use_continuous_form:
        #     # Modified law for stability analysis (Equation 30)
        #     matrix_term = (self.alpha * I_mat + 
        #                    self.beta * np.tanh(g**2 / self.epsilon) * (e3 / g**2) * J_mat)
        # else:
            # Original discontinuous law (Equation 25)
        matrix_term = self.alpha * I_mat + self.beta * (e3 / g**2) * J_mat
            
        u = -matrix_term @ BT_gradV
        return [u[0], u[1], 0]
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
    types = ["pid", "backstepping", "adaptive", "lyapunov"]
    sub_types = ["linear", "arctan", "Shen", "Nadafi_BS", "Nadafi_FNDO", "Nadafi_MFNDO", "Zarourati"]
    sub_type = None
    adaptive_gain = 0
    config = None
    wheel_module = None
    wheel_extended_state_observers = []
    results_data = None

    u_vec_prev : np.array = np.zeros(3)
    u_wheels_prev : np.array = None

    satellite = None
    nadafi_controller : NadafiController = None

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
        self.e_d_prev = col_vec(np.zeros(2))
        if self.sub_type == "Nadafi_BS" or self.sub_type == "Nadafi_FNDO" or self.sub_type == "Nadafi_MFNDO":
            self.nadafi_controller = NadafiController(config)

    def init_satellite(self, satellite):
        self.satellite = satellite
        if self.nadafi_controller is not None:
            self.nadafi_controller.satellite = satellite

    q_prev : np.quaternion = None
    def calc_pid_torque(self, q_error: np.quaternion, w : np.array, M_inertia, H_wheels, w_d):
        K = np.diag(np.full(3,self.k))
        C = np.diag(self.c)
        
        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        q_int = q_error + (self.q_prev * self.t_sample)
        self.q_prev = q_int
        q_int_vec = np.array([q_int.x, q_int.y, q_int.z])
        Hnet = M_inertia@(w) + H_wheels
        w_e = w - w_d
        # return - K@q_error_vec - C@w #+ my_utils.cross_product_M31M31(w, Hnet)

        return - self.config['controller']['kj']*M_inertia@q_error_vec + self.config['controller']['kd']*M_inertia@w_e + self.config['controller']['ki']*q_int_vec #+ my_utils.cross_product_M31M31(w, H_wheels)
        # return my_utils.sat_norm(my_utils.sat_vec(self.config['controller']['kj']*M_inertia@q_error_vec, self.config['wheels']['max_torque']) - self.config['controller']['kd']*M_inertia@w) + self.config['controller']['ki']*q_int_vec + my_utils.cross_product_M31M31(w, Hnet)
    
    h = 0

    # Shen variables
    alpha = None
    beta = None
    k = None
    eta_1 = None
    upsilon = None
    c_1 = None
    c_2 = None

    v_0 = np.zeros(2)
    chi_0 = np.zeros(2)
    chi_1 = np.zeros(2)
    # mu = np.zeros(2)
    mu = np.ones(2) * 0.00001

    phi_hat_prev = 0
    e_d_prev = np.zeros(2)
    we_u_zarourati_prev = 0.0
    dwe_u_filtered = 0.0

    init = True
    def calc_backstepping_control_torque(self, q_err: np.quaternion, q_curr : np.quaternion, q_ref : np.quaternion, w, satellite, f_est: np.array, t: float):
        
        u = np.zeros(3)
        if self.sub_type not in self.sub_types:
            raise Exception(f"sub_type {self.sub_type} is not a valid sub_type")
        if self.sub_type == "linear":
            I1 = satellite.M_inertia[0][0]
            I2 = satellite.M_inertia[1][1]
            I3 = satellite.M_inertia[2][2]
            p1 = (I2 - I3)/I1
            p2 = (I3 - I1)/I2
            p3 = (I1 - I2)/I3
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
                
        elif self.config["controller"]["sub_type"] == "Nadafi_FNDO":
        #    q_err = my_utils.get_quaternion_error_Nadafi(q_curr, q_ref)
           q_err = my_utils.get_quaternion_error_Nadafi(q_ref, q_curr)
           u = self.nadafi_controller.calc_output_BS_FNDO(q_err, w, self.u_wheels_prev, satellite.w_ref)

        elif self.config["controller"]["sub_type"] == "Nadafi_BS":
            q_err = my_utils.get_quaternion_error_Nadafi(q_ref, q_curr)
            u = self.nadafi_controller.calc_output_BS(q_err, w, satellite.w_ref)

        elif self.sub_type == "Nadafi_MFNDO":
            q_err = my_utils.get_quaternion_error_Nadafi(q_curr, q_ref)
            u = self.nadafi_controller.calc_output_BS_MFNDO(q_err, w, self.u_wheels_prev)

        
        elif self.sub_type == "Zarourati":

            """
            Implementation of UATC based on Zarourati et al. (2024).
            """
            # Nominal Inertia Matrix
            # J0 = J_nom
            J0 = satellite.M_inertia 

            # Control Gains
            k_a = 1.0
            k_u = 0.5
            k_w = 100.0
            k_d = 5.0
            k_phi = 1.0
            # Parameters for auxiliary variable xi
            gamma0, gamma1, gamma2 = (0.5, 1.0, 0.001)
            
            # Reduced representation indices for #RW2 failure (Actuated: 1, 3)
            actuated_idx = np.array([0, 2]) 
            unactuated_idx = 1
            
            # Skew-symmetric helper matrix G1
            G1 = np.array([[0, -1], [1, 0]])

            # 1. Extract transformed error states
            q_ev = np.array([q_err.x, q_err.y, q_err.z])
            e_u = q_ev[unactuated_idx]
            e_a = col_vec(np.array(q_ev[actuated_idx]))
            e4 = q_err.w
            
            w_d = col_vec(np.zeros(3)) # desired angular velocity
            w_d_r = col_vec(np.array([w_d[actuated_idx[0]], w_d[actuated_idx[1]]]))
            dw_d = col_vec(np.zeros(3))
            dw_d_r = col_vec(np.array([dw_d[actuated_idx[0]], dw_d[actuated_idx[1]]]))

            A = R.from_quat([q_err.x, q_err.y, q_err.z, q_err.w]).as_matrix()
            A_r = A[np.ix_(actuated_idx, actuated_idx)]
            w = col_vec(w)
            we = w - A@w_d
            we_u = we[unactuated_idx]
            we_a = we[actuated_idx]
            assert(we_a.shape == (2,1))

            
            B = e4 * we_u + q_err.x * w_d[2] - q_err.z * w_d[0] # @TODO generalize for any unactuated_idx
            de_u = 0.5 * (B + e_a.T @ G1 @ we_a)

            G2 = np.array([[e4, e_u], [-e_u, e4]])

            de_a = 0.5 * (G1 @ e_a * we_u + G2 @ we_a)
            assert(de_a.shape == (2,1))
            de4 = -0.5 * (e_u * we_u + e_a.T @ we_a)
            de4 = de4[0, 0]

            xi = gamma0 * np.exp(-gamma1 * t) + gamma2
            dxi = gamma0 * (-gamma1 * np.exp(-gamma1 * t))
            ddxi = gamma0 * gamma1**2 * np.exp(-gamma1 * t)
            
            # ω̇_eu via finite difference
            dwe_u = float(we_u - self.we_u_zarourati_prev) / self.t_sample
            self.we_u_zarourati_prev = we_u
            # J_inv = self.satellite.M_inertia_inv
            # dwe = - J_inv @ my_utils.skew_symmetric(w)@satellite.H + my_utils.skew_symmetric(we) @ A @ w_d - J_inv @ A @ dw_d + J_inv @ satellite.wheel_module.D @ we # how to calculate disturbance?
            # dwe_u = dwe[unactuated_idx, 0]
            
            # 2. Kinematic Controller terms
            kappa1 = (1.0 / xi**2) * (k_u * e_u + e4 * we_u)
            kappa2 = (2.0 * dxi / (e4 * xi)) + k_a * e4 # Note: text has a typo in 26, using logic from 25
            kappa1 = kappa1[0,0] # convert 1x1 numpy matrix to scalar
            Gamma = 0.5 * (k_a * e4 * e_u + e4 * kappa1 + we_u)
            Gamma = Gamma[0,0]

            dkappa1 = 1/xi**2 * (k_u * de_u + de4 * we_u + e4 * dwe_u) - 2*dxi/xi**3 * (k_u * e_u + e4 * we_u)
            dkappa2 = 2 / (e4**2 * xi ** 2) * (e4 * xi * ddxi - de4 * xi * dxi - e4 * dxi**2) + k_a * de4
            dkappa1 = dkappa1[0,0]

            # Assuming e_d follows the oscillator-like eq (24)
            # Simplified for instant computation as a vector with norm xi
            if self.init is True:
                self.e_d_prev = col_vec(np.array([np.sqrt(gamma0 + gamma2), np.sqrt(gamma0 + gamma2)]))  # ||e_d(0)|| = xi satisfies Eq. (24)
                self.init = False
            E = np.array([[0, 1], [-1, 0]])
            de_d = (dxi/xi)*self.e_d_prev + Gamma * E @ self.e_d_prev

            assert(de_d.shape == (2,1))
            e_d = self.e_d_prev + de_d * self.t_sample
            assert(e_d.shape == (2,1))
            e_d_norm = np.linalg.norm(e_d)
            if np.abs(e_d_norm - xi) > 1e-6:
                e_d = (e_d / e_d_norm) * xi
            self.e_d_prev = e_d
            
            # Virtual control input uk 
            u_k = -k_a * e4 * e_a + kappa1 * (G1 @ e_d) + kappa2 * e_d
            assert(u_k.shape == (2,1))
            du_k = -k_a * de4 * e_a - k_a * e4 * de_a + (dkappa1 * G1 @ e_d + dkappa2 * e_d) + (kappa1 * G1 @ de_d + kappa2 * de_d)
            
            assert(du_k.shape == (2, 1))
            
            # 3. Dynamic error eta
            eta = u_k - we_a
            
            # 4. Reduced Static and Dynamic components
            J_r = J0[np.ix_(actuated_idx, actuated_idx)] # J^r (Reduced inertia matrix)
            Sw = np.array([[0, -w[2,0], w[1,0]], [w[2,0], 0, -w[0,0]], [-w[1,0], w[0,0], 0]])
            Sw_r = Sw[np.ix_(actuated_idx, actuated_idx)]
            assert(Sw_r.shape == (2,2))

            Swe_r = Sw_r

            H = satellite.M_inertia @ w + col_vec(self.wheel_module.H_vec)
            # print(f"H.shape: {H.shape}")
            H_r = H[actuated_idx]
            assert(H_r.shape == (2,1))
            C_r = satellite.wheel_module.D[np.ix_(actuated_idx, actuated_idx)]
        
            e_a_tilde = e_d - e_a
            assert(e_a_tilde.shape == (2,1))
            
            # 5. Adaptive update law for phi_hat (Eq. 29)
            phi_hat_dot = (k_phi / k_d) * (np.cosh(self.phi_hat_prev)**2) * float(np.linalg.norm(eta))
            self.phi_hat_prev = self.phi_hat_prev + phi_hat_dot * self.t_sample

            # 6. Final Control Law uc^r
            u_c_r = C_r.T @ (Swe_r @ H_r - J_r @ Sw_r @ A_r @ w_d_r + J_r @ A_r @ dw_d_r + J_r @ du_k + k_w * np.tanh(eta)\
                           + 0.5 * G1 @ e_a * e_u + 0.5 * e_a_tilde + k_d * np.sign(eta) * np.tanh(self.phi_hat_prev))

            u_max = self.config['wheels']['max_torque']
            u_c_r = np.clip(np.asarray(u_c_r), -u_max, u_max)

            u[actuated_idx[0]] = u_c_r[0, 0]
            u[actuated_idx[1]] = u_c_r[1, 0]
            u[unactuated_idx] = 0 # No control input for actuated axes in this formulation
            u = -1*u
        return u

    next_t_sample : float = 0
    def calc_torque_control_output(self, t, q_curr : np.quaternion,  w_sat : np.array, q_ref : np.quaternion, satellite, w_wheels : np.array, f_est : np.array) -> np.array:
        if t >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return self.u_vec_prev, self.u_wheels_prev
        # q_sat_error = my_utils.get_quaternion_error_Nadafi(q_ref, q_curr)
        # q_sat_error = my_utils.get_quaternion_error_bong_wie(q_ref, q_curr)
        q_sat_error =  q_ref * q_curr.inverse()
        # q_sat_error =  my_utils.get_quaternion_error(q_curr, q_ref)
        # q_sat_error =  q_curr.inverse() * q_ref
        H_wheels_vec = satellite.wheel_module.H_vec # @TODO check this!!!
        if self.type == "pid":
            u = self.calc_pid_torque(q_sat_error, w_sat, satellite.M_inertia, satellite.wheel_module.H_vec, satellite.w_ref)
            if self.type == "adaptive":
                u = self.calc_adaptive_control_torque_output(u, q_curr, q_ref)
            u_vec = u
            u_wheels = satellite.wheel_module.D_psuedo_inv@u_vec

        elif self.type == "backstepping":
            u = self.calc_backstepping_control_torque(q_sat_error, q_curr, q_ref, w_sat, satellite, f_est, t)
            u_wheels = u
            u_vec = satellite.wheel_module.D@u_wheels
        elif self.type == "lyapunov":
            urakubo_controller = UrakuboController(self.config)
            u_vec = urakubo_controller.calc_output(q_sat_error)
            u_wheels = satellite.wheel_module.D_psuedo_inv@u_vec
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

    theta = 0.0
    control_adaptive_model_output = 0.0
    def calc_adaptive_control_torque_output(self, T_com, q_meas : np.quaternion, q_ref : np.quaternion):
        q_model = self.calc_adaptive_model_output(q_ref)
        self.control_adaptive_model_output = my_utils.conv_Rotation_obj_to_euler_axis_angle(my_utils.conv_numpy_to_Rotation_obj_q(q_model))
        q_error = q_meas.inverse()*q_model
        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])
        q_model_vec = np.array([q_model.x, q_model.y, q_model.z])
        
        self.theta = self.theta - q_error_vec*self.adaptive_gain*q_model_vec*self.t_sample
        return T_com + T_com*self.theta

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