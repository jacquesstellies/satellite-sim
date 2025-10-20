import numpy as np
from scipy.spatial.transform import Rotation
import my_utils as my_utils
import my_globals

from body import Body
from controller import Controller
from fault import Fault, FaultModule
from wheels import WheelModule
from orbit import Disturbances
from observer import ObserverModule

class Face():
    r_cop_to_com = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite(Body):

    logger = None
    logging_en = True
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    controller : Controller = None
    observer_module : ObserverModule = None
    wheel_module : WheelModule = None
    fault_module : FaultModule = None
    ref_q : Rotation = None
    ref_T = np.zeros(3)
    ref_q_series : list = None
    ref_t_series : list = None

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    disturbances : Disturbances = None

    config = None
    def __init__(self, wheel_module : WheelModule, controller : Controller, observer_module : ObserverModule,
                 fault_module : FaultModule, wheel_offset = 0, logger = None, logging_en=True,  config=None, results_data=None):
        self.config = config
        self.logging_en = logging_en
        self.wheel_module = wheel_module
        if wheel_module.config == "standard":
            self.wheel_module.wheels[0].position[0] = self.dimensions['x']-config
            self.wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
            self.wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        # if wheel_module.config == "pyramid":
        #     self.wheel_module.wheels[0].position[0] = self.dimensions['x']-wheel_offset
        #     self.wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
        #     self.wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        #     self.wheel_module.wheels[3].position[0] = self.dimensions['x']-wheel_offset
        self.controller = controller
        self.observer_module = observer_module
        self.next_logger_timestamp = controller.t_sample
        self._logger = logger
        self.fault_module = fault_module
        self.results_data = results_data

        for i in my_utils.q_axes:
            self.results_data[f'q_sat_ref_{i}'] = []

        for i in range(wheel_module.num_wheels):
            self.results_data[f'T_ctr_wheels_{i}'] = []

            self.results_data[f'w_wheels_{i}'] = []
            self.results_data[f'T_wheels_{i}'] = []
            self.results_data[f'E_{i}'] = []
            self.results_data[f'f_wheels_{i}'] = []
            self.results_data[f'u_a_{i}'] = []

            self.results_data[f'w_wheels_est_{i}'] = []
            self.results_data[f'f_wheels_est_{i}'] = []
            self.results_data[f'dw_wheels_est_{i}'] = []
            self.results_data[f'E_est_{i}'] = []
        
        self.dimensions, self.mass = config['satellite']['dimensions'], config['satellite']['mass']

        if self.config['satellite']['inertia_override']:
            M_inertia = np.array(config['satellite']['M_Inertia'])
            if M_inertia.shape == (3,3):
                self.M_inertia = M_inertia
            elif M_inertia.shape == (3,):
                self.M_inertia = np.diag(M_inertia)
            else:
                raise Exception("inertia override must be 3x3 or 3x1 matrix")
            self.calc_M_inertia_inv()
        else:
            self.calc_M_inertia()
        self.disturbances = Disturbances()
        self.calc_face_properties()
        self.wheels_control_enable = config['satellite']['wheels_control_enable']
        if not self.wheels_control_enable and self.controller.type == "backstepping" and self.controller.sub_type == "Shen":
            raise Exception("this backstepping controller requires wheels control enabled")

        self.T_ctr_wheels = np.zeros(self.wheel_module.num_wheels)
        self.T_ctr_vec = np.zeros(3)
        
        self.E = np.eye(self.wheel_module.num_wheels)
        self.f_wheels = np.zeros(self.wheel_module.num_wheels)

        # Control Variables
        if config['satellite']['use_ref_euler'] + config['satellite']['use_ref_q'] + config['satellite']['use_ref_series'] != 1:
            raise(Exception("exactly one of use_ref_euler, use_ref_q, use_ref_series must be true"))
        
        if config['satellite']['use_ref_euler']:
            self.ref_q = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True)
        elif config['satellite']['use_ref_q']:
            self.ref_q = Rotation.from_quat(config['satellite']['ref_q'])
        elif config['satellite']['use_ref_series']:
            t_series = config['satellite']['ref_t_series']
            q_series = config['satellite']['ref_q_series']
            if len(t_series) == len(q_series):
                self.ref_q = Rotation.from_quat(q_series[0])
                self.ref_q_series = Rotation.from_quat(q_series)
                self.ref_t_series = t_series
            else:
                raise(Exception("ref_t_series and ref_q_series must be the same length"))
        else:
            raise(Exception("no reference angle commanded"))
        
    def calc_face_properties(self):
        dim_array = np.array([self.dimensions['x'], self.dimensions['y'], self.dimensions['z']])
        for i in range(3):
            for j in range(2):
                face = Face()
                face.norm_vec = np.zeros(3)
                face.norm_vec[i] = 1*(1,-1) [j == 1]
                face.area = 1
                for k, axis in enumerate(face.norm_vec):
                    if axis == 0:
                        face.area *= dim_array[k]
                face.r_cop_to_com = face.norm_vec*0.5*dim_array              
                self.faces.append(face)

    def calc_M_inertia_body(self):

        M_inertia = np.zeros((3,3))
        # use cuboid for mass moment inertia
        M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions['y'],2)+pow(self.dimensions['z'],2))
        M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['z'],2))
        M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['y'],2))
        return M_inertia

    def calc_M_inertia_peri(self):
        M_inertia = np.zeros((3,3))
        if self.wheel_module.wheels is not None:
            M_inertia_indv_wheels = 0
            M_inertia_point_mass_wheels = 0

            for wheel in self.wheel_module.wheels:
                M_inertia_indv_wheels += wheel.M_inertia
                M_inertia_point_mass_wheels += my_utils.calc_M_inertia_point_mass(wheel.position, self.mass)

        return M_inertia

    def calc_M_inertia(self):
        self.M_inertia = self.calc_M_inertia_body() #+ self.calc_M_inertia_peri()

        self.calc_M_inertia_inv()

    t_ref_update = 0
    ref_q_series_index = 0
    def update_ref_q(self, t):
        if self.config['satellite']['use_ref_series'] is False:
            return
        if self.ref_q_series_index > len(self.ref_t_series)-1:
            return
        if t >= self.ref_t_series[self.ref_q_series_index]:
            # print("updating ref q, t=", t, " next t=", self.ref_t_series[self.ref_q_series_index+1], " index=", self.ref_q_series_index)
            self.ref_q = self.ref_q_series[self.ref_q_series_index]
            self.ref_q_series_index += 1
            print("new ref q=", self.ref_q.as_quat())


    wheels_control_enable = True
    next_logger_timestamp = 0.0
    T_ctr_vec = None
    E = None
    f = None
    f_wheels = None
    def calc_state_rates(self, t, y):

        w_sat_input = y[:3]
        q_sat_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        w_wheels_input = y[10:self.wheel_module.num_wheels + 10]

        self.update_ref_q(t)
        ### Calculate controller output
        if self.controller.enable is True:
            if self.config['observer']['feedback_en']:
                f_est = self.wheel_module.D@self.observer_module.f_wheels_est
            else:
                f_est = np.zeros(3)

            self.T_ctr_vec, self.T_ctr_wheels = self.controller.calc_torque_control_output(t, q_sat_input, w_sat_input, my_utils.conv_Rotation_obj_to_numpy_q(self.ref_q), self, w_wheels_input, f_est)

        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)
        
        if self.observer_module.enable is True:
            self.E = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
            self.f_wheels = self.observer_module.f_wheels_est
        else:
            self.f_wheels = self.fault_module.E@self.T_ctr_wheels + self.fault_module.u_a
            self.E = self.fault_module.E
        #### Calculate state rates for satellite various subsystems

        # T_aero = self.disturbances.calc_aero_torque(self, q_sat_input)
        # T_grav = self.disturbances.calc_grav_torque(self, q_sat_input)
        # T_dist = (T_aero + T_grav)

        T_dist = self.disturbances.calc_dist_torque_Shen(t)

        Hnet = self.M_inertia@(w_sat_input) + self.wheel_module.H_vec
        dw_sat_result = self.M_inertia_inv@(self.wheel_module.dH_vec + T_dist - my_utils.cross_product(w_sat_input,Hnet))

        #### Calculate the new satellite body state rates
        inertial_v_q = np.quaternion(0, w_sat_input[0], w_sat_input[1], w_sat_input[2]) # put the inertial velocity in q form

        dq_result = 0.5*q_sat_input*inertial_v_q
        dq_result = [dq_result.x, dq_result.y, dq_result.z, dq_result.w]
        control_power_result = abs(self.wheel_module.dH_vec * w_sat_input) ## @TODO fix this
        
        self.fault_module.update(t)

        if self.logging_en:
            if t >= self.next_logger_timestamp:
                # print("logging data")
                self.results_data['time'].append(t)

                # Collect results data for logging                    
                for i, axis in enumerate(my_utils.xyz_axes):
                    self.results_data['T_sat_'+ axis].append(self.T_ctr_vec[i])
                    self.results_data['dw_sat_' + axis].append(dw_sat_result[i])
                    self.results_data['T_dist_' + axis].append(T_dist[i])
                
                for i, axis in enumerate(my_utils.q_axes):
                    self.results_data['q_sat_ref_' + axis].append(self.ref_q.as_quat()[i])

                for i, wheel in enumerate(self.wheel_module.wheels):
                    self.results_data['w_wheels_' + str(i)].append(self.wheel_module.w_wheels[i])
                    self.results_data['w_wheels_est_' + str(i)].append(self.observer_module.w_wheels_est[i])
                    self.results_data['f_wheels_est_' + str(i)].append(self.observer_module.f_wheels_est[i])
                    self.results_data['T_wheels_' + str(i)].append(self.wheel_module.dw_wheels[i]*wheel.M_inertia_fast) # assuming wheel inertia is diagonal and only z component is used
                    self.results_data['dw_wheels_est_' + str(i)].append(self.observer_module.dw_wheels_est[i])
                    self.results_data['T_ctr_wheels_' + str(i)].append(self.T_ctr_wheels[i])
                    self.results_data['E_' + str(i)].append(self.fault_module.E[i][i])
                    # self.results_data['E_est_' + str(i)].append(self.observer_module.E_mul[i][i])
                    self.results_data['E_est_' + str(i)].append(self.E[i][i])
                    self.results_data['f_wheels_' + str(i)].append(self.f_wheels[i])
                    self.results_data['u_a_' + str(i)].append(self.fault_module.u_a[i])

                self.next_logger_timestamp += self.controller.t_sample

        return np.hstack([dw_sat_result, dq_result, control_power_result, self.wheel_module.dw_wheels])