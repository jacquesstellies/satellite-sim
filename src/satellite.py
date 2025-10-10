import numpy as np
from scipy.spatial.transform import Rotation
import my_utils as my_utils
import my_globals

from body import Body
from controller import Controller
from fault import Fault
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
    fault : Fault = None

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    disturbances : Disturbances = None

    config = None
    def __init__(self, wheel_module : WheelModule, controller : Controller, observer_module : ObserverModule,
                 fault : Fault, wheel_offset = 0, logger = None, logging_en=True,  config=None, results_data=None):
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
        self.fault = fault
        self. results_data = results_data
        
        for i in range(wheel_module.num_wheels):
            self.results_data['w_wheels_' + str(i)] = []
            self.results_data['T_wheels_' + str(i)] = []
            self.results_data[f'E_{i}'] = []
        
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

    ref_q = Rotation.from_quat([0,0,0,1])
    ref_T = np.zeros(3)
    wheels_control_enable = True
    next_logger_timestamp = 0.0
    T_ctr_vec = None
    E = None
    def calc_state_rates(self, t, y):

        w_sat_input = y[:3]
        q_sat_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        w_wheels_input = y[10:self.wheel_module.num_wheels + 10]

        ### Calculate controller output
        if self.controller.enable is True:
            self.T_ctr_vec, self.T_ctr_wheels = self.controller.calc_torque_control_output(t, q_sat_input, w_sat_input, self.ref_q, self, w_wheels_input, self.E)

        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)

        
        if self.observer_module.enable is True:
            self.E = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
        else:
            self.E = np.eye(self.wheel_module.num_wheels)



        #### Calculate state rates for satellite various subsystems

        # T_aero = self.disturbances.calc_aero_torque(self, q_sat_input)
        # T_grav = self.disturbances.calc_grav_torque(self, q_sat_input)
        # T_dist = (T_aero + T_grav)

        # if self.config['controller']['type'] == "adaptive" and self.config['controller']['sub_type'] == "Shen":

        T_dist = self.disturbances.calc_dist_torque_Shen(t)

        Hnet = self.M_inertia@(w_sat_input) + self.wheel_module.H_vec
        dw_sat_result = self.M_inertia_inv@(self.wheel_module.dH_vec + T_dist - my_utils.cross_product(w_sat_input,Hnet))

        #### Calculate the new satellite body state rates
        inertial_v_q = np.quaternion(0, w_sat_input[0], w_sat_input[1], w_sat_input[2]) # put the inertial velocity in q form

        dq_result = 0.5*q_sat_input*inertial_v_q
        dq_result = [dq_result.x, dq_result.y, dq_result.z, dq_result.w]
        control_power_result = abs(self.wheel_module.dH_vec * w_sat_input) ## @TODO fix this
        self.fault.update(t)

        if self.logging_en:
            if t >= self.next_logger_timestamp:
                # print("logging data")
                self.results_data['time'].append(t)

                # Collect results data for logging                    
                for i, axis in enumerate(my_utils.xyz_axes):
                    self.results_data['torque_'+ axis].append(self.T_ctr_vec[i])
                    self.results_data['angular_acc_'+ axis].append(dw_sat_result[i])
                    self.results_data['T_dist_' + axis].append(T_dist[i])

                for i, wheel in enumerate(self.wheel_module.wheels):
                    self.results_data['w_wheels_' + str(i)].append(self.wheel_module.w_wheels[i])
                    self.results_data['T_wheels_' + str(i)].append(self.wheel_module.dw_wheels[i]*wheel.M_inertia_fast) # assuming wheel inertia is diagonal and only z component is used
                    self.results_data['w_wheels_est_' + str(i)].append(self.observer_module.w_wheels_est[i])
                    self.results_data['f_wheels_est_' + str(i)].append(self.observer_module.f_wheels_est[i])
                    self.results_data['dw_wheels_est_' + str(i)].append(self.observer_module.dw_wheels_est[i])
                    self.results_data['E_' + str(i)].append(self.fault.E[i][i])
                    self.results_data['E_est_' + str(i)].append(self.observer_module.E_mul[i][i])

                self.next_logger_timestamp += self.controller.t_sample

        return np.hstack([dw_sat_result, dq_result, control_power_result, self.wheel_module.dw_wheels])