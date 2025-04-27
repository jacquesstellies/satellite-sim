import numpy as np
from scipy.spatial.transform import Rotation
import my_utils as my_utils
import my_globals

from body import Body
from controller import Controller
from fault import Fault
from wheels import WheelModule
from orbit import Disturbances

results_data = my_globals.results_data

class Face():
    r_cop_to_com = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite(Body):

    _logger = None
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    _controller : Controller = None
    _wheel_module : WheelModule = None
    _fault : Fault = None

    mode : str = ""
    modes = ["direction", "torque"]

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    _disturbances : Disturbances = None

    config = None
    def __init__(self, _wheel_module : WheelModule, controller : Controller, 
                 fault : Fault, wheel_offset = 0, logger = None, config=None):
        self.config = config
        self._wheel_module = _wheel_module
        if _wheel_module.config == "standard":
            self._wheel_module.wheels[0].position[0] = self.dimensions['x']-config
            self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
            self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        # if _wheel_module.config == "pyramid":
        #     self._wheel_module.wheels[0].position[0] = self.dimensions['x']-wheel_offset
        #     self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
        #     self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        #     self._wheel_module.wheels[3].position[0] = self.dimensions['x']-wheel_offset
        self._controller = controller
        self._next_control_time_step = controller.time_step
        self._logger = logger
        self._fault = fault
        self.mode = config['satellite']['mode']
        if self.mode not in self.modes:
            raise Exception(f"mode {self.mode} not available")
        
        self.dimensions, self.mass = config['satellite']['dimensions'], config['satellite']['mass']

        if self.config['satellite']['inertia_override']:
            M_inertia = np.array(config['satellite']['M_Inertia'])
            if M_inertia.shape == (3,3):
                self.M_inertia = M_inertia
            elif M_inertia.shape == (3,):
                self.M_inertia = np.diag(M_inertia)
            else:
                raise Exception("inertia override must be 3x3 or 3x1 matrix")
            print("Satellite inertia ",self.M_inertia)
            self.calc_M_inertia_inv()
        else:
            self.calc_M_inertia()
        self._disturbances = Disturbances()
        self.calc_face_properties()
        self.ref_T = np.array(config['satellite']['ref_T'])
        self.wheels_control_enable = config['satellite']['wheels_control_enable']
        if not self.wheels_control_enable and self._controller.type == "backstepping" and self._controller.sub_type == "Shen":
            raise Exception("this backstepping controller requires wheels control enabled")

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
        if self._wheel_module.wheels is not None:
            M_inertia_indv_wheels = 0
            M_inertia_point_mass_wheels = 0

            for wheel in self._wheel_module.wheels:
                M_inertia_indv_wheels += wheel.M_inertia
                M_inertia_point_mass_wheels += my_utils.calc_M_inertia_point_mass(wheel.position, self.mass)

        return M_inertia

    def calc_M_inertia(self):
        self.M_inertia = self.calc_M_inertia_body() #+ self.calc_M_inertia_peri()

        self.calc_M_inertia_inv()

    ref_q = Rotation.from_quat([0,0,0,1])
    ref_T = np.zeros(3)
    wheels_control_enable = True
    _next_control_time_step = 0.1
    wheels_H_rate_result = np.zeros(3)
    wheels_H_result = np.zeros(3)
    wheels_speed = np.zeros(3)
    T_controller_com = np.zeros(3)
    def calc_state_rates(self, t, y):

        sat_angular_v_input = y[:3]
        q_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        sat_angular_acc_result = [0]*3
        q_rate_result = np.quaternion(1,0,0,0)
        if self._controller.enable is True:            
            if t == 0.0:
                self.T_controller_com = self._controller.calc_torque_control_output(q_input, sat_angular_v_input, self.ref_q, self, self._wheel_module.H, t)
            if t > self._next_control_time_step:
                if self.mode == "direction":
                    self.T_controller_com = self._controller.calc_torque_control_output(q_input, sat_angular_v_input, self.ref_q, self, self._wheel_module.H, t)
                elif self.mode == "torque":
                    self.T_controller_com = self.ref_T
                self._next_control_time_step += self._controller.time_step
                if self.wheels_control_enable:
                    self.wheels_H_rate_result, self.wheels_H_result = self._wheel_module.calc_state_rates(self.T_controller_com, self._controller.time_step)

            if t > self._fault.time and self._fault.master_enable:
                self._fault.enabled = True
        if self.wheels_control_enable:
            T_controller = self.wheels_H_rate_result
        else:
            T_controller = self.T_controller_com
        T_aero = self._disturbances.calc_aero_torque(self, q_input)
        T_grav = self._disturbances.calc_grav_torque(self, q_input)
        T_dist = (T_aero + T_grav)

        if self.config['controller']['type'] == "adaptive" and self.config['controller']['sub_type'] == "Shen":
            T_dist = self._disturbances.calc_dist_torque_Shen(t)

        Hnet = self.M_inertia@(sat_angular_v_input) + self.wheels_H_result
        sat_angular_acc_result = self.M_inertia_inv@(T_controller + T_dist - np.cross(sat_angular_v_input,Hnet))

        # put the inertial velocity in q form
        inertial_v_q = np.quaternion(0, sat_angular_v_input[0], sat_angular_v_input[1], sat_angular_v_input[2])

        q_rate_result = 0.5*q_input*inertial_v_q
        q_rate_result = [q_rate_result.x, q_rate_result.y, q_rate_result.z, q_rate_result.w]
        results_data['time'].append(t)

        for i, axis in enumerate(my_utils.xyz_axes):
            results_data['torque_'+ axis].append(T_controller[i])
            results_data['angular_acc_'+ axis].append(sat_angular_acc_result[i])
            results_data['T_dist_' + axis].append(T_dist[i])
            
        for i, wheel in enumerate(self._wheel_module.wheels):
            results_data['wheel_speed_' + str(i)].append(wheel.speed)
            results_data['wheel_torque_' + str(i)].append(wheel.T)

        if self.wheels_control_enable:
            control_power_result = np.abs(self.wheels_H_rate_result * sat_angular_v_input)
        else:
            control_power_result = np.abs(self.T_controller_com * sat_angular_v_input)


        results = [sat_angular_acc_result, q_rate_result, control_power_result]
        return np.hstack(results)