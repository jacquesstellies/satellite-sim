import numpy as np
from scipy.spatial.transform import Rotation
from magt import MagtModule
import my_utils as my_utils
import my_globals

from body import Body
from controller import Controller
from fault import Fault, FaultModule
from wheels import WheelModule
from orbit import Disturbances, Orbit
from observer import ObserverModule

class DivergentRate(Exception):
    pass
class Face():
    r_cop_to_com = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite(Body):
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    controller : Controller = None
    observer_module : ObserverModule = None
    wheel_module : WheelModule = None
    fault_module : FaultModule = None
    magt_module : MagtModule = None
    orbit : Orbit = None
    logger = None

    q_ref : np.quaternion = np.quaternion(1,0,0,0)
    ref_T = np.zeros(3)
    q_ref_series : list = None
    t_ref_series : list = None

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    disturbances : Disturbances = None
    mode: str = "init"
    modes: list[str] = ["init", "ref_pointing", "nominal_day", "nominal_night", "safe"]

    q = np.quaternion(1,0,0,0)
    w = np.zeros(3)
    dw = np.zeros(3)
    H = np.zeros(3)
    dH = np.zeros(3)

    fd_w_max = 2*np.pi/180 # rad/s

    config = None
    def __init__(self, wheel_module : WheelModule, controller : Controller, observer_module : ObserverModule,
                 fault_module : FaultModule, magt_module : MagtModule, logger, wheel_offset = 0, orbit: Orbit = None, config=None):
        self.mode = config['satellite']['mode']
        
        self.config = config
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
        self.logger = logger
        self.magt_module = magt_module
        self.orbit = orbit

        self.fault_module = fault_module
        
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
        self.disturbances = Disturbances(self.orbit, config)
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
        if self.mode == "ref_pointing":
            if config['satellite']['use_ref_euler']:
                q_ref_array = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True).as_quat()
                self.q_ref = np.quaternion(q_ref_array[3], q_ref_array[0], q_ref_array[1], q_ref_array[2]) # check if valid quaternion
            elif config['satellite']['use_ref_q']:
                self.q_ref = np.quaternion(config['satellite']['ref_q'][3], config['satellite']['ref_q'][0], config['satellite']['ref_q'][1], config['satellite']['ref_q'][2]) # check if valid quaternion
                # self.q_ref = Rotation.from_quat(config['satellite']['ref_q'])
            elif config['satellite']['use_ref_series']:
                t_ref_series = config['satellite']['ref_t_series']
                q_series = config['satellite']['ref_q_series']
                if len(t_ref_series) == len(q_series):
                    # self.q_ref = Rotation.from_quat(q_series[0]).as_quat()
                    # self.q_ref_series = Rotation.from_quat(q_series)
                    for i in range(len(t_ref_series)):
                        self.q_ref_series.append(np.quaternion(q_series[i][3], q_series[i][0], q_series[i][1], q_series[i][2])) # check if valid quaternion
                    self.t_ref_series = t_ref_series
                else:
                    raise(Exception("t_ref_series and ref_q_series must be the same length"))
            else:
                raise(Exception("no reference angle commanded"))
        else:
            self.q_ref = np.quaternion(1,0,0,0)
        
        self.fd_w_max = config['FDIR']['satellite']['w_max_dps'] * my_utils.DEG_TO_RAD

        
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

    mode_latch = False
    def update_mode(self):
        mode_prev = self.mode
        if self.mode == "ref_pointing":
            return
        # if self.orbit.eclipse:
        #     self.mode = "nominal_night"
        # else:
        #     self.mode = "nominal_day"
        if self.init == True:
            print("MODE STARTUP \t ", self.mode)
        if self.mode != mode_prev:
            print("MODE SWITCH \t ", mode_prev, " -> ", self.mode)

    t_ref_update = 0
    q_ref_series_index = 0
    init = True
    def update_ref_q(self, t):
        if self.init and self.mode != "ref_pointing":
            self.init = False
            return
        if self.mode == "nominal_day":
            # nP = np.array([1,0,0]) # body frame x axis
            # qBI_I = self.qBI
            # theta_BS = np.arccos(nP @ self.orbit.nSI_I)
            # nRS = np.cross(nSI_I, nP)
            #  = self.orbit.nSI_I*np.sin()
            # qBI = self.q

            x_axis = self.orbit.nSB_I
            z_axis = self.orbit.nEB_I
            y_axis = my_utils.cross_product_M31M31(self.orbit.nEB_I, x_axis)
        
            self.q_ref = my_utils.conv_numpy_to_Rotation_obj_q(Rotation.from_matrix(np.array([x_axis, y_axis, z_axis]).T))

            # self.q_ref = Rotation.from_matrix(r_matrix).as_quat()

        if self.mode == "nominal_night":
            self.q_ref = my_utils.conv_Rotation_obj_to_numpy_q(Rotation.from_matrix(self.orbit.TOIk))
            # self.w_ref = self.orbit.TBO_B, -self.orbit.v_norm*self.orbit.radius

        if self.mode == "ref_pointing":
            if self.config['satellite']['use_ref_series'] is False:
                return
            if self.q_ref_series_index > len(self.t_ref_series)-1:
                return
            if t >= self.t_ref_series[self.q_ref_series_index]:
                # print("updating ref q, t=", t, " next t=", self.t_ref_series[self.q_ref_series_index+1], " index=", self.q_ref_series_index)
                self.q_ref = self.q_ref_series[self.q_ref_series_index]
                self.q_ref_series_index += 1
                print("new ref q=", print(self.q_ref), " at t=", t)
    
    def calc_delta_M_inertia(self, t):
        return np.diag(np.array([2*np.sin(0.1*t), 2.8*np.sin(0.2*t), 3.6*np.sin(0.3*t)]))

    wheels_control_enable = True
    T_ctr_vec = None
    T_dist = np.zeros(0)
    E = None
    f = None
    f_wheels = None
    def calc_state_rates(self, t, y):

        self.w = np.array(y[:3])
        self.q = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        w_wheels_input = y[10:self.wheel_module.num_wheels + 10]
        dcm = Rotation.from_quat([self.q.x, self.q.y, self.q.z, self.q.w]).as_matrix()

        self.update_mode()
        self.update_ref_q(t)
        ### Calculate controller output
        if self.controller.enable is True:
            if self.config['observer']['feedback_en']:
                f_est = self.wheel_module.D@self.observer_module.f_wheels_est
            else:
                f_est = np.zeros(3)

            self.T_ctr_vec, self.T_ctr_wheels = self.controller.calc_torque_control_output(t, self.q, self.w, self.q_ref, self, w_wheels_input, f_est)

        if self.magt_module.enable is True:
            q_sat_err =  self.q.inverse() * self.q_ref
            q_err_vec = np.array([q_sat_err.x, q_sat_err.y, q_sat_err.z])
            self.magt_module.calc_torque(q_err_vec, self.w, self.M_inertia@(self.w) + self.wheel_module.H_vec, t)

        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)
        
        if self.observer_module.enable is True:
            self.E = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
            self.f_wheels = self.observer_module.f_wheels_est
        else:
            self.f_wheels = self.fault_module.E@self.T_ctr_wheels + self.fault_module.u_a
            self.E = self.fault_module.E
        #### Calculate state rates for satellite various subsystems

        self.orbit.calc_orbit_state(t)

        self.T_dist = self.disturbances.calc_torque(self, dcm, t)
        # delta_M_inertia = self.calc_delta_M_inertia(t)
        # delta_M_inertia = np.zeros((3,3))
        # M_inertia_effective = self.M_inertia + delta_M_inertia
        # M_inertia_effective_inv = np.linalg.inv(M_inertia_effective)

        M_inertia_effective_inv = self.M_inertia_inv

        
        Hnet = self.M_inertia@(self.w) + self.wheel_module.H_vec
        self.dw = (M_inertia_effective_inv)@(self.wheel_module.dH_vec + self.T_dist - my_utils.cross_product_M31M31(self.w,Hnet)) + self.magt_module.T
        # dw_sat_result = (self.M_inertia_inv)@(self.wheel_module.dH_vec + T_dist - my_utils.cross_product_M31M31(elf.w,Hnet)) + T_magt

        #### Calculate the new satellite body state rates
        inertial_v_q = np.quaternion(0, self.w[0], self.w[1], self.w[2]) # put the inertial velocity in q form

        dq = 0.5*self.q*inertial_v_q
        dq = [dq.x, dq.y, dq.z, dq.w]
        control_power = abs(self.wheel_module.dH_vec * self.w) ## @TODO fix this
        
        self.fault_module.update(t)
        self.update_FDIR(t)

        self.logger.log_data(t)

        return np.hstack([self.dw, dq, control_power, self.wheel_module.dw_wheels])

    
    def update_fd(self, t):
        # print("angular")
        if not self.config['FDIR']['satellite']['enable']:
            return
        if my_utils.magnitude(self.w) > self.fd_w_max:
            raise DivergentRate(f"Divergent rate detected, w_sat = {self.w*my_utils.RAD_TO_DEG} at time {t}")
        
    def update_FDIR(self, t):
        
        self.update_fd(t)

        for wheel in self.wheel_module.wheels:
            wheel.update_fd(t)
        