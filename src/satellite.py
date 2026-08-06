import numpy as np
from scipy.spatial.transform import Rotation
from magt import MagtModule
import my_utils as my_utils
import my_globals

from controller import Controller
from fault import Fault, FaultModule
from wheels import WheelModule
from orbit import Disturbances, Orbit
from observer import ObserverModule, FNDOFaultDetector

class DivergentRate(Exception):
    pass
class Face():
    r_com_to_cop = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite():
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))

    controller : Controller = None
    observer_module : ObserverModule = None
    wheel_module : WheelModule = None
    fault_module : FaultModule = None
    magt_module : MagtModule = None
    orbit : Orbit = None
    logger = None

    q_RI : np.quaternion = np.quaternion(1,0,0,0)
    ref_T = np.zeros(3)
    q_RI_series : list = None
    t_ref_series : list = None
    _tracking_keys_cache = None
    w_RI_R = np.zeros(3)
    dw_RI_R = np.zeros(3)
    sinusoidal_ref_amplitude = np.zeros(3)
    sinusoidal_ref_frequency = np.zeros(3)

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    disturbances : Disturbances = None
    mode: str = "init"
    modes: list[str] = ["init", "ref_pointing", "nominal_day", "nominal_night", "safe"]

    q_BI = np.quaternion(1,0,0,0)
    w_BI_B = np.zeros(3)
    dw = np.zeros(3)
    H = np.zeros(3)
    H_total = np.zeros(3)
    dH = np.zeros(3)

    fd_w_max = 2*np.pi/180 # rad/s

    config = None
    def __init__(self, wheel_module : WheelModule, controller : Controller, observer_module : ObserverModule,
                 fault_module : FaultModule, magt_module : MagtModule, logger, orbit: Orbit = None, config=None):
        self.mode = config['satellite']['mode']
        
        self.config = config
        self.wheel_module = wheel_module
        self.controller = controller
        self.observer_module = observer_module
        self.logger = logger
        self.magt_module = magt_module
        self.orbit = orbit

        self.fault_module = fault_module
        
        self.dimensions, self.mass = config['satellite']['dimensions'], config['satellite']['mass']

        M_inertia = np.array(config['satellite']['M_Inertia'])
        if M_inertia.shape == (3,3):
            self.M_inertia = M_inertia
        elif M_inertia.shape == (3,):
            self.M_inertia = np.diag(M_inertia)
        else:
            raise Exception("inertia override must be 3x3 or 3x1 matrix")
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)

        self.delta_inertia_frac = config['satellite'].get('delta_inertia_frac', 0.0)
        self.delta_inertia_freq = np.array(config['satellite'].get('delta_inertia_freq', [0.1, 0.2, 0.3]))
        self.disturbances = Disturbances(self.orbit, config)
        self.calc_face_properties()
        self.wheels_control_enable = config['satellite']['wheels_control_enable']
        if not self.wheels_control_enable and self.controller.type == "backstepping" and self.controller.sub_type == "Shen":
            raise Exception("this backstepping controller requires wheels control enabled")

        self.T_ctr_wheels = np.zeros(self.wheel_module.num_wheels)
        self.T_ctr_vec = np.zeros(3)
        
        self.w_BI_B = np.array(config['satellite']['w_init_dps']) * my_utils.DEG_TO_RAD
        assert len(self.w_BI_B) == 3, "w_init_dps must be a 3 element array"
        self.H = self.M_inertia@self.w_BI_B
        self.H_total = self.H + self.wheel_module.H_vec

        self.E = np.eye(self.wheel_module.num_wheels)
        self.f_wheels = np.zeros(self.wheel_module.num_wheels)

        detection_config = config.get('detection', None)
        if detection_config is not None and detection_config.get('enable', False):
            self.fault_detector = FNDOFaultDetector(config, self.M_inertia, self.wheel_module)
        else:
            self.fault_detector = None

        # Control Variables
        if not self.mode.startswith("nominal"):
            if config['satellite']['use_ref_euler'] + config['satellite']['use_ref_q'] + config['satellite']['use_ref_series'] != 1:
                raise(Exception("exactly one of use_ref_euler, use_ref_q, use_ref_series must be true"))
        if self.mode == "ref_pointing":
            if config['satellite']['use_ref_euler']:
                q_ref_array = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True).as_quat()
                self.q_RI = np.quaternion(q_ref_array[3], q_ref_array[0], q_ref_array[1], q_ref_array[2]) # check if valid quaternion
            elif config['satellite']['use_ref_q']:
                self.q_RI = np.quaternion(config['satellite']['ref_q'][3], config['satellite']['ref_q'][0], config['satellite']['ref_q'][1], config['satellite']['ref_q'][2]) # check if valid quaternion
                # self.q_RI = Rotation.from_quat(config['satellite']['ref_q'])
            elif config['satellite']['use_ref_series']:
                t_ref_series = config['satellite']['ref_t_series']
                q_series = config['satellite']['ref_q_series']
                if len(t_ref_series) == len(q_series):
                    self.q_RI = Rotation.from_quat(q_series[0]).as_quat()
                    self.q_RI_series = []
                    # self.q_RI_series = Rotation.from_quat(q_series)
                    for i in range(len(t_ref_series)):
                        self.q_RI_series.append(np.quaternion(q_series[i][3], q_series[i][0], q_series[i][1], q_series[i][2])) # check if valid quaternion
                    self.t_ref_series = t_ref_series
                else:
                    raise(Exception("t_ref_series and ref_q_series must be the same length, got t_series {} and q_series {}".format(len(t_ref_series), len(q_series))))

                if config['simulation']['duration'] < self.t_ref_series[-1]:
                    raise(Exception("simulation duration must be longer than last time in t_ref_series"))
            else:
                raise(Exception("no reference angle commanded"))
            
        else:
            self.q_RI = np.quaternion(1,0,0,0)
        
        self.fd_w_max = config['FDIR']['satellite']['w_max_dps'] * my_utils.DEG_TO_RAD

        self.next_t_ref_update_interval = config['controller']['t_sample']

        if self.mode == "sinusoidal":
            self.sinusoidal_ref_amplitude = np.array(config['satellite']['sinusoidal_ref_amplitude_deg']) * my_utils.DEG_TO_RAD
            self.sinusoidal_ref_frequency = np.array(config['satellite']['sinusoidal_ref_frequency'])
        
    def calc_face_properties(self):
        model = self.config['satellite']['dimensions']['model']
        if model == "rectangular":
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
                    face.r_com_to_cop = face.norm_vec*0.5*dim_array              
                    self.faces.append(face)
                    print(face.norm_vec, face.area, face.r_com_to_cop)
        elif model == "EOSSAT":
            com_to_cop_array = [[0.0, 0.0, 0.995],
                                [0.0, 0.0, 0.026],
                                [0.198, 0.348, 0.514],
                                [0.198, -0.348, 0.514],
                                [-0.33, 0.0, 0.514],
                                [-0.165, 0.459, 0.514],
                                [-0.165, -0.459, 0.514]
                                ]

            norms_array = [[0, 0, 1], 
                           [0, 0, -1],
                           [0.869, 0.495, 0], # (Solar panel)
                           [0.869, -0.495, 0], # (Solar panel)
                           [-1, 0, 0],
                           [-0.82, 0.572, 0],
                           [-0.82, -0.572, 0]]
            
            areas = [0.579, 0.579, 0.763, 0.763, 0.415, 0.519, 0.519]
            for i in range(len(com_to_cop_array)):
                face = Face()
                face.r_com_to_cop = np.array(com_to_cop_array[i])
                face.norm_vec = np.array(norms_array[i])
                face.area = areas[i]
                self.faces.append(face)

    def calc_M_inertia_body(self):

        M_inertia = np.zeros((3,3))
        # use cuboid for mass moment inertia
        M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions['y'],2)+pow(self.dimensions['z'],2))
        M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['z'],2))
        M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['y'],2))
        return M_inertia

    def calc_M_inertia(self):
        self.M_inertia = self.calc_M_inertia_body()
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)

    def update_mode(self):
        mode_prev = self.mode
        if self.init == True:
            if self.config['simulation']['verbose']:
                print("MODE STARTUP \t ", self.mode)
        if self.mode == "ref_pointing":
            return
        # if self.orbit.eclipse:
        #     self.mode = "nominal_night"
        # else:
        #     self.mode = "nominal_day"

        if self.mode != mode_prev:
            print("MODE SWITCH \t ", mode_prev, " -> ", self.mode)

    t_ref_update = 0
    q_RI_series_index = 0
    init = True
    next_t_ref_update = 0.0
    next_t_ref_update_interval = None
    def update_ref_q(self, t):
        if t >= self.next_t_ref_update:
            self.next_t_ref_update = t + self.next_t_ref_update_interval
        else:
            return
        self.update_mode()

        if self.init and self.mode not in ("ref_pointing", "tracking"):
            self.w_RI_R = np.zeros(3)
            self.init = False
            return
        if self.mode == "nominal_day":
            # nP = np.array([1,0,0]) # body frame x axis
            # qBI_I = self.q_BI
            # theta_BS = np.arccos(nP @ self.orbit.nSI_I)
            # nRS = np.cross(nSI_I, nP)
            #  = self.orbit.nSI_I*np.sin()
            # qBI = self.q_BI

            # Reference (R) frame axes expressed in inertial (I): x=sun, z=nadir, y=z x x.
            # Rows of a passive DCM T_RI are the R-axes in I coords (v_R = T_RI v_I).
            x_axis = self.orbit.nSB_I
            z_axis = self.orbit.nIB_I
            y_axis = my_utils.cross_product_M31M31(self.orbit.nIB_I, x_axis)
            T_RI = np.row_stack([x_axis, y_axis, z_axis])
            self.q_RI = my_utils.dcm_to_quat(T_RI)
            self.w_RI_R = np.zeros(3)
            self.dw_RI_R = np.zeros(3)

        if self.mode == "nominal_night":
            # Nadir/LVLH pointing: reference frame R == orbit frame O, so T_RI = T_OI.
            self.q_RI = my_utils.dcm_to_quat(self.orbit.T_OI)
            # Orbital angular velocity w_OI is r x v / |r|^2 in inertial coords; the
            # controller feedforward needs it resolved in the reference (orbit) frame.
            wOI_I = my_utils.cross_product_M31M31(self.orbit.sBI_I, self.orbit.DIsBI_I) / np.linalg.norm(self.orbit.sBI_I)**2
            self.w_RI_R = self.orbit.T_OI @ wOI_I
            self.dw_RI_R = np.zeros(3)

        if self.mode == "ref_pointing":
            if self.init == True:
                self.init = False
            if self.config['satellite']['use_ref_series'] is False:
                return
            if self.q_RI_series_index > len(self.t_ref_series)-1:
                return
            if t >= self.t_ref_series[self.q_RI_series_index]:
                # print("updating ref q, t=", t, " next t=", self.t_ref_series[self.q_RI_series_index+1], " index=", self.q_RI_series_index)
                self.q_RI = self.q_RI_series[self.q_RI_series_index]
                self.q_RI_series_index += 1
                print("new ref q=", print(self.q_RI), " at t=", t)
        
        if self.mode == "sinusoidal":
            # freq = self.sinusoidal_ref_frequency
            # amp = self.sinusoidal_ref_amplitude  # already in radians (see __init__)

            # def _R_ref_at(tau):
            #     return Rotation.from_euler("xyz", [amp[0]*np.sin(freq[0]*tau),
            #                                        amp[1]*np.sin(freq[1]*tau),
            #                                        amp[2]*np.sin(freq[2]*tau)], degrees=False)

            # # Reference body angular velocity / acceleration via central finite differences.
            # # w_ref is the rate of the reference frame expressed in the reference frame, which
            # # is what the controller feedforward (C@w_d, -C_r@dw_d) expects.
            # dt = 1e-3
            # R0 = _R_ref_at(t)
            # R_fwd = (R0.inv() * _R_ref_at(t + dt)).as_rotvec() / dt
            # R_bwd = (_R_ref_at(t - dt).inv() * R0).as_rotvec() / dt
            # self.q_RI = my_utils.conv_Rotation_obj_to_numpy_q(R0)
            # self.w_RI_R = 0.5 * (R_fwd + R_bwd)
            # self.dw_RI_R = (R_fwd - R_bwd) / dt

            freq = self.sinusoidal_ref_frequency
            amp  = self.sinusoidal_ref_amplitude          # radians (see __init__)

            # Fixed axis in the body x-y plane; oscillate the angle only.
            # rotvec(t) = [amp_x, amp_y, 0] * sin(w t)  -> axis is constant, so
            #   q_ref has q1,q2 components and q3 = 0
            #   w_ref = d/dt(rotvec) is parallel to that axis -> omega_z = 0  (trackable)
            s   =  np.sin(freq[0] * t)
            c   =  freq[0]      * np.cos(freq[0] * t)
            cdd = -freq[0]**2   * np.sin(freq[0] * t)
            axis = np.array([amp[0], amp[1], 0.0])         # direction = axis, magnitude = angle amplitude

            self.q_RI  = my_utils.conv_Rotation_obj_to_numpy_q(Rotation.from_rotvec(axis * s))
            self.w_RI_R  = axis * c                          # exact body rate, z-component = 0
            self.dw_RI_R = axis * cdd                        # exact angular acceleration

        if self.mode == "tracking":
            # Zarourati-style snapshot-imaging maneuver: smoothly slew (SLERP with
            # smootherstep timing) through the ref_q_series waypoints, settling at rest at
            # each one. Reference body rate / acceleration come from central finite
            # differences so the controller feedforward (w_ref, dw_ref) stays consistent.
            # Starts at ref_q_series[0]; set euler_init to match it for zero initial error.
            self.init = False
            dt = 1e-3
            R0 = self._tracking_q_ref_at(t)
            R_fwd = (R0.inv() * self._tracking_q_ref_at(t + dt)).as_rotvec() / dt
            R_bwd = (self._tracking_q_ref_at(t - dt).inv() * R0).as_rotvec() / dt
            self.q_RI  = my_utils.conv_Rotation_obj_to_numpy_q(R0)
            self.w_RI_R  = 0.5 * (R_fwd + R_bwd)
            self.dw_RI_R = (R_fwd - R_bwd) / dt

    def _tracking_keys(self):
        if self._tracking_keys_cache is None:
            q_series = self.config['satellite']['ref_q_series']   # each [x, y, z, w]
            t_series = self.config['satellite']['ref_t_series']
            rots = [Rotation.from_quat(q) for q in q_series]
            self._tracking_keys_cache = (rots, np.array(t_series, dtype=float))
        return self._tracking_keys_cache

    def _tracking_q_ref_at(self, tau):
        rots, times = self._tracking_keys()
        if tau <= times[0]:
            return rots[0]
        if tau >= times[-1]:
            return rots[-1]
        i = int(np.searchsorted(times, tau) - 1)
        i = max(0, min(i, len(times) - 2))
        s = (tau - times[i]) / (times[i + 1] - times[i])
        s = s*s*s*(s*(s*6 - 15) + 10)          # smootherstep: C2, zero rate/accel at waypoints
        rel = (rots[i].inv() * rots[i + 1]).as_rotvec()
        return rots[i] * Rotation.from_rotvec(s * rel)

    def calc_delta_M_inertia(self, t):
        # Sinusoidal parametric uncertainty Delta J = frac * diag(J) .* sin(freq*t), applied
        # per axis. frac is set from the config so it scales with the plant: the old hard-coded
        # amplitudes [2, 2.8, 3.6] (= 0.8 * diag(J) of the Nadafi plant) drove the effective
        # inertia negative on smaller satellites (e.g. Zarourati's J = diag(0.92, 0.92, 0.44)),
        # which makes the plant unconditionally unstable. frac must stay < 1.
        if self.delta_inertia_frac == 0.0:
            return np.zeros((3,3))
        return self.delta_inertia_frac * np.diag(np.diag(self.M_inertia) * np.sin(self.delta_inertia_freq * t))

    wheels_control_enable = True
    T_ctr_vec = None
    T_dist = np.zeros(0)
    E = None
    f = None
    f_wheels = None
    def calc_state_rates(self, t, y):

        self.w_BI_B = np.array(y[:3])
        self.q_BI = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        w_wheels_input = y[10:self.wheel_module.num_wheels + 10]
        # Passive body DCM T_BI: v_B = T_BI @ v_I  (used by the disturbance models).
        T_BI = my_utils.quat_to_dcm(self.q_BI)

        self.update_ref_q(t)
        ### Calculate controller output
        if self.controller.enable is True:
            if self.config['observer']['feedback_en']:
                f_est = self.wheel_module.D@self.observer_module.f_wheels_est
            else:
                f_est = np.zeros(3)

            self.T_ctr_vec, self.T_ctr_wheels = self.controller.calc_torque_control_output(t, self.q_BI, self.w_BI_B, self.q_RI, self, w_wheels_input, f_est)

        # Body-frame attitude error q_RB (rotation B->R); vector part resolved in B.
        q_RB =  my_utils.quat_error(self.q_RI, self.q_BI)
        qv_RB = np.array([q_RB.x, q_RB.y, q_RB.z])
        
        self.magt_module.calc_torque(qv_RB, self.w_BI_B, self.H + self.wheel_module.H_vec, t)
        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)
        
        if self.observer_module.enable is True:
            self.E = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
            self.f_wheels = self.observer_module.f_wheels_est
        else:
            self.f_wheels = self.fault_module.E@self.T_ctr_wheels + self.fault_module.u_a
            self.E = self.fault_module.E
        #### Calculate state rates for satellite various subsystems

        self.orbit.calc_orbit_state(t)

        self.T_dist = self.disturbances.calc_torque(self, T_BI, t)
        
        delta_M_inertia = self.calc_delta_M_inertia(t)
        M_inertia_effective = self.M_inertia + delta_M_inertia
        M_inertia_effective_inv = np.linalg.inv(M_inertia_effective)

        # M_inertia_effective_inv = self.M_inertia_inv

        self.H = self.M_inertia@(self.w_BI_B)
        self.H_total = self.H + self.wheel_module.H_vec
        self.dw_BI_B = (M_inertia_effective_inv)@(-1*self.wheel_module.dH_vec + self.T_dist - my_utils.cross_product_M31M31(self.w_BI_B,self.H_total) + self.magt_module.T)

        #### Calculate the new satellite body state rates
        # Kinematics dq_BI = 0.5 * q_BI (x) w_BI_B  (Hamilton, body-frame rate).
        w_BI_B_quat = np.quaternion(0, self.w_BI_B[0], self.w_BI_B[1], self.w_BI_B[2])

        dq_BI = 0.5*self.q_BI*w_BI_B_quat
        dq_BI = [dq_BI.x, dq_BI.y, dq_BI.z, dq_BI.w]
        control_power = abs(self.wheel_module.dH_vec * self.w_BI_B) ## @TODO fix this
        
        self.fault_module.update(t)
        self.update_FDIR(t)

        self.logger.store_data(t)

        return np.hstack([self.dw_BI_B, dq_BI, control_power, self.wheel_module.dw_wheels])

    
    def update_fd(self, t):
        # print("angular")
        if not self.config['FDIR']['satellite']['enable']:
            return
        if my_utils.magnitude(self.w_BI_B) > self.fd_w_max:
            raise DivergentRate(f"Divergent rate detected, w_sat = {self.w_BI_B*my_utils.RAD_TO_DEG} at time {t}")
        
    def update_FDIR(self, t):

        self.update_fd(t)

        for wheel in self.wheel_module.wheels:
            wheel.update_fd(t)

        if self.fault_detector is not None:
            self.fault_detector.update(t, self.w_BI_B, self.T_ctr_wheels,
                                       self.wheel_module.H_vec, self.magt_module.T,
                                       T_dist=self.T_dist)
            if self.fault_detector.latched_wheel >= 0 and not self.fault_detector.reconfigured:
                self.fault_detector.reconfigured = True  # one-shot: never re-enters
                if self.config['detection'].get('reconfigure_en', True):
                    self.controller.activate_underactuated(t, self, self.fault_detector.latched_wheel,
                                                           chi_1_init=self.fault_detector.chi_1)
