from cmath import tau

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
from route_planner import yaw_maneuver_waypoints

# Peak |s'| and |s''| of the quintic smootherstep s = 6t^5 - 15t^4 + 10t^3 used by
# _tracking_q_ref_at, over t in [0,1]. s' = 30t^2(1-t)^2 peaks at 1.875 (t=0.5); s''
# peaks at 5.7735 (t = (3 -+ sqrt(3))/6). Exact, not fitted: _tracking_q_ref_at composes
# a fixed rotation with a one-parameter subgroup about a fixed axis, so the body-frame
# rate/accel of the path are exactly s'(t)*rel and s''(t)*rel.
SMOOTHERSTEP_PEAK_RATE = 1.875
SMOOTHERSTEP_PEAK_ACCEL = 5.7735


class DivergentRate(Exception):
    pass

class DivergentAttitude(Exception):
    pass

class Face():
    r_com_to_cop = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    solar_panel = False
    
def face_geometry(model, dimensions):
    """Surface faces (normal, area, centre of pressure) of the satellite model; the
    solar panels are flagged with solar_panel = True."""
    faces = []
    if model == "rectangular":
        dim_array = np.array([dimensions['x'], dimensions['y'], dimensions['z']])
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
                faces.append(face)
                # print(face.norm_vec, face.area, face.r_com_to_cop)
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
        solar_panels = [2, 3]
        for i in range(len(com_to_cop_array)):
            face = Face()
            face.r_com_to_cop = np.array(com_to_cop_array[i])
            face.norm_vec = np.array(norms_array[i])
            face.area = areas[i]
            face.solar_panel = i in solar_panels
            faces.append(face)
    return faces

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

        # True plant inertia. M_inertia above is the controller/model value; set
        # M_Inertia_plant to give the plant a static inertia error (defaults to M_inertia).
        M_inertia_plant = np.array(config['satellite'].get('M_Inertia_plant', self.M_inertia), dtype=float)
        if M_inertia_plant.shape == (3,):
            M_inertia_plant = np.diag(M_inertia_plant)
        self.M_inertia_plant = M_inertia_plant

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
        self.H = self.M_inertia_plant@self.w_BI_B
        self.H_total = self.H + self.wheel_module.H_vec

        self.E = np.eye(self.wheel_module.num_wheels)
        self.f_wheels = np.zeros(self.wheel_module.num_wheels)
        self.f_wheels_fault = np.zeros(self.wheel_module.num_wheels)

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
                q_ref_array = Rotation.from_euler("XYZ", config['satellite']['ref_euler'], degrees=True).as_quat()
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
        elif self.mode == "tracking":
            q_series = self.config['satellite']['ref_q_series']   # each [x, y, z, w]
            t_series = self.config['satellite']['ref_t_series']
            rots = [Rotation.from_quat(q) for q in q_series]
            self._tracking_keys_cache = (rots, np.array(t_series, dtype=float)) 
        else:
            self.q_RI = np.quaternion(1,0,0,0)

        # nominal_night: T_OI can step discretely between ref updates (e.g. LVLH frame
        # singularities near the poles); jumps bigger than this are smoothed with a
        # smootherstep SLERP over the given duration instead of applied instantly, since
        # a stepped reference otherwise spikes the FNDO observer.
        self.night_interp_threshold_deg = config['satellite'].get('nominal_night_interp_threshold_deg', 5.0)
        self.night_interp_duration = config['satellite'].get('nominal_night_interp_duration', 180.0)
        self._night_interp_enabled = config['satellite'].get('nominal_night_interp_enabled', True)

        # Yaw (z, unactuated under Nadafi_FNDO) can't be slewed to directly - when the
        # error trigger fires, drive it via route_planner's roll/pitch "box" maneuver
        # (geometric-phase yaw via the Lie bracket) instead of a naive SLERP, which would
        # demand wz through the intermediate path just like the direct jump would.
        self.night_yaw_maneuver_enable = config['satellite'].get('nominal_night_yaw_maneuver_enable', True)
        self.night_yaw_max_tilt_deg = config['satellite'].get('nominal_night_yaw_max_tilt_deg', 20.0)
        self.night_yaw_leg_duration = config['satellite'].get('nominal_night_yaw_leg_duration', 20.0)
        # One box pass leaves a residual (nadir drifts during the maneuver, and the box
        # endpoint carries roll/pitch), so converging needs successive smaller passes.
        # Wait out the cooldown between passes: the yaw metric picks up roll/pitch
        # crosstalk, so re-measuring before the cleanup interp has settled chases phantom
        # error. Give up once passes stop buying anything, rather than thrashing forever.
        self.night_yaw_cooldown = config['satellite'].get('nominal_night_yaw_cooldown', 60.0)
        self.night_yaw_max_attempts = config['satellite'].get('nominal_night_yaw_max_attempts', 4)
        self.night_yaw_min_improve = config['satellite'].get('nominal_night_yaw_min_improve', 0.7)
        
        # A box takes minutes, over which nadir rotates at orbital rate (~0.063 deg/s), so
        # sizing the yaw against nadir *now* is stale by ~10 deg before the box even ends.
        # Size it against the propagated nadir frame at maneuver end instead.
        self.night_yaw_predict_enable = config['satellite'].get('nominal_night_yaw_predict_enable', True)
        self.night_yaw_predict_iters = config['satellite'].get('nominal_night_yaw_predict_iters', 3)

        # Each box leg is a rest-to-rest single-axis slew of `tilt`, so its peak wheel
        # torque/momentum demand is set by the leg duration. Size that duration from the
        # actual wheel limits rather than trusting a hand-set constant: momentum scales
        # as 1/T and torque as 1/T^2, so a leg that is merely "slow-looking" can still be
        # the thing that saturates a wheel and winds the FNDO integrator up.
        self.night_yaw_actuator_limit_enable = config['satellite'].get('nominal_night_yaw_actuator_limit_enable', True)
        self.night_yaw_actuator_margin = config['satellite'].get('nominal_night_yaw_actuator_margin', 0.5)
        self.night_yaw_leg_duration_max = config['satellite'].get('nominal_night_yaw_leg_duration_max', 600.0)

        # nominal_day: body axis pointed at the sun (default +x, the EOSSAT solar panels'
        # mean normal); body z is kept as near nadir as the sun constraint allows.
        self.day_sun_axis = np.array(config['satellite'].get('nominal_day_sun_axis', [1.0, 0.0, 0.0]), dtype=float)
        self.day_sun_axis /= np.linalg.norm(self.day_sun_axis)
        # body counterpart of the inertial triad's second axis (sun x nadir), with z -> nadir
        z_B = np.array([0.0, 0.0, 1.0])
        b2 = my_utils.cross_product_M31M31(self.day_sun_axis, z_B)
        if np.linalg.norm(b2) < 1e-6:
            raise ValueError("nominal_day_sun_axis must not be parallel to body z (z is held toward nadir)")
        self._day_secondary_axis = b2 / np.linalg.norm(b2)

        # Sun pointing only pins 2 DOF - the rotation *about* the sun line is free, and the
        # TRIAD above spends it on "keep z near nadir". Spend it instead on making the mode
        # transitions cost nothing on the unactuated z axis: pick the clocking that zeroes
        # the z error entering sun mode, then roll (about the sun axis, which is actuated)
        # to the one that zeroes it on the way back to nadir. Turns a ~135 deg box maneuver
        # into an ~80 s roll. See _day_latch_clocking.
        self.day_clock_enable = config['satellite'].get('nominal_day_clock_enable', True)
        # wait out the mode-transition slew before re-clocking, so the two don't overlap
        self.day_reclock_delay = config['satellite'].get('nominal_day_reclock_delay',
                                                         self.night_interp_duration)

        self.fd_w_max = config['FDIR']['satellite']['w_max_dps'] * my_utils.DEG_TO_RAD
        # Pointing-error FD: principal angle of q_RB against the *commanded* reference.
        # Reference jumps are smoothed from the current attitude, so this stays small
        # through acquisition and mode switches (~2 deg peak) even when the raw target
        # jumps 140+ deg; a sustained large error means the controller has lost the ref.
        self.fd_q_max = config['FDIR']['satellite'].get('q_err_max_deg', 10.0) * my_utils.DEG_TO_RAD
        self.fd_q_persist = config['FDIR']['satellite'].get('q_err_persist_s', 10.0)
        self._fd_q_exceed_t0 = None

        self.next_t_ref_update_interval = config['controller']['t_sample']

        if self.mode == "sinusoidal":
            self.sinusoidal_ref_amplitude = np.array(config['satellite']['sinusoidal_ref_amplitude_deg']) * my_utils.DEG_TO_RAD
            self.sinusoidal_ref_frequency = np.array(config['satellite']['sinusoidal_ref_frequency'])

        self.auto_switch = config['satellite'].get('auto_switch', True)
        
    def calc_face_properties(self):
        # per instance: the class-level list would keep accumulating faces (and so
        # double the SRP/aero torques) for every Satellite built in the same process
        self.faces = face_geometry(self.config['satellite']['dimensions']['model'], self.dimensions)
        self.solar_panels = [face for face in self.faces if face.solar_panel]

    def calc_panel_cos(self, T_BI):
        """Sun incidence cosine per solar panel, clipped at 0 (sun behind the panel)
        and zeroed in eclipse. nSB_I points satellite -> sun."""
        if self.orbit.eclipse:
            return np.zeros(len(self.solar_panels))
        n_sun_B = T_BI @ self.orbit.nSB_I
        return np.array([max(0.0, face.norm_vec @ n_sun_B / np.linalg.norm(face.norm_vec))
                         for face in self.solar_panels])

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

    auto_switch = True
    def update_mode(self, t):
        mode_prev = self.mode
        if self.init == True:
            if self.config['simulation']['verbose']:
                print("MODE STARTUP \t ", self.mode)
        if self.mode == "ref_pointing":
            return
        if self.auto_switch:
            if self.mode == "nominal_day" and self.orbit.eclipse:
                self.mode = "nominal_night"
            if self.mode == "nominal_night" and not self.orbit.eclipse:
                self.mode = "nominal_day"

        if self.mode != mode_prev:
            print("MODE SWITCH \t ", mode_prev, " -> ", self.mode, " at t = ", t, "s")
            if self.mode == "nominal_day":
                # sun mode's free DOF is chosen here, against the attitude we are leaving
                self._day_latch_clocking(t)

    t_ref_update = 0
    q_RI_series_index = 0
    init = True
    next_t_ref_update = 0.0
    next_t_ref_update_interval = None
    last_q_RI_update_t = 0.0

    # nominal_night reference-jump smoothing state (see update_ref_q)
    _night_interp_enabled = False
    _night_interp_active = False
    _night_interp_t0 = 0.0
    _night_interp_t1 = 0.0
    _night_interp_offset0 = None

    # nominal_night yaw box-maneuver state (see update_ref_q)
    _night_box_active = False
    _night_box_rots = None
    _night_box_times = None
    _night_box_next_allowed_t = 0.0
    _night_box_attempts = 0
    _night_box_prev_err = None
    # last well-defined nominal_day target (see _nominal_target_rot_at)
    _day_last_rot = None

    # nominal_day clocking schedule (see _day_latch_clocking)
    _day_phi_out = 0.0
    _day_phi_in = 0.0
    _day_dphi = 0.0
    _day_rc_t0 = 0.0
    _day_rc_t1 = 0.0
    # latest raw nominal-mode target q_RI (see update_ref_q), None until first computed
    q_target_RI = None
    def update_ref_q(self, t):
        if t >= self.next_t_ref_update - my_utils.T_EPS:
            self.next_t_ref_update = t + self.next_t_ref_update_interval
        else:
            return
        self.update_mode(t)

        if self.init and self.mode not in ("ref_pointing", "tracking"):
            if self.mode == "nominal_night" or self.mode == "nominal_day":
                self.q_RI = my_utils.conv_Rotation_obj_to_numpy_q(self._nominal_target_rot_at(t))
            self.w_RI_R = np.zeros(3)
            self.last_q_RI_update_t = t
            self.init = False
            return
        if self.mode == "nominal_night" or self.mode == "nominal_day":
            # Both nominal modes are nadir pointing and differ only in the yaw target
            # (see _nominal_target_rot_at), so they share the same reference shaping.
            # The target can step discretely between ref updates (e.g. LVLH frame
            # singularities, or the day-mode yaw flip as the sun crosses nadir); feeding
            # that step straight to the controller spikes the FNDO observer, so jumps
            # bigger than night_interp_threshold_deg are smoothed with the same
            # smootherstep SLERP "tracking" mode uses between waypoints, and yaw (the
            # unactuated axis) is driven with the box maneuver.
            # The target is a function of time (it turns at ~orbit rate), so the
            # reference rate/accel feedforward comes from central differences of it.
            target_at = self._nominal_target_rot_at
            new_q_RI = my_utils.conv_Rotation_obj_to_numpy_q(target_at(t))
            # dcm_to_quat picks the sign of q arbitrarily (scipy), while the interp/box
            # branches build q_RI from q_BI - keep the same hemisphere as the previous
            # reference so q_RI stays continuous instead of flipping to -q between branches.
            if (new_q_RI.w*self.q_RI.w + new_q_RI.x*self.q_RI.x
                    + new_q_RI.y*self.q_RI.y + new_q_RI.z*self.q_RI.z) < 0:
                new_q_RI = -new_q_RI
            # raw (unsmoothed) mode target, logged for the nadir/sun comparison plot
            self.q_target_RI = new_q_RI
            dt = 1e-3
            if self._night_interp_enabled:
                if self._night_box_active:
                    if t >= self._night_box_times[-1]:
                        self.q_RI = my_utils.conv_Rotation_obj_to_numpy_q(self._night_box_rots[-1])
                        self.w_RI_R = np.zeros(3)
                        self.dw_RI_R = np.zeros(3)
                        self._night_box_active = False
                        # Re-arm after a cooldown instead of latching off for good, so
                        # successive (smaller) passes can converge the residual yaw.
                        self._night_box_next_allowed_t = t + self.night_yaw_cooldown
                    else:
                        i = self._tracking_find_time_index(t, self._night_box_times)
                        self._tracking_update_ref_project_smooth_interp(
                            t, self._night_box_times[i], self._night_box_times[i + 1],
                            self._night_box_rots[i], self._night_box_rots[i + 1], dt)
                elif self._night_interp_active:
                    if t >= self._night_interp_t1:
                        self._update_ref_from_fn(target_at, t, dt)
                        self._night_interp_active = False
                    else:
                        self._update_ref_from_fn(self._nominal_interp_rot_at, t, dt)
                else:
                    # Yaw (z) component of the body->target error, as an angle (rad for a
                    # pure-z rotation; approximate otherwise) - this is specifically the
                    # unactuated-axis error, not the full 3-axis error.
                    # err_deg_unactuated = np.degrees(2 * np.arcsin(np.clip(my_utils.quat_error(new_q_RI, self.q_BI).z, -1.0, 1.0)))
                    err_deg_unactuated = self._night_yaw_err_deg(new_q_RI)
                    err_deg_total = np.degrees(2 * np.arccos(np.clip(my_utils.quat_error(new_q_RI, self.q_BI).w, -1.0, 1.0)))

                    # In sun mode the free DOF about the sun line is already doing this
                    # job, cheaply and on an actuated axis, so the box has nothing to add
                    # there - and firing it on transition transients just throws the
                    # attitude away for the length of a box.
                    box_mode_ok = not (self.mode == "nominal_day" and self.day_clock_enable)
                    start_box = (abs(err_deg_unactuated) > self.night_interp_threshold_deg
                                 and self.night_yaw_maneuver_enable
                                 and box_mode_ok
                                 and t >= self._night_box_next_allowed_t)
                    if start_box:
                        # Stop re-arming once passes stop paying for themselves, otherwise
                        # we thrash multi-minute maneuvers forever against the residual
                        # floor set by nadir drift during the box + metric crosstalk.
                        give_up = None
                        if self._night_box_attempts >= self.night_yaw_max_attempts:
                            give_up = "attempt budget ({:d}) spent".format(self.night_yaw_max_attempts)
                        elif (self._night_box_prev_err is not None
                              and abs(err_deg_unactuated) >= self.night_yaw_min_improve * abs(self._night_box_prev_err)):
                            give_up = "no longer converging ({:.2f} -> {:.2f} deg)".format(
                                self._night_box_prev_err, err_deg_unactuated)
                        if give_up is not None:
                            print("{}: yaw error {:.2f} deg at t={:.2f}s - {}, holding "
                                  "nadir track with yaw uncorrected".format(self.mode, err_deg_unactuated, t, give_up))
                            self.night_yaw_maneuver_enable = False
                            start_box = False

                    if start_box:
                        yaw_deg, q_series, info, horizon, leg = self._night_plan_yaw_box(t)
                        rot_start = my_utils.conv_numpy_to_Rotation_obj_q(self.q_BI)
                        self._night_box_rots = [rot_start * Rotation.from_quat(q) for q in q_series]
                        n_legs = len(self._night_box_rots) - 1
                        self._night_box_times = t + leg * np.arange(len(self._night_box_rots))
                        self._night_box_active = True
                        self._night_box_attempts += 1
                        self._night_box_prev_err = err_deg_unactuated
                        print("{}: yaw error {:.2f} deg at t={:.2f}s, targeting {:.2f} deg at "
                              "t+{:.0f}s, executing {:d}-leg box (tilt={:.1f} deg, leg={:.0f}s "
                              "set by {}) over {:.1f}s [pass {:d}/{:d}]".format(
                                  self.mode, err_deg_unactuated, t, yaw_deg, horizon, n_legs,
                                  info['tilt_deg'], leg, info['leg_binding'], leg * n_legs,
                                  self._night_box_attempts, self.night_yaw_max_attempts))
                        self._tracking_update_ref_project_smooth_interp(
                            t, self._night_box_times[0], self._night_box_times[1],
                            self._night_box_rots[0], self._night_box_rots[1], dt)
                    elif abs(err_deg_total) > self.night_interp_threshold_deg:
                        print("{}: reference jump of {:.2f} deg at t={:.2f}s, smoothing over {:.2f}s".format(self.mode, err_deg_total, t, self.night_interp_duration))
                        self._night_interp_active = True
                        self._night_interp_t0 = t
                        self._night_interp_t1 = t + self.night_interp_duration
                        # Blend out the body's offset from the *moving* target rather
                        # than SLERP from a fixed inertial attitude, which would leave the
                        # reference lagging the target by the orbit rotation in the meantime.
                        self._night_interp_offset0 = target_at(t).inv() * my_utils.conv_numpy_to_Rotation_obj_q(self.q_BI)
                        self._update_ref_from_fn(self._nominal_interp_rot_at, t, dt)
                    else:
                        self._update_ref_from_fn(target_at, t, dt)
            else:
                self._update_ref_from_fn(target_at, t, dt)


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
            rots , times = self._tracking_keys_cache
            if t <= times[0]:
                self.q_RI = my_utils.conv_Rotation_obj_to_numpy_q(rots[0])
                self.w_RI_R = np.zeros(3)
                self.dw_RI_R = np.zeros(3)
            elif t >= times[-1]:
                self.q_RI = my_utils.conv_Rotation_obj_to_numpy_q(rots[-1])
                self.w_RI_R = np.zeros(3)
                self.dw_RI_R = np.zeros(3)
            else:
                i = self._tracking_find_time_index(t, times)
                t_series_last = times[i]
                t_series_next = times[i + 1]
                rot1 = rots[i]
                rot2 = rots[i + 1]
                self._tracking_update_ref_project_smooth_interp(t, t_series_last, t_series_next, rot1, rot2, dt)

    def _night_yaw_err_deg(self, q_target):
        """Yaw (z) component of the body->target error, in degrees.

        Exact for a pure-z error, approximate otherwise - a simultaneous roll+pitch error
        contributes a real z term (the same non-commutativity the box maneuver exploits),
        so this reads high while roll/pitch is still large.
        """
        return np.degrees(2 * np.arcsin(np.clip(my_utils.quat_error(q_target, self.q_BI).z, -1.0, 1.0)))

    def _night_box_leg_duration(self, tilt_deg):
        """Leg duration for a box of this tilt, floored by what the wheels can deliver.

        Every leg is a rest-to-rest single-axis slew of `tilt` about body x or y, so both
        axes are checked and the binding one wins (they share one duration, which keeps
        the waypoint schedule uniform). `nominal_night_yaw_leg_duration` stays a floor -
        the limits can only ever slow the maneuver down, never speed it past what was
        configured.
        """
        leg = self.night_yaw_leg_duration
        if not self.night_yaw_actuator_limit_enable:
            return leg, "actuator limits disabled"

        wheel_idx = self.controller.get_actuated_axes()
        tilt_rad = np.radians(abs(tilt_deg))
        binding = "configured floor"
        for axis in wheel_idx:
            t_axis, binding_axis = self.wheel_module.min_feasible_slew_duration(
                self.M_inertia, np.eye(3)[axis], tilt_rad, wheel_idx,
                SMOOTHERSTEP_PEAK_RATE, SMOOTHERSTEP_PEAK_ACCEL,
                margin=self.night_yaw_actuator_margin)
            if t_axis > leg:
                leg, binding = t_axis, "axis {:d} leg, {}".format(axis, binding_axis)

        if leg > self.night_yaw_leg_duration_max:
            print("{}: box leg wants {:.0f}s ({}) - capping at {:.0f}s, wheels will be "
                  "pushed past the {:.0%} margin".format(
                      self.mode, leg, binding, self.night_yaw_leg_duration_max,
                      self.night_yaw_actuator_margin))
            leg, binding = self.night_yaw_leg_duration_max, "duration cap"
        return leg, binding

    def _night_plan_yaw_box(self, t):
        """Size a yaw box maneuver against where nadir will be when the maneuver ends.

        The ask depends on the horizon; the horizon depends on the leg count *and* the
        per-leg duration, which in turn depends on the tilt the leg count implies - so
        iterate to a fixed point. It converges in a pass or two: moving the horizon only
        shifts the ask by the orbital drift over the maneuver, and the leg count is a
        step function of the ask, so it settles as soon as the count stops changing.

        Returns (yaw_deg, q_series, info, horizon, leg_duration) for the final plan.
        """
        horizon = 0.0
        yaw_deg = q_series = info = None
        leg = self.night_yaw_leg_duration
        for _ in range(max(1, self.night_yaw_predict_iters)):
            if self.night_yaw_predict_enable and horizon > 0.0:
                T_OI_future, _, _ = self.orbit.calc_T_OI_at(t + horizon)
                q_target = my_utils.dcm_to_quat(T_OI_future)
            else:
                q_target = my_utils.dcm_to_quat(self.orbit.T_OI)
            yaw_deg = self._night_yaw_err_deg(q_target)
            q_series, info = yaw_maneuver_waypoints(yaw_deg, max_tilt_deg=self.night_yaw_max_tilt_deg)
            leg, binding = self._night_box_leg_duration(info['tilt_deg'])
            info['leg_duration'] = leg
            info['leg_binding'] = binding
            horizon_new = leg * (len(q_series) - 1)
            if not self.night_yaw_predict_enable or horizon_new == horizon:
                horizon = horizon_new
                break
            horizon = horizon_new
        return yaw_deg, q_series, info, horizon, leg


    def _nominal_target_rot_at(self, tau):
        """Target R frame for the nominal modes at time tau, as a Rotation.
        nominal_night: plain LVLH, R == O so T_RI = T_OI.
        nominal_day: sun pointing - body day_sun_axis exactly at the sun, with the
        free rotation about the sun line set so body z is as close to nadir as it can
        get (TRIAD, sun primary, nadir secondary). When the sun is (anti-)parallel to
        nadir that rotation is undefined, so hold the last valid day target there
        (LVLH if none yet). Uses the current sun vector: it moves ~1 deg/day,
        negligible over a ref update."""
        T_OI, r_I, _ = self.orbit.calc_T_OI_at(tau)
        if self.mode == "nominal_day":
            rot_triad = self._day_triad_rot_at(tau, r_I=r_I)
            if rot_triad is not None:
                self._day_last_rot = rot_triad
            if self._day_last_rot is not None:
                # the clocking factor right-multiplies: it is a rotation about the BODY
                # sun axis, so b1 stays on the sun for any phi
                return self._day_last_rot * Rotation.from_rotvec(
                    self._day_clock_phi_at(tau) * self.day_sun_axis)
        return my_utils.conv_numpy_to_Rotation_obj_q(my_utils.dcm_to_quat(T_OI))

    def _day_triad_rot_at(self, tau, r_I=None):
        """Sun-pointing TRIAD at tau, before clocking; None when sun is (anti-)parallel
        to nadir and the triad is undefined. Uses the sun vector propagated to tau, so
        it is valid for the look-ahead the clocking planner needs, not just for now."""
        s_I, _, r_I = self.orbit.calc_sun_vector_at(tau, r_I=r_I)
        n_I = -r_I / np.linalg.norm(r_I)
        r2 = my_utils.cross_product_M31M31(s_I, n_I)
        n = np.linalg.norm(r2)
        if n <= 1e-3:
            return None
        r2 = r2 / n
        r3 = my_utils.cross_product_M31M31(s_I, r2)
        # same triad built from the body axes; T_RI maps the inertial triad onto it
        b1 = self.day_sun_axis
        b2 = self._day_secondary_axis
        b3 = my_utils.cross_product_M31M31(b1, b2)
        T_RI = np.column_stack([b1, b2, b3]) @ np.row_stack([s_I, r2, r3])
        return my_utils.conv_numpy_to_Rotation_obj_q(my_utils.dcm_to_quat(T_RI))

    def _day_clock_phi_at(self, tau):
        """Scheduled clocking angle (rad) about the body sun axis.

        A function of tau only - never of the live q_BI outside the latch instant. That
        keeps it smooth for the central differences in _update_ref_from_fn, and stops the
        reference from chasing the body (which would read zero error forever without ever
        correcting anything).
        """
        if not self.day_clock_enable:
            return 0.0
        if tau <= self._day_rc_t0:
            return self._day_phi_out
        if tau >= self._day_rc_t1:
            return self._day_phi_out + self._day_dphi
        s = (tau - self._day_rc_t0) / (self._day_rc_t1 - self._day_rc_t0)
        s = s*s*s*(s*(s*6 - 15) + 10)       # smootherstep, C2 - same profile as the slews
        return self._day_phi_out + s * self._day_dphi

    def _day_latch_clocking(self, t):
        """Pick the sun-phase clocking angles, once, at the night -> day switch.

        phi_out zeroes the unactuated-axis (z) component of the nadir -> sun error now;
        phi_in zeroes it again for the sun -> nadir handback at the next eclipse entry.
        Both are closed form: writing the error as q_e (x) q_clock(phi) makes its z
        component linear in (cos(phi/2), sin(phi/2)), so one atan2 lands the zero exactly.
        The roll between them is about the sun axis, which is actuated, so it is an
        ordinary slew sized by the wheel limits.
        """
        b1 = self.day_sun_axis
        self._day_phi_out = self._day_phi_in = self._day_dphi = 0.0
        self._day_rc_t0 = self._day_rc_t1 = t
        if not self.day_clock_enable:
            return

        rot_triad = self._day_triad_rot_at(t)
        if rot_triad is None:
            print("{}: sun near nadir at t={:.1f}s, triad undefined - clocking left at 0".format(self.mode, t))
            return

        # phi_out: zero the z error on the transition we are making right now
        q0 = my_utils.quat_error(my_utils.conv_Rotation_obj_to_numpy_q(rot_triad), self.q_BI)
        v0 = np.array([q0.x, q0.y, q0.z])
        self._day_phi_out = 2*np.arctan2(-v0[2], q0.w*b1[2] + np.cross(v0, b1)[2])

        # phi_in: zero it again on the handback to nadir at the next eclipse entry
        t_entry = self.orbit.next_eclipse_entry_after(t)
        rot_triad_entry = None if t_entry is None else self._day_triad_rot_at(t_entry)
        if rot_triad_entry is None:
            print("{}: no eclipse entry found ahead of t={:.1f}s - holding one clocking".format(self.mode, t))
            self._day_phi_in = self._day_phi_out
        else:
            q_nadir = my_utils.dcm_to_quat(self.orbit.calc_T_OI_at(t_entry)[0])
            qf = my_utils.conv_Rotation_obj_to_numpy_q(rot_triad_entry).inverse() * q_nadir
            vf = np.array([qf.x, qf.y, qf.z])
            self._day_phi_in = 2*np.arctan2(vf[2], qf.w*b1[2] + np.cross(b1, vf)[2])

        # short way round
        self._day_dphi = (self._day_phi_in - self._day_phi_out + np.pi) % (2*np.pi) - np.pi

        leg, binding = self.wheel_module.min_feasible_slew_duration(
            self.M_inertia, b1, abs(self._day_dphi), self.controller.get_actuated_axes(),
            SMOOTHERSTEP_PEAK_RATE, SMOOTHERSTEP_PEAK_ACCEL,
            margin=self.night_yaw_actuator_margin)
        leg = max(leg, 1.0)
        self._day_rc_t0 = t + self.day_reclock_delay
        self._day_rc_t1 = self._day_rc_t0 + leg
        if t_entry is not None and self._day_rc_t1 > t_entry:
            # pull it as early as possible rather than running past the handback
            self._day_rc_t0 = max(t, t_entry - leg)
            self._day_rc_t1 = self._day_rc_t0 + leg
            if self._day_rc_t1 > t_entry:
                print("{}: re-clock of {:.1f} deg needs {:.0f}s but only {:.0f}s of sun phase "
                      "remain - will not finish before handback".format(
                          self.mode, np.degrees(self._day_dphi), leg, t_entry - t))

        print("{}: clocking latched at t={:.1f}s - phi_out {:+.1f} deg, phi_in {:+.1f} deg, "
              "re-clock {:+.1f} deg over {:.0f}s from t={:.0f}s ({})".format(
                  self.mode, t, np.degrees(self._day_phi_out), np.degrees(self._day_phi_in),
                  np.degrees(self._day_dphi), leg, self._day_rc_t0, binding))

    def _nominal_interp_rot_at(self, tau):
        """Moving target composed with the start offset, smootherstepped to identity."""
        s = np.clip((tau - self._night_interp_t0) / (self._night_interp_t1 - self._night_interp_t0), 0.0, 1.0)
        s = s*s*s*(s*(s*6 - 15) + 10)
        return self._nominal_target_rot_at(tau) * Rotation.from_rotvec((1.0 - s) * self._night_interp_offset0.as_rotvec())

    def _update_ref_from_fn(self, rot_at, t, dt = 1e-3):
        """Set q_RI from rot_at(t), and w_RI_R / dw_RI_R (R frame) by central differences."""
        R0 = rot_at(t)
        R_fwd = (R0.inv() * rot_at(t + dt)).as_rotvec() / dt
        R_bwd = (rot_at(t - dt).inv() * R0).as_rotvec() / dt
        q_RI = my_utils.conv_Rotation_obj_to_numpy_q(R0)
        # keep q_RI in the previous hemisphere (q and -q are the same attitude)
        if (q_RI.w*self.q_RI.w + q_RI.x*self.q_RI.x + q_RI.y*self.q_RI.y + q_RI.z*self.q_RI.z) < 0:
            q_RI = -q_RI
        self.q_RI = q_RI
        self.w_RI_R = 0.5 * (R_fwd + R_bwd)
        self.dw_RI_R = (R_fwd - R_bwd) / dt

    def _tracking_find_time_index(self, tau, times):
        i = int(np.searchsorted(times, tau) - 1)
        i = max(0, min(i, len(times) - 2))
        return i

    def _tracking_q_ref_at(self, tau, t1, t2, rot1, rot2):
        s = (tau - t1) / (t2 - t1)
        s = s*s*s*(s*(s*6 - 15) + 10)          # smootherstep: C2, zero rate/accel at waypoints
        rel = (rot1.inv() * rot2).as_rotvec()
        return rot1 * Rotation.from_rotvec(s * rel)

    def _tracking_finite_difference(self, tau, dt, t1, t2, rot1, rot2):
        R0 = self._tracking_q_ref_at(tau, t1, t2, rot1, rot2)
        R_fwd = (R0.inv() * self._tracking_q_ref_at(tau + dt, t1, t2, rot1, rot2)).as_rotvec() / dt
        R_bwd = (self._tracking_q_ref_at(tau - dt, t1, t2, rot1, rot2).inv() * R0).as_rotvec() / dt
        return R_bwd, R_fwd

    def _tracking_update_ref_project_smooth_interp(self, t, t_series_last, t_series_next, rot1, rot2, dt = 1e-3):
        """
        Update the reference quaternion and its derivatives using smooth interpolation between waypoints.
        - t: current time
        - t_series_last: time of the last waypoint
        - t_series_next: time of the next waypoint
        - rot1: Rotation object of the last waypoint
        - rot2: Rotation object of the next waypoint
        - dt: time step for finite difference approximation of derivatives
        """
        R0 = self._tracking_q_ref_at(t, t_series_last, t_series_next, rot1, rot2)
        R_bwd, R_fwd = self._tracking_finite_difference(t, dt, t_series_last, t_series_next, rot1, rot2)
        self.q_RI  = my_utils.conv_Rotation_obj_to_numpy_q(R0)
        self.w_RI_R  = 0.5 * (R_fwd + R_bwd)
        self.dw_RI_R = (R_fwd - R_bwd) / dt

    def calc_delta_M_inertia(self, t):
        # Sinusoidal parametric uncertainty Delta J = frac * diag(J) .* sin(freq*t), applied
        # per axis. frac is set from the config so it scales with the plant: the old hard-coded
        # amplitudes [2, 2.8, 3.6] (= 0.8 * diag(J) of the Nadafi plant) drove the effective
        # inertia negative on smaller satellites (e.g. Zarourati's J = diag(0.92, 0.92, 0.44)),
        # which makes the plant unconditionally unstable. frac must stay < 1.
        if self.delta_inertia_frac == 0.0:
            return np.zeros((3,3))
        return self.delta_inertia_frac * np.diag(np.diag(self.M_inertia_plant) * np.sin(self.delta_inertia_freq * t))

    wheels_control_enable = True
    T_ctr_vec = None
    T_dist = np.zeros(0)
    E = None
    f = None
    f_wheels = None
    f_wheels_fault = None
    def calc_state_rates(self, t, y):

        w_wheels_input = self._set_state(y)
        # Passive body DCM T_BI: v_B = T_BI @ v_I  (used by the disturbance models).
        T_BI = my_utils.quat_to_dcm(self.q_BI)

        self.orbit.calc_orbit_state(t)

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

        # Orbit first so B_I is current for magnetorquers and residual dipole.
        self.orbit.calc_orbit_state(t)
        self.magt_module.calc_torque(qv_RB, self.w_BI_B, self.H + self.wheel_module.H_vec, t, T_BI)
        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)
        
        # Ground-truth per-wheel torque deviation, logged as f_wheels. Same sign convention
        # as the ESO's f_wheels_est (dw_est = (u + f)/J): f = delivered - commanded, where
        # delivered = J*dw is the net wheel torque after the fault, torque clip, speed-limit
        # clamp and friction. f_wheels_fault keeps only the fault's share of it.

        self.f_wheels = self.wheel_module.dH_wheels - self.T_ctr_wheels
        self.f_wheels_fault = self._calc_fault_torque()

        if self.observer_module.enable is True:
            # @NB FIX THIS
            self.E_est = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
            self.E = self.fault_module.E
        #### Calculate state rates for satellite various subsystems

        body_rates = self._calc_body_rates(t, T_BI)

        self.fault_module.update(t)
        self.update_FDIR(t)

        self.logger.store_data(t)

        return np.hstack([body_rates, self.wheel_module.dw_wheels])

    def _calc_fault_torque(self):
        """Fault-only wheel torque: clip(E u + u_a) - clip(u), both clipped to the wheel
        torque limit. Friction and the speed-limit clamp act on healthy and faulted wheels
        alike, so they are left out. Without the clip, (E - I) u + u_a scales with the
        unsaturated command and reports N*m-level faults whenever the controller winds up."""
        T_max = np.array([wheel.T_max for wheel in self.wheel_module.wheels])
        u = self.T_ctr_wheels
        u_faulted = self.fault_module.E @ u + self.fault_module.u_a
        return np.clip(u_faulted, -T_max, T_max) - np.clip(u, -T_max, T_max)

    def _set_state(self, y):
        self.w_BI_B = np.array(y[:3])
        self.q_BI = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        return y[10:self.wheel_module.num_wheels + 10]

    def _calc_body_rates(self, t, T_BI):
        """Rigid-body dynamics + kinematics given the current wheel rates and held
        actuator outputs. Returns [dw_BI_B, dq_BI (xyzw), control_power]."""
        self.T_dist = self.disturbances.calc_torque(self, T_BI, t)

        delta_M_inertia = self.calc_delta_M_inertia(t)
        M_inertia_effective = self.M_inertia_plant + delta_M_inertia
        M_inertia_effective_inv = np.linalg.inv(M_inertia_effective)

        # M_inertia_effective_inv = self.M_inertia_inv

        self.H = self.M_inertia_plant@(self.w_BI_B)
        self.H_total = self.H + self.wheel_module.H_vec
        self.dw_BI_B = (M_inertia_effective_inv)@(-1*self.wheel_module.dH_vec + self.T_dist - my_utils.cross_product_M31M31(self.w_BI_B,self.H_total) + self.magt_module.T)

        #### Calculate the new satellite body state rates
        # Kinematics dq_BI = 0.5 * q_BI (x) w_BI_B  (Hamilton, body-frame rate).
        w_BI_B_quat = np.quaternion(0, self.w_BI_B[0], self.w_BI_B[1], self.w_BI_B[2])

        dq_BI = 0.5*self.q_BI*w_BI_B_quat
        dq_BI = [dq_BI.x, dq_BI.y, dq_BI.z, dq_BI.w]
        control_power = abs(self.wheel_module.dH_vec * self.w_BI_B) ## @TODO fix this
        return np.hstack([self.dw_BI_B, dq_BI, control_power])

    def plant_rates(self, t, y):
        """Continuous plant only (sampled-data integrator). Controller wheel torques,
        magnetorquer torque and fault state are held from the last discrete_update
        (zero-order hold), so this is safe to evaluate at any RK stage."""
        w_wheels_input = self._set_state(y)
        T_BI = my_utils.quat_to_dcm(self.q_BI)
        self.wheel_module.calc_state_rates(t, w_wheels_input, self.T_ctr_wheels)
        body_rates = self._calc_body_rates(t, T_BI)
        return np.hstack([body_rates, self.wheel_module.dw_wheels])

    def discrete_update(self, t, y):
        """All sampled blocks, run once per base tick on the accepted state y(t). Each
        block still gates itself on its own t_sample; the integrator guarantees t lands
        exactly on tick times, so the gates fire deterministically."""
        self.fault_module.update(t)
        # refresh derived plant quantities (H_vec, H_total, T_dist, ...) from the
        # accepted state - the last RK stage left them at a trial state
        self.plant_rates(t, y)
        w_wheels_input = y[10:self.wheel_module.num_wheels + 10]
        T_BI = my_utils.quat_to_dcm(self.q_BI)

        self.orbit.calc_orbit_state(t)
        self.update_ref_q(t)

        if self.controller.enable is True:
            if self.config['observer']['feedback_en']:
                f_est = self.wheel_module.D@self.observer_module.f_wheels_est
            else:
                f_est = np.zeros(3)
            self.T_ctr_vec, self.T_ctr_wheels = self.controller.calc_torque_control_output(t, self.q_BI, self.w_BI_B, self.q_RI, self, w_wheels_input, f_est)

        q_RB =  my_utils.quat_error(self.q_RI, self.q_BI)
        qv_RB = np.array([q_RB.x, q_RB.y, q_RB.z])
        self.magt_module.calc_torque(qv_RB, self.w_BI_B, self.H + self.wheel_module.H_vec, t, T_BI)

        # derived quantities consistent with the new held inputs (logging, observers)
        self.plant_rates(t, y)

        self.f_wheels = self.wheel_module.dH_wheels - self.T_ctr_wheels
        self.f_wheels_fault = self._calc_fault_torque()

        if self.observer_module.enable is True:
            self.E_est = self.observer_module.calc_state_estimates(t, w_wheels_input, self.T_ctr_wheels)
            self.E = self.fault_module.E

        self.update_FDIR(t)
        self.logger.store_data(t)

    
    def update_fd(self, t):
        # print("angular")
        if not self.config['FDIR']['satellite']['enable']:
            return
        if my_utils.magnitude(self.w_BI_B) > self.fd_w_max:
            raise DivergentRate(f"Divergent rate detected, w_sat = {self.w_BI_B*my_utils.RAD_TO_DEG} at time {t}")
        q_error = my_utils.quat_error(self.q_RI, self.q_BI)
        err = 2*np.arccos(np.clip(abs(q_error.w), 0.0, 1.0))
        if err <= self.fd_q_max:
            self._fd_q_exceed_t0 = None
        elif self._fd_q_exceed_t0 is None:
            self._fd_q_exceed_t0 = t
        elif t - self._fd_q_exceed_t0 >= self.fd_q_persist:
            raise DivergentAttitude(f"Divergent attitude detected, pointing error {err*my_utils.RAD_TO_DEG:.2f} deg "
                                    f"above {self.fd_q_max*my_utils.RAD_TO_DEG:g} deg for {self.fd_q_persist:.0f} s at time {t}")

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
