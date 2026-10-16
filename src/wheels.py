import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
import math

import my_utils as my_utils
import my_globals
from collections import deque

from fault import Fault

class Wheel():
    mass = 0
    M_inertia = np.zeros((3,3))
    M_inertia_inv = np.zeros((3,3))
    M_inertia_fast : float = 0
    M_inertia_inv_fast : float = 0

    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    d = np.zeros(3) # direction vector
    friction_coef = 0.0
    coulomb_torque = 0.0
    coulomb_smoothing_speed = 0.1
    config = None
    index = 0

    _faults : list[Fault] = None 
    w_max = 0
    T_max = 0

    w : float = 0.0
    dw : float = 0.0
    H : float = 0.0
    dH : float = 0.0

    controller_enable = True
    t_sample = None

    _noise_std = 0

    def __init__(self, config, faults, index=0):
        self.config = config
        self.mass = self.config['wheels']['mass']
        self.dimensions['radius'] = self.config['wheels'].get('radius', 0.066)
        self.dimensions['height'] = self.config['wheels'].get('height', 0.025)
        self.w_max = my_utils.conv_rpm_to_rads_per_sec(self.config['wheels']['max_speed_rpm'])
        self.T_max = self.config['wheels']['max_torque']
        self._faults = faults
        self.index = index
        self.t_sample = self.config['controller']['t_sample']
        self.friction_coef = self.config['wheels']['friction_coef']
        # tanh-smoothed Coulomb friction: T_c*tanh(w/w_s). A hard sign(w) chatters across
        # zero speed under fixed-step integration; w_s sets the width of the smoothing
        # (see Simulation._simulate_sampled for the step-size check).
        self.coulomb_torque = self.config['wheels'].get('coulomb_torque', 0.0)
        self.coulomb_smoothing_speed = self.config['wheels'].get('coulomb_smoothing_speed', 0.1)
        self.fd_inst_time_threshold = self.config['FDIR']['wheels']['inst_time_threshold']
        self.fd_inst_count_threshold = self.config['FDIR']['wheels']['inst_count_threshold']
        if self.config['wheels']['use_inertia']:
            self.M_inertia[0][0] = 0
            self.M_inertia[1][1] = 0
            self.M_inertia[2][2] = self.config['wheels']['M_inertia']
            self.M_inertia_fast = self.M_inertia[2][2]
            # self.M_inertia_inv = np.linalg.inv(self.M_inertia)
            self.M_inertia_inv_fast = 1/self.M_inertia_fast
        else:
            self.calc_M_inertia()

        target_noise_db = 2
        target_noise_watts = 10 ** (target_noise_db / 10)
        self._noise_std = math.sqrt(target_noise_watts)
        if self.config['wheels']['saturation_enable']:
            self.T_max = self.config['wheels']['max_torque']
        else:
            self.T_max = float('inf')

        fd_inst = (False, 0)
    
    def calc_M_inertia(self):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)

        self.M_inertia_inv = np.linalg.inv(self.M_inertia)
        self.M_inertia_fast = self.M_inertia[2][2]  # For fast calculations, we only need the z-axis inertia
        self.M_inertia_inv_fast = self.M_inertia_inv[2][2]  # For fast calculations, we only need the z-axis inertia


    def calc_state_rates(self, t : float, state : float, u : float):
        self.w = state

        # Apply fault if enabled (self updated in fault class)
        for fault in self._faults:
            u *= fault.E[self.index][self.index]
            u += fault.u_a[self.index]

        # Check torque limit exceeded
        if abs(u) >= self.T_max:
            u = self.T_max*my_utils._sign(u)

            
        # Calculate the new wheel speed derivative
        T_friction = self.friction_coef*self.w
        if self.coulomb_torque != 0.0:
            T_friction += self.coulomb_torque*np.tanh(self.w/self.coulomb_smoothing_speed)
        self.dw = (u - T_friction)*self.M_inertia_inv_fast
        
        if (self.w >= self.w_max and self.dw > 0) or (self.w <= -self.w_max and self.dw < 0):
            # print("wheel speed limit reached")
            self.dw = 0
        
        self.dH = self.dw*self.M_inertia_fast
        self.H = self.w*self.M_inertia_fast
        # return self.calc_sensor_output(np.array([dw,w]))
        # return dw

    def calc_sensor_output(self, state: np.array):
        noise = np.random.normal(0, self._noise_std, size=len(state))
        return state + noise

    
    def calc_state_outputs(self, t, state):
        dw = state[0]
        w = state[1]
        H = w*self.M_inertia_fast
        dH = dw*self.M_inertia_fast
        return np.array([dH, H])

    e_prev = 0
    def calc_state_speed_control(self, t, state, w_ref):
        w = state[0]
        kp = self.config['controller']['kp_wheel_w']
        kd = self.config['controller']['kd_wheel_w']
        ki = self.config['controller']['ki_wheel_w']

        e = w_ref - w
        de = (e - self.e_prev)/self.t_sample
        e_int = e+self.e_prev*self.t_sample
        u = (kp*e + kd*de + ki*e_int)*self.M_inertia_fast

        if abs(u) > self.T_max:
            u = self.T_max*my_utils._sign(u)

        self.e_prev = e
        state = self.calc_state_rates(t, state, u)

        return state, u
    

    # fd_inst_en = False
    # fd_inst_count = 0
    fd_inst_count_threshold = 0
    fd_inst_time_threshold = 0
    fd_inst_list = deque(maxlen=50)
    dH_prev = 0
    def update_fd(self, t):
        
        if not self.config['FDIR']['wheels']['enable']:
            return
        
        # Check wheel instability
        if my_utils._sign(self.dH_prev) != my_utils._sign(self.dH):
            self.fd_inst_list.append(t)
            
            if self.fd_inst_list[0] < t - self.fd_inst_time_threshold:
                self.fd_inst_list.popleft()

            if len(self.fd_inst_list) >= self.fd_inst_count_threshold:
                raise Exception(f"Wheel instability detected at time {t}")
            
        self.dH_prev = self.dH

class WheelModule():
    wheels = None
    layout = None
    config = None
    D = None
    D_psuedo_inv = None
    _D_psuedo_inv_local = None
    num_wheels = 0
    
    def __init__(self, config,faults=None):
        self.layout = config['wheels']['config']
        self.config = config
        self.num_wheels, self.D = my_utils.get_wheel_layout(config)
        
        self.wheels = [Wheel(config,faults,i) for wheel, i in enumerate(range(self.num_wheels))]
        for i, wheel in enumerate(self.wheels):
            wheel.w_max = my_utils.conv_rpm_to_rads_per_sec(config['wheels']['max_speed_rpm'])
            wheel.T_max = config['wheels']['max_torque']
            wheel.d = np.transpose(self.D)[i]

        # self.D_psuedo_inv = np.linalg.pinv(self.D)
        if self.D.shape[1] != 3:
            self.D_psuedo_inv = np.linalg.pinv(self.D)
        else:
            self.D_psuedo_inv = np.linalg.inv(self.D)

        self._D_psuedo_inv_local = self.D_psuedo_inv

        if config['controller']['type'] == "backstepping" and config['controller']['sub_type'] == "Shen":
            self._D_psuedo_inv_local = np.eye(self.num_wheels)
        
        self.w_wheels = np.array(self.num_wheels)
        self.dw_wheels = np.array(self.num_wheels)
        self.H_wheels = np.array(self.num_wheels)
        self.dH_wheels = np.array(self.num_wheels)
        self.H_vec = np.zeros(3)
        self.dH_vec = np.zeros(3)

        self.fd_inst = np.zeros(self.num_wheels)

    w_wheels = None
    dw_wheels = None
    H_wheels = None
    dH_wheels = None
    H_vec = None
    dH_vec = None
    def calc_state_rates(self, t, state, u : np.array):
        # dw = state[:self.num_wheels]
        # w = state[self.num_wheels:]
        if len(u) != self.num_wheels:
            raise Exception(f"invalid control torque length {len(u)}")
            
        for wheel in self.wheels:
            wheel.calc_state_rates(t, state[wheel.index], u[wheel.index])

        self.w_wheels = np.array([wheel.w for wheel in self.wheels])
        self.dw_wheels = np.array([wheel.dw for wheel in self.wheels])
        self.H_wheels = np.array([wheel.H for wheel in self.wheels])
        self.dH_wheels = np.array([wheel.dH for wheel in self.wheels])

        self.H_vec = self.D @ self.H_wheels
        self.dH_vec = self.D @ self.dH_wheels

    def min_feasible_slew_duration(self, M_inertia, axis, angle_rad, wheel_idx,
                                   peak_rate_coef, peak_accel_coef, margin=0.5,
                                   use_speed_headroom=True):
        """Shortest duration for a single-axis slew that stays inside wheel limits.

        A rest-to-rest slew of `angle_rad` about body axis `axis` (a unit 3-vector) over
        duration T, with a timing profile whose peak |s'| / |s''| over the normalised leg
        are `peak_rate_coef` / `peak_accel_coef`, demands

            peak body rate   w = peak_rate_coef  * angle / T       (about `axis`)
            peak body accel  a = peak_accel_coef * angle / T^2

        so wheel i must store momentum |(J @ axis)[i]|*w and deliver torque
        |(J @ axis)[i]|*a - the off-diagonal inertia terms are why a pure-x slew still
        loads the y wheel. Each constraint is inverted for T (momentum ~ 1/T, torque ~
        1/T^2) and the binding one over `wheel_idx` (the actuated wheels) is returned.

        `margin` is the fraction of each limit the *reference* may ask for, leaving the
        rest for feedback effort, disturbances and the gyroscopic w x H term this
        open-loop sizing neglects. Assumes diagonal D (ortho layout: wheel i <-> body
        axis i), matching the nf_idx-as-body-axis assumption the UATCs already make.

        Returns (T_min, binding) where `binding` names the constraint that set it.
        """
        if not np.allclose(self.D, np.diag(np.diag(self.D))):
            raise Exception("min_feasible_slew_duration: only supports diagonal D (ortho layout)")

        angle = abs(float(angle_rad))
        T_min, binding = 0.0, "none"
        if angle == 0.0:
            return T_min, binding

        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        H_axis = M_inertia @ axis      # body momentum/torque direction for this slew

        for i in wheel_idx:
            wheel = self.wheels[i]
            coupling = abs(H_axis[i])
            if coupling == 0.0:
                continue

            # Momentum: headroom shrinks as the wheel spins up, so a maneuver that was
            # feasible from rest may not be from a biased wheel.
            if use_speed_headroom:
                dw_avail = max(wheel.w_max - abs(wheel.w), 0.0)
            else:
                dw_avail = wheel.w_max
            H_avail = margin * dw_avail * wheel.M_inertia_fast
            if H_avail <= 1e-12:
                return float('inf'), "wheel {:d} momentum (no headroom, w={:.1f}/{:.1f} rad/s)".format(
                    i, wheel.w, wheel.w_max)
            T_mom = coupling * peak_rate_coef * angle / H_avail
            if T_mom > T_min:
                T_min, binding = T_mom, "wheel {:d} momentum".format(i)

            T_avail = margin * wheel.T_max
            if np.isfinite(T_avail) and T_avail > 0.0:
                T_tor = np.sqrt(coupling * peak_accel_coef * angle / T_avail)
                if T_tor > T_min:
                    T_min, binding = T_tor, "wheel {:d} torque".format(i)

        return T_min, binding

    # calc_state_rates_optmized

    def calc_state_outputs(self, t, state):
        dw_wheels = state[:self.num_wheels]
        w_wheels = state[self.num_wheels:]
        dH_module = np.zeros(3)
        H_module = np.zeros(3)
        for wheel in self.wheels:
            y = wheel.calc_state_outputs(t, [dw_wheels[wheel.index], w_wheels[wheel.index]])
            # dH_wheels = np.zeros(self.num_wheels)
            # H_wheels = np.zeros(self.num_wheels)
            dH_module += y[0]*wheel.d 
            H_module += y[1]*wheel.d
        # return np.concatenate([dH_module, H_module, dH_wheels, H_wheels])
        return np.concatenate([dH_module, H_module])