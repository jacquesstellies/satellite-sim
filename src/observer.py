from wheels import WheelModule
from fault import Fault

import my_utils as my_utils
import my_globals

import numpy as np

class WheelExtendedStateObserver():
    t_sample = 0
    index = 0
    gain = np.zeros(2)
    M_inertia = None
    M_inertia_inv = None
    wheel = None
    def __init__(self, config, wheel: WheelModule):
        self.config = config
        self.t_sample = config['observer']['t_sample']
        self.index = wheel.index
        self.gain = config['observer']['gain']
        self.friction_coef = wheel.friction_coef
        self.M_inertia_inv = wheel.M_inertia_inv_fast
        self.M_inertia = wheel.M_inertia_fast
        self.wheel = wheel

    F = None
    g = None
    A = None
    b = None
    def calc_system_matrices(self):
        # Calculate the system matrices based on the wheel's inertia
        self.M_inertia_inv = 1/self.M_inertia
        self.A = np.array([[-self.friction_coef * self.M_inertia_inv, self.M_inertia_inv],
                           [0, 0]])
        self.b = np.array([[self.M_inertia_inv],
                           [0]])

        # self.F = np.eye(2) + self.A * self.t_sample + 0.5 * self.A**2 * self.t_sample**2 + 1/6 * self.A**3 * self.t_sample**3
        # self.g = self.t_sample * (np.eye(2) + 0.5 * self.A * self.t_sample + 1/6 * self.A**2 * self.t_sample**2) @ self.b

    e_prev = 0
    e_int = 0
    t_prev = 0
    def calc_state_estimates(self, t, state, u):

        w_est = state[0] # wheel speed
        f_est = state[1] # disturbance
        y = u[1] # measured output
        y_est = w_est

        e = y - y_est  # estimation error

        # print(f"observer {self.index} y = {y}, y_est = {y_est}, e = {e}, u = {u}, t = {t}")
        k_w = self.gain[0]
        k_f = self.gain[1]
        k_fd = self.gain[2] if len(self.gain) > 2 else 0
        k_wd = self.gain[3] if len(self.gain) > 3 else 0
        k_wi = self.gain[4] if len(self.gain) > 4 else 0
        k_fi = self.gain[5] if len(self.gain) > 5 else 0

        # dx_est = (self.A @ x_est).flatten() + self.b.flatten() * u[0] + np.array([k_w * e, k_f * e])

        # dw_est = dx_est[0]
        # df_est = dx_est[1]

        if t - self.t_prev == 0:
            de = 0
        else:
            de = (e - self.e_prev) / (t - self.t_prev)
        self.e_int += e * (t - self.t_prev)
        e_int = self.e_int
        self.e_prev = e

        if abs(u[0]) > self.wheel.T_max:
            u[0] = self.wheel.T_max*my_utils._sign(u[0])

        dw_est = (-self.friction_coef * w_est + u[0]) * self.M_inertia_inv + f_est * self.M_inertia_inv + e*k_w + de*k_wd + e_int*k_wi
        df_est = k_f * e + k_fd*de + k_fi*e_int

        if (w_est >= self.wheel.w_max and dw_est > 0) or (w_est <= -self.wheel.w_max and dw_est < 0):
            dw_est = 0
            df_est = 0

        w_est += dw_est * (t - self.t_prev)
        f_est += df_est * (t - self.t_prev)
        self.t_prev = t
        return  [w_est, f_est, dw_est]

class WheelObserver():
    t_sample = 0
    index = 0
    gain = np.zeros(2)
    M_inertia = None
    M_inertia_inv = None
    wheel = None
    def __init__(self, config, wheel: WheelModule):
        self.config = config
        self.t_sample = config['controller']['t_sample']
        self.index = 0
        self.gain = config['controller']['gain_obs']
        self.friction_coef = wheel.friction_coef
        self.M_inertia = wheel.M_inertia[2][2]
        self.wheel = wheel

        if len(self.gain) != 2:
            raise ValueError("Gain must be a 2-element vector.")
        
        self.M_inertia_inv = 1/self.M_inertia
        
    def calc_state_estimates(self, t, state, u):

        w_est = state # wheel speed
        y = u[1] # measured output
        y_est = w_est

        e = y - y_est  # estimation error
        
        k_w = self.gain[0]

        dw_est = -(self.friction_coef * self.M_inertia_inv) * w_est + u[0] * self.M_inertia_inv + e*k_w

        # T_est = self.M_inertia * dw_est
        # if np.abs(T_est) > self.wheel.T_max:
        #     dw_est = np.sign(dw_est) * self.wheel.T_max / self.M_inertia


        w_est += dw_est * self.t_sample

        # if np.abs(w_est) >= self.wheel.w_max:
        #     w_est = np.sign(w_est) * self.wheel.w_max

        return w_est


class FNDOFaultDetector():
    """Standalone 3-axis FNDO (super-twisting) wheel fault detector.

    Runs alongside the healthy 3-wheel controller (unlike the FNDO inside
    NadafiController, which only exists once the underactuated controller is
    already active). chi_1 converges to the lumped body angular acceleration the
    healthy model cannot explain:

        chi_1 -> J^-1 ( D((I-E)u - u_a) + T_dist )

    A multiplicative wheel fault makes that residual proportional to the
    commanded torque u, while external disturbances are uncorrelated with u.
    So per wheel we fit the residual torque y = D^+ J chi_1 against u with a
    forgetting-factor recursive least squares:

        y_i = theta0*u_i + theta1,   theta0 = 1 - E_i,   theta1 = bias

    Updates are gated on |u_i| >= u_min: a multiplicative fault is unobservable
    without excitation, and dividing by small u is what made the pointwise
    (u + f_est)/u estimate noisy. A wheel is latched as failed when its
    effectiveness estimate stays below E_threshold for confirm_time seconds of
    gated samples; the latch is one-shot so the reconfiguration never feeds the
    estimate back into a loop.

    Identifiability caveat: in closed loop the controller commands u ~ T_dist to
    hold attitude, so on axes where the external disturbance torque dominates
    the maneuver torque the residual is inherently u-correlated and E_est reads
    low even for a healthy wheel (with the Nadafi disturbance model this is the
    case for x and y). Effectiveness is only identifiable where commanded torque
    dominates external torque, so latching is restricted to wheels_monitored
    (default: the wheel the switch_to controller can actually drop). Set
    feed_T_dist = true to pass the simulator's exact disturbance torque to the
    observer model - an ablation showing all wheels become identifiable with
    perfect disturbance knowledge.
    """

    next_t_sample = 0.0

    def __init__(self, config, M_inertia, wheel_module: WheelModule):
        det_config = config['detection']
        self.verbose = config['simulation']['verbose']
        self.t_sample = det_config.get('t_sample', config['controller']['t_sample'])

        # super-twisting observer gains (boundary layer sat() instead of sign():
        # forward Euler + sign chatters at this sample rate)
        self.kappa_0 = det_config.get('kappa_0', 1.0)
        self.kappa_1 = det_config.get('kappa_1', 1.0)
        L = det_config.get('L', 10.0)
        self.L = np.array(L if hasattr(L, '__len__') else [L] * 3, dtype=float)
        self.bl_eps = det_config.get('bl_eps', 1e-2)

        # effectiveness RLS + decision logic
        self.u_min = det_config.get('u_min', 2e-3)
        self.rls_lambda = det_config.get('rls_lambda', 0.998)
        rls_p0 = det_config.get('rls_p0', 10.0)
        self.E_threshold = det_config.get('E_threshold', 0.7)
        self.confirm_time = det_config.get('confirm_time', 2.0)
        self.warmup_time = det_config.get('warmup_time', 2.0)
        # only latch on wheels whose axis the underactuated controller can drop
        # (and where torque excitation, not disturbance rejection, dominates)
        default_monitored = [1] if det_config.get('switch_to', 'Nadafi_FNDO') == 'Zarourati' else [2]
        self.wheels_monitored = det_config.get('wheels_monitored', default_monitored)
        self.feed_T_dist = det_config.get('feed_T_dist', False)

        # semi-implicit chi_1 update is unconditionally stable, but warn if the
        # chi_0 correction is explicit-Euler unstable inside the boundary layer
        chi_0_gain = self.kappa_0 * np.sqrt(self.L.max() * self.bl_eps) * self.t_sample / self.bl_eps
        if chi_0_gain > 1.5:
            print(f"WARNING: FNDO detector chi_0 boundary-layer gain {chi_0_gain:.2f} > 1.5, "
                  "reduce kappa_0*sqrt(L) or increase bl_eps")

        self.J = np.array(M_inertia)
        self.J_inv = np.linalg.inv(self.J)
        self.D = np.array(wheel_module.D)
        self.D_pinv = np.linalg.pinv(self.D)
        self.num_wheels = wheel_module.num_wheels

        self.chi_0 = np.zeros(3)
        self.chi_1 = np.zeros(3)
        self.theta = np.zeros((self.num_wheels, 2))  # per wheel [1 - E, torque bias]
        self.P = np.stack([np.eye(2) * rls_p0 for _ in range(self.num_wheels)])
        self.E_est = np.ones(self.num_wheels)
        self.low_time = np.zeros(self.num_wheels)
        self.latched_wheel = -1
        self.latch_time = None
        self.reconfigured = False

    def update(self, t, w, u_wheels, H_wheels_vec, T_magt, T_dist=None):
        if t >= self.next_t_sample:
            self.next_t_sample += self.t_sample
        else:
            return

        w = np.asarray(w, dtype=float).flatten()
        u = np.asarray(u_wheels, dtype=float).flatten()

        # everything the healthy model cannot explain accumulates in chi_1
        T_known = -self.D @ u \
            - my_utils.cross_product_M31M31(w, self.J @ w + np.asarray(H_wheels_vec).flatten()) \
            + np.asarray(T_magt).flatten()
        if self.feed_T_dist and T_dist is not None:
            T_known = T_known + np.asarray(T_dist).flatten()
        acc_known = self.J_inv @ T_known

        e0 = self.chi_0 - w
        sat_e0 = np.clip(e0 / self.bl_eps, -1.0, 1.0)
        v_0 = -self.kappa_0 * np.sqrt(self.L * np.abs(e0)) * sat_e0 + self.chi_1
        self.chi_0 = self.chi_0 + (v_0 + acc_known) * self.t_sample

        # semi-implicit (Acary-Brogliato) update of chi_1 -> v_0: explicit Euler
        # is unstable whenever kappa_1*L*dt > bl_eps and sawtooths at +-kappa_1*L*dt
        g = self.kappa_1 * self.L * self.t_sample
        z = self.chi_1 - v_0
        z_new = np.where(np.abs(z) >= self.bl_eps + g,
                         z - g * np.sign(z),
                         z * self.bl_eps / (self.bl_eps + g))
        self.chi_1 = v_0 + z_new

        if t < self.warmup_time:
            return

        # per-wheel residual torque: y_i = (1-E_i)*u_i - u_a_i + disturbance
        y = self.D_pinv @ (self.J @ self.chi_1)

        for i in range(self.num_wheels):
            if abs(u[i]) < self.u_min:
                continue
            phi = np.array([u[i], 1.0])
            P = self.P[i]
            K = (P @ phi) / (self.rls_lambda + phi @ P @ phi)
            self.theta[i] += K * (y[i] - phi @ self.theta[i])
            self.P[i] = (P - np.outer(K, phi @ P)) / self.rls_lambda
            self.E_est[i] = 1.0 - np.clip(self.theta[i][0], 0.0, 1.0)

            if self.latched_wheel < 0 and i in self.wheels_monitored:
                if self.E_est[i] < self.E_threshold:
                    self.low_time[i] += self.t_sample
                    if self.low_time[i] >= self.confirm_time:
                        self.latched_wheel = i
                        self.latch_time = t
                        if self.verbose:
                            print(f"FNDO detection: wheel {i} effectiveness {self.E_est[i]:.2f} "
                                  f"below {self.E_threshold} for {self.confirm_time}s, latched at t={t:.2f}")
                else:
                    self.low_time[i] = 0.0


class ObserverModule():

    wheel_module: WheelModule = None
    config : dict = None
    fault : Fault = None
    t_sample : float = 0
    wheel_extended_state_observers : list[WheelExtendedStateObserver] = []
    E_mul : np.array = None # multiplicative actuator effectiveness matrix

    w_wheels_est : np.array = None # wheel speed estimate
    f_wheels_est : np.array = None # disturbance torque estimate
    dw_wheels_est : np.array = None # wheel acceleration estimate


    def __init__(self, config : dict, wheel_module: WheelModule, fault: Fault=None):
        self.config = config
        self.fault = fault
        self.wheel_module = wheel_module
        self.t_sample : float = config['observer']['t_sample']
        self.wheel_extended_state_observers = []
        self.enable = config['observer']['enable']

        for i, wheel in enumerate(wheel_module.wheels):
            eso = WheelExtendedStateObserver(config, wheel)
            self.wheel_extended_state_observers.append(eso)

        self.w_wheels_est = np.zeros(wheel_module.num_wheels)
        self.f_wheels_est = np.zeros(wheel_module.num_wheels)
        self.dw_wheels_est = np.zeros(wheel_module.num_wheels)
        self.E_mul = np.eye(self.wheel_module.num_wheels)
        
    next_t_sample = 0
    dE = 0
    def calc_state_estimates(self, t : float, state : list[float], u_wheels : list[float]):
        if t >= self.next_t_sample:
            # if self.config['simulation']['test_mode_en'] is True:
            #     print("Observer t:", t)
            self.next_t_sample += self.t_sample
        else:
            return self.E_mul
        
        for i, wheel_extended_state_observer in enumerate(self.wheel_extended_state_observers):
            [self.w_wheels_est[i], self.f_wheels_est[i], self.dw_wheels_est[i]] \
                = wheel_extended_state_observer.calc_state_estimates(t, [self.w_wheels_est[i], self.f_wheels_est[i]], [u_wheels[i], state[i]])

            # Only update E if there was a control input
            if u_wheels[i] != 0:
                E_mul_temp = (u_wheels[i] + self.f_wheels_est[i])/u_wheels[i]
                if E_mul_temp < 1 and E_mul_temp > 0 and np.abs(u_wheels[i]) >1e-2:
                    self.E_mul[i][i] = E_mul_temp

        return self.E_mul