import numpy as np
from orbit import Orbit
import my_utils

class MagtModule():
    config : dict = None
    orbit : Orbit = None
    T = np.zeros(3)
    m = np.zeros(3)
    B_B = np.zeros(3)
    mode = None

    def __init__(self, config, orbit):
        self.config = config
        self.orbit = orbit
        magt_cfg = self.config['magt']
        self.enable = magt_cfg['enable']
        self.m_max = magt_cfg.get('mag_moment_max', 0.0)
        self.t_sample = magt_cfg.get('t_sample', 0.0)
        # Momentum-dump gain [1/s]: with T_des = -k.H this *is* the commanded decay
        # rate. Because tau = m x B is always perpendicular to B, the dump only works
        # by exploiting the rotation of B in the body frame (~2x orbit rate, i.e.
        # ~2.2e-3 rad/s at LEO). k must therefore be SLOWER than that sweep rate:
        # k >~ 1/t_sample is deadbeat and parks the residual momentum exactly along B,
        # where it is invisible to the actuator and never decays.
        self.k_dump = magt_cfg.get('dump_gain', 3e-3)
        self.T_dump_max = magt_cfg.get('dump_torque_max', 0.1)
        # Dead-axis (unactuated by the wheels) yaw regulator, used by the
        # "split_yaw_dump" mode. The wheels cannot change (H_B)_dead at all, so
        # theta_dead is the *double* integral of the dead-axis torque with nothing
        # restoring it: a body-fixed disturbance of only ~2e-6 Nm integrates to tens
        # of degrees over an orbit. The magnetorquer is the sole actuator on that
        # axis, so close a PD loop on it directly instead of relying on the isotropic
        # momentum dump to incidentally keep the axis clean.
        self.dead_axis = int(magt_cfg.get('dead_axis', 2))
        self.yaw_wn = magt_cfg.get('yaw_wn', 0.01)      # rad/s, >> orbit rate
        self.yaw_zeta = magt_cfg.get('yaw_zeta', 0.7)
        M_inertia = np.array(self.config.get('satellite', {}).get('M_Inertia', np.eye(3)), dtype=float)
        if M_inertia.shape == (3,):
            M_inertia = np.diag(M_inertia)
        self.J_dead = float(M_inertia[self.dead_axis, self.dead_axis])
        physical_cfg = magt_cfg.get('physical', {})
        self.physical = bool(physical_cfg.get('enable', False)) if isinstance(physical_cfg, dict) else False
        if self.enable:
            self.mode = magt_cfg['mode']
        self.T = np.zeros(3)
        self.m = np.zeros(3)
        self.B_B = np.zeros(3)
        self.next_t_sample = 0.0
        self.prev = np.zeros(3)
        self.T_dead_req = 0.0

    def _physical_torque(self, T_des, B_B):
        """Map a desired torque onto the achievable magnetorquer set τ = m × B.

        The unconstrained moment that realises the component of T_des perpendicular
        to B is m = (B × T_des) / |B|². The moment is then scaled (not clipped
        per-axis) to keep |m| <= mag_moment_max [A.m²]: clipping each axis
        independently rotates m away from B × T_des, so the resulting τ = m × B is
        no longer anti-parallel to the momentum being removed and can add momentum
        on some axes. Scaling preserves the torque direction and only reduces its
        magnitude. The physical torque is then recomputed from the applied moment.
        """
        B2 = float(B_B @ B_B)
        if B2 < 1e-24:
            return np.zeros(3), np.zeros(3)
        m = np.asarray(my_utils.cross_product_M31M31(B_B, T_des), dtype=float) / B2
        m_norm = float(np.linalg.norm(m))
        if self.m_max > 0.0 and m_norm > self.m_max:
            m = m * (self.m_max / m_norm)
        T = my_utils.cross_product_M31M31(m, B_B)
        return T, m

    def _physical_torque_split(self, T_in_plane, T_dead, B_B):
        """Allocate m so the dead-axis torque is delivered exactly and the in-plane
        dump uses only the freedom that is left over.

        The dead-axis torque is tau.e_d = (m x B).e_d = m.(B x e_d), so with
        u = B x e_d, only the component of m along u produces any dead-axis torque
        and every m perpendicular to u produces none. Splitting the moment on u
        therefore decouples the two objectives exactly.

        Summing the two desired torques and running them through the ordinary
        (B x T_des)/|B|^2 does NOT do this: the in-plane dump command then leaks a
        dead-axis torque of order 3e-5 Nm, more than ten times the ~2e-6 Nm
        disturbance the dead-axis loop is trying to reject, and the yaw error settles
        where the PD command balances that leakage instead of the disturbance.

        m_yaw is parallel to u and m_dump is perpendicular to it, so saturation with
        the dead axis given priority has a closed form.
        """
        B2 = float(B_B @ B_B)
        if B2 < 1e-24:
            return np.zeros(3), np.zeros(3)
        e_d = np.zeros(3)
        e_d[self.dead_axis] = 1.0
        u = np.asarray(my_utils.cross_product_M31M31(B_B, e_d), dtype=float)
        u2 = float(u @ u)

        m_dump = np.asarray(my_utils.cross_product_M31M31(B_B, T_in_plane), dtype=float) / B2
        if u2 > 1e-30:
            # strip the part of the dump moment that would twist the dead axis
            m_dump = m_dump - (float(m_dump @ u) / u2) * u
            m_yaw = (T_dead / u2) * u
        else:
            # B is parallel to the dead axis: no authority there this cycle
            m_yaw = np.zeros(3)

        n_yaw = float(np.linalg.norm(m_yaw))
        if self.m_max > 0.0:
            if n_yaw > self.m_max:
                m_yaw = m_yaw * (self.m_max / n_yaw)
                m_dump = np.zeros(3)
            else:
                n_dump = float(np.linalg.norm(m_dump))
                budget = np.sqrt(max(self.m_max**2 - n_yaw**2, 0.0))
                if n_dump > budget:
                    m_dump = m_dump * (budget / n_dump)
        m = m_yaw + m_dump
        T = my_utils.cross_product_M31M31(m, B_B)
        return T, m

    def calc_torque(self, q_err_vec, w_sat, H_sat, t, T_BI=None):
        if not self.enable:
            self.T = np.zeros(3)
            self.m = np.zeros(3)
            return
        if self.physical and self.t_sample > 0.0 and t < self.next_t_sample:
            return
        if self.physical and self.t_sample > 0.0:
            self.next_t_sample = t + self.t_sample

        ## Calculate magnetic torque command (desired, possibly unphysical)
        match self.mode:
            case "z-axis_simple":
                self.T = np.zeros(3)
                k = 1
                self.T[2] = -1 * my_utils._sign(w_sat[2]) * k
                self.T[2] = my_utils.low_pass_filter(self.T[2], self.prev[2], 0.5)
            case "momentum_dump_xyz":
                self.T = -1 * my_utils.sat_vec(self.k_dump*np.asarray(H_sat, dtype=float), self.T_dump_max)
            case "momentum_dump_xy_axis":
                self.T = -1 * my_utils.sat_vec(self.k_dump*np.asarray(H_sat, dtype=float), self.T_dump_max)
                self.T = my_utils.low_pass_filter(self.T, self.prev, 0.2)
                self.T[2] = 0
            case "split_yaw_dump":
                # Two orthogonal objectives on one actuator.
                #
                # Dead axis: PD on the attitude error. q_err_vec is the vector part
                # of q_RB, and q_err_vec[d] ~ -theta/2, so +2*J*wn^2*q_err_vec[d]
                # realises -J*wn^2*theta. The rate term uses (H_B)_d directly, since
                # w_d = (H_B)_d / J_d when the dead wheel carries no momentum.
                # Closed loop: theta_ddot + 2*zeta*wn*theta_dot + wn^2*theta = T_dist/J.
                #
                # Remaining axes: the slow isotropic dump, which has a whole orbit to
                # work and must stay well below the rate at which B sweeps through the
                # body frame (see dump_gain).
                H = np.asarray(H_sat, dtype=float)
                d = self.dead_axis
                T_dead = (-2.0*self.yaw_zeta*self.yaw_wn*H[d]
                          + 2.0*self.J_dead*self.yaw_wn**2*q_err_vec[d])
                H_in_plane = H.copy()
                H_in_plane[d] = 0.0
                self.T = -1 * my_utils.sat_vec(self.k_dump*H_in_plane, self.T_dump_max)
                # No separate torque cap here: mag_moment_max in the allocator is the
                # real (and only physical) limit, and it saturates with the dead axis
                # given priority. Clipping the torque first would distort the PD law
                # and cap how much authority yaw can claim, which is the opposite of
                # the intended priority.
                self.T[d] = T_dead
                # keep the two commands apart so they can be allocated independently
                self.T_dead_req = T_dead
            case "momentum_dump_w_z_axis_simple":
                kz = 0.001

                # this mode keeps its own tighter limit; the z axis is driven separately
                self.T = my_utils.sat_vec(self.k_dump*np.asarray(H_sat, dtype=float), 0.01)
                self.T[2] = -1 * my_utils._sign(q_err_vec[2]) * kz

        # Retain the desired (pre-projection) torque so low_pass_filter actually
        # filters; self.prev was previously initialised to 0 and never updated,
        # which turned the filter into a constant scale factor.
        self.prev = np.array(self.T, dtype=float, copy=True)

        if self.physical:
            if T_BI is None:
                raise ValueError("physical magnetorquer torque requires T_BI")
            self.B_B = T_BI @ self.orbit.B_I
            if self.mode == "split_yaw_dump":
                T_in_plane = np.array(self.T, dtype=float, copy=True)
                T_in_plane[self.dead_axis] = 0.0
                self.T, self.m = self._physical_torque_split(T_in_plane, self.T_dead_req, self.B_B)
            else:
                self.T, self.m = self._physical_torque(self.T, self.B_B)
        else:
            self.m = np.zeros(3)
