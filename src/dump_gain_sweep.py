"""Free-decay replay of the magnetic momentum-dump law over a sweep of gains.

Integrates h_{k+1} = h_k + dt * (m x B) with m = (B x T_des)/|B|^2, T_des = -k h,
|m| scaled to mag_moment_max, along the orbit's field history (inertial frame).
The torque is frame-independent and the wheels cannot change h, so attitude does
not enter. No disturbance torques.
    python dump_gain_sweep.py [h0_norm_Nms]
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys, tomllib, numpy as np
sys.path.insert(0, 'src')
from orbit import Orbit
import my_utils

cfg = tomllib.load(open('src/configs/Nadafi_FNDO_EOSSAT_magt.toml', 'rb'))
m_max = cfg['magt']['mag_moment_max']
T_dump_max = cfg['magt'].get('dump_torque_max', 0.1)

class L:
    def log(self, *a, **k): pass

orb = Orbit(cfg, L())
T = orb.period; dt = 1.0
N = int(5*T/dt) + 1
orb.t_sample = dt
B = np.zeros((N, 3))
for i in range(N):
    orb.calc_orbit_state(i*dt); B[i] = orb.B_I

h0_norm = float(sys.argv[1]) if len(sys.argv) > 1 else np.sqrt(3)*0.05
h0 = h0_norm*np.ones(3)/np.sqrt(3)
print(f"period {T:.1f} s, |h0| = {h0_norm:.4g} Nms, m_max = {m_max} Am2")

ks = [1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
nh_all = {}
print(f"{'k':>8} {'k dt':>6} {'|h|@2':>10} {'|h|@5':>10} {'peak |m|':>9}")
for k in ks:
    h = h0.copy(); nh = np.zeros(N); m_peak = 0.0
    for i in range(N):
        nh[i] = np.linalg.norm(h)
        b = B[i]
        T_des = -my_utils.sat_vec(k*h, T_dump_max)
        m = np.cross(b, T_des)/(b @ b)
        mn = np.linalg.norm(m)
        if mn > m_max:
            m *= m_max/mn
        m_peak = max(m_peak, min(mn, m_max))
        h = h + dt*np.cross(m, b)
    nh_all[k] = nh
    i2, i5 = int(2*T), int(5*T)
    print(f"{k:8.0e} {k*dt:6.3f} {nh[i2]:10.3e} {nh[i5]:10.3e} {m_peak:9.2f}")

t = np.arange(N)*dt
fig, ax = plt.subplots(figsize=(6.5, 4))
for k in ks:
    ax.semilogy(t/T, np.maximum(nh_all[k]/h0_norm, 1e-12), label=rf'$k={k:g}$')
ax.set_xlabel('Orbits'); ax.set_ylabel(r'$\|\mathbf{h}\|/\|\mathbf{h}_0\|$')
ax.set_ylim(1e-12, 2); ax.grid(alpha=.3, which='both'); ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig('dump_gain_sweep.pdf'); fig.savefig('dump_gain_sweep.png', dpi=150)
