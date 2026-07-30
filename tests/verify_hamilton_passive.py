"""Locks the Hamilton passive-rotation convention used across the sim.

Run: python scripts/verify_hamilton_passive.py  (from repo root or src/)
Verifies my_utils.quat_to_dcm / quat_error against scipy and the passive
DCM algebra, plus the plant kinematics dq = 0.5 q_BI (x) w_BI_B.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion)
from scipy.spatial.transform import Rotation
import my_utils

rng = np.random.default_rng(0)

def rand_q():
    return np.quaternion(*rng.standard_normal(4)).normalized()

def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name

print("Hamilton passive-rotation convention checks")

# 1) passive DCM == scipy active transpose, for many random q
ok = all(
    np.allclose(my_utils.quat_to_dcm(q := rand_q()),
                Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().T)
    for _ in range(100)
)
check("quat_to_dcm(q) == Rotation.from_quat(q).as_matrix().T", ok)

# 2) quat_to_dcm is orthonormal, det +1
q = rand_q(); T = my_utils.quat_to_dcm(q)
check("quat_to_dcm orthonormal", np.allclose(T @ T.T, np.eye(3)))
check("quat_to_dcm det +1", np.isclose(np.linalg.det(T), 1.0))

# 3) quat_error contract: A(quat_error(q_TI,q_FI)) == A(q_TI) @ A(q_FI).T
for _ in range(100):
    q_TI, q_FI = rand_q(), rand_q()
    lhs = my_utils.quat_to_dcm(my_utils.quat_error(q_TI, q_FI))
    rhs = my_utils.quat_to_dcm(q_TI) @ my_utils.quat_to_dcm(q_FI).T
    assert np.allclose(lhs, rhs)
check("A(quat_error(q_TI,q_FI)) == T_TI @ T_FI.T", True)

# 4) identity at zero error
q = rand_q(); e = my_utils.quat_error(q, q)
check("quat_error(q,q) == identity", np.allclose([e.w, e.x, e.y, e.z], [1, 0, 0, 0]))

# 4b) dcm_to_quat is the inverse of quat_to_dcm
for _ in range(100):
    T = my_utils.quat_to_dcm(rand_q())
    assert np.allclose(my_utils.quat_to_dcm(my_utils.dcm_to_quat(T)), T)
check("quat_to_dcm(dcm_to_quat(T)) == T", True)

# 5) plant kinematics: integrating dq = 0.5 q_BI (x) w_BI_B under a body-z spin
#    yields A(q_BI) mapping the body x-axis into inertial swept toward +y (C_{I<-B}=T_BI.T)
Omega, T_end, dt = 0.3, 1.0, 1e-4
w_BI_B = np.array([0, 0, Omega])
q_BI = np.quaternion(1, 0, 0, 0)
for _ in range(int(T_end / dt)):
    dq = 0.5 * q_BI * np.quaternion(0, *w_BI_B)
    q_BI = (q_BI + dq * dt).normalized()
theta = Omega * T_end
# body x-axis (1,0,0)_B expressed in inertial = T_BI.T @ x = T_IB @ x
x_in_I = my_utils.quat_to_dcm(q_BI).T @ np.array([1, 0, 0])
check("kinematics: body-x sweeps to +y in inertial",
      np.allclose(x_in_I, [np.cos(theta), np.sin(theta), 0], atol=1e-3))

print("ALL CHECKS PASSED")
