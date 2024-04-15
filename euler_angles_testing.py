import numpy as np
import math
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.spatial.transform import Rotation
import quaternion

def XYZ_to_ypr(inertial_angles):
    psi = inertial_angles[0]
    theta = inertial_angles[1]
    phi = inertial_angles[2]

    R1 = np.array([[1, 0, 0],
          [0, cos(psi), sin(psi)],
          [0, -sin(psi), cos(psi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
          [0, 1, 0],
          [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(phi), sin(phi), 0],
          [-sin(phi), cos(phi), 0],
          [0, 0, 1]])
    dcm = R1@R2@R3
    print("my dcm: ")
    print(np.array(dcm))

    r = Rotation.from_euler('XYZ', inertial_angles)
    print(r.as_matrix())

    # r = Rotation.from_matrix(dcm)
    # psi,theta,phi = r.as_euler('zyx')
    # print("scipy dcm: ")
    # r = Rotation.from_euler('XYZ', inertial_angles)
    # print((180/math.pi) * r.as_euler('zyx'))
    # print(np.transpose(r.as_matrix())@r.as_matrix())
    yaw, pitch, roll = r.as_euler('zyx')

    return [yaw, pitch, roll]


def dcm_to_yaw_pitch_roll(dcm):
    alpha = np.arctan(dcm[0][1]/dcm[0][0])
    beta = np.arcsin(-dcm[0][2])
    gamma = np.arctan(dcm[1][2]/dcm[2][2])
    return [alpha, beta, gamma]

# dcm = np.array([[0.64050, 0.75309, -0.15038],
#                 [0.76737, -0.63530, 0.086823],
#                 [0.30152, 0.17101, 0.98481]])

def quaternion_to_yaw_pitch_roll(q : np.quaternion):
    
    r = Rotation.from_quat([q.x, q.y, q.z, q.w])

    dcm = r.as_matrix()

    return r.as_euler('zyx', degrees=True)

# dcm = np.array([ [0.64050, 0.75309, -0.15038],
#         [0.76737, -0.63530, 0.086823],
#         [-0.30152, -0.17101, -0.98481]])

# r = Rotation.from_matrix(dcm)
# print(r.as_euler('zyx', degrees=True))
# # print(np.transpose(dcm)@dcm)

# # print(list(map(math.degrees, dcm_to_yaw_pitch_roll(dcm))))
# # XYZ_to_ypr(list(map(math.radians,[30, 45, 30])))
# # q = np.quaternion(1, 1, 1, 0)
# q = r.as_quat(r)
# quat = np.quaternion(q[3],q[0],q[1],q[2])

# print(quaternion_to_yaw_pitch_roll(quat))

def calc_control_output(q, angular_v):
    k = 1
    c1 = 1
    c2 = 1
    c3 = 1
    K = np.diag(np.full(3,k))
    C = np.diag([c1, c2, c3])

    q_vec = np.array([q.x, q.y, q.z])
    angular_v = np.array(angular_v)

    M_controller = -K@q_vec - C@angular_v
    return M_controller

# M = calc_control_output(np.quaternion(1,0,0,1), [0,0,1])
# print(M)

def rotate_M_inertia(M_inertia : np.array, direction : np.array):
    dir = Rotation.from_euler('xyz', direction, degrees=True)
    dcm = dir.as_matrix()
    
    return dcm@M_inertia@np.transpose(dcm)

print(rotate_M_inertia(np.diag([1,1,3]), [90,0,0]))
print(rotate_M_inertia(np.diag([1,1,3]), [0,90,0]))
print(rotate_M_inertia(np.diag([1,1,3]), [0,0,90]))
    