from scipy.spatial.transform import Rotation
import math
import quaternion
import numpy as np

# rotate an object's moment of inertia about the xyz axes (in degrees)
def rotate_M_inertia(M_inertia : np.array, direction : np.array):
    dir = Rotation.from_euler('xyz', direction, degrees=True)
    dcm = dir.as_matrix()
    
    return dcm@M_inertia@np.transpose(dcm)

# convert the given point mass and poisition vector to moment of inertia
def calc_M_inertia_point_mass(pos : np.array, mass : float):
    Ixx = pow(pos[1],2) + pow(pos[2],2)
    Iyy = pow(pos[0],2) + pow(pos[2],2)
    Izz = pow(pos[0],2) + pow(pos[1],2)

    Ixy = -pos[0]*pos[1]
    Ixz = -pos[0]*pos[2]
    Iyz = -pos[1]*pos[2]

    M_inertia = mass*[[Ixx, Ixy, Ixz],[Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]

    return M_inertia

def conv_scipy_to_numpy_q(q : Rotation):
    q_result = q.as_quat()
    return np.quaternion(q_result[3], q_result[0], q_result[1], q_result[2])

def conv_numpy_to_scipy_q(q : np.quaternion):
    return Rotation.from_quat([q.x, q.y, q.z, q.w])