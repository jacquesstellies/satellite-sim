from scipy.spatial.transform import Rotation
import math
import quaternion
import numpy as np

xyz_axes = ['x', 'y', 'z']
q_axes = ['x', 'y', 'z', 'w']

# rotate an object's moment of inertia about the xyz axes (in degrees)
def rotate_M_inertia(M_inertia : np.array, dir : Rotation):
    
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

    M_inertia = mass*np.array([[Ixx, Ixy, Ixz],[Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    return M_inertia

def conv_Rotation_obj_to_numpy_q(q : Rotation):
    q_result = q.as_quat()
    return np.quaternion(q_result[3], q_result[0], q_result[1], q_result[2])

def conv_numpy_to_Rotation_obj_q(q : np.quaternion):
    return Rotation.from_quat([q.x, q.y, q.z, q.w])

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def conv_Rotation_obj_to_dict(r : Rotation):
    q_result = r.as_quat()
    my_dict = {} 
    for i, axis in enumerate(q_axes):
        my_dict[axis] = q_result[i]
    return my_dict

# convert a Rotation object to angle about euler axis of rotation
def conv_Rotation_obj_to_alpha(r : Rotation):
    alpha = np.arccos(0.5*(np.trace(r.as_matrix())-1))
    return alpha

def round_dict_values(d, k):
    return {key: float(f"{value:.{k}E}") for key, value in d.items()}

def conv_rpm_to_rads_per_sec(value):
    return value*np.pi/30

def conv_rads_per_sec_to_rpm(value):
    return value*30/np.pi

def low_pass_filter(value, value_prev, coeff):
    return (coeff)*value_prev + (1 - coeff)*value