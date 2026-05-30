from scipy.spatial.transform import Rotation
import math
import quaternion
import numpy as np
import matplotlib.pyplot as plt
import os

xyz_axes = ['x', 'y', 'z']
q_axes = ['x', 'y', 'z', 'w']

RAD_TO_DEG = 180/np.pi
DEG_TO_RAD = np.pi/180
RPM_TO_RAD_PER_SEC = np.pi/30
RAD_PER_SEC_TO_RPM = 30/np.pi

###############################################################################
# Mathematical Operations
###############################################################################

SMALL = 1e-10

# Distances
KM2M = 1e3
FT2M = 0.3048
MILE2M = 1609.344
NM2M = 1852
MILE2FT = 5280
MILEPH2KMPH = 0.44704
NMPH2KMPH = 0.5144444

# Time
DAY2SEC = 86400
DAY2MIN = 1440
DAY2HR = 24
HR2SEC = 3600
MIN2SEC = 60
YR2DAY = 365.25
CENT2YR = 100
CENT2DAY = CENT2YR * YR2DAY

# Angles
HALFPI = np.pi / 2
TWOPI = 2 * np.pi
DEG2MIN = 60
DEG2ARCSEC = DEG2MIN * MIN2SEC
ARCSEC2RAD = np.radians(1 / DEG2ARCSEC)
DEG2SEC = np.degrees(TWOPI) / DAY2SEC
DEG2HR = np.degrees(TWOPI) / DAY2HR
HR2RAD = DEG2HR * np.radians(1)

###############################################################################
# Astrodynamic Operations
###############################################################################

# Time
J2000 = 2451545  # Julian date of the epoch J2000.0 (noon)
J2000_UTC = 2451544.5  # Julian date of the epoch J2000.0 in UTC (midnight)
JD_TO_MJD_OFFSET = 2400000.5  # offset between Julian and Modified Julian dates

# EGM-08 (Earth) constants used here
# fmt: off
RE = 6378.1363                      # km
FLAT = 1 / 298.257223563
EARTHROT = 7.292115e-5              # rad/s
MU = 398600.4415                    # km^3/s^2
MUM = 3.986004415e14                # m^3/s^2
J2 = 0.001082626174
J4 = -1.6198976e-06
# fmt: on

# Derived constants from the base values

# Sidereal day in seconds
SIDERALDAY_SEC = 86164.090524  # seconds

# Approximate Earth rotation
EARTHROT_APPROX = TWOPI / DAY2SEC  # rad/s

# Earth eccentricity
ECCEARTH = np.sqrt(2 * FLAT - FLAT**2)
ECCEARTHSQRD = ECCEARTH**2

# Earth radius
RENM = RE / NM2M
REFT = RE * 1e3 / FT2M

# Orbital period
TUSEC = np.sqrt(RE**3 / MU)
TUMIN = TUSEC / MIN2SEC
TUDAY = TUSEC / DAY2SEC
TUDAYSID = TUSEC / SIDERALDAY_SEC

# Earth rotation & rotational angular velocity
OMEGAARTHPTU = EARTHROT * TUSEC
OMEGAARTHPMIN = EARTHROT * MIN2SEC

# Orbital velocity
VELKPS = np.sqrt(MU / RE)
VELFPS = VELKPS * 1e3 / FT2M
VELPDMIN = VELKPS * MIN2SEC / RE
DEGSEC = (180 / np.pi) / TUSEC
RADPDAY = TWOPI * 1.002737909350795

# Astronomical distances & measurements
# fmt: off
SPEEDOFLIGHT = 299792.458           # km/s
AU2KM = 149597870.7                 # km
EARTH2MOON = 384400                 # km
MOONRADIUS = 1738                   # km
SUNRADIUS = 696000                  # km
# fmt: on

# Masses in kg
MASSSUN = 1.9891e30
MASSEARTH = 5.9742e24
MASSMOON = 7.3483e22

# Standard gravitational parameters in km^3/s^2
MUSUN = 1.32712428e11
MUMOON = 4902.799

# Obliquities
OBLIQUITYEARTH = np.radians(23.439291)

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
def conv_Rotation_obj_to_euler_axis_angle(r : Rotation):
    alpha = np.arccos(0.5*(np.trace(r.as_matrix())-1))
    return alpha

# def conv_dcm_to_quat(dcm : np.array):
    # q4 = 0.5*np.sqrt(1 + dcm[0,0] + dcm[1,1] + dcm[2,2])
    # q = np.quaternion(q4,
    #                   (dcm[1,2] - dcm[2,1])/(4*q4),
    #                   (dcm[2,0] - dcm[0,2])/(4*q4),
    #                   (dcm[0,1] - dcm[1,0])/(4*q4))
    # return q

def conv_quat_to_dcm_nadafi(q : np.quaternion):
    q0 = q.w
    q_vec = np.array([q.x, q.y, q.z])
    return (q0**2 - np.linalg.norm(q_vec))*np.eye(3) + 2*np.outer(q_vec, q_vec) - 2*q0*skew_symmetric(q_vec)
    # C = (q0**2 - np.linalg.norm(q_vec))*np.eye(3) +  - 2*q0*skew_symmetric(q_vec)

def quaternion_multiply(q1 : np.array, q2 : np.array):
    # w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    # w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z
    qv1 = q1[:3]
    qv2 = q2[:3]
    w1 = q1[3]
    w2 = q2[3]
    w = w1*w2 - np.dot(qv1, qv2)
    qv = w1*qv2 + w2*qv1 + cross_product_M31M31(qv1, qv2)

    # return np.quaternion(w, qv[0], qv[1], qv[2])
    return np.array([qv[0], qv[1], qv[2], w])

def get_quaternion_error_bong_wie(qc : np.quaternion, qd : np.quaternion):

    qe = np.array([[qc.w, qc.z, -1*qc.y, -1*qc.x],
                   [-1*qc.z, qc.w, qc.x, -1*qc.y],
                   [qc.y, -1*qc.x, qc.w, -1*qc.z],
                   [qc.x, qc.y, qc.z, qc.w]])\
        @ np.array([qd.x, qd.y, qd.z, qd.w])
    return np.quaternion(qe[3], qe[0], qe[1], qe[2])

def get_quaternion_error_Nadafi(qd : np.quaternion, q : np.quaternion):

    qe = np.array([[qd.w, qd.x, qd.y, qd.z],
                   [qd.x, -1*qd.w, -1*qd.z, qd.y],
                   [qd.y, qd.z, -1*qd.w, -1*qd.x],
                   [qd.z, -1*qd.y, qd.x, -1*qd.w]])\
        @ np.array([q.w, q.x, q.y, q.z])
    return quaternion.from_float_array([qe[0], qe[1], qe[2], qe[3]])

# assumes scalar-last format
def get_principle_angle_from_array(q):
    return 2*np.arctan2(np.linalg.norm(q[:3]), q[3])

def get_principal_angle_from_np_quaternion(q : np.quaternion):
    return 2*np.arctan2(np.linalg.norm([q.x, q.y, q.z]), q.w)

def round_dict_values(d, k):
    return {key: float(f"{value:.{k}E}") for key, value in d.items()}

# Deprecated
def conv_rpm_to_rads_per_sec(value):
    return value*np.pi/30
# Deprecated
def conv_rads_per_sec_to_rpm(value):
    return value*30/np.pi

def low_pass_filter(value, value_prev, coeff):
    return (coeff)*value_prev + (1 - coeff)*value

def cross_product_M31M31(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def mat_multiply_3x3_vec(mat : np.array, vec : np.array):
    return np.array([mat[0,0]*vec[0] + mat[0,1]*vec[1] + mat[0,2]*vec[2],
                     mat[1,0]*vec[0] + mat[1,1]*vec[1] + mat[1,2]*vec[2],
                     mat[2,0]*vec[0] + mat[2,1]*vec[1] + mat[2,2]*vec[2]])

def _sign(x: float) -> int:
    return 1 if x >= 0 else -1

def sat_delta(x: float) -> int:
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return x

def sat_delta_vec(v: np.array):
    return np.asmatrix(np.array([sat_delta(v_i) for v_i in np.asarray(v).flatten()])).T

def sat_norm(v: np.array) -> np.array:
    norm = np.linalg.norm(v)
    if norm >= 1:
        return v/norm
    else:
        return v

# Symmetric Saturation function that limits the magnitude of a scalar to 1 while preserving its direction 
def sat(v: float, max: float) -> float:
    if v > max:
        return max
    elif v < -max:
        return -max
    else:
        return v

# Symmetric Saturation function that limits the magnitude of a vector to 1 while preserving its direction 
def sat_vec(v: np.array, max: float) -> np.array:
    for i in range(len(v)):
        v[i] = sat(v[i], max)
    return v

def skew_symmetric(v: np.array) -> np.array:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def col_vec(v: np.array) -> np.array:
    return np.asmatrix(v).T

def row_vec(v: np.array) -> np.array:
    return np.asmatrix(v)

def magnitude(vector): 
    return math.sqrt(sum(pow(element, 2) for element in vector))

def angle_vec(v1: np.array, v2: np.array) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("One of the vectors has zero magnitude")
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # Clamp cos_theta to the range [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

#! @brief Create a combined plot with multiple rows and columns
# @param rows: List of tuples, each containing (row_name, [axes], label)
# @param cols: Number of columns in the plot
# @param results_data: Dictionary containing data to plot
def create_plots_combined(rows, cols, results_data, config=None, LOG_FILE_NAME=None, type='line', x_axis=None):
    if os.path.exists(fr"../data_logs/{LOG_FILE_NAME}") is False:
        os.mkdir(fr"../data_logs/{LOG_FILE_NAME}")

    fig, ax = plt.subplots(int(np.ceil(len(rows)/cols)),cols,sharex=True,figsize=(18,8))

    ax_as_np_array= np.array(ax)
    plots_axes = ax_as_np_array.flatten()
    for row_idx, row in enumerate(rows):
        row_name, axes, label = row
        current_plot : plt.Axes = plots_axes[row_idx-1]
        if axes is None:
            axes = ['none']
        for axis in axes:
            if axis != 'none': 
                name = row_name + "_" + axis
            else: 
                axis = None
                name = row_name
            if type == 'line':
                try:
                    current_plot.plot(results_data['time'], results_data[name], label=axis)
                except KeyError:
                    raise Exception(f"Warning: {name} not found in results_data")
                except Exception as e:
                    raise Exception(f"Error plotting {name}: {e}")
            elif type == 'scatter':
                if x_axis is None:
                    raise Exception("x_axis must be provided for scatter plot")
                current_plot.scatter(x_axis, results_data[name], label=axis)

        current_plot.set_xlabel('time (s)')
        current_plot.set_ylabel(label)
        if current_plot.get_legend_handles_labels()[0] != []:
            current_plot.legend()

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()
    if config is not None:
        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
            fig.savefig(fr"../data_logs/{LOG_FILE_NAME}/{LOG_FILE_NAME}_summary.pdf", bbox_inches='tight')

def create_plots_separated(rows, results_data, config=None, display=False, LOG_FILE_NAME=None):

    # Create separate figures if enabled in config
    for row in rows:
        row_name, axes, label = row
        fig_separate = plt.figure(figsize=(12,6))
        ax_separate = fig_separate.add_subplot(111)
        if axes is None:
            axes = ['none']
        for axis in axes:
            if axis != 'none':
                name = row_name + "_" + axis
            else:
                axis = None
                name = row_name
            ax_separate.plot(results_data['time'], results_data[name], label=axis)
        
        ax_separate.set_xlabel('time (s)')
        ax_separate.set_ylabel(label)
        if ax_separate.get_legend_handles_labels()[0] != []:
            ax_separate.legend()
        
        if display is True:
            plt.title(f"{label} Plot")
            plt.show()

        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
            if os.path.exists(fr"../data_logs/{LOG_FILE_NAME}") is False:
                os.mkdir(fr"../data_logs/{LOG_FILE_NAME}")
            if not os.path.exists(fr"../data_logs/{LOG_FILE_NAME}/graphs"):
                os.mkdir(fr"../data_logs/{LOG_FILE_NAME}/graphs")
            fig_separate.savefig(fr"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_{row_name}.pdf", bbox_inches='tight')
        if display is False:
            plt.close(fig_separate)