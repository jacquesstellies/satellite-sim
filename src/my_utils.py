from scipy.spatial.transform import Rotation
import math
import quaternion
import numpy as np
import matplotlib.pyplot as plt
import os
import toml

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

###############################################################################
# Plotting
###############################################################################

FIG_SIZE = (12, 8)  # 6.3" is the text width of the thesis but plots look too grainy if used for fig size

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
    return (q0**2 - np.linalg.norm(q_vec)**2)*np.eye(3) + 2*np.outer(q_vec, q_vec) - 2*q0*skew_symmetric(q_vec)
    # C = (q0**2 - np.linalg.norm(q_vec))*np.eye(3) +  - 2*q0*skew_symmetric(q_vec)

# ---------------------------------------------------------------------------
# Hamilton passive-rotation convention (single source of truth)
#
# Passive DCM T_BA maps coordinates:  v_B = T_BA @ v_A.
# A passive quaternion q_BA (rotation A->B) has attitude matrix
#     A(q) = (q0^2 - |q_v|^2) I + 2 q_v q_v^T - 2 q0 [q_v]_x      (minus sign)
# which equals Rotation.from_quat([x,y,z,w]).as_matrix().T (scipy is active).
# A(.) is an ANTI-homomorphism under the Hamilton product: A(p*q) = A(q) A(p),
# so passive composition reverses relative to DCMs. See quat_error for the
# verified order. Verified in scripts/verify_hamilton_passive.py.
# ---------------------------------------------------------------------------
def quat_to_dcm(q : np.quaternion) -> np.array:
    """Passive attitude matrix A(q_BA): v_B = quat_to_dcm(q_BA) @ v_A."""
    q0 = q.w
    q_vec = np.array([q.x, q.y, q.z])
    return (q0**2 - q_vec@q_vec)*np.eye(3) + 2*np.outer(q_vec, q_vec) - 2*q0*skew_symmetric(q_vec)

def quat_error(q_TI : np.quaternion, q_FI : np.quaternion) -> np.quaternion:
    """Relative passive quaternion q_TF (rotation F->T) from two I-referenced
    quaternions. Contract (verified): quat_to_dcm(quat_error(q_TI, q_FI))
    == quat_to_dcm(q_TI) @ quat_to_dcm(q_FI).T, and identity when q_TI == q_FI.
    e.g. quat_error(q_RI, q_BI) -> q_RB, the body-frame error (vector part in B).
    Scalar part is kept >= 0 (q and -q are the same attitude) so feedback always
    takes the short way round instead of unwinding past 180 deg."""
    q_err = q_FI.inverse() * q_TI
    if q_err.w < 0:
        q_err = -q_err
    return q_err

def dcm_to_quat(T : np.array) -> np.quaternion:
    """Inverse of quat_to_dcm: passive DCM T_BA -> passive quaternion q_BA.
    Contract (verified): quat_to_dcm(dcm_to_quat(T)) == T. scipy is active, so
    feed the transpose to recover the passive quaternion."""
    q = Rotation.from_matrix(T.T).as_quat()  # scipy active matrix == T.T
    return np.quaternion(q[3], q[0], q[1], q[2])

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

def quat_from_vectors(u, v):

    d = np.dot(u, v)
    w = cross_product_M31M31(u, v)

    qv = d + math.sqrt(d * d + np.dot(w, w))
    return np.quaternion(w, qv[0], qv[1], qv[2]).normalize()

def rotate_vector_by_quaternion(v : np.array, q : np.quaternion):
    q_conj = q.inverse()
    v_quat = np.quaternion(0, v[0], v[1], v[2])
    rotated_v_quat = q * v_quat * q_conj
    return np.array([rotated_v_quat.x, rotated_v_quat.y, rotated_v_quat.z])

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
    # Scalar part >= 0 (q and -q are the same attitude) so feedback always
    # takes the short way round instead of unwinding past 180 deg.
    if qe[0] < 0:
        qe = -qe
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

def cross_product_M21M21(a, b):
    return np.array([a[0]*b[1] - a[1]*b[0]])

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
    if np.shape(v) == (3,):
        v = np.asmatrix(v).T
    return np.array([[0, -v[2,0], v[1,0]],
                     [v[2,0], 0, -v[0,0]],
                     [-v[1,0], v[0,0], 0]])

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

def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = toml.load(f)
    return config

###############################################################################
# Wheel Layout
###############################################################################

# Wheel distribution matrices per layout. Shared by WheelModule and by the
# plotting helpers below, so a run's plots can be rebuilt from its config alone
# without instantiating the satellite.
wheel_layouts = {
    'ortho':   np.eye(3),
    'pyramid': np.array([[-1, -1,  1, 1],
                         [ 1, -1, -1, 1],
                         [ 1,  1,  1, 1]]),
    'tetra':   np.array([[ 0.9428, -0.4714, -0.4714, 0],
                         [ 0,       0.8165, -0.8165, 0],
                         [-0.3333, -0.3333, -0.3333, 1]]),
}

def get_wheel_layout(config) -> tuple:
    """(num_wheels, D) for the configured wheel layout."""
    layout = config['wheels']['config']
    if layout == 'custom':
        num_wheels = config['wheels']['num_wheels']
        D = np.array(config['wheels']['D'])
        if D.shape != (3, num_wheels):
            raise(Exception(f"invalid D matrix shape {D.shape}"))
        return num_wheels, D
    if layout not in wheel_layouts:
        raise(Exception(f"{layout} is not a valid wheel layout. \nerror unable to set up wheel layout"))
    D = wheel_layouts[layout].copy()
    return D.shape[1], D

###############################################################################
# Plot Labelling
###############################################################################

# Match the thesis' LaTeX fonts so figures drop into the document unchanged.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif"],
    "mathtext.fontset": "cm",  # matches LaTeX math rendering
    "font.size": 14,        # matches \normalsize in the thesis
    "axes.labelsize": 14,
    "legend.fontsize": 16,  # matches caption "small" size roughly
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Logged series name -> (stem, subscript) of the symbol it plots. The axis
# suffix the logger appends (_x/_y/_z/_w, or the wheel index) is folded into the
# subscript, so q_sat_error_x reads as q_{ex} rather than q_sat_error_x.
plot_symbols = {
    # Attitude
    'q_sat':                     (r'q', ''),
    'q_sat_ref':                 (r'q', 'd'),
    'q_sat_error':               (r'q', 'e'),
    'w_sat':                     (r'\omega', ''),
    'w_sat_ref':                 (r'\omega', 'd'),
    'w_sat_error':               (r'\omega', 'e'),
    'dw_sat_ref':                (r'\dot{\omega}', 'd'),
    'euler_axis_sat':            (r'\theta', ''),
    'euler_axis_sat_deg':        (r'\theta', ''),
    'euler_axis_sat_error':      (r'\theta', 'e'),
    'euler_axis_sat_error_deg':  (r'\theta', 'e'),
    'euler_int':                 (r'\theta', 'i'),
    'boresight_nadir_error_deg': (r'\theta', 'bn'),
    # Torques and energy
    'T_sat':                     (r'T', ''),
    'T_dist':                    (r'T', 'd'),
    'T_magt':                    (r'T', 'm'),
    'm_magt':                    (r'm', ''),
    'control_energy':            (r'E', 'c'),
    # Wheels
    'w_wheels':                  (r'\omega', 'w'),
    'w_wheels_est':              (r'\hat{\omega}', 'w'),
    'dw_wheels_est':             (r'\dot{\hat{\omega}}', 'w'),
    'T_wheels':                  (r'T', 'w'),
    'T_wheels_est':              (r'\hat{T}', 'w'),
    'T_ctr_wheels':              (r'T', 'c'),
    'f_wheels':                  (r'f', 'w'),
    'f_wheels_est':              (r'\hat{f}', 'w'),
    'f_wheels_error':            (r'f', 'e'),
    'u_a':                       (r'u', 'a'),
    'E':                         (r'E', ''),
    'E_est':                     (r'\hat{E}', ''),
    # Momentum, orbit and environment
    'H_total':                   (r'H', ''),
    'H_norm':                    (r'\|H\|', ''),
    'B_eci':                     (r'B', ''),
    's_sat_eci':                 (r's', ''),
    'v_sat_eci':                 (r'v', ''),
    'n_sun':                     (r'n', 's'),
    'n_nadir':                   (r'n', 'n'),
    'd':                         (r'd', ''),
    'f_wheels_acc':              (r'f', r'w\mathrm{acc}'),
    # Nadafi backstepping / FNDO auxiliaries
    'F':                         (r'F', ''),
    'Z':                         (r'Z', ''),
    'Z_norm':                    (r'\|Z\|', ''),
    'term_1':                    (r'u', '1'),
    'term_2':                    (r'u', '2'),
    'v_0':                       (r'v', '0'),
    'chi_0':                     (r'\chi', '0'),
    'chi_0_error':               (r'\chi', '0e'),
    'chi_1':                     (r'\chi', '1'),
    'chi_1_error':               (r'\chi', '1e'),
    'mu':                        (r'\mu', ''),
    # Zarourati underactuated auxiliaries
    'xi':                        (r'\xi', ''),
    'eta':                       (r'\eta', ''),
    'eta_norm':                  (r'\|\eta\|', ''),
    'kappa1':                    (r'\kappa', '1'),
    'kappa2':                    (r'\kappa', '2'),
    'we_u':                      (r'\omega', 'eu'),
    'dwe_u':                     (r'\dot{\omega}', 'eu'),
    'phi_hat':                   (r'\hat{\phi}', ''),
    # Adaptive controller
    'control_theta':             (r'\hat{\theta}', ''),
    'control_adaptive_model_output': (r'\theta', 'm'),
}

# Series whose "axis" names an angle rather than a vector component, so the axis
# carries the symbol itself instead of becoming a subscript.
plot_axis_symbols = {
    'e321_sat': {'yaw': r'\psi', 'pitch': r'\theta', 'roll': r'\phi'},
}

def _upright(text : str) -> str:
    """Multi-character names stay upright, single letters stay italic."""
    return text if len(text) == 1 else rf"\mathrm{{{text}}}"

def latex_label(row_name : str, axis = None) -> str:
    """Math-mode label for a logged series, e.g. ('w_sat_error', 'x') -> $\\omega_{ex}$.

    Unknown series fall back to the first name token as the stem and the rest as
    an upright subscript, so no raw underscores ever reach the legend."""
    axis_symbols = plot_axis_symbols.get(row_name)
    if axis_symbols is not None and axis in axis_symbols:
        return f"${axis_symbols[axis]}$"

    if row_name in plot_symbols:
        stem, sub = plot_symbols[row_name]
    else:
        tokens = row_name.split('_')
        stem = _upright(tokens[0])
        sub = ''.join(_upright(token) for token in tokens[1:])

    subs = [s for s in (sub, axis) if s not in (None, '', 'none')]
    if not subs:
        return f"${stem}$"
    return f"${stem}_{{{''.join(subs)}}}$"

#! @brief Create a combined plot with multiple rows and columns
# @param rows: List of tuples, each containing (row_name, [axes], label)
# @param cols: Number of columns in the plot
# @param results_data: Dictionary containing data to plot
def create_plots_separated(rows, 
                           results_data, 
                           config, 
                           LOG_FILE_NAME, 
                           LOG_DIR, 
                           file_name_append = ""
                           ):
    # Create separate figures if enabled in config
    names = []
    for row in rows:
        row_name, axes, label = row
        fig_separate = plt.figure(figsize=FIG_SIZE)
        ax_separate = fig_separate.add_subplot(111)
        
        for axis in axes:
            if axis != 'none':
                name = row_name + "_" + axis
            else:
                axis = None
                name = row_name
            try:
                ax_separate.plot(results_data['time'], results_data[name], label=latex_label(row_name, axis))
                names.append(name)
            except Exception as e:
                print(f"Error plotting {name}: {e}")

        
        ax_separate.set_xlabel('Time (s)')
        ax_separate.set_ylabel(label)
        ax_separate.grid(visible=True, axis='both')
        if ax_separate.get_legend_handles_labels()[0] != []:
            ax_separate.legend(loc='upper left')
        
        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None:
            if not os.path.exists(os.path.abspath(fr"{LOG_DIR}/graphs")):
                os.mkdir(os.path.abspath(fr"{LOG_DIR}/graphs"))
            fig_separate.savefig(os.path.abspath(fr"{LOG_DIR}/graphs/{LOG_FILE_NAME}_{row_name}{file_name_append}.png"), bbox_inches='tight')
        
        if config['output']['separate_plots_display'] is False:
            plt.close(fig_separate)

def create_plots_comparison(rows : list, 
                            label : str, 
                            graph_name: str, 
                            results_data : dict, 
                            config : dict, 
                            LOG_FILE_NAME : str , 
                            LOG_DIR : str, 
                            show : bool = False
                            ):
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    for row_idx, row in enumerate(rows):
        row_name, axes = row
        
        for axis_idx, axis in enumerate(axes):
            if axis != 'none':
                name = row_name + "_" + axis
            else:
                axis = None
                name = row_name
            try:
                ax.plot(results_data['time'], results_data[name], label=latex_label(row_name, axis), linestyle=['-','--',':'][row_idx%3], color=['r','g','b','y','m','gray','k'][axis_idx%7])
                
            except Exception as e:
                print(f"Error plotting {name}: {e}")
    ax.legend(loc='upper right')
    ax.grid(visible=True, axis='both')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(label)

    # Labels go on before show(), otherwise the displayed figure is unlabelled
    # while the saved png is fine.
    if show is True or config['output']['show_plots'] is True:
        try:
            plt.show()
        except Exception as e:
            print(f"Error showing plots: {e}")

    if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
        fig.savefig(os.path.abspath(f"{LOG_DIR}/graphs/{LOG_FILE_NAME}_{graph_name}.png"), bbox_inches='tight')

def create_plots_combined(rows, cols, results_data, config, LOG_FILE_NAME, LOG_DIR, type='line', x_axis=None):
    fig, ax= plt.subplots(int(np.ceil(len(rows)/cols)),cols,sharex=True,figsize=(18,8))

    ax_as_np_array= np.array(ax)
    plots_axes = ax_as_np_array.flatten()
    for row_idx, row in enumerate(rows):
        row_name, axes, label = row
        current_plot : plt.Axes = plots_axes[row_idx-1]
        for axis in axes:
            if axis != 'none': 
                name = row_name + "_" + axis
            else: 
                axis = None
                name = row_name
            try:
                if type == 'line':
                    current_plot.plot(results_data['time'], results_data[name], label=latex_label(row_name, axis))
                elif type == 'scatter':
                    if x_axis is None:
                        raise Exception("x_axis must be provided for scatter plot")
                    current_plot.scatter(x_axis, results_data[name], label=latex_label(row_name, axis))
            except Exception as e:
                print(f"Error plotting {name}: {e}")
        current_plot.grid(visible=True, axis='both')
        current_plot.set_xlabel('Time (s)')
        current_plot.set_ylabel(label)
        if current_plot.get_legend_handles_labels()[0] != []:
            current_plot.legend()

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if config['output']['show_plots'] is True:
        try:
            plt.show()
        except Exception as e:
            print(f"Error showing plots: {e}")

    if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
        fig.savefig(os.path.abspath(f"{LOG_DIR}/{LOG_FILE_NAME}_summary.png"), bbox_inches='tight')

def create_3D_quaternion_plot(results_data, config, LOG_FILE_NAME, LOG_DIR):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.plot(results_data['q_sat_x'], results_data['q_sat_y'], results_data['q_sat_z'], label='Satellite Quaternion Trajectory')
        ax.set_xlabel(latex_label('q_sat', 'x'))
        ax.set_ylabel(latex_label('q_sat', 'y'))
        ax.set_zlabel(latex_label('q_sat', 'z'))
        ax.legend()
    except Exception as e:
        print(f"Error plotting 3D quaternion trajectory: {e}")
    # plt.show()
    if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
        fig.savefig(os.path.abspath(f"{LOG_DIR}/graphs/{LOG_FILE_NAME}_quaternion_3D_trajectory.png"), bbox_inches='tight')

###############################################################################
# Standard Result Plots
###############################################################################

#! @brief Build the standard plot rows for a run from its config alone
# @return (summary_rows, detail_rows), each a list of (row_name, [axes], label)
def build_results_plots(config):
    summary = [
        ('w_sat', xyz_axes, r'Angular velocity $\omega$ (rad/s)'),
        ('q_sat', q_axes, r'Quaternion $q$'),
        ('e321_sat', ['yaw', 'pitch', 'roll'], r'Euler angle (deg)'),
        ('euler_axis_sat_deg', ['none'], r'Principal axis angle $\theta$ (deg)'),
        ('T_sat', xyz_axes, r'Torque $T$ ($\mathrm{N \cdot m}$)'),
        ('control_energy', xyz_axes, r'Control energy $E_c$ (J)'),
        ('T_dist', xyz_axes, r'Disturbance torque $T_d$ ($\mathrm{N \cdot m}$)'),
    ]

    detail = []

    if config['satellite']['wheels_control_enable']:
        num_wheels, _ = get_wheel_layout(config)
        wheel_axes = [str(i) for i in range(num_wheels)]
        detail.append(('T_wheels', wheel_axes, r'Wheel torque $T_w$ ($\mathrm{N \cdot m}$)'))
        detail.append(('w_wheels', wheel_axes, r'Wheel speed $\omega_w$ (rad/s)'))
        detail.append(('E', wheel_axes, r'Actuator authority $E$ (fraction)'))
        detail.append(('f_wheels', wheel_axes, r'Wheel disturbance torque $f_w$ ($\mathrm{N \cdot m}$)'))
        detail.append(('u_a', wheel_axes, r'Additive fault $u_a$ ($\mathrm{N \cdot m}$)'))
        if config['observer']['enable']:
            detail.append(('w_wheels_est', wheel_axes, r'Estimated wheel speed $\hat{\omega}_w$ (rad/s)'))
            detail.append(('T_wheels_est', wheel_axes, r'Estimated wheel torque $\hat{T}_w$ ($\mathrm{N \cdot m}$)'))
            detail.append(('f_wheels_est', wheel_axes, r'Estimated wheel disturbance torque $\hat{f}_w$ ($\mathrm{N \cdot m}$)'))
            detail.append(('f_wheels_error', wheel_axes, r'Wheel disturbance torque error $f_e$ ($\mathrm{N \cdot m}$)'))
            detail.append(('E_est', wheel_axes, r'Estimated actuator authority $\hat{E}$ (fraction)'))

    if config['controller']['type'] == "adaptive":
        summary.append(('control_adaptive_model_output', ['none'], r'Adaptive model output $\theta_m$ (rad)'))
        summary.append(('control_theta', xyz_axes, r'Adaptive parameter $\hat{\theta}$'))

    detail.append(('q_sat_ref', q_axes, r'Reference quaternion $q_d$'))
    detail.append(('q_sat_error', q_axes, r'Quaternion error $q_e$ (satellite to reference)'))
    detail.append(('w_sat_ref', xyz_axes, r'Reference angular velocity $\omega_d$ (rad/s)'))
    detail.append(('w_sat_error', xyz_axes, r'Angular velocity error $\omega_e$ (rad/s)'))
    detail.append(('euler_axis_sat_error_deg', ['none'], r'Principal axis angle error $\theta_e$ (deg)'))
    detail.append(('T_magt', xyz_axes, r'Magnetorquer torque $T_m$ ($\mathrm{N \cdot m}$)'))
    detail.append(('m_magt', xyz_axes, r'Magnetorquer moment $m$ ($\mathrm{A \cdot m^2}$)'))
    detail.append(('B_eci', xyz_axes, r'Magnetic field ECI $B$ (T)'))
    detail.append(('H_total', xyz_axes, r'Total angular momentum $H$ ($\mathrm{N \cdot m \cdot s}$)'))
    detail.append(('H_norm', ['none'], r'Total angular momentum norm $\|H\|$ ($\mathrm{N \cdot m \cdot s}$)'))

    detail.append(('s_sat_eci', xyz_axes, r'Satellite position ECI $s$ (km)'))
    detail.append(('v_sat_eci', xyz_axes, r'Satellite velocity ECI $v$ (km/s)'))
    detail.append(('n_sun', xyz_axes, r'Sun vector $n_s$ (unitless)'))
    detail.append(('n_nadir', xyz_axes, r'Nadir vector $n_n$ (unitless)'))

    if config['satellite']['mode'] == "nominal_night":
        # Boresight (body +z) to nadir angle - the pointing metric this mode
        # is tracking, with the (unactuated) yaw about the boresight excluded.
        detail.append(('boresight_nadir_error_deg', ['none'],
                       r'Boresight (body $z$) to nadir angle $\theta_{bn}$ (deg)'))

    if config['controller']['type'] == "backstepping":
        sub_type = config['controller'].get('sub_type', '')
        # Nadafi auxiliary variables
        if sub_type.startswith("Nadafi"):
            detail.append(('F', xyz_axes, r'$F$ ($\mathrm{rad/s^2}$)'))
            detail.append(('Z_norm', ['none'], r'$\|Z\|$ (rad/s)'))
            detail.append(('Z', xyz_axes, r'$Z$ (rad/s)'))
            detail.append(('term_1', xyz_axes, r'$u_1$ ($\mathrm{N \cdot m}$)'))
            detail.append(('term_2', xyz_axes, r'$u_2$ ($\mathrm{N \cdot m}$)'))
            detail.append(('v_0', xyz_axes, r'$v_0$ (rad/s)'))
            detail.append(('chi_0', xyz_axes, r'$\chi_0$ (rad/s)'))
            detail.append(('chi_1', xyz_axes, r'$\chi_1$ ($\mathrm{rad/s^2}$)'))
            # detail.append(('chi_0_error', xyz_axes, r'$\chi_{0e}$ ($\mathrm{rad/s}$)'))
            detail.append(('chi_1_error', xyz_axes, r'$\chi_{1e}$ ($\mathrm{rad/s^2}$)'))
            detail.append(('mu', xyz_axes, r'$\mu$ (rad/s)'))
        # Zarourati underactuated auxiliary variables
        if sub_type.startswith("Zarourati"):
            detail.append(('xi', ['none'], r'$\xi$ (unitless)'))
            detail.append(('eta_norm', ['none'], r'$\|\eta\|$ (unitless)'))
            detail.append(('kappa1', ['none'], r'$\kappa_1$ (unitless)'))
            detail.append(('kappa2', ['none'], r'$\kappa_2$ (unitless)'))
            detail.append(('we_u', ['none'], r'$\omega_{eu}$ (rad/s)'))
            detail.append(('dwe_u', ['none'], r'$\dot{\omega}_{eu}$ ($\mathrm{rad/s^2}$)'))
            detail.append(('eta', xyz_axes, r'$\eta$ (unitless)'))
            detail.append(('phi_hat', ['none'], r'$\hat{\phi}$ (unitless)'))

    return summary, detail

#! @brief Add the lumped body-frame disturbance acceleration d that chi_1 should converge to
# Wheel torque enters the body dynamics as -D @ dH_wheels (satellite.py), and the
# FNDO's known model carries the *commanded* torque with that same minus sign, so
# the per-wheel fault torque f_wheels appears in the residual as -J^-1 @ D @ f_wheels.
# f_wheels is per-wheel, so it must be mapped to the body through D first.
# T_magt is not in the FNDO's known model either, so it lands in chi_1 too. It is
# not negligible whenever the magnetorquers are actively dumping momentum, but it
# is left out here (see the commented term below).
# Columns are added in place; a run whose log lacks the inputs is left untouched.
def calc_lumped_disturbance(results_data, config):
    num_wheels, D = get_wheel_layout(config)
    needed = ['T_dist_' + axis for axis in xyz_axes] + [f'f_wheels_{i}' for i in range(num_wheels)]
    if any(column not in results_data for column in needed):
        print(f"calc_lumped_disturbance: missing columns {needed}, skipping")
        return results_data

    inertia = np.asarray(config['satellite']['M_Inertia'])
    f_wheels_body = results_data[[f'f_wheels_{i}' for i in range(num_wheels)]].to_numpy() @ np.asarray(D).T
    d = results_data[['T_dist_' + axis for axis in xyz_axes]].to_numpy() \
        - f_wheels_body
    results_data[['d_' + axis for axis in xyz_axes]] = d @ np.linalg.inv(inertia).T
    results_data[['chi_1_error_' + axis for axis in xyz_axes]] = results_data[['chi_1_' + axis for axis in xyz_axes]].to_numpy() \
    - results_data[['d_' + axis for axis in xyz_axes]].to_numpy()
    return results_data

#! @brief Create every standard plot for a run: the summary sheet, the per-signal
#         graphs and the measured-vs-estimated comparisons.
# @param results_data: results dataframe, either live or read back from a log csv
# @param config: the run's config dict
def create_results_plots(results_data,
                         config,
                         LOG_FILE_NAME,
                         LOG_DIR,
                         cols = 2,
                         show = False
                         ):
    summary_rows, detail_rows = build_results_plots(config)

    create_plots_separated(summary_rows, results_data, config, LOG_FILE_NAME, LOG_DIR)
    create_plots_combined(summary_rows, cols, results_data, config, LOG_FILE_NAME, LOG_DIR)
    create_plots_separated(detail_rows, results_data, config, LOG_FILE_NAME, LOG_DIR)
    create_3D_quaternion_plot(results_data, config, LOG_FILE_NAME, LOG_DIR)

    create_plots_comparison([('q_sat', q_axes), ('q_sat_ref', q_axes)],
                            r'Quaternion $q$', 'q_sat_vs_ref',
                            results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)
    create_plots_comparison([('q_sat', xyz_axes), ('q_sat_ref', xyz_axes)],
                            r'Quaternion $q$', 'q_sat_vs_ref_vec',
                            results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)

    if config['satellite']['wheels_control_enable'] and config['observer']['enable']:
        num_wheels, _ = get_wheel_layout(config)
        wheel_axes = [str(i) for i in range(num_wheels)]
        create_plots_comparison([('w_wheels', wheel_axes), ('w_wheels_est', wheel_axes)],
                                r'Wheel speed $\omega_w$ (rad/s)', 'wheels_speed_meas_vs_est',
                                results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)
        create_plots_comparison([('T_wheels', wheel_axes), ('T_wheels_est', wheel_axes)],
                                r'Wheel torque $T_w$ ($\mathrm{N \cdot m}$)', 'wheels_torque_meas_vs_est',
                                results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)
        create_plots_comparison([('E', wheel_axes), ('E_est', wheel_axes)],
                                r'Wheel effectiveness $E$ (fraction)', 'wheels_authority_meas_vs_est',
                                results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)

    if config['controller'].get('sub_type', '').startswith("Nadafi"):
        calc_lumped_disturbance(results_data, config)
        create_plots_comparison([('chi_1', xyz_axes), ('d', xyz_axes)],
                                r'Angular acceleration ($\mathrm{rad/s^2}$)', 'chi_1_vs_d',
                                results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)
        create_plots_comparison([('chi_0', xyz_axes), ('w_sat_error', xyz_axes)],
                                r'Angular velocity (rad/s)', 'chi_0_vs_w_sat_error',
                                results_data, config, LOG_FILE_NAME, LOG_DIR, show=show)
