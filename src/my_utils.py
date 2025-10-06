from scipy.spatial.transform import Rotation
import math
import quaternion
import numpy as np
import matplotlib.pyplot as plt
import os

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
def conv_Rotation_obj_to_euler_axis_angle(r : Rotation):
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

def cross_product(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def _sign(x: float) -> int:
    return 1 if x >= 0 else -1
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