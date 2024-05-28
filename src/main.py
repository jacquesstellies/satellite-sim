import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import math
import quaternion
import my_utils as my_utils
import logging
import pandas as pd
import argparse
import toml
import datetime
import control
import body
import my_globals
# all units are in SI (m, s, N, kg.. etc)

# logger = logging.getLogger(__name__)

results_data = my_globals.results_data


class Satellite(body.Body):

    logger = None
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    _controller : body.Controller = None
    _wheel_module : body.WheelModule = None
    _fault : body.Fault = None

    def __init__(self, _wheel_module : body.WheelModule, controller : body.Controller, fault : body.Fault, wheel_offset = 0, logger = None):
        self._wheel_module = wheel_module
        if _wheel_module.config == "standard":
            self._wheel_module.wheels[0].position[0] = self.dimensions['x']-wheel_offset
            self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
            self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        self._controller = controller
        self._next_control_time_step = controller.time_step
        self.logger = logger
        self._fault = fault


    def calc_M_inertia_body(self):

        M_inertia = np.zeros((3,3))
        # use cuboid for mass moment inertia
        M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions['y'],2)+pow(self.dimensions['z'],2))
        M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['z'],2))
        M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['y'],2))
        return M_inertia

    def calc_M_inertia_peri(self):
        M_inertia = np.zeros((3,3))
        if self._wheel_module.wheels is not None:
            M_inertia_indv_wheels = 0
            M_inertia_point_mass_wheels = 0

            for wheel in self._wheel_module.wheels:
                M_inertia_indv_wheels += wheel.M_inertia
                M_inertia_point_mass_wheels += my_utils.calc_M_inertia_point_mass(wheel.position, self.mass)

        return M_inertia

    def calc_M_inertia(self):
        self.M_inertia = self.calc_M_inertia_body() + self.calc_M_inertia_peri()

        self.calc_M_inertia_inv()

    ref_q = Rotation.from_quat([0,0,0,1])
    controller_enable = True
    wheels_control_enable = True
    _next_control_time_step = 0.1
    wheels_H_new = np.zeros(3)
    T_controller_com = np.zeros(3)
    T_controller_real = np.zeros(3)
    def calc_state(self, t, y):
        angular_v_input = y[:3]
        quaternion_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        
        angular_acc_result = [0]*3
        quaternion_rate_result = np.quaternion(1,0,0,0)
        wheels_H_curr = np.zeros(3)
        wheels_H_delta_com = np.zeros(3)

        # if self._wheel_module.wheels is not None:
        #     wheels_H_curr = self._wheel_module.get_angular_momentum()
            # for wheel in self.wheels:
            #     wheels_H_curr += wheel.calc_angular_momentum()
        
        if self.controller_enable:
        
            if t > self._next_control_time_step:
                self.T_controller_com = self._controller.calc_torque_control_output(quaternion_input, angular_v_input, self.ref_q)
                self._next_control_time_step += self._controller.time_step

            # if self.wheels_control_enable:
            #     self._wheel_module.update_angular_momentum(self.T_controller_com, self._controller.time_step)
                # wheels_H_delta_com = self.T_controller_com * _controller.time_step

                # self.wheels_H_new = wheels_H_delta_com + wheels_H_curr

                # for wheel in wheels:
                #     wheel.update_angular_velocity(wheel.M_inertia_inv@(self.wheels_H_new*wheel.shaft_axis_mask))
            
        if t > self._fault.time and self._fault.master_enable:
            self._fault.enabled = True
            # print(f"fault enabled: {self._controller.fault.enabled}")
        # self.T_controller_real = wheels_H_delta/self._controller.time_step
        # Hnet = self.M_inertia@(angular_v_input) + self._wheel_module.get_angular_momentum()
        # print("controller torque:")
        # print(self.T_controller_com)
        Hnet = self.M_inertia@(angular_v_input)
        angular_acc_result = self.M_inertia_inv@(self.T_controller_com - np.cross(angular_v_input,Hnet))

        # put the inertial velocity in quaternion form
        inertial_v_quaternion = np.quaternion(0, angular_v_input[0], angular_v_input[1], angular_v_input[2])

        quaternion_rate_result = 0.5*quaternion_input*inertial_v_quaternion
        quaternion_rate_result = [quaternion_rate_result.x, quaternion_rate_result.y, quaternion_rate_result.z, quaternion_rate_result.w]
        results_data['time'].append(t)
        for i in range(3):
            results_data['torque_'+ my_utils.xyz_axes[i]].append(self.T_controller_com[i])
            results_data['angular_acc_'+ my_utils.xyz_axes[i]].append(angular_acc_result[i])
        
        control_power_result = np.abs(self.T_controller_com * angular_v_input)
        return np.hstack([angular_acc_result, quaternion_rate_result, control_power_result])
    
# END DEF class Satellite()

def output_to_csv(file_name):
    df = pd.DataFrame().from_dict(results_data)
    file = open(fr'..\data_logs\{file_name}.csv', 'w+')
    df.to_csv(file,sep=',')
    return file

def calc_yaw_pitch_roll_rates(data_in):
    
    inertial_rates = data_in[:3]
    
    r = Rotation.from_quat([data_in[3], data_in[4], data_in[5], data_in[6]])

    yaw, pitch, roll = r.as_euler('zyx')
    
    yaw_rate = 1/np.cos(pitch)*(inertial_rates[1]*np.sin(roll)+inertial_rates[2]*np.cos(roll))
    pitch_rate = inertial_rates[1]*np.cos(roll)-inertial_rates[2]*np.sin(roll)
    roll_rate = inertial_rates[0]+inertial_rates[1]*np.tan(pitch)*np.sin(roll)+inertial_rates[2]*np.tan(pitch)*np.cos(roll)

    return [yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate]

def interpolate_data(data, time_series, time_series_new):
    return np.interp(time_series_new, time_series, data)

# Calculate info relating to the 
def calc_control_data(axes_commanded):

    control_info = {}
    accuracy = {}
    settling_time = {}

    def calc_control_info_per_axis(axis : str, valid=True):
        if axis in my_utils.q_axes:
            if valid:
                control_info[axis] = control.step_info(sysdata=results_data[f"quaternion_{axis}"], T=results_data['time'])
                ref_q_axis_value = my_utils.conv_Rotation_obj_to_dict(satellite.ref_q)[axis]
                accuracy[axis] = np.abs((control_info[axis]['SteadyStateValue']-ref_q_axis_value)/ref_q_axis_value)
                settling_time[axis] = control_info[axis]['SettlingTime']
            elif not valid:
                control_info[axis] = None
                accuracy[axis] = None
                settling_time[axis] = None
        else:
            raise(Exception(f'incorrect value for axis {axis}'))
        
    if len(axes_commanded) < 2:
        raise(Exception("no reference angle commanded"))
    elif len(axes_commanded) == 2:
        for axis in my_utils.q_axes:
            if axis in axes_commanded:
                calc_control_info_per_axis(axis, True)
            else:
                calc_control_info_per_axis(axis, False)
    else:
        for axis in my_utils.q_axes:
            calc_control_info_per_axis(axis, True)
            
    return control_info, accuracy, settling_time

if __name__ == '__main__':
    
    # Handle file parsing
    parser = argparse.ArgumentParser(prog="rigid_body_simulation")
    parser.add_argument("-l", "--log_output", help="filename to log ouptut", type=str)
    parser.add_argument("-t", "--testing", help="removes date from log file names", action='store_true')
    args = vars(parser.parse_args())
    log_file_name = args["log_output"]
    testing = args["testing"]

    if not testing and log_file_name is not None:
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        log_file_name += f"_{dt_string}"


    with open('config.toml', 'r') as config_file:
        config = toml.load(config_file)
    # logging.basicConfig(filename="logs.log", filemode="w+", level=logging.INFO)
    # logger.info("satellite simulation logs")
    results_data["time"] = []
    for axis in my_utils.xyz_axes:
        results_data["torque_" + axis] = []
        results_data["angular_acc_" + axis] = []
        results_data["angular_rate_" + axis] = []
    for axis in my_utils.q_axes:
        results_data["quaternion_" + axis] = []
    
    #------------------------------------------------------------#
    ###################### Set Up Objects ########################

    fault = body.Fault(config['fault']['time'], config['fault']['wheel_axis'], config['fault']['type'], config['fault']['torque_limit'])
    fault.master_enable = config['fault']['master_enable']
    fault.filter_coeff = config['fault']['filter_coef']

    wheel_module = body.WheelModule(mass=0.31, radius=0.066, height=0.025, config="standard")
    wheel_module.wheels[fault.wheel_num].fault = fault
    dir_init = Rotation.from_quat([0,0,0,1])
    satellite_angular_v_init = np.array([0,0,0])


    controller = body.Controller(k=config['controller']['k'], c=config['controller']['c'], T_max=config['controller']['T_max'], 
                                 T_min=config['controller']['T_min'], filter_coef=config['controller']['filter_coef'], 
                                 time_step=config['controller']['time_step'], type=config['controller']['type'],
                                 angular_v_init=np.zeros(3),quaternion_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init), adaptive_gain=1)
    controller.fault = fault

    satellite = Satellite(wheel_module, controller, fault)

    # Satellite Properties
    satellite.mass = 12 # 6U Sat weight limit
    satellite.dimensions = {'x': 0.2, 'y': 0.1, 'z': 0.3405} # 6U Sat dimension limit
    satellite.calc_M_inertia()

    # Satellite Initial Conditions
    satellite.dir_init_inertial = dir_init

    # Adaptive Controller Initialize
    satellite._controller.M_inertia_inv_model = satellite.M_inertia_inv
    satellite._controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(satellite.dir_init_inertial)

    # Control Variables
    satellite.ref_q = Rotation.from_euler("xyz", config['ref_euler_angle'], degrees=True)
    satellite.controller_enable = True
    satellite.wheels_control_enable = True

    quaternion_init = satellite.dir_init_inertial.as_quat()
    control_torque_init = [0, 0, 0]
    initial_values = np.hstack([satellite_angular_v_init, quaternion_init, control_torque_init])

    # Simulation parameters
    sim_time = config['simulation']['duration']
    sim_output_resolution_time = config['simulation']['resolution']
    sim_time_series = np.linspace(0, sim_time, int(sim_time/sim_output_resolution_time))

    #-------------------------------------------------------------#
    ###################### Simulate System ########################

    # Integrate satellite dynamics over time
    sol = solve_ivp(fun=satellite.calc_state, t_span=[0, sim_time], y0=initial_values, method="RK45",
                    t_eval=sim_time_series,
                    max_step=0.1)
    
    #----------------------------------------------------------#
    ###################### Process Data ########################
    # Put results into data object
    # print(f"adaptive model output len {len(results_data['adaptive_model_output'])}")
    # print(f"time series len {len(results_data['time'])}")
    results = [('torque', True), ('angular_acc', True)]
    if controller.type == "adaptive":
        results.append(('adaptive_model_output', False))
    for element in results:
        data, multiple_axes = element
        if multiple_axes:
            for axis in my_utils.xyz_axes:
                results_data[f'{data}_{axis}'] = interpolate_data(results_data[f'{data}_{axis}'], results_data['time'], sim_time_series)[:]
        else: 
            pass   
            # results_data[f'{data}'] = interpolate_data(results_data[f'{data}'], np.linspace(0, sim_time, int(sim_time/controller.time_step)), sim_time_series)[:]
    
    results_data['time'] = sim_time_series

    for i,axis in enumerate(my_utils.xyz_axes):
        results_data[f'angular_rate_{axis}'] = sol.y[i]

    for i,axis in enumerate(my_utils.q_axes):
        results_data[f'quaternion_{axis}'] = sol.y[i+3]
    
    results_data['euler_int'] = []
    for row in range(len(results_data['time'])):
        r : Rotation = Rotation.from_quat([results_data['quaternion_x'][row],
                                          results_data['quaternion_y'][row],
                                          results_data['quaternion_z'][row],
                                          results_data['quaternion_w'][row]])
        alpha = my_utils.conv_Rotation_obj_to_euler_int(r)
        results_data['euler_int'].append(alpha)

    if config['output']['energy_enable']:
        control_energy_arr = sol.y[7:]
        control_energy_per_axis = {}
        control_energy_total = 0
        for i,axis in enumerate(my_utils.xyz_axes):
            control_energy_per_axis[axis] = np.sum(np.abs(control_energy_arr[i]))
            control_energy_total += control_energy_per_axis[axis]
            results_data[f'control_energy_{axis}'] = control_energy_arr[i]
        print(f"control energy (J): {my_utils.round_dict_values(control_energy_per_axis,3)} | total: {round(control_energy_total,3)}")
    
    

    # axes_commanded = []
    # for i, axis in enumerate(my_utils.xyz_axes):
    #     if config['ref_euler_angle'][i] > 0:
    #         axes_commanded.append(axis)
    # axes_commanded.append('w')

    if config['output']['accuracy_enable']:
        # control_info, accuracy, settling_time = calc_control_data(axes_commanded)

        control_info = control.step_info(sysdata=results_data[f"euler_int"], T=results_data['time'])
        euler_int_ref = my_utils.conv_Rotation_obj_to_euler_int(satellite.ref_q)
        accuracy = np.abs((control_info['SteadyStateValue']-euler_int_ref)/euler_int_ref)
        settling_time = control_info['SettlingTime']

        # accuracy_percent = dict( (axis, accuracy[axis]*100 if accuracy[axis] is not None else None ) for axis in my_utils.q_axes)
        accuracy_percent = accuracy*100
        
        print(f"accuracy %: {accuracy}")
        print(f"settling_time (s): {round(settling_time,3)}")
        final_q = [results_data[f'quaternion_{axis}'][-1] for axis in my_utils.q_axes]
        print(f"final attitude (quaternion): {[round(q,3) for q in final_q]}")



    if log_file_name != None:
        output_to_csv(log_file_name)
    # yaw_pitch_roll_output = False
    # if( not yaw_pitch_roll_output):

    #--------------------------------------------------------------#
    ###################### Output to Graphs ########################
    fig = plt.figure(figsize=(18,8))
    fig.tight_layout()

    cols = 4
    rows = [ ('angular_rate','xyz'), ('quaternion','quaternion'), ('euler_int','none'), ('torque','xyz'), ('control_energy','xyz')]
    
    # if controller.type == "adaptive":
    #     rows.append(('adaptive_model_output','none'))

    for row_idx, row in enumerate(rows):
        row_name, axes_type = row
        pos = cols*row_idx+1
        if axes_type == 'quaternion':
            axes = my_utils.q_axes
        elif axes_type == 'none':
            axes = ['']
        else:
            axes = my_utils.xyz_axes
        for col_idx,axis in enumerate(axes):
            plt.subplot(len(rows),cols,pos+col_idx)
            if axes_type != 'none': 
                name = row_name + "_" + axis
            else: 
                name = row_name
            plt.plot(results_data['time'], results_data[name])
            plt.xlabel('time (s)')
            plt.ylabel(f' {name}')
        
        # else:

        #     # Convert data to yaw pitch and roll
        #     y_transpose = [list(x) for x in zip(*sol.y)] # transpose columns and rows
        #     yaw_pitch_roll_values = list(map(calc_yaw_pitch_roll_rates,y_transpose))
        #     yaw_pitch_roll_values = [list(x) for x in zip(*yaw_pitch_roll_values)]

        #     for i in range(cols):
        #         if(i==0):
        #             title = 'yaw'
        #         elif(i==1):
        #             title = 'pitch'
        #         elif(i==2):
        #             title = 'roll'
        #         plt.subplot(rows,cols,i+1)
        #         plt.plot(sol.t, yaw_pitch_roll_values[i])
        #         plt.xlabel('time (s)')
        #         plt.ylabel(f'{title} angle (rad)')

        #         plt.subplot(rows,cols,i+4)
        #         plt.plot(sol.t, yaw_pitch_roll_values[i+3]) 
        #         plt.xlabel('time (s)')
        #         plt.ylabel(f'{title} angular rate (rad/s)')

    plt.subplots_adjust(wspace=1, hspace=0.5)
    
    # Make plot full screen
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    plt.show()

    fig.savefig(fr"..\data_logs\{log_file_name}.pdf", bbox_inches='tight')


