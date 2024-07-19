import numpy as np
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
import csv
import os
# all units are in SI (m, s, N, kg.. etc)

results_data = my_globals.results_data

class Face():
    r_cop_to_com = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite(body.Body):

    logger = None
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    _controller : body.Controller = None
    _wheel_module : body.WheelModule = None
    _fault : body.Fault = None

    mode : str = ""
    modes = ["direction", "torque"]

    dir_init_obc = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    _disturbances : body.Disturbances = None
    def __init__(self, dimensions, mass, _wheel_module : body.WheelModule, controller : body.Controller, 
                 fault : body.Fault, wheel_offset = 0, logger = None, mode = "direction"):
        self._wheel_module = wheel_module
        if _wheel_module.config == "standard":
            self._wheel_module.wheels[0].position[0] = self.dimensions['x']-wheel_offset
            self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
            self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        self._controller = controller
        self._next_control_time_step = controller.time_step
        self.logger = logger
        self._fault = fault
        if mode not in self.modes:
            raise Exception(f"mode {mode} not available")
        self.mode = mode
        self.dimensions, self.mass = dimensions, mass
        self._disturbances = body.Disturbances()
        self.calc_face_properties()

    def calc_face_properties(self):
        dim_array = np.array([self.dimensions['x'], self.dimensions['y'], self.dimensions['z']])
        for i in range(3):
            for j in range(2):
                face = Face()
                face.norm_vec = np.zeros(3)
                face.norm_vec[i] = 1*(1,-1) [j == 1]
                face.area = 1
                for k, axis in enumerate(face.norm_vec):
                    if axis == 0:
                        face.area *= dim_array[k]
                face.r_cop_to_com = face.norm_vec*0.5*dim_array              
                self.faces.append(face)

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
    ref_T = np.zeros(3)
    wheels_control_enable = True
    _next_control_time_step = 0.1
    wheels_H_rate_result = np.zeros(3)
    wheels_H_result = np.zeros(3)
    wheels_speed = np.zeros(3)
    T_controller_com = np.zeros(3)
    def calc_state_rates(self, t, y):
        sat_angular_v_input = y[:3]
        q_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        sat_angular_acc_result = [0]*3
        q_rate_result = np.quaternion(1,0,0,0)
        if self._controller.enable is True:
        
            if t > self._next_control_time_step:
                if self.mode == "direction":
                    self.T_controller_com = self._controller.calc_torque_control_output(q_input, sat_angular_v_input, self.ref_q, t)
                elif self.mode == "torque":
                    self.T_controller_com = self.ref_T
                self._next_control_time_step += self._controller.time_step
                if self.wheels_control_enable:
                    self.wheels_H_rate_result, self.wheels_H_result = self._wheel_module.calc_state_rates(self.T_controller_com, controller.time_step)

            if t > self._fault.time and self._fault.master_enable:
                self._fault.enabled = True
        if self.wheels_control_enable:
            T_controller = self.wheels_H_rate_result
        else:
            T_controller = self.T_controller_com
        
        T_aero = self._disturbances.calc_aero_torque(satellite, q_input)
        T_grav = self._disturbances.calc_grav_torque(satellite, q_input)
        T_dist = T_aero + T_grav

        Hnet = self.M_inertia@(sat_angular_v_input) + self.wheels_H_result
        sat_angular_acc_result = self.M_inertia_inv@(T_controller + T_dist - np.cross(sat_angular_v_input,Hnet))

        # put the inertial velocity in q form
        inertial_v_q = np.quaternion(0, sat_angular_v_input[0], sat_angular_v_input[1], sat_angular_v_input[2])

        q_rate_result = 0.5*q_input*inertial_v_q
        q_rate_result = [q_rate_result.x, q_rate_result.y, q_rate_result.z, q_rate_result.w]
        results_data['time'].append(t)
        for i, axis in enumerate(my_utils.xyz_axes):
            results_data['torque_'+ axis].append(T_controller[i])
            results_data['angular_acc_'+ axis].append(sat_angular_acc_result[i])
            results_data['T_dist_' + axis].append(T_dist[i])
            
        for i, wheel in enumerate(self._wheel_module.wheels):
            results_data['wheel_speed_' + str(i)].append(wheel.speed)
            
        if self.wheels_control_enable:
            control_power_result = np.abs(self.wheels_H_rate_result * sat_angular_v_input)
        else:
            control_power_result = np.abs(self.T_controller_com * sat_angular_v_input)


        results = [sat_angular_acc_result, q_rate_result, control_power_result]
        return np.hstack(results)
    
# END DEF class Satellite()

def output_dict_to_csv(path, file_name, data):
    df = pd.DataFrame().from_dict(data)

    with open(fr'{path}\{file_name}.csv', 'w+') as file:
        df.to_csv(file,sep=',')

def log_to_file(path, file_name, string, print_c=True):
    if print_c:
        print(string)
    with open(fr'{path}\{file_name}.csv', 'a+') as file:
        file.write(string)
        


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

    path = fr'..\data_logs\{log_file_name}'
    if not os.path.exists(path) and log_file_name != None:
        os.mkdir(path)

    with open('config.toml', 'r') as config_file:
        config = toml.load(config_file)

    results_data["time"] = []
    for axis in my_utils.xyz_axes:
        results_data["torque_" + axis] = []
        results_data["angular_acc_" + axis] = []
        results_data["angular_rate_" + axis] = []
        results_data['T_dist_' + axis] = []
    for axis in my_utils.q_axes:
        results_data["quaternion_" + axis] = []
    
    #------------------------------------------------------------#
    ###################### Set Up Objects ########################

    fault = body.Fault(config['fault']['time'], config['fault']['wheel_axis'], config['fault']['type'], config['fault']['torque_limit'])
    fault.master_enable = config['fault']['master_enable']
    fault.filter_coeff = config['fault']['filter_coef']

    wheels_config = config['wheels']
    wheel_module = body.WheelModule(wheels_config['mass'], wheels_config['radius'], wheels_config['height'], 
                                    wheels_config['config'], my_utils.conv_rpm_to_rads_per_sec(wheels_config['max_speed_rpm']), wheels_config['max_torque'])
    wheel_module.wheels[fault.wheel_num].fault = fault
    if config['satellite']['euler_init_en']:
        dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
    else:
        dir_init = Rotation.from_quat(config['satellite']['q_init'])
    satellite_angular_v_init = np.array([0,0,0])


    controller = body.Controller(config['controller']['enable'], k=config['controller']['k'], c=config['controller']['c'], T_max=config['controller']['T_max'], 
                                 T_min=config['controller']['T_min'], filter_coef=config['controller']['filter_coef'], 
                                 time_step=config['controller']['time_step'], type=config['controller']['type'],
                                 angular_v_init=np.zeros(3),quaternion_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init), 
                                 adaptive_gain=config['controller']['adaptive_gain'])
    controller.fault = fault

    satellite = Satellite(config['satellite']['dimensions'], config['satellite']['mass'],
                          wheel_module, controller, fault, mode=config['satellite']['mode'])
    
    satellite.calc_M_inertia()

    # Satellite Initial Conditions
    satellite.dir_init_obc = dir_init

    # Adaptive Controller Initialize
    satellite._controller.M_inertia_inv_model = satellite.M_inertia_inv
    satellite._controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(satellite.dir_init_obc)

    # Control Variables
    satellite.ref_q = Rotation.from_euler("xyz", config['satellite']['ref_euler_angle'], degrees=True)
    satellite.ref_T = np.array(config['satellite']['ref_T'])
    satellite.wheels_control_enable = config['satellite']['wheels_control_enable']

    quaternion_init = satellite.dir_init_obc.as_quat()
    control_torque_init = [0, 0, 0]
    initial_values = [satellite_angular_v_init, quaternion_init, control_torque_init]
    # if satellite.wheels_control_enable is True:
    #     initial_values.append(np.zeros(3))

    initial_values = np.hstack(initial_values)

    # Simulation parameters
    sim_config = config['simulation']
    sim_time = sim_config['duration'] if not sim_config['test_mode_en'] else sim_config['test_duration']
    sim_output_resolution_time = sim_config['resolution'] if not sim_config['test_mode_en'] else sim_config['test_resolution']
    
    sim_time_series = np.linspace(0, sim_time, int(sim_time/sim_output_resolution_time))

    #-------------------------------------------------------------#
    ###################### Simulate System ########################

    # Integrate satellite dynamics over time
    sol = solve_ivp(fun=satellite.calc_state_rates, t_span=[0, sim_time], y0=initial_values, method="RK45",
                    t_eval=sim_time_series,
                    max_step=0.1)
    
    #----------------------------------------------------------#
    ###################### Process Data ########################
    
    # Put results into data object
    for key, value in results_data.items():
        if key == 'time':
            continue
        if len(value) > int(sim_time/sim_output_resolution_time):
            if key[:7] == "control":
                results_data[key] = interpolate_data(value, results_data['control_time'], sim_time_series)[:]
            else:
                results_data[key] = interpolate_data(value, results_data['time'], sim_time_series)[:]
    
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
        control_energy_log_output = f"control energy (J): {my_utils.round_dict_values(control_energy_per_axis,3)} | total: {round(control_energy_total,3)}"
        print(control_energy_log_output)
    

    if config['output']['accuracy_enable']:
        control_info = control.step_info(sysdata=results_data[f"euler_int"], T=results_data['time'])
        euler_int_ref = my_utils.conv_Rotation_obj_to_euler_int(satellite.ref_q)
        accuracy = 1-np.abs((control_info['SteadyStateValue']-euler_int_ref)/euler_int_ref)
        settling_time = control_info['SettlingTime']

        accuracy_percent = accuracy*100
        
        final_q = [results_data[f'quaternion_{axis}'][-1] for axis in my_utils.q_axes]
        print(f"accuracy %: {accuracy_percent}")
        print(f"settling_time (s): {round(settling_time,3)}")
        print(f"accuracy %: {accuracy_percent}")

    if log_file_name != None:
        output_dict_to_csv(path, log_file_name + "_log", results_data)
        output_dict_to_csv(path, log_file_name + "_config", config)
        log_to_file(path, log_file_name, f"accuracy %: {accuracy_percent}", False)
        log_to_file(path, log_file_name, f"settling_time (s): {round(settling_time,3)}",    False)
        log_to_file(path, log_file_name, f"accuracy %: {accuracy_percent}", False)
        log_to_file(path, log_file_name, control_energy_log_output, False)
        

    # yaw_pitch_roll_output = False
    # if( not yaw_pitch_roll_output):

    #--------------------------------------------------------------#
    ###################### Output to Graphs ########################
    fig = plt.figure(figsize=(18,8))
    fig.tight_layout()

    cols = 4
    rows = [ ('angular_rate',my_utils.xyz_axes), ('quaternion',my_utils.q_axes), ('euler_int',['none']), ('torque',my_utils.xyz_axes), \
            ('control_energy',my_utils.xyz_axes), ('T_dist', my_utils.xyz_axes)]
    
    if satellite.wheels_control_enable:
        rows.append(('wheel_speed', ['0','1','2']))

    if controller.type == "adaptive":
        rows.append(('control_adaptive_model_output',['none']))
        rows.append(('control_theta',my_utils.xyz_axes))

    for row_idx, row in enumerate(rows):
        row_name, axes = row
        pos = cols*row_idx+1
        for col_idx,axis in enumerate(axes):
            plt.subplot(len(rows),cols,pos+col_idx)
            if axis != 'none': 
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

    if log_file_name != None:
        fig.savefig(fr"..\data_logs\{log_file_name}\{log_file_name}.pdf", bbox_inches='tight')


