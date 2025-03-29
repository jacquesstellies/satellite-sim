import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
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
import json
from inspect import currentframe, getframeinfo

DEBUG = True

# all units are in SI (m, s, N, kg.. etc)

results_data = my_globals.results_data

class Face():
    r_cop_to_com = np.zeros(3)
    area = 0
    norm_vec = np.zeros(3)
    
class Satellite(body.Body):

    _logger = None
    wheel_offset = 0 # offset of wheel center of mass from edge of device

    _controller : body.Controller = None
    _wheel_module : body.WheelModule = None
    _fault : body.Fault = None

    mode : str = ""
    modes = ["direction", "torque"]

    dir_init = Rotation.from_quat([0,0,0,1])

    faces : list[Face] = []
    _disturbances : body.Disturbances = None

    config = None
    def __init__(self, _wheel_module : body.WheelModule, controller : body.Controller, 
                 fault : body.Fault, wheel_offset = 0, logger = None, config=None):
        self.config = config
        self._wheel_module = wheel_module
        if _wheel_module.config == "standard":
            self._wheel_module.wheels[0].position[0] = self.dimensions['x']-config
            self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
            self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        # if _wheel_module.config == "pyramid":
        #     self._wheel_module.wheels[0].position[0] = self.dimensions['x']-wheel_offset
        #     self._wheel_module.wheels[1].position[1] = self.dimensions['y']-wheel_offset
        #     self._wheel_module.wheels[2].position[2] = self.dimensions['z']-wheel_offset
        #     self._wheel_module.wheels[3].position[0] = self.dimensions['x']-wheel_offset
        self._controller = controller
        self._next_control_time_step = controller.time_step
        self._logger = logger
        self._fault = fault
        self.mode = config['satellite']['mode']
        if self.mode not in self.modes:
            raise Exception(f"mode {self.mode} not available")
        
        self.dimensions, self.mass = config['satellite']['dimensions'], config['satellite']['mass']

        if self.config['satellite']['inertia_override']:
            M_inertia = np.array(config['satellite']['M_Inertia'])
            if M_inertia.shape == (3,3):
                self.M_inertia = M_inertia
            elif M_inertia.shape == (3,):
                self.M_inertia = np.diag(M_inertia)
            else:
                raise Exception("inertia override must be 3x3 or 3x1 matrix")
            print("Satellite inertia ",self.M_inertia)
            self.calc_M_inertia_inv()
        else:
            self.calc_M_inertia()
        self._disturbances = body.Disturbances()
        self.calc_face_properties()
        self.ref_T = np.array(config['satellite']['ref_T'])
        self.wheels_control_enable = config['satellite']['wheels_control_enable']
        if not self.wheels_control_enable and self._controller.type == "backstepping" and self._controller.sub_type == "Shen":
            raise Exception("this backstepping controller requires wheels control enabled")

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
        self.M_inertia = self.calc_M_inertia_body() #+ self.calc_M_inertia_peri()

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
            if t == 0.0:
                self.T_controller_com = self._controller.calc_torque_control_output(q_input, sat_angular_v_input, self.ref_q, self, self._wheel_module.H, t)
            if t > self._next_control_time_step:
                if self.mode == "direction":
                    self.T_controller_com = self._controller.calc_torque_control_output(q_input, sat_angular_v_input, self.ref_q, self, self._wheel_module.H, t)
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
        T_dist = (T_aero + T_grav)

        if self.config['controller']['type'] == "adaptive" and self.config['controller']['sub_type'] == "Shen":
            T_dist = self._disturbances.calc_dist_torque_Shen(t)

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
            results_data['wheel_torque_' + str(i)].append(wheel.T)

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

def output_toml_to_file(path, file_name, data):
    with open(fr'{path}\{file_name}.toml', 'w+') as file:
        toml.dump(data, file)

def log_to_file(path, file_name, string, print_c=True):
    if print_c:
        print(string)
    with open(fr'{path}\{file_name}.csv', 'a+') as file:
        file.write(string+'\n')
        
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

def parse_args():
    parser = argparse.ArgumentParser(prog="rigid_body_simulation")
    parser.add_argument("-l", "--log_output", help="filename to log ouptut", type=str)
    parser.add_argument("-d", "--append_date", help="adds date to log file names", action='store_true')
    parser.add_argument("-a", "--append", help="appends text to log file names", type=str)
    args = vars(parser.parse_args())

    with open('config.toml', 'r') as config_file:
        config = toml.load(config_file)
    
    append_date = args["append_date"]
    if append_date is None:
        append_date = config['output']['append_date']
    
    if args["log_output"] is None:
        if config['output']['log_enable'] is False:
            LOG_FILE_NAME = None
        else:
            overide = config['output']['log_file_name_overide']
            if overide == "None" or overide == "":
                LOG_FILE_NAME = config['controller']['type']
                if(config['controller']['type'] != "q_feedback"):
                    LOG_FILE_NAME += ('_' + config['controller']['sub_type'])
                LOG_FILE_NAME += '_' + config['wheels']['config']
                if config['fault']['master_enable']:
                    LOG_FILE_NAME += '_fault'
    else:
        LOG_FILE_NAME = args["log_output"]


    if append_date is True and LOG_FILE_NAME is not None:
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        LOG_FILE_NAME += f"_{dt_string}"
    if args['append'] is not None:
        LOG_FILE_NAME += f"_{args['append']}"

    LOG_FOLDER_PATH = fr'..\data_logs\{LOG_FILE_NAME}'
    if not os.path.exists(LOG_FOLDER_PATH) and LOG_FILE_NAME != None:
        os.mkdir(LOG_FOLDER_PATH)
    
    print(f"output name is {LOG_FILE_NAME}")
    print(f"output folder is {LOG_FOLDER_PATH}")
    return LOG_FILE_NAME, LOG_FOLDER_PATH, config

def clear_log_file(log_file_path):
    with open(log_file_path, 'w') as file:
        file.write("")

if __name__ == '__main__':
    LOG_FILE_NAME, LOG_FOLDER_PATH, config = parse_args()

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

    fault = body.Fault(config)

    wheel_module = body.WheelModule(config, fault)
    if config['satellite']['euler_init_en']:
        dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
    else:
        dir_init = Rotation.from_quat(config['satellite']['q_init'])
    satellite_angular_v_init = np.array([0,0,0])


    controller = body.Controller(fault=fault,angular_v_init=np.zeros(3),quaternion_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init),
                                 config=config)

    satellite = Satellite(wheel_module, controller, fault,config=config)

    # Satellite Initial Conditions
    satellite.dir_init = dir_init

    # Adaptive Controller Initialize
    satellite._controller.M_inertia_inv_model = satellite.M_inertia_inv
    satellite._controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(satellite.dir_init)

    # Control Variables
    if config['satellite']['use_ref_euler']:
        satellite.ref_q = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True)
    elif config['satellite']['use_ref_q']:
        satellite.ref_q = Rotation.from_quat(config['satellite']['ref_q'])
    else:
        raise(Exception("no reference angle commanded"))

    quaternion_init = satellite.dir_init.as_quat()
    control_torque_init = [0, 0, 0]
    initial_values = [satellite_angular_v_init, quaternion_init, control_torque_init]
    # if satellite.wheels_control_enable is True:
    #     initial_values.append(np.zeros(3))

    initial_values = np.hstack(initial_values)

    test_mode_en = config['simulation']['test_mode_en']
    # Simulation parameters
    sim_config = config['simulation']
    sim_time = sim_config['duration'] if not test_mode_en else sim_config['test_duration']
    sim_output_resolution_time = sim_config['resolution'] if not test_mode_en else sim_config['test_resolution']
    
    sim_time_series = np.linspace(0, sim_time, int(sim_time/sim_output_resolution_time))

    if test_mode_en:
        print("NB ********* Test Mode is ENABLED *********")
    #-------------------------------------------------------------#
    ###################### Simulate System ########################
    if config['simulation']['enable']:
        # Integrate satellite dynamics over time
        sol = solve_ivp(fun=satellite.calc_state_rates, t_span=[0, sim_time], y0=initial_values, method="RK45",
                    t_eval=sim_time_series,
                    max_step=0.1)
    else:
        exit()
    print("Simulation Complete")
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
                try:
                    results_data[key] = interpolate_data(value, results_data['time'], sim_time_series)[:]
                except:
                    print(f"Error interpolating {key} {value}")
    
    results_data['time'] = sim_time_series

    for i,axis in enumerate(my_utils.xyz_axes):
        results_data[f'angular_rate_{axis}'] = sol.y[i]

    for i,axis in enumerate(my_utils.q_axes):
        results_data[f'quaternion_{axis}'] = sol.y[i+3]
    
    results_data['euler_axis'] = []
    for row in range(len(results_data['time'])):
        r_1 : Rotation = Rotation.from_quat([results_data['quaternion_x'][row],
                                        results_data['quaternion_y'][row],
                                        results_data['quaternion_z'][row],
                                        results_data['quaternion_w'][row]])
        # r_2 : Rotation = dir_refer
        alpha = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_1)
        results_data['euler_axis'].append(alpha)
    
    results_data['euler_int'] = cumulative_trapezoid(results_data['euler_axis'], results_data['time'], initial=0)

    df = pd.DataFrame(results_data)
    q = np.array([results_data["quaternion_x"], 
                                        results_data["quaternion_y"], 
                                        results_data["quaternion_z"], 
                                        results_data["quaternion_w"]])
    r =  Rotation.from_quat(quat=q.T)
    [results_data['euler_yaw'],results_data['euler_pitch'], results_data['euler_roll']] = r.as_euler('zyx', degrees=True).T
    # e321 = r.as_euler('zyx', degrees=True)

    # print(results_data['euler_1'])
    # exit()
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
    
    accuracy_percent = None
    settling_time = None
    if config['output']['accuracy_enable']:
        try:
            euler_axis_init = my_utils.conv_Rotation_obj_to_euler_axis_angle(satellite.dir_init)
            euler_axis_ref = my_utils.conv_Rotation_obj_to_euler_axis_angle(satellite.ref_q)
            control_info = control.step_info(sysdata=results_data[f"euler_int"], 
                                             SettlingTimeThreshold=0.002, T=results_data['time'])
            steady_state = control_info['SteadyStateValue']
            accuracy = 1-np.abs((results_data[f"euler_axis"][-1]-euler_axis_ref)/(euler_axis_ref-euler_axis_init))
            settling_time = control_info['SettlingTime']

            accuracy_percent = accuracy*100
            
            final_q = [results_data[f'quaternion_{axis}'][-1] for axis in my_utils.q_axes]
            final_euler  = Rotation.from_quat(final_q).as_euler('xyz', degrees=True)
            print(f"accuracy %: {accuracy_percent}")
            print(f"settling_time (s): {round(settling_time,3)}")
            print(f"steady_state (s): {round(steady_state,3)}")
            # print(f"final euler: {final_euler} deg xyz")
            # print(f"euler error: {euler_axis_ref - final_euler} deg xyz")

            control_info_y = control.step_info(sysdata=results_data['euler_yaw'],SettlingTimeThreshold=0.002, T=results_data['time'])
            control_info_p = control.step_info(sysdata=results_data['euler_pitch'],SettlingTimeThreshold=0.002, T=results_data['time'])
            control_info_r = control.step_info(sysdata=results_data['euler_roll'],SettlingTimeThreshold=0.002, T=results_data['time'])
            print(f"steady state error: {control_info_y['SteadyStateValue']} {control_info_p['SteadyStateValue']} {control_info_r['SteadyStateValue']} deg zyx")
        except Exception as e:
            print(f"Error calculating accuracy: {e}")

    if LOG_FILE_NAME != None and test_mode_en is False:
        LOG_FILE_NAME_RESULTS = LOG_FILE_NAME + "_results"
        clear_log_file(fr"{LOG_FOLDER_PATH}\{LOG_FILE_NAME_RESULTS}")
        output_dict_to_csv(LOG_FOLDER_PATH, LOG_FILE_NAME + "_log", results_data)
        output_toml_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME + "_config", config)
        if accuracy_percent is not None:
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"accuracy %: {accuracy_percent}", False)
        if settling_time is not None:
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"settling_time (s): {round(settling_time,3)}",    False)
        log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"accuracy %: {accuracy_percent}", False)
        log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, control_energy_log_output, False)
    
    #--------------------------------------------------------------#
    ###################### Output to Graphs ########################
    cols = 2
    rows = [ ('angular_rate',my_utils.xyz_axes), ('quaternion',my_utils.q_axes), ('euler_int',['none']), ('euler', ['yaw','pitch', 'roll']), \
            ('torque',my_utils.xyz_axes), ('control_energy',my_utils.xyz_axes), ('T_dist', my_utils.xyz_axes)]
    
    if satellite.wheels_control_enable:
        rows.append(('wheel_speed', [str(wheel.index) for wheel in wheel_module.wheels]))
        rows.append(('wheel_torque', [str(wheel.index) for wheel in wheel_module.wheels]))


    if controller.type == "adaptive":
        rows.append(('control_adaptive_model_output',['none']))
        rows.append(('control_theta',my_utils.xyz_axes))

    # Create separate figures if enabled in config
    for row_idx, row in enumerate(rows):
        row_name, axes = row
        fig_separate = plt.figure(figsize=(12,6))
        ax_separate = fig_separate.add_subplot(111)
        
        for axis in axes:
            if axis != 'none':
                name = row_name + "_" + axis
            else:
                name = row_name
            ax_separate.plot(results_data['time'], results_data[name], label=axis)
        
        ax_separate.set_xlabel('time (s)')
        ax_separate.set_ylabel(f'{row_name}')
        ax_separate.legend()
        
        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None:
            if not os.path.exists(fr"..\data_logs\{LOG_FILE_NAME}\graphs"):
                os.mkdir(fr"..\data_logs\{LOG_FILE_NAME}\graphs")
            fig_separate.savefig(fr"..\data_logs\{LOG_FILE_NAME}\graphs\{LOG_FILE_NAME}_{row_name}.pdf", bbox_inches='tight')
        if config['output']['separate_plots_display'] is False:
            plt.close(fig_separate)

    fig, ax= plt.subplots(int(np.ceil(len(rows)/cols)),cols,sharex=True,figsize=(18,8))

    ax_as_np_array= np.array(ax)
    plots_axes = ax_as_np_array.flatten()
    for row_idx, row in enumerate(rows):
        row_name, axes = row
        current_plot = plots_axes[row_idx-1]
        for axis in axes:
            if axis != 'none': 
                name = row_name + "_" + axis
            else: 
                name = row_name
            current_plot.plot(results_data['time'], results_data[name], label=axis)

        current_plot.set_xlabel('time (s)')
        current_plot.set_ylabel(f'{row_name}')
        current_plot.legend()

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()

    if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and test_mode_en is False:

        fig.savefig(fr"..\data_logs\{LOG_FILE_NAME}\{LOG_FILE_NAME}_summary.pdf", bbox_inches='tight')

