from observer import WheelExtendedStateObserver, ObserverModule
from satellite import Satellite, DivergentRate
from controller import Controller
from fault import Fault, FaultModule
from wheels import WheelModule
from magt import MagtModule
from orbit import Orbit
from logger import Logger
import visualizer as viz

import my_utils
import my_globals

import os
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import toml
import datetime
import time
import control
from inspect import currentframe, getframeinfo
import cProfile
import pyswarms
import threading

DEBUG = True
ROOT_DIR = os.popen("git rev-parse --show-toplevel").read().strip()

# all units are in SI (m, s, N, kg.. etc)

# results_data = my_globals.results_data

## Reference Frames:
# G -> ECI (Earth Centered Inertial)
# B -> Body Frame (Satellite Body Frame)
# I -> Body Inertial Frame (Non-Rotating wrt ECI)
# S -> Sun Frame

## Name conventions:
# w -> angular velocity
# d{} -> derivative of {}
# q -> quaternion
# T -> torque
# M_inertia -> moment of inertia matrix
# H -> angular momentum
# r/s -> position vector
# v -> velocity vector
# n -> unit vector

# END DEF class Satellite()

def output_dict_to_csv(path, file_name, data):
    df = pd.DataFrame().from_dict(data)

    with open(fr'{path}/{file_name}.csv', 'w+') as file:
        df.to_csv(file,sep=',')

def output_toml_to_file(path, file_name, data):
    with open(fr'{path}/{file_name}.toml', 'w+') as file:
        toml.dump(data, file)

def log_to_file(path, file_name, string, print_c=True):
    if print_c:
        print(string)
    with open(fr'{path}/{file_name}.csv', 'a+') as file:
        file.write(string+'\n')
        
def interpolate_data(data, time_series, time_series_new):
    return np.interp(time_series_new, time_series, data)

def create_default_log_file_name(config):
    
    filename = config['controller']['type']
    
    if(config['controller']['type'] != "pid"):
        filename += ('_' + config['controller']['sub_type'])
    
    filename += '_' + config['wheels']['config']
    
    if config['faults']['master_enable']:
        filename += '_fault'
    else:
        filename += '_nom'
    
    if config['simulation']['iterations'] > 1:
        filename += '_mc'

    return filename

def parse_args():
    parser = argparse.ArgumentParser(prog="rigid_body_simulation")
    parser.add_argument("-o", "--output_name", help="filename to log ouptut", type=str)
    parser.add_argument("-d", "--append_date", help="adds date to log file names", action='store_true')
    parser.add_argument("-a", "--append", help="appends text to log file names", type=str)
    parser.add_argument("-t", "--test_mode", help="enable test mode", action='store_true')
    parser.add_argument("-k", "--disable_sim", help="disable simulation", action='store_true')
    # parser.add_argument("-c", "--config_override", help="override config values with a toml file", type=str)
    parser.add_argument("-c", "--config", help="pass config file location", type=str)
    parser.add_argument("-V", "--visualize", help="visualize the results after simulating", action='store_true')
    args = vars(parser.parse_args())

    config_path = "config.toml"

    if args["config"] is not None:
        config_path = args["config"]
    
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    
    append_date = args["append_date"]
    if append_date is None:
        append_date = config['output']['append_date']
    
    if args["output_name"] is None:
        if config['output']['log_enable'] is False:
            LOG_FILE_NAME = None
        else:
            overide = config['output']['log_file_name_overide']
            if overide == "None" or overide == "":
                LOG_FILE_NAME = create_default_log_file_name(config)
    else:
        LOG_FILE_NAME = args["output_name"]

    if args["test_mode"] is True:
        config['simulation']['test_mode_en'] = True
    
    if args["disable_sim"] is True:
        config['simulation']['enable'] = False

    if append_date is True and LOG_FILE_NAME is not None:
        dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        LOG_FILE_NAME += f"_{dt_string}"
    if args['append'] is not None:
        LOG_FILE_NAME += f"_{args['append']}"

    # ROOT_DIR = os.popen("git rev-parse --show-toplevel").read().strip()
    LOG_FOLDER_BASE_PATH = os.path.abspath(f'{ROOT_DIR}/data_logs')
    LOG_FOLDER_PATH = os.path.join(LOG_FOLDER_BASE_PATH,LOG_FILE_NAME)
    if not os.path.exists(LOG_FOLDER_BASE_PATH) and LOG_FILE_NAME != None:
        raise Exception(f"Log folder {LOG_FOLDER_BASE_PATH} does not exist")
    
    print(f"output name is {LOG_FILE_NAME}")
    print(f"output folder is {LOG_FOLDER_PATH}")

    with open(f"{ROOT_DIR}/last_log.txt", "w") as file:
        file.write(LOG_FOLDER_PATH if LOG_FILE_NAME != None else "No logging")

    if args["visualize"] is True:
        print("Visualiztion enabled")
        config['output']['visualizer']['enable'] = True

    return LOG_FILE_NAME, LOG_FOLDER_PATH, config, args

def clear_log_file(log_file_path):
    with open(log_file_path, 'w+') as file:
        file.write("")

class Simulation:
    satellite : Satellite = None
    sim_time_series = None
    sim_time = 0
    config = None
    monte_carlo = False
    results_data = None
    results_df : pd.DataFrame = None
    iter = 0

    def __init__(self, config, results_data, logging_en=True):
        self.config = config
        self.results_data = results_data

        #------------------------------------------------------------#
        ###################### Set Up Objects ########################
        self.fault_module = FaultModule(config)
        wheel_module = WheelModule(config, self.fault_module.faults)
        if config['satellite']['euler_init_en']:
            dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
        else:
            dir_init = Rotation.from_quat(config['satellite']['q_init'])

        w_sat_init = np.array([0,0,0])
        controller = Controller(faults=self.fault_module.faults, wheel_module=wheel_module, results_data=results_data, w_sat_init=np.zeros(3), q_sat_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init),
                                    config=config)
        observer_module = ObserverModule(config, wheel_module)
        orbit = Orbit(config)
        magt_module = MagtModule(config, orbit)
        self.logger = Logger()
        # self.logger = Logger(config, results_data, self.satellite, logger_fields=None)
        self.satellite = Satellite(wheel_module, controller, observer_module, self.fault_module, magt_module, self.logger, orbit=orbit, config=config)

        #------------------------------------------------------------#
        ###################### Set Up Initial Conditions ########################
        # Satellite Initial Conditions
        self.satellite.dir_init = dir_init
        self.fault_module.init(wheel_module.num_wheels)
        self.logger.init(config, results_data, self.satellite, enable = config['output']['log_enable'] and logging_en, logger_fields=None)

        # Adaptive Controller Initialize
        self.satellite.controller.M_inertia_inv_model = self.satellite.M_inertia_inv
        self.satellite.controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(self.satellite.dir_init)

        q_sat_init = self.satellite.dir_init.as_quat()
        control_torque_init = np.zeros(3)
        w_wheels_init = np.zeros((self.satellite.wheel_module.num_wheels))
        self.initial_values = np.concatenate([w_sat_init, q_sat_init, w_wheels_init, control_torque_init])

        # Simulation parameters
        sim_config = config['simulation']
        self.sim_time = sim_config['duration'] if not config['simulation']['test_mode_en'] else sim_config['test_duration']
        
        if config['simulation']['test_mode_en']:
            print("NB ********* Test Mode is ENABLED *********")
    
    def clear_results_data(self):
        for entry in self.results_data:
            entry.clear()

    def simulate(self):
        t_monotonic_start_unix = time.time()
        if self.config['observer']['enable'] is True:
            max_step = np.min([self.satellite.controller.t_sample, self.satellite.observer_module.t_sample])
        else:
            max_step = self.satellite.controller.t_sample
        
        self.sim_time_series = np.arange(0, self.sim_time, max_step)
        sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="LSODA",
                        t_eval=self.sim_time_series,
                        max_step=max_step)
        
        # Integrate satellite dynamics over time
        t_monotonic_end_unix = time.time()

        print(f"Simulation took {t_monotonic_end_unix - t_monotonic_start_unix} seconds")
        return sol
    
    def simulate_monte_carlo(self):
        if self.config['observer']['enable'] is True:
            max_step = np.min([self.satellite.controller.t_sample, self.satellite.observer_module.t_sample])
        else:
            max_step = self.satellite.controller.t_sample
        self.sim_time_series = np.arange(0, self.sim_time, max_step)
        try:
            sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                            t_eval=self.sim_time_series,
                            max_step=max_step)
        except DivergentRate:
            # print("divergent rate hit")
            return -1
        
        if sol.status != 0:
            print(f"Warning: Simulation did not complete successfully, status: {sol.status}")
            
        return sol

    def collect_results(self, sol, use_only_sol=False):
        
        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'w_sat_{axis}'] = interpolate_data(sol.y[i], sol.t, self.sim_time_series)

        for i,axis in enumerate(my_utils.q_axes):
            self.results_data[f'q_sat_{axis}'] = np.clip(interpolate_data(sol.y[i+3], sol.t, self.sim_time_series), -1, 1)
            
        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'control_energy_{axis}'] = interpolate_data(sol.y[i+7], sol.t, self.sim_time_series)
        
        # Put results into data object
        for key, value in self.results_data.items():
            if key == 'time':
                continue
            if len(value) == len(self.results_data['time']):
                self.results_data[key] = interpolate_data(value, self.results_data['time'], self.sim_time_series)[:]
            
        for key, value in self.results_data.items():
            if len(value) != len(self.sim_time_series):
                print(f"Warning: {key} has length {len(value)} but time has length {len(self.sim_time_series)}")
                # self.results_data[key] = interpolate_data(value, self.results_data['time'], self.sim_time_series)[:]
            # else:
                # raise(Exception(f"Warning: {key} has length {len(value)} but time has length {len(self.results_data['time'])}"))
        
        self.results_data['time'] = self.sim_time_series
        
        self.results_df = pd.DataFrame.from_dict(self.results_data)

        self.results_df['euler_axis_sat'] = self.results_df['q_sat_w'].apply(lambda w: 2*np.arccos(w))
        self.results_df['euler_axis_sat_deg'] = self.results_df['euler_axis_sat']*180/np.pi

        self.results_df['euler_int'] = cumulative_trapezoid(self.results_df['euler_axis_sat'], self.results_df['time'], initial=0)
        
        # Quaternion to Principal Angle Error
        q_sat = np.array([self.results_data["q_sat_x"], 
                        self.results_data["q_sat_y"], 
                        self.results_data["q_sat_z"], 
                        self.results_data["q_sat_w"]])
        r_sat =  Rotation.from_quat(quat=q_sat.T)
        [self.results_df['e321_sat_yaw'],self.results_df['e321_sat_pitch'], self.results_df['e321_sat_roll']] = r_sat.as_euler('zyx', degrees=True).T

        q_sat_ref = np.array([self.results_df["q_sat_ref_x"],
                              self.results_df["q_sat_ref_y"],
                              self.results_df["q_sat_ref_z"],
                              self.results_df["q_sat_ref_w"]])
        r_sat_ref = Rotation.from_quat(q_sat_ref.T)
        # self.results_df['euler_axis_sat_ref'] = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_sat_ref)

        r_sat_error = r_sat_ref * r_sat.inv()

        [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = r_sat_error.as_quat().T
        # [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = self.results_df.apply(lambda row: my_utils.get_quaternion_error_Nadafi(row, ), axis=1).T
        self.results_df['euler_axis_sat_error'] = self.results_df['q_sat_error_w'].apply(lambda w: 2*np.arccos(w))
        self.results_df['euler_axis_sat_error_deg'] = self.results_df['euler_axis_sat_error'] * 180 / np.pi
        print(f"use_only_sol: {use_only_sol}")
        if use_only_sol == False:
            for i, wheel in enumerate(self.satellite.wheel_module.wheels):
                print(f"calculating T_wheels_est_{i}")
                self.results_data[f'T_wheels_est_{str(i)}'] = self.results_data['dw_wheels_est_' + str(i)]*wheel.M_inertia_fast
                # self.results_df['T_wheels_est'] = self.results_df['dw_wheels_est_' + str(i)]*wheel.M_inertia_fast
            for i, wheel in enumerate(self.satellite.wheel_module.wheels):
                self.results_df[f'f_wheels_error_{i}'] = self.results_df[f'f_wheels_{i}'] - self.results_df[f'f_wheels_est_{i}']

        for i,axis in enumerate(my_utils.xyz_axes):
            if self.config['output']['energy_enable']:
                self.calc_control_energy_output_results(self.results_data['control_energy_{}'.format(axis)])
        
        if self.config['output']['accuracy_enable']:
            self.calc_accuracy_output_results()    
    
    control_energy_log_output = ""
    def calc_control_energy_output_results(self, control_energy_arr):
        control_energy_per_axis = {}
        control_energy_total = 0
        for i,axis in enumerate(my_utils.xyz_axes):
            control_energy_per_axis[axis] = np.sum(np.abs(control_energy_arr[i]))
            control_energy_total += control_energy_per_axis[axis]
        self.control_energy_log_output = f"control energy (J): {my_utils.round_dict_values(control_energy_per_axis,3)} | total: {round(control_energy_total,3)}"
        # if self.monte_carlo == False:
        #     print(self.control_energy_log_output)

    accuracy_percent = None
    settling_time = None
    steady_state = None
    steady_state_euler_axis = None

    def calc_accuracy_output_results(self):
        try:
            control_info = control.step_info(sysdata=self.results_df[f"euler_axis_sat_deg"], 
                                            SettlingTimeThreshold=0.002, T=self.results_data['time'])
            self.steady_state = control_info['SteadyStateValue']

            q_final = [self.results_data[f'q_sat_{axis}'][-1] for axis in my_utils.q_axes]
            q_error = self.satellite.q_ref*np.quaternion(q_final[3], q_final[0], q_final[1], q_final[2]).inverse()
            prin_error = my_utils.get_principal_angle_from_np_quaternion(q_error)
            self.settling_time = control_info['SettlingTime']
            
            final_euler  = Rotation.from_quat(q_final).as_euler('xyz', degrees=True)
            euler_error = Rotation.from_quat([q_error.x, q_error.y, q_error.z, q_error.w]).as_euler('xyz', degrees=True)

            print(f"final euler: {final_euler} deg xyz")
            print(f"euler error: {euler_error} deg xyz")
            print(f"principal angle error: {prin_error*180/np.pi} deg")
            print(f"steady state value: {self.steady_state} deg")

            if self.config['satellite']['use_ref_series'] is True:
                return

            if self.settling_time >= (self.sim_time - 1):
                self.settling_time = None
                raise Exception("Settling time is greater than simulation time")

            
            if self.monte_carlo == False:
                print(f"settling_time (s): {round(self.settling_time,3)}")
                print(f"steady_state (s): {round(self.steady_state,3)}")


            # control_info_y = control.step_info(sysdata=self.results_df['e_sat_yaw'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            # control_info_p = control.step_info(sysdata=self.results_df['e_sat_pitch'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            # control_info_r = control.step_info(sysdata=self.results_df['e_sat_roll'],SettlingTimeThreshold=0.002, T=self.results_data['time'])

            # print(f"steady state value: {control_info_y['SteadyStateValue']} {control_info_p['SteadyStateValue']} {control_info_r['SteadyStateValue']} deg zyx")
        except Exception as e:
            print(f"Error calculating accuracy: {e}")

    def log_output_to_file(self, LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en):
        if LOG_FILE_NAME != None and test_mode_en is False:
            LOG_FILE_NAME_RESULTS = LOG_FILE_NAME + "_results"
            clear_log_file(fr"{LOG_FOLDER_PATH}/{LOG_FILE_NAME_RESULTS}")
            # output_dict_to_csv(LOG_FOLDER_PATH, LOG_FILE_NAME + "_log", self.results_data)
            with open(fr'{LOG_FOLDER_PATH}/{LOG_FILE_NAME + "_log"}.csv', 'w+') as file:
                self.results_df.to_csv(file,sep=',')
            output_toml_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME + "_config", self.config)
            if self.accuracy_percent is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"accuracy %: {self.accuracy_percent}", False)
            if self.settling_time is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"settling_time (s): {round(self.settling_time,3)}",    False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, self.control_energy_log_output, False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"{self.steady_state}", False)

    def create_plots_separated(self, rows, results_data, config, LOG_FILE_NAME, file_name_append = ""):
        # Create separate figures if enabled in config
        names = []
        for row in rows:
            row_name, axes, label = row
            fig_separate = plt.figure(figsize=(12,6))
            ax_separate = fig_separate.add_subplot(111)
            
            for axis in axes:
                if axis != 'none':
                    name = row_name + "_" + axis
                else:
                    axis = None
                    name = row_name
                try:
                    ax_separate.plot(results_data['time'], results_data[name], label=name)
                    names.append(name)
                except Exception as e:
                    print(f"Error plotting {name}: {e}")

            
            ax_separate.set_xlabel('time (s)')
            ax_separate.set_ylabel(label)
            if ax_separate.get_legend_handles_labels()[0] != []:
                ax_separate.legend(loc='upper right')
            
            if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None:
                if not os.path.exists(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs")):
                    os.mkdir(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs"))
                fig_separate.savefig(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_{row_name}{file_name_append}.png"), bbox_inches='tight')
            
            if config['output']['separate_plots_display'] is False:
                plt.close(fig_separate)
    
    def create_plots_comparison(self, rows : list, label : str, graph_name: str, results_data : dict, config : dict, LOG_FILE_NAME : str , show : bool = False):
        fig = plt.figure(figsize=(12,6))
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
                    ax.plot(results_data['time'], results_data[name], label=name, linestyle=['-','--',':'][row_idx%3], color=['r','g','b','y','m','gray','k'][axis_idx%7])
                    
                except Exception as e:
                    print(f"Error plotting {name}: {e}")
        ax.legend(loc='upper right')
        if show is True or config['output']['show_plots'] is True:
            try:
                plt.show()
            except Exception as e:
                print(f"Error showing plots: {e}")


        ax.set_xlabel('time (s)')
        ax.set_ylabel(label)

        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
            fig.savefig(os.path.abspath(f"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_{graph_name}.png"), bbox_inches='tight')

    def create_plots_combined(self, rows, cols, results_data, config, LOG_FILE_NAME, type='line', x_axis=None):
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
                        current_plot.plot(results_data['time'], results_data[name], label=axis)
                    elif type == 'scatter':
                        if x_axis is None:
                            raise Exception("x_axis must be provided for scatter plot")
                        current_plot.scatter(x_axis, results_data[name], label=axis)
                except Exception as e:
                    print(f"Error plotting {name}: {e}")

            current_plot.set_xlabel('time (s)')
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
            fig.savefig(os.path.abspath(f"../data_logs/{LOG_FILE_NAME}/{LOG_FILE_NAME}_summary.png"), bbox_inches='tight')

    def create_3D_quaternion_plot(self, results_data, config, LOG_FILE_NAME):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        try:
            ax.plot(results_data['q_sat_x'], results_data['q_sat_y'], results_data['q_sat_z'], label='Satellite Quaternion Trajectory')
            ax.set_xlabel('q_x')
            ax.set_ylabel('q_y')
            ax.set_zlabel('q_z')
            ax.legend()
        except Exception as e:
            print(f"Error plotting 3D quaternion trajectory: {e}")
        # plt.show()
        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:
            fig.savefig(os.path.abspath(f"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_quaternion_3D_trajectory.png"), bbox_inches='tight')

def generate_rand_rot():
    """Generate a 3D random rotation matrix.

    Returns:
        np.matrix: A 3D rotation matrix.

    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def generate_rand_quat() -> np.quaternion:
    """Generate a random quaternion.

    Returns:
        np.quaternion: A random quaternion.

    """
    x1, x2, x3 = np.random.rand(3)
    q = np.quaternion(np.sqrt(1 - x3) * np.cos(2 * np.pi * x1),
                    np.sqrt(1 - x3) * np.sin(2 * np.pi * x1),
                    np.sqrt(x3) * np.cos(2 * np.pi * x2),
                    np.sqrt(x3) * np.sin(2 * np.pi * x2))
    q = q.normalized()
    return q

def calc_cost(gains_list, kwargs):
    return np.array([kwargs['func'](gains_list[i, :], i, kwargs['config']) for i in range(gains_list.shape[0])])

def Nadafi_BS_controller_param_optimize(gains, particle_num, config):
    # Calculate Cost
    results_data_local = {}
    config['simulation']['duration'] = config['tuning']['duration']
    sim_obj = Simulation(config, results_data = results_data_local, logging_en=False)
    
    [sim_obj.satellite.controller.nafadi_controller.Gamma_z11,
        sim_obj.satellite.controller.nafadi_controller.Gamma_z22,
        sim_obj.satellite.controller.nafadi_controller.lambda_1,
        sim_obj.satellite.controller.nafadi_controller.lambda_2,
        sim_obj.satellite.controller.nafadi_controller.lambda_3] = gains
    sol = sim_obj.simulate_monte_carlo()
    if sol == -1:
        cost = 1e9
    else:
        SSE = sum(map(lambda w: (2*np.arccos(np.clip(w, -1, 1)))**2, sol.y[6])) # Sum of squared principal angle errors
        #control_energy = sum(map(lambda i: np.sum(np.abs(sol.y[i+7])), range(3))) # Total control energy
    
        # energy_cost = 0.001*control_energy
        # steady_state_cost = 2*np.arccos(sol.y[6][-1])*1e3
        # print(f"SSE Cost: {SSE} | Gains: {gains}, Particle: {particle_num}")
        # print(f"Energy Cost: {energy_cost}")
        # print(f"Steady State Cost: {steady_state_cost}")
        cost = SSE #+ 0.001*control_energy + 2*np.arccos(sol.y[6][-1])*1e4 #+ 1e6*(sol.y[0][-1]**2 + sol.y[1][-1]**2 + sol.y[2][-1]**2)  # Weighted cost function combining accuracy and control energy
    del sim_obj
    del results_data_local
    return cost

# simulation.results_data = {}
def Nadafi_BS_FNDO_controller_param_optimize(gains, particle_num, config):
    # Calculate Cost
    results_data_local = {}
    sim_obj = Simulation(config, results_data = results_data_local, logging_en=False)
    
    [sim_obj.satellite.controller.nafadi_controller.Gamma_z11,
        sim_obj.satellite.controller.nafadi_controller.Gamma_z22,
        sim_obj.satellite.controller.nafadi_controller.lambda_1,
        sim_obj.satellite.controller.nafadi_controller.lambda_2,
        sim_obj.satellite.controller.nafadi_controller.lambda_3,
        sim_obj.satellite.controller.nafadi_controller.L11,
        sim_obj.satellite.controller.nafadi_controller.L22,
        sim_obj.satellite.controller.nafadi_controller.kappa_0,
        sim_obj.satellite.controller.nafadi_controller.kappa_1] = gains
    sol = sim_obj.simulate_monte_carlo()
    cost = 0
    if sol == -1:
        cost = 1e9
    else:
        SSE = sum(map(lambda w: (2*np.arccos(np.clip(w, -1, 1)))**2, sol.y[6])) # Sum of squared principal angle errors
        cost = SSE
    # print(f"Cost: {SSE} | Gains: {gains}, Particle: {particle_num}")
    del sim_obj
    del results_data_local
    return cost


def main():
    LOG_FILE_NAME, LOG_FOLDER_PATH, config, args = parse_args()
    if os.path.exists(LOG_FOLDER_PATH) is False:
        os.mkdir(LOG_FOLDER_PATH)
    results_data = {}
    test_mode_en = config['simulation']['test_mode_en']

    sim_iter = config['simulation']['iterations']
    
    if config['simulation']['enable']:
        # Run Controller Tuning Setup 
        if config['simulation']['tuning']:
            # Set-up hyperparameters
            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, }
            
            # Apply function to each particle (column)
            n_particles = 8
            
            if config['controller']['sub_type'] == "Nadafi_BS":
                initial_guess = np.array([config['Nadafi']['Gamma_z11'],
                                config['Nadafi']['Gamma_z22'], 
                                config['Nadafi']['lambda_1'], 
                                config['Nadafi']['lambda_2'], 
                                config['Nadafi']['lambda_3'], 
                                ])
            elif config['controller']['sub_type'] == "Nadafi_FNDO":
                initial_guess = np.array([config['Nadafi']['Gamma_z11'],
                                config['Nadafi']['Gamma_z22'],
                                config['Nadafi']['lambda_1'], 
                                config['Nadafi']['lambda_2'], 
                                config['Nadafi']['lambda_3'],
                                config['Nadafi']['L11'],
                                config['Nadafi']['L22'], 
                                config['Nadafi']['kappa_0'],
                                config['Nadafi']['kappa_1'], 
                                ])

            # bounds = (np.zeros(len(initial_guess)), np.ones(len(initial_guess))*100)
            bounds = None
            if config['controller']['sub_type'] == "Nadafi_FNDO" or config['controller']['sub_type'] == "Nadafi_MFNDO":
                min_bound = -1*np.ones(len(initial_guess))*100
                max_bound = np.ones(len(initial_guess))*100
                min_bound[5] = 0
                min_bound[6] = 0
                bounds = (min_bound, max_bound)

            initial_guesses = np.row_stack([initial_guess * (1 + 0.5 * np.random.randn(len(initial_guess))) for _ in range(n_particles)])
            if bounds is not None:
                for i in range(len(initial_guesses)):
                    for j in range(len(initial_guess)):
                        initial_guesses[i][j] = max(initial_guesses[i][j], bounds[0][j]+1e-3) # Ensure initial guess is within bounds
            initial_guesses[0] = initial_guess # Set the first particle to the initial guess

            optimizer = pyswarms.single.GlobalBestPSO(n_particles=n_particles, dimensions=initial_guesses.shape[1], options=options, init_pos=initial_guesses, bounds=bounds)
            if config['controller']['sub_type'] == "Nadafi_BS":
                func = Nadafi_BS_controller_param_optimize
            if config['controller']['sub_type'] == "Nadafi_FNDO":
                func = Nadafi_BS_FNDO_controller_param_optimize
            
            cost, gains = optimizer.optimize(calc_cost, iters=config['tuning']['iterations'], kwargs={'config': config, 'func': func}, n_processes=min(os.cpu_count(), n_particles))
            print("Finished tuning step")
            print(f"Final Gains: {gains}")
            print(f"Final Cost: {cost}")

            # if config['controller']['sub_type'] == "Nadafi_BS":
            #     config['Nadafi']['Gamma_z11'] = gains[0]
            #     config['Nadafi']['Gamma_z22'] = gains[1]
            #     config['Nadafi']['L11'] = gains[2]
            #     config['Nadafi']['L22'] = gains[3]
            #     config['Nadafi']['lambda_1'] = gains[4]
            #     config['Nadafi']['lambda_2'] = gains[5] 
            #     config['Nadafi']['lambda_3'] = gains[6]
            # elif config['controller']['sub_type'] == "Nadafi_FNDO":
            #     config['Nadafi']['Gamma_z11'] = gains[0]
            #     config['Nadafi']['Gamma_z22'] = gains[1]
            #     config['Nadafi']['L11'] = gains[2]
            #     config['Nadafi']['L22'] = gains[3]
            #     config['Nadafi']['kappa_0'] = gains[4]
            #     config['Nadafi']['kappa_1'] = gains[5] 
            #     config['Nadafi']['lambda_1'] = gains[6]
            #     config['Nadafi']['lambda_2'] = gains[7]
            #     config['Nadafi']['lambda_3'] = gains[8]
            # else:
            #     raise Exception(f"Controller sub type {config['controller']['sub_type']} not recognized for tuning")

            config['simulation']['verbose'] = True
            simulation = Simulation(config, results_data)

            if config['controller']['sub_type'] == "Nadafi_BS":
                simulation.satellite.controller.nafadi_controller.Gamma_z11 = gains[0]
                simulation.satellite.controller.nafadi_controller.Gamma_z22 = gains[1]
                simulation.satellite.controller.nafadi_controller.lambda_1 = gains[2]
                simulation.satellite.controller.nafadi_controller.lambda_2 = gains[3] 
                simulation.satellite.controller.nafadi_controller.lambda_3 = gains[4]
            elif config['controller']['sub_type'] == "Nadafi_FNDO":
                simulation.satellite.controller.nafadi_controller.Gamma_z11 = gains[0]
                simulation.satellite.controller.nafadi_controller.Gamma_z22 = gains[1]
                simulation.satellite.controller.nafadi_controller.lambda_1 = gains[2]
                simulation.satellite.controller.nafadi_controller.lambda_2 = gains[3]
                simulation.satellite.controller.nafadi_controller.lambda_3 = gains[4]
                simulation.satellite.controller.nafadi_controller.L11 = gains[5]
                simulation.satellite.controller.nafadi_controller.L22 = gains[6]
                simulation.satellite.controller.nafadi_controller.kappa_0 = gains[7]
                simulation.satellite.controller.nafadi_controller.kappa_1 = gains[8] 
            else:
                raise Exception(f"Controller sub type {config['controller']['sub_type']} not recognized for tuning")
            simulation.logging_en = True
            # gains = np.array(gains)
            print(f"Gamma_z11 = {simulation.satellite.controller.nafadi_controller.Gamma_z11}")
            print(f"Gamma_z22 = {simulation.satellite.controller.nafadi_controller.Gamma_z22}")
            print(f"lambda_1 = {simulation.satellite.controller.nafadi_controller.lambda_1}")
            print(f"lambda_2 = {simulation.satellite.controller.nafadi_controller.lambda_2}")
            print(f"lambda_3 = {simulation.satellite.controller.nafadi_controller.lambda_3}")
            print(f"L11 = {simulation.satellite.controller.nafadi_controller.L11}")
            print(f"L22 = {simulation.satellite.controller.nafadi_controller.L22}")
            print(f"kappa_0 = {simulation.satellite.controller.nafadi_controller.kappa_0}")
            print(f"kappa_1 = {simulation.satellite.controller.nafadi_controller.kappa_1}")

            with open(fr'{LOG_FOLDER_PATH}/{LOG_FILE_NAME}_tuning_log' + ".txt", 'w+') as file:
                file.write(f"Final Gains:\n")
                file.write(f"Gamma_z11 = {simulation.satellite.controller.nafadi_controller.Gamma_z11}\n")
                file.write(f"Gamma_z22 = {simulation.satellite.controller.nafadi_controller.Gamma_z22}\n")
                file.write(f"lambda_1 = {simulation.satellite.controller.nafadi_controller.lambda_1}\n")
                file.write(f"lambda_2 = {simulation.satellite.controller.nafadi_controller.lambda_2}\n")
                file.write(f"lambda_3 = {simulation.satellite.controller.nafadi_controller.lambda_3}\n")
                file.write(f"L11 = {simulation.satellite.controller.nafadi_controller.L11}\n")
                file.write(f"L22 = {simulation.satellite.controller.nafadi_controller.L22}\n")
                file.write(f"kappa_0 = {simulation.satellite.controller.nafadi_controller.kappa_0}\n")
                file.write(f"kappa_1 = {simulation.satellite.controller.nafadi_controller.kappa_1}\n")
                # file.write(f"gamma_mu: {simulation.satellite.controller.nafadi_controller.gamma_mu}\n")
                file.write(f"Final Cost: {cost}\n")

            print(f"Running simulation with tuned gains: {gains}")
            simulation.sim_time = 300
            sol = simulation.simulate()
            print("Simulation Complete")

            simulation.collect_results(sol)
            rows = [('q_sat',my_utils.q_axes, 'Quaternion'), ('w_sat',my_utils.xyz_axes, 'Angular velocity (rad/s)' )]
            print("Creating plots of tuned parameters simulation...")
            simulation.create_plots_separated(rows, simulation.results_df, config, LOG_FILE_NAME)

        else:
            #-------------------------------------------------------------#
            ###################### Simulate System ########################
            if sim_iter == 1:
                
                simulation = Simulation(config, results_data)
                satellite = simulation.satellite
                wheel_module = satellite.wheel_module
                controller = satellite.controller
                print(f"Running Once-off simulation")
                sol = simulation.simulate()

                print("Simulation Complete")
                
                if (test_mode_en):
                    exit(0)
                simulation.collect_results(sol)

                simulation.log_output_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en)

                ## Row should be in the form of (row_name, [axes], label)
                cols = 2
                rows = [ ('w_sat',my_utils.xyz_axes, 'Angular velocity (rad/s)'), 
                        ('q_sat',my_utils.q_axes, 'Quaternion'), 
                        ('e321_sat', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), \
                        ('euler_axis_sat_deg', ['none'], 'Euler Angle about Principal Axis (deg)'),
                        ('T_sat',my_utils.xyz_axes, 'Torque (N)'), 
                        ('control_energy',my_utils.xyz_axes, 'Control Energy (J)'), 
                        ('T_dist', my_utils.xyz_axes, 'Torque Disturbance (N)')
                        ]
                

                rows_2 = []
                if satellite.wheels_control_enable:
                    _axes = [str(wheel.index) for wheel in wheel_module.wheels]
                    rows_2.append(('T_wheels', _axes, 'Wheel Torque (Nm)'))
                    rows_2.append(('w_wheels', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel speed (rad/s)'))
                    rows_2.append(('E', _axes, 'Actuator Authority Estimate (Fraction)'))
                    rows_2.append(('f_wheels', _axes, 'Disturbance Torque (Nm)'))
                    rows_2.append(('u_a', _axes, 'Additive Fault (Nm)'))

                    rows_2.append(('w_wheels_est', _axes, 'Estimated Wheel Speed (rad/s)'))
                    rows_2.append(('T_wheels_est', _axes, 'Estimated Wheel Torque (rad/s^2)'))
                    rows_2.append(('f_wheels_est', _axes, 'Estimated Disturbance Torque (Nm)'))
                    rows_2.append(('f_wheels_error', _axes, 'Disturbance Torque Error (Nm)'))
                    rows_2.append(('E_est', _axes, 'Actuator Authority Estimate (Fraction)'))
                    
                if controller.type == "adaptive":
                    rows.append(('control_adaptive_model_output',['none']))
                    rows.append(('control_theta',my_utils.xyz_axes))
                
                rows_2.append(('q_sat_ref', my_utils.q_axes, 'Reference Quaternion'))
                # rows_2.append(('q_sat_vec_ref', ['x', 'y', 'z'], 'Reference Quaternion Vector'))
                rows_2.append(('q_sat_error', my_utils.q_axes, 'Quaternion Error (satellite to reference)'))
                rows_2.append(('euler_axis_sat_error_deg', ['none'], 'Error Euler Angle about Principal Axis (deg)'))
                rows_2.append(('T_magt', my_utils.xyz_axes, 'Magnetorquer Torque (Nm)'))
                # rows_2.append(('euler_axis_sat_ref', ['none'], 'Reference Euler Angle about Principal Axis (deg)'))

                rows_2.append(('s_sat_eci', my_utils.xyz_axes, 'Satellite Position ECI (km)'))
                rows_2.append(('v_sat_eci', my_utils.xyz_axes, 'Satellite Velocity ECI (km/s)'))
                rows_2.append(('n_sun', my_utils.xyz_axes, 'Sun Vector (unitless)'))
                
                simulation.create_plots_separated(rows, simulation.results_df, config, LOG_FILE_NAME)
                simulation.create_plots_combined(rows, cols, simulation.results_df, config, LOG_FILE_NAME)

                simulation.create_plots_separated(rows_2, simulation.results_df, config, LOG_FILE_NAME)

                simulation.create_3D_quaternion_plot(simulation.results_df, config, LOG_FILE_NAME)

                if satellite.wheels_control_enable:
                    simulation.create_plots_comparison([('w_wheels', _axes),
                                        ('w_wheels_est', _axes)
                                        ], 'Wheel speed (rad/s)', 'wheels_speed_meas_vs_est', simulation.results_df, config, LOG_FILE_NAME, show=False)
                    simulation.create_plots_comparison([('T_wheels', _axes),
                                        ('T_wheels_est', _axes)
                                        ], 'Wheel torque (Nm)', 'wheels_torque_meas_vs_est', simulation.results_df, config, LOG_FILE_NAME, show=False)
                    simulation.create_plots_comparison([('E', _axes), ('E_est', _axes)
                                        ], 'Wheel effectiveness (Fraction)', 'wheels_authority_meas_vs_est', simulation.results_df, config, LOG_FILE_NAME, show=False)

                    simulation.create_plots_comparison([('q_sat', my_utils.q_axes),('q_sat_ref', my_utils.q_axes)], 'Quaternion', 'q_sat_vs_ref', simulation.results_df, config, LOG_FILE_NAME, show=False)
                    
                    simulation.create_plots_comparison([('q_sat', ['x', 'y', 'z']),('q_sat_ref', ['x', 'y', 'z'])], 'Quaternion', 'q_sat_vs_ref_vec', simulation.results_df, config, LOG_FILE_NAME, show=False)
            
                if config['output']['visualizer']['enable'] is True:
                    json_data, results_df = viz.convert_results_df_to_json(simulation.results_df, config['output']['visualizer']['t_sample'])
                    open(config['output']['visualizer']['file_path'], "w").write(json_data)
            #-------------------------------------------------------------#
            ###################### Monte Carlo ############################
            elif sim_iter > 1:
                if config['output']['visualizer']['enable'] is True:
                    print("Visualize is enabled, but Monte Carlo simulation is running. Visualization will be disabled for Monte Carlo simulation.")
                simulation.monte_carlo = True
                print(f"Running Monte Carlo simulation with {sim_iter} iterations")
                def test_q_init(q_init, results=None):

                    results['accuracy'] = []
                    results['settling_time'] = []
                    results['euler_axis_final'] = []
                    results['euler_axis_init'] = []
                    results['euler_angles_y_init'] = []
                    results['euler_angles_p_init'] = []
                    results['euler_angles_r_init'] = []
                    
                    for i in range(sim_iter):
                        print(f"Simulation Iteration {i+1} of {sim_iter}")
                        simulation.satellite.dir_init = Rotation.from_quat(q_init.as_float_array(), scalar_first=True)
                        sol = simulation.simulate()
                        simulation.collect_results(sol)
                        results['accuracy'].append(simulation.accuracy)
                        results['settling_time'].append(simulation.settling_time)
                        # results['euler_axis_final'].append(results_data['euler_axis'][-1])
                        results['euler_axis_final'].append(simulation.steady_state_euler_axis)
                        

                    rows = [('accuracy', 'none', 'Accuracy'), ('settling_time', 'none', 'Settling Time')]
                    simulation.create_plots_combined(rows, cols, results_data, config, LOG_FILE_NAME, type='scatter', x_axis=[monte_carlo_results['euler_axis_final']])
                    # simulation.log_output_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en)
                    if simulation.accuracy > 0:
                        passed = True
                    else:
                        passed = False
                    return passed

                def plot_q(passed):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('Random Quaternion')
                    # for i in range(sim_iter):
                    q_init = generate_rand_quat()
                    q_init = np.quaternion(q_init.w,q_init.x,q_init.y,q_init.z)
                    # test_q_init(q_init)
                    if passed:
                        colour = 'g'
                    else:
                        colour = 'r'
                    ax.scatter(q_init.x, q_init.y, q_init.z, label=f"Iteration {i+1}", marker='o', facecolors='none', edgecolors=colour)
                
                    # Make data
                    # u = np.linspace(0, 2 * np.pi, 100)
                    # v = np.linspace(0, np.pi, 100)
                    # x = np.outer(np.cos(u), np.sin(v))
                    # y = np.outer(np.sin(u), np.sin(v))
                    # z = np.outer(np.ones(np.size(u)), np.cos(v))
                    # ax.plot_surface(x, y, z, color='r', alpha=0.1)
                    if config['output']['show_plots'] is True:
                        try:
                            plt.show()
                        except Exception as e:
                            print(f"Error showing plots: {e}")

                monte_carlo_results = dict()
                
                for i in range(sim_iter):
                    q_init = generate_rand_quat()
                    print(q_init)
                    # quats.append(q_init)
                    passed = test_q_init(q_init, results=monte_carlo_results)

            else:
                raise Exception("Invalid simulation iteration count")

if __name__ == '__main__':

    cProfile.run('main()', os.path.abspath('../data_logs/profile_stats.prof'))



