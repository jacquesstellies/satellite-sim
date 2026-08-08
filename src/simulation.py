from observer import WheelExtendedStateObserver, ObserverModule
from satellite import Satellite, DivergentRate
from controller import Controller
from fault import Fault, FaultModule
from wheels import WheelModule
from magt import MagtModule
from orbit import Orbit
from logger import Logger
import my_utils

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time
import os
import control
import toml


def output_toml_to_file(path, file_name, data):
    with open(fr'{path}/{file_name}.toml', 'w+') as file:
        toml.dump(data, file)

class Simulation:
    satellite : Satellite = None
    sim_time_series = None
    sim_time = 0
    config = None
    monte_carlo = False
    results_data = None
    results_df : pd.DataFrame = None
    iter = 0

    def __init__(self, config, results_data, log_file_name, log_folder_path, logging_en=True):
        self.config = config
        self.results_data = results_data

        #------------------------------------------------------------#
        ###################### Set Up Objects ########################
        self.logger = Logger(config, log_file_name, log_folder_path)
        self.fault_module = FaultModule(config)
        wheel_module = WheelModule(config, self.fault_module.faults)
        if config['satellite']['euler_init_en']:
            dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
        else:
            dir_init = Rotation.from_quat(config['satellite']['q_init'])

        w_sat_init = np.array(config['satellite']['w_init_dps']) * my_utils.DEG_TO_RAD
        controller = Controller(faults=self.fault_module.faults, wheel_module=wheel_module, results_data=results_data, w_sat_init=np.zeros(3), q_sat_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init),
                                    config=config)
        observer_module = ObserverModule(config, wheel_module)
        orbit = Orbit(config, logger=self.logger)
        magt_module = MagtModule(config, orbit)
        self.satellite = Satellite(wheel_module, controller, observer_module, self.fault_module, magt_module, self.logger, orbit=orbit, config=config)

        controller.init_satellite(self.satellite)
        #------------------------------------------------------------#
        ###################### Set Up Initial Conditions ########################
        self.fault_module.init(wheel_module.num_wheels)
        self.logger.post_init(results_data, self.satellite, enable = config['output']['log_enable'] and logging_en, logger_fields=None)

        # Satellite Initial Conditions
        self.satellite.dir_init = dir_init
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
        # first_step pins scipy's automatic initial-step heuristic, which otherwise proposes
        # a trial evaluation up to t_bound away when the initial state rates are ~0 (e.g. the
        # satellite starts at rest exactly on the reference). That stray far-future call to
        # calc_state_rates permanently advances Satellite.next_t_ref_update past the whole
        # sim, freezing the reference generator for the rest of the run.
        sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                        t_eval=self.sim_time_series,
                        max_step=max_step, first_step=max_step)

        # Integrate satellite dynamics over time
        t_monotonic_end_unix = time.time()

        self.logger.log(f"Simulation took {t_monotonic_end_unix - t_monotonic_start_unix} seconds")
        return sol
    
    def simulate_monte_carlo(self):
        if self.config['observer']['enable'] is True:
            max_step = np.min([self.satellite.controller.t_sample, self.satellite.observer_module.t_sample])
        else:
            max_step = self.satellite.controller.t_sample
        self.sim_time_series = np.arange(0, self.sim_time, max_step)
        try:
            # see simulate() re: first_step
            sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                            t_eval=self.sim_time_series,
                            max_step=max_step, first_step=max_step)
        except DivergentRate:
            # print("divergent rate hit")
            return -1
        
        if sol.status != 0:
            print(f"Warning: Simulation did not complete successfully, status: {sol.status}")
            
        return sol

    def collect_results(self, sol, use_only_sol=False):
        
        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'w_sat_{axis}'] = np.interp(self.sim_time_series, sol.t, sol.y[i])

        for i,axis in enumerate(my_utils.q_axes):
            self.results_data[f'q_sat_{axis}'] = np.clip(np.interp(self.sim_time_series, sol.t, sol.y[i+3]), -1, 1)
            
        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'control_energy_{axis}'] = np.interp(self.sim_time_series, sol.t, sol.y[i+7])
        
        # Put results into data object
        for key, value in self.results_data.items():
            if key == 'time':
                continue
            if len(value) == len(self.results_data['time']):
                self.results_data[key] = np.interp(self.sim_time_series, self.results_data['time'], value)[:]
            
        for key, value in self.results_data.items():
            if len(value) != len(self.sim_time_series):
                self.logger.log(f"Warning: {key} has length {len(value)} but time has length {len(self.sim_time_series)}")
        
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
        # q_sat columns hold the passive q_BI. scipy from_quat treats a quaternion as an
        # ACTIVE rotation, so .inv() recovers the passive attitude matrix A(q_BI)=T_BI
        # (v_B = T_BI v_I); its 3-2-1 Euler angles are the body attitude.
        r_sat =  Rotation.from_quat(quat=q_sat.T).inv()
        [self.results_df['e321_sat_yaw'],self.results_df['e321_sat_pitch'], self.results_df['e321_sat_roll']] = r_sat.as_euler('zyx', degrees=True).T

        q_sat_ref = np.array([self.results_df["q_sat_ref_x"],
                              self.results_df["q_sat_ref_y"],
                              self.results_df["q_sat_ref_z"],
                              self.results_df["q_sat_ref_w"]])
        r_sat_ref = Rotation.from_quat(q_sat_ref.T).inv()  # passive: matrix T_RI
        # self.results_df['euler_axis_sat_ref'] = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_sat_ref)

        # Body-frame error (matrix T_BR); as_quat matches the controller's passive q_RB
        # = my_utils.quat_error(q_RI, q_BI) exactly (verified).
        r_sat_error = r_sat * r_sat_ref.inv()

        [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = r_sat_error.as_quat().T
        # [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = self.results_df.apply(lambda row: my_utils.get_quaternion_error_Nadafi(row, ), axis=1).T
        self.results_df['euler_axis_sat_error'] = self.results_df['q_sat_error_w'].apply(lambda w: 2*np.arccos(w))
        self.results_df['euler_axis_sat_error_deg'] = self.results_df['euler_axis_sat_error'] * 180 / np.pi
        # self.logger.log(f"use_only_sol: {use_only_sol}")
        
        for axis in my_utils.xyz_axes:
            self.results_df[f'w_sat_error_{axis}'] = self.results_df[f'w_sat_{axis}'] - self.results_df[f'w_sat_ref_{axis}']

        if use_only_sol == False:
            for i, wheel in enumerate(self.satellite.wheel_module.wheels):
                self.results_data[f'T_wheels_est_{str(i)}'] = self.results_data['dw_wheels_est_' + str(i)]*wheel.M_inertia_fast
                # self.results_df['T_wheels_est'] = self.results_df['dw_wheels_est_' + str(i)]*wheel.M_inertia_fast
            for i, wheel in enumerate(self.satellite.wheel_module.wheels):
                self.results_df[f'f_wheels_error_{i}'] = self.results_df[f'f_wheels_{i}'] - self.results_df[f'f_wheels_est_{i}']

        if self.config['output']['energy_enable']:
            self.calc_control_energy_output_results()  
    
    def calc_control_energy_output_results(self):
        control_energy_per_axis = {}
        self.control_energy_total = 0
        for i,axis in enumerate(my_utils.xyz_axes):
            control_energy_per_axis[axis] = np.sum(np.abs(self.results_df[f'control_energy_{axis}']))
            self.control_energy_total += control_energy_per_axis[axis]

    settling_time = None
    steady_state = None
    steady_state_euler_axis = None
    prin_error = None
    final_euler = None
    euler_error = None
    control_energy_total = None

    def calc_accuracy_output_results(self):
        try:
            control_info = control.step_info(sysdata=self.results_df[f"euler_axis_sat_deg"], 
                                            SettlingTimeThreshold=0.002, T=self.results_data['time'])
            self.steady_state = control_info['SteadyStateValue']

            q_final = [self.results_data[f'q_sat_{axis}'][-1] for axis in my_utils.q_axes]
            q_BI_final = np.quaternion(q_final[3], q_final[0], q_final[1], q_final[2])
            q_error = my_utils.quat_error(self.satellite.q_RI, q_BI_final)
            self.prin_error = my_utils.get_principal_angle_from_np_quaternion(q_error)
            self.settling_time = control_info['SettlingTime']
            
            self.final_euler  = Rotation.from_quat(q_final).as_euler('xyz', degrees=True)
            self.euler_error = Rotation.from_quat([q_error.x, q_error.y, q_error.z, q_error.w]).as_euler('xyz', degrees=True)

            if self.config['satellite']['use_ref_series'] is True:
                return

            if self.settling_time >= (self.sim_time - 1):
                self.settling_time = None
                raise Exception("Settling time is greater than simulation time")

            self.logger.log(f"settling_time (s): {round(self.settling_time,3)}", to_results_file=True, to_console=True)
            self.logger.log(f"steady_state (s): {round(self.steady_state,3)}", to_results_file=True, to_console=True)
            
        except Exception as e:
            self.logger.log(f"Error calculating accuracy: {e}")

    def log_data_to_file(self, LOG_FILE_NAME, LOG_FOLDER_PATH):
        self.logger.log(f"control energy (J): {round(self.control_energy_total,3)}")
        self.logger.log(f"final euler: {self.final_euler} deg xyz", to_results_file=True, to_console=True)
        self.logger.log(f"euler error: {self.euler_error} deg xyz", to_results_file=True, to_console=True)
        self.logger.log(f"principal angle error: {self.prin_error*180/np.pi} deg", to_results_file=True, to_console=True)
        self.logger.log(f"steady state value: {self.steady_state} deg", to_results_file=True, to_console=True)
        with open(fr'{LOG_FOLDER_PATH}/{LOG_FILE_NAME + "_log"}.csv', 'w+') as file:
            self.results_df.to_csv(file,sep=',')
        output_toml_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME + "_config", self.config)