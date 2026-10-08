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
from types import SimpleNamespace
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

    def _base_tick(self):
        """Fastest enabled discrete rate; every enabled block's t_sample must be an
        integer multiple of it so all gates fire exactly on tick times."""
        sat = self.satellite
        rates = {'controller': sat.controller.t_sample}
        if self.config['observer']['enable']:
            rates['observer'] = sat.observer_module.t_sample
        if sat.fault_detector is not None:
            rates['detection'] = sat.fault_detector.t_sample
        if sat.orbit.enable:
            rates['orbit'] = sat.orbit.t_sample
        if sat.magt_module.enable and sat.magt_module.physical and sat.magt_module.t_sample > 0.0:
            rates['magt'] = sat.magt_module.t_sample
        Ts = min(rates.values())
        for name, rate in rates.items():
            ratio = rate / Ts
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(f"{name}.t_sample = {rate} is not an integer multiple of the "
                                 f"base tick {Ts} (sampled integrator needs commensurate rates)")
        return Ts

    def _simulate_sampled(self):
        """Sampled-data loop: discrete blocks run once per tick on the accepted state,
        then the continuous plant is integrated over the tick with the inputs held
        (zero-order hold) using fixed-step RK4."""
        Ts = self._base_tick()
        substeps = int(self.config['simulation'].get('integrator_substeps', 1))
        h = Ts / substeps
        # tanh Coulomb friction is a stiff mode near w = 0 with rate T_c/(I_w*w_s); keep
        # rate*h well inside RK4's stability region or zero-speed crossings are inaccurate
        for wheel in self.satellite.wheel_module.wheels:
            if wheel.coulomb_torque != 0.0:
                coulomb_gain = wheel.coulomb_torque / (wheel.M_inertia_fast * wheel.coulomb_smoothing_speed) * h
                if coulomb_gain > 0.5:
                    w_s_min = 2.0 * wheel.coulomb_torque * h / wheel.M_inertia_fast
                    print(f"WARNING: wheel {wheel.index} Coulomb friction step gain {coulomb_gain:.2f} > 0.5, "
                          f"increase coulomb_smoothing_speed to >= {w_s_min:.3g} rad/s or integrator_substeps")
        n_ticks = int(round(self.sim_time / Ts))
        rates = self.satellite.plant_rates

        x = np.array(self.initial_values, dtype=float)
        t_out = np.arange(n_ticks + 1) * Ts
        y_out = np.empty((len(x), n_ticks + 1))
        y_out[:, 0] = x
        for k in range(n_ticks):
            t = k * Ts
            self.satellite.discrete_update(t, x)
            for j in range(substeps):
                tj = t + j * h
                k1 = rates(tj, x)
                k2 = rates(tj + h/2, x + h/2 * k1)
                k3 = rates(tj + h/2, x + h/2 * k2)
                k4 = rates(tj + h, x + h * k3)
                x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            x[3:7] /= np.linalg.norm(x[3:7])
            y_out[:, k + 1] = x
        return SimpleNamespace(t=t_out, y=y_out, status=0, success=True)

    def _integrate(self, max_step):
        if self.config['simulation'].get('integrator', 'rk45') == 'sampled':
            return self._simulate_sampled()
        # first_step pins scipy's automatic initial-step heuristic, which otherwise proposes
        # a trial evaluation up to t_bound away when the initial state rates are ~0 (e.g. the
        # satellite starts at rest exactly on the reference). That stray far-future call to
        # calc_state_rates permanently advances Satellite.next_t_ref_update past the whole
        # sim, freezing the reference generator for the rest of the run.
        return solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                         t_eval=self.sim_time_series,
                         max_step=max_step, first_step=max_step)

    def simulate(self):
        t_monotonic_start_unix = time.time()
        if self.config['observer']['enable'] is True:
            max_step = np.min([self.satellite.controller.t_sample, self.satellite.observer_module.t_sample])
        else:
            max_step = self.satellite.controller.t_sample

        self.sim_time_series = np.arange(0, self.sim_time, max_step)
        sol = self._integrate(max_step)

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
            sol = self._integrate(max_step)
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
            if len(value) != len(self.sim_time_series) and key != 'time':
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
        r_sat_active =  Rotation.from_quat(quat=q_sat.T)
        [self.results_df['e321_sat_yaw'],self.results_df['e321_sat_pitch'], self.results_df['e321_sat_roll']] = r_sat_active.as_euler('ZYX', degrees=True).T

        q_sat_ref = np.array([self.results_df["q_sat_ref_x"],
                              self.results_df["q_sat_ref_y"],
                              self.results_df["q_sat_ref_z"],
                              self.results_df["q_sat_ref_w"]])
        r_sat_ref = Rotation.from_quat(q_sat_ref.T).inv()  # passive: matrix T_RI
        # self.results_df['euler_axis_sat_ref'] = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_sat_ref)

        # Body-frame error (matrix T_BR); as_quat matches the controller's passive q_RB
        # = my_utils.quat_error(q_RI, q_BI) exactly (verified).
        r_sat =  Rotation.from_quat(quat=q_sat.T).inv()
        r_sat_error = r_sat * r_sat_ref.inv()

        [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = r_sat_error.as_quat().T
        # [self.results_df['q_sat_error_x'], self.results_df['q_sat_error_y'], self.results_df['q_sat_error_z'], self.results_df['q_sat_error_w']] = self.results_df.apply(lambda row: my_utils.get_quaternion_error_Nadafi(row, ), axis=1).T
        self.results_df['euler_axis_sat_error'] = self.results_df['q_sat_error_w'].apply(lambda w: 2*np.arccos(w))
        self.results_df['euler_axis_sat_error_deg'] = self.results_df['euler_axis_sat_error'] * 180 / np.pi
        # self.logger.log(f"use_only_sol: {use_only_sol}")
        
        for axis in my_utils.xyz_axes:
            self.results_df[f'w_sat_error_{axis}'] = self.results_df[f'w_sat_{axis}'] - self.results_df[f'w_sat_ref_{axis}']

        # Boresight (body +z) vs nadir - the pointing metric nominal_night is actually
        # trying to null, independent of yaw about the boresight (which q_sat_error and
        # euler_axis_sat_error_deg both fold in). r_sat is the passive matrix T_BI, so
        # applying it to the inertial nadir vector gives nadir in body coords; the angle
        # it makes with body +z is the boresight pointing error.
        n_nadir_I = self.results_df[[f'n_nadir_{axis}' for axis in my_utils.xyz_axes]].to_numpy()
        n_nadir_B = r_sat.apply(n_nadir_I)
        self.results_df['boresight_nadir_error_deg'] = np.degrees(
            np.arccos(np.clip(n_nadir_B[:, 2], -1.0, 1.0)))

        # if use_only_sol == False:
        for i, wheel in enumerate(self.satellite.wheel_module.wheels):
            self.results_data[f'H_wheels_{str(i)}'] = self.results_data['w_wheels_' + str(i)]*wheel.M_inertia_fast
            
            self.results_data[f'T_wheels_est_{str(i)}'] = self.results_data['dw_wheels_est_' + str(i)]*wheel.M_inertia_fast
            self.results_df[f'f_wheels_error_{str(i)}'] = self.results_df[f'f_wheels_{str(i)}'] - self.results_df[f'f_wheels_est_{str(i)}']

        self.results_df['H_norm'] = np.sqrt(self.results_df['H_total_x']**2 + self.results_df['H_total_y']**2 + self.results_df['H_total_z']**2)

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