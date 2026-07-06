import matplotlib.pyplot as plt
import numpy as np
import os
import my_utils
import toml
import pandas as pd
from satellite import Satellite

class Logger:

    log_file_name = ""
    log_folder_path = ""
    results_data : dict = None
    satellite : Satellite = None
    enable = True
    next_timestamp = 0.0
    initialized = False
    verbose = False
    config = None

    def init(self, config, results_data : dict, satellite, enable, logger_fields : list):
        self.initialized = True
        self.results_data = results_data
        self.satellite = satellite
        self.enable = enable
        self.verbose = config['simulation']['verbose']
        self.config = config
        
        self.results_data["time"] = []
        self.results_data["jd"] = []

        for axis in my_utils.xyz_axes:
            self.results_data["T_sat_" + axis] = []
            self.results_data["w_sat_" + axis] = []
            self.results_data['T_dist_' + axis] = []
            self.results_data['H_total_' + axis] = []
        for axis in my_utils.q_axes:
            self.results_data["q_sat_" + axis] = []

        for i in my_utils.q_axes:
            self.results_data[f'q_sat_ref_{i}'] = []

        for i in range(self.satellite.wheel_module.num_wheels):
            self.results_data[f'T_ctr_wheels_{i}'] = []

            self.results_data[f'w_wheels_{i}'] = []
            self.results_data[f'T_wheels_{i}'] = []
            self.results_data[f'E_{i}'] = []
            self.results_data[f'f_wheels_{i}'] = []
            self.results_data[f'u_a_{i}'] = []

            self.results_data[f'w_wheels_est_{i}'] = []
            self.results_data[f'f_wheels_est_{i}'] = []
            self.results_data[f'dw_wheels_est_{i}'] = []
            self.results_data[f'E_est_{i}'] = []
        
        for i in my_utils.xyz_axes:
            self.results_data[f'T_magt_{i}'] = []
            self.results_data[f's_sat_eci_{i}'] = []
            self.results_data[f'v_sat_eci_{i}'] = []
            self.results_data[f'n_sun_{i}'] = []
            self.results_data[f'w_sat_ref_{i}'] = []
            self.results_data[f'dw_sat_ref_{i}'] = []
        
        if self.config['controller']['type'] == 'backstepping':
            if self.config['controller']['sub_type'].startswith('Nadafi'):
                for axis in my_utils.xyz_axes:
                    self.results_data[f'Z_{axis}'] = []
                    self.results_data[f'F_{axis}'] = []
                    self.results_data[f'term_1_{axis}'] = []
                    self.results_data[f'term_2_{axis}'] = []
                self.results_data['Z_norm'] = []
            # if self.config['controller']['sub_type'].endswith('FNDO'):
                for axis in my_utils.xyz_axes:
                    self.results_data[f'chi_0_{axis}'] = []
                    self.results_data[f'chi_1_{axis}'] = []
                    self.results_data[f'v_0_{axis}'] = []
                # if self.config['controller']['sub_type'] == 'Nadafi_MFNDO':
                for axis in my_utils.xyz_axes:
                    self.results_data[f'mu_{axis}'] = []
            if self.config['controller']['sub_type'] == 'Zarourati':
                self.results_data['xi'] = []
                for axis in my_utils.xyz_axes:
                    self.results_data[f'kappa1_{axis}'] = []
                    self.results_data[f'kappa2_{axis}'] = []
                    self.results_data[f'eta_{axis}'] = []
                    self.results_data[f'eta_norm'] = []
                    self.results_data[f'dwe_u'] = []
                    self.results_data[f'we_u'] = []
                    self.results_data[f'']
                
                        
        self.next_timestamp = self.satellite.controller.t_sample

    def log_data(self, t):
        if self.enable and self.initialized:
            if t >= self.next_timestamp:
                # print("logging data")
                self.results_data['time'].append(t)
                self.results_data['jd'].append(self.satellite.orbit.jd + self.satellite.orbit.fr)

                # Collect results data for logging                    
                for i, axis in enumerate(my_utils.xyz_axes):
                    self.results_data['T_sat_'+ axis].append(self.satellite.T_ctr_vec[i])
                    self.results_data['T_dist_' + axis].append(self.satellite.T_dist[i])
                    self.results_data['T_magt_' + axis].append(self.satellite.magt_module.T[i])
                    self.results_data['s_sat_eci_' + axis].append(self.satellite.orbit.sBI_I[i])
                    self.results_data['v_sat_eci_' + axis].append(self.satellite.orbit.DIsBI_I[i])
                    self.results_data['n_sun_' + axis].append(self.satellite.orbit.nSB_I[i])
                    self.results_data['H_total_' + axis].append((self.satellite.H_total)[i])
                    self.results_data['w_sat_ref_' + axis].append(self.satellite.w_ref[i])
                    self.results_data['dw_sat_ref_' + axis].append(self.satellite.dw_ref[i])
                
                q_ref = [self.satellite.q_ref.x, self.satellite.q_ref.y, self.satellite.q_ref.z, self.satellite.q_ref.w]
                for i, axis in enumerate(my_utils.q_axes):
                    self.results_data['q_sat_ref_' + axis].append(q_ref[i])

                for i, wheel in enumerate(self.satellite.wheel_module.wheels):
                    self.results_data['w_wheels_' + str(i)].append(self.satellite.wheel_module.w_wheels[i])
                    self.results_data['w_wheels_est_' + str(i)].append(self.satellite.observer_module.w_wheels_est[i])
                    self.results_data['f_wheels_est_' + str(i)].append(self.satellite.observer_module.f_wheels_est[i])
                    self.results_data['T_wheels_' + str(i)].append(self.satellite.wheel_module.dw_wheels[i]*wheel.M_inertia_fast) # assuming wheel inertia is diagonal and only z component is used
                    self.results_data['dw_wheels_est_' + str(i)].append(self.satellite.observer_module.dw_wheels_est[i])
                    self.results_data['T_ctr_wheels_' + str(i)].append(self.satellite.T_ctr_wheels[i])
                    self.results_data['E_' + str(i)].append(self.satellite.fault_module.E[i][i])
                    # self.results_data['E_est_' + str(i)].append(self.observer_module.E_mul[i][i])
                    self.results_data['E_est_' + str(i)].append(self.satellite.E[i][i])
                    self.results_data['f_wheels_' + str(i)].append(self.satellite.f_wheels[i])
                    self.results_data['u_a_' + str(i)].append(self.satellite.fault_module.u_a[i])
                
                if self.config['controller']['type'] == 'backstepping':
                    if self.config['controller']['sub_type'].startswith('Nadafi'): 
                        self.results_data['Z_norm'].append(np.linalg.norm(self.satellite.controller.nadafi_controller.Z))
                        for i, axis in enumerate(my_utils.xyz_axes):
                            if i == self.satellite.controller.nadafi_controller.f_idx:
                                self.results_data['Z_' + axis].append(np.nan) # log NaN for the faulty axis
                                self.results_data['F_' + axis].append(np.nan)
                                self.results_data['term_1_' + axis].append(np.nan)
                                self.results_data['term_2_' + axis].append(np.nan)
                            else:
                                self.results_data['Z_' + axis].append(self.satellite.controller.nadafi_controller.Z[i,0])
                                self.results_data['F_' + axis].append(self.satellite.controller.nadafi_controller.F[i,0])
                                self.results_data['term_1_' + axis].append(self.satellite.controller.nadafi_controller.term_1[i,0])
                                self.results_data['term_2_' + axis].append(self.satellite.controller.nadafi_controller.term_2[i,0])
                        # if self.config['controller']['sub_type'].endswith('FNDO'):
                        for i, axis in enumerate(my_utils.xyz_axes):
                            if i == self.satellite.controller.nadafi_controller.f_idx:
                                self.results_data['chi_0_' + axis].append(np.nan)
                                self.results_data['chi_1_' + axis].append(np.nan)
                                self.results_data['v_0_' + axis].append(np.nan)
                            else:
                                self.results_data['chi_0_' + axis].append(self.satellite.controller.nadafi_controller.chi_0[i,0])
                                self.results_data['chi_1_' + axis].append(self.satellite.controller.nadafi_controller.chi_1[i,0])
                                self.results_data['v_0_' + axis].append(self.satellite.controller.nadafi_controller.v_0[i,0])
                        # if self.config['controller']['sub_type'] == 'Nadafi_MFNDO':
                        for i, axis in enumerate(my_utils.xyz_axes):
                            if i == self.satellite.controller.nadafi_controller.f_idx:
                                self.results_data['mu_' + axis].append(np.nan)
                            else:
                                self.results_data['mu_' + axis].append(self.satellite.controller.nadafi_controller.mu[i,0])


                # self.results_data['control_adaptive_model_output'] = self.satellite.controller.control_adaptive_model_output
                # for i, axis in enumerate(my_utils.xyz_axes):
                #     self.results_data[f'control_theta_{axis}'].append(self.satellite.controller.theta[i])
                self.next_timestamp += self.satellite.controller.t_sample
    
    def log(self, message):
        if self.config['simulation']['tuning']:
            return            
        if self.verbose:
            print(f"{message}")
