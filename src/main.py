from observer import WheelExtendedStateObserver
from satellite import Satellite
from controller import Controller
from fault import Fault
from wheels import WheelModule

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
import control
from inspect import currentframe, getframeinfo
import cProfile
import pyswarms
import threading

DEBUG = True

# all units are in SI (m, s, N, kg.. etc)

# results_data = my_globals.results_data

    
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
        
def interpolate_data(data, time_series, time_series_new):
    return np.interp(time_series_new, time_series, data)

def create_default_log_file_name(config):
    
    filename = config['controller']['type']
    
    if(config['controller']['type'] != "q_feedback"):
        filename += ('_' + config['controller']['sub_type'])
    
    filename += '_' + config['wheels']['config']
    
    if config['fault']['master_enable']:
        filename += '_fault'
    
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
    args = vars(parser.parse_args())

    with open('config.toml', 'r') as config_file:
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

    if args["test_mode"] is False:
        config['simulation']['test_mode_en'] = False
    else:
        config['simulation']['test_mode_en'] = True
    
    if args["disable_sim"] is True:
        config['simulation']['enable'] = False

    if append_date is True and LOG_FILE_NAME is not None:
        dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        LOG_FILE_NAME += f"_{dt_string}"
    if args['append'] is not None:
        LOG_FILE_NAME += f"_{args['append']}"

    LOG_FOLDER_BASE_PATH = os.path.abspath(r'../data_logs')
    LOG_FOLDER_PATH = os.path.join(LOG_FOLDER_BASE_PATH,LOG_FILE_NAME)
    if not os.path.exists(LOG_FOLDER_BASE_PATH) and LOG_FILE_NAME != None:
        raise Exception(f"Log folder {LOG_FOLDER_BASE_PATH} does not exist")
    
    print(f"output name is {LOG_FILE_NAME}")
    print(f"output folder is {LOG_FOLDER_PATH}")
    return LOG_FILE_NAME, LOG_FOLDER_PATH, config

def clear_log_file(log_file_path):
    with open(log_file_path, 'w') as file:
        file.write("")

class Simulation:
    satellite : Satellite = None
    sim_time_series = None
    sim_time = 0
    config = None
    monte_carlo = False
    results_data = None
    iter = 0

    def __init__(self, config, results_data, logging_en=True):
        #------------------------------------------------------------#
        ###################### Set Up Objects ########################
        self.config = config
        self.results_data = results_data

        fault = Fault(config)

        wheel_module = WheelModule(config, fault)
        if config['satellite']['euler_init_en']:
            dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
        else:
            dir_init = Rotation.from_quat(config['satellite']['q_init'])

        for i in range(wheel_module.num_wheels):
            self.results_data['wheel_speed_' + str(i)] = []
            self.results_data['wheel_torque_' + str(i)] = []

        w_sat_init = np.array([0,0,0])

        controller = Controller(fault=fault, wheel_module=wheel_module, results_data=results_data, w_sat_init=np.zeros(3), q_sat_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init),
                                    config=config)

        self.satellite = Satellite(wheel_module, controller, fault, config=config, results_data=results_data, logging_en=logging_en)

        # Satellite Initial Conditions
        self.satellite.dir_init = dir_init

        # Adaptive Controller Initialize
        self.satellite.controller.M_inertia_inv_model = self.satellite.M_inertia_inv
        self.satellite.controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(self.satellite.dir_init)

        # Control Variables
        if config['satellite']['use_ref_euler']:
            self.satellite.ref_q = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True)
        elif config['satellite']['use_ref_q']:
            self.satellite.ref_q = Rotation.from_quat(config['satellite']['ref_q'])
        else:
            raise(Exception("no reference angle commanded"))

        q_sat_init = self.satellite.dir_init.as_quat()
        control_torque_init = np.zeros(3)
        w_wheels_init = np.zeros((self.satellite.wheel_module.num_wheels))
        self.initial_values = np.concatenate([w_sat_init, q_sat_init, w_wheels_init, control_torque_init])

        # Simulation parameters
        sim_config = config['simulation']
        self.sim_time = sim_config['duration'] if not config['simulation']['test_mode_en'] else sim_config['test_duration']
        self.sim_output_resolution_time = sim_config['resolution'] if not config['simulation']['test_mode_en'] else sim_config['test_resolution']
        
        self.sim_time_series = np.linspace(0, self.sim_time, int(self.sim_time/self.sim_output_resolution_time))

        
        if config['simulation']['test_mode_en']:
            print("NB ********* Test Mode is ENABLED *********")
    
    def clear_results_data(self):
        for entry in self.results_data:
            entry.clear()

    def simulate(self):
        sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                        t_eval=self.sim_time_series,
                        max_step=self.satellite.controller.t_sample)
        # Integrate satellite dynamics over time
        return sol
    def simulate_multithread(self):
        self.results_data = None
        print(f"Starting sim {self.iter}")
        sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                        t_eval=self.sim_time_series,
                        max_step=self.satellite.controller.t_sample)
        self.results_data = sol

    def collect_results(self, sol):

        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'angular_rate_{axis}'] = interpolate_data(sol.y[i], sol.t, self.sim_time_series)

        for i,axis in enumerate(my_utils.q_axes):
            self.results_data[f'quaternion_{axis}'] = interpolate_data(sol.y[i+3], sol.t, self.sim_time_series)
        
        for i,axis in enumerate(my_utils.xyz_axes):
            self.results_data[f'control_energy_{axis}'] = interpolate_data(sol.y[i+7], sol.t, self.sim_time_series)

        # Put results into data object
        for key, value in self.results_data.items():
            if key == 'time':
                continue
            if len(value) > int(self.sim_time/self.sim_output_resolution_time):
                if key[:7] == "control":
                    self.results_data[key] = interpolate_data(value, self.results_data['control_time'], self.sim_time_series)[:]
                else:
                    try:
                        self.results_data[key] = interpolate_data(value, self.results_data['time'], self.sim_time_series)[:]
                    except:
                        print(f"Error interpolating {key}")
        
        self.results_data['time'] = self.sim_time_series
        self.results_data['control_time'] = self.sim_time_series
        
        # print(f"quaternion_x size {len(self.results_data['quaternion_x'])}")
        # print(f"time size {len(self.results_data['time'])}")
        self.results_data['euler_axis'] = []
        for row in range(len(self.results_data['time'])):
            r_1 : Rotation = Rotation.from_quat([self.results_data['quaternion_x'][row],
                                            self.results_data['quaternion_y'][row],
                                            self.results_data['quaternion_z'][row],
                                            self.results_data['quaternion_w'][row]])
            alpha = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_1)
            self.results_data['euler_axis'].append(alpha)
        
        self.results_data['euler_int'] = cumulative_trapezoid(self.results_data['euler_axis'], self.results_data['time'], initial=0)

        q = np.array([self.results_data["quaternion_x"], 
                                            self.results_data["quaternion_y"], 
                                            self.results_data["quaternion_z"], 
                                            self.results_data["quaternion_w"]])
        r =  Rotation.from_quat(quat=q.T)
        [self.results_data['euler_yaw'],self.results_data['euler_pitch'], self.results_data['euler_roll']] = r.as_euler('zyx', degrees=True).T
        
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
            euler_axis_init = my_utils.conv_Rotation_obj_to_euler_axis_angle(self.satellite.dir_init)
            euler_axis_ref = my_utils.conv_Rotation_obj_to_euler_axis_angle(self.satellite.ref_q)
            control_info = control.step_info(sysdata=self.results_data[f"euler_int"], 
                                            SettlingTimeThreshold=0.002, T=self.results_data['time'])
            self.steady_state = control_info['SteadyStateValue']

            control_info_euler_axis = control.step_info(sysdata=self.results_data['euler_axis'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            
            self.steady_state_euler_axis = control_info_euler_axis['SteadyStateValue']
            print(f"steady state error euler angle about pricipal axis: {self.steady_state_euler_axis} deg")
            self.accuracy = 1-np.abs((self.results_data[f"euler_axis"][-1]-euler_axis_ref)/(euler_axis_ref-euler_axis_init))
            self.settling_time = control_info['SettlingTime']
            
            if self.settling_time >= (self.sim_time - 1):
                self.accuracy = 0
                self.settling_time = None
                raise Exception("Settling time is greater than simulation time")

            self.accuracy_percent = self.accuracy*100
            
            final_q = [self.results_data[f'quaternion_{axis}'][-1] for axis in my_utils.q_axes]
            final_euler  = Rotation.from_quat(final_q).as_euler('xyz', degrees=True)
            if self.monte_carlo == False:
                print(f"accuracy %: {self.accuracy_percent}")
                print(f"settling_time (s): {round(self.settling_time,3)}")
                print(f"steady_state (s): {round(self.steady_state,3)}")
            print(f"final euler: {final_euler} deg xyz")
            print(f"euler error: {euler_axis_ref - final_euler} deg xyz")

            control_info_y = control.step_info(sysdata=self.results_data['euler_yaw'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            control_info_p = control.step_info(sysdata=self.results_data['euler_pitch'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            control_info_r = control.step_info(sysdata=self.results_data['euler_roll'],SettlingTimeThreshold=0.002, T=self.results_data['time'])
            if self.monte_carlo == False:
                print(f"accuracy %: {self.accuracy_percent}")
                print(f"settling_time (s): {round(self.settling_time,3)}")
                print(f"steady_state (s): {round(self.steady_state,3)}")
            print(f"steady state error: {control_info_y['SteadyStateValue']} {control_info_p['SteadyStateValue']} {control_info_r['SteadyStateValue']} deg zyx")
        except Exception as e:
            print(f"Error calculating accuracy: {e}")

    def log_output_to_file(self, LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en):
        if os.path.exists(LOG_FOLDER_PATH) is False:
            os.mkdir(LOG_FOLDER_PATH)
        if LOG_FILE_NAME != None and test_mode_en is False:
            LOG_FILE_NAME_RESULTS = LOG_FILE_NAME + "_results"
            # clear_log_file(fr"{LOG_FOLDER_PATH}\{LOG_FILE_NAME_RESULTS}")
            output_dict_to_csv(LOG_FOLDER_PATH, LOG_FILE_NAME + "_log", self.results_data)
            output_toml_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME + "_config", self.config)
            if self.accuracy_percent is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"accuracy %: {self.accuracy_percent}", False)
            if self.settling_time is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"settling_time (s): {round(self.settling_time,3)}",    False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, self.control_energy_log_output, False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"{self.steady_state}", False)

    def create_plots_separated(self, rows, results_data, config, LOG_FILE_NAME):
            # Create separate figures if enabled in config
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
                    ax_separate.plot(results_data['time'], results_data[name], label=axis)
                except Exception as e:
                    print(f"Error plotting {name}: {e}")

            
            ax_separate.set_xlabel('time (s)')
            ax_separate.set_ylabel(label)
            if ax_separate.get_legend_handles_labels()[0] != []:
                ax_separate.legend()
            
            if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None:
                if not os.path.exists(fr"..\data_logs\{LOG_FILE_NAME}\graphs"):
                    os.mkdir(fr"..\data_logs\{LOG_FILE_NAME}\graphs")
                fig_separate.savefig(fr"..\data_logs\{LOG_FILE_NAME}\graphs\{LOG_FILE_NAME}_{row_name}.pdf", bbox_inches='tight')
            if config['output']['separate_plots_display'] is False:
                plt.close(fig_separate)
    
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

            fig.savefig(fr"..\data_logs\{LOG_FILE_NAME}\{LOG_FILE_NAME}_summary.pdf", bbox_inches='tight')

# class MonteCarloSimulation(Simulation):
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

def main():
    LOG_FILE_NAME, LOG_FOLDER_PATH, config = parse_args()
    results_data = {}
    results_data["time"] = []
    for axis in my_utils.xyz_axes:
        results_data["torque_" + axis] = []
        results_data["angular_acc_" + axis] = []
        results_data["angular_rate_" + axis] = []
        results_data['T_dist_' + axis] = []
    for axis in my_utils.q_axes:
        results_data["quaternion_" + axis] = []
    
    simulation = Simulation(config, results_data)
    satellite = simulation.satellite
    wheel_module = satellite.wheel_module
    controller = satellite.controller
    test_mode_en = config['simulation']['test_mode_en']

    sim_iter = config['simulation']['iterations']
    
    if config['simulation']['enable']:
        # Run Controller Tuning Setup 
        if config['simulation']['tuning']:
            # from concurrent.futures import ThreadPoolExecutor

            simulation.satellite.logging_en = False
            # simulation.results_data = {}
            def backstepping_controller_param_optimize(gains, simulation):
                # print(f"running simulation #{simulation.iter}", end='\r', flush=True)
                # results_data['SSE'] = []
                # Set up controller gains
                threads = []
                sims = []
                # with ThreadPoolExecutor(max_workers=10) as executor:
                #     for particle in range(len(gains)):
                #         [simulation.satellite.controller.alpha,
                #         simulation.satellite.controller.beta,
                #         simulation.satellite.controller.k,
                #         simulation.satellite.controller.eta_1,
                #         simulation.satellite.controller.upsilon,
                #         simulation.satellite.controller.c_1,
                #         simulation.satellite.controller.c_2] = gains[particle]

                #     
                #     future_to_result = {executor.submit(sim_obj.simulate)}
                #     for future in concurrent.futures.as_completed(future_to_url):
                #         try:
                #             data = future.result()

                for particle in range(len(gains)):
                        [simulation.satellite.controller.alpha,
                        simulation.satellite.controller.beta,
                        simulation.satellite.controller.k,
                        simulation.satellite.controller.eta_1,
                        simulation.satellite.controller.upsilon,
                        simulation.satellite.controller.c_1,
                        simulation.satellite.controller.c_2] = gains[particle]

                        sim_obj = Simulation(config, simulation.results_data, logging_en=False)
                        thread = threading.Thread(target=sim_obj.simulate)
                        thread.daemon = True
                        thread.start()
                        threads.append(thread)
                        sims.append(sim_obj)

                for thread in threads:
                    thread.join()
                # Calculate Cost
                costs = np.zeros(gains.shape[0])
                for i, sim in enumerate(sims):
                    SSE = 0 # Sum of squared errors
                    for i, axis in enumerate(my_utils.q_axes):
                        SSE += np.linalg.norm(sim.results_data.y[i+3])**2
                    costs[i] = SSE
                
                simulation.iter += 1


                # results_data['SSE'].append(simulation.SSE)
                return SSE
            # Set-up hyperparameters
            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
            
            # Create base parameter array
            base_params = np.array([config['backstepping']['alpha'],
                                   config['backstepping']['beta'], 
                                   config['backstepping']['k'],
                                   config['backstepping']['eta_1'],
                                   config['backstepping']['upsilon'],
                                   config['backstepping']['c_1'],
                                   config['backstepping']['c_2']])
            
            # Apply function to each particle (column)
            n_particles = 10
            initial_guesses = np.column_stack([base_params * (1 + 0.1 * np.random.randn(len(base_params))) for _ in range(n_particles)])
            
            # print(f"Initial guesses: {initial_guesses}")
            optimizer = pyswarms.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(base_params), options=options, init_pos=initial_guesses.T)
            cost, gains = optimizer.optimize(backstepping_controller_param_optimize, iters=2, simulation=simulation)
            print("Finished tuning step")
            print(f"Final Gains: {gains}")
            simulation.logging_en = True
            simulation.simulate()
            print("Simulation Complete")
            cols = 2
            ## Row should be in the form of (row_name, [axes], label)
            rows = [ ('angular_rate',my_utils.xyz_axes, 'Angular velocity (rad/s)'), ('quaternion',my_utils.q_axes, 'Quaternion'), ('euler_int',['none'], 'Euler integral (deg)'), ('euler', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), \
                    ('torque',my_utils.xyz_axes, 'Torque (N)'), ('control_energy',my_utils.xyz_axes, 'Control Energy (J)'), ('T_dist', my_utils.xyz_axes, 'Torque Disturbance (N)')]
                       
            if satellite.wheels_control_enable:
                # rows.append(('wheel_speed', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel speed (rad/s)'))
                rows.append(('wheel_torque', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel Torque (N)'))

            simulation.create_plots_separated(rows, results_data, config, LOG_FILE_NAME)
            simulation.create_plots_combined(rows, cols, results_data, config, LOG_FILE_NAME)

        else:
            #-------------------------------------------------------------#
            ###################### Simulate System ########################
            if sim_iter == 1:
                print(f"Running Once-off simulation")
                sol = simulation.simulate()
                print("Simulation Complete")
                simulation.collect_results(sol)

                # simulation.log_output_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en)

                cols = 2
                rows = [ ('angular_rate',my_utils.xyz_axes, 'Angular velocity (rad/s)'), ('quaternion',my_utils.q_axes, 'Quaternion'), ('euler_int',['none'], 'Euler integral (deg)'), ('euler', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), \
                        ('torque',my_utils.xyz_axes, 'Torque (N)'), ('control_energy',my_utils.xyz_axes, 'Control Energy (J)'), ('T_dist', my_utils.xyz_axes, 'Torque Disturbance (N)')]
                
                ## Row should be in the form of (row_name, [axes], label)
                # rows = [ ('angular_rate',my_utils.xyz_axes, 'Angular rate (rad/s)'), 
                #         ('euler', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), 
                #         ('euler_int',['none'], 'Euler integral (deg)'), 
                #         ]
                
                if satellite.wheels_control_enable:
                    # rows.append(('wheel_speed', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel speed (rad/s)'))
                    rows.append(('wheel_torque', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel Torque (N)'))

                if controller.type == "adaptive":
                    rows.append(('control_adaptive_model_output',['none']))
                    rows.append(('control_theta',my_utils.xyz_axes))
                
                simulation.create_plots_separated(rows, results_data, config, LOG_FILE_NAME)
                simulation.create_plots_combined(rows, cols, results_data, config, LOG_FILE_NAME)

            #-------------------------------------------------------------#
            ###################### Monte Carlo ############################
            elif sim_iter > 1:
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
                # plot_q()
                # exit()
                # quats = []
                monte_carlo_results = dict()
                
                for i in range(sim_iter):
                    q_init = generate_rand_quat()
                    print(q_init)
                    # quats.append(q_init)
                    passed = test_q_init(q_init, results=monte_carlo_results)

            else:
                raise Exception("Invalid simulation iteration count")

if __name__ == '__main__':

    cProfile.run('main()', '../data_logs/profile_stats.prof')



