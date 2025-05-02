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

DEBUG = True

# all units are in SI (m, s, N, kg.. etc)

results_data = my_globals.results_data

    
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
                LOG_FILE_NAME = config['controller']['type']
                if(config['controller']['type'] != "q_feedback"):
                    LOG_FILE_NAME += ('_' + config['controller']['sub_type'])
                LOG_FILE_NAME += '_' + config['wheels']['config']
                if config['fault']['master_enable']:
                    LOG_FILE_NAME += '_fault'
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

    LOG_FOLDER_BASE_PATH = fr'..\data_logs'
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

    def __init__(self, config):
            #------------------------------------------------------------#
        ###################### Set Up Objects ########################
        self.config = config
        fault = Fault(config)

        wheel_module = WheelModule(config, fault)
        if config['satellite']['euler_init_en']:
            dir_init = Rotation.from_euler('xyz',config['satellite']['euler_init'],degrees=True)
        else:
            dir_init = Rotation.from_quat(config['satellite']['q_init'])
        satellite_angular_v_init = np.array([0,0,0])


        controller = Controller(fault=fault,angular_v_init=np.zeros(3),quaternion_init=my_utils.conv_Rotation_obj_to_numpy_q(dir_init),
                                    config=config)

        self.satellite = Satellite(wheel_module, controller, fault, config=config)

        # Satellite Initial Conditions
        self.satellite.dir_init = dir_init

        # Adaptive Controller Initialize
        self.satellite._controller.M_inertia_inv_model = self.satellite.M_inertia_inv
        self.satellite._controller.q_prev = my_utils.conv_Rotation_obj_to_numpy_q(self.satellite.dir_init)

        # Control Variables
        if config['satellite']['use_ref_euler']:
            self.satellite.ref_q = Rotation.from_euler("xyz", config['satellite']['ref_euler'], degrees=True)
        elif config['satellite']['use_ref_q']:
            self.satellite.ref_q = Rotation.from_quat(config['satellite']['ref_q'])
        else:
            raise(Exception("no reference angle commanded"))

        quaternion_init = self.satellite.dir_init.as_quat()
        control_torque_init = [0, 0, 0]
        self.initial_values = [satellite_angular_v_init, quaternion_init, control_torque_init]
        # if self.satellite.wheels_control_enable is True:
        #     self.initial_values.append(np.zeros(3))

        self.initial_values = np.hstack(self.initial_values)

        # Simulation parameters
        sim_config = config['simulation']
        self.sim_time = sim_config['duration'] if not config['simulation']['test_mode_en'] else sim_config['test_duration']
        self.sim_output_resolution_time = sim_config['resolution'] if not config['simulation']['test_mode_en'] else sim_config['test_resolution']
        
        self.sim_time_series = np.linspace(0, self.sim_time, int(self.sim_time/self.sim_output_resolution_time))

        
        if config['simulation']['test_mode_en']:
            print("NB ********* Test Mode is ENABLED *********")

        # return self.satellite
    
    def simulate(self):
        # Integrate satellite dynamics over time
        sol = solve_ivp(fun=self.satellite.calc_state_rates, t_span=[0, self.sim_time], y0=self.initial_values, method="RK45",
                    t_eval=self.sim_time_series,
                    max_step=0.1)
        return sol
    
    def collect_results(self, sol):

        # Put results into data object
        for key, value in results_data.items():
            if key == 'time':
                continue
            if len(value) > int(self.sim_time/self.sim_output_resolution_time):
                if key[:7] == "control":
                    results_data[key] = interpolate_data(value, results_data['control_time'], self.sim_time_series)[:]
                else:
                    try:
                        results_data[key] = interpolate_data(value, results_data['time'], self.sim_time_series)[:]
                    except:
                        print(f"Error interpolating {key} {value}")
        
        results_data['time'] = self.sim_time_series

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
            alpha = my_utils.conv_Rotation_obj_to_euler_axis_angle(r_1)
            results_data['euler_axis'].append(alpha)
        
        results_data['euler_int'] = cumulative_trapezoid(results_data['euler_axis'], results_data['time'], initial=0)

        q = np.array([results_data["quaternion_x"], 
                                            results_data["quaternion_y"], 
                                            results_data["quaternion_z"], 
                                            results_data["quaternion_w"]])
        r =  Rotation.from_quat(quat=q.T)
        [results_data['euler_yaw'],results_data['euler_pitch'], results_data['euler_roll']] = r.as_euler('zyx', degrees=True).T

        if self.config['output']['energy_enable']:
            self.calc_control_energy_output_results(sol)
        
        if self.config['output']['accuracy_enable']:
            self.calc_accuracy_output_results()
    
    control_energy_log_output = ""
    def calc_control_energy_output_results(self, sol):
        control_energy_arr = sol.y[7:]
        control_energy_per_axis = {}
        control_energy_total = 0
        for i,axis in enumerate(my_utils.xyz_axes):
            control_energy_per_axis[axis] = np.sum(np.abs(control_energy_arr[i]))
            control_energy_total += control_energy_per_axis[axis]
            results_data[f'control_energy_{axis}'] = control_energy_arr[i]
        self.control_energy_log_output = f"control energy (J): {my_utils.round_dict_values(control_energy_per_axis,3)} | total: {round(control_energy_total,3)}"
        print(self.control_energy_log_output)

    accuracy_percent = None
    settling_time = None
    steady_state = None

    def calc_accuracy_output_results(self):
        try:
            euler_axis_init = my_utils.conv_Rotation_obj_to_euler_axis_angle(self.satellite.dir_init)
            euler_axis_ref = my_utils.conv_Rotation_obj_to_euler_axis_angle(self.satellite.ref_q)
            control_info = control.step_info(sysdata=results_data[f"euler_int"], 
                                            SettlingTimeThreshold=0.002, T=results_data['time'])
            self.steady_state = control_info['SteadyStateValue']
            accuracy = 1-np.abs((results_data[f"euler_axis"][-1]-euler_axis_ref)/(euler_axis_ref-euler_axis_init))
            self.settling_time = control_info['SettlingTime']

            self.accuracy_percent = accuracy*100
            
            final_q = [results_data[f'quaternion_{axis}'][-1] for axis in my_utils.q_axes]
            final_euler  = Rotation.from_quat(final_q).as_euler('xyz', degrees=True)
            print(f"accuracy %: {self.accuracy_percent}")
            print(f"settling_time (s): {round(self.settling_time,3)}")
            print(f"steady_state (s): {round(self.steady_state,3)}")
            # print(f"final euler: {final_euler} deg xyz")
            # print(f"euler error: {euler_axis_ref - final_euler} deg xyz")

            control_info_y = control.step_info(sysdata=results_data['euler_yaw'],SettlingTimeThreshold=0.002, T=results_data['time'])
            control_info_p = control.step_info(sysdata=results_data['euler_pitch'],SettlingTimeThreshold=0.002, T=results_data['time'])
            control_info_r = control.step_info(sysdata=results_data['euler_roll'],SettlingTimeThreshold=0.002, T=results_data['time'])
            print(f"steady state error: {control_info_y['SteadyStateValue']} {control_info_p['SteadyStateValue']} {control_info_r['SteadyStateValue']} deg zyx")
        except Exception as e:
            print(f"Error calculating accuracy: {e}")

    def log_output_to_file(self, LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en):
        if os.path.exists(LOG_FOLDER_PATH) is False:
            os.mkdir(LOG_FOLDER_PATH)
        if LOG_FILE_NAME != None and test_mode_en is False:
            LOG_FILE_NAME_RESULTS = LOG_FILE_NAME + "_results"
            # clear_log_file(fr"{LOG_FOLDER_PATH}\{LOG_FILE_NAME_RESULTS}")
            output_dict_to_csv(LOG_FOLDER_PATH, LOG_FILE_NAME + "_log", results_data)
            output_toml_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME + "_config", self.config)
            if self.accuracy_percent is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"accuracy %: {self.accuracy_percent}", False)
            if self.settling_time is not None:
                log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"settling_time (s): {round(self.settling_time,3)}",    False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, self.control_energy_log_output, False)
            log_to_file(LOG_FOLDER_PATH, LOG_FILE_NAME_RESULTS, f"{self.steady_state}", False)

    def create_pdf_separated(self, rows, results_data, config, LOG_FILE_NAME):
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
                ax_separate.plot(results_data['time'], results_data[name], label=axis)
            
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
    
    def create_pdf_combined(self, rows, cols, results_data, config, LOG_FILE_NAME):
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
                current_plot.plot(results_data['time'], results_data[name], label=axis)

            current_plot.set_xlabel('time (s)')
            current_plot.set_ylabel(label)
            if current_plot.get_legend_handles_labels()[0] != []:
                current_plot.legend()

            plt.subplots_adjust(wspace=0.5, hspace=0.5)

        plt.show()

        if config['output']['pdf_output_enable'] is True and LOG_FILE_NAME != None and config['simulation']['test_mode_en'] is False:

            fig.savefig(fr"..\data_logs\{LOG_FILE_NAME}\{LOG_FILE_NAME}_summary.pdf", bbox_inches='tight')

def main():
    LOG_FILE_NAME, LOG_FOLDER_PATH, config = parse_args()

    results_data["time"] = []
    for axis in my_utils.xyz_axes:
        results_data["torque_" + axis] = []
        results_data["angular_acc_" + axis] = []
        results_data["angular_rate_" + axis] = []
        results_data['T_dist_' + axis] = []
    for axis in my_utils.q_axes:
        results_data["quaternion_" + axis] = []
    
    simulation = Simulation(config)
    satellite = simulation.satellite
    wheel_module = satellite._wheel_module
    controller = satellite._controller
    test_mode_en = config['simulation']['test_mode_en']

    #-------------------------------------------------------------#
    ###################### Simulate System ########################
    if config['simulation']['enable']:
            sol = simulation.simulate()
    else:
        exit()
    print("Simulation Complete")

    #-------------------------------------------------------------#
    ###################### Collect Results ########################
    collect_results = simulation.collect_results(sol)

    simulation.log_output_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en)

    cols = 2
    # rows = [ ('angular_rate',my_utils.xyz_axes), ('quaternion',my_utils.q_axes), ('euler_int',['none']), ('euler', ['yaw','pitch', 'roll']), \
    #         ('torque',my_utils.xyz_axes), ('control_energy',my_utils.xyz_axes), ('T_dist', my_utils.xyz_axes)]
    # rows = [ ('angular_rate',my_utils.xyz_axes, 'Angular rate (rad/s)'), ('quaternion',my_utils.q_axes, 'Quaternion'), ('euler_int',['none'], 'Euler integral (deg)'), ('euler', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), \
    #         ('torque',my_utils.xyz_axe, 'Torque (N)'), ('control_energy',my_utils.xyz_axes, 'Control Energy (J)'), ('T_dist', my_utils.xyz_axes, 'Torque Disturbance (N)')]
    
    ## Row should be in the form of (row_name, [axes], label)
    rows = [ ('angular_rate',my_utils.xyz_axes, 'Angular rate (rad/s)'), 
            ('euler', ['yaw','pitch', 'roll'], 'Euler angle (deg)'), 
            ('euler_int',['none'], 'Euler integral (deg)'), 
            ]
    
    if satellite.wheels_control_enable:
        # rows.append(('wheel_speed', [str(wheel.index) for wheel in wheel_module.wheels], 'Wheel speed (rad/s)'))
        rows.append(('wheel_torque', [str(wheel.index) for wheel in wheel_module.wheels], 'Control Torque (N)'))



    if controller.type == "adaptive":
        rows.append(('control_adaptive_model_output',['none']))
        rows.append(('control_theta',my_utils.xyz_axes))
    
    simulation.create_pdf_separated(rows, results_data, config, LOG_FILE_NAME)
    simulation.create_pdf_combined(rows, cols, results_data, config, LOG_FILE_NAME)

if __name__ == '__main__':
    # if config['simulation']['profile']:
    cProfile.run('main()', '../data_logs/profile_stats.prof')

    # else:


