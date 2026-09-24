
import visualizer as viz
import requests
from simulation import Simulation

import my_utils

import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import toml
import datetime
import cProfile
import pyswarms

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
                LOG_FILE_NAME = overide
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
    sim_obj = Simulation(config, results_data = results_data_local, logging_en=True)
    
    [sim_obj.satellite.controller.nadafi_controller.Gamma_z11,
        sim_obj.satellite.controller.nadafi_controller.Gamma_z22,
        sim_obj.satellite.controller.nadafi_controller.lambda_1,
        sim_obj.satellite.controller.nadafi_controller.lambda_2,
        sim_obj.satellite.controller.nadafi_controller.lambda_3] = gains
    sol = sim_obj.simulate_monte_carlo()
    if sol == -1:
        cost = 1e9
    else:
        sim_obj.collect_results(sol, use_only_sol=True)
        SSE = sum(sim_obj.results_df['euler_axis_sat_error']**2) # Sum of squared principal angle errors
        cost = SSE + sim_obj.results_df['euler_axis_sat_error'].iloc[-1]*1e5
    del sim_obj
    del results_data_local
    return cost

# simulation.results_data = {}
def Nadafi_BS_FNDO_controller_param_optimize(gains, particle_num, config):
    # Calculate Cost
    results_data_local = {}
    config['simulation']['duration'] = config['tuning']['duration']
    sim_obj = Simulation(config, results_data = results_data_local, logging_en=True)
    
    [sim_obj.satellite.controller.nadafi_controller.Gamma_z11,
        sim_obj.satellite.controller.nadafi_controller.Gamma_z22,
        sim_obj.satellite.controller.nadafi_controller.lambda_1,
        sim_obj.satellite.controller.nadafi_controller.lambda_2,
        sim_obj.satellite.controller.nadafi_controller.lambda_3,
        sim_obj.satellite.controller.nadafi_controller.L11,
        sim_obj.satellite.controller.nadafi_controller.L22,
        sim_obj.satellite.controller.nadafi_controller.kappa_0,
        sim_obj.satellite.controller.nadafi_controller.kappa_1] = gains
    sol = sim_obj.simulate_monte_carlo()
    cost = 0
    if sol == -1:
        cost = 1e9
    else:
        sim_obj.collect_results(sol, use_only_sol=True)
        SSE = sum(sim_obj.results_df['euler_axis_sat_error']**2) # Sum of squared principal angle errors
        cost = SSE + sim_obj.results_df['euler_axis_sat_error'].iloc[-1]*1e5
    # print(f"Cost: {SSE} | Gains: {gains}, Particle: {particle_num}")
    del sim_obj
    del results_data_local
    return cost

def Nadafi_BS_MFNDO_controller_param_optimize(gains, particle_num, config):
    # Calculate Cost
    results_data_local = {}
    config['simulation']['duration'] = config['tuning']['duration']
    sim_obj = Simulation(config, results_data = results_data_local, logging_en=True)
    
    [sim_obj.satellite.controller.nadafi_controller.Gamma_z11,
        sim_obj.satellite.controller.nadafi_controller.Gamma_z22,
        sim_obj.satellite.controller.nadafi_controller.lambda_1,
        sim_obj.satellite.controller.nadafi_controller.lambda_2,
        sim_obj.satellite.controller.nadafi_controller.lambda_3,
        sim_obj.satellite.controller.nadafi_controller.L11,
        sim_obj.satellite.controller.nadafi_controller.L22,
        sim_obj.satellite.controller.nadafi_controller.kappa_0,
        sim_obj.satellite.controller.nadafi_controller.kappa_1,
        sim_obj.satellite.controller.nadafi_controller.Gamma_mu11,
        sim_obj.satellite.controller.nadafi_controller.Gamma_mu22,
        sim_obj.satellite.controller.nadafi_controller.alpha_1,
        sim_obj.satellite.controller.nadafi_controller.alpha_2] = gains
    sol = sim_obj.simulate_monte_carlo()
    cost = 0
    if sol == -1:
        cost = 1e9
    else:
        sim_obj.collect_results(sol, use_only_sol=True)
        SSE = sum(sim_obj.results_df['euler_axis_sat_error']**2) # Sum of squared principal angle errors
        cost = SSE + sim_obj.results_df['euler_axis_sat_error'].iloc[-1]*1e5
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
            elif config['controller']['sub_type'] == "Nadafi_MFNDO":
                initial_guess = np.array([config['Nadafi']['Gamma_z11'],
                                config['Nadafi']['Gamma_z22'],
                                config['Nadafi']['lambda_1'], 
                                config['Nadafi']['lambda_2'], 
                                config['Nadafi']['lambda_3'],
                                config['Nadafi']['L11'],
                                config['Nadafi']['L22'], 
                                config['Nadafi']['kappa_0'],
                                config['Nadafi']['kappa_1'], 
                                config['Nadafi']['Gamma_mu11'],
                                config['Nadafi']['Gamma_mu22'],
                                config['Nadafi']['alpha_1'],
                                config['Nadafi']['alpha_2']
                                ])

            if config['tuning']['postive_bounds_en']:
                bounds = (np.zeros(len(initial_guess)), np.ones(len(initial_guess))*100)
            else:
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
            elif config['controller']['sub_type'] == "Nadafi_FNDO":
                func = Nadafi_BS_FNDO_controller_param_optimize
            elif config['controller']['sub_type'] == "Nadafi_MFNDO":
                func = Nadafi_BS_MFNDO_controller_param_optimize

            cost, gains = optimizer.optimize(calc_cost, iters=config['tuning']['iterations'], kwargs={'config': config, 'func': func}, n_processes=min(os.cpu_count(), n_particles))
            print("Finished tuning step")
            print(f"Final Gains: {gains}")
            print(f"Final Cost: {cost}")


            config['simulation']['verbose'] = True
            simulation = Simulation(config, results_data)

            if config['controller']['sub_type'] == "Nadafi_BS":
                simulation.satellite.controller.nadafi_controller.Gamma_z11 = gains[0]
                simulation.satellite.controller.nadafi_controller.Gamma_z22 = gains[1]
                simulation.satellite.controller.nadafi_controller.lambda_1 = gains[2]
                simulation.satellite.controller.nadafi_controller.lambda_2 = gains[3] 
                simulation.satellite.controller.nadafi_controller.lambda_3 = gains[4]
            elif config['controller']['sub_type'] == "Nadafi_FNDO":
                simulation.satellite.controller.nadafi_controller.Gamma_z11 = gains[0]
                simulation.satellite.controller.nadafi_controller.Gamma_z22 = gains[1]
                simulation.satellite.controller.nadafi_controller.lambda_1 = gains[2]
                simulation.satellite.controller.nadafi_controller.lambda_2 = gains[3]
                simulation.satellite.controller.nadafi_controller.lambda_3 = gains[4]
                simulation.satellite.controller.nadafi_controller.L11 = gains[5]
                simulation.satellite.controller.nadafi_controller.L22 = gains[6]
                simulation.satellite.controller.nadafi_controller.kappa_0 = gains[7]
                simulation.satellite.controller.nadafi_controller.kappa_1 = gains[8] 
            elif config['controller']['sub_type'] == "Nadafi_MFNDO":
                simulation.satellite.controller.nadafi_controller.Gamma_z11 = gains[0]
                simulation.satellite.controller.nadafi_controller.Gamma_z22 = gains[1]
                simulation.satellite.controller.nadafi_controller.lambda_1 = gains[2]
                simulation.satellite.controller.nadafi_controller.lambda_2 = gains[3]
                simulation.satellite.controller.nadafi_controller.lambda_3 = gains[4]
                simulation.satellite.controller.nadafi_controller.L11 = gains[5]
                simulation.satellite.controller.nadafi_controller.L22 = gains[6]
                simulation.satellite.controller.nadafi_controller.kappa_0 = gains[7]
                simulation.satellite.controller.nadafi_controller.kappa_1 = gains[8] 
                simulation.satellite.controller.nadafi_controller.Gamma_mu11 = gains[9]
                simulation.satellite.controller.nadafi_controller.Gamma_mu22 = gains[10]
                simulation.satellite.controller.nadafi_controller.alpha_1 = gains[11]
                simulation.satellite.controller.nadafi_controller.alpha_2 = gains[12]
            else:
                raise Exception(f"Controller sub type {config['controller']['sub_type']} not recognized for tuning")
            simulation.logging_en = True
            # gains = np.array(gains)
            print(f"Gamma_z11 = {simulation.satellite.controller.nadafi_controller.Gamma_z11}")
            print(f"Gamma_z22 = {simulation.satellite.controller.nadafi_controller.Gamma_z22}")
            print(f"lambda_1 = {simulation.satellite.controller.nadafi_controller.lambda_1}")
            print(f"lambda_2 = {simulation.satellite.controller.nadafi_controller.lambda_2}")
            print(f"lambda_3 = {simulation.satellite.controller.nadafi_controller.lambda_3}")
            print(f"L11 = {simulation.satellite.controller.nadafi_controller.L11}")
            print(f"L22 = {simulation.satellite.controller.nadafi_controller.L22}")
            print(f"kappa_0 = {simulation.satellite.controller.nadafi_controller.kappa_0}")
            print(f"kappa_1 = {simulation.satellite.controller.nadafi_controller.kappa_1}")
            print(f"Gamma_mu11 = {simulation.satellite.controller.nadafi_controller.Gamma_mu11}\n")
            print(f"Gamma_mu22 = {simulation.satellite.controller.nadafi_controller.Gamma_mu22}\n")
            print(f"alpha_1 = {simulation.satellite.controller.nadafi_controller.alpha_1}\n")
            print(f"alpha_2 = {simulation.satellite.controller.nadafi_controller.alpha_2}\n")

            with open(fr'{LOG_FOLDER_PATH}/{LOG_FILE_NAME}_tuning_log' + ".txt", 'w+') as file:
                file.write(f"Final Gains:\n")
                file.write(f"Gamma_z11 = {simulation.satellite.controller.nadafi_controller.Gamma_z11}\n")
                file.write(f"Gamma_z22 = {simulation.satellite.controller.nadafi_controller.Gamma_z22}\n")
                file.write(f"lambda_1 = {simulation.satellite.controller.nadafi_controller.lambda_1}\n")
                file.write(f"lambda_2 = {simulation.satellite.controller.nadafi_controller.lambda_2}\n")
                file.write(f"lambda_3 = {simulation.satellite.controller.nadafi_controller.lambda_3}\n")
                file.write(f"L11 = {simulation.satellite.controller.nadafi_controller.L11}\n")
                file.write(f"L22 = {simulation.satellite.controller.nadafi_controller.L22}\n")
                file.write(f"kappa_0 = {simulation.satellite.controller.nadafi_controller.kappa_0}\n")
                file.write(f"kappa_1 = {simulation.satellite.controller.nadafi_controller.kappa_1}\n")
                file.write(f"Gamma_mu11 = {simulation.satellite.controller.nadafi_controller.Gamma_mu11}\n")
                file.write(f"Gamma_mu22 = {simulation.satellite.controller.nadafi_controller.Gamma_mu22}\n")
                file.write(f"alpha_1 = {simulation.satellite.controller.nadafi_controller.alpha_1}\n")
                file.write(f"alpha_2 = {simulation.satellite.controller.nadafi_controller.alpha_2}\n")
                file.write(f"Final Cost: {cost}\n")

            print(f"Running simulation with tuned gains: {gains}")
            simulation.sim_time = 300
            sol = simulation.simulate()
            print("Simulation Complete")

            simulation.collect_results(sol)
            rows = [('q_sat',my_utils.q_axes, r'Quaternion $q$'), ('w_sat',my_utils.xyz_axes, r'Angular velocity $\omega$ (rad/s)' )]
            print("Creating plots of tuned parameters simulation...")
            simulation.create_plots_separated(rows, simulation.results_df, config, LOG_FILE_NAME)

            del(simulation)

        else:
            #-------------------------------------------------------------#
            ###################### Simulate System ########################
            if sim_iter == 1:

                ###############################################
                simulation = Simulation(config, results_data, log_file_name=LOG_FILE_NAME, log_folder_path=LOG_FOLDER_PATH, logging_en=True)
                satellite = simulation.satellite
                wheel_module = satellite.wheel_module
                controller = satellite.controller
                print(f"Running Once-off simulation")
                sol = simulation.simulate()

                print("Simulation Complete")
                
                if (test_mode_en):
                    exit(0)
                simulation.collect_results(sol)
                        
                if config['output']['accuracy_enable']:
                    simulation.calc_accuracy_output_results()  

                # Plot rows are built from the config, so plot_viewer.ipynb can
                # rebuild the same set from a log folder. See my_utils.
                my_utils.calc_lumped_disturbance(simulation.results_df, config)
                my_utils.create_results_plots(simulation.results_df, config, LOG_FILE_NAME, LOG_FOLDER_PATH, cols=2)

                if satellite.wheels_control_enable:
                    simulation.log_data_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH)

                if config['output']['visualizer']['enable'] is True:
                    visualizer_dict = viz.parse_results(simulation.results_df, config['output']['visualizer']['t_sample'])
                    if config['output']['visualizer']['write_to_file'] is True:
                        with open(config['output']['visualizer']['file_path'], "w") as f:
                            json.dump(visualizer_dict, f, indent=2)
                    if config['output']['visualizer']['publish'] is True:
                        url = "http://localhost:3000/api/telemetry/update"
                        response = requests.post(url, json=visualizer_dict)
                        print(f"Visualizer publish: {response.status_code}")
                        if not response.ok:
                            print(response.text)
                    else:
                        print(json.dumps(visualizer_dict, indent=2))

                    
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
                                
                        if config['output']['accuracy_enable']:
                            simulation.calc_accuracy_output_results()  
                        results['accuracy'].append(simulation.accuracy)
                        results['settling_time'].append(simulation.settling_time)
                        # results['euler_axis_final'].append(results_data['euler_axis'][-1])
                        results['euler_axis_final'].append(simulation.steady_state_euler_axis)
                        

                    rows = [('accuracy', 'none', 'Accuracy'), ('settling_time', 'none', 'Settling Time')]
                    simulation.create_plots_combined(rows, cols, results_data, config, LOG_FILE_NAME, type='scatter', x_axis=[monte_carlo_results['euler_axis_final']])
                    # simulation.log_data_to_file(LOG_FILE_NAME, LOG_FOLDER_PATH, test_mode_en)
                    if simulation.accuracy > 0:
                        passed = True
                    else:
                        passed = False
                    return passed

                def plot_q(passed):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlabel(my_utils.latex_label('q_sat', 'x'))
                    ax.set_ylabel(my_utils.latex_label('q_sat', 'y'))
                    ax.set_zlabel(my_utils.latex_label('q_sat', 'z'))
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



