import my_globals
from wheels import Wheel
from observer import WheelExtendedStateObserver, WheelObserver
from fault import Fault
import my_utils

import numpy as np
from scipy.integrate import solve_ivp
import toml
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import argparse

class Simulation:
    t_prev = 0
    t_next = 0
    y_est = None  # Initial state estimate: [disturbance, wheel speed]
    def __init__(self, type):
        self.t_prev = 0
        if type == "extended":
            self.y_est = np.zeros(3)
        else:
            self.y_est = 0

    def wheel_extended_state_observer_simulate(self, t, y, u, wheel, observer):
        y = wheel.calc_state_rates(t, y, u)

        dw = y[0]
        w = y[1]

        if t >= self.t_next:
            self.t_next += observer.t_sample
            self.y_est = observer.calc_state_rates(t, self.y_est, [u, w])

        wheel._fault.update(t)
        [dH, H] = wheel.calc_state_outputs(t, [dw, w])
        results_data['time'].append(t)
        results_data['wheel_speed'].append(w)
        results_data['wheel_torque'].append(dH)
        results_data['control_input_torque'].append(u)

        results_data['wheel_speed_est'].append(self.y_est[0])  # Estimated wheel speed
        results_data['wheel_disturbance_est'].append(self.y_est[1])  # Estimated disturbance
        results_data['wheel_torque_est'].append(self.y_est[2]*wheel.M_inertia[2][2])  # Estimated torque
        return y
    
    y_est = 0
    def wheel_observer_simulate(self, t, y, u, wheel, observer):
        y = wheel.calc_state_rates(t, y, u)

        dw = y[0]
        w = y[1]
        if t - self.t_prev >= observer.t_sample:
            self.t_prev = t
            self.y_est = observer.calc_state_rates(t, self.y_est, [u, w])
        wheel._fault.update(t)
        [dH, H] = wheel.calc_state_outputs(t, [dw, w])
        results_data['time'].append(t)
        results_data['wheel_speed'].append(w*60/2*np.pi)
        results_data['wheel_torque'].append(dH)
        results_data['wheel_speed_est'].append(self.y_est*60/2*np.pi)  # Estimated wheel speed

        return y

    def wheel_extended_state_observer_speed_control_simulate(self, t, y, w_ref, wheel, observer):
        y, u = wheel.calc_state_speed_control(t, y, w_ref)

        dw = y[0]
        w = y[1]

        if t >= self.t_next:
            self.t_next += observer.t_sample
            self.y_est = observer.calc_state_rates(t, self.y_est, [u, w])
        wheel._fault.update(t)
        [dH, H] = wheel.calc_state_outputs(t, [dw, w])
        results_data['time'].append(t)
        results_data['wheel_speed'].append(w)
        results_data['wheel_torque'].append(dw*wheel.M_inertia_fast)
        results_data['control_input_torque'].append(u)

        results_data['wheel_speed_est'].append(self.y_est[0])  # Estimated wheel speed
        results_data['wheel_disturbance_est'].append(self.y_est[1])  # Estimated disturbance
        results_data['wheel_torque_est'].append(self.y_est[2]*wheel.M_inertia[2][2])  # Estimated torque
        return y

results_data = my_globals.results_data

def wheel_simulate(t, y, u, wheel):

    print(f"t = {t}, y = {y}, u = {u}")
    y = wheel.calc_state_rates(t, y, u)

    dw = y[0]
    w = y[1]

    [dH, H] = wheel.calc_state_outputs(t, [dw, w])
    results_data['time'].append(t)
    results_data['wheel_speed'].append(w*60/2*np.pi)
    results_data['wheel_torque'].append(dw*wheel.M_inertia[2][2])

    return y

def test_wheel_constant_torque_response(config):
    fault = Fault(config)
    wheel = Wheel(config=config, fault=fault)
    duration = config['simulation']['duration']  # Simulation duration in seconds
    max_step_size = min(config['controller']['t_sample'], config['observer']['t_sample'])
    t_eval = np.linspace(0, duration, int(duration/max_step_size))
    print(f"t_eval = {t_eval}")

    print("Im here")
    T_input = 0.2  # Constant torque input

    results_data['time'] = []
    results_data['wheel_speed'] = []
    results_data['wheel_torque'] = []

    solve_ivp(
        fun=wheel_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(T_input,wheel),  # Control input: constant torque
        method='RK45',
        max_step=max_step_size,
        t_eval=t_eval)

# def test_wheel_impulse_response(config):


def test_wheel_extended_state_observer_constant_torque(config):
    fault = Fault(config)
    wheel = Wheel(config=config, fault=fault)
    observer = WheelExtendedStateObserver(config, wheel)
    simulation = Simulation("extended")
    duration = config['simulation']['duration']  # Simulation duration in seconds

    T_input = config['simulation']['input_torque']  # Constant torque input

    results_data['time'] = []
    results_data['wheel_speed'] = []
    results_data['wheel_torque'] = []
    results_data['control_input_torque'] = []

    results_data['wheel_speed_est'] = []
    results_data['wheel_disturbance_est'] = []
    results_data['wheel_torque_est'] = []

    max_step_size=min(config['controller']['t_sample'], config['observer']['t_sample'])/10

    # t_eval = np.linspace(0, duration, int(duration/max_step_size))
    t_eval = np.arange(0, duration, max_step_size)
    # print(f"t_eval = {t_eval}")

    solve_ivp(
        fun=simulation.wheel_extended_state_observer_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(T_input, wheel, observer),  # Control input: constant torque
        method='RK45',
        t_eval=t_eval,
        max_step=max_step_size,
        dense_output=True
    )
    return wheel

def test_wheel_extended_state_observer_speed_control(config):
    fault = Fault(config)
    wheel = Wheel(config=config, fault=fault)
    observer = WheelExtendedStateObserver(config, wheel)
    simulation = Simulation("extended")
    duration = config['simulation']['duration']  # Simulation duration in seconds

    w_input = config['simulation']['w_ref_rpm']*2*np.pi/60  # Constant wheel speed reference

    results_data['time'] = []
    results_data['wheel_speed'] = []
    results_data['wheel_torque'] = []
    results_data['control_input_torque'] = []

    results_data['wheel_speed_est'] = []
    results_data['wheel_disturbance_est'] = []
    results_data['wheel_torque_est'] = []

    max_step_size=min(config['controller']['t_sample'], config['observer']['t_sample'])

    solve_ivp(
        fun=simulation.wheel_extended_state_observer_speed_control_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(w_input, wheel, observer),  # Control input: constant wheel speed reference
        method='RK45',
        max_step=max_step_size,
        dense_output=True
    )
    return wheel

def test_wheel_observer(config):
    fault = Fault(config)
    wheel = Wheel(config=config, fault=fault)
    observer = WheelObserver(config, wheel)
    simulation = Simulation("simple")
    duration = config['simulation']['duration']  # Simulation duration in seconds
    t_eval = np.linspace(0, duration, int(duration/config['controller']['t_sample']))

    T_input = 0.2  # Constant torque input

    results_data['time'] = []
    results_data['wheel_speed'] = []
    results_data['wheel_torque'] = []
    results_data['wheel_speed_est'] = []

    max_step_size=min(config['controller']['t_sample'], config['observer']['t_sample'])
    solve_ivp(
        fun=simulation.wheel_observer_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(T_input, wheel, observer),  # Control input: constant torque
        method='RK45',
        # t_eval=t_eval,
        max_step=max_step_size,
        dense_output=True
    )

# def test_wheel_observer_speed_step_response(config):

def plot_results():

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(results_data['time'], results_data['wheel_speed'], label='Wheel Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (rpm/s)')
    plt.title('Wheel Speed Over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(results_data['time'], results_data['wheel_torque'], label='Wheel Torque', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Wheel Torque Over Time')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.abspath('../data_logs/wheel_test.pdf'))

def plot_results_obs():

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(results_data['time'], results_data['wheel_speed'], label='Wheel Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (rpm/s)')
    plt.title('Wheel Speed')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(results_data['time'], results_data['wheel_torque'], label='Wheel Torque', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Wheel Torque')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(results_data['time'], results_data['wheel_speed_est'], label='Wheel Speed Estimate', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (rpm)')
    plt.title('Wheel Speed Estimate')
    plt.legend()

    if 'wheel_disturbance_est' in results_data:
        plt.subplot(2, 2, 4)
        plt.plot(results_data['time'], results_data['wheel_disturbance_est'], 
                 label='Wheel Disturbance Estimate', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Wheel Disturbance Estimate')
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_results_eso(LOG_FILE_NAME):

    if not os.path.exists(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}")):
        os.mkdir(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}"))
    df = pd.DataFrame(results_data)
    df['wheel_speed_error_est'] = df['wheel_speed'] - df['wheel_speed_est']
    df['wheel_speed_error_est_rpm'] = df['wheel_speed_error_est'] * 60 / 2 * np.pi  # Speed error in rpm

    df['wheel_speed_rpm'] = df['wheel_speed'] * 60 / 2 * np.pi
    df['wheel_disturbance'] = - df['control_input_torque'] - df['wheel_torque']
    df['wheel_speed_est_rpm'] = df['wheel_speed_est'] * 60 / 2 * np.pi  # Estimated wheel speed
    df['wheel_torque_error_est'] = df['wheel_torque'] - df['wheel_torque_est']  # Estimated torque error

    df['wheel_disturbance_error_est'] = df["wheel_disturbance"] - df['wheel_disturbance_est']
    df['wheel_control_authority'] = (df['control_input_torque'] + df['wheel_disturbance_est'])/df['control_input_torque']


    # mask = df['wheel_torque_est'] <= 0.1*config['wheels']['max_torque']
    # df.loc[mask, 'wheel_control_authority'] = 1.0   

    rows = [
            ('wheel_speed_rpm', None, 'Speed (rpm)'),
            ('wheel_torque', None, 'Torque (Nm)'),
            ('control_input_torque', None, 'Input Torque (Nm)'),
            ('wheel_disturbance', None, 'Disturbance Torque Actual (Nm)'),
            ('wheel_speed_est_rpm', None, 'Speed Estimate (rpm)'),
            ('wheel_speed_error_est_rpm', None, 'Speed Error Estimate (rpm)'),
            ('wheel_disturbance_est', None, 'Disturbance Torque Estimate (Nm)'),
            ('wheel_torque_error_est', None, 'Torque Error Measured vs Estimate (Nm)'),
            ('wheel_disturbance_error_est', None, 'Disturbance Torque Estimate Error (Nm)'),
            ('wheel_control_authority', None, 'Control Authority Estimate (Dimensionless)'),
            ('wheel_torque_est', None, 'Torque Estimate (Nm)'),
            ]

    np.ndarray((2,1))
    # my_utils.create_plots_combined(
    #     rows=rows,
    #     cols=2,
    #     results_data=df,
    #     config=config,
    #     LOG_FILE_NAME=LOG_FILE_NAME
    # )
    my_utils.create_plots_separated(
        rows=rows,
        results_data=df,
        config=config,
        display=False,
        LOG_FILE_NAME=LOG_FILE_NAME
    )

    ax = plt.figure(figsize=(12, 6)).add_subplot(111)
    ax.plot(df['time'], df['wheel_speed_rpm'], label='Wheel Speed (rpm)', color='blue')
    ax.plot(df['time'], df['wheel_speed_est_rpm'], label='Wheel Speed Estimate (rpm)', linestyle='--', color='orange')
    ax.plot(df['time'], df['wheel_speed_error_est_rpm'], label='Speed Error Estimate (rpm)', linestyle='-', color='black')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (rpm)')
    ax.set_title('Wheel Speed and Estimate Over Time')
    ax.legend()

    plt.savefig(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_wheel_speed_est_vs_actual.pdf"), bbox_inches='tight')
    # plt.show)()

    ax = plt.figure(figsize=(12, 6)).add_subplot(111)
    ax.plot(df['time'], df['wheel_disturbance'], label='Wheel Disturbance (Nm)', color='blue')
    ax.plot(df['time'], df['wheel_disturbance_est'], label='Wheel Disturbance Estimate (Nm)', linestyle='--', color='orange')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Wheel Disturbance Torque vs Estimate Over Time')
    ax.legend()

    plt.savefig(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_wheel_disturbance_torque_est_vs_actual.pdf"), bbox_inches='tight')
    # plt.show)()

    ax = plt.figure(figsize=(12, 6)).add_subplot(111)
    ax.plot(df['time'], df['wheel_torque_est'], label='Wheel Torque Estimate (Nm)', color='blue', linestyle='--')
    ax.plot(df['time'], df['wheel_disturbance_est'], label='Wheel Disturbance Estimate (Nm)', linestyle='--', color='orange')
    ax.plot(df['time'], df['wheel_torque'], label='Wheel Torque Measured (Nm)', linestyle='-', color='green')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Wheel Torque vs Control Authority Estimate Over Time')
    # ax.set_ylim([-0.1, 0.25])
    ax.legend(loc='upper left')

    ax2 = ax.twinx()

    ax2.plot(df['time'], df['wheel_control_authority'], label='Torque Fraction Available', linestyle='-', color='r')
    ax2.set_ylabel('Control Authority (Dimensionless)')
    # ax2.set_ylim([0.0, 1.1])
    ax2.legend(loc='upper right')
    plt.savefig(os.path.abspath(fr"../data_logs/{LOG_FILE_NAME}/graphs/{LOG_FILE_NAME}_wheel_control_authority_vs_torque_n_disturbance.pdf"), bbox_inches='tight')




def parse_args():
    choices = ['constant_torque', 'step_response']
    parser = argparse.ArgumentParser(description="Wheel Simulation Test")
    parser.add_argument('--config', type=str, default='wheel_test.toml', help='Path to the configuration TOML file')
    parser.add_argument('--test', type=str, choices=choices, default='step_response', help='Type of test to run')
    parser.add_argument('--append', '-a', type=str, default='', help='String to append to log file names')
    parser.add_argument('--date', '-d', action='store_true', default=False, help='Append date to log file names')
    
    args = parser.parse_args()
    if args.test not in choices:
        parser.error(f"Invalid test type. Choose from {choices}.")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    log_name = "wheels_test"
    log_name += f"_{args.test}"
    if args.append:
        log_name += f"_{args.append}"
    if args.date:
        from datetime import datetime
        log_name += f"_{datetime.now().strftime(r'%Y-%m-%d')}"
    print(f"Logging to {log_name}")

    config = toml.load('wheel_test.toml')
    # test_wheel_constant_torque_response(config)
    # plot_results()

    # test_wheel_impulse_response(config)
    # plot_results
    # test_wheel_observer(config)
    # plot_results_obs()
    if args.test == 'constant_torque':
        wheel = test_wheel_extended_state_observer_constant_torque(config)
        plot_results_eso(log_name)

    if args.test == 'step_response': 
        wheel = test_wheel_extended_state_observer_speed_control(config)
        plot_results_eso(log_name)

    os.system(fr"cp {os.path.abspath('wheel_test.toml')} {os.path.abspath(fr'../data_logs/{log_name}/config.toml')}")