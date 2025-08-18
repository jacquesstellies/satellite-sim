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

class Simulation:
    t_prev = 0
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

        if t - self.t_prev >= observer.t_sample:
            self.t_prev = t
            self.y_est = observer.calc_state_rates(t, self.y_est, [u, w])

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

        [dH, H] = wheel.calc_state_outputs(t, [dw, w])
        results_data['time'].append(t)
        results_data['wheel_speed'].append(w*60/2*np.pi)
        results_data['wheel_torque'].append(dH)
        results_data['wheel_speed_est'].append(self.y_est*60/2*np.pi)  # Estimated wheel speed

        return y

results_data = my_globals.results_data

def wheel_simulate(t, y, u, wheel):

    y = wheel.calc_state_rates(t, y, u)

    dw = y[0]
    w = y[1]

    [dH, H] = wheel.calc_state_outputs(t, [dw, w])
    results_data['time'].append(t)
    results_data['wheel_speed'].append(w*60/2*np.pi)
    results_data['wheel_torque'].append(dw*wheel.M_inertia[2][2])

    return y

def test_wheel_step_response(config):
    fault = Fault(config)
    wheel = Wheel(config=config, fault=fault)
    duration = 20  # Simulation duration in seconds
    t_sample = 0.1

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
        max_step=t_sample,
        # t_eval=np.linspace(0, duration, int(duration/t_sample)), dense_output=True
    )

# def test_wheel_impulse_response(config):


def test_wheel_extended_state_observer(config):
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

    solve_ivp(
        fun=simulation.wheel_extended_state_observer_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(T_input, wheel, observer),  # Control input: constant torque
        method='RK45',
        # t_eval=t_eval,
        max_step=config['controller']['t_sample'],
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

    solve_ivp(
        fun=simulation.wheel_observer_simulate,
        t_span=(0, duration),
        y0=[0, 0],  # Initial state: [position, angular velocity]
        args=(T_input, wheel, observer),  # Control input: constant torque
        method='RK45',
        # t_eval=t_eval,
        max_step=config['controller']['t_sample'],
        dense_output=True
    )

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
    plt.show()

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

if __name__ == "__main__":
    config = toml.load('wheel_test.toml')
    # test_wheel_step_response(config)
    # plot_results()

    # test_wheel_impulse_response(config)
    # plot_results
    # test_wheel_observer(config)
    # plot_results_obs()

    wheel = test_wheel_extended_state_observer(config)

    LOG_FILE_NAME = "wheels_test"
    df = pd.DataFrame(results_data)
    df['wheel_speed_error_est'] = df['wheel_speed'] - df['wheel_speed_est']
    df['wheel_speed_error_est_rpm'] = df['wheel_speed_error_est'] * 60 / 2 * np.pi  # Speed error in rpm

    df['wheel_speed_rpm'] = df['wheel_speed'] * 60 / 2 * np.pi
    df['wheel_disturbance'] = - df['control_input_torque'] + df['wheel_torque']
    df['wheel_speed_est_rpm'] = df['wheel_speed_est'] * 60 / 2 * np.pi  # Estimated wheel speed
    df['wheel_torque_error_est'] = df['wheel_torque'] - df['wheel_torque_est']  # Estimated torque error

    df['wheel_disturbance_error_est'] = df["wheel_disturbance"] - df['wheel_disturbance_est']

    rows = [
            ('wheel_speed_rpm', None, 'Speed (rpm)'),
            ('wheel_torque', None, 'Torque (Nm)'),
            ('wheel_disturbance', None, 'Disturbance Torque Actual (Nm)'),
            ('wheel_speed_est_rpm', None, 'Speed Estimate (rpm)'),
            ('wheel_speed_error_est_rpm', None, 'Speed Error Estimate (rpm)'),
            ('wheel_disturbance_est', None, 'Disturbance Torque Estimate (Nm)'),
            ('wheel_torque_error_est', None, 'Torque Error Measured vs Estimate (Nm)'),
            ('wheel_disturbance_error_est', None, 'Disturbance Torque Error Estimate (Nm)'),
        ]

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

    plt.savefig(fr"..\data_logs\{LOG_FILE_NAME}\graphs\{LOG_FILE_NAME}_wheel_speed_est_vs_actual.pdf", bbox_inches='tight')
    plt.show()

    ax = plt.figure(figsize=(12, 6)).add_subplot(111)
    ax.plot(df['time'], df['wheel_disturbance'], label='Wheel Speed (rpm)', color='blue')
    ax.plot(df['time'], df['wheel_disturbance_est'], label='Wheel Speed Estimate (rpm)', linestyle='--', color='orange')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Wheel Disturbance Torque vs Estimate Over Time')
    ax.legend()

    plt.savefig(fr"..\data_logs\{LOG_FILE_NAME}\graphs\{LOG_FILE_NAME}_wheel_disturbance_torque_est_vs_actual.pdf", bbox_inches='tight')
    plt.show()