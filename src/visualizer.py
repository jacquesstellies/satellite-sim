from astropy.time import Time
from datetime import datetime
import json
import numpy as np
import pandas as pd


def convert_jd_to_datetime(jd):
    t = Time(jd, format='jd')
    return t.datetime


def _fmt_time(dt: datetime) -> str:
    """ISO-8601 UTC timestamp with millisecond precision."""
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def _f(value, digits: int) -> float:
    return round(float(value), digits)


## A function to convert the simulation data to a JSON format for visualization
def convert_sim_data_to_visualizer_format(results_df: pd.DataFrame):
    json_payload = {
        'satellite_count': 1,
        'start_time': _fmt_time(results_df['datetime'].iloc[0]),
        'end_time': _fmt_time(results_df['datetime'].iloc[-1]),
        'satellites': []
    }
    satellite_data = {
        'satellite_name': 'EOS-SAT1',
        'telemetry': []
    }

    for i in range(len(results_df)):
        sim_data = {
            'time': _fmt_time(results_df['datetime'].iloc[i]),
            'quaternion': [
                _f(results_df['q_sat_x'].iloc[i], 6),
                _f(results_df['q_sat_y'].iloc[i], 6),
                _f(results_df['q_sat_z'].iloc[i], 6),
                _f(results_df['q_sat_w'].iloc[i], 6),
            ],
            'position': [
                _f(results_df['s_sat_eci_x'].iloc[i] * 1e3, 3),  # km -> m
                _f(results_df['s_sat_eci_y'].iloc[i] * 1e3, 3),
                _f(results_df['s_sat_eci_z'].iloc[i] * 1e3, 3),
            ],
            'velocity': [
                _f(results_df['v_sat_eci_x'].iloc[i] * 1e3, 3),
                _f(results_df['v_sat_eci_y'].iloc[i] * 1e3, 3),
                _f(results_df['v_sat_eci_z'].iloc[i] * 1e3, 3),
            ],
        }
        satellite_data['telemetry'].append(sim_data)

    json_payload['satellites'].append(satellite_data)

    return json_payload


def parse_results(results_df: pd.DataFrame, t_sample: float = 1.0):
    """Parse the results dataframe and return a dictionary in the format expected by the visualizer."""
    # Sample the results dataframe at the given time interval
    t1 = 0
    mask = np.zeros(len(results_df), dtype=bool)
    for i in range(1, len(results_df)):
        time_diff = results_df['time'].iloc[i] - t1
        if time_diff >= t_sample:
            mask[i] = True
            t1 = t1 + t_sample

    # Use .loc[...] and .copy() to avoid SettingWithCopyWarning when assigning new columns
    results_df = results_df.loc[mask].copy()
    results_df.loc[:, 'datetime'] = results_df['jd'].apply(convert_jd_to_datetime)

    # Convert the sampled results dataframe to the visualizer format
    return convert_sim_data_to_visualizer_format(results_df)
