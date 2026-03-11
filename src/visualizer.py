from astropy.time import Time
from datetime import datetime
import json
import numpy as np
import pandas as pd

def convert_jd_to_datetime(jd):
    t = Time(jd, format='jd')
    return t.datetime

## A function to convert the simulation data to a JSON format for visualization
# def convert_sim_data_to_json(time : list[datetime], quaternion: list[np.array], position: list[np.array], velocity: list[np.array]):
def convert_sim_data_to_json(results_df: pd.DataFrame):

    json_payload = {
        'satellite_count': 1,
        'start_time': results_df['datetime'].iloc[0].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'end_time': results_df['datetime'].iloc[-1].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        'satellites': []
    }
    satellite_data = {
        'satellite_name': 'EOS-SAT1',
        'telemetry': []
    }

    for i in range(len(results_df)):
        sim_data = {
            'time': results_df['datetime'].iloc[i].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'quaternion': [results_df['q_sat_x'].iloc[i],
                           results_df['q_sat_y'].iloc[i],
                           results_df['q_sat_z'].iloc[i],
                           results_df['q_sat_w'].iloc[i]],
            'position': [results_df['s_sat_eci_x'].iloc[i]*1e3, # convert from km to m
                         results_df['s_sat_eci_y'].iloc[i]*1e3,
                         results_df['s_sat_eci_z'].iloc[i]*1e3],
            'velocity': [results_df['v_sat_eci_x'].iloc[i]*1e3,
                         results_df['v_sat_eci_y'].iloc[i]*1e3,
                         results_df['v_sat_eci_z'].iloc[i]*1e3],
        }
        satellite_data['telemetry'].append(sim_data)

    json_payload['satellites'].append(satellite_data)

    return json.dumps(json_payload, indent=4), results_df

def convert_results_df_to_json(results_df: pd.DataFrame, t_sample: float = 1.0):
    t1 = 0
    mask = np.zeros(len(results_df), dtype=bool)
    for i in range(1, len(results_df)):
        time_diff = results_df['time'].iloc[i] - t1
        if time_diff >= t_sample:
            mask[i] = True
            t1 = t1 + t_sample
            
    results_df = results_df[mask]
    # time = [convert_jd_to_datetime(jd) for jd in results_df['jd']]
    results_df['datetime'] = [convert_jd_to_datetime(jd) for jd in results_df['jd']]

    # results_df['q_sat'] = results
    # quaternion = results_df['q_sat'].tolist()
    # position = results_df['s_eci'].tolist()
    # velocity = results_df['v_eci'].tolist()

    # return time
    return convert_sim_data_to_json(results_df)