import numpy as np

class StateProvider:
    config = None
    results_data = None
    
    def __init__(self, config, results_data):
        self.config = config
        self.results_data = results_data

    def get_size(self, results_data):
        return (0, 0, 0)

    def calc_state_rates(self, t, state, u):
        return np.zeros_like(state)

    def calc_state_outputs(self, t, state, u):
        return np.array([])