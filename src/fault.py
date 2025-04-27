import numpy as np

class Fault():
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 0
    torque_limit = 1
    types = ["catastrophic", "comm_delay", "torque_limit"]
    filter_coeff = 0
    mul_fault_matrix = None
    add_fault_matrix = None

    def __init__(self, config):
        self.time = config['fault']['time']
        self.wheel_num = config['fault']['wheel_num']
        self.type=config['fault']['type']
        self.torque_limit = config['fault']['torque_limit']
        self.master_enable = config['fault']['master_enable']
        self.filter_coeff = config['fault']['filter_coef']
        self.mul_fault_matrix = np.diag(config['fault']['mul_fault'])
        self.add_fault_matrix = config['fault']['add_fault']