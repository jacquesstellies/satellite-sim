import numpy as np

class Fault():
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 0
    torque_limit_mul = 1
    mul_fault_matrix = None
    add_fault_matrix = None
    add_torque = 0

    def __init__(self, config):
        self.time = config['fault']['time']
        self.wheel_num = config['fault']['wheel_num']
        self.master_enable = config['fault']['master_enable']
        self.mul_fault_matrix = np.diag(config['fault']['mul_fault'])
        self.torque_limit_mul = self.mul_fault_matrix[self.wheel_num][self.wheel_num]
        self.add_fault_matrix = config['fault']['add_fault']
        self.add_torque = self.add_fault_matrix[self.wheel_num]

    def update(self, t):
        if not self.master_enable:
            return
        if t >= self.time:
            self.enabled = True
        else:
            self.enabled = False