import numpy as np

class Fault():
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 0
    mul_fault_matrix = None
    add_fault_matrix = None
    E = None

    def __init__(self, config):
        self.time = config['fault']['time']
        self.wheel_num = config['fault']['wheel_num']
        self.master_enable = config['fault']['master_enable']
        self.mul_fault_matrix = np.diag(config['fault']['mul_fault'])
        self.add_fault_matrix = config['fault']['add_fault']
        self.E = np.eye(len(self.mul_fault_matrix))

    count = 0
    def update(self, t):
        if t < self.time or not self.master_enable:
            return False
        if self.count == 0:
            print("fault_enabled")
            self.E = self.mul_fault_matrix
            self.count+=1
        return True