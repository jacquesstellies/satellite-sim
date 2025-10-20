import numpy as np

class Fault():
    index = 0
    time = 0
    enabled = False
    master_enable = True
    type = "catastrophic"
    wheel_num = 0
    mul_fault_matrix = None
    add_fault_matrix = None
    E = None
    u_a = None
    config = None
    f = None

    def __init__(self, config, index):
        self.config = config
        self.index = index
        self.time = self.config['fault'][f'{self.index}']['time']

    def init(self, num_wheels):
        self.master_enable = self.config['fault'][f'{self.index}']['master_enable']
        if self.config['fault'][f'{self.index}']['mul_fault'] is None:
            self.mul_fault_matrix = np.ones(num_wheels)
        else:
            self.mul_fault_matrix = np.diag(self.config['fault'][f'{self.index}']['mul_fault'])
        
        if self.config['fault'][f'{self.index}']['add_fault'] is None:
            self.add_fault_matrix = np.zeros(num_wheels)
        else:
            self.add_fault_matrix = self.config['fault'][f'{self.index}']['add_fault']

        self.u_a = np.zeros(num_wheels)
        if len(self.mul_fault_matrix) != num_wheels:
            raise Exception(f"Fault mul_fault length {len(self.mul_fault_matrix)} does not match number of wheels {num_wheels}")
        self.E = np.eye(len(self.mul_fault_matrix))

    count = 0
    def update(self, t):
        if t < self.time or not self.master_enable or not self.config['faults']['master_enable']:
            return False
        if self.count == 0:
            print("fault_enabled")
            self.E = self.mul_fault_matrix
            self.u_a = self.add_fault_matrix
            self.count+=1
        return True

class FaultModule():
    faults : list[Fault] = None
    config : dict = None

    f_wheels = None
    E = None
    u_a = None
    def __init__(self, config):
        self.config = config
        self.faults = []

    def init(self, num_wheels):

        for i, fault_config in enumerate(self.config['fault']):
            fault = Fault(self.config, i)
            fault.init(num_wheels)
            self.faults.append(fault)
        self.f_wheels = np.zeros(num_wheels)
        self.E = np.eye(num_wheels)
        self.u_a = np.zeros(num_wheels)

    def update(self, t):
        if not self.config['faults']['master_enable']:
            return
        self.f_wheels = np.zeros(len(self.f_wheels))
        self.u_a = np.zeros(len(self.u_a))
        self.E = np.eye(len(self.E))
        # self.E = np.eye(len(self.E))
        for fault in self.faults:
            fault.update(t)
            # self.f_wheels += fault.f
            self.E @= fault.E
            self.u_a += fault.u_a
