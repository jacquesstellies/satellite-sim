import numpy as np

class Body:

    mass = 0
    dimensions = {'x':0, 'y':0, 'z':0} # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))
    
    def calc_M_inertia_inv(self):
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)