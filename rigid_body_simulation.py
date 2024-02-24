import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# all units are in SI (m, s, N, kg.. etc)

class Body:
    euler_angles = np.zeros(3)
    angular_rates = np.zeros(3)
    angular_acc = np.zeros(3)

    mass = 10
    dimensions = [2.0, 3.5, 1.0] # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))

    def __init__(self) -> None:
        self.calc_M_inertia()
        print(f"M_inertia = {self.M_inertia}")

    def calc_M_inertia(self) -> None:
        # use cuboid for mass moment inertia
        self.M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions[1],2)+pow(self.dimensions[2],2))
        self.M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions[0],2)+pow(self.dimensions[2],2))
        self.M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions[0],2)+pow(self.dimensions[1],2))

        self.M_inertia_inv = np.linalg.inv(self.M_inertia)
    
    def calc_state(self, t, y):
        angle_initial = y[:3]
        angular_rate_initial = y[3:6]

        angular_rate_result = [0]*3
        angular_acc_result = [0]*3

        for axis in range(3):
            angular_rate_result[axis] = angular_rate_initial[axis]

        Hnet = self.M_inertia@(angular_rate_initial)
        angular_acc_result = torque_applied - self.M_inertia_inv@(-1*np.cross(angular_rate_initial,Hnet))
        return np.hstack([angular_rate_result, angular_acc_result])
    
torque_applied = [5, 0 , 0]
time_applied = 2 # time the force is applied

point = Body()

sim_time = 30
h = 1
point.angular_rates[0] = 4
point.angular_rates[1] = 0.1
point.angular_rates[2] = 0.1
initial_values = np.hstack([point.euler_angles, point.angular_rates]) 

sol = solve_ivp(fun=point.calc_state, t_span=[0, sim_time], y0=initial_values, method="RK45", t_eval=range(sim_time))

fig = plt.figure(figsize=(13,6))
fig.tight_layout()
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.plot(sol.t, sol.y[i])
    plt.xlabel('time (s)')
    plt.ylabel('euler angle (rad)')

    plt.subplot(2,3,i+4)
    plt.plot(sol.t, sol.y[i+3]) 
    plt.xlabel('time (s)')
    plt.ylabel('angular rate (rad/s)')

plt.subplots_adjust(wspace=1, hspace=0.2)
plt.show()