import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# all units are in SI (m, s, N, kg.. etc)

class Body:
    position = np.zeros(3)
    velocity = np.zeros(3)
    acceleration = np.zeros(3)

# expects array of dimension 6 with the structure:
    # position in y[:3] and velocity in y[3:6]

def calc_state(t, y):
    
    result = [0] * len(y)
    
    for axis in range(3):
        # position update
        result[axis] = y[axis+3]
        # velocity update
        result[axis+3] = force_applied[axis]/mass
    
    return result
    
force_applied = [5, 3, 0]
mass = 1
time_applied = 2 # time the force is applied

point = Body()

sim_time = 10
time_step = 1
point.velocity[2] = 4
_y0 = np.hstack([point.position, point.velocity])
print(_y0)
sol = solve_ivp(fun=calc_state, t_span=[0, sim_time], y0=_y0, method="RK45", t_eval=range(0, sim_time), vectorized=False)

plt.figure()
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.plot(sol.t, sol.y[i])
    plt.xlabel('time (s)')
    plt.ylabel('position (m)')

    plt.subplot(2,3,i+4)
    plt.plot(sol.t, sol.y[i+3]) 
    plt.xlabel('time (s)')
    plt.ylabel('speed (m/s)')


plt.subplots_adjust(wspace=0.5)
plt.show()

axes_3D = plt.axes(projection='3d')
axes_3D.plot3D(sol.y[0], sol.y[1], sol.y[2])
plt.show()