import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def particle_position(t, y):
    return [y[1], -(398600)/pow(y[0],2)]

t_end = 70*60
t_step = int(t_end/20)
t_range = range(0, t_end+t_step, t_step)

# if __name__ == "main":
sol = solve_ivp(fun=particle_position, t_span=[0, t_end], y0=[6500, 7.8], method="RK45", t_eval=t_range)
print(sol)
print(sol.y[0])
print(sol.y[1])
v = np.round(sol.y[1], 3)
# print(sol.y[0][1]/1e+3)
# print(np.transpose(sol.y))
plt.figure()

plt.subplot(2,1,1)
plt.plot(sol.t, sol.y[0]/1e3)
plt.xlabel('time (s)')
plt.ylabel('position (km)')

plt.subplot(2,1,2)
plt.plot(sol.t, v/1e3) 
plt.xlabel('time (s)')
plt.ylabel('speed (km/s)')

plt.subplots_adjust(wspace=0.5)
plt.show()
