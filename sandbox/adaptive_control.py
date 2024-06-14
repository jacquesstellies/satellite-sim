import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

results_data = {}

class Simulation:
    mass=12
    radius=0.7
    height=0.2
    M_inertia = 0.5*mass*pow(radius,2)
    t_step = 0.1
    pos_ref = np.pi

    def calc_controller_torque(self, pos, speed):
        k = 3
        c = 5
        e = pos - self.pos_ref
        # print(e)
        torque = (- k*e - c*speed)*3
        return torque

    t_prev = 0
    T_com = 0
    def calc_system_output(self, t, y):
        pos = y[0]
        speed = y[1]
        
        if t > self.t_prev:
            self.T_com = self.calc_controller_torque(pos, speed)
            self.T_com = self.calc_adaptive_control_torque_output(self.T_com, pos)
            self.t_prev = t + self.t_step
            results_data['time'].append(t)

        acc = self.T_com/self.M_inertia
        return speed, acc

    speed_prev = 0
    position_prev = 0
    def calc_adaptive_model_output(self):
        T_com = self.calc_controller_torque(self.position_prev, self.speed_prev)
        speed_result = self.speed_prev + (T_com/self.M_inertia)*self.t_step
        self.speed_prev = speed_result
        position = self.position_prev + speed_result*self.t_step
        self.position_prev = position
        return position

    theta_prev = 0
    adaptive_gain = 1
    def calc_adaptive_control_torque_output(self, T_com, y_meas):
        y_model = self.calc_adaptive_model_output()
        
        error = y_meas - y_model
        # print(f"y_meas {y_meas}")
        # print(f"y_model {y_model}")
        # print(f"error {error}")
        
        theta = self.theta_prev + error*self.adaptive_gain*y_model*error
        # print(f"theta {theta}")
        self.theta_prev = theta
        results_data['theta'].append(theta)
        # theta=1
        return T_com*theta + T_com

#################################### main #####################################################

results_data['theta'] = []
results_data['time'] = []
sim_time = 30
res = 0.1
t_eval_sim = np.linspace(0, sim_time, int(sim_time/res))

sim = Simulation()
sol = solve_ivp(sim.calc_system_output, t_span=[0,sim_time], y0=[0, 0], method="RK45", t_eval=t_eval_sim, max_step = 0.1)



fig = plt.figure(figsize=(10,5))


# for key, data in results_data.items():
#     if key == 'time':
#         continue
#     if len(data) > int(sim_time/res):
#         results_data[key] = np.interp(t_eval_sim, results_data['time'], data)[:]
plt.subplot(1,3,1)

plt.plot(sol.t, sol.y[0])
plt.xlabel('time (s)')
plt.ylabel('position')

plt.subplot(1,3,2)

plt.plot(sol.t, sol.y[1])
plt.xlabel('time (s)')
plt.ylabel('speed')

plt.subplot(1,3,3)

plt.plot(results_data['time'], results_data['theta'])
plt.xlabel('time (s)')
plt.ylabel('theta')

plt.subplots_adjust(wspace=1, hspace=0.2)


plt.show()