import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
import quaternion
# all units are in SI (m, s, N, kg.. etc)

class Body:
    angular_v_init = np.zeros(3) 
    q_init = Rotation.from_quat([0,0,0,1])

    mass = 0
    dimensions = np.zeros(3) # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))

    def calc_M_inertia(self) -> None:
        # use cuboid for mass moment inertia
        self.M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions[1],2)+pow(self.dimensions[2],2))
        self.M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions[0],2)+pow(self.dimensions[2],2))
        self.M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions[0],2)+pow(self.dimensions[1],2))

        self.M_inertia_inv = np.linalg.inv(self.M_inertia)
    
    ref_q = Rotation.from_quat([0,0,0,1])

    def conv_scipy_to_numpy_q(self, q : Rotation):
        q_result = q.as_quat()
        return np.quaternion(q_result[3], q_result[0], q_result[1], q_result[2])
    
    def conv_numpy_to_scipy_q(self, q : np.quaternion):
        return Rotation.from_quat(q.x, q.y, q.z, q.w)

    controller_enable = True
    def calc_state(self, t, y):
        angular_v_input = y[:3]
        quaternion_input = np.quaternion(y[3],y[4],y[5],y[6]).normalized()
        
        angular_acc_result = [0]*3
        quaternion_rate_result = np.quaternion(1,0,0,0)

        current_M_applied = [0, 0, 0]
        if t < time_applied:
            current_M_applied = M_applied

        if self.controller_enable:
            M_controller = self.calc_control_output(quaternion_input, angular_v_input)
        else:
            M_controller = 0

        Hnet = self.M_inertia@(angular_v_input)
        angular_acc_result = np.array(current_M_applied) + M_controller - self.M_inertia_inv@(-1*np.cross(angular_v_input,Hnet))

        # put the inertial velocity in quaternion form
        inertial_v_quaternion = np.quaternion(0, angular_v_input[0], angular_v_input[1], angular_v_input[2])

        quaternion_rate_result = 0.5*quaternion_input*inertial_v_quaternion
        return np.hstack([angular_acc_result, quaternion_rate_result.w, quaternion_rate_result.x, quaternion_rate_result.y, quaternion_rate_result.z])

    def calc_control_output(self, q : np.quaternion,  angular_v : list[float]) -> np.array:
        k = 10
        c1 = 10
        c2 = 10
        c3 = 10
        K = np.diag(np.full(3,k))
        C = np.diag([c1, c2, c3])

        angular_v = np.array(angular_v)
        
        q_error = q.inverse() * self.conv_scipy_to_numpy_q(self.ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        M_controller = +K@q_error_vec - C@angular_v
        return M_controller
    

def calc_yaw_pitch_roll_rates(data_in):
    
    inertial_rates = data_in[:3]
    
    r = Rotation.from_quat([data_in[4], data_in[5], data_in[6], data_in[3]])

    yaw, pitch, roll = r.as_euler('zyx')
    
    yaw_rate = 1/np.cos(pitch)*(inertial_rates[1]*np.sin(roll)+inertial_rates[2]*np.cos(roll))
    pitch_rate = inertial_rates[1]*np.cos(roll)-inertial_rates[2]*np.sin(roll)
    roll_rate = inertial_rates[0]+inertial_rates[1]*np.tan(pitch)*np.sin(roll)+inertial_rates[2]*np.tan(pitch)*np.cos(roll)

    return [yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate]



satellite = Body()

# Satellite Properties
satellite.mass = 12 # 6U Sat weight limit
satellite.dimensions = [0.2, 0.1, 0.3405] # 6U Sat dimension limit
satellite.calc_M_inertia()

# Satellite Initial Conditions
satellite.angular_v_init[0] = 0
satellite.angular_v_init[1] = 0
satellite.angular_v_init[2] = 0
satellite.q_init = Rotation.from_quat([0,0,0,1]) 

M_applied = [0, 0, 0]
time_applied = 2 # time the force is applied

# Control Variables
satellite.ref_q = Rotation.from_euler("zyx", [0, 45, 0], degrees=True)
print(satellite.ref_q.as_euler('xyz',degrees=True))
satellite.controller_enable = True

quaternion_init = satellite.rotation_dir_init.as_quat()
initial_values = np.hstack([satellite.angular_v_init, quaternion_init[3], quaternion_init[0], quaternion_init[1], quaternion_init[2]]) 

# Simulation parameters
sim_time = 15
sim_output_resolution_time = 1

# Integrate satellite dynamics over time
sol = solve_ivp(fun=satellite.calc_state, t_span=[0, sim_time], y0=initial_values, method="RK45", t_eval=range(0, sim_time, sim_output_resolution_time))

fig = plt.figure(figsize=(13,6))
fig.tight_layout()

yaw_pitch_roll_output = True

if( not yaw_pitch_roll_output):

    for i in range(4):
        if i < 3:
            plt.subplot(2,4,i+1)
            plt.plot(sol.t, sol.y[i]) 
            plt.xlabel('time (s)')
            plt.ylabel(f'inertial angular rate (rad/s)')

        plt.subplot(2,4,i+5)
        plt.plot(sol.t, sol.y[i+3]) 
        plt.xlabel('time (s)')
        plt.ylabel(f'quaternion')
    
else:
    # Convert data to yaw pitch and roll
    y_transpose = [list(x) for x in zip(*sol.y)] # transpose columns and rows
    yaw_pitch_roll_values = list(map(calc_yaw_pitch_roll_rates,y_transpose))
    yaw_pitch_roll_values = [list(x) for x in zip(*yaw_pitch_roll_values)]

    for i in range(3):
        if(i==0):
            title = 'yaw'
        elif(i==1):
            title = 'pitch'
        elif(i==2):
            title = 'roll'
        plt.subplot(2,3,i+1)
        plt.plot(sol.t, yaw_pitch_roll_values[i])
        plt.xlabel('time (s)')
        plt.ylabel(f'{title} angle (rad)')

        plt.subplot(2,3,i+4)
        plt.plot(sol.t, yaw_pitch_roll_values[i+3]) 
        plt.xlabel('time (s)')
        plt.ylabel(f'{title} angular rate (rad/s)')



plt.subplots_adjust(wspace=1, hspace=0.2)
plt.show()