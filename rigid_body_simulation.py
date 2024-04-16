import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
import quaternion
import my_utils
import logging
# all units are in SI (m, s, N, kg.. etc)

logger = logging.getLogger(__name__)
class Body:
    dir_init_inertial = Rotation.from_quat([0,0,0,1])

    mass = 0
    dimensions = {'x':0, 'y':0, 'z':0} # x, y and z dimensions
    M_inertia : np.ndarray = np.zeros((3,3))
    M_inertia_inv : np.ndarray = np.zeros((3,3))
    
    def calc_M_inertia_inv(self):
        self.M_inertia_inv = np.linalg.inv(self.M_inertia)

class Wheel(Body):
    dimensions = {'radius': 0, 'height': 0}
    position = np.zeros(3)
    angular_v = np.zeros(3)
    speed = 0
    shaft_axis_mask = None

    speed_min = 0.1
    def __init__(self, mass, radius, height, dir_init_inertial=None):
        self.mass = mass
        self.dimensions['radius'] = radius
        self.dimensions['height'] = height
        if dir_init_inertial is None:
            self.dir_init_inertial = Rotation.from_quat([0,0,0,1])
    
    def calc_M_inertia(self, direction):
        # Moment of inertia
        self.M_inertia[0][0] = 0.25*self.mass*pow(self.dimensions['radius'],2) + (1/12)*self.mass*pow(self.dimensions['height'],2)
        self.M_inertia[1][1] = self.M_inertia[0][0]
        self.M_inertia[2][2] = 0.5*self.mass*pow(self.dimensions['radius'],2)
        self.dir_init_inertial = Rotation.from_euler('xyz', direction, degrees=True) 
        self.M_inertia = my_utils.rotate_M_inertia(self.M_inertia, self.dir_init_inertial) # @TBD look at floating point error here
        self.M_inertia *= np.eye(1)
        self.calc_M_inertia_inv()
    
    # returns angular momentum
    def calc_angular_momentum(self) -> np.array:
        angular_momentum = self.M_inertia@self.angular_v
        # print(f"angular v {self.angular_v}")
        if np.isnan(angular_momentum[0]):
            raise("Nan error")
        return angular_momentum*self.shaft_axis_mask
    
    def update_angular_velocity(self, angular_v: np.array):
        self.angular_v = angular_v*self.shaft_axis_mask
        self.speed = my_utils.magnitude(angular_v)
        # if self.speed < self.speed_min:
        #     self.speed = 0
        #     self.angular_v = np.zeros(3)


    def update_speed(self, speed):
        self.speed = speed
        self.angular_v = speed
        
    
class WheelModule():
    H = 0

class Controller:
    type = "torque"
    types = ["torque", "reaction_wheel"]
    M_max = 0
    M_min = 0
    time_step = np.longdouble(0.1)
    filter_coef = 0

    def __init__(self, M_max = None, M_min = 0, type = "torque", filter_coef = 0.5, time_step=0.1):
        self.type = type
        self.M_max = M_max
        self.M_min = M_min
        if filter_coef <= 1 or filter_coef >= 0:
            self.filter_coef = filter_coef
        else:
            raise Exception("filter coef must be between 0 and 1")
        self.time_step = np.longdouble(time_step)

    M_output_prev = 0
    def calc_torque_control_output(self, curr_q : np.quaternion,  angular_v : list[float], ref_q : np.quaternion) -> np.array:
        k = 1
        c1 = 1
        c2 = 1
        c3 = 1
        K = np.diag(np.full(3,k))
        C = np.diag([c1, c2, c3])

        angular_v = np.array(angular_v)
        
        q_error = curr_q.inverse() * my_utils.conv_scipy_to_numpy_q(ref_q) # ref_q is already normalized

        q_error_vec = np.array([q_error.x, q_error.y, q_error.z])

        M_output = + K@q_error_vec - C@angular_v

        M_output = self.limit_torque(M_output)
        
        M_output = self.low_pass_filter(M_output, self.M_output_prev)

        return M_output
    
    def limit_torque(self, M):
        for i in range(3):
            if (self.M_max is not None) and (np.abs(M[i]) >= self.M_max):
                M[i] = np.sign(M[i])*self.M_max
            if np.abs(M[i]) <= self.M_min:
                M[i] = 0
        return M

    def low_pass_filter(self, value, value_prev):
        return (self.filter_coef)*value_prev + (1 - self.filter_coef)*value

    def calc_wheel_control_output(self, torque, angular_momentum) -> list[float]:
        ref_angular_momentum = torque*self.control_time_step + angular_momentum
        return ref_angular_momentum

class Satellite(Body):

    logger = None
    wheels = None # array storing wheel objects associated with the satellite
    wheel_offset = 0 # offset of wheel center of mass from edge of device
    
    controller : Controller = None
    num_wheels = 0

    def __init__(self, wheels : list[Wheel], controller : Controller, wheel_offset = 0, wheel_config = "standard", wheel_axes = 'x', logger = None):
        if wheels is not None:
            self.wheels = wheels
            self.num_wheels = len(self.wheels)
            if wheel_config == "standard":
                if len(wheels) == 3:
                    self.wheels[0].position[0] = self.dimensions['x']-wheel_offset
                    self.wheels[1].position[1] = self.dimensions['y']-wheel_offset
                    self.wheels[2].position[2] = self.dimensions['z']-wheel_offset

                    self.wheels[0].calc_M_inertia([0,90,0])
                    self.wheels[1].calc_M_inertia([90,0,0])
                    self.wheels[2].calc_M_inertia([0,0,0])

                    for idx,wheel in enumerate(wheels):
                        print(f"wheel[{idx}] \n {wheel.M_inertia} \n {wheel.M_inertia_inv}")

                    self.wheels[0].shaft_axis_mask = [1,0,0]
                    self.wheels[1].shaft_axis_mask = [0,1,0]
                    self.wheels[2].shaft_axis_mask = [0,0,1]
                elif len(wheels) == 1:
                    self.wheels[0].position[0] = self.dimensions[wheel_axes]-wheel_offset
                else:
                    raise(Exception("error unable to set up wheel configuration"))

        self.controller = controller
        self._next_control_time_step = controller.time_step
        self.logger = logger

    def calc_M_inertia_body(self):

        M_inertia = np.zeros((3,3))
        # use cuboid for mass moment inertia
        M_inertia[0][0] = 1/12*self.mass*(pow(self.dimensions['y'],2)+pow(self.dimensions['z'],2))
        M_inertia[1][1] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['z'],2))
        M_inertia[2][2] = 1/12*self.mass*(pow(self.dimensions['x'],2)+pow(self.dimensions['y'],2))
        return M_inertia

    def calc_M_inertia_peri(self):
        M_inertia = np.zeros((3,3))
        if self.wheels is not None:
            M_inertia_indv_wheels = 0
            M_inertia_point_mass_wheels = 0

            for wheel in self.wheels:
                M_inertia_indv_wheels += wheel.M_inertia
                M_inertia_point_mass_wheels += my_utils.calc_M_inertia_point_mass(wheel.position, self.mass)

        return M_inertia

    def calc_M_inertia(self):
        self.M_inertia = self.calc_M_inertia_body() + self.calc_M_inertia_peri()

        self.calc_M_inertia_inv()

    ref_q = Rotation.from_quat([0,0,0,1])
    controller_enable = True
    wheels_control_enable = True
    _next_control_time_step = np.longdouble(0.1)
    wheels_H_new = np.zeros(3)
    M_controller = np.zeros(3)
    def calc_state(self, t, y):
        angular_v_input = y[:3]
        quaternion_input = np.quaternion(y[6],y[3],y[4],y[5]).normalized()
        
        angular_acc_result = [0]*3
        quaternion_rate_result = np.quaternion(1,0,0,0)
        wheels_H_curr = np.zeros(3)
        wheels_H_delta = np.zeros(3)
        # print(self._next_control_time_step)
        # print(t)
        
        if t > self._next_control_time_step:
            if self.controller_enable:
                self.M_controller = self.controller.calc_torque_control_output(quaternion_input, angular_v_input, self.ref_q)
                # print(f"M_controller {self.M_controller}")
                if self.wheels_control_enable:
                    wheels_H_delta = self.M_controller * controller.time_step
                    self._next_control_time_step += controller.time_step
                    # print(f"entered controller time space {t}")
                    # logging.info(f"entered controller time space {t}")
                

                    if self.wheels is not None:
                        for wheel in self.wheels:
                            wheels_H_curr += wheel.calc_angular_momentum()

                    self.wheels_H_new = wheels_H_delta + wheels_H_curr

                    # print(f"wheels_H_delta {wheels_H_delta} wheels_H_curr {wheels_H_curr}")
                    
                    # print(f"wheels_H_new {self.wheels_H_new}")
        
                    for wheel in wheels:
                        wheel.update_angular_velocity(wheel.M_inertia_inv@(self.wheels_H_new*wheel.shaft_axis_mask))
                        # print(f"wheel angular v {wheel.angular_v}")
                        # print(f"wheel speed {wheel.speed}")
                    

        Hnet = self.M_inertia@(angular_v_input) + self.wheels_H_new
        angular_acc_result = self.M_inertia_inv@(self.M_controller - np.cross(angular_v_input,Hnet))

        # put the inertial velocity in quaternion form
        inertial_v_quaternion = np.quaternion(0, angular_v_input[0], angular_v_input[1], angular_v_input[2])

        quaternion_rate_result = 0.5*quaternion_input*inertial_v_quaternion
        quaternion_rate_result = [quaternion_rate_result.x, quaternion_rate_result.y, quaternion_rate_result.z, quaternion_rate_result.w]
        return np.hstack([angular_acc_result, quaternion_rate_result, self.M_controller, [wheel.speed for wheel in self.wheels] ])
        
# END DEF class Satellite()

def calc_yaw_pitch_roll_rates(data_in):
    
    inertial_rates = data_in[:3]
    
    r = Rotation.from_quat([data_in[3], data_in[4], data_in[5], data_in[6]])

    yaw, pitch, roll = r.as_euler('zyx')
    
    yaw_rate = 1/np.cos(pitch)*(inertial_rates[1]*np.sin(roll)+inertial_rates[2]*np.cos(roll))
    pitch_rate = inertial_rates[1]*np.cos(roll)-inertial_rates[2]*np.sin(roll)
    roll_rate = inertial_rates[0]+inertial_rates[1]*np.tan(pitch)*np.sin(roll)+inertial_rates[2]*np.tan(pitch)*np.cos(roll)

    return [yaw, pitch, roll, yaw_rate, pitch_rate, roll_rate]

def init_reaction_wheels(mass, radius, height):
    wheel_x = Wheel(mass, radius, height)
    wheel_y = Wheel(mass, radius, height)
    wheel_z = Wheel(mass, radius, height)
    return [wheel_x, wheel_y, wheel_z]

if __name__ == '__main__':
    # logging.basicConfig(filename="logs.log", filemode="w+", level=logging.INFO)
    # logger.info("satellite simulation logs")
    data_logs = open(file="data_logs.log",mode="w+")
    data_logs.write("")

    controller = Controller(M_max=0.2, M_min=0.001, filter_coef=0.1, time_step=0.1)
    wheels = init_reaction_wheels(mass=0.5, radius=0.03, height=0.01)

    satellite = Satellite(wheels=wheels, controller=controller)

    # Satellite Properties
    satellite.mass = 12 # 6U Sat weight limit
    satellite.dimensions = {'x': 0.2, 'y': 0.1, 'z': 0.3405} # 6U Sat dimension limit
    satellite.calc_M_inertia()

    # Satellite Initial Conditions
    satellite_angular_v = np.array([0,0,0])
    satellite.dir_init_inertial = Rotation.from_quat([0,0,0,1])

    M_init = [0, 0, 0]
    time_applied = 2 # time the moment is applied

    # Control Variables
    satellite.ref_q = Rotation.from_euler("xyz", [0, 45, 0], degrees=True)
    satellite.controller_enable = True
    satellite.wheels_control_enable = True

    quaternion_init = satellite.dir_init_inertial.as_quat()
    initial_values = np.hstack([satellite_angular_v, quaternion_init, M_init, np.zeros(3)])

    # Simulation parameters
    sim_time = 10
    sim_output_resolution_time = 1

    # Integrate satellite dynamics over time
    sol = solve_ivp(fun=satellite.calc_state, t_span=[0, sim_time], y0=initial_values, method="RK45", 
                    t_eval=range(0, sim_time, sim_output_resolution_time),
                    max_step=0.01)

    fig = plt.figure(figsize=(13,6))
    fig.tight_layout()

    angular_rate = sol.y[:3]
    sat_quaternion = sol.y[3:7]
    controller_torque_output = np.diff(sol.y[7:10], prepend=0)/np.diff(sol.t, prepend=0)
    wheels_speeds = np.diff(sol.y[10:13], prepend=0)/np.diff(sol.t, prepend=0)

    yaw_pitch_roll_output = False

    if( not yaw_pitch_roll_output):
        cols = 4
        rows = 4
        axis = ''
        for i in range(4):
            if i == 0:
                axis = 'x'
            elif i == 1:
                axis = 'y'
            elif i == 2:
                axis = 'z'
            elif i == 3:
                axis = 'w'
            if i < 3:
                plt.subplot(rows,cols,i+1)
                plt.plot(sol.t, angular_rate[i]) 
                plt.xlabel('time (s)')
                plt.ylabel(f'{axis} angular rate (rad/s)')

            plt.subplot(rows,cols,i+5)
            plt.plot(sol.t, sat_quaternion[i]) 
            plt.xlabel('time (s)')
            plt.ylabel(f'quaternion {axis}')
        
    else:
        cols = 3
        rows = 3
        # Convert data to yaw pitch and roll
        y_transpose = [list(x) for x in zip(*sol.y)] # transpose columns and rows
        yaw_pitch_roll_values = list(map(calc_yaw_pitch_roll_rates,y_transpose))
        yaw_pitch_roll_values = [list(x) for x in zip(*yaw_pitch_roll_values)]

        for i in range(cols):
            if(i==0):
                title = 'yaw'
            elif(i==1):
                title = 'pitch'
            elif(i==2):
                title = 'roll'
            plt.subplot(rows,cols,i+1)
            plt.plot(sol.t, yaw_pitch_roll_values[i])
            plt.xlabel('time (s)')
            plt.ylabel(f'{title} angle (rad)')

            plt.subplot(rows,cols,i+4)
            plt.plot(sol.t, yaw_pitch_roll_values[i+3]) 
            plt.xlabel('time (s)')
            plt.ylabel(f'{title} angular rate (rad/s)')
    
    for i in range(cols):
        if i == 0:
            axis = 'x'
        elif i == 1:
            axis = 'y'
        elif i == 2:
            axis ='z'
        if i < 3:
            pos = i+cols*2+1
            plt.subplot(rows,cols,pos)
            plt.plot(sol.t, controller_torque_output[i])
            plt.xlabel('time (s)')
            plt.ylabel(f'torque {axis} (N)')

            pos_2 = i+cols*3+1
            plt.subplot(rows,cols,pos_2)
            plt.plot(sol.t, wheels_speeds[i]*60/(2*math.pi)) 
            plt.xlabel('time (s)')
            plt.ylabel(f'wheel speed {axis}')

    plt.subplots_adjust(wspace=1, hspace=0.2)
    plt.show()