import pytest
import numpy as np
import my_utils as my_utils
from scipy.spatial.transform import Rotation
import body
import toml

config = toml.load('config.toml')

def test_torque_limiter():
    controller = body.Controller(config=config)
    limit = 0.5
    M_output = controller.limit_torque([0,0,1], 0.5)
    print(M_output)
    assert(all(i <= limit for i in M_output))


def test_wheel_M_inertia():
    # wheel = body.Wheel(0.5, 1, 0.5)
    wheel = body.Wheel(config=config)
    wheel.calc_M_inertia([0,90,0])
    cmp1 = wheel.M_inertia.round(3)
    cmp2 = my_utils.rotate_M_inertia(np.diag([0.13542, 0.13542, 0.25]), Rotation.from_euler('xyz',[0,90,0],degrees=True)).round(3)
    assert(np.all(cmp1==cmp2))

def test_calc_M_inertia_point_mass():
    cmp1 = my_utils.calc_M_inertia_point_mass([0,0,4], 0.5)
    cmp2 = np.diag([8.0,8.0,0.0])
    print(cmp1)
    print(cmp2)
    assert(np.all(cmp1==cmp2))

def test_wheel_max_torque():
    wheel = body.Wheel(config=config)
    # for i in range(100):
    print(wheel.calc_state_rates(1000))

def test_wheel_module_max_torque():
    wheels = body.WheelModule(config=config)
    for i in range(3):
        print(np.sum(np.abs(wheels.D[i]*config['wheels']['max_torque'])))
    # for i in range(100):
    # print(wheels.calc_state_rates(1000,0.1))


if __name__ == "__main__":
    # test_torque_limiter()
    # test_wheel_M_inertia()
    # test_calc_M_inertia_point_mass()
    # test_wheel_max_torque()
    test_wheel_module_max_torque()