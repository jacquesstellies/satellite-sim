import pytest
import numpy as np
import src.my_utils as my_utils
from scipy.spatial.transform import Rotation
import src.body

controller = src.body.Controller()

def test_torque_limiter():
    limit = 0.5
    M_output = controller.limit_torque([0,0,1], 0.5)
    print(M_output)
    assert(all(i <= limit for i in M_output))


def test_wheel_M_inertia():
    wheel = src.body.Wheel(0.5, 1, 0.5)
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

