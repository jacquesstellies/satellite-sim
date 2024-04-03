import rigid_body_simulation as rbs
import pytest

controller = rbs.Controller()
# sat = rbs.Satellite()


def test_torque_limiter():
    limit = 0.5
    M_output = controller.limit_torque([0,0,1], 0.5)
    print(M_output)
    assert(all(i <= limit for i in M_output))

# def test_calc():
