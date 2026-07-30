"""Route planning for the underactuated (2-wheel, z unactuated) satellite.

Geometric-phase yaw primitive
-----------------------------
Yaw (rotation about the unactuated z-axis) cannot be slewed to directly, but a
roll/pitch "box"  Rx(a) Ry(a) Rx(-a) Ry(-a)  ~=  Rz(a^2)  nets yaw via the Lie
bracket [x,y] = z. Each segment of the box is a single body-axis rotation, so it
is feasible for the controller (only one of wx, wy nonzero, wz = 0 demanded); the
*endpoint* carries the yaw. Emitting the box corners as `tracking`-mode waypoints
lets the satellite reach a pure-yaw target it cannot slew to directly.

Repeat N small boxes to accumulate larger yaw while keeping each tilt small
(better accuracy, smaller excursions). A 1-D solve sizes the tilt so the net yaw
matches the target.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import brentq


def _box(a):
    """One roll/pitch box as a scipy Rotation; ~= Rz(a^2) for small a."""
    Rx, Ry = R.from_euler("x", a), R.from_euler("y", a)
    return Rx * Ry * Rx.inv() * Ry.inv()


def _box_yaw(a):
    """Net yaw (rad, z-component of the rotation vector) produced by one box."""
    return _box(a).as_rotvec()[2]


def solve_box_angle(delta_psi):
    """Tilt angle a (=b, rad) so a single box yields exactly `delta_psi` yaw (rad)."""
    return brentq(lambda a: _box_yaw(a) - delta_psi, 1e-4, 1.0)


def yaw_maneuver_waypoints(total_yaw_deg, n_boxes=3):
    """Waypoints that net `total_yaw_deg` of yaw from identity, feasibly.

    Returns (q_series, info) where q_series is a list of [x, y, z, w] quaternions
    ready for the config's `ref_q_series`, and info is a dict of diagnostics.
    """
    dpsi = np.radians(total_yaw_deg)
    a = solve_box_angle(dpsi / n_boxes)
    Rx, Ry = R.from_euler("x", a), R.from_euler("y", a)

    wps = [R.identity()]
    base = R.identity()
    for _ in range(n_boxes):
        c1 = base * Rx
        c2 = c1 * Ry
        c3 = c2 * Rx.inv()
        c4 = c3 * Ry.inv()          # ~= base * Rz(a^2)
        wps += [c1, c2, c3, c4]
        base = c4                    # chain from the actual (not idealized) corner

    q_series = [[float(x) for x in w.as_quat()] for w in wps]   # scipy: [x, y, z, w]
    final_euler = np.degrees(wps[-1].as_euler("xyz"))
    info = {
        "tilt_deg": float(np.degrees(a)),
        "n_waypoints": len(wps),
        "final_euler_xyz_deg": [float(v) for v in final_euler],
        "achieved_yaw_deg": float(final_euler[2]),
    }
    return q_series, info


if __name__ == "__main__":
    q, info = yaw_maneuver_waypoints(20.0, n_boxes=3)
    print(info)
    print("ref_q_series =", q)
    times = np.linspace(0, 100, len(q))
    print("ref_t_series =", [float(t) for t in times])
