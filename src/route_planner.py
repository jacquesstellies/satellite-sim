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
matches the target. Reversing the box order (Ry Rx Ry^-1 Rx^-1 ~= Rz(-a^2), since
[y,x] = -[x,y]) gives negative yaw with the same tilt magnitude, so sign is handled
by leg order rather than by signing `a` itself.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import brentq


def _box(a, reverse=False):
    """One roll/pitch box as a scipy Rotation; ~= Rz(a^2) for small a (Rz(-a^2) if reverse)."""
    Rx, Ry = R.from_euler("x", a), R.from_euler("y", a)
    if reverse:
        return Ry * Rx * Ry.inv() * Rx.inv()
    return Rx * Ry * Rx.inv() * Ry.inv()


def _box_yaw_mag(a):
    """Net yaw magnitude (rad, > 0 for a in (0,1]) produced by one non-reversed box."""
    return _box(a, reverse=False).as_rotvec()[2]


def solve_box_angle(delta_psi_mag):
    """Tilt angle a (rad, > 0) so one box yields yaw magnitude `delta_psi_mag` (rad, > 0).

    Sign of the maneuver is applied separately via `_box(a, reverse=...)` - solving on the
    always-positive `_box_yaw_mag` avoids handing brentq a bracket with no sign change.
    """
    return brentq(lambda a: _box_yaw_mag(a) - delta_psi_mag, 1e-4, 1.0)


def yaw_maneuver_waypoints(total_yaw_deg, n_boxes=None, max_tilt_deg=20.0):
    """Waypoints that net `total_yaw_deg` of yaw (either sign) from identity, feasibly.

    n_boxes: fixed box count if given. If None (default), the smallest box count that
    keeps each box's tilt at or under `max_tilt_deg` is chosen automatically (more boxes
    for a larger total yaw ask) - large single-box tilts (e.g. ~44 deg for an 85 deg yaw
    at n_boxes=3) mean large, aggressive roll/pitch excursions per leg, which is exactly
    the kind of demand that risks saturating/winding up the actuated-axis controller this
    maneuver exists to avoid.

    Returns (q_series, info) where q_series is a list of [x, y, z, w] quaternions
    ready for the config's `ref_q_series` (relative to identity - compose onto the
    satellite's attitude at maneuver-start to get absolute targets), and info is a
    dict of diagnostics.
    """
    reverse = total_yaw_deg < 0
    dpsi = np.radians(abs(total_yaw_deg))

    if n_boxes is None:
        n_boxes = 1
        while True:
            try:
                a = solve_box_angle(dpsi / n_boxes)
                if np.degrees(a) <= max_tilt_deg or n_boxes >= 200:
                    break
            except ValueError:
                pass  # dpsi/n_boxes outside what a single box can net - need more boxes
            n_boxes += 1
    else:
        a = solve_box_angle(dpsi / n_boxes)

    Rx, Ry = R.from_euler("x", a), R.from_euler("y", a)

    wps = [R.identity()]
    base = R.identity()
    for _ in range(n_boxes):
        if reverse:
            c1 = base * Ry
            c2 = c1 * Rx
            c3 = c2 * Ry.inv()
            c4 = c3 * Rx.inv()          # ~= base * Rz(-a^2)
        else:
            c1 = base * Rx
            c2 = c1 * Ry
            c3 = c2 * Rx.inv()
            c4 = c3 * Ry.inv()          # ~= base * Rz(a^2)
        wps += [c1, c2, c3, c4]
        base = c4                        # chain from the actual (not idealized) corner

    q_series = [[float(x) for x in w.as_quat()] for w in wps]   # scipy: [x, y, z, w]
    final_euler = np.degrees(wps[-1].as_euler("xyz"))
    info = {
        "tilt_deg": float(np.degrees(a)),
        "n_boxes": n_boxes,
        "n_waypoints": len(wps),
        "final_euler_xyz_deg": [float(v) for v in final_euler],
        "achieved_yaw_deg": float(final_euler[2]),
    }
    return q_series, info


if __name__ == "__main__":
    # for target in (20.0, -20.0, 85.0, -85.0):
    #     q, info = yaw_maneuver_waypoints(target)
    #     print(target, "->", info)
    q, info = yaw_maneuver_waypoints(20.0, n_boxes=3)
    print(info)
    print("ref_q_series =", q)
    times = np.linspace(0, 100, len(q))
    print("ref_t_series =", [float(t) for t in times])
