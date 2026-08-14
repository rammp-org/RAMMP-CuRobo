"""Quaternion / tool-frame helpers (stdlib only).

Two quaternion orders coexist in this stack and mixing them is the classic
silent failure: cuRobo is [w, x, y, z]; ROS and the scene YAML are
[x, y, z, w]. Everything in this module says which one it takes/returns.
"""

import math


def ang_diff(a, b):
    """Smallest-magnitude angular difference a - b, in [-pi, pi).

    Joint drivers report continuous joints wrapped to (-pi, pi]: at the
    Gen3's home, joint_3 sits EXACTLY on the +/-180 deg boundary and its
    reading can flip by 2*pi between messages. Every tolerance comparison
    against a reported joint angle must go through this, or a perfectly
    tracked motion reads as a 6.283 rad "failure" (hit live on the arm)."""
    return (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi


def euler_deg_to_quat_xyzw(rpy_deg):
    """roll/pitch/yaw (degrees) -> (x, y, z, w) quaternion, ROS/xyzw order."""
    r, p, y = (math.radians(float(a)) for a in rpy_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


def xyzw_to_wxyz(q):
    x, y, z, w = q
    return [w, x, y, z]


def wxyz_to_xyzw(q):
    w, x, y, z = q
    return [x, y, z, w]


def yaw_about_world_z(xyzw, rad):
    """Rz(rad) ⊗ q: steer an orientation about the WORLD z axis. xyzw.

    Complements spin_about_tool (which rolls about the tool's OWN z):
    this keeps the attitude — e.g. a level wrist stays level — while
    swinging its heading by `rad` around vertical.
    """
    half = rad / 2.0
    c, s = math.cos(half), math.sin(half)
    x, y, z, w = (float(v) for v in xyzw)
    return (
        c * x - s * y,
        c * y + s * x,
        c * z + s * w,
        c * w - s * z,
    )


def spin_about_tool(wxyz, deg):
    """q ⊗ Rz(deg): spin an orientation about its own tool (z) axis. wxyz."""
    half = math.radians(deg) / 2.0
    sw, sz = math.cos(half), math.sin(half)
    w1, x1, y1, z1 = wxyz
    return [
        w1 * sw - z1 * sz,
        x1 * sw + y1 * sz,
        y1 * sw - x1 * sz,
        z1 * sw + w1 * sz,
    ]


def tool_axis(wxyz):
    """The tool (z) axis of a wxyz orientation: R @ [0, 0, 1]."""
    w, x, y, z = wxyz
    return [
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ]


def tip_to_tool(xyz, wxyz, offset):
    """A FINGERTIP goal -> the cuRobo tool-frame goal that realizes it.

    The 2F-85 pad-face center sits `offset` metres beyond cuRobo's tool_frame
    origin along tool z (0.021 m, measured in the RAMMP-Kinova sim), so the
    tool origin must stop `offset` short of the authored fingertip point.
    """
    ax = tool_axis(wxyz)
    return [
        xyz[0] - offset * ax[0],
        xyz[1] - offset * ax[1],
        xyz[2] - offset * ax[2],
    ]
