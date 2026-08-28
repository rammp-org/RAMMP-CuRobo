"""Small client-side helpers shared by the in-repo tools and demos."""

import time

import rclpy

# The planner node's action/service namespace — derived from the node name
# declared in planner_node.py (Node("rammp_curobo")). Defined once here for
# every in-repo client; the standalone examples carry their own copy on
# purpose (they demonstrate integration without importing this package).
NODE_NAMESPACE = "/rammp_curobo"


def spin_until_done(node, future, timeout_s):
    """Spin `node` until `future` resolves; None on timeout.

    For single-threaded clients (demos, scripts).
    """
    t0 = time.monotonic()
    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t0 > timeout_s:
            return None
    return future.result()
