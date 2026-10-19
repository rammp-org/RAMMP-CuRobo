"""A private ROS domain per test, never reused within a session.

A test node must not meet a real arm_state_publisher nearby, nor a node a
previous test left in the graph: a random domain per test met one now and
then (the planner's "another 'rammp_curobo' node is already running" check
fired), so the domains are drawn without replacement.
"""

import random

import pytest

_DOMAINS = iter(random.sample(range(100, 200), 100))


@pytest.fixture
def domain_id():
    return next(_DOMAINS)
