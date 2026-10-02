"""Refuse to test another checkout's rammp_curobo (this host has an editable install of one)."""

import os

import pytest


def pytest_configure(config):
    import rammp_curobo

    here = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.realpath(os.path.dirname(os.path.dirname(rammp_curobo.__file__)))
    if os.path.basename(src) == "core" and src != here:
        raise pytest.UsageError(
            "rammp_curobo is imported from %s, not this checkout (%s); "
            "run with PYTHONPATH=%s" % (src, here, here)
        )
