"""Refuse to test another checkout's rammp_curobo (this host has an editable install of one).

With RAMMP_CUROBO_GPU=1, which `make verify` sets, a GPU test that would be
skipped fails instead, giving the skip's reason. test_smoke.py skips itself
without torch, cuRobo or a CUDA device, and the suite would then pass on its
offline tests alone. CI has no GPU, and skips them.
"""

import os

import pytest

GPU = os.environ.get("RAMMP_CUROBO_GPU") == "1"


def pytest_configure(config):
    import rammp_curobo

    here = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.realpath(os.path.dirname(os.path.dirname(rammp_curobo.__file__)))
    if os.path.basename(src) == "core" and src != here:
        raise pytest.UsageError(
            "rammp_curobo is imported from %s, not this checkout (%s); "
            "run with PYTHONPATH=%s" % (src, here, here)
        )


def _no_gpu_skip(outcome):
    report = outcome.get_result()
    if GPU and report.skipped and os.path.basename(report.fspath) == "test_smoke.py":
        reason = report.longrepr
        if isinstance(reason, tuple):  # (path, line, reason)
            reason = reason[2]
        report.outcome = "failed"
        report.longrepr = "RAMMP_CUROBO_GPU=1 skips no GPU test: %s" % reason


# old-style wrappers, as in sim/mebot_sim/test/conftest.py: Ubuntu 22.04's
# pytest 6 has no wrapper=True
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    _no_gpu_skip((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    _no_gpu_skip((yield))
